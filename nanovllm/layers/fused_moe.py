"""Fused MoE (bf16/fp16, unquantized).

Vendored from vLLM's ``fused_moe_kernel`` (Apache-2.0,
https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py).
Stripped down to the dense unquantized path — no fp8 / int8 / int4 / mxN,
no expert_map / expert-parallel, no bias. ``moe_align_block_size`` is
reimplemented in pure torch so we don't depend on vLLM's C++/CUDA custom op.

Tensor layout (identical to vllm.fused_experts):
  hidden_states : [M, K]
  w1            : [E, 2N, K]   # gate = w1[:, :N, :], up = w1[:, N:, :]
  w2            : [E, K, N]
  topk_weights  : [M, top_k]   # float, multiplied into the w2 matmul output
  topk_ids      : [M, top_k]   # int, expert index per slot
Returns [M, K].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_moe_kernel(
    a_ptr, b_ptr, c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM,
    num_valid_tokens,
    stride_am: tl.int64, stride_ak: tl.int64,
    stride_be: tl.int64, stride_bk: tl.int64, stride_bn: tl.int64,
    stride_cm: tl.int64, stride_cn: tl.int64,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_m)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (b_ptr + off_experts * stride_be
              + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc = acc * w[:, None]

    acc = acc.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _moe_align_scatter_kernel(
    flat_ids_ptr,       # [num_valid] int32 expert id per (token, slot)
    pad_cum_ptr,        # [E+1]      int32 cumulative padded offsets
    counter_ptr,        # [E]        int32 atomic counter, zero-initialized
    out_ptr,            # [max_padded] int32, prefilled with sentinel
    num_valid,
    BLOCK: tl.constexpr,
):
    """Scatter: for each flat (token, slot), atomically claim a slot inside
    that expert's output range and write the flat index there.

    Ordering within an expert's group is non-deterministic (atomic races), but
    the downstream fused-MoE GEMM is permutation-invariant within a group, so
    end-to-end outputs are unchanged (modulo same reduction order inside each
    row's matmul, which bf16 guarantees).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_valid
    experts = tl.load(flat_ids_ptr + offs, mask=mask, other=0)
    rank = tl.atomic_add(counter_ptr + experts, 1, mask=mask)
    pad_start = tl.load(pad_cum_ptr + experts, mask=mask, other=0)
    out_pos = pad_start + rank
    tl.store(out_ptr + out_pos, offs.to(tl.int32), mask=mask)


def _moe_align_block_size(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """GPU-only moe_align_block_size — same contract as vLLM's CUDA op.

    Buffers use the static upper bound ``num_valid + E*(block-1)`` so we avoid
    a `.item()` host sync per call; the triton kernel early-returns any tiles
    past the true boundary via ``num_tokens_post_padded``.
    """
    device = topk_ids.device
    num_tokens, top_k = topk_ids.shape
    num_valid = num_tokens * top_k
    flat = topk_ids.reshape(-1).to(torch.int32)

    counts = torch.bincount(flat.long(), minlength=num_experts).to(torch.int32)
    padded = ((counts + block_size - 1) // block_size) * block_size
    pad_cum = F.pad(padded.cumsum(0), (1, 0)).to(torch.int32)   # [E+1]

    max_padded = num_valid + num_experts * (block_size - 1)
    sorted_token_ids = torch.full((max_padded,), num_valid,
                                  dtype=torch.int32, device=device)
    if num_valid > 0:
        counter = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK = 64
        grid = (triton.cdiv(num_valid, BLOCK),)
        _moe_align_scatter_kernel[grid](
            flat, pad_cum, counter, sorted_token_ids, num_valid, BLOCK=BLOCK,
        )

    num_blocks_max = max_padded // block_size
    block_starts = torch.arange(num_blocks_max, device=device, dtype=torch.int64) * block_size
    expert_ids = torch.searchsorted(pad_cum[1:].long(), block_starts, right=True).to(torch.int32)
    expert_ids.clamp_max_(num_experts - 1)

    # Keep num_tokens_post_padded on device — the kernel early-returns against it.
    num_tokens_post_padded = pad_cum[-1:].to(torch.int32)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def _default_config(M: int, E: int):
    if M <= E:
        return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}
    return {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2}


def _invoke(A, B, C, topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded, mul_routed_weight, top_k, config):
    M = A.size(0)
    num_tokens = M * top_k
    EM = sorted_token_ids.numel()
    if M < config["BLOCK_SIZE_M"]:
        EM = min(EM, M * top_k * config["BLOCK_SIZE_M"])
    grid = (triton.cdiv(EM, config["BLOCK_SIZE_M"])
            * triton.cdiv(B.size(1), config["BLOCK_SIZE_N"]),)
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16
    tw = topk_weights if topk_weights is not None else A
    _fused_moe_kernel[grid](
        A, B, C,
        tw, sorted_token_ids, expert_ids, num_tokens_post_padded,
        B.size(1), B.size(2), EM, num_tokens,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(1), C.stride(2),
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


@triton.jit
def _silu_and_mul_kernel(
    x_ptr, y_ptr,
    num_rows, N,
    BLOCK_N: tl.constexpr,
):
    """y[m, n] = silu(x[m, n]) * x[m, N + n] for m < num_rows."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= num_rows:
        return
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    base = pid_m * (2 * N)
    g = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + base + N + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-g))
    out = (g * sig).to(u.dtype) * u
    tl.store(y_ptr + pid_m * N + offs, out, mask=mask)


def _silu_and_mul(x: torch.Tensor, y: torch.Tensor):
    """x: [num_rows, 2N]; y: [num_rows, N]. Both contiguous row-major."""
    num_rows, two_N = x.shape
    N = two_N // 2
    BLOCK_N = 512 if N >= 512 else max(32, triton.next_power_of_2(N))
    grid = (num_rows, triton.cdiv(N, BLOCK_N))
    _silu_and_mul_kernel[grid](x, y, num_rows, N, BLOCK_N=BLOCK_N)


# Note: a triton moe_sum kernel exists upstream but gave slightly different
# bf16 rounding than torch's `.sum(dim=1)`, which drifts over ~50 MoE layers
# and broke bit-identity with vLLM's reference. torch's sum is already a
# fused CUDA reduction; keeping it avoids the issue with no visible speed
# difference (<1% in end-to-end bench).


def fused_experts(hidden_states: torch.Tensor,
                  w1: torch.Tensor,
                  w2: torch.Tensor,
                  topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor,
                  inplace: bool = False) -> torch.Tensor:
    """Dense bf16/fp16 fused MoE (SiLU gated)."""
    assert hidden_states.is_contiguous() and hidden_states.dtype in (torch.bfloat16, torch.float16)
    assert w1.stride(-1) == 1 and w2.stride(-1) == 1

    M, K = hidden_states.shape
    E, two_N, _ = w1.shape
    N = two_N // 2
    top_k = topk_ids.size(1)

    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous()
    cfg = _default_config(M, E)

    sorted_token_ids, expert_ids, num_post = _moe_align_block_size(
        topk_ids, cfg["BLOCK_SIZE_M"], E)

    device, dtype = hidden_states.device, hidden_states.dtype
    cache1 = torch.empty((M, top_k, 2 * N), device=device, dtype=dtype)
    cache2 = torch.empty((M * top_k, N), device=device, dtype=dtype)
    cache3 = torch.empty((M, top_k, K), device=device, dtype=dtype)

    _invoke(hidden_states, w1, cache1,
            topk_weights=topk_weights, sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids, num_tokens_post_padded=num_post,
            mul_routed_weight=False, top_k=top_k, config=cfg)

    # silu(gate) * up, fused
    _silu_and_mul(cache1.view(-1, 2 * N), cache2)

    _invoke(cache2, w2, cache3,
            topk_weights=topk_weights, sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids, num_tokens_post_padded=num_post,
            mul_routed_weight=True, top_k=1, config=cfg)

    out = cache3.sum(dim=1)
    if inplace:
        hidden_states.copy_(out)
        return hidden_states
    return out
