import torch
from torch import nn

from nanovllm.utils.context import get_context, add_graph_handle, get_graph_params
from nanovllm.utils.device import is_npu, is_cuda

# Initialize device-specific implementations at import time
# (DeviceBackend is already initialized in nanovllm.__init__)
if is_cuda():
    import triton
    import triton.language as tl
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < D
        key_offsets = idx * key_stride + offsets
        value_offsets = idx * value_stride + offsets
        key = tl.load(key_ptr + key_offsets, mask=mask)
        value = tl.load(value_ptr + value_offsets, mask=mask)
        cache_offsets = slot * D + offsets
        tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
        tl.store(v_cache_ptr + cache_offsets, value, mask=mask)

    def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N
        BLOCK_D = triton.next_power_of_2(D)
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D, BLOCK_D)

elif is_npu():
    import torch_npu

    NZ_DIM = 16

    def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
        """NPU version of KV cache storage using torch_npu.npu_scatter_pa_kv_cache."""
        N, num_kv_heads, head_dim = key.shape
        num_blocks = k_cache.shape[0]
        block_size = k_cache.shape[1]
        
        k_cache_nz = k_cache.view(num_blocks, num_kv_heads * head_dim // NZ_DIM, block_size, NZ_DIM)
        v_cache_nz = v_cache.view(num_blocks, num_kv_heads * head_dim // NZ_DIM, block_size, NZ_DIM)

        torch_npu.npu_scatter_pa_kv_cache(
            key.contiguous(),
            value.contiguous(),
            k_cache_nz,
            v_cache_nz,
            slot_mapping
        )

else:
    raise RuntimeError("No CUDA or NPU device available. DeviceBackend must be initialized before importing attention module.")


class Attention(nn.Module):

    SHARE_MASK_TRIL_SPARSE = None

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self._use_npu = is_npu()
        if self._use_npu and Attention.SHARE_MASK_TRIL_SPARSE is None:
            Attention.SHARE_MASK_TRIL_SPARSE = ~torch.tril(
                torch.ones((2048, 2048), dtype=torch.bool, device='npu')
            )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if self._use_npu:
            return self._forward_npu(q, k, v, context, k_cache, v_cache)
        else:
            return self._forward_cuda(q, k, v, context, k_cache, v_cache)

    def _forward_cuda(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """CUDA forward using Flash Attention."""
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o

    def _forward_npu(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """NPU forward using torch_npu FIA kernels."""
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache

            o = torch_npu.npu_fused_infer_attention_score_v2(
                q, k, v,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                softmax_scale=self.scale,
                sparse_mode=3,
                atten_mask=Attention.SHARE_MASK_TRIL_SPARSE,
                actual_seq_qlen=context.cu_seqlens_q[1:],
                actual_seq_kvlen=context.cu_seqlens_k[1:],
                next_tokens=0
            )[0]
            o = o.view(-1, self.num_heads, self.head_dim)
        else:
            # decode
            batch_size = q.shape[0]
            block_size = k_cache.shape[1]
            k_cache_nz = k_cache.view(-1, self.num_kv_heads, self.head_dim // NZ_DIM, block_size, NZ_DIM)
            v_cache_nz = v_cache.view(-1, self.num_kv_heads, self.head_dim // NZ_DIM, block_size, NZ_DIM)

            # Pre-allocate output tensors
            o = torch.empty(batch_size, self.num_heads, self.head_dim, dtype=q.dtype, device=q.device)
            softmax_lse = torch.empty(batch_size, dtype=torch.float32, device=q.device)

            if context.capturing:
                # ACL Graph capture mode: mark this op as updatable
                stream = torch.npu.current_stream()
                torch.npu.graph_task_group_begin(stream)

                torch_npu.npu_fused_infer_attention_score_v2.out(
                    q,
                    k_cache_nz,
                    v_cache_nz,
                    num_query_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="TND",
                    softmax_scale=self.scale,
                    block_table=context.block_tables,
                    block_size=block_size,
                    sparse_mode=3,
                    atten_mask=Attention.SHARE_MASK_TRIL_SPARSE,
                    actual_seq_qlen=context.cu_seqlens_q,
                    actual_seq_kvlen=context.context_lens,
                    out=[o, softmax_lse],
                )

                handle = torch.npu.graph_task_group_end(stream)
                # Save handle and params for later update
                add_graph_handle(
                    bs=batch_size,
                    handle=handle,
                    attn_params=(
                        q, k_cache_nz, v_cache_nz,
                        self.num_heads, self.num_kv_heads,
                        self.scale, context.block_tables, block_size,
                        Attention.SHARE_MASK_TRIL_SPARSE,
                        o, softmax_lse,
                    )
                )
            else:
                # eager mode
                torch_npu.npu_fused_infer_attention_score_v2.out(
                    q,
                    k_cache_nz,
                    v_cache_nz,
                    num_query_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="TND",
                    softmax_scale=self.scale,
                    block_table=context.block_tables,
                    block_size=block_size,
                    sparse_mode=3,
                    atten_mask=Attention.SHARE_MASK_TRIL_SPARSE,
                    actual_seq_qlen=context.cu_seqlens_q,
                    actual_seq_kvlen=context.context_lens,
                    out=[o, softmax_lse],
                )

        return o

    @staticmethod
    def update_graph_params(graph_bs: int, cu_seqlens_q: list, context_lens: list):
        """Update List parameters before ACL Graph replay.

        NPU attention ops require cu_seqlens_q and context_lens as List,
        which cannot be captured by graph. Use graph_task_update to update them.
        """
        # Pad to graph_bs size
        padded_cu_seqlens_q = list(cu_seqlens_q) + [(i + 1) for i in range(len(cu_seqlens_q), graph_bs)]
        padded_context_lens = list(context_lens) + [1] * (graph_bs - len(context_lens))

        graph_params = get_graph_params()
        stream = torch.npu.current_stream()

        for handle, attn_params in zip(graph_params.handles[graph_bs], graph_params.attn_params[graph_bs]):
            (q, k_cache_nz, v_cache_nz, num_heads, num_kv_heads,
             scale, block_table, block_size, atten_mask, o, softmax_lse) = attn_params

            torch.npu.graph_task_update_begin(stream, handle)
            torch_npu.npu_fused_infer_attention_score_v2.out(
                q, k_cache_nz, v_cache_nz,
                num_query_heads=num_heads, num_key_value_heads=num_kv_heads,
                input_layout="TND", softmax_scale=scale,
                block_table=block_table, block_size=block_size,
                sparse_mode=3, atten_mask=atten_mask,
                actual_seq_qlen=padded_cu_seqlens_q,
                actual_seq_kvlen=padded_context_lens,
                out=[o, softmax_lse],
            )
            torch.npu.graph_task_update_end(stream)
