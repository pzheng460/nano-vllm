import torch
from torch import nn

from nanovllm.utils.context import get_context
from nanovllm.utils.device import is_npu, is_cuda
from nanovllm.config import Config, get_config, set_config, reset_config

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
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)

    def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor, block_size: int = 256):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

elif is_npu():
    import torch_npu
    import torchair as tng

    NZ_DIM = 16

    def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor, block_size: int = 256):
        """NPU version of KV cache storage using torch_npu.npu_scatter_pa_kv_cache."""
        N, num_kv_heads, head_dim = key.shape
        num_blocks = k_cache.shape[0]
        
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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if self._use_npu and Attention.SHARE_MASK_TRIL_SPARSE is None:
            Attention.SHARE_MASK_TRIL_SPARSE = ~torch.tril(
                torch.ones((2048, 2048), dtype=torch.bool, device='npu')
            )

        if k_cache.numel() and v_cache.numel():
            block_size = k_cache.shape[1] if self._use_npu else 256
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, block_size)

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
        config = get_config()
        """NPU forward using torch_npu fused attention."""
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache

            o = torch_npu.npu_fused_infer_attention_score(
                q, k, v,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                scale=self.scale,
                sparse_mode=3,
                atten_mask=Attention.SHARE_MASK_TRIL_SPARSE,
                actual_seq_lengths=context.cu_seqlens_q[1:],
                actual_seq_lengths_kv=context.cu_seqlens_k[1:],
                next_tokens=0
            )[0]
            o = o.view(-1, self.num_heads, self.head_dim)
        else:
            # decode
            batch_size = q.size(0)
            block_size = k_cache.shape[1]
            block_num, block_size, num_kv_heads, head_dim = k_cache.shape
            k_cache = k_cache.reshape(-1, num_kv_heads, head_dim // NZ_DIM, block_size, NZ_DIM)
            v_cache = v_cache.reshape(-1, num_kv_heads, head_dim // NZ_DIM, block_size, NZ_DIM)

            if config.enforce_eager:
                o = torch_npu.npu_fused_infer_attention_score_v2(
                    q,
                    k_cache,
                    v_cache,
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
                )[0]
            else:
                o = tng.ops.npu_fused_infer_attention_score(
                    q,
                    k_cache,
                    v_cache,
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="TND",
                    scale=self.scale,
                    actual_seq_lengths=context.cu_seqlens_q,
                    actual_seq_lengths_kv=context.context_lens,
                    block_table=context.block_tables,
                    block_size=block_size,
                    sparse_mode=3,
                    atten_mask=Attention.SHARE_MASK_TRIL_SPARSE,
                )[0]
            o = o.view(batch_size, self.num_heads, self.head_dim)

        return o
