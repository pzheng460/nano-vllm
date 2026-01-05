import torch
from torch import nn

from nanovllm.utils.context import get_context
from nanovllm.utils.device import is_npu, is_cuda

# Lazy-initialized references for device-specific implementations
_store_kvcache_impl = None
_flash_attn_varlen_func = None
_flash_attn_with_kvcache = None


def _init_cuda_backend():
    """Initialize CUDA-specific implementations (lazy)."""
    global _store_kvcache_impl, _flash_attn_varlen_func, _flash_attn_with_kvcache

    import triton
    import triton.language as tl
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    _flash_attn_varlen_func = flash_attn_varlen_func
    _flash_attn_with_kvcache = flash_attn_with_kvcache

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

    def store_kvcache_cuda(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor, block_size: int = 256):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

    _store_kvcache_impl = store_kvcache_cuda


def _init_npu_backend():
    """Initialize NPU-specific implementations (lazy)."""
    global _store_kvcache_impl

    import torch_npu

    def store_kvcache_npu(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor, block_size: int = 256):
        """NPU version of KV cache storage using scatter_update_."""
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim

        # Calculate slot indices for scatter update
        slot_indices = torch.stack([
            slot_mapping // block_size,  # block index
            slot_mapping % block_size    # offset in block
        ], dim=1)

        # Reshape key/value for scatter update
        cast_key = key.reshape(-1, 1, D)
        cast_value = value.reshape(-1, 1, D)

        # Use scatter_update_ for in-place update
        torch_npu.scatter_update_(k_cache, slot_indices, cast_key, -2)
        torch_npu.scatter_update_(v_cache, slot_indices, cast_value, -2)

    _store_kvcache_impl = store_kvcache_npu


def _ensure_backend_initialized():
    """Ensure device-specific backend is initialized."""
    global _store_kvcache_impl
    if _store_kvcache_impl is not None:
        return

    if is_cuda():
        _init_cuda_backend()
    elif is_npu():
        _init_npu_backend()
    else:
        raise RuntimeError("Device backend not initialized. Call DeviceBackend.initialize() first.")


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor, block_size: int = 256):
    """Store key-value pairs to cache. Auto-initializes backend on first call."""
    _ensure_backend_initialized()
    _store_kvcache_impl(key, value, k_cache, v_cache, slot_mapping, block_size)


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
        self._use_npu = None  # Lazy initialization

    def _get_use_npu(self):
        """Lazy check for NPU usage."""
        if self._use_npu is None:
            self._use_npu = is_npu()
        return self._use_npu

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        use_npu = self._get_use_npu()

        if use_npu and Attention.SHARE_MASK_TRIL_SPARSE is None:
            Attention.SHARE_MASK_TRIL_SPARSE = ~torch.tril(
                torch.ones((2048, 2048), dtype=torch.bool, device='npu')
            )

        if k_cache.numel() and v_cache.numel():
            block_size = k_cache.shape[0] if use_npu else 256
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, block_size)

        if use_npu:
            return self._forward_npu(q, k, v, context, k_cache, v_cache)
        else:
            return self._forward_cuda(q, k, v, context, k_cache, v_cache)

    def _forward_cuda(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """CUDA forward using Flash Attention."""
        _ensure_backend_initialized()
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = _flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = _flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o

    def _forward_npu(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """NPU forward using torch_npu fused attention."""
        import torch_npu

        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache

            # Check if single sequence or multiple sequences
            num_seqs = len(context.cu_seqlens_q) - 1
            if num_seqs == 1:
                # Single sequence - use BSND layout
                seq_len = context.cu_seqlens_q[-1].item()
                o, _, _ = torch_npu.npu_fused_infer_attention_score(
                    q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="BSND",
                    scale=self.scale,
                    pre_tokens=65535,
                    next_tokens=0,
                    sparse_mode=0,
                )
                o = o.view(-1, self.num_heads, self.head_dim)
            else:
                # Multiple sequences - use TND layout with npu_fusion_attention
                actual_seq_qlen = (context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]).tolist()
                actual_seq_kvlen = (context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]).tolist()
                o, _, _ = torch_npu.npu_fusion_attention(
                    q, k, v,
                    head_num=self.num_heads,
                    input_layout="TND",
                    scale=self.scale,
                    pre_tokens=65535,
                    next_tokens=0,
                    actual_seq_qlen=actual_seq_qlen,
                    actual_seq_kvlen=actual_seq_kvlen,
                    sparse_mode=0,
                )
        else:    # decode
            batch_size = q.size(0)
            block_size = k_cache.shape[1]

            o, _, _ = torch_npu.npu_fused_infer_attention_score(
                q.view(batch_size, 1, self.num_heads, self.head_dim),
                k_cache,
                v_cache,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                scale=self.scale,
                actual_seq_lengths_kv=context.context_lens.tolist(),
                block_table=context.block_tables,
                block_size=block_size,
            )
            o = o.view(batch_size, self.num_heads, self.head_dim)

        return o
