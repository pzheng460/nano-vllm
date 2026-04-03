import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.attention import Attention, store_kvcache
from nanovllm.layers.linear import (
    ReplicatedLinear, MergedColumnParallelLinear, RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.context import get_context
from nanovllm.utils.device import is_cuda

class SiluAndMul(torch.nn.Module):
    """SiluAndMul without torch.compile for numerical consistency."""
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b


class RMSNorm(torch.nn.Module):
    """RMSNorm without torch.compile for numerical consistency with vLLM."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, residual=None):
        orig_dtype = x.dtype
        if residual is not None:
            x = x.float() + residual.float()
            residual = x.to(orig_dtype)
        else:
            x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(orig_dtype) * self.weight
        if residual is not None:
            return x, residual
        return x

if is_cuda():
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


# ---------------------------------------------------------------------------
# Sink Attention
# ---------------------------------------------------------------------------
class PanguSinkAttention(nn.Module):
    """Attention with learnable sink KV tokens and partial RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        qk_rope_dim: int,
        qk_nope_dim: int,
        v_channels: int,
        sink_len: int,
        max_position: int = 4096 * 32,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int | None = None,
    ) -> None:
        super().__init__()
        _tp_size = tp_size if tp_size is not None else (
            dist.get_world_size(tp_group) if tp_group is not None else dist.get_world_size())
        self.total_num_heads = num_heads
        assert self.total_num_heads % _tp_size == 0
        self.num_heads = self.total_num_heads // _tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % _tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // _tp_size
        self.tp_size = _tp_size
        self.tp_rank = 0 if _tp_size == 1 else (
            dist.get_rank(tp_group) if tp_group is not None else dist.get_rank())

        self.qk_rope_dim = qk_rope_dim
        self.qk_nope_dim = qk_nope_dim
        self.head_dim = qk_rope_dim + qk_nope_dim
        self.v_channels = v_channels
        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_kv_heads * self.head_dim
        self.v_size = self.num_kv_heads * self.v_channels
        self.scaling = self.head_dim ** -0.5
        self.sink_len = sink_len

        # QKV as merged column parallel (Q, K, V may have different dims)
        # Checkpoint already has fused qkv_proj, so we override weight_loader
        # to accept full-tensor loading (TP shard on dim 0)
        q_total = self.total_num_heads * self.head_dim
        k_total = self.total_num_kv_heads * self.head_dim
        v_total = self.total_num_kv_heads * self.v_channels
        self.qkv_proj = MergedColumnParallelLinear(
            hidden_size,
            [q_total, k_total, v_total],
            bias=qkv_bias,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        # Override: checkpoint has pre-fused qkv_proj, just TP-shard on output dim
        self.qkv_proj.weight.weight_loader = self._qkv_weight_loader
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_channels,
            hidden_size,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )

        # K layernorm (applied before RoPE)
        self.k_layernorm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Partial RoPE: only rotates qk_rope_dim dimensions, interleaved style
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.qk_rope_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )

        # V padding for FA2 (requires K/V same head_dim)
        self._pad_v = (self.v_channels != self.head_dim)
        self._effective_v_dim = self.head_dim if self._pad_v else self.v_channels

        # Attention backend (uses head_dim for both K and V due to padding)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

        # Learnable sink key parameters (loaded from checkpoint)
        if sink_len > 0:
            self.param_sink_key = nn.Parameter(
                torch.empty(sink_len, self.num_kv_heads, self.head_dim))
            self.param_sink_key.weight_loader = self._sink_weight_loader
            self.param_sink_value = nn.Parameter(
                torch.empty(sink_len, self.num_kv_heads, self.v_channels))
            self.param_sink_value.weight_loader = self._sink_weight_loader
        # Processed sink KV (after k_layernorm + v_padding)
        self.register_buffer("_sink_k", torch.zeros(0), persistent=False)
        self.register_buffer("_sink_v", torch.zeros(0), persistent=False)
        self.sink_block_ids = []  # Set by model_runner after KV cache allocation

    def _qkv_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id=None):
        """Load pre-fused qkv_proj weight with TP sharding.
        Handles both fused (no shard_id) and per-component (shard_id) loading."""
        if shard_id is not None:
            # Delegate to MergedColumnParallelLinear's original loader
            return MergedColumnParallelLinear.weight_loader(self.qkv_proj, param, loaded_weight, shard_id)
        # Fused load: shard each component (Q, K, V) by TP on dim 0
        q_total, k_total, v_total = self.qkv_proj.output_sizes
        q, k, v = loaded_weight.split([q_total, k_total, v_total], dim=0)
        q = q.chunk(self.tp_size, dim=0)[self.tp_rank]
        k = k.chunk(self.tp_size, dim=0)[self.tp_rank]
        v = v.chunk(self.tp_size, dim=0)[self.tp_rank]
        param.data.copy_(torch.cat([q, k, v], dim=0))

    def _sink_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Load sink parameters with TP sharding on the head dimension."""
        shard_size = param.data.shape[1]  # num_kv_heads per partition
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(1, start_idx, shard_size)
        param.data.copy_(loaded_weight)

    def post_weight_load(self):
        """Apply k_layernorm to sink keys and pad sink values. Called after weights loaded."""
        if self.sink_len > 0:
            sink_k = self.k_layernorm(self.param_sink_key.data)
            sink_v = self.param_sink_value.data
            if self._pad_v:
                sink_v = F.pad(sink_v, (0, self.head_dim - self.v_channels))
            self._sink_k = sink_k.contiguous()
            self._sink_v = sink_v.contiguous()

    def _populate_sink_to_cache(self, block_size: int):
        """Write sink KV into reserved cache blocks (block IDs stored in self.sink_block_ids)."""
        if not hasattr(self, 'sink_block_ids') or not self.sink_block_ids:
            return
        k_cache, v_cache = self.attn.k_cache, self.attn.v_cache
        sink_k = self._sink_k  # [sink_len, num_kv_heads, head_dim]
        sink_v = self._sink_v
        for i, block_id in enumerate(self.sink_block_ids):
            start = i * block_size
            end = min(start + block_size, self.sink_len)
            num = end - start
            k_cache[block_id, :num] = sink_k[start:end]
            v_cache[block_id, :num] = sink_v[start:end]
            # Zero-fill unfilled slots (these slots should NOT be in attention range)
            if num < block_size:
                k_cache[block_id, num:].zero_()
                v_cache[block_id, num:].zero_()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        # Reshape for attention
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.v_channels)

        # K layernorm before RoPE
        k = self.k_layernorm(k)

        # Partial RoPE
        q, k = self.rotary_emb(positions, q, k)

        # Pad V for FA2 compatibility
        if self._pad_v:
            v = F.pad(v, (0, self.head_dim - self.v_channels))

        # Delegate to _forward_with_sink
        o = self._forward_with_sink(q, k, v)

        # Slice V back from head_dim to v_channels
        if self._pad_v:
            o = o.view(-1, self.num_heads, self._effective_v_dim)
            o = o[:, :, :self.v_channels].contiguous()
            o = o.view(-1, self.num_heads * self.v_channels)
        else:
            o = o.flatten(1, -1)

        output = self.o_proj(o)
        return output

    def _forward_with_sink(self, q, k, v):
        """Attention with sink KV prepended. Unified path for prefill and decode."""
        context = get_context()
        k_cache, v_cache = self.attn.k_cache, self.attn.v_cache

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if self.sink_len == 0 or self._sink_k.numel() == 0:
            # No sink: standard attention
            if context.is_prefill:
                if context.block_tables is not None:
                    k, v = k_cache, v_cache
                return flash_attn_varlen_func(
                    q, k, v,
                    max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scaling, causal=True,
                    block_table=context.block_tables)
            else:
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1), k_cache, v_cache,
                    cache_seqlens=context.context_lens, block_table=context.block_tables,
                    softmax_scale=self.scaling, causal=True)
                return o.squeeze(1)

        # --- Sink attention: gather from cache + prepend sink, always use varlen ---
        if context.block_tables is not None:
            # Decode or verify-with-cache: gather KV from cache
            return self._gather_and_attend_with_sink(q, context, k_cache, v_cache)
        else:
            # Initial prefill (no cache): prepend sink to current batch K/V
            return self._initial_prefill_with_sink(q, k, v, context)

    def _gather_and_attend_with_sink(self, q, context, k_cache, v_cache):
        """Unified sink attention for both decode and verify/prefill-with-cache.
        Gathers KV from cache, prepends sink, uses flash_attn_varlen_func."""
        block_size = k_cache.shape[1]
        sink_k, sink_v = self._sink_k, self._sink_v

        # Get per-sequence Q lengths and cached KV lengths
        if context.is_prefill:
            cu_q_in = context.cu_seqlens_q
            num_seqs = cu_q_in.numel() - 1
            # cu_seqlens_k has NO sink adjustment — raw cached lengths
            seq_k_lens = [(context.cu_seqlens_k[i+1] - context.cu_seqlens_k[i]).item()
                          for i in range(num_seqs)]
        else:
            num_seqs = q.shape[0]
            cu_q_in = None
            seq_k_lens = context.context_lens.tolist()

        new_k_parts, new_v_parts = [], []
        new_cu_q, new_cu_k = [0], [0]
        for i in range(num_seqs):
            cached_len = seq_k_lens[i]
            # Prepend sink
            new_k_parts.append(sink_k)
            new_v_parts.append(sink_v)
            # Gather cached KV from block_table
            bt = context.block_tables[i]
            rem = cached_len
            for b in range((cached_len + block_size - 1) // block_size):
                bid = bt[b].item()
                take = min(rem, block_size)
                new_k_parts.append(k_cache[bid, :take])
                new_v_parts.append(v_cache[bid, :take])
                rem -= take
            new_cu_k.append(new_cu_k[-1] + self.sink_len + cached_len)
            q_len = (cu_q_in[i+1] - cu_q_in[i]).item() if cu_q_in is not None else 1
            new_cu_q.append(new_cu_q[-1] + q_len)

        all_k = torch.cat(new_k_parts, dim=0)
        all_v = torch.cat(new_v_parts, dim=0)
        cu_q = torch.tensor(new_cu_q, dtype=torch.int32, device=q.device)
        cu_k = torch.tensor(new_cu_k, dtype=torch.int32, device=q.device)
        max_q = max(new_cu_q[i+1] - new_cu_q[i] for i in range(num_seqs))
        max_k = max(new_cu_k[i+1] - new_cu_k[i] for i in range(num_seqs))

        return flash_attn_varlen_func(
            q, all_k, all_v,
            max_seqlen_q=max_q, cu_seqlens_q=cu_q,
            max_seqlen_k=max_k, cu_seqlens_k=cu_k,
            softmax_scale=self.scaling, causal=True)

    def _initial_prefill_with_sink(self, q, k, v, context):
        """Initial prefill (no cache): prepend sink K/V to current batch K/V."""
        cu_k = context.cu_seqlens_k
        num_seqs = cu_k.numel() - 1
        sink_k, sink_v = self._sink_k, self._sink_v

        new_k_parts, new_v_parts = [], []
        new_cu_k = [0]
        for i in range(num_seqs):
            start, end = cu_k[i].item(), cu_k[i + 1].item()
            new_k_parts.append(sink_k)
            new_k_parts.append(k[start:end])
            new_v_parts.append(sink_v)
            new_v_parts.append(v[start:end])
            new_cu_k.append(new_cu_k[-1] + self.sink_len + (end - start))

        all_k = torch.cat(new_k_parts, dim=0)
        all_v = torch.cat(new_v_parts, dim=0)
        new_cu_k = torch.tensor(new_cu_k, dtype=torch.int32, device=q.device)

        return flash_attn_varlen_func(
            q, all_k, all_v,
            max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k + self.sink_len, cu_seqlens_k=new_cu_k,
            softmax_scale=self.scaling, causal=True)


# ---------------------------------------------------------------------------
# MLP (standard, used for dense layers and shared experts)
# ---------------------------------------------------------------------------
class PanguMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False,
            tp_group=tp_group, tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
            tp_group=tp_group, tp_size=tp_size,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# MoE (Mixture of Experts) — packed weights, batched matmul
# ---------------------------------------------------------------------------
class PanguMoE(nn.Module):
    """Mixture of Experts with packed expert weights for fast batched computation."""

    def __init__(
        self,
        config,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int | None = None,
    ) -> None:
        super().__init__()
        _tp_size = tp_size if tp_size is not None else (
            dist.get_world_size(tp_group) if tp_group is not None else dist.get_world_size())
        _tp_rank = 0 if _tp_size == 1 else (
            dist.get_rank(tp_group) if tp_group is not None else dist.get_rank())

        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = getattr(config, 'n_shared_experts', 0) or 0
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', False)
        self.tp_size = _tp_size
        self.tp_rank = _tp_rank
        self.tp_group = tp_group

        hidden_size = config.hidden_size
        moe_intermediate_size = config.moe_intermediate_size
        inter_per_tp = moe_intermediate_size // _tp_size

        # Gate (replicated across TP ranks)
        self.gate = ReplicatedLinear(
            hidden_size, self.n_routed_experts, bias=False,
            tp_group=tp_group, tp_size=tp_size,
        )
        if getattr(config, 'router_enable_expert_bias', False):
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32))
            self.gate.e_score_correction_bias.weight_loader = (
                lambda param, loaded_weight: param.data.copy_(loaded_weight))
        else:
            self.gate.e_score_correction_bias = None

        # Shared experts (if any)
        if self.n_shared_experts > 0:
            shared_intermediate = moe_intermediate_size * self.n_shared_experts
            self.shared_experts = PanguMLP(
                hidden_size, shared_intermediate,
                tp_group=tp_group, tp_size=tp_size,
            )
        else:
            self.shared_experts = None

        # Packed expert weights: [n_experts, 2*inter_per_tp, hidden] and [n_experts, hidden, inter_per_tp]
        self.w13_weight = nn.Parameter(
            torch.empty(self.n_routed_experts, 2 * inter_per_tp, hidden_size))
        self.w2_weight = nn.Parameter(
            torch.empty(self.n_routed_experts, hidden_size, inter_per_tp))
        # Custom weight loaders for packed expert weights
        self.w13_weight.weight_loader = self._w13_weight_loader
        self.w2_weight.weight_loader = self._w2_weight_loader
        self._inter_per_tp = inter_per_tp
        self._hidden_size = hidden_size
        self._moe_intermediate_size = moe_intermediate_size

    def _w13_weight_loader(self, param, loaded_weight, shard_id, expert_id=None):
        """Load gate_proj (shard=0) or up_proj (shard=1) into packed w13."""
        if expert_id is None:
            expert_id = self._current_expert_id
        tp_slice = loaded_weight.chunk(self.tp_size, dim=0)[self.tp_rank]
        offset = 0 if shard_id == 0 else self._inter_per_tp
        param.data[expert_id, offset:offset + self._inter_per_tp, :] = tp_slice

    def _w2_weight_loader(self, param, loaded_weight, expert_id=None):
        """Load down_proj into packed w2 for one expert."""
        if expert_id is None:
            expert_id = self._current_expert_id
        tp_slice = loaded_weight.chunk(self.tp_size, dim=1)[self.tp_rank]
        param.data[expert_id, :, :] = tp_slice

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape

        # Routing
        router_logits = self.gate(hidden_states)
        scores = torch.sigmoid(router_logits.float())
        if self.gate.e_score_correction_bias is not None:
            biased_scores = scores + self.gate.e_score_correction_bias
            topk_ids = torch.topk(biased_scores, self.top_k, dim=-1)[1]
            topk_weights = scores.gather(1, topk_ids)
        else:
            topk_weights, topk_ids = torch.topk(scores, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_states.dtype)

        # Shared experts
        shared_output = None
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)

        # Expert computation
        NK = num_tokens * self.top_k
        flat_ids = topk_ids.view(-1)
        flat_w = topk_weights.view(-1)
        if NK <= 256:
            routed_output = self._forward_bmm(hidden_states, flat_ids, flat_w,
                                              num_tokens, hidden_dim)
        else:
            routed_output = self._forward_grouped(hidden_states, flat_ids, flat_w,
                                                  num_tokens, hidden_dim)

        # All-reduce for TP
        if self.tp_size > 1:
            dist.all_reduce(routed_output, group=self.tp_group)

        # Scaling
        if hidden_states.dtype != torch.float16:
            routed_output = routed_output * self.routed_scaling_factor
        elif shared_output is not None:
            shared_output = shared_output * (1.0 / self.routed_scaling_factor)

        if shared_output is not None:
            routed_output = routed_output + shared_output
        return routed_output

    def _forward_bmm(self, x, flat_ids, flat_w, N, H):
        """Decode path: bmm-based expert computation."""
        NK = flat_ids.shape[0]
        flat_x = x.repeat_interleave(self.top_k, dim=0)
        sel_w13 = self.w13_weight[flat_ids]
        gate_up = torch.bmm(sel_w13, flat_x.unsqueeze(-1)).squeeze(-1)
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        sel_w2 = self.w2_weight[flat_ids]
        out = torch.bmm(sel_w2, act.unsqueeze(-1)).squeeze(-1)
        out = out * flat_w.unsqueeze(-1)
        return out.view(N, self.top_k, H).sum(dim=1)

    def _forward_grouped(self, x, flat_ids, flat_w, N, H):
        """Prefill path: sort tokens by expert, contiguous per-expert matmul."""
        flat_x = x.repeat_interleave(self.top_k, dim=0)
        sort_idx = flat_ids.argsort()
        sorted_ids = flat_ids[sort_idx]
        sorted_x = flat_x[sort_idx]
        sorted_w = flat_w[sort_idx]
        flat_out = torch.empty_like(sorted_x)
        unique_experts, counts = sorted_ids.unique_consecutive(return_counts=True)
        offset = 0
        for eid, cnt in zip(unique_experts.tolist(), counts.tolist()):
            end = offset + cnt
            x_e = sorted_x[offset:end]
            gate_up = x_e @ self.w13_weight[eid].T
            gate, up = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up
            flat_out[offset:end] = act @ self.w2_weight[eid].T
            offset = end
        flat_out = flat_out * sorted_w.unsqueeze(-1)
        result = torch.empty_like(flat_out)
        result[sort_idx] = flat_out
        return result.view(N, self.top_k, H).sum(dim=1)


# ---------------------------------------------------------------------------
# Decoder Layer (with sandwich norm)
# ---------------------------------------------------------------------------
class PanguDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_idx: int,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = config.hidden_size

        # Attention
        sink_len = getattr(config, 'param_sink_number', 0) or 0
        qk_rope_dim = getattr(config, 'qk_rope_dim', None)
        qk_nope_dim = getattr(config, 'qk_nope_dim', None)
        v_channels = getattr(config, 'v_channels', None)

        if qk_rope_dim is not None and qk_nope_dim is not None:
            # PanGu sink attention with partial RoPE
            self.self_attn = PanguSinkAttention(
                hidden_size=hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                qk_rope_dim=qk_rope_dim,
                qk_nope_dim=qk_nope_dim,
                v_channels=v_channels or (qk_rope_dim + qk_nope_dim),
                sink_len=sink_len,
                max_position=getattr(config, 'max_position_embeddings', 131072),
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=getattr(config, 'qkv_bias', False),
                rope_theta=getattr(config, 'rope_theta', 10000),
                rope_scaling=getattr(config, 'rope_scaling', None),
                tp_group=tp_group,
                tp_size=tp_size,
            )
        else:
            # Fallback to standard attention (no sink, no partial RoPE)
            from nanovllm.models.qwen3 import Qwen3Attention
            self.self_attn = Qwen3Attention(
                hidden_size=hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                max_position=getattr(config, 'max_position_embeddings', 131072),
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=getattr(config, 'attention_bias', True),
                head_dim=getattr(config, 'head_dim', None),
                rope_theta=getattr(config, 'rope_theta', 10000),
                rope_scaling=getattr(config, 'rope_scaling', None),
                tp_group=tp_group,
                tp_size=tp_size,
            )

        # MLP or MoE
        n_routed_experts = getattr(config, 'n_routed_experts', None)
        first_k_dense = getattr(config, 'first_k_dense_replace', config.num_hidden_layers)
        if n_routed_experts is not None and layer_idx >= first_k_dense:
            self.mlp = PanguMoE(config, tp_group=tp_group, tp_size=tp_size)
            self._is_moe = True
        else:
            self.mlp = PanguMLP(
                hidden_size, config.intermediate_size,
                tp_group=tp_group, tp_size=tp_size,
            )
            self._is_moe = False

        # Norms
        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        # Sandwich norm: extra pre/post MLP norms
        self.sandwich_norm = getattr(config, 'sandwich_norm', False)
        if self.sandwich_norm:
            self.pre_mlp_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
            self.post_mlp_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', None)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-attention norm
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self-attention
        hidden_states = self.self_attn(positions, hidden_states)

        # FP16 overflow fix: scale hidden_states and residual before norm
        if (self.routed_scaling_factor is not None
                and hidden_states.dtype == torch.float16):
            hidden_states = hidden_states * (1.0 / self.routed_scaling_factor)
            if self.layer_idx == 0:
                residual = residual * (1.0 / self.routed_scaling_factor)

        # Post-attention norm (sandwich vs standard)
        if self.sandwich_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, residual = self.pre_mlp_layernorm(hidden_states, residual)
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MLP / MoE
        hidden_states = self.mlp(hidden_states)

        # FP16 fix for dense MLP layers
        if (self.routed_scaling_factor is not None
                and not self._is_moe
                and hidden_states.dtype == torch.float16):
            hidden_states = hidden_states * (1.0 / self.routed_scaling_factor)

        # Post-MLP sandwich norm
        if self.sandwich_norm:
            hidden_states = self.post_mlp_layernorm(hidden_states)

        return hidden_states, residual


# ---------------------------------------------------------------------------
# Shared Head (for MTP - independent norm + lm_head)
# ---------------------------------------------------------------------------
class PanguSharedHead(nn.Module):
    """Independent norm and LM head for MTP layers."""

    def __init__(self, config, tp_group=None, tp_size=None):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            tp_group=tp_group, tp_size=tp_size,
        )

    def forward(self, hidden_states, residual=None):
        """Apply norm (optionally with residual) and return normed hidden."""
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# MTP Layer
# ---------------------------------------------------------------------------
class PanguMTPLayer(nn.Module):
    """Multi-token prediction layer with independent head/norm."""

    def __init__(self, config, layer_idx: int, tp_group=None, tp_size=None):
        super().__init__()
        hidden_size = config.hidden_size

        self.enorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            hidden_size * 2, hidden_size, bias=False,
            tp_group=tp_group, tp_size=tp_size,
        )
        # Full decoder layer as MTP block
        self.mtp_block = PanguDecoderLayer(
            config, layer_idx=layer_idx,
            tp_group=tp_group, tp_size=tp_size,
        )
        # Independent head with its own norm
        self.shared_head = PanguSharedHead(config, tp_group=tp_group, tp_size=tp_size)

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (normed, prenorm). normed for logits, prenorm for chaining."""
        # Mask position 0 embeddings (not needed by MTP - no prior token to predict)
        input_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, input_embeds)
        input_embeds = self.enorm(input_embeds)
        previous_hidden = self.hnorm(hidden_states)
        hidden = self.eh_proj(torch.cat([input_embeds, previous_hidden], dim=-1))

        hidden, residual = self.mtp_block(positions, hidden, residual=None)

        # Match vLLM: compute unnormed sum first, then norm separately
        # (vLLM returns residual + hidden_states, then shared_head norms it)
        prenorm = residual + hidden if residual is not None else hidden
        normed = self.shared_head(prenorm)  # norm without residual addition
        return normed, prenorm


# ---------------------------------------------------------------------------
# PanGu Model
# ---------------------------------------------------------------------------
class PanguModel(nn.Module):

    def __init__(self, config, tp_group=None, tp_size=None):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size,
            tp_group=tp_group, tp_size=tp_size,
        )
        self.layers = nn.ModuleList([
            PanguDecoderLayer(config, layer_idx=i, tp_group=tp_group, tp_size=tp_size)
            for i in range(config.num_hidden_layers)
        ])
        self.mtp_layers = nn.ModuleList([
            PanguMTPLayer(
                config,
                layer_idx=config.num_hidden_layers + i,
                tp_group=tp_group, tp_size=tp_size,
            )
            for i in range(getattr(config, 'num_nextn_predict_layers', 0))
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        return_penultimate: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        penultimate = None
        num_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if return_penultimate and i == num_layers - 2:
                penultimate = (hidden_states + residual).clone()
        hidden_states, residual = self.norm(hidden_states, residual)
        if return_penultimate:
            return residual, penultimate
        return residual


# ---------------------------------------------------------------------------
# PanGu CausalLM
# ---------------------------------------------------------------------------
class PanguForCausalLM(nn.Module):
    # Checkpoint already has fused qkv_proj, only need gate/up → gate_up_proj mapping
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, tp_group=None, tp_size=None):
        super().__init__()
        self.config = config
        self.tp_group = tp_group
        self.tp_size = tp_size if tp_size is not None else (
            dist.get_world_size(tp_group) if tp_group is not None else dist.get_world_size())
        self.tp_rank = 0 if self.tp_size == 1 else (
            dist.get_rank(tp_group) if tp_group is not None else dist.get_rank())
        self.model = PanguModel(config, tp_group=tp_group, tp_size=tp_size)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            tp_group=tp_group, tp_size=tp_size,
        )
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        return_penultimate: bool = False,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, return_penultimate=return_penultimate)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.norm(hidden_states)
        return self.lm_head(hidden_states)

    def compute_logits_all(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_states, self.lm_head.weight)
        if self.tp_size > 1:
            all_logits = ([torch.empty_like(logits) for _ in range(self.tp_size)]
                          if self.tp_rank == 0 else None)
            dist.gather(logits, all_logits, 0, group=self.tp_group)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits

    def load_weights(self, path: str):
        """Fast weight loader: direct GPU loading + pre-built name mapping."""
        import os
        import json
        from glob import glob
        from safetensors import safe_open
        from nanovllm.utils.loader import default_weight_loader

        params_dict = dict(self.named_parameters())
        num_hidden = self.config.num_hidden_layers
        num_mtp = getattr(self.config, 'num_nextn_predict_layers', 0)
        packed = self.packed_modules_mapping
        device = str(next(self.parameters()).device)

        # --- Phase 1: build ckpt_name → (param, loader, shard_id) mapping ---
        # Read safetensors index to get ckpt_name list without opening every file
        index_file = os.path.join(path, "model.safetensors.index.json")
        if os.path.isfile(index_file):
            with open(index_file) as f:
                weight_map = json.load(f)["weight_map"]  # ckpt_name → filename
        else:
            weight_map = {}
            for sf in sorted(glob(os.path.join(path, "*.safetensors"))):
                with safe_open(sf, "pt", "cpu") as f:
                    for k in f.keys():
                        weight_map[k] = os.path.basename(sf)

        name_mapping = {}  # ckpt_name → (param_name, param, weight_loader, shard_id|None)
        for ckpt_name in weight_map:
            mapped = self._resolve_weight_name(
                ckpt_name, packed, params_dict, num_hidden, num_mtp)
            if mapped is not None:
                name_mapping[ckpt_name] = mapped

        # --- Phase 2: load weights file-by-file, direct to GPU ---
        # Group by file for sequential read
        file_to_keys = {}
        for ckpt_name, _ in name_mapping.items():
            fname = weight_map[ckpt_name]
            file_to_keys.setdefault(fname, []).append(ckpt_name)

        for fname, keys in file_to_keys.items():
            fpath = os.path.join(path, fname)
            with safe_open(fpath, "pt", device) as f:
                for ckpt_name in keys:
                    entry = name_mapping[ckpt_name]
                    tensor = f.get_tensor(ckpt_name)
                    if len(entry) == 5:
                        # Expert weight: (param_name, param, loader, shard_id, expert_id)
                        _, param, loader, shard_id, expert_id = entry
                        if shard_id is not None:
                            loader(param, tensor, shard_id, expert_id=expert_id)
                        else:
                            loader(param, tensor, expert_id=expert_id)
                    else:
                        _, param, loader, shard_id = entry
                        if shard_id is not None:
                            loader(param, tensor, shard_id)
                        else:
                            loader(param, tensor)

        # Post weight load (e.g., apply k_layernorm to sink keys)
        for module in self.modules():
            if hasattr(module, 'post_weight_load') and module is not self:
                module.post_weight_load()

    def _resolve_weight_name(self, name, packed, params_dict, num_hidden, num_mtp):
        """Map a checkpoint weight name to (param_name, param, weight_loader, shard_id).
        Returns None if the weight should be skipped.
        For expert weights, returns tuple with extra expert_id."""
        import re
        from nanovllm.utils.loader import default_weight_loader

        if "rotary_emb.inv_freq" in name:
            return None
        if getattr(self.config, 'tie_word_embeddings', False) and "lm_head.weight" in name:
            return None

        # MTP layer weight remapping
        if "layers" in name and num_mtp > 0:
            layer_idx = int(name.split("layers.")[-1].split(".")[0])
            mtp_idx = layer_idx - num_hidden
            if 0 <= mtp_idx < num_mtp:
                name = self._rewrite_mtp_weight_name(layer_idx, mtp_idx, name)
            elif mtp_idx >= num_mtp:
                return None

        # e_score_correction_bias remapping
        if name.endswith("e_score_correction_bias") and "gate." not in name:
            name = name.replace("e_score_correction_bias",
                                "gate.e_score_correction_bias")

        # --- Routed expert weights: experts.{N}.gate_proj/up_proj/down_proj ---
        m = re.search(r'\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$', name)
        if m:
            expert_id = int(m.group(1))
            proj_type = m.group(2)
            # Find the MoE module's packed weight
            moe_prefix = name[:m.start()] + ".mlp."
            if proj_type in ("gate_proj", "up_proj"):
                param_name = moe_prefix + "w13_weight"
                shard_id = 0 if proj_type == "gate_proj" else 1
            else:  # down_proj
                param_name = moe_prefix + "w2_weight"
                shard_id = None
            if param_name in params_dict:
                param = params_dict[param_name]
                loader = getattr(param, "weight_loader")
                # Return 5-tuple for expert weights
                return (param_name, param, loader, shard_id, expert_id)
            return None

        # Packed modules mapping (gate_proj→gate_up_proj for shared_experts/dense MLP)
        for ckpt_name, (param_name, shard_id) in packed.items():
            dotted = f".{ckpt_name}."
            if dotted in name:
                mapped_name = name.replace(ckpt_name, param_name)
                if mapped_name in params_dict:
                    param = params_dict[mapped_name]
                    loader = getattr(param, "weight_loader")
                    return (mapped_name, param, loader, shard_id)
                break

        # Direct parameter load
        if name in params_dict:
            param = params_dict[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            return (name, param, loader, None)

        return None

    def _rewrite_mtp_weight_name(self, layer_idx, mtp_idx, name):
        """Rewrite checkpoint MTP weight name to model parameter name.

        Checkpoint format: model.layers.{num_hidden+mtp_idx}.{component}
        Model format: model.mtp_layers.{mtp_idx}.{component}

        Components like self_attn, mlp, input_layernorm, etc. go under .mtp_block
        Components like enorm, hnorm, eh_proj, shared_head stay at top level.
        embed_tokens is shared with the main model.
        """
        spec_layer_names = ["enorm", "hnorm", "eh_proj", "shared_head"]
        shared_names = ["embed_tokens"]

        # Check if this is a spec layer weight or shared weight
        is_spec = any(w in name for w in spec_layer_names)
        is_shared = any(w in name for w in shared_names)

        old_prefix = f"model.layers.{layer_idx}."
        if is_shared:
            # Shared weights map to model-level
            name = name.replace(old_prefix, "model.")
        elif is_spec:
            # Spec layer weights (enorm, hnorm, eh_proj, shared_head)
            name = name.replace(old_prefix, f"model.mtp_layers.{mtp_idx}.")
        else:
            # Transformer block weights (self_attn, mlp, norms)
            name = name.replace(old_prefix,
                                f"model.mtp_layers.{mtp_idx}.mtp_block.")
        return name
