"""Llama model family (Llama 1/2/3, Vicuna, and derivatives).

Key differences vs Qwen3:
- No QK normalization
- No QKV bias (LLaMA uses bias=False on all projections)
- rope_theta defaults to 1e4 (vs Qwen3's 1e6). LLaMA-3/3.1 use 5e5, which is
  recorded in config.json and picked up via getattr.
- Supports rope_scaling={"rope_type":"llama3",...} for LLaMA-3.1 long-context.
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 2048,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int | None = None,
    ) -> None:
        super().__init__()
        _tp_size = tp_size if tp_size is not None else (
            dist.get_world_size(tp_group) if tp_group is not None else dist.get_world_size()
        )
        self.total_num_heads = num_heads
        assert self.total_num_heads % _tp_size == 0
        self.num_heads = self.total_num_heads // _tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % _tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // _tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
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
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        return self.down_proj(x)


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, 'head_dim', None),
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, "rope_theta", 10000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(self, config, tp_group=None, tp_size=None):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, tp_group=tp_group, tp_size=tp_size,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, tp_group=tp_group, tp_size=tp_size)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, tp_group=None, tp_size=None):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = tp_size if tp_size is not None else (
            dist.get_world_size(tp_group) if tp_group is not None else dist.get_world_size()
        )
        self.tp_rank = 0 if self.tp_size == 1 else (
            dist.get_rank(tp_group) if tp_group is not None else dist.get_rank()
        )
        self.model = LlamaModel(config, tp_group=tp_group, tp_size=tp_size)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, tp_group=tp_group, tp_size=tp_size,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def compute_logits_all(self, hidden_states):
        logits = F.linear(hidden_states, self.lm_head.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0, group=self.tp_group)
            if self.tp_rank == 0:
                logits = torch.cat(all_logits, dim=-1)
        return logits
