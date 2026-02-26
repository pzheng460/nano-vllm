import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    QKVParallelLinear, RowParallelLinear, ReplicatedLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.qwen3 import Qwen3MLP


class EAGLEAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
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

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class EAGLEDecoderLayer(nn.Module):
    """EAGLE decoder layer: no input_layernorm, only post_attention_layernorm.
    Matches the EAGLE-Qwen2-7B-Instruct checkpoint structure."""

    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = EAGLEAttention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_attention_heads,  # full MHA
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, 'head_dim', None),
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class EAGLEModel(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, embed_tokens, lm_head):
        super().__init__()
        hidden_size = config.hidden_size
        self.embed_tokens = embed_tokens   # shared, frozen
        self.lm_head = lm_head             # shared, frozen
        self.fc = ReplicatedLinear(hidden_size * 2, hidden_size, bias=True)
        self.layers = nn.ModuleList([EAGLEDecoderLayer(config)])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        token_embeds = self.embed_tokens(input_ids)
        hidden = self.fc(torch.cat([token_embeds, target_hidden], dim=-1))
        hidden = self.layers[0](positions, hidden)
        return hidden

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
