import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.models.qwen3 import Qwen3Attention, Qwen3MLP, Qwen3DecoderLayer


class MiMoMTPLayer(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size

        self.token_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.hidden_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.input_proj = ReplicatedLinear(hidden_size * 2, hidden_size, bias=False)

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 640000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.final_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (normed, prenorm). normed for logits, prenorm for chaining."""
        input_embeds = self.token_layernorm(input_embeds)
        previous_hidden = self.hidden_layernorm(hidden_states)
        hidden = self.input_proj(torch.cat([previous_hidden, input_embeds], dim=-1))

        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn(positions, hidden)
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        hidden = residual + hidden

        normed = self.final_layernorm(hidden)
        return normed, hidden


class MiMoModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.mtp_layers = nn.ModuleList([
            MiMoMTPLayer(config)
            for _ in range(getattr(config, 'num_nextn_predict_layers', 0))
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Returns pre-norm hidden states (before final RMSNorm)."""
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, residual = self.norm(hidden_states, residual)
        return residual


class MiMoForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = MiMoModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model.norm(hidden_states)
        return self.lm_head(hidden_states)

    def compute_logits_all(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits for ALL positions. Caller must ensure hidden_states are already normed."""
        tp_size = dist.get_world_size()
        tp_rank = dist.get_rank()
        logits = F.linear(hidden_states, self.lm_head.weight)
        if tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(tp_size)] if tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if tp_rank == 0 else None
        return logits
