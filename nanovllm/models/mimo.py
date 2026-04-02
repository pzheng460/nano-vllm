import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.models.qwen3 import Qwen3Attention, Qwen3MLP, Qwen3DecoderLayer


class MiMoMTPLayer(nn.Module):

    def __init__(self, config, tp_group=None, tp_size=None) -> None:
        super().__init__()
        hidden_size = config.hidden_size

        self.token_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.hidden_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.input_proj = ReplicatedLinear(hidden_size * 2, hidden_size, bias=False,
                                           tp_group=tp_group, tp_size=tp_size)

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
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            tp_group=tp_group,
            tp_size=tp_size,
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
        # Mask inputs at position 0 (no prior context for MTP), matching vLLM
        input_embeds = input_embeds.clone()
        input_embeds[positions == 0] = 0
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

    def __init__(self, config, tp_group=None, tp_size=None) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, tp_group=tp_group, tp_size=tp_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, tp_group=tp_group, tp_size=tp_size) for _ in range(config.num_hidden_layers)])
        self.mtp_layers = nn.ModuleList([
            MiMoMTPLayer(config, tp_group=tp_group, tp_size=tp_size)
            for _ in range(getattr(config, 'num_nextn_predict_layers', 0))
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        return_penultimate: bool = False,
    ) -> torch.Tensor:
        """Returns pre-norm hidden states (before final RMSNorm).
        If return_penultimate=True, also returns hidden states from the
        second-to-last layer (for SSD early MTP prediction).
        """
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


class MiMoForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, tp_group=None, tp_size=None) -> None:
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = tp_size if tp_size is not None else (dist.get_world_size(tp_group) if tp_group is not None else dist.get_world_size())
        self.tp_rank = 0 if self.tp_size == 1 else (dist.get_rank(tp_group) if tp_group is not None else dist.get_rank())
        self.model = MiMoModel(config, tp_group=tp_group, tp_size=tp_size)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, tp_group=tp_group, tp_size=tp_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        return_penultimate: bool = False,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, return_penultimate=return_penultimate)

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
        logits = F.linear(hidden_states, self.lm_head.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0, group=self.tp_group)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
