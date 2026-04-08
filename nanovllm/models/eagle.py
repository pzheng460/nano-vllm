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
        bias: bool = True,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int | None = None,
    ) -> None:
        super().__init__()
        _tp_size = tp_size if tp_size is not None else (dist.get_world_size(tp_group) if tp_group is not None else dist.get_world_size())
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
            bias=bias,
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
    """EAGLE decoder layer: no input_layernorm, only post_attention_layernorm."""

    def __init__(self, config, tp_group=None, tp_size=None) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        # Detect bias: Qwen2 has QKV bias, Llama/Qwen3 don't
        model_type = getattr(config, 'model_type', '')
        if hasattr(config, 'attention_bias'):
            qkv_bias = config.attention_bias
        elif hasattr(config, 'qkv_bias'):
            qkv_bias = config.qkv_bias
        else:
            # Fallback by model type: qwen2 has bias, llama/qwen3 don't
            qkv_bias = model_type in ('qwen2',)
        self.self_attn = EAGLEAttention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, 'head_dim', None),
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            bias=qkv_bias,
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


class Eagle3Attention(nn.Module):
    """EAGLE3 attention: GQA, input_dim=2*hidden (takes concatenated input), no QKV bias."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, max_position=4096*32,
                 head_dim=None, rms_norm_eps=1e-6, rope_theta=1000000, rope_scaling=None,
                 tp_group=None, tp_size=None):
        super().__init__()
        _tp_size = tp_size if tp_size is not None else (dist.get_world_size(tp_group) if tp_group is not None else dist.get_world_size())
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // _tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads // _tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        # EAGLE3: QKV input is 2*hidden (concatenated fc_output + hidden)
        self.qkv_proj = QKVParallelLinear(
            hidden_size * 2, self.head_dim, self.total_num_heads, self.total_num_kv_heads,
            bias=False, tp_group=tp_group, tp_size=tp_size)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, hidden_size, bias=False,
            tp_group=tp_group, tp_size=tp_size)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim,
            max_position=max_position, base=rope_theta, rope_scaling=rope_scaling)
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

    def forward(self, positions, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class Eagle3DecoderLayer(nn.Module):
    """EAGLE3 decoder layer: input_layernorm(embeds) + hidden_norm(fc_out) → cat → attn → post_attn_ln → mlp."""

    def __init__(self, config, tp_group=None, tp_size=None):
        super().__init__()
        H = config.hidden_size
        self.input_layernorm = RMSNorm(H, eps=config.rms_norm_eps)
        self.self_attn = Eagle3Attention(
            hidden_size=H, num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, 'head_dim', None),
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            rope_scaling=getattr(config, 'rope_scaling', None),
            tp_group=tp_group, tp_size=tp_size)
        self.mlp = Qwen3MLP(hidden_size=H, intermediate_size=config.intermediate_size,
                             hidden_act=config.hidden_act, tp_group=tp_group, tp_size=tp_size)
        self.post_attention_layernorm = RMSNorm(H, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(H, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, embeds):
        # vLLM EAGLE3 layer 0: norm embeds + norm hidden → cat → attn → fused residual
        normed_embeds = self.input_layernorm(embeds)
        normed_hidden = self.hidden_norm(hidden_states)
        residual = hidden_states  # pre-norm hidden as residual
        attn_input = torch.cat([normed_embeds, normed_hidden], dim=-1)
        hidden_states = self.self_attn(positions, attn_input)
        # Fused post_attention_layernorm: norm(attn_out + residual)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Eagle3Model(nn.Module):
    """EAGLE3 model: fc(3*H) + decoder(2*H attn) + own lm_head (reduced vocab) + d2t/t2d mapping."""
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, draft_config, embed_tokens, tp_group=None, tp_size=None):
        super().__init__()
        H = config.hidden_size
        draft_vocab = getattr(draft_config, 'draft_vocab_size', config.vocab_size)
        self.embed_tokens = embed_tokens  # shared with target
        self.fc = ReplicatedLinear(H * 3, H, bias=False, tp_group=tp_group, tp_size=tp_size)
        self.midlayer = Eagle3DecoderLayer(draft_config, tp_group=tp_group, tp_size=tp_size)
        self.norm = RMSNorm(H, eps=config.rms_norm_eps)
        # Own lm_head with reduced vocab
        from nanovllm.layers.embed_head import ParallelLMHead
        self.lm_head = ParallelLMHead(draft_vocab, H, tp_group=tp_group, tp_size=tp_size)
        # d2t: draft_id → target_id mapping (int64)
        # t2d_mask: target_id → bool (is in draft vocab)
        # t2d_map: target_id → draft_id (built from d2t after loading)
        self.register_buffer('d2t', torch.zeros(draft_vocab, dtype=torch.int64))
        self.register_buffer('t2d_mask', torch.zeros(config.vocab_size, dtype=torch.bool))
        self.register_buffer('t2d_map', torch.zeros(config.vocab_size, dtype=torch.int64))
        self.draft_vocab_size = draft_vocab

    def forward(self, input_ids, positions, target_hidden, aux_hiddens=None):
        token_embeds = self.embed_tokens(input_ids)
        # Step 0: fc(aux_hiddens) → hidden, then midlayer(hidden, embeds)
        # Step 1+: no fc, midlayer(previous_output, embeds) directly
        if aux_hiddens is not None:
            fc_hidden = self.fc(aux_hiddens)
        else:
            fc_hidden = target_hidden
        hidden_states, residual = self.midlayer(positions, fc_hidden, token_embeds)
        hidden = hidden_states + residual
        return hidden


    def compute_logits(self, hidden_states):
        """Compute draft logits in reduced vocab space."""
        import torch.nn.functional as F
        hidden_normed = self.norm(hidden_states)
        return F.linear(hidden_normed, self.lm_head.weight)

    def draft_to_target(self, draft_token_ids):
        """Map draft token IDs (16k) to target token IDs (152k)."""
        return self.d2t[draft_token_ids]

    def target_to_draft(self, target_token_ids):
        """Map target token IDs (152k) to draft token IDs (16k)."""
        return self.t2d[target_token_ids]

    def load_weights(self, path):
        """Custom weight loader handling d2t/t2d buffers and packed modules."""
        import os
        from glob import glob
        from safetensors import safe_open
        from nanovllm.utils.loader import _load_weight
        packed = self.packed_modules_mapping

        def _load_tensor(name, tensor):
            if name == 'd2t':
                self.d2t.copy_(tensor.to(self.d2t.dtype))
            elif name == 't2d':
                self.t2d_mask.copy_(tensor.to(torch.bool))
            else:
                try:
                    _load_weight(self, packed, name, tensor)
                except (AttributeError, KeyError):
                    pass

        safetensor_files = glob(os.path.join(path, "*.safetensors"))
        if safetensor_files:
            for file in safetensor_files:
                with safe_open(file, "pt", "cpu") as f:
                    for name in f.keys():
                        _load_tensor(name, f.get_tensor(name))
        else:
            import torch
            bin_files = glob(os.path.join(path, "pytorch_model*.bin"))
            for file in bin_files:
                state_dict = torch.load(file, map_location="cpu", weights_only=True)
                for name, tensor in state_dict.items():
                    _load_tensor(name, tensor)
                del state_dict

        # Build reverse mapping: target_id → draft_id (from d2t)
        for draft_id in range(self.d2t.shape[0]):
            target_id = self.d2t[draft_id].item()
            self.t2d_map[target_id] = draft_id


class EAGLEModel(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, embed_tokens, lm_head, tp_group=None, tp_size=None):
        super().__init__()
        hidden_size = config.hidden_size
        self.embed_tokens = embed_tokens   # shared, frozen
        self.lm_head = lm_head             # shared, frozen
        model_type = getattr(config, 'model_type', '')
        fc_bias = model_type in ('qwen2',)  # Qwen2 EAGLE has fc bias, Llama/Qwen3 don't
        self.fc = ReplicatedLinear(hidden_size * 2, hidden_size, bias=fc_bias, tp_group=tp_group, tp_size=tp_size)
        self.layers = nn.ModuleList([EAGLEDecoderLayer(config, tp_group=tp_group, tp_size=tp_size)])

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
