import os
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from glob import glob
from safetensors import safe_open

from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    QKVParallelLinear, RowParallelLinear, ReplicatedLinear,
)
from nanovllm.layers.rotary_embedding import get_rope


def _default_rope_theta(config) -> float:
    """Pick rope_theta default by model family when the config doesn't set it.
    Qwen2/Qwen3 use 1e6; LLaMA (and derivatives like Vicuna) use 1e4."""
    if hasattr(config, 'rope_theta') and config.rope_theta is not None:
        return float(config.rope_theta)
    mt = getattr(config, 'model_type', '').lower()
    return 1_000_000.0 if mt.startswith('qwen') else 10_000.0
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
            rope_theta=_default_rope_theta(config),
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
    """EAGLE3 attention: GQA, input_dim=2*H (takes concatenated input), no QKV bias."""

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
    """EAGLE3 decoder layer (layer 0 only): norm embeds + norm hidden -> cat -> attn -> post_attn_ln -> mlp."""

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
            rope_theta=_default_rope_theta(config),
            rope_scaling=getattr(config, 'rope_scaling', None),
            tp_group=tp_group, tp_size=tp_size)
        self.mlp = Qwen3MLP(hidden_size=H, intermediate_size=config.intermediate_size,
                             hidden_act=config.hidden_act, tp_group=tp_group, tp_size=tp_size)
        self.post_attention_layernorm = RMSNorm(H, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(H, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, embeds):
        # vLLM EAGLE3 layer 0: norm embeds + norm hidden -> cat -> attn -> fused residual
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
    """EAGLE-3 speculative decoding model.

    Structure mirrors vLLM's Eagle3LlamaForCausalLM:
      self.model   -- inner model (embed_tokens, fc, layers, norm)
      self.lm_head -- reduced-vocab head (top-level)
      self.d2t     -- draft->target vocab mapping (top-level)
    """
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, draft_config, embed_tokens=None, tp_group=None, tp_size=None):
        super().__init__()
        H = config.hidden_size
        draft_vocab = getattr(draft_config, 'draft_vocab_size', config.vocab_size)
        self.draft_vocab_size = draft_vocab
        self.vocab_size = config.vocab_size

        # Inner model (vLLM: self.model = LlamaModel(...))
        self.model = nn.Module()
        if embed_tokens is not None:
            self.model.embed_tokens = embed_tokens  # shared with target
        else:
            from nanovllm.layers.embed_head import VocabParallelEmbedding
            self.model.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, H, tp_group=tp_group, tp_size=tp_size)
        self.model.fc = ReplicatedLinear(H * 3, H, bias=False, tp_group=tp_group, tp_size=tp_size)
        self.model.layers = nn.ModuleList([
            Eagle3DecoderLayer(draft_config, tp_group=tp_group, tp_size=tp_size)])
        self.model.norm = RMSNorm(H, eps=config.rms_norm_eps)

        # Top-level: lm_head and vocab mapping (same as vLLM)
        from nanovllm.layers.embed_head import ParallelLMHead
        self.lm_head = ParallelLMHead(draft_vocab, H, tp_group=tp_group, tp_size=tp_size)
        self.register_buffer('d2t', torch.zeros(draft_vocab, dtype=torch.int64))
        self.register_buffer('t2d_mask', torch.zeros(config.vocab_size, dtype=torch.bool))
        self.register_buffer('t2d_map', torch.zeros(config.vocab_size, dtype=torch.int64))

    # -- vLLM-style API --

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states):
        """Combine auxiliary hidden states from target layers. [N, 3*H] -> [N, H]."""
        return self.model.fc(hidden_states)

    def forward(self, input_ids, positions, hidden_states, aux_hiddens=None):
        input_embeds = self.embed_input_ids(input_ids)
        if aux_hiddens is not None:
            hidden_states = self.combine_hidden_states(aux_hiddens)
        hidden_states, residual = self.model.layers[0](positions, hidden_states, input_embeds)
        return hidden_states + residual

    def compute_logits(self, hidden_states):
        """Compute logits in draft vocab space."""
        hidden_normed = self.model.norm(hidden_states)
        return F.linear(hidden_normed, self.lm_head.weight)

    def draft_to_target(self, draft_token_ids):
        """Map draft token IDs (reduced vocab) -> target token IDs."""
        return self.d2t[draft_token_ids]

    def target_to_draft(self, target_token_ids):
        """Map target token IDs -> draft token IDs."""
        return self.t2d_map[target_token_ids]

    # -- Weight loading (vLLM-style name remapping) --

    def load_weights(self, path):
        """Load weights with vLLM-style name remapping.

        EAGLE-3 checkpoint names:  fc.*, midlayer.*, lm_head.*, norm.*, d2t, t2d
        Target checkpoint names:   model.embed_tokens.*

        Remapping: non-lm_head names get 'model.' prefix; midlayer -> layers.0
        """
        from nanovllm.utils.loader import _load_weight

        def _remap_and_load(name, tensor):
            # Vocab-mapping buffers
            if name == 'd2t':
                # d2t stores offsets: target_id = draft_id + d2t[draft_id]
                base = torch.arange(tensor.shape[0], dtype=tensor.dtype, device="cpu")
                direct = (tensor + base).to(dtype=self.d2t.dtype, device=self.d2t.device)
                self.d2t.copy_(direct)
                return
            if name == 't2d':
                return  # reverse mapping built from d2t below
            # vLLM convention: everything except lm_head lives under model.*
            if 'lm_head' not in name and not name.startswith('model.'):
                name = 'model.' + name
            if 'midlayer.' in name:
                name = name.replace('midlayer.', 'layers.0.')
            try:
                _load_weight(self, self.packed_modules_mapping, name, tensor)
            except (AttributeError, KeyError, RuntimeError):
                pass  # skip weights that don't match (e.g., target model layers)

        safetensor_files = glob(os.path.join(path, "*.safetensors"))
        if safetensor_files:
            for file in safetensor_files:
                with safe_open(file, "pt", "cpu") as f:
                    for name in f.keys():
                        _remap_and_load(name, f.get_tensor(name))
        else:
            bin_files = glob(os.path.join(path, "pytorch_model*.bin"))
            for file in bin_files:
                state_dict = torch.load(file, map_location="cpu", weights_only=True)
                for name, tensor in state_dict.items():
                    _remap_and_load(name, tensor)
                del state_dict

        # Build reverse mapping: target_id -> draft_id
        for draft_id in range(self.d2t.shape[0]):
            target_id = self.d2t[draft_id].item()
            if target_id < self.t2d_map.shape[0]:
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
        # Some EAGLE-1 checkpoints ship an fc.bias (Vicuna, Qwen2); others omit
        # it (EAGLE-LLaMA3.1). Create the param unconditionally, but zero it so
        # that a checkpoint lacking fc.bias leaves the fused add as a no-op
        # (torch.empty() would leave garbage and break the draft forward).
        self.fc = ReplicatedLinear(hidden_size * 2, hidden_size, bias=True, tp_group=tp_group, tp_size=tp_size)
        torch.nn.init.zeros_(self.fc.bias)
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
