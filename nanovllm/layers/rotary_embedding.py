from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb_neox(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Neox style: split into halves [x1, x2].
    Matches vLLM: compute in input dtype (bf16), NOT float32."""
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


def apply_rotary_emb_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Interleaved (GPT-J) style: rotate paired elements (x0,x1), (x2,x3), ..."""
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    y_even = x_even * cos - x_odd * sin
    y_odd = x_odd * cos + x_even * sin
    return torch.stack([y_even, y_odd], dim=-1).reshape_as(x)


# Backward compat alias
apply_rotary_emb = apply_rotary_emb_neox


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool = True,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.is_partial = rotary_dim < head_size
        self.is_neox_style = is_neox_style
        self._apply_fn = apply_rotary_emb_neox if is_neox_style else apply_rotary_emb_interleaved
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype or cache.device != query.device:
            cache = cache.to(device=query.device, dtype=query.dtype)
            self.cos_sin_cache = cache
        cos_sin = cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_partial:
            query = _apply_partial(query, cos, sin, self.rotary_dim, self._apply_fn)
            key = _apply_partial(key, cos, sin, self.rotary_dim, self._apply_fn)
        else:
            query = self._apply_fn(query, cos, sin)
            key = self._apply_fn(key, cos, sin)
        return query, key


def _apply_partial(x, cos, sin, rotary_dim, apply_fn):
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x_rot = apply_fn(x_rot, cos, sin)
    return torch.cat((x_rot, x_pass), dim=-1)


def _compute_llama3_inv_freq(base, rotary_dim, rope_scaling):
    """Compute inv_freq for llama3 rope type with frequency scaling."""
    factor = rope_scaling.get("factor", 8.0)
    low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
    old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)

    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelens = 2 * torch.pi / inv_freq
    # Scale frequencies based on wavelength
    inv_freq_scaled = inv_freq / factor
    smooth = (old_context_len / wavelens - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smooth = smooth.clamp(0, 1)
    inv_freq = torch.where(wavelens > low_freq_wavelen, inv_freq_scaled,
                torch.where(wavelens < high_freq_wavelen, inv_freq,
                             (1 - smooth) * inv_freq_scaled + smooth * inv_freq))
    return inv_freq


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
    is_neox_style: bool = True,
):
    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", "default")
        if rope_type == "llama3":
            inv_freq = _compute_llama3_inv_freq(base, rotary_dim, rope_scaling)
            return _get_rope_with_inv_freq(head_size, rotary_dim, max_position, inv_freq, is_neox_style)
        elif rope_type != "default":
            raise ValueError(f"Unsupported rope_type: {rope_type}")
    return _get_rope_cached(head_size, rotary_dim, max_position, base, is_neox_style)


def _get_rope_with_inv_freq(head_size, rotary_dim, max_position, inv_freq, is_neox_style):
    """Create RotaryEmbedding with precomputed inv_freq (for llama3 scaling)."""
    rope = RotaryEmbedding(head_size, rotary_dim, max_position, 10000.0, is_neox_style)
    # Override cos_sin_cache with scaled frequencies
    t = torch.arange(max_position, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
    rope.cos_sin_cache = cache
    return rope


@lru_cache(8)
def _get_rope_cached(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool = True,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, is_neox_style)
    return rotary_emb
