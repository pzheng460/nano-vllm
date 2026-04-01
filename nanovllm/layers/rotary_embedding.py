from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb_neox(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Neox style: split into halves [x1, x2]."""
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def apply_rotary_emb_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Interleaved (GPT-J) style: rotate paired elements (x0,x1), (x2,x3), ..."""
    orig_dtype = x.dtype
    x = x.float()
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    y_even = x_even * cos - x_odd * sin
    y_odd = x_odd * cos + x_even * sin
    return torch.stack([y_even, y_odd], dim=-1).reshape_as(x).to(orig_dtype)


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
        cos_sin = self.cos_sin_cache[positions]
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
        assert rope_type == "default", f"Unsupported rope_type: {rope_type}"
    return _get_rope_cached(head_size, rotary_dim, max_position, base, is_neox_style)


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
