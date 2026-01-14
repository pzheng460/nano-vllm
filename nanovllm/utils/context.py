from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | list | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | list | None = None
    block_tables: torch.Tensor | None = None
    # For ACL Graph: whether we are capturing
    capturing: bool = False


@dataclass
class GraphParams:
    """Parameters for ACL Graph task update."""
    # handles[bs] = list of handles for each attention layer
    handles: dict[int, list[Any]] = field(default_factory=dict)
    # attn_params[bs] = list of (q, k_cache, v_cache, ...) for each attention layer
    attn_params: dict[int, list[tuple]] = field(default_factory=dict)


_CONTEXT = Context()
_GRAPH_PARAMS: GraphParams | None = None


def get_context():
    global _CONTEXT
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, capturing=False):
    global _CONTEXT
    _CONTEXT.is_prefill = is_prefill
    _CONTEXT.cu_seqlens_q = cu_seqlens_q
    _CONTEXT.cu_seqlens_k = cu_seqlens_k
    _CONTEXT.max_seqlen_q = max_seqlen_q
    _CONTEXT.max_seqlen_k = max_seqlen_k
    _CONTEXT.slot_mapping = slot_mapping
    _CONTEXT.context_lens = context_lens
    _CONTEXT.block_tables = block_tables
    _CONTEXT.capturing = capturing


def reset_context():
    global _CONTEXT
    _CONTEXT.is_prefill = False
    _CONTEXT.cu_seqlens_q = None
    _CONTEXT.cu_seqlens_k = None
    _CONTEXT.max_seqlen_q = 0
    _CONTEXT.max_seqlen_k = 0
    _CONTEXT.slot_mapping = None
    _CONTEXT.context_lens = None
    _CONTEXT.block_tables = None
    _CONTEXT.capturing = False


def init_graph_params(batch_sizes: list[int]):
    """Initialize graph params for ACL Graph capture."""
    global _GRAPH_PARAMS
    _GRAPH_PARAMS = GraphParams(
        handles={bs: [] for bs in batch_sizes},
        attn_params={bs: [] for bs in batch_sizes},
    )


def get_graph_params() -> GraphParams | None:
    return _GRAPH_PARAMS


def add_graph_handle(bs: int, handle, attn_params: tuple):
    """Add a handle and params for a specific batch size."""
    global _GRAPH_PARAMS
    if _GRAPH_PARAMS is not None:
        _GRAPH_PARAMS.handles[bs].append(handle)
        _GRAPH_PARAMS.attn_params[bs].append(attn_params)
