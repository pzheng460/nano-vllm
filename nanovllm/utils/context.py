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
    # For ACL Graph: whether we are capturing (to mark updatable ops)
    capturing: bool = False

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, capturing=False):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, capturing)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()


# NPU-specific: Store handles for graph_task_update (List params cannot be captured by graph)
@dataclass
class GraphParams:
    handles: dict[int, list[Any]] = field(default_factory=dict)
    attn_params: dict[int, list[tuple]] = field(default_factory=dict)

_GRAPH_PARAMS: GraphParams | None = None
def init_graph_params(batch_sizes: list[int]):
    global _GRAPH_PARAMS
    _GRAPH_PARAMS = GraphParams(
        handles={bs: [] for bs in batch_sizes},
        attn_params={bs: [] for bs in batch_sizes},
    )


def get_graph_params() -> GraphParams | None:
    return _GRAPH_PARAMS


def add_graph_handle(bs: int, handle, attn_params: tuple):
    global _GRAPH_PARAMS
    if _GRAPH_PARAMS is not None:
        _GRAPH_PARAMS.handles[bs].append(handle)
        _GRAPH_PARAMS.attn_params[bs].append(attn_params)
