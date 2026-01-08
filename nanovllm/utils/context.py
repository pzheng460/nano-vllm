from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()
_GRAPH_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

def get_graph_context():
    return _GRAPH_CONTEXT

def set_graph_context(cu_seqlens_q=None, slot_mapping=None, context_lens=None, block_tables=None):
    """Update context tensor references without creating new Context object.

    This is used for GE Graph where we need to use the same tensor objects
    that were used during compilation.
    """
    global _GRAPH_CONTEXT
    if cu_seqlens_q is not None:
        _GRAPH_CONTEXT.cu_seqlens_q = cu_seqlens_q
    if slot_mapping is not None:
        _GRAPH_CONTEXT.slot_mapping = slot_mapping
    if context_lens is not None:
        _GRAPH_CONTEXT.context_lens = context_lens
    if block_tables is not None:
        _GRAPH_CONTEXT.block_tables = block_tables

def reset_graph_context():
    global _GRAPH_CONTEXT
    _GRAPH_CONTEXT.cu_seqlens_q = None
    _GRAPH_CONTEXT.slot_mapping = None
    _GRAPH_CONTEXT.context_lens = None
    _GRAPH_CONTEXT.block_tables = None
