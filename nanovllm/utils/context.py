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

def get_context():
    global _CONTEXT
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    # _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)
    _CONTEXT.is_prefill = is_prefill
    _CONTEXT.cu_seqlens_q = cu_seqlens_q
    _CONTEXT.cu_seqlens_k = cu_seqlens_k
    _CONTEXT.max_seqlen_q = max_seqlen_q
    _CONTEXT.max_seqlen_k = max_seqlen_k
    _CONTEXT.slot_mapping = slot_mapping
    _CONTEXT.context_lens = context_lens
    _CONTEXT.block_tables = block_tables

def reset_context():
    global _CONTEXT
    # _CONTEXT = Context()
    _CONTEXT.is_prefill = False
    _CONTEXT.cu_seqlens_q = None
    _CONTEXT.cu_seqlens_k = None
    _CONTEXT.max_seqlen_q = 0
    _CONTEXT.max_seqlen_k = 0
    _CONTEXT.slot_mapping = None
    _CONTEXT.context_lens = None
    _CONTEXT.block_tables = None
