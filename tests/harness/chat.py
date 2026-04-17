"""Chat-template application for multi-turn Spec-Bench prompts.

Spec-Bench mt_bench entries have 2 turns per question; other categories have 1.
For each turn we build the full conversation (prior turns + assistant replies
observed so far) and call tokenizer.apply_chat_template to produce the input
token ids for the next assistant reply.
"""
from __future__ import annotations


def build_turn_input_ids(tokenizer, turns: list[str], prior_replies: list[str]) -> list[int]:
    """Build input_ids for the next assistant turn.

    turns[i]          is the user message for turn i.
    prior_replies[i]  is the assistant reply already produced for turn i.

    We expect len(prior_replies) < len(turns); the next assistant turn to
    generate is turns[len(prior_replies)].
    """
    t = len(prior_replies)
    assert t < len(turns), "all turns have replies already"
    messages: list[dict] = []
    for i in range(t):
        messages.append({"role": "user", "content": turns[i]})
        messages.append({"role": "assistant", "content": prior_replies[i]})
    messages.append({"role": "user", "content": turns[t]})

    try:
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        return list(tokenizer(text, add_special_tokens=True)["input_ids"])
    # Some transformers versions return a BatchEncoding dict instead of list[int].
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):  # batched shape [[...]]
        ids = ids[0]
    return list(ids)


def tokenizer_encode_fallback(tokenizer, text: str) -> list[int]:
    out = tokenizer(text, add_special_tokens=True)
    return list(out["input_ids"])


def count_ref_tokens(tokenizer, text: str) -> int:
    """Count tokens in a reference text, minus BOS-like prefix.

    Matches Spec-Bench speed.py which does `tokenizer(text).input_ids[1:]` to
    drop the leading BOS when measuring baseline throughput.
    """
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    return max(len(ids) - 1, 1)
