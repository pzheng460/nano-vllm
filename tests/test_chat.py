"""Unit tests for tests/harness/chat.py (no GPU required, just a tokenizer)."""
from __future__ import annotations

import os
import pytest

from tests.harness.chat import build_turn_input_ids


@pytest.fixture(scope="module")
def tokenizer():
    path = os.environ.get("NANO_QWEN2_TARGET", "/mnt/data/peizhen/Qwen2-7B-Instruct")
    if not os.path.isdir(path):
        pytest.skip(f"Qwen2 tokenizer not available at {path}")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)


def test_single_turn_prompt_roundtrips(tokenizer):
    ids = build_turn_input_ids(tokenizer, ["Hello, how are you?"], [])
    assert isinstance(ids, list)
    assert ids and all(isinstance(x, int) for x in ids)
    text = tokenizer.decode(ids)
    assert "how are you" in text.lower()


def test_multi_turn_appends_prior_reply(tokenizer):
    turns = ["What is 2+2?", "Now add 5 to that."]
    prior = ["2+2 equals 4."]
    ids = build_turn_input_ids(tokenizer, turns, prior)
    text = tokenizer.decode(ids)
    # Prior assistant reply should appear in the built context
    assert "4" in text
    # Current user turn should be present
    assert "add 5" in text.lower()


def test_encoding_differs_across_turns(tokenizer):
    turns = ["first", "second"]
    ids0 = build_turn_input_ids(tokenizer, turns, [])
    ids1 = build_turn_input_ids(tokenizer, turns, ["first reply"])
    assert ids0 != ids1
    assert len(ids1) > len(ids0)


def test_error_when_all_turns_consumed(tokenizer):
    with pytest.raises(AssertionError):
        build_turn_input_ids(tokenizer, ["only turn"], ["reply"])
