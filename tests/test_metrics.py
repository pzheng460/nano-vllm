"""Unit tests for tests/harness/metrics.py (no GPU)."""
from __future__ import annotations

import pytest

from tests.harness.metrics import compute_speed, equivalence, speedup, speedup_table


def _ans(qid, cat, turns, new_tokens, wall, accepts, steps=None):
    steps = steps if steps is not None else [len(a) for a in (accepts if isinstance(accepts[0], list) else [accepts])]
    return {
        "question_id": qid,
        "category": cat,
        "model_id": "x",
        "answer_id": "a",
        "tstamp": 0.0,
        "choices": [{
            "index": 0,
            "turns": turns,
            "decoding_steps": steps,
            "new_tokens": new_tokens,
            "wall_time": wall,
            "accept_lengths": accepts if not isinstance(accepts[0], list) else sum(accepts, []),
        }],
    }


def test_compute_speed_single_answer():
    ans = _ans(1, "math", ["hello"], [10], [1.0], [2, 2, 2, 2, 2])
    r = compute_speed([ans])
    assert r["overall"]["num_answers"] == 1
    assert r["overall"]["total_new_tokens"] if "total_new_tokens" in r["overall"] else r["overall"]["total_new_tokens"] == 10
    assert r["overall"]["tokens_per_second"] == pytest.approx(10.0)
    assert r["overall"]["mean_accept_length"] == pytest.approx(2.0)
    assert "math" in r["by_category"]


def test_compute_speed_multi_category():
    a = _ans(1, "math", ["x"], [5], [10], [0.5], [2] * 5)  # steps=[5] but we pass new_tokens via kwargs
    # Build manually to be explicit:
    ans_math = {
        "question_id": 1, "category": "math", "model_id": "x",
        "answer_id": "a", "tstamp": 0,
        "choices": [{"index": 0, "turns": ["x"], "decoding_steps": [5],
                     "new_tokens": [10], "wall_time": [0.5],
                     "accept_lengths": [2, 2, 2, 2, 2]}],
    }
    ans_writing = {
        "question_id": 2, "category": "writing", "model_id": "x",
        "answer_id": "b", "tstamp": 0,
        "choices": [{"index": 0, "turns": ["y"], "decoding_steps": [10],
                     "new_tokens": [10], "wall_time": [1.0],
                     "accept_lengths": [1] * 10}],
    }
    r = compute_speed([ans_math, ans_writing])
    # Combined tps = 20 new tokens / 1.5s
    assert r["overall"]["tokens_per_second"] == pytest.approx(20.0 / 1.5)
    assert r["by_category"]["math"]["tokens_per_second"] == pytest.approx(20.0)
    assert r["by_category"]["writing"]["tokens_per_second"] == pytest.approx(10.0)


def test_speedup_matches_by_question_id():
    base = {
        "question_id": 1, "category": "math", "model_id": "baseline",
        "answer_id": "a", "tstamp": 0,
        "choices": [{"index": 0, "turns": ["x"], "decoding_steps": [10],
                     "new_tokens": [10], "wall_time": [1.0],
                     "accept_lengths": [1] * 10}],
    }
    spec = {
        "question_id": 1, "category": "math", "model_id": "spec",
        "answer_id": "b", "tstamp": 0,
        "choices": [{"index": 0, "turns": ["x"], "decoding_steps": [3],
                     "new_tokens": [9], "wall_time": [0.3],
                     "accept_lengths": [3, 3, 3]}],
    }
    r = speedup([spec], [base])
    assert r["overall"]["speedup"] == pytest_approx_equal(r["overall"]["spec_tps"] / r["overall"]["base_tps"])
    assert r["overall"]["mean_accept_length"] == pytest.approx(3.0)
    assert r["overall"]["num_answers"] == 1


def pytest_approx_equal(x):
    return pytest.approx(x, rel=1e-6)


def test_speedup_table_is_readable():
    base = [{
        "question_id": 1, "category": "math", "model_id": "baseline",
        "answer_id": "a", "tstamp": 0,
        "choices": [{"index": 0, "turns": ["x"], "decoding_steps": [10],
                     "new_tokens": [10], "wall_time": [1.0],
                     "accept_lengths": [1] * 10}],
    }]
    spec = [{
        "question_id": 1, "category": "math", "model_id": "spec",
        "answer_id": "b", "tstamp": 0,
        "choices": [{"index": 0, "turns": ["x"], "decoding_steps": [4],
                     "new_tokens": [10], "wall_time": [0.5],
                     "accept_lengths": [3, 3, 2, 2]}],
    }]
    text = __import__("tests.harness.metrics", fromlist=["speedup_table"]).speedup_table(spec, base)
    assert "overall" in text
    assert "speedup" in text.lower()


def test_equivalence_exact_match():
    def make(qid, text):
        return {
            "question_id": qid, "category": "math", "model_id": "x",
            "answer_id": str(qid), "tstamp": 0,
            "choices": [{"index": 0, "turns": [text], "decoding_steps": [1],
                         "new_tokens": [1], "wall_time": [0.1],
                         "accept_lengths": [1]}],
        }
    a = [make(1, "hello world"), make(2, "foo bar")]
    b = [make(1, "hello world"), make(2, "foo bar")]
    r = equivalence(a, b)
    assert r["num_turns"] == 2
    assert r["matched_turns"] == 2
    assert r["divergences"] == []


def test_equivalence_detects_divergence():
    def make(qid, text):
        return {
            "question_id": qid, "category": "math", "model_id": "x",
            "answer_id": str(qid), "tstamp": 0,
            "choices": [{"index": 0, "turns": [text], "decoding_steps": [1],
                         "new_tokens": [1], "wall_time": [0.1],
                         "accept_lengths": [1]}],
        }
    a = [make(1, "hello world")]
    b = [make(1, "hello there")]
    r = equivalence(a, b)
    assert r["matched_turns"] == 0
    assert len(r["divergences"]) == 1
    d = r["divergences"][0]
    assert d["question_id"] == 1


def test_equivalence_non_strict_prefix():
    def make(qid, text):
        return {
            "question_id": qid, "category": "math", "model_id": "x",
            "answer_id": str(qid), "tstamp": 0,
            "choices": [{"index": 0, "turns": [text], "decoding_steps": [1],
                         "new_tokens": [1], "wall_time": [0.1],
                         "accept_lengths": [1]}],
        }
    a = [make(1, "hello world, how are you")]
    b = [make(1, "hello world")]
    strict = equivalence(a, b, strict=True)
    loose = equivalence(a, b, strict=False)
    assert strict["matched_turns"] == 0
    assert loose["matched_turns"] == 1
