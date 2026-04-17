"""Unit tests for tests/harness/specbench.py (no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

from tests.harness.specbench import (
    Choice,
    ModelAnswer,
    Question,
    load_questions,
    read_model_answers,
    write_model_answers,
)

SMOKE_JSONL = Path(__file__).parent / "data" / "specbench_smoke.jsonl"


def test_load_smoke_questions():
    qs = load_questions(str(SMOKE_JSONL))
    assert len(qs) >= 5
    assert all(isinstance(q, Question) for q in qs)
    assert all(q.turns and isinstance(q.turns, list) for q in qs)
    assert all(isinstance(t, str) and t.strip() for q in qs for t in q.turns)
    # categories cover multiple Spec-Bench tasks
    cats = {q.category for q in qs}
    assert "math" in cats
    assert len(cats) >= 4


def test_mt_bench_multi_turn_parsing():
    qs = load_questions(str(SMOKE_JSONL))
    multi = [q for q in qs if len(q.turns) >= 2]
    assert multi, "expected at least one multi-turn question in smoke set"
    for q in multi:
        assert all(isinstance(t, str) for t in q.turns)


def test_roundtrip_model_answer(tmp_path):
    ans = ModelAnswer(
        question_id=1,
        category="math",
        model_id="qwen2-7b::sync-eagle-k3",
        choices=[Choice(
            index=0,
            turns=["hello world"],
            decoding_steps=[5],
            new_tokens=[20],
            wall_time=[0.5],
            accept_lengths=[4, 4, 4, 4, 4],
        )],
    )
    out = tmp_path / "answers.jsonl"
    write_model_answers(str(out), [ans])
    reloaded = read_model_answers(str(out))
    assert len(reloaded) == 1
    a = reloaded[0]
    assert a["question_id"] == 1
    assert a["model_id"] == "qwen2-7b::sync-eagle-k3"
    assert a["choices"][0]["accept_lengths"] == [4, 4, 4, 4, 4]
    # schema keys Spec-Bench depends on
    required = {"question_id", "category", "answer_id", "model_id", "choices", "tstamp"}
    assert required.issubset(a.keys())
    choice_required = {
        "index", "turns", "decoding_steps", "new_tokens", "wall_time", "accept_lengths"
    }
    assert choice_required.issubset(a["choices"][0].keys())


def test_answer_id_unique(tmp_path):
    ids = set()
    for _ in range(10):
        ans = Question.from_dict({"question_id": 1, "category": "x", "turns": ["hi"]})
        model_answer = ModelAnswer(
            question_id=ans.question_id, category=ans.category,
            model_id="m", choices=[Choice(0, [""], [0], [0], [0.0], [])],
        )
        ids.add(model_answer.answer_id)
    assert len(ids) == 10


def test_jsonl_schema_parseable(tmp_path):
    """Ensure the written jsonl is strict one-json-per-line (what upstream
    Spec-Bench's speed.py expects)."""
    ans = ModelAnswer_like()
    out = tmp_path / "answers.jsonl"
    write_model_answers(str(out), [ans, ans])
    for line in open(out):
        assert line.strip(), "no empty lines"
        json.loads(line)  # must parse


def ModelAnswer_like() -> ModelAnswer:
    return ModelAnswer(
        question_id=42,
        category="qa",
        model_id="m",
        choices=[Choice(0, ["t"], [1], [1], [0.1], [1])],
    )
