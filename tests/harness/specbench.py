"""Spec-Bench jsonl I/O.

Matches the schema in https://github.com/hemingkx/Spec-Bench.

Input (question.jsonl):
    {"question_id": int, "category": str, "turns": [str, ...], "reference"?: [...]}

Output (model_answer/<model_id>.jsonl), one line per question:
    {"question_id": ..., "category": ..., "answer_id": str, "model_id": str,
     "choices": [{"index": 0, "turns": [str, ...],
                  "decoding_steps": [int, ...], "new_tokens": [int, ...],
                  "wall_time": [float, ...], "accept_lengths": [int, ...]}],
     "tstamp": float}

`accept_lengths` is a flat list of per-decoding-step accepted token counts
concatenated across all turns. `decoding_steps[t]` is the number of steps for
turn t, so the split can be recovered if needed.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Question:
    question_id: int | str
    category: str
    turns: list[str]
    reference: list[str] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(
            question_id=d["question_id"],
            category=d["category"],
            turns=list(d["turns"]),
            reference=d.get("reference"),
        )


@dataclass
class Choice:
    index: int
    turns: list[str]
    decoding_steps: list[int]
    new_tokens: list[int]
    wall_time: list[float]
    accept_lengths: list[int]

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "turns": self.turns,
            "decoding_steps": self.decoding_steps,
            "new_tokens": self.new_tokens,
            "wall_time": self.wall_time,
            "accept_lengths": self.accept_lengths,
        }


@dataclass
class ModelAnswer:
    question_id: int | str
    category: str
    model_id: str
    choices: list[Choice]
    answer_id: str = ""
    tstamp: float = 0.0

    def __post_init__(self):
        if not self.answer_id:
            self.answer_id = uuid.uuid4().hex
        if not self.tstamp:
            self.tstamp = time.time()

    def as_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "answer_id": self.answer_id,
            "model_id": self.model_id,
            "category": self.category,
            "choices": [c.as_dict() for c in self.choices],
            "tstamp": self.tstamp,
        }


def load_questions(path: str | os.PathLike) -> list[Question]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Question.from_dict(json.loads(line)))
    return out


def write_model_answers(path: str | os.PathLike, answers: list[ModelAnswer]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ans in answers:
            f.write(json.dumps(ans.as_dict(), ensure_ascii=False) + "\n")


def read_model_answers(path: str | os.PathLike) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


