"""Spec-Bench compatible evaluation harness for nano-vllm.

Layout:
- specbench.py  - question.jsonl loader + model-answer jsonl writer
- chat.py       - chat-template helpers (Qwen2 default)
- runner.py     - batch=1 instrumented runner (produces per-step accept_lengths)
- metrics.py    - speedup, mean-accept, greedy-equivalence
- run_spec_bench.py - CLI entrypoint
"""

from tests.harness.specbench import (
    Question,
    Choice,
    ModelAnswer,
    load_questions,
    write_model_answers,
    read_model_answers,
)
from tests.harness.runner import RunnerConfig, run_questions

__all__ = [
    "Question",
    "Choice",
    "ModelAnswer",
    "load_questions",
    "write_model_answers",
    "read_model_answers",
    "RunnerConfig",
    "run_questions",
]
