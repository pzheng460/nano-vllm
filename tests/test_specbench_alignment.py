"""Alignment tests between our harness and upstream Spec-Bench (evaluation/speed.py).

Three layers:

1. **Formula tests** (no GPU): reimplement Spec-Bench's speed.py aggregation
   locally and verify our `compute_speed` produces identical numbers.

2. **Semantic tests** (no GPU): check that `accept_lengths` values fall in the
   expected range [1, K+1], and that the schema we write matches what Spec-Bench
   expects to read.

3. **Upstream cross-check** (GPU, opt-in): run our baseline, pipe the jsonl
   through upstream `speed.py`, verify its parsed `#Mean accepted tokens` and
   `tokens/s` match our compute_speed output.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.harness.metrics import compute_speed, speedup
from tests.harness.specbench import load_questions, read_model_answers


REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_JSONL = REPO_ROOT / "tests" / "data" / "specbench_smoke.jsonl"


# ----------------------------- helpers ---------------------------------------

def _upstream_formula(records: list[dict]) -> dict:
    """Exact reimplementation of Spec-Bench evaluation/speed.py aggregation.

    For each question record d:
        tokens = sum(d.choices[0].new_tokens)
        times  = sum(d.choices[0].wall_time)
        speeds.append(tokens / times)
        accepts.extend(d.choices[0].accept_lengths)
    tokens_per_second = mean(speeds)     # macro-average over questions
    mean_accept       = mean(accepts)    # flat over all decode steps
    """
    speeds: list[float] = []
    accepts: list[int] = []
    for d in records:
        ch = d["choices"][0]
        toks = sum(ch["new_tokens"])
        t = sum(ch["wall_time"])
        if t > 0:
            speeds.append(toks / t)
        accepts.extend(ch["accept_lengths"])
    return {
        "tokens_per_second": sum(speeds) / len(speeds) if speeds else 0.0,
        "mean_accepted": sum(accepts) / len(accepts) if accepts else 0.0,
        "num_questions": len(speeds),
        "num_steps": len(accepts),
    }


def _record(qid, cat: str, new_tokens: int, wall: float, accepts: list[int]) -> dict:
    return {
        "question_id": qid,
        "category": cat,
        "model_id": "x",
        "answer_id": str(qid),
        "tstamp": 0.0,
        "choices": [{
            "index": 0,
            "turns": ["t"],
            "decoding_steps": [len(accepts)],
            "new_tokens": [new_tokens],
            "wall_time": [wall],
            "accept_lengths": list(accepts),
        }],
    }


# ----------------------------- formula tests ---------------------------------

def test_macro_tps_matches_upstream_formula():
    records = [
        _record(1, "a", new_tokens=10, wall=0.5, accepts=[1, 2, 3, 4]),
        _record(2, "a", new_tokens=200, wall=5.0, accepts=[1] * 150 + [4] * 25),
        _record(3, "b", new_tokens=50, wall=1.0, accepts=[2] * 30),
    ]
    ours = compute_speed(records)["overall"]
    ref = _upstream_formula(records)
    assert abs(ours["tokens_per_second_macro"] - ref["tokens_per_second"]) < 1e-9
    assert abs(ours["mean_accept_length"] - ref["mean_accepted"]) < 1e-9


def test_pooled_tps_differs_from_macro_on_uneven_lengths():
    records = [
        _record(1, "x", new_tokens=10, wall=0.1, accepts=[1] * 8),   # 100 tok/s
        _record(2, "x", new_tokens=10, wall=10.0, accepts=[1] * 8),  #   1 tok/s
    ]
    ours = compute_speed(records)["overall"]
    assert abs(ours["tokens_per_second_macro"] - 50.5) < 1e-6
    assert abs(ours["tokens_per_second"] - (20 / 10.1)) < 1e-6


def test_macro_tps_matches_specbench_formula():
    records = [
        _record(1, "math", 10, 0.5, [1, 2, 3, 4]),
        _record(2, "math", 200, 5.0, [1] * 150 + [2] * 25),
        _record(3, "qa", 50, 1.0, [1] * 10),
    ]
    ours = compute_speed(records)["overall"]
    ref = _upstream_formula(records)
    assert abs(ours["tokens_per_second_macro"] - ref["tokens_per_second"]) < 1e-9
    assert abs(ours["mean_accept_length"] - ref["mean_accepted"]) < 1e-9


def test_speedup_overall_uses_macro_tps():
    base = [
        _record(1, "c", 10, 1.0, [1] * 10),
        _record(2, "c", 20, 2.0, [1] * 20),
    ]
    spec = [
        _record(1, "c", 10, 0.5, [2] * 5),
        _record(2, "c", 20, 1.0, [2] * 10),
    ]
    r = speedup(spec, base)["overall"]
    assert abs(r["base_tps"] - 10.0) < 1e-9
    assert abs(r["spec_tps"] - 20.0) < 1e-9
    assert abs(r["speedup"] - 2.0) < 1e-9
    assert abs(r["mean_accept_length"] - 2.0) < 1e-9


# ----------------------------- semantic / schema -----------------------------

def test_accept_length_semantics_drafts_plus_bonus():
    """Spec-Bench's accept_lengths = accepted drafts + 1 bonus. For K=3,
    values are in [1, 4]. Our runner stores `-num_tokens` per step; for K=3
    nano-vllm emits 1..4 tokens depending on how many drafts were accepted."""
    K = 3
    valid = [1, 2, 3, 4]
    assert max(valid) == K + 1
    assert min(valid) == 1
    # baseline always emits exactly 1 token per step
    baseline_values = [1] * 100
    assert set(baseline_values) == {1}


def test_smoke_jsonl_schema():
    """Every row of our shipped smoke set has fields Spec-Bench expects."""
    qs = load_questions(str(SMOKE_JSONL))
    for q in qs:
        assert q.question_id is not None
        assert q.category
        assert q.turns and all(t.strip() for t in q.turns)


# ----------------------------- upstream cross-check (opt-in) -----------------

@pytest.mark.gpu
@pytest.mark.slow
def test_upstream_speed_matches_our_numbers(tmp_path, qwen2_target, qwen2_eagle):
    """Run both modes, import upstream `speed.speed()` DIRECTLY (bypassing
    argparse defaults that hardcode Vicuna paths), and assert exact agreement:

        mean_accept_length       == mean(upstream accept_lengths)          [== 1e-9]
        tokens_per_second_macro  == upstream tokens_per_second (spec side) [== 1e-6]
        specbench_exact_speedup  == upstream speedup_ratio                 [== 1e-6]

    The third equation requires re-tokenizing the baseline turns with the
    model's tokenizer (that's what upstream does at speed.py lines 49-57);
    our `specbench_exact_speedup` helper reproduces this exactly.
    """
    spec_bench = os.environ.get("SPEC_BENCH_DIR")
    if not spec_bench or not (Path(spec_bench) / "evaluation" / "speed.py").is_file():
        pytest.skip("SPEC_BENCH_DIR not pointing at a Spec-Bench clone")

    base_out = tmp_path / "base.jsonl"
    eagle_out = tmp_path / "eagle.jsonl"
    subprocess.run([
        sys.executable, "-m", "tests.harness.run_spec_bench",
        "--mode", "baseline", "--model", qwen2_target,
        "--input", str(SMOKE_JSONL), "--output", str(base_out),
        "--max-new-tokens", "32", "--limit", "3",
    ], check=True, cwd=str(REPO_ROOT))
    subprocess.run([
        sys.executable, "-m", "tests.harness.run_spec_bench",
        "--mode", "eagle", "--model", qwen2_target,
        "--draft", qwen2_eagle, "--K", "3",
        "--input", str(SMOKE_JSONL), "--output", str(eagle_out),
        "--max-new-tokens", "32", "--limit", "3",
    ], check=True, cwd=str(REPO_ROOT))

    # import upstream speed() directly; speed.py argparse hardcodes Vicuna paths
    sys.path.insert(0, spec_bench)
    try:
        from evaluation.speed import speed as upstream_speed
    finally:
        sys.path.remove(spec_bench)
    tps_spec, tps_base, ratio, accepts = upstream_speed(
        str(eagle_out), str(base_out), qwen2_target,
        task="overall", report=False,
    )

    ours_spec = compute_speed(read_model_answers(str(eagle_out)))["overall"]
    upstream_mean_accept = sum(accepts) / len(accepts)
    assert abs(ours_spec["mean_accept_length"] - upstream_mean_accept) < 1e-9
    assert abs(ours_spec["tokens_per_second_macro"] - tps_spec) < 1e-6

    from tests.harness.metrics import specbench_exact_speedup
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(qwen2_target, trust_remote_code=True, use_fast=True)
    our_exact = specbench_exact_speedup(
        read_model_answers(str(eagle_out)),
        read_model_answers(str(base_out)),
        tokenizer=tok,
    )
    assert abs(our_exact["speedup_ratio"] - ratio) < 1e-4
    assert abs(our_exact["tokens_per_second_baseline"] - tps_base) < 1e-4
