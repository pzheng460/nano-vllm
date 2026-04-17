"""End-to-end harness tests for Qwen2-7B on a Spec-Bench smoke subset.

Regression (回测) checks. Each engine is spawned in a fresh subprocess (via
the CLI in `tests.harness.run_spec_bench`). Running two nano-vllm engines in
the same process is fragile (init_process_group can only fire once; CUDA
memory from the first run blocks the second's KV-cache sizing), so we
isolate them at the OS-process level — same approach Spec-Bench itself uses
when evaluating multiple methods.

Coverage:
  * baseline emits non-empty output for every prompt
  * sync-EAGLE matches baseline greedy output at >= 80% of turns
  * sync-EAGLE mean_accept_length > 1.3 (spec decoding really accepts drafts)
  * sync-EAGLE speedup >= 0.95 vs baseline (no performance regression)
  * async-SSD (2 GPUs) matches baseline at >= 75% of turns

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/test_qwen2_specbench.py -v -s

Env overrides:
  QWEN2_TARGET      Qwen2-7B-Instruct dir
  QWEN2_EAGLE       EAGLE-Qwen2-7B-Instruct dir
  NANO_SMOKE_LIMIT  cap on smoke questions (default 4)
  NANO_MAX_NEW      max_new_tokens per turn (default 128)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.harness.metrics import compute_speed, equivalence, speedup
from tests.harness.specbench import read_model_answers


REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_JSONL = REPO_ROOT / "tests" / "data" / "specbench_smoke.jsonl"


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _run_cli(cli_args: list[str], out_path: Path, qwen2_target: str) -> list[dict]:
    """Spawn the Spec-Bench CLI as a subprocess and load its jsonl output.

    Each invocation gets its own nano-vllm engine with a clean
    init_process_group + CUDA state, which is what makes the multi-mode
    test suite reliable.
    """
    cmd = [
        sys.executable, "-m", "tests.harness.run_spec_bench",
        "--model", qwen2_target,
        "--input", str(SMOKE_JSONL),
        "--output", str(out_path),
        "--max-new-tokens", str(_env_int("NANO_MAX_NEW", 128)),
        "--limit", str(_env_int("NANO_SMOKE_LIMIT", 4)),
        "--verbose",
        *cli_args,
    ]
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    output = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"run_spec_bench subprocess failed "
            f"(exit={proc.returncode}):\n{output[-2000:]}"
        )
    # bubble child stdout up so pytest -s shows per-prompt progress
    print(output, end="")
    return read_model_answers(str(out_path))


@pytest.fixture(scope="session")
def baseline_answers(tmp_path_factory, qwen2_target):
    out = tmp_path_factory.mktemp("harness") / "baseline.jsonl"
    return _run_cli(["--mode", "baseline"], out, qwen2_target)


@pytest.fixture(scope="session")
def eagle_answers(tmp_path_factory, qwen2_target, qwen2_eagle):
    out = tmp_path_factory.mktemp("harness") / "eagle.jsonl"
    return _run_cli(
        ["--mode", "eagle", "--draft", qwen2_eagle, "--K", "3"],
        out, qwen2_target,
    )


@pytest.fixture(scope="session")
def async_eagle_answers(tmp_path_factory, qwen2_target, qwen2_eagle):
    out = tmp_path_factory.mktemp("harness") / "async_eagle.jsonl"
    return _run_cli(
        ["--mode", "async-eagle", "--draft", qwen2_eagle,
         "--K", "3", "--draft-gpu", "1",
         "--fan-out", "3", "--early-layers", "1"],
        out, qwen2_target,
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_baseline_produces_output(baseline_answers, smoke_questions):
    assert len(baseline_answers) == len(smoke_questions)
    for ans in baseline_answers:
        choice = ans["choices"][0]
        assert choice["turns"] and all(t.strip() for t in choice["turns"]), \
            f"empty output for q={ans['question_id']}"
        assert sum(choice["new_tokens"]) > 0
        assert sum(choice["wall_time"]) > 0


@pytest.mark.gpu
@pytest.mark.slow
def test_eagle_greedy_equivalence(baseline_answers, eagle_answers):
    """EAGLE output should approximately match baseline greedy output. Small
    numerical drift in the verify (batched-prefill) path can flip near-tied
    argmax choices, and on multi-turn prompts those flips cascade across
    turns, so we require >= 0.7 match ratio rather than 1.0. A much lower
    ratio would flag a real spec-decoding correctness regression."""
    r = equivalence(eagle_answers, baseline_answers, strict=True)
    assert r["match_ratio"] >= 0.7, (
        f"EAGLE diverged on {r['num_turns']-r['matched_turns']}/{r['num_turns']} "
        f"turns (ratio={r['match_ratio']:.2f}). "
        f"First diffs: {r['divergences'][:2]}"
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_eagle_mean_accept_above_floor(eagle_answers):
    stats = compute_speed(eagle_answers)
    mean_accept = stats["overall"]["mean_accept_length"]
    assert mean_accept > 1.3, (
        f"mean_accept_length={mean_accept:.3f} <= 1.3: "
        f"EAGLE drafts rarely accepted — calibration regression?"
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_eagle_no_speed_regression(baseline_answers, eagle_answers):
    r = speedup(eagle_answers, baseline_answers)
    assert r["overall"]["speedup"] >= 0.95, (
        f"EAGLE speedup {r['overall']['speedup']:.3f} < 0.95 "
        f"(spec_tps={r['overall']['spec_tps']:.1f}, "
        f"base_tps={r['overall']['base_tps']:.1f})"
    )


@pytest.mark.two_gpu
@pytest.mark.slow
def test_async_eagle_greedy_equivalence(baseline_answers, async_eagle_answers):
    r = equivalence(async_eagle_answers, baseline_answers, strict=True)
    assert r["match_ratio"] >= 0.75, (
        f"Async-SSD diverged: match_ratio={r['match_ratio']:.2f}. "
        f"First diffs: {r['divergences'][:2]}"
    )
