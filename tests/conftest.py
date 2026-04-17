"""Pytest fixtures + skip markers for the nano-vllm harness suite.

Environment variables:
  NANO_QWEN2_PATH   - path to Qwen2-7B-Instruct (default: /mnt/data/peizhen/Qwen2-7B-Instruct)
  NANO_EAGLE_PATH   - path to EAGLE-Qwen2-7B-Instruct (default: /mnt/data/peizhen/EAGLE-Qwen2-7B-Instruct)
  SPEC_BENCH_DIR    - optional path to a cloned Spec-Bench repo (for full eval)
  NANO_SMOKE_LIMIT  - cap number of smoke-test questions (default 4)

Markers:
  gpu               - requires CUDA
  two_gpu           - requires >=2 CUDA devices
  slow              - end-to-end runs that load the full model
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_JSONL = REPO_ROOT / "tests" / "data" / "specbench_smoke.jsonl"


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA")
    config.addinivalue_line("markers", "two_gpu: requires >=2 CUDA devices")
    config.addinivalue_line("markers", "slow: loads full 7B model, seconds-to-minutes")


def _cuda_device_count() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def pytest_collection_modifyitems(config, items):
    n_gpus = _cuda_device_count()
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    skip_two = pytest.mark.skip(reason=f"need 2+ GPUs, found {n_gpus}")
    for item in items:
        if "gpu" in item.keywords and n_gpus < 1:
            item.add_marker(skip_gpu)
        if "two_gpu" in item.keywords and n_gpus < 2:
            item.add_marker(skip_two)


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture(scope="session")
def qwen2_target() -> str:
    path = os.environ.get("QWEN2_TARGET", "/mnt/data/peizhen/Qwen2-7B-Instruct")
    if not os.path.isdir(path):
        pytest.skip(f"Qwen2 target not found at {path}; set QWEN2_TARGET env var")
    return path


@pytest.fixture(scope="session")
def qwen2_eagle() -> str:
    path = os.environ.get("QWEN2_EAGLE", "/mnt/data/peizhen/EAGLE-Qwen2-7B-Instruct")
    if not os.path.isdir(path):
        pytest.skip(f"EAGLE draft not found at {path}; set QWEN2_EAGLE env var")
    return path


@pytest.fixture(scope="session")
def smoke_questions():
    from tests.harness.specbench import load_questions
    qs = load_questions(str(SMOKE_JSONL))
    limit = int(os.environ.get("NANO_SMOKE_LIMIT", "4"))
    return qs[:limit]


@pytest.fixture(scope="session")
def spec_bench_questions():
    """Full Spec-Bench question.jsonl, if a local clone is configured."""
    root = os.environ.get("SPEC_BENCH_DIR")
    if not root:
        pytest.skip("SPEC_BENCH_DIR not set; skipping full Spec-Bench test")
    qpath = Path(root) / "data" / "spec_bench" / "question.jsonl"
    if not qpath.is_file():
        pytest.skip(f"Spec-Bench question.jsonl not at {qpath}")
    from tests.harness.specbench import load_questions
    return load_questions(str(qpath))
