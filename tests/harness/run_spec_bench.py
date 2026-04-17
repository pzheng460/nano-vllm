"""CLI: run nano-vllm through a Spec-Bench question.jsonl and emit a
model_answer jsonl compatible with upstream Spec-Bench (speed.py, equal.py).

Examples
--------

Baseline (no spec) on a smoke subset:
    .venv/bin/python -m tests.harness.run_spec_bench \\
        --mode baseline --model /mnt/data/peizhen/Qwen2-7B-Instruct \\
        --input tests/data/specbench_smoke.jsonl \\
        --output /tmp/smoke_baseline.jsonl

Sync EAGLE K=3:
    .venv/bin/python -m tests.harness.run_spec_bench \\
        --mode eagle --model /mnt/data/peizhen/Qwen2-7B-Instruct \\
        --draft /mnt/data/peizhen/EAGLE-Qwen2-7B-Instruct --K 3 \\
        --input tests/data/specbench_smoke.jsonl \\
        --output /tmp/smoke_eagle.jsonl \\
        --baseline-path /tmp/smoke_baseline.jsonl

Compare against upstream Spec-Bench after cloning:
    python Spec-Bench/evaluation/speed.py \\
        --file-path /tmp/smoke_eagle.jsonl \\
        --base-path  /tmp/smoke_baseline.jsonl \\
        --tokenizer-path /mnt/data/peizhen/Qwen2-7B-Instruct
"""
from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="target HF model dir")
    p.add_argument("--mode", choices=["baseline", "eagle", "async-eagle"], required=True)
    p.add_argument("--draft", default=None, help="EAGLE draft model dir")
    p.add_argument("--K", dest="num_speculative_tokens", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--input", required=True, help="Spec-Bench question.jsonl path")
    p.add_argument("--output", required=True, help="output model_answer jsonl path")
    p.add_argument("--baseline-path", default=None,
                   help="optional baseline jsonl; prints speedup table if given")
    p.add_argument("--categories", nargs="+", default=None,
                   help="filter to these Spec-Bench categories")
    p.add_argument("--limit", type=int, default=None, help="cap on question count")
    p.add_argument("--draft-gpu", type=int, default=1)
    p.add_argument("--fan-out", type=int, default=3)
    p.add_argument("--early-layers", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.mode != "baseline" and args.draft is None:
        p.error("--draft is required when --mode != baseline")
    return args


def main() -> int:
    args = _parse_args()

    # Late imports so --help doesn't initialize torch/CUDA.
    from tests.harness.runner import RunnerConfig, run_questions
    from tests.harness.specbench import (
        load_questions, write_model_answers, read_model_answers,
    )
    from tests.harness.metrics import speedup_table

    questions = load_questions(args.input)
    if args.categories:
        wanted = set(args.categories)
        questions = [q for q in questions if q.category in wanted]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit("No questions after filtering.")

    cfg = RunnerConfig(
        model=args.model,
        draft_model=None if args.mode == "baseline" else args.draft,
        num_speculative_tokens=args.num_speculative_tokens,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        draft_async=(args.mode == "async-eagle"),
        draft_gpu=args.draft_gpu,
        async_fan_out=args.fan_out,
        ssd_early_layers=args.early_layers,
        ssd_tree_decode=True,
    )

    answers = run_questions(cfg, questions, verbose=args.verbose)
    write_model_answers(args.output, answers)
    print(f"[run_spec_bench] wrote {len(answers)} answers -> {args.output}")

    if args.baseline_path:
        base = read_model_answers(args.baseline_path)
        ours = [a.as_dict() for a in answers]
        print(speedup_table(ours, base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
