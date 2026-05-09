"""PanGu MTP bench — sync (single-process) or async (push-mode Latent SD).

Cache miss falls through to baseline 1-token verify by default. Set
`--fallback` to re-enable the cmd=0 round-trip path.

`--prompts` accepts:
  • a local jsonl path with one of:
      - {"prompt": "..."}             (our format)
      - {"turns": ["...", ...], ...}  (Spec-Bench schema; turns[0] used)
  • an HF dataset spec:  hf:<dataset>[:<split>[:<field>]]
    e.g.  hf:hendrydong/gpqa_diamond            (auto-detects 'problem')
          hf:hendrydong/gpqa_diamond:test
          hf:hendrydong/gpqa_diamond:test:problem
    Common text fields are auto-detected: prompt | problem | question |
    instruction | text | turns[0]. Requires `pip install datasets`.

Usage:
    # Sync MTP K=1 baseline on Spec-Bench
    rm -f /dev/shm/nanovllm
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -u \\
        experiments/pangu_lsd/bench.py --mode sync \\
            --prompts tests/data/specbench_480.jsonl

    # Async (4 target + 1 draft GPU). NCCL_PORT must be unique per run.
    rm -f /dev/shm/nanovllm
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_PORT=2510 .venv/bin/python -u \\
        experiments/pangu_lsd/bench.py --mode async --K 1 --F 1 --early -1

    # HF dataset (e.g. GPQA Diamond)
    .venv/bin/python -u experiments/pangu_lsd/bench.py --mode sync \\
        --prompts hf:hendrydong/gpqa_diamond --max-tokens 256

    # Profile (small trace → profile_merged.json.gz)
    rm -f /dev/shm/nanovllm target_trace.json draft_trace.json profile_merged.json.gz
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_PORT=2520 .venv/bin/python -u \\
        experiments/pangu_lsd/bench.py --mode async --profile \\
            --num-prompts 1 --max-tokens 8 --K 2 --F 1 --early -3
"""
import argparse
import json
import os
from time import perf_counter


DEFAULT_MODEL = "/home/zhengpeizhen/weights/openPangu-R-72B-2512"
DEFAULT_PROMPTS = os.path.join(os.path.dirname(__file__), "prompts.jsonl")
TP = 4              # target tensor parallel (also drives draft GPU index for async)
MAX_MODEL_LEN = 2048
MAX_NUM_SEQS = 1

# Auto-detection order for HF datasets when no explicit field is given.
_HF_TEXT_FIELDS = ("prompt", "problem", "question", "instruction", "text")


def _row_to_text(row, field=None):
    """Pull a prompt string out of a dict-like row from a jsonl line or HF row."""
    if field:
        v = row.get(field)
        if isinstance(v, list) and v:
            return v[0]
        if isinstance(v, str):
            return v
        raise ValueError(f"field '{field}' not found or not a string in row keys={list(row)}")
    # Auto-detect.
    for k in _HF_TEXT_FIELDS:
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    turns = row.get("turns")
    if isinstance(turns, list) and turns:
        return turns[0]
    raise ValueError(f"no recognized text field in row keys={list(row)}; "
                     f"pass an explicit field via --prompts hf:NAME:SPLIT:FIELD")


def load_prompts(spec):
    """spec: either a local jsonl path, or 'hf:DATASET[:SPLIT[:FIELD]]'."""
    if spec.startswith("hf:"):
        parts = spec[3:].split(":")
        ds_name = parts[0]
        split = parts[1] if len(parts) > 1 and parts[1] else "test"
        field = parts[2] if len(parts) > 2 and parts[2] else None
        try:
            from datasets import load_dataset
        except ImportError:
            raise SystemExit(
                "HF dataset spec used but `datasets` not installed. "
                "Install with: pip install datasets"
            )
        ds = load_dataset(ds_name, split=split)
        return [_row_to_text(row, field) for row in ds]
    out = []
    with open(spec) as f:
        for line in f:
            obj = json.loads(line)
            out.append(_row_to_text(obj))
    return out


def main():
    p = argparse.ArgumentParser(description="PanGu MTP bench (sync / async push-mode Latent SD)")
    p.add_argument("--mode", choices=["sync", "async"], required=True,
                   help="sync = single-process TP=4; async = TP=4 target + 1 draft GPU")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"model dir (default: {DEFAULT_MODEL})")
    p.add_argument("--prompts", default=DEFAULT_PROMPTS, help="jsonl with {'prompt': ...}/line")
    p.add_argument("--num-prompts", type=int, default=0, help="0 = all in file")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--K", type=int, default=1, help="num_speculative_tokens (>=1)")
    p.add_argument("--F", type=int, default=1, help="async_fan_out (async only)")
    p.add_argument("--early", type=int, default=-1,
                   help="latent_early_layers (async only, must be <0; -1 = last layer)")
    p.add_argument("--fallback", action="store_true",
                   help="async only: re-enable cmd=0 fallback on cache miss")
    p.add_argument("--profile", action="store_true",
                   help="torch.profiler trace → profile_merged.json.gz in cwd")
    args = p.parse_args()

    if args.mode == "async" and args.early >= 0:
        p.error("--early must be < 0 (negative = layers from the end)")

    from nanovllm import LLM, SamplingParams
    prompts = load_prompts(args.prompts)
    if args.num_prompts > 0:
        prompts = prompts[:args.num_prompts]
    print(f"loaded {len(prompts)} prompts from {args.prompts}", flush=True)

    common = dict(
        tensor_parallel_size=TP,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        num_speculative_tokens=args.K,
        profile=args.profile,
    )

    if args.mode == "sync":
        llm = LLM(args.model, mtp_early_layers=-1, **common)
        tag = f"sync K={args.K}"
    else:
        # Tree-decode is required for K>1; chain mode (K=1) keeps it off.
        llm = LLM(
            args.model,
            draft_async=True,
            draft_gpu=TP,                       # draft sits on the GPU after TP ranks
            async_fan_out=args.F,
            latent_early_layers=args.early,
            latent_tree_decode=(args.K > 1),
            enable_fallback=args.fallback,
            **common,
        )
        tag = f"async K={args.K} F={args.F} early={args.early}" + (" fb" if args.fallback else "")

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    print("warmup", flush=True)
    llm.generate(prompts, sp)
    print("benchmark", flush=True)
    t = perf_counter()
    out = llm.generate(prompts, sp)
    elapsed = perf_counter() - t
    total = sum(len(o["token_ids"]) for o in out)
    print(f"PanGu {tag}: {total} tok in {elapsed:.2f}s = {total/elapsed:.2f} tok/s")


if __name__ == "__main__":
    main()
