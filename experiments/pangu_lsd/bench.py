"""PanGu MTP bench — sync (single-process) or async (push-mode Latent SD).

Cache miss falls through to baseline 1-token verify by default. Set
`--fallback` to re-enable the cmd=0 round-trip path.

Usage:
    # Sync MTP K=1 baseline
    rm -f /dev/shm/nanovllm
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -u \\
        experiments/pangu_lsd/bench.py --mode sync

    # Async (4 target + 1 draft GPU). NCCL_PORT must be unique per run.
    rm -f /dev/shm/nanovllm
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_PORT=2510 .venv/bin/python -u \\
        experiments/pangu_lsd/bench.py --mode async --K 1 --F 1 --early -1

    # Profile (small trace → profile_merged.json.gz)
    rm -f /dev/shm/nanovllm target_trace.json draft_trace.json profile_merged.json.gz
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_PORT=2520 .venv/bin/python -u \\
        experiments/pangu_lsd/bench.py --mode async --profile \\
            --num-prompts 1 --max-tokens 8 --K 2 --F 1 --early -3

    # GPQA Diamond accept-rate sanity check (~89%)
    .venv/bin/python -u experiments/pangu_lsd/bench.py --mode sync \\
        --prompts experiments/pangu_lsd/gpqa_diamond.jsonl --max-tokens 256
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
    prompts = [json.loads(l)["prompt"] for l in open(args.prompts)]
    if args.num_prompts > 0:
        prompts = prompts[:args.num_prompts]

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
