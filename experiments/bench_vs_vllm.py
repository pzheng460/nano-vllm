"""nano-vllm vs vLLM throughput comparison.

Usage:
  # nano-vllm (single GPU, CUDA graph)
  CUDA_VISIBLE_DEVICES=0 python experiments/bench_vs_vllm.py --engine nano --bs 10

  # nano-vllm + EAGLE sync (single GPU, eager)
  CUDA_VISIBLE_DEVICES=0 python experiments/bench_vs_vllm.py --engine nano --mode sync --bs 10

  # nano-vllm + SSD async (two GPUs, eager)
  CUDA_VISIBLE_DEVICES=0,1 python experiments/bench_vs_vllm.py --engine nano --mode ssd --bs 10

  # vLLM baseline
  CUDA_VISIBLE_DEVICES=0 python experiments/bench_vs_vllm.py --engine vllm --bs 10

  # vLLM + EAGLE
  CUDA_VISIBLE_DEVICES=0 python experiments/bench_vs_vllm.py --engine vllm --mode eagle --bs 10
"""
import argparse
import json
import os
from time import perf_counter

MODEL = "/mnt/data/peizhen/Llama-3.1-8B-Instruct"
DRAFT = "/mnt/data/peizhen/EAGLE-LLaMA3.1-Instruct-8B"


def load_prompts(bs):
    prompts = json.load(open("experiments/gpqa_prompts.json"))
    if bs > len(prompts):
        prompts = (prompts * (bs // len(prompts) + 1))[:bs]
    else:
        prompts = prompts[:bs]
    return prompts


def bench_nano(prompts, mode, max_tokens=256):
    from nanovllm import LLM, SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    kwargs = dict(model=MODEL, tensor_parallel_size=1, max_model_len=4096)

    if mode == "baseline":
        kwargs["enforce_eager"] = False
    elif mode == "sync":
        kwargs["enforce_eager"] = True
        kwargs["draft_model"] = DRAFT
        kwargs["num_speculative_tokens"] = 3
    elif mode == "ssd":
        kwargs["enforce_eager"] = True
        kwargs["draft_model"] = DRAFT
        kwargs["num_speculative_tokens"] = 3
        kwargs["draft_async"] = True
        kwargs["draft_gpu"] = 1
        kwargs["async_fan_out"] = 5
        kwargs["ssd_early_layers"] = 2
        kwargs["ssd_tree_decode"] = True

    llm = LLM(**kwargs)
    llm.generate(prompts[:min(2, len(prompts))], sp, use_tqdm=False)  # warmup

    t0 = perf_counter()
    out = llm.generate(prompts, sp, use_tqdm=False)
    elapsed = perf_counter() - t0
    total = sum(len(o["token_ids"]) for o in out)
    return total, elapsed


def bench_vllm(prompts, mode, max_tokens=256):
    from vllm import LLM, SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    kwargs = dict(model=MODEL, tensor_parallel_size=1, max_model_len=4096)

    if mode == "eagle":
        kwargs["speculative_config"] = {
            "model": DRAFT,
            "num_speculative_tokens": 3,
            "method": "eagle",
        }

    llm = LLM(**kwargs)
    llm.generate(prompts[:min(2, len(prompts))], sp)  # warmup

    t0 = perf_counter()
    out = llm.generate(prompts, sp)
    elapsed = perf_counter() - t0
    total = sum(len(o.outputs[0].token_ids) for o in out)
    return total, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["nano", "vllm"], required=True)
    parser.add_argument("--mode", choices=["baseline", "sync", "ssd", "eagle"], default="baseline")
    parser.add_argument("--bs", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    prompts = load_prompts(args.bs)

    if args.engine == "nano":
        total, elapsed = bench_nano(prompts, args.mode, args.max_tokens)
    else:
        total, elapsed = bench_vllm(prompts, args.mode, args.max_tokens)

    tps = total / elapsed
    print(f"{args.engine:5s} {args.mode:8s} bs={args.bs:3d}: {total:5d} tok  {elapsed:6.2f}s  {tps:7.1f} tok/s")


if __name__ == "__main__":
    main()
