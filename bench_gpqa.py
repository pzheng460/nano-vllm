#!/usr/bin/env python3
"""GPQA Diamond benchmark for nanovllm: compare sync MTP, SSD chain, SSD tree, SSD K=1."""
import argparse
import time
from nanovllm import LLM, SamplingParams


def load_gpqa_diamond(num_prompts=0):
    from datasets import load_dataset
    ds = load_dataset("hendrydong/gpqa_diamond", split="test")
    prompts = [item["problem"] for item in ds]
    if num_prompts > 0:
        prompts = prompts[:num_prompts]
    return prompts


def bench(name, llm, prompts, sp):
    # Warmup
    llm.generate(prompts[:3], sp)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    elapsed = time.perf_counter() - t0
    total_tokens = sum(len(o['token_ids']) for o in outputs)
    print(f'{name}: {total_tokens} tokens in {elapsed:.2f}s = {total_tokens/elapsed:.1f} tok/s')
    return total_tokens, elapsed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['sync1', 'sync3', 'async3', 'async3-tree', 'async1'], required=True)
    parser.add_argument('--model', default='/mnt/data/peizhen/MiMo-7B-Base/')
    parser.add_argument('--early-layers', type=int, default=2)
    parser.add_argument('--fan-out', type=int, default=3)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--num-prompts', type=int, default=0, help='0=all (~198)')
    args = parser.parse_args()

    prompts = load_gpqa_diamond(args.num_prompts)
    sp = SamplingParams(temperature=0.001, max_tokens=args.max_tokens)
    print(f"Loaded {len(prompts)} GPQA Diamond prompts, max_tokens={args.max_tokens}")

    if args.mode == 'sync1':
        llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1, max_model_len=4096)
        bench('Sync MTP K=1', llm, prompts, sp)
    elif args.mode == 'sync3':
        llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
                   num_speculative_tokens=args.K)
        bench(f'Sync MTP K={args.K}', llm, prompts, sp)
    elif args.mode == 'async3':
        llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
                   draft_async=True, draft_gpu=1,
                   num_speculative_tokens=args.K,
                   async_fan_out=args.fan_out,
                   ssd_early_layers=args.early_layers)
        bench(f'Async SSD K={args.K} early={args.early_layers} fan={args.fan_out} chain', llm, prompts, sp)
    elif args.mode == 'async3-tree':
        llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
                   draft_async=True, draft_gpu=1,
                   num_speculative_tokens=args.K,
                   async_fan_out=args.fan_out,
                   ssd_early_layers=args.early_layers,
                   ssd_tree_decode=True)
        bench(f'Async SSD K={args.K} early={args.early_layers} fan={args.fan_out} tree', llm, prompts, sp)
    elif args.mode == 'async1':
        llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
                   draft_async=True, draft_gpu=1,
                   num_speculative_tokens=1,
                   async_fan_out=args.fan_out,
                   ssd_early_layers=args.early_layers)
        bench(f'Async SSD K=1 early={args.early_layers} fan={args.fan_out}', llm, prompts, sp)
