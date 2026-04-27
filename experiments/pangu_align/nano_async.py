"""Smoke test: nano-vllm async (SSD-MTP) on PanGu-72B.

Runs target with TP=4 on the first 4 visible GPUs and draft on the 5th.

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7,0 NCCL_PORT=2400 \
    .venv/bin/python experiments/pangu_align/nano_async.py
"""
import sys
import time

MODEL = "/mnt/data/weights/openPangu-R-72B-2512"
PROMPTS = [
    "The capital of France is",
    "Write a Python function that computes the nth Fibonacci number.",
    "List the planets in our solar system.",
    "What are the advantages and disadvantages of TCP vs UDP?",
    "Explain how attention works in a transformer model.",
    "Summarize the theory of relativity in one paragraph.",
]
MAX_TOKENS = 128


def main():
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        MODEL,
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=2048,
        enforce_eager=True,
        num_speculative_tokens=1,
        draft_async=True,
        async_fan_out=3,
        ssd_early_layers=2,
        ssd_tree_decode=False,  # K=1: tree adds build cost without helping hit rate
    )
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    from time import perf_counter
    t0 = perf_counter()
    outs = llm.generate(PROMPTS, sp)
    elapsed = perf_counter() - t0

    total = sum(len(o["token_ids"]) for o in outs)
    print(f"\n[async PanGu] {total} tokens in {elapsed:.2f}s = {total/elapsed:.1f} tok/s")
    for i, o in enumerate(outs):
        print(f"  q{i} ({len(o['token_ids'])} tok): {o.get('text', '')[:90]!r}")


if __name__ == "__main__":
    main()
