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
        max_num_seqs=8,  # default for the 6-prompt smoke; spec_graph silently falls back to eager when batch overflows BMM
        enforce_eager=True,
        num_speculative_tokens=1,
        draft_async=True,
        # Grid-sweep sweet spot: extract early hidden at 倒数第2层 (e=1)
        # and predict top-3 candidates (f=3). Cache hit ~95%, ~91 tok/s
        # on PanGu 4×H100+1draft (within ~10% of sync's 100 tok/s).
        # Tuneable via NANO_EARLY / NANO_FAN env.
        # ssd_early_layers convention: K must be < 0; |K| = 倒数第|K|层.
        #   K=-1 → last layer (sync MTP equivalent)
        #   K=-3 → 倒数第3 (matches old K=2 default)
        async_fan_out=int(__import__('os').environ.get('NANO_FAN', 3)),
        ssd_early_layers=int(__import__('os').environ.get('NANO_EARLY', -2)),
        ssd_tree_decode=False,
        profile=__import__('os').environ.get('NANO_PROFILE', '0') == '1',
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
