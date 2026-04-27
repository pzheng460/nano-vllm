"""Benchmark nano-vllm vs vLLM on PanGu MTP K=1 (decode tok/s).

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7 python bench_both.py nano
  CUDA_VISIBLE_DEVICES=4,5,6,7 python bench_both.py vllm
"""
import json
import sys
import time

MODEL = "/mnt/data/weights/openPangu-R-72B-2512"
PROMPTS = [
    "The capital of France is",
    "Write a Python function that computes the nth Fibonacci number efficiently.",
    "Explain how attention works in transformers.",
    "What is the chemical formula for glucose and how is it synthesized?",
    "Summarize the theory of relativity in one paragraph.",
    "Describe the difference between TCP and UDP.",
]
MAX_TOKENS = 128


def run_nano(out_path):
    from nanovllm import LLM, SamplingParams
    llm = LLM(
        MODEL,
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=2048,
        enforce_eager=True,
        num_speculative_tokens=1,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # Warm-up single short run so weights / kernels are hot.
    _ = llm.generate(["hi"], SamplingParams_from(sp, MAX_TOKENS=8))

    from time import perf_counter
    t0 = perf_counter()
    outs = llm.generate(PROMPTS, sp)
    elapsed = perf_counter() - t0

    total_toks = sum(len(o["token_ids"]) for o in outs)
    data = {
        "framework": "nano-vllm",
        "elapsed_sec": elapsed,
        "total_output_tokens": total_toks,
        "tok_per_sec": total_toks / elapsed,
        "per_prompt": [
            {"prompt": p, "n_tokens": len(o["token_ids"]),
             "text_prefix": o.get("text", "")[:80]}
            for p, o in zip(PROMPTS, outs)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))


def SamplingParams_from(sp, MAX_TOKENS):
    from nanovllm import SamplingParams
    return SamplingParams(temperature=sp.temperature, max_tokens=MAX_TOKENS)


def run_vllm(out_path):
    import os
    os.environ['VLLM_USE_V1'] = '1'
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=2048,
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
    )
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # warmup
    llm.generate(["Hello"], SamplingParams(temperature=0.0, max_tokens=8))

    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp)
    elapsed = time.perf_counter() - t0

    total_toks = sum(len(o.outputs[0].token_ids) for o in outs)
    data = {
        "framework": "vllm",
        "elapsed_sec": elapsed,
        "total_output_tokens": total_toks,
        "tok_per_sec": total_toks / elapsed,
        "per_prompt": [
            {
                "prompt": o.prompt,
                "n_tokens": len(o.outputs[0].token_ids),
                "text_prefix": o.outputs[0].text[:100],
            }
            for o in outs
        ],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    framework = sys.argv[1] if len(sys.argv) > 1 else "nano"
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/pangu_{framework}_bench.json"
    PROMPTS_list = PROMPTS  # global

    if framework == "nano":
        run_nano(out_path)
    elif framework == "vllm":
        run_vllm(out_path)
    else:
        sys.exit(f"unknown framework: {framework}")
