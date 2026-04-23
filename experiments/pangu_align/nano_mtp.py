"""Run nano-vllm on PanGu with MTP K=1, dump outputs and accept stats."""
import json
import sys

MODEL = "/mnt/data/weights/openPangu-R-72B-2512"
PROMPTS = [
    "The capital of France is",
    "Write a Python fibonacci function:",
    "List the planets in our solar system:",
]
MAX_TOKENS = 50  # aligned with vLLM MTP test

if __name__ == "__main__":
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.sequence import Sequence
    from time import perf_counter

    llm = LLM(
        MODEL,
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=2048,
        enforce_eager=True,
        num_speculative_tokens=1,  # MTP K=1
    )
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # Use llm.generate to get outputs (this aggregates stats internally)
    out = llm.generate(PROMPTS, sp)

    data = []
    for p, o in zip(PROMPTS, out):
        data.append({
            "prompt": p,
            "output_token_ids": list(o["token_ids"]),
            "output_text": o.get("text", ""),
        })

    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pangu_nano_mtp.json"
    with open(out_path, "w") as f:
        import json as _j
        _j.dump(data, f, indent=2)
    print(f"Saved: {out_path}")
    for i, d in enumerate(data):
        print(f"  q{i}: {len(d['output_token_ids'])} tok; text: {d['output_text'][:100]!r}")
