"""Run nano-vllm on PanGu with NO MTP (disable_mtp=True) greedy. Match vllm script."""
import json
import sys

MODEL = "/mnt/data/weights/openPangu-R-72B-2512"
PROMPTS = [
    "The capital of France is",
    "Write a Python fibonacci function:",
    "List the planets in our solar system:",
]
MAX_TOKENS = 50

if __name__ == "__main__":
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        MODEL,
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=2048,
        enforce_eager=True,
        disable_mtp=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    out = llm.generate(PROMPTS, sp)

    data = []
    for p, o in zip(PROMPTS, out):
        data.append({
            "prompt": p,
            "output_token_ids": list(o["token_ids"]),
            "output_text": o.get("text", ""),
        })
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pangu_nano_baseline.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {out_path}")
    for i, d in enumerate(data):
        print(f"  q{i} {d['prompt'][:40]!r}: {len(d['output_token_ids']) if False else len(d['output_token_ids']) if False else len(d['output_token_ids'])} tokens")
        print(f"    first 10 ids: {d['output_token_ids'][:10]}")
        print(f"    text: {d['output_text'][:80]!r}")
