"""Run vLLM on PanGu with MTP K=1 and measure acceptance."""
import json
import sys

MODEL = "/mnt/data/weights/openPangu-R-72B-2512"
PROMPTS = [
    "The capital of France is",
    "Write a Python fibonacci function:",
    "List the planets in our solar system:",
]
MAX_TOKENS = 50


def main():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=2048,
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 1,
        },
    )
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    outs = llm.generate(PROMPTS, sp)

    data = []
    for o in outs:
        sr = getattr(o, 'spec_result', None)
        metrics = {}
        if hasattr(o, 'metrics'):
            for k in dir(o.metrics):
                v = getattr(o.metrics, k, None)
                if isinstance(v, (int, float, str)) and not k.startswith('_'):
                    metrics[k] = v
        data.append({
            "prompt": o.prompt,
            "prompt_token_ids": list(o.prompt_token_ids),
            "output_token_ids": list(o.outputs[0].token_ids),
            "output_text": o.outputs[0].text,
            "metrics": metrics,
        })

    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pangu_vllm_mtp.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {out_path}")
    for i, d in enumerate(data):
        print(f"  q{i} {len(d['output_token_ids'])} tok: {d['output_text'][:100]!r}")


if __name__ == "__main__":
    main()
