"""vLLM MTP K=1 with detailed accept rate via spec_output_dump."""
import json
import os
import sys


def main():
    os.environ["VLLM_LOG_LEVEL"] = "DEBUG"
    from vllm import LLM, SamplingParams
    from vllm.v1.metrics.stats import SchedulerStats

    MODEL = "/mnt/data/weights/openPangu-R-72B-2512"
    PROMPTS = [
        "The capital of France is",
        "Write a Python fibonacci function:",
        "List the planets in our solar system:",
    ]

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=2048,
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
        disable_log_stats=False,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=50)
    outs = llm.generate(PROMPTS, sp)

    data = []
    for o in outs:
        d = {
            "prompt": o.prompt,
            "output_token_ids": list(o.outputs[0].token_ids),
            "output_text": o.outputs[0].text,
        }
        # try to extract spec-related metrics
        if hasattr(o, 'metrics'):
            m = o.metrics
            for attr in ('scheduled_time', 'model_forward_time', 'first_token_time'):
                v = getattr(m, attr, None)
                if v is not None:
                    d[attr] = v
        data.append(d)

    with open("/tmp/pangu_vllm_mtp_v2.json", "w") as f:
        json.dump(data, f, indent=2)

    # Try to get engine-level spec stats
    try:
        engine = llm.llm_engine if hasattr(llm, 'llm_engine') else llm
        print("Engine class:", type(engine).__name__)
    except Exception as e:
        print("engine check err:", e)

    for d in data:
        print(f"  {len(d['output_token_ids'])} tok: {d['output_text'][:80]!r}")


if __name__ == "__main__":
    main()
