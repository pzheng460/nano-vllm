"""vLLM MTP run + extract accept rate via engine.get_metrics()."""
import json
import sys


def main():
    from vllm import LLM, SamplingParams

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

    # Dump outputs
    data = []
    for o in outs:
        data.append({
            "prompt": o.prompt,
            "output_token_ids": list(o.outputs[0].token_ids),
            "output_text": o.outputs[0].text,
        })
    with open("/tmp/pangu_vllm_mtp.json", "w") as f:
        json.dump(data, f, indent=2)

    # Extract accept stats from engine
    engine = llm.llm_engine
    print("=== Engine type:", type(engine).__name__)
    # V1 engine stats
    try:
        stats = engine.get_metrics() if hasattr(engine, 'get_metrics') else None
        if stats:
            print("Metrics:")
            for k, v in stats.items():
                if 'spec' in str(k).lower() or 'accept' in str(k).lower():
                    print(f"  {k}: {v}")
    except Exception as e:
        print("get_metrics failed:", e)
    # Try stats via engine_core
    try:
        if hasattr(engine, 'engine_core'):
            ec = engine.engine_core
            print("engine_core type:", type(ec).__name__)
            if hasattr(ec, 'get_metrics'):
                m = ec.get_metrics()
                for k, v in m.items():
                    if 'spec' in str(k).lower() or 'accept' in str(k).lower():
                        print(f"  ec:{k}: {v}")
    except Exception as e:
        print("engine_core get_metrics failed:", e)
    # Try prometheus client
    try:
        from prometheus_client import REGISTRY
        for c in REGISTRY.collect():
            if 'spec_decode' in c.name or 'accept' in c.name:
                for m in c.samples:
                    print(f"  {m.name}: {m.value}")
    except Exception as e:
        print("Prometheus REGISTRY not available:", e)


if __name__ == "__main__":
    main()
