"""Sweep early_layers × fan_out: cache hit rate, acceptance, throughput.

Usage:
  CUDA_VISIBLE_DEVICES=4,5 python experiments/sweep_cache.py --model llama31
  CUDA_VISIBLE_DEVICES=6,7 python experiments/sweep_cache.py --model qwen2
"""
import subprocess, os, re, json, sys, itertools, tempfile

MODELS = {
    "qwen2": ("/mnt/data/peizhen/Qwen2-7B-Instruct", "/mnt/data/peizhen/EAGLE-Qwen2-7B-Instruct"),
    "llama31": ("/mnt/data/peizhen/Llama-3.1-8B-Instruct", "/mnt/data/peizhen/EAGLE-LLaMA3.1-Instruct-8B"),
}

EARLY_LAYERS = [1, 2, 3, 4]
FAN_OUTS = [1, 2, 3, 4, 5]


_port_counter = [0]

def run_one(model_name, early, fan):
    _port_counter[0] += 1
    # Use GPU-based offset to avoid port collision between parallel sweeps
    gpu_base = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    port = 3000 + gpu_base * 100 + _port_counter[0]
    model, draft = MODELS[model_name]
    code = f"""
import json
from time import perf_counter
from nanovllm import LLM, SamplingParams

prompts = json.load(open("experiments/gpqa_prompts.json"))[:10]
sp = SamplingParams(temperature=0.0, max_tokens=64)
llm = LLM("{model}", draft_model="{draft}",
           enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
           num_speculative_tokens=3,
           draft_async=True, draft_gpu=1,
           async_fan_out={fan}, ssd_early_layers={early}, ssd_tree_decode=True)
llm.generate(prompts[:2], sp, use_tqdm=False)
t0 = perf_counter()
out = llm.generate(prompts, sp, use_tqdm=False)
elapsed = perf_counter() - t0
total = sum(len(o["token_ids"]) for o in out)
ch = getattr(llm.model_runner, "_cache_hit", 0)
cm = getattr(llm.model_runner, "_cache_miss", 0)
print(f"RESULT tps={{total/elapsed:.1f}} ch={{ch}} cm={{cm}}")
"""
    # Write code to temp file to avoid quoting issues
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmpfile = f.name

    try:
        r = subprocess.run(
            [sys.executable, "-u", tmpfile],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=300,
            env={**os.environ, "PYTHONPATH": "/mnt/data/peizhen/nano-vllm",
                 "NCCL_PORT": str(port)},
            cwd="/mnt/data/peizhen/nano-vllm",
        )
        out = r.stdout
    except subprocess.TimeoutExpired:
        return None, None, None, "timeout"
    finally:
        os.unlink(tmpfile)

    m = re.search(r"RESULT tps=([\d.]+) ch=(\d+) cm=(\d+)", out)
    if not m:
        err_lines = [l for l in out.strip().split("\n") if "Error" in l or "assert" in l.lower()]
        return None, None, None, (err_lines[-1][:60] if err_lines else "unknown")

    tps = float(m.group(1))
    ch, cm = int(m.group(2)), int(m.group(3))
    total = ch + cm
    rate = ch / total * 100 if total > 0 else 0
    return tps, rate, f"{ch}/{total}", None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    args = parser.parse_args()

    # Ensure prompts file exists
    if not os.path.exists("experiments/gpqa_prompts.json"):
        from datasets import load_dataset
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        json.dump([q["Question"] for q in ds][:50], open("experiments/gpqa_prompts.json", "w"))

    results = []
    print(f"\n{'early':>5} {'F':>3} {'hit%':>8} {'ch/total':>12} {'tok/s':>8}")
    print("-" * 42)

    for early, fan in itertools.product(EARLY_LAYERS, FAN_OUTS):
        tps, rate, ct, err = run_one(args.model, early, fan)
        results.append({"early": early, "fan": fan, "tps": tps, "cache_hit": rate, "detail": ct, "error": err})
        if err:
            print(f"{early:>5} {fan:>3} {'ERR':>8} {'':>12} {'':>8}  {err}", flush=True)
        else:
            print(f"{early:>5} {fan:>3} {rate:>7.1f}% {ct:>12} {tps:>7.1f}", flush=True)

    with open(f"experiments/sweep_cache_{args.model}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/sweep_cache_{args.model}.json")
