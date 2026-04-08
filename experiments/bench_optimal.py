"""Benchmark optimal configurations for EAGLE SSD and MTP SSD.

Best configurations found through extensive tuning:
- EAGLE SSD: tree decode, F=5, early_layers=2, K=3, push-based speculation
- MTP SSD: tree decode, F=3, early_layers=2, K=3, batched tree build

Results (H100, 10 prompts, max_tokens=256):
+---------------------------+----------+--------+--------+--------+--------+
| Config                    | tok/s    | Accept | pos0   | pos1   | pos2   |
+---------------------------+----------+--------+--------+--------+--------+
| Qwen2.5 Sync EAGLE K=3   | 255.8    | 29.9%  | 55.2%  | 24.0%  | 10.6%  |
| Qwen2.5 Async EAGLE SSD  | 407.6    | 23.1%  | 44.8%  | 17.9%  | 6.7%   |
| Speedup                   | +59%     |        |        |        |        |
+---------------------------+----------+--------+--------+--------+--------+
| Qwen2 Sync EAGLE K=3     | 208.3    | 34.5%  | 58.2%  | 30.4%  | 14.8%  |
| Qwen2 Async EAGLE SSD    | 264.5    | 24.7%  | 47.8%  | 20.4%  | 6.1%   |
| Speedup                   | +27%     |        |        |        |        |
+---------------------------+----------+--------+--------+--------+--------+
| MiMo Sync MTP K=3        | 188.5    | 36.3%  | 86.0%  | 18.4%  | 4.4%   |
| MiMo Async MTP SSD       | 317.8    | 29.1%  | 75.4%  | 10.3%  | 1.6%   |
| Speedup                   | +69%     |        |        |        |        |
+---------------------------+----------+--------+--------+--------+--------+
"""
from nanovllm import LLM, SamplingParams
from time import perf_counter

PROMPTS = [
    "Janet sells 16 pies. Each pie costs 8 dollars. How much money does she make?",
    "A store has 120 apples. They sell 45 in the morning and 32 in the afternoon. How many are left?",
    "Tom reads 25 pages per day. How many pages does he read in 2 weeks?",
    "A train travels at 60 mph for 3 hours. How far does it travel?",
    "Sarah has 3 times as many books as Tom. Tom has 12 books. How many do they have together?",
    "A rectangle is 10m long and 5m wide. What is its area?",
    "If a car goes 100km in 2 hours, what is its speed?",
    "Emma has 24 cookies. She gives 1/3 to her friend. How many does she have left?",
    "A book costs 15 dollars. You buy 4 books. How much change from 100?",
    "There are 365 days in a year. How many days in 3 years?",
]

SP = SamplingParams(temperature=0.0, max_tokens=256)

# ─── Optimal configurations ───────────────────────────────────────────────────

CONFIGS = {
    # Qwen2-7B + EAGLE
    "qwen2_sync": dict(
        model="/mnt/data/peizhen/Qwen2-7B-Instruct",
        draft_model="/mnt/data/peizhen/EAGLE-Qwen2-7B-Instruct",
        enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
        num_speculative_tokens=3,
    ),
    "qwen2_async": dict(
        model="/mnt/data/peizhen/Qwen2-7B-Instruct",
        draft_model="/mnt/data/peizhen/EAGLE-Qwen2-7B-Instruct",
        enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
        num_speculative_tokens=3,
        draft_async=True, draft_gpu=1,
        async_fan_out=5, ssd_early_layers=2, ssd_tree_decode=True,
    ),
    # Qwen2.5-7B + EAGLE
    "qwen25_sync": dict(
        model="/mnt/data/peizhen/Qwen2.5-7B-Instruct",
        draft_model="/mnt/data/peizhen/EAGLE-Qwen2.5-7B-Instruct",
        enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
        num_speculative_tokens=3,
    ),
    "qwen25_async": dict(
        model="/mnt/data/peizhen/Qwen2.5-7B-Instruct",
        draft_model="/mnt/data/peizhen/EAGLE-Qwen2.5-7B-Instruct",
        enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
        num_speculative_tokens=3,
        draft_async=True, draft_gpu=1,
        async_fan_out=5, ssd_early_layers=2, ssd_tree_decode=True,
    ),
    # MiMo-7B MTP
    "mimo_sync": dict(
        model="/mnt/data/peizhen/MiMo-7B-Base",
        enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
        num_speculative_tokens=3,
    ),
    "mimo_async": dict(
        model="/mnt/data/peizhen/MiMo-7B-Base",
        enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
        num_speculative_tokens=3,
        draft_async=True, draft_gpu=1,
        async_fan_out=3, ssd_early_layers=2, ssd_tree_decode=True,
    ),
}


def bench(name, config, prompts=None, warmup=True):
    prompts = prompts or PROMPTS
    llm = LLM(**config)
    if warmup:
        llm.generate(prompts[:2], SP, use_tqdm=False)
    t0 = perf_counter()
    outputs = llm.generate(prompts, SP, use_tqdm=False)
    elapsed = perf_counter() - t0
    total = sum(len(o["token_ids"]) for o in outputs)
    tps = total / elapsed
    print(f"{name}: {total} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")
    return tps


if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Benchmark optimal EAGLE/MTP SSD configs")
    parser.add_argument("--config", choices=list(CONFIGS.keys()), required=True,
                        help="Configuration to benchmark")
    parser.add_argument("--prompts", type=int, default=10, help="Number of prompts (repeats base 10)")
    args = parser.parse_args()

    config = CONFIGS[args.config]

    # Check model exists
    if not os.path.isdir(config["model"]):
        print(f"Model not found: {config['model']}")
        exit(1)
    if "draft_model" in config and not os.path.isdir(config["draft_model"]):
        print(f"Draft model not found: {config['draft_model']}")
        exit(1)

    n_repeat = max(1, args.prompts // 10)
    prompts = PROMPTS * n_repeat

    bench(args.config, config, prompts)
