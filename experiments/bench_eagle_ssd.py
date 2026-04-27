"""Benchmark: sync EAGLE vs async EAGLE SSD."""
from nanovllm import LLM, SamplingParams
from time import perf_counter

MODEL = '/mnt/data/peizhen/Qwen2-7B-Instruct'
EAGLE = '/mnt/data/peizhen/EAGLE-Qwen2-7B-Instruct'

prompts = [
    'Janet sells 16 pies. Each pie costs 8 dollars. How much money does she make?',
    'A store has 120 apples. They sell 45 in the morning and 32 in the afternoon. How many are left?',
    'Tom reads 25 pages per day. How many pages does he read in 2 weeks?',
    'A train travels at 60 mph for 3 hours. How far does it travel?',
    'Sarah has 3 times as many books as Tom. Tom has 12 books. How many do they have together?',
] * 10

sp = SamplingParams(temperature=0.0, max_tokens=256)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['sync', 'async'], required=True)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--early-layers', type=int, default=-3)
    parser.add_argument('--fan-out', type=int, default=3)
    parser.add_argument('--tree-decode', action='store_true')
    parser.add_argument('--num-prompts', type=int, default=50)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()

    prompts = prompts[:args.num_prompts]

    if args.mode == 'sync':
        llm = LLM(MODEL, draft_model=EAGLE, enforce_eager=True,
                   tensor_parallel_size=1, max_model_len=4096,
                   num_speculative_tokens=args.K,
                   profile=args.profile)
        name = f'Sync EAGLE K={args.K}'
    else:
        llm = LLM(MODEL, draft_model=EAGLE, enforce_eager=True,
                   tensor_parallel_size=1, max_model_len=4096,
                   num_speculative_tokens=args.K,
                   draft_async=True, draft_gpu=1,
                   async_fan_out=args.fan_out,
                   ssd_early_layers=args.early_layers,
                   ssd_tree_decode=args.tree_decode,
                   profile=args.profile)
        td = ' tree' if args.tree_decode else ' chain'
        name = f'Async EAGLE SSD K={args.K} early={args.early_layers} fan={args.fan_out}{td}'

    t0 = perf_counter()
    outputs = llm.generate(prompts, sp)
    elapsed = perf_counter() - t0
    total_tokens = sum(len(o['token_ids']) for o in outputs)
    print(f'{name}: {total_tokens} tokens in {elapsed:.2f}s = {total_tokens/elapsed:.1f} tok/s')
