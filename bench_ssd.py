"""Benchmark sync MTP vs async SSD-MTP."""
from nanovllm import LLM, SamplingParams
from time import perf_counter

prompts = [
    'Janet sells 16 pies. Each pie costs 8 dollars. How much money does she make?',
    'A store has 120 apples. They sell 45 in the morning and 32 in the afternoon. How many are left?',
    'Tom reads 25 pages per day. How many pages does he read in 2 weeks?',
    'A train travels at 60 mph for 3 hours. How far does it travel?',
    'Sarah has 3 times as many books as Tom. Tom has 12 books. How many do they have together?',
] * 10

sp = SamplingParams(temperature=0.001, max_tokens=256)


def bench(name, llm):
    # Warmup
    llm.generate(prompts[:5], sp)
    t0 = perf_counter()
    outputs = llm.generate(prompts, sp)
    elapsed = perf_counter() - t0
    total_tokens = sum(len(o['token_ids']) for o in outputs)
    print(f'{name}: {total_tokens} tokens in {elapsed:.2f}s = {total_tokens/elapsed:.1f} tok/s')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['sync1', 'sync3', 'async3'], required=True)
    parser.add_argument('--model', default='/mnt/data/peizhen/MiMo-7B-Base/')
    parser.add_argument('--early-layers', type=int, default=2,
                        help='SSD: extract early hidden at layer N-X (default: 2)')
    parser.add_argument('--fan-out', type=int, default=3)
    parser.add_argument('--K', type=int, default=3)
    args = parser.parse_args()

    if args.mode == 'sync1':
        llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1, max_model_len=4096)
        bench('Sync MTP K=1', llm)
    elif args.mode == 'sync3':
        llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
                   num_speculative_tokens=args.K)
        bench(f'Sync MTP K={args.K}', llm)
    elif args.mode == 'async3':
        llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
                   draft_async=True, draft_gpu=1,
                   num_speculative_tokens=args.K,
                   async_fan_out=args.fan_out,
                   ssd_early_layers=args.early_layers)
        bench(f'Async SSD K={args.K} early={args.early_layers} fan={args.fan_out}', llm)
