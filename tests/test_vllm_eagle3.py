"""Compare EAGLE-3 acceptance rate: vLLM vs nano-vllm."""
import os, sys
# V1 engine for eagle3 support

TARGET = '/mnt/data/peizhen/Qwen2.5-7B-Instruct'
EAGLE3 = '/mnt/data/peizhen/EAGLE3-Qwen2.5-7B-Instruct'

prompts = [
    'A train travels at 60 mph for 3 hours. How far does it travel?',
    'Janet sells 16 pies. Each pie costs 8 dollars. How much money does she make?',
    'A store has 120 apples. They sell 45 in the morning and 32 in the afternoon. How many are left?',
    'Tom reads 25 pages per day. How many pages does he read in 2 weeks?',
    'Sarah has 3 times as many books as Tom. Tom has 12 books. How many do they have together?',
]

K = 3
MAX_TOKENS = 256

mode = sys.argv[1] if len(sys.argv) > 1 else 'both'

if mode in ('vllm', 'both'):
    from vllm import LLM, SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    print(f"\n{'='*60}")
    print(f"vLLM EAGLE-3 K={K}")
    print(f"{'='*60}")
    llm = LLM(TARGET,
              speculative_config={
                  'method': 'eagle3',
                  'model': EAGLE3,
                  'num_speculative_tokens': K,
              },
              enforce_eager=True,
              max_model_len=4096,
              gpu_memory_utilization=0.85,
              disable_log_stats=False)

    from time import perf_counter
    t0 = perf_counter()
    outputs = llm.generate(prompts, sp)
    elapsed = perf_counter() - t0
    total_toks = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"vLLM: {total_toks} tokens in {elapsed:.2f}s = {total_toks/elapsed:.1f} tok/s")

    # Get spec decode metrics
    try:
        from prometheus_client import REGISTRY
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                if 'spec_decode' in sample.name and sample.value != 0:
                    print(f"  {sample.name} {sample.labels}: {sample.value}")
    except Exception as e:
        print(f"  Metrics error: {e}")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

if True and mode in ('nano', 'both'):
    pass

if mode in ('nano', 'both'):
    from time import perf_counter
    print(f"\n{'='*60}")
    print("nano-vllm EAGLE-3")
    from nanovllm import LLM as NanoLLM
    from nanovllm import SamplingParams as NanoSP
    nllm = NanoLLM('/mnt/data/peizhen/Qwen2.5-7B-Instruct',
                    draft_model=EAGLE3,
                    enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
                    num_speculative_tokens=K)
    nsp = NanoSP(temperature=0.0, max_tokens=MAX_TOKENS)
    t0 = perf_counter()
    nout = nllm.generate(prompts, nsp)
    elapsed = perf_counter() - t0
    ntok = sum(len(o['token_ids']) for o in nout)
    print(f"nano-vllm: {ntok} tokens in {elapsed:.2f}s = {ntok/elapsed:.1f} tok/s")
