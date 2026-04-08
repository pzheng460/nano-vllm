"""Test EAGLE speculative decoding with Qwen2-7B-Instruct."""
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

sp = SamplingParams(temperature=0.0, max_tokens=128)

print("=== Loading Qwen2-7B + EAGLE ===")
llm = LLM(MODEL, draft_model=EAGLE, enforce_eager=True,
          tensor_parallel_size=1, max_model_len=4096,
          num_speculative_tokens=5)

# Warmup
llm.generate(prompts[:2], sp)

print("\n=== Running benchmark ===")
t0 = perf_counter()
outputs = llm.generate(prompts, sp)
elapsed = perf_counter() - t0
total_tokens = sum(len(o['token_ids']) for o in outputs)
print(f'EAGLE: {total_tokens} tokens in {elapsed:.2f}s = {total_tokens/elapsed:.1f} tok/s')

for i, o in enumerate(outputs):
    print(f"\nPrompt {i}: {prompts[i][:60]}...")
    print(f"Output ({len(o['token_ids'])} tokens): {o['text'][:200]}")
