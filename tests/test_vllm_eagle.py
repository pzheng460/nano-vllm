"""Compare EAGLE acceptance rate: vLLM baseline."""
import os
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams

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

print("=== Loading vLLM with EAGLE ===")
llm = LLM(MODEL,
          speculative_config={
              "model": EAGLE,
              "num_speculative_tokens": 5,
              "method": "eagle",
          },
          enforce_eager=True,
          tensor_parallel_size=1,
          max_model_len=4096,
          gpu_memory_utilization=0.85,
          disable_log_stats=False)

print("\n=== Running benchmark ===")
outputs = llm.generate(prompts, sp)

total_toks = 0
for i, o in enumerate(outputs):
    text = o.outputs[0].text
    toks = len(o.outputs[0].token_ids)
    total_toks += toks
    print(f"\nPrompt {i}: {prompts[i][:60]}...")
    print(f"Output ({toks} tokens): {text[:200]}")

print(f"\nTotal tokens: {total_toks}")

# Get spec decode metrics from prometheus
from prometheus_client import REGISTRY
for metric in REGISTRY.collect():
    for sample in metric.samples:
        if 'spec' in sample.name or 'draft' in sample.name or 'accept' in sample.name:
            if sample.value != 0:
                print(f"  {sample.name} {sample.labels}: {sample.value}")
