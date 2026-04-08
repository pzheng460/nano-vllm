"""Verify EAGLE correctness: greedy output must match non-speculative baseline."""
import sys, json

MODEL = '/mnt/data/peizhen/Qwen2-7B-Instruct'
EAGLE = '/mnt/data/peizhen/EAGLE-Qwen2-7B-Instruct'

prompts = [
    'Janet sells 16 pies. Each pie costs 8 dollars. How much money does she make?',
    'A store has 120 apples. They sell 45 in the morning and 32 in the afternoon. How many are left?',
    'Tom reads 25 pages per day. How many pages does he read in 2 weeks?',
    'A train travels at 60 mph for 3 hours. How far does it travel?',
    'Sarah has 3 times as many books as Tom. Tom has 12 books. How many do they have together?',
]

from nanovllm import LLM, SamplingParams
sp = SamplingParams(temperature=0.0, max_tokens=128)

if sys.argv[1] == 'base':
    llm = LLM(MODEL, enforce_eager=True, tensor_parallel_size=1, max_model_len=4096)
    outputs = llm.generate(prompts, sp)
    result = [o['token_ids'] for o in outputs]
    json.dump(result, open('/tmp/eagle_base.json', 'w'))
    print("Baseline saved")
elif sys.argv[1] == 'eagle':
    llm = LLM(MODEL, draft_model=EAGLE, enforce_eager=True,
              tensor_parallel_size=1, max_model_len=4096, num_speculative_tokens=5)
    outputs = llm.generate(prompts, sp)
    result = [o['token_ids'] for o in outputs]
    json.dump(result, open('/tmp/eagle_spec.json', 'w'))
    print("EAGLE saved")
elif sys.argv[1] == 'compare':
    base = json.load(open('/tmp/eagle_base.json'))
    spec = json.load(open('/tmp/eagle_spec.json'))
    all_match = True
    for i in range(len(prompts)):
        match = base[i] == spec[i]
        status = "MATCH" if match else "MISMATCH"
        print(f"Prompt {i}: {status} (base={len(base[i])} tokens, eagle={len(spec[i])} tokens)")
        if not match:
            all_match = False
            for j in range(min(len(base[i]), len(spec[i]))):
                if base[i][j] != spec[i][j]:
                    print(f"  First divergence at pos {j}: base={base[i][j]}, eagle={spec[i][j]}")
                    break
    print(f"\n{'ALL OUTPUTS MATCH - EAGLE is correct!' if all_match else 'OUTPUTS DIFFER'}")
