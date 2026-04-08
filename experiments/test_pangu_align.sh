#!/bin/bash
# PanGu MTP alignment test: baseline vs MTP on 10 prompts, 100 tokens each
# Usage: CUDA_VISIBLE_DEVICES=4,5,6,7 bash test_pangu_align.sh
set -e
PYTHON=".venv/bin/python"
MODEL="/mnt/data/weights/openPangu-R-72B-2512"

PROMPTS='[
    "What is the capital of France?",
    "Write a Python fibonacci function:",
    "Explain quantum computing in simple terms:",
    "List the planets in our solar system:",
    "How does photosynthesis work?",
    "What is machine learning?",
    "Describe the water cycle:",
    "How do computers store data?",
    "What causes earthquakes?",
    "Explain how vaccines work:"
]'

echo "=== Step 1: Baseline (no MTP) ==="
$PYTHON -c "
if __name__ == '__main__':
    import json
    from nanovllm import LLM, SamplingParams
    prompts = $PROMPTS
    sp = SamplingParams(temperature=0, max_tokens=100)
    llm = LLM('$MODEL', enforce_eager=True, tensor_parallel_size=4, max_model_len=4096, disable_mtp=True)
    out = llm.generate(prompts, sp)
    tids = [o['token_ids'] for o in out]
    json.dump(tids, open('/tmp/pangu_base.json','w'))
    print(f'Baseline done: {len(tids)} seqs, {sum(len(t) for t in tids)} tokens')
"

echo ""
echo "=== Step 2: MTP K=1 ==="
$PYTHON -c "
if __name__ == '__main__':
    import json
    from nanovllm import LLM, SamplingParams
    prompts = $PROMPTS
    sp = SamplingParams(temperature=0, max_tokens=100)
    llm = LLM('$MODEL', enforce_eager=True, tensor_parallel_size=4, max_model_len=4096, num_speculative_tokens=1)
    out = llm.generate(prompts, sp)
    tids = [o['token_ids'] for o in out]
    json.dump(tids, open('/tmp/pangu_mtp.json','w'))
    print(f'MTP done: {len(tids)} seqs, {sum(len(t) for t in tids)} tokens')
"

echo ""
echo "=== Step 3: Compare ==="
$PYTHON -c "
import json
base = json.load(open('/tmp/pangu_base.json'))
mtp = json.load(open('/tmp/pangu_mtp.json'))
full = 0
prefixes = []
for i in range(len(base)):
    ml = 0
    for j in range(min(len(base[i]),len(mtp[i]))):
        if base[i][j]==mtp[i][j]: ml+=1
        else: break
    prefixes.append(ml)
    if ml==len(base[i])==len(mtp[i]):
        full+=1
        print(f'  seq[{i}]: FULL MATCH ({ml} tokens)')
    else:
        print(f'  seq[{i}]: diverge@{ml}/{min(len(base[i]),len(mtp[i]))}')
print(f'')
print(f'Full match: {full}/{len(base)}')
print(f'Prefix match: min={min(prefixes)} max={max(prefixes)} mean={sum(prefixes)/len(prefixes):.1f}')
print(f'Total tokens compared: base={sum(len(t) for t in base)} mtp={sum(len(t) for t in mtp)}')
"
