#!/bin/bash
# Compare PanGu MTP acceptance rate: nano-vllm vs vLLM
# Usage: CUDA_VISIBLE_DEVICES=4,5,6,7 bash test_pangu_acceptance.sh
set -e
NANO_PY=".venv/bin/python"
VLLM_PY="/mnt/data/z00929669/vllm/.venv/bin/python"
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

echo "=== vLLM MTP K=1 ==="
$VLLM_PY -c "
if __name__ == '__main__':
    import json
    from vllm import LLM, SamplingParams
    prompts = $PROMPTS
    sp = SamplingParams(temperature=0, max_tokens=100)
    llm = LLM('$MODEL', enforce_eager=True, tensor_parallel_size=4, max_model_len=4096,
              trust_remote_code=True, speculative_config={'method': 'mtp', 'num_speculative_tokens': 1})
    out = llm.generate(prompts, sp)
    tids = [list(o.outputs[0].token_ids) for o in out]
    json.dump(tids, open('/tmp/vllm_pangu.json','w'))
    print(f'vLLM done: {len(tids)} seqs, {sum(len(t) for t in tids)} tokens')
"

echo ""
echo "=== nano-vllm MTP K=1 ==="
$NANO_PY -c "
if __name__ == '__main__':
    import json
    from nanovllm import LLM, SamplingParams
    prompts = $PROMPTS
    sp = SamplingParams(temperature=0, max_tokens=100)
    llm = LLM('$MODEL', enforce_eager=True, tensor_parallel_size=4, max_model_len=4096, num_speculative_tokens=1)
    out = llm.generate(prompts, sp)
    tids = [o['token_ids'] for o in out]
    json.dump(tids, open('/tmp/nano_mtp_pangu.json','w'))
    print(f'nano MTP done: {len(tids)} seqs, {sum(len(t) for t in tids)} tokens')
"

echo ""
echo "=== Compare vLLM vs nano-vllm ==="
$NANO_PY -c "
import json
vllm = json.load(open('/tmp/vllm_pangu.json'))
nano = json.load(open('/tmp/nano_mtp_pangu.json'))

full = 0
prefixes = []
for i in range(len(vllm)):
    ml = 0
    for j in range(min(len(vllm[i]), len(nano[i]))):
        if vllm[i][j] == nano[i][j]: ml += 1
        else: break
    prefixes.append(ml)
    if ml == len(vllm[i]) == len(nano[i]):
        full += 1
        print(f'  seq[{i}]: FULL MATCH ({ml} tokens)')
    else:
        print(f'  seq[{i}]: diverge@{ml}/{min(len(vllm[i]),len(nano[i]))}  vllm={vllm[i][ml] if ml<len(vllm[i]) else \"END\"}  nano={nano[i][ml] if ml<len(nano[i]) else \"END\"}')

print()
print(f'Full match: {full}/{len(vllm)}')
print(f'Prefix: min={min(prefixes)} max={max(prefixes)} mean={sum(prefixes)/len(prefixes):.1f}')
total_match = sum(prefixes)
total_tokens = sum(min(len(v),len(n)) for v,n in zip(vllm,nano))
print(f'Token-level match rate: {total_match}/{total_tokens} ({total_match/total_tokens*100:.1f}%)')
"
