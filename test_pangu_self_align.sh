#!/bin/bash
# Test: does nano MTP match nano baseline? (self-consistency check)
# Usage: CUDA_VISIBLE_DEVICES=4,5,6,7 bash test_pangu_self_align.sh
set -e
PYTHON=".venv/bin/python"
MODEL="/mnt/data/weights/openPangu-R-72B-2512"
RESULT="pangu_align_result.json"

echo "=== nano baseline (no MTP) ==="
$PYTHON -c "
if __name__ == '__main__':
    import json
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained('$MODEL', trust_remote_code=True)
    raw_prompts = [
        'What is the capital of France?',
        'Write a Python fibonacci function',
        'Explain quantum computing in simple terms',
        'List the first 10 prime numbers',
        'How does photosynthesis work?',
        'What is machine learning?',
        'Describe the water cycle',
        'How do computers store data?',
        'What causes earthquakes?',
        'Explain how vaccines work',
        'What is the speed of light?',
        'How does gravity work?',
        'Write a bubble sort algorithm',
        'What is the meaning of life?',
        'Explain the theory of relativity',
        'How do airplanes fly?',
        'What is DNA?',
        'How does the internet work?',
        'What causes rain?',
        'Explain how a battery works',
    ]
    prompts = []
    for p in raw_prompts:
        text = tokenizer.apply_chat_template([{'role':'user','content':p}], tokenize=False, add_generation_prompt=True)
        prompts.append(text)
    sp = SamplingParams(temperature=0, max_tokens=256)
    llm = LLM('$MODEL', enforce_eager=True, tensor_parallel_size=4, max_model_len=4096, disable_mtp=True)
    out = llm.generate(prompts, sp)
    data = {
        'prompts': raw_prompts,
        'base_tids': [o['token_ids'] for o in out],
        'base_texts': [o['text'] for o in out],
    }
    json.dump(data, open('/mnt/data/peizhen/nano-vllm/pangu_base_tmp.json','w'), ensure_ascii=False)
    print(f'done: {len(out)} seqs, {sum(len(o[\"token_ids\"]) for o in out)} tokens')
"

echo ""
echo "=== nano MTP K=1 ==="
$PYTHON -c "
if __name__ == '__main__':
    import json
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained('$MODEL', trust_remote_code=True)
    raw_prompts = [
        'What is the capital of France?',
        'Write a Python fibonacci function',
        'Explain quantum computing in simple terms',
        'List the first 10 prime numbers',
        'How does photosynthesis work?',
        'What is machine learning?',
        'Describe the water cycle',
        'How do computers store data?',
        'What causes earthquakes?',
        'Explain how vaccines work',
        'What is the speed of light?',
        'How does gravity work?',
        'Write a bubble sort algorithm',
        'What is the meaning of life?',
        'Explain the theory of relativity',
        'How do airplanes fly?',
        'What is DNA?',
        'How does the internet work?',
        'What causes rain?',
        'Explain how a battery works',
    ]
    prompts = []
    for p in raw_prompts:
        text = tokenizer.apply_chat_template([{'role':'user','content':p}], tokenize=False, add_generation_prompt=True)
        prompts.append(text)
    sp = SamplingParams(temperature=0, max_tokens=256)
    llm = LLM('$MODEL', enforce_eager=True, tensor_parallel_size=4, max_model_len=4096, num_speculative_tokens=1)
    out = llm.generate(prompts, sp)
    base_data = json.load(open('/mnt/data/peizhen/nano-vllm/pangu_base_tmp.json'))
    base_data['mtp_tids'] = [o['token_ids'] for o in out]
    base_data['mtp_texts'] = [o['text'] for o in out]
    json.dump(base_data, open('$RESULT','w'), ensure_ascii=False, indent=2)
    print(f'done: {len(out)} seqs, {sum(len(o[\"token_ids\"]) for o in out)} tokens')
"

echo ""
echo "=========================================="
echo "=== Compare ==="
echo "=========================================="
$PYTHON -c "
import json
data = json.load(open('$RESULT'))
prompts = data['prompts']
base = data['base_tids']
mtp = data['mtp_tids']
base_texts = data['base_texts']
mtp_texts = data['mtp_texts']

full = 0
total_tokens = 0
for i in range(len(base)):
    minlen = min(len(base[i]),len(mtp[i]))
    total_tokens += minlen
    ml = 0
    for j in range(minlen):
        if base[i][j]==mtp[i][j]: ml+=1
        else: break
    match = ml==len(base[i])==len(mtp[i])
    if match: full+=1
    status = 'MATCH' if match else f'DIVERGE@{ml}/{minlen}'
    print(f'[{i:2d}] {status:20s} | {prompts[i]}')
    print(f'     base: {base_texts[i][:120]}')
    print(f'     mtp:  {mtp_texts[i][:120]}')
    print()

print(f'Full match: {full}/{len(base)} | Total tokens: {total_tokens}')
print(f'Result saved to: $RESULT')
if full == len(base):
    print('PASS')
else:
    print('FAIL')
"
