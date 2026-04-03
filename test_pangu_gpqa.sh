#!/bin/bash
# PanGu sync MTP K=1 on GPQA Diamond
# Usage: CUDA_VISIBLE_DEVICES=4,5,6,7 bash test_pangu_gpqa.sh
set -e
PYTHON=".venv/bin/python"
MODEL="/mnt/data/weights/openPangu-R-72B-2512"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"

echo "=== PanGu MTP K=1 on GPQA Diamond ==="
HF_TOKEN=$HF_TOKEN $PYTHON -c "
if __name__ == '__main__':
    import os, json
    os.environ['HF_TOKEN'] = '$HF_TOKEN'
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams
    from time import perf_counter

    # Load GPQA Diamond
    ds = load_dataset('hendrydong/gpqa_diamond', split='test')
    raw_prompts = [item['problem'] for item in ds]
    print(f'Loaded {len(raw_prompts)} GPQA Diamond prompts')

    # Apply chat template
    tokenizer = AutoTokenizer.from_pretrained('$MODEL', trust_remote_code=True)
    prompts = []
    for p in raw_prompts:
        text = tokenizer.apply_chat_template([{'role':'user','content':p}], tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    sp = SamplingParams(temperature=0, max_tokens=256)
    llm = LLM('$MODEL', enforce_eager=True, tensor_parallel_size=4, max_model_len=4096, num_speculative_tokens=1)

    # Warmup
    llm.generate(prompts[:3], sp)

    # Benchmark
    t0 = perf_counter()
    outputs = llm.generate(prompts, sp)
    elapsed = perf_counter() - t0
    total_tokens = sum(len(o['token_ids']) for o in outputs)

    print(f'')
    print(f'PanGu MTP K=1 GPQA Diamond:')
    print(f'  Prompts: {len(prompts)}')
    print(f'  Tokens: {total_tokens}')
    print(f'  Time: {elapsed:.2f}s')
    print(f'  Throughput: {total_tokens/elapsed:.1f} tok/s')

    # Save results
    result = {
        'model': '$MODEL',
        'num_prompts': len(prompts),
        'total_tokens': total_tokens,
        'elapsed': elapsed,
        'throughput': total_tokens/elapsed,
        'outputs': [{'token_ids': o['token_ids'], 'text': o['text'][:200]} for o in outputs],
    }
    json.dump(result, open('pangu_gpqa_result.json','w'), ensure_ascii=False, indent=2)
    print(f'Results saved to pangu_gpqa_result.json')
"
