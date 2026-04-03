#!/bin/bash
# PanGu-R-72B MTP benchmark: baseline vs sync MTP vs async SSD
# Usage: bash bench_pangu.sh [baseline|sync|async|all]

MODEL="/mnt/data/weights/openPangu-R-72B-2512"
PYTHON=".venv/bin/python"
MAX_TOKENS=128

PROMPTS='
prompts = [
    "What is the capital of France?",
    "Write a Python fibonacci function:",
    "Explain quantum computing in simple terms:",
    "List the planets in our solar system:",
    "How does photosynthesis work?",
] * 10
'

BENCH='
from time import perf_counter
sp = SamplingParams(temperature=0.001, max_tokens='$MAX_TOKENS')
llm.generate(prompts[:5], sp)
t0 = perf_counter()
outputs = llm.generate(prompts, sp)
elapsed = perf_counter() - t0
total_tokens = sum(len(o["token_ids"]) for o in outputs)
'

run_baseline() {
    echo "=== PanGu baseline TP4 ==="
    CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -c "
from nanovllm import LLM, SamplingParams
$PROMPTS
llm = LLM('$MODEL', enforce_eager=True, tensor_parallel_size=4, max_model_len=4096, disable_mtp=True)
$BENCH
print(f'PanGu baseline TP4: {total_tokens} tokens in {elapsed:.2f}s = {total_tokens/elapsed:.1f} tok/s')
"
}

run_sync() {
    echo "=== PanGu sync MTP K=1 TP4 ==="
    CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -c "
from nanovllm import LLM, SamplingParams
$PROMPTS
llm = LLM('$MODEL', enforce_eager=True, tensor_parallel_size=4, max_model_len=4096, num_speculative_tokens=1)
$BENCH
print(f'PanGu MTP K=1 TP4: {total_tokens} tokens in {elapsed:.2f}s = {total_tokens/elapsed:.1f} tok/s')
"
}

run_async() {
    echo "=== PanGu async SSD K=3 TP4+draft ==="
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 $PYTHON -c "
from nanovllm import LLM, SamplingParams
$PROMPTS
llm = LLM('$MODEL', enforce_eager=True, tensor_parallel_size=4, max_model_len=4096,
           draft_async=True, draft_gpu=4, num_speculative_tokens=1, async_fan_out=3, ssd_early_layers=2)
$BENCH
print(f'PanGu SSD K=1 TP4: {total_tokens} tokens in {elapsed:.2f}s = {total_tokens/elapsed:.1f} tok/s')
"
}

MODE=${1:-all}
case $MODE in
    baseline) run_baseline ;;
    sync)     run_sync ;;
    async)    run_async ;;
    all)      run_baseline; run_sync; run_async ;;
    *)        echo "Usage: bash bench_pangu.sh [baseline|sync|async|all]" ;;
esac
