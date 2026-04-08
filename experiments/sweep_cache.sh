#!/bin/bash
# Sweep early_layers × fan_out for EAGLE SSD cache hit analysis
# Usage: CUDA_VISIBLE_DEVICES=4,5 bash experiments/sweep_cache.sh llama31
# Usage: CUDA_VISIBLE_DEVICES=6,7 bash experiments/sweep_cache.sh qwen2

set -e
MODEL=$1
cd /mnt/data/peizhen/nano-vllm

if [ "$MODEL" = "llama31" ]; then
    TARGET="/mnt/data/peizhen/Llama-3.1-8B-Instruct"
    DRAFT="/mnt/data/peizhen/EAGLE-LLaMA3.1-Instruct-8B"
    PORT_BASE=3400
elif [ "$MODEL" = "qwen2" ]; then
    TARGET="/mnt/data/peizhen/Qwen2-7B-Instruct"
    DRAFT="/mnt/data/peizhen/EAGLE-Qwen2-7B-Instruct"
    PORT_BASE=3600
else
    echo "Usage: $0 {llama31|qwen2}"
    exit 1
fi

echo ""
printf "%5s %3s %8s %12s %8s\n" "early" "F" "hit%" "ch/total" "tok/s"
echo "------------------------------------------"

PORT=$PORT_BASE
for EARLY in 1 2 3 4; do
for FAN in 1 2 3 4 5; do
    PORT=$((PORT + 1))
    sleep 3  # Wait for previous NCCL port release

    RESULT=$(NCCL_PORT=$PORT .venv/bin/python -u -c "
import json
from time import perf_counter
from nanovllm import LLM, SamplingParams

prompts = json.load(open('experiments/gpqa_prompts.json'))[:10]
sp = SamplingParams(temperature=0.0, max_tokens=64)
llm = LLM('$TARGET', draft_model='$DRAFT',
           enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
           num_speculative_tokens=3,
           draft_async=True, draft_gpu=1,
           async_fan_out=$FAN, ssd_early_layers=$EARLY, ssd_tree_decode=True)
llm.generate(prompts[:2], sp, use_tqdm=False)
t0 = perf_counter()
out = llm.generate(prompts, sp, use_tqdm=False)
elapsed = perf_counter() - t0
total = sum(len(o['token_ids']) for o in out)
ch = getattr(llm.model_runner, '_cache_hit', 0)
cm = getattr(llm.model_runner, '_cache_miss', 0)
ct = ch + cm
rate = ch / ct * 100 if ct > 0 else 0
print(f'RESULT {total/elapsed:.1f} {rate:.1f} {ch}/{ct}')
" 2>/dev/null | grep "^RESULT" | tail -1)

    if [ -n "$RESULT" ]; then
        TPS=$(echo $RESULT | awk '{print $2}')
        HIT=$(echo $RESULT | awk '{print $3}')
        CT=$(echo $RESULT | awk '{print $4}')
        printf "%5d %3d %7s%% %12s %7s\n" "$EARLY" "$FAN" "$HIT" "$CT" "$TPS"
    else
        printf "%5d %3d %8s %12s %8s  ERR\n" "$EARLY" "$FAN" "" "" ""
    fi
done
done

echo ""
echo "Done: $MODEL"
