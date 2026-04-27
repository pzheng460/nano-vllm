#!/bin/bash
# Grid: ssd_early_layers ∈ {0, 3}, async_fan_out ∈ {1, 3, 5}
# 0 = output of last layer (倒数第1)
# 3 = output of 倒数第4 layer (early_layer = N-3-1 = N-4)
set -e
cd /mnt/data/peizhen/nano-vllm

results_file=/tmp/async_grid_results.txt
> "$results_file"

for early in 0 3; do
  for fan in 1 3 5; do
    log=/tmp/async_grid_e${early}_f${fan}.log
    echo "=== early=$early fan=$fan ===" | tee -a "$results_file"
    NCCL_PORT=$((2500 + early * 10 + fan)) \
    CUDA_VISIBLE_DEVICES=4,5,6,7,0 \
    NANO_EARLY=$early NANO_FAN=$fan \
      .venv/bin/python experiments/pangu_align/nano_async.py >"$log" 2>&1 \
      || true
    log=$log
    grep -E "Speculative decoding|Cache hit:|tok/s|tokens in" "$log" | tail -5 | tee -a "$results_file"
    echo "" | tee -a "$results_file"
  done
done

cat "$results_file"
