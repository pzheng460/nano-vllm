#!/usr/bin/env bash
# Sweep sync + async early-layer configs for PanGu MTP K=1.
# Run from repo root.
set -euo pipefail

VENV=".venv/bin/python"
SCRIPT="experiments/pangu_lsd/bench.py"
LOG_DIR="experiments/pangu_lsd/logs"
mkdir -p "$LOG_DIR"

echo "=== sync K=1 baseline ==="
rm -f /dev/shm/nanovllm
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    $VENV $SCRIPT --mode sync \
    > "$LOG_DIR/sync_K1.log" 2>&1
grep -E "Speculative|PanGu" "$LOG_DIR/sync_K1.log" | tail -3

PORT=2510
for EARLY in -1 -2 -3; do
    echo "=== async K=1 F=1 early=${EARLY} ==="
    rm -f /dev/shm/nanovllm
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_PORT=$PORT \
        $VENV $SCRIPT --mode async --K 1 --F 1 --early $EARLY \
        > "$LOG_DIR/async_K1_F1_early${EARLY}.log" 2>&1
    grep -E "Speculative|Cache hit:|PanGu" \
        "$LOG_DIR/async_K1_F1_early${EARLY}.log" | tail -5
    PORT=$((PORT + 1))
    sleep 2
done
