#!/usr/bin/env bash
# Smoke / regression test on Spec-Bench prompts. Runs sync MTP K=1
# baseline + async push-mode (no fallback), reports tok/s + accept rate.
#
# Customize via env vars (defaults work for the dev box):
#   MODEL          model dir (REQUIRED if you didn't edit bench.py default)
#   GPUS           CUDA_VISIBLE_DEVICES (default: 0,1,2,3,4 = TP=4 + draft)
#   NCCL_PORT      NCCL rendezvous port (default: 2530)
#   PROMPTS        Local jsonl path or 'hf:DATASET[:SPLIT[:FIELD]]'.
#                  Local options:
#                    tests/data/specbench_smoke.jsonl  — 10 prompts (sanity)
#                    tests/data/specbench_120.jsonl    — 120-prompt subset
#                    tests/data/specbench_480.jsonl    — full upstream set (default)
#                  HF examples (needs `pip install -e '.[hf]'`):
#                    hf:hendrydong/gpqa_diamond
#                    hf:Idavidrein/gpqa:test:Question
#   MAX_TOKENS     per-prompt cap (default: 256)
#   EARLY          latent_early_layers (must be <0; default: -4)
#   K              num_speculative_tokens (default: 1)
#   F              async_fan_out (default: 1)
#   NUM_PROMPTS    cap on prompts read (default: 0 = all)
#   MAX_NUM_SEQS   concurrent sequences (default: 1)
#
# Usage:
#   cd nano-vllm
#   MODEL=/path/to/your/pangu-mtp ./experiments/pangu_lsd/test.sh
#
# Examples:
#   # Quick sanity run on 10 prompts
#   MODEL=/path PROMPTS=tests/data/specbench_smoke.jsonl ./experiments/pangu_lsd/test.sh
#
#   # K=2, fan-out=4, early=-3
#   MODEL=/path EARLY=-3 K=2 F=4 ./experiments/pangu_lsd/test.sh
#
#   # Cap at first 50 prompts of the full set
#   MODEL=/path NUM_PROMPTS=50 ./experiments/pangu_lsd/test.sh
set -euo pipefail

# ---------- env defaults ----------
MODEL="${MODEL:-/home/zhengpeizhen/weights/openPangu-R-72B-2512}"
GPUS="${GPUS:-0,1,2,3,4}"
NCCL_PORT="${NCCL_PORT:-2530}"
PROMPTS="${PROMPTS:-tests/data/specbench_480.jsonl}"
MAX_TOKENS="${MAX_TOKENS:-256}"
EARLY="${EARLY:--4}"
K="${K:-1}"
F="${F:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

# ---------- python interpreter ----------
# Prefer:
#   1. $PYTHON env var (explicit override)
#   2. ./.venv/bin/python  (uv venv / python -m venv layout)
#   3. `python` from PATH  (conda activate / system)
if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="$(command -v python || true)"
fi
[ -n "$PY" ] && "$PY" -c "import nanovllm" 2>/dev/null || {
    echo "ERROR: no working Python with nanovllm found."
    echo "  Install one of:"
    echo "    uv venv && uv pip install -e '.[cuda,hf]'"
    echo "    conda create -n nanovllm python=3.12 && conda activate nanovllm && pip install -e '.[cuda,hf]'"
    echo "  Or set PYTHON=/path/to/python."
    exit 1
}
[ -d "$MODEL" ] || { echo "ERROR: MODEL not a directory: $MODEL"; exit 1; }
[ -f "$PROMPTS" ] || { echo "ERROR: PROMPTS file missing: $PROMPTS"; exit 1; }

rm -f /dev/shm/nanovllm

NUM_LINES=$(wc -l < "$PROMPTS")
echo "================================================================"
echo "  Python      : $PY"
echo "  Model       : $MODEL"
echo "  GPUs        : $GPUS"
echo "  Prompts     : $PROMPTS  ($NUM_LINES total$( [ "$NUM_PROMPTS" -gt 0 ] && echo ", first $NUM_PROMPTS used" ))"
echo "  max_tokens  : $MAX_TOKENS"
echo "  max_num_seqs: $MAX_NUM_SEQS"
echo "  K F early   : $K  $F  $EARLY"
echo "  Fallback    : OFF"
echo "================================================================"

# Tee both runs into log files so we can extract tok/s afterwards.
LOG_DIR="${LOG_DIR:-experiments/pangu_lsd/logs}"
mkdir -p "$LOG_DIR"
SYNC_LOG="$LOG_DIR/sync_K1.log"
ASYNC_LOG="$LOG_DIR/async_K${K}_F${F}_early${EARLY}.log"

# ---------- 1. Sync MTP K=1 baseline ----------
echo
echo "=== [1/2] sync MTP K=1 baseline ==="
CUDA_VISIBLE_DEVICES="$GPUS" $PY -u experiments/pangu_lsd/bench.py \
    --mode sync \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --num-prompts "$NUM_PROMPTS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-tokens "$MAX_TOKENS" 2>&1 | tee "$SYNC_LOG"

# ---------- 2. Async push-mode (no fallback) ----------
rm -f /dev/shm/nanovllm
echo
echo "=== [2/2] async K=$K F=$F early=$EARLY (no fallback) ==="
CUDA_VISIBLE_DEVICES="$GPUS" NCCL_PORT="$NCCL_PORT" $PY -u experiments/pangu_lsd/bench.py \
    --mode async \
    --K "$K" --F "$F" --early "$EARLY" \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --num-prompts "$NUM_PROMPTS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-tokens "$MAX_TOKENS" 2>&1 | tee "$ASYNC_LOG"

# ---------- Comparison ----------
# Pull "<total> tok in <secs>s = <tok/s> tok/s" from each log.
grep_perf() { grep -oE "[0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$1" | tail -1; }
grep_accept() { grep -oE "draft tokens accepted \([0-9.]+%" "$1" | tail -1 | grep -oE "[0-9.]+%"; }
grep_cache_hit() { grep -oE "Cache hit: [0-9]+/[0-9]+ \([0-9.]+%\)" "$1" | tail -1; }

SYNC_PERF=$(grep_perf "$SYNC_LOG")
ASYNC_PERF=$(grep_perf "$ASYNC_LOG")
SYNC_ACC=$(grep_accept "$SYNC_LOG")
ASYNC_ACC=$(grep_accept "$ASYNC_LOG")
ASYNC_HIT=$(grep_cache_hit "$ASYNC_LOG")

# Speedup (async tok/s ÷ sync tok/s)
SYNC_TOKS=$(echo "$SYNC_PERF" | grep -oE "= [0-9.]+ tok/s" | grep -oE "[0-9.]+")
ASYNC_TOKS=$(echo "$ASYNC_PERF" | grep -oE "= [0-9.]+ tok/s" | grep -oE "[0-9.]+")
SPEEDUP=$(awk -v a="$ASYNC_TOKS" -v s="$SYNC_TOKS" 'BEGIN{ if (s+0>0) printf "%.3f", a/s; else print "n/a" }')

echo
echo "================================================================"
echo "  Comparison (logs in $LOG_DIR/)"
echo "----------------------------------------------------------------"
printf "  %-32s %s\n" "sync MTP K=1"            "$SYNC_PERF   accept=$SYNC_ACC"
printf "  %-32s %s\n" "async K=$K F=$F early=$EARLY"  "$ASYNC_PERF   accept=$ASYNC_ACC   $ASYNC_HIT"
echo "----------------------------------------------------------------"
echo "  speedup (async / sync)        : ${SPEEDUP}x"
echo "================================================================"
