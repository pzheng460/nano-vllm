# PanGu Latent SD bench scripts

Async two-GPU Latent Speculative Decoding for PanGu-72B MTP. Cache miss steps
fall through to baseline 1-token verify by default; set `--fallback` (or
`enable_fallback=True` in the config) to re-enable the cmd=0 round-trip path.

## Files

| File | Purpose |
|------|---------|
| `prompts.jsonl` | 10 chat-style prompts |
| `gpqa_diamond.jsonl` | 50 GPQA Diamond prompts (accept-rate sanity check) |
| `bench.py` | Unified bench: `--mode {sync, async}`, `--profile`, `--fallback` |
| `analyze_trace.py` | Parse `profile_merged.json.gz`, report compute/NCCL overlap |
| `run_grid.sh` | Sweep sync + async early ∈ {-1, -2, -3} |

Hard-coded inside `bench.py`: `TP=4`, `MAX_NUM_SEQS=1`, `MAX_MODEL_LEN=2048`,
draft sits on GPU index `TP` (= 4). Edit the constants at the top of the file
if your hardware differs.

## Quick start

```bash
# Sync MTP K=1 baseline (TP=4, single process)
rm -f /dev/shm/nanovllm
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -u \
    experiments/pangu_lsd/bench.py --mode sync

# Async (4 target GPUs + 1 draft GPU)
rm -f /dev/shm/nanovllm
CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_PORT=2510 .venv/bin/python -u \
    experiments/pangu_lsd/bench.py --mode async --K 1 --F 1 --early -1

# Profile (1 prompt × 8 tok = small clean trace)
rm -f /dev/shm/nanovllm target_trace.json draft_trace.json profile_merged.json.gz
CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_PORT=2520 .venv/bin/python -u \
    experiments/pangu_lsd/bench.py --mode async --profile \
        --num-prompts 1 --max-tokens 8 --K 2 --F 1 --early -3
.venv/bin/python experiments/pangu_lsd/analyze_trace.py
# drag profile_merged.json.gz into https://ui.perfetto.dev/

# GPQA Diamond accept-rate sanity check (~89% on PanGu sync MTP K=1)
.venv/bin/python -u experiments/pangu_lsd/bench.py --mode sync \
    --prompts experiments/pangu_lsd/gpqa_diamond.jsonl --max-tokens 256

# Sweep
./experiments/pangu_lsd/run_grid.sh
```

## CLI

| arg | default | meaning |
|-----|---------|---------|
| `--mode` | required | `sync` or `async` |
| `--model PATH` | bench.py constant | model dir |
| `--prompts FILE` | `prompts.jsonl` | jsonl with `{"prompt": "..."}` per line |
| `--num-prompts N` | 0 (= all) | take first N prompts |
| `--max-tokens N` | 64 | per-prompt generation cap |
| `--K N` | 1 | num_speculative_tokens (≥1) |
| `--F N` | 1 | (async) async_fan_out (tree branching) |
| `--early N` | -1 | (async) latent_early_layers (must be <0; -1 = last layer) |
| `--fallback` | off | (async) enable cmd=0 fallback on cache miss |
| `--profile` | off | torch.profiler → `profile_merged.json.gz` |

`latent_tree_decode` is auto-on for K>1, off for K=1. Edit `TP`/`MAX_NUM_SEQS`/
draft GPU index at the top of `bench.py` for hardware migration.

## Reference numbers (PanGu-R-72B-2512, H100×4 target + 1 draft)

| Config | Prompts | tok/s | Accept | Cache hit |
|--------|---------|-------|--------|-----------|
| sync K=1 | 10 chat | 16.88 | 83.2% | — |
| sync K=1 | **GPQA Diamond ×50** | **17.74** | **89.3%** | — |
| async K=1 F=1 early=-1 | 10 chat | 17.61 | 83.2% | 100% |
| async K=1 F=1 early=-2 | 10 chat | 16.34 | 68.2% | 83% |

**Takeaways**
- `early=-1` is best at batch=1 (cache 100% hit, MTP offloaded to draft GPU)
- `early=-2/-3` hurts at batch=1: cache miss steps revert to baseline
  (no fallback). Use `--fallback` to compare against the cmd=0 path
- Larger batch (edit `MAX_NUM_SEQS` constant) is where async should pay off
  — target verify time amortizes draft latency
