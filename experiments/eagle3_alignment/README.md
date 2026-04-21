# EAGLE-3 op-level alignment harness

Tools for comparing nano-vllm's `Eagle3Model` chain K=3 forward against
Spec-Bench's upstream `EaModel3.eagenerate` configured for chain decoding.
Used to locate the residual ~10–18 pp per-position accept-rate gap after the
EAGLE prefill shift fix landed.

## Running

Use **different venvs** for the two sides — Spec-Bench needs its own pinned
environment (`/mnt/data/peizhen/Spec-Bench/.venv-specbench`, transformers 4.48).

```bash
# 1. Spec-Bench side (Spec-Bench venv + Spec-Bench EaModel3)
CUDA_VISIBLE_DEVICES=4 /mnt/data/peizhen/Spec-Bench/.venv-specbench/bin/python \
    experiments/eagle3_alignment/capture_specbench_eagle3.py \
    --model /mnt/data/peizhen/Qwen2.5-7B-Instruct \
    --draft /mnt/data/peizhen/EAGLE3-Qwen2.5-7B-Instruct \
    --prompt "<fixed prompt>" \
    --output /tmp/cap_spec.pt --max-new 40

# 2. nano-vllm side (nano-vllm venv)
CUDA_VISIBLE_DEVICES=5 NCCL_PORT=2389 /mnt/data/peizhen/nano-vllm/.venv/bin/python \
    experiments/eagle3_alignment/capture_nano_eagle3.py \
    --model /mnt/data/peizhen/Qwen2.5-7B-Instruct \
    --draft /mnt/data/peizhen/EAGLE3-Qwen2.5-7B-Instruct \
    --prompt "<same prompt>" \
    --output /tmp/cap_nano.pt --max-new 40

# 3. Diff
.venv/bin/python experiments/eagle3_alignment/diff_eagle3.py \
    --nano /tmp/.../cap_nano.pt --spec /tmp/.../cap_spec.pt
```

## What's captured

Each framework hooks its draft-model sub-modules and records a dict with:
- `fc` — list of `{in, out}` for each `fc` / `combine_hidden_states` call
- `layer` (nano) / `midlayer` (spec) — the `Eagle3DecoderLayer` / `LlamaDecoderLayeremb` output
- `norm` — post-layer RMSNorm output (what `lm_head` sees)
- `meta` — prompt, accept_lens, step counts, generated ids, wall time

## Status

- Simple prompts ("capital of France"): nano-vllm and Spec-Bench produce
  **identical** output tokens with identical 2-round accept pattern after the
  EAGLE prefill shift fix.
- Medium prompts (summarization): mean_accept rates converge (~2.2 on both).
- 120q aggregate: nano 1.95 vs Spec-Bench 2.41 — gap concentrated on specific
  prompt classes, not systematic per-step error. Harness ready to be pointed
  at individual failing prompts.

## Reference: Spec-Bench chain K=3 config

Spec-Bench `topK_genrate` with `depth=3, top_k=1, total_token=4` is the
apples-to-apples config for nano-vllm `num_speculative_tokens=3`:
- 1 initial draft from last_hidden + 3 chain iters = 4 draft tokens
- max accept_len = 4 (same as nano-vllm K=3 + bonus)
