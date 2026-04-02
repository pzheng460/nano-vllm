# nano-vllm

Lightweight vLLM-compatible inference engine with speculative decoding support.

## Project Structure

- `nanovllm/engine/llm_engine.py` — Main engine, orchestrates prefill/decode/spec-decode loop
- `nanovllm/engine/model_runner.py` — Target model runner, handles sync MTP and SSD decode
- `nanovllm/engine/draft_runner.py` — Async draft runner (separate GPU), tree cache + MTP
- `nanovllm/engine/scheduler.py` — Request scheduling and block management
- `nanovllm/models/mimo.py` — MiMo model with MTP layer
- `nanovllm/models/pangu.py` — PanGu model with MTP + MoE
- `nanovllm/config.py` — All configuration parameters

## Speculative Decoding Algorithms

### 1. Sync MTP (Multi-Token Prediction)

Single-GPU speculative decoding using the model's built-in MTP layers.

**Flow (per decode step):**
1. **Draft**: Run MTP layer K times recursively on target GPU
   - Input: `(embed(token), hidden_state)` → MTP → draft token
   - Chain: use `final_layernorm` output (normed) as next hidden (matching vLLM)
   - First step uses unnormed target hidden (MiMo has `hidden_layernorm`)
2. **Verify**: Target model processes `[last_token, d0, d1, ..., d_{K-1}]` in one prefill-like forward
3. **Accept**: Greedy match — accept consecutive matching tokens, append bonus token
4. **MTP KV update**: After accept, update MTP KV cache with accepted tokens (shifted)

**Key details:**
- MiMo MTP expects **unnormed** hidden from target (has its own `hidden_layernorm`)
- PanGu MTP expects **normed** hidden (no `hidden_layernorm`)
- Position 0 masking: `input_embeds[positions == 0] = 0` (matching vLLM)
- MTP KV cache uses shifted token IDs: position i gets `embed(token_{i+1})`
- `num_speculative_tokens` auto-set to `num_nextn_predict_layers` unless user overrides

### 2. SSD-MTP (Speculative Streaming Decoding)

Two-GPU async speculative decoding. Target model on GPU 0, MTP draft on GPU 1.

**Architecture:**
```
GPU 0 (Target)                    GPU 1 (Draft)
┌─────────────────┐              ┌──────────────────┐
│ Prefill/Decode   │   NCCL      │ MTP Draft Model   │
│ Full Model       │◄──────────►│ embed + MTP layer │
│ MTP KV cache     │             │ MTP KV cache      │
│ (for cache miss) │             │ Tree Cache        │
└─────────────────┘              └──────────────────┘
```

**SSD Decode Flow:**

```
Step T:
  1. Speculation: cache hit → NCCL lookup (fast) | miss → local MTP on target
  2. Verify: target model forward on [last_token, d0..d_{K-1}]
     - At layer N-X: extract early hidden, batch NCCL send to draft
     - Draft receives and pre-computes candidates for Step T+1 (async)
  3. Accept: greedy match, update last_hidden
  4. Draft stores tree cache entries: (seq_id, position, candidate) → draft_token
```

**Tree Cache (on draft GPU):**
- **Populate**: During verify, draft receives early hidden at K+1 positions.
  For each position: `norm(early_hidden) → lm_head → topF candidates → embed → MTP → draft_token`.
  Stores `(K+1)*F` entries per seq.
- **Lookup**: Chain K lookups: `(sid, 0, recovery_token) → d0`, `(sid, 1, d0) → d1`, `(sid, 2, d1) → d2`
- **Hit rate**: ~99% with `ssd_early_layers=2`, `async_fan_out=3`

**NCCL Communication Protocol (cmd IDs):**
| cmd | Name | Direction | Data |
|-----|------|-----------|------|
| 0 | speculate | T→D | hidden + block_table → D returns K draft tokens |
| 1 | prefill | T→D | shifted_tokens + hidden + positions + block_table |
| 2 | exit | T→D | shutdown draft loop |
| 3 | cleanup | T→D | seq_id (finished sequence) |
| 4 | cache_update | T→D | (seq_id, acc_len, token) + K draft tokens |
| 5 | early_speculate | T→D | batched: num_seqs + per-seq meta + hidden + bt + pos |
| 6 | cache_lookup | T→D | (seq_id, acc_len, token) → D returns K tokens |

**Key optimizations:**
- Batch NCCL: all seqs' early hidden sent in one message (not per-seq)
- Skip target MTP KV update in SSD mode (99% cache hit makes it unnecessary)
- Draft KV cache capped to `max_num_seqs * max_blocks_per_seq` (not fill GPU)
- `ssd_early_layers` config: extract early hidden at layer `N-X` (default 2)

### 3. EAGLE Speculative Decoding

External draft model (`nanovllm/models/eagle.py`). Uses a lightweight single-layer transformer as draft.

## MiMo MTP Layer Details

```python
def forward(input_embeds, hidden_states, positions):
    input_embeds[positions == 0] = 0          # position 0 masking
    input_embeds = token_layernorm(input_embeds)
    hidden = hidden_layernorm(hidden_states)   # expects UNNORMED hidden
    hidden = input_proj(cat([hidden, input_embeds]))  # [hidden, embed] order
    # Transformer block: pre-norm attn + post-norm MLP
    hidden = attn(input_layernorm(hidden)) + hidden
    hidden = mlp(post_attn_layernorm(hidden)) + hidden
    return final_layernorm(hidden), hidden     # (normed, prenorm)
```

- **Logits**: `F.linear(normed_output, lm_head.weight)` — uses main model's lm_head (no shared_head)
- **Chain**: recursive MTP uses `normed` output as next `hidden_states`
- **First step**: uses target model's unnormed hidden (from residual stream)

## Build & Test

```bash
# Sync MTP K=1 (single GPU)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_ssd.py --mode sync1

# Sync MTP K=3 (single GPU)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_ssd.py --mode sync3

# Async SSD K=3 (two GPUs)
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python bench_ssd.py --mode async3 --early-layers 2 --fan-out 3
```

## Performance (MiMo-7B-Base, H100, 50 prompts, max_tokens=256)

| Mode | tok/s | pos0 accept | pos1 | pos2 |
|------|-------|-------------|------|------|
| Sync MTP K=1 | 381.9 | 86.5% | - | - |
| Sync MTP K=3 | 257.6 | 84.0% | 22.7% | 5.9% |
| Async SSD K=3 | 406.5 | 72.0% | 8.2% | 0.2% |
