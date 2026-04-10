# nano-vllm

Lightweight vLLM-compatible inference engine with speculative decoding support.

## Documentation Rules

When updating project documentation, always keep all three files in sync:
1. `README.md` — project readme
2. `CLAUDE.md` — English technical docs
3. `CLAUDE_zh.md` — Chinese translation of CLAUDE.md

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
   - First step uses unnormed target hidden (from residual stream)
   - Chain: use `final_layernorm` output (normed) as next hidden (matching vLLM)
   - Input: `(embed(token), hidden_state)` → MTP → draft token
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
     - At layer N-X: extract early hidden, async NCCL send to draft
     - Draft receives and pre-computes candidates for Step T+1
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
| 1 | prefill | T→D | prefill hidden + shifted tokens + positions + block_table |
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

### SSD-MTP Algorithm Details

#### Architecture

Two-GPU async speculative decoding. Core idea: extract hidden state from an intermediate layer during target verify forward and send it to Draft GPU early, so Draft pre-computes a candidate token tree for near-zero-latency speculation in the next decode step.

```
GPU 0 (Target)                         GPU 1 (Draft)
┌──────────────────────┐               ┌──────────────────────┐
│  Full target model    │    NCCL      │  MTP Draft model      │
│  - Prefill/Decode     │◄────────────►│  - embed + MTP layer  │
│  - Early hidden extr. │              │  - Tree Cache         │
│  - Verify             │              │  - Async tree build   │
└──────────────────────┘               └──────────────────────┘
```

#### Key Config Parameters

| Parameter | Meaning |
|-----------|---------|
| `draft_async=True` | Enable async two-GPU mode |
| `num_speculative_tokens (K)` | Draft tokens per step |
| `ssd_early_layers` | Extract early hidden at layer `N-X` (default 2) |
| `async_fan_out (F)` | Tree cache branching factor (default 3) |

#### Decode Flow (per step)

**Step 1: Speculation**

Target sends cache lookup request (NCCL cmd=6) with `(seq_id, accepted_len, last_token)`:
- **Cache Hit (~99%)**: Draft returns K tokens from Tree Cache via chain lookup, near-zero latency
- **Cache Miss**: Fallback to local MTP layer on Target GPU to generate K draft tokens

Tree Cache lookup:
```
lookup(sid, pos=0, recovery_token) → d₀
lookup(sid, pos=1, d₀) → d₁
lookup(sid, pos=2, d₁) → d₂
... K chain lookups total
```

**Step 2: Verify**

Target runs prefill-like forward on `[last_token, d₀, d₁, ..., d_{K-1}]`:
- At layer `N - ssd_early_layers`:
  - Extract early hidden state
  - Non-blocking async NCCL send (cmd=5) to Draft with all seqs' early hidden, block tables, positions
  - Target continues remaining layers **in parallel** with Draft's tree construction

**Step 3: Draft Async Tree Cache Build (parallel with Step 2 remaining layers)**

Draft receives early hidden and:
1. **Glue decode**: Run MTP layer on K+1 positions' hidden state to get prenorm hidden
2. **Fork**: Take top-F candidate tokens per position (excluding actual token)
3. **Tree decode**: Embed each candidate, chain MTP layer K steps
4. **Store in Tree Cache**: key = `(seq_id, position, candidate_token)` → value = `K draft tokens`

Each seq stores `(K+1) × F` cache entries.

**Step 4: Greedy Accept**

After verify forward completes, compare K+1 logits positions:
- Check `argmax(logits[i]) == d_i` left-to-right
- Stop at first mismatch; take argmax as recovery token
- If all K match, get bonus token

#### Key Optimizations

1. **Early hidden extraction**: Send hidden X layers before output; Draft and Target last X layers run in parallel
2. **Skip MTP KV update on target**: 99% cache hit makes local MTP KV maintenance unnecessary
3. **Batch NCCL**: All seqs' early hidden packed in one message
4. **Tree Cache pre-computation**: Fan-out=3 means 3 branches per position, near-certain hit next step
5. **Non-blocking async send**: `dist.isend()` doesn't block Target's remaining layer computation

#### Timeline

```
Target:  [==== Verify Forward (layers 0..N-X) ====][send hidden][layers N-X+1..N][Accept]
Draft:                                              [recv hidden][Build Tree Cache =========]
                                                                  ^ parallel with Target last X layers

Next Step:
Target:  [Cache Lookup (fast!)][==== Verify Forward ====] ...
Draft:   [Respond K tokens    ][                        ] ...
```

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
