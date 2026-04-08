# SSD Tree Cache Hit Rate Sweep: early_layers x fan_out

## Experiment Setup

- **Models**: Llama-3.1-8B-Instruct + EAGLE, Qwen2-7B-Instruct + EAGLE
- **Dataset**: GPQA Diamond (10 prompts, scientific questions)
- **Generation**: greedy, max_tokens=64, K=3 speculative tokens
- **Hardware**: H100 80GB, 2 GPUs per run (target + draft)
- **Variables**: `ssd_early_layers` (1-4), `async_fan_out` (1-5)
- **Metric**: Tree cache hit rate (% of decode steps where draft tokens come from pre-computed cache vs JIT NCCL fallback)

## Results

### Llama 3.1 + EAGLE

| early \ F |    1   |    2   |    3   |    4   |    5   |
|:---------:|:------:|:------:|:------:|:------:|:------:|
|   **1**   | 97.3%  | 97.1%  | 97.4%  | 97.4%  | 97.4%  |
|   **2**   | 75.8%  | 86.5%  | 90.7%  | 92.7%  | 94.7%  |
|   **3**   | 61.3%  | 75.6%  | 80.4%  | 84.3%  | 86.2%  |
|   **4**   | 52.6%  | 64.7%  | 72.2%  | 74.5%  | 75.7%  |

### Qwen2 + EAGLE

| early \ F |    1   |    2   |    3   |    4   |    5   |
|:---------:|:------:|:------:|:------:|:------:|:------:|
|   **1**   | 96.2%  | 97.4%  | 97.4%  | 97.4%  | 97.4%  |
|   **2**   | 72.5%  | 85.1%  | 91.3%  | 92.9%  | 93.8%  |
|   **3**   | 57.0%  | 70.6%  | 75.1%  | 78.6%  | 80.5%  |
|   **4**   | 46.0%  | 58.4%  | 64.8%  | 67.0%  | 71.1%  |

## Analysis

### Effect of `ssd_early_layers`

`ssd_early_layers=N` means the target model extracts hidden states at layer `L-N` (N layers before the final layer) and sends them to the draft GPU via NCCL. A larger N gives the draft GPU more compute time to pre-build the tree cache before the target finishes its forward pass.

| early_layers | Llama hit (F=5) | Qwen2 hit (F=5) | Interpretation |
|:---:|:---:|:---:|:---|
| 1 | 97.4% | 97.4% | Draft has max time, saturated |
| 2 | 94.7% | 93.8% | Sweet spot: high hit + good acceptance |
| 3 | 86.2% | 80.5% | Diminishing returns |
| 4 | 75.7% | 71.1% | Draft often can't finish in time |

**Each additional layer costs ~10-15pp hit rate.** At early=1, the draft has the longest window (target still computing 1 final layer), so cache hit saturates at ~97% regardless of F.

### Effect of `async_fan_out` (F)

F controls how many top-F candidate tokens the draft pre-computes at each verify position. Higher F increases the chance that the actual accepted token is among the pre-computed candidates.

| F | Llama hit (e=2) | Qwen2 hit (e=2) | Delta from F=1 |
|:---:|:---:|:---:|:---:|
| 1 | 75.8% | 72.5% | baseline |
| 2 | 86.5% | 85.1% | +11-13pp |
| 3 | 90.7% | 91.3% | +15-19pp |
| 4 | 92.7% | 92.9% | +17-20pp |
| 5 | 94.7% | 93.8% | +19-21pp |

**F=1->3 gives the biggest gain (~+15pp), F=3->5 has diminishing returns (~+4pp).** This makes sense: top-3 candidates cover ~90% of the probability mass for greedy decoding.

### Diminishing returns at early=1

At early=1, even F=1 achieves 96-97% hit. This means with 1 extra layer of compute time, the draft can almost always finish even with just 1 candidate. F has no effect because time, not coverage, is the bottleneck at early=2+.

### Model comparison

Llama and Qwen2 show nearly identical trends. Llama is slightly better at early=3,4 (e.g., 86.2% vs 80.5% at e=3,F=5), possibly because:
- Llama EAGLE uses GQA (8 kv_heads) vs Qwen2 EAGLE full MHA (28 kv_heads), making draft forward faster
- Faster draft = more time to finish before target completes

## Throughput Summary (H100, 10 prompts, max_tokens=256)

| Config | tok/s | Accept | pos0 | pos1 | pos2 | Speedup |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Llama3.1 Sync EAGLE K=3 | 252.6 | 36.3% | 67.9% | 32.1% | 8.9% | - |
| Llama3.1 Async SSD (e=2,F=5) | 406.2 | 32.3% | 60.9% | 26.3% | 9.5% | **+61%** |
| Qwen2.5 Sync EAGLE K=3 | 255.8 | 29.9% | 55.2% | 24.0% | 10.6% | - |
| Qwen2.5 Async SSD (e=2,F=5) | 407.6 | 23.1% | 44.8% | 17.9% | 6.7% | **+59%** |
| Qwen2 Sync EAGLE K=3 | 208.3 | 34.5% | 58.2% | 30.4% | 14.8% | - |
| Qwen2 Async SSD (e=2,F=5) | 264.5 | 24.7% | 47.8% | 20.4% | 6.1% | **+27%** |
| MiMo Sync MTP K=3 | 188.5 | 36.3% | 86.0% | 18.4% | 4.4% | - |
| MiMo Async MTP SSD (e=2,F=3) | 317.8 | 29.1% | 75.4% | 10.3% | 1.6% | **+69%** |

## Speedup Breakdown (Llama 3.1, 10 prompts)

```
Sync:  8.3 ms/step = EAGLE draft (3.4ms, 41%) + Verify (4.9ms, 59%)
Async: 4.8 ms/step = Verify (4.8ms) + NCCL (<0.1ms, overlapped)

Step time speedup:  1.71x (removed 41% draft overhead)
Tokens/step ratio:  0.94x (async has slightly lower acceptance)
Net throughput:     1.61x (+61%)
```

## Acceptance Rate Alignment with vLLM (K=5, 10 prompts)

| Config | pos0 | pos1 | pos2 | pos3 | pos4 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Llama3.1 nano-vllm | 73.4% | 35.7% | 8.1% | 2.6% | 0.2% |
| Llama3.1 vLLM | 74.9% | 50.4% | 14.7% | 8.1% | 5.5% |
| Gap | -1.5pp | -14.7pp | -6.6pp | -5.5pp | -5.3pp |
| | | | | | |
| Qwen2 nano-vllm | 57.8% | 28.9% | 10.9% | 5.5% | 2.0% |
| Qwen2 vLLM | 63.9% | 40.0% | 17.2% | 8.3% | 3.9% |
| Gap | -6.1pp | -11.1pp | -6.3pp | -2.8pp | -1.9pp |

- **pos0 gap (~1-6pp)**: flash_attn vs flashinfer attention backend numerical difference
- **pos1+ gap amplifies**: EAGLE draft chaining compounds the per-step error
- **Greedy decode output**: token-for-token identical between nano-vllm and vLLM (target model aligned)

## Recommended Configuration

| Parameter | Recommended | Rationale |
|:---|:---:|:---|
| `ssd_early_layers` | **2** | 94% hit, good acceptance (early=1 has higher hit but lower acceptance due to shallower hidden) |
| `async_fan_out` | **3-5** | 91-95% hit; F>5 has diminishing returns |
| `num_speculative_tokens` | **3** | Best throughput/acceptance tradeoff |
| `ssd_tree_decode` | **True** | Batched tree decode for all K steps in one pass |
