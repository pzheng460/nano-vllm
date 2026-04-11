# nano-vllm

轻量级 vLLM 兼容推理引擎，支持投机解码。

## 项目结构

- `nanovllm/engine/llm_engine.py` — 主引擎，协调 prefill/decode/投机解码循环
- `nanovllm/engine/model_runner.py` — 目标模型运行器，处理同步 MTP 和 SSD 解码
- `nanovllm/engine/draft_runner.py` — 异步草稿运行器（独立 GPU），树缓存 + MTP
- `nanovllm/engine/scheduler.py` — 请求调度和块管理
- `nanovllm/models/mimo.py` — MiMo 模型，含 MTP 层
- `nanovllm/models/pangu.py` — PanGu 模型，含 MTP + MoE
- `nanovllm/config.py` — 所有配置参数

## 投机解码算法

### 1. 同步 MTP（多 Token 预测）

单 GPU 投机解码，使用模型内置的 MTP 层。

**流程（每个 decode 步骤）：**
1. **草稿阶段**：在目标 GPU 上递归运行 MTP 层 K 次
   - 输入：`(embed(token), hidden_state)` → MTP → 草稿 token
   - 链式传递：使用 `final_layernorm` 输出（normed）作为下一步的 hidden（与 vLLM 对齐）
   - 第一步使用未归一化的目标 hidden（MiMo 自带 `hidden_layernorm`）
2. **验证阶段**：目标模型以 prefill 方式一次处理 `[last_token, d0, d1, ..., d_{K-1}]`
3. **接受阶段**：贪心匹配——接受连续匹配的 token，追加 bonus token
4. **MTP KV 更新**：接受后，用 shifted token 逐步更新 MTP KV cache

**关键细节：**
- MiMo MTP 期望接收**未归一化**的目标 hidden（自带 `hidden_layernorm`）
- PanGu MTP 期望接收**已归一化**的 hidden（无 `hidden_layernorm`）
- Position 0 遮蔽：`input_embeds[positions == 0] = 0`（与 vLLM 对齐）
- MTP KV cache 使用 shifted token ID：位置 i 使用 `embed(token_{i+1})`
- `num_speculative_tokens` 自动设为 `num_nextn_predict_layers`，除非用户显式覆盖

### 2. SSD-MTP（投机流式解码）

双 GPU 异步投机解码。目标模型在 GPU 0，MTP 草稿在 GPU 1。

**架构：**
```
GPU 0（目标）                      GPU 1（草稿）
┌─────────────────┐              ┌──────────────────┐
│ Prefill/Decode   │   NCCL      │ MTP 草稿模型      │
│ 完整模型          │◄──────────►│ embed + MTP 层    │
│ MTP KV cache     │             │ MTP KV cache      │
│（用于 cache miss）│             │ 树缓存            │
└─────────────────┘              └──────────────────┘
```

**SSD 解码流程：**

```
第 T 步：
  1. 推测：cache 命中 → NCCL 查找（快速）| 未命中 → 目标 GPU 本地 MTP
  2. 验证：目标模型前向 [last_token, d0..d_{K-1}]
     - 在第 N-X 层：提取 early hidden，批量 NCCL 发送到草稿 GPU
     - 草稿 GPU 接收并异步预计算第 T+1 步的候选（与目标剩余层并行）
  3. 接受：贪心匹配，更新 last_hidden
  4. 草稿 GPU 存储树缓存条目：(seq_id, position, candidate) → draft_token
```

**树缓存（草稿 GPU 上）：**
- **填充**：验证阶段，草稿 GPU 接收 K+1 个位置的 early hidden。
  对每个位置：`norm(early_hidden) → lm_head → topF 候选 → embed → MTP → draft_token`。
  每个 seq 存储 `(K+1)*F` 个条目。
- **查找**：链式 K 步查找：`(sid, 0, recovery_token) → d0`，`(sid, 1, d0) → d1`，`(sid, 2, d1) → d2`
- **命中率**：`ssd_early_layers=2`，`async_fan_out=3` 时约 99%

**NCCL 通信协议（cmd ID）：**
| cmd | 名称 | 方向 | 数据 |
|-----|------|------|------|
| 0 | speculate | T→D | hidden + block_table → D 返回 K 个草稿 token |
| 1 | prefill | T→D | shifted_tokens + hidden + positions + block_table |
| 2 | exit | T→D | 关闭草稿循环 |
| 3 | cleanup | T→D | seq_id（已完成序列） |
| 4 | cache_update | T→D | (seq_id, acc_len, token) + K 个草稿 token |
| 5 | early_speculate | T→D | 批量：num_seqs + 每 seq 元数据 + hidden + bt + pos |
| 6 | cache_lookup | T→D | (seq_id, acc_len, token) → D 返回 K 个 token |

**关键优化：**
- 批量 NCCL：所有 seq 的 early hidden 一次发送（非逐 seq）
- SSD 模式跳过目标端 MTP KV 更新（99% cache 命中，更新无必要）
- 草稿 KV cache 上限为 `max_num_seqs * max_blocks_per_seq`（不填满 GPU）
- `ssd_early_layers` 配置：在第 `N-X` 层提取 early hidden（默认 2）

### SSD-MTP 算法详解

#### 一、整体架构

SSD-MTP 是一种**双 GPU 异步推测解码**方案：

```
GPU 0 (Target)                         GPU 1 (Draft)
┌──────────────────────┐               ┌──────────────────────┐
│  完整目标模型          │    NCCL      │  MTP Draft 模型       │
│  - Prefill/Decode     │◄────────────►│  - embed + MTP layer  │
│  - Early hidden 抽取  │              │  - Tree Cache         │
│  - 投机验证           │              │  - 异步构建候选树      │
└──────────────────────┘               └──────────────────────┘
```

核心思想：**在目标模型做 verify forward 的同时，利用中间层（early layer）的 hidden state 提前发送给 Draft GPU，让 Draft 异步预计算下一步的候选 token 树，从而在下一个 decode step 中实现接近零延迟的投机。**

#### 关键配置参数

| 参数 | 含义 |
|------|------|
| `draft_async=True` | 启用异步双 GPU 模式 |
| `num_speculative_tokens (K)` | 每步投机 token 数 |
| `ssd_early_layers` | 从第 `N-X` 层抽取 early hidden（默认 2） |
| `async_fan_out (F)` | 树缓存扇出因子（默认 3），每个位置取 top-F 候选 |

#### 二、算法流程（每个 Decode Step）

**Step 1: 投机（Speculation）**

Target 向 Draft 发送 **cache lookup 请求**（NCCL cmd=6），查询格式为 `(seq_id, accepted_len, last_token)`：

- **Cache Hit（~99% 情况）**：Draft 直接从 Tree Cache 链式查找 K 个 token 返回，**几乎零延迟**
- **Cache Miss**：回退到 Target GPU 本地用 MTP layer 链式生成 K 个 draft token（较慢）

Tree Cache 查找逻辑：
```
lookup(sid, pos=0, recovery_token) → d₀
lookup(sid, pos=1, d₀) → d₁
lookup(sid, pos=2, d₁) → d₂
...共 K 步链式查找
```

**Step 2: 验证（Verify）**

Target 模型对 `[last_token, d₀, d₁, ..., d_{K-1}]` 做一次 **prefill-like forward**：

- 逐层计算，到达第 `N - ssd_early_layers` 层时：
  - **抽取 early hidden state**
  - **非阻塞异步 NCCL 发送**（cmd=5）给 Draft，包含所有 seq 的 early hidden、block table、positions
  - Target **继续计算剩余层**，与 Draft 的树构建**并行执行**

**Step 3: Draft 异步构建 Tree Cache（与 Step 2 剩余层并行）**

Draft 收到 early hidden 后：
1. **Glue decode**：对 verify 的 K+1 个位置的 hidden state，通过 MTP layer 得到 prenorm hidden
2. **Fork**：对每个位置取 **top-F 候选 token**（排除已选 token）
3. **Tree decode**：每个候选 token embed 后，通过 MTP layer 链式展开 K 步
4. **存入 Tree Cache**：key = `(seq_id, position, candidate_token)` → value = `K 个后续 draft tokens`

每个 seq 存储 `(K+1) × F` 条缓存记录，覆盖下一步所有可能的 recovery token。

**Step 4: 贪心接受（Accept）**

Target 的 verify forward 完成后，得到 K+1 个位置的 logits：

```
logits[0] → 验证 d₀ 是否 = argmax(logits[0])
logits[1] → 验证 d₁ 是否 = argmax(logits[1])
...
第一个不匹配处停止，取 argmax 作为 recovery token
若全部匹配，额外获得 bonus token
```

接受后更新 `last_hidden`，将 recovery token 保存供下一轮 cache lookup 使用。

#### 三、关键优化

1. **Early hidden extraction**：倒数第 X 层就把 hidden 发出去，Draft 和 Target 最后几层**并行计算**
2. **Skip MTP KV update on target**：SSD 模式下 Target 不维护 MTP 的 KV cache（hit rate ~99%，miss 时用 local MTP 兜底）
3. **批量 NCCL**：所有 seq 的 early hidden 打包成一条消息发送，避免多次 NCCL 通信开销
4. **Tree Cache 预计算**：Fan-out=3 意味着每个位置预计算 3 个分支，下一步几乎必然命中
5. **非阻塞异步发送**：`dist.isend()` 不阻塞 Target 的后续层计算

#### 四、时间线

```
Target:  [==== Verify Forward (layers 0..N-X) ====][send hidden][layers N-X+1..N][Accept]
Draft:                                              [recv hidden][Build Tree Cache =========]
                                                                  ↑ 与 Target 最后 X 层并行

Next Step:
Target:  [Cache Lookup (fast!)][==== Verify Forward ====] ...
Draft:   [Respond K tokens    ][                        ] ...
```

核心优势在于：Draft 的树缓存构建与 Target 的最后几层计算**重叠执行**，使得下一步的投机查找几乎免费，从而实现比同步 MTP 更高的吞吐量（长序列场景下可达 2.6x 加速）。

### 3. EAGLE 投机解码

外部草稿模型（`nanovllm/models/eagle.py`）。使用轻量单层 Transformer 作为草稿。

## MiMo MTP 层细节

```python
def forward(input_embeds, hidden_states, positions):
    input_embeds[positions == 0] = 0          # position 0 遮蔽
    input_embeds = token_layernorm(input_embeds)
    hidden = hidden_layernorm(hidden_states)   # 期望未归一化的 hidden
    hidden = input_proj(cat([hidden, input_embeds]))  # [hidden, embed] 拼接顺序
    # Transformer 块：pre-norm attn + post-norm MLP
    hidden = attn(input_layernorm(hidden)) + hidden
    hidden = mlp(post_attn_layernorm(hidden)) + hidden
    return final_layernorm(hidden), hidden     # (normed, prenorm)
```

- **Logits 计算**：`F.linear(normed_output, lm_head.weight)` — 使用主模型的 lm_head（无 shared_head）
- **链式传递**：递归 MTP 使用 `normed` 输出作为下一步的 `hidden_states`
- **第一步**：使用目标模型未归一化的 hidden（来自残差流）

## 构建与测试

```bash
# 同步 MTP K=1（单卡）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_ssd.py --mode sync1

# 同步 MTP K=3（单卡）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_ssd.py --mode sync3

# 异步 SSD K=3（双卡）
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python bench_ssd.py --mode async3 --early-layers 2 --fan-out 3
```

## 性能分析与优化流程

### 使用 torch.profiler 生成火焰图

生成 Chrome trace JSON 文件，在 [Perfetto UI](https://ui.perfetto.dev/) 或 `chrome://tracing` 中查看 timeline 和火焰图：

```bash
# 分析同步 MTP
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/profile_ssd.py --mode sync1 --prompts 3

# 分析异步 SSD（双卡）
NCCL_PORT=2345 CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python experiments/profile_ssd.py --mode async1 --prompts 3
```

输出 `*.json.gz` trace 文件，拖入 Perfetto UI 查看 timeline 和火焰图。

### 优化检查清单

- 检查 GPU 利用率：若 Self CPU >> Self CUDA，瓶颈在 CPU 侧（tensor 创建、H2D 拷贝）
- 检查 NCCL 开销：profiler 输出中 `ncclDevKernel_SendRecv` 和 `nccl:recv` 的占比
- 排查 dead code：计算结果从未被读取的代码（如 `_last_candidates` 的 norm→lm_head→topk 预计算耗费 6ms/step 但从未使用）
- 评估 MTP KV update 必要性：单层 MTP 的 KV cache 在 draft 阶段已写入，verify 后无需重跑

## 性能（MiMo-7B-Base, H100, 50 prompts, max_tokens=256）

| 模式 | tok/s | pos0 接受率 | pos1 | pos2 |
|------|-------|------------|------|------|
| 同步 MTP K=1 | 381.9 | 86.5% | - | - |
| 同步 MTP K=3 | 257.6 | 84.0% | 22.7% | 5.9% |
| 异步 SSD K=3 | 406.5 | 72.0% | 8.2% | 0.2% |

## 性能（Qwen2-7B + EAGLE, H100, bs=1, max_tokens=256）

| 模式 | tok/s | pos0 | pos1 | pos2 |
|------|-------|------|------|------|
| 基线（无投机） | 52.4 | - | - | - |
| 同步 EAGLE K=3（batched） | 78.4 | 63% | 37% | 10% |
| 异步 EAGLE SSD K=3（tree） | 78.3 | 66% | 40% | 13% |
| 异步 EAGLE SSD K=2（tree） | 75.3 | 65% | 32% | - |

### 性能（Qwen2-7B + EAGLE, H100, bs=50, max_tokens=256）

| 模式 | tok/s | pos0 | pos1 | pos2 |
|------|-------|------|------|------|
| 同步 EAGLE K=2（batched） | 1541 | 48% | 17% | - |
| 同步 EAGLE K=3（batched） | 1493 | 46% | 18% | 9% |
| 异步 EAGLE SSD K=3（tree） | 1079 | 48% | 21% | 7% |

注意事项：
- **bs=1 时异步 K=3（80.0）超过同步 K=3（78.4）** — draft 完全被掩盖，比基线快 53%
- bs=50 时同步更快（batched EAGLE draft 在大 batch 下高效）
- K>1 时必须使用 `ssd_tree_decode=True`（chain 模式只做 1 步 draft）
- EAGLE 对 `ssd_early_layers=1` 最优（多层时接受率大幅下降）
- 同步 EAGLE draft 已 batch 化（所有 seq 一起处理，非逐 seq 串行）
