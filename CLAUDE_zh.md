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

## 性能（MiMo-7B-Base, H100, 50 prompts, max_tokens=256）

| 模式 | tok/s | pos0 接受率 | pos1 | pos2 |
|------|-------|------------|------|------|
| 同步 MTP K=1 | 381.9 | 86.5% | - | - |
| 同步 MTP K=3 | 257.6 | 84.0% | 22.7% | 5.9% |
| 异步 SSD K=3 | 406.5 | 72.0% | 8.2% | 0.2% |
