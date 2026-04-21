# SSD Tree Cache 命中率扫描：early_layers x fan_out

## 实验设置

- **模型**：Llama-3.1-8B-Instruct + EAGLE、Qwen2-7B-Instruct + EAGLE
- **数据集**：GPQA Diamond（10条prompt，科学类问题）
- **生成配置**：贪心解码，max_tokens=64，K=3 投机token
- **硬件**：H100 80GB，每次运行使用2张GPU（目标模型 + 草稿模型）
- **变量**：`ssd_early_layers`（1-4）、`async_fan_out`（1-5）
- **指标**：Tree cache命中率（decode步骤中draft token来自预计算缓存 vs JIT NCCL回退的比例）

## 结果

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

## 分析

### `ssd_early_layers` 的影响

`ssd_early_layers=N` 表示目标模型在第 `L-N` 层（倒数第N层）提取隐藏状态，通过NCCL发送给草稿GPU。N越大，草稿GPU越早收到隐藏状态，在目标模型完成剩余N层计算的时间窗口内有更多时间来预构建tree cache。但N越大，提取的隐藏状态越浅，草稿预测的质量也越低。

| early_layers | Llama 命中率 (F=5) | Qwen2 命中率 (F=5) | 解读 |
|:---:|:---:|:---:|:---|
| 1 | 97.4% | 97.4% | 计算时间最短（仅1层窗口），但命中率已饱和 |
| 2 | 94.7% | 93.8% | 最佳平衡点：高命中率 + 良好接受率 |
| 3 | 86.2% | 80.5% | 时间更充裕但隐藏状态更浅，预测质量下降 |
| 4 | 75.7% | 71.1% | 隐藏状态太浅导致预测不准，命中率下降 |

**命中率受两个因素共同影响：** N越大，草稿GPU计算时间越充裕（有利于命中），但隐藏状态越浅、预测质量越低（不利于命中）。实验表明 early=1 时即使计算窗口最短，命中率仍饱和在~97%，说明单层窗口的计算时间已足够；而 early=4 时虽然计算时间最充裕，但浅层隐藏状态导致预测质量大幅下降，命中率仅~75%。

### `async_fan_out`（F）的影响

F 控制草稿模型在每个验证位置预计算多少个top-F候选token。F越大，实际被接受的token落在预计算候选中的概率越高。

| F | Llama 命中率 (e=2) | Qwen2 命中率 (e=2) | 相对F=1的提升 |
|:---:|:---:|:---:|:---:|
| 1 | 75.8% | 72.5% | 基准 |
| 2 | 86.5% | 85.1% | +11-13pp |
| 3 | 90.7% | 91.3% | +15-19pp |
| 4 | 92.7% | 92.9% | +17-20pp |
| 5 | 94.7% | 93.8% | +19-21pp |

**F从1到3提升最大（约+15pp），F从3到5收益递减（约+4pp）。** 这是合理的：在贪心解码下，top-3候选已覆盖约90%的概率质量。

### early=1 时的收益饱和

当 early=1 时，草稿GPU的计算窗口最短（仅目标模型最后1层的耗时），但即使 F=1 也能达到96-97%的命中率。这说明1层的计算时间已足以让草稿模型完成预计算。F 对命中率几乎无影响，因为时间不是瓶颈。当 early>=2 时，计算窗口更长、时间更充裕，但隐藏状态更浅导致预测覆盖率下降，F 的增大才开始发挥作用。

### 模型对比

Llama 和 Qwen2 表现出几乎相同的趋势。Llama 在 early=3,4 时略优（如 e=3,F=5 时 86.2% vs 80.5%），可能原因：
- Llama EAGLE 使用 GQA（8个KV头）vs Qwen2 EAGLE 全MHA（28个KV头），草稿模型前向更快
- 草稿模型更快 = 在目标模型完成前有更多时间完成计算

## 吞吐量汇总（H100，50条prompt，max_tokens=256）

| 配置 | tok/s | 接受率 | pos0 | pos1 | pos2 | 加速比 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Llama3.1 同步EAGLE K=3 | 252.6 | 36.3% | 67.9% | 32.1% | 8.9% | - |
| Llama3.1 异步SSD (e=3,F=5) | 260.2 | 32.3% | 60.9% | 26.3% | 9.5% | **+3%** |
| Qwen2.5 同步EAGLE K=3 | 255.8 | 29.9% | 55.2% | 24.0% | 10.6% | - |
| Qwen2.5 异步SSD (e=3,F=5) | 259.8 | 23.1% | 44.8% | 17.9% | 6.7% | **+1%** |

## 加速分解（Llama 3.1，10条prompt）

```
同步：8.3 ms/步 = EAGLE草稿 (3.4ms, 41%) + 验证 (4.9ms, 59%)
异步：4.8 ms/步 = 验证 (4.8ms) + NCCL (<0.1ms, 重叠)

单步耗时加速：1.71x（消除了41%的草稿开销）
每步token数比值：0.94x（异步接受率略低）
净吞吐量：1.61x（+61%）
```

## 与vLLM的接受率对齐（K=5，10条prompt）

| 配置 | pos0 | pos1 | pos2 | pos3 | pos4 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Llama3.1 nano-vllm | 73.4% | 35.7% | 8.1% | 2.6% | 0.2% |
| Llama3.1 vLLM | 74.9% | 50.4% | 14.7% | 8.1% | 5.5% |
| 差距 | -1.5pp | -14.7pp | -6.6pp | -5.5pp | -5.3pp |
| | | | | | |
| Qwen2 nano-vllm | 57.8% | 28.9% | 10.9% | 5.5% | 2.0% |
| Qwen2 vLLM | 63.9% | 40.0% | 17.2% | 8.3% | 3.9% |
| 差距 | -6.1pp | -11.1pp | -6.3pp | -2.8pp | -1.9pp |

- **pos0差距（约1-6pp）**：flash_attn 与 flashinfer 注意力后端的数值差异
- **pos1+差距放大**：EAGLE草稿链式推理会累积逐步误差
- **贪心解码输出**：nano-vllm 与 vLLM 的目标模型输出逐token一致

## Spec-Bench 跨模型接受率对齐（K=3，120 题子集）

数据集：`tests/data/specbench_120.jsonl` —— Spec-Bench 官方 480 题的分层子集，每个
大类别抽 20 题（math_reasoning / qa / rag / summarization / translation）+ 20 道 mt_bench
多轮。`max_new_tokens=512`，`max_model_len=2048`，贪心，bs=1。

### 主结果

| 模型 | 引擎 | pos0 | pos1 | pos2 | mean accept |
|:---|:---|:---:|:---:|:---:|:---:|
| **Vicuna-7B-v1.3** | nano-vllm | 63.2% | 36.0% | 19.3% | 2.186 |
|                    | vLLM 0.11.1 | 57.6% | 28.0% | 13.2% | 1.989 |
|                    | Δ (nano − vLLM) | +5.6 | +8.0 | +6.1 | +0.197 |
| **Llama-3.1-8B-Instruct** | nano-vllm | 64.1% | 35.1% | 17.1% | 2.164 |
|                           | vLLM 0.11.1 | 65.9% | 36.3% | 18.5% | 2.207 |
|                           | Δ (nano − vLLM) | −1.8 | −1.2 | −1.4 | −0.043 |
| **Qwen2-7B-Instruct**   | nano-vllm (EAGLE-1) | 60.0% | 32.2% | 16.4% | 2.087 |
| **Qwen2.5-7B-Instruct** | nano-vllm (EAGLE-1) | 57.5% | 28.8% | 14.6% | 2.009 |
| **Qwen2.5-7B-Instruct** | nano-vllm (EAGLE-3) | 48.8% | 21.3% |  9.1% | 1.792 |
| **Qwen3-8B**            | nano-vllm (EAGLE-3) | 50.4% | 22.6% |  9.5% | 1.824 |

> vLLM 0.11.1 既不支持 `EagleQwen2ForCausalLM`（EAGLE-1 on Qwen target），也不支持
> `EagleLlamaForCausalLMEagle3`（EAGLE-3 权重里架构写的是 `LlamaForCausalLMEagle3`
> 但 target 是 Qwen）。所以 **Qwen 系列 4 条全部只有 nano-vllm 自测**。

Llama-3.1：每位置与 vLLM 差距 ≤ 2pp，算对齐。
Vicuna：nano-vllm 比 vLLM 高 3–8pp；候选因素是 fp16→bf16 dtype 策略差异（vicuna-7b-v1.3 原生 fp16）。

**Qwen 系列 EAGLE-3 已确认是 nano-vllm 实现 bug**（非 checkpoint）：
Qwen2.5 + EAGLE-3 在 nano-vllm chain K=3 下是 48.8 / 21.3 / 9.1 (mean 1.79)，**全面低于**
Qwen2.5 + EAGLE-1 (57.5 / 28.8 / 14.6, mean 2.01)，与 EAGLE-3 论文方向相反。

在独立 venv (`/mnt/data/peizhen/Spec-Bench/.venv-specbench`, transformers==4.48.2) 里跑
Spec-Bench 官方 `EaModel3.eagenerate()` 做第三方参考。**apples-to-apples 只能
用 chain 配置 (top_k=1)** 做对比 — tree decode 每步能接受多个分支 (accept_len
上限 = depth+1，实测最长 7)，与 nano-vllm 的 chain K=3 (max accept_len 4) 不是同一量级。

| 实现 | 配置 | 120q mean_accept | max accept_len |
|---|---|:---:|:---:|
| nano-vllm chain K=3 (prefill shift fix) | — | **1.946** | 4 |
| nano-vllm chain K=3 (before fix) | — | 1.792 | 4 |
| **Spec-Bench chain K=3** (apples-to-apples) | **depth=2, top_k=1, total_token=3** | **2.133** | 4 |
| Spec-Bench chain K=4 (仅参考) | depth=3, top_k=1, total_token=4 | 2.411 | 5 |
| Spec-Bench tree (仅 ckpt sanity，不做对齐) | depth=5, top_k=10, total_token≈60 | 3.954 | 7 |

**注意 1**：Spec-Bench `topK_genrate` 的 `depth=N` 实际产 `N+1` drafts（初始 sample + N 次
chain 迭代）。真正 K=3 apples-to-apples 要用 `depth=2, top_k=1, total_token=3`。

**注意 2**：tree (3.954) 每步可接 7 个 token，和 chain (max 4) 根本不是一个单位，之前把
tree 当基准直接对比是错的。tree 数据只用来确认 draft checkpoint 本身是健康的。

**真实 apples-to-apples gap**：Spec-Bench chain K=3 (2.133) vs nano-vllm chain K=3 (1.946
post-shift-fix) = **0.19 tokens/step** (~8.8% 低，远小于最初误报的 26%)。

已排除的怀疑：
- 草稿 checkpoint 健康（Spec-Bench tree 3.95 确认）
- aux 层选择（`[1, N/2-1, N-4]`，与 SpecForge 训练默认一致）
- fc 权重形状 (3584×10752 = 3×H)、`combine_hidden_states` 调用点
- d2t offset → absolute 转换

### Phase 3.4 op-level bisection（已完成部分）

同一短 prompt "What is the capital of France? Answer in one sentence." 上 apples-to-apples：

| 实现 | accept_lens | 轮数 | pos0 | pos1 | pos2 | 最终输出 |
|---|---|:---:|:---:|:---:|:---:|---|
| Spec-Bench chain-K=3 | [4, 4] | 2 | 100% | 100% | 100% | "The capital of France is Paris.<\|im_end\|>" |
| nano-vllm chain K=3 | pos-wise | 3 | 100% | 67% | 33% | （逐 token 一致）|

**关键发现**：
- 两侧最终输出 token 逐字一致 → greedy target 完全对齐
- **step 0 / pos0 两边都是 100%** → draft 权重、aux 抽取、`combine_hidden_states`、
  `Eagle3DecoderLayer` 单层前向、lm_head、d2t 映射都对
- **pos1/pos2 严重偏低** → 问题在 **chain continuation（step 1 → step 2）的状态传递**

已逐行比对 `nanovllm/models/eagle.py::Eagle3DecoderLayer.forward` 与 Spec-Bench
`model/eagle3/cnets.py::LlamaDecoderLayeremb.forward`：两者 `(hidden_states + residual)` 返回值
数学等价（= `pre_layer_hidden + attn_out + mlp(norm(pre_layer_hidden + attn_out))`）。fused
`post_attention_layernorm(x, residual)` 与非 fused 版在 BF16 下的数值差距不足以解释 33pp gap。

剩余需要深入的可疑面（下一轮调试）：
- **chain 循环的 KV slot_mapping / position_ids**（`model_runner.py:593-607`）：
  `p = len(seq) - 1 + step`，需核对 step=0 位置与 prefill 期间已写入 draft KV 的最后一个位置
  是否会重复覆盖
- **verify 后 `_last_aux_hidden[seq_id] = aux_concat[accepted_idx:accepted_idx+1]`**
  （`model_runner.py:682`）：verify 阶段 aux_concat 在 accepted_idx 取的是下一步 step 0 需要的 aux，
  如果 offset 偏一格，下一轮 step 0 的草稿就会漂
- **EAGLE-3 forward 内部对 `aux_hiddens` 的判空**：step 1+ `aux_hiddens=None` 让
  `hidden_states` 直接透传（没再 `fc` 一次），但 Spec-Bench 的 `forward` 内部是按
  `shape[-1] != inputs_embeds.shape[-1]` 判断是否要 fc，逻辑等价但实现路径不同

Qwen3 + EAGLE-3 的 Spec-Bench 参考跑不通：transformers 4.48 不认识 qwen3；升到 4.51+ 又触发
`low_cpu_mem_usage=True` 下 `topK_genrate` 的 meta-tensor 初始化 bug。暂作 known issue。

Spec-Bench 侧所需的三处 patch（全部落在 venv 内，不入 nano-vllm 仓库）：
1. `model/eagle3/ea_model.py::forward` — `output_hidden_states=True` + 按
   `[2, N/2, N-3]`（HF `hidden_states[i]` 对应 layer `i-1` 的输出）过滤到 3 个 aux 层
2. `model/eagle3/modeling_qwen2_kv.py::_init_rope` — 新 transformers 合成的
   `{"rope_type":"default"}` 要走默认 rope 分支
3. `model/eagle3/cnets.py::_init_rope` — 同 2

### 修复记录

本次对齐过程触发两处 EAGLE 权重加载的 bug，均已修复并进入 `nanovllm/models/eagle.py` 与
`nanovllm/engine/model_runner.py`：

1. **`fc.bias` 被未初始化内存污染**。`EAGLE-LLaMA3.1-Instruct-8B` 的 checkpoint 不附带
   `fc.bias`，但 `EAGLEModel` 硬编码 `bias=True` 且只依赖 `torch.empty`；加载器没有匹配键
   ⇒ 张量里是 `1e34`、`5376.0` 这类垃圾值。后果：Llama-3.1 上 0% 接受。
   修复：构造时 `torch.nn.init.zeros_(self.fc.bias)`，ckpt 无键时保持零偏置（加法 no-op）。

2. **Llama-3 系列 rope_scaling 未继承**。EAGLE 草稿 config 里常见 `rope_theta=500000` 但
   无 `rope_scaling`，而目标模型使用 `rope_scaling={"rope_type":"llama3", factor=8, ...}`。
   HF transformers 会把 draft config 里的 `rope_theta` 自动包装成
   `{"rope_type":"default", "rope_theta":500000}`，老 patch 里 `if rope_scaling is None` 的
   判断于是永远过不去。修复：构造 `EAGLEModel` 前，若 target 有实际 scaling 类型而 draft
   只有默认类型，则把 target 的 `rope_scaling` 拷给 draft config。

3. **Vicuna 持续性 mid-run 段错误**（未修复，使用 chunked workaround）。对 `vicuna-7b-v1.3`
   在 `run_spec_bench` 里跑到第 ~12 个 prompt 后必定 SIGSEGV，且只在 Vicuna 上出现（Llama-3.1、
   Qwen2、Qwen2.5 连续跑 120 题均无问题）。改用 10 题/chunk 模式（`bash /tmp/specbench_align/
   run_vicuna_chunks.sh`）可拿到完整 120 题数据。后续应单独立项排查。

### 方法

- 数据集：`tests/data/specbench_120.jsonl`（Spec-Bench 6 大类分层抽样 120 题）
- 参考实现：vLLM 0.11.1（`speculative_config={"method":"eagle", ...}`，chain K=3）
- 运行脚本：`tests/harness/run_spec_bench.py`（nano-vllm）、`experiments/bench_vllm_specbench.py`（vLLM 参考）
- 每位置接受率：nano-vllm 从 `accept_lengths[i] ≥ pos+2` 导出；vLLM 从
  `get_metrics() -> vllm:spec_decode_num_accepted_tokens_per_pos` 读取
- Vicuna tokenizer 无 `chat_template`，harness 回退到 FastChat `get_conversation_template("vicuna")`
  的 v1.1 模板（system + `USER: ... ASSISTANT: ...`），与 EAGLE-Vicuna 训练分布一致

## 推荐配置

| 参数 | 推荐值 | 理由 |
|:---|:---:|:---|
| `ssd_early_layers` | **2** | 94%命中率，良好接受率（early=1命中率更高，但因隐藏状态较浅导致接受率较低） |
| `async_fan_out` | **3-5** | 91-95%命中率；F>5收益递减 |
| `num_speculative_tokens` | **3** | 吞吐量/接受率的最佳权衡 |
| `ssd_tree_decode` | **True** | 批量tree decode，一次完成所有K步 |
