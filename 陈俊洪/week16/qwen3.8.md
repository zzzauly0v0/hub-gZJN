---
title: Qwen3.8-27B
model_id: Qwen/Qwen3.8-27B
publisher: Qwen Team (Alibaba)
license: Apache-2.0
source: https://huggingface.co/Qwen/Qwen3.8-27B
---

# Qwen3.8-27B

> 官方自称 "the most capable generation in the **Qwen open-model family** to date" 的**稠密（Dense）多模态模型**，270 亿参数，主打**编程、专业办公、研究与长周期 Agent 任务**。文本 + 图像 + 视频输入、文本输出，原生 **256K** 上下文并可 YaRN 扩展至 **1M**。
>
> 标注说明：带 `[cfg]` 的条目取自仓库 `config.json`，模型卡正文未直接列出；带 `[推断]` 的为笔者依据配置做的解读，非官方表述。

---

## 1. 模型定位与核心亮点

- **能力上限优先，不谈本地部署：** 与 [Muse Glimmer-30B](Muse-Glimmer-30B.md) 把"消费级硬件"作为第一性设计目标不同，这张模型卡的叙事重心是**跨编程、办公、研究、Agent 的全面能力提升**，通篇未提显存需求、量化方案或本地推理性能。

- **稠密架构（Dense Model）：** 推理时激活全部参数。`config.json` 中**不存在任何 MoE / expert 相关字段**，可确认非混合专家结构 `[cfg]`。

- **原生视觉语言理解：** 内置视觉编码器，原生支持图像与视频，覆盖范围官方表述为"from STEM diagrams and documents to **hour-scale videos**"——**小时级视频**是三张卡里唯一明确承诺的（Gemma 4 与 Muse Glimmer 均只能逐帧处理）。

- **Agent 执行力强化：** 官方强调"更强的自主规划能力（stronger autonomous planning）"与"更好的环境反馈处理（better handling of environment feedback）"，目标是端到端长流程任务的可靠完成率。

- **生态兼容性被单列为亮点：** 官方明确扩大了对主流 harness 与开发工具链的支持，便于接入既有技术栈——这与它逐条给出框架启动命令的做法一致（见 §7）。

- **灵活思考控制：** 思考模式默认开启、可按请求关闭；推理深度用独立请求字段 `reasoning_effort` 调节，历史思考上下文由 `preserve_thinking` 保留（见 §5）。

- **开源协议：** Apache 2.0。

> ⚠️ **不要引用的表述：** "紧凑、易于部署的密集型模型"这类说法**不是模型卡原文**。卡片只是按 dense 模型呈现，从未以"易部署"或"紧凑"作为卖点，也未给出任何显存数字。

---

## 2. 模型规模与核心维度参数

| 项目 | 数值 |
| --- | --- |
| 模型类型 | Causal Language Model with Vision Encoder |
| 训练阶段 | Pre-training & Post-training |
| 语言模型参数量 | 27B |
| 网络层数（Layers） | 64 |
| 隐藏层维度（hidden_size） | 5,120 |
| FFN 中间维度（intermediate_size） | 17,408 |
| 词表 / Token Embedding | 248,320（已填充 Padded） |
| LM Output | 248,320（已填充 Padded） |
| Gated DeltaNet 头数 | V：48 个线性注意力头；QK：16 个 |
| Gated DeltaNet 头维度 | 128 |
| Gated DeltaNet 卷积核维度 | `linear_conv_kernel_dim = 4` `[cfg]` |
| Gated Attention 头数 | Q：24；KV：4（GQA，6:1） |
| Gated Attention 头维度 | 256 |
| 旋转位置嵌入维度 | 64（= 256 × `partial_rotary_factor` 0.25）`[cfg]` |
| 上下文长度 | 262,144 tokens（`max_position_embeddings = 262144`） |
| Embedding 权重共享 | 否（`tie_word_embeddings = false`）`[cfg]` |
| 权重精度 | BF16（SSM 状态以 float32 计算，`mamba_ssm_dtype`）`[cfg]` |
| 输入 / 输出模态 | 文本 + 图像 + 视频 / 文本 |
| HF 任务标签 | `Image-Text-to-Text` |
| 特殊 token id | bos/eos 248044 / image 248056 / video 248057 / vision_start 248053 / vision_end 248054 `[cfg]` |
| 依赖版本声明 | `transformers_version = 5.8.0.dev0` `[cfg]` |

**几个容易被忽略的实现相关字段** `[cfg]`：

| 字段 | 值 | 作用 |
| --- | --- | --- |
| `attn_output_gate` | `true` | 注意力输出加门控（Gated Attention 的"Gated"所指） |
| `output_gate_type` | `swish` | 门控激活函数 |
| `full_attention_interval` | 4 | 每 4 层插入一层全注意力 |
| `mtp_num_hidden_layers` | 1 | 多词元预测头层数（见 §7.3） |
| `language_model_only` | `false` | 默认以多模态模式加载 |

> ⚠️ 与 Muse Glimmer 那种改了 softcapping / QK scale 的情况不同，Qwen3.8 没有非标准数值稳定性字段；但 `linear_attention` 层需要 **Gated DeltaNet（线性注意力 + SSM 状态）** 内核支持，**这本身就不是标准 Transformer 结构**，自行实现推理栈的成本反而更高。

> 📌 **命名注意：** `config.json` 里 `model_type = qwen3_5`、架构名 `Qwen3_5ForConditionalGeneration`——沿用上一代代码路径命名，不是笔误。

---

## 3. 注意力机制设计

### 3.1 线性 / 全局交替布局

64 层采用固定的 4 层一组循环：

```
16 × ( 3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN) )
```

即每 4 层中前 3 层为 `linear_attention`（Gated DeltaNet），第 4 层为 `full_attention`（Gated Attention），全注意力层共 16 层（`full_attention_interval = 4`）`[cfg]`。

> **与两个同级模型的对比：** 三者都是"局部 3 + 全局 1"的四层循环，但**局部层的实现路线完全不同**——
>
> | 模型 | 局部层机制 | 全局层数 / 总层数 |
> | --- | --- | --- |
> | **Qwen3.8-27B** | **Gated DeltaNet（线性注意力）** | 16 / 64 |
> | Gemma 4 31B | 滑动窗口注意力（window = 1024） | 交替，末层强制全局 |
> | Muse Glimmer-30B | 滑动窗口注意力（window = 2048） | 13 / 52 |
>
> 滑动窗口只是**限制注意力范围**，复杂度仍是窗口内的二次；Gated DeltaNet 是**真正的线性注意力 / 状态空间**路线，KV Cache 被固定大小的状态替代。这是 Qwen3.8 敢把原生上下文做到 256K、并宣称可扩到 1M 的结构基础 `[推断]`。

### 3.2 位置编码：多模态 mRoPE

| 项目 | 值 |
| --- | --- |
| RoPE 类型 | mRoPE（多模态旋转位置编码），`rope_type = default` |
| `rope_theta` | 10,000,000（1e7） |
| `mrope_section` | `[11, 11, 10]`（时间 / 高 / 宽 三段） |
| `mrope_interleaved` | `true`（三段交错排布） |
| `partial_rotary_factor` | 0.25（仅 1/4 维度施加旋转，即 256 → 64 维） |

`[cfg]` 全部来自 `text_config.rope_parameters`。

- **`mrope_section = [11, 11, 10]`** 表示位置编码被拆成时间、高度、宽度三个子空间，是原生视频理解的位置编码基础。
- **`partial_rotary_factor = 0.25`** 意味着 256 维的头里只有 64 维带旋转位置信息，其余 64×3 维为无位置先验（NoPE 式）分量。`[推断]` 这与 Muse Glimmer 把全局层 `rope_theta` 设为 0 的思路同向——都在为长上下文外推削弱 RoPE 频率衰减的约束，只是 Qwen3.8 做在**维度**上，Muse 做在**层**上。

### 3.3 视觉编码器 `[cfg]`

模型卡正文**未给出任何视觉塔规格**（连参数量都没写），以下全部由 `vision_config` 得出：

| 项目 | 数值 |
| --- | --- |
| 层数（depth） | 27 |
| 隐藏维度 | 1,152 |
| 注意力头数 | 16（head_dim = 72） |
| FFN 中间维度 | 4,304 |
| 投影输出维度 | `out_hidden_size = 5120`（对齐语言模型 hidden_size ✅） |
| Patch 大小 | 16 |
| 空间合并 / 时间 patch | `spatial_merge_size = 2`；`temporal_patch_size = 2` |
| 位置嵌入数 | 2,304（48 × 48 网格）`[推断]` |
| 激活函数 | `gelu_pytorch_tanh` |
| DeepStack 视觉索引 | `[]`（空，本 checkpoint 未启用） |

- **`temporal_patch_size = 2`** 表示每 2 帧在时间维度合并，这是"小时级视频"能力的结构支撑——与 Muse Glimmer 那种"结构上留了能力但官方不保证视频"的情况不同，Qwen3.8 是**明确承诺**的。
- **投影维度自洽**（1152 → 5120 = 语言模型 hidden_size），不存在 Muse Glimmer 那种 `out_hidden_size` 与 `hidden_size` 对不上的疑点。
- 视觉塔**未采用**语言模型的"局部 3 + 全局 1"布局，是常规全注意力 ViT；这点与 Muse Glimmer（视觉塔复用同一套混合注意力）不同。

---

## 4. 上下文长度

- **原生上下文：** 262,144 tokens（256K）。
- **扩展上限：** 通过 **YaRN**（RoPE 缩放）可扩展至 **1,000,000 tokens**——三张卡里**唯一给出完整扩展配置**的。

修改 `config.json` 中的 `rope_parameters`：

```json
{
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10],
  "rope_type": "yarn",
  "rope_theta": 10000000,
  "partial_rotary_factor": 0.25,
  "factor": 4.0,
  "original_max_position_embeddings": 262144
}
```

各框架需同时放开长度上限：

| 框架 | 环境变量 | 长度参数 |
| --- | --- | --- |
| vLLM | `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | `--max-model-len 1000000` |
| SGLang | `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` | `--context-length 1000000` |
| TokenSpeed | `TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` | `--max-model-len 1000000` |

> ⚠️ **官方提示的代价：** 这是**静态（static）YaRN**，缩放系数恒定生效、不随输入长度动态调整，因此**可能损害短文本上的表现**。日常输入远低于 256K 时不要盲目开启。

> 对比参考：Qwen3.8-27B 原生 256K / 可扩 1M；Gemma 4 31B 原生 256K（未给扩展方案）；Muse Glimmer-30B 原生 128K（未给扩展方案）。**上下文长度是 Qwen3.8 相对两者最明确的结构优势。**

---

## 5. 推理强度控制

| 参数 | 位置 | 取值 | 默认 |
| --- | --- | --- | --- |
| 思考模式 | 请求级开关 | 开 / 关（输出包裹在 `<think>...</think>`） | **开启** |
| `reasoning_effort` | **独立请求字段** | `xhigh` / `medium` / `low` | **`xhigh`** |
| `preserve_thinking` | `chat_template_kwargs` | `true` / `false` | **`true`** |

- **`reasoning_effort`：** 默认 `xhigh`，官方描述面向"需要充分分析的复杂任务"；降档用于在推理深度与成本之间取平衡。
- **`preserve_thinking`：** 默认保留**全部历史消息**中的思考块，官方表述为"maintaining a complete reasoning trace across the conversation"——对长线 Agent 任务的逻辑一致性关键，但会显著吃上下文预算。
- **官方反直觉提示：** 在多轮 Agent 任务中，**降低 reasoning effort 并不总能缩短整体任务完成时间**（推理省下的 token 可能被更多轮试错抵消）。

> ⚠️ **与两个同级模型的机制差异（选型时容易踩）：**
>
> | 模型 | 控制方式 | 档位 | 历史思考保留 |
> | --- | --- | --- | --- |
> | **Qwen3.8-27B** | **独立请求字段** `reasoning_effort` | 3 档（`low`/`medium`/`xhigh`） | ✅ `preserve_thinking`，默认保留 |
> | Gemma 4 31B | System Prompt 前置 `<\|think\|>` / `enable_thinking` | 二元开关 | ❌ 规范要求**丢弃**（工具调用轮除外） |
> | Muse Glimmer-30B | System Prompt 文本行 `Reasoning strength: <value>` | 4 档（多一个 `high`） | ❌ 未提供开关 |
>
> - Qwen3.8 是三者中**唯一把推理强度做成结构化 API 字段**的，好处是不占 System Prompt、不会被用户指令冲淡；代价是**依赖推理框架显式支持**，OpenAI 兼容端点若未透传该字段就会静默失效。
> - `preserve_thinking` 也是三者中唯一的历史思考保留开关。**注意它与 Gemma 4 的规范正好相反**——Gemma 4 明确要求多轮时丢弃上轮思考，Qwen3.8 默认全保留。跨模型迁移 Agent 代码时这条必须改。
> - 档位命名不通用：Qwen3.8 **没有 `high`**（直接跳到 `xhigh`），Muse Glimmer 有 4 档。照抄字符串会报错或落回默认。

**推荐采样参数：**

| 模式 | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
| --- | --- | --- | --- | --- | --- | --- |
| Thinking（思考模式） | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Instruct（非思考模式） | 0.7 | 0.80 | 20 | 0.0 | **1.5** | 1.0 |

> 与另两个模型的差异：Gemma 4 31B 与 Muse Glimmer 均推荐 `temperature=1.0 / top_p=0.95 / top_k=64`。Qwen3.8 思考模式的 `top_k` 只有 **20**（收窄 3 倍），且非思考模式额外要求 `presence_penalty=1.5`——这个值相当激进，是三张卡里唯一非零的惩罚项，**不要沿用其他模型的参数直接跑**。

**输出长度建议：**

- **推理内容（Reasoning Content）：** 最大输出长度设为 **262,144** tokens。
- **最终回答（Final Response）：** 最大输出长度设为 **131,072** tokens。

---

## 6. 基准测试表现

对比机型：**Qwen3.6-27B**（上一代同尺寸）、**Qwen3.7-Plus**（自家更大模型）、**Muse Glimmer-30B**、**Opus4.6 Max**。**加粗**为该行最高分；`--` 为官方未提供数据。

### 6.1 文本能力（Text）

| 基准 | Qwen3.8-27B | Qwen3.6-27B | Qwen3.7-Plus | Muse Glimmer-30B | Opus4.6 Max |
| --- | --- | --- | --- | --- | --- |
| 终端 Agent 编程（Terminal Bench 2.1） | 73.0 | 63.4 | 64.0 | 51.7 | **78.2** |
| Agent 编程（SWE-bench Pro） | **61.7** | 53.5 | 57.6 | 51.2 | 53.4 |
| 仓库级代码生成（NL2Repo-Bench） | 42.3 | 36.2 | 41.1 | -- | **47.6** |
| Agent 代码修复（DeepSWE 1.1） | **42.2** | 13.3 | 14.2 | -- | -- |
| 企业级软件工程（QwenSWEBench） | **79.0** | 49.3 | 59.2 | -- | 63.8 |
| 长流程协同办公（CoWorkBench） | **70.7** | 61.0 | 65.1 | -- | 68.2 |
| 专业岗位任务（JobBench） | **33.4** | 21.8 | 27.6 | -- | -- |
| 前沿 Agent 任务（Agents' Last Exam）Pass@1 | **20.4** | 10.6 | 13.2 | -- | -- |
| 前沿 Agent 任务（Agents' Last Exam）Score | **42.9** | 27.3 | 33.6 | -- | -- |
| 复杂指令遵循（IFBench） | **79.5** | 69.1 | 79.1 | 77.0 | 62.5 |
| 专家级科学推理（GPQA Diamond） | 89.2 | 87.8 | 90.3 | 83.5 | **91.3** |
| 多学科综合难题（HLE） | 30.8 | 24.0 | 34.7 | 22.0 | **40.0** |
| 竞赛编程（LiveCodeBench v6） | **90.3** | 83.9 | 89.6 | -- | 88.8 |

### 6.2 多模态 Agent 能力（Agentic Multimodal）

| 基准 | Qwen3.8-27B | Qwen3.6-27B | Qwen3.7-Plus | Muse Glimmer-30B | Opus4.6 Max |
| --- | --- | --- | --- | --- | --- |
| 电脑桌面操作（OSWorld-Verified） | **84.3** | 63.9 | 73.3 | 65.9 | 72.7 |
| 浏览器操作（WebArena-Verified） | **64.8** | 48.8 | 55.3 | -- | -- |
| 移动端操作（AndroidWorld） | **81.9** | 70.3 | 81.0 | -- | 62.0 |
| 应用复现（RecreationBench） | **47.1** | 29.8 | 30.2 | -- | -- |
| 多模态工具调用（ClawEval-MM）Pass@3 | **57.4** | 42.6 | 57.4 | -- | 52.5 |
| 多模态工具调用（ClawEval-MM）Average | 56.9 | 50.4 | **60.1** | -- | 54.7 |
| 多模态软件工程（SWE-MM） | **38.6** | 25.7 | 30.0 | -- | 27.1 |
| 视觉网页开发（Vision2Web） | **62.9** | 45.0 | 42.1 | -- | -- |

### 6.3 通用多模态能力（General Multimodal）

`With CI` = 开启代码解释器（Code Interpreter）。

| 基准 | Qwen3.8-27B | Qwen3.6-27B | Qwen3.7-Plus | Muse Glimmer-30B | Opus4.6 Max |
| --- | --- | --- | --- | --- | --- |
| 视觉数学解题（MathVision）无 CI | 90.0 | 85.1 | **90.3** | -- | 65.5 |
| 视觉数学解题（MathVision）With CI | **94.6** | -- | -- | -- | -- |
| 通用视觉推理（BabyVision）无 CI | **65.7** | 28.9 | 64.7 | -- | 12.6 |
| 通用视觉推理（BabyVision）With CI | **85.6** | -- | 70.4 | -- | -- |
| 科学图表解析（CharXiv RQ）无 CI | 83.7 | 78.4 | **85.8** | -- | 66.0 |
| 科学图表解析（CharXiv RQ）With CI | **90.2** | -- | 85.9 | 78.8 | -- |
| 文档智能（OmniDocBench 1.5） | 91.1 | 89.4 | **91.4** | 75.8 | 86.6 |
| 真实世界感知（RealWorldQA） | 85.9 | 84.1 | **86.9** | -- | 73.9 |
| 具身智能（ERQA） | 65.5 | 62.5 | **69.8** | -- | 40.8 |

### 6.4 读表要点

- **强项在 Agent 与工程落地：** SWE-bench Pro、QwenSWEBench、DeepSWE、CoWorkBench、OSWorld、AndroidWorld、WebArena、Vision2Web、SWE-MM 全面第一。**QwenSWEBench 从上一代 49.3 → 79.0** 是全表最大跃升（+29.7），DeepSWE 1.1 更是 13.3 → 42.2（3 倍）。
- **纯知识推理是明确短板：** GPQA Diamond（89.2）与 HLE（30.8）**既落后 Opus4.6 Max（91.3 / 40.0），也落后自家 Qwen3.7-Plus（90.3 / 34.7）**。HLE 差 9.2 分是全表最大逆差——不适合当"百科型专家"用。这与 Muse Glimmer 在 GPQA/HLE 上被双杀的形态一致，是小尺寸开源模型的共性天花板。
- **终端任务尚未登顶：** Terminal Bench 2.1 的 73.0 低于 Opus4.6 Max 的 78.2（−5.2），但已大幅甩开 Qwen3.6-27B（63.4）与 Muse Glimmer（51.7）。
- **代码解释器收益极大：** BabyVision 65.7 → **85.6**（+19.9）、CharXiv RQ 83.7 → **90.2**（+6.5）、MathVision 90.0 → **94.6**（+4.6）。**视觉数理场景务必挂上 CI 工具**，且注意 CharXiv 只有开了 CI 才反超 Qwen3.7-Plus。
- **被自家更大模型压制的都是"静态感知"项：** OmniDocBench、RealWorldQA、ERQA、MathVision（无 CI）四项均小幅落后 Qwen3.7-Plus，差距 0.3–4.3 分。**规律是：越偏动态交互/Agent，27B 越占优；越偏静态感知与知识，参数量越吃紧。**

### 6.5 ⚠️ 跨模型卡数据交叉核对（重要）

本卡引用的 **Muse Glimmer-30B** 分数与 [Muse 自家模型卡](Muse-Glimmer-30B.md) 的自报值**完全一致**（TerminalBench 51.7、SWE-bench Pro 51.2、IFBench 77.0、GPQA 83.5、HLE 22.0、OSWorld 65.9、CharXiv 78.8、OmniDocBench 75.8）——**Qwen 直接采信了对手的自报数字，未做复测。**

反过来，Muse 的卡对 **Qwen3.6-27B** 做了自行复测，与 Qwen 官方数字有出入：

| 基准 | Muse 卡给 Qwen3.6 的分 | 本卡给 Qwen3.6 的分 | 差值 |
| --- | --- | --- | --- |
| OSWorld-Verified | 75.6 | 63.9 | **+11.7** |
| GPQA Diamond | 84.2 | 87.8 | −3.6 |
| TerminalBench 2.1 | 60.7 | 63.4 | −2.7 |
| SWE-Bench Pro | 50.2 | 53.5 | −3.3 |
| IFBench | 70.8 | 69.1 | +1.7 |
| HLE (Text) | 23.1 | 24.0 | −0.9 |

**结论：** Meta 在 OSWorld 上甚至复测出**比 Qwen 官方更高**的分（说明并非刻意压低）。**选型时不要把两张卡的数字混进同一张表比较**，跨卡对比只能看趋势、不能看小数点。

> 另需注意：本卡**未包含 Gemma 4 31B** 作为对比机型，Gemma 卡也未包含 Qwen3.8。三者的直接可比数据只有经由 Muse 卡的间接链路。

---

## 7. 部署与优化

### 7.1 启动命令

模型卡逐条给出了启动命令——**这是它相对 Muse Glimmer 卡的明显优势**（后者一条命令都没给）：

```bash
# vLLM
vllm serve "Qwen/Qwen3.8-27B"

# SGLang
python3 -m sglang.launch_server --model-path "Qwen/Qwen3.8-27B" --host 0.0.0.0 --port 30000

# Docker Model Runner
docker model run hf.co/Qwen/Qwen3.8-27B
```

官方列出的支持框架：**Transformers、vLLM、SGLang、TokenSpeed**。

> ⚠️ 模型卡**未标注任何框架的最低版本号**。但 `config.json` 声明 `transformers_version = 5.8.0.dev0`，意味着 Transformers 路线**必须 dev 版 / 源码安装** `[cfg]`；且 Gated DeltaNet 线性注意力内核的支持进度需按框架逐一验证。

### 7.2 快速上手（Transformers）

模型卡使用 `AutoProcessor` + `AutoModelForMultimodalLM`（与 Gemma 4 31B 相同的多模态类路径）：

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "Qwen/Qwen3.8-27B"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, device_map="auto")
```

多模态输入通过 OpenAI 兼容的 Chat Completions API 传入，`content` 数组使用 `type: "image_url"` 与 `type: "video_url"`。

> 注：上述片段依据模型卡的类名与调用形式整理，**非逐字复制**；完整示例请以模型卡页面为准。

### 7.3 多词元预测（MTP）

| 项目 | 值 |
| --- | --- |
| 模型卡原文 | 仅一句 "trained with **multiple steps**" |
| `mtp_num_hidden_layers` | 1 `[cfg]` |
| `mtp_use_dedicated_embeddings` | `false`（复用主干 embedding）`[cfg]` |

> ⚠️ **不要当成"必然提速"：** 官方对 MTP **没有给出任何加速比或吞吐数字**。实际收益取决于推理框架是否用这个头做投机解码（speculative decoding）。
>
> 对比 Muse Glimmer：后者发布了独立的 **DFlash drafter head** 并给出实测吞吐（RTX 5090 上 74.9 → 233.4 tok/s，3.1×）。**Qwen3.8 在"投机解码可用性"这件事上的交付完整度明显更低**——有结构、无数据、无配套工具。

### 7.4 量化与硬件

| 项目 | 状态 |
| --- | --- |
| 官方量化权重（GGUF / FP8 / AWQ / Int4） | **未发布** |
| 社区量化 | HF 页面 "Browse Quantizations" 列出 **667** 个衍生模型，覆盖 llama.cpp / Ollama / LM Studio / Jan |
| 显存需求 | 模型卡**未给出任何数字** |
| 投机解码 drafter | 未发布（仅内置 MTP 头，见 §7.3） |

> `[推断]` 27B × 2 bytes（BF16）≈ **54 GB 权重**，加上视觉塔、KV/SSM 状态与激活，BF16 全精度实际需求应在 **64 GB 级别**（可参照 Muse Glimmer ~29.6B 官方标注的 64 GB）。**这是笔者估算，官方未背书。**
>
> ⚠️ **本地部署是本卡最大的文档缺口：** Muse Glimmer 逐项给出了 BF16 / 4-bit 两档量化的显存需求与精度损失（64 / 32 / 24 GB，损失 0.2% / 1.0%），Qwen3.8 **完全依赖社区量化**且无官方精度损失数据。消费级硬件场景需自行评估。

### 7.5 云端服务

官方 **Qwen Cloud API** 将提供托管版本，附带生产特性（**默认 1M 上下文**、官方内置工具等），状态为 **coming soon**。

---

## 8. 适用场景与安全

### 8.1 官方声明的适用场景

- **编程与软件工程**：Agent 式代码修复、仓库级生成、终端操作
- **专业办公与研究**：长流程协同办公、专业岗位任务
- **长周期 Agent 任务**：自主规划 + 环境反馈处理
- **多端操作**：电脑桌面 / 浏览器 / 移动端
- **多模态工具调用**
- **文档智能**：复杂文档与报表解析

### 8.2 安全与责任声明

> ⚠️ **模型卡未提供**任何安全章节：无 Safety SFT / Safety RL 说明、无 Preparedness 风险等级、无责任使用条款、无年龄或用途限制声明。
>
> 与两个同级模型对照：Muse Glimmer 给出了完整的训练期安全措施与三域风险等级（Chem/Bio、Cyber、Loss of Control 均 Moderate 或更低）；Gemma 4 至少说明了训练数据经 CSAM 过滤与敏感信息剔除。**Qwen3.8 在合规文档完整度上弱于两者**，面向受监管场景落地时需自行补充评估。

---

## 9. 局限性

- **纯知识推理落后：** GPQA Diamond 与 HLE 同时落后 Opus4.6 Max 和自家 Qwen3.7-Plus（见 §6.4）。
- **静态 YaRN 的短文本代价：** 官方明确承认恒定缩放系数**可能损害短文本表现**，长短混合负载需要跑两套配置或放弃 1M。
- **降低推理强度不一定更快：** 官方提示多轮 Agent 任务中降档可能因试错增多而延长总时长（见 §5）。
- **MTP 无收益数据：** 官方未给出任何加速比，也未发布配套 drafter（见 §7.3）。
- **无官方量化与显存指引：** 本地部署完全依赖社区量化（见 §7.4）。
- **框架版本要求不明：** 未标注最低版本；`transformers` 需 dev 版；Gated DeltaNet 内核支持度需逐框架验证。
- **文档缺项（模型卡未提供）：**

  | 缺失项 | 状态 |
  | --- | --- |
  | 知识截止日期 | 未提供（Gemma 4 给到 2025-01，Muse 给到 2026-01-04） |
  | 支持语言数量 | 未提供（Gemma 4 称 140+ / 开箱 35+，Muse 称 100+） |
  | 视觉编码器规格与参数量 | 未提供（本文 §3.3 由 `config.json` 反推） |
  | 训练数据来源、token 量、训练阶段细分 | 未提供（仅 "Pre-training & Post-training"） |
  | 安全 / 风险等级 / 责任使用 | 未提供（见 §8.2） |
  | 音频支持 | 未提及（推定不支持，与另两个模型相同） |

---

## 10. 选型建议（相对 Gemma 4 31B / Muse Glimmer-30B）

**选它：** 核心需求是 **Agent 与软件工程能力上限**（SWE-bench Pro / OSWorld / AndroidWorld / WebArena 全面第一）；需要 **256K 原生、乃至 1M 扩展**的超长上下文；需要**原生小时级视频**理解；希望用**结构化 API 字段**（`reasoning_effort`）而非 System Prompt 控制推理强度，并需要 `preserve_thinking` 维持多轮 Agent 的推理链路；看重**开箱可用的框架启动命令**。

**别选它：** 显存预算 ≤ 32 GB 且要求官方背书的量化方案（→ Muse Glimmer，24 GB 可跑且标注了精度损失）；需要**投机解码的实测性能保证**（→ Muse Glimmer 的 DFlash 有完整数据）；核心需求是**知识密集型专家问答**（GPQA / HLE 双重落后）；需要**音频输入**（三者均不支持）；需要**明确的安全评级、知识截止与语言覆盖披露**以满足合规审查（→ Muse Glimmer 或 Gemma 4）。

**一句话概括三者分工：** Qwen3.8-27B 拼**能力上限与超长上下文**，Muse Glimmer-30B 拼**消费级硬件可落地**，Gemma 4 31B 拼**多语言覆盖与视觉 Token 预算可调**。

---

## 11. 许可与引用

- **License：** Apache-2.0

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```

> 注：官方 citation 条目的标题写的是 **Qwen3.8-Max**（面向整个 Qwen3.8 发布，而非 27B 单个 checkpoint）；模型卡未提供 arXiv 论文，仅指向官方博客。

---

**数据来源：** [Qwen/Qwen3.8-27B · Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B)（模型卡 + `config.json`，抓取日期：2026-08-20）
