---
title: Muse Glimmer-30B
model_id: meta-models/Muse-Glimmer-30B
publisher: Meta Superintelligence Lab
license: Apache-2.0
source: https://huggingface.co/meta-models/Muse-Glimmer-30B
---

# Muse Glimmer-30B

> Meta Superintelligence Lab 于 2026 年 8 月发布的**稠密（Dense）多模态 Agent 模型**，官方定位一句话概括为"purpose-built for autonomous agentic tasks on **consumer hardware**"——即**为消费级硬件上的端到端自主 Agent 任务而生**。文本 + 图像输入、文本输出，原生 128K 上下文。
>
> 标注说明：带 `[cfg]` 的条目取自仓库 `config.json`，模型卡正文未直接列出；带 `[推断]` 的为笔者依据配置做的解读，非官方表述。

---

## 1. 模型定位与核心亮点

- **本地优先（Local-first）是第一性设计目标：** 与 Qwen3.8、Gemma 4 把"能力上限"作为卖点不同，这张模型卡的叙事重心是**不依赖云基础设施也能跑完整 Agent 流程**。量化版本与投机解码 drafter 是与主权重**同批发布**的一等公民，而非社区后补产物。

- **Agent 能力打包交付：** 官方强调把**多步推理、工具调用、多模态理解、失败恢复（failure recovery）**四件事整合进同一个模型，而不是只优化单轮问答。"失败恢复"被单独列为亮点，指向长流程任务中出错后自我纠偏的能力。

- **来自更大模型的蒸馏：** Glimmer 由家族中更大的 **Muse Spark** 蒸馏而来（模型卡在风险等级章节提到 "Muse Spark 1.0"）。因此它的定位是旗舰模型的**可本地部署下放版**，而非独立训练的小模型。

- **可控推理强度：** 支持 `low / medium / high / xhigh` 四档，通过 **System Prompt 文本**指定（详见 §5），官方推荐复杂问题、编程与 Agent 任务用 `high` 或 `xhigh`。

- **知识截止：** 2026 年 1 月 4 日（精确到日，比多数模型卡给的"某年某月"更细）。

- **语言覆盖：** 100+ 种语言。

- **开源协议：** Apache 2.0，且**全部产物**（BF16 权重、两个 4-bit 量化版、DFlash drafter、冻结的视觉编码器）统一采用该协议。

> ⚠️ **参数量口径需注意：** 模型卡正文自称 "a **30-billion-parameter** causal language model"，规格表写 **~29.6B**（说明含视觉编码器），而视觉塔又被单列为 **~1.8B**。三个数字无法严格自洽，引用时建议直接写"约 30B 级别、含 ~1.8B 视觉编码器"，不要把 29.6B 当作纯语言主干的参数量。

---

## 2. 模型规模与核心维度参数

| 项目 | 数值 |
| --- | --- |
| 总参数量（含视觉编码器） | ~29.6B |
| 网络层数（Layers） | 52 |
| 隐藏层维度（hidden_size） | 6,656 |
| FFN 中间维度（intermediate_size） | 19,968（= 3 × hidden_size）`[cfg]` |
| 词表大小（vocab_size） | 202,048 |
| 注意力头数 | Q：32；KV：2（GQA，16:1 极致压缩） |
| 单头维度（head_dim） | 128 `[cfg]` |
| FFN 类型 | SwiGLU（`hidden_activation = silu`）`[cfg]` |
| 上下文长度 | 131,072 tokens（`max_position_embeddings = 131072`） |
| 视觉编码器 | ~1.8B，ViT-G/14，**训练中冻结** |
| 单图最大视觉 Token | 4,096 |
| 权重精度 | BF16 `[cfg]` |
| 输入 / 输出模态 | 文本 + 图像 / 文本 |
| Embedding 权重共享 | 否（`tie_word_embeddings = false`）`[cfg]` |
| 特殊 token id | bos 200000 / eos 200001 / image 200092 / video 200091 `[cfg]` |
| 依赖版本声明 | `transformers_version = 5.15.0.dev0` `[cfg]` |

**几个容易被忽略的稳定性相关字段** `[cfg]`：

| 字段 | 值 | 作用 |
| --- | --- | --- |
| `final_logit_softcapping` | 20.0 | 输出 logits 软截断，抑制极端分布 |
| `qk_scale_factor` | 3.87 | QK 缩放系数，非默认 `1/√d` |
| `output_multiplier` | 0.19611613 | 输出缩放（≈ 1/√26） |
| `post_norm_eps` | 1e-08 | Post-Norm 的 eps，比 `rms_norm_eps`（1e-05）小三个数量级 |

> ⚠️ 这几个字段意味着**自行实现推理内核时不能照抄标准 Llama 结构**，softcapping 与两套不同的 norm eps 都必须还原，否则长输出容易出现数值漂移。

> ⚠️ **配置疑点：** 顶层 `out_hidden_size = 6144`，而文本侧 `hidden_size = 6656`，二者不一致（`projector_hidden_size = 4096`）。按常理视觉投影输出应对齐语言模型维度，此处存疑，接入自定义推理栈前建议以实际权重 shape 为准。`[cfg]`

---

## 3. 注意力机制设计

### 3.1 局部 / 全局交替布局

52 层采用固定的 4 层一组循环：

```
13 × ( 3 × sliding_attention → 1 × full_attention )
```

即每 4 层中前 3 层为**滑动窗口注意力**（`sliding_window = 2048`），第 4 层为**全局注意力**，全局层共 13 层 `[cfg]`。设计意图与 Gemma 4 的混合注意力一致：用小窗口层承担绝大部分计算以压低 KV Cache 与长序列开销，再由周期性全局层兜住长程依赖。

> 与 Gemma 4 31B 的差异：Gemma 4 窗口为 1024 且**最后一层强制全局**；Muse Glimmer 窗口更大（2048），但最后一层是 `full_attention`（第 52 层恰好落在循环的第 4 位）`[cfg]`。

### 3.2 全局层不加位置编码（NoPE）

`layer_rope_theta` 是一个长度 52 的数组：滑动窗口层为 `500000.0`，**全局层为 `0`** `[cfg]`。

- 也就是说：局部层用 RoPE（θ = 500,000），**全局层完全不施加旋转位置编码**。
- `[推断]` 这与 Meta 在 Llama 4 上使用的 **iRoPE**（interleaved RoPE + NoPE 全局层）思路一致——去掉全局层的位置先验，让长上下文外推不受 RoPE 频率衰减限制。模型卡把上下文写作 "131,072**+**"（带加号），大概也是暗示该结构具备一定超原生长度的外推余量，但**官方未给出任何 YaRN / RoPE Scaling 配置，也未承诺具体可用长度**，超 128K 使用请自行评估。

### 3.3 视觉编码器结构 `[cfg]`

模型卡正文只给了 "~1.8B ViT-G/14"，以下由 `vision_config` 得出：

| 项目 | 数值 |
| --- | --- |
| 层数 | 50 |
| 隐藏维度 | 1,536 |
| 注意力头数 | 16（head_dim = 96） |
| FFN 中间维度 | 8,960 |
| Patch 大小 | 14 |
| 注意力布局 | `3 × window_attention → 1 × full_attention` 循环，**末层为 full_attention** |
| 空间合并 / 时间 patch | `merge_size = 2`；`patch_temporal = 2` |
| 位置嵌入网格 | 32 × 32（`max_position_embeddings = 1024`） |
| RoPE θ | 10,000 |

有趣的是**视觉塔复用了与语言模型相同的"局部 3 + 全局 1"哲学**，说明这套混合注意力被当作跨模态统一的架构选择。`patch_temporal = 2` 表明结构上留了时间维度合并能力，但官方明确**未针对视频优化**（见 §8）。

---

## 4. 上下文长度

- **原生上下文：** 131,072 tokens。
- **扩展方案：** 模型卡**未提供** YaRN 等 RoPE 缩放配置，也未声明可扩展上限。
- **长上下文实测成绩：** `AA-LCR` 80.0、`Beam128K` 65.1，两项均为三方对比中的最高分（见 §6.4）——在 128K 级别的实际检索与推理任务上，它是这一档里表现最好的。

> 对比参考：Qwen3.8-27B 原生 256K、可 YaRN 扩到 1M；Gemma 4 31B 原生 256K。**Muse Glimmer 的 128K 是三者中最短的**，长上下文是它为"塞进消费级显存"付出的代价之一。但从 AA-LCR / Beam128K 看，它在自己支持的长度内**利用率更高**。

---

## 5. 推理强度控制

| 项目 | 说明 |
| --- | --- |
| 控制方式 | **System Prompt 文本行**：`Reasoning strength: <value>` |
| 可选档位 | `low` / `medium` / `high` / `xhigh` |
| 官方推荐 | 复杂问题求解、编程、Agent 任务用 `high` 或 `xhigh` |

> ⚠️ **与 Qwen3.8 的关键差异：** Qwen3.8 的 `reasoning_effort` 是**独立请求字段**，Muse Glimmer 则是**写进 System Prompt 的自然语言指令**。这意味着：
> - 不需要推理框架专门支持，任何 OpenAI 兼容端点都能用；
> - 但它**占用 System Prompt 空间**，且与用户自定义 System Prompt 拼接时需要自行约定顺序，容易被后续指令冲淡；
> - 模型卡**未提供** `preserve_thinking` 之类的历史思考保留开关，多轮 Agent 场景下的思考块处理策略需自行验证。

**推荐采样参数：**

```
temperature = 1.0
top_p       = 0.95
top_k       = 64
```

> 与 Gemma 4 31B 的推荐值完全相同；Qwen3.8 思考模式为 `temperature=1.0 / top_p=0.95 / top_k=20`，差别主要在 `top_k`。

---

## 6. 基准测试表现

对比机型：**Gemma4-31B（Thinking Mode）**、**Qwen3.6-27B（Thinking Mode）**；Muse Glimmer 一列为 **High Reasoning** 档。**加粗**为该行最高分。

### 6.1 通用 Agent 能力

| 基准 | Muse Glimmer-30B | Gemma4-31B | Qwen3.6-27B |
| --- | --- | --- | --- |
| MCP 工具调用（MCP Atlas, Public） | **75.5** | 54.2 | 62.5 |
| 深度检索问答（DeepSearch QA） | **74.6** | 61.7 | 71.1 |
| 多轮业务对话（τ3-Banking） | **23.5** | 15.1 | 16.7 |
| 综合工具任务（WildClawBench） | **47.6** | 37.6 | 43.2 |
| 经济价值任务（GDPVal-AA v2） | 953 | 811 | **1141** |
| Agent 综合（Gaia2） | **43.3** | 36.4 | 40.0 |
| 技能型任务（SkillsBench） | 44.3 | 32.4 | **46.6** |
| 电脑桌面操作（OSWorld-Verified） | 65.9 | 58.5 | **75.6** |

### 6.2 Agent 编程能力

| 基准 | Muse Glimmer-30B | Gemma4-31B | Qwen3.6-27B |
| --- | --- | --- | --- |
| SWE-Bench Pro | **51.2** | 36.9 | 50.2 |
| SWE-Bench Verified | 76.0 | 66.6 | **77.2** |
| 终端 Agent（TerminalBench 2.1） | 51.7 | 43.4 | **60.7** |
| 科研代码（SciCode） | **43.6** | 43.4 | 39.8 |

### 6.3 多模态能力

| 基准 | Muse Glimmer-30B | Gemma4-31B | Qwen3.6-27B |
| --- | --- | --- | --- |
| 科学图表推理（Charxiv Reasoning） | **78.8** | 77.7 | 78.4 |
| GUI 元素定位（ScreenSpot Pro） | 75.4 | 75.9 | **76.1** |
| 文档解析（OmniDocBench v1.5） | 75.8 | 72.5 | **77.8** |
| 学科视觉推理（MMMU Pro） | 74 | 73 | **75** |

### 6.4 通用能力与推理

| 基准 | Muse Glimmer-30B | Gemma4-31B | Qwen3.6-27B |
| --- | --- | --- | --- |
| 复杂指令遵循（IFBench） | **77.0** | 76.0 | 70.8 |
| 数学竞赛（AIME 2026） | **94.7** | 89.2 | 94.1 |
| 专家级科学推理（GPQA Diamond） | 83.5 | **85.7** | 84.2 |
| 多学科难题（HLE Text） | 22.0 | **23.6** | 23.1 |
| 长上下文推理（AA-LCR） | **80.0** | 68.3 | 73.3 |
| 长上下文（Beam128K） | **65.1** | 58.2 | 63.0 |

### 6.5 读表要点

- **强项在"工具调用 + 检索"这条链上：** MCP Atlas（75.5 vs 62.5/54.2）、DeepSearch QA、WildClawBench、Gaia2、τ3-Banking 全面领先，且 MCP Atlas 领先第二名 13 分——这是全表最大优势项，与"本地 Agent"的定位完全吻合。
- **纯知识推理是明确短板：** GPQA Diamond（83.5）与 HLE Text（22.0）**同时低于两个对手**，是全表唯一被双杀的能力项。它不适合当"百科型专家"用。
- **编程能力分化明显：** SWE-Bench Pro 领先（51.2），但 SWE-Bench Verified 与 TerminalBench 2.1 落后 Qwen3.6 一个身位（尤其终端任务 51.7 vs 60.7）。**倾向于"能改复杂仓库、但不擅长长链终端操作"**。
- **多模态属于"够用不出彩"：** 四项里只赢下 CharXiv 且优势不足 1 分，ScreenSpot Pro / OmniDocBench / MMMU Pro 全部小幅落后。视觉塔是冻结的 ViT-G/14，没有为多模态做深度联合优化，这个结果合理。
- **长上下文利用率是隐藏亮点：** 上下文长度最短（128K vs 256K），但 AA-LCR 与 Beam128K 双双最高。**"窗口短但用得实"**，长文档任务不必因为 128K 就直接排除它。
- **GDPVal-AA v2 需谨慎解读：** 953 vs Qwen3.6 的 1141，该指标量纲与其余百分制不同（类 ELO/评分），差距幅度不能与百分点差直接类比。

### 6.6 ⚠️ 跨模型卡数据交叉核对（重要）

把本卡数字与 [Qwen3.8-27B 模型卡](Qwen3.8.md) 对照，发现**同一个 Qwen3.6-27B 在两张卡上分数不一致**：

| 基准 | 本卡给 Qwen3.6 的分 | Qwen 自家卡给 Qwen3.6 的分 | 差值 |
| --- | --- | --- | --- |
| OSWorld-Verified | 75.6 | 63.9 | **+11.7** |
| GPQA Diamond | 84.2 | 87.8 | −3.6 |
| TerminalBench 2.1 | 60.7 | 63.4 | −2.7 |
| SWE-Bench Pro | 50.2 | 53.5 | −3.3 |
| IFBench | 70.8 | 69.1 | +1.7 |
| HLE (Text) | 23.1 | 24.0 | −0.9 |

而反过来，**Qwen3.8 卡里引用的 Muse Glimmer 分数与本卡自报值完全一致**（TerminalBench 51.7、SWE-Bench Pro 51.2、IFBench 77.0、GPQA 83.5、HLE 22.0、OSWorld 65.9、CharXiv 78.8、OmniDocBench 75.8）。

**结论：** Qwen 直接引用了 Muse 的自报数字，而 Meta 对 Qwen3.6 做了自行复测（OSWorld 上甚至复测出**比 Qwen 官方更高**的分数，说明并非刻意压低）。选型时**不要把两张卡的数字混在一张表里比**，跨卡对比只能看趋势不能看小数点。

---

## 7. 本地部署与优化（本模型最大差异点）

### 7.1 量化版本与显存需求

| 版本 | 显存需求 | 精度损失 |
| --- | --- | --- |
| Full Precision（BF16） | 64 GB VRAM | — |
| K-Quant-Dynamic（4-bit） | **32 GB VRAM** | 0.2% |
| K-Quant-17GB（4-bit） | **24 GB VRAM** | 1.0% |

> 这是它区别于 Qwen3.8 / Gemma 4 的核心竞争力：**24GB 单卡（RTX 4090 / 5090）即可跑完整多模态 Agent**，且官方标注了精度损失量级。K-Quant-Dynamic 用 0.2% 的代价换掉一半显存，性价比极高——**除非有 64GB 以上显存，否则默认就该用它**。

### 7.2 投机解码（DFlash）

官方同时发布了配套的 **DFlash drafter head**，配合量化 drafter 的实测吞吐：

| 硬件 | 基线 | 启用 DFlash | 加速比 |
| --- | --- | --- | --- |
| Nvidia RTX 5090 | 74.9 tok/s | 233.4 tok/s | **3.1×** |
| Apple M4 Max | 23.7 tok/s | 37.8 tok/s | 1.5× |
| Apple M5 Max | 26.6 tok/s | 50.2 tok/s | 1.8× |

**读数要点：**

- **加速比与硬件强绑定：** N 卡 3.1× 而 Apple Silicon 仅 1.5–1.8×。投机解码收益依赖并行验证的算力冗余，带宽受限的统一内存平台拿不到同等红利，**不要按 3.1× 去估 Mac 上的体验**。
- **绝对吞吐差距更值得关注：** RTX 5090 开启后 233 tok/s，M4 Max 仅 37.8 tok/s，相差 6 倍。Mac 端即使开了 DFlash 也只是"可用"级别。
- Nvidia 侧数据由 **llama.cpp** 测得，Apple 侧由 **ExecuTorch** 测得——**两组数字来自不同推理栈，严格来说不可直接横比**。

### 7.3 部署路径

模型卡列出的可用路径：**vLLM、SGLang、Docker，以及本地应用 llama.cpp / Ollama / LM Studio**。

> ⚠️ **文档缺口：** 与 Qwen3.8 模型卡逐条给出 `vllm serve ...` / `sglang.launch_server ...` 启动命令不同，**本卡没有提供任何框架的具体启动命令，正文也没有 Python 推理示例**。llama.cpp 与 ExecuTorch 只是作为性能测试工具被提及，并非附带教程。落地时需要自行摸索参数，尤其考虑到 `config.json` 声明 `transformers_version = 5.15.0.dev0`（**必须 dev 版 / 源码安装 Transformers**）以及 §2 提到的 softcapping 等非标准字段，**各推理框架的支持进度需要逐一验证**。

Hugging Face 页面的 "Use this model" 面板给出的最小调用形式为：

```python
from transformers import pipeline

pipe = pipeline("image-text-to-text", model="meta-models/Muse-Glimmer-30B")
```

> 注：这段来自 HF 平台自动生成的 UI 面板，**不是模型卡正文内容**，仅表明其 task 类型为 `image-text-to-text`。

---

## 8. 适用场景与安全

### 8.1 官方声明的适用场景

面向**商业与研究用途**，具体包括：

- **本地 AI Agent**：多步规划类任务
- **编程 Agent**：调试与软件工程
- **工具调用 / Function Calling**
- **多模态推理**：带视觉内容的推理任务
- **合成数据生成**
- **LLM-as-judge**：作为评估裁判

### 8.2 训练期安全措施

- **Safety SFT**：基于精选样本的安全监督微调
- **Safety RL**：使用安全专项奖励的强化学习
- **信息流原则内化进权重**：数据敏感性识别、最小化采集、**本地优先执行（local-first execution）**

### 8.3 Preparedness 风险等级

| 风险域 | 等级 |
| --- | --- |
| 化学 / 生物（Chem/Bio） | Moderate 或更低 |
| 网络安全（Cyber） | Moderate 或更低 |
| 失控风险（Loss of Control） | Moderate 或更低 |

> 注：该评级挂在 **Muse Spark 1.0**（上游更大模型）名下，Glimmer 作为蒸馏产物继承该结论。

### 8.4 已发布产物

| 产物 | 说明 |
| --- | --- |
| Full-precision weights | BF16 |
| 4-bit 量化权重 | 2 个变体（K-Quant-Dynamic / K-Quant-17GB） |
| DFlash drafter head | 投机解码用 |
| Perception Encoder | ~1.8B，训练中冻结 |

---

## 9. 局限性

- **不支持音频：** 音频输入与输出均**明确排除**在支持范围外（与 Gemma 4 31B 相同）。
- **未针对视频优化：** 视频只能**逐帧**作为图像处理，尽管 `patch_temporal = 2` 在结构上留了时间维度合并能力 `[cfg]`。
- **可能产生不准确、有偏或令人反感的输出**；多步推理链条中也可能出错。
- **多语言表现不均：** 支持 100+ 语言，但在支持列表外的语言上性能会下降。
- **量化有边界代价：** 官方承认量化版本在**边缘 case** 上存在细微质量差异（对应 §7.1 的 0.2% / 1.0%）。
- **知识截止 2026 年 1 月 4 日**之后的事实需外部检索补足。
- **不面向 18 岁以下用户。**
- **禁止用于违反适用法律的任何用途。**
- **训练数据来源较模糊：** 原文为"公开可得数据、第三方提供数据，以及来自 **Meta 产品与服务的信息**，由外部供应商网络清洗与增强"。**未披露训练 token 量、训练算力，也未说明预训练 / SFT / RL 的阶段划分**（仅在安全章节提到 Safety SFT 与 Safety RL）。

---

## 10. 选型建议（相对 Qwen3.8-27B / Gemma 4 31B）

**选它：** 显存预算 ≤ 32GB 且要跑完整多模态 Agent；核心需求是 MCP / 工具调用 / 深度检索；需要官方背书的量化与投机解码方案；重视本地优先的数据不出境。

**别选它：** 需要知识密集型专家问答（GPQA / HLE 双输）；需要 256K 以上上下文；主要做长链终端操作（TerminalBench 落后 9 分）；需要音频或原生视频；**期望开箱即用的部署文档**（本卡在这点上明显弱于 Qwen3.8）。

---

## 11. 许可与引用

- **License：** Apache-2.0（覆盖全部已发布产物）

模型卡引用的两篇论文：

| 论文 | arXiv | 用途 |
| --- | --- | --- |
| Perception Encoder | [2504.13181](https://arxiv.org/abs/2504.13181) | 视觉编码器（ViT-G/14） |
| DFlash | [2602.06036](https://arxiv.org/abs/2602.06036) | 投机解码 drafter |

---

**数据来源：** [meta-models/Muse-Glimmer-30B · Hugging Face](https://huggingface.co/meta-models/Muse-Glimmer-30B)（模型卡 + `config.json`，抓取日期：2026-08-20）
