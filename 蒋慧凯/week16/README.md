# Week 16 作业：几个未讲授开源模型的结构特点调研

> 课程重点讲授的旗舰模型：DeepSeek-V3 / V4、GLM-5.2、Kimi-K3、Qwen3.6。  
> 本报告选取 2025 年发布、且课堂上未展开讲解的三个开源模型：Meta **Llama 4**、Google **Gemma 3**、Microsoft **Phi-4-mini / Phi-4-Multimodal**，从结构角度做横向调研。

---

## 1. 调研对象与选取理由

| 模型 | 发布方 / 时间 | 核心结构标签 | 选取原因 |
|------|--------------|-------------|---------|
| Llama 4 | Meta / 2025.04 | MoE + iRoPE + 原生多模态 | Llama 系列首次全面转向 MoE，并凭 iRoPE 把上下文推到 10M 量级 |
| Gemma 3 | Google / 2025.03 | 5:1 局部/全局注意力 + GQA + 蒸馏 | 小尺寸开源多模态模型中，用「局部-全局交错」替代复杂 MLA/线性注意力的典型 |
| Phi-4-mini / Phi-4-Multimodal | Microsoft / 2025.02 | SLM + 25% partial RoPE + Mixture-of-LoRAs | 展示小模型如何在 3.8B~5.6B 规模通过数据与模块化设计实现跨模态能力 |

---

## 2. Meta Llama 4：Llama 家族首次全面 MoE 化

### 2.1 整体定位

Llama 4 是 Meta 首次在 Llama 家族中采用稀疏 MoE 架构的一代，同时实现原生多模态（文本 + 图像/视频 early fusion）。首发两个开放权重版本：

| 版本 | 总参数量 | 激活参数量 | 专家数 | 上下文长度 |
|------|---------|-----------|--------|-----------|
| Scout | ~109B | 17B | 16 | 10M |
| Maverick | ~400B | 17B | 128 | 1M |

（还预告了 2T/288B 激活的 Behemoth 作为教师模型）[[1]](https://siyaz.com.tr/en/blog/meta-llama-4-scout-maverick)[[2]](https://codersera.com/blog/llama-4-complete-guide-2026/)

### 2.2 结构特点

#### （1）MoE：细粒度专家 + 共享专家

- 每层 FFN 被替换为多个前馈「专家」加轻量路由器；每个 token 只激活极少数专家。
- Scout 16 专家、Maverick 128 专家，但激活量均控制在 17B 左右，实现「大记忆、小计算」。
- 保留共享专家（shared expert），兜底通用语言知识与基础推理能力。

#### （2）iRoPE：用 NoPE 层做长上下文「信息高速公路」

Llama 4 把「位置编码按需分配」推到更极致：

- 每 4 层为一组：3 层使用标准 RoPE + **Chunked Local Attention**（8K 局部窗口），1 层使用 **NoPE + Full Attention**（无位置编码、全局可见）。
- NoPE 层允许模型在长程上做「内容匹配」而不受 RoPE 角度外推限制；局部层负责保近邻精度。
- Scout 预训练 256K，Instruct 版通过该技术支持到 **10M token**；Maverick 支持到 1M。这相当于课程中讲的 partial RoPE / NoPE 思想的工业落地版本。  [[3]](https://arxiv.org/pdf/2508.08192)[[4]](https://www.mmntm.net/articles/context-window-race)

#### （3）原生多模态：early fusion

- 文本 token 与视觉 token 在预训练阶段就进入同一个序列联合训练，而非后期冻结视觉编码器再对齐。
- 配合 MetaP 超参数搜索确定初始化尺度，目标是让多模态训练更稳定。

### 2.3 与课堂内容的联系

- 课程中 DS-V4 用 partial RoPE（只旋转 64 维）、Kimi-K3 MLA 层完全 NoPE；Llama 4 的 iRoPE 是另一种更工程化的折中：用层间交错而非头间/维间交错。
- 与 DS-V4 的「压缩 + 稀疏」、Kimi-K3 的「线性注意力」不同，Llama 4 在长文本上走「局部 + 全局」路线，更像 Gemma 3 的思路，但用 NoPE 层替换了全局层。

---

## 3. Google Gemma 3：小模型上的局部-全局注意力工程

### 3.1 整体定位

Gemma 3 是 Google DeepMind 在 2025 年 3 月发布的轻量开源多模态模型，参数规模 1B / 4B / 12B / 27B，主打端侧与低成本部署。与 Llama 4 的「巨兽 MoE」路线相反，它证明**不引入 MLA、不引入线性注意力**，仅靠局部-全局注意力交错就能把上下文扩展到 128K。 [[5]](https://arxiv.org/abs/2503.19786)

### 3.2 结构特点

#### （1）5:1 局部/全局注意力交错

- 每 6 层中：5 层局部注意力 + 1 层全局注意力。
- 局部层使用滑窗注意力，窗口较短（如 1024 token），KV cache 只存窗口内的 token。
- 全局层负责跨段信息汇总；信息通过堆叠的局部层逐层「传播」，虽然不如全局注意力精确，但显著降低长文本推理的显存与计算开销。  [[6]](https://debuggercafe.com/gemma-3-advancing-open-lightweight-multimodal-ai/)[[7]](https://melchi.me/posts/kv-cache/)

#### （2）GQA + QK-Norm + RMSNorm

- 使用 Grouped-Query Attention（GQA）压缩 KV cache。
- 用 QK-Norm 替代 Gemma 2 的 soft-capping，稳定注意力 logits。
- 保持 RMSNorm，符合课程中提到的「LayerNorm → RMSNorm」的标配趋势。

#### （3）原生多模态与蒸馏

- 4B/12B/27B 接入 SigLIP 视觉编码器，1B 为纯文本。
- 训练上大量使用蒸馏：Gemma3-4B-IT 在部分任务上可与 Gemma2-27B-IT 相当；Gemma3-27B-IT 对标 Gemini-1.5-Pro。

### 3.3 与课堂内容的联系

- 课程中 DS-V4 用 CSA/HCA「压缩历史」、Kimi-K3 用 KDA 线性注意力；Gemma 3 走的是第三条更保守的路线——**滑窗局部注意力 + 少量全局层兜底**。
- 它的设计哲学与 GLM-5.2 有相似之处：站在已被验证的机制上（GQA、局部注意力、QK-Norm）做工程压榨，而不赌激进的新范式。

---

## 4. Microsoft Phi-4-mini / Phi-4-Multimodal：小模型的模块化跨模态设计

### 4.1 整体定位

Phi 系列一直强调「小参数 + 高质量数据」。Phi-4-mini 是 3.8B 的 dense decoder-only Transformer，支持 128K 上下文；Phi-4-Multimodal 在其基础上用 **Mixture-of-LoRAs** 扩展出视觉与语音/音频能力，总参数仅约 5.6B。 [[8]](https://arxiv.org/abs/2503.01743)

### 4.2 结构特点

#### （1）语言骨干：dense + GQA + partial RoPE

- 32 层，hidden size 3072，采用 **GQA 24Q/8KV**（KV cache 压缩为 MHA 的 1/3）。
- 使用 **fractional RoPE**：25% 的 head 维度保留位置无关，辅助长上下文外推（LongRoPE 支持 128K）。
- 词汇表扩展到 200K，提升多语言与多模态 token 效率。

#### （2）Mixture-of-LoRAs：冻结 LLM，按需挂载适配器

这是 Phi-4-Multimodal 最核心的结构创新：

- **完全冻结 Phi-4-mini 语言模型**，避免多模态训练对文本能力的灾难性遗忘。
- 视觉任务挂载 LoRA_V，语音/音频任务挂载 LoRA_A，各自配套独立的 encoder + projector。
- 路由器（modality-specific routers）决定激活哪些 LoRA，支持纯文本、图文、图音、纯语音/音频等多种组合，而不会互相干扰。

各模块规模：

| 模块 | 参数量 | 说明 |
|------|--------|------|
| Phi-4-mini 骨干 | 3.8B | 冻结 |
| 视觉 encoder (SigLIP-400M) + projector | 440M | 可训练 |
| 视觉 LoRA_V | 370M | 可训练 |
| 语音/音频 encoder (Conformer) + projector | 460M | 可训练 |
| 语音/音频 LoRA_A | 460M | 可训练 |
| 总计（Phi-4-Multimodal） | ~5.6B | 相对同能力模型极轻量 |

### 4.3 训练阶段

- 语言模型先在 5T 高质量 token 上预训练，强调数学、代码、合成数据。
- 多模态阶段：冻结语言模型 → 分别训练视觉/语音模块 → 最后做图文-语音联合训练。
- 实验版 Phi-4-mini-reasoning 还展示了小模型做长思维链蒸馏的可能性。

### 4.4 与课堂内容的联系

- 课程中多模态部分（PPT 6.4）讲 ViT → projector → LLM 的通用三步；Phi-4-Multimodal 把 projector 升级为「LoRA + 路由器」组合，让同一 LLM 在不改权重的情况下服务多种模态。
- 与 Qwen3.6/Kimi-K3 的「原生联合训练」不同，Phi-4 走「冻结骨干 + 轻量适配器」路线，更适合端侧增量扩展新模态。

---

## 5. 横向对比：三条路线的取舍

| 维度 | Llama 4 | Gemma 3 | Phi-4-mini / Multimodal |
|------|---------|---------|--------------------------|
| **规模路线** | 稀疏 MoE（109B~400B 总参） | Dense（1B~27B） | Dense + LoRA（3.8B~5.6B） |
| **注意力策略** | iRoPE：3 层 RoPE 局部 + 1 层 NoPE 全局 | 5 层局部滑窗 + 1 层全局 | 标准 GQA + partial RoPE 25% |
| **长上下文** | Scout 10M / Maverick 1M | 128K（4B/12B/27B） | 128K（LongRoPE） |
| **MoE 化** | 是，首次在 Llama 使用 | 否 | 否 |
| **多模态方式** | early fusion，联合预训练 | SigLIP + projector | 冻结 LLM + Mixture-of-LoRAs |
| **位置编码** | 层间 RoPE/NoPE 交错 | 标准 RoPE（局部层用短窗） | fractional RoPE（25% 无位置） |
| **设计哲学** | 用稀疏 MoE 支撑大记忆与多模态 | 用成熟组件做工程压榨 | 小模型 + 模块化适配器 |

---

## 6. 个人小结

1. **注意力路线没有银弹**：课堂上 DS-V4 走压缩（CSA/HCA）、Kimi-K3 走线性注意力（KDA）、GLM-5.2 走稀疏选择（DSA/IndexShare）；课外 Llama 4 和 Gemma 3 又展示了「局部 + 全局」交错这条更保守、但工程可控性更强的路线。

2. **MoE 已成旗舰标配，但不是唯一选择**：Llama 4 证明 Llama 家族必须 MoE 化才能进入 100B+ 总参与原生多模态；但 Gemma 3 / Phi-4-mini 说明 dense 小模型仍可通过数据质量与结构优化（局部-全局、GQA、partial RoPE）在各自定位上保持竞争力。

3. **多模态接口正在分化**：
   - 原生联合训练（Llama 4、Qwen3.6、Kimi-K3）追求端到端能力上限；
   - 冻结骨干 + LoRA/适配器（Phi-4-Multimodal）追求可扩展性与端侧部署。
   这与课程中 PPT 6.4 提到的「从外挂对齐到原生联合训练」并不矛盾，只是不同资源约束下的两种极值。

4. **位置编码越来越像「可调旋钮」**：从 RoPE → YaRN → partial RoPE → NoPE → iRoPE，模型不再统一给所有层/所有头加位置信息，而是按层、按头、按维度甚至按任务需要分配位置感知能力。

---

## 参考资料

1. [Meta Llama 4 Scout and Maverick: 10 Million Token Context with MoE Architecture](https://siyaz.com.tr/en/blog/meta-llama-4-scout-maverick) —— 参数与上下文概述
2. [Llama 4 Guide: Scout, Maverick, Behemoth Status & Muse Spark (2026)](https://codersera.com/blog/llama-4-complete-guide-2026/) —— iRoPE 与 MoE 细节
3. [arXiv:2508.08192 — iRoPE for Llama 4](https://arxiv.org/pdf/2508.08192) —— iRoPE 与推理实现
4. [The Context Window Race: Why 10 Million Tokens Doesn't Mean 10 Million Useful Tokens](https://www.mmntm.net/articles/context-window-race) —— iRoPE 3:1 局部/全局分析
5. [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) —— 官方技术报告
6. [Gemma 3 - Advancing Open, Lightweight, Multimodal AI](https://debuggercafe.com/gemma-3-advancing-open-lightweight-multimodal-ai/) —— 5:1 局部/全局注意力
7. [Understanding KV Cache: The Hidden Memory Cost of Serving LLMs](https://melchi.me/posts/kv-cache/) —— Gemma 3 KV cache 工程分析
8. [Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs](https://arxiv.org/abs/2503.01743) —— 官方技术报告
