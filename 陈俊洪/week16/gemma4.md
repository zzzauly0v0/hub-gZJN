---
title: Gemma 4 31B-it
model_id: google/gemma-4-31B-it
publisher: Google DeepMind
license: Apache-2.0
source: https://huggingface.co/google/gemma-4-31B-it
---

# Gemma 4 31B-it

> Gemma 4 家族中的**稠密（Dense）旗舰版本**，指令微调（it）模型，支持文本 + 图像输入并生成文本，原生 256K 上下文。

## 1. 架构类型与模型定位

- **密集型架构（Dense Model）：** 属于 Gemma 4 家族的稠密旗舰版本（非 MoE 架构），推理时激活全部参数。

- **家族定位：** Gemma 4 提供 **E2B**、**E4B**、**12B Unified**、**26B A4B（MoE）** 与 **31B Dense** 五个规格，本模型为其中参数量最大的稠密版本。

- **多模态输入支持：** 支持 **文本（Text）** 与 **图像（Image）** 输入，视频可按帧（frame-by-frame）处理后送入。需要注意的是，31B 版本**未集成音频编码器**——音频输入仅在 E2B、E4B、12B 版本原生支持（单段最长 30 秒）。

- **语言覆盖：** 预训练覆盖 **140+ 种语言**，开箱可用（out-of-the-box）支持 **35+ 种语言**。

- **知识截止：** 2025 年 1 月。

- **开源协议：** 采用 Apache 2.0 开源许可。

---

## 2. 模型规模与核心维度参数

| 项目 | 数值 |
| --- | --- |
| 总参数量（Total Parameters） | 30.7B（约 307 亿） |
| 网络层数（Layers） | 60 |
| 词表大小（Vocabulary Size） | 262K |
| 视觉编码器（Vision Encoder） | 约 550M |
| 上下文长度（Context Length） | 原生 **256K tokens** |
| 权重精度 | BF16（safetensors） |

---

## 3. 注意力机制设计

- **混合注意力结构（Hybrid Attention）：** 采用交替结构，将局部滑动窗口注意力（Sliding Window Attention）与全局注意力（Full Global Attention）结合，且**最后一层始终保持全局注意力**。

- **滑动窗口大小：** 1024 tokens。局部窗口降低了显存开销与计算复杂度，全局层则保证长程依赖的捕捉。

- **显存与长文本优化：** 全局注意力层采用统一的 Keys/Values 设计，并引入 **Proportional RoPE (p-RoPE)** 优化长文本推理。

---

## 4. 关键技术与特色功能

### 4.1 可配置的思考机制（Thinking Mode）

- 支持内置思维链推理，通过在 System Prompt 前添加 `<|think|>` 触发；使用 `apply_chat_template` 时对应 `enable_thinking=True/False`。

- 输出结构为 `<|channel>thought\n[推理过程]<channel|>[最终回答]`。

- **多轮对话规范：** 历史轮次只应保留**最终回答**，上一轮的思考过程不得拼接到下一轮用户输入之前；**唯一例外是工具调用（Tool Call）轮次**——此时思考内容需要保留。

### 4.2 动态视觉 Token 预算（Variable Image Resolution）

- 支持多种纵横比与分辨率，图像 Token 预算可配置：**70 / 140 / 280 / 560 / 1120 tokens**。

- 细粒度任务（OCR、复杂文档解析、图表理解）分配高预算；通用图像理解与视频逐帧处理可使用低预算以提升吞吐。

### 4.3 推荐采样参数

```
temperature = 1.0
top_p = 0.95
top_k = 64
```

---

## 5. 基准测试表现

### 5.1 推理与代码

| Benchmark | 成绩 |
| --- | --- |
| AIME 2026（无工具） | 89.2% |
| LiveCodeBench v6 | 80.0% |
| Codeforces ELO | 2150 |
| GPQA Diamond | 84.3% |
| MMLU Pro | 85.2% |

> Codeforces ELO 2150 与 AIME 89.2% 表明其算法与复杂编程能力处于同规模开源模型的顶尖水平。

### 5.2 多模态

| Benchmark | 成绩 |
| --- | --- |
| MMMU Pro（视觉推理） | 76.9% |
| MATH-Vision | 85.6% |
| OmniDocBench 1.5（文档解析，越低越好） | 0.131 |

### 5.3 多语言与长上下文

| Benchmark | 成绩 |
| --- | --- |
| MMMLU（多语言） | 88.4% |
| MRCR v2（8 needle @ 128K） | 66.4% |

---

## 6. 快速上手

依赖：`transformers`（需包含 `AutoModelForMultimodalLM` 的版本）、`torch`、`accelerate`、`torchvision`。

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-31B-it"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, dtype="auto", device_map="auto"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True,
    return_tensors="pt", add_generation_prompt=True,
    enable_thinking=False,   # 置为 True 开启思考模式
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

---

## 7. 局限性

- 对**微妙语义、讽刺与比喻**的理解仍可能出错。

- 受训练数据分布影响，可能生成**不准确或过时的事实性陈述**（知识截止为 2025 年 1 月）。

- 31B Dense 不支持音频输入；视频需自行抽帧。

- 训练数据经过 CSAM 过滤、敏感信息剔除与质量/安全过滤，但不构成对输出安全性的保证。

---

**数据来源：** [google/gemma-4-31B-it · Hugging Face](https://huggingface.co/google/gemma-4-31B-it)（抓取日期：2026-08-20）
