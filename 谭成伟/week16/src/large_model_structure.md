# 国产开源大模型结构分析

> 分析对象：Qwen3.8-27B、MiniMax-M3、Hunyuan Hy3
> 数据来源：ModelScope 官方仓库的 config.json / 配置类代码 / README
> 本地结构代码路径：D:/trae-project/checkreservation/model-structure-research/

---

## 一、Qwen3.8-27B（阿里通义千问）

### 1.1 基本信息

| 属性 | 值 |
|---|---|
| 模型类 | `Qwen3_5ForConditionalGeneration` |
| model_type | `qwen3_5` |
| 模态 | 原生多模态（图 + 视频 + 文） |
| 架构类型 | dense（非 MoE） |
| 总参数 | 27B |
| 上下文 | 262K 原生，YaRN 外推至 1M |
| 授权 | Apache 2.0 |
| ModelScope ID | `Qwen/Qwen3.8-27B` |

### 1.2 整体架构：多模态三段式

顶层类是 `ForConditionalGeneration`（非 `ForCausalLM`），属于条件生成式多模态架构，由视觉编码器、投影层和语言模型三部分组成。

视觉编码器采用 ViT 结构，27 层、hidden 1152、patch 16。`temporal_patch_size: 2` 表示时间维 2 帧合并，原生支持视频输入；`spatial_merge_size: 2` 做 2x2 patch 空间合并以降低视觉 token 数量。`out_hidden_size: 5120` 与文本 hidden_size 完全一致，视觉特征可直接拼入文本序列。视觉特征通过 `image_token_id` / `video_token_id` / `vision_start_token_id` / `vision_end_token_id` 等特殊 token 插入文本流，是 Qwen-VL 系列一贯做法。

```json
"vision_config": {
    "depth": 27,
    "hidden_size": 1152,
    "num_heads": 16,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "out_hidden_size": 5120
}
```

### 1.3 语言模型核心：混合线性注意力 + 全注意力

64 层语言模型采用 3:1 的线性注意力与全注意力混合模式，`layer_types` 数组以 3 个 `linear_attention` 加 1 个 `full_attention` 循环 16 组构成 64 层，`full_attention_interval: 4` 印证每 4 层插入一层全注意力。README 原文描述为 `16 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))`。

```json
"layer_types": [
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ...
]
"full_attention_interval": 4
```

线性注意力层（Gated DeltaNet）配置 16 个 QK 头加 48 个 V 头，head_dim 128。`linear_conv_kernel_dim: 4` 用 1D 卷积做短程衰减建模，`mamba_ssm_dtype: float32` 暗示引入了 Mamba/SSM 风格的状态空间机制。线性注意力复杂度为 O(n)，长序列下显著节省 KV cache。

```json
"linear_num_key_heads": 16,
"linear_num_value_heads": 48,
"linear_key_head_dim": 128,
"linear_value_head_dim": 128,
"linear_conv_kernel_dim": 4,
"mamba_ssm_dtype": "float32"
```

全注意力层（Gated Attention）配置 24 头 Q 与 4 个 KV 头（GQA，6 倍分组共享），head_dim 256。每 4 层插一层全注意力保证全局信息混合，弥补线性注意力表达力不足。`attn_output_gate: true` 配合 `output_gate_type: swish` 给注意力输出加门控（GLU 风格），用于稳定混合架构训练。

```json
"num_attention_heads": 24,
"num_key_value_heads": 4,
"head_dim": 256,
"attn_output_gate": true,
"output_gate_type": "swish"
```

### 1.4 位置编码：M-RoPE

位置编码采用 M-RoPE（多模态旋转位置编码），`mrope_section: [11, 11, 10]` 把 RoPE 维度分成三段，分别对应视频的时间 T、高度 H、宽度 W 三个维度，天然支持视频时空建模。`partial_rotary_factor: 0.25` 表示仅 25% 的 head_dim 走 RoPE，节省位置编码开销。

```json
"rope_parameters": {
    "mrope_interleaved": true,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000
}
```

### 1.5 MTP

`mtp_num_hidden_layers: 1` 配置 1 层 MTP（多 token 预测），且 `mtp_use_dedicated_embeddings: false` 复用 token embedding 而非独立 embedding，用于推测解码加速。

### 1.6 小结

Qwen3.8-27B 的设计思路是用混合线性/全注意力把 dense 模型的长上下文做便宜，同时保持原生多模态。3:1 的 DeltaNet 与 Attention 比例让大部分层是 O(n) 线性注意力，每 4 层用一次全注意力兜底全局建模；M-RoPE 让位置编码天然适配视频时空。整体定位是 dense 27B、多模态、长程 Agent 导向。

---

## 二、MiniMax-M3（MiniMax）

### 2.1 基本信息

| 属性 | 值 |
|---|---|
| 模型类 | `MiniMaxM3SparseForConditionalGeneration` |
| model_type | `minimax_m3_vl` |
| 模态 | 原生多模态（图 + 视频 + 文） |
| 架构类型 | MoE |
| 总参数 / 激活 | 428B / 23B |
| 上下文 | 1M（1048576） |
| 授权 | minimax-community |
| ModelScope ID | `MiniMax/MiniMax-M3` |

### 2.2 整体架构：多模态 + MoE 文本骨干

顶层 `ForConditionalGeneration` 为 VL 架构。视觉编码器是 CLIP 风格 ViT，32 层、hidden 1280、patch 14，`image_size: 2016` 支持高分辨率输入。`rope_mode: "3d"` 用 3D RoPE 给视频 T/H/W 维度编码。`patch_merge` 配合 `spatial_merge_size: 2` 和 `temporal_patch_size: 2` 对视觉 token 做 2x2 空间加 2 帧时间合并，大幅压缩视觉 token 数。`image_grid_pinpoints` 列出 36 种分辨率组合（336~2016）做动态分辨率适配。

```json
"vision_config": {
    "hidden_size": 1280,
    "num_hidden_layers": 32,
    "patch_size": 14,
    "image_size": 2016,
    "position_embedding_type": "rope",
    "rope_mode": "3d",
    "img_token_compression_config": {
        "image_token_compression_method": "patch_merge",
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
    }
}
```

### 2.3 文本骨干：MoE

文本骨干 60 层，`moe_layer_freq` 数组前 3 个为 0、后续为 1，表示前 3 层是 dense、后 57 层是 MoE。128 个路由专家加 1 个共享专家，每 token 选 4 个。`scoring_func: sigmoid` 配合 `use_routing_bias: true` 采用 sigmoid 路由加路由偏置项。专家 FFN 的 `intermediate_size: 3072` 远小于 dense 层的 `dense_intermediate_size: 12288`，属于细粒度专家设计。

```json
"hidden_size": 6144,
"num_hidden_layers": 60,
"num_local_experts": 128,
"num_experts_per_tok": 4,
"n_shared_experts": 1,
"scoring_func": "sigmoid",
"use_routing_bias": true,
"moe_layer_freq": [0, 0, 0, 1, 1, 1, ... 1]
```

### 2.4 注意力：GQA + MSA

基础注意力是 GQA，64 头 Q 配 4 个 KV 头（16 倍分组共享），head_dim 128，走标准 GQA 压 KV，与 GLM 的 MHA+MLA 路线不同。`use_qk_norm: true` 配合 `qk_norm_type: "per_head"` 做 per-head QK 归一化稳定训练。

```json
"num_attention_heads": 64,
"num_key_value_heads": 4,
"head_dim": 128,
"partial_rotary_factor": 0.5,
"use_qk_norm": true,
"qk_norm_type": "per_head"
```

MSA（MiniMax Sparse Attention）稀疏注意力是 1M 上下文的关键。`sparse_disable_index_value` 和 `sparse_attention_freq` 都是前 3 层为 0、第 4 层起为 1，与 MoE 的 dense/MoE 分界线一致。`sparse_num_index_heads: 4` 配置 4 个 indexer head 学习哪些 block 重要。`sparse_topk_blocks: 16` 配合 `sparse_block_size: 128` 表示每个 token 只 attend 16 个 block（每 block 128 token），即 2048 token 的稀疏窗口。`sparse_score_type: "max"` 用 indexer 的最大分选 block，`sparse_local_block: 1` 保留 1 个局部 block 做局部注意力兜底。

```json
"sparse_attention_config": {
    "use_sparse_attention": true,
    "sparse_index_dim": 128,
    "sparse_num_index_heads": 4,
    "sparse_topk_blocks": 16,
    "sparse_block_size": 128,
    "sparse_disable_index_value": [0, 0, 0, 1, 1, ... 1],
    "sparse_score_type": "max",
    "sparse_local_block": 1,
    "sparse_attention_freq": [0, 0, 0, 1, 1, ... 1]
}
```

与 GLM DSA 相比，GLM 是每 4 层共享 indexer（IndexShare，省 indexer 算力），MiniMax 是每层独立 4 个 indexer head；GLM 是 token 级 top-2048，MiniMax 是 block 级 top-16（16x128=2048 token），量级相同但组织方式不同。

### 2.5 其他细节

`hidden_act: "swigluoai"` 配合 `swiglu_alpha: 1.702` 和 `swiglu_limit: 7.0` 是自研 SwiGLU 变体，带限幅。`use_gemma_norm: true` 用 Gemma 风格 RMSNorm 变体。`num_mtp_modules: 7` 配合 `num_nextn_predict_layers: 1` 配置了 7 个 MTP 模块，比 GLM 和 Hunyuan 的 1 层多很多，推测解码力度最大。

### 2.6 小结

MiniMax-M3 走的是多模态加 MSA 稀疏注意力把 1M 上下文做便宜、并押注推测解码的路线。和 GLM 同样是 MoE 加稀疏注意力加 1M 上下文，但稀疏注意力实现思路（MSA，block 级加每层独立 indexer）与 GLM 的 DSA/IndexShare 不同；7 个 MTP 模块说明在推测解码上押注最重。

---

## 三、Hunyuan Hy3（腾讯混元）

### 3.1 基本信息

| 属性 | 值 |
|---|---|
| 模型类 | `HYV3ForCausalLM` |
| model_type | `hy_v3` |
| 模态 | 纯文本 |
| 架构类型 | MoE |
| 总参数 / 激活 | 295B / 21B（+ 3.8B MTP） |
| 上下文 | 256K（262144） |
| 授权 | Apache 2.0（无地域限制） |
| ModelScope ID | `Tencent-Hunyuan/Hy3` |

### 3.2 整体架构：纯文本 MoE

顶层 `ForCausalLM`，纯文本，无视觉编码器。80 层中 `first_k_dense_replace: 1` 表示只有第 1 层是 dense，后 79 层全 MoE。对比 GLM 前 3 层 dense、MiniMax 前 3 层 dense，Hunyuan 仅前 1 层 dense，把算力几乎全交给 MoE，MoE 激进程度最高。192 个路由专家加 1 个共享专家，每 token 选 8。专家 FFN 的 `moe_intermediate_size: 1536` 只有 dense 层 `intermediate_size: 13312` 的约 1/9，属于细粒度专家。

```json
"hidden_size": 4096,
"num_hidden_layers": 80,
"first_k_dense_replace": 1,
"num_experts": 192,
"num_experts_per_tok": 8,
"num_shared_experts": 1,
"moe_intermediate_size": 1536,
"intermediate_size": 13312
```

### 3.3 路由

`moe_router_use_sigmoid: true` 采用 sigmoid 路由，与 DeepSeek、GLM、MiniMax 同流。`moe_router_enable_expert_bias: true` 加专家偏置项辅助负载均衡。`router_scaling_factor: 2.826` 是三个 MoE 模型里最高的路由缩放因子。`route_norm: true` 做路由归一化稳定 MoE 训练。

```json
"moe_router_use_sigmoid": true,
"moe_router_enable_expert_bias": true,
"router_scaling_factor": 2.826,
"route_norm": true,
"output_router_logits": true
```

### 3.4 注意力：标准 GQA

注意力是标准 GQA，64 头 Q 配 8 个 KV 头（8 倍分组共享），head_dim 128，无 MLA、无稀疏注意力，结构上最保守。`qk_norm: true` 做 QK 归一化稳定训练。`rope_theta: 11158840.0`（约 11M）是三个模型里最高的，支持 256K 上下文。

```json
"num_attention_heads": 64,
"num_key_value_heads": 8,
"head_dim": 128,
"qk_norm": true,
"rope_parameters": {
    "rope_theta": 11158840.0,
    "rope_type": "default"
}
```

### 3.5 MTP 与推测解码

`num_nextn_predict_layers: 1` 配置 1 层 MTP。README 部署示例配合 vLLM/SGLang 的 EAGLE 式投机解码，`--speculative-config.num_speculative_tokens 2` 一次推测 2 个 token。

```json
"num_nextn_predict_layers": 1
```

```bash
vllm serve tencent/Hy3 \
  --tensor-parallel-size 8 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2
```

### 3.6 其他细节

`enable_lm_head_fp32: true` 让 LM head 用 fp32 保精度，其余 bf16。`enable_attention_fp32_softmax: false` 和 `enable_moe_fp32_combine: false` 让注意力和 MoE 合并都用低精度省算力。`initializer_range: 0.006` 初始化标准差较小。

### 3.7 小结

Hunyuan Hy3 不追求结构创新，把 MoE 加 GQA 加 MTP 这套成熟组合做到极致工程化。结构上最保守（标准 GQA，无稀疏注意力，仅 1 层 dense），重点在 Agent 能力和工具调用的后训练。README 强调 tool-call 稳定性、抗幻觉（幻觉率 12.5% 降到 5.4%）、多轮意图保持（问题率 17.4% 降到 7.9%）。Apache 2.0 授权商用最友好。

---

## 四、三模型横向对比

| 维度 | Qwen3.8-27B | MiniMax-M3 | Hunyuan Hy3 |
|---|---|---|---|
| 模态 | 图 + 视频 + 文 | 图 + 视频 + 文 | 纯文本 |
| 架构类型 | dense | MoE | MoE |
| 总参 / 激活 | 27B dense | 428B / 23B | 295B / 21B |
| 层数 | 64 | 60 | 80 |
| dense 层数 | 全 dense | 前 3 层 | 前 1 层 |
| 专家配置 | 无 | 128 路由 + 1 共享，选 4 | 192 路由 + 1 共享，选 8 |
| 注意力机制 | Gated DeltaNet（线性）+ Gated Attention（全，GQA 24/4） | GQA（64/4）+ MSA 稀疏 | GQA（64/8）标准 |
| 稀疏注意力 | 无（靠线性注意力降复杂度） | MSA（block 级 top-16，每层 4 indexer） | 无 |
| 上下文 | 262K 原生 / 1M 外推 | 1M 原生 | 256K |
| MTP | 1 层 | 7 个模块 | 1 层 |
| 位置编码 | M-RoPE [11,11,10]，theta 10M | 3D RoPE，theta 5M | RoPE，theta 11M |
| 路由评分 | 无 | sigmoid + routing_bias | sigmoid + expert_bias |
| 授权 | Apache 2.0 | minimax-community | Apache 2.0 |

---

## 五、三条技术路线

三款模型代表了 2026 年国产开源大模型的三条不同技术路线。

### 5.1 Qwen3.8-27B：混合注意力 + 原生多模态

用 3:1 的线性注意力（Gated DeltaNet）与全注意力（Gated Attention）混合架构，让 dense 模型在不增加参数的情况下获得 O(n) 的长序列效率。每 4 层用一次全注意力兜底全局建模，M-RoPE 天然适配视频时空。适合中等规模、多模态、长程 Agent 任务，消费级硬件可部署，27B dense 量化后能跑笔记本。

### 5.2 MiniMax-M3：MoE + 稀疏注意力 + 押注推测解码

MoE（128 专家选 4）撑容量，MSA 稀疏注意力（block 级 top-16，每层 4 个 indexer head）把 1M 上下文注意力成本压下来，GQA 压 KV。最大亮点是 7 个 MTP 模块，在推测解码上押注最重。与 GLM 同样走 MoE 加稀疏注意力加 1M 上下文，但稀疏注意力实现不同，GLM 用 IndexShare（每 4 层共享 indexer 省 indexer 算力），MiniMax 用每层独立 indexer（更灵活但算力更高）。适合超大上下文、多模态、高吞吐推理场景。

### 5.3 Hunyuan Hy3：保守结构 + 极致后训练

结构上最保守，标准 GQA，无 MLA、无稀疏注意力，仅 1 层 dense，把所有创新预算留给后训练。重点在 Agent 能力和工具调用的稳定性，靠后训练工程（SFT + RL）取胜。不追求 1M 上下文（256K 够用），不追求结构创新，但 dense 层最少（仅 1 层），MoE 激进程度最高。Apache 2.0 授权商用最友好，适合 Agent 和工具调用场景的企业商用部署。

---

## 六、关键结构创新点

| 创新点 | 所属模型 | 作用 |
|---|---|---|
| Gated DeltaNet 线性注意力 | Qwen3.8-27B | O(n) 复杂度，长序列省 KV |
| 3:1 线性/全注意力混合 | Qwen3.8-27B | 兼顾长序列效率与全局建模 |
| M-RoPE [11,11,10] | Qwen3.8-27B | 视频时空位置编码 |
| MSA（block 级稀疏注意力） | MiniMax-M3 | 1M 上下文注意力降本 |
| 7 个 MTP 模块 | MiniMax-M3 | 推测解码力度最大 |
| 3D RoPE 视觉编码 | MiniMax-M3 | 视频视觉 token 时空编码 |
| 仅 1 层 dense + 79 层 MoE | Hunyuan Hy3 | MoE 激进程度最高 |
| sigmoid + expert_bias 路由 | Hunyuan Hy3 | 负载均衡 |
| per-head QK norm | MiniMax-M3 / Hunyuan Hy3 | 稳定训练 |

---

## 七、参考文件

所有结构代码均下载自 ModelScope，Qwen3.8-27B 目录下含 config.json、README.md、preprocessor_config.json、video_preprocessor_config.json、chat_template.jinja。MiniMax-M3 目录下含 config.json、configuration_minimax_m3_vl.py（配置类）、processing_minimax.py、image_processor.py、video_processor.py、README.md。Hunyuan-3.0 目录下含 config.json、README.md 与 README_CN.md、finetune 微调脚本、rl RL 训练文档、chat_template.jinja。
