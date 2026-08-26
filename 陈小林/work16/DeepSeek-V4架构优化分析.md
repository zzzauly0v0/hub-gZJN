# DeepSeek V4 架构优化深度分析

> 基于 `transformers/models/deepseek_v4` 源码（`modeling_deepseek_v4.py` + `configuration_deepseek_v4.py`），从**多头注意力**、**归一化+残差连接**、**前馈网络**三个维度，逐层拆解 DeepSeek V4 的优化方案。每个优化点都给出"为什么需要"、"怎么做的"、"效果是什么"的完整解释。

---

## 零、整体架构鸟瞰

在进入细节之前，先看 V4 的全局参数（V4-Flash 配置）：

```
hidden_size = 4096          # 模型隐藏维度
num_hidden_layers = 43      # Transformer 层数
num_attention_heads = 64    # Query 头数量
num_key_value_heads = 1     # KV 头数量（仅 1 个！）
head_dim = 512              # 每个注意力头的维度
q_lora_rank = 1024          # Query 的 LoRA 中间维度
n_routed_experts = 256      # MoE 路由专家总数
num_experts_per_tok = 6     # 每个 token 激活的专家数
moe_intermediate_size = 2048 # 每个专家的中间维度
hc_mult = 4                 # 超连接的并行流数
```

**一句话总结 V4 的设计哲学**：用尽可能大的模型容量（256 个专家、64 个注意力头、512 维 head_dim），但通过**稀疏激活**（MoE）、**压缩注意力**（CSA/HCA）、**流形约束残差**（mHC）三大机制，让每个 token 实际执行的计算量远低于同等容量的密集模型。

---

## 一、多头注意力（Multi-Head Attention）

### 标准 Transformer 注意力的问题

经典 MHA 中，每个注意力头都有独立的 Q、K、V 投影。对于一个 64 头、512 维 head_dim 的模型：
- KV cache 需要存 64 个 K 头 + 64 个 V 头 = 128 个张量/层
- 长序列时 KV cache 内存爆炸
- $O(n^2)$ 的注意力计算在 100 万 token 上下文时不可行

V4 针对这些问题逐一给出了优化方案。

---

### 1.1 Shared-KV Multi-Query Attention（共享 KV 多查询注意力）

#### 是什么？

标准 MHA 中每个 head 有独立的 K 和 V。MQA（Multi-Query Attention）让所有 query head 共享同一组 K/V。V4 把这个思路推到极致：**只有 1 个 KV 头**，且 K 和 V 是同一个张量。

#### 代码实现

```python
# DeepseekV4Attention.__init__
self.num_key_value_groups = config.num_attention_heads  # = 64
# 含义：1 个 KV 头需要"复制"64 次才能和 64 个 Q 头对齐

self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
# 只投影到 1 个 head_dim = 512 维，而非 num_heads * head_dim
```

```python
# DeepseekV4Attention.forward
kv = self.kv_norm(self.kv_proj(hidden_states))  # [B, S, 1, 512]
kv = apply_rotary_pos_emb(kv, cos, sin)
# ...
attn_output, _ = attention_interface(self, q, kv, kv, ...)  # 注意：K=V 是同一个张量
```

在 `repeat_kv` 函数中，单个 KV 头被广播到所有 64 个 query head：
```python
def repeat_kv(hidden_states, n_rep):
    # [B, 1, S, D] → [B, 64, S, D]
    hidden_states[:, :, None, :, :].expand(batch, 1, n_rep, slen, head_dim)
```

#### 为什么有效？

| 对比项 | 标准 MHA (64头) | GQA (8 KV头) | V4 MQA (1 KV头) |
|--------|----------------|--------------|----------------|
| KV cache/层 | 64K + 64V = 128张量 | 8K + 8V = 16张量 | **1张量 (K=V)** |
| 内存节省 | 基准 | 8× | **128×** |
| 带宽压力 | 高 | 中 | 极低 |

**直觉理解**：想象一个会议室里 64 个人（query heads）各自提问，但大家共享同一块白板（KV）上的信息。每个人关注白板上不同部分（通过不同的 Q），但白板上只有一份内容。这大幅减少了"搬运白板"的内存带宽开销。

**K=V 的额外好处**：Key 和 Value 共享同一投影，不仅减少一半 KV cache，还隐式约束了注意力机制——"检索什么"（K）和"取回什么"（V）被绑定在一起，模型必须学会在同一个表示中同时编码"可检索性"和"信息内容"。

---

### 1.2 Query 的 LoRA 投影

#### 是什么？

虽然 KV 只有 1 个头，但 Q 仍然有 64 个头，每个 512 维，总计 $64 \times 512 = 32768$ 维。直接从 4096 维投影到 32768 维是一个巨大的矩阵乘法。V4 用两步投影（类似 LoRA）来降低开销。

#### 代码实现

```python
# 第一步：降维到 q_lora_rank = 1024
self.q_a_proj = nn.Linear(hidden_size=4096, q_lora_rank=1024)
self.q_a_norm = DeepseekV4RMSNorm(1024)

# 第二步：从低维展开到全部 head
self.q_b_proj = nn.Linear(1024, 64 * 512 = 32768)
self.q_b_norm = DeepseekV4UnweightedRMSNorm()  # 无参数归一化
```

```python
# forward 过程
q_residual = self.q_a_norm(self.q_a_proj(hidden_states))  # [B, S, 1024] — 低秩中间态
q = self.q_b_proj(q_residual).view(*hidden_shape)          # [B, S, 64, 512] — 展开
q = self.q_b_norm(q)                                        # 归一化
q = apply_rotary_pos_emb(q, cos, sin)                       # 施加位置编码
```

#### 为什么有效？

直接投影：$4096 \times 32768 = 1.34 \times 10^8$ 参数

LoRA 两步投影：$4096 \times 1024 + 1024 \times 32768 = 4.2 \times 10^6 + 3.36 \times 10^7 ≈ 3.78 \times 10^7$ 参数

**参数减少约 72%**，同时保留了足够的表达能力（1024 维的瓶颈层仍然很宽）。

**`q_residual` 的双重身份**：这个 1024 维的中间态不仅用于生成 Q，还被传给 CSA 的 Lightning Indexer（`q_b_proj` 共享使用），让 indexer 的 query 与主注意力的 query 共享低秩表示，减少冗余计算。

---

### 1.3 分层压缩注意力（Compressed Attention）

这是 V4 注意力最核心的创新。标准注意力在长序列上面临 $O(n^2)$ 的计算瓶颈，V4 的解决方案是**让不同层看到不同"粒度"的历史信息**。

#### 三种注意力层类型

V4 定义了三种层类型，每层只使用其中一种：

| 层类型 | 压缩率 | 覆盖范围 | 类比 |
|--------|--------|---------|------|
| `sliding_attention` | 无压缩 | 最近 128 个 token | "近视"——只看近处 |
| `compressed_sparse_attention` (CSA) | 每 4 token → 1 | 中等距离 + 索引检索 | "中距离眼镜"——适度压缩 + 智能检索 |
| `heavily_compressed_attention` (HCA) | 每 128 token → 1 | 全局远距离 | "望远镜"——高度压缩看全局 |

**默认排布**（43 层）：

```
层 0: HCA (bootstrap)      ← 最底层先建立全局视野
层 1: HCA (bootstrap)      ← 两层 bootstrap 确保底层有远距离信息
层 2: CSA                  ← 之后 CSA/HCA 交替
层 3: HCA
层 4: CSA
层 5: HCA
...交替直到层 42
```

**直觉理解**：想象你在阅读一篇长文章。底层（HCA）帮你记住"文章大致讲了什么"（全局概要），中层（CSA）帮你定位"第三章的论点在哪里"（中等距离检索），高层（滑动窗口）帮你理解"当前这句话的语法结构"（局部细节）。不同层负责不同粒度的信息获取。

---

#### 1.3.1 滑动窗口注意力（Sliding Window）

最简单的情况：每个 query 只关注最近 $n_{win}=128$ 个 token。

```python
# 缓存更新
self.keys = full[:, :, -self.sliding_window + 1:, :]  # 只保留最近 127 个旧 token
self.values = self.keys  # K = V
```

**效果**：计算量固定为 $O(n_{win})$，与总序列长度无关。适合捕捉局部依赖（语法、短语结构）。

---

#### 1.3.2 HCA 压缩器（Heavily Compressed Attention）

##### 核心思想

每 128 个 token 被压缩为 1 个"超级 token"。压缩方式不是简单平均，而是**学习一个门控加权聚合**。

##### 压缩过程（代码逐步解析）

```python
class DeepseekV4HCACompressor:
    def __init__(self, config):
        self.compress_rate = 128  # 每 128 个 token 压缩为 1 个
        self.kv_proj  = nn.Linear(4096, 512)   # 投影到 head_dim
        self.gate_proj = nn.Linear(4096, 512)  # 门控投影
        self.position_bias = nn.Parameter(torch.empty(128, 512))  # 位置内位置偏置
        self.kv_norm = DeepseekV4RMSNorm(512)
```

```python
def forward(self, hidden_states, ...):
    # 步骤 1：投影
    kv   = self.kv_proj(hidden_states)    # [B, T, 512] — 内容表示
    gate = self.gate_proj(hidden_states)  # [B, T, 512] — 门控信号

    # 步骤 2：按窗口 reshape
    chunk_kv   = kv.view(batch, n_windows, 128, 512)    # 分成 n_windows 个窗口
    chunk_gate = gate.view(batch, n_windows, 128, 512) + self.position_bias
    #                                                    ↑ 加上窗口内的位置偏置
    #    让模型知道这 128 个 token 中"谁是第几个"

    # 步骤 3：Softmax 加权聚合
    compressed = kv_norm(
        (chunk_kv * chunk_gate.softmax(dim=2)).sum(dim=2)
    )
    # softmax(gate) 产生 128 个权重，加权和把 128 个 token 融合成 1 个
    # 结果形状：[B, n_windows, 512]

    # 步骤 4：对压缩条目施加 RoPE
    positions = arange(n_windows) * 128 + first_window_position
    cos, sin = rotary_emb(compressed, positions)
    compressed = apply_rotary_pos_emb(compressed, cos, sin)
```

**直觉理解**：把 128 个 token 的窗口想象成一个"会议"。`kv_proj` 提取每个参会者的"发言内容"，`gate_proj` 决定每个参会者的"发言权重"。`softmax` 归一化后，加权求和就得到"会议纪要"——一个浓缩了 128 个 token 核心信息的压缩表示。`position_bias` 让模型知道每句话在会议中的时间顺序。

##### 因果性保证

```python
# query 在位置 t 只能看到 position ≤ t 的压缩条目
causal_threshold = (position_ids + 1) // compress_rate
block_bias = masked_fill(entry_indices >= causal_threshold, -inf)
```

这确保了位置 7 的 query 看不到压缩了位置 8-135 的压缩条目（因为它包含了"未来"信息）。

---

#### 1.3.3 CSA 压缩器 + Lightning Indexer

CSA 比 HCA 更精细：压缩率更低（4:1），但引入了 **Lightning Indexer** 做智能检索。

##### CSA 双序列重叠压缩

```python
class DeepseekV4CSACompressor:
    kv_proj  = nn.Linear(4096, 2 * 512)  # 注意：2 倍 head_dim
    gate_proj = nn.Linear(4096, 2 * 512)
```

每个 token 投影到 $2 \times 512 = 1024$ 维，分为两个系列：

- **Ca**（前 512 维）：贡献给**下一个**窗口的压缩条目
- **Cb**（后 512 维）：贡献给**当前**窗口的压缩条目

```
窗口 w-1:     [... Ca tokens ...]
窗口 w:                    [... Cb tokens ...]
压缩条目 w:   ← 合并(softmax(Ca_{w-1}, Cb_w)) →
```

**有效宽度 = $2 \times 4 = 8$，步长 = 4**。相邻窗口有 50% 重叠，类似于滑动窗口的效果，但通过 softmax 加权实现了更灵活的融合。

**直觉理解**：如果 HCA 是"年度总结"（128→1），CSA 就是"周报"（4→1）。周报之间还有交叉——本周报告的结尾和下周报告的开头有重叠部分，确保边界处的信息不丢失。

##### Lightning Indexer（闪电索引器）

当序列很长时（如 100 万 token），即使压缩率 4:1，也有 25 万个压缩条目。全部做注意力仍然很贵。Lightning Indexer 的作用是为每个 query **只挑选最相关的 top-k 个压缩条目**。

```python
class DeepseekV4Indexer:
    # 索引器有自己的缩小版压缩器
    kv_proj = nn.Linear(4096, 2 * 128)    # index_head_dim = 128（远小于主 head_dim 512）
    q_b_proj = nn.Linear(1024, 64 * 128)  # 从 q_lora_rank 投影

    def forward(self, ...):
        # 步骤 1：对压缩 KV 打分
        scores = relu(q · compressed_kv^T) * scale   # [B, S, H, T]
        weights = weights_proj(hidden_states)          # [B, S, H]
        index_scores = (scores * weights).sum(dim=2)   # [B, S, T]

        # 步骤 2：选 top-k
        top_k_indices = index_scores.topk(512, dim=-1) # 每 query 只保留 512 个
```

**打分公式**：

$$\text{score}_{t,s} = \sum_{h=1}^{H} w_{t,h} \cdot \text{ReLU}(q_{t,h} \cdot K^{IComp}_s)$$

其中 $H = 64$（indexer 头数），$w_{t,h}$ 是从 hidden_states 学习的每头权重。

**直觉理解**：Indexer 就像一个"图书馆索引系统"。面对 25 万本书（压缩条目），它先快速浏览每本书的摘要（低维 compressed key），然后结合你当前的问题（query），给出相关度评分。最终只把最相关的 512 本书取出来供你精读（主注意力计算）。

**因果性 + 有效性过滤**：

```python
# 如果 query 位置太小，不够看到某些压缩条目，用 -1 标记为无效
invalid = top_k_indices >= causal_threshold
return torch.where(invalid, -1, top_k_indices)
```

---

### 1.4 分组低秩输出投影（Grouped Output Projection）

#### 问题

注意力计算完成后，需要把 $64 \times 512 = 32768$ 维的输出投影回 $4096$ 维。直接做需要 $32768 \times 4096 ≈ 1.34 \times 10^8$ 参数的矩阵乘法——**这个投影可能比注意力本身还贵**。

#### 解决方案：两步分组投影

```python
# 第一步：分组块对角投影 (o_a_proj)
self.o_a_proj = DeepseekV4GroupedLinear(
    in_features_per_group = 64 * 512 // 8 = 4096,  # 每组 8 个头
    out_features = 1024,                              # 每组压缩到 1024
    n_groups = 8
)
# 8 组 × (4096 → 1024) = 8 个独立的小投影

# 第二步：全局混合投影 (o_b_proj)
self.o_b_proj = nn.Linear(8 * 1024 = 8192, 4096)
```

```python
class DeepseekV4GroupedLinear(nn.Linear):
    def forward(self, x):
        # x: [B, S, 8_groups, 4096_per_group]
        w = self.weight.view(n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, n_groups, hidden_dim).transpose(0, 1)  # [8, B*S, 4096]
        y = torch.bmm(x, w)  # 批量矩阵乘：8 个组并行投影
        return y.reshape(*input_shape, n_groups, -1)  # [B, S, 8, 1024]
```

#### 参数量对比

| 方案 | 参数量 |
|------|--------|
| 直接投影 32768 → 4096 | $1.34 \times 10^8$ |
| 分组投影 8×(4096→1024) + 8192→4096 | $3.36 \times 10^7 + 3.35 \times 10^7 ≈ 6.7 \times 10^7$ |

**参数减少约 50%**，同时第一步是块对角的（各组独立），天然适合并行计算。

**直觉理解**：相当于先把 64 个专家的意见分 8 个小组讨论（每组 8 人），每组形成一份小组报告（1024 维）。然后把 8 份小组报告汇总为最终决策（4096 维）。比让 64 个人同时发言形成决策高效得多。

---

### 1.5 Partial RoPE（部分旋转位置编码）

#### 是什么？

标准 RoPE 对整个 head_dim 施加旋转。V4 只对每个 head 的 **最后 64 维**（共 512 维的 12.5%）施加 RoPE，前 448 维不含位置信息（称为 nope）。

```python
partial_rotary_factor = 64 / 512 = 0.125

def apply_rotary_pos_emb(x, cos, sin):
    rope_dim = cos.shape[-1]          # = 32（半维，repeat_interleave 后变 64）
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]  # 切分
    rotated = rope * cos + rotate_half(rope) * sin         # 只对 rope 部分旋转
    return torch.cat([nope, rotated], dim=-1)              # 拼接回去
```

#### 为什么？

- **nope 部分**（448 维）：编码与位置无关的语义信息（"这个词是什么意思"）
- **rope 部分**（64 维）：编码与位置相关的关系信息（"这个词在哪个位置"）

让模型自由分配"位置敏感"和"位置无关"的容量，而不是强制所有维度都携带位置信息。

---

### 1.6 Reverse RoPE on Output（输出反向旋转）

这是一个精巧的设计。由于 K=V，V 也携带了 RoPE。在注意力输出中，V 的位置编码会"残留"——但我们需要的是**相对**位置信息，而非绝对位置。

```python
# 在注意力输出上施加 -sin（共轭旋转），抵消 V 上的 RoPE
attn_output = apply_rotary_pos_emb(attn_output, cos, -sin)
```

**数学解释**：

$$\text{output} = \sum_j \alpha_j \cdot V_j \quad (V_j \text{ 带 RoPE}(j))$$

施加 $\text{RoPE}(-i)$（query 位置的反向旋转）后：

$$\text{output}' = \text{RoPE}(-i) \cdot \sum_j \alpha_j \cdot \text{RoPE}(j) \cdot v_j = \sum_j \alpha_j \cdot \text{RoPE}(j - i) \cdot v_j$$

结果仅依赖于**相对距离** $j - i$，与绝对位置无关。这样后续的输出投影就可以在位置无关的空间中混合各头信息。

---

### 1.7 可学习注意力汇聚（Learnable Attention Sink）

```python
self.sinks = nn.Parameter(torch.empty(self.num_heads))  # 每头一个标量

# 在 softmax 前拼接
combined_logits = torch.cat([attn_weights, sinks], dim=-1)
probs = softmax(combined_logits, dim=-1)
scores = probs[..., :-1]  # softmax 后丢弃 sink
```

**直觉理解**：有时候"什么都不关注"才是最佳选择。例如处理虚词"的"时，模型可能不需要关注任何具体 token，而是直接传递上一层的信息。Sink 提供了一个"安全出口"，让注意力权重可以"浪费"在这个虚拟 token 上，而不是被迫关注不相关的 token。

---

## 二、归一化 + 残差连接

### 标准 Transformer 残差块回顾

经典 Pre-Norm Transformer 的残差块：

```
x = x + Attention(LayerNorm(x))    # 注意力子层
x = x + MLP(LayerNorm(x))          # 前馈子层
```

问题：
1. 残差系数恒为 1，信号只能"累加"不能"衰减"，深层堆叠容易信号膨胀
2. 只有一条残差流，信息传递路径单一
3. 随着层数增加，梯度可能爆炸或消失

V4 用两个层面的优化来解决这些问题。

---

### 2.1 RMSNorm 的精细分层使用

V4 使用了**两种** RMSNorm 变体，在不同位置各有讲究：

#### 带权重 RMSNorm（`DeepseekV4RMSNorm`）

```python
class DeepseekV4RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习缩放
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)
```

等价于 T5LayerNorm，先归一化再逐元素乘以可学习权重。

#### 无权重 RMSNorm（`DeepseekV4UnweightedRMSNorm`）

```python
class DeepseekV4UnweightedRMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)
```

纯归一化，**零学习参数**。

#### 各处的使用策略与原因

| 位置 | 类型 | 维度 | 为什么选择这种 |
|------|------|------|--------------|
| `input_layernorm` | **带权重** | 4096 | 子层入口，需要逐维度缩放来调节不同特征的强度 |
| `post_attention_layernorm` | **带权重** | 4096 | 同上 |
| `q_a_norm` | **带权重** | 1024 | Q 的 LoRA 中间态，维度较小，学习参数代价低 |
| `q_b_norm` | **无权重** | 512/per head | Q 展开后是 $64 \times 512 = 32768$ 维，如果带权重则引入 32768 个参数，性价比低；且此处只需消除量纲差异 |
| `kv_norm` | **带权重** | 512 | KV 只有 1 个头，仅 512 维，学习参数代价极低 |
| 压缩器 `kv_norm` | **带权重** | 512 | 压缩后的 KV，需要学习"哪些维度更重要" |
| HC 内部 `input_norm` | **无权重** | $4 \times 4096$ | 用于混合多条流之前的纯归一化，不需要逐维度偏好 |

**设计哲学**：在**参数代价低且信息增益大**的地方用带权重版本（如 KV norm 只有 512 个参数），在**参数代价高且只需消除量纲**的地方用无权重版本（如 q_b_norm 展开后有 32768 维）。

#### 强制 FP32 精度

```python
_keep_in_fp32_modules_strict = [
    "attn_hc", "ffn_hc", "hc_head",    # 超连接模块
    "sinks",                              # 注意力汇聚
    "position_bias",                      # 压缩器位置偏置
    "e_score_correction_bias",            # MoE 路由校正
    "q_a_norm", "kv_norm",               # 注意力归一化
    "input_layernorm", "post_attention_layernorm", "norm",  # 层归一化
]
```

所有归一化层和超连接模块**强制 FP32 计算**，即使模型主体使用 BF16/FP8。这是因为归一化中的方差计算和 Sinkhorn 迭代对数值精度极其敏感——BF16 的有限精度可能导致方差下溢或 Sinkhorn 不收敛。

---

### 2.2 流形约束超连接（mHC）—— 替代传统残差

这是 V4 最具创新性的组件，彻底重新设计了残差连接的方式。

#### 2.2.1 从单流到多流：为什么需要多条并行流？

**传统残差的问题**：

```
层 L:   h_{L+1} = h_L + f_L(Norm(h_L))
```

这里只有**一条**信息流 $h$。所有子层（Attention 和 MLP）都在同一条流上"叠加"修改。这意味着：
- 如果某层的 $f_L$ 输出过大，信号就会膨胀
- 如果某层的 $f_L$ 输出接近 $-h_L$，信号就会被"消除"
- 没有"备用通道"来保留原始信息

**mHC 的解决方案**：维护 $H = 4$ 条并行流，形状始终为 `[B, S, 4, D]`。

```
输入：[B, S, 4, 4096]  —  4 条独立的"信息河流"
```

类比：传统残差像一条公路——所有车辆（信息）走同一条路，一旦堵车（信号问题）就全堵。mHC 像四条并行高速公路——信息可以在不同公路之间流转，某条路堵了可以走其他路。

#### 2.2.2 mHC 模块的完整数据流

每个子层（Attention/MLP）前后各有一个 mHC 模块。源码中的 ASCII 图非常清晰：

```
              hidden_streams        flatten(2)        RMSNorm + Linear(fn)
         [B, S, H, D]  ──────────►  [B, S, H*D]  ─────────────────────────►
                                                            mix-logits
                                                            [B, S, (2+H)*H]
                                                                   │
                            ┌──────────────────────────────────────┴───────────────────────┐
                            ▼                          ▼                                    ▼
                        pre logits                post logits                         comb logits
                        [B, S, H]                 [B, S, H]                           [B, S, H, H]
                        × scale[0]                × scale[1]                          × scale[2]
                        + base[:H]                + base[H:2H]                        + base[2H:]
                        σ() + eps                 2·σ()                               softmax(-1) + eps
                        │                         │                                   │
                        pre                       post                                Sinkhorn(iters)
```

逐步解析（$H = 4$）：

**步骤 1：全局感知**

```python
flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
# [B, S, 4, 4096] → [B, S, 16384] → 归一化
# 把 4 条流展平为一个向量，让后续线性层能"看到所有流的全貌"
```

**步骤 2：线性映射生成三组 logits**

```python
mix = (2 + H) * H = (2 + 4) * 4 = 24  # 总输出维度

pre_w, post_w, comb_w = F.linear(flat, self.fn).split([4, 4, 16], dim=-1)
#                         ↑ 一次线性映射产出所有需要的原始 logits
#                         pre: 4 维（每条流一个坍缩权重）
#                         post: 4 维（每条流一个放置权重）
#                         comb: 16 维（4×4 混合矩阵）
```

**步骤 3：生成 pre（坍缩权重）**

```python
pre = sigmoid(pre_w * pre_scale + pre_b) + eps   # 值域: (eps, 1+eps)
# sigmoid 保证正数，eps 防止完全为零
# pre[i] 表示"第 i 条流对子层输入的贡献权重"
```

```python
collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2)
# 加权求和：4 条流 → 1 条流，作为子层的输入
# [B, S, 4, D] × [B, S, 4, 1] → sum → [B, S, D]
```

**直觉**：pre 就像一个"选稿编辑"——从 4 条信息流中按比例混合，选出送给子层处理的"稿件"。

**步骤 4：生成 post（放置权重）**

```python
post = 2 * sigmoid(post_w * post_scale + post_b)  # 值域: [0, 2]
# post[i] 表示"子层输出写入第 i 条流的强度"
# 乘以 2 使得最大值为 2，允许信号放大（但不超过 2 倍）
```

**步骤 5：生成 comb（混合矩阵）+ Sinkhorn 投影**

```python
# 初始：softmax + eps 保证严格正数
comb = softmax(comb_logits, dim=-1) + eps

# Sinkhorn-Knopp 迭代：交替行归一化和列归一化
for _ in range(20 - 1):
    comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)  # 行归一化
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)  # 列归一化
# 经过 20 次迭代，comb 收敛到双随机矩阵（行和=1，列和=1）
```

**步骤 6：更新 hidden_states**

```python
# 子层输出写入 + 残差混合
hidden_states = post.unsqueeze(-1) * sublayer_output.unsqueeze(-2) + matmul(comb.T, hidden_states)
#                  ↑ 子层输出按 post 权重写入各流        ↑ 旧流按 comb 矩阵混合
# [B, S, 4, D] = [B, S, 4, 1]*[B, S, 1, D]  +  [4, 4] @ [B, S, 4, D]
```

展开写就是：

$$h'_k = \text{post}_k \cdot f(\text{collapsed}) + \sum_j \text{comb}_{j,k} \cdot h_j$$

其中：
- 第一项：子层输出按 `post[k]` 的权重写入第 $k$ 条流
- 第二项：旧的四条流通过双随机矩阵 `comb` 混合后贡献给第 $k$ 条流

#### 2.2.3 Sinkhorn 双随机约束的深层含义

**双随机矩阵**：行和与列和均为 1 的非负矩阵。

**为什么重要？**

1. **信号守恒**：双随机矩阵作用于向量后，向量各分量之和不变。即 $\sum_k h'_k = \sum_j h_j$（忽略 post 项）。信号不会在流混合过程中膨胀或衰减。

2. **非膨胀性**（Non-expansive）：双随机矩阵的谱范数 ≤ 1，所以 $\|comb \cdot h\| \leq \|h\|$。这意味着残差部分的信号**永远不会放大**。

3. **信息守恒**：双随机矩阵是可逆的（正交矩阵的特殊情况），信息不会在混合中丢失。

4. **梯度稳定性**：反向传播时梯度也经过双随机矩阵，不会放大，避免梯度爆炸。

**直觉理解**：想象 4 个水杯（4 条流），每个杯子里有不同量的水（信号强度）。双随机矩阵就像一组管道，把水在杯子之间重新分配——但总水量（信号总量）不变。不管经过多少层（多少次重新分配），总水量始终守恒。这就是"流形约束"的含义——把混合操作限制在"双随机矩阵流形"上。

#### 2.2.4 与传统残差的定量对比

| 特性 | 传统残差 $h' = h + f(h)$ | mHC |
|------|-------------------------|-----|
| 残差系数 | 恒为 1 | 自适应学习（由 comb 决定） |
| 信号放大 | 可能（$f(h)$ 无约束） | 不可能（双随机约束 + post ≤ 2） |
| 信息路径 | 单一 | 4 条并行流 + 自适应混合 |
| 深层稳定性 | 需要额外技巧（如 $\frac{1}{\sqrt{L}}$ 缩放） | 天然稳定（信号守恒） |
| 表达力 | $h + f(h)$ | $post \cdot f(collapse(h)) + comb^T \cdot h$ |

---

### 2.3 最终流坍缩（`DeepseekV4HyperHead`）

43 层 decoder 结束后，4 条流需要合并为 1 条输出：

```python
class DeepseekV4HyperHead:
    def forward(self, x):  # x: [B, S, 4, D]
        flat = self.input_norm(x.flatten(2).float())   # 全局感知
        mixes = F.linear(flat, self.hc_fn)              # [B, S, 4]
        pre = sigmoid(mixes * scale + base) + eps       # 坍缩权重
        return (pre.unsqueeze(-1) * x).sum(dim=2)       # 加权求和 → [B, S, D]
```

模型最终输出：

```python
hidden_states = self.norm(self.hc_head(hidden_streams))
#               ↑ 最终 RMSNorm  ↑ 4流→1流
```

**注意**：最终坍缩比层间 mHC 简单——只需要 pre（坍缩权重），不需要 post 和 comb（因为没有后续子层了）。

---

## 三、前馈网络（Feed-Forward Network）

### 标准 FFN 的问题

经典 Transformer 的 FFN 是一个两层 MLP：$x \mapsto W_2 \cdot \text{ReLU}(W_1 \cdot x)$。
- 参数量：$2 \times d_{model} \times d_{ff}$，通常占模型参数的 2/3
- 对所有 token 执行相同计算，没有"分工"

MoE 的思路是：**用多个专家 MLP 替代单个 MLP，每个 token 只激活少量专家**。

---

### 3.1 稀疏混合专家（Sparse MoE）

#### 架构

```python
class DeepseekV4SparseMoeBlock:
    gate = TopKRouter / HashRouter    # 路由器：决定每个 token 去哪些专家
    experts = DeepseekV4Experts       # 256 个路由专家
    shared_experts = DeepseekV4MLP    # 1 个共享专家（始终激活）
```

```python
def forward(self, hidden_states, input_ids=None):
    # 路由：决定每个 token 去哪些专家
    _, weights, indices = self.gate(hidden_states)

    # 稀疏计算：只激活被选中的专家
    routed = self.experts(flat, indices, weights)

    # 共享专家：对所有 token 都执行
    return routed + self.shared_experts(residual)
    #      ↑ 稀疏路径              ↑ 密集路径
```

#### 参数与计算量对比

| 项目 | 密集 MLP (intermediate=16384) | V4 MoE (256专家, intermediate=2048, top-6) |
|------|-----|------|
| 总参数量 | $2 \times 4096 \times 16384 ≈ 1.34 \times 10^8$ | $256 \times 3 \times 2048 \times 4096 ≈ 6.44 \times 10^9$ |
| 每 token 计算量 | $1.34 \times 10^8$ 次运算 | $(6 + 1) \times 3 \times 2048 \times 4096 ≈ 1.76 \times 10^8$ 次运算 |
| **参数/计算比** | 1:1 | **36:1** |

**关键洞察**：MoE 让模型参数量增长了约 48 倍，但每 token 计算量只增长了约 1.3 倍。这就是"稀疏激活"的威力——大容量、小计算。

---

### 3.2 专家权重存储与计算

```python
class DeepseekV4Experts:
    gate_up_proj = nn.Parameter(torch.empty(256, 2*2048, 4096))  # [专家数, 2×中间维, 隐藏维]
    down_proj = nn.Parameter(torch.empty(256, 4096, 2048))
```

**为什么合并 gate 和 up？**

```python
def _apply_gate(self, gate_up):
    gate, up = gate_up.chunk(2, dim=-1)  # 拆分
    gate = gate.clamp(max=10.0)
    up = up.clamp(min=-10.0, max=10.0)
    return silu(gate) * up
```

SwiGLU 激活需要两个投影：$gate\_proj(x)$ 和 $up\_proj(x)$。它们可以合并为一次矩阵乘法：

```
分开做：gate_proj(x) + up_proj(x)  →  2 次 kernel launch
合并做：gate_up_proj(x)            →  1 次 kernel launch，然后 chunk 拆分
```

在 GPU 上，kernel launch 有固定开销，合并能显著减少开销。这在 MoE 中尤其重要——每个专家已经很小了，kernel launch 开销占比更高。

---

### 3.3 Hash-MoE 引导层（前 3 层）

#### 问题

训练初期，路由器（gate）还没有学会如何分配 token 到专家。可能出现"路由崩塌"：所有 token 被路由到同一批热门专家，冷门专家完全得不到训练。

#### 解决方案

前 3 层使用**冻结的 hash 表**来决定路由：

```python
class DeepseekV4HashRouter:
    tid2eid = torch.zeros(vocab_size=129280, top_k=6, dtype=long)
    # 一张大表：每个 token id 映射到固定的 6 个专家 id

    def forward(self, hidden_states, input_ids):
        indices = self.tid2eid[input_ids].long()   # 专家选择完全由 token id 决定
        # 但权重仍然由学习的 gate 产出
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        weights = scores.gather(1, indices)        # 只对选中的专家打分
```

**直觉理解**：想象一家公司的新人培训。前 3 天（前 3 层）按"学号"（token id）固定分组，确保每组人数均匀。之后（后续层）再让"项目经理"（gate）根据能力（hidden state）自主分配。这样避免了初期所有人涌向同一个"明星经理"的问题。

**关键设计**：hash 只决定"去哪个专家"（indices），"专家贡献多大"（weights）仍由学习的 gate 决定。这保证了即使在 hash 层，梯度仍然可以通过 weights 流向 gate 参数，为后续层的路由学习做预热。

---

### 3.4 Top-K 路由器（后续层）

```python
class DeepseekV4TopKRouter:
    weight = nn.Parameter(torch.empty(256, 4096))  # 每专家一个线性分类器
    score_fn = ACT2FN["sqrtsoftplus"]               # 打分激活函数
    e_score_correction_bias = zeros(256)             # 可学习偏置

    def forward(self, hidden_states):
        logits = F.linear(flat, self.weight)          # [tokens, 256]
        scores = self.score_fn(logits)                # sqrt(softplus(x))
        indices = topk(scores + e_score_correction_bias, 6)  # 选 top-6
        weights = scores.gather(1, indices)
        weights = weights / weights.sum(dim=-1)       # 归一化
        return logits, weights * 1.5, indices         # × routed_scaling_factor
```

#### `sqrtsoftplus` 打分函数

$$\text{score}(x) = \sqrt{\text{softplus}(x)} = \sqrt{\ln(1 + e^x)}$$

**为什么用 sqrtsoftplus 而不是 softmax？**

- **Softmax**：分数归一化后竞争性强，但梯度在饱和区接近 0
- **Sigmoid**：各专家独立打分，但缺乏竞争性
- **sqrtsoftplus**：
  - 始终非负（softplus 保证）
  - 单调递增，无饱和区（梯度不会消失）
  - sqrt 压缩了大值、放大了小值的差异，使路由更加均匀
  - 不像 softmax 那样强制所有分数之和为 1，允许"多个专家都挺好"

#### `e_score_correction_bias`（分数校正偏置）

```python
indices = topk(scores + self.e_score_correction_bias, 6)
```

这是一个可学习的 per-expert 偏置，用于**长期平衡**。即使某个专家的 `weight` 学得不够好导致分数偏低，`e_score_correction_bias` 可以补偿它，防止该专家被"饿死"。

#### `routed_scaling_factor = 1.5`

```python
return logits, weights * 1.5, indices
```

归一化后的权重之和为 1，乘以 1.5 放大后之和为 1.5。这**有意让专家贡献略大于残差连接**，强化专家的影响力，防止专家信号被残差流"淹没"。

---

### 3.5 共享专家（Shared Expert）

```python
self.shared_experts = DeepseekV4MLP(config)
# 标准 SwiGLU MLP：gate_proj + up_proj + down_proj

def forward(self, x):
    gate = self.gate_proj(x).clamp(max=10.0)
    up = self.up_proj(x).clamp(min=-10.0, max=10.0)
    return self.down_proj(silu(gate) * up)
```

**为什么需要共享专家？**

路由专家是稀疏激活的——每个 token 只走 6/256 个专家。但有些"通用知识"（如基本的语言理解、常见推理模式）是**每个 token 都需要**的。如果只依赖路由专家，这些通用知识可能分散在多个专家中，需要"凑齐"多个专家才能用——浪费路由配额。

共享专家始终激活，承担"通用基础变换"的角色，让路由专家可以专注于**专业化**的知识。

**类比**：共享专家是"全科医生"（所有人都可以看），路由专家是"专科医生"（按需转诊）。

---

### 3.6 SwiGLU 激活 + 数值裁剪

```python
gate = gate.clamp(max=10.0)            # gate 最大 10
up = up.clamp(min=-10.0, max=10.0)     # up 在 [-10, 10]
return silu(gate) * up
```

**为什么要裁剪？**

在 FP8/BF16 混合精度训练中：
- `gate_proj(x)` 可能产生极大值（如 50），经过 `silu` 后仍很大
- `up_proj(x)` 可能产生极大负值
- 两者相乘可能溢出 FP8 的表示范围（$\pm 448$）

裁剪到 $[-10, 10]$ 后：$|\text{silu}(10) \times 10| \approx |10 \times 10| = 100$，远低于溢出阈值。

**gate 只裁上界（max=10）不裁下界**：`silu` 在负值区间趋近 0，大负值不会产生大输出，所以下界不需要裁剪。

---

### 3.7 负载均衡辅助损失

```python
def load_balancing_loss_func(gate_logits, num_experts, top_k, attention_mask):
    routing_weights = softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = topk(routing_weights, top_k)
    expert_mask = one_hot(selected_experts, num_experts)

    tokens_per_expert = mean(expert_mask, dim=0)          # 每专家被选中比例
    router_prob_per_expert = mean(routing_weights, dim=0) # 每专家平均路由概率

    loss = sum(tokens_per_expert * router_prob_per_expert) * num_experts
```

这是 Switch Transformer 风格的负载均衡损失：

- `tokens_per_expert`：衡量实际分配（"多少 token 去了这个专家"）
- `router_prob_per_expert`：衡量路由倾向（"路由器多想送 token 去这个专家"）
- 两者乘积之和在**均匀分配时最小**

这迫使路由器均匀利用所有 256 个专家，防止"赢者通吃"。

---

## 四、总结：V4 优化全景图

```
┌───────────────────────────────────────────────────────────────────┐
│                    DeepSeek V4 Decoder Layer                       │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  输入: [B, S, 4, 4096] — 4条并行流                                │
│                                                                    │
│  ┌─── mHC (Attention Site) ───────────────────────────────┐       │
│  │  pre: 4流 → 1流 (加权坍缩)                              │       │
│  │  post: 子层输出放置权重 [0, 2]                            │       │
│  │  comb: 4×4 双随机混合矩阵 (Sinkhorn 20次迭代)            │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                    │
│  ┌─── Attention ──────────────────────────────────────────┐       │
│  │  Q: 4096 → 1024 → 64×512  (LoRA 两步投影)              │       │
│  │  KV: 4096 → 512  (单头, K=V)                           │       │
│  │  压缩: Sliding / CSA+Indexer / HCA                      │       │
│  │  输出: GroupedLinear(8组) → Linear → 4096               │       │
│  │  Reverse RoPE + Attention Sink                          │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                    │
│  h' = post ⊙ attn_output + comb^T @ h                             │
│                                                                    │
│  ┌─── mHC (MLP Site) ────────────────────────────────────┐       │
│  │  (同上)                                                 │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                    │
│  ┌─── MoE MLP ───────────────────────────────────────────┐       │
│  │  路由: HashRouter(前3层) / TopKRouter(后续层)           │       │
│  │  专家: 256路由专家(top-6) + 1共享专家                   │       │
│  │  激活: SwiGLU + clamp(±10)                              │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                    │
│  h' = post ⊙ mlp_output + comb^T @ h                              │
│                                                                    │
│  输出: [B, S, 4, 4096] — 4条并行流                                │
└───────────────────────────────────────────────────────────────────┘
```

### 各优化对应的核心瓶颈

| 瓶颈 | V4 优化方案 | 关键创新点 |
|------|------------|-----------|
| **KV cache 内存爆炸** | Shared-KV MQA + K=V | 128× 内存压缩 |
| **长序列 $O(n^2)$ 计算** | CSA (4:1) + HCA (128:1) + Lightning Indexer | 分层粒度压缩 + 智能检索 |
| **大输出投影开销** | Grouped Low-Rank Output Projection | 50% 参数减少 |
| **深层信号膨胀/消失** | mHC + Sinkhorn 双随机约束 | 信号守恒的自适应残差 |
| **路由崩塌** | Hash-MoE 引导层 + sqrtsoftplus + 分数校正偏置 | 确定性引导 + 均匀打分 |
| **参数容量 vs 计算量** | Sparse MoE (256专家, top-6) + 共享专家 | 36:1 的参数/计算比 |
| **混合精度数值溢出** | SwiGLU clamp + FP32 归一化 | 精准控制数值范围 |
| **RoPE 限制语义表达** | Partial RoPE (12.5%) + Reverse RoPE on Output | 分离位置与语义 |

### 设计哲学总结

> **"大处大方，小处节省"**
>
> V4 的参数量很大（256 个专家、64 个注意力头），但每个 token 的实际计算路径很窄（6 个专家、1 个 KV 头、分组投影）。多条并行流（mHC）增加了信息通道，但双随机约束保证了训练稳定性。压缩注意力（CSA/HCA）保留了远距离信息，但 Lightning Indexer 控制了检索成本。
>
> 最终效果：**在保持大模型容量的同时，实现接近小模型的推理效率**。
