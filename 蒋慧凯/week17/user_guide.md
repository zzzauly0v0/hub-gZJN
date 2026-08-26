# GRPO 算术题强化学习作业

## 1. 项目简介

本项目在消费级显卡（RTX 4060 Laptop 8GB）上完整演示 GRPO（Group Relative Policy Optimization）强化学习训练：以 Qwen2-0.5B-Instruct 为基座，通过可程序化验证的规则奖励，让模型同时学会按 `<answer>` 标签输出格式和提高算术题正确率。训练前后通过同一评估集（seed=42，6 难度 × 50 题）进行配对比较。

## 2. 目录结构

```
grpo_arithmetic/
├── src/
│   ├── probe_baseline.py      # 基线摸底 / 训练后评估
│   ├── train_grpo.py          # GRPO 训练主脚本
│   ├── compare_results.py     # 训练前后对比 + 训练曲线
│   └── trl_compat.py          # trl 0.21 + transformers 5.x 兼容补丁
├── outputs/
│   ├── baseline_probe.json           # 基线指标
│   ├── post_train_probe.json         # 全量训练后指标
│   ├── post_train_probe_lora.json    # LoRA 训练后指标
│   ├── train_log.json                # 全量训练日志
│   ├── train_log_lora.json           # LoRA 训练日志
│   ├── figures/train_curves.png      # 训练曲线对比图
│   ├── grpo_ckpt/                    # 全量微调 checkpoint
│   └── grpo_lora_ckpt/               # LoRA adapter checkpoint
├── requirements.txt
└── user_guide.md
```

## 3. 环境要求

- Python 3.10 ~ 3.12（项目依赖的 torch 2.6.0 未提供 Python 3.14 wheel）
- CUDA 12.6 及以上（训练需 GPU；纯 CPU 运行训练极慢不推荐）
- 关键依赖：
  - torch==2.6.0+cu126
  - transformers==5.5.3
  - trl==0.21.0
  - peft==0.15.0
  - accelerate==1.5.2
  - datasets
  - matplotlib

预训练模型：`pretrain_models/Qwen2-0.5B-Instruct`，脚本中已按项目相对路径自动定位。

## 4. 输入输出与运行方式

### 安装依赖

```bash
pip install -r requirements.txt
```

若 PyTorch wheel 索引不可用，可改用：

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 运行全流程

```bash
# 1. 基线摸底
python src/probe_baseline.py

# 2. 全量 GRPO 训练（默认 200 步，约 3 分钟）
python src/train_grpo.py

# 3. LoRA 训练（显存不足时使用）
python src/train_grpo.py --lora

# 4. 训练后评估（必须与基线同一 seed）
python src/probe_baseline.py --model outputs/grpo_ckpt --out outputs/post_train_probe.json --seed 42
python src/probe_baseline.py --model outputs/grpo_lora_ckpt --out outputs/post_train_probe_lora.json --seed 42

# 5. 生成对比报告与训练曲线
python src/compare_results.py
```

### 输出结果

- 指标 JSON：`outputs/*_probe.json`、`outputs/train_log*.json`
- 对比图表：`outputs/figures/train_curves.png`
- 模型 checkpoint：`outputs/grpo_ckpt/`、`outputs/grpo_lora_ckpt/`

对比脚本会在终端打印训练前后格式率 / greedy 正确率 / pass@8 三方对照表，以及典型样例的 greedy 输出变化。
