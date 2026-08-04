# Week 14 作业：Skill 生成与优化对比

## 项目简介

本作业演示如何让大模型生成一个客服 Skill，然后让大模型从 **token 消耗** 角度优化它，并对比优化前后的效果。

以"数字商品退款"为场景，覆盖电子书、软件激活码、游戏点卡、会员卡等常见数字商品类型。

## 目录结构

```
homework/
├── data/
│   └── test_cases.json          # 10 道测试题及 ground truth
├── src/
│   ├── generate_skill.py        # 生成初始 Skill
│   ├── optimize_skill.py        # 优化 Skill（减少 token 消耗）
│   ├── evaluate_skill.py        # 评估 Skill 准确率与 token 数
│   └── run_all.py               # 完整流程
├── outputs/                     # 运行后生成
│   ├── skill_v1.md              # 初始 Skill
│   ├── skill_v2.md              # 优化后 Skill
│   ├── eval_v1.json             # 优化前评估结果
│   ├── eval_v2.json             # 优化后评估结果
│   ├── comparison_report.md     # 对比报告
│   └── logs/
│       └── run_log.json         # 运行摘要
├── requirements.txt
└── user_guide.md
```

## 环境要求

- Python >= 3.10
- 依赖：`openai`、`tiktoken`
- 需要 DeepSeek API Key（通过环境变量 `DEEPSEEK_API_KEY` 传入）

## 安装依赖

```bash
cd "week14自进化agent/week14 自进化agent/homework"
pip install -r requirements.txt
```

## 运行方式

### 设置 API Key

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-xxxxxx"

# Windows cmd
set DEEPSEEK_API_KEY=sk-xxxxxx

# Linux / Mac
export DEEPSEEK_API_KEY="sk-xxxxxx"
```

### 运行完整流程

```bash
python src/run_all.py
```

### 单独运行各步骤

```bash
# 仅生成初始 Skill
python src/generate_skill.py

# 仅优化 Skill（需先生成 skill_v1.md）
python src/optimize_skill.py

# 仅评估某个 Skill
python src/evaluate_skill.py outputs/skill_v1.md
python src/evaluate_skill.py outputs/skill_v2.md
```

## 输出说明

运行结束后：

- `outputs/skill_v1.md`：大模型生成的初始数字商品退款 Skill
- `outputs/skill_v2.md`：大模型优化后的 Skill
- `outputs/comparison_report.md`：包含 token 数、准确率、响应时间的对比报告
- `outputs/eval_v1.json` / `outputs/eval_v2.json`：每题详细评估结果
- `outputs/logs/run_log.json`：运行摘要

核心对比指标：

| 指标 | 说明 |
|------|------|
| Skill token 数 | Skill 文本的 token 估算值，衡量输入成本 |
| 测试集准确率 | 在 10 道固定测试题上的正确率 |
| 平均响应时间 | 每题 LLM 推理耗时 |
