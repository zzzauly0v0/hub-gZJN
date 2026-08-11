# 并行 Subagent Agent 作业

## 项目简介

实现一个可下发 subagent 的 Agent：主 Agent 接收整体任务后拆分为多个子任务，通过 `ThreadPoolExecutor` 并行派发多个 subagent 执行，最后聚合结果输出综合报告。

本作业重点展示：

- 主 Agent 的任务拆分与派发
- 多个 subagent 的并行执行
- 并行 vs 串行模式的量化对比（wall-clock 与加速比）

## 文件结构

```
homework/
├── parallel_agent.py          # 主程序：ParallelAgent 实现
├── outputs/
│   └── parallel_agent_result.json  # 运行结果统计
└── user_guide.md              # 使用说明
```

## 安装依赖

仅使用 Python 标准库：

- `concurrent.futures`
- `uuid`
- `hashlib`
- `json`
- `time`
- `os`
- `typing`

无需安装第三方包。

## 运行方式

```bash
cd homework
python parallel_agent.py
```

## 输出说明

1. **控制台输出**：
   - 主 Agent 拆分子任务的日志
   - 每个 subagent 完成任务的日志
   - 综合报告（含各子任务结果）
   - 并行统计：`wall_clock`、`serial_sum`、`speedup`

2. **结果文件**：
   - 路径：`outputs/parallel_agent_result.json`
   - 内容：包含并行模式和串行模式的完整统计信息

## 示例运行结果

| 模式 | wall-clock | 串行时长之和 | 加速比 |
|------|-----------|------------|--------|
| 并行 | 3.0s      | 9.0s       | 3.0×   |
| 串行 | 9.0s      | 9.0s       | 1.0×   |

> 注意：加速比取决于子任务数量与单个任务耗时。子任务越多、越独立，并行收益越明显。
