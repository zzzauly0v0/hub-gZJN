# Week 15 作业：并行 Subagent Agent

## 作业目标

实现一个可以下发 subagent 的 Agent，能够并行完成多项工作。通过本次作业理解：

1. 为什么需要 subagent 并行？
2. 主 Agent 和 subagent 的职责分别是什么？
3. 并行执行相比串行执行的收益和局限。

## 核心设计

### 1. 主 Agent

- 接收整体任务，例如"2024 年中国新能源汽车市场调研"
- 根据任务自动拆分为多个子课题，例如"销量规模"、"主要厂商竞争格局"、"政策趋势"
- 通过 `dispatch_subagents` 方法派发 subagent
- 收齐所有 subagent 结果后，聚合输出综合报告

### 2. Subagent

- 每个 subagent 只负责一个子课题
- 当前实现使用 `mock_research` 模拟耗时调研任务
- 任务耗时 1~3 秒不等，由子课题哈希值决定，保证可复现

### 3. 并行机制

- 使用 `ThreadPoolExecutor` 同时运行多个 subagent
- 支持 `serial=True` 参数切换为串行模式，用于对比基线
- 统计指标：
  - `wall_clock`：实际 wall-clock 时间
  - `serial_sum`：所有 subagent 耗时之和（等价于串行总时间）
  - `speedup`：串行总时间 / wall-clock 时间

## 运行结果

```text
模式一：并行执行
并行统计：{"wall_clock": 3.0, "serial_sum": 9.0, "speedup": 3.0}

模式二：串行执行（作为并行基线对比）
并行统计：{"wall_clock": 9.0, "serial_sum": 9.0, "speedup": 1.0}
```

结果文件：`outputs/parallel_agent_result.json`

## 关键学习点

- **并行收益只在可并行的子任务部分**：本例中 3 个 subagent 完全独立，所以接近 3× 加速。
- **串行段是瓶颈**：如果主 Agent 的规划/聚合本身很耗时，总加速比会被拉低（Amdahl 定律）。
- **subagent 越多、越独立，并行收益越大**；但线程池 `max_workers` 不宜无限扩大。

## 薄弱点 / 待改进

1. 当前 subagent 用的是 mock 函数，没有真实联网搜索或 LLM 推理。
2. 主 Agent 的拆分逻辑是硬编码的，没有让 LLM 自动决策。
3. 缺少错误处理：如果某个 subagent 失败，主 Agent 如何降级？
4. 缺少 SSE/可视化，无法实时观察每个 subagent 的执行过程。

## 扩展方向

1. 把 `mock_research` 替换为真实的 Tavily 搜索或 LLM 调用。
2. 让主 Agent 根据问题语义自动判断需要拆成几个子课题。
3. 增加子任务失败重试、结果去重、结果置信度评估。
4. 对比课程项目 `market_research_subagents`，学习真实 ReAct 引擎中的 subagent 派发与 trace 捕获。
