# sports_hotspot_subagents — 体育热点分析
## 1. 核心范式（动态 Orchestrator-Workers）

```
用户热点问题
   ↓
主 agent（ReAct，2 个工具）
   ├─ 单一事实  → web_search → Final Answer
   └─ 多个侧面  → dispatch_subagents("赛况 | 球员 | 舆论")
                     ↓  asyncio.gather 并发
            ┌ sub1 ReAct(web_search) ┐
            ├ sub2 ReAct(web_search) ┤  墙钟 ≈ max，不是 sum
            └ sub3 ReAct(web_search) ┘
                     ↓ 汇总（截短后回灌）
            主 agent 综合成热点分析报告
```

关键点：**派几个、派什么由主 agent 的 LLM 自己决定**，拓扑运行时生长；
子 agent 的 tools 里没有 `dispatch_subagents`，所以不会无限套娃。

## 2. 文件（4 个源文件，共 ~230 行）

| 文件 | 职责 |
|------|------|
| `src/agents.py` | **重点**：`MAIN_SYSTEM` 决策提示 + `dispatch_subagents` 并发下发 + `analyze()` 入口 |
| `src/react_loop.py` | 通用异步 ReAct 循环，主 agent / 子 agent 共用，区别只在 tools |
| `src/search.py` | `web_search` 工具，httpx 异步调 Tavily |
| `src/llm_client.py` | `AsyncOpenAI` 单轮对话；无 key 时降级为离线 mock |
| `src/compare.py` | 并发 vs 顺序 A/B，量化 dispatch 收益 |

异步是贯穿的：`llm_chat` / `web_search` / `ReActLoop.run` 全是协程，
所以 `dispatch_subagents` 里一行 `asyncio.gather(...)` 就让 N 条 ReAct 链真正同时跑。

## 3. 运行

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-xxx      # 主/子 agent 推理
export TAVILY_API_KEY=tvly-xxx      # 联网搜索

python src/agents.py                                   # 默认热点
python src/agents.py "梅西美职联热点：进球数据、球队战绩、舆论评价"
python src/agents.py --serial                          # 子 agent 退化为顺序（对照）
python src/compare.py                                  # 并发 vs 顺序对照表
```

**无 key 也能跑**（验证下发+并发链路）：加 `MOCK=1`，LLM 和搜索都走离线假实现。
Windows 下若中文乱码，加 `PYTHONIOENCODING=utf-8`。

## 4. 实测（MOCK=1 离线，2 题）

| 指标 | 并发(gather) | 顺序(for await) |
|------|-------------|----------------|
| 平均总墙钟 | **2.45s** | 6.12s |
| dispatch 段加速 | **3.00×** | 1.0×（基线） |

主 agent 每次都自主选择了 `dispatch_subagents` 并拆成 3 个子课题，
动作序列 `['dispatch_subagents', 'Final Answer']`。

总墙钟加速（2.5×）略低于 dispatch 段加速（3.0×）：主 agent 自身的
「规划 + 综合」两次 LLM 调用是串行段，不可并发——Amdahl 定律，符合预期。

## 5. 相比 market_research_subagents 砍掉的部分

只保留下发 subagent 这一条主线，去掉了 SSE 服务、拓扑可视化前端、
trace 持久化、失败重试；`on_*` 回调接口保留了（`analyze(..., on_dispatch=..., on_subagent_step=...)`），
需要接可视化时直接往上挂即可。
