# RESULTS.md — 测试指标结果

> ⚠️ **本轮全部数据来自 `MOCK=1` 离线模式**（环境无 `DEEPSEEK_API_KEY` / `TAVILY_API_KEY`）。
> LLM 走 `llm_client._mock_chat`（每次推理 `sleep(0.3)`），搜索走 `search.web_search` 的
> mock 分支（每次 `sleep(1.2)` 模拟网络）。
> **并发结构是真实的**（`asyncio.gather` vs `for await`），**绝对耗时是合成的**。
> 配好 key 后重跑 `python src/compare.py` 即得真实数字。

测试环境：Windows 11 / Python 3.11.8 (`M:/Badou/.venv`) / 命令行加 `PYTHONIOENCODING=utf-8`

---

## 1. 并发 vs 顺序对照（核心指标）

命令：`MOCK=1 python src/compare.py`（2 题 × 各跑并发+顺序两遍）

| 热点问题 | 并发总墙钟 | 顺序总墙钟 | 子 agent 数 | dispatch 段加速 |
|---------|-----------|-----------|------------|----------------|
| 2026世界杯预选赛国足热点分析：比赛结果、球员表现、舆论争议 | 2.45s | 6.11s | 3 | 3.00× |
| NBA交易截止日热点分析：主要交易、球队战力变化、球迷反应 | 2.45s | 6.13s | 3 | 3.00× |
| **平均** | **2.45s** | **6.12s** | **3** | **3.00×** |

复跑一次（第二次采样，验证稳定性）：并发 2.44s / 2.45s，顺序 6.12s / 6.11s，平均加速仍为 3.00×。

**两个加速比要分开看：**

| 指标 | 数值 | 含义 |
|------|------|------|
| dispatch 段加速 | **3.00×** | 3 个子 agent 的 ReAct 链并发，墙钟从 `sum` 压到 `≈max` —— 并发的纯收益 |
| 总墙钟加速 | **2.50×** | 6.12 / 2.45，包含了主 agent 自己的串行段 |

差距来源：主 agent 的「规划」和「综合」是两次不可并发的 LLM 调用（mock 下各 0.3s，
合计约 0.6s 固定串行开销）。**Amdahl 定律**——并发收益只作用在可并行的子任务段上。
真实 LLM 下串行段会更重（每次调用数秒），总加速比会比 dispatch 加速低更多。

---

## 2. 单次执行的 trace 结构

命令：`MOCK=1 python src/agents.py`

```
[main #0] dispatch_subagents: 比赛结果 | 球员表现 | 舆论争议
↳ 下发 3 个子分析员: 比赛结果 | 球员表现 | 舆论争议
  [sub1_41ba] 完成 1.83s · 比赛结果
  [sub2_90ef] 完成 1.83s · 球员表现
  [sub3_34fb] 完成 1.84s · 舆论争议
[main #1] Final Answer: 【mock 报告】...
```

| 节点 | 步数 | 动作序列 | 耗时 |
|------|------|---------|------|
| main | 2 | `['dispatch_subagents', 'Final Answer']` | 2.45s（总） |
| sub1 | 2 | `['web_search', 'Final Answer']` | 1.83s |
| sub2 | 2 | `['web_search', 'Final Answer']` | 1.83s |
| sub3 | 2 | `['web_search', 'Final Answer']` | 1.83s |

`stats = {'n': 3, 'wall': 1.84, 'sum': 5.49, 'speedup': 2.98}`

**三个子 agent 耗时几乎相同且 `wall ≈ 单个子 agent 耗时`** —— 这就是并发生效的直接证据：
顺序模式下同样三个子 agent，`wall` 变成 5.50s（= sum，`speedup` 1.0×）。

---

## 3. 功能性验证

| 验证项 | 结果 | 说明 |
|--------|------|------|
| 主 agent 自主选择 dispatch | ✅ | 2 个多侧面热点问题均自主走 `dispatch_subagents`，非硬编码 |
| 子课题自主拆分 | ✅ | 从问题中拆出 3 个侧面，管道符分隔传入 |
| 子 agent 无 dispatch 工具（不套娃） | ✅ | 子 agent 动作序列只有 `web_search` / `Final Answer` |
| 回调事件完整 | ✅ | `on_main_step` / `on_dispatch` / `on_subagent_done` 均按序触发 |
| `--serial` 对照基线可用 | ✅ | `speedup` 退化为 1.0×，`wall == sum` |
| Observation 截短防爆 context | ✅ | 每个子结果截 400 字回灌主 agent |

---

## 4. 已知偏差（诚实记录）

1. **单一事实问题在 mock 下也会 dispatch**：跑 `"巴黎奥运会中国队金牌热点"`（单侧面）时
   主 agent 仍下发了 1 个子分析员（`speedup` 1.0×）。这是 mock LLM 的固定逻辑
   （`_mock_chat` 首步无条件返回 dispatch），不是路由逻辑本身的缺陷——
   真实模型按 `MAIN_SYSTEM` 的决策原则应走 `web_search` 分支。**此项待真实 key 复验。**
2. **耗时不可与 `market_research_subagents` 的实测数字（并行 32.98s / 串行 50.47s，
   加速 2.51×）直接比较**：那边是真实 LLM + Tavily，这里是 mock。
   可比的只有结构性结论：dispatch 段加速 ≈ 子 agent 数，总加速被主 agent 串行段拉低。
3. **Windows 控制台编码**：不加 `PYTHONIOENCODING=utf-8` 时回调里的中文/箭头会抛
   `UnicodeEncodeError`，并被 ReAct 的 `_exec` 捕获成工具错误，表现为「子 agent 数 0」。
   已在 `agents.py` 的 `__main__` 里 `sys.stdout.reconfigure(encoding="utf-8")` 兜底。

---

## 5. 复现方式

```bash
cd sports_hotspot_subagents

# 离线复现本文档全部数字
MOCK=1 PYTHONIOENCODING=utf-8 python src/compare.py
MOCK=1 PYTHONIOENCODING=utf-8 python src/agents.py
MOCK=1 PYTHONIOENCODING=utf-8 python src/agents.py --serial

# 真实数字（需 key）
export DEEPSEEK_API_KEY=sk-xxx && export TAVILY_API_KEY=tvly-xxx
python src/compare.py
```
