"""主 agent + 下发 subagent（体育热点分析 · 最小单元）

这个文件就是本项目的全部重点：
  1. 主 agent 也是一个 ReAct 循环，手里有 2 个工具
       web_search           —— 单一事实，自己搜一次就够
       dispatch_subagents   —— 多侧面热点，下发 N 个子分析员
     用哪个由 LLM 自己决定（不是硬编码拓扑）。
  2. dispatch_subagents 里 asyncio.gather 并发跑 N 个子 agent 的 ReAct，
     墙钟从 sum(各子 agent) 压到 ≈max(各子 agent)。
  3. 子 agent 就是同一个 ReActLoop，只是 tools 里少了 dispatch —— 所以不会无限套娃。

运行：
  python src/agents.py                          # 默认热点，并发
  python src/agents.py --serial                 # 退化成顺序（对照基线）
  python src/agents.py "梅西美职联表现：进球数、球队战绩、舆论评价"
"""
import sys, uuid, time, asyncio, argparse

from react_loop import ReActLoop
from search import web_search

MAIN_SYSTEM = """你是体育热点主分析师。可用工具：
{tools_desc}

【决策原则】
- 热点只要涉及 2 个以上侧面（赛况数据 / 人物表现 / 舆论争议 / 商业影响 / 后续走势…），
  必须用 dispatch_subagents 把侧面拆开下发给子分析员并发调研，不要自己一次次搜。
- 只有单一事实（如「昨晚湖人比分」）才自己 web_search。
- 拿到子分析员的汇总后，综合成一份分维度、带来源、末尾有结论的热点分析。

【示例】
Question: 世俱杯决赛热点分析：比赛过程、关键球员、舆论反响
Thought: 三个侧面，下发子分析员并发调研
Action: dispatch_subagents
Action Input: 世俱杯决赛比赛过程与关键数据 | 决赛关键球员表现 | 球迷与媒体舆论反响
Observation: 并发调研完成：3 个子分析员……
Thought: 三个维度都齐了
Final Answer: （分维度的热点分析报告）"""

SUB_TOOLS = {"web_search": (web_search, "联网搜索一次，参数=查询词")}


async def _run_subagent(sid: str, topic: str, ctx: dict) -> tuple:
    """一个子分析员：独立的 ReAct 循环，只有 web_search。"""
    sub = ReActLoop(sid, SUB_TOOLS, max_steps=3)
    on_step = ctx.get("on_subagent_step")
    res = await sub.run(topic, on_step=(lambda s: on_step(sid, s)) if on_step else None)
    ctx["subagents"][sid] = {"topic": topic, **res}
    if cb := ctx.get("on_subagent_done"):
        cb(sid, topic, res["duration"])
    return sid, topic, res


async def dispatch_subagents(action_input: str, ctx: dict = None, **_) -> str:
    """主 agent 的核心工具：把「课题1 | 课题2 | 课题3」下发成 N 个子 agent。

    并发（默认）：asyncio.gather，N 个子 agent 的 ReAct 交错跑，墙钟 ≈ max
    顺序（--serial）：for + await，墙钟 = sum —— A/B 对照基线
    """
    ctx = ctx if ctx is not None else {}
    ctx.setdefault("subagents", {})
    topics = [t.strip() for t in action_input.split("|") if t.strip()][:5]
    if not topics:
        return "未解析出子课题"

    jobs = [(f"sub{i+1}_{uuid.uuid4().hex[:4]}", t) for i, t in enumerate(topics)]
    if cb := ctx.get("on_dispatch"):
        cb([{"id": sid, "topic": t} for sid, t in jobs])      # 拓扑可视化：动态加节点

    t0 = time.time()
    if ctx.get("serial"):
        done = [await _run_subagent(sid, t, ctx) for sid, t in jobs]
    else:
        done = await asyncio.gather(*(_run_subagent(sid, t, ctx) for sid, t in jobs))
    wall = round(time.time() - t0, 2)

    total = round(sum(r["duration"] for _, _, r in done), 2)
    ctx["stats"] = {"n": len(jobs), "wall": wall, "sum": total,
                    "speedup": round(total / wall, 2) if wall else 0}
    body = "\n\n".join(f"【{t}】({r['duration']}s)\n{r['final_answer'][:400]}"
                       for _, t, r in done)                   # 截短，别撑爆主 agent context
    return (f"并发调研完成：{len(jobs)} 个子分析员，墙钟 {wall}s"
            f"（顺序需 {total}s，加速 {ctx['stats']['speedup']}×）\n\n{body}")


async def analyze(question: str, *, serial: bool = False, **callbacks) -> dict:
    """跑一次体育热点分析。callbacks: on_main_step / on_subagent_step / on_subagent_done / on_dispatch"""
    ctx = {"subagents": {}, "stats": None, "serial": serial, **callbacks}
    main = ReActLoop("main", max_steps=4, system=MAIN_SYSTEM, tools={
        "web_search": (web_search, "联网搜索一次，参数=查询词"),
        "dispatch_subagents": (dispatch_subagents,
                               "下发多个子分析员并发调研，参数=用 | 分隔的多个子课题"),
    })
    res = await main.run(question, on_step=callbacks.get("on_main_step"), ctx=ctx)
    return {"final_answer": res["final_answer"], "main_trace": res["trace"],
            "subagents": ctx["subagents"], "stats": ctx["stats"],
            "duration": res["duration"]}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows 控制台默认 gbk，中文会炸
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?",
                    default="2026世界杯预选赛国足热点分析：比赛结果、球员表现、舆论争议")
    ap.add_argument("--serial", action="store_true", help="子 agent 改顺序执行（对照基线）")
    a = ap.parse_args()

    r = asyncio.run(analyze(
        a.question, serial=a.serial,
        on_main_step=lambda s: print(f"[main #{s['idx']}] {s['action']}: {s['action_input'][:60]}"),
        on_dispatch=lambda subs: print(f"↳ 下发 {len(subs)} 个子分析员: "
                                       + " | ".join(s["topic"][:16] for s in subs)),
        on_subagent_done=lambda sid, t, d: print(f"  [{sid}] 完成 {d}s · {t[:24]}")))

    print(f"\n{'='*60}\n主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"子 agent 数: {len(r['subagents'])} | 统计: {r['stats']} | 总墙钟: {r['duration']}s")
    print(f"\n报告:\n{r['final_answer'][:600]}")
