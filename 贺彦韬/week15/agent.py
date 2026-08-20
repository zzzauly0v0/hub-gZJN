"""
旅行规划 Agent：主 agent 自主判断"这个问题要不要拆给多个子助手并行查"，
子助手只能联网搜索、不能再往下派发——是"动态 Orchestrator-Workers"架构的一个具体应用
（主 agent 当指挥官决定怎么拆、拆几份，子 agent 当工人各自把分到的活干完，不再转包）。
"""
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from search_tool import web_search, format_search_result

logger = logging.getLogger(__name__)

MAIN_SYSTEM = """你是一个旅行规划助手。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词），用于能一次查到答案的单一事实问题
- dispatch_subagents：派发多个子助手并行查不同方面的信息（参数=用 | 分隔的多个子课题）

【决策原则】
- 只要是"规划一次旅行"这种涉及 2 个及以上方面的问题（交通、住宿、行程景点、美食/注意事项等），
  必须用 dispatch_subagents 把各方面拆给子助手并行查，不要自己一个一个 web_search
- 只有单一事实问题（比如"东京现在几点"、"日本签证要多久"）才直接 web_search
- 拿到子助手的结果后，把它们综合成一份分板块的行程建议，末尾提醒不确定或需要临行前确认的信息

【示例】
Question: 帮我规划一次3天的大阪旅行，需要交通、住宿、景点行程、当地美食建议
Thought: 这是多方面的旅行规划问题，必须派发子助手并行查，不能自己一个个搜
Action: dispatch_subagents
Action Input: 大阪往返交通方式与大致票价 | 大阪3天住宿区域与酒店推荐 | 大阪3天经典景点行程安排 | 大阪当地美食与就餐建议
Observation: 并行调研完成：4 个子助手...(各子课题结果)
Thought: 四个方面信息都齐了，可以综合成完整行程
Final Answer: (分板块的完整行程建议)"""


def _dispatch_subagents(action_input: str, shared_state: dict = None,
                         on_subagent_step: Callable = None,
                         on_subagent_done: Callable = None,
                         serial: bool = False) -> str:
    """dispatch_subagents 工具的具体实现：拆子课题 -> 造子 agent -> 并行/串行跑完 -> 汇总成一段文字。

    并行的价值不是"少做事"，是把 N 个独立子任务的墙钟时间从"一个个加起来"压到
    "约等于其中最慢的那一个"。serial=True 时退化成 for 循环，作为跟并行版本
    量化对比用的基线（见 main.py 的 --compare）。
    """
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:5]
    if not subtopics:
        return "未解析出子课题"

    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    # 每个子课题造一个新的 ReActLoop 实例，tools 只给 web_search 一个 —— 子 agent
    # "不能再往下派发"的唯一原因就在这里：没有别的拦截机制，就是没把 dispatch_subagents
    # 这个函数塞进它的 tools 字典，它的系统提示里也就压根不会提到这个工具存在。
    defs = []
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        sub = ReActLoop(
            agent_id=sid,
            tools={"web_search": (lambda q, **_: format_search_result(web_search(q)),
                                   "联网搜索，参数是查询词")},
            max_steps=4,
        )
        defs.append((sid, sub, topic))

    def _run_one(sid: str, sub: ReActLoop, topic: str):
        return sid, sub.run(topic, on_step=(
            lambda step, sid=sid: on_subagent_step(sid, step) if on_subagent_step else None))

    def _record(sid: str, topic: str, res: dict):
        shared_state["subagents"][sid] = {"subtopic": topic, **res}
        if on_subagent_done:
            on_subagent_done(sid, res["duration"], topic)

    t0 = time.time()
    results: dict[str, tuple[str, dict]] = {}

    if serial:
        # 串行：一个接一个跑完才跑下一个，专门用来跟并行版本对比耗时
        for sid, sub, topic in defs:
            sid, res = _run_one(sid, sub, topic)
            results[sid] = (topic, res)
            _record(sid, topic, res)
    else:
        # 并行：ThreadPoolExecutor 同时把所有子 agent 的任务扔进线程池，
        # as_completed 按"谁先跑完"的顺序把结果吐出来
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, sid, sub, topic): sid for sid, sub, topic in defs}
            for fut in as_completed(futs):
                sid, res = fut.result()
                topic = next(t for s, _, t in defs if s == sid)  # fut.result() 不带 topic，回 defs 里查
                results[sid] = (topic, res)
                _record(sid, topic, res)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    speedup = round(serial_sum / wall, 2) if wall else 0
    shared_state.setdefault("parallel_stats", []).append(
        {"n_subagents": len(defs), "wall_clock": wall, "serial_sum": serial_sum, "speedup": speedup})

    # 每个子助手的完整答案截短到 600 字再拼进汇总，避免主 agent 的上下文被撑爆
    parts = [f"【{topic}】(用时{r['duration']}s)\n{r['final_answer'][:600]}"
             for sid, (topic, r) in results.items()]
    return (f"并行调研完成：{len(defs)} 个子助手，wall-clock {wall}s"
            f"（若串行需 {serial_sum}s，加速 {speedup}×）\n\n" + "\n\n".join(parts))


def plan_trip(question: str, serial: bool = False,
              on_main_step: Callable = None,
              on_subagent_step: Callable = None,
              on_subagent_done: Callable = None) -> dict:
    """对外入口：跑一次旅行规划，返回 {final_answer, main_trace, subagents, parallel_stats}。"""
    shared_state = {"subagents": {}, "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        return _dispatch_subagents(action_input, shared_state=shared_state,
                                    on_subagent_step=on_subagent_step,
                                    on_subagent_done=on_subagent_done, serial=serial)

    main = ReActLoop(
        agent_id="main",
        tools={
            "web_search": (lambda q, **_: format_search_result(web_search(q)),
                           "联网搜索一次，参数=查询词"),
            "dispatch_subagents": (dispatch_tool,
                                   "派发多个子助手并行查不同方面，参数=用 | 分隔的多个子课题"),
        },
        max_steps=8,
        system_prompt=MAIN_SYSTEM,
    )
    result = main.run(question, on_step=on_main_step, shared_state=shared_state)
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
    }
