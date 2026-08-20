"""
主Agent + 并行子Agent编排
智能旅行规划助手
"""
import os
import re
import time
import json
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List

from react_loop import ReActLoop
from llm_client import LLMClient
from travel_api import (
    search_attractions, format_attractions,
    get_weather, format_weather,
    search_food, format_food
)

logger = logging.getLogger(__name__)

# ============================================================
# 主Agent系统提示
# ============================================================
MAIN_SYSTEM = """你是智能旅行规划师。你的任务是为用户规划旅行攻略。

你有以下工具可以调用：

1. **search_attractions**: 搜索城市的景点信息
   - 参数: 城市名称
   - 返回: 景点列表（名称、地址、类型、评分、费用）

2. **get_weather**: 查询城市的天气预报
   - 参数: 城市名称（格式："城市,天数"，如"北京,3"）
   - 返回: 未来几天的天气

3. **search_food**: 搜索城市的美食推荐
   - 参数: 城市名称
   - 返回: 美食餐厅列表

【决策原则】
- 如果用户问题涉及**多个方面**（如"规划X日游"、"去X地旅游"、"X地攻略"），
  必须**在同一轮并行调用**多个工具获取信息，然后综合成攻略。
  示例：用户说"帮我规划北京3日游" 
  → 同时调用 search_attractions(北京)、get_weather(北京,3)、search_food(北京)
  
- 如果用户只问单一问题（如"北京有什么景点"），只需调用对应工具。

【执行机制】
- 当你在一步中**同时调用多个工具**时，系统会自动把它们派发给多个子Agent并行执行；
  当你**只调用一个工具**时，将由你自己直接执行。无需担心执行方式，只需正确选择工具即可。

【报告格式】
最终输出必须是结构化的旅行攻略，包含：
1. 📸 景点推荐（每日行程建议）
2. 🌤 天气提醒（穿衣建议）
3. 🍜 美食推荐
4. 💡 实用建议（交通、住宿等）

要求：信息丰富、条理清晰、有实用价值。
"""


# ============================================================
# 工具函数包装（兼容ReActLoop的调用格式）
# ============================================================
def _search_attractions(query: str) -> str:
    """搜索景点"""
    result = search_attractions(query)
    return format_attractions(result)


def _get_weather(query: str) -> str:
    """查询天气，格式：城市,天数"""
    parts = query.split(",")
    city = parts[0].strip()
    days = int(parts[1].strip()) if len(parts) > 1 else 3
    result = get_weather(city, days)
    return format_weather(result)


def _search_food(query: str) -> str:
    """搜索美食"""
    result = search_food(query)
    return format_food(result)


# ============================================================
# 子Agent调度（并行执行）
# 多问题场景：一步多个工具调用 → 每个工具调用派发一个子Agent
# ============================================================
_TOOL_MAP = {
    "search_attractions": (_search_attractions, "搜索景点，参数=城市名称"),
    "get_weather": (_get_weather, "查询天气，参数=城市,天数"),
    "search_food": (_search_food, "搜索美食，参数=城市名称"),
}
_TASK_DESC = {
    "search_attractions": "搜索景点",
    "get_weather": "查询天气",
    "search_food": "搜索美食",
}


def _dispatch_tool_calls(
    calls: List[tuple],
    shared_state: Dict = None,
    on_subagent_step: Callable = None,
    on_subagent_done: Callable = None,
    on_dispatch: Callable = None,
    serial: bool = False
) -> List[str]:
    """
    将多个工具调用派发给子Agent并行执行
    
    calls: [(工具名, 查询参数), ...]
    返回: 与calls顺序一致的观察结果列表，供主Agent继续推理
    """
    if not calls:
        return []
    
    shared_state = shared_state or {}
    shared_state.setdefault("subagents", {})
    
    # 创建子Agent：每个工具调用一个，只持有对应的单一工具
    defs = []  # (sid, sub, task_desc)
    for tool_name, query in calls:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        if tool_name not in _TOOL_MAP:
            defs.append((sid, None, f"{tool_name}: {query}"))
            continue
        sub = ReActLoop(
            agent_name=sid,
            tools={tool_name: _TOOL_MAP[tool_name]},
            max_steps=3,
            model_tag="deepseek-chat(子)",
            system_prompt="你是专业的旅行信息调研员，根据用户需求调用工具获取信息，给出简洁准确的回答。"
        )
        defs.append((sid, sub, f"{_TASK_DESC.get(tool_name, tool_name)}: {query}"))
    
    # 记录派发信息
    dispatch_info = {
        "subtasks": [task for _, _, task in defs],
        "subagent_ids": [sid for sid, _, _ in defs]
    }
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)
    
    # ── 执行 ──
    t0 = time.time()
    results = {}
    runnable = [(sid, sub, task) for sid, sub, task in defs if sub is not None]
    
    def _run_one(sid, sub, task):
        return sid, sub.run(task, on_step=lambda step: on_subagent_step(sid, step) if on_subagent_step else None)
    
    if serial:
        # 串行执行
        for sid, sub, task in runnable:
            sid, res = _run_one(sid, sub, task)
            results[sid] = (task, res)
            shared_state["subagents"][sid] = {
                "subtask": task,
                "trace": res["trace"],
                "duration": res["duration"],
                "final_answer": res["final_answer"]
            }
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], task)
    else:
        # 并行执行（核心优势）
        with ThreadPoolExecutor(max_workers=len(runnable)) as pool:
            futures = {pool.submit(_run_one, sid, sub, task): sid for sid, sub, task in runnable}
            for future in as_completed(futures):
                sid, res = future.result()
                task = next(t for s, _, t in defs if s == sid)
                results[sid] = (task, res)
                shared_state["subagents"][sid] = {
                    "subtask": task,
                    "trace": res["trace"],
                    "duration": res["duration"],
                    "final_answer": res["final_answer"]
                }
                if on_subagent_done:
                    on_subagent_done(sid, res["duration"], task)
    
    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    speedup = round(serial_sum / wall, 2) if wall > 0 else 0
    
    shared_state.setdefault("parallel_stats", []).append({
        "n_subagents": len(runnable),
        "wall_clock": wall,
        "serial_sum": serial_sum,
        "speedup": speedup
    })
    
    # 按调用顺序组装观察结果（与tool_calls一一对应）
    observations = []
    for sid, sub, task in defs:
        if sub is None:
            observations.append(f"[未知工具] {task} 执行失败：该工具不可用")
        else:
            _, res = results[sid]
            observations.append(f"[子任务] {task}（用时{res['duration']}s）\n{res['final_answer'][:500]}")
    
    return observations


# ============================================================
# 查询意图分类：判断是"综合旅游需求"还是"单一查询"
# 综合需求 → 派发多个子Agent；单一查询 → 主Agent自己处理
# ============================================================
# 综合旅游规划意图关键词
_PLAN_KEYWORDS = [
    "旅游", "规划", "攻略", "度假", "旅行", "几日游",
    "行程", "怎么玩", "游玩", "带我去", "安排", "路线", "怎么安排"
]

# 各查询方面关键词（用于判断涉及几个方面）
_ASPECT_KEYWORDS = {
    "search_attractions": ["景点", "景区", "好玩", "游玩的地方", "博物馆", "打卡", "逛逛", "去哪"],
    "get_weather": ["天气", "气温", "温度", "下雨", "穿衣", "天冷"],
    "search_food": ["美食", "好吃", "餐厅", "小吃", "饭店", "吃什么", "吃啥"],
}

_KNOWN_CITIES = [
    "北京", "上海", "广州", "深圳", "杭州", "成都", "重庆",
    "西安", "南京", "武汉", "苏州", "天津", "长沙", "厦门",
    "青岛", "三亚", "昆明", "哈尔滨", "大连", "桂林"
]


def _needs_dispatch(query: str) -> bool:
    """判断是否属于综合旅游需求（需要派发多个子Agent并行调研）"""
    if any(k in query for k in _PLAN_KEYWORDS):
        return True
    # 即使没有规划类关键词，若同时涉及多个方面（如景点+美食），也视为综合需求
    aspects = [name for name, kws in _ASPECT_KEYWORDS.items() if any(k in query for k in kws)]
    return len(aspects) >= 2


def _extract_city(query: str) -> str:
    """从查询中提取目的地城市名（优先关键词匹配，其次LLM兜底）"""
    for city in _KNOWN_CITIES:
        if city in query:
            return city
    try:
        resp = LLMClient().simple_chat(
            f"从下面的句子中提取目的地城市名，只输出城市名本身，不要输出任何其他文字：\n{query}"
        ).strip()
        if resp and "调用失败" not in resp:
            return resp.split("，")[0].split(",")[0].strip()
    except Exception:
        pass
    return ""


def _extract_days(query: str) -> int:
    """从查询中提取游玩天数（默认3天，上限7天）"""
    m = re.search(r"(\d+)\s*日游", query)
    if m:
        return max(1, min(int(m.group(1)), 7))
    m = re.search(r"(\d+)\s*天", query)
    if m:
        return max(1, min(int(m.group(1)), 7))
    return 3


# 主Agent总结归纳专用系统提示（子Agent已调研完毕）
MAIN_SUMMARY_SYSTEM = """你是智能旅行规划师。子Agent已经完成了景点、天气、美食等信息的并行调研。
请根据用户需求和给定的调研结果，综合成一份完整、结构化的旅行攻略，包含：
1. 📸 景点推荐（每日行程建议）
2. 🌤 天气提醒（穿衣建议）
3. 🍜 美食推荐
4. 💡 实用建议（交通、住宿等）

要求：只使用调研结果中给出的信息，不要编造，信息丰富、条理清晰、有实用价值。
"""


# ============================================================
# 主入口函数
# ============================================================
def plan_travel(
    query: str,
    on_main_step: Callable = None,
    on_subagent_step: Callable = None,
    on_subagent_done: Callable = None,
    on_dispatch: Callable = None,
    serial: bool = False
) -> Dict:
    """
    执行旅行规划
    
    Args:
        query: 用户查询，如"帮我规划北京3日游"
        on_main_step: 主Agent每一步的回调
        on_subagent_step: 子Agent每一步的回调
        on_subagent_done: 子Agent完成的回调
        on_dispatch: 派发子Agent的回调
        serial: 是否串行执行（用于对比测试）
    
    Returns:
        {
            "final_answer": str,
            "main_trace": list,
            "subagents": dict,
            "parallel_stats": list,
            "dispatches": list
        }
    """
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}
    
    # ── 多工具调用派发器（保险机制）：若主Agent一步调用多个工具，也派发给子Agent并行执行 ──
    def multi_tool_dispatch(calls):
        return _dispatch_tool_calls(
            calls,
            shared_state=shared_state,
            on_subagent_step=on_subagent_step,
            on_subagent_done=on_subagent_done,
            on_dispatch=on_dispatch,
            serial=serial
        )
    
    # 主Agent（带工具）：用于单一查询场景
    def build_main_agent():
        return ReActLoop(
            agent_name="main",
            tools={
                "search_attractions": (_search_attractions, "搜索景点，参数=城市名称"),
                "get_weather": (_get_weather, "查询天气，参数=城市,天数"),
                "search_food": (_search_food, "搜索美食，参数=城市名称"),
            },
            max_steps=8,
            model_tag="deepseek-chat(主)",
            system_prompt=MAIN_SYSTEM,
            multi_tool_dispatch=multi_tool_dispatch
        )
    
    # ── 意图判定 ──
    if _needs_dispatch(query):
        # 综合旅游需求：拆分景点/天气/美食子任务 → 派发多个子Agent并行查询
        city = _extract_city(query)
        if city:
            days = _extract_days(query)
            calls = [
                ("search_attractions", city),
                ("get_weather", f"{city},{days}"),
                ("search_food", city),
            ]
            observations = _dispatch_tool_calls(
                calls,
                shared_state=shared_state,
                on_subagent_step=on_subagent_step,
                on_subagent_done=on_subagent_done,
                on_dispatch=on_dispatch,
                serial=serial
            )
            # 主Agent归纳总结
            main = ReActLoop(
                agent_name="main",
                tools={},
                max_steps=2,
                model_tag="deepseek-chat(主)",
                system_prompt=MAIN_SUMMARY_SYSTEM
            )
            summary_query = (
                f"用户需求：{query}\n\n"
                f"以下是子Agent并行调研到的信息：\n" + "\n\n".join(observations) +
                f"\n\n请综合以上信息，输出完整、结构化的旅行攻略。"
            )
            result = main.run(summary_query, on_step=on_main_step, shared_state=shared_state)
        else:
            # 未识别出城市 → 退化为普通模式，由主Agent自行处理
            result = build_main_agent().run(query, on_step=on_main_step, shared_state=shared_state)
    else:
        # 单一查询（如"北京天气"）→ 主Agent自己直接查询
        result = build_main_agent().run(query, on_step=on_main_step, shared_state=shared_state)
    
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
    }


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    
    # 简单测试
    print("=" * 60)
    print("智能旅行规划助手 - 测试")
    print("=" * 60)
    
    query = "帮我规划北京3日游，要包含景点、天气和美食推荐"
    print(f"\n用户问题: {query}\n")
    
    result = plan_travel(query)
    
    print("\n" + "=" * 60)
    print("主Agent执行轨迹:")
    for step in result["main_trace"]:
        print(f"  Step {step['step']}: {step['action']}")
    
    print(f"\n派发子Agent: {len(result['dispatches'])} 次")
    print(f"并行统计: {result['parallel_stats']}")
    
    print("\n" + "=" * 60)
    print("📋 旅行攻略:")
    print("=" * 60)
    print(result["final_answer"])
