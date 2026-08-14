"""LangGraph 图流程定义

图流程：scrape → dispatch → analyze（并行 fan-out）→ summarize
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from baidu_hotspot_agent.agents.analyst import analyze_hotspot
from baidu_hotspot_agent.agents.orchestrator import (
    dispatch_items,
    generate_summary,
    scrape_hotspots,
)
from baidu_hotspot_agent.state import AgentState


def _should_summarize(state: dict) -> str:
    """条件边：根据是否有热搜数据决定下一步

    - 有数据 → 进入 dispatch（分发）节点
    - 无数据 → 直接进入 summarize（汇总）节点
    """
    items = state.get("hotspot_items", [])
    if items:
        return "dispatch"
    return "summarize"


def build_graph() -> StateGraph:
    """构建并编译 LangGraph 图流程"""

    graph = StateGraph(AgentState)

    # ── 添加节点 ──
    graph.add_node("scrape", scrape_hotspots)
    graph.add_node("dispatch", dispatch_items)
    graph.add_node("analyze_hotspot", analyze_hotspot)
    graph.add_node("summarize", generate_summary)

    # ── 添加边 ──
    # 起点 → 抓取
    graph.set_entry_point("scrape")

    # 抓取 → 条件判断（有数据走分发，无数据走汇总）
    graph.add_conditional_edges(
        "scrape",
        _should_summarize,
        {"dispatch": "dispatch", "summarize": "summarize"},
    )

    # 分发 → 子 Agent（Send API 自动 fan-out 并行）
    graph.add_edge("dispatch", "analyze_hotspot")

    # 子 Agent → 汇总
    graph.add_edge("analyze_hotspot", "summarize")

    # 汇总 → 结束
    graph.add_edge("summarize", END)

    return graph.compile()


# 导出编译好的图实例
app = build_graph()
