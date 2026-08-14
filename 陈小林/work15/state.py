"""状态定义与数据结构"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict


@dataclass
class HotspotItem:
    """热搜条目"""

    title: str
    hot_score: str
    url: str
    desc: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "hot_score": self.hot_score,
            "url": self.url,
            "desc": self.desc,
        }


@dataclass
class AnalysisResult:
    """子 Agent 分析结果"""

    title: str
    background: str
    key_points: list[str] = field(default_factory=list)
    brief_comment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "background": self.background,
            "key_points": self.key_points,
            "brief_comment": self.brief_comment,
        }


class AgentState(TypedDict):
    """LangGraph 图状态"""

    # 热搜条目列表（抓取结果）
    hotspot_items: list[dict[str, str]]
    # 子 Agent 分析结果列表（通过 reducer 自动合并）
    analysis_results: Annotated[list[dict[str, Any]], operator.add]
    # 最终汇总摘要
    final_summary: str
