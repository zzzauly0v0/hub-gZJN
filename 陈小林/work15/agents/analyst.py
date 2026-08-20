"""子 Agent：热搜条目内容分析

每个子 Agent 接收一个热搜条目，调用 LLM 生成结构化分析结果。
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from baidu_hotspot_agent.config import config
from baidu_hotspot_agent.state import AgentState

ANALYSIS_PROMPT = """\
你是一个专业的新闻分析师。请针对以下百度热搜条目进行深度分析。

## 热搜条目
- **标题**: {title}
- **热度**: {hot_score}
- **描述**: {desc}
- **链接**: {url}

## 要求
请以 JSON 格式返回分析结果，包含以下字段：
1. "background": 事件背景概述（100-200字）
2. "key_points": 关键信息要点（列表，3-5条）
3. "brief_comment": 简要评论和分析（100-150字）

{limitation_note}

请只返回 JSON，不要包含其他文字。
"""


def _get_llm() -> ChatOpenAI:
    """创建 LLM 实例"""
    return ChatOpenAI(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
        temperature=0.7,
        max_tokens=1000,
    )


async def analyze_hotspot(state: dict) -> dict:
    """分析单个热搜条目的节点函数（异步）

    由 LangGraph Send API 调用，每个实例处理一个热搜条目。
    使用 ainvoke 实现真正的协程并行。
    """
    item = state.get("hotspot_item", {})
    title = item.get("title", "未知")
    hot_score = item.get("hot_score", "0")
    url = item.get("url", "")
    desc = item.get("desc", "")

    # 判断是否有足够信息
    has_enough_info = bool(desc and len(desc) > 10)
    limitation_note = "" if has_enough_info else "注意：该条目描述信息有限，请主要基于标题进行合理推断和分析。"

    prompt = ANALYSIS_PROMPT.format(
        title=title,
        hot_score=hot_score,
        desc=desc or "无详细描述",
        url=url or "无链接",
        limitation_note=limitation_note,
    )

    try:
        llm = _get_llm()
        response = await llm.ainvoke(prompt)
        content = response.content

        # 解析 LLM 返回的 JSON
        import json

        # 处理可能的 markdown 代码块包裹
        if content.startswith("```"):
            lines = content.strip().split("\n")
            content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        result = json.loads(content)

        return {
            "analysis_results": [
                {
                    "title": title,
                    "background": result.get("background", "分析失败"),
                    "key_points": result.get("key_points", []),
                    "brief_comment": result.get("brief_comment", ""),
                }
            ]
        }

    except Exception as e:
        # 降级处理：返回错误信息但不中断流程
        return {
            "analysis_results": [
                {
                    "title": title,
                    "background": f"分析失败: {str(e)}",
                    "key_points": [],
                    "brief_comment": "由于 LLM 调用异常，未能生成分析结果",
                }
            ]
        }
