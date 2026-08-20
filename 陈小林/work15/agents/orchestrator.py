"""主 Agent：热搜抓取、分发与汇总

包含三个核心节点：
1. scrape_hotspots - 抓取热搜列表
2. dispatch_items - 分发条目给子 Agent（Send API fan-out）
3. generate_summary - 汇总分析结果生成报告
"""

from __future__ import annotations

from datetime import datetime

from langchain_openai import ChatOpenAI
from langgraph.types import Send

from baidu_hotspot_agent.config import config
from baidu_hotspot_agent.tools.scraper import fetch_baidu_hotspot

# ── 全局变量：由 main.py 在启动时设置 ──
_scrape_limit: int = config.default_limit


def set_scrape_limit(limit: int) -> None:
    """设置抓取条目数量（由 main.py 调用）"""
    global _scrape_limit
    _scrape_limit = limit


# ── 节点 1：抓取热搜 ──


def scrape_hotspots(state: dict) -> dict:
    """抓取百度热搜列表（单次请求，无需 async）"""
    print(f"🔍 [主Agent] 正在抓取百度热搜 (top {_scrape_limit})...")
    try:
        items = fetch_baidu_hotspot(limit=_scrape_limit)
        print(f"✅ [主Agent] 成功抓取 {len(items)} 条热搜")
        for i, item in enumerate(items, 1):
            print(f"   {i}. {item['title']} (热度: {item['hot_score']})")
        return {"hotspot_items": items}
    except Exception as e:
        print(f"❌ [主Agent] 抓取失败: {e}")
        return {"hotspot_items": []}


# ── 节点 2：分发条目（Send API fan-out） ──


def dispatch_items(state: dict) -> list[Send]:
    """将热搜条目分发给子 Agent 并行分析

    使用 LangGraph Send API 实现动态 fan-out：
    每个热搜条目生成一个 Send 对象，框架自动并行执行。
    """
    items = state.get("hotspot_items", [])
    if not items:
        print("⚠️ [主Agent] 热搜列表为空，跳过分发")
        return []

    print(f"📤 [主Agent] 分发 {len(items)} 条热搜给子 Agent 并行分析...")
    return [
        Send("analyze_hotspot", {"hotspot_item": item})
        for item in items
    ]


# ── 节点 3：汇总生成报告 ──

SUMMARY_PROMPT = """\
你是一位资深新闻编辑。以下是当前百度热搜的分析结果，请生成一份结构化的热点摘要报告。

## 热搜分析结果
{analysis_text}

## 要求
1. 先用一段话总结当前热搜的整体趋势和主题分布（overview）
2. 然后对每个热点给出精炼的摘要

请用 JSON 格式返回：
{{
  "overview": "整体趋势概述（150-200字）",
  "highlights": [
    {{
      "title": "热搜标题",
      "summary": "精炼摘要（50-80字）"
    }}
  ]
}}

请只返回 JSON，不要包含其他文字。
"""


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
        temperature=0.5,
        max_tokens=2000,
    )


def generate_summary(state: dict) -> dict:
    """汇总所有子 Agent 的分析结果，生成 Markdown 报告（单次 LLM 调用，无需 async）"""
    results = state.get("analysis_results", [])
    items = state.get("hotspot_items", [])

    print(f"📊 [主Agent] 汇总分析结果 ({len(results)}/{len(items)} 条成功)...")

    if not results:
        return {"final_summary": "# 百度热搜摘要报告\n\n未能获取或分析任何热搜条目。"}

    # 构建分析文本
    analysis_parts = []
    for i, r in enumerate(results, 1):
        kp = "\n".join(f"  - {p}" for p in r.get("key_points", []))
        analysis_parts.append(
            f"### {i}. {r['title']}\n"
            f"背景: {r.get('background', '无')}\n"
            f"关键点:\n{kp}\n"
            f"评论: {r.get('brief_comment', '无')}"
        )
    analysis_text = "\n\n".join(analysis_parts)

    # 调用 LLM 生成概述
    try:
        llm = _get_llm()
        prompt = SUMMARY_PROMPT.format(analysis_text=analysis_text)
        response = llm.invoke(prompt)
        content = response.content

        import json

        if content.startswith("```"):
            lines = content.strip().split("\n")
            content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        summary_data = json.loads(content)
        overview = summary_data.get("overview", "")
        highlights = summary_data.get("highlights", [])
    except Exception:
        overview = ""
        highlights = []

    # 组装 Markdown 报告
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        f"# 百度热搜摘要报告",
        f"",
        f"> 生成时间: {now}  ",
        f"> 分析条目: {len(results)}/{len(items)}",
        f"",
    ]

    if overview:
        report_lines.extend(["## 整体趋势", "", overview, ""])

    if highlights:
        report_lines.append("## 热点速览")
        report_lines.append("")
        for h in highlights:
            report_lines.append(f"- **{h['title']}**: {h['summary']}")
        report_lines.append("")

    report_lines.extend(["## 详细分析", ""])
    report_lines.append(analysis_text)
    report_lines.append("")

    # 标注失败条目
    failed_count = len(items) - len(results)
    if failed_count > 0:
        report_lines.append(f"---")
        report_lines.append(f"*注: {failed_count} 条热搜分析失败*")

    final_summary = "\n".join(report_lines)
    print("✅ [主Agent] 报告生成完成")
    return {"final_summary": final_summary}
