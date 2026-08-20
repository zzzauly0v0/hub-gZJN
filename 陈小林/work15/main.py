"""百度热搜多智能体分析系统 - 命令行入口

用法:
    python -m baidu_hotspot_agent.main              # 默认抓取 top 10
    python -m baidu_hotspot_agent.main --limit 5     # 抓取 top 5
    python -m baidu_hotspot_agent.main --output report.md  # 保存到文件
"""

from __future__ import annotations

import argparse
import asyncio

from baidu_hotspot_agent.agents.orchestrator import set_scrape_limit
from baidu_hotspot_agent.graph import app


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="百度热搜多智能体分析系统 - 自动抓取热搜并生成摘要报告",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="抓取热搜条目数量 (默认: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="报告输出文件路径 (如: report.md)",
    )
    args = parser.parse_args()

    # 设置抓取数量
    set_scrape_limit(args.limit)

    print("=" * 60)
    print("  百度热搜多智能体分析系统")
    print("=" * 60)
    print()

    # 异步执行图流程（子 Agent 通过协程并行分析）
    initial_state = {
        "hotspot_items": [],
        "analysis_results": [],
        "final_summary": "",
    }

    result = await app.ainvoke(initial_state)

    # 输出报告
    report = result.get("final_summary", "未能生成报告")

    print()
    print("=" * 60)
    print(report)
    print("=" * 60)

    # 保存到文件
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n📄 报告已保存到: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
