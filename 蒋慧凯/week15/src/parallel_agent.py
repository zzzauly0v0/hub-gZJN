# -*- coding: utf-8 -*-
"""
作业：实现一个可以下发 subagent 的 agent，并行完成多项工作

核心设计：
1. 主 Agent 接收整体任务，拆分为多个子任务。
2. 通过 ThreadPoolExecutor 并行派发多个 subagent 执行。
3. subagent 完成任务后，主 Agent 聚合结果并输出综合报告。
4. 支持串行/并行两种模式对比，量化并行加速收益。

本实现先用 mock 工具函数模拟耗时调研，避免依赖外部 API。
后续可把 mock_research 替换为真实 LLM/搜索工具。
"""

import os
import time
import json
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Dict


def mock_research(topic: str) -> str:
    """模拟 subagent 执行的耗时调研任务。

    通过 topic 的哈希值决定"执行时长"和"结果内容"，保证可复现。
    """
    h = int(hashlib.md5(topic.encode("utf-8")).hexdigest(), 16)
    duration = 1 + (h % 3)  # 1~3 秒
    time.sleep(duration)
    market_size = 100 + (h % 900)
    return (
        f"关于「{topic}」的调研结果：市场规模约 {market_size} 亿元，"
        f"主要参与者包括 A、B、C，预计年增长率 {(5 + h % 15)}%。"
    )


class ParallelAgent:
    """可下发 subagent 并行完成多项工作的 Agent。"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def dispatch_subagents(
        self,
        topics: List[str],
        on_dispatch: Callable[[List[str]], None] = None,
        on_subagent_done: Callable[[str, str, str], None] = None,
        serial: bool = False,
    ) -> Dict:
        """派发多个 subagent。

        Args:
            topics: 子任务主题列表。
            on_dispatch: 派发时回调，参数为子任务列表。
            on_subagent_done: 单个 subagent 完成时回调，参数为 (sid, topic, result)。
            serial: 为 True 时串行执行，用于和并行模式做 A/B 对比。

        Returns:
            包含 results、wall_clock、serial_sum、speedup 的字典。
        """
        subagents = [
            {"sid": f"sub_{uuid.uuid4().hex[:6]}", "topic": topic}
            for topic in topics
        ]

        if on_dispatch:
            on_dispatch([s["topic"] for s in subagents])

        results = {}
        t0 = time.time()

        if serial:
            # 串行基线：一个接一个执行
            for s in subagents:
                t_start = time.time()
                res = mock_research(s["topic"])
                dur = time.time() - t_start
                results[s["sid"]] = {
                    "topic": s["topic"],
                    "result": res,
                    "duration": round(dur, 2),
                }
                if on_subagent_done:
                    on_subagent_done(s["sid"], s["topic"], res)
        else:
            # 并行：用 ThreadPoolExecutor 同时跑多个 subagent
            def run_one(s):
                t_start = time.time()
                res = mock_research(s["topic"])
                dur = time.time() - t_start
                return s, res, dur

            max_w = min(self.max_workers, len(subagents))
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                futures = {pool.submit(run_one, s): s for s in subagents}
                for future in as_completed(futures):
                    s, res, dur = future.result()
                    results[s["sid"]] = {
                        "topic": s["topic"],
                        "result": res,
                        "duration": round(dur, 2),
                    }
                    if on_subagent_done:
                        on_subagent_done(s["sid"], s["topic"], res)

        wall_clock = round(time.time() - t0, 2)
        serial_sum = round(sum(r["duration"] for r in results.values()), 2)
        speedup = round(serial_sum / wall_clock, 2) if wall_clock else 0

        return {
            "results": results,
            "wall_clock": wall_clock,
            "serial_sum": serial_sum,
            "speedup": speedup,
        }

    def run(self, task: str, topics: List[str], serial: bool = False) -> Dict:
        """主 Agent 入口：理解任务、派发 subagent、聚合结果。"""
        print(f"[主Agent] 收到任务: {task}")
        print(f"[主Agent] 拆分为 {len(topics)} 个子任务: {topics}")

        stats = self.dispatch_subagents(
            topics,
            on_dispatch=lambda ts: print(f"[主Agent] 派发 {len(ts)} 个 subagent..."),
            on_subagent_done=lambda sid, topic, res: print(
                f"[{sid}] 完成「{topic}」"
            ),
            serial=serial,
        )

        # 主 Agent 聚合各 subagent 结果
        summary_parts = [
            f"【{r['topic']}】\n{r['result']}"
            for r in stats["results"].values()
        ]
        summary = "\n\n".join(summary_parts)

        final_answer = (
            f"综合报告：{task}\n\n"
            f"{summary}\n\n"
            f"并行统计：{json.dumps({
                'wall_clock': stats['wall_clock'],
                'serial_sum': stats['serial_sum'],
                'speedup': stats['speedup'],
            }, ensure_ascii=False)}"
        )

        return {
            "task": task,
            "topics": topics,
            "final_answer": final_answer,
            "stats": stats,
        }


if __name__ == "__main__":
    agent = ParallelAgent(max_workers=4)
    task = "2024年中国新能源汽车市场调研"
    topics = ["销量规模", "主要厂商竞争格局", "政策趋势"]

    print("=" * 60)
    print("模式一：并行执行")
    r_parallel = agent.run(task, topics, serial=False)
    print(r_parallel["final_answer"])

    print("\n" + "=" * 60)
    print("模式二：串行执行（作为并行基线对比）")
    r_serial = agent.run(task, topics, serial=True)
    print(r_serial["final_answer"])

    # 保存结果到 outputs
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/parallel_agent_result.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "parallel": r_parallel["stats"],
                "serial": r_serial["stats"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("\n结果已保存至 outputs/parallel_agent_result.json")
