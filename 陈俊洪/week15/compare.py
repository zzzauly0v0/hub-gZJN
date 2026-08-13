"""并发 vs 顺序对照：量化「下发 subagent 并发」的收益

同一个热点跑两遍，唯一区别是 dispatch_subagents 内部用
asyncio.gather（并发）还是 for + await（顺序）。

  python src/compare.py            # 真实 LLM + Tavily
  MOCK=1 python src/compare.py     # 无 key 离线跑，也能看出 sum → max
"""
import sys, asyncio
from agents import analyze

QUESTIONS = [
    "2026世界杯预选赛国足热点分析：比赛结果、球员表现、舆论争议",
    "NBA交易截止日热点分析：主要交易、球队战力变化、球迷反应",
]


async def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = []
    for q in QUESTIONS:
        p = await analyze(q, serial=False)
        s = await analyze(q, serial=True)
        rows.append((q, p, s))
        print(f"{q[:20]:<22} 并发 {p['duration']}s / 顺序 {s['duration']}s "
              f"(子agent {len(p['subagents'])}, dispatch 加速 {p['stats']['speedup']}×)")

    n = len(rows)
    print(f"\n{'='*56}")
    print(f"平均总墙钟：并发 {sum(p['duration'] for _, p, _ in rows)/n:.2f}s"
          f" vs 顺序 {sum(s['duration'] for _, _, s in rows)/n:.2f}s")
    print(f"平均 dispatch 加速：{sum(p['stats']['speedup'] for _, p, _ in rows)/n:.2f}×")
    print("注：总墙钟加速小于 dispatch 加速——主 agent 自己的规划/综合两次 LLM 调用"
          "不可并发（Amdahl 定律）。")


if __name__ == "__main__":
    asyncio.run(main())
