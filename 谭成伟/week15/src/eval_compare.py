"""Parallel vs Serial 量化对比（异步版，凸显 subagent 并行优势）。"""
import sys
import time
import json
import asyncio
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EVAL_QUESTIONS = [
    "东京5天自由行规划：必去景点、特色美食、市内交通、住宿区域",
    "大阪3天自由行规划：景点、美食、交通卡、亲子注意事项",
    "成都4天自由行规划：景点、美食、川菜、大熊猫基地、市内交通、周边一日游",
    "京都2天自由行规划：寺庙景点、抹茶美食、交通、和服体验",
]


async def run_one(question: str, serial: bool) -> dict:
    import agents
    t0 = time.time()
    r = await agents.run_research(question, serial=serial)
    wall = time.time() - t0
    ps = r["parallel_stats"][-1] if r["parallel_stats"] else None
    return {
        "wall": round(wall, 2),
        "n_subagents": ps["n_subagents"] if ps else 0,
        "dispatch_wall": ps["wall_clock"] if ps else 0,
        "serial_sum": ps["serial_sum"] if ps else 0,
        "speedup": ps["speedup"] if ps else 0,
        "mode": ps["mode"] if ps else "—",
        "dispatched": len(r["dispatches"]) > 0,
    }


async def amain():
    parser = argparse.ArgumentParser(description="parallel vs serial 对比（异步）")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    qs = EVAL_QUESTIONS[:args.limit] if args.limit else EVAL_QUESTIONS

    results = []
    for i, q in enumerate(qs):
        logger.warning(f"[{i+1}/{len(qs)}] {q}")
        p = await run_one(q, serial=False)
        s = await run_one(q, serial=True)
        results.append({"question": q, "parallel": p, "serial": s})
        print(f"  {q[:22]:<24} 并行 {p['wall']}s vs 串行 {s['wall']}s "
              f"(subagent {p['n_subagents']}, 加速 {p['speedup']}×, "
              f"模式 {p['mode']})")

    avg_p = sum(r["parallel"]["wall"] for r in results) / len(results)
    avg_s = sum(r["serial"]["wall"] for r in results) / len(results)
    avg_spd = sum(r["parallel"]["speedup"] for r in results) / len(results)

    print(f"\n{'='*60}\nParallel vs Serial 对比（异步，{len(results)} 题）\n{'='*60}")
    print(f"{'指标':<16} {'并行(async gather)':<20} {'串行(逐个 await)':<20}")
    print(f"{'平均墙钟(s)':<16} {avg_p:<20.2f} {avg_s:<20.2f}")
    print(f"{'平均加速':<16} {avg_spd:<20.2f}× {'—':<20}")
    print(f"\n结论：asyncio.gather 把 N 个独立子任务的墙钟从 sum 压到 ≈max，"
          f"平均加速 {avg_spd:.2f}×。")

    out = {"summary": {"avg_parallel_s": round(avg_p, 2),
                       "avg_serial_s": round(avg_s, 2),
                       "avg_speedup": round(avg_spd, 2),
                       "concurrency_model": "asyncio"},
           "details": results}
    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "eval_compare.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(amain())
