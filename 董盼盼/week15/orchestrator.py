"""
主 Agent（Orchestrator）：旅游规划总调度

职责：
  1. 接收用户约束（目的地/日期/人数/预算/偏好），归一化为标准约束 dict
  2. 并行派发 5 个子 Agent（景点/住宿/交通/美食/风险）
  3. 收集结果并校验预算：超出时让 住宿/交通 Agent 按更紧预算二次协商
  4. 整合所有信息，生成最终行程（含每日安排 + 费用明细 + 风险提示）

设计要点：
  - ThreadPoolExecutor 并行调用 5 个子 Agent，降低端到端时延
  - 预算校验是主 Agent 的核心职责，子 Agent 不感知总预算
  - 最终行程由一次 LLM 整合生成（带结构化输入），失败时走模板兜底
"""

import os
import sys
import math
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Windows OpenMP 冲突修复
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client import llm_chat
from sub_agents import (
    research_attractions,
    filter_accommodation,
    plan_transport,
    recommend_food,
    assess_risks,
)

logger = logging.getLogger(__name__)

# 预算分项默认占比（用户只给总额时使用）
BUDGET_RATIO = {
    "accommodation": 0.35,
    "transport":     0.25,
    "food":          0.20,
    "tickets":       0.15,
    "other":         0.05,
}


# ── 约束解析 ─────────────────────────────────────────────────────────────────
def parse_constraints(raw: dict) -> dict:
    """把用户输入归一化为标准约束 dict，并推导天数、分项预算"""
    from datetime import date
    start = raw["start_date"]
    end = raw["end_date"]
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)
    days = max((end - start).days, 1)

    total = float(raw["budget_total"])
    c = {
        "destination":   raw["destination"],
        "origin_city":   raw.get("origin_city", "未指定"),
        "start_date":    start.isoformat(),
        "end_date":      end.isoformat(),
        "days":          days,
        "travelers":     int(raw.get("travelers", 1)),
        "budget_total":  total,
        "budget_accommodation": float(raw.get("budget_accommodation", total * BUDGET_RATIO["accommodation"])),
        "budget_transport":     float(raw.get("budget_transport",     total * BUDGET_RATIO["transport"])),
        "budget_food":          float(raw.get("budget_food",          total * BUDGET_RATIO["food"])),
        "budget_tickets":       float(raw.get("budget_tickets",       total * BUDGET_RATIO["tickets"])),
        "budget_other":         float(raw.get("budget_other",         total * BUDGET_RATIO["other"])),
        "preferences":   raw.get("preferences", []),
        "special_needs": raw.get("special_needs", "无"),
    }
    return c


# ── 预算计算 ─────────────────────────────────────────────────────────────────
def _rooms(travelers: int, capacity: int = 2) -> int:
    return max(1, math.ceil(travelers / max(capacity, 1)))


def compute_cost(results: dict, c: dict) -> dict:
    """根据各子 Agent 返回的结构化数据，计算分项与总费用"""
    # 住宿
    rec = results["accommodation"].get("recommended", {})
    cap = rec.get("room_capacity", 2) or 2
    accom_cost = rec.get("price_per_night", 0) * c["days"] * _rooms(c["travelers"], cap)

    # 交通：跨城按人 × 往返段 + 城内按天
    inter = sum(s.get("price_per_person", 0) for s in results["transport"].get("intercity", []))
    intra_daily = sum(s.get("cost_per_day_total", 0) for s in results["transport"].get("intracity", []))
    trans_cost = inter * c["travelers"] + intra_daily * c["days"]

    # 美食：按人按天
    food_per_day = results["food"].get("estimated_cost_per_person_per_day", 0)
    food_cost = food_per_day * c["travelers"] * c["days"]

    # 门票：按所有景点单人票 × 人数（保守上界，实际按行程选中的算）
    ticket_cost = sum(a.get("ticket_price", 0) for a in results["attractions"].get("attractions", [])) * c["travelers"]

    total = accom_cost + trans_cost + food_cost + ticket_cost
    return {
        "accommodation": accom_cost,
        "transport":     trans_cost,
        "food":          food_cost,
        "tickets":       ticket_cost,
        "other":         0,
        "total":         total,
    }


# ── 并行派发 ─────────────────────────────────────────────────────────────────
def _dispatch_all(c: dict) -> dict:
    """并行调用 5 个子 Agent"""
    tasks = {
        "attractions":    research_attractions,
        "accommodation":  filter_accommodation,
        "transport":      plan_transport,
        "food":           recommend_food,
        "risk":           assess_risks,
    }
    results = {}
     # 创建线程池，最多同时跑5个线程
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fn, c): key for key, fn in tasks.items()}
        # as_completed：**哪个线程先执行完，就先处理哪个结果**，不保证顺序
        for fut in as_completed(futures):
            key = futures[fut] # 通过future拿回对应的任务名称
            try:
                 # fut.result() 获取线程函数返回值；如果线程内部抛异常，这里会重新抛出
                results[key] = fut.result()
            except Exception as e:
                logger.error(f"子 Agent {key} 异常: {e}")
                results[key] = {"_error": str(e)}
    return results


# ── 最终行程生成 ─────────────────────────────────────────────────────────────
def _build_itinerary_llm(c: dict, results: dict, cost: dict) -> str:
    """用一次 LLM 调用把所有结构化结果整合为每日行程 Markdown"""
    sys = (
        "你是旅游行程整合专家。根据提供的景点/住宿/交通/美食/风险数据，"
        "生成一份完整的行程方案，包含：\n"
        "1. 行程概览（目的地/日期/人数/总预算）\n"
        "2. 每日安排（按天数分配景点+餐饮+住宿，时间合理）\n"
        "3. 交通方案（跨城往返 + 城内出行）\n"
        "4. 费用明细表（住宿/交通/美食/门票/合计，对照预算）\n"
        "5. 风险提示（天气/避坑/注意事项）\n"
        "使用 Markdown 格式，表格呈现费用，条理清晰。"
    )
    user = (
        f"【用户约束】\n{json.dumps(c, ensure_ascii=False, indent=2)}\n\n"
        f"【子 Agent 结果】\n{json.dumps(results, ensure_ascii=False, indent=2)}\n\n"
        f"【费用计算】\n{json.dumps(cost, ensure_ascii=False, indent=2)}\n\n"
        f"请生成最终行程。"
    )
    text = llm_chat(sys, user, enable_search=False, max_tokens=2500)
    return text


def _build_itinerary_template(c: dict, results: dict, cost: dict, over_budget: bool) -> str:
    """LLM 不可用时的模板兜底"""
    lines = []
    lines.append(f"# {c['destination']} {c['days']}日行程方案\n")
    lines.append(f"**日期**：{c['start_date']} ~ {c['end_date']}　**人数**：{c['travelers']}　"
                 f"**总预算**：{c['budget_total']:.0f} 元　**预估花费**：{cost['total']:.0f} 元\n")
    if over_budget:
        lines.append(f"> ⚠️ 预算超标：预估超出 {cost['total'] - c['budget_total']:.0f} 元，已尝试收紧住宿/交通。\n")

    # 每日安排（按天数轮转分配景点）
    lines.append("## 每日安排\n")
    attrs = results["attractions"].get("attractions", [])
    rests = results["food"].get("restaurants", [])
    for d in range(1, c["days"] + 1):
        lines.append(f"### Day {d}")
        a = attrs[(d - 1) % len(attrs)] if attrs else None
        if a:
            lines.append(f"- 上午：{a['name']}（{a.get('type','')}，约{a.get('duration_hours',2)}h，门票 {a.get('ticket_price',0)} 元）")
        a2 = attrs[(d) % len(attrs)] if attrs else None
        if a2:
            lines.append(f"- 下午：{a2['name']}（{a2.get('type','')}，约{a2.get('duration_hours',2)}h）")
        r = rests[(d - 1) % len(rests)] if rests else None
        if r:
            lines.append(f"- 午餐/晚餐：{r['name']}（{r.get('cuisine','')}，人均 {r.get('avg_price_per_person_per_meal',0)} 元，推荐：{', '.join(r.get('must_try',[]))})")
        lines.append("")

    # 住宿
    rec = results["accommodation"].get("recommended", {})
    lines.append("## 住宿\n")
    lines.append(f"- 推荐：{rec.get('name','')}（{rec.get('price_per_night',0):.0f} 元/晚，{rec.get('location','')}）")
    for alt in results["accommodation"].get("alternatives", []):
        lines.append(f"- 备选：{alt.get('name','')}（{alt.get('price_per_night',0):.0f} 元/晚，{alt.get('location','')}）")

    # 交通
    lines.append("\n## 交通\n")
    for s in results["transport"].get("intercity", []):
        lines.append(f"- 跨城 {s.get('segment','')}：{s.get('mode','')}，{s.get('price_per_person',0)} 元/人，约 {s.get('duration_hours',0)}h")
    for s in results["transport"].get("intracity", []):
        lines.append(f"- 城内：{s.get('type','')}，约 {s.get('cost_per_day_total',0)} 元/天")

    # 费用明细
    lines.append("\n## 费用明细\n")
    lines.append("| 分项 | 预算 | 预估 | 差额 |")
    lines.append("|------|------|------|------|")
    for key, label in [("accommodation","住宿"),("transport","交通"),("food","美食"),("tickets","门票"),("other","其他")]:
        budget = c.get(f"budget_{key}", 0)
        actual = cost.get(key, 0)
        lines.append(f"| {label} | {budget:.0f} | {actual:.0f} | {actual-budget:+.0f} |")
    lines.append(f"| **合计** | **{c['budget_total']:.0f}** | **{cost['total']:.0f}** | **{cost['total']-c['budget_total']:+.0f}** |")

    # 风险
    risk = results.get("risk", {})
    lines.append("\n## 风险提示\n")
    lines.append(f"- 天气：{risk.get('weather','')}")
    for w in risk.get("warnings", []):
        lines.append(f"- ⚠️ {w}")
    for t in risk.get("tips", []):
        lines.append(f"- 💡 {t}")

    return "\n".join(lines)


# ── 主流程 ───────────────────────────────────────────────────────────────────
def plan_trip(c: dict, verbose: bool = True) -> str:
    """
    主 Agent 入口：派发 → 校验预算 → 整合 → 输出最终行程

    返回 Markdown 字符串。
    """
    if verbose:
        print(f"\n[主 Agent] 接收约束：{c['destination']} {c['days']}天 {c['travelers']}人 预算{c['budget_total']:.0f}元")
        print("[主 Agent] 并行派发 5 个子 Agent ...")

    # Round 1：并行派发
    results = _dispatch_all(c)
    if verbose:
        for k, v in results.items():
            tag = " (Mock)" if isinstance(v, dict) and v.get("_mock") else (" (Error)" if isinstance(v, dict) and v.get("_error") else "")
            print(f"  - {k} 完成{tag}")

    # 预算校验
    cost = compute_cost(results, c)
    over_budget = cost["total"] > c["budget_total"] * 1.05  # 5% 容差

    if over_budget and verbose:
        print(f"[主 Agent] 预算超标：预估 {cost['total']:.0f} > 预算 {c['budget_total']:.0f}，启动二次协商 ...")

    # Round 2：若超标，收紧住宿/交通重新协商
    if over_budget:
        rooms = _rooms(c["travelers"])
        max_nightly = (c["budget_accommodation"] * 0.75) / max(c["days"] * rooms, 1)
        max_trans_total = c["budget_transport"] * 0.85
        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(filter_accommodation, c, max_nightly)
            ft = pool.submit(plan_transport, c, max_trans_total)
            results["accommodation"] = fa.result()
            results["transport"] = ft.result()
        cost = compute_cost(results, c)
        over_budget = cost["total"] > c["budget_total"] * 1.05
        if verbose:
            print(f"[主 Agent] 二次协商后预估 {cost['total']:.0f} 元，{'仍超标' if over_budget else '已落入预算'}")

    # 整合输出
    if verbose:
        print("[主 Agent] 整合最终行程 ...")
    itinerary = _build_itinerary_llm(c, results, cost)
    if not itinerary:
        itinerary = _build_itinerary_template(c, results, cost, over_budget)

    return itinerary
