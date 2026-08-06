"""
把实验结果渲染成 Markdown 对比报告（本任务新增）。
"""

from __future__ import annotations

from metrics import pct_delta


def _fmt_delta(v: float, unit: str = "%") -> str:
    return f"{v:+.1f}{unit}" if v else f"0.0{unit}"


def render_markdown(report: dict) -> str:
    L: list[str] = []
    add = L.append

    add("# Skill 优化实验报告")
    add("")
    add(f"- 生成时间：{report['generated_at'][:19]}")
    add(f"- Provider：`{report['provider']}`　应答模型：`{report['agent_model']}`　"
        f"优化模型：`{report['meta_model']}`")
    add(f"- 变体选择综合分：`{report['score_formula']}`")
    add(f"- token 口径：doc_tokens = {report['token_counting']['doc_tokens']}；"
        f"prompt_tokens = {report['token_counting']['prompt_tokens']}")
    add("")

    # ── 总览 ────────────────────────────────────────────────────────────────
    add("## 一、总览：各业务域优化前后（test 集，dev 未见过的题）")
    add("")
    add("| 业务域 | 胜出策略 | 准确率 v0 → 优化后 | 文档 token | 每题 prompt token | 每 100 题成本 |")
    add("|---|---|---|---|---|---|")
    for d in report["domains"]:
        w = d["winner"]
        t0 = d["test"]["v0_baseline"]
        tw = d["test"].get(w, t0)
        doc0 = d["variants"]["v0_baseline"]["doc_tokens"]
        docw = d["variants"][w]["doc_tokens"]
        acc_txt = f"{t0['accuracy']:.0%} → {tw['accuracy']:.0%}"
        if tw["accuracy"] != t0["accuracy"]:
            acc_txt += f" ({_fmt_delta((tw['accuracy'] - t0['accuracy']) * 100, 'pt')})"
        add(f"| {d['label']} | `{w}` | {acc_txt} | "
            f"{doc0} → {docw} ({_fmt_delta(pct_delta(docw, doc0))}) | "
            f"{t0['avg_prompt_tokens']:.0f} → {tw['avg_prompt_tokens']:.0f} "
            f"({_fmt_delta(pct_delta(tw['avg_prompt_tokens'], t0['avg_prompt_tokens']))}) | "
            f"${t0['cost_usd_per_100q']:.4f} → ${tw['cost_usd_per_100q']:.4f} |")
    add("")

    # ── 各域细节 ────────────────────────────────────────────────────────────
    add("## 二、各业务域详情")
    for d in report["domains"]:
        add("")
        add(f"### {d['label']}（`{d['skill_name']}`）")
        add("")
        add(f"- dev 题号：{d['dev_ids']}　test 题号：{d['test_ids']}")
        add(f"- 给优化 Agent 的 token 预算：{d['token_budget']}")
        add(f"- dev 上的 Pareto 前沿（准确率↑ / 每题 prompt token↓）："
            f"{', '.join('`'+v+'`' for v in d['pareto_front_dev'])}")
        add("")
        add("**dev 集全变体对比**")
        add("")
        add("| 变体 | 说明 | 准确率 | 文档token | 每题prompt | 每题输出 | 综合分 | 失败原因 |")
        add("|---|---|---|---|---|---|---|---|")
        score_map = {s["variant"]: s["score"] for s in d["dev_scores"]}
        for key, r in d["dev"].items():
            mark = " ★" if key == d["winner"] else ""
            reasons = "、".join(f"{k}×{v}" for k, v in r["fail_reasons"].items()) or "—"
            add(f"| `{key}`{mark} | {d['variants'][key]['label']} | {r['accuracy']:.0%} | "
                f"{d['variants'][key]['doc_tokens']} | {r['avg_prompt_tokens']:.0f} | "
                f"{r['avg_completion_tokens']:.0f} | {score_map.get(key, 0):.4f} | {reasons} |")
        add("")
        add("**test 集复评（检验是否只是过拟合 dev）**")
        add("")
        add("| 变体 | 准确率 | 每题prompt | 每题输出 | 平均延迟(真实调用) | 失败原因 |")
        add("|---|---|---|---|---|---|")
        for key, r in d["test"].items():
            reasons = "、".join(f"{k}×{v}" for k, v in r["fail_reasons"].items()) or "—"
            add(f"| `{key}` | {r['accuracy']:.0%} | {r['avg_prompt_tokens']:.0f} | "
                f"{r['avg_completion_tokens']:.0f} | {r['avg_live_latency_s']:.2f}s | {reasons} |")
    add("")

    # ── 成本外推 ────────────────────────────────────────────────────────────
    add("## 三、成本外推")
    add("")
    add("按 test 集每题 prompt/输出 token 与当期估算价外推（`metrics.PRICING`，会随官方调价过期）：")
    add("")
    add("| 业务域 | v0 每万题成本 | 优化后每万题成本 | 节省 |")
    add("|---|---|---|---|")
    tot0 = totw = 0.0
    for d in report["domains"]:
        t0 = d["test"]["v0_baseline"]
        tw = d["test"].get(d["winner"], t0)
        c0 = t0["cost_usd_per_100q"] * 100
        cw = tw["cost_usd_per_100q"] * 100
        tot0 += c0
        totw += cw
        add(f"| {d['label']} | ${c0:.2f} | ${cw:.2f} | {_fmt_delta(pct_delta(cw, c0))} |")
    add(f"| **合计** | **${tot0:.2f}** | **${totw:.2f}** | **{_fmt_delta(pct_delta(totw, tot0))}** |")
    add("")

    u = report["llm_usage"]
    add("## 四、本次实验自身开销")
    add("")
    add(f"- Agent 应答调用：{u['agent_calls']['calls']} 次，"
        f"{u['agent_calls']['total_tokens']} tokens（缓存命中不计入）")
    add(f"- 写/优化 Skill 的元调用：{u['meta_calls']['calls']} 次，"
        f"{u['meta_calls']['total_tokens']} tokens")
    add(f"- 估算总成本：约 ${u['estimated_cost_usd']:.4f}")
    add("")
    return "\n".join(L)
