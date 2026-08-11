"""
Skill 优化实验主程序（本任务新增，不改动 src/ 与 skills/ 下任何原有文件）。

实验问题：**让大模型自己优化一份它写出来的 Skill，能不能在不掉准确率的前提下省 token？**

流程：
  1) 作者 Agent 读 policies.md，为 3 个初始 Skill 库未覆盖的业务域各写一份 v0 Skill
     （logistics / payment_account / promotion_refund —— 原项目基线里这三类接近 0%，
       优化空间最大，且不与 skills/ 里已有的 refund / vip_benefits 冲突）
  2) v0 在 dev 集上评估 → 拿到准确率、真实 token、失败样本
  3) 三种优化策略各产出一个变体：
       opt_compress        纯压缩
       opt_structured      重构为查表式决策表
       opt_failure_guided  看 dev 失败样本 + token 预算的反思式优化
  4) 每个变体在 dev 集评估，选出综合得分最高者
  5) **胜者与 v0 一起在 test 集（dev 未见过的题）复评** —— 防止只是过拟合 dev
  6) 输出对比表、Pareto 前沿、成本外推，落盘 JSON + Markdown 报告

用法：
  cd self_evolving_agent
  python skill_opt/run_optimization.py                 # 全量（3 域 × 4 变体）
  python skill_opt/run_optimization.py --domain logistics   # 只跑一个域，最省
  python skill_opt/run_optimization.py --no-cache      # 忽略答案缓存，全部重打
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluator import Evaluator                       # noqa: E402  复用原项目判分

from authoring import OPTIMIZERS, author_skill, optimize_skill  # noqa: E402
from harness import AnswerCache, SkillHarness                   # noqa: E402
from llm import LLM                                             # noqa: E402
from metrics import count_tokens, estimate_cost_usd, pareto_front, pct_delta  # noqa: E402
from report import render_markdown                              # noqa: E402

OUT_DIR = ROOT / "outputs" / "skill_optimization"
POLICIES = ROOT / "data" / "policies.md"
EVAL_SET = ROOT / "data" / "eval_set.json"


# ── 实验域定义 ───────────────────────────────────────────────────────────────
# 选这三个域的理由：原项目 ARCHITECTURE 第五章记录它们基线 ~0%（初始 Skill 完全不覆盖），
# 所以"作者 Agent 写一份 Skill"能带来大幅提升，优化前后的差异才看得清。
# 同时它们与 skills/ 里已有的 refund / vip_benefits 不重叠，实验互不干扰。
DOMAINS = {
    "logistics": {
        "label": "物流配送与订单取消",
        "skill_name": "logistics",
        "scope_hint": "政策手册第三章（配送方式、时效、运费门槛、次日达城市与截单时间、"
                      "偏远地区、未发货订单取消的 48 小时规则与退款到账时间）",
    },
    "payment_account": {
        "label": "支付、账户余额与积分退款",
        "skill_name": "payment_account",
        "scope_hint": "政策手册第五章与 1.5 节（余额充值/提现/转赠上限、各支付方式退款"
                      "到账时间、积分抵扣退款的 1 元=80 积分换算、账户冻结的影响与解冻流程）",
    },
    "promotion_refund": {
        "label": "促销商品与优惠券退款",
        "skill_name": "promotion_refund",
        "scope_hint": "政策手册 1.3 节与第四章（限时特惠商品的退货规则及 VIP 例外的 7 天期限、"
                      "满减门槛与部分退货补差价、新人券不补发、限时特惠不可叠加优惠券）",
    },
}

DIFFICULTY_ORDER = ("hard", "medium", "easy")


def split_dev_test(evaluator: Evaluator, category: str) -> tuple[list[int], list[int]]:
    """
    按难度分层，把某类别的题交替分入 dev / test。

    为什么要分层而不是简单对半切：eval_set 里 easy/medium/hard 混杂，
    随手切很容易让 dev 全是简单题——v0 直接 100%，就没有失败样本可供
    opt_failure_guided 反思，也看不出优化到底有没有修好难题。
    交替分配（hard 先分给 dev）保证两半难度分布接近，且 dev 一定含难题。

    计数器 k 跨难度组累加而不是每组归零，否则每组都从 dev 开始分，
    奇数长度的组累积起来会让 dev 明显多于 test（实测 7:5）。
    确定性：只依赖题号排序，不用随机数，实验可复现。
    """
    dev: list[int] = []
    test: list[int] = []
    k = 0
    for diff in DIFFICULTY_ORDER:
        ids = sorted(
            qid for qid, q in evaluator.questions.items()
            if q["category"] == category and q["difficulty"] == diff
        )
        for qid in ids:
            (dev if k % 2 == 0 else test).append(qid)
            k += 1
    return sorted(dev), sorted(test)


def score(acc: float, doc_tokens: int, base_doc_tokens: int) -> float:
    """
    变体选择用的综合分：准确率为主，token 省下来算加分。

    权重 0.15 的含义：把文档压到 0 token 最多加 0.15 分，
    约等于"愿意用 1.5 个百分点的准确率换一半 token"。准确率始终占主导。
    """
    saved_ratio = 1 - doc_tokens / base_doc_tokens if base_doc_tokens else 0
    return round(acc + 0.15 * saved_ratio, 4)


def run_domain(
    domain_key: str,
    cfg: dict,
    author_llm: LLM,
    opt_llm: LLM,
    harness: SkillHarness,
    optimizers: list[dict],
) -> dict:
    dev_ids, test_ids = split_dev_test(harness.evaluator, domain_key)
    print("\n" + "=" * 74)
    print(f"  业务域：{cfg['label']}  (skill={cfg['skill_name']})")
    print(f"  dev 题号 {dev_ids}   test 题号 {test_ids}")
    print("=" * 74)

    policies = POLICIES.read_text(encoding="utf-8")
    sample_qs = [harness.evaluator.questions[i]["question"] for i in dev_ids]

    # ── 步骤 1：作者 Agent 写 v0 ──────────────────────────────────────────
    print("\n[1/5] 作者 Agent 撰写 v0 Skill …")
    v0 = author_skill(
        author_llm, policies, cfg["skill_name"], cfg["label"],
        cfg["scope_hint"], sample_qs,
    )
    v0_tokens = count_tokens(v0)
    print(f"      ✓ v0 写成，{v0_tokens} tokens（{len(v0)} 字符）")

    variants: dict[str, dict] = {
        "v0_baseline": {"label": "v0 基线（大模型初次撰写）", "content": v0}
    }

    # ── 步骤 2：v0 在 dev 集评估 ──────────────────────────────────────────
    print("\n[2/5] v0 在 dev 集评估 …")
    dev_results = {
        "v0_baseline": harness.evaluate(
            {cfg["skill_name"]: v0}, dev_ids, "v0_baseline", variants["v0_baseline"]["label"], "dev"
        )
    }
    v0_dev = dev_results["v0_baseline"]
    print(f"      失败原因分布：{v0_dev.fail_reasons or '（全对）'}")

    # ── 步骤 3：三种策略各产出一个变体 ────────────────────────────────────
    print(f"\n[3/5] {len(optimizers)} 种优化策略产出变体 …")
    budget = max(120, int(v0_tokens * 0.6))
    for opt in optimizers:
        print(f"      · {opt['key']}: {opt['label']}", end="", flush=True)
        content = optimize_skill(
            opt_llm, opt, v0, cfg["skill_name"], version=2,
            cur_tokens=v0_tokens, budget=budget,
            failures=v0_dev.failures(), passes=v0_dev.passes(),
        )
        t = count_tokens(content)
        variants[opt["key"]] = {"label": opt["label"], "content": content}
        print(f" → {t} tokens ({pct_delta(t, v0_tokens):+.1f}%)")

    # ── 步骤 4：变体在 dev 集评估并选优 ───────────────────────────────────
    print("\n[4/5] 各变体在 dev 集评估 …")
    for key, v in variants.items():
        if key in dev_results:
            continue
        dev_results[key] = harness.evaluate(
            {cfg["skill_name"]: v["content"]}, dev_ids, key, v["label"], "dev"
        )

    scored = []
    for key, r in dev_results.items():
        scored.append({
            "variant": key,
            "score": score(r.accuracy, r.doc_tokens, v0_dev.doc_tokens),
            "accuracy": r.accuracy,
            "doc_tokens": r.doc_tokens,
        })
    scored.sort(key=lambda x: (-x["score"], x["doc_tokens"]))
    winner = scored[0]["variant"]
    print("\n      dev 综合分排名（准确率 + 0.15×token节省率）:")
    for s in scored:
        mark = "★" if s["variant"] == winner else " "
        print(f"      {mark} {s['variant']:<20} score={s['score']:.4f} "
              f"acc={s['accuracy']:.1%} doc={s['doc_tokens']}tok")
    if winner == "v0_baseline":
        print("      ⚠ dev 上没有变体优于 v0（本域优化未取得净收益）")

    # ── 步骤 5：test 集复评（防过拟合） ───────────────────────────────────
    print(f"\n[5/5] test 集复评（v0 vs 胜者 {winner}）…")
    test_targets = ["v0_baseline"] if winner == "v0_baseline" else ["v0_baseline", winner]
    test_results = {
        key: harness.evaluate(
            {cfg["skill_name"]: variants[key]["content"]}, test_ids,
            key, variants[key]["label"], "test",
        )
        for key in test_targets
    }

    # 落盘每个变体的 Skill 文档，方便人工 diff
    skills_dir = OUT_DIR / "variants" / domain_key
    skills_dir.mkdir(parents=True, exist_ok=True)
    for key, v in variants.items():
        (skills_dir / f"{key}.md").write_text(v["content"], encoding="utf-8")

    model = harness.llm.model
    return {
        "domain": domain_key,
        "label": cfg["label"],
        "skill_name": cfg["skill_name"],
        "dev_ids": dev_ids,
        "test_ids": test_ids,
        "token_budget": budget,
        "winner": winner,
        "dev_scores": scored,
        "variants": {
            k: {
                "label": v["label"],
                "doc_tokens": count_tokens(v["content"]),
                "chars": len(v["content"]),
                "file": f"variants/{domain_key}/{k}.md",
            }
            for k, v in variants.items()
        },
        "dev": {k: r.to_dict(model) for k, r in dev_results.items()},
        "test": {k: r.to_dict(model) for k, r in test_results.items()},
        "dev_detail": {k: r.per_question for k, r in dev_results.items()},
        "test_detail": {k: r.per_question for k, r in test_results.items()},
        "pareto_front_dev": pareto_front(
            [
                {"variant": k, "acc": r.accuracy, "tok": r.avg_prompt_tokens}
                for k, r in dev_results.items()
            ],
            gain_key="acc", cost_key="tok",
        ),
    }


def main():
    ap = argparse.ArgumentParser(description="Skill 优化实验（准确率 / token 双目标）")
    ap.add_argument("--domain", action="append", choices=list(DOMAINS),
                    help="只跑指定业务域（可重复传）；默认全部 3 个")
    ap.add_argument("--optimizer", action="append", choices=[o["key"] for o in OPTIMIZERS],
                    help="只跑指定优化策略；默认全部 3 种")
    ap.add_argument("--provider", default=None, help="deepseek / gemini / openai，默认按环境变量自动选")
    ap.add_argument("--model", default=None, help="覆盖 Agent 应答用的模型")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--no-cache", action="store_true", help="不使用答案缓存，全部重新调用")
    args = ap.parse_args()

    domains = args.domain or list(DOMAINS)
    optimizers = (
        [o for o in OPTIMIZERS if o["key"] in args.optimizer] if args.optimizer else OPTIMIZERS
    )

    # Agent 应答：think=False，token 数才只反映 Skill 本身
    agent_llm = LLM(provider=args.provider, model=args.model, think=False)
    # 写/优化 Skill 是元调用，只有几次，允许思考换质量
    meta_llm = LLM(provider=args.provider, model=args.model, think=True)

    print("=" * 74)
    print("  Skill 优化实验：让大模型优化自己写的 Skill（准确率 / token 双目标）")
    print("=" * 74)
    print(f"  provider={agent_llm.provider}  model={agent_llm.model}")
    print(f"  业务域={domains}")
    print(f"  优化策略={[o['key'] for o in optimizers]}")
    print(f"  答案缓存={'关闭' if args.no_cache else '开启'}  并发={args.concurrency}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(str(EVAL_SET))
    cache = AnswerCache(OUT_DIR / "answer_cache.json", enabled=not args.no_cache)
    harness = SkillHarness(agent_llm, evaluator, cache=cache, concurrency=args.concurrency)

    domain_results = [
        run_domain(k, DOMAINS[k], meta_llm, meta_llm, harness, optimizers)
        for k in domains
    ]

    report = {
        "generated_at": datetime.now().isoformat(),
        "provider": agent_llm.provider,
        "agent_model": agent_llm.model,
        "meta_model": meta_llm.model,
        "token_counting": {
            "doc_tokens": "tiktoken cl100k_base 离线计数，跨变体可比",
            "prompt_tokens": "API 真实 usage，含系统提示模板与问题",
        },
        "score_formula": "accuracy + 0.15 * (1 - doc_tokens / v0_doc_tokens)",
        "domains": domain_results,
        "llm_usage": {
            "agent_calls": agent_llm.meter.summary(),
            "meta_calls": meta_llm.meter.summary(),
            "estimated_cost_usd": round(
                estimate_cost_usd(agent_llm.model,
                                  agent_llm.meter.prompt_tokens + meta_llm.meter.prompt_tokens,
                                  agent_llm.meter.completion_tokens + meta_llm.meter.completion_tokens),
                4,
            ),
        },
    }

    json_path = OUT_DIR / "optimization_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = render_markdown(report)
    md_path = OUT_DIR / "OPTIMIZATION_REPORT.md"
    md_path.write_text(md, encoding="utf-8")

    print("\n" + "=" * 74)
    print("  实验完成")
    print("=" * 74)
    print(md)
    print(f"\n✓ JSON  : {json_path}")
    print(f"✓ 报告  : {md_path}")
    print(f"✓ 变体  : {OUT_DIR / 'variants'}")


if __name__ == "__main__":
    main()
