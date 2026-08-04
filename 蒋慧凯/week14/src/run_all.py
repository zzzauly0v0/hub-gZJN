"""
Week 14 作业主流程：
  1. 生成初始数字商品退款 Skill
  2. 评估初始 Skill（准确率 + token 消耗）
  3. 优化 Skill（减少 token 消耗）
  4. 评估优化后 Skill
  5. 生成对比报告

输出：
  - outputs/skill_v1.md
  - outputs/skill_v2.md
  - outputs/eval_v1.json
  - outputs/eval_v2.json
  - outputs/comparison_report.md
  - outputs/logs/run_log.json
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate_skill import generate_initial_skill
from optimize_skill import optimize_skill
from evaluate_skill import evaluate_skill


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_report(v1: dict, v2: dict) -> str:
    """生成 Markdown 对比报告。"""
    token_change = v2["skill_tokens"] - v1["skill_tokens"]
    token_change_pct = token_change / v1["skill_tokens"] * 100 if v1["skill_tokens"] else 0
    time_change = v2["avg_response_time"] - v1["avg_response_time"]
    acc_change = v2["accuracy"] - v1["accuracy"]
    acc_change_pct = acc_change * 100

    lines = [
        "# Skill 优化前后对比报告",
        "",
        f"生成时间：{datetime.now().isoformat()}",
        "",
        "## 核心指标对比",
        "",
        "| 指标 | 优化前 (v1) | 优化后 (v2) | 变化 |",
        "|------|------------|------------|------|",
        f"| Skill token 数 | {v1['skill_tokens']} | {v2['skill_tokens']} | {token_change:+d} ({token_change_pct:+.1f}%) |",
        f"| Skill 字符数 | {v1['skill_chars']} | {v2['skill_chars']} | {v2['skill_chars'] - v1['skill_chars']:+d} |",
        f"| 测试集准确率 | {v1['accuracy']:.1%} | {v2['accuracy']:.1%} | {acc_change:+.1%} |",
        f"| 平均响应时间 | {v1['avg_response_time']:.3f}s | {v2['avg_response_time']:.3f}s | {time_change:+.3f}s |",
        f"| 总响应时间 | {v1['total_response_time']:.3f}s | {v2['total_response_time']:.3f}s | {v2['total_response_time'] - v1['total_response_time']:+.3f}s |",
        "",
        "## 逐题结果",
        "",
        "| 题号 | 问题 | v1 答案 | v1 对错 | v2 答案 | v2 对错 |",
        "|------|------|---------|--------|---------|--------|",
    ]

    for r1, r2 in zip(v1["results"], v2["results"]):
        q = r1["question"]
        mark1 = "[OK]" if r1["correct"] else "[FAIL]"
        mark2 = "[OK]" if r2["correct"] else "[FAIL]"
        ans1 = r1["answer"].replace("\n", " ")[:40]
        ans2 = r2["answer"].replace("\n", " ")[:40]
        lines.append(f"| {r1['id']} | {q[:30]}... | {ans1} | {mark1} | {ans2} | {mark2} |")

    lines.extend([
        "",
        "## 结论",
        "",
    ])

    if token_change < 0 and acc_change >= 0:
        lines.append(f"优化成功：token 数减少 {abs(token_change)} ({abs(token_change_pct):.1f}%)，准确率保持或提升 {acc_change_pct:+.1f}个百分点。")
    elif token_change < 0:
        lines.append(f"token 数减少 {abs(token_change)} ({abs(token_change_pct):.1f}%)，但准确率下降 {abs(acc_change_pct):.1f}个百分点，需要权衡。")
    else:
        lines.append(f"token 数未减少，优化效果不明显。")

    lines.extend([
        "",
        "## 原始文件",
        "",
        "- `outputs/skill_v1.md`：优化前 Skill",
        "- `outputs/skill_v2.md`：优化后 Skill",
        "- `outputs/eval_v1.json`：优化前评估详情",
        "- `outputs/eval_v2.json`：优化后评估详情",
    ])

    return "\n".join(lines)


def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：请先设置 DEEPSEEK_API_KEY 环境变量")
        sys.exit(1)

    print("=" * 60)
    print(" Week 14 作业：Skill 生成与优化对比")
    print("=" * 60)

    # 1. 生成初始 Skill
    print("\n[1/5] 生成初始 Skill...")
    v1_path = ROOT / "outputs" / "skill_v1.md"
    skill_v1 = generate_initial_skill(api_key)
    v1_path.write_text(skill_v1, encoding="utf-8")
    print(f"  [OK] 已保存: {v1_path}")

    # 2. 评估初始 Skill
    print("\n[2/5] 评估初始 Skill...")
    test_path = ROOT / "data" / "test_cases.json"
    eval_v1 = evaluate_skill(str(v1_path), str(test_path), api_key)
    save_json(eval_v1, ROOT / "outputs" / "eval_v1.json")
    print(f"  [OK] 准确率: {eval_v1['accuracy']:.1%}, token 数: {eval_v1['skill_tokens']}")

    # 3. 优化 Skill
    print("\n[3/5] 优化 Skill（减少 token 消耗）...")
    v2_path = ROOT / "outputs" / "skill_v2.md"
    skill_v2 = optimize_skill(skill_v1, api_key)
    v2_path.write_text(skill_v2, encoding="utf-8")
    print(f"  [OK] 已保存: {v2_path}")

    # 4. 评估优化后 Skill
    print("\n[4/5] 评估优化后 Skill...")
    eval_v2 = evaluate_skill(str(v2_path), str(test_path), api_key)
    save_json(eval_v2, ROOT / "outputs" / "eval_v2.json")
    print(f"  [OK] 准确率: {eval_v2['accuracy']:.1%}, token 数: {eval_v2['skill_tokens']}")

    # 5. 生成对比报告
    print("\n[5/5] 生成对比报告...")
    report = build_report(eval_v1, eval_v2)
    report_path = ROOT / "outputs" / "comparison_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  [OK] 已保存: {report_path}")

    # 运行日志
    run_log = {
        "timestamp": datetime.now().isoformat(),
        "skill_v1_tokens": eval_v1["skill_tokens"],
        "skill_v2_tokens": eval_v2["skill_tokens"],
        "v1_accuracy": eval_v1["accuracy"],
        "v2_accuracy": eval_v2["accuracy"],
        "v1_avg_time": eval_v1["avg_response_time"],
        "v2_avg_time": eval_v2["avg_response_time"],
    }
    save_json(run_log, ROOT / "outputs" / "logs" / "run_log.json")

    print("\n" + "=" * 60)
    print(" 完成")
    print("=" * 60)
    print(f"  初始 Skill token 数: {eval_v1['skill_tokens']}")
    print(f"  优化后 Skill token 数: {eval_v2['skill_tokens']}")
    print(f"  初始准确率: {eval_v1['accuracy']:.1%}")
    print(f"  优化后准确率: {eval_v2['accuracy']:.1%}")
    print(f"  报告: {report_path}")


if __name__ == "__main__":
    main()
