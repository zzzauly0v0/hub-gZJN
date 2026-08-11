"""
知识库问答 Skill 进化系统（SKILL.md 自更新版）

进化逻辑：
1. 大模型仅根据 SKILL.md 回答问题
2. 测试发现回答出错时，从 knowledge_base.md 找到正确知识
3. 将正确知识写入 SKILL.md，使下次回答正确
4. 重复直到所有测试通过，或 SKILL.md 已无法继续更新

支持 DeepSeek 真实模型和模拟模式
"""

import json
import os
from datetime import datetime
from typing import List, Dict

from qa_skill import ask_question, set_llm_quality, is_using_real_model, load_skill_md, get_skill_qa_pairs
from evaluator import run_tests, print_report
from skill_updater import update_from_failures, find_relevant_entries
from knowledge_loader import load_knowledge_base


def analyze_failures(report: Dict) -> Dict:
    """
    收集失败用例，分析失败原因，生成失败分析报告。

    Args:
        report: 测试报告字典

    Returns:
        失败分析报告
    """
    failures = []
    for detail in report.get("failed_details", []):
        failure = {
            "id": detail["id"],
            "question": detail["question"],
            "llm_answer": detail["llm_answer"],
            "expected_answer": detail["expected_answer"],
            "missed_keywords": detail["missed_keywords"],
            "match_ratio": detail["match_ratio"],
            "reason": _analyze_reason(detail),
        }
        failures.append(failure)

    return {
        "total_failures": len(failures),
        "failures": failures,
        "common_missed_keywords": _find_common_missed(failures),
    }


def _analyze_reason(detail: Dict) -> str:
    """分析单个失败用例的原因"""
    if "抱歉" in detail["llm_answer"] or "没有相关信息" in detail["llm_answer"]:
        return "SKILL.md 中缺少该知识点，回答为空"

    missed = detail["missed_keywords"]
    if len(missed) > len(detail.get("hit_keywords", [])):
        return f"SKILL.md 中该知识点不完整，未命中 {len(missed)} 个关键词"

    return f"SKILL.md 中有部分内容但遗漏了关键词: {', '.join(missed)}"


def _find_common_missed(failures: List[Dict]) -> List[str]:
    """找出所有失败用例中共同缺失的关键词"""
    all_missed = {}
    for f in failures:
        for kw in f["missed_keywords"]:
            all_missed[kw] = all_missed.get(kw, 0) + 1
    return sorted(all_missed.keys(), key=lambda k: all_missed[k], reverse=True)


def evolve(
    max_iterations: int = 5,
    threshold: float = 0.6,
    stagnant_limit: int = 3,
    skill_file_path: str = "SKILL.md",
    knowledge_base_path: str = "knowledge_base.md",
) -> Dict:
    """
    Skill 进化主循环（SKILL.md 自更新版）。

    每轮迭代：
    1. 运行测试（基于当前 SKILL.md）
    2. 分析失败用例
    3. 从 knowledge_base.md 找到正确知识，更新 SKILL.md
    4. 重新测试

    终止条件：
    - 通过率达到 100%
    - 本轮无任何 SKILL.md 更新（说明知识库中也没有缺失的知识）
    - 连续 stagnant_limit 次迭代通过率无提升

    Args:
        max_iterations: 最大迭代次数
        threshold: 通过阈值
        stagnant_limit: 连续无提升次数限制
        skill_file_path: SKILL.md 路径
        knowledge_base_path: knowledge_base.md 路径

    Returns:
        进化结果摘要
    """
    use_real = is_using_real_model()

    print("\n" + "=" * 60)
    print("开始 Skill 进化流程（SKILL.md 自更新模式）")
    if use_real:
        print("模式：DeepSeek 真实模型")
    else:
        print("模式：模拟 LLM")
    print("=" * 60)

    evolution_log = []
    prev_pass_rate = -1.0
    stagnant_count = 0

    # 模拟模式下的质量参数
    if not use_real:
        current_quality = 0.55
        quality_step = 0.15
        set_llm_quality(current_quality)

    for iteration in range(1, max_iterations + 1):
        quality_info = f" (LLM质量: {current_quality:.0%})" if not use_real else ""
        skill_qa_count = len(get_skill_qa_pairs())
        print(f"\n{'=' * 60}")
        print(f"第 {iteration} 轮进化{quality_info}")
        print(f"SKILL.md 当前知识条目数: {skill_qa_count}")
        print(f"{'=' * 60}")

        # 1. 运行测试（基于当前 SKILL.md）
        report = run_tests(threshold=threshold, skill_file_path=skill_file_path)
        print_report(report)

        current_pass_rate = report["pass_rate"]

        # 2. 记录迭代信息
        iteration_record = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "pass_rate": current_pass_rate,
            "passed": report["passed"],
            "failed": report["failed"],
            "total": report["total"],
            "skill_qa_count": skill_qa_count,
            "mode": "deepseek" if use_real else "simulate",
        }

        # 3. 检查是否全部通过
        if current_pass_rate >= 1.0:
            print(f"\n所有测试通过！进化完成！")
            iteration_record["status"] = "全部通过"
            iteration_record["action"] = "无需更新"
            evolution_log.append(iteration_record)
            break

        # 4. 分析失败用例
        analysis = analyze_failures(report)
        iteration_record["analysis"] = {
            "total_failures": analysis["total_failures"],
            "common_missed_keywords": analysis["common_missed_keywords"][:5],
        }

        # 5. 从知识库找到正确知识，更新 SKILL.md
        print(f"\n正在从 knowledge_base.md 查找缺失知识并更新 SKILL.md...")
        update_result = update_from_failures(
            analysis["failures"],
            skill_file_path=skill_file_path,
            knowledge_base_path=knowledge_base_path,
        )

        iteration_record["update_result"] = {
            "updated_count": update_result["updated_count"],
            "details": [
                {"question": d["question"][:40], "action": d.get("action", "no_match")}
                for d in update_result["details"]
            ],
        }

        if update_result["updated_count"] == 0:
            print(f"SKILL.md 无法继续更新（知识库中没有更多相关知识），进化停止。")
            iteration_record["status"] = "知识库无更多内容"
            iteration_record["action"] = "无更新"
            evolution_log.append(iteration_record)
            break

        print(f"已更新 {update_result['updated_count']} 条知识到 SKILL.md：")
        for d in update_result["details"]:
            if d.get("updated"):
                for q in d.get("questions", []):
                    print(f"  {q}")

        # 模拟模式下同步提升质量参数（让模拟回答更完整）
        if not use_real:
            current_quality = min(1.0, current_quality + quality_step)
            set_llm_quality(current_quality)
            iteration_record["new_quality"] = current_quality

        # 检查是否陷入停滞
        if current_pass_rate <= prev_pass_rate:
            stagnant_count += 1
        else:
            stagnant_count = 0

        if stagnant_count >= stagnant_limit:
            print(f"\n连续 {stagnant_limit} 次迭代通过率未提升，进化收敛。建议人工审查。")
            iteration_record["status"] = "进化收敛"
            evolution_log.append(iteration_record)
            break

        prev_pass_rate = current_pass_rate
        evolution_log.append(iteration_record)

    # 保存进化日志
    log_path = "evolution_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(evolution_log, f, ensure_ascii=False, indent=2)
    print(f"\n进化日志已保存到 {log_path}")

    # 返回摘要
    final_rate = evolution_log[-1]["pass_rate"] if evolution_log else 0
    return {
        "total_iterations": len(evolution_log),
        "initial_pass_rate": evolution_log[0]["pass_rate"] if evolution_log else 0,
        "final_pass_rate": final_rate,
        "improvement": round(final_rate - evolution_log[0]["pass_rate"], 2) if evolution_log else 0,
        "mode": "deepseek" if use_real else "simulate",
    }


def main():
    """主函数 - 完整流程演示"""
    print("=" * 60)
    print("知识库问答 Skill 进化系统（SKILL.md 自更新版）")
    print("=" * 60)

    # 检查运行模式
    use_real = is_using_real_model()
    mode_str = "DeepSeek 真实模型" if use_real else "模拟模式（未设置 DEEPSEEK_API_KEY）"
    print(f"运行模式: {mode_str}")
    if not use_real:
        print("提示: 设置环境变量 DEEPSEEK_API_KEY 或在 qa_skill.py 中填入 API Key 以使用真实模型")
    print()

    # 步骤1：加载并展示初始 SKILL.md
    print("[步骤 1] 加载 SKILL.md（问答唯一知识源）...")
    load_skill_md()
    qa_pairs = get_skill_qa_pairs()
    print(f"SKILL.md 当前包含 {len(qa_pairs)} 条知识：")
    for qa in qa_pairs:
        print(f"  - Q: {qa['question'][:40]}...")

    # 步骤2：展示问答（仅基于 SKILL.md）
    print("\n[步骤 2] 问答演示（仅基于 SKILL.md 内容）...")
    if not use_real:
        set_llm_quality(0.55)

    demo_questions = [
        "什么是RESTful API？",  # SKILL.md 中有，应能回答
        "Git是什么？",           # SKILL.md 中没有，应答"没有相关信息"
    ]
    for q in demo_questions:
        answer = ask_question(q)
        print(f"Q: {q}")
        print(f"A: {answer[:80]}...")
        print()

    # 步骤3：初始测试（进化前）
    print("[步骤 3] 初始测试（进化前，SKILL.md 知识不完整）...")
    if not use_real:
        set_llm_quality(0.55)
    initial_report = run_tests()
    print_report(initial_report)

    # 步骤4：进化（通过更新 SKILL.md 来提升通过率）
    print("\n[步骤 4] 开始 Skill 进化（从 knowledge_base.md 补充知识到 SKILL.md）...")
    result = evolve(max_iterations=5, threshold=0.6, stagnant_limit=3)

    # 步骤5：输出最终结果
    print("\n" + "=" * 60)
    print("进化结果摘要")
    print("=" * 60)
    print(f"运行模式: {result['mode']}")
    print(f"总迭代次数: {result['total_iterations']}")
    print(f"初始通过率: {result['initial_pass_rate']:.0%}")
    print(f"最终通过率: {result['final_pass_rate']:.0%}")
    print(f"提升幅度: {result['improvement']:.0%}")

    # 展示最终 SKILL.md 知识条目
    final_qa = get_skill_qa_pairs()
    print(f"\nSKILL.md 最终包含 {len(final_qa)} 条知识：")
    for qa in final_qa:
        print(f"  - Q: {qa['question'][:40]}...")
    print("=" * 60)


if __name__ == "__main__":
    main()
