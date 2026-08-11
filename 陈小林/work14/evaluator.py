"""评测模块 - 通过关键词匹配判断 LLM 回答是否正确
仅基于 SKILL.md 内容进行评测，不再直接读取 knowledge_base.md 作答
"""

import json
from typing import Dict, List

from qa_skill import ask_question, load_skill_md


def evaluate_answer(
    llm_answer: str,
    keywords: List[str],
    threshold: float = 0.6
) -> Dict:
    """
    通过关键词匹配判断 LLM 回答是否正确。

    匹配度 = 命中关键词数 / 总关键词数

    Args:
        llm_answer: LLM 生成的回答文本
        keywords: 标准答案的关键词列表
        threshold: 通过阈值（默认 0.6）

    Returns:
        评测结果字典
    """
    hit_keywords = []
    missed_keywords = []

    answer_lower = llm_answer.lower()

    for keyword in keywords:
        if keyword.lower() in answer_lower:
            hit_keywords.append(keyword)
        else:
            missed_keywords.append(keyword)

    match_ratio = len(hit_keywords) / len(keywords) if keywords else 0.0
    passed = match_ratio >= threshold

    return {
        "passed": passed,
        "hit_keywords": hit_keywords,
        "missed_keywords": missed_keywords,
        "match_ratio": round(match_ratio, 2),
        "threshold": threshold,
    }


def run_tests(
    test_cases_path: str = "test_cases.json",
    threshold: float = 0.6,
    skill_file_path: str = "SKILL.md",
) -> Dict:
    """
    批量执行测试用例，基于 SKILL.md 内容进行问答并评测。

    Args:
        test_cases_path: 测试用例 JSON 文件路径
        threshold: 通过阈值
        skill_file_path: SKILL.md 文件路径

    Returns:
        测试报告字典
    """
    # 加载 SKILL.md（每次测试前重新加载，确保使用最新内容）
    load_skill_md(skill_file_path)

    # 加载测试用例
    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []
    passed_count = 0
    failed_details = []

    for tc in test_cases:
        # 仅基于 SKILL.md 回答问题
        llm_answer = ask_question(tc["question"])

        # 评测
        eval_result = evaluate_answer(llm_answer, tc["keywords"], threshold)

        detail = {
            "id": tc["id"],
            "question": tc["question"],
            "llm_answer": llm_answer,
            "expected_answer": tc["expected_answer"],
            **eval_result,
        }
        results.append(detail)

        if eval_result["passed"]:
            passed_count += 1
        else:
            failed_details.append(detail)

    total = len(test_cases)
    pass_rate = round(passed_count / total, 2) if total > 0 else 0.0

    report = {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": pass_rate,
        "details": results,
        "failed_details": failed_details,
    }

    return report


def print_report(report: Dict):
    """格式化输出测试报告"""
    print("=" * 60)
    print(f"测试报告")
    print(f"{'=' * 60}")
    print(f"总用例数: {report['total']}")
    print(f"通过: {report['passed']}")
    print(f"失败: {report['failed']}")
    print(f"通过率: {report['pass_rate']:.0%}")
    print(f"{'=' * 60}")

    if report["failed_details"]:
        print("\n失败用例详情:")
        print("-" * 60)
        for detail in report["failed_details"]:
            print(f"\n用例 #{detail['id']}: {detail['question']}")
            print(f"  LLM 回答: {detail['llm_answer'][:80]}...")
            print(f"  匹配度: {detail['match_ratio']:.0%} (阈值: {detail['threshold']:.0%})")
            print(f"  未命中关键词: {', '.join(detail['missed_keywords'])}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    report = run_tests()
    print_report(report)
