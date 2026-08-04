"""
评估 Skill 在测试集上的准确率和 token 消耗。

教学点：
  1. 用同一测试集对比优化前后的准确率
  2. 用 tiktoken 估算 Skill 文件的 token 数
  3. 记录响应时间作为执行效率参考
"""

import os
import json
import time
import tiktoken
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


SYSTEM_TEMPLATE = """你是云购商城的智能客服助手。请严格基于以下 Skill 内容回答问题，不要自行推断。

## 回答规则
- 如果 Skill 覆盖了问题：直接给出完整具体的答案
- 如果 Skill 没有覆盖：仅回答"需要联系人工客服"

{skill_content}
"""


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """用 tiktoken 估算 token 数。"""
    try:
        enc = tiktoken.get_encoding(model)
    except Exception:
        # 兜底：按字符数估算
        return len(text) // 4
    return len(enc.encode(text))


def evaluate_skill(skill_path: str, test_path: str, api_key: str | None = None) -> dict:
    """评估单个 Skill 的效果。"""
    client = OpenAI(
        api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    skill_content = Path(skill_path).read_text(encoding="utf-8")
    test_data = json.loads(Path(test_path).read_text(encoding="utf-8"))
    questions = test_data["questions"]

    system_prompt = SYSTEM_TEMPLATE.format(skill_content=skill_content)
    skill_tokens = count_tokens(skill_content)

    correct = 0
    total = len(questions)
    results = []
    total_time = 0.0

    for q in questions:
        start = time.time()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q["question"]},
            ],
            temperature=0,
            max_tokens=300,
        )
        elapsed = time.time() - start
        total_time += elapsed

        answer = response.choices[0].message.content.strip()
        ok = _check_answer(answer, q["required"], q.get("forbidden", []))
        if ok:
            correct += 1

        results.append({
            "id": q["id"],
            "question": q["question"],
            "answer": answer,
            "correct": ok,
            "required": q["required"],
            "forbidden": q.get("forbidden", []),
        })

    return {
        "skill_path": skill_path,
        "skill_tokens": skill_tokens,
        "skill_chars": len(skill_content),
        "total_questions": total,
        "correct": correct,
        "accuracy": round(correct / total, 3),
        "avg_response_time": round(total_time / total, 3),
        "total_response_time": round(total_time, 3),
        "results": results,
    }


# 否定前缀：如果 forbidden 关键词前 NEG_WINDOW 字内出现这些字，视为被否定
NEG_PREFIXES = ("不", "无", "非", "未", "没")
NEG_WINDOW = 4


def _normalize(text: str) -> str:
    """简单归一化：小写、去空格。"""
    return text.lower().replace(" ", "").replace("，", ",").replace("？", "?")


def _forbidden_hits(text: str, keyword: str) -> bool:
    """
    检查 forbidden 关键词是否"真正"出现：
    - 若关键词前 NEG_WINDOW 字内有否定词，视为被否定，不算命中
    - 所有出现位置都被否定 → 未命中；任一出现未被否定 → 命中
    """
    idx = 0
    while True:
        pos = text.find(keyword, idx)
        if pos == -1:
            return False
        prefix = text[max(0, pos - NEG_WINDOW):pos]
        if not any(neg in prefix for neg in NEG_PREFIXES):
            return True
        idx = pos + 1


def _check_answer(answer: str, required: list[str], forbidden: list[str]) -> bool:
    """检查答案是否满足 required 且不含 forbidden。"""
    ans_norm = _normalize(answer)

    for kw in required:
        if _normalize(kw) not in ans_norm:
            return False

    for kw in forbidden:
        if _forbidden_hits(ans_norm, _normalize(kw)):
            return False

    return True


def main():
    import sys
    if len(sys.argv) < 2:
        print("用法: python evaluate_skill.py <skill_path>")
        return

    skill_path = sys.argv[1]
    test_path = Path(__file__).parent.parent / "data" / "test_cases.json"

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：请先设置 DEEPSEEK_API_KEY 环境变量")
        return

    result = evaluate_skill(skill_path, str(test_path), api_key)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
