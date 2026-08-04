"""
用 LLM 优化已有 Skill，目标：减少 token 消耗，同时保持准确率。

教学点：
  1. 展示如何让 LLM 做"压缩"优化
  2. 优化不是删减信息，而是减少冗余表达
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


OPTIMIZE_PROMPT = """你是电商客服系统的技能优化专家。

请对下面这份数字商品退款 Skill 进行优化，目标：
1. **减少 token 消耗**：删除冗余表述、重复内容、不必要的空话
2. **保持信息完整**：不能丢失任何影响判断的关键规则
3. **保持结构清晰**：仍然用 Markdown + YAML frontmatter，便于 LLM 读取
4. 输出格式要求：
   ---
   name: digital_goods_refund
   description: 数字商品退款规则（优化版）
   version: 2
   ---

原始 Skill：

{skill_content}

请直接输出优化后的 Skill 内容，不要有多余解释。"""


def optimize_skill(skill_content: str, api_key: str | None = None) -> str:
    """调用 DeepSeek 优化 Skill。"""
    client = OpenAI(
        api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": OPTIMIZE_PROMPT.format(skill_content=skill_content)}],
        temperature=0.3,
        max_tokens=1500,
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r"^```(?:markdown)?\n?", "", content)
    content = re.sub(r"\n?```$", "", content)
    return content.strip()


def main():
    input_path = Path(__file__).parent.parent / "outputs" / "skill_v1.md"
    output_path = Path(__file__).parent.parent / "outputs" / "skill_v2.md"

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：请先设置 DEEPSEEK_API_KEY 环境变量")
        return

    if not input_path.exists():
        print(f"错误：找不到 {input_path}，请先运行 generate_skill.py")
        return

    skill_v1 = input_path.read_text(encoding="utf-8")
    skill_v2 = optimize_skill(skill_v1, api_key)
    output_path.write_text(skill_v2, encoding="utf-8")
    print(f"[OK] 优化后 Skill 已保存: {output_path}")


if __name__ == "__main__":
    main()
