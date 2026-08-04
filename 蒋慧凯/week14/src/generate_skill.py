"""
用 LLM 生成初始数字商品退款 Skill。

教学点：
  1. 展示如何让 LLM 从零写一个 Skill
  2. 生成的 Skill 可以比较冗长，为后续优化留出空间
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


PROMPT = """你是电商客服系统的设计师。请为"云购商城"编写一份数字商品退款规则 Skill，
供客服 Agent 读取后回答用户问题。

要求：
1. 以 Markdown 格式输出，开头有 YAML frontmatter：
   ---
   name: digital_goods_refund
   description: 数字商品退款规则
   version: 1
   ---
2. 覆盖以下场景：电子书、软件激活码、游戏点卡、平台会员卡、虚拟会员充值卡
3. 明确区分"未激活"和"已激活"软件的不同处理方式
4. 说明数字商品有质量问题时的处理原则
5. 强调数字商品退款政策与普通商品 30 天退货政策不同
6. 内容要详细、完整，便于客服准确回答

请只输出 Skill 内容，不要有多余解释。"""


def generate_initial_skill(api_key: str | None = None) -> str:
    """调用 DeepSeek 生成初始 Skill。"""
    client = OpenAI(
        api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.7,
        max_tokens=1500,
    )
    content = response.choices[0].message.content.strip()
    # 如果 LLM 用 ```markdown 包裹，去掉代码块标记
    content = re.sub(r"^```(?:markdown)?\n?", "", content)
    content = re.sub(r"\n?```$", "", content)
    return content.strip()


def main():
    output_path = Path(__file__).parent.parent / "outputs" / "skill_v1.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：请先设置 DEEPSEEK_API_KEY 环境变量")
        return

    skill = generate_initial_skill(api_key)
    output_path.write_text(skill, encoding="utf-8")
    print(f"[OK] 初始 Skill 已保存: {output_path}")


if __name__ == "__main__":
    main()
