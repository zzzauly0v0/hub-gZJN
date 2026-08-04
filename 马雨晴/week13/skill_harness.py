from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

import argparse
import json
import os
import re
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any

from openai import OpenAI


# ========== 配置 ==========
DEFAULT_SKILL_PATH = Path(".cursor/skills/flash-card/SKILL.md")
DEFAULT_MODEL = os.getenv("AGENT_MODEL", "qwen-max")

# 默认使用 OpenAI API，可通过环境变量切换
BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://api.openai.com/v1",
)


# ========== 核心函数 ==========

def load_skill(skill_path: Path) -> str:
    """
    读取 SKILL.md，提取核心指令。
    如果文件包含 YAML frontmatter (---)，自动去除。
    """
    if not skill_path.exists():
        raise FileNotFoundError(
            f"找不到 SKILL.md：{skill_path.resolve()}\n"
            "请确认路径是否正确，或使用 --skill 指定路径"
        )

    content = skill_path.read_text(encoding="utf-8")
    
    # 去除 YAML frontmatter
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            # parts[0] 是空字符串，parts[1] 是 frontmatter，parts[2] 是实际内容
            return parts[2].strip()
    
    return content


def create_client() -> OpenAI:
    """创建 OpenAI 兼容客户端。"""
    # 优先使用 OPENAI_API_KEY，其次 DASHSCOPE_API_KEY
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "没有检测到 API Key。\n"
            "请设置环境变量：\n"
            "  export OPENAI_API_KEY='sk-xxx'  # 使用 OpenAI\n"
            "  export DASHSCOPE_API_KEY='sk-xxx'  # 使用通义千问"
        )

    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )


def extract_word_from_request(text: str) -> str:
    """从用户请求中提取目标单词。"""
    # 匹配英文单词（允许连字符和撇号）
    match = re.search(r'\b([a-z][a-z\'-]*)\b', text.lower())
    if match:
        return match.group(1)
    return ""


def remove_markdown_fence(text: str) -> str:
    """移除模型可能返回的 ```json 代码块。"""
    text = text.strip()
    match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else text


def generate_flashcard_data(
    client: OpenAI,
    skill_instruction: str,
    user_request: str,
    model: str,
) -> dict[str, Any]:
    """
    调用 AI 生成闪卡数据。
    完全遵循 SKILL.md 中的 JSON 格式要求。
    """
    
    # 从用户请求中提取目标单词，帮助 AI 聚焦
    target_word = extract_word_from_request(user_request)
    
    system_prompt = f"""
你是一个单词闪卡生成专家。请严格遵循以下规则：

{skill_instruction}

重要提示：
- 用户请求中的目标单词是：{target_word if target_word else "需要你识别"}
- 必须只输出合法的 JSON 对象，不要有任何额外文字
- 不要使用 Markdown 代码块标记
- examples 必须恰好 3 条
- synonyms 包含 4-6 个近义词

JSON 格式：
{{
  "word": "英文单词（小写）",
  "phonetic": "音标（如 /rɪˈzɪliənt/）",
  "pos": "词性（如 adj.）",
  "definition": "中文释义（包含核心含义）",
  "examples": [
    {{"en": "英文例句1", "zh": "中文翻译1"}},
    {{"en": "英文例句2", "zh": "中文翻译2"}},
    {{"en": "英文例句3", "zh": "中文翻译3"}}
  ],
  "synonyms": ["近义词1", "近义词2", "近义词3", "近义词4"]
}}

现在请根据用户请求生成闪卡数据。
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_request},
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("模型返回了空内容。")

    # 清理并解析 JSON
    content = remove_markdown_fence(content)
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"模型没有返回合法 JSON。\n"
            f"原始内容：\n{content[:500]}..."
        ) from exc

    # 验证数据
    validate_flashcard_data(data)
    return data


def validate_flashcard_data(data: dict[str, Any]) -> None:
    """验证闪卡数据是否符合要求。"""
    required = {"word", "phonetic", "pos", "definition", "examples", "synonyms"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"JSON 缺少字段：{', '.join(sorted(missing))}")

    # 验证 word
    word = str(data["word"]).strip().lower()
    if not re.fullmatch(r"[a-z][a-z'-]*", word):
        raise ValueError(f"word 不是合法英文单词：{word!r}")
    data["word"] = word

    # 验证 examples（必须恰好 3 条）
    examples = data["examples"]
    if not isinstance(examples, list) or len(examples) != 3:
        raise ValueError("examples 必须恰好包含 3 条。")
    
    for i, ex in enumerate(examples, 1):
        if not isinstance(ex, dict):
            raise ValueError(f"第 {i} 条例句必须是对象。")
        if not ex.get("en") or not ex.get("zh"):
            raise ValueError(f"第 {i} 条例句缺少 en 或 zh 字段。")

    # 验证 synonyms（4-6 个）
    synonyms = data["synonyms"]
    if not isinstance(synonyms, list):
        raise ValueError("synonyms 必须是列表。")
    if not 4 <= len(synonyms) <= 6:
        raise ValueError("synonyms 必须包含 4-6 个近义词。")


def save_flashcard_json(data: dict[str, Any], skill_dir: Path) -> Path:
    """保存闪卡数据到 skill 的 data 目录。"""
    data_dir = skill_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    word = data["word"]
    output_path = data_dir / f"{word}.json"

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path


def run_html_generator(skill_dir: Path, json_path: Path, output_dir: Path) -> Path:
    """调用 Skill 的 HTML 生成脚本。"""
    generator_path = skill_dir / "scripts" / "make_flashcard.py"
    
    if not generator_path.exists():
        raise FileNotFoundError(
            f"找不到 HTML 生成脚本：{generator_path.resolve()}\n"
            "请确保 skill 目录结构正确。"
        )

    word = json_path.stem
    html_path = output_dir / f"{word}.html"

    print(f"正在生成 HTML：{html_path}")

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(generator_path.resolve()),
                str(json_path.resolve()),
                "-o",
                str(html_path.resolve()),
            ],
            cwd=output_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"HTML 生成脚本执行失败：\n{e.stderr}"
        ) from e

    if not html_path.exists():
        raise RuntimeError(f"脚本执行完成，但没有生成 HTML：{html_path}")

    return html_path


# ========== 主程序 ==========

def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成英语单词闪卡",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python skill_harness.py "给我做一个 resilient 的闪卡"
  python skill_harness.py --skill custom/SKILL.md "生成 crazy 的闪卡"
  python skill_harness.py --no-open "做 meticulous 的闪卡"
        """
    )

    parser.add_argument(
        "request",
        nargs="?",
        help='用户请求，例如："给我做一个 resilient 的闪卡"',
    )

    parser.add_argument(
        "--skill",
        type=Path,
        default=DEFAULT_SKILL_PATH,
        help=f"SKILL.md 的路径（默认：{DEFAULT_SKILL_PATH}）",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"使用的 AI 模型（默认：{DEFAULT_MODEL}）",
    )

    parser.add_argument(
        "--no-open",
        action="store_true",
        help="生成后不自动打开浏览器",
    )

    args = parser.parse_args()

    # 获取用户请求
    user_request = args.request
    if not user_request:
        user_request = input("请输入闪卡请求：").strip()
    if not user_request:
        print("错误：没有输入请求。")
        sys.exit(1)

    # 准备路径
    skill_path = args.skill.resolve()
    skill_dir = skill_path.parent
    output_dir = Path.cwd()

    print(f"\n📚 加载 Skill：{skill_path}")
    print(f"💬 用户请求：{user_request}")
    print(f"🤖 使用模型：{args.model}")
    print()

    try:
        # 1. 加载 skill 指令
        skill_instruction = load_skill(skill_path)
        
        # 2. 创建 AI 客户端
        client = create_client()
        
        # 3. 生成闪卡数据
        print("⏳ 正在生成闪卡数据...")
        data = generate_flashcard_data(
            client=client,
            skill_instruction=skill_instruction,
            user_request=user_request,
            model=args.model,
        )
        print(f"✅ 已生成单词：{data['word']}")
        
        # 4. 保存 JSON
        json_path = save_flashcard_json(data, skill_dir)
        print(f"✅ JSON 已保存：{json_path}")
        
        # 5. 生成 HTML
        print("⏳ 正在生成 HTML...")
        html_path = run_html_generator(skill_dir, json_path, output_dir)
        print(f"✅ HTML 已生成：{html_path}")
        
        # 6. 打开浏览器
        if not args.no_open:
            webbrowser.open(html_path.resolve().as_uri())
            print("✅ 已在浏览器中打开")
        
        print("\n🎉 完成！")
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件错误：{e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ 运行时错误：{e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ 数据验证错误：{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 未知错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()