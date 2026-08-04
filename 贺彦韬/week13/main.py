"""CLI 入口。

不带 query 或加 --list 时只打印 Level 1 索引，不需要 API key，方便离线检查 skill 是否被正确发现。
带 query 时会真正调用阿里云百炼（DashScope，OpenAI 兼容模式）跑一次渐进式加载的 agent 循环，
需要设置环境变量 DASHSCOPE_API_KEY（详见 harness.py 顶部说明）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from skill_loader import SkillLibrary

DEFAULT_SKILLS_DIR = Path(__file__).parent / "skills"


def main() -> None:
    parser = argparse.ArgumentParser(description="渐进式 skill 加载 harness")
    parser.add_argument("query", nargs="?", help="要问助手的问题")
    parser.add_argument("--skills-dir", default=str(DEFAULT_SKILLS_DIR))
    parser.add_argument("--model", default=None, help="默认读 DASHSCOPE_MODEL 环境变量，否则用 harness.DEFAULT_MODEL")
    parser.add_argument("--list", action="store_true", help="只打印 Level 1 索引，不调用 API")
    args = parser.parse_args()

    skills_dir = Path(args.skills_dir)

    if args.list or not args.query:
        library = SkillLibrary.discover(skills_dir)
        print(library.index_text())
        return

    from harness import run  # 延迟导入：没有 openai 依赖时 --list 也能跑

    answer = run(args.query, skills_dir, model=args.model)
    print("\n=== 最终回答 ===")
    print(answer)


if __name__ == "__main__":
    main()
