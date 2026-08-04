"""
Flash Card Harness — 入口
==========================
用法：
    python run.py "给我做张 crazy 词的闪卡"
    python run.py                        # 交互模式
"""

import sys
import os
from pathlib import Path

# Windows 终端 UTF-8
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from harness import agent_loop

# skill 目录（相对于本文件）
SKILL_DIR = Path(__file__).parent.parent / "skills" / "flash-card"


def main():
    args = sys.argv[1:]

    if args:
        # 命令行模式
        user_input = " ".join(args)
        result = agent_loop(user_input, SKILL_DIR)
        print(f"\n最终结果: {result}")
    else:
        # 交互模式
        print("Flash Card Harness（输入 quit 退出）")
        print(f"Skill 目录: {SKILL_DIR}\n")
        while True:
            try:
                user_input = input("用户> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            result = agent_loop(user_input, SKILL_DIR)
            print(f"\n结果: {result}\n")


if __name__ == "__main__":
    main()
