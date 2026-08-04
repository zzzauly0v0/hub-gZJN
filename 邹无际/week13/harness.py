

import sys
import argparse
from pathlib import Path

# 终端颜色
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
DIM = "\033[2m"


# 将 agent_memory_system 加入 sys.path，使 src_mode 可被 import
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent  # agent_memory_system/
sys.path.insert(0, str(_PROJECT_ROOT))

from .skill_parser import SkillParser
from .skill_registry import SkillRegistry
from .skill_executor import SkillExecutor


BANNER = r"""
  ╔══════════════════════════════════════════════════════╗
  ║     Progressive Skill Harness  v1.0                 ║
  ║     渐进式 Skill 加载与执行引擎                      ║
  ╚══════════════════════════════════════════════════════╝
"""


def resolve_skills_dir(cli_dir: str = None) -> Path:
    """
    确定 skills 目录路径，优先级：
      1. CLI 参数 --skills-dir
      2. 环境变量 HARNESS_SKILLS_DIR
      3. 默认相对路径：../skills（相对于 agent_memory_system/）
    """
    if cli_dir:
        return Path(cli_dir).resolve()

    import os
    env_dir = os.environ.get("HARNESS_SKILLS_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    # 默认：agent_memory_system/../skills
    default = _PROJECT_ROOT.parent / "skills"
    return default.resolve()


def print_help():
    print(f"""
{CYAN}可用命令：{RESET}
  {BOLD}/skills{RESET}                   列出所有已发现的 skill（Phase 1 信息）
  {BOLD}/run <name> [args]{RESET}        手动执行某个 skill（Phase 2+3）
  {BOLD}/info <name>{RESET}              查看某个 skill 的详细信息
  {BOLD}/help{RESET}                     显示此帮助
  {BOLD}/exit{RESET}                     退出

{CYAN}直接输入触发 Skill：{RESET}
  输入包含 skill 名称或触发词的话语，自动匹配并执行。
  例如：{DIM}"给我做张 crazy 的闪卡"{RESET}、{DIM}"flash card resilient"{RESET}
""")


def interactive_mode(registry: SkillRegistry, executor: SkillExecutor):
    """交互式主循环"""
    print_help()

    while True:
        try:
            user_input = input(f"\n{BOLD}> {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue

        # ── 命令处理 ─────────────────────────────────────────
        if user_input == "/exit":
            print("再见！")
            break

        if user_input == "/help":
            print_help()
            continue

        if user_input == "/skills":
            registry.list_skills()
            continue

        if user_input.startswith("/run "):
            parts = user_input[5:].strip().split(maxsplit=1)
            name = parts[0] if parts else ""
            args = parts[1] if len(parts) > 1 else ""
            full = registry.activate(name)
            if full:
                executor.execute(full, user_args=args)
            else:
                print(f"  {YELLOW}未找到 skill: {name}{RESET}")
            continue

        if user_input.startswith("/info "):
            name = user_input[6:].strip()
            full = registry.activate(name)
            if full:
                print(f"\n{CYAN}=== Skill: {full.meta.name} ==={RESET}")
                print(f"版本: {full.meta.version}")
                print(f"描述: {full.meta.description}")
                print(f"目录: {full.meta.skill_dir}")
                print(f"\n触发模式 ({len(full.trigger_patterns)} 条):")
                for t in full.trigger_patterns:
                    print(f"  - {t}")
                print(f"\n执行步骤 ({len(full.execution_steps)} 步):")
                for step in full.execution_steps:
                    cmd_tag = f" {DIM}[cmd]{RESET}" if step.command else ""
                    print(f"  {step.index}. {BOLD}{step.title}{RESET}{cmd_tag}")
                    print(f"     {DIM}{step.detail[:80]}{RESET}")
                if full.data_dir:
                    print(f"\n数据目录: {full.data_dir}")
                if full.scripts_dir:
                    print(f"脚本目录: {full.scripts_dir}")
            else:
                print(f"  {YELLOW}未找到 skill: {name}{RESET}")
            continue

        # ── 自动匹配 Skill ────────────────────────────────────
        print(f"\n  {DIM}[匹配中...]{RESET}")
        matched = registry.find_matching(user_input)
        if matched:
            print(f"  {GREEN}匹配到 skill: {BOLD}{matched.meta.name}{RESET}")

            # 从用户输入中提取参数（去掉 skill 名称和常见触发词）
            args = _extract_args(user_input, matched.meta.name)
            executor.execute(matched, user_args=args)
        else:
            print(f"  {YELLOW}没有匹配到任何 skill。{RESET}")
            print(f"  {DIM}输入 /skills 查看可用 skill，/help 查看帮助{RESET}")


def _extract_args(user_input: str, skill_name: str) -> str:
    """从用户输入中提取传给 skill 的参数"""
    text = user_input.lower()

    # 移除 skill 名称
    text = text.replace(skill_name.lower(), "")

    # 移除常见触发词
    trigger_words = [
        "给我做", "做一张", "做一个", "帮我生成", "生成", "的闪卡", "闪卡",
        "的flash card", "flash card", "的单词卡", "单词卡", "for", "make",
        "create", "a", "an", "the", "张", "个", "请", "帮", "我"
    ]
    for tw in trigger_words:
        text = text.replace(tw, " ")

    # 提取剩余的英文单词作为参数
    words = text.split()
    english_words = [w for w in words if w.isalpha() and len(w) > 1]

    if english_words:
        return english_words[-1]  # 取最后一个英文单词作为目标词

    # 如果没有找到英文单词，返回原始输入中去掉中文后的部分
    remaining = " ".join(english_words)
    return remaining.strip()


def main():
    parser = argparse.ArgumentParser(description="渐进式 Skill 执行 Harness")
    parser.add_argument(
        "--skills-dir",
        help="skills 目录路径（默认: ../skills）"
    )
    parser.add_argument(
        "--run",
        nargs="+",
        metavar=("SKILL", "ARGS"),
        help="直接执行某个 skill（非交互模式）"
    )
    parser.add_argument(
        "--working-dir",
        help="工作目录（skill 输出文件的位置）"
    )
    args = parser.parse_args()

    # 打印 Banner
    print(f"{CYAN}{BANNER}{RESET}")

    # 确定 skills 目录
    skills_dir = resolve_skills_dir(args.skills_dir)
    print(f"  Skills 目录: {DIM}{skills_dir}{RESET}")

    # 确定工作目录
    working_dir = Path(args.working_dir).resolve() if args.working_dir else Path.cwd()
    print(f"  工作目录:   {DIM}{working_dir}{RESET}\n")

    # ── Phase 1: 发现所有 skills ──────────────────────────────
    print(f"{CYAN}── Phase 1: Discover ──{RESET}")
    registry = SkillRegistry(skills_dir)
    discovered = registry.discover()

    if not discovered:
        print(f"{YELLOW}没有发现任何 skill，请检查 skills 目录。{RESET}")
        sys.exit(1)

    # 创建执行器
    executor = SkillExecutor(working_dir=working_dir)

    # ── 直接执行模式 ─────────────────────────────────────────
    if args.run:
        skill_name = args.run[0]
        skill_args = " ".join(args.run[1:]) if len(args.run) > 1 else ""

        print(f"\n{CYAN}── Phase 2: Activate ──{RESET}")
        full = registry.activate(skill_name)
        if not full:
            print(f"{YELLOW}未找到 skill: {skill_name}{RESET}")
            sys.exit(1)

        print(f"\n{CYAN}── Phase 3: Execute ──{RESET}")
        success = executor.execute(full, user_args=skill_args)
        sys.exit(0 if success else 1)

    # ── 交互模式 ─────────────────────────────────────────────
    print(f"{CYAN}── 进入交互模式 ──{RESET}")
    print(f"  输入包含 skill 名称的话语自动触发，或 /help 查看命令\n")
    interactive_mode(registry, executor)


if __name__ == "__main__":
    main()
