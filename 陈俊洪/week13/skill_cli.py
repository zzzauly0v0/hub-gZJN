"""
Skill Harness CLI — 渐进式加载执行演示

  用彩色终端把"三级加载"一步步打出来，让学生亲眼看到 token 是逐级进入上下文的：
    Level 1 元信息目录（常驻）→ 路由命中 → Level 2 正文 → Level 3 资源 → LLM 执行

使用方式：
  python src/skill_cli.py

命令：
  /skills            列出所有 skill 的元信息（Level 1）
  /trace <你的输入>  只做加载+组装，打印三级加载轨迹，不调用 LLM
  /exit              退出

直接输入任意文本 = 路由 → 渐进加载 → 调用 LLM 执行。

依赖：
  pip install openai
  export DEEPSEEK_API_KEY="sk-xxx"   # 或 DASHSCOPE_API_KEY（见 llm_config.py）
"""

import os
import sys
from pathlib import Path

# Windows OpenMP 冲突修复（与项目其余脚本一致）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 让 src/ 内模块可相互 import
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.skill_loader import SkillRegistry
from src.skill_router import SkillRouter
from src.skill_harness import SkillHarness
from src.llm_config import get_chat_client, current_model_info

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
DIM = "\033[2m"

LEVEL_COLOR = {1: CYAN, 2: GREEN, 3: MAGENTA}


def print_catalog(registry: SkillRegistry):
    metas = registry.list_metadata()
    print(f"\n{CYAN}{'─'*64}{RESET}")
    print(f"{CYAN}  Level 1 — 已注册 skill 元信息（常驻上下文，用于路由）{RESET}")
    print(f"{CYAN}{'─'*64}{RESET}")
    if not metas:
        print(f"  {DIM}（skills/ 目录下没有 skill）{RESET}")
    for m in metas:
        print(f"  🧩 {BOLD}{m.name}{RESET}  {DIM}[~{m.token_estimate} tokens]{RESET}")
        print(f"     {DIM}{m.description}{RESET}")
    print(f"{CYAN}{'─'*64}{RESET}\n")


def print_trace(result):
    d = result.decision
    print(f"\n{YELLOW}{'═'*64}{RESET}")
    print(f"{YELLOW}  渐进式加载轨迹{RESET}")
    print(f"{YELLOW}{'═'*64}{RESET}")
    method_desc = {"keyword": "关键词初筛", "llm": "LLM 兜底", "none": "无匹配"}
    print(f"  路由结果：{BOLD}{d.skill_name or '（不使用 skill）'}{RESET}  "
          f"{DIM}方式={method_desc.get(d.method, d.method)}  得分={d.score}{RESET}")
    if d.reason:
        print(f"  {DIM}理由：{d.reason}{RESET}")
    print()
    for step in result.trace:
        color = LEVEL_COLOR.get(step.level, RESET)
        print(f"  {color}L{step.level}{RESET} {step.detail}  {DIM}[{step.char_count} 字符]{RESET}")
    print(f"\n  {DIM}上下文总字符数：{result.total_chars}{RESET}")
    print(f"{YELLOW}{'═'*64}{RESET}\n")


def main():
    info = current_model_info()
    print(f"\n{BOLD}Skill Harness — 渐进式加载执行演示{RESET}")
    print(f"当前模型：{CYAN}{info['display']}{RESET}  "
          f"{DIM}（切换：LLM_PROVIDER=deepseek 或 qwen）{RESET}")
    print("命令：/skills  /trace <文本>  /exit\n")

    registry = SkillRegistry()
    # 有 API key 才启用 LLM 兜底路由，否则纯关键词，保证离线可跑
    use_llm = False
    try:
        get_chat_client()
        use_llm = True
    except EnvironmentError as e:
        print(f"{YELLOW}{e}{RESET}")
        print(f"{DIM}（未设置 API Key：路由仅用关键词，/trace 仍可用，但无法真正调用 LLM）{RESET}\n")

    router = SkillRouter(registry, use_llm=use_llm)
    harness = SkillHarness(registry=registry, router=router)

    print_catalog(registry)

    while True:
        try:
            user_input = input(f"{BOLD}你：{RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("再见！")
            break

        if user_input == "/skills":
            print_catalog(registry)
            continue

        if user_input.startswith("/trace"):
            query = user_input[len("/trace"):].strip()
            if not query:
                print(f"{YELLOW}用法：/trace <你要执行的内容>{RESET}")
                continue
            result = harness.assemble(query)
            print_trace(result)
            continue

        # 正常执行：路由 → 渐进加载 → 调 LLM
        result = harness.run(user_input)
        print_trace(result)
        if result.error:
            print(f"{YELLOW}{result.error}{RESET}\n")
            continue
        label = result.skill.name if result.skill else "通用"
        print(f"{GREEN}[{label}] 回复：{RESET}\n{result.answer}\n")


if __name__ == "__main__":
    main()
