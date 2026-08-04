
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import get_client
from tools import create_default_registry


# ════════════════════════════════════════════════════════════
#  渐进式读取 SKILL.md
# ════════════════════════════════════════════════════════════

def read_skill_header(skill_md_path: Path) -> dict:
    """
    第一层：只读取 SKILL.md 的 frontmatter（头部元数据）。
    返回 {"name": ..., "description": ..., "raw_header": ...}
    占用极少 token，始终可驻留 Context。
    """
    text = skill_md_path.read_text(encoding="utf-8")

    # frontmatter 在两个 --- 之间
    parts = text.split("---")
    if len(parts) < 3:
        return {"name": skill_md_path.parent.name, "description": "", "raw_header": ""}

    header = parts[1].strip()  # frontmatter 内容
    # 简单按行解析 key: value
    info = {"name": "", "description": "", "raw_header": header}
    for line in header.split("\n"):
        if line.startswith("name:"):
            info["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("description:"):
            info["description"] = line.split(":", 1)[1].strip()

    return info


def read_skill_full(skill_md_path: Path) -> str:
    """
    第二层：读取 SKILL.md 完整内容（含执行流程）。
    触发条件满足后才调用，注入 Context 指导大模型执行。
    """
    return skill_md_path.read_text(encoding="utf-8")


def check_trigger(user_input: str, skill_header: dict) -> bool:
    """
    检查用户输入是否匹配 Skill 的触发条件。
    简单判断：输入中包含 skill 名称相关关键词即触发。
    """
    text = user_input.lower()
    # flash-card skill 的触发词
    triggers = ["闪卡", "flash card", "flashcard", "单词卡"]
    return any(t in text for t in triggers)


# ════════════════════════════════════════════════════════════
#  工具定义（OpenAI function calling 格式）
# ════════════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "读取指定路径的文件内容",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "文件路径"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "将内容写入指定路径的文件（自动创建目录）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": "执行 Shell 命令并返回输出（用于运行 Python 脚本等）",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "要执行的命令"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_browser",
            "description": "用默认浏览器打开文件",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "文件路径"}},
                "required": ["path"],
            },
        },
    },
]


# ════════════════════════════════════════════════════════════
#  Agent Loop（ReAct 循环）
# ════════════════════════════════════════════════════════════

def agent_loop(user_input: str, skill_dir: Path, verbose: bool = True) -> str:
    """
    Agent Loop：
      1. 渐进式读取 SKILL.md 头部 → 判断是否触发
      2. 触发 → 加载完整 SKILL.md 注入 system prompt
         未触发 → 正常对话，大模型自由回复
      3. ReAct 循环：调用大模型 → 工具调用 → 执行 → 回传 → 循环
    """
    skill_md = skill_dir / "SKILL.md"

    # ── 渐进式披露：读取头部，判断是否触发 ──
    triggered = False
    if skill_md.exists():
        header = read_skill_header(skill_md)
        triggered = check_trigger(user_input, header)
        if verbose:
            print(f"\n[渐进式披露] 读取 SKILL.md 头部: name={header['name']}")
            print(f"  触发判断: {' 匹配' if triggered else ' 未匹配，走普通对话,请尝试说：给我做张crazy词的闪卡'}")

    # ── 组装 system prompt ──
    if triggered:
        # 触发：加载完整 SKILL.md（第二层）
        skill_content = read_skill_full(skill_md)
        if verbose:
            print(f"  加载完整 SKILL.md（{len(skill_content)} 字符）")
        system_prompt = (
            "你是一个 Agent 助手。请严格按照下面的 Skill 定义执行任务。\n"
            "你可以调用工具来完成任务。\n\n"
            f"## Skill 定义\n{skill_content}\n\n"
            f"## Skill 目录: {skill_dir.resolve()}\n"
            f"## 数据目录: {skill_dir.resolve() / 'data'}\n"
            f"## 脚本: {skill_dir.resolve() / 'scripts' / 'make_flashcard.py'}\n"
        )
    else:
        # 未触发：普通对话，大模型自由回复
        system_prompt = (
            "你是一个有用的 AI 助手。你可以调用工具来帮助用户完成任务。\n"
            "如果用户的问题不需要工具，直接回答即可。"
        )

    # ── 初始化 ──
    client, model = get_client()
    registry = create_default_registry()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    # ── ReAct 循环 ──
    if verbose:
        print("\n[Agent Loop] 开始...")

    max_rounds = 10
    for round_num in range(1, max_rounds + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )
        msg = response.choices[0].message

        # 无工具调用 → 大模型直接回复，结束
        if not msg.tool_calls:
            final = msg.content or ""
            if verbose:
                print(f"\n[完成] {final[:200]}")
            return final

        # 有工具调用 → 执行并回传
        if verbose:
            print(f"\n--- Round {round_num} ---")
        messages.append(msg.model_dump())

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            if verbose:
                print(f"  工具: {fn_name}({fn_args})")

            result = registry.execute(fn_name, fn_args)
            if verbose:
                print(f"  结果: {result[:150]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return f"达到最大轮次 {max_rounds}，任务未完成"
