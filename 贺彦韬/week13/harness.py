"""基于 OpenAI 兼容接口（阿里云百炼 DashScope）的渐进式 skill 加载 agent 循环。

阿里云百炼提供了兼容 OpenAI Chat Completions 的网关，可以直接用官方 openai SDK
指定 base_url 调用 DeepSeek / 通义千问等模型，价格通常比原生海外 API 便宜很多，
适合学生做练习和演示用。

System prompt 里只放 Level 1 索引（所有 skill 的 name + description）。
模型判断需要某个 skill 时，通过 load_skill 工具主动请求 Level 2 正文；
正文里如果提示还要看某个资源文件，模型再通过 load_resource 请求 Level 3 内容。

需要的环境变量：
- DASHSCOPE_API_KEY（必填）：百炼控制台「API-KEY 管理」里创建。
- DASHSCOPE_BASE_URL（可选）：默认是公共 compatible-mode 网关；如果你用的模型
  在控制台给出的是专属 workspace 网关地址，把完整 URL 填到这里覆盖默认值。
- DASHSCOPE_MODEL（可选）：模型 id。去控制台「模型广场」确认当前可用模型和价格，
  想换更便宜的模型（比如 deepseek-v4-flash 或 qwen-flash）改这一个环境变量就行，
  不用改代码。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

from skill_loader import SkillLibrary

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# run_python 直接在本机子进程里跑模型生成的代码，没有沙箱隔离，只有超时限制。
# 只在你信任 skills/ 目录下内容、本机自己用的场景下开启。
PYTHON_EXEC_TIMEOUT_SECONDS = 30
MAX_TOOL_OUTPUT_CHARS = 4000

# 到期前需要把 DASHSCOPE_MODEL 换成 deepseek-v4-flash 之类的替代模型。
DEFAULT_MODEL = "deepseek-v4-pro"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": (
                "加载某个 skill 的完整正文内容（Level 2）。"
                "当你根据 skill 列表里的 name/description 判断当前任务需要用到某个 skill 时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "skill 的 name，例如 pdf-tools"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_resource",
            "description": (
                "加载某个 skill 目录下、被正文引用到的资源文件内容（Level 3），"
                "例如 references/xxx.md 或 scripts/xxx.py。必须先 load_skill 才知道有哪些资源文件。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "skill 的 name"},
                    "resource_path": {
                        "type": "string",
                        "description": "相对于该 skill 目录的相对路径，例如 references/form_fields.md",
                    },
                },
                "required": ["skill_name", "resource_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "执行一段 Python 代码，用来真正完成 skill 正文里描述的操作（比如读写文件、转换图片）。"
                "代码在本机子进程里运行，工作目录和启动 harness 时的当前目录一致，"
                "所以用户给的相对路径可以直接用。运行有超时限制。"
                "只有调用这个工具并且返回结果显示成功，才能认为任务真的完成了；"
                "光是描述该怎么做、或者只加载了 skill 内容，都不代表任务已经完成。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的完整 Python 代码"},
                },
                "required": ["code"],
            },
        },
    },
]


def build_system_prompt(library: SkillLibrary) -> str:
    return "\n".join(
        [
            "你是一个具备渐进式 skill 加载（progressive disclosure）能力的助手。",
            "下面只列出了每个 skill 的名字和简介（Level 1），完整内容需要你主动调用工具才能看到。",
            "",
            library.index_text(),
            "",
            "工作方式：",
            "1. 先根据 name/description 判断本次任务是否需要某个 skill，不相关的 skill 不要理会。",
            "2. 需要时调用 load_skill 获取该 skill 的完整正文（Level 2）。",
            "3. 如果正文中提到了额外的参考文件或脚本、且确实需要查看，再调用 load_resource 获取（Level 3）。",
            "4. 只加载完成任务真正需要的部分，不要把所有 skill 和所有资源文件都加载一遍。",
            "5. 如果任务需要真正操作文件（生成、转换、修改文件等），必须调用 run_python 实际执行代码，"
            "不能只是把 skill 里的示例代码讲一遍就说完成了。",
            "6. 在 run_python 返回结果确认成功之前，不要在回复里宣称任务已经完成；"
            "如果执行报错，如实告诉用户报错内容，不要编造成功结果。",
        ]
    )


def run_python_code(code: str) -> str:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=PYTHON_EXEC_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"错误: 执行超过 {PYTHON_EXEC_TIMEOUT_SECONDS} 秒，已终止"

    stdout = proc.stdout[-MAX_TOOL_OUTPUT_CHARS:]
    stderr = proc.stderr[-MAX_TOOL_OUTPUT_CHARS:]
    return (
        f"returncode: {proc.returncode}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )


def dispatch_tool(library: SkillLibrary, name: str, tool_input: dict, verbose: bool = True) -> str:
    try:
        if name == "load_skill":
            skill_name = tool_input["name"]
            if verbose:
                print(f"[Level 2] 加载 skill 正文: {skill_name}")
            return library.load_skill_body(skill_name)
        if name == "load_resource":
            skill_name = tool_input["skill_name"]
            resource_path = tool_input["resource_path"]
            if verbose:
                print(f"[Level 3] 加载资源文件: {skill_name}/{resource_path}")
            return library.load_resource(skill_name, resource_path)
        if name == "run_python":
            if verbose:
                print("[执行] 运行 run_python:")
                print(tool_input.get("code", ""))
            return run_python_code(tool_input["code"])
        return f"未知工具: {name}"
    except (KeyError, FileNotFoundError, ValueError) as exc:
        return f"错误: {exc}"


def run(
    query: str,
    skills_dir: Path,
    model: str | None = None,
    max_turns: int = 8,
    verbose: bool = True,
) -> str:
    library = SkillLibrary.discover(skills_dir)
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
    )
    model = model or os.environ.get("DASHSCOPE_MODEL", DEFAULT_MODEL)

    system = build_system_prompt(library)
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]

    if verbose:
        print(f"[Level 1] 已加载 {len(library.entries)} 个 skill 的索引（仅 name + description）")
        print(f"[模型] {model}")

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return message.content or ""

        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")
            result_text = dispatch_tool(library, tool_call.function.name, args, verbose=verbose)
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result_text}
            )

    raise RuntimeError(f"超过最大轮数 {max_turns} 仍未得到最终回答")
