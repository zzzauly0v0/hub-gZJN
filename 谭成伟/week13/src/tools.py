"""
Built-in Tools — Agent 可调用的基础工具集
============================================
最小工具集：
  - file_read:    读取文件内容
  - file_write:   写入文件（LLM 生成 JSON 数据时用）
  - shell_exec:   执行 Shell 命令（运行 make_flashcard.py 脚本）
  - open_browser: 用默认浏览器打开文件
"""

import subprocess
import webbrowser
from pathlib import Path


class Tool:
    """工具：名称、描述、执行函数"""

    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func

    def execute(self, **kwargs) -> str:
        try:
            return self.func(**kwargs)
        except Exception as e:
            return f"[ERROR] {type(e).__name__}: {e}"


class ToolRegistry:
    """工具注册表：管理所有可用工具"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def execute(self, name: str, args: dict) -> str:
        if name not in self._tools:
            return f"[ERROR] 未知工具: {name}"
        return self._tools[name].execute(**args)

    def get_schemas(self) -> list[dict]:
        """返回所有工具的 schema（注入 system prompt 供 LLM 感知可用工具）"""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]


# ── 内置工具实现 ──────────────────────────────────────────────

def _file_read(path: str) -> str:
    """读取文件内容"""
    p = Path(path)
    if not p.exists():
        return f"[ERROR] 文件不存在: {path}"
    return p.read_text(encoding="utf-8")



def _file_write(path: str, content: str) -> str:
    """写入文件（自动创建父目录）"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"[OK] 已写入 {p.name}"


def _shell_exec(command: str) -> str:
    """执行 Shell 命令，返回 stdout"""
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    output = (result.stdout or "").strip()
    if result.returncode != 0:
        err = (result.stderr or "").strip()
        if err:
            output += f"\n[STDERR] {err}" if output else f"[STDERR] {err}"
        output += f"\n[EXIT CODE] {result.returncode}"
    return output or "(无输出)"


def _open_browser(path: str) -> str:
    """用默认浏览器打开文件"""
    abs_path = str(Path(path).resolve())
    # Windows 路径需要正斜杠
    url = f"file:///{abs_path.replace(chr(92), '/')}"
    webbrowser.open(url)
    return f"[OK] 已在浏览器中打开: {abs_path}"


def create_default_registry() -> ToolRegistry:
    """创建默认工具注册表"""
    registry = ToolRegistry()
    registry.register(Tool("file_read", "读取指定路径的文件内容", _file_read))
    registry.register(Tool("file_write", "将内容写入文件（自动创建目录）", _file_write))
    registry.register(Tool("shell_exec", "执行 Shell 命令并返回输出", _shell_exec))
    registry.register(Tool("open_browser", "用默认浏览器打开文件", _open_browser))
    return registry
