"""
Skills ReAct Agent 工具集

使用方式：
  from tools import TOOLS_MAP, TOOLS_SCHEMA
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
SKILLS_DIR = BASE_DIR / "skills"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def _parse_frontmatter(content: str) -> dict:
    """解析 SKILL.md 的 YAML frontmatter，返回 name/description 等字段"""
    if not content.startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    fm_text = parts[1]
    # 优先用 yaml 库，否则简单按行解析
    try:
        import yaml
        return yaml.safe_load(fm_text) or {}
    except ImportError:
        result: dict = {}
        current_key = None
        current_lines = []
        for line in fm_text.split("\n"):
            # 新 key（行首非空白且含冒号）
            if line and not line[0].isspace() and ":" in line:
                if current_key:
                    result[current_key] = "\n".join(current_lines).strip().strip('"')
                key, _, val = line.partition(":")
                current_key = key.strip()
                current_lines = [val.strip().strip('"')] if val.strip() else []
            elif current_key:
                stripped = line.strip()
                if stripped:
                    current_lines.append(stripped)
        if current_key:
            result[current_key] = "\n".join(current_lines).strip().strip('"')
        return result


def get_skill_des() -> str:
    """获取所有 skill 的摘要信息（名称 + 描述），用于模型了解可选技能"""
    if not SKILLS_DIR.exists():
        return "当前没有可用的 Skill。"

    lines = []
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        content = skill_md.read_text(encoding="utf-8")
        fm = _parse_frontmatter(content)
        name = fm.get("name", skill_dir.name)
        desc = fm.get("description", "(无描述)")
        # 截断过长的描述
        if len(desc) > 120:
            desc = desc[:120] + "..."
        lines.append(f"- **{name}**：{desc}")

    return "\n".join(lines) if lines else "当前没有可用的 Skill。"


def get_skill_all(skill_name: str) -> str:
    """根据 skill 名称获取完整的 SKILL.md 内容"""
    if not SKILLS_DIR.exists():
        return f"Skill 目录不存在：{SKILLS_DIR}"

    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        content = skill_md.read_text(encoding="utf-8")
        fm = _parse_frontmatter(content)
        if fm.get("name", "").lower() == skill_name.lower():
            return content

    # 也尝试用目录名匹配
    target = SKILLS_DIR / skill_name / "SKILL.md"
    if target.exists():
        return target.read_text(encoding="utf-8")

    return f"未找到名为 '{skill_name}' 的 Skill。"

def tool_use_skill(skill_name: str) -> str:
    """加载指定 skill 的完整使用说明，模型拿到后按说明执行"""
    return get_skill_all(skill_name)


def tool_execute_skill(skill_name: str, params_json: str) -> str:
    """执行 skill 脚本：按参数形状自动适配传参方式"""
    script = {
        "flash-card": SKILLS_DIR / "flash-card" / "scripts" / "make_flashcard.py",
        "PPT Reader": SKILLS_DIR / "ppt-reader" / "scripts" / "read_ppt.py",
    }.get(skill_name)
    if script is None:
        return f"未找到 skill: '{skill_name}'，可用: flash-card, PPT Reader"

    # 解析参数，失败则当原始字符串用
    try:
        params = json.loads(params_json)
    except json.JSONDecodeError:
        params = params_json

    # 字典 → 落盘 JSON，传 JSON 路径；字符串 → 直接传
    if isinstance(params, dict):
        key = params.get("word") or params.get("name") or skill_name
        data_dir = script.parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        data_file = data_dir / f"{key}.json"
        data_file.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
        cli_arg = str(data_file)
    else:
        cli_arg = str(params)

    try:
        proc = subprocess.run(
            [sys.executable, str(script), cli_arg],
            cwd=str(OUTPUTS_DIR), capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "脚本执行超时"

    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        return f"执行失败: {err[-1] if err else f'退出码 {proc.returncode}'}"

    output_file = OUTPUTS_DIR / f"{key if isinstance(params, dict) else 'output'}.html"
    if output_file.exists():
        return f"执行成功，产物: {output_file}"
    return f"执行成功\n{proc.stdout.strip()}"


# ── 统一工具注册表 ─────────────────────────────────────────────────────────────

TOOLS_MAP: dict[str, Any] = {
    "use_skill":     tool_use_skill,
    "execute_skill": tool_execute_skill,
}

# Function Calling 版所需的 JSON Schema 描述
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "use_skill",
            "description": "加载一个 Skill 的完整使用说明。当需要执行复杂任务时，先调此工具获取操作指南。",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Skill 名称"},
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_skill",
            "description": "执行 Skill 脚本。先调 use_skill 获取说明，按说明准备好数据后调此工具跑脚本。参数以 JSON 字符串传入。",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Skill 名称"},
                    "params_json": {"type": "string", "description": "传给脚本的参数，JSON 字符串"},
                },
                "required": ["skill_name", "params_json"],
            },
        },
    },
]
