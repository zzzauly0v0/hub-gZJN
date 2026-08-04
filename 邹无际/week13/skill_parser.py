

import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SkillMeta:
    """Phase 1 轻量信息：仅 frontmatter"""
    name: str
    description: str
    version: str = "1.0.0"
    skill_dir: Optional[Path] = None
    skill_md_path: Optional[Path] = None

    @property
    def base_dir(self) -> str:
        """SKILL.md 所在目录，供脚本路径引用"""
        return str(self.skill_dir) if self.skill_dir else ""


@dataclass
class ExecutionStep:
    """一个执行步骤"""
    index: int
    title: str
    detail: str
    command: Optional[str] = None      # 若有 shell 命令
    input_required: bool = False        # 是否需要用户输入
    output_file: Optional[str] = None   # 预期输出文件


@dataclass
class SkillFull:
    """Phase 2 完整信息：包含执行流程"""
    meta: SkillMeta
    trigger_patterns: list[str] = field(default_factory=list)
    execution_steps: list[ExecutionStep] = field(default_factory=list)
    raw_content: str = ""
    data_dir: Optional[str] = None
    scripts_dir: Optional[str] = None


class SkillParser:
    """解析 SKILL.md 文件"""

    # frontmatter 正则：--- ... ---
    FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
    # 执行步骤：数字. **标题**：内容
    STEP_RE = re.compile(r"^(\d+)\.\s+\*\*(.+?)\*\*[：:]\s*(.*)", re.MULTILINE)
    # 触发场景 bullet
    TRIGGER_RE = re.compile(r'^[-•]\s+"?(.+?)"?\s*$', re.MULTILINE)
    # bash 代码块
    CMD_RE = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)

    def parse_meta(self, skill_md_path: Path) -> SkillMeta:
        """Phase 1：只解析 frontmatter，快速扫描"""
        text = skill_md_path.read_text(encoding="utf-8")
        m = self.FRONTMATTER_RE.match(text)
        if not m:
            raise ValueError(f"SKILL.md 缺少 frontmatter: {skill_md_path}")

        fm = yaml.safe_load(m.group(1))
        skill_dir = skill_md_path.parent

        return SkillMeta(
            name=fm.get("name", skill_dir.name),
            description=fm.get("description", ""),
            version=str(fm.get("version", "1.0.0")),
            skill_dir=skill_dir,
            skill_md_path=skill_md_path,
        )

    def parse_full(self, skill_md_path: Path) -> SkillFull:
        """Phase 2：完整解析执行流程"""
        text = skill_md_path.read_text(encoding="utf-8")
        meta = self.parse_meta(skill_md_path)

        # 提取触发场景
        trigger_section = self._extract_section(text, "触发场景")
        triggers = self.TRIGGER_RE.findall(trigger_section) if trigger_section else []

        # 提取执行步骤
        steps = self._parse_steps(text)

        # 确定 data/ 和 scripts/ 目录
        data_dir = str(skill_dir / "data") if (skill_dir := meta.skill_dir) and (skill_dir / "data").exists() else None
        scripts_dir = str(skill_dir / "scripts") if meta.skill_dir and (skill_dir / "scripts").exists() else None

        return SkillFull(
            meta=meta,
            trigger_patterns=triggers,
            execution_steps=steps,
            raw_content=text,
            data_dir=data_dir,
            scripts_dir=scripts_dir,
        )

    def _extract_section(self, text: str, heading: str) -> str:
        """提取某个二级标题下的内容，直到下一个二级标题"""
        pattern = rf"## {re.escape(heading)}\s*\n(.*?)(?=\n## |\Z)"
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1) if m else ""

    def _parse_steps(self, text: str) -> list[ExecutionStep]:
        """从「执行流程」章节解析步骤"""
        section = self._extract_section(text, "执行流程")
        if not section:
            return []

        steps = []
        # 匹配 1. **标题**：内容（可能跨行到下一个数字步骤）
        parts = re.split(r"(?=^\d+\.\s+\*\*)", section, flags=re.MULTILINE)
        for part in parts:
            m = self.STEP_RE.match(part)
            if not m:
                continue
            idx = int(m.group(1))
            title = m.group(2).strip()
            detail = m.group(3).strip()

            # 检查是否有 bash 命令
            cmd = None
            cmd_match = self.CMD_RE.search(part)
            if cmd_match:
                cmd = cmd_match.group(1).strip()

            steps.append(ExecutionStep(
                index=idx,
                title=title,
                detail=detail,
                command=cmd,
            ))

        return steps

    def match_trigger(self, skill: SkillFull, user_input: str) -> bool:
        """判断用户输入是否匹配该 skill 的触发条件"""
        user_lower = user_input.lower()
        name = skill.meta.name.lower()

        # 直接包含 skill 名称
        if name in user_lower:
            return True

        # 匹配触发模式中的关键词
        for pattern in skill.trigger_patterns:
            # 提取模式中的核心词（去掉引号和修饰词）
            core_words = re.findall(r"[a-zA-Z\u4e00-\u9fff]+", pattern.lower())
            # 如果用户输入包含模式中 60% 以上的词，视为匹配
            if core_words:
                matched = sum(1 for w in core_words if w in user_lower)
                if matched / len(core_words) >= 0.5:
                    return True

        return False
