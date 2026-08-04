import json
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Skill:
    """
    Skill 数据模型。

    字段分两类：
      - 启动时即加载（用于注册和触发匹配）：
          name, description, directory, trigger_examples, trigger_patterns
      - 懒加载（activate() 后才有）：
          execution_steps, word_index, data_files, data_dir, scripts_dir
    """
    name: str
    description: str
    directory: Path
    # -- 启动时加载 --
    trigger_examples: list[str] = field(default_factory=list)
    trigger_patterns: list[str] = field(default_factory=list)
    # -- 懒加载（activate 后填充）--
    execution_steps: list[str] = field(default_factory=list)
    llm_prompt: str = ""  # 从 skill 目录下的 llm_prompt.txt 读取
    word_index: dict[str, dict] = field(default_factory=dict)
    data_files: list[Path] = field(default_factory=list)
    data_dir: Path = Path()
    scripts_dir: Path = Path()
    activated: bool = field(default=False, init=False, repr=False)

    def activate(self) -> "Skill":
        """懒加载：读取 SKILL.md 完整内容 + data/*.json。"""
        if self.activated:
            return self

        md_path = self.directory / "SKILL.md"
        content = md_path.read_text(encoding="utf-8")

        # 执行流程
        self.execution_steps = self._parse_execution_flow(content)

        # LLM 生成指令（从 skill 目录下的 llm_prompt.txt 读取）
        prompt_path = self.directory / "llm_prompt.txt"
        self.llm_prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else ""

        # data 目录
        self.data_dir = self.directory / "data"
        self.scripts_dir = self.directory / "scripts"
        self.data_files, self.word_index = self._load_data_files(self.data_dir)

        self.activated = True
        return self

    @staticmethod
    def _parse_execution_flow(content: str) -> list[str]:
        section = Skill._extract_section(content, "执行流程")
        if not section:
            return []
        steps = re.findall(
            r'\d+\.\s*\*\*(.+?)\*\*[：:]\s*(.+?)(?=\n\d+\.|\n\n|\Z)',
            section, re.DOTALL
        )
        return [f"{title}：{detail.strip()}" for title, detail in steps]

    @staticmethod
    def _load_data_files(data_dir: Path) -> tuple[list[Path], dict[str, dict]]:
        data_files, word_index = [], {}
        if not data_dir.exists():
            return data_files, word_index
        for json_file in sorted(data_dir.glob("*.json")):
            data_files.append(json_file)
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                word = data.get("word", json_file.stem).lower()
                word_index[word] = data
            except (json.JSONDecodeError, KeyError):
                continue
        return data_files, word_index

    @staticmethod
    def _extract_section(content: str, heading: str) -> str:
        pattern = rf'##\s+{re.escape(heading)}\s*\n(.*?)(?=\n##|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""


class SkillLoader:
    """扫描 skills/ 目录，按需加载技能。"""

    def __init__(self, skills_root: Path):
        self.skills_root = Path(skills_root)
        self.skills: dict[str, Skill] = {}

    def discover(self) -> dict[str, Skill]:
        """
        启动时调用：只解析每个 SKILL.md 的 frontmatter + 触发场景，
        不加载 data/ 和执行流程。轻量快速。
        """
        if not self.skills_root.exists():
            return self.skills

        for md_file in sorted(self.skills_root.glob("*/SKILL.md")):
            skill = self._parse_header(md_file)
            if skill:
                self.skills[skill.name] = skill
        return self.skills

    def activate(self, name: str) -> Optional[Skill]:
        """触发时调用：懒加载指定 skill 的完整内容。"""
        skill = self.skills.get(name)
        if skill and not skill.activated:
            skill.activate()
        return skill

    def _parse_header(self, md_path: Path) -> Optional[Skill]:
        """只解析 SKILL.md 的 frontmatter 和触发场景。"""
        content = md_path.read_text(encoding="utf-8")
        frontmatter = self._parse_frontmatter(content)
        if not frontmatter:
            return None

        name = frontmatter.get("name", md_path.parent.name)
        description = frontmatter.get("description", "").replace("\n", " ")
        trigger_examples, trigger_patterns = self._parse_triggers(content)

        return Skill(
            name=name,
            description=description,
            directory=md_path.parent,
            trigger_examples=trigger_examples,
            trigger_patterns=trigger_patterns,
        )

    def _parse_triggers(self, content: str) -> tuple[list[str], list[str]]:
        section = self._extract_section(content, "触发场景")
        if not section:
            return [], []

        examples = re.findall(r'[-*]\s*[""](.+?)[""]', section)
        patterns = [self._example_to_pattern(ex) for ex in examples]
        patterns = [p for p in patterns if p]
        return examples, patterns

    def _example_to_pattern(self, example: str) -> str:
        """把触发例句转为灵活的正则模板。"""
        # 在例句中找已知单词
        known_words = set()
        for s in self.skills.values():
            known_words.update(
                w for w in s.word_index.keys()
                if s.activated  # 只有已激活的 skill 才有 word_index
            )

        example_lower = example.lower()
        for w in sorted(known_words, key=len, reverse=True):
            idx = example_lower.find(w)
            if idx < 0:
                continue

            before_esc = _escape_ascii_regex(example[:idx].strip())
            after_esc = _escape_ascii_regex(example[idx + len(w):].strip())
            before_flex = _apply_chinese_flex(before_esc)
            after_flex = _apply_chinese_flex(after_esc)
            return before_flex + r"\s*([a-zA-Z]+)\s*" + after_flex

        # fallback: 中英文边界
        m = re.search(r"[\u4e00-\u9fa5]\s*([a-zA-Z]+)", example)
        if m:
            w = m.group(1)
            idx = example.index(w)
            before_esc = _escape_ascii_regex(example[:idx].strip())
            after_esc = _escape_ascii_regex(example[idx + len(w):].strip())
            before_flex = _apply_chinese_flex(before_esc)
            after_flex = _apply_chinese_flex(after_esc)
            return before_flex + r"\s*([a-zA-Z]+)\s*" + after_flex
        return ""

    def _extract_section(self, content: str, heading: str) -> str:
        pattern = rf'##\s+{re.escape(heading)}\s*\n(.*?)(?=\n##|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _parse_frontmatter(content: str) -> Optional[dict]:
        match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not match:
            return None
        try:
            return yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            return None

    # ---- 查询接口 ----

    def get_skill(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def list_skills(self) -> list[Skill]:
        return list(self.skills.values())

    def list_activated_skills(self) -> list[Skill]:
        return [s for s in self.skills.values() if s.activated]


def _escape_ascii_regex(s: str) -> str:
    """只对 ASCII 正则元字符转义。"""
    meta = {ord(c): "\\" + c for c in ".^$*+?{}[]\\|()"}
    return s.translate(meta)


def _apply_chinese_flex(s: str) -> str:
    """对中文触发句片段做容错。"""
    # 1) 多字组合先处理
    s = s.replace("闪卡", "(闪卡|flash\\s*card|flashcard|单词卡)")
    s = s.replace("单词卡", "(单词卡|闪卡|flash\\s*card|flashcard)")
    # 2) 张/个 互通
    s = s.replace("张", "\x00X\x00").replace("个", "\x00X\x00")
    s = s.replace("\x00X\x00", "[张个]")
    # 3) [张个] 前补一?
    s = s.replace("[张个]", "一?[张个]")
    # 4) 单字可选
    s = s.replace("词", "词?")
    s = s.replace("的", "的?")
    s = s.replace("一?一?", "一?")
    if s.endswith("做"):
        s += "一?[张个]?"
    return s
