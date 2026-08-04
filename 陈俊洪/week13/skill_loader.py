"""
Skill 渐进式加载器 — 三级按需加载的核心

  1. Skill 就是"能力的 Markdown 配置"：一个 SKILL.md = frontmatter(元信息) + body(操作指令)
  2. 渐进式加载（Progressive Disclosure）分三级，越晚加载越省 token：
       Level 1  只加载 name + description（元信息），常驻上下文，用于路由
       Level 2  命中某个 skill 后，才把它的 SKILL.md 正文注入上下文
       Level 3  skill 正文里用 `::LOAD 相对路径` 声明的资源，执行时才读进来
  3. 类比 Layer 3 的 Markdown 记忆：同样是"人类可读的 Markdown 即配置"，
     但 skill 关注"怎么做一件事"，记忆关注"关于用户/世界的事实"。

目录约定：
  skills/
    <skill-name>/
      SKILL.md              # frontmatter + 正文（Level 1 + Level 2）
      references/*.md        # 可选资源（Level 3，按需 ::LOAD）
      scripts/*.py           # 可选脚本（Level 3，按需执行）

使用方式：
  from src.skill_loader import SkillRegistry
  reg = SkillRegistry()
  for meta in reg.list_metadata():        # Level 1
      print(meta.name, meta.description)
  skill = reg.load("commit-message")      # Level 2：读 body
  text = skill.load_resource("references/conventional-commits.md")  # Level 3
"""

import re
from pathlib import Path
from dataclasses import dataclass, field

SKILLS_DIR = Path(__file__).parent.parent / "skills"

# 正文里声明 Level 3 资源的指令：`::LOAD references/xxx.md`
LOAD_DIRECTIVE = re.compile(r"::LOAD\s+([^\s`]+)")


@dataclass
class SkillMetadata:
    """Level 1：只有元信息，进入上下文用于路由，成本极低。"""
    name: str
    description: str
    path: Path

    @property
    def token_estimate(self) -> int:
        # 粗略估算：中英文混合约 1.5 字符/token
        return int((len(self.name) + len(self.description)) / 1.5)


@dataclass
class Skill:
    """Level 2：已加载 SKILL.md 正文的完整 skill。"""
    name: str
    description: str
    body: str
    path: Path
    declared_resources: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.body)

    def load_resource(self, rel_path: str) -> str:
        """Level 3：读取 skill 目录下的资源文件（references/xxx.md 等）。"""
        target = (self.path / rel_path).resolve()
        # 防目录穿越：资源必须落在 skill 目录内
        if not str(target).startswith(str(self.path.resolve())):
            raise ValueError(f"资源路径越界：{rel_path}")
        if not target.exists():
            raise FileNotFoundError(f"资源不存在：{rel_path}")
        return target.read_text(encoding="utf-8")

    def resource_path(self, rel_path: str) -> Path:
        target = (self.path / rel_path).resolve()
        if not str(target).startswith(str(self.path.resolve())):
            raise ValueError(f"资源路径越界：{rel_path}")
        return target


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    解析 SKILL.md 顶部的 YAML frontmatter（--- 包裹）。
    只支持 `key: value` 单行键值，够 skill 元信息用，无需引入 PyYAML。
    返回 (元信息字典, 去掉 frontmatter 后的正文)。
    """
    if not text.startswith("---"):
        return {}, text.strip()
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text.strip()

    raw = text[3:end].strip()
    body = text[end + len("\n---"):].lstrip("\n")

    meta: dict = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, body.strip()


class SkillRegistry:
    """扫描 skills/ 目录，管理三级加载。"""

    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills_dir = skills_dir

    def _skill_dirs(self) -> list[Path]:
        if not self.skills_dir.exists():
            return []
        return sorted(
            d for d in self.skills_dir.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        )

    def list_metadata(self) -> list[SkillMetadata]:
        """Level 1：只读每个 SKILL.md 的 frontmatter，不读正文。"""
        metas: list[SkillMetadata] = []
        for d in self._skill_dirs():
            text = (d / "SKILL.md").read_text(encoding="utf-8")
            # 只需 frontmatter，读文件后立即丢弃正文，体现"只加载元信息"
            fm, _ = _parse_frontmatter(text)
            name = fm.get("name") or d.name
            description = fm.get("description", "")
            metas.append(SkillMetadata(name=name, description=description, path=d))
        return metas

    def load(self, name: str) -> Skill:
        """Level 2：加载指定 skill 的 SKILL.md 正文。"""
        for d in self._skill_dirs():
            text = (d / "SKILL.md").read_text(encoding="utf-8")
            fm, body = _parse_frontmatter(text)
            skill_name = fm.get("name") or d.name
            if skill_name == name:
                resources = LOAD_DIRECTIVE.findall(body)
                return Skill(
                    name=skill_name,
                    description=fm.get("description", ""),
                    body=body,
                    path=d,
                    declared_resources=resources,
                )
        raise KeyError(f"未找到 skill：{name}")

    def get(self, name: str) -> Skill | None:
        try:
            return self.load(name)
        except KeyError:
            return None
