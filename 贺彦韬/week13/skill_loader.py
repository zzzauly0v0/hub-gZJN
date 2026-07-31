"""三级渐进式 skill 加载器。

Level 1: 目录下所有 SKILL.md 的 frontmatter（name + description）—— 始终加载，成本很低。
Level 2: 某个 skill 的 SKILL.md 正文 —— 判断确实需要该 skill 时才加载。
Level 3: 正文中引用的资源文件（references/*.md、scripts/*.py 等）—— 正文提示需要时才加载。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析形如 `---\\nkey: value\\n---\\nbody` 的简单单行 YAML frontmatter。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md 缺少 frontmatter（应以 --- 开头）")

    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        raise ValueError("SKILL.md 的 frontmatter 未正确闭合（缺少结尾的 ---）")

    metadata: dict = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        metadata[key.strip()] = value.strip()

    body = "\n".join(lines[end + 1 :]).strip("\n")
    return metadata, body


@dataclass
class SkillEntry:
    name: str
    description: str
    dir_path: Path
    skill_md_path: Path
    _body: str | None = field(default=None, repr=False)

    def body(self) -> str:
        """Level 2：懒加载并缓存 SKILL.md 正文。"""
        if self._body is None:
            _, self._body = parse_frontmatter(self.skill_md_path.read_text(encoding="utf-8"))
        return self._body

    def load_resource(self, relative_path: str) -> str:
        """Level 3：读取该 skill 目录下的资源文件，禁止越权访问目录外的路径。"""
        target = (self.dir_path / relative_path).resolve()
        root = self.dir_path.resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"拒绝访问 skill 目录之外的路径: {relative_path}")
        if not target.is_file():
            raise FileNotFoundError(f"资源文件不存在: {relative_path}")
        return target.read_text(encoding="utf-8")


class SkillLibrary:
    def __init__(self, entries: list[SkillEntry]):
        self.entries = entries
        self._by_name = {e.name: e for e in entries}

    @classmethod
    def discover(cls, skills_dir: Path) -> "SkillLibrary":
        skills_dir = Path(skills_dir)
        if not skills_dir.is_dir():
            raise FileNotFoundError(f"skills 目录不存在: {skills_dir}")

        entries: list[SkillEntry] = []
        for child in sorted(skills_dir.iterdir()):
            skill_md = child / "SKILL.md"
            if not child.is_dir() or not skill_md.is_file():
                continue
            metadata, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            name = metadata.get("name", child.name)
            description = metadata.get("description", "")
            entries.append(
                SkillEntry(name=name, description=description, dir_path=child, skill_md_path=skill_md)
            )

        return cls(entries)

    def index_text(self) -> str:
        """Level 1 索引：仅 name + description，供 system prompt 使用。"""
        return "\n".join(f"- {e.name}: {e.description}" for e in self.entries)

    def load_skill_body(self, name: str) -> str:
        if name not in self._by_name:
            raise KeyError(f"未知 skill: {name}（可用: {', '.join(self._by_name)}）")
        return self._by_name[name].body()

    def load_resource(self, skill_name: str, resource_path: str) -> str:
        if skill_name not in self._by_name:
            raise KeyError(f"未知 skill: {skill_name}")
        return self._by_name[skill_name].load_resource(resource_path)


if __name__ == "__main__":
    # 不依赖 API key 的自测：只验证三级加载机制本身。
    library = SkillLibrary.discover(Path(__file__).parent / "skills")

    print("=== Level 1: 索引 ===")
    print(library.index_text())

    print("\n=== Level 2: 加载 pdf-tools 正文 ===")
    print(library.load_skill_body("pdf-tools"))

    print("\n=== Level 3: 加载 pdf-tools 的 references/form_fields.md ===")
    print(library.load_resource("pdf-tools", "references/form_fields.md"))

    print("\n=== 安全检查：尝试越权访问目录外文件 ===")
    try:
        library.load_resource("pdf-tools", "../excel-tools/SKILL.md")
    except ValueError as exc:
        print(f"按预期被拒绝: {exc}")
