"""
Progressive Skill Loading Harness (渐进式技能加载执行器)

核心思想：
  1. 启动阶段   — 仅扫描 SKILL.md 的 YAML 头部元数据（名字+描述），不加载正文
  2. 匹配阶段   — 用户查询只与已加载的轻量元数据匹配
  3. 渐进加载   — 匹配到某个 skill 后，才加载其完整 SKILL.md 正文
  4. 级联加载   — 若该 skill 引用了其他 skill，被引用者同样渐进加载
  5. 执行阶段   — 全部所需 skill 加载完毕后，按依赖顺序执行

Usage: python harness/harness.py              # 演示模式
       python harness/harness.py -i           # 交互模式
"""

import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SkillMeta:
    """轻量元数据 —— 启动时即加载，非常廉价"""
    name: str
    description: str
    path: Path
    source: str


@dataclass
class SkillFull:
    """完整 skill —— 渐进加载后才拥有"""
    meta: SkillMeta
    body: str
    referenced_skills: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# 第1层：轻量注册表 —— 仅解析 YAML frontmatter
# ──────────────────────────────────────────────────────────────────────

class SkillRegistry:

    def __init__(self):
        self._skills: dict[str, SkillMeta] = {}

    @staticmethod
    def _parse_frontmatter(filepath: Path) -> Optional[dict]:
        """仅解析 YAML frontmatter。支持 BOM、多行描述（>- / | 语法）。"""
        try:
            text = filepath.read_text(encoding="utf-8-sig")  # utf-8-sig strips BOM
        except Exception:
            return None

        m = re.match(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
        if not m:
            return None

        fm_raw = m.group(1)
        fm = {}
        current_key = None
        current_value_lines = []

        for line in fm_raw.split('\n'):
            kv = re.match(r'^(\w+):\s*(.*)', line)
            if kv and not line.startswith(' '):
                if current_key:
                    fm[current_key] = ' '.join(current_value_lines).strip()
                current_key = kv.group(1)
                val = kv.group(2)
                if val in ('>-', '|', '>', '|-', '|+'):
                    current_value_lines = []
                else:
                    current_value_lines = [val]
            elif current_key and line.startswith('  '):
                current_value_lines.append(line.strip())

        if current_key:
            fm[current_key] = ' '.join(current_value_lines).strip()

        return fm if 'name' in fm else None

    def scan(self, *search_dirs: Path, source_label: str = "project"):
        for base in search_dirs:
            if not base.exists():
                continue
            for skill_md in base.rglob("SKILL.md"):
                fm = self._parse_frontmatter(skill_md)
                if fm:
                    meta = SkillMeta(
                        name=fm['name'],
                        description=fm.get('description', ''),
                        path=skill_md,
                        source=source_label,
                    )
                    # cursor skills 优先保留（后扫描的 project 不覆盖）
                    if meta.name not in self._skills:
                        self._skills[meta.name] = meta
        return self

    def list_all(self) -> list[SkillMeta]:
        return list(self._skills.values())

    def get_meta(self, name: str) -> Optional[SkillMeta]:
        return self._skills.get(name)

    def match_by_keywords(self, query: str) -> list[SkillMeta]:
        q = query.lower()
        results = []
        for meta in self._skills.values():
            score = 0
            desc = meta.description.lower()
            name = meta.name.lower()
            # 完整短语匹配最高分
            if q in desc:
                score += 10
            if q in name:
                score += 8
            # 逐词匹配
            for word in q.split():
                if word in name:
                    score += 3
                if word in desc:
                    score += 1
            # 中文子串匹配
            for i in range(len(q)):
                for j in range(i + 2, len(q) + 1):
                    if q[i:j] in desc:
                        score += 1
                        break
            if score > 0:
                results.append((score, meta))
        results.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in results]


# ──────────────────────────────────────────────────────────────────────
# 第2层：渐进加载器
# ──────────────────────────────────────────────────────────────────────

class SkillLoader:

    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self._loaded: dict[str, SkillFull] = {}
        self._load_count = 0

    @property
    def load_count(self) -> int:
        return self._load_count

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def load(self, name: str) -> Optional[SkillFull]:
        if name in self._loaded:
            print(f"  [SKIP] '{name}' 已加载，复用缓存")
            return self._loaded[name]

        meta = self.registry.get_meta(name)
        if not meta:
            print(f"  [MISS] '{name}' 未在注册表中找到")
            return None

        self._load_count += 1
        print(f"  [LOAD #{self._load_count}] 渐进加载 '{name}' <- {meta.path.name}")

        raw = meta.path.read_text(encoding="utf-8-sig")
        body = re.sub(r'^---.*?---\s*', '', raw, count=1, flags=re.DOTALL).strip()

        refs = self._detect_skill_refs(body, exclude_self=name)
        full = SkillFull(meta=meta, body=body, referenced_skills=refs)
        self._loaded[name] = full

        if refs:
            print(f"         `-> 检测到引用: {refs}")
        return full

    def load_cascade(self, name: str, visited: set | None = None) -> list[SkillFull]:
        if visited is None:
            visited = set()
        if name in visited:
            return []
        visited.add(name)

        full = self.load(name)
        if not full:
            return []

        result = [full]
        for ref_name in full.referenced_skills:
            result.extend(self.load_cascade(ref_name, visited))
        return result

    def _detect_skill_refs(self, body: str, exclude_self: str = "") -> list[str]:
        refs = []
        all_names = {m.name for m in self.registry.list_all()}
        for name in all_names:
            if name == exclude_self:
                continue
            pattern = re.compile(r'(?<![-\w])' + re.escape(name) + r'(?![-\w])')
            if pattern.search(body):
                refs.append(name)
        return refs


# ──────────────────────────────────────────────────────────────────────
# 第3层：执行器
# ──────────────────────────────────────────────────────────────────────

class SkillHarness:

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.registry = SkillRegistry()
        self.loader: Optional[SkillLoader] = None
        self._cursor_skills = project_root / ".cursor" / "skills"
        self._project_skills = project_root / "skills"

    def boot(self):
        self._banner("阶段1：启动 — 扫描 Skill 元数据（仅 frontmatter）")

        # cursor 先扫描（优先级高）
        self.registry.scan(self._cursor_skills, source_label="cursor")
        self.registry.scan(self._project_skills, source_label="project")
        self.loader = SkillLoader(self.registry)

        metas = self.registry.list_all()
        print(f"  发现 {len(metas)} 个 skill（仅元数据）：")
        for m in metas:
            desc = m.description[:60] + "..." if len(m.description) > 60 else m.description
            print(f"    * {m.name} [{m.source}]  {desc}")
        print(f"  当前内存：仅 {len(metas)} 条轻量记录，未加载任何正文\n")

    def execute(self, user_query: str):
        self._banner(f'阶段2：匹配 — 查询: "{user_query}"')

        matches = self.registry.match_by_keywords(user_query)
        if not matches:
            print("  未匹配到任何 skill\n")
            return

        print(f"  匹配到 {len(matches)} 个 skill（基于轻量描述）：")
        for m in matches:
            print(f"    * {m.name} [{m.source}]")
        print()

        self._banner("阶段3：渐进加载 — 仅加载匹配的 skill 正文")

        all_loaded = []
        for m in matches:
            loaded = self.loader.load_cascade(m.name)
            all_loaded.extend(loaded)

        seen = set()
        unique = []
        for s in all_loaded:
            if s.meta.name not in seen:
                seen.add(s.meta.name)
                unique.append(s)

        print(f"\n  本次共渐进加载 {self.loader.load_count} 次")
        print(f"  最终参与执行: {[s.meta.name for s in unique]}\n")

        self._banner("阶段4：执行 — 打印每个 skill 的摘要")

        for i, skill in enumerate(unique, 1):
            self._exec_one(i, skill)

    def _exec_one(self, index: int, skill: SkillFull):
        lines = skill.body.split('\n')
        refs_str = ""
        if skill.referenced_skills:
            loaded_refs = [r for r in skill.referenced_skills if self.loader.is_loaded(r)]
            unloaded_refs = [r for r in skill.referenced_skills if not self.loader.is_loaded(r)]
            refs_str = f"  已加载引用={loaded_refs}  未加载引用={unloaded_refs}"

        print(f"  [{index}] {skill.meta.name} [{skill.meta.source}]")
        print(f"       文件: {skill.meta.path}")
        print(f"       正文: {len(lines)} 行, {len(skill.body)} 字符{refs_str}")
        print(f"       预览: {lines[0][:80] if lines else '(空)'}")
        print()

    def show_status(self):
        self._banner("加载状态")
        total = len(self.registry.list_all())
        loaded = self.loader.load_count if self.loader else 0
        still = total - loaded
        print(f"  总 skill 数: {total}")
        print(f"  已渐进加载: {loaded}")
        print(f"  未加载（仅元数据）: {still}")
        if still > 0 and self.loader:
            unloaded = [m.name for m in self.registry.list_all()
                        if m.name not in self.loader._loaded]
            print(f"    -> {unloaded}")
        print()

    @staticmethod
    def _banner(title: str):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")


# ──────────────────────────────────────────────────────────────────────
# 演示
# ──────────────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    "画一个架构图",
    "帮我做一张 flash card",
    "提取 PPT 内容并总结成网页",
    "diagram",
]


def demo():
    root = Path(__file__).resolve().parent.parent
    harness = SkillHarness(root)

    print()
    print("=" * 60)
    print("  渐进式 Skill 加载 Harness — 演示")
    print("  核心：技能只在需要时才加载，启动仅持轻量元数据")
    print("=" * 60)

    harness.boot()

    for i, q in enumerate(DEMO_QUERIES, 1):
        print(f"\n  >>> 演示 {i}/{len(DEMO_QUERIES)}: \"{q}\"")
        print(f"  {'-' * 55}")
        harness.execute(q)
        harness.show_status()

    print("=" * 60)
    print("  演示结束。")
    print("  关键：未匹配到的 skill 始终只持轻量元数据，从未加载正文。")
    print("  这就是渐进式加载。")
    print("=" * 60)
    print()


def interactive():
    root = Path(__file__).resolve().parent.parent
    harness = SkillHarness(root)

    print()
    print("=" * 60)
    print("  渐进式 Skill 加载 Harness — 交互模式")
    print("=" * 60)
    harness.boot()
    print("  命令: /status | /demo | /exit\n")

    while True:
        try:
            q = input("  query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见!")
            break
        if not q:
            continue
        if q == "/exit":
            break
        if q == "/status":
            harness.show_status()
            continue
        if q == "/demo":
            for dq in DEMO_QUERIES:
                harness.execute(dq)
            continue
        harness.execute(q)


if __name__ == "__main__":
    if "-i" in sys.argv or "--interactive" in sys.argv:
        interactive()
    else:
        demo()
