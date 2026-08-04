

from pathlib import Path
from typing import Optional
from .skill_parser import SkillParser, SkillMeta, SkillFull


# 终端颜色
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"


class SkillRegistry:
    """
    Skill 注册表 —— 渐进式加载

    Phase 1 (discover): 只读 frontmatter，建立 skill 索引
    Phase 2 (activate): 按需完整解析，获取执行步骤
    """

    def __init__(self, skills_dir: Path, parser: Optional[SkillParser] = None):
        self.skills_dir = skills_dir
        self.parser = parser or SkillParser()

        # Phase 1 缓存：name -> SkillMeta
        self._meta_cache: dict[str, SkillMeta] = {}
        # Phase 2 缓存：name -> SkillFull
        self._full_cache: dict[str, SkillFull] = {}

    def discover(self) -> list[SkillMeta]:
        """
        Phase 1：扫描 skills_dir 下所有含 SKILL.md 的子目录
        只解析 frontmatter，建立轻量索引
        """
        discovered = []
        if not self.skills_dir.exists():
            print(f"{YELLOW}[Registry] skills 目录不存在: {self.skills_dir}{RESET}")
            return discovered

        for child in sorted(self.skills_dir.iterdir()):
            skill_md = child / "SKILL.md"
            if child.is_dir() and skill_md.exists():
                try:
                    meta = self.parser.parse_meta(skill_md)
                    self._meta_cache[meta.name] = meta
                    discovered.append(meta)
                    print(f"  {GREEN}[Phase 1]{RESET} 发现 skill: {BOLD}{meta.name}{RESET}  "
                          f"{DIM}v{meta.version}{RESET}")
                except Exception as e:
                    print(f"  {YELLOW}[Phase 1]{RESET} 解析失败 {child.name}: {e}")

        print(f"\n  共发现 {CYAN}{len(discovered)}{RESET} 个 skill\n")
        return discovered

    def activate(self, name: str) -> Optional[SkillFull]:
        """
        Phase 2：按需完整加载某个 skill 的执行细节
        如果已经加载过，直接返回缓存
        """
        # 检查缓存
        if name in self._full_cache:
            print(f"  {DIM}[Phase 2] {name} 已加载（缓存命中）{RESET}")
            return self._full_cache[name]

        # 检查是否在 Phase 1 中发现
        if name not in self._meta_cache:
            print(f"  {YELLOW}[Phase 2] 未找到 skill: {name}{RESET}")
            return None

        meta = self._meta_cache[name]
        print(f"  {CYAN}[Phase 2]{RESET} 完整加载: {BOLD}{name}{RESET} ...")

        try:
            full = self.parser.parse_full(meta.skill_md_path)
            self._full_cache[name] = full

            # 打印加载信息
            print(f"    触发模式: {len(full.trigger_patterns)} 条")
            print(f"    执行步骤: {len(full.execution_steps)} 步")
            if full.data_dir:
                print(f"    数据目录: {full.data_dir}")
            if full.scripts_dir:
                print(f"    脚本目录: {full.scripts_dir}")

            return full
        except Exception as e:
            print(f"    {YELLOW}加载失败: {e}{RESET}")
            return None

    def find_matching(self, user_input: str) -> Optional[SkillFull]:
        """
        根据用户输入，找到匹配的 skill 并完整加载
        渐进式：先 Phase 1 匹配名称，再 Phase 2 加载详情
        """
        # 先尝试直接名称匹配（最快）
        for name in self._meta_cache:
            if name.lower() in user_input.lower():
                return self.activate(name)

        # 再尝试触发模式匹配（需要 Phase 2 数据）
        for name in self._meta_cache:
            full = self.activate(name)
            if full and self.parser.match_trigger(full, user_input):
                return full

        return None

    def list_skills(self) -> None:
        """打印所有已发现的 skill 列表"""
        if not self._meta_cache:
            print(f"  {YELLOW}暂无已注册的 skill{RESET}")
            return

        print(f"\n{CYAN}{'─'*60}{RESET}")
        print(f"{CYAN}  已注册 Skills ({len(self._meta_cache)} 个){RESET}")
        print(f"{CYAN}{'─'*60}{RESET}")
        for name, meta in self._meta_cache.items():
            loaded = "[loaded]" if name in self._full_cache else "[ready]"
            desc_short = meta.description[:50].replace("\n", " ") + "..." if len(meta.description) > 50 else meta.description.replace("\n", " ")
            print(f"  [{loaded}] {BOLD}{name}{RESET}  {DIM}v{meta.version}{RESET}")
            print(f"      {desc_short}")
        print(f"{CYAN}{'─'*60}{RESET}\n")

    def get_meta(self, name: str) -> Optional[SkillMeta]:
        return self._meta_cache.get(name)

    def get_full(self, name: str) -> Optional[SkillFull]:
        return self._full_cache.get(name)
