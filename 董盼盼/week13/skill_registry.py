# =============================================================================
# 文件: skill_registry.py
# 意图: Skill 注册表 - 自动发现和管理系统中所有 Skill
# 功能: 扫描 skills/ 目录，识别包含 SKILL.md 的 Skill 并注册到注册表
#   - discover():      扫描目录，自动发现并注册所有 Skill
#   - _parse_skill():  解析 SKILL.md 文件，提取元数据、数据文件、脚本文件
#   - _extract_metadata(): 从 SKILL.md 的 YAML front matter 提取元数据
#   - get_skill():     按名称获取 Skill 对象
#   - has_skill():     检查指定 Skill 是否存在
# 依赖: os, re, yaml, pathlib, typing, .skill
# =============================================================================

import os
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from .skill import Skill, SkillStatus


class SkillRegistry:
    SKILL_MARKER = "SKILL.md"
    
    def __init__(self, skills_dir: Optional[str] = None):
        self.skills_dir = Path(skills_dir) if skills_dir else Path(__file__).parent.parent / "skills"
        self._registry: Dict[str, Skill] = {}
        self._discovered: bool = False
    
    @property
    def skills(self) -> List[Skill]:
        return list(self._registry.values())
    
    @property
    def skill_names(self) -> List[str]:
        return list(self._registry.keys())
    
    def discover(self, force: bool = False) -> int:
        if self._discovered and not force:
            return len(self._registry)
        
        self._registry.clear()
        
        if not self.skills_dir.exists() or not self.skills_dir.is_dir():
            return 0
        
        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            
            skill_md = skill_dir / self.SKILL_MARKER
            if not skill_md.exists():
                continue
            
            skill = self._parse_skill(skill_dir, skill_md)
            if skill:
                self._registry[skill.name] = skill
        
        self._discovered = True
        return len(self._registry)
    
    def _parse_skill(self, skill_dir: Path, skill_md: Path) -> Optional[Skill]:
        try:
            content = skill_md.read_text(encoding="utf-8")
            
            metadata = self._extract_metadata(content)
            name = metadata.get("name", skill_dir.name)
            
            data_files = []
            data_dir = skill_dir / "data"
            if data_dir.exists() and data_dir.is_dir():
                for ext in ("*.json", "*.yaml", "*.yml", "*.txt", "*.csv"):
                    data_files.extend(data_dir.glob(ext))
            
            script_files = []
            script_dir = skill_dir / "scripts"
            if script_dir.exists() and script_dir.is_dir():
                for ext in ("*.py", "*.ts", "*.js", "*.sh"):
                    script_files.extend(script_dir.glob(ext))
            
            return Skill(
                name=name,
                path=skill_dir,
                metadata=metadata,
                data_files=data_files,
                script_files=script_files,
                status=SkillStatus.DISCOVERED
            )
        except Exception as e:
            return None
    
    def _extract_metadata(self, content: str) -> Dict:
        match = re.match(r"^---\n(.+?)\n---", content, re.DOTALL)
        if not match:
            return {}
        
        try:
            return yaml.safe_load(match.group(1))
        except:
            return {}
    
    def get_skill(self, name: str) -> Optional[Skill]:
        return self._registry.get(name)
    
    def has_skill(self, name: str) -> bool:
        return name in self._registry
    
    def get_by_status(self, status: SkillStatus) -> List[Skill]:
        return [s for s in self._registry.values() if s.status == status]
    
    def iter_skills(self) -> Iterator[Skill]:
        return iter(self._registry.values())
    
    def __len__(self) -> int:
        return len(self._registry)
    
    def __contains__(self, name: str) -> bool:
        return self.has_skill(name)
    
    def __getitem__(self, name: str) -> Skill:
        skill = self.get_skill(name)
        if skill is None:
            raise KeyError(f"Skill '{name}' not found")
        return skill
