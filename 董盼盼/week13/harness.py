# =============================================================================
# 文件: harness.py
# 意图: Harness 核心类 - Skill 管理的统一入口
# 功能: 整合 SkillRegistry 和 ProgressiveLoader，提供统一的 Skill 管理接口
#   - discover():        自动发现 skills/ 目录下所有 Skill
#   - load_skill():      渐进式加载指定 Skill，支持进度回调
#   - load_all():        批量加载所有 Skill，支持并发/串行
#   - load_progressive(): 按指定顺序渐进式加载 Skill
#   - execute_flashcard(): 执行 flash-card Skill 生成 HTML 闪卡
#   - create_flashcard_data(): 创建新的 flash-card 数据文件
#   - list_flashcard_words(): 列出所有可用的闪卡单词
# 依赖: asyncio, subprocess, json, pathlib, typing, .skill, .skill_registry, .progressive_loader
# =============================================================================

import asyncio
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from .skill import Skill, SkillStatus
from .skill_registry import SkillRegistry
from .progressive_loader import ProgressiveLoader


class Harness:
    def __init__(self, skills_dir: Optional[str] = None):
        self._registry = SkillRegistry(skills_dir)
        self._loader = ProgressiveLoader(self._registry)
        self._loaded_skills: Dict[str, Skill] = {}
        self._on_skill_discovered: Optional[Callable[[Skill], None]] = None
        self._on_skill_loaded: Optional[Callable[[Skill], None]] = None
        self._on_skill_load_failed: Optional[Callable[[str, str], None]] = None
    
    @property
    def registry(self) -> SkillRegistry:
        return self._registry
    
    @property
    def loader(self) -> ProgressiveLoader:
        return self._loader
    
    @property
    def loaded_skills(self) -> Dict[str, Skill]:
        return self._loaded_skills
    
    def set_discovered_callback(self, callback: Callable[[Skill], None]) -> 'Harness':
        self._on_skill_discovered = callback
        return self
    
    def set_loaded_callback(self, callback: Callable[[Skill], None]) -> 'Harness':
        self._on_skill_loaded = callback
        return self
    
    def set_load_failed_callback(self, callback: Callable[[str, str], None]) -> 'Harness':
        self._on_skill_load_failed = callback
        return self
    
    def discover(self) -> int:
        count = self._registry.discover()
        
        for skill in self._registry.iter_skills():
            if self._on_skill_discovered:
                self._on_skill_discovered(skill)
        
        return count
    
    async def load_skill(self, name: str, 
                        on_progress: Optional[Callable[[float], None]] = None) -> Skill:
        skill = await self._loader.load_skill(
            name,
            on_progress=on_progress,
            on_complete=self._on_skill_loaded,
            on_failed=self._on_skill_load_failed
        )
        
        if skill.is_ready():
            self._loaded_skills[name] = skill
        
        return skill
    
    async def load_all(self, concurrent: bool = True,
                      on_progress: Optional[Callable[[str, float], None]] = None) -> List[Skill]:
        skills = await self._loader.load_all(
            concurrent=concurrent,
            on_progress=on_progress
        )
        
        for skill in skills:
            if skill.is_ready():
                self._loaded_skills[skill.name] = skill
        
        return skills
    
    async def load_progressive(self, names: List[str], 
                              on_progress: Optional[Callable[[str, float], None]] = None) -> List[Skill]:
        self._loader.set_load_order(names)
        return await self.load_all(concurrent=False, on_progress=on_progress)
    
    def get_skill(self, name: str) -> Optional[Skill]:
        return self._loaded_skills.get(name) or self._registry.get_skill(name)
    
    def has_skill(self, name: str) -> bool:
        return name in self._loaded_skills or self._registry.has_skill(name)
    
    def execute_flashcard(self, word: str, output_dir: str = ".") -> Optional[str]:
        skill = self.get_skill("flash-card")
        if not skill or not skill.is_ready():
            raise RuntimeError("flash-card skill is not loaded")
        
        data_dir = skill.path / "data"
        data_file = data_dir / f"{word.lower()}.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"No data file found for '{word}'")
        
        script_file = None
        for sf in skill.script_files:
            if sf.name == "make_flashcard.py":
                script_file = sf
                break
        
        if not script_file:
            raise RuntimeError("make_flashcard.py script not found")
        
        output_path = Path(output_dir) / f"{word.lower()}.html"
        output_path = output_path.resolve()
        
        try:
            cmd = ["python", str(script_file), str(data_file), "-o", str(output_path)]
            cwd_dir = str(skill.path.parent.parent)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=False,
                cwd=cwd_dir
            )
            
            if result.returncode == 0 and output_path.exists():
                return str(output_path)
            else:
                stdout = self._decode_output(result.stdout)
                stderr = self._decode_output(result.stderr)
                error_detail = (
                    f"returncode={result.returncode}, "
                    f"output_exists={output_path.exists()}, "
                    f"stdout='{stdout.strip()}', "
                    f"stderr='{stderr.strip()}', "
                    f"cmd={cmd}, "
                    f"cwd={cwd_dir}"
                )
                raise RuntimeError(f"Script execution failed: {error_detail}")
        except Exception as e:
            raise RuntimeError(f"Failed to execute flashcard: {str(e)}")
    
    def _decode_output(self, output_bytes: bytes) -> str:
        if not output_bytes:
            return ""
        for codec in ["utf-8", "gbk", "gb2312", "latin-1"]:
            try:
                return output_bytes.decode(codec)
            except UnicodeDecodeError:
                continue
        return output_bytes.decode("latin-1")
    
    def create_flashcard_data(self, word: str, phonetic: str, pos: str, 
                              definition: str, examples: List[Dict[str, str]],
                              synonyms: List[str]) -> str:
        skill = self.get_skill("flash-card")
        if not skill:
            raise RuntimeError("flash-card skill not found")
        
        data_dir = skill.path / "data"
        data_dir.mkdir(exist_ok=True)
        
        data = {
            "word": word,
            "phonetic": phonetic,
            "pos": pos,
            "definition": definition,
            "examples": examples,
            "synonyms": synonyms
        }
        
        data_file = data_dir / f"{word.lower()}.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return str(data_file)
    
    def list_flashcard_words(self) -> List[str]:
        skill = self.get_skill("flash-card")
        if not skill:
            return []
        
        data_dir = skill.path / "data"
        if not data_dir.exists():
            return []
        
        words = []
        for json_file in data_dir.glob("*.json"):
            words.append(json_file.stem)
        
        return sorted(words)
    
    async def ensure_flashcard_ready(self) -> Skill:
        if "flash-card" not in self._loaded_skills:
            await self.load_skill("flash-card")
        
        return self.get_skill("flash-card")
    
    def __len__(self) -> int:
        return len(self._loaded_skills)
    
    def __contains__(self, name: str) -> bool:
        return self.has_skill(name)
    
    def __getitem__(self, name: str) -> Skill:
        skill = self.get_skill(name)
        if skill is None:
            raise KeyError(f"Skill '{name}' not found")
        return skill
