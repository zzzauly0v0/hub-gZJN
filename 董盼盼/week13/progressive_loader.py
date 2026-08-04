# =============================================================================
# 文件: progressive_loader.py
# 意图: 渐进式 Skill 加载器
# 功能: 支持 Skill 的分步渐进式加载，提供进度回调、并发加载、加载顺序控制
#   - load_skill():       加载单个 Skill，分 4 步执行（元数据→数据文件→脚本初始化→完成）
#   - _progressive_load(): 内部方法，按权重分配的步骤执行加载
#   - load_all():         批量加载所有 Skill，支持并发/串行模式
#   - load_batch():       批量加载指定列表的 Skill
#   - set_load_order():   设置加载顺序
#   - prioritize_skill(): 将指定 Skill 优先加载
# 依赖: json, asyncio, pathlib, typing, .skill
# =============================================================================

import json
import subprocess
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from .skill import Skill, SkillStatus


class ProgressiveLoader:
    def __init__(self, registry: 'SkillRegistry'):
        self._registry = registry
        self._loading_tasks: Dict[str, asyncio.Task] = {}
        self._load_order: List[str] = []
    
    def set_load_order(self, order: List[str]) -> 'ProgressiveLoader':
        self._load_order = order
        return self
    
    def prioritize_skill(self, name: str) -> 'ProgressiveLoader':#动态提升技能到最高优先级
        if name in self._load_order:
            self._load_order.remove(name)
        self._load_order.insert(0, name)
        return self
    
    async def load_skill(self, name: str, 
                        on_progress: Optional[Callable[[float], None]] = None,
                        on_complete: Optional[Callable[[Skill], None]] = None,
                        on_failed: Optional[Callable[[str, str], None]] = None) -> Skill:
        skill = self._registry.get_skill(name)
        if not skill:
            raise ValueError(f"Skill '{name}' not found in registry")
        
        if skill.status == SkillStatus.READY:
            if on_complete:
                on_complete(skill)
            return skill
        
        if skill.status == SkillStatus.LOADING:
            while skill.status == SkillStatus.LOADING:
                await asyncio.sleep(0.1)
            return skill
        
        skill.status = SkillStatus.LOADING
        
        try:
            await self._progressive_load(skill, on_progress)
            
            if on_complete:
                on_complete(skill)
            return skill
        
        except Exception as e:
            skill.mark_failed(str(e))
            if on_failed:
                on_failed(name, str(e))
            return skill
    
    async def _progressive_load(self, skill: Skill, 
                               on_progress: Optional[Callable[[float], None]] = None) -> None:
        steps = [
            (20, "Loading metadata", self._load_metadata),
            (30, "Loading data files", self._load_data_files),
            (30, "Initializing scripts", self._init_scripts),
            (20, "Finalizing", self._finalize),
        ]
        
        total_progress = 0
        
        for weight, step_name, step_func in steps:
            if on_progress:
                on_progress(total_progress)
            
            try:
                await step_func(skill)
                total_progress += weight
            except Exception as e:
                raise RuntimeError(f"Failed at '{step_name}': {str(e)}")
        
        if on_progress:
            on_progress(total_progress)
        
        skill.status = SkillStatus.READY
    
    async def _load_metadata(self, skill: Skill) -> None:
        await asyncio.sleep(0.1)
    
    async def _load_data_files(self, skill: Skill) -> None:
        skill_data = {}
        
        for data_file in skill.data_files:
            try:
                if data_file.suffix == ".json":
                    with open(data_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        skill_data[data_file.stem] = data
                elif data_file.suffix in (".yaml", ".yml"):
                    import yaml
                    with open(data_file, "r", encoding="utf-8") as f:
                        skill_data[data_file.stem] = yaml.safe_load(f)
                else:
                    with open(data_file, "r", encoding="utf-8") as f:
                        skill_data[data_file.stem] = f.read()
            except Exception:
                pass
        
        if skill_data:
            skill.metadata["loaded_data"] = skill_data
        
        await asyncio.sleep(0.15)
    
    async def _init_scripts(self, skill: Skill) -> None:
        for script_file in skill.script_files:
            if script_file.suffix == ".py":
                try:
                    await self._validate_python_script(script_file)
                except Exception:
                    pass
        
        await asyncio.sleep(0.1)
    
    async def _validate_python_script(self, script_file: Path) -> None:
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "py_compile", str(script_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.wait()
    
    async def _finalize(self, skill: Skill) -> None:
        
        await asyncio.sleep(0.05)
    
    async def load_all(self, concurrent: bool = True,
                      on_progress: Optional[Callable[[str, float], None]] = None,
                      on_complete: Optional[Callable[[List[Skill]], None]] = None) -> List[Skill]:
        skills_to_load = self._load_order.copy()
        
        for skill in self._registry.iter_skills():
            if skill.name not in skills_to_load:
                skills_to_load.append(skill.name)
        
        if concurrent:
            tasks = []
            for name in skills_to_load:
                task = asyncio.create_task(
                    self.load_skill(
                        name,
                        on_progress=lambda p, n=name: on_progress(n, p) if on_progress else None
                    )
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        else:
            results = []
            for name in skills_to_load:
                skill = await self.load_skill(
                    name,
                    on_progress=lambda p, n=name: on_progress(n, p) if on_progress else None
                )
                results.append(skill)
        
        if on_complete:
            on_complete(results)
        
        return results
    
    async def load_batch(self, names: List[str], concurrent: bool = True) -> List[Skill]:
        if concurrent:
            tasks = [self.load_skill(name) for name in names]
            return await asyncio.gather(*tasks)
        
        results = []
        for name in names:
            results.append(await self.load_skill(name))
        return results
    
    def cancel_load(self, name: str) -> bool:
        task = self._loading_tasks.get(name)
        if task and not task.done():
            task.cancel()
            return True
        return False
    
    def is_loading(self) -> bool:
        return any(not task.done() for task in self._loading_tasks.values())
