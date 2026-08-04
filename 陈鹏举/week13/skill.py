"""
技能系统核心：Registry、Loader、Executor
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)

# 技能目录
SKILLS_DIR = Path(__file__).parent.parent / "skills"


class Skill:
    """技能接口"""
    name: str
    description: str
    triggers: List[str]       # 关键词触发（可选）
    execute: Callable         # 异步函数 async def execute(ctx, ...)

    def __init__(self, name, description, triggers, execute):
        self.name = name
        self.description = description
        self.triggers = triggers
        self.execute = execute


class SkillRegistry:
    """技能注册表（启动时加载元数据）"""
    def __init__(self):
        self._skills: Dict[str, Skill] = {}   # name -> Skill
        self._load_metadata()

    def _load_metadata(self):
        """扫描 skills/ 目录，导入模块但仅提取元数据"""
        if not SKILLS_DIR.exists():
            return
        for py_file in SKILLS_DIR.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # 查找模块中定义的 Skill 实例或函数
                for attr in dir(module):
                    obj = getattr(module, attr)
                    if isinstance(obj, Skill):
                        self._skills[obj.name] = obj
                    elif callable(obj) and hasattr(obj, "skill_meta"):
                        # 支持装饰器注册，见后面
                        meta = getattr(obj, "skill_meta")
                        skill = Skill(
                            name=meta["name"],
                            description=meta["description"],
                            triggers=meta.get("triggers", []),
                            execute=obj
                        )
                        self._skills[skill.name] = skill
            except Exception as e:
                logger.warning(f"加载技能 {py_file.name} 失败: {e}")

    def get_all(self) -> List[Skill]:
        return list(self._skills.values())

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def match(self, text: str) -> Optional[Skill]:
        """基于触发词匹配（简单关键词）"""
        for skill in self._skills.values():
            for trigger in skill.triggers:
                if trigger in text:
                    return skill
        return None


# ── 装饰器辅助（供技能作者使用）─────────────────────────────────────────────

def skill(name: str, description: str, triggers: List[str] = None):
    """装饰器，将普通函数注册为技能"""
    def decorator(func):
        func.skill_meta = {
            "name": name,
            "description": description,
            "triggers": triggers or []
        }
        return func
    return decorator


class SkillLoader:
    """按需加载技能模块（若未加载，则动态 import）"""
    _loaded_modules = {}

    @staticmethod
    def load(name: str) -> Optional[Skill]:
        """加载技能（若元数据已存在，直接从注册表返回 Skill 对象）"""
        registry = get_registry()
        skill = registry.get(name)
        if not skill:
            return None
        # 若 skill 的 execute 已经是函数（非类实例），直接返回
        # 若需要动态导入完整模块，可在此处理（但 registry 已经导入过）
        return skill


# 单例
_registry = None

def get_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry
