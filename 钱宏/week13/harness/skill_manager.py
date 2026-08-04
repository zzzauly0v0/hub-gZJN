"""
Skill 管理器 - 支持渐进式加载

核心功能：
1. 注册 Skill（只注册元数据，不加载完整功能）
2. 根据用户输入匹配相关 Skill
3. 按需加载 Skill 完整功能（渐进式加载）
4. 管理 Skill 生命周期（加载/卸载）
"""

import os
import importlib
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .skill import Skill, SkillMetadata, SkillState, create_skill_from_config, load_skill_from_markdown

logger = logging.getLogger(__name__)


class SkillManager:
    """
    Skill 管理器
    
    渐进式加载策略：
    1. 启动时只扫描并注册 Skill 元数据（轻量）
    2. 用户提问时，根据关键词匹配可能相关的 Skill
    3. 只加载被选中的 Skill（加载 executor、tools、prompt）
    4. 可配置自动卸载超时，释放资源
    """
    
    def __init__(self, skills_dir: str = "skills", auto_unload_seconds: int = 300):
        """
        初始化 Skill 管理器
        
        Args:
            skills_dir: Skills 目录路径
            auto_unload_seconds: 自动卸载时间（秒），0 表示不自动卸载
        """
        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, Skill] = {}
        self.keyword_index: Dict[str, List[str]] = {}  # 关键词 -> skill names
        self.auto_unload_seconds = auto_unload_seconds
        
        # 初始化时只注册元数据
        self._discover_skills()
    
    def _discover_skills(self) -> None:
        """
        扫描并注册所有 Skill（只注册元数据）
        """
        if not self.skills_dir.exists():
            logger.warning(f"Skills 目录不存在: {self.skills_dir}")
            return
        
        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                logger.debug(f"跳过无 SKILL.md 的目录: {skill_dir}")
                continue
            
            try:
                # 读取 SKILL.md内容
                config = load_skill_from_markdown(str(skill_md))
                metadata = SkillMetadata(
                    name=config.get("name", skill_dir.name),
                    description=config.get("description", ""),
                    keywords=config.get("keywords", []),
                    version=config.get("version", "1.0.0"),
                    author=config.get("author", ""),
                    tags=config.get("tags", []),
                    priority=config.get("priority", 0),
                )
                
                skill = Skill(metadata=metadata)
                self.skills[skill.name] = skill
                
                # 建立关键词索引
                for keyword in metadata.keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower not in self.keyword_index:
                        self.keyword_index[keyword_lower] = []
                    self.keyword_index[keyword_lower].append(skill.name)
                
                logger.info(f"注册 Skill: {skill.name} (状态: {skill.state.value})")
                
            except Exception as e:
                logger.error(f"注册 Skill 失败 {skill_dir}: {e}")
    
    def register_skill(self, skill: Skill) -> None:
        """手动注册一个 Skill"""
        self.skills[skill.name] = skill
        
        for keyword in skill.metadata.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in self.keyword_index:
                self.keyword_index[keyword_lower] = []
            self.keyword_index[keyword_lower].append(skill.name)
    
    def match_skills(self, query: str, top_k: int = 3) -> List[Tuple[Skill, float]]:
        """
        根据用户查询匹配相关 Skill
        
        使用关键词匹配 + 评分策略：
        - 关键词完全匹配：+1.0 分
        - 关键词部分匹配：+0.5 分
        - 标签匹配：+0.3 分
        - 优先级加成：+priority * 0.1
        
        Args:
            query: 用户查询文本
            top_k: 返回前 k 个结果
        
        Returns:
            [(Skill, score), ...] 按分数降序排列
        """
        query_lower = query.lower()
        scores: Dict[str, float] = {}
        
        # 关键词匹配
        for keyword, skill_names in self.keyword_index.items():
            if keyword in query_lower:
                for skill_name in skill_names:
                    scores[skill_name] = scores.get(skill_name, 0) + 1.0
        
        # 标签匹配
        for name, skill in self.skills.items():
            for tag in skill.metadata.tags:
                if tag.lower() in query_lower:
                    scores[name] = scores.get(name, 0) + 0.3
        
        # 描述匹配（简单包含匹配）
        for name, skill in self.skills.items():
            desc_lower = skill.metadata.description.lower()
            words = query_lower.split()
            match_count = sum(1 for w in words if w in desc_lower)
            if match_count > 0:
                scores[name] = scores.get(name, 0) + match_count * 0.1
        
        # 优先级加成
        for name in scores:
            skill = self.skills.get(name)
            if skill:
                scores[name] += skill.metadata.priority * 0.1
        
        # 排序并返回
        sorted_skills = sorted(
            [(self.skills[name], score) for name, score in scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        
        return sorted_skills[:top_k]
    
    def load_skill(self, skill_name: str) -> Optional[Skill]:
        """
        加载指定 Skill（渐进式加载）
        
        加载步骤：
        1. 查找 Skill 配置
        2. 尝试加载 skill.py 中的执行器
        3. 设置为 LOADED 状态
        
        Args:
            skill_name: Skill 名称
        
        Returns:
            加载后的 Skill 或 None
        """
        skill = self.skills.get(skill_name)
        if not skill:
            logger.error(f"Skill 不存在: {skill_name}")
            return None
        
        if skill.is_loaded:
            return skill
        
        skill.load()
        
        try:
            # 尝试加载 skill.py 模块
            skill_dir = self.skills_dir / skill_name
            skill_py = skill_dir / "skill.py"
            
            if skill_py.exists():
                module_path = f"skills.{skill_name}.skill"
                try:
                    module = importlib.import_module(module_path)
                    
                    # 查找 create_skill 函数
                    if hasattr(module, "create_skill"):
                        loaded_skill = module.create_skill()
                        if isinstance(loaded_skill, dict):
                            skill.tools = loaded_skill.get("tools", [])
                            skill.system_prompt = loaded_skill.get("system_prompt", "")
                            skill.executor = loaded_skill.get("executor")
                        elif callable(loaded_skill):
                            skill.executor = loaded_skill
                except ImportError as e:
                    logger.warning(f"无法导入 Skill 模块 {module_path}: {e}")
            
            # 从 SKILL.md 加载系统提示词
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                config = load_skill_from_markdown(str(skill_md))
                if not skill.system_prompt:
                    skill.system_prompt = config.get("markdown_content", "")
                if not skill.tools:
                    skill.tools = config.get("tools", [])
            
            skill.state = SkillState.LOADED
            logger.info(f"Skill 加载完成: {skill_name}")
            
        except Exception as e:
            skill.state = SkillState.ERROR
            logger.error(f"Skill 加载失败 {skill_name}: {e}")
            return None
        
        return skill
    
    def load_matched_skills(self, query: str, top_k: int = 3) -> List[Skill]:
        """
        匹配并加载相关 Skill
        
        Args:
            query: 用户查询
            top_k: 最大加载数量
        
        Returns:
            已加载的 Skill 列表
        """
        matched = self.match_skills(query, top_k)
        loaded_skills = []
        
        for skill, score in matched:
            loaded = self.load_skill(skill.name)
            if loaded:
                loaded_skills.append(loaded)
        
        return loaded_skills
    
    def unload_skill(self, skill_name: str) -> bool:
        """卸载指定 Skill"""
        skill = self.skills.get(skill_name)
        if not skill:
            return False
        skill.unload()
        logger.info(f"Skill 已卸载: {skill_name}")
        return True
    
    def unload_all(self) -> None:
        """卸载所有已加载的 Skill"""
        for skill in self.skills.values():
            if skill.is_loaded:
                skill.unload()
    
    def get_skill(self, skill_name: str) -> Optional[Skill]:
        """获取 Skill（不加载）"""
        return self.skills.get(skill_name)
    
    def list_skills(self) -> List[Dict]:
        """列出所有 Skill 元数据"""
        return [skill.to_dict() for skill in self.skills.values()]
    
    def get_loaded_skills(self) -> List[Skill]:
        """获取所有已加载的 Skill"""
        return [s for s in self.skills.values() if s.is_loaded]
    
    def get_status(self) -> Dict:
        """获取管理器状态"""
        total = len(self.skills)
        loaded = len(self.get_loaded_skills())
        return {
            "total_skills": total,
            "loaded_skills": loaded,
            "unloaded_skills": total - loaded,
            "skills": {
                name: skill.state.value
                for name, skill in self.skills.items()
            }
        }
