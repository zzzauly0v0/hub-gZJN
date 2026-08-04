"""
Skill 定义模块

Skill（技能）是 Agent 可调用的能力单元，包含：
- 元数据：名称、描述、触发关键词
- 加载状态：未加载、已加载
- 执行器：实际执行技能的函数
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import json


class SkillState(Enum):
    """Skill 加载状态"""
    UNLOADED = "unloaded"      # 未加载（只知道元数据）
    LOADED = "loaded"          # 已加载（完整功能可用）
    LOADING = "loading"        # 正在加载
    ERROR = "error"            # 加载出错


@dataclass
class SkillMetadata:
    """Skill 元数据（轻量级，用于注册和匹配）"""
    name: str                           # 技能名称（唯一标识）
    description: str                    # 技能描述
    keywords: List[str] = field(default_factory=list)  # 触发关键词
    version: str = "1.0.0"             # 版本号
    author: str = ""                    # 作者
    tags: List[str] = field(default_factory=list)    # 标签
    priority: int = 0                   # 优先级（数字越大越优先）
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "keywords": self.keywords,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "priority": self.priority,
        }


@dataclass
class Skill:
    """完整 Skill 定义"""
    metadata: SkillMetadata
    state: SkillState = SkillState.UNLOADED
    executor: Optional[Callable] = None  # 执行函数
    tools: List[Dict] = field(default_factory=list)  # 可用工具列表
    system_prompt: str = ""              # 系统提示词
    
    @property
    def name(self) -> str:
        return self.metadata.name
    
    @property
    def is_loaded(self) -> bool:
        return self.state == SkillState.LOADED
    
    def load(self) -> None:
        """加载 Skill（实际加载 executor、tools、prompt）"""
        self.state = SkillState.LOADING
    
    def unload(self) -> None:
        """卸载 Skill（释放资源）"""
        self.state = SkillState.UNLOADED
        self.executor = None
        self.tools = []
        self.system_prompt = ""
    
    def execute(self, **kwargs) -> Any:
        """执行 Skill"""
        if not self.is_loaded or not self.executor:
            raise RuntimeError(f"Skill '{self.name}' 未加载，无法执行")
        return self.executor(**kwargs)
    
    def to_dict(self) -> Dict:
        return {
            "metadata": self.metadata.to_dict(),
            "state": self.state.value,
            "is_loaded": self.is_loaded,
            "tools_count": len(self.tools),
        }


def create_skill_from_config(config: Dict, executor: Optional[Callable] = None) -> Skill:
    """
    从配置字典创建 Skill
    
    配置格式：
    {
        "name": "skill_name",
        "description": "技能描述",
        "keywords": ["关键词1", "关键词2"],
        "tools": [...],
        "system_prompt": "...",
    }
    """
    metadata = SkillMetadata(
        name=config["name"],
        description=config["description"],
        keywords=config.get("keywords", []),
        version=config.get("version", "1.0.0"),
        author=config.get("author", ""),
        tags=config.get("tags", []),
        priority=config.get("priority", 0),
    )
    
    skill = Skill(
        metadata=metadata,
        state=SkillState.LOADED if executor else SkillState.UNLOADED,
        executor=executor,
        tools=config.get("tools", []),
        system_prompt=config.get("system_prompt", ""),
    )
    
    return skill


def load_skill_from_markdown(md_path: str) -> Dict:
    """
    从 SKILL.md 文件加载 Skill 配置
    
    SKILL.md 格式（Front Matter + Markdown）：
    ---
    name: skill_name
    description: 技能描述
    keywords: [关键词1, 关键词2]
    version: 1.0.0
    ---
    
    # Skill 详细说明
    
    （技能的详细使用说明）
    """
    import re
    
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 解析 Front Matter
    fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if fm_match:
        fm_content = fm_match.group(1)
        config = {}
        
        for line in fm_content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # 处理数组
                if value.startswith('[') and value.endswith(']'):
                    value = [v.strip().strip("'\"") for v in value[1:-1].split(',') if v.strip()]
                elif value:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            value = value.strip("'\"")
                
                config[key] = value
        
        # 附加 markdown 内容
        config['markdown_content'] = content[fm_match.end():]
        return config
    
    # 无 Front Matter，返回基本信息
    return {
        "name": md_path.split('/')[-2] if '/' in md_path else md_path.replace('.md', ''),
        "description": content[:200] if content else "",
        "keywords": [],
        "markdown_content": content,
    }
