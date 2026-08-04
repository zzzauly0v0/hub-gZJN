# =============================================================================
# 文件: skill.py
# 意图: Skill 数据模型定义
# 功能: 定义 Skill 的数据结构、状态枚举和回调机制
#   - SkillStatus:  Skill 加载状态枚举（DISCOVERED/LOADING/LOADED/READY/FAILED）
#   - Skill:        Skill 数据类，包含名称、路径、元数据、文件列表、加载进度
#                  支持三种回调：加载进度、加载完成、加载失败
# 依赖: dataclasses, pathlib, typing, enum
# =============================================================================

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from enum import Enum


class SkillStatus(Enum):
    DISCOVERED = "discovered"
    LOADING = "loading"
    LOADED = "loaded"
    READY = "ready"
    FAILED = "failed"


@dataclass
class Skill:
    name: str
    path: Path
    status: SkillStatus = SkillStatus.DISCOVERED
    metadata: Dict[str, Any] = field(default_factory=dict)
    data_files: List[Path] = field(default_factory=list)
    script_files: List[Path] = field(default_factory=list)
    load_progress: float = 0.0
    load_error: Optional[str] = None
    
    _on_load_progress: Optional[Callable[[float], None]] = None
    _on_load_complete: Optional[Callable[[], None]] = None
    _on_load_failed: Optional[Callable[[str], None]] = None
    
    def set_progress_callback(self, callback: Callable[[float], None]) -> 'Skill':
        self._on_load_progress = callback
        return self
    
    def set_complete_callback(self, callback: Callable[[], None]) -> 'Skill':
        self._on_load_complete = callback
        return self
    
    def set_failed_callback(self, callback: Callable[[str], None]) -> 'Skill':
        self._on_load_failed = callback
        return self
    
    def update_progress(self, progress: float) -> None:
        self.load_progress = progress
        if self._on_load_progress:
            self._on_load_progress(progress)
    
    def mark_loaded(self) -> None:
        self.status = SkillStatus.LOADED
        self.load_progress = 1.0
        if self._on_load_complete:
            self._on_load_complete()
    
    def mark_failed(self, error: str) -> None:
        self.status = SkillStatus.FAILED
        self.load_error = error
        if self._on_load_failed:
            self._on_load_failed(error)
    
    def is_ready(self) -> bool:
        return self.status == SkillStatus.READY
    
    def __repr__(self) -> str:
        return f"<Skill(name='{self.name}', status='{self.status.value}', progress={self.load_progress:.1%})>"
