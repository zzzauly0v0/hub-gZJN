# =============================================================================
# 文件: __init__.py
# 意图: Harness 模块初始化入口
# 功能: 导出 Harness 框架的核心类，供外部统一调用
#   - Harness:        核心管理器，统一接口发现、加载、执行 Skill
#   - Skill:          Skill 数据模型，描述 Skill 的状态、元数据、回调
#   - SkillRegistry:  Skill 注册表，自动扫描 skills/ 目录并注册所有 Skill
#   - ProgressiveLoader: 渐进式加载器，支持分步加载、进度回调、并发加载
# 用法: from harness import Harness
# =============================================================================

from .harness import Harness
from .skill import Skill
from .skill_registry import SkillRegistry
from .progressive_loader import ProgressiveLoader

__all__ = ['Harness', 'Skill', 'SkillRegistry', 'ProgressiveLoader']
__version__ = '1.0.0'
