"""
对话上下文管理
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class ConversationContext:
    """
    对话上下文
    
    管理多轮对话的历史消息、已加载的 Skill、会话状态
    """
    
    messages: List[Dict[str, Any]] = field(default_factory=list)
    loaded_skills: List[str] = field(default_factory=list)  # 当前会话已加载的 Skill
    session_id: str = ""
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.session_id:
            import uuid
            self.session_id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """添加一条消息"""
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        self.messages.append(msg)
    
    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str, **kwargs) -> None:
        """添加助手消息"""
        self.add_message("assistant", content, **kwargs)
    
    def add_system_message(self, content: str) -> None:
        """添加系统消息"""
        self.add_message("system", content)
    
    def add_skill_context(self, skill_name: str, skill_data: Dict) -> None:
        """添加 Skill 上下文"""
        self.add_message("system", f"[Skill: {skill_name} 已激活]", skill_context=skill_data)
        if skill_name not in self.loaded_skills:
            self.loaded_skills.append(skill_name)
    
    def get_messages(self, include_system: bool = True) -> List[Dict]:
        """获取消息列表"""
        if include_system:
            return self.messages
        return [m for m in self.messages if m.get("role") != "system"]
    
    def get_history(self, last_n: int = 10) -> List[Dict]:
        """获取最近 n 条消息"""
        return self.messages[-last_n:]
    
    def clear(self) -> None:
        """清空上下文"""
        self.messages = []
        self.loaded_skills = []
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "messages_count": len(self.messages),
            "loaded_skills": self.loaded_skills,
            "messages": self.messages,
        }
