"""
LLM客户端封装
支持OpenAI兼容的API（DeepSeek/通义千问等）
"""
import os
import json
from typing import List, Dict, Optional
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
DEFAULT_MODEL = "deepseek-chat"  # 或其他兼容模型
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"  # DeepSeek API

# 从环境变量读取Key
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "api_key")


class LLMClient:
    """LLM客户端（单例模式）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=DEFAULT_BASE_URL,
            timeout=30
        )
        self._initialized = True
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: List[Dict] = None,
        tool_choice: str = "auto"
    ) -> Dict:
        """
        调用LLM
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            model: 模型名称
            temperature: 温度
            max_tokens: 最大token数
            tools: 工具定义列表
            tool_choice: 工具选择策略
        
        Returns:
            {
                "content": str,  # 文本回复
                "tool_calls": [...],  # 工具调用
                "finish_reason": str
            }
        """
        model = model or DEFAULT_MODEL
        
        try:
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            if tools:
                params["tools"] = tools
                params["tool_choice"] = tool_choice
            
            response = self.client.chat.completions.create(**params)
            message = response.choices[0].message
            
            result = {
                "content": message.content or "",
                "tool_calls": [],
                "finish_reason": response.choices[0].finish_reason
            }
            
            # 解析工具调用
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            return result
            
        except Exception as e:
            print(f"[LLM] 调用失败: {e}")
            return {
                "content": f"LLM调用失败: {e}",
                "tool_calls": [],
                "finish_reason": "error"
            }
    
    def simple_chat(self, prompt: str, system: str = None, model: str = None) -> str:
        """简单对话（无工具）"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self.chat(messages, model=model)
        return result.get("content", "")


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    client = LLMClient()
    resp = client.simple_chat("你好，请介绍一下你自己")
    print(resp)
