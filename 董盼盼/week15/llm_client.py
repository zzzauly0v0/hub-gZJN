"""
统一 LLM 客户端封装（DashScope / Qwen，OpenAI 兼容协议）

教学重点：
  1. 复用同一个 OpenAI 兼容客户端，5 个子 Agent + 主 Agent 共享
  2. enable_search=True 让 Qwen 联网搜索实时信息（天气/景点/票价/航班）
  3. llm_json() 统一处理模型返回中的 ```json``` 代码块，解析失败时回退
  4. 未配置 DASHSCOPE_API_KEY 时返回 None，调用方走 Mock 降级路径

环境变量：
  DASHSCOPE_API_KEY  必填（不填则使用 Mock 数据演示）
  AGENT_MODEL        默认 qwen-max
"""

import os
import re
import json
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("AGENT_MODEL", "qwen-max")
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_client = None


def get_client():
    """懒加载 OpenAI 兼容客户端；未配置 API Key 时返回 None"""
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.warning("未设置 DASHSCOPE_API_KEY，将使用 Mock 数据演示")
        return None
    _client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE)
    return _client


def llm_chat(system_prompt, user_prompt, model=None,
             enable_search=True, temperature=0.3, max_tokens=2000):
    """调用 LLM 返回纯文本；失败或未配置时返回 None"""
    client = get_client()
    if client is None:
        return None
    try:
        kwargs = {
            "model": model or DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if enable_search:
            # Qwen 联网搜索：通过 extra_body 透传给 DashScope
            kwargs["extra_body"] = {"enable_search": True}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"LLM 调用失败: {e}")
        return None


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def llm_json(system_prompt, user_prompt, **kwargs):
    """
    调用 LLM 并解析为 JSON（dict 或 list）。
    支持 ```json``` 包裹和裸 JSON 两种返回格式；解析失败返回 None。
    """
    text = llm_chat(system_prompt, user_prompt, **kwargs)
    if text is None:
        return None
    # 1. 先尝试提取代码块
    m = _JSON_BLOCK_RE.search(text)
    candidate = m.group(1) if m else text
    # 2. 直接解析
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # 3. 兜底：截取第一个 {...} 或 [...]
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pattern, candidate)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    logger.warning(f"JSON 解析失败，原始返回前200字: {text[:200]}")
    return None
