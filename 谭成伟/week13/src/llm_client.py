"""
LLM 客户端 — 大模型连接配置
=============================
参考 agent_memory_system/src/llm_config.py 的写法，
使用 OpenAI 兼容接口连接大模型。

切换方式（环境变量）：
  LLM_PROVIDER=qwen        # 默认，使用 DashScope qwen-plus
  LLM_PROVIDER=deepseek    # 使用 DeepSeek

对应 API Key：
  DASHSCOPE_API_KEY=sk-xxx   （qwen）
  DEEPSEEK_API_KEY=sk-xxx    （deepseek）
"""

import os
from openai import OpenAI

# ── 支持的模型提供商 ──────────────────────────────────────────
PROVIDERS = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "qwen": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


def get_client() -> tuple[OpenAI, str]:
    """
    获取 LLM 客户端和模型名称。
    由环境变量 LLM_PROVIDER 决定使用哪个提供商（默认 qwen）。
    """
    provider = os.getenv("LLM_PROVIDER", "qwen").lower()
    if provider not in PROVIDERS:
        raise ValueError(f"未知 LLM_PROVIDER='{provider}'，可选: {list(PROVIDERS)}")

    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        raise EnvironmentError(f"请设置环境变量 {cfg['api_key_env']}")

    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    return client, cfg["model"]
