import os
from typing import Optional

from openai import OpenAI

# ---- Provider 配置 ----

PROVIDERS: dict[str, dict] = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "chat_model": "deepseek-v4-flash",
        "display_name": "DeepSeek V4 Flash",
    },
    "qwen": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_model": "qwen-plus",
        "display_name": "Qwen Plus (DashScope)",
    },
}


class LLMClient:
    """通用 LLM 客户端，不绑定任何业务逻辑。"""

    def __init__(self, provider: str = "deepseek"):
        if provider not in PROVIDERS:
            raise ValueError(f"不支持的 provider: {provider}，可选: {list(PROVIDERS.keys())}")

        cfg = PROVIDERS[provider]
        api_key = os.getenv(cfg["api_key_env"], "")
        if not api_key:
            raise ValueError(
                f"请设置环境变量 {cfg['api_key_env']}（provider: {cfg['display_name']}）"
            )

        self.client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
        self.model = cfg["chat_model"]
        self.provider = provider
        self.display_name = cfg["display_name"]

    def chat(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> Optional[str]:
        """
        发送 prompt 到 LLM，返回原始文本。

        prompt 内容由调用方（skill）决定，LLM 只负责执行。
        失败返回 None。
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[LLM::{self.provider}] 调用失败: {e}")
            return None
