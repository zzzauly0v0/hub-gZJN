"""异步 LLM 客户端（AsyncOpenAI，DeepSeek 兼容接口）。"""
from __future__ import annotations
import os
import asyncio
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise EnvironmentError("请设置 DEEPSEEK_API_KEY")
        _client = AsyncOpenAI(api_key=key, base_url=DEEPSEEK_URL)
    return _client


async def llm_chat(system, user, *, temperature=0.0, max_tokens=1024,
                   stop=None, retries=3) -> str:
    for attempt in range(retries):
        try:
            resp = await get_client().chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
            logger.warning(f"LLM 异步重试({attempt + 1}): {str(e)[:80]}")


if __name__ == "__main__":
    import asyncio as _a

    async def _t():
        r = await llm_chat("你是助手", "用一句话介绍东京")
        print(r[:120])

    _a.run(_t())
