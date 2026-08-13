"""极简异步 LLM 客户端（体育热点分析 subagent 最小单元）

教学重点：主 agent 和子 agent 的每一步推理都是 `await llm_chat(...)`，
所以整条 ReAct 链是协程——N 个子 agent 直接丢给 asyncio.gather 就并发了。

无 DEEPSEEK_API_KEY（或 MOCK=1）时自动降级为离线假 LLM，
保证「下发 subagent + 异步并发」这条链路无 key 也能跑通演示。
"""
import os, re, asyncio
from openai import AsyncOpenAI

MODEL = "deepseek-chat"
_client = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                              base_url="https://api.deepseek.com")
    return _client


def mock_mode() -> bool:
    """没 key 就走离线 mock，不报错（教学演示优先）。"""
    return os.getenv("MOCK") == "1" or not os.getenv("DEEPSEEK_API_KEY")


async def llm_chat(system: str, user: str, *, max_tokens: int = 768, stop=None) -> str:
    """单轮异步对话。stop=["Observation:"] 让 ReAct 在 Action Input 后停下。"""
    if mock_mode():
        return await _mock_chat(system, user)
    resp = await _get_client().chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=max_tokens, stop=stop,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return resp.choices[0].message.content or ""


async def _mock_chat(system: str, user: str) -> str:
    """离线假 LLM：够演示就好——主 agent 首步必派发，子 agent 首步搜一次，之后收尾。"""
    await asyncio.sleep(0.3)                                  # 模拟推理延迟
    question = user.split("Question:")[-1].split("\n")[0].strip()
    if "Observation:" not in user:                            # 第一步：决定动作
        if "dispatch_subagents" in system:                    # 主 agent
            facets = [s.strip() for s in re.split(r"[：:，,、]", question)
                      if len(s.strip()) > 3][1:4]
            return ("Thought: 这个热点有多个侧面，下发子分析员并行调研\n"
                    f"Action: dispatch_subagents\n"
                    f"Action Input: {' | '.join(facets) or question}\n")
        return f"Thought: 先联网搜一次\nAction: web_search\nAction Input: {question}\n"
    obs = user.rsplit("Observation:", 1)[-1].strip()          # 已有观察 → 收尾
    return f"Thought: 资料已够，综合收尾\nFinal Answer: 【mock 报告】{question}\n{obs[:400]}"
