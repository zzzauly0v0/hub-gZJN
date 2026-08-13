"""异步联网搜索（Tavily REST，httpx 直调，不引 SDK）

体育热点强时效，必须联网。这里只有一个工具函数 web_search，
子 agent 手里就这一件武器；异步版本让 N 个子 agent 的搜索真正同时发出。

无 TAVILY_API_KEY（或 MOCK=1）时返回假摘要 + sleep 模拟网络耗时，
离线也能看出并发把墙钟从 sum 压到 ≈max。
"""
import os, asyncio, httpx

TAVILY_URL = "https://api.tavily.com/search"


async def web_search(query: str, ctx=None, **_) -> str:
    """ReAct 工具：搜一次，返回喂给 LLM 的纯文本。失败不抛异常，返回错误串让 ReAct 兜底。"""
    key = os.getenv("TAVILY_API_KEY")
    if not key or os.getenv("MOCK") == "1":
        await asyncio.sleep(1.2)                              # 模拟一次搜索的网络耗时
        return (f"摘要: 【mock】关于「{query}」的检索要点：赛况/数据/舆论各一条\n"
                f"[1] mock 来源 example.com/{abs(hash(query)) % 999}")
    try:
        async with httpx.AsyncClient(timeout=30) as cli:
            resp = await cli.post(TAVILY_URL, json={
                "api_key": key, "query": query,
                "max_results": 4, "search_depth": "basic", "include_answer": True})
            data = resp.json()
    except Exception as e:
        return f"搜索失败: {type(e).__name__}: {str(e)[:80]}"
    parts = [f"摘要: {data['answer']}"] if data.get("answer") else []
    parts += [f"[{i}] {r.get('title', '')} ({r.get('url', '')})\n    {(r.get('content') or '')[:220]}"
              for i, r in enumerate(data.get("results", [])[:4], 1)]
    return "\n".join(parts) or "无结果"
