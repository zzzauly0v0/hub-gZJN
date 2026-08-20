"""Tavily 联网搜索异步封装（标准库 urllib + asyncio.to_thread，零额外依赖）。"""
import os
import json
import asyncio
import urllib.request
import logging

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"


def _tavily_search_sync(query: str, max_results: int = 5) -> dict:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return {"error": "未设置 TAVILY_API_KEY"}
    payload = {
        "api_key": key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": True,
    }
    try:
        req = urllib.request.Request(
            TAVILY_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = [{"title": r.get("title", ""), "url": r.get("url", ""),
                    "content": (r.get("content") or "")[:600]}
                   for r in data.get("results", [])]
        return {"answer": data.get("answer") or "",
                "results": results,
                "response_time": data.get("response_time")}
    except Exception as e:
        logger.warning(f"Tavily 搜索失败 '{query}': {e}")
        return {"error": f"{type(e).__name__}: {str(e)[:100]}"}


async def tavily_search(query: str, max_results: int = 5) -> dict:
    return await asyncio.to_thread(_tavily_search_sync, query, max_results)


def format_search_result(r: dict) -> str:
    if "error" in r:
        return f"搜索失败: {r['error']}"
    parts = []
    if r.get("answer"):
        parts.append(f"摘要: {r['answer']}")
    for i, res in enumerate(r.get("results", []), 1):
        parts.append(f"[{i}] {res['title']}\n    {res['content'][:300]}")
    return "\n".join(parts) if parts else "无结果"


if __name__ == "__main__":
    import asyncio as _a

    async def _t():
        r = await tavily_search("东京必去景点 2024")
        print(format_search_result(r)[:400])

    _a.run(_t())
