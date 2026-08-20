"""百度热搜抓取工具

使用百度热搜 JSON API 接口获取实时热搜数据，比 HTML 解析更稳定。
API: https://top.baidu.com/api/board?tab=realtime
"""

import httpx

from baidu_hotspot_agent.config import config


def _parse_hotspot_response(data: dict, limit: int) -> list[dict[str, str]]:
    """解析百度热搜 API 响应数据"""
    cards = data.get("data", {}).get("cards", [])
    if not cards:
        raise ValueError("百度热搜 API 返回数据异常: 未找到 cards 字段")

    content_list = cards[0].get("content", [])
    if not content_list:
        raise ValueError("百度热搜 API 返回空列表")

    items = []
    for entry in content_list[:limit]:
        item = {
            "title": entry.get("word", entry.get("query", "")),
            "hot_score": str(entry.get("hotScore", entry.get("desc", "0"))),
            "url": entry.get("url", entry.get("rawUrl", "")),
            "desc": entry.get("desc", ""),
        }
        if item["title"]:
            items.append(item)

    return items


async def async_fetch_baidu_hotspot(limit: int | None = None) -> list[dict[str, str]]:
    """异步抓取百度实时热搜列表（推荐在 async 节点中使用）"""
    if limit is None:
        limit = config.default_limit

    headers = {
        "User-Agent": config.user_agent,
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://top.baidu.com/board?tab=realtime",
    }

    async with httpx.AsyncClient(timeout=config.request_timeout) as client:
        response = await client.get(config.baidu_hotspot_url, headers=headers)
        response.raise_for_status()

    return _parse_hotspot_response(response.json(), limit)


def fetch_baidu_hotspot(limit: int | None = None) -> list[dict[str, str]]:
    """同步抓取百度实时热搜列表

    Args:
        limit: 返回条目数量上限，默认使用 config.default_limit

    Returns:
        热搜条目列表，每条包含 title、hot_score、url、desc

    Raises:
        httpx.HTTPStatusError: HTTP 请求返回非 200 状态码
        httpx.TimeoutException: 请求超时
        ValueError: API 返回数据格式异常
    """
    if limit is None:
        limit = config.default_limit

    headers = {
        "User-Agent": config.user_agent,
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://top.baidu.com/board?tab=realtime",
    }

    with httpx.Client(timeout=config.request_timeout) as client:
        response = client.get(config.baidu_hotspot_url, headers=headers)
        response.raise_for_status()

    return _parse_hotspot_response(response.json(), limit)
