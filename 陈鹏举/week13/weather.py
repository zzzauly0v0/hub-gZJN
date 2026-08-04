from src.skill import skill
import asyncio
import random

@skill(
    name="weather",
    description="查询指定城市的天气（模拟）",
    triggers=["天气", "温度", "下雨"]
)
async def weather_skill(ctx: dict, city: str = None) -> str:
    """ctx 包含 session_id, db, loader, llm_client, ..."""
    if not city:
        # 可以从 ctx 或用户消息中提取
        return "请告诉我城市名称。"
    # 模拟天气查询
    await asyncio.sleep(1)
    temp = random.randint(0, 35)
    condition = random.choice(["晴", "多云", "小雨", "阴"])
    return f"{city} 今天 {condition}，气温 {temp}°C。"
