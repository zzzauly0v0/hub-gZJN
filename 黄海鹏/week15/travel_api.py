"""
旅行相关API封装
只用高德地图（一个Key搞定所有）
"""
import requests
import json
import random
from typing import List, Dict, Optional

# ============================================================
# 配置（替换成你自己的高德Key）
# ============================================================
AMAP_KEY = os.environ.get("AMAP_API_KEY", "api_key") # 去 https://lbs.amap.com/ 注册

# ============================================================
# 通用请求函数
# ============================================================
def _amap_request(url: str, params: dict) -> dict:
    """发送高德API请求"""
    params["key"] = AMAP_KEY
    try:
        resp = requests.get(url, params=params, timeout=5)
        return resp.json()
    except Exception as e:
        print(f"[高德API] 请求失败: {e}")
        return {"status": "0"}


# ============================================================
# 1. 景点搜索（关键字搜索API）
# ============================================================
def search_attractions(city: str, limit: int = 8) -> List[Dict]:
    """搜索城市景点"""
    url = "https://restapi.amap.com/v3/place/text"
    params = {
        "keywords": "景点",
        "city": city,
        "citylimit": "true",  # 只返回指定城市
        "offset": limit,
        "extensions": "all"
    }
    
    data = _amap_request(url, params)
    
    if data.get("status") == "1" and data.get("pois"):
        return [
            {
                "name": p["name"],
                "address": p.get("pname", ""),
                "type": p.get("type", ""),
                "rating": p.get("biz_ext", {}).get("rating", "暂无"),
                "cost": p.get("biz_ext", {}).get("cost", "暂无")
            }
            for p in data["pois"][:limit]
        ]
    
    return _mock_attractions(city)


def _mock_attractions(city: str) -> List[Dict]:
    """降级方案：本地数据"""
    mock = {
        "北京": [
            {"name": "故宫博物院", "address": "东城区", "type": "历史建筑", "rating": "4.9", "cost": "60元"},
            {"name": "八达岭长城", "address": "延庆区", "type": "历史遗迹", "rating": "4.8", "cost": "40元"},
            {"name": "天安门广场", "address": "东城区", "type": "广场", "rating": "4.7", "cost": "免费"},
            {"name": "颐和园", "address": "海淀区", "type": "皇家园林", "rating": "4.8", "cost": "30元"},
        ],
        "上海": [
            {"name": "外滩", "address": "黄浦区", "type": "江景", "rating": "4.9", "cost": "免费"},
            {"name": "东方明珠塔", "address": "浦东新区", "type": "地标", "rating": "4.7", "cost": "199元"},
            {"name": "豫园", "address": "黄浦区", "type": "古典园林", "rating": "4.6", "cost": "40元"},
        ],
        "成都": [
            {"name": "宽窄巷子", "address": "青羊区", "type": "历史文化街区", "rating": "4.7", "cost": "免费"},
            {"name": "大熊猫基地", "address": "成华区", "type": "动物园", "rating": "4.9", "cost": "55元"},
            {"name": "锦里古街", "address": "武侯区", "type": "古街", "rating": "4.6", "cost": "免费"},
        ]
    }
    return mock.get(city, [
        {"name": f"{city}著名景点{i+1}", "address": city, "type": "景点", "rating": "4.5", "cost": "免费"}
        for i in range(5)
    ])


# ============================================================
# 2. 天气查询（天气查询API）⭐ 高德自带
# ============================================================
def get_weather(city: str, days: int = 3) -> List[Dict]:
    """
    查询天气（高德天气API）
    返回未来3天天气预报
    """
    # 先获取城市adcode
    city_code = _get_city_code(city)
    if not city_code:
        return _mock_weather(city, days)
    
    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {
        "city": city_code,
        "extensions": "all"  # all=预报天气
    }
    
    data = _amap_request(url, params)
    
    if data.get("status") == "1" and data.get("forecasts"):
        forecast = data["forecasts"][0]
        result = []
        for i, cast in enumerate(forecast.get("casts", [])[:days]):
            result.append({
                "date": cast.get("date", ""),
                "weather": cast.get("dayweather", ""),
                "temp_max": cast.get("daytemp_high", ""),
                "temp_min": cast.get("nighttemp_low", ""),
                "wind": cast.get("daywind", ""),
            })
        return result
    
    return _mock_weather(city, days)


def _get_city_code(city: str) -> str:
    """获取城市adcode（行政区划代码）"""
    url = "https://restapi.amap.com/v3/config/district"
    params = {
        "keywords": city,
        "subdistrict": 0,
        "extensions": "base"
    }
    data = _amap_request(url, params)
    
    if data.get("status") == "1" and data.get("districts"):
        return data["districts"][0].get("adcode", "")
    return ""


def _mock_weather(city: str, days: int) -> List[Dict]:
    """降级方案：本地天气数据"""
    weathers = ["晴", "多云", "阴", "小雨"]
    result = []
    for i in range(days):
        w = random.choice(weathers)
        result.append({
            "date": f"2026-08-{12+i}",
            "weather": w,
            "temp_max": str(random.randint(25, 35)),
            "temp_min": str(random.randint(15, 25)),
            "wind": random.choice(["南风", "北风", "东风", "无持续风向"])
        })
    return result


# ============================================================
# 3. 美食搜索（关键字搜索API）
# ============================================================
def search_food(city: str, limit: int = 6) -> List[Dict]:
    """搜索城市美食"""
    url = "https://restapi.amap.com/v3/place/text"
    params = {
        "keywords": "美食",
        "city": city,
        "citylimit": "true",
        "offset": limit,
        "extensions": "all"
    }
    
    data = _amap_request(url, params)
    
    if data.get("status") == "1" and data.get("pois"):
        return [
            {
                "name": p["name"],
                "address": p.get("pname", ""),
                "type": p.get("type", ""),
                "rating": p.get("biz_ext", {}).get("rating", "暂无"),
                "cost": p.get("biz_ext", {}).get("cost", "暂无")
            }
            for p in data["pois"][:limit]
        ]
    
    return _mock_food(city)


def _mock_food(city: str) -> List[Dict]:
    """降级方案：本地美食数据"""
    mock = {
        "北京": [
            {"name": "全聚德烤鸭店", "address": "东城区", "type": "北京菜", "rating": "4.8", "cost": "200元"},
            {"name": "老北京炸酱面", "address": "西城区", "type": "面食", "rating": "4.5", "cost": "50元"},
        ],
        "上海": [
            {"name": "南翔馒头店", "address": "黄浦区", "type": "小吃", "rating": "4.6", "cost": "30元"},
            {"name": "老正兴菜馆", "address": "黄浦区", "type": "本帮菜", "rating": "4.5", "cost": "120元"},
        ],
        "成都": [
            {"name": "小龙坎火锅", "address": "锦江区", "type": "火锅", "rating": "4.8", "cost": "150元"},
            {"name": "陈麻婆豆腐", "address": "青羊区", "type": "川菜", "rating": "4.6", "cost": "80元"},
        ]
    }
    return mock.get(city, [
        {"name": f"{city}特色餐厅{i+1}", "address": city, "type": "本地菜", "rating": "4.3", "cost": "100元"}
        for i in range(3)
    ])


# ============================================================
# 统一格式化函数
# ============================================================
def format_attractions(result: List[Dict]) -> str:
    if not result:
        return "未找到景点信息"
    lines = ["📸 推荐景点："]
    for i, item in enumerate(result, 1):
        lines.append(f"  {i}. {item['name']} | {item['address']} | 评分:{item['rating']} | 费用:{item['cost']}")
    return "\n".join(lines)


def format_weather(result: List[Dict]) -> str:
    if not result:
        return "未找到天气信息"
    lines = ["🌤 天气预报："]
    for item in result:
        lines.append(f"  {item['date']}: {item['weather']} | {item['temp_min']}~{item['temp_max']}℃ | {item['wind']}")
    return "\n".join(lines)


def format_food(result: List[Dict]) -> str:
    if not result:
        return "未找到美食信息"
    lines = ["🍜 推荐美食："]
    for i, item in enumerate(result, 1):
        lines.append(f"  {i}. {item['name']} | {item['address']} | 评分:{item['rating']} | 人均:{item['cost']}")
    return "\n".join(lines)
