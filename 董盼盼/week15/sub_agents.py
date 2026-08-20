"""
5 个子 Agent：景点 / 住宿 / 交通 / 美食 / 风险

每个子 Agent：
  1. 接收统一格式的用户约束 dict（见 orchestrator.parse_constraints）
  2. 用专属 System Prompt 调用 LLM（开启联网搜索，获取实时票价/天气）
  3. 返回结构化 dict（字段见各函数注释）
  4. LLM 不可用时回退到 Mock 数据，保证演示不中断

设计要点：
  - 每个 Agent 只关心自己负责的领域，Prompt 短而聚焦
  - 返回 JSON 结构固定，方便主 Agent 整合与预算校验
  - accommodation / transport 支持传入收紧预算，用于预算超标时的二次协商
"""

import math
import logging

from llm_client import llm_json

logger = logging.getLogger(__name__)


# ── 共用：把约束拼成简要文本，塞进每个 Agent 的 user prompt ──────────────────
def _brief(c: dict, extra: str = "") -> str:
    prefs = ", ".join(c.get("preferences") or ["综合"])
    text = (
        f"目的地：{c['destination']}\n"
        f"出发城市：{c.get('origin_city', '未指定')}\n"
        f"出行日期：{c['start_date']} ~ {c['end_date']}（共 {c['days']} 天）\n"
        f"出行人数：{c['travelers']} 人\n"
        f"用户偏好：{prefs}\n"
        f"特殊需求：{c.get('special_needs', '无')}\n"
    )
    if extra:
        text += f"\n{extra}\n"
    return text


# ── 1. 景点调研 Agent ────────────────────────────────────────────────────────
def research_attractions(c: dict) -> dict:
    """
    搜集景点并匹配用户偏好。
    返回：{"attractions": [
        {"name", "type", "duration_hours", "ticket_price", "tags", "reason"}
    ]}
    ticket_price 为人民币单人单次。
    """
    sys = (
        "你是景点调研专家。根据用户偏好搜集目的地热门景点，"
        "返回严格 JSON：{\"attractions\":[{\"name\":\"景点名\","
        "\"type\":\"自然/人文/美食/休闲等\",\"duration_hours\":2.5,"
        "\"ticket_price\":80,\"tags\":[\"标签\"],\"reason\":\"匹配偏好的理由\"}]}。\n"
        "要求：数量 6~10 个，覆盖用户所有偏好；门票为人民币单人单次（免票填0）；"
        "只返回 JSON，不要解释。"
    )
    data = llm_json(sys, _brief(c), max_tokens=1600)
    if data and isinstance(data.get("attractions"), list) and data["attractions"]:
        return data
    # Mock 兜底
    return {"attractions": [
        {"name": f"{c['destination']}古城地标", "type": "人文",
         "duration_hours": 3.0, "ticket_price": 80,
         "tags": ["必打卡", "历史文化"], "reason": "城市地标，首次到访必去"},
        {"name": f"{c['destination']}近郊自然景区", "type": "自然",
         "duration_hours": 4.0, "ticket_price": 60,
         "tags": ["自然风光"], "reason": "体验本地自然特色"},
        {"name": f"{c['destination']}特色街区", "type": "休闲",
         "duration_hours": 2.0, "ticket_price": 0,
         "tags": ["美食", "市井"], "reason": "感受本地生活气息"},
        {"name": f"{c['destination']}博物馆", "type": "人文",
         "duration_hours": 2.0, "ticket_price": 0,
         "tags": ["历史文化", "亲子"], "reason": "免费了解城市脉络"},
    ], "_mock": True}


# ── 2. 住宿筛选 Agent ───────────────────────────────────────────────────────
def filter_accommodation(c: dict, max_price_per_night: float = None) -> dict:
    """
    按预算、位置筛选住宿。
    返回：{"recommended": {"name","price_per_night","location","features","room_capacity"},
           "alternatives": [...]}
    room_capacity 默认 2（一间住 2 人）；价格为人名币每晚每间。
    """
    extra = ""
    if max_price_per_night:
        extra = f"硬性约束：单间每晚价格 ≤ {max_price_per_night:.0f} 元，必须给出更便宜的方案。"
    sys = (
        "你是住宿筛选专家。按预算、位置、出行人结构筛选住宿，"
        "返回严格 JSON：{\"recommended\":{"
        "\"name\":\"酒店名\",\"price_per_night\":360,\"location\":\"位置\","
        "\"features\":[\"特征\"],\"room_capacity\":2},"
        "\"alternatives\":[{\"name\",\"price_per_night\",\"location\",\"features\"}]}\n"
        "price_per_night 为人民币每晚每间；room_capacity 默认 2；"
        "推荐项必须满足预算硬约束；alternatives 给 2~3 个备选；只返回 JSON。"
    )
    data = llm_json(sys, _brief(c, extra), max_tokens=1200)
    if data and data.get("recommended"):
        return data
    # Mock 兜底
    price = min(max_price_per_night or 400, 400)
    return {"recommended": {
        "name": f"{c['destination']}市中心连锁酒店",
        "price_per_night": price,
        "location": "市中心，交通便利",
        "features": ["近地铁", "含早", "免费WiFi"],
        "room_capacity": 2,
    }, "alternatives": [
        {"name": "经济型快捷酒店", "price_per_night": price * 0.7,
         "location": "主干道旁", "features": ["性价比高"]},
        {"name": "特色民宿", "price_per_night": price * 1.2,
         "location": "景区附近", "features": ["本地风情"]},
    ], "_mock": True}


# ── 3. 交通规划 Agent ───────────────────────────────────────────────────────
def plan_transport(c: dict, max_total: float = None) -> dict:
    """
    规划跨城 + 城市内交通。
    返回：{"intercity": [{"segment","mode","price_per_person","duration_hours"}],
           "intracity": [{"type","cost_per_day_total"}]}
    跨城按往返算（去程+返程）；城内 cost_per_day_total 为整队每天合计。
    """
    extra = ""
    if max_total:
        extra = f"硬性约束：交通总费用 ≤ {max_total:.0f} 元（含往返+城内），请优先推荐高性价比方案。"
    sys = (
        "你是交通规划专家。给出跨城往返与目的地城内交通方案，"
        "返回严格 JSON：{\"intercity\":[{\"segment\":\"北京->成都\","
        "\"mode\":\"高铁/飞机\",\"price_per_person\":800,\"duration_hours\":7}],"
        "\"intracity\":[{\"type\":\"地铁+打车\",\"cost_per_day_total\":60}]}\n"
        "跨城必须包含去程和返程两条；price_per_person 为人民币单人单程；"
        "intracity 的 cost_per_day_total 为整队每天合计；只返回 JSON。"
    )
    data = llm_json(sys, _brief(c, extra), max_tokens=1200)
    if data and (data.get("intercity") or data.get("intracity")):
        return data
    # Mock 兜底
    origin = c.get("origin_city", "出发地")
    dest = c["destination"]
    return {"intercity": [
        {"segment": f"{origin}->{dest}", "mode": "高铁",
         "price_per_person": 550, "duration_hours": 6},
        {"segment": f"{dest}->{origin}", "mode": "高铁",
         "price_per_person": 550, "duration_hours": 6},
    ], "intracity": [
        {"type": "地铁+网约车", "cost_per_day_total": 60},
    ], "_mock": True}


# ── 4. 美食推荐 Agent ───────────────────────────────────────────────────────
def recommend_food(c: dict) -> dict:
    """
    推荐本地美食与餐馆。
    返回：{"restaurants": [{"name","cuisine","avg_price_per_person_per_meal","must_try"}],
           "estimated_cost_per_person_per_day": 180}
    """
    sys = (
        "你是本地美食推荐专家。根据目的地推荐特色美食与餐馆，"
        "返回严格 JSON：{\"restaurants\":[{\"name\":\"餐馆名\","
        "\"cuisine\":\"菜系\",\"avg_price_per_person_per_meal\":80,"
        "\"must_try\":[\"招牌菜\"]}],"
        "\"estimated_cost_per_person_per_day\":180}\n"
        "数量 5~8 家；价格为人名币单人单餐；estimated 按三餐合计估算；只返回 JSON。"
    )
    data = llm_json(sys, _brief(c), max_tokens=1200)
    if data and (data.get("restaurants") or data.get("estimated_cost_per_person_per_day")):
        return data
    return {"restaurants": [
        {"name": f"{c['destination']}老字号面馆", "cuisine": "本地小吃",
         "avg_price_per_person_per_meal": 35, "must_try": ["招牌面", "卤味"]},
        {"name": f"{c['destination']}特色川菜馆", "cuisine": "地方菜",
         "avg_price_per_person_per_meal": 90, "must_try": ["招牌菜A", "招牌菜B"]},
    ], "estimated_cost_per_person_per_day": 180, "_mock": True}


# ── 5. 风险提醒 Agent ───────────────────────────────────────────────────────
def assess_risks(c: dict) -> dict:
    """
    评估出行风险：天气、避坑、注意事项。
    返回：{"weather": "...", "warnings": ["..."], "tips": ["..."]}
    """
    sys = (
        "你是出行风险提醒专家。基于出行日期查询目的地天气，并给出避坑提示与注意事项，"
        "返回严格 JSON：{\"weather\":\"未来天气概述（温度/降雨/穿衣建议）\","
        "\"warnings\":[\"避坑提示\"],\"tips\":[\"出行建议\"]}\n"
        "warnings 3~5 条（常见坑/宰客/高峰/安全）；tips 3~5 条实用建议；只返回 JSON。"
    )
    data = llm_json(sys, _brief(c), max_tokens=1000)
    if data and (data.get("weather") or data.get("warnings") or data.get("tips")):
        return data
    return {"weather": f"{c['destination']} 出行期天气宜人，建议带一件薄外套，关注临近预报。",
            "warnings": ["景区周边警惕黑导游", "打车优先网约车避免绕路", "高峰时段提前预约门票"],
            "tips": ["随身带身份证", "保留电子票据", "错峰游览体验更佳"], "_mock": True}
