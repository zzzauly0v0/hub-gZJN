"""
多工单交叉作业风险检测 Skill 适配层

本 Skill 的核心功能是：接收传入的作业工单信息，
与 data/work_tickets.json 中的已有工单进行比对，检测交叉作业风险。

工作流程：
1. 接收用户传入的作业工单参数（作业类型、区域、位置、时间窗口）
2. 从 data/work_tickets.json 加载所有已有工单
3. 筛选同区域 + 时间重叠的工单
4. 根据冲突规则检测风险（动火+装卸料、动火+盲板抽堵等）
5. 返回风险信息字符串

目录结构：
- SKILL.md: Skill 元数据和使用说明
- skill.py: 本文件，Skill 的 Python 实现
- data/work_tickets.json: 模拟工单数据
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

# ── 路径常量 ──────────────────────────────────────────────────────────────────
SKILL_DIR = Path(__file__).parent
DATA_DIR = SKILL_DIR / "data"
DATA_FILE = DATA_DIR / "work_tickets.json"

# ── 作业类型映射（中文名 → 代码）──────────────────────────────────────────────
# 用于将用户传入的中文作业类型转换为内部代码，以便匹配冲突规则
WORK_TYPE_MAP = {
    "动火作业": "hot_work", "动火": "hot_work",
    "装卸料作业": "loading_unloading", "装卸料": "loading_unloading",
    "盲板抽堵作业": "blind_plate", "盲板抽堵": "blind_plate",
    "检修作业": "maintenance", "检修": "maintenance",
    "高处作业": "height_work", "高处": "height_work",
}

# ── 风险检测规则定义 ──────────────────────────────────────────────────────────
# 格式: {(类型1, 类型2): {风险等级, 风险类型, 风险描述, 建议措施}}
CONFLICT_RULES = {
    # 动火 + 装卸料 = 火灾爆炸风险（最严重）
    ("hot_work", "loading_unloading"): {
        "level": "严重",
        "type": "火灾爆炸风险",
        "description": "动火作业产生的明火/火花与装卸料作业中的易燃易爆物料接触，可能引发火灾或爆炸",
        "action": "立即停止其中一项作业，或确保安全距离≥30米并设置防火隔离措施",
    },
    # 动火 + 盲板抽堵 = 有毒气体泄漏风险
    ("hot_work", "blind_plate"): {
        "level": "严重",
        "type": "有毒气体泄漏风险",
        "description": "盲板抽堵作业可能导致管道内有毒气体泄漏，遇动火作业明火可能引发中毒或二次爆炸",
        "action": "禁止同时作业，须在盲板抽堵完成并确认无泄漏后再进行动火作业",
    },
    # 动火 + 检修（含易燃物料）= 安全管控风险
    ("hot_work", "maintenance"): {
        "level": "中等",
        "type": "安全管控风险",
        "description": "检修作业可能涉及易燃物料残留，与动火作业同时进行存在安全隐患",
        "action": "确认检修设备已清洗置换合格，动火前进行可燃气体检测",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载与时间处理工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _load_tickets() -> List[Dict]:
    """
    模拟
    从 JSON 文件加载所有已有工单数据
    """
    if not DATA_FILE.exists():
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tickets", [])


def _parse_time(time_str: str) -> datetime:
    """解析时间字符串为 datetime 对象，支持 'YYYY-MM-DD HH:MM' 格式"""
    for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(time_str.strip(), fmt)
        except ValueError:
            continue
    return datetime.now()


def _time_overlap(t1_start: str, t1_end: str, t2_start: str, t2_end: str) -> bool:
    """
    检测两个时间窗口是否有重叠
    
    原理: A.end > B.start 且 A.start < B.end → 重叠
    """
    s1, e1 = _parse_time(t1_start), _parse_time(t1_end)
    s2, e2 = _parse_time(t2_start), _parse_time(t2_end)
    return s1 < e2 and s2 < e1


def _resolve_work_type_code(work_type: str) -> str:
    """
    将中文作业类型名称解析为内部代码
    
    如果传入的已经是代码（如 'hot_work'），直接返回。
    否则在 WORK_TYPE_MAP 中查找对应的中文名。
    """
    if work_type in WORK_TYPE_MAP.values():
        return work_type
    return WORK_TYPE_MAP.get(work_type, work_type)


def _format_ticket_brief(ticket: Dict) -> str:
    """格式化工单摘要信息"""
    return (
        f"  • {ticket.get('ticket_id', '?')} | {ticket.get('work_type', '?')} | "
        f"{ticket.get('location', '?')} | {ticket.get('start_time', '?')}~{ticket.get('end_time', '?')}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 四个核心工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def query_work_ticket_by_area_time(
    area: str = "",
    start_time: str = "",
    end_time: str = ""
) -> str:
    """
    按作业区域 + 时间窗口查询已有工单
    
    Args:
        area: 作业区域名称（如 "化工一车间"），为空则查全部
        start_time: 时间窗口起始（如 "2024-01-15 08:00"），为空则不限制
        end_time: 时间窗口结束（如 "2024-01-15 18:00"），为空则不限制
    
    Returns:
        工单列表字符串
    """
    tickets = _load_tickets()
    
    # 按区域筛选
    if area:
        tickets = [t for t in tickets if area in t.get("area", "")]
    
    # 按时间窗口筛选
    if start_time and end_time:
        tickets = [
            t for t in tickets
            if _time_overlap(t["start_time"], t["end_time"], start_time, end_time)
        ]
    
    if not tickets:
        return f"📭 未找到符合条件的工单（区域: {area or '全部'}, 时间: {start_time or '不限'} ~ {end_time or '不限'}）"
    
    lines = [f"📋 查询结果：共 {len(tickets)} 张工单"]
    lines.append(f"   区域: {area or '全部'} | 时间: {start_time or '不限'} ~ {end_time or '不限'}")
    lines.append("-" * 60)
    for t in tickets:
        lines.append(_format_ticket_brief(t))
    
    return "\n".join(lines)


def detect_cross_work_risks(
    work_type: str = "",
    area: str = "",
    location: str = "",
    start_time: str = "",
    end_time: str = ""
) -> str:
    """
    根据传入的作业工单，与 JSON 中已有工单比对，检测交叉作业风险
    
    执行步骤：
    1. 将传入的作业类型解析为内部代码
    2. 加载 JSON 中所有已有工单
    3. 筛选同区域 + 时间重叠的工单
    4. 逐个检查是否命中冲突规则（动火+装卸料、动火+盲板抽堵等）
    5. 检测同区域同时有3处以上作业的交叉风险
    6. 返回风险信息字符串
    
    Args:
        work_type: 作业类型（如"动火作业"、"装卸料作业"、"动火"），也可传代码如"hot_work"
        area: 作业区域（如"化工一车间"）
        location: 具体位置（如"A区-反应釜R-101"）
        start_time: 开始时间（如"2024-01-15 09:00"）
        end_time: 结束时间（如"2024-01-15 12:00"）
    
    Returns:
        风险检测结果字符串，包含冲突列表和风险描述
    """
    # ── Step 1: 解析作业类型代码 ────────────────────────────────────
    input_code = _resolve_work_type_code(work_type)
    
    # ── Step 2: 加载已有工单 ────────────────────────────────────────
    all_tickets = _load_tickets()
    if not all_tickets:
        return "📭 工单数据库为空，无法进行比对。"
    
    # ── Step 3: 筛选同区域 + 时间重叠的工单 ─────────────────────────
    # 构造传入工单的虚拟记录，用于统一比对逻辑
    input_ticket = {
        "ticket_id": "输入工单",
        "work_type": work_type or "未知",
        "work_type_code": input_code,
        "area": area,
        "location": location or area,
        "start_time": start_time,
        "end_time": end_time,
    }
    
    # 筛选同区域且时间重叠的已有工单
    related_tickets = []
    for t in all_tickets:
        # 区域匹配（模糊匹配，传入区域名包含在工单区域中或反过来）
        if area and area not in t.get("area", "") and t.get("area", "") not in area:
            continue
        # 时间重叠
        if start_time and end_time and t.get("start_time") and t.get("end_time"):
            if not _time_overlap(start_time, end_time, t["start_time"], t["end_time"]):
                continue
        related_tickets.append(t)
    
    if not related_tickets:
        return (
            f"✅ 风险检测完成\n\n"
            f"传入工单: {work_type} | {area} | {location} | {start_time}~{end_time}\n"
            f"比对工单数: 0（同区域无时间重叠工单）\n"
            f"冲突数量: 0\n\n"
            f"结论: 未发现交叉作业风险。"
        )
    
    # ── Step 4: 逐个检查冲突规则 ────────────────────────────────────
    conflicts: List[Dict] = []
    
    for existing in related_tickets:
        existing_code = existing.get("work_type_code", "")
        
        # 在 CONFLICT_RULES 中查找（双向检查）
        rule = CONFLICT_RULES.get((input_code, existing_code)) or \
               CONFLICT_RULES.get((existing_code, input_code))
        
        if rule:
            conflicts.append({
                "input_ticket": input_ticket,
                "existing_ticket": existing,
                "risk_level": rule["level"],
                "risk_type": rule["type"],
                "risk_description": rule["description"],
                "action": rule["action"],
            })
    
    # ── Step 5: 检测3处以上交叉作业 ─────────────────────────────────
    # 如果同区域时间重叠的工单数 >= 2（加上传入工单共3处以上），报告交叉风险
    if len(related_tickets) >= 2:
        # 检查这些工单是否真正同时重叠（三张工单两两重叠）
        has_triple_overlap = True
        for i in range(len(related_tickets)):
            for j in range(i + 1, len(related_tickets)):
                t1, t2 = related_tickets[i], related_tickets[j]
                if not _time_overlap(t1["start_time"], t1["end_time"],
                                     t2["start_time"], t2["end_time"]):
                    has_triple_overlap = False
                    break
            if not has_triple_overlap:
                break
        
        # 也要检查传入工单与每个已有工单都重叠
        if has_triple_overlap and start_time and end_time:
            for t in related_tickets:
                if not _time_overlap(start_time, end_time, t["start_time"], t["end_time"]):
                    has_triple_overlap = False
                    break
        
        if has_triple_overlap:
            conflicts.append({
                "input_ticket": input_ticket,
                "existing_ticket": related_tickets[0],  # 代表性工单
                "extra_tickets": related_tickets[1:],
                "risk_level": "中等",
                "risk_type": "交叉作业人员伤害风险",
                "risk_description": f"同一区域({area})同时有{len(related_tickets) + 1}处作业（含传入工单），存在交叉作业人员伤害风险",
                "action": "增设现场监护人员，协调作业时序，避免同时进行多方作业",
            })
    
    # ── Step 6: 返回风险信息字符串 ──────────────────────────────────
    if not conflicts:
        return (
            f"✅ 风险检测完成\n\n"
            f"传入工单: {work_type} | {area} | {location} | {start_time}~{end_time}\n"
            f"比对工单数: {len(related_tickets)} 张（同区域时间重叠）\n"
            f"冲突数量: 0\n\n"
            f"结论: 未发现交叉作业风险。"
        )
    
    # 统计
    severe = sum(1 for c in conflicts if c["risk_level"] == "严重")
    medium = sum(1 for c in conflicts if c["risk_level"] == "中等")
    
    lines = [
        f"🚨 风险检测报告",
        f"{'=' * 60}",
        f"传入工单: {work_type} | {area} | {location} | {start_time}~{end_time}",
        f"比对工单数: {len(related_tickets)} 张（同区域时间重叠）",
        f"发现冲突: {len(conflicts)} 处 (严重: {severe}, 中等: {medium})",
        f"{'=' * 60}",
    ]
    
    for i, c in enumerate(conflicts, 1):
        ex = c["existing_ticket"]
        lines.append(f"\n【冲突 {i}】{c['risk_level']} - {c['risk_type']}")
        lines.append(f"  传入工单: {work_type} | {location or area} | {start_time}~{end_time}")
        lines.append(f"  冲突工单: {ex['ticket_id']} | {ex['work_type']} | {ex['location']} | {ex['start_time']}~{ex['end_time']}")
        if "extra_tickets" in c:
            for et in c["extra_tickets"]:
                lines.append(f"  关联工单: {et['ticket_id']} | {et['work_type']} | {et['location']} | {et['start_time']}~{et['end_time']}")
        lines.append(f"  风险描述: {c['risk_description']}")
        lines.append(f"  建议措施: {c['action']}")
    
    return "\n".join(lines)


def list_all_work_tickets() -> str:
    """列出所有已有工单（按区域分组）"""
    tickets = _load_tickets()
    if not tickets:
        return "📭 暂无工单数据"
    
    area_groups: Dict[str, List[Dict]] = {}
    for t in tickets:
        area = t.get("area", "未知")
        if area not in area_groups:
            area_groups[area] = []
        area_groups[area].append(t)
    
    lines = [f"📋 工单列表（共 {len(tickets)} 张）"]
    for area, area_tickets in area_groups.items():
        lines.append(f"\n{'─' * 50}")
        lines.append(f"【{area}】({len(area_tickets)} 张)")
        lines.append(f"{'─' * 50}")
        for t in area_tickets:
            lines.append(_format_ticket_brief(t))
    
    return "\n".join(lines)


def get_work_ticket_detail(ticket_id: str) -> str:
    """查看指定工单的详细信息"""
    tickets = _load_tickets()
    ticket = None
    for t in tickets:
        if t.get("ticket_id", "").upper() == ticket_id.upper():
            ticket = t
            break
    
    if not ticket:
        return f"❌ 未找到工单 '{ticket_id}'。请检查工单编号是否正确。"
    
    return (
        f"📋 工单详情 - {ticket['ticket_id']}\n"
        f"{'─' * 50}\n"
        f"  工单编号: {ticket['ticket_id']}\n"
        f"  作业类型: {ticket['work_type']} ({ticket.get('work_type_code', '')})\n"
        f"  作业区域: {ticket['area']}\n"
        f"  具体位置: {ticket['location']}\n"
        f"  开始时间: {ticket['start_time']}\n"
        f"  结束时间: {ticket['end_time']}\n"
        f"  审批状态: {ticket['status']}\n"
        f"  申请人:   {ticket['applicant']}\n"
        f"  危险等级: {ticket.get('hazard_level', '未知')}\n"
        f"  作业内容: {ticket['description']}\n"
        f"{'─' * 50}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Skill 配置导出（供 Harness 引擎加载）
# ═══════════════════════════════════════════════════════════════════════════════

def create_skill() -> Dict[str, Any]:
    """
    创建风险检测 Skill 配置
    
    返回:
        Skill 配置字典（tools + system_prompt + executor）
    """
    return {
        "tools": [
            # 工具1: query_work_ticket_by_area_time
            {
                "type": "function",
                "function": {
                    "name": "query_work_ticket_by_area_time",
                    "description": "按作业区域和时间窗口查询已有工单",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "area": {
                                "type": "string",
                                "description": "作业区域名称，如 '化工一车间'、'储罐区'",
                            },
                            "start_time": {
                                "type": "string",
                                "description": "时间窗口起始，格式 'YYYY-MM-DD HH:MM'",
                            },
                            "end_time": {
                                "type": "string",
                                "description": "时间窗口结束，格式 'YYYY-MM-DD HH:MM'",
                            },
                        },
                    },
                },
            },
            # 工具2: detect_cross_work_risks
            {
                "type": "function",
                "function": {
                    "name": "detect_cross_work_risks",
                    "description": "根据传入的作业工单信息（作业类型、区域、位置、时间），与已有工单比对，检测交叉作业风险，返回风险信息字符串",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "work_type": {
                                "type": "string",
                                "description": "作业类型，如 '动火作业'、'装卸料作业'、'盲板抽堵作业'、'检修作业'、'高处作业'，也可传代码如 'hot_work'",
                            },
                            "area": {
                                "type": "string",
                                "description": "作业区域，如 '化工一车间'、'储罐区'、'公用工程区'",
                            },
                            "location": {
                                "type": "string",
                                "description": "具体位置，如 'A区-反应釜R-101'",
                            },
                            "start_time": {
                                "type": "string",
                                "description": "开始时间，格式 'YYYY-MM-DD HH:MM'",
                            },
                            "end_time": {
                                "type": "string",
                                "description": "结束时间，格式 'YYYY-MM-DD HH:MM'",
                            },
                        },
                        "required": ["work_type", "area", "start_time", "end_time"],
                    },
                },
            },
            # 工具3: list_all_work_tickets
            {
                "type": "function",
                "function": {
                    "name": "list_all_work_tickets",
                    "description": "列出所有已有工单（按区域分组显示）",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            # 工具4: get_work_ticket_detail
            {
                "type": "function",
                "function": {
                    "name": "get_work_ticket_detail",
                    "description": "查看指定工单的详细信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticket_id": {
                                "type": "string",
                                "description": "工单编号，如 'WT-2024-001'",
                            },
                        },
                        "required": ["ticket_id"],
                    },
                },
            },
        ],
        
        "system_prompt": """你是园区安全风险检测助手，使用 risk-detection 技能检测多工单交叉作业风险。

## 可用工具

### 1. query_work_ticket_by_area_time - 按区域+时间查询工单
查询指定区域和时间窗口内的已有工单。

### 2. detect_cross_work_risks - 检测交叉作业风险（核心功能）
根据传入的作业工单信息，与已有工单比对，检测交叉作业风险。
需要提供：work_type（作业类型）、area（区域）、start_time、end_time。

### 3. list_all_work_tickets - 列出所有工单
### 4. get_work_ticket_detail - 查看工单详情

## 使用场景
- 用户传入作业工单信息 → 使用 detect_cross_work_risks
- 用户查询已有工单 → 使用 query_work_ticket_by_area_time
- 用户列出所有工单 → 使用 list_all_work_tickets
- 用户查看工单详情 → 使用 get_work_ticket_detail

## 园区区域
- 化工一车间（A区）
- 储罐区（B区）
- 公用工程区（C区）

## 注意事项
- 时间格式统一为 'YYYY-MM-DD HH:MM'
- 风险检测直接返回文本结果，不生成HTML""",

        "executor": {
            "query_work_ticket_by_area_time": query_work_ticket_by_area_time,
            "detect_cross_work_risks": detect_cross_work_risks,
            "list_all_work_tickets": list_all_work_tickets,
            "get_work_ticket_detail": get_work_ticket_detail,
        },
    }
