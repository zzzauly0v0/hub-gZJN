"""
安全作业票智能助手 —— 承包商人员评估数据源（假数据）

教学用：每个查询方法都用「列表」构造假数据模拟数据库表，
按 person_id 过滤返回。仅供 week15 人员评估 demo 使用。
"""

# 表1：人员基本信息 + 证书（列表构造假数据）
PERSON_BASIC_INFO = [
    {
        "person_id": "C1234",
        "name": "张工",
        "company": "XX石化工程建设有限公司",
        "job": "焊工",
        "work_years": 12,
        "position_status": "在岗",
        "blacklist": False,
        "certificates": [
            {
                "cert_name": "特种作业操作证（熔化焊接与热切割）",
                "cert_no": "T1101****1234",
                "valid_from": "2020-07-01",
                "valid_to": "2026-06-30",
                "status": "已过期（未复审）",
            },
            {
                "cert_name": "危险化学品企业从业人员安全培训合格证",
                "cert_no": "AQ2023****88",
                "valid_from": "2023-01-15",
                "valid_to": "2027-01-14",
                "status": "有效",
            },
        ],
    },
]

# 表2：培训记录
TRAINING_RECORDS = [
    {
        "person_id": "C1234",
        "records": [
            {"course": "入厂三级安全教育", "date": "2023-03-01", "result": "合格"},
            {"course": "一般动火作业安全培训", "date": "2024-06-10", "result": "合格"},
            {"course": "受限空间作业安全培训", "date": "2025-09-20", "result": "合格"},
            {"course": "特级动火作业专项培训", "date": "无记录", "result": "未参加"},
            {"course": "特种作业操作证复审培训", "date": "无记录", "result": "未参加（证书已过期）"},
        ],
    },
]

# 表3：历史安全绩效
SAFETY_VIOLATIONS = [
    {
        "person_id": "C1234",
        "violations": [
            {
                "violation_id": "WF-2026-0512",
                "date": "2026-05-12",
                "category": "作业违章",
                "detail": "动火作业中未佩戴防护面罩",
                "level": "一般违章",
                "status": "整改中（未闭环）",
            },
        ],
        "accidents": [],  # 无事故记录
        "blacklist_history": [
            {"period": "2019-01 ~ 2019-06", "reason": "轻微违章累计", "current_status": "已解除"},
        ],
    },
]

# 表4：当前参与中的作业票（无时间冲突 → 动态状态维度可通过）
CURRENT_ASSIGNMENTS = [
    {
        "person_id": "C1234",
        "assignments": [
            {
                "ticket_no": "TK-20260813-03",
                "work_type": "受限空间作业",
                "location": "302罐区",
                "time": "2026-08-13 09:00 ~ 16:00",
                "role": "作业人",
                "status": "进行中",
            },
        ],
    },
]


def _find(table: list, person_id: str) -> dict:
    for row in table:
        if row["person_id"] == person_id:
            return row
    return {"error": f"未找到工号 {person_id} 的记录"}


def get_person_basic_info(person_id: str) -> dict:
    """获取人员基本信息和证书。"""
    return _find(PERSON_BASIC_INFO, person_id)


def get_training_records(person_id: str) -> dict:
    """获取培训记录。"""
    return _find(TRAINING_RECORDS, person_id)


def get_safety_violations(person_id: str) -> dict:
    """获取历史违章记录。"""
    return _find(SAFETY_VIOLATIONS, person_id)


def get_current_assignments(person_id: str) -> dict:
    """获取当前参与中的作业票。"""
    return _find(CURRENT_ASSIGNMENTS, person_id)


if __name__ == "__main__":
    import json
    pid = "C1234"
    for fn in (get_person_basic_info, get_training_records,
               get_safety_violations, get_current_assignments):
        print(json.dumps(fn(pid), ensure_ascii=False, indent=1)[:200], "\n")
