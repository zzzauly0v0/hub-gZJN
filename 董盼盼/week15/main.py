"""
多角色旅游规划智能体 —— CLI 入口

主 Agent：接收约束、分配任务、校验预算、整合行程
子 Agent：景点 / 住宿 / 交通 / 美食 / 风险

使用方式：
  # 1) 交互式（逐项输入）
  python main.py

  # 2) 命令行参数（一键运行示例）
  python main.py --destination 成都 --origin 北京 \
      --start 2026-09-01 --end 2026-09-05 --travelers 2 \
      --budget 6000 --preferences 美食 历史文化 自然风光

  # 3) 无参数直接运行示例
  python main.py --demo

环境变量：
  DASHSCOPE_API_KEY  设置后使用 Qwen 联网搜索；不设则走 Mock 演示
  AGENT_MODEL        默认 qwen-max

依赖：
  pip install -r requirements.txt
"""

import os
import sys
import argparse
import logging

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import parse_constraints, plan_trip

logging.basicConfig(level=logging.WARNING)

# 颜色输出（Windows 终端兼容）
CYAN, GREEN, YELLOW, MAGENTA, RESET = "\033[36m", "\033[32m", "\033[33m", "\033[35m", "\033[0m"


def _demo_constraints() -> dict:
    return {
        "destination":  "成都",
        "origin_city":  "北京",
        "start_date":   "2026-09-01",
        "end_date":     "2026-09-05",
        "travelers":    2,
        "budget_total": 6000,
        "preferences":  ["美食", "历史文化", "自然风光"],
        "special_needs": "带老人，行程不要太紧",
    }


def _interactive_input() -> dict:
    print(f"{CYAN}=== 多角色旅游规划智能体 ==={RESET}\n请输入出行约束：")
    destination = input("目的地城市：").strip() or "成都"
    origin_city = input("出发城市（回车跳过）：").strip() or "北京"
    start_date  = input("出发日期 (YYYY-MM-DD)：").strip() or "2026-09-01"
    end_date    = input("返回日期 (YYYY-MM-DD)：").strip() or "2026-09-05"
    travelers   = input("出行人数 (默认2)：").strip() or "2"
    budget      = input("总预算 (元，默认6000)：").strip() or "6000"
    prefs_raw   = input("偏好（空格分隔，如 美食 历史 自然）：").strip()
    special     = input("特殊需求（回车跳过）：").strip() or "无"
    return {
        "destination":  destination,
        "origin_city":  origin_city,
        "start_date":   start_date,
        "end_date":     end_date,
        "travelers":    int(travelers),
        "budget_total": float(budget),
        "preferences":  prefs_raw.split() if prefs_raw else [],
        "special_needs": special,
    }


def main():
    parser = argparse.ArgumentParser(description="多角色旅游规划智能体")
    parser.add_argument("--demo", action="store_true", help="运行示例用例")
    parser.add_argument("--destination", help="目的地城市")
    parser.add_argument("--origin", dest="origin_city", default="北京", help="出发城市")
    parser.add_argument("--start", dest="start_date", help="出发日期 YYYY-MM-DD")
    parser.add_argument("--end", dest="end_date", help="返回日期 YYYY-MM-DD")
    parser.add_argument("--travelers", type=int, default=2, help="出行人数")
    parser.add_argument("--budget", type=float, default=6000, help="总预算 (元)")
    parser.add_argument("--preferences", nargs="*", default=[], help="偏好列表")
    parser.add_argument("--special", dest="special_needs", default="无", help="特殊需求")
    parser.add_argument("--save", help="保存到指定文件路径")
    args = parser.parse_args()

    if args.demo or not args.destination:
        raw = _demo_constraints() if args.demo else _interactive_input()
    else:
        raw = {
            "destination":  args.destination,
            "origin_city":  args.origin_city,
            "start_date":   args.start_date,
            "end_date":     args.end_date,
            "travelers":    args.travelers,
            "budget_total": args.budget,
            "preferences":  args.preferences,
            "special_needs": args.special_needs,
        }

    if not raw.get("start_date") or not raw.get("end_date"):
        print(f"{YELLOW}日期不完整，使用示例日期{RESET}")
        raw["start_date"] = raw.get("start_date") or "2026-09-01"
        raw["end_date"]   = raw.get("end_date")   or "2026-09-05"

    constraints = parse_constraints(raw)

    print(f"\n{MAGENTA}{'='*60}{RESET}")
    print(f"{MAGENTA}  主 Agent 开始规划{RESET}")
    print(f"{MAGENTA}{'='*60}{RESET}")

    itinerary = plan_trip(constraints, verbose=True)

    print(f"\n{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}  最终行程{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")
    print(itinerary)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(itinerary)
        print(f"\n{YELLOW}行程已保存至：{args.save}{RESET}")


if __name__ == "__main__":
    main()
