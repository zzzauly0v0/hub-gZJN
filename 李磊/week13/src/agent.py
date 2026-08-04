"""
Skills ReAct Agent 统一入口（Function Calling 版）

使用方式：
  python agent.py
  python agent.py --question "给我做一张 resilient 词的闪卡"
  python agent.py --question "..." --max_steps 8

环境变量：
  DEEPSEEK_API_KEY   必填（用于 LLM 调用）
"""

import argparse

DEFAULT_QUESTION = "给我做一张 crazy 词的闪卡"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skills ReAct Agent (Function Calling)")
    parser.add_argument("--question",  default=DEFAULT_QUESTION)
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()

    from react_function_calling import run_and_print

    run_and_print(args.question, args.max_steps)
