"""
CLI 终端版旅行规划
只在终端打印，不启动 Web 服务
python cli.py                 # 交互式输入问题
python cli.py "北京天气怎么样"  # 命令行直接传问题
"""
import sys
from agents import plan_travel


def print_result(query: str, serial: bool = False):
    print()
    print("=" * 70)
    print(f"🧳 用户问题: {query}")
    print(f"⚙️  模式: {'串行' if serial else '并行'}")
    print("=" * 70)

    main_steps = []
    sub_steps = {}
    sub_results = {}
    dispatch_info = [None]

    def on_main_step(step):
        main_steps.append(step)
        tag = "🧠 主"
        action = step.get("action", "")
        thought = step.get("thought", "")
        obs = step.get("observation", "")
        if action == "final_answer":
            pass  # final 单独打印
        else:
            print(f"\n{tag} [Step {step['step']}] action={action}")
            if thought and thought != "（工具调用）":
                print(f"    💭 thought: {thought[:200]}")
            if obs:
                first_obs = obs.split("\n")[0][:200]
                print(f"    👁  obs    : {first_obs}")

    def on_subagent_step(sid, step):
        if sid not in sub_steps:
            sub_steps[sid] = []
        sub_steps[sid].append(step)
        topic = sub_results.get(sid, {}).get("task", sid)
        tag = f"📌 {topic[:8]}"
        action = step.get("action", "")
        thought = step.get("thought", "")
        if action == "final_answer":
            pass
        else:
            print(f"    {tag} [{sid}] Step {step['step']} action={action}")
            if thought and thought != "（工具调用）":
                print(f"            💭 {thought[:120]}")

    def on_subagent_done(sid, duration, task):
        sub_results[sid] = {"duration": duration, "task": task}
        print(f"    ✅ [{sid}] 完成: {task} ({duration}s)")

    def on_dispatch(info):
        dispatch_info[0] = info
        print(f"\n🚀 派发 {len(info['subtasks'])} 个子任务并行执行:")
        for i, t in enumerate(info["subtasks"], 1):
            print(f"    {i}. {t}")

    print("\n▶️  开始规划...")
    result = plan_travel(
        query=query,
        on_main_step=on_main_step,
        on_subagent_step=on_subagent_step,
        on_subagent_done=on_subagent_done,
        on_dispatch=on_dispatch,
        serial=serial,
    )

    print("\n" + "=" * 70)
    print("📊 执行总结")
    print("=" * 70)
    print(f"主 Agent 步数: {len(main_steps)}")
    if dispatch_info[0]:
        print(f"派发次数: {len(result['dispatches'])}  |  子 Agent 数量: {len(result['subagents'])}")
    else:
        print("派发次数: 0（单一查询，主 Agent 直接执行）")
    if result["parallel_stats"]:
        s = result["parallel_stats"][0]
        print(f"并行: {s['n_subagents']} 个子Agent | wall {s['wall_clock']}s | "
              f"串行 {s['serial_sum']}s | 加速 {s['speedup']}x")

    print("\n" + "=" * 70)
    print("📋 最终旅行攻略")
    print("=" * 70)
    print(result["final_answer"])
    print()


def main():
    serial = False
    args = sys.argv[1:]
    if "--serial" in args:
        serial = True
        args.remove("--serial")

    if args:
        query = " ".join(args).strip()
    else:
        print("=" * 70)
        print("🧳 智能旅行规划助手（终端版）")
        print("=" * 70)
        print("示例：")
        print("  · 单一查询：北京天气怎么样")
        print("  · 单一查询：杭州西湖附近有什么好玩的？")
        print("  · 综合规划：帮我规划北京3日游，要包含景点、天气和美食")
        print("  · 综合规划：我想去成都玩4天，推荐景点和美食")
        print("输入空行或 q 退出。")
        print()
        while True:
            try:
                q = input("❓ 请输入问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if q in ("", "q", "quit", "exit"):
                return
            print_result(q, serial=serial)
        return

    print_result(query, serial=serial)


if __name__ == "__main__":
    main()
