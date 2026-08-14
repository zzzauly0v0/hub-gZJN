"""
对比测试：串行 vs 并行执行
展示并行调用的性能优势
"""
import time
import logging
from agents import plan_travel

logging.basicConfig(level=logging.WARNING)


def test_compare(query: str, city: str):
    """对比串行和并行的执行时间"""
    
    print("=" * 70)
    print(f"📍 对比测试: {query}")
    print("=" * 70)
    
    # ── 并行执行 ──
    print("\n[并行模式] 开始...")
    start = time.time()
    result_parallel = plan_travel(query, serial=False)
    parallel_time = time.time() - start
    
    # ── 串行执行 ──
    print("\n[串行模式] 开始...")
    start = time.time()
    result_serial = plan_travel(query, serial=True)
    serial_time = time.time() - start
    
    # ── 结果对比 ──
    print("\n" + "=" * 70)
    print("📊 性能对比结果")
    print("=" * 70)
    print(f"  并行模式: {parallel_time:.2f}s")
    print(f"  串行模式: {serial_time:.2f}s")
    print(f"  加速比: {serial_time / parallel_time:.2f}x")
    
    # 获取详细统计
    stats_p = result_parallel.get("parallel_stats", [])[0] if result_parallel.get("parallel_stats") else {}
    stats_s = result_serial.get("parallel_stats", [])[0] if result_serial.get("parallel_stats") else {}
    
    if stats_p:
        print(f"\n  并行详情: {stats_p.get('n_subagents', 0)}个子Agent, "
              f"wall {stats_p.get('wall_clock', 0)}s, "
              f"理论串行 {stats_p.get('serial_sum', 0)}s, "
              f"加速 {stats_p.get('speedup', 0)}x")
    
    if stats_s:
        print(f"  串行详情: {stats_s.get('n_subagents', 0)}个子Agent, "
              f"wall {stats_s.get('wall_clock', 0)}s")
    
    # 显示结果预览
    print("\n" + "=" * 70)
    print("📋 并行生成攻略预览（前200字）:")
    print("=" * 70)
    print(result_parallel.get("final_answer", "")[:200] + "...")
    
    return {
        "parallel": {"time": parallel_time, "result": result_parallel},
        "serial": {"time": serial_time, "result": result_serial}
    }


if __name__ == "__main__":
    # 运行对比测试
    test_compare("帮我规划北京3日游，要包含景点、天气和美食", "北京")
    
    print("\n" + "=" * 70)
    print("✅ 对比测试完成")
    print("=" * 70)
