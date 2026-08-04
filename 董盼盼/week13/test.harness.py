# =============================================================================
# 文件: test_harness.py
# 意图: Harness 框架测试脚本
# 功能: 验证 Harness 的核心功能
#   - test_flashcard_progressive_load(): 测试 flash-card Skill 的自动发现、渐进式加载、执行
#   - test_progressive_load_all():      测试所有 Skill 的批量并发加载
# 运行: python harness/test_harness.py
# =============================================================================

import asyncio
import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import Harness


async def test_flashcard_progressive_load():
    print("=" * 60)
    print("Harness 渐进式加载 flash-card Skill 测试")
    print("=" * 60)
    
    harness = Harness()
    
    def on_discovered(skill):
        print(f"[*] 发现 Skill: {skill.name}")
    
    def on_loaded(skill):
        print(f"[OK] Skill 加载完成: {skill.name}")
    
    def on_load_failed(name, error):
        print(f"[FAIL] Skill 加载失败: {name} - {error}")
    
    harness.set_discovered_callback(on_discovered)
    harness.set_loaded_callback(on_loaded)
    harness.set_load_failed_callback(on_load_failed)
    
    print("\n--- 1. 发现 Skills ---")
    count = harness.discover()
    print(f"共发现 {count} 个 Skills")
    
    print("\n--- 2. 渐进式加载 flash-card ---")
    skill = await harness.load_skill("flash-card", on_progress=lambda p: print(f"   加载进度: {p:.0f}%"))
    
    if skill.is_ready():
        print(f"\n--- 3. flash-card Skill 信息 ---")
        print(f"   名称: {skill.name}")
        print(f"   状态: {skill.status.value}")
        print(f"   路径: {skill.path}")
        print(f"   描述: {skill.metadata.get('description', '无')[:60]}...")
        print(f"   数据文件: {len(skill.data_files)} 个")
        print(f"   脚本文件: {len(skill.script_files)} 个")
        
        words = harness.list_flashcard_words()
        print(f"   可用单词: {', '.join(words)}")
        
        print("\n--- 4. 执行 flash-card 生成 HTML ---")
        for word in words:
            try:
                output_path = harness.execute_flashcard(word)
                print(f"   [OK] 生成 {word}.html -> {output_path}")
            except Exception as e:
                print(f"   [FAIL] 生成 {word}.html 失败: {e}")
    
    else:
        print(f"\n[FAIL] flash-card Skill 加载失败: {skill.load_error}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


async def test_progressive_load_all():
    print("\n\n" + "=" * 60)
    print("Harness 渐进式加载所有 Skills 测试")
    print("=" * 60)
    
    harness = Harness()
    
    def on_progress(name, p):
        print(f"   {name}: {p:.1%}")
    
    print("\n--- 并行加载所有 Skills ---")
    skills = await harness.load_all(concurrent=True, on_progress=on_progress)
    
    print(f"\n--- 加载结果 ---")
    for skill in skills:
        status_mark = "[OK]" if skill.is_ready() else "[FAIL]"
        print(f"   {status_mark} {skill.name}: {skill.status.value}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_flashcard_progressive_load())
    asyncio.run(test_progressive_load_all())
