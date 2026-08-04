import sys
import re
import json
import subprocess
import webbrowser
from pathlib import Path

# 添加当前目录到 Python 路径（关键修复）
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 现在可以直接导入
from skill_registry import SkillRegistry
from skill_executor import SkillExecutor


class SkillHarness:
    def __init__(self, skills_root: Path | str | None = None):
        if skills_root is None:
            skills_root = Path(__file__).parent.parent / "skills"
        self.skills_root = Path(skills_root).resolve()
        self.registry = SkillRegistry()
        self.executor = SkillExecutor()
        self.registry.scan(self.skills_root)

    def reload(self):
        self.registry = SkillRegistry()
        self.registry.scan(self.skills_root)
        return len(self.registry)

    def match_intent(self, user_input: str, top_k: int = 3):
        return self.registry.match(user_input, top_k=top_k)

    def load_prompt(self, name: str, read_references: bool = True) -> str | None:
        meta = self.registry.get(name)
        if not meta:
            return None
        return self.executor.get_full_prompt(meta, read_references)

    def execute_script(self, name: str, script_name: str, args: list[str] | None = None):
        meta = self.registry.get(name)
        if not meta:
            return -1, "", f"Skill 未找到: {name}"
        return self.executor.run_script(meta, script_name, args)

    def __repr__(self) -> str:
        return f"SkillHarness(root={self.skills_root}, skills={len(self.registry)})"

    @staticmethod
    def _get_word_info(word: str) -> dict | None:
        """获取单词信息（模拟词典查询）"""
        word_data = {
            "good": {
                "word": "good",
                "phonetic": "/ɡʊd/",
                "pos": "adj.",
                "definition": "好的，优秀的；有益的；愉快的；相当的",
                "examples": [
                    {"en": "She is a good student who always gets high marks.", "zh": "她是个好学生，总是取得高分。"},
                    {"en": "This is a good opportunity to learn something new.", "zh": "这是一个学习新东西的好机会。"},
                    {"en": "Have a good day and see you tomorrow!", "zh": "祝你有美好的一天，明天见！"}
                ],
                "synonyms": ["excellent", "great", "fine", "wonderful", "nice", "positive"]
            },
            "happy": {
                "word": "happy",
                "phonetic": "/ˈhæpi/",
                "pos": "adj.",
                "definition": "快乐的，幸福的；满足的；幸运的",
                "examples": [
                    {"en": "She looks very happy today.", "zh": "她今天看起来很开心。"},
                    {"en": "I'm happy to help you with your project.", "zh": "我很乐意帮助你完成项目。"},
                    {"en": "They lived happily ever after.", "zh": "他们从此过上了幸福的生活。"}
                ],
                "synonyms": ["joyful", "cheerful", "glad", "delighted", "pleased", "content"]
            },
            "beautiful": {
                "word": "beautiful",
                "phonetic": "/ˈbjuːtɪfl/",
                "pos": "adj.",
                "definition": "美丽的，漂亮的；出色的，极好的",
                "examples": [
                    {"en": "What a beautiful sunset!", "zh": "多么美丽的日落啊！"},
                    {"en": "She is a beautiful woman inside and out.", "zh": "她是一个内外兼修的美丽女人。"},
                    {"en": "This is a beautiful piece of music.", "zh": "这是一首优美的音乐。"}
                ],
                "synonyms": ["gorgeous", "lovely", "stunning", "attractive", "pretty", "charming"]
            },
            "crazy": {
                "word": "crazy",
                "phonetic": "/ˈkreɪzi/",
                "pos": "adj.",
                "definition": "疯狂的，疯狂的；着迷的；荒唐的",
                "examples": [
                    {"en": "You're crazy to go out in this weather!", "zh": "这种天气出去你真是疯了！"},
                    {"en": "She's crazy about dancing.", "zh": "她对跳舞很着迷。"},
                    {"en": "It's crazy how fast time flies.", "zh": "时间过得真快，真让人难以置信。"}
                ],
                "synonyms": ["insane", "mad", "wild", "furious", "passionate", "obsessed"]
            },
            "resilient": {
                "word": "resilient",
                "phonetic": "/rɪˈzɪliənt/",
                "pos": "adj.",
                "definition": "能迅速从困难、挫折中恢复过来的；有韧性的，适应力强的",
                "examples": [
                    {"en": "She is a resilient child who bounces back quickly from setbacks.", "zh": "她是个有韧性的孩子，遇到挫折能很快恢复过来。"},
                    {"en": "The economy proved remarkably resilient during the crisis.", "zh": "在危机期间，经济表现出了惊人的韧性。"},
                    {"en": "A resilient mindset helps you cope with life's challenges.", "zh": "一种有韧性的心态能帮你应对生活中的挑战。"}
                ],
                "synonyms": ["tough", "flexible", "strong", "hardy", "buoyant", "springy"]
            },
            "agent": {
                "word": "agent",
                "phonetic": "/ˈeɪdʒənt/",
                "pos": "n.",
                "definition": "代理人，代理商；特工；动因，原因",
                "examples": [
                    {"en": "He works as a real estate agent.", "zh": "他是一名房地产经纪人。"},
                    {"en": "James Bond is a secret agent.", "zh": "詹姆斯·邦德是一名特工。"},
                    {"en": "Rain is the agent of erosion.", "zh": "雨水是侵蚀的动因。"}
                ],
                "synonyms": ["representative", "proxy", "delegate", "middleman", "broker", "spy"]
            },
            "thrill": {
                "word": "thrill",
                "phonetic": "/θrɪl/",
                "pos": "n./v.",
                "definition": "激动，兴奋；使激动，使兴奋",
                "examples": [
                    {"en": "The roller coaster gave me a thrill.", "zh": "过山车让我感到非常刺激。"},
                    {"en": "She thrilled the audience with her performance.", "zh": "她的表演让观众激动不已。"},
                    {"en": "It thrilled me to see them again.", "zh": "再次见到他们让我激动万分。"}
                ],
                "synonyms": ["excitement", "delight", "rapture", "excite", "electrify", "captivate"]
            },
            "deepseek": {
                "word": "deepseek",
                "phonetic": "/ˈdiːpsiːk/",
                "pos": "v.",
                "definition": "深度探索，深入追寻",
                "examples": [
                    {"en": "Scientists deepseek the mysteries of the ocean.", "zh": "科学家深入探索海洋的奥秘。"},
                    {"en": "We need to deepseek the root cause of the problem.", "zh": "我们需要深入追寻问题的根源。"},
                    {"en": "The team will deepseek new solutions.", "zh": "团队将深入探索新的解决方案。"}
                ],
                "synonyms": ["explore", "investigate", "probe", "research", "delve", "search"]
            }
        }
        return word_data.get(word.lower())


if __name__ == "__main__":
    harness = SkillHarness()
    print(harness.registry.summary())

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\n匹配意图: {query}")
        matches = harness.match_intent(query)
        
        for m in matches:
            print(f"  [{m.score:.1f}] {m.skill.name} — {m.skill.description[:60]}")
            if m.matched_keywords:
                print(f"       命中关键词: {', '.join(m.matched_keywords)}")

        if matches:
            top = matches[0].skill
            print(f"\n--- {top.name} 完整 Prompt（前 1000 字）---")
            prompt = harness.load_prompt(top.name)
            if prompt:
                print(prompt[:1000] + "...")

            if top.name == "flash-card":
                word_match = re.search(r'([a-zA-Z]+)', query)
                if word_match:
                    word = word_match.group(1).lower()
                    print(f"\n--- 正在为 '{word}' 生成闪卡... ---")
                    
                    data_path = top.base_dir / "data" / f"{word}.json"
                    output_path = top.base_dir / "data" / f"{word}.html"
                    
                    if not data_path.exists():
                        print(f"📝 JSON 文件不存在，正在自动生成...")
                        word_info = harness._get_word_info(word)
                        if word_info:
                            data_path.write_text(
                                json.dumps(word_info, ensure_ascii=False, indent=2),
                                encoding="utf-8"
                            )
                            print(f"✅ JSON 数据已生成: {data_path}")
                        else:
                            print(f"❌ 无法获取 '{word}' 的单词信息。")
                            sys.exit(1)
                    else:
                        print(f"📋 JSON 文件已存在: {data_path}")
                    
                    code, stdout, stderr = harness.execute_script(
                        "flash-card", "make_flashcard.py",
                        [str(data_path), "-o", str(output_path)]
                    )
                    
                    if code == 0:
                        print(f"✅ 闪卡生成成功: {output_path}")
                        
                        # 使用 webbrowser 模块打开预览
                        print(f"🌐 正在打开预览...")
                        try:
                            webbrowser.open(f"file:///{output_path.resolve()}")
                        except Exception as e:
                            print(f"⚠ 无法自动打开浏览器，请手动打开: {output_path}")
                    else:
                        print(f"❌ 生成失败")
                        if stderr:
                            print(f"  错误: {stderr}")
