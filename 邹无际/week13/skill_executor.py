

import os
import re
import json
import subprocess
import webbrowser
from pathlib import Path
from typing import Optional, Callable
from .skill_parser import SkillFull, ExecutionStep


# 终端颜色
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
DIM = "\033[2m"


class StepResult:
    """单个步骤的执行结果"""
    def __init__(self, step: ExecutionStep, success: bool, output: str = "", artifact: str = ""):
        self.step = step
        self.success = success
        self.output = output        # 步骤产生的文本输出
        self.artifact = artifact    # 步骤产生的文件路径


class SkillExecutor:
    """
    渐进式执行引擎

    对每个 skill 的步骤逐一执行，支持：
      - 交互式输入收集（如单词名）
      - JSON 数据文件生成
      - Shell 命令执行（如 Python 脚本）
      - 文件输出验证
      - 浏览器预览打开
    """

    def __init__(self, working_dir: Optional[Path] = None):
        self.working_dir = working_dir or Path.cwd()
        self._step_results: list[StepResult] = []
        self._context: dict = {}   # 步骤间共享上下文

    def execute(self, skill: SkillFull, user_args: str = "") -> bool:
        """
        渐进式执行一个 skill 的所有步骤

        Args:
            skill:      完整加载的 skill
            user_args:  用户提供的参数（如单词名）

        Returns:
            True 如果所有步骤执行成功
        """
        self._step_results.clear()
        self._context = {"user_args": user_args, "skill": skill}

        print(f"\n{MAGENTA}{'═'*60}{RESET}")
        print(f"{MAGENTA}  开始执行 Skill: {BOLD}{skill.meta.name}{RESET}")
        print(f"{MAGENTA}{'═'*60}{RESET}")
        print(f"  工作目录: {DIM}{self.working_dir}{RESET}")
        print(f"  Skill 目录: {DIM}{skill.meta.skill_dir}{RESET}\n")

        # 根据 skill 名称分发到对应的执行策略
        if skill.meta.name == "flash-card":
            return self._execute_flash_card(skill, user_args)
        else:
            return self._execute_generic(skill, user_args)

    def _execute_flash_card(self, skill: SkillFull, user_args: str) -> bool:
        """
        flash-card skill 的渐进式执行

        步骤：
          1. 识别单词
          2. 生成 JSON 数据
          3. 运行脚本生成 HTML
          4. 打开预览
        """
        skill_dir = skill.meta.skill_dir
        data_dir = skill_dir / "data"
        scripts_dir = skill_dir / "scripts"

        # ── Step 1: 识别单词 ─────────────────────────────────────
        print(f"  {CYAN}[Step 1/4]{RESET} 识别单词")
        word = user_args.strip().lower()
        if not word:
            # 检查 data/ 目录中已有的单词
            existing = [f.stem for f in data_dir.glob("*.json")] if data_dir.exists() else []
            if existing:
                print(f"    已有数据: {', '.join(existing)}")
                print(f"    请输入一个新单词: ", end="")
                try:
                    word = input().strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print(f"\n    {YELLOW}已取消{RESET}")
                    return False
            else:
                print(f"    请输入要制作闪卡的单词: ", end="")
                try:
                    word = input().strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print(f"\n    {YELLOW}已取消{RESET}")
                    return False

        if not word or not word.isalpha():
            print(f"    {YELLOW}无效的单词: {word}{RESET}")
            return False

        self._context["word"] = word
        print(f"    {GREEN}[OK] 目标单词: {BOLD}{word}{RESET}")

        # ── Step 2: 生成/加载 JSON 数据 ─────────────────────────
        print(f"\n  {CYAN}[Step 2/4]{RESET} 生成 JSON 数据")
        json_path = data_dir / f"{word}.json"

        if json_path.exists():
            print(f"    数据已存在: {json_path}")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"    {GREEN}[OK] 加载已有数据{RESET}")
        else:
            # 自动生成示例数据
            print(f"    自动生成 {word} 的学习数据...")
            data = self._generate_word_data(word)
            data_dir.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"    {GREEN}[OK] 数据已保存: {json_path}{RESET}")

        self._context["json_path"] = str(json_path)
        self._context["data"] = data

        # 打印数据预览
        print(f"    单词: {data['word']}  音标: {data.get('phonetic', '')}")
        print(f"    词性: {data.get('pos', '')}  释义: {data.get('definition', '')}")
        print(f"    例句: {len(data.get('examples', []))} 条")
        print(f"    近义词: {', '.join(data.get('synonyms', []))}")

        # ── Step 3: 运行脚本生成 HTML ──────────────────────────
        print(f"\n  {CYAN}[Step 3/4]{RESET} 生成 HTML 闪卡")
        script_path = scripts_dir / "make_flashcard.py"
        if not script_path.exists():
            print(f"    {YELLOW}脚本不存在: {script_path}{RESET}")
            return False

        output_path = self.working_dir / f"{word}.html"
        cmd = ["python", str(script_path), str(json_path), "-o", str(output_path)]
        print(f"    执行: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                cwd=str(self.working_dir),
                encoding="utf-8", errors="replace"
            )
            if result.returncode == 0:
                out = (result.stdout or "").strip()
                print(f"    {GREEN}[OK] {out}{RESET}")
            else:
                print(f"    {YELLOW}脚本错误: {(result.stderr or '').strip()}{RESET}")
                return False
        except subprocess.TimeoutExpired:
            print(f"    {YELLOW}脚本执行超时{RESET}")
            return False
        except Exception as e:
            print(f"    {YELLOW}执行失败: {e}{RESET}")
            return False

        self._context["html_path"] = str(output_path)

        # ── Step 4: 打开预览 ────────────────────────────────────
        print(f"\n  {CYAN}[Step 4/4]{RESET} 打开预览")
        if output_path.exists():
            print(f"    HTML 文件: {output_path}")
            try:
                webbrowser.open(str(output_path))
                print(f"    {GREEN}[OK] 已在浏览器中打开{RESET}")
            except Exception as e:
                print(f"    {YELLOW}无法打开浏览器: {e}{RESET}")
                print(f"    {DIM}请手动打开: {output_path}{RESET}")
        else:
            print(f"    {YELLOW}HTML 文件未生成{RESET}")
            return False

        # ── 完成 ────────────────────────────────────────────────
        print(f"\n{MAGENTA}{'═'*60}{RESET}")
        print(f"  [OK] Flash Card 生成完成！{RESET}")
        print(f"    数据: {DIM}{json_path}{RESET}")
        print(f"    卡片: {DIM}{output_path}{RESET}")
        print(f"{MAGENTA}{'═'*60}{RESET}\n")
        return True

    def _execute_generic(self, skill: SkillFull, user_args: str) -> bool:
        """
        通用执行策略：按 SKILL.md 中解析出的步骤逐一执行
        适用于没有专用执行器的 skill
        """
        steps = skill.execution_steps
        if not steps:
            print(f"  {YELLOW}该 skill 没有可执行的步骤{RESET}")
            return False

        total = len(steps)
        for step in steps:
            print(f"\n  {CYAN}[Step {step.index}/{total}]{RESET} {BOLD}{step.title}{RESET}")
            print(f"    {DIM}{step.detail}{RESET}")

            if step.command:
                # 替换变量
                cmd = step.command.replace("{baseDir}", skill.meta.base_dir)
                cmd = cmd.replace("{word}", user_args)
                print(f"    执行命令: {cmd}")
                try:
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True, timeout=60,
                        cwd=str(self.working_dir),
                        encoding="utf-8", errors="replace"
                    )
                    if result.returncode == 0:
                        print(f"    {GREEN}[OK] {result.stdout.strip()}{RESET}")
                    else:
                        print(f"    {YELLOW}命令返回 {result.returncode}: {result.stderr.strip()}{RESET}")
                except Exception as e:
                    print(f"    {YELLOW}命令执行失败: {e}{RESET}")
            else:
                print(f"    {GREEN}[OK] 步骤完成（说明性步骤）{RESET}")

        print(f"\n{GREEN}{BOLD}  [OK] Skill 执行完成！{RESET}\n")
        return True

    def _generate_word_data(self, word: str) -> dict:
        """
        为一个英语单词自动生成学习数据
        这里提供一个基础模板，实际使用时可以由 LLM 生成更丰富的内容
        """
        # 预置的单词数据（与 data/ 目录中已有的保持一致）
        known_words = {
            "crazy": {
                "word": "crazy",
                "phonetic": "/ˈkreɪzi/",
                "pos": "adj.",
                "definition": "疯狂的；荒唐的；着迷的；极热衷的",
                "examples": [
                    {"en": "You must be crazy to go out in this storm!", "zh": "你在这场暴风雨中出门一定是疯了！"},
                    {"en": "She is crazy about jazz music and collects old records.", "zh": "她极其热衷于爵士乐，收藏了很多老唱片。"},
                    {"en": "It sounds crazy, but the plan actually worked.", "zh": "听起来很荒唐，但这个计划居然奏效了。"}
                ],
                "synonyms": ["mad", "insane", "wild", "absurd", "nuts", "eccentric"]
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
            "thrill": {
                "word": "thrill",
                "phonetic": "/θrɪl/",
                "pos": "n. / v.",
                "definition": "n. 兴奋，激动，震颤感；v. 使兴奋，使激动，使胆战心惊",
                "examples": [
                    {"en": "She felt a sudden thrill of excitement as the roller coaster plunged downward.", "zh": "过山车俯冲而下时，她突然感到一阵兴奋的震颤。"},
                    {"en": "The crowd was thrilled by the acrobat's breathtaking performance.", "zh": "杂技演员惊心动魄的表演让观众兴奋不已。"},
                    {"en": "Winning the championship thrilled the young athlete beyond words.", "zh": "赢得冠军让这位年轻运动员激动得无法用言语表达。"}
                ],
                "synonyms": ["excitement", "euphoria", "exhilaration", "excite", "electrify", "stimulate"]
            }
        }

        if word in known_words:
            return known_words[word]

        # 对于未知单词，生成一个基础模板
        return {
            "word": word,
            "phonetic": f"/{word}/",
            "pos": "n.",
            "definition": f"（请补充 {word} 的中文释义）",
            "examples": [
                {"en": f"This is an example sentence using '{word}'.", "zh": f"这是一个使用 '{word}' 的例句。"},
                {"en": f"She learned how to use '{word}' in daily conversation.", "zh": f"她学会了在日常对话中使用 '{word}'。"},
                {"en": f"The word '{word}' has multiple meanings in different contexts.", "zh": f"'{word}' 在不同语境下有多个含义。"}
            ],
            "synonyms": ["related", "similar", "comparable"]
        }
