"""
Flash Card Skill 适配层

本 Skill 的核心功能是为英语单词生成学习闪卡（HTML 格式）。

工作流程：
1. 接收 LLM 或规则引擎的工具调用请求
2. 将单词数据保存为 JSON 文件（持久化）
3. 根据数据生成漂亮的 HTML 闪卡文件
4. 返回生成结果（含文件路径）

目录结构：
- SKILL.md: Skill 元数据和使用说明
- skill.py: 本文件，Skill 的 Python 实现
- data/: 存放单词的 JSON 数据文件
- scripts/make_flashcard.py: 独立的 HTML 生成脚本

三个核心工具：
- generate_flashcard: 生成单词闪卡（JSON + HTML）
- list_flashcards: 列出所有已有闪卡
- show_flashcard: 查看指定单词的闪卡详情
"""

import json
import html
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

# Skill 目录路径（相对于本文件）
SKILL_DIR = Path(__file__).parent           # skills/flash-card/
DATA_DIR = SKILL_DIR / "data"               # skills/flash-card/data/
SCRIPTS_DIR = SKILL_DIR / "scripts"         # skills/flash-card/scripts/


def _load_existing_word(word: str) -> Optional[Dict]:
    """
    从 data/ 目录加载已有的单词 JSON 数据
    
    Args:
        word: 单词（会转为小写作为文件名）
    
    Returns:
        单词数据字典，或 None（文件不存在时）
    """
    data_file = DATA_DIR / f"{word.lower()}.json"
    if data_file.exists():
        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_word_data(word: str, data: Dict) -> str:
    """
    将单词数据保存为 JSON 文件
    
    Args:
        word: 单词
        data: 单词数据字典
    
    Returns:
        保存后的文件路径字符串
    """
    # 确保 data 目录存在
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 写入 JSON 文件
    data_file = DATA_DIR / f"{word.lower()}.json"
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return str(data_file)


def _generate_html_content(word: str, phonetic: str, pos: str, 
                           definition: str, examples: list, 
                           synonyms: list) -> str:
    """
    生成 Flash Card 的 HTML 内容
    
    使用内联的 HTML 模板（与 scripts/make_flashcard.py 保持一致），
    生成美观的学习卡片页面。
    
    Args:
        word: 单词
        phonetic: 音标
        pos: 词性
        definition: 中文释义
        examples: 例句列表 [{en, zh}, ...]
        synonyms: 近义词列表
    
    Returns:
        完整的 HTML 字符串
    """
    # ── 渲染例句列表 ──────────────────────────────────────────────────
    # 固定 3 条：不足补占位，多余截断
    fixed_examples = list(examples[:3]) + [{}] * (3 - len(examples))
    examples_html = ""
    for ex in fixed_examples:
        en = html.escape(ex.get("en", "") or "（待补充例句）")
        zh = html.escape(ex.get("zh", "") or "（待补充翻译）")
        examples_html += (
            f'<li>'
            f'<div class="en">{en}</div>'
            f'<div class="zh">{zh}</div>'
            f'</li>\n        '
        )
    
    # ── 渲染近义词标签 ────────────────────────────────────────────────
    synonyms_html = " ".join(
        f'<span class="tag">{html.escape(s)}</span>' for s in synonyms
    )
    
    # ── 拼装完整 HTML ────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(word)} - Flash Card</title>
<style>
  /* ── 基础样式 ── */
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
    background: #f5f7fb;
    color: #1f2937;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }}
  
  /* ── 卡片容器 ── */
  .card {{
    width: 100%;
    max-width: 720px;
    background: #fff;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(17, 24, 39, 0.08);
    overflow: hidden;
  }}
  
  /* ── 头部区域（渐变背景） ── */
  .header {{
    padding: 32px 36px 24px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: #fff;
  }}
  .word {{ margin: 0; font-size: 44px; font-weight: 700; }}
  .phonetic {{ margin-top: 8px; font-size: 18px; opacity: 0.92; font-style: italic; }}
  
  /* ── 内容区域 ── */
  .body {{ padding: 28px 36px 36px; }}
  
  /* ── 释义 ── */
  .definition {{
    font-size: 20px; line-height: 1.6;
    padding: 14px 16px; background: #eef2ff;
    border-left: 4px solid #4f46e5; border-radius: 8px;
  }}
  .definition .pos {{ color: #4f46e5; font-weight: 600; margin-right: 6px; }}
  
  /* ── 小标题 ── */
  h2 {{
    margin: 28px 0 14px;
    font-size: 16px; font-weight: 600;
    color: #6b7280; text-transform: uppercase; letter-spacing: 1px;
  }}
  
  /* ── 近义词标签 ── */
  .synonyms {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .synonyms .tag {{
    padding: 6px 14px; background: #eef2ff; color: #4f46e5;
    border-radius: 999px; font-size: 14px; font-weight: 500;
  }}
  
  /* ── 例句列表 ── */
  .examples {{ list-style: none; padding: 0; margin: 0; }}
  .examples li {{
    padding: 14px 16px; margin-bottom: 10px;
    background: #fafafa; border: 1px solid #e5e7eb; border-radius: 10px;
  }}
  .examples .en {{ font-size: 17px; line-height: 1.55; }}
  .examples .zh {{ margin-top: 6px; font-size: 14px; color: #6b7280; line-height: 1.55; }}
  
  /* ── 页脚 ── */
  .footer {{
    margin-top: 28px; padding-top: 16px;
    border-top: 1px dashed #e5e7eb;
    font-size: 12px; color: #6b7280; text-align: center;
  }}
</style>
</head>
<body>
  <div class="card">
    <!-- 头部：单词 + 音标 -->
    <div class="header">
      <h1 class="word">{html.escape(word)}</h1>
      <div class="phonetic">{html.escape(phonetic)}</div>
    </div>
    
    <!-- 内容区 -->
    <div class="body">
      <!-- 释义 -->
      <div class="definition">
        <span class="pos">{html.escape(pos)}</span>{html.escape(definition)}
      </div>
      
      <!-- 近义词 -->
      <h2>近义词</h2>
      <div class="synonyms">{synonyms_html}</div>
      
      <!-- 例句 -->
      <h2>例句</h2>
      <ul class="examples">{examples_html}</ul>
      
      <!-- 页脚 -->
      <div class="footer">Flash Card · 学一个词，记一组词</div>
    </div>
  </div>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# 三个核心工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def generate_flashcard(word: str, phonetic: str, pos: str, 
                       definition: str, examples: Optional[list] = None,
                       synonyms: Optional[list] = None) -> str:
    """
    为指定单词生成 HTML 闪卡文件
    
    执行步骤：
    1. 校验参数完整性
    2. 补全例句到 3 条
    3. 保存 JSON 数据到 data/ 目录
    4. 生成 HTML 文件到 skill 根目录
    5. 返回生成结果摘要
    
    Args:
        word: 单词（必填）
        phonetic: 音标，如 '/ˈkreɪzi/'（必填）
        pos: 词性，如 'adj.'（必填）
        definition: 中文释义（必填）
        examples: 例句列表，每项含 en（英文）和 zh（中文）
        synonyms: 近义词列表
    
    Returns:
        生成结果描述字符串（含文件路径）
    """
    # ── 参数校验 ──────────────────────────────────────────────────────
    if not word or not word.strip():
        return "❌ 错误：必须提供单词"
    
    word = word.strip().lower()
    examples = examples or []
    synonyms = synonyms or []
    
    # ── 补全例句到 3 条 ──────────────────────────────────────────────
    while len(examples) < 3:
        examples.append({
            "en": f"Example sentence for the word '{word}'.",
            "zh": f"这是单词 '{word}' 的例句。"
        })
    
    # ── Step 1: 保存 JSON 数据 ──────────────────────────────────────
    word_data = {
        "word": word,
        "phonetic": phonetic,
        "pos": pos,
        "definition": definition,
        "examples": examples,
        "synonyms": synonyms,
    }
    json_path = _save_word_data(word, word_data)
    
    # ── Step 2: 生成 HTML 文件 ──────────────────────────────────────
    html_content = _generate_html_content(
        word=word,
        phonetic=phonetic,
        pos=pos,
        definition=definition,
        examples=examples,
        synonyms=synonyms,
    )
    
    # HTML 输出到 skill 根目录
    html_path = SKILL_DIR / f"{word}.html"
    html_path.write_text(html_content, encoding="utf-8")
    
    # ── Step 3: 返回结果 ──────────────────────────────────────────
    synonyms_str = ", ".join(synonyms[:5]) if synonyms else "无"
    examples_count = len(examples)
    
    return (
        f"✅ 闪卡生成成功！\n\n"
        f"📝 单词信息：\n"
        f"  • 单词: {word}\n"
        f"  • 音标: {phonetic}\n"
        f"  • 词性: {pos}\n"
        f"  • 释义: {definition}\n"
        f"  • 近义词: {synonyms_str}\n"
        f"  • 例句数: {examples_count}\n\n"
        f"📁 文件路径：\n"
        f"  • JSON 数据: {json_path}\n"
        f"  • HTML 闪卡: {html_path}\n\n"
        f"💡 提示：在浏览器中打开 HTML 文件即可查看闪卡效果。"
    )


def list_flashcards() -> str:
    """
    列出所有已生成的闪卡
    
    扫描 data/ 目录下的所有 JSON 文件，汇总显示。
    
    Returns:
        闪卡列表字符串
    """
    # ── 扫描 data 目录 ──────────────────────────────────────────────
    if not DATA_DIR.exists():
        return "📭 暂无闪卡数据（data/ 目录不存在）"
    
    json_files = list(DATA_DIR.glob("*.json"))
    if not json_files:
        return "📭 暂无闪卡数据"
    
    # ── 逐个读取并汇总 ──────────────────────────────────────────────
    cards = []
    for f in sorted(json_files):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            
            # 检查对应 HTML 是否存在
            html_file = SKILL_DIR / f"{data.get('word', f.stem)}.html"
            html_status = "✅" if html_file.exists() else "⚠️"
            
            cards.append(
                f"  {html_status} {data.get('word', f.stem)} "
                f"({data.get('pos', '?')}) "
                f"- {data.get('definition', '')[:40]}"
            )
        except Exception as e:
            cards.append(f"  ❌ {f.stem} (数据损坏: {e})")
    
    return f"📚 共有 {len(cards)} 张闪卡：\n" + "\n".join(cards)


def show_flashcard(word: str) -> str:
    """
    显示指定单词的闪卡详细信息
    
    Args:
        word: 要查看的英语单词
    
    Returns:
        闪卡详情字符串
    """
    word = word.strip().lower()
    
    # ── 加载 JSON 数据 ──────────────────────────────────────────────
    data = _load_existing_word(word)
    if not data:
        return f"❌ 未找到 '{word}' 的闪卡数据。请先使用 generate_flashcard 生成。"
    
    # ── 检查 HTML 文件 ──────────────────────────────────────────────
    html_file = SKILL_DIR / f"{word}.html"
    html_status = "✅ 已生成" if html_file.exists() else "⚠️ 未生成"
    
    # ── 格式化输出 ──────────────────────────────────────────────
    examples_text = ""
    for i, ex in enumerate(data.get("examples", []), 1):
        examples_text += f"\n  [{i}] {ex.get('en', '')}\n      {ex.get('zh', '')}"
    
    synonyms_text = ", ".join(data.get("synonyms", [])) or "无"
    
    return (
        f"📖 闪卡详情 - {word}\n"
        f"{'─' * 40}\n"
        f"  音标: {data.get('phonetic', '?')}\n"
        f"  词性: {data.get('pos', '?')}\n"
        f"  释义: {data.get('definition', '?')}\n"
        f"  近义词: {synonyms_text}\n"
        f"  HTML 文件: {html_status}\n"
        f"{'─' * 40}\n"
        f"  例句:{examples_text}\n"
        f"{'─' * 40}\n"
        f"  HTML 路径: {html_file if html_file.exists() else '(未生成)'}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Skill 配置导出（供 Harness 引擎加载）
# ═══════════════════════════════════════════════════════════════════════════════

def create_skill() -> Dict[str, Any]:
    """
    创建 Flash Card Skill 配置
    
    本函数是 Harness 引擎加载 Skill 时调用的入口。
    返回的配置包含：
    - tools: 工具定义列表（OpenAI Function Calling 格式）
    - system_prompt: 系统提示词（指导 LLM 如何使用这些工具）
    - executor: 工具执行器映射（工具名 → 函数）
    
    Returns:
        Skill 配置字典
    """
    return {
        # ── 工具定义 ──────────────────────────────────────────────
        "tools": [
            # 工具1: generate_flashcard
            {
                "type": "function",
                "function": {
                    "name": "generate_flashcard",
                    "description": "为一个英语单词生成学习闪卡（HTML 文件），需要提供音标、词性、释义、例句和近义词",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "word": {
                                "type": "string",
                                "description": "英语单词，如 'crazy'、'resilient'",
                            },
                            "phonetic": {
                                "type": "string",
                                "description": "音标，如 '/ˈkreɪzi/'",
                            },
                            "pos": {
                                "type": "string",
                                "description": "词性，如 'adj.'、'n.'、'v.'",
                            },
                            "definition": {
                                "type": "string",
                                "description": "中文释义",
                            },
                            "examples": {
                                "type": "array",
                                "description": "例句列表（建议 3 条），每项含 en（英文）和 zh（中文翻译）",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "en": {"type": "string", "description": "英文例句"},
                                        "zh": {"type": "string", "description": "中文翻译"},
                                    },
                                },
                            },
                            "synonyms": {
                                "type": "array",
                                "description": "近义词列表（4-6 个为宜）",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["word", "phonetic", "pos", "definition"],
                    },
                },
            },
            # 工具2: list_flashcards
            {
                "type": "function",
                "function": {
                    "name": "list_flashcards",
                    "description": "列出所有已生成的闪卡及其状态",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            # 工具3: show_flashcard
            {
                "type": "function",
                "function": {
                    "name": "show_flashcard",
                    "description": "查看指定单词的闪卡详细信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "word": {
                                "type": "string",
                                "description": "要查看的英语单词",
                            },
                        },
                        "required": ["word"],
                    },
                },
            },
        ],
        
        # ── 系统提示词 ──────────────────────────────────────────
        "system_prompt": """你是一个英语学习助手，使用 flash-card 技能为用户生成单词学习闪卡。

## 可用工具

### 1. generate_flashcard - 生成闪卡
为指定单词生成 HTML 闪卡文件。需要提供：
- word: 单词（必填）
- phonetic: 音标，如 '/ˈkreɪzi/'（必填）
- pos: 词性，如 'adj.'（必填）
- definition: 中文释义（必填）
- examples: 例句列表，每项含 en 和 zh（建议 3 条）
- synonyms: 近义词列表（建议 4-6 个）

### 2. list_flashcards - 列出闪卡
查看所有已生成的闪卡。

### 3. show_flashcard - 查看闪卡详情
查看指定单词的闪卡信息。

## 使用场景
- 用户说"做一张XX的闪卡"→ 使用 generate_flashcard
- 用户问"有哪些闪卡"→ 使用 list_flashcards
- 用户问"查看XX闪卡"→ 使用 show_flashcard

## 例句要求
- 地道、长度适中、体现典型用法
- 中英文对应准确
- 近义词贴近核心含义""",
        
        # ── 工具执行器 ──────────────────────────────────────────
        "executor": {
            "generate_flashcard": generate_flashcard,
            "list_flashcards": list_flashcards,
            "show_flashcard": show_flashcard,
        },
    }
