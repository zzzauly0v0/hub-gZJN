"""知识库加载模块 - 解析 knowledge_base.md 文件"""

import re
from pathlib import Path
from typing import List, Dict


def load_knowledge_base(file_path: str = "knowledge_base.md") -> List[Dict[str, str]]:
    """
    解析知识库 Markdown 文件，返回结构化的知识条目列表。

    每个条目为字典结构，包含：
    - title: 标题
    - category: 分类
    - question: 问题
    - answer: 答案

    Args:
        file_path: 知识库文件路径

    Returns:
        知识条目列表
    """
    content = Path(file_path).read_text(encoding="utf-8")

    # 使用 --- 分隔符拆分条目
    sections = re.split(r"\n---\n", content)

    entries = []
    for section in sections:
        section = section.strip()
        if not section or section.startswith("#"):
            continue

        entry = {}
        # 解析 key: value 格式的字段
        for field in ["title", "category", "question", "answer"]:
            pattern = rf"^{field}:\s*(.+)$"
            match = re.search(pattern, section, re.MULTILINE)
            if match:
                entry[field] = match.group(1).strip()

        if all(k in entry for k in ["title", "category", "question", "answer"]):
            entries.append(entry)

    return entries


def format_knowledge_for_prompt(entries: List[Dict[str, str]]) -> str:
    """
    将知识条目格式化为可注入 prompt 的文本。

    Args:
        entries: 知识条目列表

    Returns:
        格式化后的知识库文本
    """
    parts = []
    for i, entry in enumerate(entries, 1):
        parts.append(
            f"【知识条目 {i}】\n"
            f"主题：{entry['title']}\n"
            f"分类：{entry['category']}\n"
            f"问题：{entry['question']}\n"
            f"答案：{entry['answer']}"
        )
    return "\n\n".join(parts)


if __name__ == "__main__":
    entries = load_knowledge_base()
    print(f"共加载 {len(entries)} 条知识条目：")
    for entry in entries:
        print(f"  - [{entry['category']}] {entry['title']}")
