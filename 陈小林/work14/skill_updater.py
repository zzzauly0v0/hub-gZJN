"""
SKILL.md 自动更新模块

当问答系统回答出错时，从 knowledge_base.md 中查找相关知识，
并将其写入（新增或替换）SKILL.md 的「知识内容」部分，实现自我进化。
"""

import re
from pathlib import Path
from typing import List, Dict, Optional

from knowledge_loader import load_knowledge_base
from qa_skill import load_skill_md, get_skill_content, get_skill_qa_pairs

SKILL_FILE_PATH = "SKILL.md"
KNOWLEDGE_BASE_PATH = "knowledge_base.md"

# 常见的中文功能词/双字停用词，在匹配时应忽略
_STOP_WORDS = {"什么", "哪些", "如何", "怎么", "为什么", "是不是", "有没有",
               "问题", "作用", "主要", "介绍", "分别", "其中", "可以",
               "是否", "能否", "还是", "以及", "或者", "关于", "它的",
               "它是", "它有", "它们", "创建", "解决", "区别"}


def _topic_tokenize(text: str) -> set:
    """
    对文本进行主题级分词：
    - 中文：滑动窗口 2-gram（每个相邻 2 字符对），避免单字功能词和整句粘连
    - 英文：整词匹配
    过滤停用词后返回 token 集合。
    """
    text = text.lower()
    # 英文单词
    tokens = set(re.findall(r"[a-zA-Z]+[a-zA-Z0-9]*", text))
    # 中文 2-gram 滑动窗口
    zh_chars = re.findall(r"[一-鿿]+", text)
    for seg in zh_chars:
        for i in range(len(seg) - 1):
            tokens.add(seg[i:i + 2])
    return tokens - _STOP_WORDS


def find_relevant_entries(
    question: str,
    entries: List[Dict[str, str]],
    top_k: int = 1,
) -> List[Dict[str, str]]:
    """
    根据失败问题，从知识库条目中找出最相关的条目。

    Args:
        question: 回答失败的问题
        entries: knowledge_base.md 中解析出的知识条目列表
        top_k: 返回的最相关条目数量

    Returns:
        按相关度降序排列的知识条目列表
    """
    question_tokens = _topic_tokenize(question)
    scored = []

    for entry in entries:
        entry_q = entry['question']
        entry_q_tokens = _topic_tokenize(entry_q)
        entry_a_tokens = _topic_tokenize(entry['answer'])
        title_tokens = _topic_tokenize(entry['title'])

        q_overlap = question_tokens & entry_q_tokens
        a_overlap = question_tokens & entry_a_tokens
        title_overlap = question_tokens & title_tokens
        score = len(title_overlap) * 5 + len(q_overlap) * 3 + len(a_overlap)

        # 问题子串匹配强信号：测试问题包含 KB 问题（或反之），通常意味着讨论同一主题
        if entry_q and (entry_q in question or question in entry_q):
            score += 15

        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:top_k] if _ > 0]


def _parse_skill_sections(content: str) -> tuple:
    """
    将 SKILL.md 内容拆分为：头部（## 知识内容 之前）和 Q&A 块列表。

    Returns:
        (header, qa_blocks)
        - header: "## 知识内容" 之前的所有内容（含 "## 知识内容\n" 本身）
        - qa_blocks: 每个 Q&A 块的 (question, answer) 列表
    """
    # 找到 "## 知识内容" 的位置
    match = re.search(r"^## 知识内容\s*\n", content, re.MULTILINE)
    if not match:
        # 如果没有「知识内容」section，用整个内容作为 header
        return content, []

    header_end = match.end()
    header = content[:header_end]
    qa_section = content[header_end:]

    # 按 ### Q: 拆分 Q&A 块
    qa_blocks = []
    # 找到所有 ### Q: 的位置
    qa_pattern = re.compile(r"^###\s*Q:\s*(.+?)\nA:\s*(.*?)(?=^###\s*Q:|\Z)",
                            re.MULTILINE | re.DOTALL)
    for m in qa_pattern.finditer(qa_section):
        q = m.group(1).strip()
        a = m.group(2).strip()
        qa_blocks.append({"question": q, "answer": a})

    return header, qa_blocks


def _reassemble_skill(header: str, qa_blocks: List[Dict[str, str]]) -> str:
    """
    将 header 和 Q&A 块列表重新组装为 SKILL.md 内容。

    Args:
        header: 头部内容（以 "## 知识内容\n" 结尾）
        qa_blocks: Q&A 块列表

    Returns:
        组装后的 SKILL.md 文本
    """
    parts = [header]
    for i, block in enumerate(qa_blocks):
        if i > 0:
            parts.append("\n")  # Q&A 块之间的空行
        parts.append(f"\n### Q: {block['question']}\nA: {block['answer']}\n")
    return "".join(parts)


def _find_existing_qa_index(question: str, qa_blocks: List[Dict[str, str]]) -> Optional[int]:
    """
    在现有 Q&A 块中，找到与给定问题主题相同的条目索引。
    仅当两个问题讨论的是同一主题时才返回匹配。

    判断标准：主题级关键词重叠数 >= 2

    Args:
        question: 待匹配的问题
        qa_blocks: Q&A 块列表

    Returns:
        匹配的索引，无匹配时返回 None
    """
    q_tokens = _topic_tokenize(question)
    best_idx = None
    best_score = 0

    for i, block in enumerate(qa_blocks):
        block_tokens = _topic_tokenize(block['question'])
        overlap = q_tokens & block_tokens
        score = len(overlap)
        if score > best_score:
            best_score = score
            best_idx = i

    # 至少需要 2 个主题级 token 重叠才认为是同一条
    return best_idx if best_score >= 2 else None


def update_skill_md(
    failed_question: str,
    knowledge_entries: List[Dict[str, str]],
    skill_file_path: str = SKILL_FILE_PATH,
) -> Dict:
    """
    根据失败问题和匹配到的知识条目，更新 SKILL.md。

    - 如果 SKILL.md 中已有该问题的不完整条目 → 替换为完整答案
    - 如果 SKILL.md 中没有该问题 → 追加新的 Q&A 条目

    Args:
        failed_question: 回答失败的问题
        knowledge_entries: 从 knowledge_base.md 中找到的相关知识条目
        skill_file_path: SKILL.md 文件路径

    Returns:
        更新结果字典
    """
    if not knowledge_entries:
        return {"updated": False, "action": "no_match", "entries_added": 0, "questions": []}

    # 解析当前 SKILL.md 为 header + Q&A 块
    skill_content = Path(skill_file_path).read_text(encoding="utf-8")
    header, qa_blocks = _parse_skill_sections(skill_content)

    updated_questions = []
    action = "appended"

    for entry in knowledge_entries:
        existing_idx = _find_existing_qa_index(entry['question'], qa_blocks)
        new_block = {"question": entry['question'], "answer": entry['answer']}

        if existing_idx is not None:
            # 替换已有的不完整条目
            qa_blocks[existing_idx] = new_block
            action = "replaced"
            updated_questions.append(f"[替换] {entry['question']}")
        else:
            # 追加新条目
            qa_blocks.append(new_block)
            action = "appended"
            updated_questions.append(f"[新增] {entry['question']}")

    # 重新组装并写入
    new_content = _reassemble_skill(header, qa_blocks)
    Path(skill_file_path).write_text(new_content, encoding="utf-8")

    # 重新加载全局状态
    load_skill_md(skill_file_path)

    return {
        "updated": True,
        "action": action,
        "entries_added": len(updated_questions),
        "questions": updated_questions,
    }


def update_from_failures(
    failures: List[Dict],
    skill_file_path: str = SKILL_FILE_PATH,
    knowledge_base_path: str = KNOWLEDGE_BASE_PATH,
) -> Dict:
    """
    批量处理所有失败用例，从知识库找到相关知识并更新 SKILL.md。

    Args:
        failures: 失败用例列表，每项需包含 question 字段
        skill_file_path: SKILL.md 路径
        knowledge_base_path: knowledge_base.md 路径

    Returns:
        批量更新结果摘要
    """
    kb_entries = load_knowledge_base(knowledge_base_path)
    details = []
    updated_count = 0

    for failure in failures:
        question = failure["question"]
        relevant = find_relevant_entries(question, kb_entries, top_k=1)

        if relevant:
            result = update_skill_md(question, relevant, skill_file_path)
            details.append({"question": question, **result})
            if result["updated"]:
                updated_count += 1
        else:
            details.append({"question": question, "updated": False, "action": "no_match"})

    return {
        "total_failures": len(failures),
        "updated_count": updated_count,
        "details": details,
    }


if __name__ == "__main__":
    load_skill_md()
    print("[更新前] SKILL.md Q&A 数量:", len(get_skill_qa_pairs()))
    for qa in get_skill_qa_pairs():
        print(f"  - Q: {qa['question'][:30]}...")

    mock_failures = [
        {"question": "Git是什么？谁创建了Git？"},
        {"question": "Docker是什么？它解决了什么问题？"},
    ]

    print("\n开始更新 SKILL.md...")
    result = update_from_failures(mock_failures)
    print(f"更新结果: {result['updated_count']}/{result['total_failures']} 条已更新")
    for d in result["details"]:
        print(f"  {d['question'][:30]}... -> {d.get('action', 'no_match')}")

    print("\n[更新后] SKILL.md Q&A 数量:", len(get_skill_qa_pairs()))
    for qa in get_skill_qa_pairs():
        print(f"  - Q: {qa['question'][:30]}...")
