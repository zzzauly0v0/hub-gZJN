"""
问答 Skill 模块 - 仅基于 SKILL.md 内容的问答系统
支持 DeepSeek 真实模型调用（通过 langchain-deepseek）和模拟 LLM 两种模式
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ============================================================
# DeepSeek 配置
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")  # 填入你的 API Key（格式：sk-xxx）
DEEPSEEK_MODEL = "deepseek-chat"  # 可选：deepseek-chat / deepseek-reasoner

# 是否强制使用模拟模式（True=始终模拟，False=有 API Key 时用真实模型）
FORCE_SIMULATE = False

# ============================================================
# 全局状态
# ============================================================
_skill_file_path: str = "SKILL.md"
_skill_content: str = ""
_skill_qa_pairs: List[Dict[str, str]] = []
_llm_quality: float = 0.6
_llm_client = None


def load_skill_md(file_path: str = "SKILL.md") -> str:
    """
    加载 SKILL.md 文件内容，并解析出 Q&A 对。

    Args:
        file_path: SKILL.md 文件路径

    Returns:
        SKILL.md 的完整文本内容
    """
    global _skill_file_path, _skill_content, _skill_qa_pairs
    _skill_file_path = file_path
    _skill_content = Path(file_path).read_text(encoding="utf-8")
    _skill_qa_pairs = _parse_skill_qa(_skill_content)
    return _skill_content


def _parse_skill_qa(content: str) -> List[Dict[str, str]]:
    """
    从 SKILL.md 内容中解析 Q&A 对。
    格式：### Q: 问题\nA: 答案

    Args:
        content: SKILL.md 文本内容

    Returns:
        Q&A 对列表，每项包含 question 和 answer
    """
    qa_pairs = []
    # 匹配 ### Q: ... 和 A: ... 的模式
    pattern = r"###\s*Q:\s*(.+?)\nA:\s*(.+?)(?=\n###\s*Q:|\n##\s*[^#]|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)
    for question, answer in matches:
        qa_pairs.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return qa_pairs


def get_skill_qa_pairs() -> List[Dict[str, str]]:
    """获取当前已解析的 SKILL.md Q&A 对列表"""
    return _skill_qa_pairs


def get_skill_content() -> str:
    """获取当前加载的 SKILL.md 原始内容"""
    return _skill_content


def _get_llm_client():
    """获取或创建 DeepSeek LLM 客户端（延迟初始化）"""
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    api_key = DEEPSEEK_API_KEY
    if not api_key or FORCE_SIMULATE:
        return None

    try:
        from langchain_deepseek import ChatDeepSeek
        _llm_client = ChatDeepSeek(
            model=DEEPSEEK_MODEL,
            api_key=api_key,
            temperature=0.1,
            max_tokens=1024,
        )
        print(f"[LLM] 已初始化 DeepSeek 模型: {DEEPSEEK_MODEL}")
        return _llm_client
    except Exception as e:
        print(f"[LLM] DeepSeek 初始化失败，将使用模拟模式: {e}")
        return None


def is_using_real_model() -> bool:
    """检查当前是否使用真实 DeepSeek 模型"""
    return _get_llm_client() is not None


def build_system_prompt(skill_content: str) -> str:
    """
    构建仅包含 SKILL.md 内容的 system prompt。
    大模型只能根据 SKILL.md 的知识来回答问题。

    Args:
        skill_content: SKILL.md 的完整文本

    Returns:
        完整的 system prompt
    """
    # 提取「知识内容」部分（## 知识内容 之后的所有内容）
    knowledge_section = ""
    match = re.search(r"## 知识内容\s*\n(.*)", skill_content, re.DOTALL)
    if match:
        knowledge_section = match.group(1).strip()
    else:
        knowledge_section = "（暂无知识内容）"

    prompt = f"""你是一个专业的知识库问答助手。你只能根据以下 SKILL 文件中的知识内容回答问题。

## 回答规则
1. 你的回答必须严格基于下方知识内容中的信息，不得添加任何额外信息
2. 如果知识内容中没有相关信息，你必须回答："抱歉，当前知识库中没有相关信息"
3. 回答时要包含知识内容中该条目的所有关键事实（人名、时间、技术术语等）
4. 绝对不要编造知识内容中不存在的信息
5. 回答要完整，涵盖知识内容中该主题的全部要点

## 知识内容（SKILL.md）

{knowledge_section}
"""
    return prompt


def _tokenize(text: str) -> set:
    """将文本分词：中文 2-gram 滑动窗口 + 英文整词，不包含单字中文"""
    text = text.lower()
    tokens = set(re.findall(r"[a-zA-Z]+[a-zA-Z0-9]*", text))
    zh_chars = re.findall(r"[一-鿿]+", text)
    for seg in zh_chars:
        for i in range(len(seg) - 1):
            tokens.add(seg[i:i + 2])
    return tokens


def set_llm_quality(quality: float):
    """设置模拟 LLM 的回答质量（0.0-1.0，仅在模拟模式下生效）"""
    global _llm_quality
    _llm_quality = quality


def call_deepseek(system_prompt: str, user_question: str) -> str:
    """
    调用真实 DeepSeek 模型获取回答。

    Args:
        system_prompt: system prompt（仅含 SKILL.md 知识内容）
        user_question: 用户问题

    Returns:
        DeepSeek 模型的回答文本
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    client = _get_llm_client()
    if client is None:
        raise RuntimeError("DeepSeek 客户端未初始化，请检查 API Key 配置")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question),
    ]
    response = client.invoke(messages)
    return response.content


def simulate_llm_response(user_question: str, qa_pairs: List[Dict[str, str]] = None) -> str:
    """
    模拟 LLM 回答（fallback 模式）。
    通过关键词匹配从 SKILL.md 的 Q&A 对中检索最相关的回答。

    Args:
        user_question: 用户问题
        qa_pairs: SKILL.md 中解析出的 Q&A 对列表

    Returns:
        模拟的 LLM 回答
    """
    if qa_pairs is None:
        qa_pairs = _skill_qa_pairs

    if not qa_pairs:
        return "抱歉，当前知识库中没有相关信息"

    question_tokens = _tokenize(user_question)
    best_match = None
    best_score = 0

    for qa in qa_pairs:
        entry_text = f"{qa['question']} {qa['answer']}"
        entry_tokens = _tokenize(entry_text)
        overlap = question_tokens & entry_tokens
        title_tokens = _tokenize(qa['question'])
        title_overlap = question_tokens & title_tokens
        score = len(overlap) + len(title_overlap) * 2

        if score > best_score:
            best_score = score
            best_match = qa['answer'].strip()

    if best_score < 3 or best_match is None:
        return "抱歉，当前知识库中没有相关信息"

    # 模拟 LLM 回答不完整（根据 _llm_quality 截断）
    if _llm_quality < 1.0 and best_match:
        sentences = re.split(r'[。！？]', best_match)
        sentences = [s.strip() for s in sentences if s.strip()]
        keep_count = max(1, int(len(sentences) * _llm_quality))
        truncated = '。'.join(sentences[:keep_count])
        if truncated and not truncated.endswith('。'):
            truncated += '。'
        return truncated

    return best_match


def ask_question(question: str) -> str:
    """
    向问答 Skill 提问。仅使用 SKILL.md 中的知识内容回答。
    优先使用真实 DeepSeek 模型，无 API Key 时降级为模拟模式。

    调用前需先调用 load_skill_md() 加载 SKILL.md。

    Args:
        question: 用户问题

    Returns:
        LLM 生成的回答（仅基于 SKILL.md 内容）
    """
    if not _skill_content:
        load_skill_md()

    system_prompt = build_system_prompt(_skill_content)

    # 优先使用真实 DeepSeek 模型
    client = _get_llm_client()
    if client is not None:
        try:
            return call_deepseek(system_prompt, question)
        except Exception as e:
            print(f"[LLM] DeepSeek 调用失败，降级为模拟模式: {e}")

    # Fallback: 模拟模式（基于 SKILL.md Q&A 对匹配）
    return simulate_llm_response(question, _skill_qa_pairs)


if __name__ == "__main__":
    load_skill_md()

    mode = "DeepSeek 真实模型" if is_using_real_model() else "模拟模式"
    print(f"[模式] {mode}")
    print(f"[SKILL.md] 已加载 {len(_skill_qa_pairs)} 条 Q&A 知识")
    print()

    test_questions = [
        "Python是什么编程语言？",
        "Git的主要作用是什么？",
        "Docker解决了什么问题？",
        "机器学习的三大类型是什么？",
        "什么是RESTful API？",
        "关系型数据库和非关系型数据库有什么区别？",
    ]

    for q in test_questions:
        answer = ask_question(q)
        print(f"Q: {q}")
        print(f"A: {answer}")
        print()
