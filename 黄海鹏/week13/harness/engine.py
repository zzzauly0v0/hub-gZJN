import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .skill_loader import SkillLoader, Skill
from .llm import LLMClient


@dataclass
class WordResult:
    word: str
    phonetic: str = ""
    pos: str = ""
    definition: str = ""
    examples: list[dict] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    skill_name: str = ""
    matched_by: str = ""          # trigger / english / chinese
    llm_generated: bool = False   # 是否由 LLM 生成


@dataclass
class TriggerMatch:
    skill: Skill
    word: str
    matched_pattern: str


class QueryEngine:
    """Harness 核心查询引擎。"""

    def __init__(self, loader: SkillLoader, llm: Optional[LLMClient] = None):
        self.loader = loader
        self.llm = llm

    async def query(self, text: str) -> list[WordResult]:
        text = text.strip()
        if not text:
            return []

        # 1) 自然语言触发匹配
        trigger_match = self._match_trigger(text)
        if trigger_match:
            # 懒加载：触发后才激活 skill，加载 data/ 和执行流程
            skill = self.loader.activate(trigger_match.skill.name)
            if not skill:
                return []

            # 查本地词库（已通过 activate 加载）
            results = self._lookup_word(trigger_match.word)
            if results:
                for r in results:
                    r.matched_by = "trigger"
                    r.skill_name = skill.name
                return results

            # 本地没有 → 尝试 LLM 生成
            if self.llm:
                return [await self._generate_via_llm(trigger_match.word, skill)]

            return [WordResult(
                word=trigger_match.word,
                definition=f"「{trigger_match.word}」尚未收录，且未配置 LLM。",
                skill_name=skill.name,
                matched_by="trigger",
            )]

        # 2) 中文搜索
        if re.search(r"[\u4e00-\u9fa5]", text):
            self._ensure_activated()
            return self._search_by_chinese(text)

        # 3) 英文搜索
        self._ensure_activated()
        return self._search_by_english(text)

    # ---- LLM 生成 ----

    async def _generate_via_llm(self, word: str, skill: Skill) -> WordResult:
        """
        用 skill 的 llm_prompt 模板调用 LLM 生成单词数据。

        skill.llm_prompt 来自 SKILL.md 的 ## LLM 生成指令 段，
        engine 只负责填充 {word} 占位符，调用 LLM，解析 JSON 结果。
        """
        if not skill.llm_prompt:
            return WordResult(
                word=word,
                definition=f"Skill「{skill.name}」未定义 LLM 生成指令，无法生成「{word}」。",
                skill_name=skill.name,
                matched_by="trigger",
            )

        # 用 skill 的 prompt 模板，填充单词
        prompt = skill.llm_prompt.replace("{word}", word)
        raw = self.llm.chat(prompt)  # type: ignore[union-attr]

        if not raw:
            return WordResult(
                word=word,
                definition=f"LLM 生成「{word}」失败，请稍后重试。",
                skill_name=skill.name,
                matched_by="trigger",
            )

        # 从 LLM 返回中解析 JSON（通用解析逻辑，与具体 skill 解耦）
        data = self._parse_json_from_llm(raw, word)
        if not data:
            return WordResult(
                word=word,
                definition=f"LLM 返回解析失败，原始内容: {raw[:200]}...",
                skill_name=skill.name,
                matched_by="trigger",
            )

        # 保存 JSON 到 skill 的 data/ 目录
        self._save_word_data(skill, data)

        # 写入内存索引，下次直接命中
        skill.word_index[word.lower()] = data

        return WordResult(
            word=data.get("word", word),
            phonetic=data.get("phonetic", ""),
            pos=data.get("pos", data.get("part_of_speech", "")),
            definition=self._extract_definition(data),
            examples=data.get("examples", [])[:3],
            synonyms=data.get("synonyms", [])[:8],
            skill_name=skill.name,
            matched_by="trigger",
            llm_generated=True,
        )

    @staticmethod
    def _parse_json_from_llm(raw: str, fallback_word: str) -> Optional[dict]:
        """从 LLM 返回的原始文本中提取 JSON 对象（与 skill 无关的通用逻辑）。"""
        # 提取 ```json ... ``` 或 ``` 代码块
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        json_str = m.group(1).strip() if m else raw

        # 提取第一个 { 到最后一个 }
        m = re.search(r"\{.*\}", json_str, re.DOTALL)
        if m:
            json_str = m.group(0)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"[Engine] JSON 解析失败，原始内容: {raw[:300]}")
            return None

    @staticmethod
    def _extract_definition(data: dict) -> str:
        """容错：LLM 可能返回 definition(字符串) 或 definitions(数组)。"""
        d = data.get("definition", "")
        if d and isinstance(d, str):
            return d
        defs = data.get("definitions", [])
        if isinstance(defs, list) and defs:
            return "；".join(str(x) for x in defs if x)
        return ""

    @staticmethod
    def _save_word_data(skill: Skill, data: dict):
        """将 LLM 生成的数据写入 JSON 文件。"""
        skill.data_dir.mkdir(parents=True, exist_ok=True)
        word = data.get("word", "unknown").lower()
        filepath = skill.data_dir / f"{word}.json"
        filepath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[Engine] 已保存: {filepath}")

    # ---- 辅助 ----

    def _ensure_activated(self):
        """确保所有 skill 都已激活（加载 data/），用于非触发类搜索。"""
        for s in self.loader.list_skills():
            if not s.activated:
                self.loader.activate(s.name)

    # ---- 触发匹配 ----

    def _match_trigger(self, text: str) -> TriggerMatch | None:
        for skill in self.loader.list_skills():
            for pattern in skill.trigger_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and match.group(1):
                    return TriggerMatch(
                        skill=skill,
                        word=match.group(1).lower(),
                        matched_pattern=pattern,
                    )
        return None

    # ---- 单词查询 ----

    def _lookup_word(self, word: str) -> list[WordResult]:
        word_lower = word.lower()
        results = []
        for skill in self.loader.list_skills():
            for w, data in skill.word_index.items():
                if w == word_lower or w.startswith(word_lower) or word_lower in w:
                    results.append(self._to_result(data, skill.name, "english"))
        return results

    def _search_by_english(self, word: str) -> list[WordResult]:
        results = self._lookup_word(word)
        if results:
            return results
        return [WordResult(
            word=word,
            definition=f"「{word}」未收录，当前词库: {', '.join(self._all_words())}。试试用自然语言触发 LLM 生成：如「给我做张 {word} 词的闪卡」",
            matched_by="english",
        )]

    def _search_by_chinese(self, query: str) -> list[WordResult]:
        results = []
        for skill in self.loader.list_skills():
            for data in skill.word_index.values():
                if self._match_chinese(data, query):
                    results.append(self._to_result(data, skill.name, "chinese"))
        if results:
            return results
        return [WordResult(
            word=query,
            definition=f"未找到与「{query}」相关的单词。当前词库: {', '.join(self._all_words())}",
            matched_by="chinese",
        )]

    def _all_words(self) -> list[str]:
        words = []
        for s in self.loader.list_skills():
            words.extend(s.word_index.keys())
        return sorted(words)

    @staticmethod
    def _match_chinese(data: dict, query: str) -> bool:
        fields = [data.get("definition", "")]
        fields.extend(ex.get("zh", "") for ex in data.get("examples", []))
        return any(query in f for f in fields)

    @staticmethod
    def _to_result(data: dict, skill_name: str, matched_by: str) -> WordResult:
        return WordResult(
            word=data.get("word", ""),
            phonetic=data.get("phonetic", ""),
            pos=data.get("pos", ""),
            definition=data.get("definition", ""),
            examples=data.get("examples", []),
            synonyms=data.get("synonyms", []),
            skill_name=skill_name,
            matched_by=matched_by,
        )
