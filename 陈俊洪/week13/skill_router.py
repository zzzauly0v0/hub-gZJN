"""
Skill 路由 — 只看 Level 1 元信息，决定该激活哪个 skill

教学重点：
  1. 路由只依赖 Level 1（name + description），这正是元信息常驻上下文的目的：
     用极少 token 就能判断"这句话要不要用某个能力"。
  2. 两段式路由（对齐 heartbeat_parser 的"正则初筛 + LLM 判断"）：
       Step 1  关键词初筛：description 与 query 的重叠词，零成本、可解释
       Step 2  可选 LLM 兜底：初筛不确定时，让 LLM 在候选里做最终选择
  3. 命中后才交给 SkillRegistry.load() 做 Level 2，避免一次性把所有 skill 正文塞进上下文。

使用方式：
  from src.skill_loader import SkillRegistry
  from src.skill_router import SkillRouter
  router = SkillRouter(SkillRegistry())
  decision = router.route("帮我写个 commit message")
  if decision.skill_name:
      ...  # 激活该 skill
"""

import re
from dataclasses import dataclass

from src.skill_loader import SkillRegistry, SkillMetadata

# 用于关键词初筛的停用词：这些词区分度低，不参与匹配
_STOPWORDS = {
    "的", "了", "我", "你", "他", "她", "它", "们", "个", "帮", "请", "一下", "一个",
    "怎么", "如何", "什么", "这", "那", "是", "在", "和", "与", "生成", "写", "做",
    "a", "an", "the", "to", "of", "for", "and", "or", "help", "me", "please", "write",
}

# 从中英文文本里抽取"词"：连续 ASCII 词 或 单个汉字
_TOKEN = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]")

# 阈值：初筛得分高于此值直接命中，低于此值视为无匹配，中间地带交给 LLM
_HIGH_CONFIDENCE = 0.34
_LOW_CONFIDENCE = 0.08


@dataclass
class RouteDecision:
    skill_name: str | None       # 命中的 skill；None 表示无需 skill
    score: float                 # 初筛得分（0~1）
    method: str                  # "keyword" | "llm" | "none"
    reason: str = ""
    candidates: list[str] = None


def _tokenize(text: str) -> set[str]:
    toks = _TOKEN.findall(text.lower())
    return {t for t in toks if t not in _STOPWORDS and len(t) >= 1}


class SkillRouter:
    def __init__(self, registry: SkillRegistry, use_llm: bool = False):
        self.registry = registry
        self.use_llm = use_llm  # 是否允许 LLM 兜底（离线/无 key 时置 False）

    def _keyword_scores(self, query: str, metas: list[SkillMetadata]) -> list[tuple[SkillMetadata, float]]:
        q_tokens = _tokenize(query)
        scored: list[tuple[SkillMetadata, float]] = []
        for m in metas:
            # 用 name + description 组成 skill 的"关键词画像"
            profile = _tokenize(f"{m.name} {m.description}")
            if not profile or not q_tokens:
                scored.append((m, 0.0))
                continue
            overlap = q_tokens & profile
            # 得分 = 命中词数 / query 词数（衡量 query 有多少被这个 skill 覆盖）
            score = len(overlap) / max(len(q_tokens), 1)
            scored.append((m, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def route(self, query: str) -> RouteDecision:
        metas = self.registry.list_metadata()
        if not metas:
            return RouteDecision(None, 0.0, "none", "没有已注册的 skill")

        scored = self._keyword_scores(query, metas)
        best, best_score = scored[0]
        candidates = [m.name for m, s in scored if s > 0]

        # 高置信：关键词初筛直接命中
        if best_score >= _HIGH_CONFIDENCE:
            return RouteDecision(
                best.name, round(best_score, 3), "keyword",
                f"query 与「{best.name}」描述关键词重叠度高", candidates,
            )

        # 低置信：几乎无重叠，判定为无需 skill
        if best_score < _LOW_CONFIDENCE:
            return RouteDecision(None, round(best_score, 3), "none",
                                 "无 skill 与 query 明显相关", candidates)

        # 中间地带：可选 LLM 兜底
        if self.use_llm:
            picked = self._llm_route(query, metas)
            if picked:
                return RouteDecision(picked, round(best_score, 3), "llm",
                                     "关键词不确定，LLM 判定命中", candidates)
            return RouteDecision(None, round(best_score, 3), "llm",
                                 "关键词不确定，LLM 判定无需 skill", candidates)

        # 未启用 LLM：中间地带保守地取初筛最优
        return RouteDecision(best.name, round(best_score, 3), "keyword",
                             "关键词初筛（中等置信，未启用 LLM 兜底）", candidates)

    def _llm_route(self, query: str, metas: list[SkillMetadata]) -> str | None:
        """LLM 只看 Level 1 元信息，在候选里选一个或返回 none。"""
        from src.llm_config import get_chat_client

        catalog = "\n".join(f"- {m.name}: {m.description}" for m in metas)
        sys_prompt = (
            "你是一个 skill 路由器。下面是可用 skill 的名称与用途。"
            "根据用户输入，只回复最匹配的一个 skill 名称；"
            "若没有任何 skill 合适，只回复 none。不要输出其他内容。\n\n"
            f"可用 skill：\n{catalog}"
        )
        try:
            client, model = get_chat_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
            )
            answer = (resp.choices[0].message.content or "").strip().lower()
        except Exception:
            return None  # LLM 不可用时安静降级为"无匹配"

        valid = {m.name.lower(): m.name for m in metas}
        for key, original in valid.items():
            if key in answer:
                return original
        return None
