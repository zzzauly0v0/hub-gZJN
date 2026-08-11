"""
token 计数、成本估算、Pareto 前沿（本任务新增）。

两种 token 数各有用途，报告里都会出现：
  doc_tokens        —— 用 tiktoken 离线数 Skill 文档本身，确定性、不花钱、可反复算，
                       适合做"压缩了多少"的口径。
  avg_prompt_tokens —— 来自 API 真实 usage，包含系统提示模板 + 问题，
                       适合算"每题真实花多少钱"。
"""

from __future__ import annotations

# ── token 计数 ───────────────────────────────────────────────────────────────

_ENC = None
_ENC_TRIED = False
DOC_ENCODING = "cl100k_base"


def _encoder():
    global _ENC, _ENC_TRIED
    if not _ENC_TRIED:
        _ENC_TRIED = True
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding(DOC_ENCODING)
        except Exception:
            _ENC = None
    return _ENC


def count_tokens(text: str) -> int:
    """
    离线 token 估算，口径固定为 cl100k_base。
    注意：这是**跨模型可比的代理指标**，不等于 DeepSeek/Gemini 的真实分词数
    （中文尤其会偏大）。用它比较"变体之间压缩了多少"是可靠的；
    要绝对值就看 avg_prompt_tokens（真实 usage）。
    tiktoken 缺失时退化为经验公式：中文按 1 字≈1 token，其余按 4 字符≈1 token。
    """
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return cjk + max(1, (len(text) - cjk) // 4)


# ── 成本 ─────────────────────────────────────────────────────────────────────

# 美元 / 百万 token，(input, output)。**近似值，会随官方调价过期**，
# 仅用于给出数量级；要精确成本请按当期官网价改这张表。
PRICING = {
    "deepseek-chat":         (0.27, 1.10),
    "gemini-2.5-flash":      (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gpt-4o-mini":           (0.15, 0.60),
}
DEFAULT_PRICE = (0.30, 1.00)


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = PRICING.get(model, DEFAULT_PRICE)
    return prompt_tokens / 1e6 * pin + completion_tokens / 1e6 * pout


# ── Pareto 前沿 ──────────────────────────────────────────────────────────────

def pareto_front(items: list[dict], gain_key: str, cost_key: str) -> list[str]:
    """
    双目标（gain 越大越好 / cost 越小越好）的非支配集合，返回变体名列表。
    A 支配 B ⇔ A 的 gain ≥ B 且 cost ≤ B，且至少一维严格更优。
    """
    front = []
    for a in items:
        dominated = any(
            b is not a
            and b[gain_key] >= a[gain_key]
            and b[cost_key] <= a[cost_key]
            and (b[gain_key] > a[gain_key] or b[cost_key] < a[cost_key])
            for b in items
        )
        if not dominated:
            front.append(a["variant"])
    return front


def pct_delta(new: float, old: float) -> float:
    """相对变化百分比；old 为 0 时返回 0，避免除零。"""
    if not old:
        return 0.0
    return round((new - old) / old * 100, 1)
