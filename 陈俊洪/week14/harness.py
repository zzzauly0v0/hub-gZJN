"""
评估台（本任务新增）：给定一份 Skill 内容，测它的 **准确率 / token / 延迟**。

复用原项目的两块代码，保证口径一致、结论可与原实验对齐：
  src/agent.py     的 SYSTEM_TEMPLATE / SKILLS_SECTION_TEMPLATE —— 系统提示逐字一致
  src/evaluator.py 的 Evaluator                                 —— 判分规则逐字一致

与原项目 CustomerServiceAgent 的唯一差别：
  它从 skills/ 目录读文件，我们直接把 Skill 内容以字符串传入（变体不落地、互不污染），
  并且记录每次调用的真实 token 用量。
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent as project_agent          # noqa: E402  复用系统提示模板
from evaluator import Evaluator        # noqa: E402  复用判分规则

from llm import LLM, TokenMeter, Usage  # noqa: E402
from metrics import count_tokens, estimate_cost_usd  # noqa: E402


def build_system_prompt(skills: dict[str, str]) -> str:
    """
    用原项目的模板拼系统提示。逐字复用是刻意的：
    只要模板一致，本实验测出的 token 数就能直接对应原项目的真实开销。
    """
    if not skills:
        section = "（暂无技能文档，请依据通用客服原则回答）"
    else:
        parts = [f"### 技能：{n}\n{c}" for n, c in sorted(skills.items())]
        section = project_agent.SKILLS_SECTION_TEMPLATE.format(
            count=len(skills),
            skills_content="\n\n---\n\n".join(parts),
        )
    return project_agent.SYSTEM_TEMPLATE.format(skills_section=section)


# ── 答案缓存 ─────────────────────────────────────────────────────────────────

class AnswerCache:
    """
    (model, system, user) → (答案, 用量) 的磁盘缓存。

    temperature=0 下同一输入的答案本就应当稳定，缓存让"改报告不用重花钱"，
    也让评测可复现。**token 数照实取缓存里记录的真实 usage**，不是估算；
    只有延迟统计会跳过缓存命中（见 EvalResult.live_calls）。
    """

    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        if enabled and path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    @staticmethod
    def key(model: str, system: str, user: str) -> str:
        h = hashlib.sha256()
        h.update(model.encode()); h.update(b"\x00")
        h.update(system.encode()); h.update(b"\x00")
        h.update(user.encode())
        return h.hexdigest()[:32]

    def get(self, k: str) -> tuple[str, Usage] | None:
        if not self.enabled:
            return None
        with self._lock:
            rec = self._data.get(k)
        if not rec:
            return None
        return rec["answer"], Usage(
            prompt_tokens=rec["prompt_tokens"],
            completion_tokens=rec["completion_tokens"],
            total_tokens=rec["total_tokens"],
            latency_s=0.0,      # 缓存命中不计延迟
        )

    def put(self, k: str, answer: str, u: Usage):
        if not self.enabled:
            return
        with self._lock:
            self._data[k] = {
                "answer": answer,
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            }

    def flush(self):
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8"
            )


# ── 评估结果 ─────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    variant: str
    label: str
    split: str
    total: int = 0
    correct: int = 0
    doc_tokens: int = 0
    system_prompt_tokens: int = 0
    live_calls: int = 0
    live_latency_s: float = 0.0
    meter: TokenMeter = field(default_factory=TokenMeter)
    per_question: dict[str, dict] = field(default_factory=dict)
    by_category: dict[str, dict] = field(default_factory=dict)
    fail_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return round(self.correct / self.total, 4) if self.total else 0.0

    @property
    def avg_prompt_tokens(self) -> float:
        return round(self.meter.prompt_tokens / self.total, 1) if self.total else 0.0

    @property
    def avg_total_tokens(self) -> float:
        return round(self.meter.total_tokens / self.total, 1) if self.total else 0.0

    @property
    def avg_live_latency_s(self) -> float:
        """只统计真实发起的调用；缓存命中不参与，避免把延迟压成 0。"""
        return round(self.live_latency_s / self.live_calls, 2) if self.live_calls else 0.0

    def failures(self) -> list[dict]:
        return [v for v in self.per_question.values() if not v["correct"]]

    def passes(self) -> list[dict]:
        return [v for v in self.per_question.values() if v["correct"]]

    def to_dict(self, model: str) -> dict:
        return {
            "variant": self.variant,
            "label": self.label,
            "split": self.split,
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "doc_tokens": self.doc_tokens,
            "system_prompt_tokens": self.system_prompt_tokens,
            "avg_prompt_tokens": self.avg_prompt_tokens,
            "avg_completion_tokens": round(self.meter.completion_tokens / self.total, 1) if self.total else 0.0,
            "avg_total_tokens": self.avg_total_tokens,
            "sum_prompt_tokens": self.meter.prompt_tokens,
            "sum_completion_tokens": self.meter.completion_tokens,
            "sum_total_tokens": self.meter.total_tokens,
            "avg_live_latency_s": self.avg_live_latency_s,
            "live_calls": self.live_calls,
            "cost_usd_per_100q": round(
                estimate_cost_usd(
                    model,
                    self.meter.prompt_tokens,
                    self.meter.completion_tokens,
                ) / self.total * 100, 5
            ) if self.total else 0.0,
            "by_category": self.by_category,
            "fail_reasons": self.fail_reasons,
        }


# ── 评估台 ───────────────────────────────────────────────────────────────────

class SkillHarness:
    def __init__(
        self,
        llm: LLM,
        evaluator: Evaluator,
        cache: AnswerCache | None = None,
        concurrency: int = 6,
        max_answer_tokens: int = 400,
    ):
        self.llm = llm
        self.evaluator = evaluator
        self.cache = cache
        self.concurrency = concurrency
        self.max_answer_tokens = max_answer_tokens

    def answer_one(self, system: str, question: str, meter: TokenMeter) -> tuple[str, Usage, bool]:
        """返回 (答案, 用量, 是否缓存命中)。"""
        k = AnswerCache.key(self.llm.model, system, question) if self.cache else ""
        if self.cache:
            hit = self.cache.get(k)
            if hit:
                answer, usage = hit
                meter.add(usage)
                return answer, usage, True
        answer, usage = self.llm.chat(
            system, question, max_tokens=self.max_answer_tokens, meter=meter
        )
        if self.cache:
            self.cache.put(k, answer, usage)
        return answer, usage, False

    def evaluate(
        self,
        skills: dict[str, str],
        question_ids: list[int],
        variant: str,
        label: str,
        split: str,
        progress: bool = True,
    ) -> EvalResult:
        """对指定题号集合评估一份 Skill（并发跑题，逐题记录 token）。"""
        system = build_system_prompt(skills)
        doc_tokens = sum(count_tokens(c) for c in skills.values())
        res = EvalResult(
            variant=variant, label=label, split=split,
            doc_tokens=doc_tokens,
            system_prompt_tokens=count_tokens(system),
        )
        lock = threading.Lock()

        def work(qid: int):
            q = self.evaluator.questions[qid]
            answer, usage, cached = self.answer_one(system, q["question"], res.meter)
            ok, reason = self.evaluator.evaluate_answer(answer, qid)
            with lock:
                res.total += 1
                res.correct += int(ok)
                if not cached:
                    res.live_calls += 1
                    res.live_latency_s += usage.latency_s
                cat = q["category"]
                res.by_category.setdefault(cat, {"total": 0, "correct": 0})
                res.by_category[cat]["total"] += 1
                res.by_category[cat]["correct"] += int(ok)
                if not ok:
                    tag = _reason_tag(reason)
                    res.fail_reasons[tag] = res.fail_reasons.get(tag, 0) + 1
                res.per_question[str(qid)] = {
                    "id": qid,
                    "category": cat,
                    "question": q["question"],
                    "answer": answer,
                    "correct": ok,
                    "fail_reason": "" if ok else reason,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                }
                if progress:
                    done = res.total
                    print(f"\r    [{variant}/{split}] {done}/{len(question_ids)} "
                          f"正确 {res.correct}", end="", flush=True)

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            list(ex.map(work, question_ids))
        if progress:
            print(f"\r    [{variant}/{split}] {res.total}/{len(question_ids)} "
                  f"正确 {res.correct} → 准确率 {res.accuracy:.1%}"
                  f"  avg_prompt={res.avg_prompt_tokens:.0f}tok", flush=True)

        for c in res.by_category.values():
            c["accuracy"] = round(c["correct"] / c["total"], 3)
        if self.cache:
            self.cache.flush()
        return res


def _reason_tag(reason: str) -> str:
    """把评估器的失败原因归到三类互斥标签（原项目 ARCHITECTURE 第四章的口径）。"""
    if "推脱" in reason:
        return "Agent推脱"
    if "缺少关键词" in reason:
        return "缺少关键词"
    if "禁止词" in reason:
        return "出现禁止词"
    return "其他"
