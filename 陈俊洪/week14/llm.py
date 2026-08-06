"""
带 token 计量的 LLM 通道（本任务新增，不改动 src/ 下原有代码）。

为什么需要它：
  原 src/agent.py 直接 new OpenAI(...) 并丢弃 response.usage，
  而本任务的优化目标之一就是 **token 消耗**，必须拿到每次调用的真实用量。
  因此这里封装一层：调用行为与原项目一致（OpenAI 兼容接口、temperature=0），
  额外把 prompt/completion/total token 和延迟累计进 TokenMeter。

Provider 自动选择（按环境变量优先级）：
  DEEPSEEK_API_KEY → deepseek-chat（原项目默认）
  GEMINI_API_KEY   → gemini-2.5-flash（OpenAI 兼容端点）
  OPENAI_API_KEY   → gpt-4o-mini

使用方式：
  from llm import LLM
  llm = LLM()                      # 自动挑 provider
  text, usage = llm.chat("你是客服", "30天能退吗")
  print(llm.meter.summary())
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field

from openai import OpenAI


# ── Provider 表 ──────────────────────────────────────────────────────────────
# extra:    该 provider 需要的额外参数
# no_think: 关闭思考预算的参数。gemini 2.5 系列默认开 thinking，会额外烧 token 且
#           可能把 max_tokens 全用在思考上导致 content 为空；跑 Agent 应答时必须关，
#           这样 prompt token 数才真实反映 Skill 文档本身的开销。
# model / meta_model: 应答用便宜高配额的型号，写/优化 Skill 的元调用用强一点的型号。
# rpm:      免费档每分钟请求数上限，用于本地限流（宁可慢，不要 429 打断实验）。
PROVIDERS = {
    "deepseek": {
        "env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "meta_model": "deepseek-chat",
        "extra": {},
        "no_think": {},
        "rpm": 0,          # 0 = 不限流
    },
    "gemini": {
        "env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash-lite",   # 免费档 ~15 RPM
        "meta_model": "gemini-2.5-flash",   # 免费档 ~5 RPM，只用于少量元调用
        "extra": {},
        "no_think": {"reasoning_effort": "none"},
        "rpm": 13,
    },
    "openai": {
        "env": "OPENAI_API_KEY",
        "base_url": None,
        "model": "gpt-4o-mini",
        "meta_model": "gpt-4o-mini",
        "extra": {},
        "no_think": {},
        "rpm": 0,
    },
}

PROVIDER_ORDER = ("deepseek", "gemini", "openai")


def resolve_provider(name: str | None = None) -> str:
    """按环境变量挑一个可用 provider；name 显式指定时校验其 key 是否存在。"""
    if name:
        if name not in PROVIDERS:
            raise ValueError(f"未知 provider: {name}（可选 {list(PROVIDERS)}）")
        if not os.getenv(PROVIDERS[name]["env"]):
            raise RuntimeError(f"provider '{name}' 需要环境变量 {PROVIDERS[name]['env']}")
        return name
    for cand in PROVIDER_ORDER:
        if os.getenv(PROVIDERS[cand]["env"]):
            return cand
    envs = " / ".join(PROVIDERS[p]["env"] for p in PROVIDER_ORDER)
    raise RuntimeError(f"未检测到任何可用 API Key，请设置其中之一：{envs}")


# ── 用量计量 ─────────────────────────────────────────────────────────────────

@dataclass
class Usage:
    """单次调用的用量。thinking_tokens = total - prompt - completion（Gemini 等有思考预算的模型）。"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0

    @property
    def thinking_tokens(self) -> int:
        return max(0, self.total_tokens - self.prompt_tokens - self.completion_tokens)


@dataclass
class TokenMeter:
    """
    一组调用的累计用量。评估会用线程池并发跑题，`x += 1` 是读-改-写不是原子操作，
    所以累加统一加锁——否则并发下 token 总数会少算。
    """
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    retries: int = 0
    per_call: list[Usage] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, u: Usage):
        with self._lock:
            self.calls += 1
            self.prompt_tokens += u.prompt_tokens
            self.completion_tokens += u.completion_tokens
            self.total_tokens += u.total_tokens
            self.latency_s += u.latency_s
            self.per_call.append(u)

    def bump_retries(self, n: int = 1):
        with self._lock:
            self.retries += n

    def summary(self) -> dict:
        n = max(1, self.calls)
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_prompt_tokens": round(self.prompt_tokens / n, 1),
            "avg_completion_tokens": round(self.completion_tokens / n, 1),
            "avg_total_tokens": round(self.total_tokens / n, 1),
            "avg_latency_s": round(self.latency_s / n, 2),
            "retries": self.retries,
        }


# ── 限流 ─────────────────────────────────────────────────────────────────────

class RateLimiter:
    """
    进程内滑动窗口限流：保证任意 60s 内不超过 rpm 次调用。

    需要它是因为免费档配额很低（gemini flash 仅 5 RPM），并发跑题会瞬间打满，
    退避重试虽然能救回来但会白等很久。主动限流后实验一次跑通，不会中途炸掉。
    同一个 model 的多个 LLM 实例共享一把限流器（按 model 名注册）。
    """

    _registry: dict[str, "RateLimiter"] = {}
    _registry_lock = threading.Lock()

    @classmethod
    def get(cls, key: str, rpm: int) -> "RateLimiter | None":
        if not rpm:
            return None
        with cls._registry_lock:
            if key not in cls._registry:
                cls._registry[key] = cls(rpm)
            return cls._registry[key]

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._lock = threading.Lock()
        self._stamps: list[float] = []
        self._blocked_until = 0.0

    def acquire(self):
        while True:
            with self._lock:
                now = time.time()
                # 服务端明确让我们等（429 的 retryDelay）→ 全局阻塞到该时刻
                wait = self._blocked_until - now
                if wait <= 0:
                    self._stamps = [t for t in self._stamps if now - t < 60.0]
                    if len(self._stamps) < self.rpm:
                        self._stamps.append(now)
                        return
                    wait = 60.0 - (now - self._stamps[0]) + 0.05
            time.sleep(max(0.05, min(wait, 65.0)))

    def penalize(self, seconds: float):
        """收到 429 后，让所有线程一起等够服务端要求的时间。"""
        with self._lock:
            self._blocked_until = max(self._blocked_until, time.time() + seconds)


_RETRY_DELAY_RE = re.compile(r"[Rr]etry in ([0-9.]+)s")
_RETRY_INFO_RE = re.compile(r"'retryDelay':\s*'(\d+)s'")


def parse_retry_delay(msg: str, default: float = 20.0) -> float:
    """从 429 报文里抠出服务端建议的等待秒数（拿不到就用默认值）。"""
    for rx, scale in ((_RETRY_DELAY_RE, 1.0), (_RETRY_INFO_RE, 1.0)):
        m = rx.search(msg)
        if m:
            return min(90.0, float(m.group(1)) * scale + 1.0)
    return default


# ── LLM 通道 ─────────────────────────────────────────────────────────────────

class LLM:
    """
    OpenAI 兼容通道 + 用量计量 + 退避重试。

    think=False（默认，用于 Agent 应答）：关闭 provider 的思考预算，
      保证 token 数只反映"Skill 文档 + 问题 + 答案"，这是本实验要测的量。
    think=True（用于写 Skill / 优化 Skill 的元调用）：允许模型思考，元调用只有几次，
      开销可忽略，但质量差别明显。
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        think: bool = False,
        max_retries: int = 6,
    ):
        self.provider = resolve_provider(provider)
        cfg = PROVIDERS[self.provider]
        # think=True 的是写/优化 Skill 的元调用，默认用 meta_model（更强）
        default_model = cfg["meta_model"] if think else cfg["model"]
        self.model = model or os.getenv("SKILL_OPT_MODEL") or default_model
        self.think = think
        self.max_retries = max_retries
        self._cfg = cfg
        self.meter = TokenMeter()
        self.limiter = RateLimiter.get(f"{self.provider}:{self.model}", cfg.get("rpm", 0))

        kwargs = {"api_key": os.getenv(cfg["env"])}
        if cfg["base_url"]:
            kwargs["base_url"] = cfg["base_url"]
        self.client = OpenAI(**kwargs)

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 400,
        temperature: float = 0.0,
        meter: TokenMeter | None = None,
    ) -> tuple[str, Usage]:
        """
        单轮调用（system + user，不带历史——与原项目 agent.answer() 的无状态设计一致）。
        返回 (文本, 本次用量)，并把用量累计进 self.meter 以及可选的额外 meter。
        """
        extra = dict(self._cfg["extra"])
        if not self.think:
            extra.update(self._cfg["no_think"])

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if self.limiter:
                    self.limiter.acquire()
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                )
                latency = time.time() - t0
                raw = resp.choices[0].message.content
                u = resp.usage
                usage = Usage(
                    prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                    total_tokens=getattr(u, "total_tokens", 0) or 0,
                    latency_s=latency,
                )
                self.meter.add(usage)
                if meter is not None:
                    meter.add(usage)

                text = (raw or "").strip()
                if not text:
                    # 思考预算吃光了 max_tokens 的典型症状：下一轮强制关思考重试
                    last_err = RuntimeError("模型返回空内容")
                    extra.update(self._cfg["no_think"])
                    self.meter.bump_retries()
                    if meter is not None:
                        meter.bump_retries()
                    continue
                return text, usage

            except Exception as e:  # 限流 / 网络 / 服务端 5xx 一律退避重试
                last_err = e
                self.meter.bump_retries()
                if meter is not None:
                    meter.bump_retries()
                if attempt >= self.max_retries - 1:
                    break
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate" in msg.lower():
                    # 配额打满：按服务端给的 retryDelay 全局等待，别让其他线程继续撞墙
                    delay = parse_retry_delay(msg)
                    if self.limiter:
                        self.limiter.penalize(delay)
                    time.sleep(delay)
                else:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"LLM 调用失败（重试 {self.max_retries} 次）: {last_err}")
