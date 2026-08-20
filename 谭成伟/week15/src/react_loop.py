"""通用异步 ReAct 循环引擎。"""
from __future__ import annotations
import time
import re
import asyncio
import logging
from typing import Callable, Awaitable, Optional

from llm_client import llm_chat

logger = logging.getLogger(__name__)

REACT_SYSTEM = """你是旅行规划助手，能用以下工具联网搜索与估算。

可用工具：
{tools_desc}

按如下格式严格输出（每轮一次 Thought/Action/Action Input）：
Thought: 你的推理，分析还需查什么
Action: 工具名
Action Input: 工具参数（字符串）

工具执行后会得到 Observation。多轮调用直到能给出完整答案，最后用：
Thought: 我已收集足够信息
Final Answer: 综合答案（带来源要点）

规则：
- Action 必须是上面列出的工具名之一
- Action Input 是该工具的参数字符串
- 每轮只调一次工具，等 Observation 再决定下一步"""


def build_tools_desc(tools: dict) -> str:
    lines = []
    for name, (fn, desc) in tools.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


class AsyncReActLoop:
    def __init__(self, agent_name: str, tools: dict,
                 max_steps: int = 6, model_tag: str = "deepseek-chat",
                 system_prompt: Optional[str] = None):
        self.agent_name = agent_name
        self.tools = tools
        self.max_steps = max_steps
        self.model_tag = model_tag
        self._system_template = system_prompt or REACT_SYSTEM
        self.trace: list[dict] = []

    async def run(self, question: str,
                  on_step: Optional[Callable[[dict], Awaitable[None]]] = None,
                  shared_state: dict = None) -> dict:
        self.trace = []
        t0 = time.time()
        system = self._system_template.format(tools_desc=build_tools_desc(self.tools))
        history = f"Question: {question}\n\n"
        final_answer = ""

        for step_idx in range(self.max_steps):
            llm_out = await llm_chat(system, history, temperature=0.0,
                                     max_tokens=768, stop=["Observation:"])
            thought, action, action_input = self._parse(llm_out)

            step = {"idx": step_idx, "agent": self.agent_name,
                    "thought": thought, "action": action,
                    "action_input": action_input, "observation": None}

            if action == "Final Answer":
                step["final"] = True
                final_answer = action_input
                self.trace.append(step)
                if on_step:
                    await on_step(step)
                break

            step["final"] = False
            if on_step:
                await on_step(step)

            observation = await self._exec_tool(action, action_input, shared_state)

            step["observation"] = observation
            step["done"] = True
            self.trace.append(step)
            if on_step:
                await on_step(step)

            history += llm_out + f"Observation: {observation[:1200]}\n"
        else:
            final_answer = "（已达最大步数）" + (self.trace[-1].get("observation", "") or "")
            step = {"idx": self.max_steps, "agent": self.agent_name,
                    "thought": "达到步数上限", "action": "Final Answer",
                    "action_input": final_answer, "observation": None, "final": True}
            self.trace.append(step)
            if on_step:
                await on_step(step)

        duration = round(time.time() - t0, 2)
        return {"final_answer": final_answer, "trace": self.trace,
                "duration": duration}

    def _parse(self, text: str) -> tuple[str, str, str]:
        thought = ""
        m = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.S)
        if m:
            thought = m.group(1).strip()[:400]

        mfa = re.search(r"Final Answer:\s*(.*)", text, re.S)
        if mfa:
            return thought, "Final Answer", mfa.group(1).strip()

        ma = re.search(r"Action:\s*(.*)", text)
        mi = re.search(r"Action Input:\s*(.*)", text)
        if ma:
            action = ma.group(1).strip()
            action_input = (mi.group(1).strip() if mi else "")
            return thought, action, action_input

        if text.strip():
            return thought or "综合调研结果给出方案", "Final Answer", text.strip()
        return thought, "", ""

    async def _exec_tool(self, action: str, action_input: str,
                         shared_state: dict) -> str:
        if action not in self.tools:
            return f"工具 '{action}' 不存在，可选: {list(self.tools.keys())}"
        fn, _ = self.tools[action]
        try:
            if asyncio.iscoroutinefunction(fn):
                result = await (fn(action_input, shared_state=shared_state)
                               if shared_state is not None
                               else fn(action_input))
            else:
                def _call():
                    return fn(action_input, shared_state=shared_state) \
                        if shared_state is not None else fn(action_input)
                result = await asyncio.to_thread(_call)
            return str(result)
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {str(e)[:120]}"


if __name__ == "__main__":
    import asyncio as _a
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    from tavily_search import tavily_search, format_search_result

    async def web_search(q, **_):
        return format_search_result(await tavily_search(q))

    async def _t():
        loop = AsyncReActLoop("test",
                              tools={"web_search": (web_search, "联网搜索，参数是查询词")},
                              max_steps=4)
        r = await loop.run("东京晴空塔的门票价格和开放时间？")
        print(f"\n答案: {r['final_answer'][:120]}")
        print(f"trace {len(r['trace'])} 步:")
        for s in r["trace"]:
            print(f"  [{s['idx']}] {s['action']}({s['action_input'][:40]}) → "
                  f"{(s.get('observation') or '')[:50]}")

    _a.run(_t())
