"""通用异步 ReAct 循环（主 agent / 子 agent 共用同一个类）

教学重点：
  1. ReAct = LLM 输出 Thought/Action/Action Input → runner 执行工具得 Observation → 续写
  2. 主 agent 和子 agent 的唯一区别是手里的 tools：
       主 agent {web_search, dispatch_subagents}   子 agent {web_search}
  3. run() 是 async def，工具也 await —— 所以整条链可被 asyncio.gather 并发

一个类 ~60 行，够体现范式即可，不追求鲁棒性。
"""
import re, time, asyncio
from llm_client import llm_chat

SYSTEM = """你是体育热点分析助手，可用工具：
{tools_desc}

严格按格式输出，每轮一次：
Thought: 你的推理
Action: 工具名
Action Input: 参数字符串

工具执行后你会看到 Observation。信息够了就输出：
Thought: 信息已足够
Final Answer: 你的分析结论（要点带来源）"""


class ReActLoop:
    """异步 ReAct 循环。tools: {name: (async_fn, desc)}"""

    def __init__(self, name: str, tools: dict, max_steps: int = 4, system: str = None):
        self.name, self.tools, self.max_steps = name, tools, max_steps
        self.system_tpl = system or SYSTEM

    async def run(self, question: str, on_step=None, ctx: dict = None) -> dict:
        """跑完一轮，返回 {final_answer, trace, duration}。on_step 每步回调（可视化/日志用）。"""
        t0 = time.time()
        system = self.system_tpl.format(
            tools_desc="\n".join(f"- {n}: {d}" for n, (_, d) in self.tools.items()))
        history, trace, final = f"Question: {question}\n\n", [], ""

        for i in range(self.max_steps):
            out = await llm_chat(system, history, stop=["Observation:"])
            thought, action, action_input = self._parse(out)
            step = {"idx": i, "agent": self.name, "thought": thought,
                    "action": action, "action_input": action_input, "observation": None}

            if action == "Final Answer":
                final, step["final"] = action_input, True
                trace.append(step)
                if on_step: on_step(step)
                break

            if on_step: on_step(step)                        # 先报「决定做什么」，不等工具返回
            obs = await self._exec(action, action_input, ctx)
            step["observation"] = obs
            trace.append(step)
            if on_step: on_step(step)                        # 工具回来后带 observation 再报一次
            history += out + f"Observation: {obs[:1200]}\n"
        else:
            final = "（达到步数上限）" + (trace[-1].get("observation") or "" if trace else "")

        return {"final_answer": final, "trace": trace, "duration": round(time.time() - t0, 2)}

    def _parse(self, text: str):
        """解析 Thought/Action/Action Input；无格式标记但有文本 → 当 Final Answer（防空转）。"""
        m = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", text, re.S)
        thought = m.group(1).strip()[:300] if m else ""
        if fa := re.search(r"Final Answer:\s*(.*)", text, re.S):
            return thought, "Final Answer", fa.group(1).strip()
        if act := re.search(r"Action:\s*(.*)", text):
            ai = re.search(r"Action Input:\s*(.*)", text)
            return thought, act.group(1).strip(), (ai.group(1).strip() if ai else "")
        return thought or "直接给结论", "Final Answer", text.strip()

    async def _exec(self, action: str, action_input: str, ctx: dict) -> str:
        if action not in self.tools:
            return f"无此工具 '{action}'，可选 {list(self.tools)}"
        try:
            return str(await self.tools[action][0](action_input, ctx=ctx))
        except Exception as e:
            return f"工具出错: {type(e).__name__}: {str(e)[:100]}"
