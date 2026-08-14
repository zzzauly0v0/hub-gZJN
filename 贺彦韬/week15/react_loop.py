"""
通用 ReAct 循环引擎：Thought -> Action -> Observation 反复循环，直到 Final Answer。

三个设计要点（写作业报告/答辩时用得上）：

  1. 主 agent 和子 agent 共用同一个 ReActLoop 类，区别只在造实例时给了什么 tools。
     能力边界是"造它的时候手里塞了什么工具"决定的，不是运行时判断"你是主还是子"再
     拦下来——子 agent 天生就没有 dispatch_subagents 这个工具，它自己的系统提示里
     也压根不会提到这个工具存在，它无从"想到"要用。

  2. LLM 每次生成只允许写到 Thought/Action/Action Input，用 stop=["Observation:"]
     卡住生成——Observation 必须来自代码真正执行工具之后拿到的真实结果，由这里手动
     拼进对话历史（history += llm_out + "Observation: ..."），不是 LLM 自己写出来的。
     少了这一步，LLM 会自己瞎编"搜索结果"，等于压根没真的去查资料。

  3. max_steps 是安全阀：万一 LLM 一直不给 Final Answer，强制在 N 轮后收尾，
     保证函数一定有返回值，不会真的死循环。
"""
import re
import time
import logging
from typing import Callable, Optional

from llm_client import llm_chat

logger = logging.getLogger(__name__)

REACT_SYSTEM_TEMPLATE = """你是一个能使用工具的助手。

可用工具：
{tools_desc}

严格按下面的格式，每轮只输出一次 Thought/Action/Action Input：
Thought: 你的推理，说明还需要做什么
Action: 工具名（必须是上面列出的名字之一）
Action Input: 这个工具的参数（一段文字）

工具执行后你会看到 Observation。可以循环多轮，直到信息足够，最后输出：
Thought: 我已经有足够信息了
Final Answer: 完整答案"""


def build_tools_desc(tools: dict) -> str:
    """把 tools 字典（{名字: (函数, 描述)}）整理成给 LLM 看的说明文字。"""
    return "\n".join(f"- {name}: {desc}" for name, (_, desc) in tools.items())


class ReActLoop:
    """通用 ReAct 循环。主 agent / 子 agent 各自 new 一个实例，只是传的 tools 不同。"""

    def __init__(self, agent_id: str, tools: dict, max_steps: int = 6,
                 system_prompt: Optional[str] = None):
        """
        agent_id: 这个实例的标识，日志/可视化用。
        tools:    {工具名: (fn(参数)->str, 一句话描述)}——决定了这个 agent 能做什么。
        system_prompt: 不传就用默认模板；主 agent 会传一份带自己决策规则的自定义提示。
        """
        self.agent_id = agent_id
        self.tools = tools
        self.max_steps = max_steps
        self._system_template = system_prompt or REACT_SYSTEM_TEMPLATE
        self.trace: list[dict] = []

    def run(self, question: str, on_step: Callable = None, shared_state: dict = None) -> dict:
        """跑一次完整的 ReAct 循环，返回 {final_answer, trace, duration}。

        on_step(step): 每一步的回调（想接可视化/流式展示时用，不需要就不传）。
        shared_state:  一个可以在多个 agent 之间共享的字典，派发子 agent 时用来
                       让 dispatch 工具函数拿到"共享记事本"。
        """
        self.trace = []
        system = self._system_template.format(tools_desc=build_tools_desc(self.tools))
        history = f"Question: {question}\n\n"
        final_answer = ""
        t0 = time.time()

        for step_idx in range(self.max_steps):
            llm_out = llm_chat(system, history, max_tokens=700, stop=["Observation:"])
            thought, action, action_input = self._parse(llm_out)
            step = {"idx": step_idx, "agent": self.agent_id, "thought": thought,
                    "action": action, "action_input": action_input, "observation": None}

            if action == "Final Answer":
                step["final"] = True
                final_answer = action_input
                self.trace.append(step)
                if on_step:
                    on_step(step)
                break

            step["final"] = False
            if on_step:
                on_step(step)  # 先让外部看到"决定要做什么了"，工具可能要跑一阵子

            observation = self._exec_tool(action, action_input, shared_state)
            step["observation"] = observation
            self.trace.append(step)
            if on_step:
                on_step(step)  # 再补一次，带上真实结果

            history += llm_out + f"Observation: {observation[:1200]}\n"
        else:
            # for 循环正常跑满没被 break，说明 max_steps 轮都没等到 Final Answer
            last_obs = self.trace[-1].get("observation") if self.trace else ""
            final_answer = "（已达最大推理步数，以下是目前收集到的信息）\n" + (last_obs or "")

        return {"final_answer": final_answer, "trace": self.trace,
                "duration": round(time.time() - t0, 2)}

    def _parse(self, text: str) -> tuple[str, str, str]:
        """从 LLM 输出的自由文本里，用正则抠出 Thought/Action/Action Input，
        或者识别出 Final Answer。"""
        thought = ""
        # 非贪婪匹配到 "下一个 Action:" 或 "下一个 Final Answer:" 或字符串末尾为止，
        # 三个都要列上——否则遇到"只有 Thought+Final Answer、没有 Action"的输出时，
        # 后面的 Final Answer 内容会被一起吞进 thought 里。
        m = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", text, re.S)
        if m:
            thought = m.group(1).strip()[:400]

        mfa = re.search(r"Final Answer:\s*(.*)", text, re.S)
        if mfa:
            return thought, "Final Answer", mfa.group(1).strip()

        ma = re.search(r"Action:\s*(.*)", text)
        mi = re.search(r"Action Input:\s*(.*)", text)
        if ma:
            action = ma.group(1).strip()
            action_input = mi.group(1).strip() if mi else ""
            return thought, action, action_input

        # 兜底：LLM 拿到足够信息后经常直接写答案、忘了加 "Final Answer:" 前缀，
        # 这里把"没匹配到任何格式但确实写了东西"也当成 Final Answer 处理，
        # 避免因为解析不到 Action 而一直空转到 max_steps。
        if text.strip():
            return thought or "汇总信息给出答案", "Final Answer", text.strip()
        return thought, "", ""

    def _exec_tool(self, action: str, action_input: str, shared_state: dict) -> str:
        """真正执行一次工具调用，返回 Observation 文本。"""
        if action not in self.tools:
            return f"工具 '{action}' 不存在，可选：{list(self.tools.keys())}"
        fn, _ = self.tools[action]
        try:
            if shared_state is not None:
                return str(fn(action_input, shared_state=shared_state))
            return str(fn(action_input))
        except Exception as e:
            return f"工具执行出错：{type(e).__name__}: {str(e)[:150]}"
