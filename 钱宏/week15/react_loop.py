"""
通用 ReAct 循环引擎

教学重点：
  1. ReAct = Reason + Act：LLM 生成 Thought(推理) → Action(选工具) → Action Input(参数)，
     runner 执行工具得 Observation，再喂回 LLM 继续，直到 Final Answer
  2. 主 agent 和 subagent 都是 ReAct 循环——区别只在「有哪些工具」：
     主 agent 有 dispatch_subagents（派发子课题并行调研），
     subagent 各绑定一个数据查询方法（如 get_person_basic_info）
  3. 完整 trace 捕获：每步 Thought/Action/ActionInput/Observation 存下来，
     供可视化「点节点看 ReAct 过程」用

用 stop=["Observation:"] 让 LLM 在生成完 Action Input 后停下，runner 执行工具
再补 Observation 续写——这是 ReAct 的经典实现技巧。

依赖：仅 llm_client + 工具函数，无外部库
"""

import time, re, json, logging
from typing import Callable, Optional
from llm_client import llm_chat

logger = logging.getLogger(__name__)

REACT_SYSTEM = """你是安全作业票智能助手，能用以下工具查询真实数据（严禁虚构）。

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
- 每轮只调一次工具，等 Observation 再决定下一步
- 拿到 Observation 后必须原样引用数据，不得自行编造字段"""


def build_tools_desc(tools: dict) -> str:
    """把 tools 字典格式化成工具说明。tools: {name: (fn, description)}"""
    lines = []
    for name, (fn, desc) in tools.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


class ReActLoop:
    """通用 ReAct 循环。主 agent / subagent 各自实例化一个。"""

    def __init__(self, agent_name: str, tools: dict,
                 max_steps: int = 6, model_tag: str = "deepseek-chat",
                 system_prompt: Optional[str] = None,
                 require_action_first: bool = False,
                 max_tool_calls: Optional[int] = None):
        """
        tools: {tool_name: (fn(arg)->str, description_str)}
        system_prompt: 自定义系统提示（主 agent 用 MAIN_SYSTEM 引导派发）。
                       None 时用默认 REACT_SYSTEM。{tools_desc} 占位符会被替换。
        require_action_first: 若为 True，则在本 agent 调用过任何工具之前，
                        LLM 直接给出的 Final Answer 会被忽略，强制其先走 Action。
                        用于主 agent——避免 LLM 偷懒直接编答案而不派发 subagent。
        max_tool_calls: 最多允许调用工具的次数。达到上限后若 LLM 仍想调工具，
                        则强制把最近一次 Observation 当作 Final Answer 收尾
                        （subagent 查一次即"原样返回"，杜绝重复查询/无限循环）。
        """
        self.agent_name = agent_name
        self.tools = tools          # {name: (fn, desc)}
        self.max_steps = max_steps
        self.model_tag = model_tag
        self._system_template = system_prompt or REACT_SYSTEM
        self.require_action_first = require_action_first
        self.max_tool_calls = max_tool_calls
        self.trace: list[dict] = []  # 本轮执行 trace（点节点查看用）

    def run(self, question: str, on_step: Callable = None,
            shared_state: dict = None) -> dict:
        """
        执行 ReAct 循环。
        on_step(step_dict): 每步回调（SSE 流式用）。
        shared_state: 共享状态 dict（主 agent 派发 subagent 时往里塞 subagent trace）。
        返回 {final_answer, trace, duration}。
        """
        self.trace = []
        t0 = time.time()
        system = self._system_template.format(tools_desc=build_tools_desc(self.tools))
        # 对话历史：累积 Thought/Action/ActionInput/Observation
        history = f"Question: {question}\n\n"
        final_answer = ""
        has_called_tool = False   # 是否至少调用过一次工具
        tool_calls = 0            # 工具调用次数（受 max_tool_calls 约束）
        last_observation = ""     # 最近一次工具返回的 Observation

        for step_idx in range(self.max_steps):
            # 调 LLM 生成下一步（停在 Observation: 前）
            llm_out = llm_chat(system, history, temperature=0.0,
                               max_tokens=768, stop=["Observation:"])
            # 解析 Action 或 Final Answer
            thought, action, action_input = self._parse(llm_out)

            step = {"idx": step_idx, "agent": self.agent_name,
                    "thought": thought, "action": action,
                    "action_input": action_input, "observation": None}

            # 强制先调工具：本 agent 还没调过任何工具，却直接给了 Final Answer
            # → 视为偷懒，忽略并把"必须先调用工具"的指令塞回历史，让其重试
            if (action == "Final Answer" and self.require_action_first
                    and not has_called_tool):
                history += (llm_out
                    + "\nObservation: 错误：你还没有调用任何工具查询真实数据，"
                      "根据规则必须先调用 dispatch_subagents（或你的查询工具）"
                      "获取真实数据，禁止直接给出结论。请重新输出 Thought/Action。\n")
                continue

            if action == "Final Answer":
                step["final"] = True
                final_answer = action_input   # Final Answer 内容放 action_input
                self.trace.append(step)
                if on_step: on_step(step)     # final：单次回调
                break

            # 达到工具调用上限：不再执行工具，强制把最近 Observation 当结论收尾
            # （subagent 查一次即"原样返回"；主 agent 已拿到汇总，转交 LLM 综合）
            if self.max_tool_calls is not None and tool_calls >= self.max_tool_calls:
                final_answer = last_observation
                step["final"] = True
                step["action"] = "Final Answer"
                step["action_input"] = final_answer
                step["observation"] = last_observation
                step["note"] = f"达到工具调用上限({self.max_tool_calls})，原样返回数据"
                self.trace.append(step)
                if on_step: on_step(step)
                break

            # ── pre 执行：立即发 step（observation=None），让前端马上看到
            #    "主 agent 决定派发 / subagent 决定搜索" 的决策，不用等工具返回 ──
            step["final"] = False
            if on_step: on_step(step)

            # 执行工具（可能很慢，如 dispatch_subagents 要等所有子 agent 跑完）
            observation = self._exec_tool(action, action_input, shared_state)
            has_called_tool = True   # 已调用过工具，后续才允许 Final Answer
            tool_calls += 1
            last_observation = observation

            # ── post 执行：同一 idx 再发一次，带真实 observation，前端原地更新 ──
            step["observation"] = observation
            step["done"] = True
            self.trace.append(step)
            if on_step: on_step(step)

            # 续写历史
            history += llm_out + f"Observation: {observation[:1200]}\n"

        else:
            # 超过 max_steps，强制收尾
            final_answer = "（已达最大步数）" + (last_observation or "")
            step = {"idx": self.max_steps, "agent": self.agent_name,
                    "thought": "达到步数上限", "action": "Final Answer",
                    "action_input": final_answer, "observation": last_observation,
                    "final": True}
            self.trace.append(step)
            if on_step: on_step(step)

        duration = round(time.time() - t0, 2)
        return {"final_answer": final_answer, "trace": self.trace,
                "duration": duration}

    def _parse(self, text: str) -> tuple[str, str, str]:
        """从 LLM 输出解析 Thought/Action/Action Input。
        返回 (thought, action, action_input)。Final Answer 时 action='Final Answer'。
        兜底：若没匹配到 Action 也没 Final Answer，但有实质文本，当作 Final Answer
        （LLM 拿到子调研结果后常直接写报告、不带 Final Answer 前缀）。"""
        thought = ""
        m = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.S)
        if m: thought = m.group(1).strip()[:400]

        # Final Answer 优先检测
        mfa = re.search(r"Final Answer:\s*(.*)", text, re.S)
        if mfa:
            return thought, "Final Answer", mfa.group(1).strip()

        # Action / Action Input
        ma = re.search(r"Action:\s*(.*)", text)
        mi = re.search(r"Action Input:\s*(.*)", text)
        if ma:
            action = ma.group(1).strip()
            action_input = (mi.group(1).strip() if mi else "")
            return thought, action, action_input

        # 兜底：有实质文本但无格式标记 → 当作 Final Answer
        if text.strip():
            return thought or "综合调研结果给出报告", "Final Answer", text.strip()
        return thought, "", ""

    def _exec_tool(self, action: str, action_input: str, shared_state: dict) -> str:
        """执行工具，返回 observation 文本。未知工具返回错误说明。"""
        if action not in self.tools:
            return f"工具 '{action}' 不存在，可选: {list(self.tools.keys())}"
        fn, _ = self.tools[action]
        try:
            # 工具可能需要 shared_state（dispatch_subagents 用）
            return str(fn(action_input, shared_state=shared_state)
                       if shared_state is not None else fn(action_input))
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {str(e)[:120]}"
