"""
ReAct循环实现
Agent的核心：思考-行动-观察循环
"""
import json
import time
import logging
from typing import Dict, List, Callable, Optional, Any

from llm_client import LLMClient

logger = logging.getLogger(__name__)


class ReActLoop:
    """
    ReAct循环Agent
    
    流程：
    1. Thought: LLM思考下一步做什么
    2. Action: 执行工具调用
    3. Observation: 观察工具返回结果
    4. 重复直到得出Final Answer
    """
    
    def __init__(
        self,
        agent_name: str,
        tools: Dict[str, tuple],
        max_steps: int = 8,
        model_tag: str = "",
        system_prompt: str = None,
        multi_tool_dispatch: Callable = None
    ):
        """
        Args:
            agent_name: Agent名称
            tools: 工具字典 {"工具名": (函数, "描述")}
            max_steps: 最大循环步数
            model_tag: 模型标签（用于日志）
            system_prompt: 自定义系统提示
            multi_tool_dispatch: 多工具调用派发器。
                当一步中出现多个工具调用（多个问题）时调用，格式:
                f(calls: [(工具名, 查询参数), ...]) -> List[str]（观察结果，顺序与calls一致）。
                若为None，则多个工具调用由本Agent直接串行执行。
        """
        self.agent_name = agent_name
        self.tools = tools
        self.max_steps = max_steps
        self.model_tag = model_tag
        self.system_prompt = system_prompt
        self.multi_tool_dispatch = multi_tool_dispatch
        self.llm = LLMClient()
        
        # 构建工具定义（OpenAI Function Calling格式）
        self.tool_defs = []
        self.tool_funcs = {}
        for name, (func, desc) in tools.items():
            # 自动推断参数schema
            self.tool_defs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": f"调用{name}的查询参数"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
            self.tool_funcs[name] = func
    
    def run(
        self,
        query: str,
        on_step: Callable = None,
        shared_state: Dict = None
    ) -> Dict:
        """
        执行ReAct循环
        
        Args:
            query: 用户查询
            on_step: 每一步的回调函数，用于可视化
            shared_state: 共享状态（用于多Agent通信）
        
        Returns:
            {
                "final_answer": str,
                "trace": [{"step": int, "thought": str, "action": str, "observation": str}],
                "duration": float
            }
        """
        start_time = time.time()
        trace = []
        
        # 初始化消息列表
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": query})
        
        step = 0
        
        while step < self.max_steps:
            step += 1
            
            # 1. Thought: 调用LLM
            response = self.llm.chat(
                messages=messages,
                tools=self.tool_defs,
                tool_choice="auto"
            )
            
            # 获取LLM回复
            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])
            
            # 2. 如果没有工具调用，说明已经得到最终答案
            if not tool_calls and content:
                trace.append({
                    "step": step,
                    "thought": content,
                    "action": "final_answer",
                    "observation": "任务完成"
                })
                
                if on_step:
                    on_step({
                        "step": step,
                        "thought": content,
                        "action": "final_answer",
                        "observation": "任务完成"
                    })
                
                return {
                    "final_answer": content,
                    "trace": trace,
                    "duration": round(time.time() - start_time, 2)
                }
            
            # 3. 执行工具调用
            observations = []
            
            # 多个工具调用（多个问题）→ 派发给子Agent并行执行
            if len(tool_calls) > 1 and self.multi_tool_dispatch:
                calls = []
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    try:
                        tool_args = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        tool_args = {}
                    calls.append((tool_name, tool_args.get("query", "")))
                observations = self.multi_tool_dispatch(calls) or []
            else:
                # 单个工具调用（单一问题）→ 主Agent直接执行
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])
                    
                    # 执行工具
                    if tool_name in self.tool_funcs:
                        try:
                            # 调用工具函数
                            func = self.tool_funcs[tool_name]
                            # 支持共享状态注入
                            if "shared_state" in func.__code__.co_varnames:
                                result = func(tool_args.get("query", ""), shared_state=shared_state)
                            else:
                                result = func(tool_args.get("query", ""))
                            observations.append(f"[{tool_name}] {result}")
                        except Exception as e:
                            observations.append(f"[{tool_name}] 执行失败: {e}")
                    else:
                        observations.append(f"[{tool_name}] 工具不存在")
            
            observation_text = "\n".join(observations)
            
            # 记录trace
            thought_text = content or "（工具调用）"
            action_text = ", ".join([f"{tc['function']['name']}({tc['function']['arguments']})" 
                                     for tc in tool_calls])
            
            trace.append({
                "step": step,
                "thought": thought_text,
                "action": action_text,
                "observation": observation_text[:500]  # 截断避免过长
            })
            
            if on_step:
                on_step({
                    "step": step,
                    "thought": thought_text,
                    "action": action_text,
                    "observation": observation_text
                })
            
            # 将工具结果加入对话历史
            # 先添加assistant的消息（包含工具调用）
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.get("id", f"call_{step}_{i}"),
                        "type": "function",
                        "function": tc["function"]
                    }
                    for i, tc in enumerate(tool_calls)
                ]
            })
            
            # 再添加工具返回结果
            for i, obs in enumerate(observations):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_calls[i].get("id", f"call_{step}_{i}"),
                    "content": obs
                })
        
        # 达到最大步数
        final_msg = f"达到最大步数({self.max_steps})，请简化问题重试"
        return {
            "final_answer": final_msg,
            "trace": trace,
            "duration": round(time.time() - start_time, 2)
        }
