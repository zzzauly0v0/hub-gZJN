import asyncio
import random
import time
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# ---------- 1. 数据结构定义 ----------
@dataclass
class Task:
    """子任务实体"""
    id: int
    description: str
    assigned_agent: str = ""
    result: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed

@dataclass
class AgentContext:
    """子Agent上下文"""
    name: str
    expertise: str
    config: Dict[str, Any] = field(default_factory=dict)


# ---------- 2. 子Agent抽象基类 ----------
class BaseSubAgent(ABC):
    """所有子Agent必须继承此类"""
    
    def __init__(self, context: AgentContext):
        self.context = context
        self.name = context.name

    @abstractmethod
    async def execute(self, task: Task) -> str:
        """执行具体任务，返回结果字符串"""
        pass

    async def __call__(self, task: Task) -> Task:
        """使Agent可调用，自动更新任务状态"""
        task.status = "running"
        task.assigned_agent = self.name
        try:
            start = time.perf_counter()
            result = await self.execute(task)
            elapsed = time.perf_counter() - start
            task.result = result
            task.status = "completed"
            print(f"  ✅ [{self.name}] 完成任务 (耗时 {elapsed:.2f}s): {task.description[:20]}...")
        except Exception as e:
            task.status = "failed"
            task.result = f"Error: {str(e)}"
            print(f"  ❌ [{self.name}] 任务失败: {e}")
        return task


# ---------- 3. 具体子Agent实现示例 ----------
class ResearcherAgent(BaseSubAgent):
    """擅长检索、搜集信息"""
    async def execute(self, task: Task) -> str:
        # 模拟异步IO调用（如搜索引擎API、数据库查询）
        await asyncio.sleep(random.uniform(1.0, 2.5)) 
        return f"【研究结果】针对 '{task.description}'，找到了3篇相关论文和5条最新动态。"

class AnalystAgent(BaseSubAgent):
    """擅长数据分析、逻辑推理"""
    async def execute(self, task: Task) -> str:
        await asyncio.sleep(random.uniform(1.5, 3.0))
        return f"【分析结果】对 '{task.description}' 进行了SWOT分析，结论倾向于积极。"

class WriterAgent(BaseSubAgent):
    """擅长文案撰写、内容生成"""
    async def execute(self, task: Task) -> str:
        await asyncio.sleep(random.uniform(1.0, 2.0))
        return f"【撰写结果】基于调研数据，完成了 '{task.description}' 的初稿，共1200字。"

class CoderAgent(BaseSubAgent):
    """擅长代码编写、调试"""
    async def execute(self, task: Task) -> str:
        await asyncio.sleep(random.uniform(2.0, 3.5))
        return f"【编码结果】针对 '{task.description}' 编写了Python实现，并附有单元测试。"


# ---------- 4. 主控Agent (Orchestrator) ----------
class MasterAgent:
    """
    主控Agent：负责任务分解、并行调度、结果聚合。
    支持动态并发控制，防止资源耗尽。
    """
    
    def __init__(self, max_concurrent_tasks: int = 5):
        self.subagents: Dict[str, BaseSubAgent] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.task_counter = 0

    def register_agent(self, agent: BaseSubAgent):
        """注册子Agent到系统中"""
        self.subagents[agent.name] = agent
        print(f"📌 已注册子Agent: {agent.name} (擅长: {agent.context.expertise})")

    def _plan_tasks(self, goal: str) -> List[Task]:
        """
        规划阶段：将复杂目标拆解为多个子任务。
        这里使用硬编码策略（演示），实际生产可用LLM根据Agent能力动态拆分。
        """
        print(f"\n🧠 主控Agent正在规划目标: {goal}")
        tasks = []
        
        # 根据关键词模拟任务拆分
        if "市场调研" in goal or "新产品" in goal:
            tasks.append(Task(id=len(tasks)+1, description="收集竞品市场数据"))
            tasks.append(Task(id=len(tasks)+1, description="分析用户痛点与需求"))
            tasks.append(Task(id=len(tasks)+1, description="撰写市场调研报告"))
        elif "开发" in goal or "代码" in goal:
            tasks.append(Task(id=len(tasks)+1, description="设计系统架构"))
            tasks.append(Task(id=len(tasks)+1, description="编写核心功能模块"))
            tasks.append(Task(id=len(tasks)+1, description="编写单元测试与文档"))
        else:
            # 通用拆解
            tasks.append(Task(id=len(tasks)+1, description=f"调研: {goal} 的背景信息"))
            tasks.append(Task(id=len(tasks)+1, description=f"分析: {goal} 的可行性"))
            tasks.append(Task(id=len(tasks)+1, description=f"总结: 关于 {goal} 的报告"))
        
        print(f"📋 任务拆解完成，共 {len(tasks)} 个子任务")
        return tasks

    def _dispatch(self, tasks: List[Task]) -> List[Task]:
        """
        分发阶段：根据任务描述匹配最合适的子Agent。
        这里用简单的关键词匹配，生产场景可用向量检索或LLM Router。
        """
        for task in tasks:
            desc = task.description.lower()
            if "调研" in desc or "收集" in desc or "数据" in desc:
                task.assigned_agent = "Researcher"
            elif "分析" in desc or "swot" in desc or "痛点" in desc:
                task.assigned_agent = "Analyst"
            elif "撰写" in desc or "报告" in desc or "总结" in desc:
                task.assigned_agent = "Writer"
            elif "代码" in desc or "开发" in desc or "架构" in desc:
                task.assigned_agent = "Coder"
            else:
                # 默认分配给分析师
                task.assigned_agent = "Analyst"
        return tasks

    async def _execute_single(self, task: Task) -> Task:
        """执行单个任务（带并发控制）"""
        async with self.semaphore:
            agent = self.subagents.get(task.assigned_agent)
            if not agent:
                task.status = "failed"
                task.result = f"未找到Agent: {task.assigned_agent}"
                return task
            return await agent(task)  # 调用 __call__

    async def _execute_parallel(self, tasks: List[Task]) -> List[Task]:
        """并行执行所有任务"""
        print(f"\n⚡ 开始并行执行 {len(tasks)} 个子任务 (最大并发: {self.semaphore._value})...")
        start = time.perf_counter()
        
        # 核心并行逻辑：asyncio.gather 并发执行所有协程
        completed_tasks = await asyncio.gather(
            *[self._execute_single(task) for task in tasks],
            return_exceptions=False  # 设为True可容错，但为了清晰这里抛出异常
        )
        
        elapsed = time.perf_counter() - start
        print(f"\n⏱️  所有任务执行完毕，总耗时: {elapsed:.2f}秒")
        return list(completed_tasks)

    def _aggregate(self, tasks: List[Task], original_goal: str) -> Dict[str, Any]:
        """聚合阶段：汇总所有子任务结果，生成最终输出"""
        print(f"\n📊 主控Agent正在聚合结果...")
        success_count = sum(1 for t in tasks if t.status == "completed")
        failed_count = len(tasks) - success_count
        
        summary = {
            "goal": original_goal,
            "total_tasks": len(tasks),
            "success": success_count,
            "failed": failed_count,
            "details": [
                {
                    "agent": t.assigned_agent,
                    "description": t.description,
                    "result": t.result,
                    "status": t.status
                }
                for t in tasks
            ],
            "final_answer": f"基于 {success_count} 个成功子任务，已完成对 '{original_goal}' 的综合处理。"
        }
        
        # 如果有写入Agent，尝试生成最终总结（这里简单拼接）
        writer_results = [t.result for t in tasks if t.assigned_agent == "Writer" and t.result]
        if writer_results:
            summary["final_answer"] = f"最终报告摘要: {writer_results[0]}"
            
        return summary

    async def run(self, goal: str) -> Dict[str, Any]:
        """
        主控Agent运行入口（Plan -> Dispatch -> Execute -> Aggregate）
        """
        print("\n" + "="*60)
        print(f"🚀 主控Agent启动，目标: {goal}")
        print("="*60)
        
        # 1. Plan
        tasks = self._plan_tasks(goal)
        
        # 2. Dispatch
        tasks = self._dispatch(tasks)
        
        # 3. Execute (Parallel)
        completed_tasks = await self._execute_parallel(tasks)
        
        # 4. Aggregate
        final_result = self._aggregate(completed_tasks, goal)
        
        return final_result


# ---------- 5. 真实LLM扩展（可选） ----------
class OpenAISubAgent(BaseSubAgent):
    """
    真实调用OpenAI API的子Agent示例（需安装openai库）
    实际使用时取消注释并配置API Key
    """
    async def execute(self, task: Task) -> str:
        # from openai import AsyncOpenAI
        # client = AsyncOpenAI(api_key="your-key")
        # response = await client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": f"你是一个{self.context.expertise}专家，请完成: {task.description}"}]
        # )
        # return response.choices[0].message.content
        await asyncio.sleep(1)  # 模拟
        return f"【OpenAI】针对 {task.description} 的生成结果。"


# ---------- 6. 启动示例 ----------
async def main():
    # 1. 初始化主控Agent（最大并发数设为3）
    master = MasterAgent(max_concurrent_tasks=3)

    # 2. 注册子Agent（可插拔、可扩展）
    master.register_agent(ResearcherAgent(AgentContext("Researcher", "信息检索")))
    master.register_agent(AnalystAgent(AgentContext("Analyst", "逻辑分析")))
    master.register_agent(WriterAgent(AgentContext("Writer", "内容撰写")))
    master.register_agent(CoderAgent(AgentContext("Coder", "代码开发")))

    # 3. 下发任务并并行执行
    goal = "针对2026年AI Agent市场进行调研，并开发一个简易的Demo原型"
    result = await master.run(goal)

    # 4. 打印最终聚合结果
    print("\n" + "="*60)
    print("📝 最终聚合报告")
    print("="*60)
    print(f"目标: {result['goal']}")
    print(f"状态: 成功 {result['success']} / 总数 {result['total_tasks']}")
    print(f"结论: {result['final_answer']}")
    print("\n详细子任务输出:")
    for detail in result['details']:
        print(f"  - [{detail['agent']}] {detail['description']}: {detail['result'][:50]}...")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
