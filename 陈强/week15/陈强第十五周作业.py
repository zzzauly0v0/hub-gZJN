"""并行 Subagent Agent

主 Agent 接收多个任务后，通过 ThreadPoolExecutor 把它们并行下发给 SubAgent，
再汇总各 SubAgent 的执行结果。

用法：
    from simple_parallel_agent import Agent
    agent = Agent(worker=my_worker)
    result = agent.dispatch(["任务A", "任务B", "任务C"])
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def default_worker(task: str) -> str:
    """默认子代理：调用 LLM 完成单个任务。"""
    from llm_client import llm_chat

    return llm_chat(
        "你是子代理，请直接完成任务并返回简洁结果。",
        task,
        temperature=0.0,
        max_tokens=512,
    )


class Agent:
    """主 Agent：负责拆分任务、并行下发、收集结果。"""

    def __init__(self, worker=None, max_workers=None):
        self.worker = worker or default_worker
        self.max_workers = max_workers

    @staticmethod
    def normalize_tasks(tasks):
        """把列表或管道分隔的字符串统一转换为任务列表。"""
        if isinstance(tasks, str):
            tasks = tasks.split("|")
        return [str(task).strip() for task in tasks if str(task).strip()]

    def dispatch(self, tasks):
        """并行下发多个任务，并返回各任务结果和总耗时。"""
        tasks = self.normalize_tasks(tasks)
        if not tasks:
            return {"results": {}, "task_count": 0, "wall_time": 0.0}

        max_workers = self.max_workers or max(1, len(tasks))
        results = {}
        start = time.time()

        # 并行执行所有子任务
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.worker, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                results[task] = future.result()

        return {
            "results": results,
            "task_count": len(tasks),
            "wall_time": round(time.time() - start, 3),
        }

    def run(self, tasks):
        """dispatch 的别名，方便直接调用。"""
        return self.dispatch(tasks)


if __name__ == "__main__":
    # 演示 worker：不依赖外部 API，方便本地快速验证
    def demo_worker(task: str) -> str:
        time.sleep(0.2 if task.startswith("慢") else 0.05)
        return task + " 完成"

    demo = Agent(worker=demo_worker)
    print(demo.dispatch(["任务A", "慢任务B", "任务C"]))
