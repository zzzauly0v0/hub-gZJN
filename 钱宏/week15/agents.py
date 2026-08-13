"""
安全作业票智能助手 —— 承包商人员评估（主 Agent + 并行 Subagent 硬编码编排）

教学重点：
  1. 去掉"派发工具"：run_assessment 直接【硬编码】并行启动 4 个评估维度 subagent
     （ThreadPoolExecutor），不再依赖主 agent 的 ReAct 决策是否派发
  2. 每个 subagent 只绑定一个「数据查询方法」（列表构造假数据，data_provider.py），
     拿到数据后【原样返回】
  3. 4 份数据并行查完后，直接汇总喂给主 agent（无工具 ReAct）做最终综合评估
  4. 一票否决：4 个维度中只要有一项不符合，判定【不允许】；Final Answer 须列明"哪项没过"
  5. 并行优势：wall-clock ≈ max(单agent时长) 而非 sum，统计加速比
"""

import time, json, logging, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from react_loop import ReActLoop
import data_provider

logger = logging.getLogger(__name__)

# 固定派发的 4 个评估维度 → (维度名, 数据查询方法名, 工具描述)
DIMENSIONS = [
    ("基础资质合规", "get_person_basic_info",
     "获取人员基本信息和证书（评估：证书有效期、作业范围）"),
    ("岗位匹配性", "get_training_records",
     "获取培训记录（评估：技能、专项培训）"),
    ("历史安全绩效", "get_safety_violations",
     "获取历史违章记录（评估：违章、事故、黑名单）"),
    ("当前动态状态", "get_current_assignments",
     "获取当前参与中的作业票（评估：位置、时间冲突、工时）"),
]

# 主 agent：无工具，直接拿 4 份汇总数据做综合评估
MAIN_SYSTEM = """你是"安全作业票智能助手"中的承包商人员评估主分析师。
你的任务：根据下面提供的「4 个评估维度真实查询数据」和「作业信息」，评估该人员是否可以实施本次作业。
数据已由子代理并行查得并原样列出，严禁再虚构任何字段。

【决策规则】一票否决：4 个评估维度中只要有一项不符合，就判定为【不允许】实施本次作业。
判断细则：
- 证书过期或作业范围不含本次作业类型 → 基础资质不合规
- 缺少本次作业对应的专项培训（如特级动火） → 岗位不匹配
- 有未闭环违章 / 事故 / 在黑名单 → 历史安全绩效不达标
- 作业时间与当前作业票冲突 → 当前动态状态不符

【返回格式】必须按维度说明评估结果，明确指出"哪项评估没有过"：
- 未通过维度：维度名 + 具体原因
- 若全部通过，则说明各维度均符合，允许作业。
- 结尾给出明确结论：允许作业 / 不允许作业（含一票否决理由）"""

SUBAGENT_SYSTEM_TMPL = """你是"安全作业票智能助手"中的{name}查询子代理。

可用工具：
{tools_desc}

你只有一个工具，调用它查询真实数据（严禁虚构）。
调用工具拿到 Observation 后，请把返回的数据【原样完整】作为 Final Answer 输出，
不要自行判断、不要补充评价、不要遗漏任何字段。

示例：
Thought: 我需要调用工具获取{name}相关数据
Action: {tool_name}
Action Input: 查询
（拿到 Observation 后）
Final Answer: 原样数据"""




def _fmt(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, indent=1)


def _run_dimension(dim_name: str, fn_name: str, desc: str, person_id: str,
                   on_subagent_step: Optional[Callable] = None,
                   on_subagent_done: Optional[Callable] = None):
    """硬编码启动单个评估维度 subagent：只查一次数据，原样返回。"""
    fn = getattr(data_provider, fn_name)
    sid = f"sub_{uuid.uuid4().hex[:6]}"

    # 闭包锁定 person_id：无论 LLM 传什么，都查询正确的人
    tool = (lambda _q, _fn=fn, _pid=person_id, **kw: _fmt(_fn(_pid)),
            f"{desc}（人员已锁定 {person_id}）")
    sub = ReActLoop(
        agent_name=sid,
        tools={fn_name: tool},
        max_steps=2,
        model_tag=f"deepseek-chat({dim_name})",
        system_prompt=SUBAGENT_SYSTEM_TMPL.format(
            name=dim_name, tool_name=fn_name,
            tools_desc=f"- {fn_name}: {desc}（人员已锁定 {person_id}）"),
        max_tool_calls=1,   # subagent 只查一次，原样返回，杜绝重复查询
    )

    def _on_step(step):
        if on_subagent_step:
            on_subagent_step(sid, step)

    res = sub.run(dim_name, on_step=_on_step)
    if on_subagent_done:
        on_subagent_done(sid, res["duration"], dim_name)
    return sid, {
        "subtopic": dim_name, "trace": res["trace"],
        "duration": res["duration"], "final_answer": res["final_answer"],
    }


def _dispatch_dimensions(person_id: str, serial: bool = False,
                         on_subagent_step: Optional[Callable] = None,
                         on_subagent_done: Optional[Callable] = None):
    """硬编码并行派发 4 个评估维度 subagent（不再作为主 agent 的工具）。

    返回 (results_dict, parallel_stats)。results_dict: {sid: {subtopic, trace, duration, final_answer}}
    """
    t0 = time.time()
    results: dict = {}

    if serial:
        # 串行基线（A/B 对比用）
        for dim_name, fn_name, desc in DIMENSIONS:
            sid, r = _run_dimension(dim_name, fn_name, desc, person_id,
                                    on_subagent_step, on_subagent_done)
            results[sid] = r
    else:
        # 并行（默认）：wall-clock ≈ max 而非 sum
        with ThreadPoolExecutor(max_workers=len(DIMENSIONS)) as pool:
            futs = {pool.submit(_run_dimension, dim_name, fn_name, desc, person_id,
                                on_subagent_step, on_subagent_done): dim_name
                    for dim_name, fn_name, desc in DIMENSIONS}
            for fut in as_completed(futs):
                sid, r = fut.result()
                results[sid] = r

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for r in results.values()), 2)
    stats = {"n_subagents": len(DIMENSIONS), "wall_clock": wall,
             "serial_sum": serial_sum, "speedup": round(serial_sum / wall, 2) if wall else 0}
    return results, stats


def _build_summary(results: dict) -> str:
    """把 4 份子代理结果拼成汇总文本，喂给主 agent 当评估输入。"""
    parts = [f"【{r['subtopic']}】(用时{r['duration']}s)\n{r['final_answer'][:800]}"
             for r in results.values()]
    return "\n\n".join(parts)


def run_assessment(question: str, person_id: str = "C1234",
                   work_context: dict = None,
                   on_main_step: Optional[Callable] = None,
                   on_subagent_step: Optional[Callable] = None,
                   on_subagent_done: Optional[Callable] = None,
                   serial: bool = False) -> dict:
    """执行一次承包商人员评估。

    流程：硬编码并行启动 4 个维度 subagent → 汇总数据 → 主 agent 综合评估。
    返回 {final_answer, main_trace, subagents, parallel_stats, work_context}。
    """
    work_context = work_context or {}

    """"① 硬编码并行派发 4 个 subagent（不再经过工具/ReAct 派发）"""
    subagents, parallel_stats = _dispatch_dimensions(
        person_id, serial=serial,
        on_subagent_step=on_subagent_step,
        on_subagent_done=on_subagent_done)

    # ② 汇总 4 份数据
    summary = _build_summary(subagents)
    wc = "\n".join(f"- {k}: {v}" for k, v in work_context.items())

    # ③ 主 agent 无工具，直接拿汇总 + 作业信息做综合评估
    main_question = (
        f"{question}\n\n"
        f"【作业信息】\n{wc}\n\n"
        f"【4 个评估维度真实查询数据】\n{summary}\n\n"
        f"请按 MAIN_SYSTEM 的决策规则做一票否决综合评估，并明确指出哪项评估没有过。"
    )
    main = ReActLoop(
        agent_name="main",
        tools={},   # 无工具：数据已由 subagent 查好，直接综合
        max_steps=2,
        model_tag="deepseek-chat(主)",
        system_prompt=MAIN_SYSTEM,
    )
    result = main.run(main_question, on_step=on_main_step)

    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "subagents": subagents,
        "parallel_stats": [parallel_stats],
        "work_context": work_context,
    }



if __name__ == "__main__":
    """
    这一块可以直接使用规则判断，整个作业安全助手，包含llm,以及规则判断的使用。判断完成后，使用llm统一输出
    """
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    q = ("请评估承包商人员张工（工号C1234）是否可执行以下作业：\n"
         "- 作业类型：特级动火\n"
         "- 地点：303罐区\n"
         "- 介质：石脑油（闪点<-18℃）\n"
         "- 时间：2026-08-12 09:00 ~ 12:00")
    r = run_assessment(q, person_id="C1234",
                       work_context={"作业类型": "特级动火", "地点": "303罐区",
                                     "介质": "石脑油（闪点<-18℃）",
                                     "时间": "2026-08-12 09:00 ~ 12:00"})
    print(f"\nsubagent 数: {len(r['subagents'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n最终评估结论:\n{r['final_answer']}")
