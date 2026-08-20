"""
因为本专业是做雷达信号处理的所以写了一个简单版的信号分类器，看信号类别并且得到信号参数。这个也许真可以迁移到我的毕设上替换传统算法。
用户任务
   ↓
主Agent
   ↓
LLM 拆解任务
   ↓
并行下发 3 个 SubAgent
   ├─ SpectrumAgent
   │    ├─ 调 spectrum_analysis_tool
   │    └─ LLM 解释频域结果
   │
   ├─ TimeDomainAgent
   │    ├─ 调 time_domain_analysis_tool
   │    └─ LLM 解释时域结果
   │
   └─ TypeAgent
        ├─ 调 type_feature_tool
        └─ LLM 判断 LFM / CW / Pulse     
"""

import argparse
import asyncio
import json
import os
import re
import time
from typing import Any

import numpy as np
from openai import AsyncOpenAI


# ============================================================
# 1. LLM 配置
# ============================================================

DEFAULT_MODEL = os.getenv("AGENT_MODEL", "qwen-max")

client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.2,
) -> str:
    """统一的大模型调用函数。"""

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError(
            "未检测到 DASHSCOPE_API_KEY。\n"
            'PowerShell 中执行：$env:DASHSCOPE_API_KEY="你的 API Key"'
        )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content or ""


# ============================================================
# 2. 测试信号生成
# ============================================================

def generate_signal(
    signal_type: str,
    fs: int = 20000,
    duration: float = 0.1,
):
    """
    生成简单测试信号：
    - lfm   : 1 kHz -> 4 kHz 线性调频
    - cw    : 2 kHz 单频连续波
    - pulse : 2 kHz 脉冲调制信号
    """

    t = np.arange(0, duration, 1 / fs)

    if signal_type == "lfm":
        f_start = 1000
        f_end = 4000
        k = (f_end - f_start) / duration

        phase = 2 * np.pi * (
            f_start * t
            + 0.5 * k * t**2
        )

        signal = np.cos(phase)

        truth = {
            "type": "LFM",
            "f_start_hz": f_start,
            "f_end_hz": f_end,
            "duration_s": duration,
        }

    elif signal_type == "cw":
        freq = 2000

        signal = np.cos(
            2 * np.pi * freq * t
        )

        truth = {
            "type": "CW",
            "frequency_hz": freq,
            "duration_s": duration,
        }

    elif signal_type == "pulse":
        carrier = 2000

        # 10 ms 周期，2 ms 脉冲宽度
        period = 0.01
        pulse_width = 0.002

        gate = (
            np.mod(t, period) < pulse_width
        ).astype(float)

        signal = (
            gate
            * np.cos(2 * np.pi * carrier * t)
        )

        truth = {
            "type": "Pulse",
            "carrier_hz": carrier,
            "period_s": period,
            "pulse_width_s": pulse_width,
            "duration_s": duration,
        }

    else:
        raise ValueError(
            "signal_type 只支持：lfm / cw / pulse"
        )

    return t, signal, truth


# ============================================================
# 3. Tool：频域分析
# ============================================================

def spectrum_analysis_tool(
    signal: np.ndarray,
    fs: int,
) -> dict[str, Any]:
    """
    SpectrumAgent 使用的本地工具。
    计算：
    - 峰值频率
    - 起始频率
    - 终止频率
    - 中心频率
    - 估计带宽
    """

    window = np.hanning(len(signal))
    spectrum = np.abs(
        np.fft.rfft(signal * window)
    )

    freqs = np.fft.rfftfreq(
        len(signal),
        d=1 / fs,
    )

    if np.max(spectrum) == 0:
        return {
            "peak_frequency_hz": 0,
            "start_frequency_hz": 0,
            "end_frequency_hz": 0,
            "center_frequency_hz": 0,
            "bandwidth_hz": 0,
        }

    peak_index = int(
        np.argmax(spectrum)
    )

    peak_freq = float(
        freqs[peak_index]
    )

    # 用最大谱幅的 20% 作为简单阈值
    threshold = (
        np.max(spectrum) * 0.2
    )

    valid = np.where(
        spectrum >= threshold
    )[0]

    if len(valid) == 0:
        f_start = peak_freq
        f_end = peak_freq
    else:
        f_start = float(
            freqs[valid[0]]
        )
        f_end = float(
            freqs[valid[-1]]
        )

    center = (
        f_start + f_end
    ) / 2

    bandwidth = (
        f_end - f_start
    )

    return {
        "peak_frequency_hz": round(
            peak_freq, 2
        ),
        "start_frequency_hz": round(
            f_start, 2
        ),
        "end_frequency_hz": round(
            f_end, 2
        ),
        "center_frequency_hz": round(
            center, 2
        ),
        "bandwidth_hz": round(
            bandwidth, 2
        ),
    }


# ============================================================
# 4. Tool：时域分析
# ============================================================

def time_domain_analysis_tool(
    signal: np.ndarray,
    fs: int,
) -> dict[str, Any]:
    """
    TimeDomainAgent 使用的本地工具。
    """

    duration = len(signal) / fs

    peak = float(
        np.max(np.abs(signal))
    )

    rms = float(
        np.sqrt(
            np.mean(signal**2)
        )
    )

    crest_factor = (
        peak / rms
        if rms > 0
        else 0
    )

    # 简单的有效样本比例
    activity_threshold = (
        peak * 0.15
    )

    active_ratio = float(
        np.mean(
            np.abs(signal)
            > activity_threshold
        )
    )

    return {
        "duration_s": round(
            duration, 6
        ),
        "peak_amplitude": round(
            peak, 4
        ),
        "rms": round(
            rms, 4
        ),
        "crest_factor": round(
            crest_factor, 4
        ),
        "active_sample_ratio": round(
            active_ratio, 4
        ),
    }


# ============================================================
# 5. Tool：类型特征提取
# ============================================================

def dominant_frequency(
    segment: np.ndarray,
    fs: int,
) -> float:
    """计算一段信号的主频。"""

    if len(segment) == 0:
        return 0.0

    window = np.hanning(
        len(segment)
    )

    spectrum = np.abs(
        np.fft.rfft(
            segment * window
        )
    )

    freqs = np.fft.rfftfreq(
        len(segment),
        d=1 / fs,
    )

    return float(
        freqs[
            np.argmax(spectrum)
        ]
    )


def type_feature_tool(
    signal: np.ndarray,
    fs: int,
) -> dict[str, Any]:
    """
    TypeAgent 使用的 Tool。

    把信号分成 4 段，观察主频随时间是否变化；
    同时提取时域活动比例，为 LLM 提供分类依据。
    """

    segments = np.array_split(
        signal,
        4,
    )

    dominant_freqs = [
        dominant_frequency(
            seg,
            fs,
        )
        for seg in segments
    ]

    freq_change = (
        dominant_freqs[-1]
        - dominant_freqs[0]
    )

    freq_span = (
        max(dominant_freqs)
        - min(dominant_freqs)
    )

    peak = float(
        np.max(np.abs(signal))
    )

    threshold = peak * 0.15

    active_ratio = float(
        np.mean(
            np.abs(signal)
            > threshold
        )
    )

    return {
        "segment_dominant_frequencies_hz": [
            round(f, 2)
            for f in dominant_freqs
        ],
        "frequency_change_hz": round(
            freq_change, 2
        ),
        "frequency_span_hz": round(
            freq_span, 2
        ),
        "active_sample_ratio": round(
            active_ratio, 4
        ),
    }


# ============================================================
# 6. Supervisor Agent：任务规划
# ============================================================

DEFAULT_PLAN = {
    "spectrum_task": (
        "分析信号的频域特征，包括主要频率范围、"
        "中心频率和带宽。"
    ),
    "time_task": (
        "分析信号的时域特征，包括持续时间、"
        "峰值、RMS 和活动比例。"
    ),
    "type_task": (
        "根据分段主频变化和时域活动特征，"
        "判断信号更接近 LFM、CW 还是 Pulse。"
    ),
}


def extract_json(text: str) -> dict | None:
    """
    从 LLM 输出中提取 JSON。
    """

    text = text.strip()

    # 去掉 ```json
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        try:
            data = json.loads(
                text[start:end + 1]
            )

            if isinstance(data, dict):
                return data

        except Exception:
            pass

    return None


async def supervisor_plan(
    user_request: str,
    model: str,
) -> dict[str, str]:
    """
    Supervisor 先用 LLM 拆解任务。
    """

    system_prompt = """
你是一个信号分析系统中的 Supervisor Agent。

你的任务是把用户的信号分析请求拆成 3 个并行子任务，
分别交给：

1. SpectrumAgent：负责频域特征分析。
2. TimeDomainAgent：负责时域特征分析。
3. TypeAgent：负责根据信号特征判断 LFM / CW / Pulse。

只输出 JSON，不要输出 Markdown。

格式必须为：
{
  "spectrum_task": "...",
  "time_task": "...",
  "type_task": "..."
}

要求：
- 每个任务一句话即可。
- 不要自己计算信号参数。
- 不要提前给最终结论。
""".strip()

    try:
        output = await call_llm(
            system_prompt,
            user_request,
            model,
        )

        plan = extract_json(output)

        if not plan:
            return DEFAULT_PLAN

        required = {
            "spectrum_task",
            "time_task",
            "type_task",
        }

        if not required.issubset(
            plan.keys()
        ):
            return DEFAULT_PLAN

        return {
            key: str(plan[key])
            for key in required
        }

    except Exception:
        # 为避免规划 JSON 格式异常导致整个流程停止，
        # 保留固定任务作为兜底。
        return DEFAULT_PLAN


# ============================================================
# 7. Spectrum SubAgent
# ============================================================

async def spectrum_agent(
    task: str,
    signal: np.ndarray,
    fs: int,
    model: str,
) -> dict[str, Any]:
    """
    SpectrumAgent：
    Tool -> LLM -> Result
    """

    print(
        "[SpectrumAgent] "
        "调用 spectrum_analysis_tool..."
    )

    tool_result = (
        spectrum_analysis_tool(
            signal,
            fs,
        )
    )

    system_prompt = """
你是 SpectrumAgent，一名频域信号分析专家。

你会收到：
1. Supervisor 分配给你的任务；
2. spectrum_analysis_tool 的真实计算结果。

请只根据 Tool 结果进行分析，不要虚构数值。
用简洁中文给出 2-4 句话的频域分析结论。
""".strip()

    user_prompt = (
        f"任务：{task}\n\n"
        "Tool 结果：\n"
        + json.dumps(
            tool_result,
            ensure_ascii=False,
            indent=2,
        )
    )

    analysis = await call_llm(
        system_prompt,
        user_prompt,
        model,
    )

    return {
        "agent": "SpectrumAgent",
        "task": task,
        "tool_result": tool_result,
        "analysis": analysis.strip(),
    }


# ============================================================
# 8. TimeDomain SubAgent
# ============================================================

async def time_domain_agent(
    task: str,
    signal: np.ndarray,
    fs: int,
    model: str,
) -> dict[str, Any]:
    """
    TimeDomainAgent：
    Tool -> LLM -> Result
    """

    print(
        "[TimeDomainAgent] "
        "调用 time_domain_analysis_tool..."
    )

    tool_result = (
        time_domain_analysis_tool(
            signal,
            fs,
        )
    )

    system_prompt = """
你是 TimeDomainAgent，一名时域信号分析专家。

你会收到：
1. Supervisor 分配给你的任务；
2. time_domain_analysis_tool 的真实计算结果。

请只根据 Tool 结果进行分析，不要虚构数值。
用简洁中文说明信号持续时间、幅度和活动特征。
""".strip()

    user_prompt = (
        f"任务：{task}\n\n"
        "Tool 结果：\n"
        + json.dumps(
            tool_result,
            ensure_ascii=False,
            indent=2,
        )
    )

    analysis = await call_llm(
        system_prompt,
        user_prompt,
        model,
    )

    return {
        "agent": "TimeDomainAgent",
        "task": task,
        "tool_result": tool_result,
        "analysis": analysis.strip(),
    }


# ============================================================
# 9. Type SubAgent
# ============================================================

async def type_agent(
    task: str,
    signal: np.ndarray,
    fs: int,
    model: str,
) -> dict[str, Any]:
    """
    TypeAgent：
    Tool -> LLM -> Result
    """

    print(
        "[TypeAgent] "
        "调用 type_feature_tool..."
    )

    tool_result = (
        type_feature_tool(
            signal,
            fs,
        )
    )

    system_prompt = """
你是 TypeAgent，一名简单信号类型识别专家。

需要在 LFM、CW、Pulse 三种类型中进行判断。

参考规则：
- 如果不同时间段的主频持续明显变化，通常更符合 LFM；
- 如果各时间段主频基本稳定，通常更符合 CW；
- 如果信号只在部分时间内明显存在、活动比例较低，则可能更符合 Pulse。

你必须基于 Tool 返回的数据判断，不要虚构特征。

输出格式：

类型：LFM / CW / Pulse
理由：一句或两句解释
""".strip()

    user_prompt = (
        f"任务：{task}\n\n"
        "Tool 结果：\n"
        + json.dumps(
            tool_result,
            ensure_ascii=False,
            indent=2,
        )
    )

    analysis = await call_llm(
        system_prompt,
        user_prompt,
        model,
    )

    return {
        "agent": "TypeAgent",
        "task": task,
        "tool_result": tool_result,
        "analysis": analysis.strip(),
    }


# ============================================================
# 10. Supervisor Agent：并行调度
# ============================================================

async def run_subagents_parallel(
    plan: dict[str, str],
    signal: np.ndarray,
    fs: int,
    model: str,
):
    """
    asyncio.gather 同时运行 3 个 SubAgent。
    """

    print("\n" + "=" * 60)
    print(
        "SupervisorAgent："
        "开始并行下发 SubAgent"
    )
    print("=" * 60)

    start = time.perf_counter()

    results = await asyncio.gather(
        spectrum_agent(
            plan["spectrum_task"],
            signal,
            fs,
            model,
        ),
        time_domain_agent(
            plan["time_task"],
            signal,
            fs,
            model,
        ),
        type_agent(
            plan["type_task"],
            signal,
            fs,
            model,
        ),
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return results, elapsed


# ============================================================
# 11. Supervisor Agent：结果融合
# ============================================================

async def supervisor_summarize(
    user_request: str,
    results: list[dict[str, Any]],
    model: str,
) -> str:
    """
    将 3 个 SubAgent 的分析结果交回 Supervisor LLM，
    生成最终综合报告。
    """

    system_prompt = """
你是 Supervisor Agent。

你已经收到多个 SubAgent 的结果：
- SpectrumAgent
- TimeDomainAgent
- TypeAgent

请综合这些结果生成最终信号分析报告。

要求：
1. 所有具体数值必须来自 SubAgent 的 Tool 结果；
2. 不要虚构额外参数；
3. 给出最终信号类型判断；
4. 简洁说明主要频域、时域和类型特征；
5. 使用中文；
6. 控制在 6-10 句话以内。
""".strip()

    user_prompt = (
        f"用户原始任务：{user_request}\n\n"
        "SubAgent 结果：\n"
        + json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        )
    )

    return await call_llm(
        system_prompt,
        user_prompt,
        model,
    )


# ============================================================
# 12. 主流程
# ============================================================

async def main():
    parser = argparse.ArgumentParser(
        description=(
            "Supervisor + 并行 SubAgent "
            "信号分析 Demo"
        )
    )

    parser.add_argument(
        "--signal",
        choices=[
            "lfm",
            "cw",
            "pulse",
        ],
        default="lfm",
        help="测试信号类型",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="LLM 模型名称",
    )

    parser.add_argument(
        "--question",
        default=(
            "请分析这个信号的频域、时域特征，"
            "并判断它属于 LFM、CW 还是 Pulse。"
        ),
        help="发送给 Supervisor 的用户任务",
    )

    args = parser.parse_args()

    fs = 20000

    _, signal, truth = (
        generate_signal(
            args.signal,
            fs=fs,
            duration=0.1,
        )
    )

    print("=" * 60)
    print("Multi-Agent Signal Analysis")
    print("=" * 60)

    print(
        f"模型：{args.model}"
    )

    print(
        f"测试信号：{args.signal}"
    )

    print(
        "仿真真值："
        + json.dumps(
            truth,
            ensure_ascii=False,
        )
    )

    print(
        f"\n用户任务：{args.question}"
    )

    # --------------------------------------------------------
    # Step 1：Supervisor LLM 规划
    # --------------------------------------------------------

    print("\n[Step 1]")
    print(
        "SupervisorAgent "
        "调用 LLM 拆解任务..."
    )

    plan = await supervisor_plan(
        args.question,
        args.model,
    )

    print(
        json.dumps(
            plan,
            ensure_ascii=False,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Step 2：并行 SubAgent
    # --------------------------------------------------------

    print("\n[Step 2]")

    results, parallel_time = (
        await run_subagents_parallel(
            plan,
            signal,
            fs,
            args.model,
        )
    )

    # --------------------------------------------------------
    # Step 3：打印各 SubAgent 结果
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SubAgent 执行结果")
    print("=" * 60)

    for result in results:
        print(
            f"\n[{result['agent']}]"
        )

        print(
            "Tool Result:"
        )

        print(
            json.dumps(
                result["tool_result"],
                ensure_ascii=False,
                indent=2,
            )
        )

        print(
            "\nLLM Analysis:"
        )

        print(
            result["analysis"]
        )

    # --------------------------------------------------------
    # Step 4：Supervisor LLM 汇总
    # --------------------------------------------------------

    print("\n[Step 3]")
    print(
        "SupervisorAgent "
        "调用 LLM 汇总..."
    )

    final_answer = (
        await supervisor_summarize(
            args.question,
            results,
            args.model,
        )
    )

    print("\n" + "=" * 60)
    print("Supervisor 最终结果")
    print("=" * 60)

    print(final_answer)

    print("\n" + "=" * 60)
    print("执行信息")
    print("=" * 60)

    print(
        f"SubAgent 并行阶段耗时："
        f"{parallel_time:.2f} s"
    )

    print(
        "并行 SubAgent 数量：3"
    )

    print(
        "LLM 调用流程："
        "Supervisor规划 -> "
        "3个SubAgent并行 -> "
        "Supervisor汇总"
    )


if __name__ == "__main__":
    asyncio.run(main())
