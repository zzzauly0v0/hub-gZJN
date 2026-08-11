"""
让大模型「写 Skill」和「优化 Skill」（本任务新增）。

与原项目 src/background_reviewer.py 的分工区别：
  Reviewer  —— 运行时 Nudge，输入失败样本，输出 create/patch 的最小改动，目标是**补覆盖**
  本模块    —— 离线优化，输入一份完整 Skill，输出等价但更省的重写版，目标是**降开销**

三个优化变体各代表一种优化手段（对应 GEPA 的不同突变算子）：
  opt_compress       纯压缩：删冗余表达，信息量不变
  opt_structured     重构：散文 → 查表式决策表，让 LLM 少推理
  opt_failure_guided 反思式：看 dev 集失败样本 + token 预算，边修边压
"""

from __future__ import annotations

import re

from llm import LLM

# ── 作者 Agent：从政策文档写出 v0 Skill ──────────────────────────────────────

AUTHOR_SYSTEM = """你是「云购商城」客服系统的 Skill 文档作者。

你的任务：阅读平台政策手册，为指定业务域写一份 SKILL.md。这份文档会被塞进客服
Agent 的系统提示，是 Agent 回答该业务域问题的**唯一**知识来源——政策手册本身
Agent 看不到。

## 输出要求
- 输出**纯 Markdown**，不要套 ``` 代码块
- 开头必须是 YAML frontmatter：
---
name: {skill_name}
description: 一句话说明覆盖范围
type: knowledge
version: 1
---
- 正文用 `## 小节` 组织，把政策里的**具体数字**（天数、金额、工作日、时间点、
  城市名）全部写进去——Agent 答不出具体数字就等于答错
- 只覆盖指定业务域，不要把其他业务域的政策抄进来

## 政策手册（你的信息来源）
{policies}
"""

AUTHOR_USER = """请为业务域「{domain_label}」写一份 SKILL.md，文件名 `{skill_name}`。

该业务域需要覆盖的政策范围：
{scope_hint}

Agent 将会被问到这类问题（**仅用于理解覆盖范围，不要在文档里逐条回答它们**）：
{sample_questions}

现在输出完整的 SKILL.md。"""


# ── 优化 Agent：三种优化策略 ─────────────────────────────────────────────────

OPT_COMMON_RULES = """## 不可违背的约束（违背即视为优化失败）

1. **信息无损**：原文里所有具体数字、阈值、城市名、专有名词（如"平台承担"、
   "解冻"、"待发货"）必须保留。这些词是评估的判定依据，删一个就少答对一题。
2. **保持 SKILL.md 格式**：YAML frontmatter（name 不变、version 递增）+ `##` 小节。
3. **不要新增政策**：不许推断、补充、举例扩写原文没有的规则。
4. 输出**纯 Markdown**，不要套 ``` 代码块，不要写任何解释性前言后语。"""

OPT_COMPRESS = {
    "key": "opt_compress",
    "label": "压缩（token 优先）",
    "system": """你是 Prompt/Skill 压缩专家。给你一份 SKILL.md，请在**信息无损**的前提下
把它的 token 数压到最低——它每次都被塞进系统提示，每一个 token 每题都要付费。

""" + OPT_COMMON_RULES + """

## 压缩手法
- 删礼貌语、过渡句、"需要注意的是"这类元话语
- 删对 Agent 决策没影响的解释、背景、举例（**除数字示例外**）
- 合并同义条目；能一行说完的不要分三行
- 用符号代替长句：`≤48h → 可取消`、`银卡/金卡 → 平台承担`
- 删掉重复强调同一规则的段落，只留最醒目的一处

目标：token 数减少 40% 以上，覆盖能力不降。""",
    "user": """## 待压缩的 SKILL.md（当前约 {tokens} tokens）

{skill_content}

输出压缩后的完整 SKILL.md。""",
}

OPT_STRUCTURED = {
    "key": "opt_structured",
    "label": "重构（查表式决策表）",
    "system": """你是 Skill 结构化专家。给你一份 SKILL.md，请把它从"散文式描述"改写为
"**查表式决策表**"。

动机：LLM 读散文时需要自己做条件推理（容易漏分支、被先验带偏）；读决策表时只需
定位行列后照抄，既更准也更快。

""" + OPT_COMMON_RULES + """

## 重构手法
- 每个业务域一张 Markdown 表：**条件列 → 结论列**，结论里直接写具体数字
- 条件要覆盖完整判定空间，含边界（如 `≤48h` 和 `>48h` 两行都要有）
- 容易被 LLM 先验带偏的规则，在表后加一行 `⚠️` 强制说明（如"此规则优先于VIP特权"）
- 表格本身要紧凑：列名短，单元格不写完整句子

目标：结构清晰可查表，同时 token 不高于原文。""",
    "user": """## 待重构的 SKILL.md（当前约 {tokens} tokens）

{skill_content}

输出重构后的完整 SKILL.md。""",
}

OPT_FAILURE_GUIDED = {
    "key": "opt_failure_guided",
    "label": "反思式（失败样本 + token 预算双目标）",
    "system": """你是 Skill 优化专家，同时背两个指标：**答对率**和**token 开销**。

给你一份 SKILL.md、它在开发集上的**失败样本**、以及一个 token 预算。请给出一版
既修掉失败、又更省 token 的 Skill。

""" + OPT_COMMON_RULES + """

## 优化步骤（按序思考后再动笔）
1. 逐条读失败样本，判断根因属于哪一类：
   - **缺信息**：文档里根本没写这条规则 → 从失败样本反推需要补的规则（但只补
     失败样本直接指向的，不要顺手扩写）
   - **写了但没被用**：文档有这条规则，Agent 却答错/答偏 → 说明表述不够醒目或
     被别的段落盖住了，**改表述强度和位置**，不是加字数
   - **表述不匹配**：Agent 用了同义说法但没用政策原词 → 把政策专有词写成
     必须使用的措辞（如明确要求答"平台承担"而不是"免运费"）
2. 修完后做一遍压缩：把预算腾给真正影响判定的内容
3. 自查：失败样本里出现过的每一个具体数字，在新文档里都能找到

## Token 预算
新文档不得超过 **{budget} tokens**。修 bug 要靠"换位置、换措辞、删冗余"腾空间，
不是靠加长度。""",
    "user": """## 当前 SKILL.md（约 {tokens} tokens）

{skill_content}

## 开发集失败样本（共 {n_fail} 条，Agent 答错或推脱）

{failures}

## 开发集上已答对的题（{n_pass} 条，**不要改坏它们**）

{passes}

输出优化后的完整 SKILL.md（≤{budget} tokens）。""",
}

OPTIMIZERS = [OPT_COMPRESS, OPT_STRUCTURED, OPT_FAILURE_GUIDED]


# ── 清洗与规范化 ─────────────────────────────────────────────────────────────

FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def clean_skill_md(raw: str, skill_name: str, version: int) -> str:
    """
    把 LLM 输出规范成合法 SKILL.md：
      1. 脱掉外层 ``` 代码块（模型很爱加）
      2. 缺 frontmatter 就补一个——原项目的 _extract_description / _bump_version
         都靠 frontmatter 里的 description / version 字段工作
      3. 强制 name 与 version 为我们指定的值，避免变体之间 name 打架
    """
    text = raw.strip()
    m = FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()

    if not text.startswith("---"):
        text = (
            f"---\nname: {skill_name}\ndescription: (由模型生成，未提供描述)\n"
            f"type: knowledge\nversion: {version}\n---\n\n{text}"
        )
        return text

    # 已有 frontmatter：只替换 name / version 两个字段，其余保留
    end = text.find("\n---", 3)
    if end == -1:
        return text
    head, body = text[: end + 4], text[end + 4 :]
    if re.search(r"^name:", head, re.MULTILINE):
        head = re.sub(r"^name:.*$", f"name: {skill_name}", head, count=1, flags=re.MULTILINE)
    else:
        head = head.replace("---", f"---\nname: {skill_name}", 1)
    if re.search(r"^version:", head, re.MULTILINE):
        head = re.sub(r"^version:.*$", f"version: {version}", head, count=1, flags=re.MULTILINE)
    else:
        head = head.rstrip("\n-") + f"\nversion: {version}\n---"
    return head + body


# ── 对外入口 ─────────────────────────────────────────────────────────────────

def author_skill(
    llm: LLM,
    policies: str,
    skill_name: str,
    domain_label: str,
    scope_hint: str,
    sample_questions: list[str],
    max_tokens: int = 2000,
) -> str:
    """让大模型从政策手册写出 v0 Skill（未优化的起点）。"""
    system = AUTHOR_SYSTEM.format(skill_name=skill_name, policies=policies)
    user = AUTHOR_USER.format(
        domain_label=domain_label,
        skill_name=skill_name,
        scope_hint=scope_hint,
        sample_questions="\n".join(f"- {q}" for q in sample_questions),
    )
    raw, _ = llm.chat(system, user, max_tokens=max_tokens)
    return clean_skill_md(raw, skill_name, version=1)


def optimize_skill(
    llm: LLM,
    optimizer: dict,
    skill_content: str,
    skill_name: str,
    version: int,
    cur_tokens: int,
    budget: int | None = None,
    failures: list[dict] | None = None,
    passes: list[dict] | None = None,
    max_tokens: int = 2500,
) -> str:
    """按指定优化策略产出一个 Skill 变体。"""
    fmt = {
        "skill_content": skill_content,
        "tokens": cur_tokens,
        "budget": budget or int(cur_tokens * 0.7),
        "n_fail": len(failures or []),
        "n_pass": len(passes or []),
        "failures": _format_failures(failures or []),
        "passes": _format_passes(passes or []),
    }
    system = optimizer["system"].format(**{k: v for k, v in fmt.items() if f"{{{k}}}" in optimizer["system"]})
    user = optimizer["user"].format(**{k: v for k, v in fmt.items() if f"{{{k}}}" in optimizer["user"]})
    raw, _ = llm.chat(system, user, max_tokens=max_tokens)
    return clean_skill_md(raw, skill_name, version=version)


def _format_failures(failures: list[dict]) -> str:
    if not failures:
        return "（无失败样本）"
    lines = []
    for i, f in enumerate(failures, 1):
        lines.append(f"[{i}] 用户问：{f['question']}")
        lines.append(f"    Agent 答：{f['answer'][:180]}")
        lines.append(f"    ✗ 判定：{f['fail_reason']}")
    return "\n".join(lines)


def _format_passes(passes: list[dict]) -> str:
    if not passes:
        return "（无）"
    return "\n".join(f"- {p['question']}" for p in passes)
