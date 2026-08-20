"""
A/B 评估 flash-card Skill 升级前后的执行质量。

目录结构（默认）:
skills/
├── skill_compare.py
└── flash_card/
    ├── SKILL_v1.md
    ├── SKILL_v2.md
    ├── data/
    └── script/
        └── make_flashcard.py
        
输出:
    skill_comparison_results.csv
    skill_comparison_summary.csv
    skill_comparison_report.html

输出结果:最后显示响应耗时和平均token数有优化

"""

import argparse
import csv
import html
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
FLASH_DIR = BASE_DIR / "flash_card"
DEFAULT_V1 = FLASH_DIR / "SKILL_v1.md"
DEFAULT_V2 = FLASH_DIR / "SKILL_v2.md"
SCRIPT_PATH = FLASH_DIR / "script" / "make_flashcard.py"

TEST_CASES = [
    {"id": "T01", "request": "给我做一个 resilient 的闪卡", "word": "resilient"},
    {"id": "T02", "request": "帮我生成 meticulous 的单词卡", "word": "meticulous"},
    {"id": "T03", "request": "给我做 crazy 的 flash card", "word": "crazy"},
    {"id": "T04", "request": "生成一个 ambitious 的英语单词闪卡", "word": "ambitious"},
    {"id": "T05", "request": "make a flash card for fragile", "word": "fragile"},
    {"id": "T06", "request": "请做一张 GENEROUS 的单词卡，谢谢", "word": "generous"},
    {"id": "T07", "request": "我在背单词，能不能帮我把 persistent 做成 flash card？", "word": "persistent"},
    {"id": "T08", "request": "Flash card please: sophisticated", "word": "sophisticated"},
    {"id": "T09", "request": "做个 concise 的闪卡，例句要中英对照", "word": "concise"},
    {"id": "T10", "request": "帮我生成一个 versatile 单词卡", "word": "versatile"},
]

REQUIRED_FIELDS = {"word", "phonetic", "pos", "definition", "examples", "synonyms"}


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    return path.read_text(encoding="utf-8")


def make_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未检测到 DASHSCOPE_API_KEY。PowerShell 示例：\n"
            '$env:DASHSCOPE_API_KEY="你的 API Key"'
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def strip_fence(text: str) -> str:
    text = (text or "").strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    return m.group(1).strip() if m else text


def extract_json_object(text: str) -> str:
    """先按纯 JSON 解析；失败时仅截取最外层 {...}，不修补字段。"""
    text = strip_fence(text)
    try:
        json.loads(text)
        return text
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start:end + 1]
        return text


def call_skill(
    client: OpenAI,
    model: str,
    skill_text: str,
    request: str,
    temperature: float,
) -> tuple[str, float, dict[str, int]]:
    # 公平性关键：这里只规定“输出评估所需 JSON”，不重复 v2 中的具体字段约束。
    system = (
        "你是一个 Skill harness。严格按照下面的 SKILL.md 处理用户请求。\n\n"
        "===== SKILL.md =====\n"
        f"{skill_text}\n"
        "===== END SKILL =====\n\n"
        "为了便于自动评估，本次不要实际打开浏览器，也不要输出解释。"
        "只输出该 Skill 在生成 HTML 前应写入 data/<word>.json 的 JSON 对象。"
    )

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": request},
        ],
        temperature=temperature,
    )
    elapsed = time.perf_counter() - t0
    text = resp.choices[0].message.content or ""

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if getattr(resp, "usage", None):
        usage["prompt_tokens"] = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
        usage["completion_tokens"] = int(getattr(resp.usage, "completion_tokens", 0) or 0)
        usage["total_tokens"] = int(getattr(resp.usage, "total_tokens", 0) or 0)

    return text, elapsed, usage


def is_nonempty_string(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def is_english_word(x: Any) -> bool:
    return isinstance(x, str) and bool(re.fullmatch(r"[A-Za-z][A-Za-z'-]*", x.strip()))


def evaluate_json(raw: str, expected_word: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    metrics: dict[str, Any] = {
        "json_valid": 0,
        "required_fields": 0,
        "word_match": 0,
        "phonetic_valid": 0,
        "pos_valid": 0,
        "definition_valid": 0,
        "examples_exact3": 0,
        "examples_bilingual": 0,
        "examples_use_word": 0,
        "synonyms_4_6": 0,
        "synonyms_unique": 0,
    }

    try:
        data = json.loads(extract_json_object(raw))
        if not isinstance(data, dict):
            return metrics, None
        metrics["json_valid"] = 1
    except Exception:
        return metrics, None

    metrics["required_fields"] = int(REQUIRED_FIELDS.issubset(data.keys()))

    word = data.get("word")
    if is_nonempty_string(word):
        metrics["word_match"] = int(word.strip().lower() == expected_word.lower())

    phonetic = data.get("phonetic")
    metrics["phonetic_valid"] = int(
        is_nonempty_string(phonetic)
        and phonetic.strip().startswith("/")
        and phonetic.strip().endswith("/")
        and len(phonetic.strip()) > 2
    )

    pos = data.get("pos")
    metrics["pos_valid"] = int(
        is_nonempty_string(pos)
        and bool(re.fullmatch(r"(n|v|adj|adv|prep|conj|pron|det|interj)\.", pos.strip()))
    )

    metrics["definition_valid"] = int(is_nonempty_string(data.get("definition")))

    examples = data.get("examples")
    metrics["examples_exact3"] = int(isinstance(examples, list) and len(examples) == 3)

    if isinstance(examples, list) and len(examples) == 3:
        bilingual_ok = True
        use_word_ok = True
        for ex in examples:
            if not isinstance(ex, dict) or not is_nonempty_string(ex.get("en")) or not is_nonempty_string(ex.get("zh")):
                bilingual_ok = False
                use_word_ok = False
                break
            # 允许目标词大小写变化；不强制词形变化（这是保守的确定性指标）
            if expected_word.lower() not in ex["en"].lower():
                use_word_ok = False
        metrics["examples_bilingual"] = int(bilingual_ok)
        metrics["examples_use_word"] = int(use_word_ok)

    syns = data.get("synonyms")
    if isinstance(syns, list):
        cleaned = [s.strip().lower() for s in syns if is_english_word(s)]
        metrics["synonyms_4_6"] = int(4 <= len(syns) <= 6 and len(cleaned) == len(syns))
        metrics["synonyms_unique"] = int(len(cleaned) == len(set(cleaned)) == len(syns) and len(cleaned) > 0)

    return metrics, data


def html_generation_success(data: dict[str, Any] | None) -> int:
    if data is None or not SCRIPT_PATH.exists():
        return 0

    try:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            json_path = td_path / "card.json"
            html_path = td_path / "card.html"
            json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    str(json_path),
                    "-o",
                    str(html_path),
                ],
                cwd=td_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
            return int(proc.returncode == 0 and html_path.exists() and html_path.stat().st_size > 0)
    except Exception:
        return 0


QUALITY_KEYS = [
    "json_valid",
    "required_fields",
    "word_match",
    "phonetic_valid",
    "pos_valid",
    "definition_valid",
    "examples_exact3",
    "examples_bilingual",
    "examples_use_word",
    "synonyms_4_6",
    "synonyms_unique",
    "html_success",
]


def evaluate_version(
    version: str,
    skill_path: Path,
    client: OpenAI,
    model: str,
    runs: int,
    temperature: float,
) -> list[dict[str, Any]]:
    skill_text = read_text(skill_path)
    rows: list[dict[str, Any]] = []

    for case in TEST_CASES:
        for run_idx in range(1, runs + 1):
            print(f"[{version}] {case['id']} run={run_idx}: {case['request']}")
            row: dict[str, Any] = {
                "version": version,
                "case_id": case["id"],
                "run": run_idx,
                "request": case["request"],
                "expected_word": case["word"],
            }

            try:
                raw, elapsed, usage = call_skill(
                    client, model, skill_text, case["request"], temperature
                )
                metrics, data = evaluate_json(raw, case["word"])
                metrics["html_success"] = html_generation_success(data)

                row.update(metrics)
                row["latency_s"] = round(elapsed, 4)
                row.update(usage)
                row["overall_success"] = int(all(metrics[k] == 1 for k in QUALITY_KEYS))
                row["quality_score"] = round(
                    100 * sum(metrics[k] for k in QUALITY_KEYS) / len(QUALITY_KEYS), 2
                )
                row["error"] = ""
            except Exception as exc:
                for k in QUALITY_KEYS:
                    row[k] = 0
                row["overall_success"] = 0
                row["quality_score"] = 0.0
                row["latency_s"] = 0.0
                row["prompt_tokens"] = 0
                row["completion_tokens"] = 0
                row["total_tokens"] = 0
                row["error"] = str(exc)

            rows.append(row)

    return rows


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows]
    return statistics.mean(vals) if vals else 0.0


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    versions = sorted({r["version"] for r in rows})
    by_version = {v: [r for r in rows if r["version"] == v] for v in versions}

    metric_specs = [
        ("overall_success", "综合成功率", "%"),
        ("quality_score", "平均质量得分", "score"),
        ("json_valid", "JSON 合法率", "%"),
        ("required_fields", "字段完整率", "%"),
        ("word_match", "目标词提取正确率", "%"),
        ("examples_exact3", "3 条例句满足率", "%"),
        ("examples_bilingual", "中英例句完整率", "%"),
        ("examples_use_word", "例句包含目标词率", "%"),
        ("synonyms_4_6", "近义词数量正确率", "%"),
        ("synonyms_unique", "近义词去重正确率", "%"),
        ("html_success", "HTML 生成成功率", "%"),
        ("latency_s", "平均响应耗时", "s"),
        ("total_tokens", "平均 Token 数", "tokens"),
    ]

    out = []
    for key, label, unit in metric_specs:
        vals = {}
        for v in versions:
            value = mean(by_version[v], key)
            if unit == "%":
                value *= 100
            vals[v] = value

        v1 = vals.get("V1", 0.0)
        v2 = vals.get("V2", 0.0)
        delta = v2 - v1

        out.append({
            "metric": key,
            "label": label,
            "unit": unit,
            "V1": round(v1, 3),
            "V2": round(v2, 3),
            "delta": round(delta, 3),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_value(v: float, unit: str) -> str:
    if unit == "%":
        return f"{v:.1f}%"
    if unit == "s":
        return f"{v:.3f}s"
    if unit == "score":
        return f"{v:.1f}"
    if unit == "tokens":
        return f"{v:.1f}"
    return f"{v:.3f}"


def delta_class(metric: str, delta: float) -> str:
    # latency/token 越低越好，其余指标越高越好
    lower_is_better = metric in {"latency_s", "total_tokens"}
    improved = delta < 0 if lower_is_better else delta > 0
    degraded = delta > 0 if lower_is_better else delta < 0
    if improved:
        return "good"
    if degraded:
        return "bad"
    return "same"


def write_html_report(
    path: Path,
    summary: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    model: str,
    runs: int,
    temperature: float,
) -> None:
    table_rows = []
    for s in summary:
        cls = delta_class(s["metric"], s["delta"])
        sign = "+" if s["delta"] > 0 else ""
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(s['label'])}</td>"
            f"<td>{format_value(s['V1'], s['unit'])}</td>"
            f"<td>{format_value(s['V2'], s['unit'])}</td>"
            f"<td class='{cls}'>{sign}{format_value(s['delta'], s['unit'])}</td>"
            "</tr>"
        )

    case_rows = []
    grouped = {}
    for r in rows:
        grouped.setdefault((r["version"], r["case_id"]), []).append(r)

    case_ids = [c["id"] for c in TEST_CASES]
    for cid in case_ids:
        v1 = grouped.get(("V1", cid), [])
        v2 = grouped.get(("V2", cid), [])
        q1 = mean(v1, "quality_score") if v1 else 0
        q2 = mean(v2, "quality_score") if v2 else 0
        s1 = 100 * mean(v1, "overall_success") if v1 else 0
        s2 = 100 * mean(v2, "overall_success") if v2 else 0
        case_rows.append(
            "<tr>"
            f"<td>{cid}</td>"
            f"<td>{q1:.1f}</td><td>{q2:.1f}</td>"
            f"<td>{s1:.1f}%</td><td>{s2:.1f}%</td>"
            "</tr>"
        )

    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Flash Card Skill 升级前后性能对比</title>
<style>
  body {{
    font-family: Arial, "Microsoft YaHei", sans-serif;
    max-width: 1100px; margin: 36px auto; padding: 0 20px;
    color: #1f2937; background: #f7f8fa;
  }}
  h1, h2 {{ color: #111827; }}
  .meta {{ color: #6b7280; margin-bottom: 24px; }}
  .card {{
    background: white; border: 1px solid #e5e7eb; border-radius: 12px;
    padding: 20px; margin: 18px 0;
  }}
  table {{ width: 100%; border-collapse: collapse; background: white; }}
  th, td {{ padding: 11px 12px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #f3f4f6; }}
  .good {{ color: #067647; font-weight: 700; }}
  .bad {{ color: #b42318; font-weight: 700; }}
  .same {{ color: #667085; }}
  .barrow {{ display:grid; grid-template-columns: 190px 1fr 70px; gap:10px; align-items:center; margin:9px 0; }}
  .track {{ height:12px; background:#eef0f3; border-radius:999px; overflow:hidden; }}
  .fill {{ height:100%; background:#4f46e5; border-radius:999px; }}
  .note {{ font-size: 14px; color:#667085; }}
</style>
</head>
<body>
<h1>Flash Card Skill 升级前后性能对比</h1>
<div class="meta">
模型：{html.escape(model)} ｜ 每个测试重复：{runs} 次 ｜ temperature：{temperature} ｜ 测试用例：{len(TEST_CASES)} 个
</div>

<div class="card">
<h2>总体指标</h2>
<table>
<thead><tr><th>指标</th><th>升级前 V1</th><th>升级后 V2</th><th>变化</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
<p class="note">绿色表示改善，红色表示退化。响应耗时和 Token 数越低越好，其余指标越高越好。</p>
</div>

<div class="card">
<h2>核心质量得分可视化</h2>
"""
    # Use summary quality_score values for a simple visual.
    qs = next((s for s in summary if s["metric"] == "quality_score"), None)
    if qs:
        for label, val in [("升级前 V1", qs["V1"]), ("升级后 V2", qs["V2"])]:
            width = max(0, min(100, float(val)))
            content += (
                f"<div class='barrow'><div>{label}</div>"
                f"<div class='track'><div class='fill' style='width:{width:.2f}%'></div></div>"
                f"<div>{val:.1f}</div></div>"
            )

    content += f"""
</div>

<div class="card">
<h2>逐测试用例对比</h2>
<table>
<thead>
<tr><th>用例</th><th>V1 质量得分</th><th>V2 质量得分</th><th>V1 综合成功率</th><th>V2 综合成功率</th></tr>
</thead>
<tbody>{''.join(case_rows)}</tbody>
</table>
</div>

<div class="card">
<h2>评估说明</h2>
<ul>
<li>同一模型、同一测试集、同一 temperature 下进行 A/B 对比。</li>
<li>质量得分由 12 个确定性指标等权平均，包括 JSON、字段、目标词、例句、近义词和 HTML 生成。</li>
<li>本报告不使用另一个 LLM 作为主观裁判，因此重点测量格式稳定性、执行可靠性和可自动验证约束。</li>
<li>大模型具有随机性，建议每个用例至少重复 3 次；课程报告建议使用 5 次。</li>
</ul>
</div>
</body></html>"""

    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, default=DEFAULT_V1)
    parser.add_argument("--v2", type=Path, default=DEFAULT_V2)
    parser.add_argument("--model", default=os.getenv("AGENT_MODEL", "qwen-max"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs 必须 >= 1")

    if not SCRIPT_PATH.exists():
        print(f"警告：未找到 HTML 生成脚本：{SCRIPT_PATH}")
        print("HTML 生成成功率将记为 0。")

    client = make_client()

    all_rows = []
    all_rows += evaluate_version("V1", args.v1, client, args.model, args.runs, args.temperature)
    all_rows += evaluate_version("V2", args.v2, client, args.model, args.runs, args.temperature)

    summary = build_summary(all_rows)

    detail_csv = BASE_DIR / "skill_comparison_results.csv"
    summary_csv = BASE_DIR / "skill_comparison_summary.csv"
    html_report = BASE_DIR / "skill_comparison_report.html"

    write_csv(detail_csv, all_rows)
    write_csv(summary_csv, summary)
    write_html_report(html_report, summary, all_rows, args.model, args.runs, args.temperature)

    print("\n" + "=" * 80)
    print("Flash Card Skill 升级前后对比")
    print("=" * 80)
    print(f"{'指标':<22}{'V1':>14}{'V2':>14}{'变化':>14}")
    print("-" * 80)
    for s in summary:
        sign = "+" if s["delta"] > 0 else ""
        print(
            f"{s['label']:<22}"
            f"{format_value(s['V1'], s['unit']):>14}"
            f"{format_value(s['V2'], s['unit']):>14}"
            f"{(sign + format_value(s['delta'], s['unit'])):>14}"
        )
    print("=" * 80)
    print(f"明细 CSV：{detail_csv}")
    print(f"汇总 CSV：{summary_csv}")
    print(f"可视化报告：{html_report}")


if __name__ == "__main__":
    main()
