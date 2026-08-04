"""
FastAPI 路由：提供 REST API 和静态页面。
"""
import json
import os
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pathlib import Path

from .skill_loader import SkillLoader
from .engine import QueryEngine
from .llm import LLMClient, PROVIDERS

app = FastAPI(title="英语单词学习 Harness", version="0.2.0")

BASE_DIR = Path(__file__).resolve().parent.parent
SKILLS_DIR = BASE_DIR / "skills"
TEMPLATE_FILE = BASE_DIR / "templates" / "index.html"


def _init_llm() -> "LLMClient | None":
    """
    初始化 LLM 客户端。优先级：
    1. 环境变量 HARNESS_LLM_PROVIDER 指定
    2. 自动检测本地已配置的 API Key（deepseek → qwen）
    3. 都没有则返回 None
    """
    # 1) 显式指定
    explicit = os.getenv("HARNESS_LLM_PROVIDER", "").strip()
    if explicit and explicit in PROVIDERS:
        try:
            return LLMClient(provider=explicit)
        except ValueError as e:
            print(f"[Server] {explicit} 初始化失败: {e}")

    # 2) 自动检测：按优先级遍历
    for name in ("deepseek", "qwen"):
        cfg = PROVIDERS[name]
        if os.getenv(cfg["api_key_env"], "").strip():
            try:
                client = LLMClient(provider=name)
                print(f"[Server] 自动检测到 {cfg['display_name']} API Key，已启用")
                return client
            except Exception:
                continue

    print("[Server] 未检测到可用的 LLM API Key，LLM 功能不可用")
    print("[Server] 可设置环境变量 HARNESS_LLM_PROVIDER=deepseek 或 qwen")
    return None


# ---- LLM 初始化 ----
llm = _init_llm()

# ---- 核心组件 ----
loader = SkillLoader(SKILLS_DIR)
loader.discover()  # 懒加载：此时只加载 header（name/description/触发场景）
# 为了页面展示完整信息，预激活所有 skill（加载 data/ 和执行流程）
# 实际生产中可按需激活，这里演示完整功能
for _skill in loader.list_skills():
    loader.activate(_skill.name)
engine = QueryEngine(loader, llm=llm)

# 预读 HTML 模板
_HTML_TEMPLATE = TEMPLATE_FILE.read_text(encoding="utf-8") if TEMPLATE_FILE.exists() else ""


# ---- 页面 ----

@app.get("/", response_class=HTMLResponse)
async def index():
    """首页：单词查询界面。"""
    skills_data = []
    for s in loader.list_skills():
        skills_data.append({
            "name": s.name,
            "word_count": len(s.word_index),
            "triggers": s.trigger_examples[:3],
            "steps": s.execution_steps,
        })

    skills_json = json.dumps(skills_data, ensure_ascii=False)

    triggers_html = ""
    for s in loader.list_skills():
        for ex in s.trigger_examples[:3]:
            triggers_html += f'<button class="tip-tag trigger" onclick="doSearch(\'{ex}\')">{ex}</button>'

    html = _HTML_TEMPLATE
    html = html.replace("__SKILLS_JSON__", skills_json)
    html = html.replace("__TRIGGER_BUTTONS__", triggers_html)
    html = html.replace("__LLM_PROVIDER__", llm.display_name if llm else "")

    return HTMLResponse(content=html)


# ---- API ----

@app.get("/api/skills")
async def list_skills():
    """列出所有已注册的 skill 及其完整信息。"""
    return {
        "skills": [
            {
                "name": s.name,
                "description": s.description,
                "word_count": len(s.word_index),
                "words": list(s.word_index.keys()),
                "trigger_examples": s.trigger_examples,
                "execution_steps": s.execution_steps,
            }
            for s in loader.list_skills()
        ],
        "llm": {
            "enabled": llm is not None,
            "provider": llm.display_name if llm else None,
        },
    }


@app.get("/api/query")
async def query_word(q: str = Query(..., description="查询关键词或自然语言指令")):
    """查询单词，支持自然语言触发和关键词搜索。"""
    results = await engine.query(q)
    return {
        "query": q,
        "count": len(results),
        "results": [
            {
                "word": r.word,
                "phonetic": r.phonetic,
                "pos": r.pos,
                "definition": r.definition,
                "examples": r.examples,
                "synonyms": r.synonyms,
                "skill_name": r.skill_name,
                "matched_by": r.matched_by,
                "llm_generated": r.llm_generated,
            }
            for r in results
        ],
    }
