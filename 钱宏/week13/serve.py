"""
Harness Engineering - Skills 渐进式加载演示服务

API 接口：
  POST /chat           - 发送消息，触发 Harness 流程
  GET  /skills         - 获取所有 Skill 列表
  GET  /skills/status  - 获取 Skill 加载状态
  POST /skills/{name}/load   - 手动加载指定 Skill
  POST /skills/{name}/unload - 手动卸载指定 Skill
  GET  /engine/status  - 获取引擎状态
  POST /engine/reset   - 重置引擎上下文

使用方式：
  cd week13
  python -m uvicorn serve:app --host 0.0.0.0 --port 8001
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# 确保可以导入 harness 模块
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from harness import SkillManager, HarnessEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── 全局实例 ──────────────────────────────────────────────────────────────────
skill_manager: Optional[SkillManager] = None
engine: Optional[HarnessEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化 Skill 管理器和 Harness 引擎"""
    global skill_manager, engine
    
    skills_dir = str(BASE_DIR / "skills")
    logger.info(f"初始化 Skill 管理器，扫描目录: {skills_dir}")
    
    skill_manager = SkillManager(skills_dir=skills_dir)
    engine = HarnessEngine(
        skill_manager=skill_manager,
        auto_load_skills=True,
    )
    
    # 设置工具执行器（从 Skill 配置中加载）
    for skill in skill_manager.skills.values():
        if skill.executor and isinstance(skill.executor, dict):
            for tool_name, executor in skill.executor.items():
                engine.register_tool_executor(tool_name, executor)
    
    # 如果有 LLM API Key，配置真实的 LLM 函数
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
            )
            model = os.getenv("LLM_MODEL", "deepseek-v4-flash")
            
            def llm_func(messages: List[Dict], tools: List[Dict]):
                """LLM 调用函数（同步，OpenAI SDK 本身是同步的）"""
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None,
                    temperature=0,
                )
                return response.choices[0].message
            
            engine.set_llm_func(llm_func)
            logger.info(f"已配置 LLM: {model}")
        except Exception as e:
            logger.warning(f"LLM 配置失败，使用演示模式: {e}")
    
    logger.info("Harness 引擎初始化完成")
    yield


app = FastAPI(title="Harness Engineering - Skills 渐进式加载", lifespan=lifespan)


# ── 请求模型 ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    reset_context: bool = False


class ChatResponse(BaseModel):
    type: str
    message: Optional[str] = None
    data: Optional[Dict] = None


# ── SSE 辅助 ──────────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    """发送消息，触发 Harness 流程（流式响应）"""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    if req.reset_context:
        engine.reset_context()
    
    async def generate():
        loop = asyncio.get_event_loop()
        
        def _run():
            return list(engine.run(req.question))
        
        steps = await loop.run_in_executor(None, _run)
        
        for step in steps:
            yield _sse(step)
        
        yield _sse({"type": "stream_done"})
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/skills")
async def list_skills():
    """获取所有 Skill 列表"""
    if not skill_manager:
        raise HTTPException(status_code=500, detail="Skill manager not initialized")
    return {"skills": skill_manager.list_skills()}


@app.get("/skills/status")
async def skills_status():
    """获取 Skill 加载状态"""
    if not skill_manager:
        raise HTTPException(status_code=500, detail="Skill manager not initialized")
    return skill_manager.get_status()


@app.post("/skills/{skill_name}/load")
async def load_skill(skill_name: str):
    """手动加载指定 Skill"""
    if not skill_manager:
        raise HTTPException(status_code=500, detail="Skill manager not initialized")
    
    skill = skill_manager.load_skill(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found or load failed")
    
    # 注册工具执行器
    if skill.executor and isinstance(skill.executor, dict) and engine:
        for tool_name, executor in skill.executor.items():
            engine.register_tool_executor(tool_name, executor)
    
    return {"status": "loaded", "skill": skill.to_dict()}


@app.post("/skills/{skill_name}/unload")
async def unload_skill(skill_name: str):
    """手动卸载指定 Skill"""
    if not skill_manager:
        raise HTTPException(status_code=500, detail="Skill manager not initialized")
    
    success = skill_manager.unload_skill(skill_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")
    
    return {"status": "unloaded", "skill": skill_name}


@app.get("/engine/status")
async def engine_status():
    """获取引擎状态"""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    return engine.get_status()


@app.post("/engine/reset")
async def engine_reset():
    """重置引擎上下文"""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    engine.reset_context()
    engine.skill_manager.unload_all()
    return {"status": "reset"}


# ── 托管前端页面 ──────────────────────────────────────────────────────────────
HTML_PATH = BASE_DIR / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Harness Engineering - Skills 渐进式加载演示</h2>")


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "engine_ready": engine is not None,
        "skills_ready": skill_manager is not None,
    }
