"""
FastAPI HTTP 服务 — 带 SSE 事件流的可视化后端 + 技能系统
"""
# 增加技能导入
import os
import sys
import json
import sqlite3
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.session_db import SessionDB
from src.memory_loader import MemoryLoader
from src.vector_store import VectorStore
from src.fts_store import FTSStore
from src.retrieval import HybridRetriever
from src.memory_flush import MemoryFlusher
from src.llm_config import get_chat_client, current_model_info
from src.heartbeat_parser import HeartbeatParser
from src.scheduler import HeartbeatScheduler

# ========== 新增：技能系统导入 ==========
from src.skill import get_registry, SkillExecutor
# =======================================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ... 全局变量定义不变 ...

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, loader, vs, fts, retriever, flusher, current_session_id, hb_parser, hb_scheduler
    db = SessionDB()
    loader = MemoryLoader()
    vs = VectorStore()
    fts = FTSStore()
    retriever = HybridRetriever(vs, fts)
    flusher = MemoryFlusher()
    hb_parser = HeartbeatParser()
    current_session_id = db.new_session()
    logger.info(f"服务启动，会话 #{current_session_id}")
    logger.info(f"FTS5/BM25 可用：{fts.available}（{'混合检索' if fts.available else '退化为纯向量'}）")

    # ========== 新增：打印已加载技能 ==========
    registry = get_registry()
    logger.info(f"已加载 {len(registry.get_all())} 个技能")
    # ========================================

    hb_scheduler = HeartbeatScheduler()
    hb_scheduler.start(broadcast)
    logger.info("HEARTBEAT 调度器已启动")

    yield

    hb_scheduler.stop()
    if current_session_id:
        db.close_session(current_session_id)


app = FastAPI(title="Agent 记忆系统", lifespan=lifespan)

# ... 请求模型等保持不变 ...

@app.post("/chat")
async def chat(req: ChatRequest):
    sid = req.session_id or current_session_id

    async def stream():
        # Step 1-2: 记忆加载与检索（不变）
        prompt_result = loader.build_system_prompt(recent_memory_limit=10)
        layers_info = [...]
        yield sse_event("memory_load", {...})
        await asyncio.sleep(0)

        semantic_results = retriever.search(req.message, top_k=3)
        yield sse_event("semantic_search", {...})
        await asyncio.sleep(0)

        # ========== 新增：技能匹配与执行 ==========
        registry = get_registry()
        matched_skill = registry.match(req.message)
        if matched_skill:
            yield sse_event("skill_triggered", {"name": matched_skill.name})
            ctx = {
                "session_id": sid,
                "db": db,
                "loader": loader,
                "vs": vs,
                "fts": fts,
                "retriever": retriever,
                "flusher": flusher,
                "llm_client": get_chat_client,
            }
            executor = SkillExecutor(ctx)
            try:
                # 简单参数抽取：如果技能需要 city，直接把整个消息传进去
                skill_result = await executor.execute(matched_skill.name, city=req.message)
                if skill_result:
                    # 技能直接回复，跳过 LLM
                    db.add_message(sid, "assistant", skill_result)
                    yield sse_event("skill_result", {
                        "name": matched_skill.name,
                        "result": skill_result
                    })
                    yield sse_event("done", {
                        "response": skill_result,
                        "session_id": sid,
                        "message_count": db.get_message_count(sid),
                        "auto_flush_threshold": 20,
                        "from_skill": True,
                    })
                    return  # 直接结束流，不走后续 LLM
            except Exception as e:
                logger.error(f"技能执行失败: {e}")
                yield sse_event("skill_error", {"name": matched_skill.name, "error": str(e)})
                # 技能失败后继续走 LLM 流程（降级）
        # ========================================

        # Step 3: 组装 Context（不变）
        semantic_context = ""
        if semantic_results:
            snippets = [...]
            semantic_context = "## 语义检索到的相关记忆\n" + "\n".join(snippets)

        system_prompt = prompt_result.system_prompt
        if semantic_context:
            system_prompt += "\n\n" + semantic_context

        history = db.get_session_messages(sid)
        history_for_api = [{"role": m["role"], "content": m["content"]} for m in history]

        yield sse_event("context_assembly", {...})
        await asyncio.sleep(0)

        # Step 4: LLM 流式生成（不变）
        api_messages = ([{"role": "system", "content": system_prompt}] +
                        history_for_api +
                        [{"role": "user", "content": req.message}])
        client, model = get_chat_client()
        stream_resp = client.chat.completions.create(...)
        full_response = ""
        for chunk in stream_resp:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_response += delta
                yield sse_event("token", {"text": delta})

        db.add_message(sid, "user", req.message)
        db.add_message(sid, "assistant", full_response)

        msg_count = db.get_message_count(sid)
        yield sse_event("done", {...})

        # Step 5: 后台检测调度意图（不变）
        if hb_parser and hb_parser.may_contain_schedule_intent(req.message):
            asyncio.create_task(_check_schedule_intent(req.message))

    return StreamingResponse(stream(), media_type="text/event-stream")


# 其余接口（/flush, /memories, /stream, /reset, /health, /session/new）均保持不变
# 不再重复列出，避免冗余。
