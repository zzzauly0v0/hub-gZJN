"""旅行规划 Subagent HTTP 服务（FastAPI + 异步 SSE 流式）。"""
import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app):
    logger.info("旅行规划 subagent 服务就绪（异步架构）")
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QueryRequest(BaseModel):
    question: str
    serial: bool = False


@app.get("/health")
def health():
    return {"status": "ok",
            "tavily": bool(os.getenv("TAVILY_API_KEY")),
            "llm": bool(os.getenv("DEEPSEEK_API_KEY")),
            "arch": "async"}


@app.post("/query")
async def query(req: QueryRequest):
    import agents

    async def event_stream():
        q: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        async def push(ev):
            await q.put(ev)

        async def on_main_step(step):
            await push({"type": "main_step", **step})

        async def on_dispatch(info):
            await push({"type": "dispatch", **info})

        async def on_subagent_step(sid, step):
            await push({"type": "subagent_step", "subagent_id": sid, **step})

        async def on_subagent_done(sid, duration, topic):
            await push({"type": "subagent_done", "subagent_id": sid,
                        "duration": duration, "subtopic": topic})

        async def run():
            try:
                r = await agents.run_research(
                    req.question,
                    on_main_step=on_main_step,
                    on_dispatch=on_dispatch,
                    on_subagent_step=on_subagent_step,
                    on_subagent_done=on_subagent_done,
                    serial=req.serial,
                )
                await push({"type": "final", "answer": r["final_answer"],
                            "parallel_stats": r["parallel_stats"],
                            "main_trace_len": len(r["main_trace"]),
                            "subagent_count": len(r["subagents"])})
            except Exception as e:
                await push({"type": "error",
                            "message": f"{type(e).__name__}: {str(e)[:200]}"})
            finally:
                await q.put(SENTINEL)

        task = asyncio.create_task(run())

        yield "data: " + json.dumps({"type": "start", "question": req.question},
                                   ensure_ascii=False) + "\n\n"
        while True:
            ev = await q.get()
            if ev is SENTINEL:
                yield "data: " + json.dumps({"type": "done"},
                                           ensure_ascii=False) + "\n\n"
                break
            yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
        await task

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8003, reload=False)
