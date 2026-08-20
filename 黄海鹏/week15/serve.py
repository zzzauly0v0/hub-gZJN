"""
Web界面 - 旅行规划助手可视化
展示主Agent + 并行子Agent的调用链路
"""
from flask import Flask, render_template, request, jsonify, Response
import json
import logging
import time
from agents import plan_travel

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# ============================================================
# SSE 事件流（实时推送步骤）
# ============================================================
@app.route("/")
def index():
    """首页"""
    return render_template("index.html")


@app.route("/plan", methods=["POST"])
def plan():
    """规划接口 - 返回JSON"""
    data = request.get_json()
    query = data.get("query", "")
    serial = data.get("serial", False)
    
    if not query:
        return jsonify({"error": "请输入问题"}), 400
    
    # 记录执行过程
    main_steps = []
    sub_steps = {}
    sub_results = {}
    dispatch_info = None
    
    def on_main_step(step):
        main_steps.append(step)
    
    def on_subagent_step(sid, step):
        if sid not in sub_steps:
            sub_steps[sid] = []
        sub_steps[sid].append(step)
    
    def on_subagent_done(sid, duration, task):
        sub_results[sid] = {"duration": duration, "task": task}
        print(f"[{sid}] 完成: {task} ({duration}s)")
    
    def on_dispatch(info):
        nonlocal dispatch_info
        dispatch_info = info
        print(f"[主Agent] 派发 {len(info['subtasks'])} 个子任务")
    
    # 执行
    result = plan_travel(
        query=query,
        on_main_step=on_main_step,
        on_subagent_step=on_subagent_step,
        on_subagent_done=on_subagent_done,
        on_dispatch=on_dispatch,
        serial=serial
    )
    
    return jsonify({
        "final_answer": result["final_answer"],
        "main_steps": main_steps,
        "sub_steps": sub_steps,
        "sub_results": sub_results,
        "parallel_stats": result["parallel_stats"],
        "dispatches": result["dispatches"],
        "dispatch_info": dispatch_info
    })


# ============================================================
# SSE 流式接口（实时推送）
# ============================================================
@app.route("/plan/stream", methods=["POST"])
def plan_stream():
    """流式规划接口 - SSE"""
    data = request.get_json()
    query = data.get("query", "")
    serial = data.get("serial", False)
    
    if not query:
        return jsonify({"error": "请输入问题"}), 400
    
    def generate():
        # 事件队列：plan_travel 在后台线程执行，回调把事件放入队列，此处逐个 yield
        import queue
        import threading
        ev_queue = queue.Queue()
        
        def emit(ev):
            ev_queue.put(ev)
        
        def on_main_step(step):
            emit({"type": "main_step", **step})
        
        def on_subagent_step(sid, step):
            emit({"type": "subagent_step", "subagent_id": sid, **step})
        
        def on_subagent_done(sid, duration, task):
            emit({"type": "subagent_done", "subagent_id": sid, "duration": duration, "task": task})
        
        def on_dispatch(info):
            emit({"type": "dispatch", "subtasks": info.get("subtasks", []), "subagent_ids": info.get("subagent_ids", [])})
        
        def run_plan():
            result = plan_travel(
                query=query,
                on_main_step=on_main_step,
                on_subagent_step=on_subagent_step,
                on_subagent_done=on_subagent_done,
                on_dispatch=on_dispatch,
                serial=serial
            )
            ev_queue.put({"type": "done", "result": result})
        
        threading.Thread(target=run_plan, daemon=True).start()
        
        # 逐个发送事件，直到规划完成
        while True:
            ev = ev_queue.get()
            if ev.get("type") == "done":
                result = ev["result"]
                yield f"data: {json.dumps({'type': 'final', 'answer': result.get('final_answer', ''), 'parallel_stats': result.get('parallel_stats', [])})}\n\n"
                break
            yield f"data: {json.dumps(ev)}\n\n"
    
    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
