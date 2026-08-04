"""
英语单词学习 Harness 入口

用法:
    python app.py                  # 开发模式启动（http://localhost:8000）
    uvicorn app:harness_app --reload
"""
import uvicorn
from harness.server import app as harness_app

if __name__ == "__main__":
    uvicorn.run("harness.server:app", host="0.0.0.0", port=8000, reload=True)
