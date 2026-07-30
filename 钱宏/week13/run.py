"""
Harness Engineering - Skills 渐进式加载演示

启动方式：
    # 方式1：命令行（推荐）
    python -m uvicorn serve:app --host 0.0.0.0 --port 8001
    
    # 方式2：PyCharm 调试模式（支持断点）
    在 PyCharm 中点击 Debug 运行本文件

访问：
    http://localhost:8001

环境变量：
    DEEPSEEK_API_KEY  # 可选，配置后启用真实 LLM
    LLM_BASE_URL      # 可选，默认 https://api.deepseek.com
    LLM_MODEL         # 可选，默认 deepseek-v4-flash
"""

import os
import sys

# ── 启动配置 ──
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8001"))

print(f"启动 Harness 演示服务...")
print(f"访问地址: http://localhost:{PORT}")
print(f"提示：在浏览器中打开 http://localhost:{PORT} 查看界面")

# ── 启动方式：兼容 PyCharm 调试器 ──
if __name__ == "__main__":
    import uvicorn
    
    # 创建 Config
    config = uvicorn.Config(
        "serve:app",
        host=HOST,
        port=PORT,
        log_level="info",
        reload=False,
    )
    
    # 创建 Server
    server = uvicorn.Server(config)
    
    # 使用 asyncio 直接运行，避免 loop_factory 参数问题
    # 这使得 PyCharm 调试器可以正常工作
    import asyncio
    
    # 检查是否在调试器中运行
    if 'pydevd' in sys.modules:
        # 在调试器中：直接运行，支持断点
        print("检测到 PyCharm 调试器，启用断点支持...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()
    else:
        # 正常模式
        asyncio.run(server.serve())
