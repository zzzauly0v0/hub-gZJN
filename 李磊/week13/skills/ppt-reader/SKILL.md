---
name: "PPT Reader"
description: "读取 .pptx 文件中的文字内容。当用户需要读取、查看、提取 PPT/幻灯片内容时触发。使用 python-pptx 库提取所有文本框和表格中的文字，逐页输出。"
when_to_use: "用户提到 PPT、pptx、幻灯片、PowerPoint、读取PPT、提取PPT内容时自动触发"
argument-hint: "[ppt文件路径]"
allowed-tools: "Read, Write, Bash, Glob, Grep"
---

# PPT Reader

读取 PowerPoint (.pptx) 文件的文字内容，逐页输出所有文本框和表格中的文本。

## 依赖检查

执行前先检查 `python-pptx` 是否可用：

```bash
D:/conda/python.exe -c "from pptx import Presentation; print('OK')"
```

如果报错，提示用户安装：`conda activate base && pip install python-pptx`

## 执行

使用 `${CLAUDE_SKILL_DIR}/scripts/read_ppt.py` 读取 PPT：

```bash
D:/conda/python.exe -X utf8 ${CLAUDE_SKILL_DIR}/scripts/read_ppt.py "$ARGUMENTS"
```

## 输出说明

- 输出内容包括：文件名、总页数、每页的文本框和表格内容
- 仅提取文字，不提取矢量图形和嵌入图片
- 如果输出过长，Read 工具读取临时输出文件
