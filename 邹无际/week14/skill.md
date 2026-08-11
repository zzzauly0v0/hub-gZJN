---
name: "md-to-docx"
description: "Converts Markdown files to formatted Word (docx) documents with unified fonts, styled tables, and clear hierarchy. Invoke when user says '生成doc', '生成docx', '转docx', '导出doc', '转word', or asks to convert md to docx."
---

# MD to DOCX Converter

将 Markdown 文件转换为格式统一的 Word (docx) 文档。转换逻辑已封装在本目录 `md_to_docx.py` 中，**直接运行即可，不要复制或重写脚本**。

## 触发条件

当用户说以下任意关键词时启动本 skill：
- 生成 doc / 生成 docx / 生成 word
- 转 doc / 转 docx / 转 word
- 导出 doc / 导出 docx
- 将 md 转(换/导出)为 docx

## 执行步骤

1. 确认要转换的 `.md` 文件路径（可一次传多个，空格分隔）。
2. 直接运行脚本（工作目录为项目根目录）：
   ```powershell
   python ".trae\skills\md-to-docx\md_to_docx.py" "<md文件1>" ["<md文件2>" ...]
   ```
   - 路径含空格/中文必须加双引号。
   - docx 默认生成在与 md 同目录、同名；目标被占用时脚本自动加 `_v2` 重试。
   - 需指定输出位置时追加 `--out "<目录或docx路径>"`。
3. 若报 `ModuleNotFoundError: docx`，先执行 `pip install python-docx` 再重试。
4. 把生成的 docx 路径告知用户，并简述格式规范（统一微软雅黑、标题层级、表格样式等）。

## 内置格式规范（勿改动脚本）

- 全文字体微软雅黑，代码 Consolas（#C7254E）
- 文档标题 22pt / 二级 16pt / 三级 14pt / 正文 12pt，标题深藏青（#1A3C6E / #2B5797）
- 表头深蓝底（#1A3C6E）白字加粗居中，数据行交替浅蓝底（#E8EFF8）
- 引用块：左侧蓝边框 + 浅蓝底（#EDF2F9）
- 页边距：上下 2cm，左右 2.5cm
- 文档标题与 docx 文件名均取 md 文件名
