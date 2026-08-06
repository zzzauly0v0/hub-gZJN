---
name: pptx-summary
description: >-
  从 PowerPoint (.pptx) 幻灯片中提取文字，生成精美的深色主题 HTML 总结页。
  可选使用 baoyu-diagram 设计系统创建架构 SVG 图。当用户要求提取、总结或
  可视化 PowerPoint 内容时使用，例如"把 pptx 文字提取出来"、"总结这个PPT并
  生成HTML"、"把slides转成网页"、"extract slides to HTML"。
---
# PPTX Summary — 幻灯片提取与HTML总结

从 .pptx 文件中提取所有文字，生成精美的深色主题 HTML 总结，可选 SVG 图表。

## 工作流程

### 步骤 1：定位 PPTX 文件

确认文件路径。如果用户未提供完整路径，递归搜索：

`powershell
Get-ChildItem -Recurse -Filter "*.pptx" -ErrorAction SilentlyContinue | Select-Object FullName
`

### 步骤 2：提取文字

运行附带的提取脚本。该脚本使用 PowerPoint COM 自动化（仅限 Windows）：

`powershell
powershell -ExecutionPolicy Bypass -File "{skillDir}/scripts/extract_pptx.ps1" -PptxPath "<路径>" -OutputPath "<输出>.md"
`

脚本输出一个 Markdown 文件，每页一个章节（## 第 N 页），以 --- 分隔。

### 步骤 3：阅读并分析提取的内容

读取生成的 .md 文件以理解幻灯片结构：
- 从幻灯片标题和 Part 标记中识别章节/部分分组
- 将相关幻灯片归入逻辑章节
- 提取关键概念、定义、数据点和结论

### 步骤 4：生成 HTML 总结

读取 {skillDir}/references/html-design.md 了解深色主题设计系统（颜色、
字体、卡片样式、响应式网格）。构建一个自包含的 HTML 文件，包含：

- **Hero 区域**：标题、副标题、关键词标签
- **章节区域**：每个逻辑章节一个 <section>，标注 Part N
- **卡片与网格**：使用 .grid2、.grid3、.grid4 组织概念组；使用 .card.cyan/.green/.purple 等添加彩色左边框
- **表格**：用于对比数据（如 FC vs MCP vs RAG vs Skills）
- **代码块**：使用 <pre> 和等宽字体展示 JSON/YAML/代码示例
- **高亮框**：使用 .highlight 突出核心结论和要点
- **响应式**：通过 CSS 媒体查询适配移动端

HTML 必须完全自包含（除 Google Fonts 外无外部 CSS）。

### 步骤 5（可选）：创建 SVG 图表

如果内容涉及架构关系、演进路径或工作流程，使用 baoyu-diagram 技能的设计系统
创建辅助 SVG 图表：

- 深色背景（#0f172a）配合微妙网格
- 通过 Google Fonts 引入 JetBrains Mono 字体
- 语义化配色方案（cyan/emerald/amber/purple/rose/blue）
- 将 SVG 保存至 diagram/ 子目录

### 步骤 6：链接图表并展示结果

使用按钮样式锚点将图表链接插入 HTML：
`html
<div class="diagram-link">
  <a href="diagram/xxx.svg" target="_blank" class="btn btn-cyan">📐 查看架构图 (SVG)</a>
</div>
`

在默认浏览器中打开 HTML：
`powershell
Start-Process "<html路径>"
`

## 输出文件

| 文件 | 描述 |
|------|------|
| <名称>_extracted.md | 原始文字提取，每页一个章节 |
| <名称>_summary.html | 精美的深色主题 HTML 总结 |
| diagram/*.svg | 可选的辅助图表（如适用） |

## 依赖

- 已安装 PowerPoint（Windows COM 自动化）
- baoyu-diagram 技能（用于可选的 SVG 图表）
