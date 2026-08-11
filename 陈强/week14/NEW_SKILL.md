---
name: pptx-summary
description: 提取 PowerPoint 文字并生成深色主题 HTML 总结页，可选 SVG 架构图。当用户需要提取、总结或可视化 PPTX 内容时使用，如"总结PPT生成HTML"、"extract slides to HTML"。
---

# PPTX 摘要 — 幻灯片提取与 HTML 总结

提取 .pptx 文字，生成深色主题 HTML 总结，可选 SVG 图表。

## 工作流程

### 1. 定位与提取

若用户未指定路径，先搜索 .pptx 文件；运行提取脚本（仅 Windows，依赖 PowerPoint COM）：

```powershell
powershell -ExecutionPolicy Bypass -File "{skillDir}/scripts/extract_pptx.ps1" -PptxPath "<路径>" -OutputPath "<输出>.md"
```

输出：Markdown 格式，每页 `## 第 N 页`，`---` 分隔。

### 2. 分析内容

读取生成的 .md 文件，识别章节分组、提取关键概念/定义/数据/结论。

### 3. 生成 HTML

参考 `{skillDir}/references/html-design.md` 中的深色主题设计规范，构建自包含的 HTML 文件：

- **Hero 区**：标题 + 副标题 + 关键词标签
- **章节区**：每逻辑章一个 `<section>`，标注 Part N
- **组件**：卡片（.card.cyan/.green/.purple...）、网格（.grid2/.grid3/.grid4）、表格、代码块（<pre>）、高亮框（.highlight）、原则列表（.principle）、数值卡片（.value-card）
- **响应式**：媒体查询适配移动端（网格→单列，标题→32px）
- 完全自包含（仅 Google Fonts 外部引用）

### 4. 图表与展示（可选）

架构/流程图用 baoyu-diagram 技能生成 SVG。通过按钮链接插入 HTML：

```html
<div class="diagram-link">
  <a href="diagram/x.svg" target="_blank" class="btn btn-cyan">📐 图表名 (SVG)</a>
</div>
```

打开 HTML：`Start-Process "<html路径>"`

## 输出文件

| 文件 | 描述 |
|------|------|
| `<名称>_extracted.md` | 每页文字提取 |
| `<名称>_summary.html` | 深色主题 HTML 总结 |
| `diagram/*.svg` | 可选的辅助图表 |

## 依赖

- PowerPoint 已安装（Windows COM 自动化）
- baoyu-diagram 技能（可选，用于 SVG 图表）
