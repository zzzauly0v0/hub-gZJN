---
name: ppt-to-web-fast
description: >-
  将 PPTX 课件高效提炼为 HTML 网页（单步直接生成，暗色主题）。
  Use when user asks to convert PPT/PPTX to web page / 网页.
---

# PPT to Web Fast（高效版）

单步将 PPTX 直接转为 HTML，无中间文件，CSS 精简，执行效率高。

## 执行流程

```bash
python skills/ppt-to-web-fast/scripts/ppt_to_web_fast.py <input.pptx> -o <output>.html
```

## 特性

- 单步直接生成（PPTX → HTML，无中间 JSON）
- 暗色主题 + 侧边导航 + 卡片布局
- 自动跳过水印文本（"八斗学院"、"盗版"）
- 支持 Python 3.8+，依赖 python-pptx
