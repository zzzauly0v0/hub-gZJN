---
name: "project-flow-canvas"
description: "分析任意项目完整流程或某功能端到端逻辑，生成 Canvas 风格 HTML 画布（指标/概念澄清/流水线/痛点对照/功能卡片：测试用例+调用链路+代码位置+前置逻辑+面试问答/落地顺序/面试叙事）。当用户想可视化理解项目流程、梳理代码流转、做面试准备时调用。"
---

# 项目流程梳理画布生成器 (Project Flow Canvas)

把任意项目的「完整流程」或「某个功能」分析成一张 **Canvas 风格 HTML 画布**，帮助快速理清从前端到后端的代码流转，可用于自学、复盘、面试准备。

## 何时调用
- 用户说"梳理这个项目的流程""这个功能怎么实现的""生成一个画布/HTML""帮我理清代码流转""做个面试准备文档"等
- 用户想理解任意项目的完整流程 或 某个功能的端到端逻辑
- 用户明确想要可视化的 HTML 画布（而非纯 Markdown 聊天回复）
- 用户提到"像上次那样生成""用 Cursor Canvas 画布输出"

## 输入（向用户确认或从上下文推断）
1. **项目根目录路径**（必须）
2. **分析范围**：完整项目 / 某个功能 / 某几个技术要点（必须）
3. **排除项**：用户明确说不分析的部分（如"已实现的传统能力不分析""跳过 docs/ 简历文件夹"）
4. **输出位置**：默认放 `<项目根>/docs/` 下，文件名 `技术选型与流程梳理-Canvas画布.html`

## 输出规范
单个 HTML 文件，Canvas 白板卡片风格，**必须包含以下 7 大区块（按顺序）+ N 个功能详解卡片**。不要省略任何区块。

---

## 执行方法论（严格按此步骤）

### 第1步：明确范围与排除项
- 若用户已指定范围/排除项，直接采用；若不明确，用 AskUserQuestion 确认
- 如果用户提供简历/技术要点列表，以该列表为准逐条分析
- 记住排除项：分析全程不得读取/引用被排除的文件

### 第2步：并行探索代码（用 Agent 工具，关键提效步骤）
启动 **多个 Explore agent 并行**，每个负责一条链路。这是提速核心——不要自己逐文件读。
典型分工（按项目架构调整）：
- Agent A：前端入口 → API 调用 → SSE/请求机制 → 事件处理 → 用户交互（确认/点击）
- Agent B：后端服务入口 → 路由/分流/预处理逻辑
- Agent C：核心业务逻辑（Agent/检索/工具/算法）
- Agent D：数据层（DB/向量库/缓存/ETL/建索引）
- Agent E：离线预处理（训练/ETL/同步/配置）

每个 agent 的 prompt 要求返回：
- 完整文件路径
- 关键代码片段（带行号）
- 该文件/函数在系统中的作用
- 该链路的入口→出口顺序

### 第3步：精读关键文件
- 对每个 agent 返回的核心文件，用 **Read** 读取完整内容（不要只看摘录）
- 提取**精确行号**，统一用 `文件:行号` 格式（如 `agent.py:285`），便于用户在 IDE 搜索定位
- 识别**前置代码**：链路中需事先完成的离线/初始化逻辑（如 chunking、建索引、训练、数据集构造）

### 第4步：梳理每个功能的 4 要素（核心产出）
对每个要分析的功能/技术点，整理以下 4 部分：

**① 测试用例（页面操作步骤）**
- 写成用户视角的操作序列，如"进入X页→双击Y→输入Z→预期..."
- 用 testcase 样式块，带"用例标题 + 有序步骤 + 预期结果"
- 一个功能给 1-3 个用例，覆盖正常路径和验证点

**② 完整调用链路（按实现顺序）**
- 用 chain 列表，每步一行：`文件:行号 函数名 → 说明`
- **前置步骤用 `pre` 样式 + "前置"标签**标记（离线/初始化代码）
- 按真实执行顺序排，从用户操作到最终返回
- 如果一个功能分离线/在线两阶段，分阶段列表

**③ 前置代码逻辑（决策点详解，最重要）**
- 对链路中的"决策形选择"必须说明：**具体做法 + 为什么这样做**
- 用 decision 样式块（黄底虚线框）突出决策点
- 例如：chunking 策略为什么按语义单元切而非固定 token；为什么用质心而非 SVM；为什么持久化到 DB；LoRA 为什么只训某些层
- 每个决策点配 1-3 个前置子项，说明实现过程

**④ 面试问答**
- 2-4 个该功能的高频追问 + 解答
- 用 callout 样式块，Q 加粗，A 跟随
- 覆盖"为什么选这个方案""和 X 有什么区别""参数怎么定""失败怎么办"

### 第5步：生成 HTML 画布
- 按下方【HTML 模板】生成单个 HTML 文件
- 用 Write 写入项目 docs/ 目录
- 所有代码位置用 `<code>文件:行号</code>` 格式，便于搜索

---

## HTML 模板（直接套用，填入项目实际内容）

模板含完整 CSS（Canvas 白板风格）+ 7 区块骨架 + 功能卡片骨架。
**保留所有 CSS 不变**，只替换 `{{...}}` 占位符为实际内容。功能卡片按实际功能数复制。

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{项目名}} · 技术选型与流程梳理画布</title>
<style>
  :root{
    --bg:#eef1f6; --card:#ffffff; --ink:#1f2330; --sub:#5b6478;
    --line:#e3e8f0; --accent:#3b6ef0; --green:#16a34a; --orange:#ea8a1a;
    --red:#dc2626; --purple:#7c3aed; --teal:#0d9488; --pink:#db2777;
    --p0:#dc2626; --p1:#ea8a1a;
    --code-bg:#0f172a; --code-ink:#e2e8f0;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--ink);font-family:-apple-system,"PingFang SC","Microsoft YaHei",Segoe UI,Roboto,sans-serif;line-height:1.65;font-size:15px}
  .canvas{max-width:1280px;margin:0 auto;padding:32px 28px 80px}
  .title-bar{background:linear-gradient(135deg,#3b6ef0,#7c3aed);color:#fff;border-radius:16px;padding:26px 32px;margin-bottom:24px;box-shadow:0 8px 24px rgba(59,110,240,.25)}
  .title-bar h1{font-size:26px;letter-spacing:.5px}
  .title-bar p{opacity:.9;margin-top:6px;font-size:14px}
  .card{background:var(--card);border-radius:14px;padding:22px 26px;margin-bottom:22px;box-shadow:0 2px 12px rgba(30,40,70,.06);border:1px solid var(--line)}
  .card h2{font-size:19px;margin-bottom:16px;padding-bottom:10px;border-bottom:2px solid var(--line);display:flex;align-items:center;gap:8px}
  .card h2 .tag{font-size:12px;font-weight:600;padding:2px 10px;border-radius:20px;background:#eef2ff;color:var(--accent)}
  .card h3{font-size:16px;margin:18px 0 10px;color:var(--accent)}
  .card h4{font-size:14px;margin:14px 0 8px}
  .stats{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:24px}
  .stat{background:var(--card);border-radius:12px;padding:18px 16px;text-align:center;border:1px solid var(--line);box-shadow:0 2px 8px rgba(30,40,70,.05)}
  .stat .num{font-size:30px;font-weight:800;color:var(--accent)}
  .stat .num.g{color:var(--green)}.stat .num.o{color:var(--orange)}.stat .num.p{color:var(--purple)}.stat .num.t{color:var(--teal)}
  .stat .lbl{font-size:12px;color:var(--sub);margin-top:4px}
  table{width:100%;border-collapse:collapse;font-size:13.5px;margin:10px 0}
  th{background:#f5f7fb;text-align:left;padding:10px 12px;font-weight:600;color:var(--ink);border-bottom:2px solid var(--line);white-space:nowrap}
  td{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
  tr:hover td{background:#fafbfe}
  .callout{border-left:4px solid var(--accent);background:#f5f8ff;padding:12px 16px;border-radius:0 8px 8px 0;margin:12px 0;font-size:13.5px}
  .callout.warn{border-color:var(--orange);background:#fff7ed}
  .callout.danger{border-color:var(--red);background:#fef2f2}
  .callout.ok{border-color:var(--green);background:#f0fdf4}
  .callout b{color:var(--ink)}
  .pbadge{display:inline-block;font-size:11px;font-weight:700;padding:1px 8px;border-radius:6px;color:#fff}
  .P0{background:var(--p0)}.P1{background:var(--p1)}
  code,.code{font-family:"JetBrains Mono","Fira Code",Consolas,monospace;font-size:12.5px;background:#f1f3f9;padding:1px 6px;border-radius:4px;color:#be185d}
  pre{background:var(--code-bg);color:var(--code-ink);padding:14px 16px;border-radius:10px;overflow-x:auto;font-size:12.5px;line-height:1.6;margin:10px 0;font-family:Consolas,monospace}
  .chain{list-style:none;counter-reset:step;margin:10px 0}
  .chain li{counter-increment:step;position:relative;padding:10px 14px 10px 46px;margin-bottom:6px;background:#f8fafc;border-radius:8px;border:1px solid var(--line);font-size:13.5px}
  .chain li::before{content:counter(step);position:absolute;left:12px;top:10px;width:24px;height:24px;background:var(--accent);color:#fff;border-radius:50%;font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center}
  .chain li .pos{color:var(--accent);font-weight:600;font-size:12px}
  .chain li.pre{background:#fffbeb;border-color:#fde68a}
  .chain li.pre::before{background:var(--orange)}
  .chain li.pre .tag-pre{font-size:10px;background:var(--orange);color:#fff;padding:1px 6px;border-radius:4px;margin-left:6px}
  .feature{border:1px solid var(--line);border-radius:14px;overflow:hidden;margin-bottom:24px;background:#fff;box-shadow:0 2px 12px rgba(30,40,70,.06)}
  .feature .hd{padding:14px 22px;color:#fff;font-weight:700;font-size:16px;display:flex;align-items:center;gap:10px}
  .feature .hd .no{background:rgba(255,255,255,.25);width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px}
  .f1 .hd{background:linear-gradient(135deg,#3b6ef0,#5b8def)}
  .f2 .hd{background:linear-gradient(135deg,#7c3aed,#a78bfa)}
  .f3 .hd{background:linear-gradient(135deg,#0d9488,#2dd4bf)}
  .f4 .hd{background:linear-gradient(135deg,#ea8a1a,#fbbf24)}
  .f5 .hd{background:linear-gradient(135deg,#db2777,#f472b6)}
  .f6 .hd{background:linear-gradient(135deg,#4f46e5,#818cf8)}
  .feature .bd{padding:20px 24px}
  .testcase{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:14px 18px;margin:12px 0;font-size:13.5px}
  .testcase .t-title{font-weight:700;color:#15803d;margin-bottom:8px}
  .testcase ol{margin-left:18px}
  .testcase li{margin:4px 0}
  .meta-row{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0;font-size:12.5px}
  .meta-row .m{background:#f1f3f9;padding:3px 10px;border-radius:6px;color:var(--sub)}
  .meta-row .m b{color:var(--ink)}
  .decision{background:#fefce8;border:1px dashed #facc15;border-radius:8px;padding:10px 14px;margin:10px 0;font-size:13px}
  .decision b{color:#92400e}
  ul.tight{margin:6px 0 6px 20px}ul.tight li{margin:3px 0;font-size:13.5px}
  .small{font-size:12.5px;color:var(--sub)}
  .pill{display:inline-block;font-size:11px;padding:1px 8px;border-radius:5px;background:#eef2ff;color:var(--accent);margin:0 3px}
  hr.sep{border:0;border-top:1px dashed var(--line);margin:18px 0}
  .legend span{display:inline-block;font-size:12px;margin-right:18px}
  .legend .dot{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:middle;margin-right:4px}
</style>
</head>
<body>
<div class="canvas">

<!-- 标题栏 -->
<div class="title-bar">
  <h1>{{项目名}} · 技术选型与流程梳理画布</h1>
  <p>{{范围说明 + 排除项说明}}</p>
</div>

<!-- ① 顶部关键结论指标 -->
<div class="stats">
  <!-- 5个 stat 卡片，num 用 .g/.o/.p/.t 换色 -->
  <div class="stat"><div class="num">{{数字}}</div><div class="lbl">{{标签}}</div></div>
  <div class="stat"><div class="num g">{{数字}}</div><div class="lbl">{{标签}}</div></div>
  <div class="stat"><div class="num p">{{数字}}</div><div class="lbl">{{标签}}</div></div>
  <div class="stat"><div class="num o">{{数字}}</div><div class="lbl">{{标签}}</div></div>
  <div class="stat"><div class="num t">{{数字}}</div><div class="lbl">{{标签}}</div></div>
</div>

<!-- ② 易混淆概念澄清 -->
<div class="card">
  <h2>易混淆概念澄清 <span class="tag">防面试翻车</span></h2>
  <table>
    <tr><th>易混淆对</th><th>区别</th><th>本项目中的角色</th></tr>
    <!-- 每行一对易混淆概念，至少 4-6 对 -->
    <tr><td><b>{{概念A}}</b> vs <b>{{概念B}}</b></td><td>{{区别}}</td><td>{{本项目角色}}</td></tr>
  </table>
  <div class="callout warn"><b>⚠ 最易答错点：</b>{{最易答错的概念辨析}}</div>
</div>

<!-- ③ 推荐端到端流水线 -->
<div class="card">
  <h2>推荐端到端流水线 <span class="tag">阶段表</span></h2>
  <table>
    <tr><th>阶段</th><th>环节</th><th>技术</th><th>关键代码</th><th>产物</th></tr>
    <!-- 按离线准备→请求接入→分流→执行→沉淀 分阶段 -->
    <tr><td rowspan="N"><b>{{阶段名}}</b></td><td>{{环节}}</td><td>{{技术}}</td><td><code>{{文件:行号}}</code></td><td>{{产物}}</td></tr>
  </table>
</div>

<!-- ④ 任务×痛点对照表 -->
<div class="card">
  <h2>任务 × 痛点对照表 <span class="tag">P0/P1 优先级</span></h2>
  <table>
    <tr><th>优先级</th><th>任务</th><th>传统痛点</th><th>技术方案</th><th>落地核心</th></tr>
    <tr><td><span class="pbadge P0">P0</span></td><td>{{任务}}</td><td>{{痛点}}</td><td>{{方案}}</td><td>{{文件}}</td></tr>
  </table>
  <div class="callout ok"><b>优先级原则：</b>{{P0/P1 划分依据}}</div>
</div>

<!-- ⑤ 建议落地顺序 -->
<div class="card">
  <h2>建议落地顺序 <span class="tag">实现时的依赖链</span></h2>
  <table>
    <tr><th>顺序</th><th>阶段</th><th>做什么</th><th>为什么这个顺序</th><th>验证标准</th></tr>
    <tr><td>{{N}}</td><td>{{阶段}}</td><td>{{内容}}</td><td>{{依赖原因}}</td><td>{{如何验证跑通}}</td></tr>
  </table>
</div>

<!-- ⑥ 面试一句话叙事 -->
<div class="card">
  <h2>面试 / 对外一句话叙事 <span class="tag">电梯演讲</span></h2>
  <div class="callout ok" style="font-size:15px;line-height:1.8">
    <b>一句话：</b>{{30字内概括：在什么上叠加什么，用什么技术解决什么痛点}}
  </div>
  <div class="callout" style="font-size:14px">
    <b>展开版（30秒）：</b>{{端到端走一遍主流程，点名关键机制}}
  </div>
</div>

<!-- ⑦ 分卡片详解说明 -->
<div class="card">
  <h2>分卡片详解：{{N}} 个能力的完整落地链路 <span class="tag">含测试用例+代码位置+前置逻辑</span></h2>
  <div class="callout warn"><b>阅读说明：</b>每个功能卡含 ①测试用例(页面操作步骤) ②完整调用链路(按实现顺序,带代码位置) ③前置代码逻辑(链路中需事先完成的部分,含决策点详解) ④面试问答。</div>
</div>

<!-- ========== 功能卡片模板（按实际功能数复制，f1/f2/.../f6 轮换颜色） ========== -->
<div class="feature f1">
  <div class="hd"><span class="no">{{序号}}</span>{{功能名}}</div>
  <div class="bd">
    <div class="meta-row">
      <span class="m"><b>痛点</b>{{痛点}}</span>
      <span class="m"><b>技术</b>{{技术}}</span>
      <span class="m"><b>优先级</b><span class="pbadge P0">P0</span></span>
      <span class="m"><b>核心文件</b>{{文件列表}}</span>
    </div>

    <h3>① 测试用例（页面操作）</h3>
    <div class="testcase">
      <div class="t-title">用例：{{用例名}}</div>
      <ol>
        <li>{{操作步骤1}}</li>
        <li>{{操作步骤2}}</li>
        <li>预期：{{预期结果}}</li>
      </ol>
    </div>

    <h3>② 完整调用链路（按实现顺序）</h3>
    <!-- 分阶段时用 h4 分组 -->
    <ul class="chain">
      <li class="pre">{{文件:行号}} <span class="pos">{{函数}}</span> {{说明}} <span class="tag-pre">前置</span></li>
      <li>{{文件:行号}} <span class="pos">{{函数}}</span> {{说明}}</li>
    </ul>

    <h3>③ 前置代码逻辑（决策点详解）</h3>
    <div class="decision">
      <b>决策点：{{决策问题}}</b><br>
      {{为什么这样选，对比替代方案}}
    </div>
    <h4>前置N：{{子项名}} <code>{{文件:行号}}</code></h4>
    <p class="small">{{实现过程 + 为什么这样做}}</p>

    <h3>④ 面试问答</h3>
    <div class="callout"><b>Q：{{问题}}</b><br>A：{{解答}}</div>
  </div>
</div>

<!-- 全局面试问答汇总 -->
<div class="card">
  <h2>面试官可能问到的任何问题 · 汇总解答 <span class="tag">速查</span></h2>
  <table>
    <tr><th>维度</th><th>问题</th><th>解答要点</th></tr>
    <tr><td>{{维度}}</td><td>{{问题}}</td><td>{{解答}}</td></tr>
  </table>
</div>

<div class="card" style="text-align:center;color:var(--sub);font-size:13px">
  <p>本画布基于 {{项目路径}} 真实代码生成 · 所有代码位置均可在 IDE 中搜索定位</p>
</div>

</div>
</body>
</html>
```

---

## 7 大区块内容规范

| 区块 | 内容要求 |
|------|---------|
| ① 顶部 Stat 指标 | 5 个关键数字卡片（模块数/工具数/表数/层数/步数等），用不同颜色区分 |
| ② 易混淆概念澄清 | 4-6 对易混淆概念表格 + 1 个"最易答错点"warn callout |
| ③ 端到端流水线 | 分阶段表（离线→接入→分流→执行→沉淀），每行含技术/代码/产物 |
| ④ 任务×痛点对照 | 每任务带 P0/P1 徽章，列：痛点→技术→落地核心 + 优先级原则 callout |
| ⑤ 建议落地顺序 | 有依赖关系的实现顺序表，每步含"为什么这个顺序"+"验证标准" |
| ⑥ 面试一句话叙事 | 一句话版 + 30秒展开版，点名关键技术机制 |
| ⑦ 功能分卡片 | 每功能一卡，含 4 要素（测试用例/链路/前置逻辑/面试问答） |

## 功能卡片 4 要素规范（核心，不可省略）

1. **测试用例**：必须是具体页面操作步骤（"进入X页→双击Y→输入Z"），不是抽象描述。用 testcase 绿底块。
2. **完整调用链路**：用 chain 列表，每步 `文件:行号 函数 → 说明`。前置/离线步骤用 `pre` 样式 + "前置"标签。按真实执行顺序。多阶段用 h4 分组。
3. **前置代码逻辑**：用 decision 黄底虚线框写"决策点"（为什么这样选）；前置子项用 h4 + small 段落说明实现过程。**决策点必须答"为什么这样做"而非"做了什么"**。
4. **面试问答**：2-4 个 callout，Q 加粗 A 跟随，覆盖方案选型/区别/参数/失败兜底。

## 质量检查清单（生成后逐项核对）
- [ ] 每个代码位置都是 `文件:行号` 格式，可在 IDE 搜索定位
- [ ] 测试用例是具体页面操作步骤，有"预期"结果
- [ ] 调用链路按实现顺序，前置步骤有橙色"前置"标记
- [ ] 决策点说明了"为什么这样做"+ 对比替代方案，不只是"做了什么"
- [ ] 面试问答覆盖该功能的高频追问
- [ ] 痛点对应真实业务问题，不是凑数
- [ ] 排除项全程未引用
- [ ] 7 大区块齐全，功能卡片 4 要素齐全
- [ ] HTML 可在浏览器正常打开，样式无错乱

## 关键提效技巧
- **第2步必须用 Agent 并行探索**：启动 4-5 个 Explore agent 同时跑，每个负责一条链路，比逐文件读快 5 倍。Agent 返回结论（文件+行号+作用），你再 Read 关键文件精读。
- **行号要精确**：不要写"agent.py 的 run 方法"，要写 `agent.py:147`。用 Read 读文件后按行号引用。
- **前置逻辑是亮点**：用户最常困惑的是"链路中那段代码凭什么能跑"——主动识别并解释前置依赖（如"检索前向量库是提前 chunking 的，chunking 做法是…"）。
- **决策点要对比**：写"为什么用 A 不用 B"，而不是"用了 A"。面试官追问的正是替代方案。
