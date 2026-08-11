# self-improving-agent：效率与 Token 消耗对比（原版 vs 优化版）

> 生成时间：2026-08-05
> 优化目标：降低技能加载时的上下文 token 消耗，并提升 hook 脚本执行效率。
> 测量口径：token 用 `tiktoken` 的 `cl100k_base` 编码估算；运行时取 200 行命令样本、各 30 次平均。

## 一、总览

| 指标 | 原版 | 优化版 | 变化 |
|------|------|--------|------|
| **SKILL.md 行数** | 479 | 139 | **-71%** |
| **SKILL.md 体积** | 18,171 B | 6,264 B | -65% |
| **SKILL.md token（加载成本）** | 6,373 | 2,270 | **-64.4%** |
| references 文件数 | 7 | 8 | +1（外移所致）|
| references 总 token | 10,085 | 10,524 | +4%（均为按需懒加载，不计入技能加载成本）|
| activator 每轮注入 token | 92 | 87 | -5.4% |
| activator 每轮注入字节 | 453 | 259 | -43% |
| error-detector 单次耗时 | 162.6 ms | 32.0 ms | **-80.3%（提速 5×）** |
| redact 单次耗时 | — | 22.2 ms | 微秒级 |

## 二、关键结论

### 1. 上下文 token 消耗（最重要）
- **技能每次被加载进上下文时，主文件从 6,373 token 降到 2,270 token**，每次触发少占约 **4,100 token**。
- references 略增 (+439 token) 是因为把 OpenClaw 专属细节与 `simplify-and-harden` 小众流程**外移**到 `references/`，但 references 是**懒加载**（AI 用不到就不读），所以**不影响加载成本**。
- 净效果：技能"激活成本"下降约 2/3。

### 2. 执行效率
- `error-detector.sh` 从"每次命令 20 次管道 fork"改为"**单次 `grep -qF -e` 多模式匹配**"，耗时 **163ms → 32ms**，提速 80%，且**误报率不变（11/11 用例仍全过）**。
- `activator.sh` 提醒文本压缩，注入字节 -43%。

### 3. 故意失败验证（端到端）
- 在新版 skill 下执行 `python3 -c "raise FileNotFoundError('demo: config/build.conf not found ...')"`，`error-detector.sh` 成功捕获，并按格式写入 `.learnings/ERRORS.md` 新条目 `ERR-20260805-002`，"失败 → 捕获 → 记录"闭环在新版下依旧有效。
- 原版（优化前）此前已验证过 `ERR-20260805-001`（RuntimeError 场景）。

## 三、本次具体改动

| 文件 | 改了什么 | 收益 |
|------|----------|------|
| `SKILL.md` | 删去 OpenClaw 专属大段、合并重复的"升级目标"表、把 `Simplify & Harden Feed` 整段外移、压缩示例与说明 | 主文件 -64% token |
| `references/simplify-harden.md` | **新增**：承接从 SKILL.md 外移的小众摄取流程 | 主文件瘦身、细节可查 |
| `scripts/activator.sh` | 提醒文本大幅压缩为中文短句 | 注入字节 -43% |
| `scripts/error-detector.sh` | 20 次管道 → 单次 `grep -qF -e` 多模式；保留全部误报防护 | 耗时 -80%，准确率不变 |

## 四、交付物

- **原版（优化前）**：`self-improving-agent-original/`（SKILL.md 479 行 / 6,373 token）
- **优化版（新版）**：`self-improving-agent/`（SKILL.md 139 行 / 2,270 token）
- **演示记录**：`week14SKILL/.learnings/ERRORS.md`（含 `ERR-20260805-001` 原版验证、`ERR-20260805-002` 新版验证）
- **原始测量数据**：`_compare_raw.json`

## 五、使用建议

- 若你主要用 WorkBuddy / Cursor / VS Code+Copilot（无 Claude 式 hook），**activator/error-detector 不会自动触发**，但技能激活后 AI 仍会按 SKILL.md 记录；此时本次优化的 token 收益（加载即省 ~4,100 token/次）直接生效。
- 仅在 Claude Code / OpenClaw 启 hook 时，activator 的每轮 -5 token 才累积体现。
