---
name: self-improving-agent
description: "捕获错误、纠正与经验教训，实现持续自我改进。当：命令/操作失败、用户纠正 AI（'不对…'）、用户想要缺失功能、API/工具调用失败、知识过时、发现更好的做法时使用；重大任务前先回顾已记录经验。"
metadata:
---

# 自我改进技能

把错误与经验教训写入 Markdown 日志，并升级为项目记忆（CLAUDE.md / AGENTS.md），让编码智能体越用越聪明。

## 初始化（首次使用）

```bash
mkdir -p .learnings
[ -f .learnings/LEARNINGS.md ] || printf "# Learnings\n\nCategories: correction | insight | knowledge_gap | best_practice\n\n---\n" > .learnings/LEARNINGS.md
[ -f .learnings/ERRORS.md ] || printf "# Errors\n\n---\n" > .learnings/ERRORS.md
[ -f .learnings/FEATURE_REQUESTS.md ] || printf "# Feature Requests\n\n---\n" > .learnings/FEATURE_REQUESTS.md
```

不覆盖已有文件。**记录前先脱敏**：对可能含敏感信息的输出运行 `scripts/redact.sh`（规范见 `references/sanitize.md`）。

## 速查表

| 触发 | 写入 |
|------|------|
| 命令/操作失败、API 出错 | `ERRORS.md` |
| 用户纠正你 | `LEARNINGS.md`（correction）|
| 用户想要缺失功能 | `FEATURE_REQUESTS.md` |
| 知识过时 | `LEARNINGS.md`（knowledge_gap）|
| 发现更好做法 | `LEARNINGS.md`（best_practice）|
| 与已有条目相似 | 加 `**See Also**` 并提升优先级 |
| 经验广泛适用 | 升级到 `CLAUDE.md` / `AGENTS.md` / `.github/copilot-instructions.md` |

## 安装位置（各平台路径）

复制整个 `self-improving-agent/` 到对应目录（**文件夹名须与 `name` 一致**）：

| 平台 | 用户级 | 项目级 | 自动 hook |
|------|--------|--------|-----------|
| Claude Code | `~/.claude/skills/...` | `项目/.claude/skills/...` | ✅ |
| Codex | `~/.codex/skills/...` | `项目/.codex/skills/...` | ✅ |
| OpenClaw | `~/.openclaw/skills/...` | 工作区注入 | ✅ |
| Cursor | `~/.cursor/skills/...` | `项目/.cursor/skills/...` | ⚠️ 兜底 |
| WorkBuddy | `~/.workbuddy/skills/...` | `项目/.workbuddy/skills/...` | ⚠️ 兜底 |
| VS Code+Copilot | 仓库 `skills/` | 同上 | ⚠️ 兜底 |
| Trae / Qoder CN | 不支持标准格式 | 改写 | ❌ |

> ⚠️ 平台无 Claude 式 hook 时，用 `references/no-hook-fallback.md` 的可复制规则片段兜底；OpenClaw 详细配置见 `references/openclaw-integration.md`。

## 日志格式

完整模板与 ID 规则见 `references/log-format.md`。速记：
- **ID**：`TYPE-YYYYMMDD-XXX`，TYPE = `LRN` / `ERR` / `FEAT`。
- **最小示例**：
  ```markdown
  ## [LRN-20260805-001] correction
  **Logged**: 2026-08-05T14:30:00+08:00
  **Priority**: high
  **Status**: pending
  **Area**: config
  ### Summary
  用 pnpm 而非 npm 安装依赖。
  ### Details
  npm install 失败，锁定文件是 pnpm-lock.yaml。
  ### Suggested Action
  改用 `pnpm install`。
  ```
- 记录前务必脱敏（见 `references/sanitize.md`）。

## 升级到项目记忆

当经验广泛适用（多文件/多人应知/可防重复错/项目约定）时，提炼为简短规则写入：

| 目标 | 适合内容 |
|------|----------|
| `CLAUDE.md` | 项目事实、约定、陷阱 |
| `AGENTS.md` | 工作流、工具使用模式、自动化 |
| `.github/copilot-instructions.md` | Copilot 上下文与约定 |
| `SOUL.md` / `TOOLS.md` | OpenClaw 工作区：行为准则 / 工具陷阱 |

原条目改 `**Status**: promoted` 并加 `**Promoted**: <file>`。

## 重复模式检测

1. `grep -r "keyword" .learnings/` 搜相似条目
2. 加 `**See Also**: <ID>` 关联
3. 反复出现 → 提升优先级，考虑系统性修复（补文档 / 自动化 / 建技术债工单）

## 检测触发器（自动记录）

- **纠正** → `correction`：用户说"不对 / 其实应该… / 你搞错了 / 那过时了"
- **功能请求** → `FEATURE_REQUESTS`："你能不能也… / 有没有办法…"
- **知识缺口** → `knowledge_gap`：用户给了你不知道的信息、文档过时
- **错误** → `ERRORS`：非零退出码、异常/堆栈、非预期输出、超时

## 优先级

`critical`（阻塞/数据丢失/安全）> `high`（重大影响/常见/重复）> `medium`（有变通）> `low`（边缘）。

## 领域标签

`frontend` / `backend` / `infra` / `tests` / `docs` / `config`。

## 最佳实践

1. 立即记录，上下文最鲜　2. 具体 + 复现步骤　3. 关联文件与具体修复　4. 一致分类　5. 积极升级　6. 定期回顾　7. 记录前脱敏

## 周期性回顾

自然断点（新任务前 / 功能完成 / 每周）回顾 `.learnings/`：解决已修复项、升级适用经验、关联重复项。

```bash
grep -h "Status\*\*: pending" .learnings/*.md | wc -l                      # pending 数
grep -B5 "Priority\*\*: high" .learnings/*.md | grep "^## \["            # 高优先级
```

## Gitignore

本地私有（默认）：忽略 `.learnings/`。团队共享：不忽略。混合：忽略 `*.md` 保留 `.gitkeep`。

## Hook 集成（opt-in）

| 脚本 | Hook | 用途 |
|------|------|------|
| `scripts/activator.sh` | UserPromptSubmit | 任务后提醒评估经验（注入极短提示）|
| `scripts/error-detector.sh` | PostToolUse(Bash) | 命令出错时触发（已降误报）|
| `scripts/redact.sh` | 手动/管道 | 记录前脱敏 |

Claude Code / Codex：在 `.claude/settings.json`（或 `.codex/`）配置 hook；详细 JSON 与排错见 `references/hooks-setup.md`。推荐默认仅启用 activator；PostToolUse 仅在愿检查命令输出时启用。

## 自动技能抽取

高价值且可复用经验用 `scripts/extract-skill.sh skill-name [--dry-run]` 抽取为新技能骨架，原条目标 `promoted_to_skill` 并加 `Skill-Path`。完整标准见 `references/skill-extraction.md`。

## 多智能体支持

- **Claude Code / Codex**：Hook 自动检测。
- **OpenClaw**：工作区注入 + 跨会话工具；详情见 `references/openclaw-integration.md`；`simplify-and-harden` 摄取流程见 `references/simplify-harden.md`。
- **Copilot / Cursor / WorkBuddy / 其他**：技能被激活后按本文件记录；自动提醒用项目规则/指令兜底，片段见 `references/no-hook-fallback.md`。
