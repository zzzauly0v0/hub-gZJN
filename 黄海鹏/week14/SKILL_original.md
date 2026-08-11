---
name: self-improving-agent
description: "捕获经验教训、错误与纠正，实现持续自我改进。当以下情况时使用：(1) 命令或操作意外失败，(2) 用户纠正 AI（'不对，那是错的…'、'其实…'），(3) 用户请求一个不存在的能力，(4) 外部 API 或工具调用失败，(5) AI 意识到自己的知识已过时或不正确，(6) 发现针对某类重复任务的更好做法。在执行重大任务前也应先回顾已记录的经验。"
metadata:
---

# 自我改进技能（Self-Improving Agent）

将经验教训与错误记录到 Markdown 文件中，用于持续自我改进。编码智能体后续可将这些记录处理成修复方案，重要经验会被升级为项目记忆（如 `CLAUDE.md` / `AGENTS.md`）。

## 首次使用初始化

在记录任何内容之前，确保项目或工作区根目录下已存在 `.learnings/` 目录及其文件。若有缺失则创建：

```bash
mkdir -p .learnings
[ -f .learnings/LEARNINGS.md ] || printf "# Learnings\n\nCorrections, insights, and knowledge gaps captured during development.\n\n**Categories**: correction | insight | knowledge_gap | best_practice\n\n---\n" > .learnings/LEARNINGS.md
[ -f .learnings/ERRORS.md ] || printf "# Errors\n\nCommand failures and integration errors.\n\n---\n" > .learnings/ERRORS.md
[ -f .learnings/FEATURE_REQUESTS.md ] || printf "# Feature Requests\n\nCapabilities requested by the user.\n\n---\n" > .learnings/FEATURE_REQUESTS.md
```

不要覆盖已有文件。如果 `.learnings/` 已初始化，上述命令为空操作。

除非用户明确要求记录到该详细程度，否则不要记录密钥、令牌、私钥、环境变量或完整源码/配置文件。优先使用简短摘要或脱敏片段，而非原始命令输出或完整对话记录。**记录前先脱敏**：对可能含敏感信息的输出，运行 `scripts/redact.sh`，或参考 `references/sanitize.md`。

如果需要自动提醒或初始化辅助，请使用 [Hook 集成](#hook-集成) 中描述的按需启用（opt-in）hook 工作流。

## 速查表

| 场景 | 操作 |
|------|------|
| 命令/操作失败 | 记录到 `.learnings/ERRORS.md` |
| 用户纠正你 | 以 `correction` 分类记录到 `.learnings/LEARNINGS.md` |
| 用户想要缺失的功能 | 记录到 `.learnings/FEATURE_REQUESTS.md` |
| API/外部工具失败 | 带集成细节记录到 `.learnings/ERRORS.md` |
| 知识已过时 | 以 `knowledge_gap` 分类记录到 `.learnings/LEARNINGS.md` |
| 发现更好做法 | 以 `best_practice` 分类记录到 `.learnings/LEARNINGS.md` |
| 与已有条目相似 | 用 `**See Also**` 关联，并考虑提升优先级 |
| 具有广泛适用性的经验 | 升级到 `CLAUDE.md`、`AGENTS.md` 和/或 `.github/copilot-instructions.md` |
| 工作流改进 | 升级到 `AGENTS.md`（OpenClaw 工作区） |
| 工具使用陷阱 | 升级到 `TOOLS.md`（OpenClaw 工作区） |
| 行为模式 | 升级到 `SOUL.md`（OpenClaw 工作区） |

## OpenClaw 配置（推荐）

OpenClaw 是本技能的主要平台。它采用基于工作区的提示注入，并自动加载技能。

### 安装

**通过 ClawdHub（推荐）：**
```bash
clawdhub install self-improving-agent
```

**手动：**
```bash
git clone https://github.com/peterskoett/self-improving-agent.git ~/.openclaw/skills/self-improving-agent
```

### 工作区结构

OpenClaw 会在每次会话中注入以下文件：

```
~/.openclaw/workspace/
├── AGENTS.md          # 多智能体工作流、委派模式
├── SOUL.md            # 行为准则、性格、原则
├── TOOLS.md           # 工具能力、集成陷阱
├── MEMORY.md          # 长期记忆（仅主会话）
├── memory/            # 每日记忆文件
│   └── YYYY-MM-DD.md
└── .learnings/        # 本技能的日志文件
    ├── LEARNINGS.md
    ├── ERRORS.md
    └── FEATURE_REQUESTS.md
```

### 创建学习文件

```bash
mkdir -p ~/.openclaw/workspace/.learnings
```

随后创建日志文件（或从 `assets/` 复制）：
- `LEARNINGS.md` — 纠正、知识缺口、最佳实践
- `ERRORS.md` — 命令失败、异常
- `FEATURE_REQUESTS.md` — 用户请求的能力

### 升级目标

当经验被证明具有广泛适用性时，将其升级到工作区文件：

| 学习类型 | 升级到 | 示例 |
|----------|--------|------|
| 行为模式 | `SOUL.md` | "保持简洁，避免前置声明" |
| 工作流改进 | `AGENTS.md` | "长任务派生子智能体处理" |
| 工具陷阱 | `TOOLS.md` | "git push 需要先配置认证" |

### 跨会话通信

OpenClaw 提供在会话间共享经验的工具：

- **sessions_list** — 查看活动/近期会话
- **sessions_history** — 读取其他会话的记录
- **sessions_send** — 将一条经验发送给其他会话
- **sessions_spawn** — 派生子智能体执行后台任务

仅在可信环境中、且用户明确希望跨会话共享时使用。优先发送简短脱敏摘要及相关文件路径，而非原始记录、密钥或完整命令输出。

### 可选：启用 Hook

用于在会话开始时自动提醒：

```bash
# 将 hook 复制到 OpenClaw 的 hooks 目录
cp -r hooks/openclaw ~/.openclaw/hooks/self-improving-agent

# 启用
openclaw hooks enable self-improving-agent
```

完整细节见 `references/openclaw-integration.md`。

---

## 通用配置（其他智能体）

对于 Claude Code、Codex、Copilot、Cursor、VS Code、WorkBuddy 或其他智能体，在项目或工作区根目录创建 `.learnings/`：

```bash
mkdir -p .learnings
```

使用上方所示的标题内联创建文件。除非明确信任该路径，否则避免从当前仓库或工作区读取模板。

### 在智能体文件中添加引用（作为 hook 提醒的替代方案）

在 `AGENTS.md`、`CLAUDE.md` 或 `.github/copilot-instructions.md` 中添加引用，提醒自己记录经验。

#### 自我改进工作流

当发生错误或纠正时：
1. 记录到 `.learnings/ERRORS.md`、`LEARNINGS.md` 或 `FEATURE_REQUESTS.md`
2. 回顾并将具有广泛适用性的经验升级到：
   - `CLAUDE.md` - 项目事实与约定
   - `AGENTS.md` - 工作流与自动化
   - `.github/copilot-instructions.md` - Copilot 上下文

## 安装位置（各平台路径）

把整个 `self-improving-agent/` 文件夹复制到对应目录即可。**文件夹名必须与 `SKILL.md` 的 `name` 字段一致**（即 `self-improving-agent`），否则部分加载器会出问题。

| 平台 | 用户级（所有项目通用） | 项目级（仅当前项目） | 自动提醒 hook | 备注 |
|------|------------------------|----------------------|----------------|------|
| **Claude Code** | `~/.claude/skills/self-improving-agent/` | `项目/.claude/skills/self-improving-agent/` | ✅ 原生 | 参考实现，hook 最完整 |
| **Codex CLI** | `~/.codex/skills/self-improving-agent/` | `项目/.codex/skills/self-improving-agent/` | ✅ 原生 | hook 用 `.codex/settings.json` |
| **OpenClaw** | `~/.openclaw/skills/self-improving-agent/` | （工作区自动注入） | ✅ 原生 | 主推平台，`clawdhub install` |
| **Cursor** | `~/.cursor/skills/self-improving-agent/` | `项目/.cursor/skills/self-improving-agent/` | ⚠️ 无 | 用 Cursor Rules 兜底（见下） |
| **WorkBuddy** | `~/.workbuddy/skills/self-improving-agent/` | `项目/.workbuddy/skills/self-improving-agent/` | ⚠️ 无 | 技能原生加载，用项目规则兜底 |
| **VS Code + Copilot** | 仓库 `skills/self-improving-agent/` 或 `.github/skills/` | 同上（项目内） | ⚠️ 无 | 配 `.github/copilot-instructions.md` 兜底 |
| **Trae CN** | 不支持标准格式 | `.trae/rules/` | ❌ | 需改写为项目规则（见兜底文档） |
| **Qoder CN** | 不支持标准格式 | `.qoder/skills/`（改写） | ❌ | 需改写为 Qoder Skill/Prompt（见兜底文档） |

> 路径以各平台当前版本为准。带 ⚠️ 的平台虽不支持 Claude 式 hook，但技能被激活后仍可手动/按需记录；**具体兜底做法（可复制的项目规则片段）见 `references/no-hook-fallback.md`**。

## 日志记录格式

完整的三类条目模板（学习 / 错误 / 功能请求）、ID 生成规则、解决条目写法，见 **`references/log-format.md`**。

速记：

- **ID 格式**：`TYPE-YYYYMMDD-XXX`，`TYPE` 取 `LRN` / `ERR` / `FEAT`。
- **最小示例（学习）**：
  ```markdown
  ## [LRN-20260805-001] correction

  **Logged**: 2026-08-05T14:30:00+08:00
  **Priority**: high
  **Status**: pending
  **Area**: config

  ### Summary
  用 pnpm 而非 npm 安装依赖。

  ### Details
  运行 npm install 失败，锁定文件是 pnpm-lock.yaml。

  ### Suggested Action
  改用 `pnpm install`。
  ```
- 记录前务必脱敏（见 `references/sanitize.md`）。

## 升级到项目记忆

当某条经验具有广泛适用性（而非一次性修复）时，将其升级为永久项目记忆。

### 何时升级

- 经验适用于多个文件/功能
- 任何贡献者（人或 AI）都应知晓的知识
- 可防止重复犯错
- 记录了项目特定的约定

### 升级目标

| 目标 | 适合放入的内容 |
|------|----------------|
| `CLAUDE.md` | 项目事实、约定、所有 Claude 交互的陷阱 |
| `AGENTS.md` | 智能体特定的工作流、工具使用模式、自动化规则 |
| `.github/copilot-instructions.md` | 给 GitHub Copilot 的项目上下文与约定 |
| `SOUL.md` | 行为准则、沟通风格、原则（OpenClaw 工作区）|
| `TOOLS.md` | 工具能力、使用模式、集成陷阱（OpenClaw 工作区）|

### 如何升级

1. **提炼**经验为简洁的规则或事实
2. **添加**到目标文件的相应章节（如需要则创建文件）
3. **更新**原始条目：
   - 将 `**Status**: pending` 改为 `**Status**: promoted`
   - 添加 `**Promoted**: CLAUDE.md`、`AGENTS.md` 或 `.github/copilot-instructions.md`

### 升级示例

**学习**（冗长）：
> 项目使用 pnpm workspaces。尝试 `npm install` 失败。锁定文件是 `pnpm-lock.yaml`，必须使用 `pnpm install`。

**写入 CLAUDE.md**（简洁）：
```markdown
## 构建与依赖
- 包管理器：pnpm（非 npm）- 使用 `pnpm install`
```

## 重复模式检测

若记录内容与已有条目相似：

1. **先搜索**：`grep -r "keyword" .learnings/`
2. **关联条目**：在 Metadata 中添加 `**See Also**: ERR-20250110-001`
3. **提升优先级**（若问题反复出现）
4. **考虑系统性修复**：重复出现的问题往往意味着：
   - 文档缺失（→ 升级到 CLAUDE.md 或 .github/copilot-instructions.md）
   - 自动化缺失（→ 添加到 AGENTS.md）
   - 架构问题（→ 创建技术债务工单）

## 简化与加固反馈（Simplify & Harden Feed）

使用此工作流，从 `simplify-and-harden` 技能中汲取重复模式，并将其转化为持久的提示指导。

### 摄取工作流

1. 读取任务摘要中的 `simplify_and_harden.learning_loop.candidates`。
2. 对每个候选，使用 `pattern_key` 作为稳定的去重键。
3. 在 `.learnings/LEARNINGS.md` 中搜索已有该键的条目：
   - `grep -n "Pattern-Key: <pattern_key>" .learnings/LEARNINGS.md`
4. 若找到：
   - 递增 `Recurrence-Count`
   - 更新 `Last-Seen`
   - 添加 `See Also` 关联相关条目/任务
5. 若未找到：
   - 创建新的 `LRN-...` 条目
   - 设置 `Source: simplify-and-harden`
   - 设置 `Pattern-Key`、`Recurrence-Count: 1` 与 `First-Seen`/`Last-Seen`

### 升级规则（系统提示反馈）

当全部满足以下条件时，将重复模式升级到智能体上下文/系统提示文件：

- `Recurrence-Count >= 3`
- 出现在至少 2 个不同任务中
- 在 30 天窗口内发生

升级目标：
- `CLAUDE.md`
- `AGENTS.md`
- `.github/copilot-instructions.md`
- 适用时，OpenClaw 工作区级指导写入 `SOUL.md` / `TOOLS.md`

将升级后的规则写成简短的预防性规则（在编码前/中该做什么），而非冗长的事故记录。

## 周期性回顾

在自然断点处回顾 `.learnings/`：

### 何时回顾

- 开始新的重大任务前
- 完成某个功能后
- 在存在历史经验的领域工作时
- 活跃开发期间每周

### 快速状态检查

```bash
# 统计 pending 项数量
grep -h "Status\*\*: pending" .learnings/*.md | wc -l

# 列出 pending 的高优先级项
grep -B5 "Priority\*\*: high" .learnings/*.md | grep "^## \["

# 查找特定领域的经验
grep -l "Area\*\*: backend" .learnings/*.md
```

### 回顾动作

- 解决已修复的项
- 升级适用的经验
- 关联相关条目
- 上报重复出现的问题

## 检测触发器

当注意到以下情况时自动记录：

**纠正**（→ 以 `correction` 分类记录）：
- "不，那不对…"
- "其实应该是…"
- "你搞错了…"
- "那过时了…"

**功能请求**（→ 功能请求）：
- "你能不能也…"
- "我希望你能…"
- "有没有办法…"
- "你为什么不能…"

**知识缺口**（→ 以 `knowledge_gap` 分类记录）：
- 用户提供了你不知道的信息
- 你引用的文档已过时
- API 行为与你理解的不一致

**错误**（→ 错误条目）：
- 命令返回非零退出码
- 异常或堆栈跟踪
- 非预期的输出或行为
- 超时或连接失败

## 优先级指南

| 优先级 | 使用时机 |
|--------|----------|
| `critical` | 阻塞核心功能、有数据丢失风险、安全问题 |
| `high` | 影响重大、影响常见工作流、重复出现的问题 |
| `medium` | 中等影响，存在变通方案 |
| `low` | 轻微不便、边缘情况、锦上添花 |

## 领域标签

用于按代码库区域过滤经验：

| 区域 | 范围 |
|------|------|
| `frontend` | UI、组件、客户端代码 |
| `backend` | API、服务、服务端代码 |
| `infra` | CI/CD、部署、Docker、云 |
| `tests` | 测试文件、测试工具、覆盖率 |
| `docs` | 文档、注释、README |
| `config` | 配置文件、环境、设置 |

## 最佳实践

1. **立即记录** - 问题发生后上下文最新鲜
2. **具体明确** - 未来的智能体需要快速理解
3. **包含复现步骤** - 尤其针对错误
4. **关联相关文件** - 便于修复
5. **给出具体修复** - 而非仅写"调查一下"
6. **使用一致的分类** - 便于过滤
7. **积极升级** - 如有疑问，就加到 CLAUDE.md 或 .github/copilot-instructions.md
8. **定期回顾** - 陈旧的经验会失去价值
9. **记录前脱敏** - 密钥/令牌/私钥/口令一律先经 `scripts/redact.sh` 处理

## Gitignore 选项

**本地保留经验**（每位开发者私有）：
```gitignore
.learnings/
```

本仓库默认采用此方式，避免意外提交敏感或嘈杂的本地日志。

**仓库内跟踪经验**（团队共享）：
不要加入 .gitignore —— 经验成为共享知识。

**混合**（跟踪模板、忽略条目）：
```gitignore
.learnings/*.md
!.learnings/.gitkeep
```

## Hook 集成

通过智能体 hook 启用自动提醒。这是**按需启用（opt-in）**的 —— 你必须显式配置 hook。

### 快速设置（Claude Code / Codex）

在项目下创建 `.claude/settings.json`：

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "./skills/self-improving-agent/scripts/activator.sh"
      }]
    }]
  }
}
```

这会在每次提示后注入一条经验评估提醒（约 50-100 token 开销）。

### 可用 Hook 脚本

| 脚本 | Hook 类型 | 用途 |
|------|-----------|------|
| `scripts/activator.sh` | UserPromptSubmit | 任务后提醒评估经验 |
| `scripts/error-detector.sh` | PostToolUse (Bash) | 命令出错时触发（已降低误报）|
| `scripts/redact.sh` | 手动/管道 | 记录前对输出脱敏 |

**高级配置（含错误检测的 `PostToolUse` hook）、排错、各平台路径**见 **`references/hooks-setup.md`**。

推荐默认仅用 activator 设置；只有在你愿意让 hook 脚本检查命令输出中的错误模式时，才启用 `PostToolUse`。

## 自动技能抽取

当某条经验足够有价值、可成为可复用技能时，使用提供的辅助脚本抽取它。

### 抽取工作流（速记）

1. 识别候选（满足任一标准：重复出现 / 已验证 resolved / 非显而易见 / 广泛适用 / 用户标记）
2. 运行：
   ```bash
   ./skills/self-improving-agent/scripts/extract-skill.sh skill-name --dry-run
   ./skills/self-improving-agent/scripts/extract-skill.sh skill-name
   ```
3. 用经验内容填充生成的 `SKILL.md` 模板
4. 更新原经验：状态 `promoted_to_skill`，添加 `Skill-Path`

**完整标准、手动抽取、质量门禁、检测触发器**见 **`references/skill-extraction.md`**。

## 多智能体支持

本技能支持不同 AI 编码智能体，采用智能体特定的激活方式。

### Claude Code

**激活**：Hooks（UserPromptSubmit、PostToolUse）
**设置**：`.claude/settings.json` 配置 hook
**检测**：通过 hook 脚本自动

### Codex CLI

**激活**：Hooks（与 Claude Code 相同模式）
**设置**：`.codex/settings.json` 配置 hook
**检测**：通过 hook 脚本自动

### GitHub Copilot

**激活**：手动（无 hook 支持）
**设置**：在 `.github/copilot-instructions.md` 添加：

```markdown
## 自我改进

解决非显而易见的问题后，考虑记录到 `.learnings/`：
1. 使用 self-improving-agent 技能中的格式
2. 用 See Also 关联相关条目
3. 将高价值经验升级为技能

在聊天中询问："要把这个记录为一条经验吗？"
```

**检测**：会话结束时的手动回顾

### VS Code / Cursor / WorkBuddy 等支持 Agent Skills 的平台

这些平台的技能运行时可直接加载本技能包（放入对应 `skills/` 目录即可，路径见上方「安装位置」表）。它们**不支持 Claude 式 hook**，自动提醒需靠"在 `CLAUDE.md` / `AGENTS.md` / 项目规则 / 指令文件中加一段自我改进提示"来兜底——核心的"写 Markdown 日志 + 升级到项目记忆"逻辑任何能写文件的 AI 都能执行。**各平台的可复制兜底片段见 `references/no-hook-fallback.md`**。
