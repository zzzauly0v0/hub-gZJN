# SKILL_HARNESS.md — 渐进式加载执行 Skill 的 Harness

> 本文档描述新增的 skill harness 子系统。它**不改动**原有四层记忆代码，
> 只在 `src/` 下新增几个模块，复用现有的 `llm_config.py`。

## 一、这是什么

一个演示 **Progressive Disclosure（渐进式加载）** 的 skill 执行框架。
核心思想：**能力（skill）的内容按需、分级进入上下文，越晚加载越省 token。**

这与本项目原有的"四层记忆分层注入"是同一哲学的延伸：
- 记忆系统（`memory_loader.py`）分层注入**关于用户/世界的事实**；
- skill harness 分级加载**关于怎么做一件事的指令**。

## 二、三级加载模型

```
┌────────────────────────────────────────────────────────────┐
│  Level 1  元信息目录（常驻上下文）                            │
│    只有每个 skill 的 name + description                       │
│    极短，用于"路由"：判断这句话要不要用某个能力                │
├────────────────────────────────────────────────────────────┤
│  Level 2  Skill 正文（命中后才加载）                          │
│    路由命中某个 skill 后，才把它的 SKILL.md 正文注入 system   │
├────────────────────────────────────────────────────────────┤
│  Level 3  Skill 资源（执行时才加载）                          │
│    正文里用 `::LOAD 相对路径` 声明的 references/*.md          │
│    真正需要时才读进上下文，平时零占用                          │
└────────────────────────────────────────────────────────────┘
```

## 三、目录约定

```
skills/
  <skill-name>/
    SKILL.md            # frontmatter(name/description) + 正文
    references/*.md     # 可选资源，正文用 ::LOAD 按需引用
```

SKILL.md 结构：

```markdown
---
name: commit-message
description: 一句话说明这个 skill 干什么、什么时候用（路由靠它）
---

# 正文：给 LLM 的操作指令
...
1. `::LOAD references/xxx.md`   # 声明一个 Level 3 资源
```

## 四、新增文件（均为新增，无改动）

| 文件 | 职责 |
|------|------|
| `src/skill_loader.py`  | 扫描 `skills/`，解析 frontmatter，实现三级加载（元信息/正文/资源）|
| `src/skill_router.py`  | 只看 Level 1 元信息做路由：关键词初筛 + 可选 LLM 兜底 |
| `src/skill_harness.py` | 编排器：把三级拼成 system prompt 并调用 LLM，记录逐级加载轨迹 |
| `src/skill_cli.py`     | 彩色 CLI 演示，把三级加载过程逐步打印出来 |
| `skills/commit-message/` | 示例 skill：生成 Conventional Commits 提交信息 |
| `skills/weekly-review/`  | 示例 skill：把零散记录整理成周报 |

## 五、整体流水线

```
用户输入
    │
    ▼  [skill_harness] build_catalog()
Level 1  组装所有 skill 的元信息目录（常驻，很短）
    │
    ▼  [skill_router] route(query)
关键词初筛（零成本）→ 高置信直接命中 / 低置信判无匹配 / 中间地带可选 LLM 兜底
    │  命中 skill？
    ├── 否 → 只带 Level 1 目录，按通用助手回答
    │
    ▼  是
Level 2  registry.load(name) → 注入该 skill 的 SKILL.md 正文
    │
    ▼  [skill_harness] _resolve_resources()
Level 3  解析正文里的 ::LOAD → 读取 references/*.md → 追加进上下文
    │
    ▼  [llm_config] get_chat_client() → 调用 LLM
用户收到回复 + 逐级加载轨迹（哪级加了多少字符）
```

## 六、运行

```bash
# 交互式演示（有 API Key 时可真正执行，无 Key 时 /trace 仍可看加载过程）
python src/skill_cli.py

# 命令：
#   /skills            列出所有 skill 元信息（Level 1）
#   /trace <文本>      只做加载+组装，打印三级加载轨迹，不调 LLM
#   /exit              退出
```

程序化调用：

```python
from src.skill_harness import SkillHarness

harness = SkillHarness()
result = harness.run("帮我写个 commit：修复了检索降级的空指针")
print(result.answer)
for step in result.trace:        # 逐级加载轨迹
    print(f"L{step.level} {step.detail} [{step.char_count} 字符]")
```

## 七、加一个新 skill

1. 新建目录 `skills/<你的skill名>/`；
2. 写 `SKILL.md`：frontmatter 填 `name` / `description`（description 决定路由准不准），正文写操作指令；
3. 需要大段参考资料时，放到 `references/xxx.md`，在正文用 `::LOAD references/xxx.md` 引用；
4. 无需改任何代码——`SkillRegistry` 启动时自动扫描。

## 八、设计取舍

| 决策 | 理由 |
|------|------|
| frontmatter 只做单行 `key: value` 解析 | 避免引入 PyYAML，元信息够用，与项目"零多余依赖"一致 |
| 路由先关键词后 LLM | 对齐 `heartbeat_parser` 的"正则初筛 + LLM 判断"，零成本优先，可解释 |
| 无 API Key 时纯关键词路由 | 保证离线可运行，`/trace` 完整展示加载过程不依赖网络 |
| 资源加载做目录穿越校验 | `::LOAD` 只能读 skill 自己目录内的文件，防止 `../` 逃逸 |
| Harness 复用 `llm_config` | 与全项目统一走 DeepSeek/Qwen 切换，不重复造轮子 |
```
