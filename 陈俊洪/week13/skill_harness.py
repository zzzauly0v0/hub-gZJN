"""
Skill Harness — 渐进式加载执行 skill 的编排器

  1. 把三级加载串成一次完整执行，学生能看到 token 是"逐级"进入上下文的：
       Level 1  组装所有 skill 的元信息目录（很短，常驻）
       Level 2  路由命中后，只把那一个 skill 的正文注入 system prompt
       Level 3  执行前解析正文里的 `::LOAD`，把资源正文追加进上下文
  2. 对齐 memory_loader 的分层思路：同样用 dataclass 记录"每一步加了多少字符"，
     方便前端/CLI 把加载过程可视化。
  3. Harness 不自己调模型的花活，复用 llm_config.get_chat_client()，与全项目一致。

一次执行的上下文构成：
  [system] = 基础指令 + (Level 2) skill 正文 + (Level 3) 已加载资源
  [user]   = 用户输入

使用方式：
  from src.skill_harness import SkillHarness
  harness = SkillHarness()
  result = harness.run("帮我写个 commit：修复了检索降级的空指针")
  print(result.answer)
  for step in result.trace:      # 逐级加载轨迹
      print(step.level, step.detail, step.char_count)
"""

from dataclasses import dataclass, field

from src.skill_loader import SkillRegistry, Skill, LOAD_DIRECTIVE
from src.skill_router import SkillRouter, RouteDecision

BASE_INSTRUCTION = (
    "你是一个具备可插拔技能（skill）的助手。"
    "当某个 skill 被激活时，严格遵循该 skill 的操作指令完成任务。"
)


@dataclass
class LoadStep:
    """一次加载动作的记录，用于可视化渐进式加载过程。"""
    level: int                 # 1 / 2 / 3
    action: str                # "metadata_catalog" | "skill_body" | "resource"
    detail: str                # 人类可读描述
    char_count: int = 0


@dataclass
class HarnessResult:
    query: str
    decision: RouteDecision
    skill: Skill | None
    system_prompt: str
    answer: str | None
    trace: list[LoadStep] = field(default_factory=list)
    error: str | None = None

    @property
    def total_chars(self) -> int:
        return sum(s.char_count for s in self.trace)


class SkillHarness:
    def __init__(self, registry: SkillRegistry | None = None,
                 router: SkillRouter | None = None, use_llm_router: bool = False):
        self.registry = registry or SkillRegistry()
        self.router = router or SkillRouter(self.registry, use_llm=use_llm_router)

    # ── Level 1：元信息目录 ────────────────────────────────────────────────
    def build_catalog(self) -> tuple[str, LoadStep]:
        """把所有 skill 的 name + description 拼成一个简短目录（常驻上下文）。"""
        metas = self.registry.list_metadata()
        lines = [f"- {m.name}: {m.description}" for m in metas]
        catalog = "可用技能目录（仅元信息）：\n" + "\n".join(lines) if lines else "（无已注册 skill）"
        step = LoadStep(1, "metadata_catalog",
                        f"加载 {len(metas)} 个 skill 的元信息", len(catalog))
        return catalog, step

    # ── Level 3：解析并加载正文里声明的资源 ────────────────────────────────
    def _resolve_resources(self, skill: Skill) -> tuple[str, list[LoadStep]]:
        steps: list[LoadStep] = []
        chunks: list[str] = []
        for rel in LOAD_DIRECTIVE.findall(skill.body):
            try:
                content = skill.load_resource(rel)
            except (FileNotFoundError, ValueError) as e:
                steps.append(LoadStep(3, "resource", f"资源加载失败 {rel}：{e}", 0))
                continue
            block = f"### 资源：{rel}\n{content}"
            chunks.append(block)
            steps.append(LoadStep(3, "resource", f"加载资源 {rel}", len(content)))
        return ("\n\n".join(chunks)).strip(), steps

    # ── 组装：把三级拼成最终 system prompt（不调用 LLM，可单测）───────────────
    def assemble(self, query: str) -> HarnessResult:
        trace: list[LoadStep] = []

        # Level 1
        catalog, catalog_step = self.build_catalog()
        trace.append(catalog_step)

        # 路由（只依赖 Level 1）
        decision = self.router.route(query)

        parts = [BASE_INSTRUCTION, catalog]
        skill: Skill | None = None

        if decision.skill_name:
            # Level 2：加载命中 skill 的正文
            skill = self.registry.load(decision.skill_name)
            parts.append(f"## 已激活 Skill：{skill.name}\n{skill.body}")
            trace.append(LoadStep(2, "skill_body",
                                  f"加载 skill「{skill.name}」正文", skill.char_count))

            # Level 3：加载正文声明的资源
            resource_text, res_steps = self._resolve_resources(skill)
            trace.extend(res_steps)
            if resource_text:
                parts.append(f"## Skill 资源（按需加载）\n{resource_text}")

        system_prompt = "\n\n---\n\n".join(parts)
        return HarnessResult(
            query=query, decision=decision, skill=skill,
            system_prompt=system_prompt, answer=None, trace=trace,
        )

    # ── 完整执行：组装上下文 + 调用 LLM ────────────────────────────────────
    def run(self, query: str, temperature: float = 0.5) -> HarnessResult:
        result = self.assemble(query)
        try:
            from src.llm_config import get_chat_client
            client, model = get_chat_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": result.system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=temperature,
            )
            result.answer = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            result.error = f"LLM 调用失败：{e}"
        return result
