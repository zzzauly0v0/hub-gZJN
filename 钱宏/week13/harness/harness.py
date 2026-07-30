"""
Harness 引擎 - 协调 Skill 执行的核心

核心流程（ReAct 模式）：
1. 接收用户输入
2. 匹配相关 Skill（渐进式加载）
3. 构建完整上下文（系统提示 + Skill 工具 + 历史对话）
4. 调用 LLM 或规则引擎执行任务
5. 处理工具调用和结果（循环）
6. 返回响应

支持两种执行模式：
- LLM 模式：配置了 llm_func 时，使用大模型进行推理和工具调用
- 规则模式：未配置 LLM 时，使用关键词匹配 + 规则引擎进行工具调用
"""

import json
import re
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

from .skill import Skill, SkillState
from .skill_manager import SkillManager
from .context import ConversationContext

logger = logging.getLogger(__name__)


class HarnessEngine:
    """
    Harness 引擎
    
    设计理念：
    - 渐进式加载：启动时只注册元数据，使用时才加载完整功能
    - 上下文感知：根据对话历史动态调整激活的 Skill
    - 可插拔：支持自定义 LLM 调用函数和工具执行器
    - ReAct 循环：思考(Thought) → 行动(Action) → 观察(Observation) → 最终答案(Answer)
    """
    
    def __init__(
        self,
        skill_manager: SkillManager,
        llm_func: Optional[Callable] = None,
        max_tool_rounds: int = 10,
        auto_load_skills: bool = True,
    ):
        """
        初始化 Harness 引擎
        
        Args:
            skill_manager: Skill 管理器，负责 Skill 的注册、匹配和加载
            llm_func: LLM 调用函数，签名为 (messages, tools) -> response
            max_tool_rounds: 最大工具调用轮数（防止死循环）
            auto_load_skills: 是否自动加载匹配的 Skill
        """
        self.skill_manager = skill_manager
        self.llm_func = llm_func
        self.max_tool_rounds = max_tool_rounds
        self.auto_load_skills = auto_load_skills
        self.context = ConversationContext()  # 对话上下文
        
        # 内置工具执行器：工具名 → 执行函数
        self._tool_executors: Dict[str, Callable] = {}
    
    def set_llm_func(self, func: Callable) -> None:
        """设置 LLM 调用函数"""
        self.llm_func = func
    
    def register_tool_executor(self, tool_name: str, executor: Callable) -> None:
        """
        注册工具执行器
        
        当 Skill 被加载时，会自动将其工具执行器注册到引擎中
        """
        self._tool_executors[tool_name] = executor
    
    def reset_context(self) -> None:
        """重置会话上下文（新对话）"""
        self.context = ConversationContext()
    
    def _build_system_prompt(self) -> str:
        """
        构建系统提示词
        
        组合：基础系统提示 + 已加载 Skill 的提示
        """
        base_prompt = """你是一个智能助手，可以使用已加载的技能（Skills）来完成各种任务。
请根据用户需求，选择合适的技能并调用相应的工具。

工作流程：
1. 理解用户意图
2. 选择合适的工具
3. 调用工具并观察结果
4. 根据结果决定是否继续调用其他工具
5. 给出最终答案"""
        
        # 拼接已加载 Skill 的系统提示
        loaded_skills = self.skill_manager.get_loaded_skills()
        skill_prompts = []
        
        for skill in loaded_skills:
            if skill.system_prompt:
                skill_prompts.append(f"\n--- Skill: {skill.name} ---\n{skill.system_prompt}")
        
        return base_prompt + "\n" + "\n".join(skill_prompts)
    
    def _build_tools_schema(self) -> List[Dict]:
        """
        构建可用工具列表（OpenAI Function Calling 格式）
        
        从所有已加载的 Skill 中收集工具定义
        """
        all_tools = []
        seen_names = set()
        
        for skill in self.skill_manager.get_loaded_skills():
            for tool in skill.tools:
                tool_name = tool.get("function", {}).get("name", "")
                if tool_name and tool_name not in seen_names:
                    all_tools.append(tool)
                    seen_names.add(tool_name)
        
        # 独立注册的工具也加入
        for tool_name, executor in self._tool_executors.items():
            if tool_name not in seen_names:
                all_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": executor.__doc__ or f"执行 {tool_name} 操作",
                        "parameters": {
                            "type": "object",
                            "properties": {"input": {"type": "string"}},
                        },
                    },
                })
                seen_names.add(tool_name)
        
        return all_tools
    
    def _match_and_load_skills(self, query: str) -> List[Skill]:
        """
        匹配并加载相关 Skill（渐进式加载核心）
        
        流程：
        1. 根据用户查询匹配相关 Skill（关键词匹配 + 评分）
        2. 加载匹配的 Skill（加载 executor、tools、prompt）
        3. 将 Skill 的工具执行器注册到引擎
        """
        if not self.auto_load_skills:
            return self.skill_manager.get_loaded_skills()
        
        # Step 1: 匹配相关 Skill
        matched = self.skill_manager.match_skills(query, top_k=3)
        
        # Step 2: 加载匹配的 Skill
        for skill, score in matched:
            if not skill.is_loaded:
                loaded = self.skill_manager.load_skill(skill.name)
                if loaded:
                    self.context.add_skill_context(skill.name, skill.to_dict())
                    
                    # Step 3: 注册工具执行器
                    if loaded.executor and isinstance(loaded.executor, dict):
                        for tool_name, executor in loaded.executor.items():
                            self.register_tool_executor(tool_name, executor)
                    
                    logger.info(f"渐进式加载 Skill: {skill.name} (得分: {score:.2f})")
        
        return self.skill_manager.get_loaded_skills()
    
    def execute_tool(self, tool_name: str, tool_args: Dict) -> Any:
        """
        执行工具调用
        
        优先级：
        1. 直接注册的工具执行器
        2. Skill 内置的 executor 字典
        3. 未找到则返回错误信息
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数（字典）
        
        Returns:
            工具执行结果
        """
        # 1. 检查直接注册的执行器
        if tool_name in self._tool_executors:
            logger.info(f"执行工具: {tool_name}, 参数: {tool_args}")
            return self._tool_executors[tool_name](**tool_args)
        
        # 2. 从已加载的 Skill 中查找
        for skill in self.skill_manager.get_loaded_skills():
            if skill.executor and isinstance(skill.executor, dict):
                if tool_name in skill.executor:
                    logger.info(f"执行工具 (from skill {skill.name}): {tool_name}")
                    return skill.executor[tool_name](**tool_args)
        
        # 3. 未找到执行器
        logger.warning(f"工具 '{tool_name}' 未注册执行器")
        return f"工具 '{tool_name}' 未注册执行器，无法执行"
    
    def _rule_based_execute(self, question: str) -> Generator[Dict, None, None]:
        """
        基于规则的工具调用引擎（无 LLM 时使用）
        
        意图识别优先级（从高到低）：
        1. "列出/所有/清单" → list_flashcards（查看全部）
        2. "查看/详情/信息" + 单词 → show_flashcard（查看详情）
        3. "做/生成/创建" + "闪卡/flashcard" + 单词 → generate_flashcard（生成）
        4. 只有"闪卡" + 单词 → generate_flashcard（默认生成）
        """
        loaded_skills = self.skill_manager.get_loaded_skills()
        skill_names = [s.name for s in loaded_skills]
        
        logger.info(f"规则引擎执行: question='{question}', 已加载技能: {skill_names}")
        question_lower = question.lower()
        
        # ════════════════════════════════════════════════════════════════
        # 意图识别（优先级从高到低）
        # ════════════════════════════════════════════════════════════════
        
        has_list_kw = any(kw in question for kw in ["列出", "所有", "哪些", "全部", "清单", "列表"])
        has_view_kw = any(kw in question for kw in ["查看", "详情", "信息", "看看", "内容", "显示"])
        has_make_kw = any(kw in question for kw in ["做", "生成", "创建", "制作", "建立", "弄", "写"])
        has_card_kw = any(kw in question_lower for kw in ["闪卡", "flashcard", "flash card", "单词卡"])
        
        # ── 意图1: 列出所有闪卡（优先级最高）────────────────────────
        if has_list_kw:
            yield {"type": "thinking", "message": "识别到用户想列出所有闪卡..."}
            
            executor = self._tool_executors.get("list_flashcards")
            if executor:
                result = executor()
                yield {
                    "type": "tool_call",
                    "step": 1,
                    "tool_name": "list_flashcards",
                    "tool_args": {},
                    "tool_result": str(result)[:300],
                }
                yield {
                    "type": "final",
                    "answer": result,
                    "skills_used": skill_names,
                }
            else:
                yield {
                    "type": "final",
                    "answer": "工具 list_flashcards 不可用",
                    "skills_used": skill_names,
                }
            return
        
        # ── 意图2: 查看闪卡详情（有查看关键词）────────────────────────
        if has_view_kw:
            yield {"type": "thinking", "message": "识别到用户想查看闪卡详情..."}
            
            word = self._extract_word(question)
            if not word:
                yield {
                    "type": "final",
                    "answer": "请告诉我要查看哪个单词的闪卡，例如：'查看 crazy 闪卡详情'",
                    "skills_used": skill_names,
                }
                return
            
            executor = self._tool_executors.get("show_flashcard")
            if executor:
                result = executor(word=word)
                yield {
                    "type": "tool_call",
                    "step": 1,
                    "tool_name": "show_flashcard",
                    "tool_args": {"word": word},
                    "tool_result": str(result)[:300],
                }
                yield {
                    "type": "final",
                    "answer": result,
                    "skills_used": skill_names,
                }
            else:
                yield {
                    "type": "final",
                    "answer": f"工具 show_flashcard 不可用，无法查看 '{word}' 的闪卡",
                    "skills_used": skill_names,
                }
            return
        
        # ── 意图3: 生成闪卡（有"做/生成"关键词 + "闪卡"）────────────
        if has_make_kw and has_card_kw:
            yield {"type": "thinking", "message": "识别到用户想生成单词闪卡..."}
            
            word = self._extract_word(question)
            if not word:
                yield {
                    "type": "final",
                    "answer": "我没听清楚您要为哪个单词制作闪卡。请告诉我具体的英语单词，例如：'给我做一张 crazy 的闪卡'。",
                    "skills_used": skill_names,
                }
                return
            
            yield {"type": "thinking", "message": f"为单词 '{word}' 生成闪卡..."}
            
            # 尝试从已有 JSON 数据中读取并生成 HTML
            data_file = Path(self.skill_manager.skills_dir) / "flash-card" / "data" / f"{word}.json"
            if data_file.exists():
                import json as _json
                with open(data_file, "r", encoding="utf-8") as f:
                    word_data = _json.load(f)
                
                executor = self._tool_executors.get("generate_flashcard")
                if executor:
                    result = executor(
                        word=word_data["word"],
                        phonetic=word_data["phonetic"],
                        pos=word_data["pos"],
                        definition=word_data["definition"],
                        examples=word_data["examples"],
                        synonyms=word_data["synonyms"],
                    )
                    yield {
                        "type": "tool_call",
                        "step": 1,
                        "tool_name": "generate_flashcard",
                        "tool_args": {k: v for k, v in word_data.items() if k != "word"},
                        "tool_result": str(result)[:300],
                    }
                    yield {
                        "type": "final",
                        "answer": f"闪卡已为您生成完成！\n\n{result}\n\n💡 提示：在浏览器中打开 HTML 文件即可查看精美的闪卡效果。",
                        "skills_used": skill_names,
                    }
                else:
                    yield {
                        "type": "final",
                        "answer": f"工具 generate_flashcard 不可用",
                        "skills_used": skill_names,
                    }
                return
            else:
                # 没有数据文件
                yield {
                    "type": "final",
                    "answer": (
                        f"抱歉，还没有 '{word}' 的数据。\n\n"
                        f"要生成闪卡，我需要以下信息：\n"
                        f"- 📝 音标（如 /ˈkreɪzi/）\n"
                        f"- 🏷️ 词性（如 adj.）\n"
                        f"- 📖 中文释义\n"
                        f"- 💬 3 条中英例句\n"
                        f"- 🔗 4-6 个近义词\n\n"
                        f"或者您可以告诉我一个已有数据的单词，如 crazy、resilient、thrill。"
                    ),
                    "skills_used": skill_names,
                }
                return
        
        # ── 意图4: 只有"闪卡"关键词（默认生成）────────────────────
        if has_card_kw:
            yield {"type": "thinking", "message": "识别到用户提到闪卡，尝试生成..."}
            
            word = self._extract_word(question)
            if not word:
                yield {
                    "type": "final",
                    "answer": "请告诉我具体要生成哪个单词的闪卡，例如：'crazy'、'resilient'",
                    "skills_used": skill_names,
                }
                return
            
            # 检查数据是否存在
            data_file = Path(self.skill_manager.skills_dir) / "flash-card" / "data" / f"{word}.json"
            if data_file.exists():
                import json as _json
                with open(data_file, "r", encoding="utf-8") as f:
                    word_data = _json.load(f)
                
                executor = self._tool_executors.get("generate_flashcard")
                if executor:
                    result = executor(
                        word=word_data["word"],
                        phonetic=word_data["phonetic"],
                        pos=word_data["pos"],
                        definition=word_data["definition"],
                        examples=word_data["examples"],
                        synonyms=word_data["synonyms"],
                    )
                    yield {
                        "type": "tool_call",
                        "step": 1,
                        "tool_name": "generate_flashcard",
                        "tool_args": {k: v for k, v in word_data.items() if k != "word"},
                        "tool_result": str(result)[:300],
                    }
                    yield {
                        "type": "final",
                        "answer": f"闪卡已为您生成完成！\n\n{result}\n\n💡 提示：在浏览器中打开 HTML 文件即可查看精美的闪卡效果。",
                        "skills_used": skill_names,
                    }
                return
            else:
                yield {
                    "type": "final",
                    "answer": f"抱歉，还没有 '{word}' 的数据。请使用已有单词：crazy、resilient、thrill",
                    "skills_used": skill_names,
                }
                return
        
        # ── 无法识别意图 ──────────────────────────────────────────
        if loaded_skills:
            yield {
                "type": "final",
                "answer": (
                    f"我已加载技能：{', '.join(skill_names)}。\n\n"
                    f"您可以尝试以下操作：\n"
                    f"  📝 生成单词闪卡：'给我做一张 crazy 的闪卡'\n"
                    f"  📋 列出所有闪卡：'列出所有闪卡'\n"
                    f"  🔍 查看闪卡详情：'查看 crazy 闪卡详情'\n"
                    f"  ➕ 新增闪卡：'为新单词生成闪卡，音标/词性/释义是...'"
                ),
                "skills_used": skill_names,
            }
        else:
            yield {
                "type": "final",
                "answer": "抱歉，我还没有加载任何技能。请检查 Skills 配置是否正确。",
                "skills_used": skill_names,
            }
    
    def _extract_word(self, text: str) -> Optional[str]:
        """
        从用户输入中提取目标英语单词
        
        支持的格式：
        - "crazy 的闪卡"
        - "做一张 crazy 闪卡"
        - "给我做 crazy 的 flash card"
        - "查看 crazy 详情"
        """
        # 移除常见的中文前缀/后缀
        patterns = [
            r'([a-zA-Z]+(?:\s[a-zA-Z]+)*)\s*(?:的|词)?\s*(?:闪卡|flashcard|flash card|单词卡|详情|信息)',
            r'(?:给我|帮我)?\s*(?:做|生成|创建|制作)?\s*(?:一张|一个|张|个)?\s*([a-zA-Z]+(?:\s[a-zA-Z]+)*)',
            r'(?:查看|看看|显示)\s*([a-zA-Z]+(?:\s[a-zA-Z]+)*)',
            r'([a-zA-Z]+(?:\s[a-zA-Z]+)*)',  # 兜底：直接提取
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                word = match.group(1).strip()
                # 过滤掉一些常见的非单词词汇
                skip_words = {"flash", "card", "flashcard", "闪卡", "单词", "英语", "英文"}
                if word.lower() not in skip_words and len(word) >= 2:
                    return word.lower()
        
        return None
    
    def run(self, question: str) -> Generator[Dict, None, None]:
        """
        执行一次完整的 Harness 流程（ReAct 模式）
        
        Yields 结构化的步骤数据，便于前端展示：
        - loading_skills: 正在加载技能
        - skills_loaded: 技能加载完成
        - thinking: 正在思考
        - tool_call: 工具调用（含参数和结果）
        - final: 最终答案
        - error: 错误信息
        
        流程：
        1. 先检查是否匹配 Skill 关键词
        2. 如果不匹配 → 普通对话模式（不加载 Skill）
        3. 如果匹配 → 渐进式加载 Skill 并执行
        """
        # ═══════════════════════════════════════════════════════════════════
        # Step 0: 判断是否需要触发 Skill
        # ═══════════════════════════════════════════════════════════════════
        
        # 先用 match_skills 检测是否有关键词匹配
        matched = self.skill_manager.match_skills(question, top_k=1)
        is_skill_triggered = len(matched) > 0 and matched[0][1] > 0
        
        # 如果没有触发任何 Skill，走普通对话
        if not is_skill_triggered:
            yield {"type": "thinking", "message": "识别为普通对话..."}
            yield from self._handle_normal_chat(question)
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 1: 渐进式加载 Skill（仅在触发时）
        # ═══════════════════════════════════════════════════════════════════
        yield {
            "type": "loading_skills",
            "message": "正在匹配相关技能...",
        }
        
        loaded_skills = self._match_and_load_skills(question)
        skill_names = [s.name for s in loaded_skills]
        
        yield {
            "type": "skills_loaded",
            "skills": skill_names,
            "message": f"已加载 {len(loaded_skills)} 个技能: {', '.join(skill_names) if skill_names else '无'}",
        }
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 2: 保存用户消息到上下文
        # ═══════════════════════════════════════════════════════════════════
        self.context.add_user_message(question)
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 3: 判断执行模式
        # ═══════════════════════════════════════════════════════════════════
        
        # 如果有 LLM 函数，使用 LLM 模式
        if self.llm_func and loaded_skills:
            yield {"type": "thinking", "message": "正在调用 LLM 进行推理..."}
            yield from self._llm_execute(question, loaded_skills)
            return
        
        # 否则使用规则引擎
        yield {"type": "thinking", "message": "使用规则引擎分析用户意图..."}
        yield from self._rule_based_execute(question)
    
    def _handle_normal_chat(self, question: str) -> Generator[Dict, None, None]:
        """
        处理普通对话（不触发任何 Skill 时）
        
        模式：
        - 有 LLM：调用 LLM 进行通用对话（不带 tools，支持多轮上下文）
        - 无 LLM：返回提示用户配置 API Key 的回复
        """
        # 先保存用户消息
        self.context.add_user_message(question)
        
        # 如果有 LLM，使用 LLM 进行普通对话
        if self.llm_func:
            yield {"type": "thinking", "message": "正在思考您的问题..."}
            
            # 构建对话消息（带系统提示，包含历史上下文）
            system_prompt = (
                "你是 Harness 智能助手，一个友好且专业的 AI 助手。\n"
                "你可以：\n"
                "- 回答各种通用问题\n"
                "- 进行英语教学和单词学习辅导\n"
                "- 帮助生成英语单词闪卡（当用户说'做闪卡'、'生成闪卡'时）\n\n"
                "请用简洁、准确、自然的方式回答用户问题。"
            )
            
            # 获取历史消息（排除之前注入的 system 提示，避免重复）
            history = self.context.get_messages(include_system=False)
            messages = [{"role": "system", "content": system_prompt}] + history
            
            try:
                # 调用 LLM（不传 tools，纯对话模式）
                response = self.llm_func(messages, [])
                answer = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")
                answer = f"抱歉，对话服务暂时不可用。错误信息：{str(e)}"
            
            self.context.add_assistant_message(answer)
            yield {
                "type": "final",
                "answer": answer,
                "skills_used": [],
                "mode": "normal_chat",
            }
            return
        
        # 无 LLM 时的回复（提示用户配置 API Key）
        has_skills = len(self.skill_manager.list_skills()) > 0
        skills_hint = ""
        if has_skills:
            skill_names = [s.get("name", "") for s in self.skill_manager.list_skills()]
            skills_hint = f"\n\n💡 可用技能：{', '.join(skill_names)}"
        
        answer = (
            f"⚠️ 当前未配置 LLM API Key，无法进行智能对话。\n\n"
            f"📌 配置方法：\n"
            f"  设置环境变量 DEEPSEEK_API_KEY\n"
            f"  例如：$env:DEEPSEEK_API_KEY='your-api-key'"
            f"{skills_hint}\n\n"
            f"� 若要使用技能，请尝试：\n"
            f"  • '做一张 crazy 的闪卡'\n"
            f"  • '列出所有闪卡'\n"
            f"  • '查看 crazy 闪卡详情'"
        )
        
        yield {
            "type": "final",
            "answer": answer,
            "skills_used": [],
            "mode": "normal_chat",
        }
    
    def _llm_execute(self, question: str, loaded_skills: List[Skill]) -> Generator[Dict, None, None]:
        """
        基于 LLM 的工具调用流程
        
        ReAct 循环：
        1. 构建 messages 和 tools
        2. 调用 LLM
        3. 如果 LLM 返回 tool_calls → 执行工具，追加到 messages，继续循环
        4. 如果 LLM 返回 content → 输出最终答案，结束
        """
        skill_names = [s.name for s in loaded_skills]
        system_prompt = self._build_system_prompt()
        tools = self._build_tools_schema()
        
        # 构建消息历史
        messages = self.context.get_messages(include_system=False)
        messages_with_system = [{"role": "system", "content": system_prompt}] + messages
        messages_with_system.append({"role": "user", "content": question})
        
        current_messages = messages_with_system
        
        for step in range(self.max_tool_rounds):
            # ── 调用 LLM ──────────────────────────────────────────────
            try:
                response = self.llm_func(current_messages, tools)
            except Exception as e:
                yield {
                    "type": "error",
                    "message": f"LLM 调用失败: {str(e)}",
                }
                return
            
            # ── 检查工具调用 ──────────────────────────────────────────
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # LLM 返回了工具调用
                assistant_msg = {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in response.tool_calls
                    ],
                }
                current_messages.append(assistant_msg)
                
                # 逐个执行工具调用
                for tc in response.tool_calls:
                    # 解析参数
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        tool_args = {}
                    
                    # 执行工具
                    tool_result = self.execute_tool(tc.function.name, tool_args)
                    
                    # 发送工具调用事件
                    yield {
                        "type": "tool_call",
                        "step": step + 1,
                        "tool_name": tc.function.name,
                        "tool_args": tool_args,
                        "tool_result": str(tool_result)[:300],
                    }
                    
                    # 将工具结果追加到消息列表
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(tool_result),
                    })
            
            elif hasattr(response, 'content') and response.content:
                # LLM 返回了最终答案
                self.context.add_assistant_message(response.content)
                
                yield {
                    "type": "final",
                    "answer": response.content,
                    "skills_used": skill_names,
                    "total_steps": step + 1,
                }
                return
            
            else:
                break
        
        # 超过最大步数
        yield {
            "type": "max_steps_reached",
            "message": f"已达到最大工具调用轮数 {self.max_tool_rounds}",
        }
    
    def get_status(self) -> Dict:
        """获取引擎状态"""
        return {
            "context": self.context.to_dict(),
            "skill_manager": self.skill_manager.get_status(),
            "has_llm_func": self.llm_func is not None,
            "registered_tools": list(self._tool_executors.keys()),
        }
