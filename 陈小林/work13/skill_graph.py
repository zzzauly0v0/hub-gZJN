"""SQL generation LangGraph workflow – query rewrite → hybrid retrieval → SQL generation → validation."""

import json
import logging
import re
from datetime import datetime
from typing import Annotated, AsyncGenerator

import sqlparse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.database import get_db_context
from app.models.schema_analysis import NLQueryLog, SchemaAnalysis
from app.services.db_connection import get_connection_engine
from app.models.skill import Skill
from app.services.knowledge_base import create_llm, _call_llm_with_retry, _parse_json_response
from app.services.skill_matching import match_skills_by_tags, rank_skills_by_llm, get_step_titles, get_single_step
from app.services.vector_kb import query_hybrid

logger = logging.getLogger(__name__)


# ─── State ────────────────────────────────────────────────────────────────────

class SQLGenerationState(TypedDict):
    """State for the SQL generation LangGraph workflow."""
    analysis_id: int
    user_id: int
    question: str
    rewritten_question: str | None
    intent: str | None
    key_entities: list[str] | None
    context_docs: list[dict] | None
    context_text: str | None
    matched_skill_ids: list[int] | None
    skill_steps: list[dict] | None        # flat list: [{skill_id, skill_name, sort_order, title}]
    pending_tool_call: dict | None          # agent hub sets this: {tool, args}
    tool_context: str                       # accumulated tool results + loaded step content
    generated_sql: str | None
    validation_passed: bool | None
    validation_message: str | None
    error: str | None
    current_step: str
    retry_count: int
    messages: Annotated[list, add_messages]


# ─── Prompts ──────────────────────────────────────────────────────────────────

QUERY_REWRITE_PROMPT = """你是一个查询意图分析专家。分析用户的自然语言问题，输出：
1. rewritten_question: 消除歧义后的完整查询描述（补充时间范围、状态条件等）
2. intent: 查询意图（如：统计查询、明细查询、聚合对比、排行查询）
3. key_entities: 涉及的关键实体（表名、字段名、业务概念）

输出 JSON 格式：
{
  "rewritten_question": "...",
  "intent": "统计查询|明细查询|聚合对比|排行查询|其他",
  "key_entities": ["entity1", "entity2"]
}"""

SQL_GENERATION_PROMPT = """你是一个 SQL 生成专家。根据用户的自然语言需求，结合提供的数据库结构信息（表结构、字段语义、表间关系），生成对应的 SQL 查询语句。

要求：
1. 只生成 SELECT 查询
2. 使用明确的表别名
3. 对于多表查询，使用正确的 JOIN 类型
4. 添加必要的注释说明

直接输出 SQL，不要添加 markdown 代码块标记。"""

SELF_CORRECT_PROMPT = """你是一个 SQL 修正专家。以下 SQL 存在语法问题，请修正后输出正确的 SQL。

错误信息：{error_message}

原始 SQL：
{sql}

请直接输出修正后的 SQL，不要添加 markdown 代码块标记。"""

AGENT_HUB_PROMPT = """你是一个 SQL 生成 Agent。你通过调用工具来收集领域知识，最终辅助 SQL 生成。

## 当前状态

用户问题：{question}
查询意图：{intent}
数据库上下文摘要：{context_summary}

## 可用技能步骤
{available_steps}

## 已收集的信息（工具执行结果）
{loaded_context}

## 可用工具
1. **load_step** - 加载一个技能步骤的完整内容。参数: {{"index": 步骤编号}}
   用于阅读领域规则和指令。你应该基于已加载内容判断哪些步骤有帮助。

2. **run_query** - 在目标数据库上执行 SELECT 查询，返回结果。参数: {{"sql": "SELECT ..."}}
   用于探查数据分布、验证字段值、检查数据量等。只允许 SELECT 语句。

3. **get_table_info** - 获取表的详细结构信息（字段、类型、描述）。参数: {{"table": "表名"}}
   用于查看不在上下文中的表结构，或确认字段详情。

4. **finish** - 已收集足够信息，开始生成 SQL。参数: {{}}

## 决策原则
- 每步都要思考：这个操作对当前用户问题有帮助吗？
- 加载的步骤内容可能提示你需要查询数据或查看其他表
- run_query 的结果可能帮助你决定是否需要加载其他步骤
- 不相关的步骤不需要加载，不必要的查询不需要执行
- 当你认为已有足够信息时，立即 finish

输出 JSON 格式（仅输出 JSON）：
{{"reasoning": "你的推理过程...", "action": "工具名", "args": {{...}}}}
如果 action 是 "finish"，args 为 {{}}。"""


# ─── Node Functions ───────────────────────────────────────────────────────────

def query_rewrite_node(state: SQLGenerationState) -> dict:
    """Node: Rewrite ambiguous natural language question into a precise query intent."""
    logger.info("Node: query_rewrite – analysing query intent...")
    try:
        llm = create_llm()

        # Load skill steps (no skill context at rewrite stage)
        skill_block = ""

        messages = [
            SystemMessage(content=QUERY_REWRITE_PROMPT + skill_block),
            HumanMessage(content=f"用户问题：{state['question']}"),
        ]
        response = _call_llm_with_retry(llm, messages)
        result = _parse_json_response(response.content)

        rewritten = result.get("rewritten_question") or state["question"]
        intent = result.get("intent", "其他")
        key_entities = result.get("key_entities", [])

        return {
            "rewritten_question": rewritten,
            "intent": intent,
            "key_entities": key_entities,
            "current_step": "query_rewrite_done",
            "messages": [AIMessage(content=f"查询重写完成：意图={intent}")],
        }
    except Exception as e:
        logger.warning(f"Query rewrite failed, falling back to original question: {e}")
        return {
            "rewritten_question": state["question"],
            "intent": "其他",
            "key_entities": [],
            "current_step": "query_rewrite_fallback",
            "messages": [AIMessage(content=f"查询重写降级：{e}")],
        }


def retrieve_context_node(state: SQLGenerationState) -> dict:
    """Node: Retrieve relevant schema fragments via hybrid (vector + BM25) search."""
    if state.get("error"):
        return {"current_step": "skipped_due_to_error"}

    logger.info("Node: retrieve_context – fetching schema context...")
    try:
        query = state.get("rewritten_question") or state["question"]
        key_entities = state.get("key_entities") or []
        if key_entities:
            query = f"{query} {' '.join(key_entities)}"

        result = query_hybrid(state["analysis_id"], question=query, top_k=15)
        context_text = _format_retrieved_context(result["results"])

        return {
            "context_docs": result["results"],
            "context_text": context_text,
            "current_step": "retrieve_context_done",
            "messages": [AIMessage(content=f"检索到 {len(result['results'])} 条 Schema 片段")],
        }
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        return {
            "error": f"Schema 上下文检索失败：{e}。请确认向量知识库已构建完成。",
            "current_step": "retrieve_context_error",
            "messages": [AIMessage(content=f"上下文检索失败：{e}")],
        }


def agent_hub_node(state: SQLGenerationState) -> dict:
    """Node: ReAct agent hub – LLM reasons and decides next tool call.

    Supports multiple tools:
      - load_step(index): load a skill step's content
      - run_query(sql): execute SELECT query for data exploration
      - get_table_info(table): get detailed table structure
      - finish: proceed to SQL generation
    """
    skill_steps = state.get("skill_steps") or []
    tool_context = state.get("tool_context") or ""

    # Parse loaded step indices
    loaded_indices: set[int] = set()
    for m in re.finditer(r"\[step-(\d+)\]", tool_context):
        loaded_indices.add(int(m.group(1)))

    available = [(i, s) for i, s in enumerate(skill_steps) if i not in loaded_indices]

    # Build prompt
    steps_text = "\n".join(
        f"  [{i}] {s['skill_name']} — 步骤 {s['sort_order']}: {s['title']}"
        for i, s in available
    ) if available else "（所有步骤已加载）"

    ctx_display = tool_context if tool_context else "（无）"
    question = state.get("rewritten_question") or state["question"]
    intent = state.get("intent") or "未分析"
    context_text = state.get("context_text") or ""
    context_summary = context_text[:300] + "..." if len(context_text) > 300 else (context_text or "无")

    try:
        llm = create_llm()
        prompt = AGENT_HUB_PROMPT.format(
            available_steps=steps_text,
            loaded_context=ctx_display,
            question=question,
            intent=intent,
            context_summary=context_summary,
        )
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="请根据当前状态决定下一步操作"),
        ]
        response = _call_llm_with_retry(llm, messages)
        result = _parse_json_response(response.content)

        action = result.get("action", "finish")
        args = result.get("args", {})
        reasoning = result.get("reasoning", "")

        if action == "load_step":
            idx = args.get("index", -1)
            if isinstance(idx, int) and 0 <= idx < len(skill_steps) and idx not in loaded_indices:
                logger.info(f"Agent hub → load_step [{idx}]: {skill_steps[idx]['title']} | {reasoning}")
                return {
                    "pending_tool_call": {"tool": "load_step", "args": {"index": idx}},
                    "current_step": "agent_hub_tool_call",
                    "messages": [AIMessage(content=f"Agent: 加载步骤 [{idx}] {skill_steps[idx]['title']}\n推理: {reasoning}")],
                }

        elif action == "run_query":
            sql = args.get("sql", "")
            if sql and len(sql) < 5000:
                logger.info(f"Agent hub → run_query: {sql[:100]}... | {reasoning}")
                return {
                    "pending_tool_call": {"tool": "run_query", "args": {"sql": sql}},
                    "current_step": "agent_hub_tool_call",
                    "messages": [AIMessage(content=f"Agent: 执行查询\nSQL: {sql[:200]}\n推理: {reasoning}")],
                }

        elif action == "get_table_info":
            table = args.get("table", "")
            if table:
                logger.info(f"Agent hub → get_table_info: {table} | {reasoning}")
                return {
                    "pending_tool_call": {"tool": "get_table_info", "args": {"table": table}},
                    "current_step": "agent_hub_tool_call",
                    "messages": [AIMessage(content=f"Agent: 查看表结构 [{table}]\n推理: {reasoning}")],
                }

        # Default: finish
        logger.info(f"Agent hub → finish | reasoning: {reasoning}")
        return {
            "pending_tool_call": None,
            "current_step": "agent_hub_finish",
            "messages": [AIMessage(content=f"Agent: 完成知识收集\n推理: {reasoning}")],
        }
    except Exception as e:
        logger.warning(f"Agent hub failed, proceeding to generation: {e}")
        return {"pending_tool_call": None, "current_step": "agent_hub_finish"}


def execute_tool_node(state: SQLGenerationState) -> dict:
    """Node: Execute the pending tool call from agent hub.

    Routes to the appropriate tool implementation:
      - load_step: load skill step content
      - run_query: execute SQL query
      - get_table_info: get table structure

    Results are accumulated into tool_context for the agent to observe.
    """
    tool_call = state.get("pending_tool_call") or {}
    tool_name = tool_call.get("tool", "")
    tool_args = tool_call.get("args", {})
    existing = state.get("tool_context") or ""

    if tool_name == "load_step":
        return _execute_load_step(state, tool_args, existing)
    elif tool_name == "run_query":
        return _execute_run_query(state, tool_args, existing)
    elif tool_name == "get_table_info":
        return _execute_get_table_info(state, tool_args, existing)
    else:
        logger.warning(f"Unknown tool: {tool_name}")
        return {"pending_tool_call": None, "current_step": "tool_unknown"}


def _execute_load_step(state: SQLGenerationState, args: dict, existing: str) -> dict:
    """Load a skill step's full content."""
    idx = args.get("index", 0)
    skill_steps = state.get("skill_steps") or []

    if idx < 0 or idx >= len(skill_steps):
        logger.warning(f"Invalid step index [{idx}]")
        return {"pending_tool_call": None, "current_step": "tool_error"}

    step_info = skill_steps[idx]
    try:
        with get_db_context() as db:
            content = get_single_step(step_info["skill_id"], step_info["sort_order"], db)
        if not content:
            logger.warning(f"Step [{idx}] not found in DB")
            return {"pending_tool_call": None, "current_step": "tool_error"}

        new_block = f"\n\n[step-{idx}]\n{content}"
        logger.info(f"Loaded step [{idx}]: {step_info['title']} ({len(content)} chars)")
        return {
            "tool_context": existing + new_block,
            "pending_tool_call": None,
            "current_step": "tool_done",
            "messages": [AIMessage(content=f"步骤 [{idx}] 已加载: {step_info['title']} ({len(content)} 字符)")],
        }
    except Exception as e:
        logger.warning(f"Failed to load step [{idx}]: {e}")
        return {"pending_tool_call": None, "current_step": "tool_error"}


def _execute_run_query(state: SQLGenerationState, args: dict, existing: str) -> dict:
    """Execute a SELECT query against the target database and return results."""
    sql = args.get("sql", "")
    if not sql.strip().upper().startswith("SELECT"):
        result_text = "错误: 只允许执行 SELECT 查询"
        return {
            "tool_context": existing + f"\n\n[query-result]\n{result_text}",
            "pending_tool_call": None,
            "current_step": "tool_done",
        }

    analysis_id = state.get("analysis_id")
    try:
        with get_db_context() as db:
            analysis = db.query(SchemaAnalysis).filter_by(id=analysis_id).first()
            if not analysis:
                raise ValueError(f"Analysis {analysis_id} not found")
            conn = analysis.connection
            if not conn:
                raise ValueError(f"No database connection for analysis {analysis_id}")

            db_conn = get_connection_engine(conn)
            try:
                cursor = db_conn.cursor()
                # Set read timeout
                if conn.db_type == "mysql":
                    cursor.execute("SET SESSION MAX_EXECUTION_TIME=30000")
                cursor.execute(sql)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchmany(50)  # Limit results
                total = cursor.rowcount if cursor.rowcount >= 0 else len(rows)
            finally:
                db_conn.close()

        # Format results
        lines = [f"查询: {sql}", f"列: {', '.join(columns)}", f"行数: {total}（显示前 {len(rows)} 行）", ""]
        if rows:
            lines.append(" | ".join(columns))
            lines.append("-" * min(100, len(" | ".join(columns))))
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))
        else:
            lines.append("(无结果)")

        result_text = "\n".join(lines)
        logger.info(f"run_query returned {len(rows)} rows for: {sql[:80]}")
        return {
            "tool_context": existing + f"\n\n[query-result]\n{result_text}",
            "pending_tool_call": None,
            "current_step": "tool_done",
            "messages": [AIMessage(content=f"查询执行完成: {len(rows)} 行结果")],
        }
    except Exception as e:
        error_text = f"查询: {sql}\n错误: {e}"
        logger.warning(f"run_query failed: {e}")
        return {
            "tool_context": existing + f"\n\n[query-result]\n{error_text}",
            "pending_tool_call": None,
            "current_step": "tool_done",
            "messages": [AIMessage(content=f"查询执行失败: {e}")],
        }


def _execute_get_table_info(state: SQLGenerationState, args: dict, existing: str) -> dict:
    """Get detailed table structure information from the analysis."""
    table_name = args.get("table", "")
    analysis_id = state.get("analysis_id")

    try:
        with get_db_context() as db:
            from app.models.schema_analysis import TableAnalysis
            analysis = db.query(SchemaAnalysis).filter_by(id=analysis_id).first()
            if not analysis:
                raise ValueError(f"Analysis {analysis_id} not found")

            table = db.query(TableAnalysis).filter_by(
                analysis_id=analysis_id, table_name=table_name
            ).first()

            if not table:
                result_text = f"表 '{table_name}' 在当前分析中不存在"
                return {
                    "tool_context": existing + f"\n\n[table-info]\n{result_text}",
                    "pending_tool_call": None,
                    "current_step": "tool_done",
                }

            lines = [
                f"表名: {table.table_name}",
                f"描述: {table.llm_description or table.table_comment or '无'}",
                "",
                "字段列表:",
            ]
            for field in table.fields:
                key_info = f" [{field.column_key}]" if field.column_key else ""
                desc = f" — {field.llm_description}" if field.llm_description else ""
                lines.append(f"  {field.field_name}: {field.data_type}{key_info}{desc}")

            result_text = "\n".join(lines)
            logger.info(f"get_table_info: {table_name} ({len(table.fields)} fields)")
            return {
                "tool_context": existing + f"\n\n[table-info]\n{result_text}",
                "pending_tool_call": None,
                "current_step": "tool_done",
                "messages": [AIMessage(content=f"表结构已获取: {table_name} ({len(table.fields)} 字段)")],
            }
    except Exception as e:
        error_text = f"获取表 '{table_name}' 信息失败: {e}"
        logger.warning(f"get_table_info failed: {e}")
        return {
            "tool_context": existing + f"\n\n[table-info]\n{error_text}",
            "pending_tool_call": None,
            "current_step": "tool_done",
            "messages": [AIMessage(content=f"获取表信息失败: {e}")],
        }


def match_skills_node(state: SQLGenerationState) -> dict:
    """Node: Match relevant Agent Skills and store their IDs for per-step loading."""
    if state.get("error"):
        return {"current_step": "skipped_due_to_error"}

    logger.info("Node: match_skills – matching agent skills...")
    try:
        key_entities = state.get("key_entities") or []
        question = state.get("rewritten_question") or state["question"]

        if not key_entities:
            logger.info("No key_entities for skill matching, skipping.")
            return {"matched_skill_ids": [], "current_step": "match_skills_skip"}

        # Load all skills from DB
        with get_db_context() as db:
            all_skills = db.query(Skill).all()
            skill_data = [
                Skill(id=s.id, name=s.name, description=s.description,
                      tags=s.tags, content=s.content, created_by=s.created_by)
                for s in all_skills
            ]

        if not skill_data:
            logger.info("No skills defined, skipping match.")
            return {"matched_skill_ids": [], "current_step": "match_skills_empty"}

        # Phase 1: tags pre-filter
        candidates = match_skills_by_tags(key_entities, skill_data)
        if not candidates:
            logger.info("No skill matched by tags, skipping LLM ranking.")
            return {"matched_skill_ids": [], "current_step": "match_skills_no_match"}

        # Phase 2: LLM semantic ranking
        selected_ids = rank_skills_by_llm(question, candidates)
        if not selected_ids:
            logger.info("LLM found no relevant skills.")
            return {"matched_skill_ids": [], "current_step": "match_skills_llm_empty"}

        # Validate IDs against candidates
        valid_ids = {s.id for s in candidates}
        matched_ids = [sid for sid in selected_ids if sid in valid_ids]

        id_to_name = {s.id: s.name for s in candidates}
        logger.info(f"Matched {len(matched_ids)} skill(s): {[id_to_name.get(i) for i in matched_ids]}")

        # Load step titles for progressive loading
        flat_steps: list[dict] = []
        try:
            with get_db_context() as db:
                titles = get_step_titles(matched_ids, db)
                for group in titles:
                    for s in group["steps"]:
                        flat_steps.append({
                            "skill_id": group["skill_id"],
                            "skill_name": group["skill_name"],
                            "sort_order": s["sort_order"],
                            "title": s["title"],
                        })
        except Exception as e:
            logger.warning(f"Failed to load step titles: {e}")

        return {
            "matched_skill_ids": matched_ids,
            "skill_steps": flat_steps,
            "pending_tool_call": None,
            "tool_context": "",
            "current_step": "match_skills_done",
            "messages": [AIMessage(content=f"匹配到 {len(matched_ids)} 个 Skill，共 {len(flat_steps)} 个步骤")],
        }
    except Exception as e:
        logger.warning(f"Skill matching failed, degrading gracefully: {e}")
        return {
            "matched_skill_ids": [],
            "skill_steps": [],
            "pending_tool_call": None,
            "tool_context": "",
            "current_step": "match_skills_error",
        }


def generate_sql_node(state: SQLGenerationState) -> dict:
    """Node: Generate SQL using LLM with retrieved context and step-based skill knowledge."""
    if state.get("error"):
        return {"current_step": "skipped_due_to_error"}

    logger.info("Node: generate_sql – generating SQL...")
    try:
        context_text = state.get("context_text") or ""
        question = state.get("rewritten_question") or state["question"]

        context_block = f"数据库结构上下文：\n{context_text}" if context_text else "（无可用上下文）"

        # Use accumulated tool context from progressive loading
        tool_ctx = state.get("tool_context") or ""
        # Separate skill instructions from tool results
        skill_parts = re.sub(r"\n?\[step-\d+\]\n?", "\n", tool_ctx)
        # Extract tool results (query-result, table-info) separately
        tool_results_parts = []
        for m in re.finditer(r"\[(query-result|table-info)\]\n(.+?)(?=\n\n\[|$)", skill_parts, re.DOTALL):
            tool_results_parts.append(m.group(2).strip())
        # Remove tool results from skill parts
        skill_parts = re.sub(r"\[(query-result|table-info)\]\n.+?(?=\n\n\[|$)", "", skill_parts, flags=re.DOTALL)
        skill_ctx_clean = skill_parts.strip()

        blocks = []
        if skill_ctx_clean:
            blocks.append(f"\n\n领域技能指令：\n{skill_ctx_clean}")
        if tool_results_parts:
            blocks.append(f"\n\n工具执行结果：\n" + "\n\n".join(tool_results_parts))
        extra_blocks = "".join(blocks)

        llm = create_llm()
        messages = [
            SystemMessage(content=SQL_GENERATION_PROMPT),
            HumanMessage(content=f"{context_block}{extra_blocks}\n\n用户需求：{question}"),
        ]

        response = _call_llm_with_retry(llm, messages)
        sql_text = _clean_sql(response.content)

        return {
            "generated_sql": sql_text,
            "current_step": "generate_sql_done",
            "messages": [AIMessage(content="SQL 生成完成")],
        }
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return {
            "error": f"SQL 生成失败：{e}",
            "current_step": "generate_sql_error",
        }


def validate_sql_node(state: SQLGenerationState) -> dict:
    """Node: Validate generated SQL with sqlparse."""
    if state.get("error"):
        return {"current_step": "skipped_due_to_error"}

    logger.info("Node: validate_sql – checking SQL syntax...")
    sql = state.get("generated_sql", "")
    try:
        parsed = sqlparse.parse(sql)
        if not parsed or not parsed[0].tokens:
            return {
                "validation_passed": False,
                "validation_message": "无法解析 SQL，请检查语法",
                "current_step": "validate_sql_fail",
            }
        return {
            "validation_passed": True,
            "validation_message": "SQL 语法检查通过",
            "current_step": "validate_sql_done",
            "messages": [AIMessage(content="SQL 校验通过")],
        }
    except Exception as e:
        return {
            "validation_passed": False,
            "validation_message": str(e),
            "current_step": "validate_sql_error",
        }


def self_correct_sql_node(state: SQLGenerationState) -> dict:
    """Node: Ask LLM to fix SQL based on validation error message."""
    logger.info("Node: self_correct_sql – attempting auto-correction...")
    try:
        sql = state.get("generated_sql", "")
        error_msg = state.get("validation_message", "")

        # Skill context already accumulated via progressive step loading
        skill_block = ""

        llm = create_llm()
        prompt = SELF_CORRECT_PROMPT.format(error_message=error_msg, sql=sql)
        messages = [
            SystemMessage(content=prompt + skill_block),
            HumanMessage(content="请修正上述 SQL"),
        ]

        response = _call_llm_with_retry(llm, messages)
        corrected_sql = _clean_sql(response.content)

        return {
            "generated_sql": corrected_sql,
            "retry_count": state.get("retry_count", 0) + 1,
            "current_step": "self_correct_done",
            "messages": [AIMessage(content="SQL 自修正完成，重新校验...")],
        }
    except Exception as e:
        logger.warning(f"Self-correction failed: {e}")
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "current_step": "self_correct_error",
        }


# ─── Graph Construction ───────────────────────────────────────────────────────

def _validate_router(state: SQLGenerationState) -> str:
    """Conditional edge: route after validate_sql_node."""
    if state.get("validation_passed"):
        return "end"
    if state.get("retry_count", 0) < 1:
        return "self_correct"
    return "end"


graph = StateGraph(SQLGenerationState)
graph.add_node("query_rewrite", query_rewrite_node)
graph.add_node("retrieve_context", retrieve_context_node)
graph.add_node("match_skills", match_skills_node)
graph.add_node("agent_hub", agent_hub_node)
graph.add_node("execute_tool", execute_tool_node)
graph.add_node("generate_sql", generate_sql_node)
graph.add_node("validate_sql", validate_sql_node)
graph.add_node("self_correct_sql", self_correct_sql_node)

graph.set_entry_point("query_rewrite")
graph.add_edge("query_rewrite", "retrieve_context")
graph.add_edge("retrieve_context", "match_skills")

# match_skills → agent_hub (if skills matched) or generate_sql (if not)
graph.add_conditional_edges(
    "match_skills",
    lambda s: "agent_hub" if (s.get("skill_steps") or []) else "generate_sql",
    {"agent_hub": "agent_hub", "generate_sql": "generate_sql"},
)

# agent_hub → execute_tool (any tool call) or generate_sql (finish)
graph.add_conditional_edges(
    "agent_hub",
    lambda s: "execute_tool" if s.get("current_step") == "agent_hub_tool_call" else "generate_sql",
    {"execute_tool": "execute_tool", "generate_sql": "generate_sql"},
)

# execute_tool → agent_hub (loop back to reason with new tool results)
graph.add_edge("execute_tool", "agent_hub")

graph.add_edge("generate_sql", "validate_sql")
graph.add_conditional_edges(
    "validate_sql",
    _validate_router,
    {"self_correct": "self_correct_sql", "end": END},
)
graph.add_edge("self_correct_sql", "validate_sql")

sql_generation_graph = graph.compile()


# ─── Execution Entry Points ───────────────────────────────────────────────────

def run_sql_generation_sync(
    analysis_id: int,
    question: str,
    user_id: int,
    db=None,
) -> dict:
    """Synchronous SQL generation – invoke the full graph and persist the log."""
    initial_state: SQLGenerationState = {
        "analysis_id": analysis_id,
        "user_id": user_id,
        "question": question,
        "rewritten_question": None,
        "intent": None,
        "key_entities": None,
        "context_docs": None,
        "context_text": None,
        "matched_skill_ids": None,
        "skill_steps": None,
        "pending_tool_call": None,
        "tool_context": "",
        "generated_sql": None,
        "validation_passed": None,
        "validation_message": None,
        "error": None,
        "current_step": "init",
        "retry_count": 0,
        "messages": [],
    }

    final_state = sql_generation_graph.invoke(initial_state)

    generated_sql = final_state.get("generated_sql", "")
    error = final_state.get("error")

    # Persist to nl_query_logs
    log_id = None
    created_at = None
    if generated_sql:
        def _save(db_session):
            log = NLQueryLog(
                analysis_id=analysis_id,
                question=question,
                generated_sql=generated_sql,
                created_by=user_id,
            )
            db_session.add(log)
            db_session.flush()
            return log.id, log.created_at

        if db is not None:
            log_id, created_at = _save(db)
        else:
            with get_db_context() as _db:
                log_id, created_at = _save(_db)

    return {
        "generated_sql": generated_sql,
        "log_id": log_id,
        "created_at": created_at,
        "error": error,
    }


async def run_sql_generation_stream(
    analysis_id: int,
    question: str,
    user_id: int,
) -> AsyncGenerator[dict, None]:
    """Streaming SQL generation – manually step through nodes, yielding events."""
    state: SQLGenerationState = {
        "analysis_id": analysis_id,
        "user_id": user_id,
        "question": question,
        "rewritten_question": None,
        "intent": None,
        "key_entities": None,
        "context_docs": None,
        "context_text": None,
        "matched_skill_ids": None,
        "skill_steps": None,
        "pending_tool_call": None,
        "tool_context": "",
        "generated_sql": None,
        "validation_passed": None,
        "validation_message": None,
        "error": None,
        "current_step": "init",
        "retry_count": 0,
        "messages": [],
    }

    # ── 1. Query Rewrite ─────────────────────────────────────────────────
    yield {"type": "node_start", "data": {"step": "query_rewrite", "message": "正在分析查询意图..."}}
    updates = query_rewrite_node(state)
    state = {**state, **{k: v for k, v in updates.items() if k != "messages"}}

    if state.get("error"):
        yield {"type": "error", "data": {"message": state["error"]}}
        yield {"type": "done", "data": {}}
        return

    yield {
        "type": "node_done",
        "data": {
            "step": "query_rewrite",
            "rewritten_question": state.get("rewritten_question"),
            "intent": state.get("intent"),
        },
    }

    # ── 2. Retrieve Context ──────────────────────────────────────────────
    yield {"type": "node_start", "data": {"step": "retrieve_context", "message": "正在检索 Schema 上下文..."}}
    updates = retrieve_context_node(state)
    state = {**state, **{k: v for k, v in updates.items() if k != "messages"}}

    if state.get("error"):
        yield {"type": "error", "data": {"message": state["error"]}}
        yield {"type": "done", "data": {}}
        return

    yield {
        "type": "node_done",
        "data": {
            "step": "retrieve_context",
            "doc_count": len(state.get("context_docs") or []),
        },
    }

    # ── 2b. Match Skills ─────────────────────────────────────────────────
    yield {"type": "node_start", "data": {"step": "match_skills", "message": "正在匹配 Agent Skills..."}}
    updates = match_skills_node(state)
    state = {**state, **{k: v for k, v in updates.items() if k != "messages"}}

    yield {
        "type": "node_done",
        "data": {
            "step": "match_skills",
            "matched_skill_ids": state.get("matched_skill_ids", []),
            "skill_step_count": len(state.get("skill_steps") or []),
        },
    }

    # ── 2c. ReAct Agent Hub ─────────────────────────────────────────
    if True:  # Always enter agent hub (even without skills, agent may want to run queries)
        yield {"type": "node_start", "data": {"step": "agent_hub", "message": "Agent 开始推理..."}}
        max_iterations = max(len(state.get('skill_steps') or []) * 3 + 5, 8)
        iteration = 0
        while iteration < max_iterations:
            iteration += 1

            updates = agent_hub_node(state)
            state = {**state, **{k: v for k, v in updates.items() if k != "messages"}}

            if state.get("current_step") != "agent_hub_tool_call":
                break  # Agent finished or errored

            # Tool call
            tool_call = state.get("pending_tool_call") or {}
            tool_name = tool_call.get("tool", "unknown")
            tool_args = tool_call.get("args", {})
            yield {
                "type": "tool_call",
                "data": {
                    "tool": tool_name,
                    "args": tool_args,
                    "message": f"Agent 调用 {tool_name}({tool_args})",
                },
            }

            updates = execute_tool_node(state)
            state = {**state, **{k: v for k, v in updates.items() if k != "messages"}}

            yield {
                "type": "tool_result",
                "data": {
                    "tool": tool_name,
                    "status": state.get("current_step"),
                },
            }

        yield {
            "type": "node_done",
            "data": {
                "step": "agent_hub",
                "action": "finish",
                "iterations": iteration,
            },
        }

    # ── 3. Generate SQL ──────────────────────────────────────────────────
    yield {"type": "node_start", "data": {"step": "generate_sql", "message": "正在生成 SQL..."}}
    updates = generate_sql_node(state)
    state = {**state, **{k: v for k, v in updates.items() if k != "messages"}}

    if state.get("error"):
        yield {"type": "error", "data": {"message": state["error"]}}
        yield {"type": "done", "data": {}}
        return

    # ── 4. Validate SQL ───────────────────────────────────────────────────
    yield {"type": "node_start", "data": {"step": "validate_sql", "message": "正在校验 SQL..."}}
    updates = validate_sql_node(state)
    state = {**state, **{k: v for k, v in updates.items() if k != "messages"}}

    # ── 4b. Self-correct if needed ───────────────────────────────────────
    if not state.get("validation_passed") and state.get("retry_count", 0) < 1:
        yield {
            "type": "node_done",
            "data": {"step": "validate_sql", "passed": False, "message": state.get("validation_message")},
        }
        yield {"type": "node_start", "data": {"step": "self_correct_sql", "message": "正在修正 SQL..."}}
        updates = self_correct_sql_node(state)
        state = {**state, **{k: v for k, v in updates.items() if k != "messages"}}

        yield {"type": "node_start", "data": {"step": "validate_sql", "message": "重新校验 SQL..."}}
        updates = validate_sql_node(state)
        state = {**state, **{k: v for k, v in updates.items() if k != "messages"}}

    yield {
        "type": "node_done",
        "data": {
            "step": "validate_sql",
            "passed": state.get("validation_passed", False),
            "message": state.get("validation_message"),
        },
    }

    # ── Save log ─────────────────────────────────────────────────────────
    generated_sql = state.get("generated_sql", "")
    log_id = None
    created_at = None
    if generated_sql:
        try:
            with get_db_context() as db:
                log = NLQueryLog(
                    analysis_id=analysis_id,
                    question=question,
                    generated_sql=generated_sql,
                    created_by=user_id,
                )
                db.add(log)
                db.flush()
                log_id = log.id
                created_at = log.created_at
        except Exception as e:
            logger.warning(f"Failed to save NL query log: {e}")

    yield {"type": "sql_generated", "data": {"generated_sql": generated_sql, "log_id": log_id}}
    yield {"type": "done", "data": {}}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_sql(sql_text: str) -> str:
    """Strip markdown code-block wrappers from LLM SQL output."""
    sql_text = sql_text.strip()
    if sql_text.startswith("```"):
        lines = sql_text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        sql_text = "\n".join(lines).strip()
    return sql_text


def _format_retrieved_context(results: list[dict]) -> str:
    """Format hybrid-retrieval results into structured Markdown for the LLM.

    Groups chunks by type (table / field_group / relationship) and limits
    total character count to 12 000.
    """
    if not results:
        return ""

    tables: list[str] = []
    field_groups: list[str] = []
    relationships: list[str] = []

    for r in results:
        meta = r.get("metadata", {})
        chunk_type = meta.get("chunk_type", "")
        content = r.get("content", "")

        if chunk_type == "table":
            tables.append(content)
        elif chunk_type == "field_group":
            field_groups.append(content)
        elif chunk_type == "relationship":
            relationships.append(content)
        else:
            tables.append(content)

    parts: list[str] = []
    if tables:
        parts.append("### 表结构\n")
        parts.extend(tables)
    if field_groups:
        parts.append("\n### 字段分组\n")
        parts.extend(field_groups)
    if relationships:
        parts.append("\n### 关联关系\n")
        parts.extend(relationships)

    text = "\n".join(parts)

    if len(text) > 12000:
        text = text[:12000] + "\n... (上下文已截断)"

    return text
