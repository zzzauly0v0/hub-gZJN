
"""
generate_and_optimize_skill.py

演示如何让大模型生成一个客服 Skill，再优化其内容以减少 Token 消耗，
并对比优化前后的效果。
"""

import os
import json
import re
from pathlib import Path
from openai import OpenAI
from evaluator import Evaluator  # 使用项目自带的评估器

# ---------- 系统提示模板 ----------
SYSTEM_TEMPLATE = """你是云购商城的智能客服助手。

你的所有知识来源于以下技能文档，严格基于文档内容回答，不要自行推断或编造政策。

## 回答规则（严格遵守）
- 【能回答】如果技能文档覆盖了用户问题：直接给出完整具体的答案（含具体天数/金额/
  工作日数等政策细节）。**不要在答案中加"建议联系人工客服"之类的推脱话**。
- 【不能回答】如果技能文档确实不覆盖：**仅回答一句** "需要联系人工客服"，
  不要编造答案，也不要列举可能的情况。

{skills_section}
"""

SKILLS_SECTION_TEMPLATE = """## 当前知识库（共{count}个技能）

{skills_content}
"""

# ---------- 调用 LLM 的通用函数 ----------
def call_llm(client, system_prompt, user_prompt, model="deepseek-chat", temperature=0):
    """调用 LLM 并返回响应文本和 usage 信息"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=3000,
    )
    content = response.choices[0].message.content.strip()
    usage = response.usage
    return content, usage

# ---------- 构建系统提示 ----------
def build_system_prompt(skill_name, skill_content):
    """根据一个 Skill 构建完整的 system prompt"""
    skills_section = SKILLS_SECTION_TEMPLATE.format(
        count=1,
        skills_content=f"### 技能：{skill_name}\n{skill_content}"
    )
    return SYSTEM_TEMPLATE.format(skills_section=skills_section)

# ---------- 主流程 ----------
def main():
    # 初始化客户端
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：请设置环境变量 DEEPSEEK_API_KEY")
        return
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # ---------- 第一步：生成初始 Skill ----------
    print("=" * 60)
    print("Step 1: 让大模型生成一个关于“退货政策”的 Skill")
    print("=" * 60)

    gen_system = "你是一位资深的客服知识库编写专家。请根据云购商城的退货政策，编写一个完整的 SKILL.md 文件。"
    gen_user = """请严格按照以下格式编写，包含 frontmatter 和正文：

---
name: return_policy
description: 云购商城退货政策，涵盖退货条件、时限、运费承担等
version: 1
---

# 退货政策

（在这里编写具体政策内容，要详细、准确，包括：
- 退货时限（如 7 天无理由）
- 退货条件（商品完好、包装齐全等）
- 运费承担规则
- 退款方式（原路返回）
- 特殊商品例外（如电子书、虚拟商品等）
）

请确保内容完整，可直接用于客服回答。"""

    initial_skill_raw, usage_gen = call_llm(client, gen_system, gen_user)
    # 提取 SKILL.md 内容（可能前后有文字，但通常大模型会直接输出 markdown）
    # 用正则提取 ```...``` 或者直接取全文
    # 简单起见，如果包含 ``` 则提取代码块，否则全部作为内容
    if "```" in initial_skill_raw:
        # 取第一个代码块
        match = re.search(r"```(?:md)?\s*(.*?)```", initial_skill_raw, re.DOTALL)
        if match:
            initial_skill = match.group(1).strip()
        else:
            initial_skill = initial_skill_raw
    else:
        initial_skill = initial_skill_raw

    print("生成初始 Skill 成功。")
    print(f"初始 Skill 长度（字符数）: {len(initial_skill)}")
    print(f"生成消耗输入 Token: {usage_gen.prompt_tokens}, 输出 Token: {usage_gen.completion_tokens}\n")

    # ---------- 第二步：优化 Skill ----------
    print("=" * 60)
    print("Step 2: 要求大模型优化该 Skill，减少 Token 消耗")
    print("=" * 60)

    opt_system = "你是一位内容精简专家，擅长压缩文本长度而不损失信息。"
    opt_user = f"""请优化下面的 SKILL.md，目标是**大幅减少 Token 数量**（即缩短文本长度），同时**保持所有政策信息完整、清晰**。

优化策略：
- 合并重复或类似的条款
- 使用更简洁的表述（例如“7天内”代替“自签收之日起7个自然日内”）
- 删除冗余的修饰词和例子
- 保持 frontmatter 不变（name, description, version 可以不改，但可以适当精简 description）

请输出优化后的完整 SKILL.md（含 frontmatter）。

原始 SKILL.md：
{initial_skill}
"""

    optimized_skill_raw, usage_opt = call_llm(client, opt_system, opt_user)
    # 同样提取内容
    if "```" in optimized_skill_raw:
        match = re.search(r"```(?:md)?\s*(.*?)```", optimized_skill_raw, re.DOTALL)
        if match:
            optimized_skill = match.group(1).strip()
        else:
            optimized_skill = optimized_skill_raw
    else:
        optimized_skill = optimized_skill_raw

    print("优化 Skill 完成。")
    print(f"优化后 Skill 长度（字符数）: {len(optimized_skill)}")
    print(f"优化消耗输入 Token: {usage_opt.prompt_tokens}, 输出 Token: {usage_opt.completion_tokens}\n")

    # 保存两个版本到文件以便查看
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "initial_skill.md").write_text(initial_skill, encoding="utf-8")
    (out_dir / "optimized_skill.md").write_text(optimized_skill, encoding="utf-8")
    print("两个版本的 Skill 已保存到 outputs/ 目录。\n")

    # ---------- 第三步：加载评估集，选取退货相关的问题 ----------
    print("=" * 60)
    print("Step 3: 加载评估数据集，选取退款/退货相关的问题")
    print("=" * 60)

    eval_path = Path("data/eval_set.json")
    if not eval_path.exists():
        print("错误：找不到 data/eval_set.json，请确保在项目根目录运行。")
        return
    evaluator = Evaluator(str(eval_path))

    # 选取所有类别包含 "refund" 的问题（即退货退款相关）
    refund_questions = [q for q in evaluator.questions.values() 
                        if "refund" in q.get("category", "").lower()]
    print(f"找到 {len(refund_questions)} 个退货相关的问题。\n")

    if not refund_questions:
        print("警告：没有找到退货相关问题，请检查 eval_set.json 的类别标签。")
        return

    # ---------- 第四步：用两个 Skill 分别回答问题，并统计结果 ----------
    def evaluate_skill(skill_content, skill_name="return_policy"):
        """使用给定的 Skill 回答所有退货问题，返回准确率和 Token 消耗"""
        system_prompt = build_system_prompt(skill_name, skill_content)
        total_input_tokens = 0
        total_output_tokens = 0
        correct = 0
        details = []

        for q in refund_questions:
            # 调用 LLM 获取回答
            answer, usage = call_llm(
                client, 
                system_prompt, 
                q["question"],
                temperature=0,
                max_tokens=400
            )
            total_input_tokens += usage.prompt_tokens
            total_output_tokens += usage.completion_tokens

            # 评估回答是否正确
            ok, reason = evaluator.evaluate_answer(answer, q["id"])
            if ok:
                correct += 1
            details.append({
                "id": q["id"],
                "question": q["question"],
                "answer": answer,
                "correct": ok,
                "reason": reason
            })
        return {
            "total": len(refund_questions),
            "correct": correct,
            "accuracy": correct / len(refund_questions),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "details": details
        }

    print("评估初始 Skill……")
    result_initial = evaluate_skill(initial_skill)
    print("评估优化后 Skill……")
    result_optimized = evaluate_skill(optimized_skill)

    # ---------- 第五步：输出对比报告 ----------
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)

    print(f"{'指标':<20} {'初始 Skill':<20} {'优化后 Skill':<20}")
    print("-" * 60)
    print(f"{'总问题数':<20} {result_initial['total']:<20} {result_optimized['total']:<20}")
    print(f"{'正确数':<20} {result_initial['correct']:<20} {result_optimized['correct']:<20}")
    print(f"{'准确率':<20} {result_initial['accuracy']:.1%} {'':<14} {result_optimized['accuracy']:.1%}")
    print(f"{'输入 Token 总数':<20} {result_initial['input_tokens']:<20} {result_optimized['input_tokens']:<20}")
    print(f"{'输出 Token 总数':<20} {result_initial['output_tokens']:<20} {result_optimized['output_tokens']:<20}")
    print(f"{'总 Token 数':<20} {result_initial['input_tokens'] + result_initial['output_tokens']:<20} {result_optimized['input_tokens'] + result_optimized['output_tokens']:<20}")
    print(f"{'Token 节省 (输入)':<20} {'':<20} {result_initial['input_tokens'] - result_optimized['input_tokens']}")
    print(f"{'Token 节省 (输出)':<20} {'':<20} {result_initial['output_tokens'] - result_optimized['output_tokens']}")
    print(f"{'总节省':<20} {'':<20} {(result_initial['input_tokens'] + result_initial['output_tokens']) - (result_optimized['input_tokens'] + result_optimized['output_tokens'])}")

    # 可选：打印每个问题的回答差异（如果准确率下降，可以查看原因）
    if result_initial['accuracy'] != result_optimized['accuracy']:
        print("\n注意：准确率有变化，以下是回答差异详情（仅显示判定不一致的问题）：")
        for init, opt in zip(result_initial['details'], result_optimized['details']):
            if init['correct'] != opt['correct']:
                print(f"\n问题ID {init['id']}: {init['question']}")
                print(f"  初始回答: {init['answer'][:100]}... (正确={init['correct']})")
                print(f"  优化回答: {opt['answer'][:100]}... (正确={opt['correct']})")

    # 保存详细对比结果到 JSON
    report = {
        "initial_skill": {
            "content": initial_skill,
            "length_chars": len(initial_skill),
            "evaluation": result_initial
        },
        "optimized_skill": {
            "content": optimized_skill,
            "length_chars": len(optimized_skill),
            "evaluation": result_optimized
        },
        "comparison": {
            "accuracy_diff": result_optimized['accuracy'] - result_initial['accuracy'],
            "input_token_saved": result_initial['input_tokens'] - result_optimized['input_tokens'],
            "output_token_saved": result_initial['output_tokens'] - result_optimized['output_tokens'],
            "total_token_saved": (result_initial['input_tokens'] + result_initial['output_tokens']) - (result_optimized['input_tokens'] + result_optimized['output_tokens'])
        }
    }
    report_path = out_dir / "skill_optimization_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细报告已保存到 {report_path}")

if __name__ == "__main__":
    main()
