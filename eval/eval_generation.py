"""Generation evaluation via LLM-as-Judge: faithfulness, answer relevancy, context relevancy, hallucination detection."""

import json
import logging
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("eval.generation")


@dataclass
class GenerationMetrics:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_relevancy: float = 0.0
    hallucination_score: float = 0.0
    faithfulness_reason: str = ""
    answer_relevancy_reason: str = ""
    context_relevancy_reason: str = ""
    hallucination_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


FAITHFULNESS_PROMPT = """你是一个严格的内容审查员。请评估以下"回答"是否忠实于"检索到的知识"或"标准答案"。

评分标准：
- 1.0: 回答完全基于检索到的知识或标准答案，没有添加任何知识库中不存在的信息
- 0.8: 回答基本基于知识，仅有少量合理推断
- 0.5: 回答部分基于知识，部分内容无法在知识中找到依据
- 0.2: 回答大部分与知识无关，自行编造
- 0.0: 回答完全脱离检索知识，纯属编造

检索到的知识：
{context}

标准答案：
{expected}

用户问题：{query}

回答：{answer}

请返回JSON格式：{{"score": 0.0-1.0, "reason": "简短评分理由"}}。只返回JSON。"""

ANSWER_RELEVANCY_PROMPT = """你是一个回答质量评估员。请评估以下回答是否切题、有帮助。

评分标准：
- 1.0: 回答完全切题，直接回答了用户问题，内容有帮助
- 0.8: 回答基本切题，覆盖了主要问题
- 0.5: 回答部分切题，但遗漏了关键信息
- 0.2: 回答偏离主题，答非所问
- 0.0: 回答完全无关

用户问题：{query}

回答：{answer}

请返回JSON格式：{{"score": 0.0-1.0, "reason": "简短评分理由"}}。只返回JSON。"""

CONTEXT_RELEVANCY_PROMPT = """你是一个上下文相关性评估员。请评估检索到的知识与用户问题的相关程度。

评分标准：
- 1.0: 检索结果完全相关，每段都与问题直接相关
- 0.8: 大部分检索结果相关
- 0.5: 部分结果相关，有些是噪音
- 0.2: 少量结果相关，大部分无关
- 0.0: 检索结果完全无关

用户问题：{query}

检索到的知识：
{context}

请返回JSON格式：{{"score": 0.0-1.0, "reason": "简短评分理由"}}。只返回JSON。"""

HALLUCINATION_PROMPT = """你是一个幻觉检测器。请检查"回答"中是否存在幻觉（即编造的、不在"检索到的知识"中的事实性声明）。

检测规则：
- 识别回答中具体的数字、名称、功能描述等事实性声明
- 检查每个事实性声明是否能在检索知识中找到依据
- 忽略礼貌用语、过渡句等非事实性内容

检索到的知识：
{context}

回答：{answer}

请返回JSON格式：
{{"hallucination_score": 0.0-1.0, "hallucinated_claims": ["声明1", "声明2"], "reason": "简短说明"}}
其中 hallucination_score: 0.0=无幻觉, 1.0=严重幻觉。只返回JSON。"""


async def _llm_judge(client, prompt: str) -> dict:
    """Call LLM as judge, return parsed JSON result."""
    if not client:
        return {"score": 0.5, "reason": "LLM not available, default score"}

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.warning(f"LLM judge call failed: {e}")
        return {"score": 0.5, "reason": f"LLM judge error: {e}"}


async def evaluate_faithfulness(client, query: str, context: str, answer: str, expected: str = "") -> tuple[float, str]:
    """Is the response grounded in retrieved knowledge or matches expected answer?"""
    prompt = FAITHFULNESS_PROMPT.format(
        query=query,
        context=context or "（无检索知识）",
        expected=expected or "（无标准答案）",
        answer=answer
    )
    result = await _llm_judge(client, prompt)
    score = float(result.get("score", 0.5))
    reason = result.get("reason", "")
    return score, reason


async def evaluate_answer_relevancy(client, query: str, answer: str) -> tuple[float, str]:
    """Does the response address the user's question?"""
    prompt = ANSWER_RELEVANCY_PROMPT.format(query=query, answer=answer)
    result = await _llm_judge(client, prompt)
    score = float(result.get("score", 0.5))
    reason = result.get("reason", "")
    return score, reason


async def evaluate_context_relevancy(client, query: str, context: str) -> tuple[float, str]:
    """Is the retrieved context relevant to the query?"""
    prompt = CONTEXT_RELEVANCY_PROMPT.format(query=query, context=context or "（无检索知识）")
    result = await _llm_judge(client, prompt)
    score = float(result.get("score", 0.5))
    reason = result.get("reason", "")
    return score, reason


async def detect_hallucination(client, context: str, answer: str) -> tuple[float, str, list[str]]:
    """Detect hallucinated claims in the response."""
    prompt = HALLUCINATION_PROMPT.format(context=context or "（无检索知识）", answer=answer)
    result = await _llm_judge(client, prompt)
    score = float(result.get("hallucination_score", 0.5))
    reason = result.get("reason", "")
    claims = result.get("hallucinated_claims", [])
    return score, reason, claims


async def evaluate_generation(
    client,
    query: str,
    context: str,
    answer: str,
    expected: str = "",
) -> GenerationMetrics:
    """Run all generation evaluation metrics."""
    m = GenerationMetrics()

    m.faithfulness, m.faithfulness_reason = await evaluate_faithfulness(client, query, context, answer, expected)
    m.answer_relevancy, m.answer_relevancy_reason = await evaluate_answer_relevancy(client, query, answer)
    m.context_relevancy, m.context_relevancy_reason = await evaluate_context_relevancy(client, query, context)
    m.hallucination_score, m.hallucination_reason, _ = await detect_hallucination(client, context, answer)

    logger.info(json.dumps(m.to_dict(), ensure_ascii=False))
    return m
