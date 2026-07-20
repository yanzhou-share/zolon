"""End-to-end evaluation: answer correctness, response quality, user satisfaction prediction."""

import json
import logging
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("eval.e2e")


@dataclass
class E2EMetrics:
    correctness: float = 0.0
    correctness_reason: str = ""
    response_quality: float = 0.0
    response_quality_reason: str = ""
    satisfaction_prediction: float = 0.0
    satisfaction_reason: str = ""
    overall_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


CORRECTNESS_PROMPT = """你是一个答案正确性评估员。请评估"回答"与"标准答案"的一致程度。

评分标准：
- 1.0: 回答完全正确，与标准答案一致
- 0.8: 回答基本正确，包含标准答案的核心信息
- 0.5: 回答部分正确，但有遗漏或轻微错误
- 0.2: 回答有严重错误或偏差
- 0.0: 回答完全错误

标准答案：{ground_truth}

回答：{answer}

请返回JSON格式：{{"score": 0.0-1.0, "reason": "简短评分理由"}}。只返回JSON。"""

QUALITY_PROMPT = """你是一个回答质量评估专家。请综合评估以下销售助手回答的质量。

评估维度：
1. 准确性：信息是否正确
2. 完整性：是否覆盖了问题的所有方面
3. 专业性：是否体现了专业销售顾问的素养
4. 清晰度：表达是否清晰易懂
5. 礼貌性：语气是否礼貌友好

用户问题：{query}

回答：{answer}

请返回JSON格式：{{"score": 0.0-1.0, "reason": "简短评分理由"}}。只返回JSON。"""

SATISFACTION_PROMPT = """你是一个用户满意度预测器。请基于对话内容预测用户对此回答的满意程度。

评估因素：
- 回答是否解决了用户的问题
- 回答是否及时、有用
- 回答语气是否友好专业
- 回答长度是否合适（不过长也不过短）

用户问题：{query}

回答：{answer}

请返回JSON格式：{{"score": 0.0-1.0, "reason": "简短评分理由"}}。只返回JSON。"""


async def _llm_judge(client, prompt: str) -> dict:
    """Call LLM as judge."""
    if not client:
        return {"score": 0.5, "reason": "LLM not available"}
    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.warning(f"E2E LLM judge failed: {e}")
        return {"score": 0.5, "reason": f"Error: {e}"}


async def evaluate_correctness(client, query: str, answer: str, ground_truth: str) -> tuple[float, str]:
    """Evaluate answer correctness against ground truth."""
    prompt = CORRECTNESS_PROMPT.format(ground_truth=ground_truth, answer=answer)
    result = await _llm_judge(client, prompt)
    return float(result.get("score", 0.5)), result.get("reason", "")


async def evaluate_response_quality(client, query: str, answer: str) -> tuple[float, str]:
    """Evaluate overall response quality."""
    prompt = QUALITY_PROMPT.format(query=query, answer=answer)
    result = await _llm_judge(client, prompt)
    return float(result.get("score", 0.5)), result.get("reason", "")


async def evaluate_satisfaction(client, query: str, answer: str) -> tuple[float, str]:
    """Predict user satisfaction."""
    prompt = SATISFACTION_PROMPT.format(query=query, answer=answer)
    result = await _llm_judge(client, prompt)
    return float(result.get("score", 0.5)), result.get("reason", "")


async def evaluate_e2e(
    client,
    query: str,
    answer: str,
    ground_truth: Optional[str] = None,
) -> E2EMetrics:
    """Run end-to-end evaluation."""
    m = E2EMetrics()

    m.response_quality, m.response_quality_reason = await evaluate_response_quality(client, query, answer)
    m.satisfaction_prediction, m.satisfaction_reason = await evaluate_satisfaction(client, query, answer)

    if ground_truth:
        m.correctness, m.correctness_reason = await evaluate_correctness(client, query, answer, ground_truth)

    scores = [m.response_quality, m.satisfaction_prediction]
    if ground_truth:
        scores.append(m.correctness)
    m.overall_score = round(sum(scores) / len(scores), 4)

    logger.info(json.dumps(m.to_dict(), ensure_ascii=False))
    return m
