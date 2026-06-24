"""
AI Agent 节点模块

LangGraph 6 节点：input_collector, sentiment_analyzer, query_optimizer,
knowledge_retriever, response_generator, handover_response。
"""

import json
import logging
from typing import Optional

from config import llm as llm_cfg, sentiment as sent_cfg
from models import AgentState

logger = logging.getLogger(__name__)

# LLM 客户端（延迟初始化）
llm_client = None

def init_llm():
    global llm_client
    if llm_cfg.API_KEY:
        from openai import AsyncOpenAI
        llm_client = AsyncOpenAI(api_key=llm_cfg.API_KEY, base_url=llm_cfg.BASE_URL)

# Embedding 模型（延迟加载）
embed_model = None

def _load_embed_model():
    global embed_model
    try:
        from sentence_transformers import SentenceTransformer
        from config import embed as embed_cfg
        logger.info(f"Loading embedding model: {embed_cfg.MODEL_NAME}...")
        embed_model = SentenceTransformer(embed_cfg.MODEL_NAME, cache_folder=embed_cfg.CACHE_DIR)
        logger.info(f"Embedding model loaded")
    except Exception as e:
        logger.warning(f"Failed to load BGE model: {e}")

def embed_texts(texts: list) -> list:
    if embed_model:
        return embed_model.encode(texts, normalize_embeddings=True).tolist()
    return None

# ========== 情感分析 ==========
def _keyword_sentiment_fallback(text: str) -> str:
    text_lower = text.lower()
    for keyword in sent_cfg.INSULT_KEYWORDS:
        if keyword in text_lower:
            return "angry"
    for keyword in sent_cfg.ANGRY_KEYWORDS:
        if keyword in text_lower:
            return "angry"
    return "normal"


async def sentiment_analyzer(state: AgentState) -> dict:
    """情感分析节点"""
    from eval_tracer import TraceContext, extract_token_counts
    text = state["merged_message"]
    trace = state.get("_trace")

    # 优先关键词
    score = _keyword_sentiment_fallback(text)
    if score == "angry":
        if trace:
            trace.end_node("sentiment_analyzer")
        return {"sentiment_score": "angry", "sentiment_label": "angry", "sentiment_confidence": 0.9}

    if not llm_client:
        if trace:
            trace.end_node("sentiment_analyzer")
        return {"sentiment_score": "normal", "sentiment_label": "neutral", "sentiment_confidence": 0.6}

    try:
        response = await llm_client.chat.completions.create(
            model=llm_cfg.MODEL,
            messages=[
                {"role": "system", "content": """你是一个情绪分析器。分析客户消息的情绪，返回JSON格式：{"label": "angry|neutral|positive", "confidence": 0.0-1.0, "reason": "简短原因"}。

重要规则：
- 只有当客户明确表达愤怒、威胁、辱骂时才判定为 angry
- 产品反馈、投诉、建议、疑问都属于 neutral，不是 angry
- "太重了"、"不好用"、"有问题"等产品反馈是 neutral
- 只有"我要投诉"、"你们是骗子"、"垃圾产品"等才是 angry

只返回JSON，不要其他内容。"""},
                {"role": "user", "content": f"分析以下客户消息的情绪：\n\n{text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=100,
        )
        result = json.loads(response.choices[0].message.content)
        label = result.get("label", "neutral")
        confidence = result.get("confidence", 0.5)
        score = "angry" if label == "angry" else "normal"
        if trace:
            tokens_in, tokens_out = extract_token_counts(response.usage)
            trace.record_llm_call("sentiment_analyzer", llm_cfg.MODEL, tokens_in, tokens_out)
            trace.end_node("sentiment_analyzer")
        return {"sentiment_score": score, "sentiment_label": label, "sentiment_confidence": confidence}
    except Exception as e:
        logger.warning(f"Sentiment LLM failed: {e}")
        score = _keyword_sentiment_fallback(text)
        label = "angry" if score == "angry" else "neutral"
        if trace:
            trace.end_node("sentiment_analyzer", error=str(e))
        return {"sentiment_score": score, "sentiment_label": label, "sentiment_confidence": 0.8 if score == "angry" else 0.6}


# ========== 查询优化 ==========
async def query_optimizer(state: AgentState) -> dict:
    """查询优化节点"""
    from eval_tracer import TraceContext, extract_token_counts
    text = state["merged_message"]
    trace = state.get("_trace")

    # 简单查询跳过
    if len(text) < 10 or any(p in text for p in ["MP12", "M8100", "KDS", "你好", "价格"]):
        if trace:
            trace.end_node("query_optimizer")
        return {"intent": "other", "optimized_query": text, "keywords": []}

    if not llm_client:
        if trace:
            trace.end_node("query_optimizer")
        return {"intent": "other", "optimized_query": text, "keywords": []}

    try:
        response = await llm_client.chat.completions.create(
            model=llm_cfg.MODEL,
            messages=[
                {"role": "system", "content": "你是一个查询优化器。分析客户消息，返回JSON格式：{\"intent\": \"pricing_query|feature_inquiry|comparison|support|greeting|other\", \"optimized_query\": \"优化后的中文搜索查询\", \"keywords\": [\"关键词1\", \"关键词2\"]}。只返回JSON。"},
                {"role": "user", "content": f"分析以下客户消息：\n\n{text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=200,
        )
        result = json.loads(response.choices[0].message.content)
        if trace:
            tokens_in, tokens_out = extract_token_counts(response.usage)
            trace.record_llm_call("query_optimizer", llm_cfg.MODEL, tokens_in, tokens_out)
            trace.end_node("query_optimizer")
        return {
            "intent": result.get("intent", "other"),
            "optimized_query": result.get("optimized_query", text),
            "keywords": result.get("keywords", []),
        }
    except Exception as e:
        logger.warning(f"Query optimizer failed: {e}")
        if trace:
            trace.end_node("query_optimizer", error=str(e))
        return {"intent": "other", "optimized_query": text, "keywords": []}


# ========== 知识检索 ==========
async def knowledge_retriever(state: AgentState, collection=None) -> dict:
    """知识检索节点"""
    from eval_tracer import TraceContext, extract_token_counts, extract_distances
    base_query = state.get("optimized_query") or state["merged_message"]
    search_query = base_query
    trace = state.get("_trace")

    # 使用默认 collection
    if collection is None:
        from backend import knowledge_collection
        collection = knowledge_collection

    # 简单查询跳过改写
    if len(base_query) > 10 and llm_client:
        try:
            rewrite_response = await llm_client.chat.completions.create(
                model=llm_cfg.MODEL,
                messages=[{"role": "user", "content": f"将以下搜索查询改写为更适合向量搜索的版本，保持简洁中文，只返回改写后的查询文本：\n\n{base_query}"}],
                temperature=0.1, max_tokens=100,
            )
            search_query = rewrite_response.choices[0].message.content.strip()
            if trace:
                tokens_in, tokens_out = extract_token_counts(rewrite_response.usage)
                trace.record_llm_call("knowledge_retriever_rewrite", llm_cfg.MODEL, tokens_in, tokens_out)
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")

    if collection.count() == 0:
        if trace:
            trace.record_retrieval("knowledge_retriever", [], 0)
            trace.end_node("knowledge_retriever")
        return {"retrieved_knowledge": ""}

    # 语义搜索
    results = collection.query(query_texts=[search_query], n_results=5)
    candidates = []
    distances = []
    if results and results["documents"] and results["documents"][0]:
        candidates = results["documents"][0]
        distances = extract_distances(results)

    # 关键词补充
    keywords = [kw for kw in search_query.split() if len(kw) > 1]
    all_docs = collection.get()
    if all_docs and all_docs["documents"]:
        for doc in all_docs["documents"]:
            if any(kw in doc for kw in keywords) and doc not in candidates:
                candidates.append(doc)
                distances.append(0.3)

    if trace:
        trace.record_retrieval("knowledge_retriever", distances, len(candidates))

    # Rerank
    if len(candidates) > 3 and llm_client:
        avg_dist = sum(distances) / len(distances) if distances else 1.0
        if avg_dist < 0.3:
            candidates = candidates[:3]
        else:
            try:
                rerank_response = await llm_client.chat.completions.create(
                    model=llm_cfg.MODEL,
                    messages=[
                        {"role": "system", "content": "根据客户查询对检索结果进行相关性排序。返回JSON：{\"ranked\": [索引列表]}，从最相关到最不相关。"},
                        {"role": "user", "content": f"客户查询：{base_query}\n\n检索结果：{json.dumps(candidates, ensure_ascii=False)}"}
                    ],
                    response_format={"type": "json_object"}, temperature=0.0, max_tokens=50,
                )
                order = json.loads(rerank_response.choices[0].message.content).get("ranked", list(range(len(candidates))))
                candidates = [candidates[i] for i in order[:3] if i < len(candidates)]
                if trace:
                    tokens_in, tokens_out = extract_token_counts(rerank_response.usage)
                    trace.record_llm_call("knowledge_retriever_rerank", llm_cfg.MODEL, tokens_in, tokens_out)
            except Exception as e:
                logger.warning(f"Rerank failed: {e}")
                candidates = candidates[:3]
    else:
        candidates = candidates[:3]

    knowledge = "\n".join(candidates)
    if trace:
        trace.end_node("knowledge_retriever")
    return {"retrieved_knowledge": knowledge}


# ========== 响应生成 ==========
SALES_SOP = {
    "greeting": "您好！很高兴为您服务！",
    "professional": "作为专业销售顾问，我将为您提供最准确的产品信息。",
    "closing": "请问还有什么可以帮助您的吗？"
}


async def response_generator(state: AgentState) -> dict:
    """非流式响应生成"""
    from eval_tracer import TraceContext, extract_token_counts
    knowledge = state["retrieved_knowledge"]
    sentiment_label = state.get("sentiment_label", "neutral")
    chat_history = state.get("chat_history", [])
    trace = state.get("_trace")

    if not llm_client:
        reply = f"{SALES_SOP['greeting']}\n\n{knowledge if knowledge else '抱歉，我暂时没有找到相关信息。'}\n\n{SALES_SOP['closing']}"
        return {"reply": reply, "status": "normal"}

    cleaned_history = await _clean_chat_history(chat_history, state.get("merged_message", ""))
    system_prompt = f"""你是一个智能销售助手。根据知识库中的信息回答客户问题。

严格规则：
1. 始终保持礼貌和专业
2. **必须严格基于"检索到的知识"回答，禁止编造**
3. 如果知识库没有相关信息，告知客户建议联系人工客服
4. 回复简洁清晰，不超过 200 字
5. 结尾使用：{SALES_SOP['closing']}
6. 客户情绪：{sentiment_label}"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in cleaned_history:
        role = "assistant" if msg.get("role") == "assistant" else "user"
        content = msg.get("content", "")
        if content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": f"客户消息：{state['merged_message']}\n\n检索到的知识：\n{knowledge or '暂无相关信息'}\n\n请基于以上知识回答。"})

    try:
        response = await llm_client.chat.completions.create(model=llm_cfg.MODEL, messages=messages, temperature=llm_cfg.TEMPERATURE, max_tokens=llm_cfg.MAX_TOKENS)
        reply = response.choices[0].message.content
        if trace:
            tokens_in, tokens_out = extract_token_counts(response.usage)
            trace.record_llm_call("response_generator", llm_cfg.MODEL, tokens_in, tokens_out)
            trace.end_node("response_generator")
        return {"reply": reply, "status": "normal"}
    except Exception as e:
        logger.warning(f"Response generator failed: {e}")
        reply = f"{SALES_SOP['greeting']}\n\n抱歉，我暂时无法回答。\n\n{SALES_SOP['closing']}"
        return {"reply": reply, "status": "normal"}


async def response_generator_stream(state: AgentState):
    """流式响应生成"""
    from eval_tracer import TraceContext, extract_token_counts
    knowledge = state["retrieved_knowledge"]
    sentiment_label = state.get("sentiment_label", "neutral")
    chat_history = state.get("chat_history", [])

    if not llm_client:
        full_reply = f"{SALES_SOP['greeting']}\n\n{knowledge or '抱歉，我暂时没有找到相关信息。'}\n\n{SALES_SOP['closing']}"
        yield {"event": "token", "data": full_reply}
        yield {"event": "done", "data": json.dumps({"reply": full_reply, "status": "normal", "sentiment_label": state.get("sentiment_label", "neutral"), "sentiment_confidence": state.get("sentiment_confidence", 0.5), "intent": state.get("intent", "other"), "retrieved_knowledge": state.get("retrieved_knowledge", "")})}
        return

    cleaned_history = await _clean_chat_history(chat_history, state.get("merged_message", ""))
    system_prompt = f"""你是一个智能销售助手。根据知识库中的信息回答客户问题。

严格规则：
1. 始终保持礼貌和专业
2. **必须严格基于"检索到的知识"回答，禁止编造**
3. 如果知识库没有相关信息，告知客户建议联系人工客服
4. 回复简洁清晰，不超过 200 字
5. 结尾使用：{SALES_SOP['closing']}
6. 客户情绪：{sentiment_label}"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in cleaned_history:
        role = "assistant" if msg.get("role") == "assistant" else "user"
        content = msg.get("content", "")
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": f"客户消息：{state['merged_message']}\n\n检索到的知识：\n{knowledge or '暂无相关信息'}\n\n请基于以上知识回答。"})

    full_reply = ""
    try:
        stream = await llm_client.chat.completions.create(model=llm_cfg.MODEL, messages=messages, temperature=llm_cfg.TEMPERATURE, max_tokens=llm_cfg.MAX_TOKENS, stream=True)
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_reply += token
                yield {"event": "token", "data": token}
    except Exception as e:
        logger.warning(f"Stream response failed: {e}")
        full_reply = f"{SALES_SOP['greeting']}\n\n抱歉，我暂时无法回答。\n\n{SALES_SOP['closing']}"
        yield {"event": "token", "data": full_reply}

    yield {"event": "done", "data": json.dumps({"reply": full_reply, "status": "normal", "sentiment_label": state.get("sentiment_label", "neutral"), "sentiment_confidence": state.get("sentiment_confidence", 0.5), "intent": state.get("intent", "other"), "retrieved_knowledge": state.get("retrieved_knowledge", "")})}

    # 记录用量
    api_key = state.get("_api_key", "")
    if api_key:
        from database import log_usage
        trace = state.get("_trace")
        if trace:
            trace.finish()
        log_usage(api_key, "", 0, 0, trace.trace.total_duration_ms if trace else 0, "normal")


# ========== 人工转接 ==========
def handover_response(state: AgentState) -> dict:
    """人工转接响应"""
    reply = "您好，检测到您可能有不满情绪。正在为您转接人工客服，请稍候..."
    return {"reply": reply, "status": "human"}


def human_handover_router(state: AgentState) -> str:
    """转人工路由"""
    if state["sentiment_score"] == "angry":
        return "handover"
    return "continue"


# ========== 辅助函数 ==========
async def _clean_chat_history(chat_history: list, current_query: str) -> list:
    """清洗对话历史"""
    if len(chat_history) <= 6:
        return chat_history

    topic_switch = await _detect_topic_switch(chat_history, current_query)
    if topic_switch:
        return chat_history[-6:]

    summary = await _summarize_history(chat_history)
    if summary:
        return [{"role": "system", "content": f"[历史摘要] {summary}"}] + chat_history[-6:]
    return chat_history[-10:]


async def _detect_topic_switch(chat_history: list, current_query: str) -> bool:
    """检测话题转换"""
    if len(chat_history) < 4 or not llm_client:
        return False
    last_query = ""
    for msg in reversed(chat_history):
        if msg.get("role") == "user":
            last_query = msg.get("content", "")
            break
    if not last_query:
        return False
    try:
        response = await llm_client.chat.completions.create(
            model=llm_cfg.MODEL,
            messages=[{"role": "user", "content": f"判断以下两个问题是否在讨论同一个话题。\n\n上一个问题：{last_query[:100]}\n当前问题：{current_query[:100]}\n\n如果是同一话题返回\"same\"，如果是新话题返回\"different\"。只返回一个词。"}],
            temperature=0.0, max_tokens=10,
        )
        return "different" in response.choices[0].message.content.lower()
    except Exception:
        return False


async def _summarize_history(chat_history: list) -> str:
    """对话摘要"""
    if len(chat_history) <= 6 or not llm_client:
        return ""
    try:
        history_text = json.dumps(chat_history[-10:], ensure_ascii=False)
        response = await llm_client.chat.completions.create(
            model=llm_cfg.MODEL,
            messages=[{"role": "user", "content": f"请将以下对话历史压缩为简短摘要，保留：\n1. 用户询问过的主要产品/问题\n2. 关键的决策或结论\n3. 未解决的问题\n\n对话历史：{history_text}\n\n返回格式：简短的中文摘要，不超过100字。"}],
            temperature=0.1, max_tokens=150,
        )
        return response.choices[0].message.content
    except Exception:
        return ""
