import json
import logging
from typing import Optional

from config import llm as llm_cfg, sentiment as sent_cfg, chat as chat_cfg
from models import AgentState

logger = logging.getLogger(__name__)

# ========== Token 计数工具 ==========
def count_tokens(text: str) -> int:
    """统计文本 token 数（使用 BGE 模型的 tokenizer）"""
    if not text:
        return 0
    if embed_model is None:
        # 模型未加载时粗略估算
        cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - cn_chars
        return int(cn_chars * 1.5 + other_chars / 4)
    return len(embed_model.tokenizer.encode(text))

# ========== 上下文治理参数 ==========
MODEL_MAX_CONTEXT = chat_cfg.MODEL_MAX_CONTEXT
SAFETY_MARGIN = chat_cfg.SAFETY_MARGIN
RESPONSE_TOKEN_BUDGET = chat_cfg.RESPONSE_TOKEN_BUDGET
SYSTEM_PROMPT_TOKENS = 150  # 系统提示估算

# LLM 客户端（延迟初始化）
llm_client = None

def init_llm():
    global llm_client
    if llm_cfg.API_KEY:
        from openai import AsyncOpenAI
        llm_client = AsyncOpenAI(api_key=llm_cfg.API_KEY, base_url=llm_cfg.BASE_URL)

# Embedding 模型（延迟加载）
embed_model = None
embed_ready = None  # threading.Event，模型加载完成后 set

def _load_embed_model():
    global embed_model, embed_ready
    try:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer
        from config import embed as embed_cfg
        logger.info(f"Loading embedding model: {embed_cfg.MODEL_NAME}...")
        embed_model = SentenceTransformer(embed_cfg.MODEL_NAME, cache_folder=embed_cfg.CACHE_DIR)
        logger.info(f"Embedding model loaded")
    except Exception as e:
        logger.warning(f"Failed to load BGE model: {e}")
    finally:
        if embed_ready:
            embed_ready.set()

def embed_texts(texts: list) -> list:
    if embed_model:
        return embed_model.encode(texts, normalize_embeddings=True).tolist()
    return None


# BGE 检索指令前缀（P0 优化：提升检索质量）
DOC_PREFIX = "为这个句子生成表示以用于检索相关内容："
QUERY_PREFIX = "为这个句子生成表示以用于检索相关内容："


class BGEEncodingFn:
    """ChromaDB 自定义 EmbeddingFunction，包装 BGE 模型"""
    is_legacy = False

    def __call__(self, input: list[str]) -> list[list[float]]:
        if embed_model:
            prefixed = [DOC_PREFIX + t for t in input]
            return embed_model.encode(prefixed, normalize_embeddings=True).tolist()
        raise RuntimeError("BGE model not loaded")

    def embed_query(self, input) -> list:
        if embed_model:
            texts = [input] if isinstance(input, str) else input
            prefixed = [QUERY_PREFIX + t for t in texts]
            return embed_model.encode(prefixed, normalize_embeddings=True).tolist()
        raise RuntimeError("BGE model not loaded")

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def name(self) -> str:
        return "bge-small-zh"

    def json(self):
        return {"name": self.name()}

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
    from eval.eval_tracer import TraceContext, extract_token_counts
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
    from eval.eval_tracer import TraceContext, extract_token_counts
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
    from eval.eval_tracer import TraceContext, extract_token_counts, extract_distances
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

    # ========== 阶段 1：语义搜索粗筛 TOP20 ==========
    COARSE_TOP_K = 20
    RELEVANCE_THRESHOLD = 0.6
    results = collection.query(
        query_texts=[search_query],
        n_results=COARSE_TOP_K,
        where={"status": "active"}
    )

    # all_data = collection.get()
    # for idx, doc_id in enumerate(all_data["ids"]):
    #     text = all_data["documents"][idx]
    #     meta = all_data["metadatas"][idx]
    #     print(f"\n【ID】{doc_id}")
    #     print(f"【文本】{text}")
    #     print(f"【元数据】{meta}")

    candidates = []
    distances = []
    if results and results["documents"] and results["documents"][0]:
        for doc, dist in zip(
            results["documents"][0],
            extract_distances(results),
        ):
            if dist <= RELEVANCE_THRESHOLD:
                candidates.append(doc)
                distances.append(dist)

    if trace:
        trace.record_retrieval("knowledge_retriever_coarse", distances, len(candidates))

    if not candidates:
        if trace:
            trace.end_node("knowledge_retriever")
        return {"retrieved_knowledge": ""}

    # ========== 阶段 2：LLM Rerank 精排 TOP5 ==========
    FINE_TOP_K = 5
    if len(candidates) > FINE_TOP_K and llm_client:
        try:
            rerank_response = await llm_client.chat.completions.create(
                model=llm_cfg.MODEL,
                messages=[
                    {"role": "system", "content": "根据客户查询对检索结果进行相关性排序。返回JSON：{\"ranked\": [索引列表]}，从最相关到最不相关。"},
                    {"role": "user", "content": f"客户查询：{base_query}\n\n检索结果：{json.dumps(candidates[:20], ensure_ascii=False)}"}
                ],
                response_format={"type": "json_object"}, temperature=0.0, max_tokens=100,
            )
            order = json.loads(rerank_response.choices[0].message.content).get("ranked", list(range(len(candidates))))
            candidates = [candidates[i] for i in order[:FINE_TOP_K] if i < len(candidates)]
            if trace:
                tokens_in, tokens_out = extract_token_counts(rerank_response.usage)
                trace.record_llm_call("knowledge_retriever_rerank", llm_cfg.MODEL, tokens_in, tokens_out)
        except Exception as e:
            logger.warning(f"Rerank failed: {e}")
            candidates = candidates[:FINE_TOP_K]
    else:
        candidates = candidates[:FINE_TOP_K]

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
    from eval.eval_tracer import TraceContext, extract_token_counts
    knowledge = state["retrieved_knowledge"]
    sentiment_label = state.get("sentiment_label", "neutral")
    chat_history = state.get("chat_history", [])
    trace = state.get("_trace")

    if not llm_client:
        reply = f"{SALES_SOP['greeting']}\n\n{knowledge if knowledge else '抱歉，我暂时没有找到相关信息。'}\n\n{SALES_SOP['closing']}"
        return {"reply": reply, "status": "normal"}

    cleaned_history = await _clean_chat_history(chat_history, state.get("merged_message", ""), knowledge)
    system_prompt = f"""你是一个智能销售助手。

严格规则：
1. 始终保持礼貌和专业
2. 如果"检索到的知识"包含与客户问题相关的信息，基于该知识回答
3. 如果"检索到的知识"与客户问题无关或为空，**不要提及知识库内容**，直接友好地回答
4. 回复简洁清晰，不超过 200 字
5. 客户情绪：{sentiment_label}"""

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
    from eval.eval_tracer import TraceContext, extract_token_counts
    knowledge = state["retrieved_knowledge"]
    sentiment_label = state.get("sentiment_label", "neutral")
    chat_history = state.get("chat_history", [])

    if not llm_client:
        full_reply = f"{SALES_SOP['greeting']}\n\n{knowledge or '抱歉，我暂时没有找到相关信息。'}\n\n{SALES_SOP['closing']}"
        yield {"event": "token", "data": full_reply}
        yield {"event": "done", "data": json.dumps({"reply": full_reply, "status": "normal", "sentiment_label": state.get("sentiment_label", "neutral"), "sentiment_confidence": state.get("sentiment_confidence", 0.5), "intent": state.get("intent", "other"), "retrieved_knowledge": state.get("retrieved_knowledge", "")})}
        return

    cleaned_history = await _clean_chat_history(chat_history, state.get("merged_message", ""), knowledge)
    system_prompt = f"""你是一个智能销售助手。

严格规则：
1. 始终保持礼貌和专业
2. 如果"检索到的知识"包含与客户问题相关的信息，基于该知识回答
3. 如果"检索到的知识"与客户问题无关或为空，**不要提及知识库内容**，直接友好地回答
4. 回复简洁清晰，不超过 200 字
5. 客户情绪：{sentiment_label}"""

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
def _truncate_by_tokens(messages: list, budget: int) -> list:
    """按 token 预算从尾部保留消息"""
    result = []
    used = 0
    for msg in reversed(messages):
        msg_tokens = count_tokens(msg.get("content", ""))
        if used + msg_tokens > budget:
            break
        result.append(msg)
        used += msg_tokens
    result.reverse()
    return result


async def _clean_chat_history(chat_history: list, current_query: str, knowledge: str = "") -> list:
    """基于 token 预算的上下文清洗"""
    if not chat_history:
        return chat_history

    # 1. 计算当前历史 token 数
    history_tokens = sum(count_tokens(msg.get("content", "")) for msg in chat_history)

    # 2. 计算可用空间
    reserved = SYSTEM_PROMPT_TOKENS + RESPONSE_TOKEN_BUDGET + count_tokens(knowledge) + count_tokens(current_query)
    available = int(MODEL_MAX_CONTEXT * SAFETY_MARGIN) - reserved

    logger.info(f"Context治理: history={history_tokens} tokens, available={available} tokens, margin={MODEL_MAX_CONTEXT}×{SAFETY_MARGIN}")

    # 3. 未超限，原样返回
    if history_tokens <= available:
        return chat_history

    logger.info(f"Context超限，触发压缩: {history_tokens} > {available}")

    # 4. 超限，触发压缩
    # 方案 A：话题切换 → 截断
    topic_switch = await _detect_topic_switch(chat_history, current_query)
    if topic_switch:
        truncated = _truncate_by_tokens(chat_history, available)
        logger.info(f"话题切换截断: {len(chat_history)} → {len(truncated)} 条")
        return truncated

    # 方案 B：摘要 + 最近N条
    summary = await _summarize_history(chat_history)
    if summary:
        summary_tokens = count_tokens(f"[历史摘要] {summary}")
        remaining_budget = available - summary_tokens
        recent = _truncate_by_tokens(chat_history, remaining_budget)
        logger.info(f"摘要压缩: 摘要={summary_tokens} tokens, 保留{len(recent)}条原始消息")
        return [{"role": "system", "content": f"[历史摘要] {summary}"}] + recent

    # 方案 C：直接截断
    truncated = _truncate_by_tokens(chat_history, available)
    logger.info(f"直接截断: {len(chat_history)} → {len(truncated)} 条")
    return truncated


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
