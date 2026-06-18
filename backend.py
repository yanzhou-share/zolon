import asyncio
import json
import logging
import os
import time
import hashlib
import threading
from collections import defaultdict
from datetime import datetime
from typing import TypedDict, List, Tuple

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from sse_starlette.sse import EventSourceResponse
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, END

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Sales Agent")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_MODEL = "deepseek-chat"
llm_client = None

if DEEPSEEK_API_KEY:
    llm_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Service exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=200,
        content={"reply": f"服务处理异常: {str(exc)}", "status": "normal"}
    )


EMBED_MODEL_NAME = "BAAI/bge-small-zh"
embed_model = None

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "hf_cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)


def _load_embed_model():
    global embed_model
    try:
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}...")
        embed_model = SentenceTransformer(EMBED_MODEL_NAME, cache_folder=os.environ["HF_HOME"])
        logger.info(f"Embedding model loaded: {EMBED_MODEL_NAME}")
    except Exception as e:
        logger.warning(f"Failed to load BGE model: {e}. Using ChromaDB default embedding.")


threading.Thread(target=_load_embed_model, daemon=True).start()


def embed_texts(texts: list) -> list:
    if embed_model:
        return embed_model.encode(texts, normalize_embeddings=True).tolist()
    return None


CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
knowledge_collection = chroma_client.get_or_create_collection(
    name="uploaded_knowledge",
    metadata={"hnsw:space": "cosine"}
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

FILE_SIZE_LIMIT = 10 * 1024 * 1024

SALES_SOP = {
    "greeting": "您好！很高兴为您服务！",
    "professional": "作为专业销售顾问，我将为您提供最准确的产品信息。",
    "closing": "请问还有什么可以帮助您的吗？"
}


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def parse_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_docx(file_path: str) -> str:
    from docx import Document
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.size and file.size > FILE_SIZE_LIMIT:
        return JSONResponse(status_code=413, content={"error": "文件大小超过10MB限制"})

    suffix = file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""
    if suffix not in ("txt", "docx"):
        return JSONResponse(status_code=400, content={"error": "仅支持 .txt 和 .docx 格式"})

    file_id = hashlib.md5(f"{file.filename}-{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        if suffix == "txt":
            text = parse_txt(save_path)
        else:
            text = parse_docx(save_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"文件解析失败: {str(e)}"})

    if not text.strip():
        return JSONResponse(status_code=400, content={"error": "文件内容为空"})

    chunks = chunk_text(text)
    upload_time = datetime.now().isoformat()

    ids = [f"{file_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": file.filename,
            "upload_time": upload_time,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        for i in range(len(chunks))
    ]

    knowledge_collection.add(ids=ids, documents=chunks, metadatas=metadatas)

    logger.info(f"Uploaded '{file.filename}': {len(chunks)} chunks indexed.")
    return {"status": "success", "filename": file.filename, "chunks": len(chunks), "file_id": file_id}


@app.get("/upload/list")
async def list_uploads():
    results = knowledge_collection.get()
    files = {}
    if results and results["metadatas"]:
        for meta in results["metadatas"]:
            name = meta["source"]
            if name not in files:
                files[name] = {"filename": name, "upload_time": meta["upload_time"], "chunks": 0}
            files[name]["chunks"] += 1
    return {"files": list(files.values())}


@app.delete("/upload/{filename}")
async def delete_upload(filename: str):
    results = knowledge_collection.get()
    ids_to_delete = []
    if results and results["metadatas"]:
        for id_, meta in zip(results["ids"], results["metadatas"]):
            if meta["source"] == filename:
                ids_to_delete.append(id_)
    if ids_to_delete:
        knowledge_collection.delete(ids=ids_to_delete)
    return {"status": "success", "deleted_chunks": len(ids_to_delete)}


message_buffer: dict[str, list[tuple[float, str]]] = defaultdict(list)
CONTEXT_MERGE_WINDOW = 5.0

INSULT_KEYWORDS = [
    "傻逼", "操你", "他妈", "去死", "废物", "白痴", "智障",
    "脑残", "煞笔", "混蛋", "王八蛋", "fuck", "shit", "stupid",
    "idiot", "damn"
]

ANGRY_KEYWORDS = ["生气", "愤怒", "不满", "投诉", "差评", "垃圾", "退货", "退款"]


class AgentState(TypedDict):
    session_id: str
    raw_message: str
    merged_message: str
    sentiment_score: str
    sentiment_label: str
    sentiment_confidence: float
    intent: str
    optimized_query: str
    keywords: List[str]
    retrieved_knowledge: str
    reply: str
    status: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    status: str
    sentiment_score: str = "normal"
    sentiment_label: str = "neutral"
    sentiment_confidence: float = 0.5
    intent: str = "other"
    retrieved_knowledge: str = ""


def _keyword_sentiment_fallback(text: str) -> str:
    text_lower = text.lower()
    for keyword in INSULT_KEYWORDS:
        if keyword in text_lower:
            return "angry"
    for keyword in ANGRY_KEYWORDS:
        if keyword in text_lower:
            return "angry"
    return "normal"


def input_collector(state: AgentState) -> dict:
    session_id = state["session_id"]
    new_message = state["raw_message"]
    now = time.time()
    buffer = message_buffer[session_id]

    buffer = [(ts, msg) for ts, msg in buffer if now - ts < CONTEXT_MERGE_WINDOW]
    message_buffer[session_id] = buffer

    buffer.append((now, new_message))

    if len(buffer) >= 2:
        merged = " ".join([msg for _, msg in buffer])
        message_buffer[session_id] = []
    else:
        merged = new_message

    logger.info(f"Input_Collector: merged_message={merged}")
    return {"merged_message": merged}


async def sentiment_analyzer(state: AgentState) -> dict:
    text = state["merged_message"]

    if not llm_client:
        score = _keyword_sentiment_fallback(text)
        label = "angry" if score == "angry" else "neutral"
        return {
            "sentiment_score": score,
            "sentiment_label": label,
            "sentiment_confidence": 0.8 if score == "angry" else 0.6,
        }

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个情绪分析器。分析客户消息的情绪，返回JSON格式：{\"label\": \"angry|neutral|positive\", \"confidence\": 0.0-1.0, \"reason\": \"简短原因\"}。只返回JSON，不要其他内容。"},
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

        logger.info(f"Sentiment_Analyzer (LLM): label={label}, confidence={confidence}")
        return {
            "sentiment_score": score,
            "sentiment_label": label,
            "sentiment_confidence": confidence,
        }
    except Exception as e:
        logger.warning(f"Sentiment LLM failed, using fallback: {e}")
        score = _keyword_sentiment_fallback(text)
        label = "angry" if score == "angry" else "neutral"
        return {
            "sentiment_score": score,
            "sentiment_label": label,
            "sentiment_confidence": 0.8 if score == "angry" else 0.6,
        }


def human_handover_router(state: AgentState) -> str:
    if state["sentiment_score"] == "angry":
        return "handover"
    return "continue"


async def query_optimizer(state: AgentState) -> dict:
    text = state["merged_message"]

    if not llm_client:
        return {
            "intent": "other",
            "optimized_query": text,
            "keywords": [],
        }

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个查询优化器。分析客户消息，返回JSON格式：{\"intent\": \"pricing_query|feature_inquiry|comparison|support|greeting|other\", \"optimized_query\": \"优化后的中文搜索查询\", \"keywords\": [\"关键词1\", \"关键词2\"]}。只返回JSON。"},
                {"role": "user", "content": f"分析以下客户消息：\n\n{text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=200,
        )
        result = json.loads(response.choices[0].message.content)

        logger.info(f"Query_Optimizer (LLM): intent={result.get('intent')}, query={result.get('optimized_query', '')[:50]}")
        return {
            "intent": result.get("intent", "other"),
            "optimized_query": result.get("optimized_query", text),
            "keywords": result.get("keywords", []),
        }
    except Exception as e:
        logger.warning(f"Query optimizer LLM failed, using fallback: {e}")
        return {
            "intent": "other",
            "optimized_query": text,
            "keywords": [],
        }


async def knowledge_retriever(state: AgentState) -> dict:
    base_query = state.get("optimized_query") or state["merged_message"]
    search_query = base_query

    if llm_client:
        try:
            rewrite_response = await llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": f"将以下搜索查询改写为更适合向量搜索的版本，保持简洁中文，只返回改写后的查询文本：\n\n{base_query}"}
                ],
                temperature=0.1,
                max_tokens=100,
            )
            search_query = rewrite_response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Query rewrite LLM failed: {e}")

    logger.info(f"Knowledge_Retriever: query={search_query[:50]}, collection_count={knowledge_collection.count()}")

    if knowledge_collection.count() == 0:
        logger.info("Knowledge_Retriever: no documents in knowledge base")
        return {"retrieved_knowledge": ""}

    results = knowledge_collection.query(
        query_texts=[search_query], n_results=5
    )

    candidates = []
    if results and results["documents"] and results["documents"][0]:
        candidates = results["documents"][0]

    logger.info(f"Knowledge_Retriever: found {len(candidates)} candidates")

    if len(candidates) > 3 and llm_client:
        try:
            rerank_response = await llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "根据客户查询对检索结果进行相关性排序。返回JSON：{\"ranked\": [索引列表]}，从最相关到最不相关。"},
                    {"role": "user", "content": f"客户查询：{base_query}\n\n检索结果：{json.dumps(candidates, ensure_ascii=False)}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=50,
            )
            order = json.loads(rerank_response.choices[0].message.content).get("ranked", list(range(len(candidates))))
            candidates = [candidates[i] for i in order[:3] if i < len(candidates)]
        except Exception as e:
            logger.warning(f"Rerank LLM failed: {e}")
            candidates = candidates[:3]
    else:
        candidates = candidates[:3]

    knowledge = "\n".join(candidates)
    logger.info(f"Knowledge_Retriever: final {len(candidates)} results")
    return {"retrieved_knowledge": knowledge}


async def response_generator(state: AgentState) -> dict:
    knowledge = state["retrieved_knowledge"]
    sentiment_label = state.get("sentiment_label", "neutral")

    if not llm_client:
        if not knowledge:
            reply = f"{SALES_SOP['greeting']}\n\n抱歉，我暂时没有找到相关信息。请问您想了解什么？\n\n{SALES_SOP['closing']}"
        else:
            reply = f"{SALES_SOP['greeting']}\n\n{knowledge}\n\n{SALES_SOP['closing']}"
        return {"reply": reply, "status": "normal"}

    system_prompt = f"""你是一个智能销售助手。根据知识库中的信息回答客户问题。

规则：
1. 始终保持礼貌和专业
2. 优先使用检索到的知识回答问题
3. 如果检索到相关知识，基于知识内容详细回答
4. 如果没有检索到相关知识，告知客户您会尽快查找
5. 回复简洁清晰
6. 结尾使用：{SALES_SOP['closing']}
7. 客户情绪：{sentiment_label}"""

    user_msg = f"""客户消息：{state['merged_message']}

检索到的知识：
{knowledge if knowledge else '知识库中暂无相关信息'}

请基于以上知识回答客户问题。"""

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=800,
        )
        reply = response.choices[0].message.content
        logger.info(f"Response_Generator (LLM): reply={reply[:100]}...")
        return {"reply": reply, "status": "normal"}
    except Exception as e:
        logger.warning(f"Response generator LLM failed: {e}")
        if not knowledge:
            reply = f"{SALES_SOP['greeting']}\n\n抱歉，我暂时无法回答。请稍后重试。\n\n{SALES_SOP['closing']}"
        else:
            reply = f"{SALES_SOP['greeting']}\n\n{knowledge}\n\n{SALES_SOP['closing']}"
        return {"reply": reply, "status": "normal"}


async def response_generator_stream(state: AgentState):
    knowledge = state["retrieved_knowledge"]
    sentiment_label = state.get("sentiment_label", "neutral")

    if not llm_client:
        if not knowledge:
            full_reply = f"{SALES_SOP['greeting']}\n\n抱歉，我暂时没有找到相关信息。\n\n{SALES_SOP['closing']}"
        else:
            full_reply = f"{SALES_SOP['greeting']}\n\n{knowledge}\n\n{SALES_SOP['closing']}"
        yield {"event": "token", "data": full_reply}
        yield {"event": "done", "data": json.dumps({
            "reply": full_reply,
            "status": "normal",
            "sentiment_score": state.get("sentiment_score", "normal"),
            "sentiment_label": state.get("sentiment_label", "neutral"),
            "sentiment_confidence": state.get("sentiment_confidence", 0.5),
            "intent": state.get("intent", "other"),
            "retrieved_knowledge": state.get("retrieved_knowledge", "")
        })}
        return

    system_prompt = f"""你是一个智能销售助手。根据知识库中的信息回答客户问题。

规则：
1. 始终保持礼貌和专业
2. 优先使用检索到的知识回答问题
3. 如果检索到相关知识，基于知识内容详细回答
4. 如果没有检索到相关知识，告知客户您会尽快查找
5. 回复简洁清晰
6. 结尾使用：{SALES_SOP['closing']}
7. 客户情绪：{sentiment_label}"""

    user_msg = f"""客户消息：{state['merged_message']}

检索到的知识：
{knowledge if knowledge else '知识库中暂无相关信息'}

请基于以上知识回答客户问题。"""

    full_reply = ""
    try:
        stream = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=800,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_reply += token
                yield {"event": "token", "data": token}
    except Exception as e:
        logger.warning(f"Stream response LLM failed: {e}")
        if not knowledge:
            full_reply = f"{SALES_SOP['greeting']}\n\n抱歉，我暂时无法回答。\n\n{SALES_SOP['closing']}"
        else:
            full_reply = f"{SALES_SOP['greeting']}\n\n{knowledge}\n\n{SALES_SOP['closing']}"
        yield {"event": "token", "data": full_reply}

    yield {"event": "done", "data": json.dumps({
        "reply": full_reply,
        "status": "normal",
        "sentiment_score": state.get("sentiment_score", "normal"),
        "sentiment_label": state.get("sentiment_label", "neutral"),
        "sentiment_confidence": state.get("sentiment_confidence", 0.5),
        "intent": state.get("intent", "other"),
        "retrieved_knowledge": state.get("retrieved_knowledge", "")
    })}


def handover_response(state: AgentState) -> dict:
    reply = "您好，检测到您可能有不满情绪。正在为您转接人工客服，请稍候..."
    logger.info(f"Human_Handover_Router: triggering handover")
    return {"reply": reply, "status": "human"}


workflow = StateGraph(AgentState)

workflow.add_node("input_collector", input_collector)
workflow.add_node("sentiment_analyzer", sentiment_analyzer)
workflow.add_node("query_optimizer", query_optimizer)
workflow.add_node("knowledge_retriever", knowledge_retriever)
workflow.add_node("response_generator", response_generator)
workflow.add_node("handover_response", handover_response)

workflow.set_entry_point("input_collector")

workflow.add_edge("input_collector", "sentiment_analyzer")

workflow.add_conditional_edges(
    "sentiment_analyzer",
    human_handover_router,
    {
        "handover": "handover_response",
        "continue": "query_optimizer"
    }
)

workflow.add_edge("query_optimizer", "knowledge_retriever")
workflow.add_edge("knowledge_retriever", "response_generator")
workflow.add_edge("response_generator", END)
workflow.add_edge("handover_response", END)

graph = workflow.compile()


def make_initial_state(request: ChatRequest) -> AgentState:
    return {
        "session_id": request.session_id,
        "raw_message": request.message,
        "merged_message": "",
        "sentiment_score": "normal",
        "sentiment_label": "neutral",
        "sentiment_confidence": 0.5,
        "intent": "other",
        "optimized_query": "",
        "keywords": [],
        "retrieved_knowledge": "",
        "reply": "",
        "status": "normal"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"Input: session_id={request.session_id}, message={request.message}")

    initial_state = make_initial_state(request)
    result = await graph.ainvoke(initial_state)

    logger.info(f"Final: status={result['status']}, sentiment={result['sentiment_label']}")

    return ChatResponse(
        reply=result["reply"],
        status=result["status"],
        sentiment_score=result["sentiment_score"],
        sentiment_label=result.get("sentiment_label", "neutral"),
        sentiment_confidence=result.get("sentiment_confidence", 0.5),
        intent=result.get("intent", "other"),
        retrieved_knowledge=result["retrieved_knowledge"]
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    logger.info(f"Input (stream): session_id={request.session_id}, message={request.message}")

    initial_state = make_initial_state(request)

    merged_state = input_collector(initial_state)
    initial_state.update(merged_state)

    sentiment_result = await sentiment_analyzer(initial_state)
    initial_state.update(sentiment_result)

    if initial_state["sentiment_score"] == "angry":
        handover_result = handover_response(initial_state)
        initial_state.update(handover_result)

        async def handover_stream():
            yield {"event": "token", "data": initial_state["reply"]}
            yield {"event": "done", "data": json.dumps({
                "reply": initial_state["reply"],
                "status": "human",
                "sentiment_score": initial_state["sentiment_score"],
                "sentiment_label": initial_state["sentiment_label"],
                "sentiment_confidence": initial_state["sentiment_confidence"],
                "intent": "other",
                "retrieved_knowledge": ""
            })}

        return EventSourceResponse(handover_stream())

    query_result = await query_optimizer(initial_state)
    initial_state.update(query_result)

    knowledge_result = await knowledge_retriever(initial_state)
    initial_state.update(knowledge_result)

    return EventSourceResponse(response_generator_stream(initial_state))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
