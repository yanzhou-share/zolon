import asyncio
import json
import logging
import os
import time
import hashlib
import secrets
import threading
from collections import defaultdict
from datetime import datetime
from typing import TypedDict, List, Tuple, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, UploadFile, File, Header, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import AsyncOpenAI
from sse_starlette.sse import EventSourceResponse
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, END

from eval_tracer import TraceContext, extract_token_counts, extract_distances
from eval_pipeline import auto_evaluate_request, batch_evaluate, get_eval_stats
from eval_retrieval import evaluate_retrieval_batch
from database import log_usage, get_dashboard_stats, get_merchant_stats, calculate_billing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== SaaS 多商户管理 ==========
MERCHANTS_FILE = os.path.join(os.path.dirname(__file__), "merchants.json")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "admin123")

def load_merchants() -> dict:
    if os.path.exists(MERCHANTS_FILE):
        with open(MERCHANTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"merchants": {}}

def save_merchants(data: dict):
    with open(MERCHANTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_api_key() -> str:
    return "zk_" + secrets.token_hex(6)

def validate_api_key(api_key: str) -> Optional[dict]:
    merchants = load_merchants()
    merchant = merchants.get("merchants", {}).get(api_key)
    if merchant and merchant.get("status") == "active":
        return merchant
    return None

def validate_origin(api_key: str, origin: str, referer: str) -> bool:
    merchants = load_merchants()
    merchant = merchants.get("merchants", {}).get(api_key)
    if not merchant:
        return False
    allowed = merchant.get("allowed_origins", [])
    if not allowed:
        return True
    source = origin or referer or ""
    for allowed_origin in allowed:
        if source.startswith(allowed_origin):
            return True
    return False

# 限流: {api_key: [timestamp, ...]}
_request_counts = defaultdict(list)

def check_rate_limit(api_key: str, limit: int = 100) -> bool:
    now = time.time()
    _request_counts[api_key] = [t for t in _request_counts[api_key] if now - t < 60]
    if len(_request_counts[api_key]) >= limit:
        return False
    _request_counts[api_key].append(now)
    return True

app = FastAPI(title="AI Sales Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/widget", response_class=HTMLResponse)
async def serve_widget():
    widget_path = os.path.join(STATIC_DIR, "widget.html")
    with open(widget_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/demo", response_class=HTMLResponse)
async def serve_demo():
    demo_path = os.path.join(STATIC_DIR, "demo.html")
    with open(demo_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

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

def get_tenant_collection(api_key: str):
    return chroma_client.get_or_create_collection(
        name=f"knowledge_{api_key}",
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


def parse_pdf(file_path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def _delete_file_chunks(filename: str, collection=None) -> int:
    col = collection or knowledge_collection
    results = col.get()
    ids_to_delete = []
    if results and results["metadatas"]:
        for id_, meta in zip(results["ids"], results["metadatas"]):
            if meta["source"] == filename:
                ids_to_delete.append(id_)
    if ids_to_delete:
        col.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def _delete_disk_file(filename: str):
    import glob
    pattern = os.path.join(UPLOAD_DIR, f"*_{filename}")
    for f in glob.glob(pattern):
        try:
            os.remove(f)
            logger.info(f"Deleted disk file: {f}")
        except Exception as e:
            logger.warning(f"Failed to delete disk file {f}: {e}")


@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: str = Header(alias="X-API-Key")
):
    merchant = validate_api_key(x_api_key)
    if not merchant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    origin = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")
    if not validate_origin(x_api_key, origin, referer):
        raise HTTPException(status_code=403, detail="Domain not allowed")

    collection = get_tenant_collection(x_api_key)

    if file.size and file.size > FILE_SIZE_LIMIT:
        return JSONResponse(status_code=413, content={"error": "文件大小超过10MB限制"})

    suffix = file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""
    if suffix not in ("txt", "docx", "md", "pdf"):
        return JSONResponse(status_code=400, content={"error": "仅支持 .txt、.docx、.md 和 .pdf 格式"})

    content = await file.read()
    content_hash = hashlib.md5(content).hexdigest()[:12]

    existing_count = _delete_file_chunks(file.filename, collection)
    if existing_count > 0:
        _delete_disk_file(file.filename)
        logger.info(f"Replaced existing file '{file.filename}': deleted {existing_count} old chunks")

    save_path = os.path.join(UPLOAD_DIR, f"{content_hash}_{file.filename}")
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        if suffix in ("txt", "md"):
            text = parse_txt(save_path)
        elif suffix == "pdf":
            text = parse_pdf(save_path)
        else:
            text = parse_docx(save_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"文件解析失败: {str(e)}"})

    if not text.strip():
        return JSONResponse(status_code=400, content={"error": "文件内容为空"})

    chunks = chunk_text(text)
    upload_time = datetime.now().isoformat()

    ids = [f"{content_hash}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": file.filename,
            "content_hash": content_hash,
            "upload_time": upload_time,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        for i in range(len(chunks))
    ]

    collection.add(ids=ids, documents=chunks, metadatas=metadatas)

    logger.info(f"Uploaded '{file.filename}' to tenant {merchant.get('name')}: {len(chunks)} chunks indexed.")
    return {"status": "success", "filename": file.filename, "chunks": len(chunks), "file_id": content_hash}


@app.get("/upload/list")
async def list_uploads(x_api_key: str = Header(alias="X-API-Key")):
    merchant = validate_api_key(x_api_key)
    if not merchant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    collection = get_tenant_collection(x_api_key)
    results = collection.get()
    files = {}
    if results and results["metadatas"]:
        for meta in results["metadatas"]:
            name = meta["source"]
            if name not in files:
                files[name] = {"filename": name, "upload_time": meta["upload_time"], "chunks": 0}
            files[name]["chunks"] += 1
    return {"files": list(files.values())}


@app.delete("/upload/{filename}")
async def delete_upload(filename: str, x_api_key: str = Header(alias="X-API-Key")):
    merchant = validate_api_key(x_api_key)
    if not merchant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    collection = get_tenant_collection(x_api_key)
    deleted_chunks = _delete_file_chunks(filename, collection)
    _delete_disk_file(filename)
    return {"status": "success", "deleted_chunks": deleted_chunks}


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
    chat_history: List[dict]
    _trace: Optional[object]
    _api_key: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    chat_history: List[dict] = []


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
    return {"merged_message": merged, "_trace": TraceContext(session_id)}


async def sentiment_analyzer(state: AgentState) -> dict:
    text = state["merged_message"]
    trace: Optional[TraceContext] = state.get("_trace")
    if trace:
        trace.start_node("sentiment_analyzer")

    if not llm_client:
        score = _keyword_sentiment_fallback(text)
        label = "angry" if score == "angry" else "neutral"
        if trace:
            trace.end_node("sentiment_analyzer")
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

        if trace:
            tokens_in, tokens_out = extract_token_counts(response.usage)
            trace.record_llm_call("sentiment_analyzer", LLM_MODEL, tokens_in, tokens_out)
            trace.end_node("sentiment_analyzer")

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
        if trace:
            trace.end_node("sentiment_analyzer", error=str(e))
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
    trace: Optional[TraceContext] = state.get("_trace")
    if trace:
        trace.start_node("query_optimizer")

    if not llm_client:
        if trace:
            trace.end_node("query_optimizer")
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

        if trace:
            tokens_in, tokens_out = extract_token_counts(response.usage)
            trace.record_llm_call("query_optimizer", LLM_MODEL, tokens_in, tokens_out)
            trace.end_node("query_optimizer")

        logger.info(f"Query_Optimizer (LLM): intent={result.get('intent')}, query={result.get('optimized_query', '')[:50]}")
        return {
            "intent": result.get("intent", "other"),
            "optimized_query": result.get("optimized_query", text),
            "keywords": result.get("keywords", []),
        }
    except Exception as e:
        logger.warning(f"Query optimizer LLM failed, using fallback: {e}")
        if trace:
            trace.end_node("query_optimizer", error=str(e))
        return {
            "intent": "other",
            "optimized_query": text,
            "keywords": [],
        }


async def knowledge_retriever(state: AgentState) -> dict:
    base_query = state.get("optimized_query") or state["merged_message"]
    search_query = base_query
    trace: Optional[TraceContext] = state.get("_trace")
    if trace:
        trace.start_node("knowledge_retriever")

    api_key = state.get("_api_key", "")
    collection = get_tenant_collection(api_key) if api_key else knowledge_collection

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
            if trace:
                tokens_in, tokens_out = extract_token_counts(rewrite_response.usage)
                trace.record_llm_call("knowledge_retriever_rewrite", LLM_MODEL, tokens_in, tokens_out)
        except Exception as e:
            logger.warning(f"Query rewrite LLM failed: {e}")

    logger.info(f"Knowledge_Retriever: query={search_query[:50]}, collection_count={collection.count()}")

    if collection.count() == 0:
        logger.info("Knowledge_Retriever: no documents in knowledge base")
        if trace:
            trace.record_retrieval("knowledge_retriever", [], 0)
            trace.end_node("knowledge_retriever")
        return {"retrieved_knowledge": ""}

    results = collection.query(
        query_texts=[search_query], n_results=5
    )

    candidates = []
    distances = []
    if results and results["documents"] and results["documents"][0]:
        candidates = results["documents"][0]
        distances = extract_distances(results)

    if trace:
        trace.record_retrieval("knowledge_retriever", distances, len(candidates))

    logger.info(f"Knowledge_Retriever: found {len(candidates)} candidates")

    if len(candidates) > 3 and llm_client:
        avg_dist = sum(distances) / len(distances) if distances else 1.0
        skip_rerank = avg_dist < 0.3
        if skip_rerank:
            logger.info(f"Knowledge_Retriever: skip rerank (avg_dist={avg_dist:.4f} < 0.3)")
            candidates = candidates[:3]
        else:
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
                if trace:
                    tokens_in, tokens_out = extract_token_counts(rerank_response.usage)
                    trace.record_llm_call("knowledge_retriever_rerank", LLM_MODEL, tokens_in, tokens_out)
            except Exception as e:
                logger.warning(f"Rerank LLM failed: {e}")
                candidates = candidates[:3]
    else:
        candidates = candidates[:3]

    knowledge = "\n".join(candidates)
    logger.info(f"Knowledge_Retriever: final {len(candidates)} results")
    if trace:
        trace.end_node("knowledge_retriever")
    return {"retrieved_knowledge": knowledge}


async def response_generator(state: AgentState) -> dict:
    knowledge = state["retrieved_knowledge"]
    sentiment_label = state.get("sentiment_label", "neutral")
    chat_history = state.get("chat_history", [])
    trace: Optional[TraceContext] = state.get("_trace")
    if trace:
        trace.start_node("response_generator")

    if not llm_client:
        if not knowledge:
            reply = f"{SALES_SOP['greeting']}\n\n抱歉，我暂时没有找到相关信息。请问您想了解什么？\n\n{SALES_SOP['closing']}"
        else:
            reply = f"{SALES_SOP['greeting']}\n\n{knowledge}\n\n{SALES_SOP['closing']}"
        if trace:
            trace.end_node("response_generator")
        return {"reply": reply, "status": "normal"}

    system_prompt = f"""你是一个智能销售助手。根据知识库中的信息回答客户问题。

严格规则（必须遵守）：
1. 始终保持礼貌和专业
2. **必须严格基于"检索到的知识"回答，禁止编造、推测或添加知识库中不存在的信息**
3. 如果知识库中有相关信息，仅基于该信息回答，不要添加额外内容
4. 如果知识库中没有相关信息，直接告知客户"抱歉，我暂时没有找到相关信息，建议您联系人工客服"
5. 不要猜测、不要推理、不要补充，只复述知识库中的内容
6. 回复简洁清晰，不超过 200 字
7. 结尾使用：{SALES_SOP['closing']}
8. 客户情绪：{sentiment_label}
9. 结合对话历史理解上下文，如果客户的问题指代之前的对话内容，请结合历史回答"""

    MAX_HISTORY = 10
    messages = [{"role": "system", "content": system_prompt}]

    history = chat_history[-MAX_HISTORY:] if len(chat_history) > MAX_HISTORY else chat_history
    for msg in history:
        role = "assistant" if msg.get("role") == "assistant" else "user"
        content = msg.get("content", "")
        if content:
            messages.append({"role": role, "content": content})

    user_msg = f"""客户消息：{state['merged_message']}

检索到的知识：
{knowledge if knowledge else '知识库中暂无相关信息'}

请基于以上知识和对话历史回答客户问题。"""

    messages.append({"role": "user", "content": user_msg})

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )
        reply = response.choices[0].message.content
        if trace:
            tokens_in, tokens_out = extract_token_counts(response.usage)
            trace.record_llm_call("response_generator", LLM_MODEL, tokens_in, tokens_out)
            trace.end_node("response_generator")
        logger.info(f"Response_Generator (LLM): reply={reply[:100]}...")
        return {"reply": reply, "status": "normal"}
    except Exception as e:
        logger.warning(f"Response generator LLM failed: {e}")
        if trace:
            trace.end_node("response_generator", error=str(e))
        if not knowledge:
            reply = f"{SALES_SOP['greeting']}\n\n抱歉，我暂时无法回答。请稍后重试。\n\n{SALES_SOP['closing']}"
        else:
            reply = f"{SALES_SOP['greeting']}\n\n{knowledge}\n\n{SALES_SOP['closing']}"
        return {"reply": reply, "status": "normal"}


async def response_generator_stream(state: AgentState):
    knowledge = state["retrieved_knowledge"]
    sentiment_label = state.get("sentiment_label", "neutral")
    chat_history = state.get("chat_history", [])

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

严格规则（必须遵守）：
1. 始终保持礼貌和专业
2. **必须严格基于"检索到的知识"回答，禁止编造、推测或添加知识库中不存在的信息**
3. 如果知识库中有相关信息，仅基于该信息回答，不要添加额外内容
4. 如果知识库中没有相关信息，直接告知客户"抱歉，我暂时没有找到相关信息，建议您联系人工客服"
5. 不要猜测、不要推理、不要补充，只复述知识库中的内容
6. 回复简洁清晰，不超过 200 字
7. 结尾使用：{SALES_SOP['closing']}
8. 客户情绪：{sentiment_label}
9. 结合对话历史理解上下文，如果客户的问题指代之前的对话内容，请结合历史回答"""

    MAX_HISTORY = 10
    messages = [{"role": "system", "content": system_prompt}]

    history = chat_history[-MAX_HISTORY:] if len(chat_history) > MAX_HISTORY else chat_history
    for msg in history:
        role = "assistant" if msg.get("role") == "assistant" else "user"
        content = msg.get("content", "")
        if content:
            messages.append({"role": role, "content": content})

    user_msg = f"""客户消息：{state['merged_message']}

检索到的知识：
{knowledge if knowledge else '知识库中暂无相关信息'}

请基于以上知识和对话历史回答客户问题。"""

    messages.append({"role": "user", "content": user_msg})

    full_reply = ""
    try:
        stream = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
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

    api_key = state.get("_api_key", "")
    if api_key:
        trace = state.get("_trace")
        if trace:
            trace.finish()
        tokens_in = trace.trace.total_llm_tokens_in if trace else 0
        tokens_out = trace.trace.total_llm_tokens_out if trace else 0
        latency = trace.trace.total_duration_ms if trace else 0
        from database import log_usage
        log_usage(api_key, "", tokens_in, tokens_out, latency, "normal")


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


def make_initial_state(request: ChatRequest, api_key: str = "") -> AgentState:
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
        "status": "normal",
        "chat_history": request.chat_history,
        "_trace": None,
        "_api_key": api_key,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_req: ChatRequest, x_api_key: str = Header(alias="X-API-Key")):
    merchant = validate_api_key(x_api_key)
    if not merchant:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    origin = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")
    if not validate_origin(x_api_key, origin, referer):
        raise HTTPException(status_code=403, detail="Domain not allowed")
    if not check_rate_limit(x_api_key, merchant.get("rate_limit", 100)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    logger.info(f"Input: session_id={chat_req.session_id}, message={chat_req.message}, merchant={merchant.get('name')}")

    initial_state = make_initial_state(chat_req, api_key=x_api_key)
    result = await graph.ainvoke(initial_state)

    trace: Optional[TraceContext] = result.get("_trace")
    if trace:
        trace.finish()
        trace.log_summary()
        try:
            await auto_evaluate_request(
                client=llm_client,
                trace=trace.trace,
                query=chat_req.message,
                answer=result["reply"],
                context=result["retrieved_knowledge"],
                retrieved_distances=[],
                skip_llm_judge=False,
            )
        except Exception as e:
            logger.warning(f"Auto-evaluation failed: {e}")

    logger.info(f"Final: status={result['status']}, sentiment={result['sentiment_label']}")

    trace: Optional[TraceContext] = result.get("_trace")
    tokens_in = trace.trace.total_llm_tokens_in if trace else 0
    tokens_out = trace.trace.total_llm_tokens_out if trace else 0
    latency = trace.trace.total_duration_ms if trace else 0
    log_usage(x_api_key, merchant.get("name", ""), tokens_in, tokens_out, latency, result["status"])

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
async def chat_stream(request: Request, chat_req: ChatRequest, x_api_key: str = Header(alias="X-API-Key")):
    merchant = validate_api_key(x_api_key)
    if not merchant:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    origin = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")
    if not validate_origin(x_api_key, origin, referer):
        raise HTTPException(status_code=403, detail="Domain not allowed")
    if not check_rate_limit(x_api_key, merchant.get("rate_limit", 100)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    logger.info(f"Input (stream): session_id={chat_req.session_id}, message={chat_req.message}, merchant={merchant.get('name')}")

    initial_state = make_initial_state(chat_req, api_key=x_api_key)

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
            trace = initial_state.get("_trace")
            if trace:
                trace.finish()
            log_usage(x_api_key, merchant.get("name", ""), 0, 0, trace.trace.total_duration_ms if trace else 0, "human")

        return EventSourceResponse(handover_stream())

    query_result = await query_optimizer(initial_state)
    initial_state.update(query_result)

    knowledge_result = await knowledge_retriever(initial_state)
    initial_state.update(knowledge_result)

    return EventSourceResponse(response_generator_stream(initial_state))


# ========== 商户管理 API ==========
@app.post("/admin/merchants")
async def create_merchant(
    name: str,
    admin_key: str = Header(alias="X-Admin-Key")
):
    if admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    api_key = generate_api_key()
    merchants = load_merchants()
    merchants["merchants"][api_key] = {
        "name": name,
        "status": "active",
        "created_at": datetime.now().isoformat(),
        "rate_limit": 100,
        "daily_limit": 5000
    }
    save_merchants(merchants)
    logger.info(f"Created merchant: {name}, api_key={api_key}")
    return {"api_key": api_key, "name": name}


@app.get("/admin/merchants")
async def list_merchants(admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    merchants = load_merchants()
    result = []
    for key, info in merchants.get("merchants", {}).items():
        result.append({"api_key": key, **info})
    return {"merchants": result}


@app.delete("/admin/merchants/{api_key}")
async def deactivate_merchant(api_key: str, admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    merchants = load_merchants()
    if api_key in merchants.get("merchants", {}):
        merchants["merchants"][api_key]["status"] = "inactive"
        save_merchants(merchants)
        return {"status": "deactivated", "api_key": api_key}
    raise HTTPException(status_code=404, detail="Merchant not found")


@app.put("/admin/merchants/{api_key}/origins")
async def update_origins(api_key: str, origins: list[str], admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    merchants = load_merchants()
    if api_key in merchants.get("merchants", {}):
        merchants["merchants"][api_key]["allowed_origins"] = origins
        save_merchants(merchants)
        return {"api_key": api_key, "allowed_origins": origins}
    raise HTTPException(status_code=404, detail="Merchant not found")


@app.get("/admin/dashboard")
async def get_dashboard(days: int = 30, admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return get_dashboard_stats(days)


@app.get("/admin/dashboard/merchant/{api_key}")
async def get_merchant_dashboard(api_key: str, days: int = 30, admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return get_merchant_stats(api_key, days)


@app.get("/admin/billing/{api_key}")
async def get_merchant_billing(api_key: str, month: str = None, admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return calculate_billing(api_key, month)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/eval/stats")
async def eval_stats():
    return get_eval_stats()


class BatchEvalRequest(BaseModel):
    test_cases: list[dict]


@app.post("/eval/batch")
async def eval_batch(request: BatchEvalRequest):
    report = await batch_evaluate(
        client=llm_client,
        test_cases=request.test_cases,
        knowledge_collection=knowledge_collection,
    )
    return report.to_dict()


@app.get("/eval/dataset")
async def eval_dataset():
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Dataset not found"}


@app.post("/eval/run-dataset")
async def eval_run_dataset():
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    if not os.path.exists(dataset_path):
        return JSONResponse(status_code=404, content={"error": "Dataset not found"})

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    test_cases = dataset.get("test_cases", [])
    report = await batch_evaluate(
        client=llm_client,
        test_cases=test_cases,
        knowledge_collection=knowledge_collection,
    )
    return report.to_dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
