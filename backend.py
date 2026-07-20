"""
AI 客服 SaaS 平台 - 主入口

FastAPI 应用入口，导入各模块组装完整后端。
"""

import json
import logging
import os
import time
import hashlib
from datetime import datetime
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, Header, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from config import llm, embed, chroma, upload, chat as chat_cfg, chunk as chunk_cfg, merchant as merchant_cfg, dashboard as dash_cfg
from models import ChatRequest, ChatResponse, BatchEvalRequest
from merchants import load_merchants, save_merchants, generate_api_key, validate_api_key, validate_origin, check_rate_limit
from document_parser import chunk_text, chunk_text_semantic, parse_txt, parse_docx, parse_pdf
import chat_agent
from chat_agent import (
    init_llm, _load_embed_model, embed_texts,
    AgentState, sentiment_analyzer, query_optimizer, knowledge_retriever,
    response_generator, response_generator_stream, handover_response, human_handover_router,
    _keyword_sentiment_fallback, SALES_SOP,
)
from database import log_usage, get_dashboard_stats, get_merchant_stats, calculate_billing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== FastAPI 应用 ==========
app = FastAPI(title="AI 客服 SaaS 平台")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 静态文件 ==========
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


# ========== 异常处理 ==========
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Service exception: {str(exc)}", exc_info=True)
    return JSONResponse(status_code=200, content={"reply": f"服务处理异常: {str(exc)}", "status": "normal"})


# ========== ChromaDB ==========
os.makedirs(chroma.PERSIST_DIR, exist_ok=True)
import chromadb

# ========== Embedding (必须在 ChromaDB collection 之前加载) ==========
import threading
from chat_agent import _load_embed_model, BGEEncodingFn
import chat_agent

chat_agent.embed_ready = threading.Event()
threading.Thread(target=_load_embed_model, daemon=True).start()

bge_ef = BGEEncodingFn()


def _semantic_embed_fn(texts):
    """语义分块用的 embedding 函数（不加前缀，纯向量）"""
    from chat_agent import embed_model
    if embed_model is None:
        raise RuntimeError("BGE model not loaded")
    return embed_model.encode(texts, normalize_embeddings=True).tolist()


chroma_client = chromadb.PersistentClient(path=chroma.PERSIST_DIR)


def _get_or_recreate_collection(name: str, ef, meta=None):
    """获取 collection，若嵌入函数冲突则删除重建"""
    try:
        col = chroma_client.get_or_create_collection(name=name, metadata=meta, embedding_function=ef)
        logger.info(f"Collection '{name}' loaded, count={col.count()}")
        return col
    except ValueError:
        logger.warning(f"Collection '{name}' embedding conflict, recreating...")
        chroma_client.delete_collection(name)
        return chroma_client.create_collection(name=name, metadata=meta, embedding_function=ef)


knowledge_collection = _get_or_recreate_collection(
    chroma.DEFAULT_COLLECTION, bge_ef, {"hnsw:space": chroma.HNSW_SPACE}
)


def get_tenant_collection(api_key: str):
    return _get_or_recreate_collection(
        f"knowledge_{api_key}", bge_ef, {"hnsw:space": chroma.HNSW_SPACE}
    )

from eval_tracer import TraceContext, extract_token_counts, extract_distances


# ========== 会话管理 ==========
message_buffer: dict[str, list[tuple[float, str]]] = defaultdict(list)


def input_collector(state: AgentState) -> dict:
    session_id = state["session_id"]
    new_message = state["raw_message"]
    now = time.time()
    buffer = message_buffer[session_id]
    buffer = [(ts, msg) for ts, msg in buffer if now - ts < chat_cfg.CONTEXT_MERGE_WINDOW]
    message_buffer[session_id] = buffer
    buffer.append((now, new_message))
    if len(buffer) >= 2:
        merged = " ".join([msg for _, msg in buffer])
        message_buffer[session_id] = []
    else:
        merged = new_message
    return {"merged_message": merged, "_trace": TraceContext(session_id)}


# ========== 文件删除辅助 ==========
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
    pattern = os.path.join(upload.UPLOAD_DIR, f"*_{filename}")
    for f in glob.glob(pattern):
        try:
            os.remove(f)
        except Exception:
            pass


# ========== 初始状态 ==========
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


# ========== 认证依赖 ==========
async def require_api_key(
    request: Request,
    x_api_key: str = Header(alias="X-API-Key")
):
    merchant_info = validate_api_key(x_api_key)
    if not merchant_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    origin = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")
    if not validate_origin(x_api_key, origin, referer):
        raise HTTPException(status_code=403, detail="Domain not allowed")
    return merchant_info


# ========== 上传端点 ==========
@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...), x_api_key: str = Header(alias="X-API-Key")):
    merchant_info = validate_api_key(x_api_key)
    if not merchant_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not validate_origin(x_api_key, request.headers.get("origin", ""), request.headers.get("referer", "")):
        raise HTTPException(status_code=403, detail="Domain not allowed")

    collection = get_tenant_collection(x_api_key)

    if file.size and file.size > upload.FILE_SIZE_LIMIT:
        return JSONResponse(status_code=413, content={"error": "文件大小超过10MB限制"})

    suffix = file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""
    if suffix not in upload.ALLOWED_FORMATS:
        return JSONResponse(status_code=400, content={"error": f"仅支持 {', '.join('.' + f for f in upload.ALLOWED_FORMATS)} 格式"})

    content = await file.read()
    content_hash = hashlib.md5(content).hexdigest()[:12]

    # 版本管理：不再删除旧 chunks，保留所有版本
    save_path = os.path.join(upload.UPLOAD_DIR, f"{content_hash}_{file.filename}")
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
    print(text)
    if not text.strip():
        return JSONResponse(status_code=400, content={"error": "文件内容为空"})

    if chunk_cfg.USE_SEMANTIC_CHUNK:
        chunks = chunk_text_semantic(text, _semantic_embed_fn)
    else:
        chunks = chunk_text(text)
    upload_time = datetime.now().isoformat()

    # ========== 版本管理（文件存储） ==========
    versions_file = os.path.join(upload.UPLOAD_DIR, f"{file.filename}.versions.json")
    versions = {}
    if os.path.exists(versions_file):
        with open(versions_file, "r") as f:
            versions = json.load(f)

    # 查找最大版本号
    max_version = 0
    for v in versions:
        if v.isdigit() and int(v) > max_version:
            max_version = int(v)
    new_version = max_version + 1

    # 旧版本标记 deprecated
    if str(max_version) in versions:
        versions[str(max_version)]["status"] = "deprecated"

    # 记录新版本
    versions[str(new_version)] = {
        "version": new_version,
        "status": "active",
        "upload_time": upload_time,
        "file_type": suffix,
        "file_size": file.size or 0,
        "char_count": len(text),
        "content_hash": content_hash,
    }

    with open(versions_file, "w") as f:
        json.dump(versions, f, ensure_ascii=False, indent=2)

    logger.info(f"Version: file={file.filename}, new=v{new_version}")

    # 新版本 chunks
    ids = [f"{file.filename}_v{new_version}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{
        "source": file.filename,
        "version": new_version,
        "status": "active",
        "content_hash": content_hash,
        "upload_time": upload_time,
        "chunk_index": i,
        "total_chunks": len(chunks),
        "file_type": suffix,
        "file_size": file.size or 0,
        "char_count": len(text),
    } for i in range(len(chunks))]

    collection.add(ids=ids, documents=chunks, metadatas=metadatas)

    # 更新旧版本 chunks 的 status 为 deprecated
    if max_version > 0:
        old_results = collection.get(where={"$and": [{"source": file.filename}, {"status": "active"}]})
        if old_results and old_results["ids"]:
            new_id_set = set(ids)
            ids_to_deprecated = [i for i in old_results["ids"] if i not in new_id_set]
            if ids_to_deprecated:
                collection.update(
                    ids=ids_to_deprecated,
                    metadatas=[{"status": "deprecated"}] * len(ids_to_deprecated)
                )
                logger.info(f"Deprecated {len(ids_to_deprecated)} old chunks for '{file.filename}'")

    logger.info(f"Uploaded '{file.filename}' v{new_version} to tenant {merchant_info.get('name')}: {len(chunks)} chunks.")
    return {"status": "success", "filename": file.filename, "version": new_version, "chunks": len(chunks), "file_id": content_hash}


@app.get("/upload/list")
async def list_uploads(x_api_key: str = Header(alias="X-API-Key")):
    merchant_info = validate_api_key(x_api_key)
    if not merchant_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    collection = get_tenant_collection(x_api_key)
    results = collection.get()
    files = {}
    if results and results["metadatas"]:
        for meta in results["metadatas"]:
            name = meta["source"]
            status = meta.get("status", "active")
            version = meta.get("version", 1)
            if name not in files:
                files[name] = {"filename": name, "upload_time": meta["upload_time"], "chunks": 0, "latest_version": 0, "active_chunks": 0}
            files[name]["chunks"] += 1
            if version > files[name]["latest_version"]:
                files[name]["latest_version"] = version
            if status == "active":
                files[name]["active_chunks"] += 1
    return {"files": list(files.values())}


@app.delete("/upload/{filename}")
async def delete_upload(filename: str, x_api_key: str = Header(alias="X-API-Key")):
    merchant_info = validate_api_key(x_api_key)
    if not merchant_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    collection = get_tenant_collection(x_api_key)
    deleted_chunks = _delete_file_chunks(filename, collection)
    _delete_disk_file(filename)
    # 删除版本 JSON 文件
    versions_file = os.path.join(upload.UPLOAD_DIR, f"{filename}.versions.json")
    if os.path.exists(versions_file):
        os.remove(versions_file)
    # 大小写不敏感删除
    import glob
    for f in glob.glob(os.path.join(upload.UPLOAD_DIR, "*.versions.json")):
        if os.path.basename(f).lower() == f"{filename}.versions.json".lower():
            os.remove(f)
            break
    return {"status": "success", "deleted_chunks": deleted_chunks}


@app.get("/upload/{filename}/versions")
async def get_file_versions(filename: str, x_api_key: str = Header(alias="X-API-Key")):
    """获取文件版本历史"""
    merchant_info = validate_api_key(x_api_key)
    if not merchant_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    versions_file = os.path.join(upload.UPLOAD_DIR, f"{filename}.versions.json")
    if not os.path.exists(versions_file):
        return {"filename": filename, "versions": []}
    with open(versions_file, "r") as f:
        versions = json.load(f)
    return {"filename": filename, "versions": sorted(versions.values(), key=lambda x: x["version"], reverse=True)}


@app.post("/upload/{filename}/rollback/{version}")
async def rollback_version(filename: str, version: int, x_api_key: str = Header(alias="X-API-Key")):
    """回滚到指定版本"""
    merchant_info = validate_api_key(x_api_key)
    if not merchant_info:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 文件名大小写不敏感：查找实际存在的版本文件
    import glob
    versions_file = None
    # 精确匹配
    exact_path = os.path.join(upload.UPLOAD_DIR, f"{filename}.versions.json")
    if os.path.exists(exact_path):
        # 获取实际文件路径（处理大小写）
        for f in glob.glob(os.path.join(upload.UPLOAD_DIR, "*.versions.json")):
            if os.path.basename(f).lower() == f"{filename}.versions.json".lower():
                versions_file = f
                break
    # 大小写不敏感匹配
    if not versions_file:
        for f in glob.glob(os.path.join(upload.UPLOAD_DIR, "*.versions.json")):
            if os.path.basename(f).lower() == f"{filename}.versions.json".lower():
                versions_file = f
                break
    if not versions_file:
        raise HTTPException(status_code=404, detail="File not found")

    with open(versions_file, "r") as f:
        versions = json.load(f)

    target_key = str(version)
    if target_key not in versions:
        raise HTTPException(status_code=404, detail=f"Version {version} not found")

    # 所有版本设为 deprecated，目标版本设为 active
    for v in versions.values():
        v["status"] = "deprecated"
    versions[target_key]["status"] = "active"

    with open(versions_file, "w") as f:
        json.dump(versions, f, ensure_ascii=False, indent=2)

    # 同步更新 ChromaDB metadata
    collection = get_tenant_collection(x_api_key)
    actual_filename = os.path.basename(versions_file).replace(".versions.json", "")
    logger.info(f"Rollback sync: filename={actual_filename}, target_version={version}")
    all_data = collection.get()
    logger.info(f"ChromaDB chunks: {len(all_data['ids']) if all_data and all_data.get('ids') else 0}")

    ids_to_update = []
    metadatas_to_update = []
    if all_data and all_data["ids"]:
        for id_, meta in zip(all_data["ids"], all_data["metadatas"]):
            if meta.get("source") == actual_filename:
                if meta.get("version") == version:
                    ids_to_update.append(id_)
                    metadatas_to_update.append({"status": "active"})
                else:
                    ids_to_update.append(id_)
                    metadatas_to_update.append({"status": "deprecated"})

    logger.info(f"Chunks to update: {len(ids_to_update)}")
    if ids_to_update:
        collection.update(ids=ids_to_update, metadatas=metadatas_to_update)
        logger.info("ChromaDB metadata updated")

    # 使用实际文件名（从版本文件路径提取）
    return {"status": "success", "filename": actual_filename, "rolled_back_to": version}

@app.get("/test/embedding")
async def test_embedding():
    x_api_key = "zk_test12345678"
    query = "如何处理模型输出失败"
    collection = get_tenant_collection(x_api_key)

    count = collection.count()
    if count == 0:
        return {"status": "error", "message": "知识库为空，请先上传文件"}

    # 测试向量检索
    results = collection.query(query_texts=[query], n_results=5)
    logger.info(f"Test embedding query: {query}")
    logger.info(f"Results count: {len(results['documents'][0]) if results['documents'] else 0}")

    docs = results["documents"][0] if results and results["documents"] else []
    distances = results["distances"][0] if results and results["distances"] else []

    return {
        "status": "success",
        "query": query,
        "collection_count": count,
        "results": [
            {"document": doc[:100], "distance": round(dist, 4)}
            for doc, dist in zip(docs, distances)
        ]
    }

# ========== 聊天端点 ==========
@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_req: ChatRequest, x_api_key: str = Header(alias="X-API-Key")):
    merchant_info = await require_api_key(request, x_api_key)
    if not check_rate_limit(x_api_key, merchant_info.get("rate_limit", 100)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    initial_state = make_initial_state(chat_req, api_key=x_api_key)

    merged_state = input_collector(initial_state)
    initial_state.update(merged_state)

    # 并行执行
    # 情感分析
    sentiment_task = sentiment_analyzer(initial_state)
    query_task = query_optimizer(initial_state)
    sentiment_result, query_result = await asyncio.gather(sentiment_task, query_task)
    initial_state.update(sentiment_result)
    initial_state.update(query_result)

    if initial_state["sentiment_score"] == "angry":
        handover_result = handover_response(initial_state)
        initial_state.update(handover_result)
        trace = initial_state.get("_trace")
        if trace:
            trace.finish()
        log_usage(x_api_key, merchant_info.get("name", ""), 0, 0, trace.trace.total_duration_ms if trace else 0, "human")
        return ChatResponse(
            reply=initial_state["reply"], status="human",
            sentiment_score=initial_state["sentiment_score"],
            sentiment_label=initial_state.get("sentiment_label", "neutral"),
            sentiment_confidence=initial_state.get("sentiment_confidence", 0.5),
            intent="other", retrieved_knowledge=""
        )

    knowledge_result = await knowledge_retriever(initial_state, get_tenant_collection(x_api_key))
    initial_state.update(knowledge_result)

    response_result = await response_generator(initial_state)
    initial_state.update(response_result)

    trace = initial_state.get("_trace")
    if trace:
        trace.finish()
        trace.log_summary()
    tokens_in = trace.trace.total_llm_tokens_in if trace else 0
    tokens_out = trace.trace.total_llm_tokens_out if trace else 0
    latency = trace.trace.total_duration_ms if trace else 0
    log_usage(x_api_key, merchant_info.get("name", ""), tokens_in, tokens_out, latency, initial_state.get("status", "normal"))

    return ChatResponse(
        reply=initial_state["reply"], status=initial_state.get("status", "normal"),
        sentiment_score=initial_state["sentiment_score"],
        sentiment_label=initial_state.get("sentiment_label", "neutral"),
        sentiment_confidence=initial_state.get("sentiment_confidence", 0.5),
        intent=initial_state.get("intent", "other"),
        retrieved_knowledge=initial_state["retrieved_knowledge"]
    )


@app.post("/chat/stream")
async def chat_stream(request: Request, chat_req: ChatRequest, x_api_key: str = Header(alias="X-API-Key")):
    merchant_info = await require_api_key(request, x_api_key)
    if not check_rate_limit(x_api_key, merchant_info.get("rate_limit", 100)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    initial_state = make_initial_state(chat_req, api_key=x_api_key)

    merged_state = input_collector(initial_state)
    initial_state.update(merged_state)

    # 并行执行
    # 情感分析
    sentiment_task = sentiment_analyzer(initial_state)
    query_task = query_optimizer(initial_state)
    sentiment_result, query_result = await asyncio.gather(sentiment_task, query_task)
    initial_state.update(sentiment_result)
    initial_state.update(query_result)

    if initial_state["sentiment_score"] == "angry":
        handover_result = handover_response(initial_state)
        initial_state.update(handover_result)

        async def handover_stream():
            yield {"event": "token", "data": initial_state["reply"]}
            yield {"event": "done", "data": json.dumps({
                "reply": initial_state["reply"], "status": "human",
                "sentiment_score": initial_state["sentiment_score"],
                "sentiment_label": initial_state["sentiment_label"],
                "sentiment_confidence": initial_state["sentiment_confidence"],
                "intent": "other", "retrieved_knowledge": ""
            })}
            trace = initial_state.get("_trace")
            if trace:
                trace.finish()
            log_usage(x_api_key, merchant_info.get("name", ""), 0, 0, trace.trace.total_duration_ms if trace else 0, "human")

        return EventSourceResponse(handover_stream())
    
    # 知识库检索
    knowledge_result = await knowledge_retriever(initial_state, get_tenant_collection(x_api_key))
    initial_state.update(knowledge_result)

    return EventSourceResponse(response_generator_stream(initial_state))


# ========== 商户管理 API ==========
@app.post("/admin/merchants")
async def create_merchant(name: str, admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != merchant_cfg.ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    api_key = generate_api_key()
    merchants = load_merchants()
    merchants["merchants"][api_key] = {"name": name, "status": "active", "created_at": datetime.now().isoformat(), "allowed_origins": [], "rate_limit": 100, "daily_limit": 5000}
    save_merchants(merchants)
    return {"api_key": api_key, "name": name}


@app.get("/admin/merchants")
async def list_merchants(admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != merchant_cfg.ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    merchants = load_merchants()
    return {"merchants": [{"api_key": k, **v} for k, v in merchants.get("merchants", {}).items()]}


@app.delete("/admin/merchants/{api_key}")
async def deactivate_merchant(api_key: str, admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != merchant_cfg.ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    merchants = load_merchants()
    if api_key in merchants.get("merchants", {}):
        merchants["merchants"][api_key]["status"] = "inactive"
        save_merchants(merchants)
        return {"status": "deactivated", "api_key": api_key}
    raise HTTPException(status_code=404, detail="Merchant not found")


@app.put("/admin/merchants/{api_key}/origins")
async def update_origins(api_key: str, origins: list[str], admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != merchant_cfg.ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    merchants = load_merchants()
    if api_key in merchants.get("merchants", {}):
        merchants["merchants"][api_key]["allowed_origins"] = origins
        save_merchants(merchants)
        return {"api_key": api_key, "allowed_origins": origins}
    raise HTTPException(status_code=404, detail="Merchant not found")


# ========== 仪表盘 API ==========
@app.get("/admin/dashboard")
async def get_dashboard(days: int = 30, admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != merchant_cfg.ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return get_dashboard_stats(days)


@app.get("/admin/dashboard/merchant/{api_key}")
async def get_merchant_dashboard(api_key: str, days: int = 30, admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != merchant_cfg.ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return get_merchant_stats(api_key, days)


@app.get("/admin/billing/{api_key}")
async def get_merchant_billing(api_key: str, month: str = None, admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != merchant_cfg.ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return calculate_billing(api_key, month)


# ========== 评估 API ==========
@app.get("/admin/eval/trends")
async def get_eval_trends(days: int = 30, api_key: str = None, admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != merchant_cfg.ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    from eval_trends import get_trend
    return {"trends": get_trend(days, api_key)}


@app.get("/admin/eval/alerts")
async def get_eval_alerts(days: int = 7, api_key: str = None, admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != merchant_cfg.ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    from eval_alerts import get_recent_alerts, get_alert_stats
    return {"alerts": get_recent_alerts(days, api_key), "stats": get_alert_stats(days)}


@app.post("/admin/eval/run")
async def run_manual_eval(admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != merchant_cfg.ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    from eval_scheduler import daily_eval_task
    report = await daily_eval_task(chat_agent.llm_client, knowledge_collection)
    if report:
        return {"status": "success", "report": report.to_dict()}
    return {"status": "no_data"}


@app.get("/admin/eval/report")
async def get_latest_eval_report(report_type: str = "daily", admin_key: str = Header(alias="X-Admin-Key")):
    if admin_key != merchant_cfg.ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    from eval_scheduler import get_latest_report
    return get_latest_report(report_type)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/eval/stats")
async def eval_stats():
    from eval_pipeline import get_eval_stats
    return get_eval_stats()


@app.post("/eval/batch")
async def eval_batch(request: BatchEvalRequest):
    from eval_pipeline import batch_evaluate
    report = await batch_evaluate(client=chat_agent.llm_client, test_cases=request.test_cases, knowledge_collection=knowledge_collection)
    return report.to_dict()


@app.get("/eval/dataset")
async def eval_dataset():
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Dataset not found"}


@app.post("/eval/run-dataset")
async def eval_run_dataset(api_key: str = "zk_dftest12345"):
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    if not os.path.exists(dataset_path):
        return JSONResponse(status_code=404, content={"error": "Dataset not found"})
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    from eval_pipeline import batch_evaluate
    collection = get_tenant_collection(api_key)
    report = await batch_evaluate(client=chat_agent.llm_client, test_cases=dataset.get("test_cases", []), knowledge_collection=collection)
    return report.to_dict()


# ========== 启动 ==========
import asyncio
init_llm()
os.makedirs(upload.UPLOAD_DIR, exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
