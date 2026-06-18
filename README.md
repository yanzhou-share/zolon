# AI 智能销售终端

基于 FastAPI + Streamlit + ChromaDB + DeepSeek LLM 的 AI 销售助手系统。

## 功能特性

- **RAG 问答**: 基于 ChromaDB 向量数据库的知识库问答
- **文件上传**: 支持 .txt 和 .docx 文件，自动切分、向量化存储
- **情感分析**: 使用 DeepSeek LLM 实时判定用户情绪
- **流式响应**: SSE 流式输出，逐 token 渲染
- **会话持久化**: 对话历史保存到本地，刷新不丢失
- **转人工**: 检测到愤怒情绪时自动转接人工客服

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置

创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-xxx  # DeepSeek API Key（可选）
HF_ENDPOINT=https://hf-mirror.com  # HuggingFace 镜像（加速模型下载）
```

## 启动服务

### 1. 启动后端

```bash
python backend.py
```

后端运行在 http://localhost:8000

### 2. 启动前端

```bash
streamlit run app.py
```

前端运行在 http://localhost:8501

## 运行测试

```bash
pytest test_backend.py -v
```

## API 接口

### POST /chat

请求：
```json
{
  "session_id": "session_123",
  "message": "你好"
}
```

响应：
```json
{
  "reply": "您好！很高兴为您服务！...",
  "status": "normal",
  "sentiment_label": "neutral",
  "intent": "greeting"
}
```

### POST /chat/stream

SSE 流式响应，事件格式：
```
event: token
data: 您

event: token
data: 好

event: done
data: {"reply": "您好！", "status": "normal", ...}
```

### POST /upload

上传文件（multipart/form-data）：

```bash
curl -X POST http://localhost:8000/upload -F "file=@产品手册.txt"
```

响应：
```json
{
  "status": "success",
  "filename": "产品手册.txt",
  "chunks": 12
}
```

### GET /upload/list

响应：
```json
{
  "files": [
    {"filename": "产品手册.txt", "upload_time": "2026-01-01T00:00:00", "chunks": 12}
  ]
}
```

### DELETE /upload/{filename}

### GET /health

## 项目结构

```
zolon/
├── backend.py          # FastAPI 后端 + LangGraph 节点
├── app.py              # Streamlit 前端
├── test_backend.py     # 测试用例
├── requirements.txt    # 依赖列表
├── .env                # 环境变量
├── chroma_db/          # ChromaDB 持久化存储
├── uploads/            # 上传的文件
├── sessions/           # 会话数据
├── hf_cache/           # BGE 模型缓存
└── README.md           # 项目说明
```
