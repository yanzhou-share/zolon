# 智能客服 SaaS 平台

基于 RAG 架构的多商户智能客服 SaaS 平台，支持企业快速部署 AI 客服系统。

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Python + FastAPI |
| AI Agent | LangGraph（6 节点状态图） |
| LLM | DeepSeek API（可配置模型/BaseURL） |
| 向量数据库 | ChromaDB（多租户隔离） |
| Embedding | BGE-small-zh |
| 前端 | Streamlit（管理后台）+ 嵌入式 Widget |
| 数据库 | SQLite（用量统计、计费、告警） |
| 流式响应 | SSE（Server-Sent Events） |
| 文档解析 | python-docx + pdfplumber |

## 功能特性

### 核心功能
- **RAG 知识库问答**: 递归分块 + 语义分块 + 多阶段检索（向量搜索 + LLM Rerank）
- **多租户 SaaS**: API Key 认证、域名白名单、租户知识库隔离
- **文档支持**: txt/docx/md/pdf，表格自动转 Markdown，Q&A 配对
- **流式响应**: SSE 逐 token 渲染
- **情感分析**: 关键词优先 + LLM 兜底，区分产品反馈和愤怒
- **长对话治理**: 话题检测 + 对话摘要 + 滑动窗口

### 运营功能
- **管理后台**: 商户 CRUD、文档管理、版本控制
- **运营仪表盘**: 用量统计、每日趋势、商户排行
- **计费系统**: 每千 Token 计价、月度免费额度

### 评估系统
- **多维度评估**: 检索质量（Recall/Precision/MRR/NDCG）、生成质量（Faithfulness/Relevancy/Hallucination）、端到端质量（Correctness/Quality/Satisfaction）
- **批量评估**: 35 条测试用例，覆盖 6 个文档
- **趋势追踪**: 指标趋势图表，支持按商户筛选
- **自动告警**: 指标异常检测（忠实度低、幻觉率高、延迟高）

### 嵌入式 Widget
- 一行代码集成到任意网站
- 支持 API Key 认证
- 流式逐字渲染 + 移动端适配

## 项目结构

```
zolon/
├── backend.py              # 主入口（FastAPI + LangGraph）
├── config.py               # 配置管理
├── models.py               # 数据模型
├── merchants.py            # 商户管理
├── document_parser.py      # 文档解析（递归分块 + 语义分块）
├── chat_agent.py           # Agent 节点（情感/查询/检索/生成）
├── database.py             # SQLite 数据库
├── text_cleaner.py         # 文本清洗
├── app.py                  # Streamlit 前端
│
├── eval/                   # 评估模块
│   ├── eval_pipeline.py    # 评估编排
│   ├── eval_retrieval.py   # 检索评估指标
│   ├── eval_generation.py  # 生成质量评估
│   ├── eval_e2e.py         # 端到端评估
│   ├── eval_tracer.py      # 请求追踪
│   ├── eval_trends.py      # 指标趋势追踪
│   ├── eval_alerts.py      # 告警系统
│   ├── eval_scheduler.py   # 定时评估任务
│   └── eval_dataset.json   # 评估测试集（35条）
│
├── tests/                  # 测试用例
│   ├── test_backend.py     # 后端测试（35个）
│   ├── test_eval.py        # 评估系统测试（40个）
│   ├── test_retrieval_recall.py
│   └── ...
│
├── rag_doc/                # RAG 测试文档
├── eval_metrics/           # 评估指标输出
├── requirements.txt        # 依赖列表
├── .env.example            # 配置示例
└── merchants.json          # 商户配置
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入 API Key
```

### 3. 启动服务

```bash
# 后端
python backend.py

# 前端（新终端）
streamlit run app.py
```

### 4. 访问

- **管理后台**: http://localhost:8501
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## 配置说明

所有配置可通过环境变量或 `.env` 文件设置：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DEEPSEEK_API_KEY` | - | DeepSeek API Key |
| `LLM_MODEL` | deepseek-chat | LLM 模型 |
| `LLM_BASE_URL` | https://api.deepseek.com | LLM Base URL |
| `LLM_TEMPERATURE` | 0.3 | 温度参数 |
| `ADMIN_SECRET` | admin123 | 管理员密钥 |
| `CHUNK_SIZE` | 500 | 文档分块大小 |
| `CHUNK_OVERLAP` | 50 | 分块重叠 |
| `USE_SEMANTIC_CHUNK` | false | 启用语义分块 |
| `SEMANTIC_THRESHOLD` | 0.5 | 语义断点阈值 |
| `MAX_HISTORY` | 10 | 最大对话历史 |

## API 接口

### 聊天接口

```bash
# 非流式
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: zk_xxx" \
  -d '{"session_id": "s1", "message": "你好"}'

# 流式
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: zk_xxx" \
  -d '{"session_id": "s1", "message": "你好"}'
```

### 文件上传

```bash
curl -X POST http://localhost:8000/upload \
  -H "X-API-Key: zk_xxx" \
  -F "file=@产品手册.docx"
```

### 商户管理

```bash
# 创建商户
curl -X POST "http://localhost:8000/admin/merchants?name=商户A" \
  -H "X-Admin-Key: admin123"

# 列出商户
curl http://localhost:8000/admin/merchants \
  -H "X-Admin-Key: admin123"
```

### 评估接口

```bash
# 运行数据集评估
curl -X POST http://localhost:8000/eval/run-dataset \
  -H "X-Admin-Key: admin123"

# 查看评估统计
curl http://localhost:8000/eval/stats

# 查看趋势
curl http://localhost:8000/admin/eval/trends \
  -H "X-Admin-Key: admin123"
```

## 运行测试

```bash
# 评估系统测试
pytest tests/test_eval.py -v

# 后端测试
pytest tests/test_backend.py -v

# 所有测试
pytest tests/ -v
```

## 嵌入式 Widget

在任意 HTML 页面中引入：

```html
<script>window.AI_CHAT_API_URL = 'http://your-server:8000';</script>
<script>window.AI_CHAT_API_KEY = 'zk_xxx';</script>
<script src="http://your-server:8000/static/widget.js"></script>
```

## License

MIT
