# AGENTS.md

## Rules

- 产品文档是唯一的真理来源，编码前先分析文档
- 实现功能后必须写测试用例
- 重启服务时不需要跑测试用例，用户会手动验证
- 代码修改后主动重启服务让用户测试
- 遇到问题时提供截图反馈，直接根据截图定位问题并修复
- **禁止将 .env 文件提交到代码仓库**，.env 包含 API Key 等敏感信息

## 技术栈

- Python + FastAPI (后端)
- Streamlit (前端)
- LangGraph (AI Agent 图状态机)
- ChromaDB (向量数据库，持久化存储)
- DeepSeek LLM (可选，无 API Key 时降级到关键词匹配)
- BGE-small-zh (Embedding 模型，后台异步加载)
- SSE 流式响应
- python-docx (Word 文件解析)

## 项目结构

```
backend.py    - FastAPI 后端 + LangGraph 6 节点 + 文件上传 API
app.py        - Streamlit 前端 + 会话持久化 + 文件上传 UI
test_backend.py - pytest 测试用例
eval_tracer.py - 观测层: 请求追踪、节点计时、Token统计
eval_retrieval.py - 检索评估: Recall@K, Precision@K, MRR, Hit Rate
eval_generation.py - 生成评估: LLM-as-Judge (faithfulness, relevancy, hallucination)
eval_e2e.py    - 端到端评估: correctness, quality, satisfaction
eval_pipeline.py - 评估流水线: 自动评估、批量评估、指标导出
eval_dataset.json - 评估测试数据集 (10条样本)
test_eval.py   - 评估系统测试 (40个用例)
eval_metrics/  - 评估指标输出目录 (JSONL日志 + 批量报告)
sessions/      - 会话数据存储 (JSON)
chroma_db/     - ChromaDB 持久化存储 (向量数据库)
uploads/       - 上传的文件存储
hf_cache/      - BGE 模型缓存
.env           - 环境变量 (DEEPSEEK_API_KEY, HF_ENDPOINT)
产品文档.txt   - 产品需求文档
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/chat` | POST | 非流式聊天 |
| `/chat/stream` | POST | SSE 流式聊天 |
| `/upload` | POST | 上传文件 (.txt/.docx) |
| `/upload/list` | GET | 列出已上传文件 |
| `/upload/{filename}` | DELETE | 删除指定文件 |
| `/health` | GET | 健康检查 |
| `/eval/stats` | GET | 获取评估统计指标 |
| `/eval/batch` | POST | 批量评估测试用例 |
| `/eval/dataset` | GET | 获取评估数据集 |
| `/eval/run-dataset` | POST | 运行完整数据集评估 |

## 开发流程

1. 分析产品文档，理解需求
2. 修改代码
3. 运行测试: `python -m pytest test_backend.py -v`
4. 重启服务让用户手动测试
5. 根据反馈修复问题

## 服务重启命令

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Process python -ArgumentList "backend.py" -WorkingDirectory "D:\xiaomi-mimo-test\zolon" -WindowStyle Hidden
Start-Process python -ArgumentList "-m", "streamlit", "run", "app.py" -WorkingDirectory "D:\xiaomi-mimo-test\zolon" -WindowStyle Hidden
```

## 注意事项

- BGE 模型首次加载需要从 HuggingFace 下载（约 183MB），后续启动从缓存加载
- 知识库为空时，系统会提示用户先上传文件
- 流式响应通过 SSE 实现，前端逐 token 渲染
