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
sessions/     - 会话数据存储 (JSON)
chroma_db/    - ChromaDB 持久化存储 (向量数据库)
uploads/      - 上传的文件存储
hf_cache/     - BGE 模型缓存
.env          - 环境变量 (DEEPSEEK_API_KEY, HF_ENDPOINT)
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
