# 智能客服 SaaS 平台 — 项目简历描述

## 项目概述

基于 RAG（检索增强生成）架构的多商户智能客服 SaaS 平台，支持企业快速部署 AI 客服系统。采用 LangGraph 构建 6 节点 AI Agent 流水线，实现知识库问答、情感分析、意图识别、人工转接等能力。

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **后端框架** | FastAPI 0.104 | 异步 Web 框架，支持 async/await |
| **AI Agent** | LangGraph | 6 节点状态图，支持条件路由和并行执行 |
| **LLM** | DeepSeek API | 通过 OpenAI SDK 兼容接口调用 |
| **向量数据库** | ChromaDB 0.4.22 | 嵌入式向量数据库，HNSW 索引（cosine） |
| **Embedding** | BGE-small-zh | Sentence Transformers，512 维向量 |
| **前端** | Streamlit 1.29 | 管理后台 + 仪表盘 |
| **数据库** | SQLite | 用量统计、计费、告警 |
| **流式响应** | SSE (sse-starlette) | Server-Sent Events 逐 token 推送 |
| **文档解析** | python-docx + pdfplumber | Word/PDF 解析，表格转 Markdown |

## 核心功能

### 1. RAG 知识库问答

**递归分块算法：**
- 按分隔符优先级递归分割：`\n\n` → `\n` → `。！？` → `，` → 空格 → 字符
- 保证句子完整性，避免在句子中间切断
- 支持 overlap（默认 50 字符）保持上下文连贯

**语义分块（可选）：**
- 基于 BGE embedding 计算相邻句子余弦相似度
- 相似度骤降处（< 0.5 阈值）识别为语义断点
- 在断点处分割，保证每个 chunk 语义完整

**多阶段检索：**
- 阶段 1：ChromaDB 向量搜索 TOP20，过滤 `status=active` 的 chunks
- 阶段 2：LLM Rerank 精排 TOP5，基于查询相关性排序
- 查询改写：LLM 自动优化搜索查询，提升检索质量

**版本管理：**
- 同名文件上传自动创建新版本
- 旧版本 chunks 标记为 `deprecated`，新版本标记为 `active`
- 支持版本回滚，自动同步 ChromaDB metadata

### 2. 多租户 SaaS 架构

- API Key 认证 + 域名白名单验证
- 租户级知识库隔离（独立 ChromaDB Collection: `knowledge_{api_key}`）
- 多商户管理（添加/编辑/删除）
- 请求限流 + 用量计量

### 3. AI Agent 流水线（LangGraph 6 节点）

```
用户输入 → 消息合并 → [情感分析, 查询优化] → 意图路由
                                              ↓
                                    愤怒 → 人工转接
                                    正常 → 知识检索 → 响应生成 → 输出
```

- **情感分析**：关键词优先（侮辱词/愤怒词）+ LLM 兜底，区分产品反馈和愤怒
- **意图识别**：价格查询/功能咨询/对比/售后/问候
- **查询优化**：自动改写 + 简单查询跳过（< 10 字符）
- **知识检索**：两阶段检索 + 查询改写
- **响应生成**：流式输出 + 幻觉控制（强化 prompt 约束）
- **人工转接**：愤怒情绪自动触发

### 4. 长对话污染治理

- **消息合并**：5 秒窗口内多条消息自动合并
- **话题转换检测**：LLM 判断当前问题是否切换话题
- **对话摘要压缩**：超限时 LLM 生成摘要 + 保留最近 N 条
- **滑动窗口**：Token 预算管理，system prompt 150 + response 800 + knowledge + history

### 5. 运营仪表盘

- 实时用量统计（请求量/Token/延迟）
- 每日趋势图表
- 商户排行
- 月度计费账单（每千 Token 计价）

### 6. 评估系统（完整闭环）

**三维评估体系：**

| 维度 | 指标 | 说明 |
|------|------|------|
| 检索质量 | Recall@K, Precision@K, MRR, NDCG, Hit Rate | 向量检索效果 |
| 生成质量 | Faithfulness, Answer Relevancy, Context Relevancy, Hallucination | LLM-as-Judge |
| 端到端 | Correctness, Response Quality, Satisfaction | 整体效果 |

**自动化流程：**
- 批量评估：35 条测试用例，覆盖 6 个文档
- 趋势追踪：SQLite 存储每日指标，支持按商户筛选
- 自动告警：忠实度 < 0.5、幻觉率 > 0.3、延迟 > 10s
- 仪表盘展示：4 组 Tab 图表（检索/生成/端到端/性能）

### 7. 嵌入式 Widget

- 一行代码集成到任意网站
- 流式逐字渲染
- 多轮对话历史
- 移动端适配
- 商户可自定义助手名称

## 项目亮点

1. **RAG 优化**：递归分块保证语义完整性，多阶段检索（向量+LLM Rerank）提升命中率
2. **延迟优化**：并行化 LLM 调用（情感+查询优化并行）、简单查询跳过改写、关键词情感优先
3. **幻觉控制**：强化 prompt 约束、temperature 0.3、faithfulness 评估闭环
4. **SaaS 架构**：多租户隔离、API Key 认证、域名白名单、限流计费
5. **评估闭环**：检索+生成+端到端三维评估，趋势追踪+自动告警，持续监控质量

## 代码规模

- **后端**: ~1500 行（backend.py + database.py + eval/*.py）
- **前端**: ~550 行（app.py + widget.js）
- **测试**: 75 个单元测试（test_backend 35 + test_eval 40）
- **API 端点**: 17 个
- **评估测试集**: 35 条用例，覆盖 6 个文档

## 适用场景

- 企业产品客服（电商/SaaS/硬件）
- 内部知识库问答系统
- 多商户客服 SaaS 平台
