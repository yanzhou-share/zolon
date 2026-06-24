"""
数据模型定义

Pydantic 模型和 TypedDict 定义。
"""

from typing import TypedDict, List, Optional
from pydantic import BaseModel


class AgentState(TypedDict):
    """Agent 状态"""
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
    """聊天请求"""
    session_id: str
    message: str
    chat_history: List[dict] = []


class ChatResponse(BaseModel):
    """聊天响应"""
    reply: str
    status: str
    sentiment_score: str
    sentiment_label: str
    sentiment_confidence: float
    intent: str
    retrieved_knowledge: str


class BatchEvalRequest(BaseModel):
    """批量评估请求"""
    test_cases: list[dict]
