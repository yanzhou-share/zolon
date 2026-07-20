"""Observability layer: request tracing, per-node timing, ChromaDB distance scores, token counting, structured JSON logging."""

import json
import time
import uuid
import logging
from typing import Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("eval.tracer")


@dataclass
class NodeTrace:
    node_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    llm_model: str = ""
    retrieval_distances: list = field(default_factory=list)
    retrieval_count: int = 0
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)


@dataclass
class RequestTrace:
    trace_id: str
    session_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    total_duration_ms: float = 0.0
    nodes: list = field(default_factory=list)
    total_llm_tokens_in: int = 0
    total_llm_tokens_out: int = 0
    total_llm_calls: int = 0
    intent: str = ""
    sentiment_label: str = ""
    status: str = ""
    has_retrieval: bool = False
    retrieval_doc_count: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["nodes"] = [asdict(n) for n in self.nodes]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class TraceContext:
    """Per-request tracing context. Created at request start, attached to AgentState."""

    def __init__(self, session_id: str, trace_id: Optional[str] = None):
        self.trace = RequestTrace(
            trace_id=trace_id or str(uuid.uuid4())[:12],
            session_id=session_id,
            start_time=time.time(),
        )
        self._node_map: dict[str, NodeTrace] = {}

    def start_node(self, node_name: str) -> NodeTrace:
        nt = NodeTrace(node_name=node_name, start_time=time.time())
        self._node_map[node_name] = nt
        return nt

    def end_node(self, node_name: str, error: Optional[str] = None):
        nt = self._node_map.get(node_name)
        if nt:
            nt.end_time = time.time()
            nt.duration_ms = round((nt.end_time - nt.start_time) * 1000, 2)
            nt.error = error

    def record_llm_call(self, node_name: str, model: str, tokens_in: int, tokens_out: int):
        nt = self._node_map.get(node_name)
        if nt:
            nt.llm_model = model
            nt.llm_tokens_in = tokens_in
            nt.llm_tokens_out = tokens_out

    def record_retrieval(self, node_name: str, distances: list, count: int):
        nt = self._node_map.get(node_name)
        if nt:
            nt.retrieval_distances = distances
            nt.retrieval_count = count

    def finish(self) -> RequestTrace:
        self.trace.end_time = time.time()
        self.trace.total_duration_ms = round(
            (self.trace.end_time - self.trace.start_time) * 1000, 2
        )
        self.trace.nodes = list(self._node_map.values())
        for nt in self.trace.nodes:
            self.trace.total_llm_tokens_in += nt.llm_tokens_in
            self.trace.total_llm_tokens_out += nt.llm_tokens_out
            if nt.llm_model:
                self.trace.total_llm_calls += 1
            if nt.node_name == "knowledge_retriever":
                self.trace.has_retrieval = nt.retrieval_count > 0
                self.trace.retrieval_doc_count = nt.retrieval_count
        return self.trace

    def log_summary(self):
        t = self.trace
        summary = {
            "trace_id": t.trace_id,
            "session_id": t.session_id,
            "total_ms": t.total_duration_ms,
            "llm_calls": t.total_llm_calls,
            "tokens_in": t.total_llm_tokens_in,
            "tokens_out": t.total_llm_tokens_out,
            "intent": t.intent,
            "sentiment": t.sentiment_label,
            "status": t.status,
            "retrieval_docs": t.retrieval_doc_count,
            "node_breakdown": [
                {"name": n.node_name, "ms": n.duration_ms, "tokens_in": n.llm_tokens_in, "tokens_out": n.llm_tokens_out}
                for n in t.nodes
            ],
        }
        logger.info(json.dumps(summary, ensure_ascii=False))


def extract_token_counts(usage) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from OpenAI usage object."""
    if usage is None:
        return 0, 0
    return getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0


def extract_distances(query_results) -> list[float]:
    """Extract cosine distances from ChromaDB query results."""
    if not query_results or not query_results.get("distances"):
        return []
    return query_results["distances"][0] if query_results["distances"] else []
