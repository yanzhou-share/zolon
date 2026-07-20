"""Tests for the RAG evaluation system."""

import pytest
import json
import time
import os
from unittest.mock import AsyncMock, MagicMock, patch
from eval.eval_tracer import TraceContext, RequestTrace, NodeTrace, extract_token_counts, extract_distances
from eval.eval_retrieval import (
    RetrievalMetrics, BatchRetrievalReport,
    compute_recall_at_k, compute_precision_at_k, compute_mrr, compute_hit_rate,
    evaluate_retrieval_single, evaluate_retrieval_batch,
)
from eval.eval_generation import GenerationMetrics, evaluate_generation
from eval.eval_e2e import E2EMetrics, evaluate_e2e
from eval.eval_pipeline import SingleEvalResult, BatchEvalReport, auto_evaluate_request, get_eval_stats


# ─── TraceContext Tests ───

class TestTraceContext:
    def test_trace_creation(self):
        ctx = TraceContext("session1")
        assert ctx.trace.session_id == "session1"
        assert len(ctx.trace.trace_id) == 12
        assert ctx.trace.start_time > 0

    def test_trace_custom_id(self):
        ctx = TraceContext("s", trace_id="custom-id")
        assert ctx.trace.trace_id == "custom-id"

    def test_start_end_node(self):
        ctx = TraceContext("s")
        nt = ctx.start_node("test_node")
        assert nt.node_name == "test_node"
        assert nt.start_time > 0
        time.sleep(0.01)
        ctx.end_node("test_node")
        assert nt.duration_ms > 0
        assert nt.error is None

    def test_end_node_with_error(self):
        ctx = TraceContext("s")
        ctx.start_node("n1")
        ctx.end_node("n1", error="something broke")
        t = ctx.finish()
        assert t.nodes[0].error == "something broke"

    def test_record_llm_call(self):
        ctx = TraceContext("s")
        ctx.start_node("n1")
        ctx.record_llm_call("n1", "deepseek-chat", 100, 50)
        ctx.end_node("n1")
        t = ctx.finish()
        assert t.total_llm_tokens_in == 100
        assert t.total_llm_tokens_out == 50
        assert t.total_llm_calls == 1

    def test_record_retrieval(self):
        ctx = TraceContext("s")
        ctx.start_node("knowledge_retriever")
        ctx.record_retrieval("knowledge_retriever", [0.1, 0.2, 0.3], 3)
        ctx.end_node("knowledge_retriever")
        t = ctx.finish()
        assert t.has_retrieval is True
        assert t.retrieval_doc_count == 3

    def test_finish_calculates_totals(self):
        ctx = TraceContext("s")
        ctx.start_node("a")
        ctx.record_llm_call("a", "model", 10, 20)
        ctx.end_node("a")
        ctx.start_node("b")
        ctx.record_llm_call("b", "model", 30, 40)
        ctx.end_node("b")
        t = ctx.finish()
        assert t.total_llm_tokens_in == 40
        assert t.total_llm_tokens_out == 60
        assert t.total_llm_calls == 2
        assert t.total_duration_ms >= 0

    def test_to_dict(self):
        ctx = TraceContext("s")
        ctx.start_node("n1")
        ctx.end_node("n1")
        t = ctx.finish()
        d = t.to_dict()
        assert "trace_id" in d
        assert "nodes" in d
        assert len(d["nodes"]) == 1

    def test_to_json(self):
        ctx = TraceContext("s")
        ctx.start_node("n1")
        ctx.end_node("n1")
        t = ctx.finish()
        j = t.to_json()
        parsed = json.loads(j)
        assert parsed["trace_id"] == t.trace_id


class TestExtractHelpers:
    def test_extract_token_counts_with_usage(self):
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        assert extract_token_counts(usage) == (100, 50)

    def test_extract_token_counts_none(self):
        assert extract_token_counts(None) == (0, 0)

    def test_extract_distances(self):
        results = {"distances": [[0.1, 0.2, 0.3]]}
        assert extract_distances(results) == [0.1, 0.2, 0.3]

    def test_extract_distances_empty(self):
        assert extract_distances(None) == []
        assert extract_distances({}) == []


# ─── Retrieval Evaluation Tests ───

class TestRetrievalMetrics:
    def test_recall_at_k_perfect(self):
        assert compute_recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_recall_at_k_partial(self):
        assert compute_recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5

    def test_recall_at_k_none_relevant(self):
        assert compute_recall_at_k(["x", "y"], {"a", "b"}, 3) == 0.0

    def test_recall_at_k_no_relevant_ids(self):
        assert compute_recall_at_k(["a", "b"], set(), 3) == 1.0

    def test_precision_at_k(self):
        assert compute_precision_at_k(["a", "b", "c"], {"a", "b"}, 3) == pytest.approx(2/3, abs=0.001)

    def test_precision_at_k_zero_k(self):
        assert compute_precision_at_k([], {"a"}, 0) == 0.0

    def test_mrr_first_relevant(self):
        assert compute_mrr(["a", "b"], {"a"}) == 1.0

    def test_mrr_second_relevant(self):
        assert compute_mrr(["x", "a"], {"a"}) == 0.5

    def test_mrr_no_relevant(self):
        assert compute_mrr(["x", "y"], {"a"}) == 0.0

    def test_hit_rate_below_threshold(self):
        assert compute_hit_rate([0.1, 0.2], threshold=0.5) == 1.0

    def test_hit_rate_above_threshold(self):
        assert compute_hit_rate([0.6, 0.7], threshold=0.5) == 0.0

    def test_hit_rate_empty(self):
        assert compute_hit_rate([], threshold=0.5) == 0.0

    def test_evaluate_single(self):
        m = evaluate_retrieval_single(
            ["a", "b", "c"],
            [0.1, 0.2, 0.6],
            {"a", "b"},
            k=3,
        )
        assert m.recall_at_k == 1.0
        assert m.precision_at_k == round(2/3, 4)
        assert m.mrr == 1.0
        assert m.hit_rate == 1.0
        assert m.avg_distance == round((0.1 + 0.2 + 0.6) / 3, 4)

    def test_evaluate_batch(self):
        data = [
            {"retrieved_ids": ["a", "b"], "retrieved_distances": [0.1, 0.3], "relevant_ids": ["a"]},
            {"retrieved_ids": ["c", "d"], "retrieved_distances": [0.6, 0.7], "relevant_ids": ["a"]},
        ]
        report = evaluate_retrieval_batch(data, k=2)
        assert report.total_queries == 2
        assert report.avg_recall_at_k >= 0
        assert report.hit_rate >= 0

    def test_batch_empty(self):
        report = evaluate_retrieval_batch([])
        assert report.total_queries == 0


# ─── Generation Evaluation Tests (mocked LLM) ───

class TestGenerationEval:
    @pytest.mark.asyncio
    async def test_evaluate_generation_mock(self):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "score": 0.9, "reason": "good"
        })
        mock_client.chat.completions.create.return_value = mock_response

        m = await evaluate_generation(mock_client, "question", "context", "answer")
        assert isinstance(m, GenerationMetrics)
        assert m.faithfulness == 0.9
        assert m.answer_relevancy == 0.9
        assert m.context_relevancy == 0.9

    @pytest.mark.asyncio
    async def test_evaluate_generation_no_client(self):
        m = await evaluate_generation(None, "q", "c", "a")
        assert m.faithfulness == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_generation_llm_error(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        m = await evaluate_generation(mock_client, "q", "c", "a")
        assert m.faithfulness == 0.5


# ─── E2E Evaluation Tests (mocked LLM) ───

class TestE2EEval:
    @pytest.mark.asyncio
    async def test_evaluate_e2e_mock(self):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "score": 0.85, "reason": "good answer"
        })
        mock_client.chat.completions.create.return_value = mock_response

        m = await evaluate_e2e(mock_client, "question", "answer", "ground truth")
        assert isinstance(m, E2EMetrics)
        assert m.response_quality == 0.85
        assert m.correctness == 0.85
        assert m.overall_score > 0

    @pytest.mark.asyncio
    async def test_evaluate_e2e_no_ground_truth(self):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "score": 0.7, "reason": "ok"
        })
        mock_client.chat.completions.create.return_value = mock_response

        m = await evaluate_e2e(mock_client, "q", "a")
        assert m.correctness == 0.0
        assert m.response_quality == 0.7

    @pytest.mark.asyncio
    async def test_evaluate_e2e_no_client(self):
        m = await evaluate_e2e(None, "q", "a", "gt")
        assert m.overall_score > 0


# ─── Pipeline Tests ───

class TestEvalPipeline:
    @pytest.mark.asyncio
    async def test_auto_evaluate_request_no_llm(self):
        ctx = TraceContext("s")
        ctx.start_node("knowledge_retriever")
        ctx.record_retrieval("knowledge_retriever", [0.1, 0.2], 2)
        ctx.end_node("knowledge_retriever")
        trace = ctx.finish()

        result = await auto_evaluate_request(
            client=None,
            trace=trace,
            query="test query",
            answer="test answer",
            context="test context",
            retrieved_distances=[0.1, 0.2],
            skip_llm_judge=True,
        )
        assert isinstance(result, SingleEvalResult)
        assert result.retrieval is not None
        assert result.generation is None

    @pytest.mark.asyncio
    async def test_auto_evaluate_with_retrieval(self):
        ctx = TraceContext("s")
        ctx.start_node("knowledge_retriever")
        ctx.record_retrieval("knowledge_retriever", [0.05, 0.15], 2)
        ctx.end_node("knowledge_retriever")
        trace = ctx.finish()

        result = await auto_evaluate_request(
            client=None,
            trace=trace,
            query="query",
            answer="answer",
            context="context",
            retrieved_distances=[0.05, 0.15],
            skip_llm_judge=True,
        )
        assert result.retrieval["hit_rate"] == 1.0
        assert result.retrieval["min_distance"] == 0.05

    def test_get_eval_stats_empty(self):
        stats = get_eval_stats()
        assert "total_requests" in stats


# ─── Data Schema Tests ───

class TestSchemas:
    def test_retrieval_metrics_fields(self):
        m = RetrievalMetrics()
        d = m.to_dict()
        assert "recall_at_k" in d
        assert "precision_at_k" in d
        assert "mrr" in d
        assert "hit_rate" in d
        assert "avg_distance" in d

    def test_generation_metrics_fields(self):
        m = GenerationMetrics()
        d = m.to_dict()
        assert "faithfulness" in d
        assert "answer_relevancy" in d
        assert "context_relevancy" in d
        assert "hallucination_score" in d

    def test_e2e_metrics_fields(self):
        m = E2EMetrics()
        d = m.to_dict()
        assert "correctness" in d
        assert "response_quality" in d
        assert "satisfaction_prediction" in d
        assert "overall_score" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
