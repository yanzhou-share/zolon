"""Evaluation pipeline: auto-evaluation per request, batch evaluation, metrics export."""

import json
import time
import logging
import os
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

from eval_tracer import TraceContext, RequestTrace
from eval_retrieval import RetrievalMetrics, evaluate_retrieval_single
from eval_generation import GenerationMetrics, evaluate_generation
from eval_e2e import E2EMetrics, evaluate_e2e

logger = logging.getLogger("eval.pipeline")

METRICS_DIR = os.path.join(os.path.dirname(__file__), "eval_metrics")
os.makedirs(METRICS_DIR, exist_ok=True)


@dataclass
class SingleEvalResult:
    trace_id: str
    timestamp: str
    query: str
    answer: str
    context: str
    retrieval: Optional[dict] = None
    generation: Optional[dict] = None
    e2e: Optional[dict] = None
    latency_ms: float = 0.0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchEvalReport:
    batch_id: str
    timestamp: str
    total_queries: int = 0
    avg_retrieval_score: float = 0.0
    avg_generation_score: float = 0.0
    avg_e2e_score: float = 0.0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    per_query_results: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


async def auto_evaluate_request(
    client,
    trace: RequestTrace,
    query: str,
    answer: str,
    context: str,
    retrieved_distances: list[float] = None,
    ground_truth: Optional[str] = None,
    skip_llm_judge: bool = False,
) -> SingleEvalResult:
    """Lightweight auto-evaluation called on every request.

    Evaluates retrieval (distance-based, no LLM needed) and optionally generation quality.
    Set skip_llm_judge=True for low-overhead per-request eval.
    """
    start = time.time()
    eval_start_time = datetime.now().isoformat()

    result = SingleEvalResult(
        trace_id=trace.trace_id,
        timestamp=eval_start_time,
        query=query,
        answer=answer,
        context=context[:500],
        latency_ms=trace.total_duration_ms,
        total_tokens=trace.total_llm_tokens_in + trace.total_llm_tokens_out,
    )

    if retrieved_distances and context:
        result.retrieval = RetrievalMetrics(
            avg_distance=round(sum(retrieved_distances) / len(retrieved_distances), 4) if retrieved_distances else 0,
            min_distance=round(min(retrieved_distances), 4) if retrieved_distances else 1,
            num_results=len(retrieved_distances),
            hit_rate=1.0 if min(retrieved_distances) <= 0.5 else 0.0 if retrieved_distances else 0,
        ).to_dict()

    if not skip_llm_judge and client and answer and context:
        try:
            gen_metrics = await evaluate_generation(client, query, context, answer)
            result.generation = gen_metrics.to_dict()
        except Exception as e:
            logger.warning(f"Generation eval failed: {e}")

        try:
            e2e_metrics = await evaluate_e2e(client, query, answer, ground_truth)
            result.e2e = e2e_metrics.to_dict()
        except Exception as e:
            logger.warning(f"E2E eval failed: {e}")

    result.latency_ms = round((time.time() - start) * 1000 + trace.total_duration_ms, 2)

    _append_eval_log(result)
    return result


async def batch_evaluate(
    client,
    test_cases: list[dict],
    knowledge_collection=None,
    k: int = 5,
) -> BatchEvalReport:
    """Run batch evaluation on a test dataset.

    test_cases: [{"query": str, "expected_answer": str, "relevant_ids": [str], "retrieved_ids": [str], "context": str}]
    """
    import hashlib
    batch_id = hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8]
    report = BatchEvalReport(
        batch_id=batch_id,
        timestamp=datetime.now().isoformat(),
        total_queries=len(test_cases),
    )

    total_retrieval = 0.0
    total_generation = 0.0
    total_e2e = 0.0
    total_latency = 0.0
    total_tok = 0

    for i, tc in enumerate(test_cases):
        query = tc["query"]
        expected = tc.get("expected_answer", "")
        context = tc.get("context", "")

        # 如果没有 context，从知识库查询
        if not context and knowledge_collection and knowledge_collection.count() > 0:
            try:
                results = knowledge_collection.query(query_texts=[query], n_results=3)
                if results and results["documents"] and results["documents"][0]:
                    context = "\n".join(results["documents"][0])
            except Exception as e:
                logger.warning(f"Knowledge query failed for eval: {e}")

        retrieval_metrics = None
        if tc.get("retrieved_ids") and tc.get("relevant_ids"):
            rm = evaluate_retrieval_single(
                retrieved_ids=tc["retrieved_ids"],
                retrieved_distances=tc.get("retrieved_distances", []),
                relevant_ids=set(tc["relevant_ids"]),
                k=k,
            )
            retrieval_metrics = rm.to_dict()
            total_retrieval += rm.recall_at_k

        gen_metrics = None
        if context and client:
            try:
                gm = await evaluate_generation(client, query, context, expected or "", expected)
                gen_metrics = gm.to_dict()
                total_generation += gm.faithfulness
            except Exception as e:
                logger.warning(f"Batch gen eval query {i} failed: {e}")

        e2e_metrics = None
        if expected and client:
            try:
                em = await evaluate_e2e(client, query, expected, expected)
                e2e_metrics = em.to_dict()
                total_e2e += em.overall_score
            except Exception as e:
                logger.warning(f"Batch e2e eval query {i} failed: {e}")

        report.per_query_results.append(SingleEvalResult(
            trace_id=f"batch_{batch_id}_{i}",
            timestamp=datetime.now().isoformat(),
            query=query,
            answer=expected,
            context=context[:500],
            retrieval=retrieval_metrics,
            generation=gen_metrics,
            e2e=e2e_metrics,
        ).to_dict())

    n = max(len(test_cases), 1)
    report.avg_retrieval_score = round(total_retrieval / n, 4)
    report.avg_generation_score = round(total_generation / n, 4)
    report.avg_e2e_score = round(total_e2e / n, 4)
    report.total_tokens = total_tok

    report_path = os.path.join(METRICS_DIR, f"batch_{batch_id}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info(f"Batch eval report saved: {report_path}")

    return report


def _append_eval_log(result: SingleEvalResult):
    """Append single eval result to the daily log file."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(METRICS_DIR, f"eval_{today}.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def get_eval_stats(days: int = 7) -> dict:
    """Read recent eval logs and compute aggregate stats."""
    import glob as glob_mod
    all_results = []
    for fpath in sorted(glob_mod.glob(os.path.join(METRICS_DIR, "eval_*.jsonl"))):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not all_results:
        return {"total_requests": 0, "message": "No evaluation data yet"}

    total = len(all_results)
    avg_latency = sum(r.get("latency_ms", 0) for r in all_results) / total
    avg_tokens = sum(r.get("total_tokens", 0) for r in all_results) / total

    faith_scores = [r["generation"]["faithfulness"] for r in all_results if r.get("generation")]
    relevancy_scores = [r["generation"]["answer_relevancy"] for r in all_results if r.get("generation")]
    hallucination_scores = [r["generation"]["hallucination_score"] for r in all_results if r.get("generation")]

    return {
        "total_requests": total,
        "avg_latency_ms": round(avg_latency, 2),
        "avg_tokens_per_request": round(avg_tokens, 1),
        "avg_faithfulness": round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else None,
        "avg_answer_relevancy": round(sum(relevancy_scores) / len(relevancy_scores), 4) if relevancy_scores else None,
        "avg_hallucination": round(sum(hallucination_scores) / len(hallucination_scores), 4) if hallucination_scores else None,
        "generation_eval_count": len(faith_scores),
    }
