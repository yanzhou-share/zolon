"""Retrieval evaluation metrics: Recall@K, Precision@K, MRR, Hit Rate, relevance threshold."""

import json
import logging
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("eval.retrieval")


@dataclass
class RetrievalMetrics:
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    hit_rate: float = 0.0
    avg_distance: float = 0.0
    min_distance: float = 1.0
    max_distance: float = 0.0
    num_results: int = 0
    num_relevant: int = 0
    k: int = 5

    def to_dict(self) -> dict:
        return asdict(self)


def compute_recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k results."""
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    relevant_in_top = top_k & relevant_ids
    return round(len(relevant_in_top) / len(relevant_ids), 4)


def compute_precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_in_top = top_k & relevant_ids
    return round(len(relevant_in_top) / k, 4)


def compute_mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return round(1.0 / (i + 1), 4)
    return 0.0


def compute_ndcg(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    import math
    # Ideal DCG: all relevant docs at top positions
    ideal_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant))
    if idcg == 0:
        return 0.0
    # Actual DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)
    return round(dcg / idcg, 4)


def compute_hit_rate(retrieved_distances: list[float], threshold: float = 0.5) -> float:
    """1.0 if any result is below the distance threshold, else 0.0. Cosine distance: lower = more relevant."""
    if not retrieved_distances:
        return 0.0
    return 1.0 if min(retrieved_distances) <= threshold else 0.0


def evaluate_retrieval_single(
    retrieved_ids: list[str],
    retrieved_distances: list[float],
    relevant_ids: set[str],
    k: int = 5,
    distance_threshold: float = 0.5,
) -> RetrievalMetrics:
    """Evaluate a single query's retrieval results."""
    m = RetrievalMetrics(k=k, num_results=len(retrieved_ids))
    m.num_relevant = len(relevant_ids)

    # 计算召回率@K：检索结果中相关文档占所有相关文档的比例
    m.recall_at_k = compute_recall_at_k(retrieved_ids, relevant_ids, k)
    # 计算精确率@K：检索结果中相关文档占前K个结果的比例
    m.precision_at_k = compute_precision_at_k(retrieved_ids, relevant_ids, k)
    # 计算平均倒数排名：第一个相关文档排名的倒数
    m.mrr = compute_mrr(retrieved_ids, relevant_ids)
    # 计算归一化折损累计增益：考虑排名位置的加权相关性评分
    m.ndcg = compute_ndcg(retrieved_ids, relevant_ids, k)
    # 计算命中率：是否有检索结果的距离低于阈值
    m.hit_rate = compute_hit_rate(retrieved_distances, distance_threshold)

    if retrieved_distances:
        m.avg_distance = round(sum(retrieved_distances) / len(retrieved_distances), 4)
        m.min_distance = round(min(retrieved_distances), 4)
        m.max_distance = round(max(retrieved_distances), 4)

    return m


@dataclass
class BatchRetrievalReport:
    total_queries: int = 0
    avg_recall_at_k: float = 0.0
    avg_precision_at_k: float = 0.0
    avg_mrr: float = 0.0
    avg_ndcg: float = 0.0
    hit_rate: float = 0.0
    avg_distance: float = 0.0
    per_query: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def evaluate_retrieval_batch(query_results: list[dict], k: int = 5, distance_threshold: float = 0.5) -> BatchRetrievalReport:
    """Evaluate retrieval for a batch of queries.

    Each item in query_results:
        {
            "retrieved_ids": ["id1", "id2", ...],
            "retrieved_distances": [0.1, 0.2, ...],
            "relevant_ids": ["id1", "id3", ...]
        }
    """
    report = BatchRetrievalReport(total_queries=len(query_results))
    if not query_results:
        return report

    total_recall = 0.0
    total_precision = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0
    hits = 0
    all_distances = []

    for qr in query_results:
        metrics = evaluate_retrieval_single(
            retrieved_ids=qr["retrieved_ids"],
            retrieved_distances=qr.get("retrieved_distances", []),
            relevant_ids=set(qr["relevant_ids"]),
            k=k,
            distance_threshold=distance_threshold,
        )
        total_recall += metrics.recall_at_k
        total_precision += metrics.precision_at_k
        total_mrr += metrics.mrr
        total_ndcg += metrics.ndcg
        hits += metrics.hit_rate
        all_distances.extend(qr.get("retrieved_distances", []))
        report.per_query.append(metrics.to_dict())

    n = len(query_results)
    report.avg_recall_at_k = round(total_recall / n, 4)
    report.avg_precision_at_k = round(total_precision / n, 4)
    report.avg_mrr = round(total_mrr / n, 4)
    report.avg_ndcg = round(total_ndcg / n, 4)
    report.hit_rate = round(hits / n, 4)
    report.avg_distance = round(sum(all_distances) / len(all_distances), 4) if all_distances else 0.0

    logger.info(json.dumps(report.to_dict(), ensure_ascii=False))
    return report
