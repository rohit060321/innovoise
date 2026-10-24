"""Metrics suite for retrieval quality, latency, utilization, and topology."""

from dataclasses import dataclass
from typing import Any, Dict, List, Set
import numpy as np


@dataclass
class EvaluationMetrics:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    avg_latency_ms: float
    p95_latency_ms: float
    queried_modules_per_query: float
    module_count: int
    edge_count: int
    shortcut_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall_at_1": round(self.recall_at_1, 4),
            "recall_at_3": round(self.recall_at_3, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "queried_modules_per_query": round(self.queried_modules_per_query, 2),
            "module_count": self.module_count,
            "edge_count": self.edge_count,
            "shortcut_count": self.shortcut_count,
        }


def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str] | List[str], k: int) -> float:
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k.intersection(gt_set))
    return hits / float(len(gt_set))


def mean_reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: Set[str] | List[str]) -> float:
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gt_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str] | List[str], k: int) -> float:
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if doc_id in gt_set else 0.0
        dcg += rel / np.log2(i + 1)

    # Ideal DCG
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gt_set), k) + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_retrieval_batch(
    results: List[Any],  # List[RetrievalResult]
    ground_truths: List[List[str]],
    module_count: int = 0,
    edge_count: int = 0,
    shortcut_count: int = 0,
) -> EvaluationMetrics:
    if not results:
        return EvaluationMetrics(
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            mrr=0.0,
            ndcg_at_5=0.0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            queried_modules_per_query=0.0,
            module_count=module_count,
            edge_count=edge_count,
            shortcut_count=shortcut_count,
        )

    r1_list, r3_list, r5_list = [], [], []
    mrr_list, ndcg_list = [], []
    latencies = []
    queried_counts = []

    for res, gt in zip(results, ground_truths):
        retrieved_ids = [r["id"] for r in res.records]
        r1_list.append(recall_at_k(retrieved_ids, gt, k=1))
        r3_list.append(recall_at_k(retrieved_ids, gt, k=3))
        r5_list.append(recall_at_k(retrieved_ids, gt, k=5))
        mrr_list.append(mean_reciprocal_rank(retrieved_ids, gt))
        ndcg_list.append(ndcg_at_k(retrieved_ids, gt, k=5))
        latencies.append(res.latency_ms)
        queried_counts.append(len(res.queried_modules))

    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

    return EvaluationMetrics(
        recall_at_1=float(np.mean(r1_list)),
        recall_at_3=float(np.mean(r3_list)),
        recall_at_5=float(np.mean(r5_list)),
        mrr=float(np.mean(mrr_list)),
        ndcg_at_5=float(np.mean(ndcg_list)),
        avg_latency_ms=float(np.mean(latencies)),
        p95_latency_ms=p95,
        queried_modules_per_query=float(np.mean(queried_counts)),
        module_count=module_count,
        edge_count=edge_count,
        shortcut_count=shortcut_count,
    )

