"""Evaluation package for MORPHE."""

from .metrics import (
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    evaluate_retrieval_batch,
    EvaluationMetrics,
)
from .dataset import BenchmarkDataset, get_standard_benchmark
from .evaluator import SystemEvaluator

__all__ = [
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "evaluate_retrieval_batch",
    "EvaluationMetrics",
    "BenchmarkDataset",
    "get_standard_benchmark",
    "SystemEvaluator",
]

