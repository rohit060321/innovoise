"""Evaluation harness to measure retrieval quality, latency, and co-access."""

from typing import Any, Dict, List, Optional, Tuple

from ..graph.memory_graph import MemoryGraph
from ..retrieval.router import RoutedRouter, RetrievalResult
from ..storage.db import SQLiteStorage
from .dataset import BenchmarkQuery
from .metrics import EvaluationMetrics, evaluate_retrieval_batch


class SystemEvaluator:
    """Evaluates the routed memory system against benchmark query suites."""

    def __init__(
        self,
        router: RoutedRouter,
        storage: Optional[SQLiteStorage] = None,
    ) -> None:
        self.router = router
        self.storage = storage

    def evaluate(
        self,
        queries: List[BenchmarkQuery],
        top_k: int = 5,
        log_coaccess: bool = True,
    ) -> Tuple[EvaluationMetrics, List[RetrievalResult]]:
        results: List[RetrievalResult] = []
        ground_truths: List[List[str]] = []

        for bq in queries:
            res = self.router.retrieve(bq.text, top_k_records=top_k)
            results.append(res)
            ground_truths.append(bq.ground_truth_doc_ids)

            if log_coaccess and self.storage and len(res.queried_modules) > 1:
                self.storage.log_coaccess(bq.text, res.queried_modules)

        graph = self.router.graph
        mod_count = len(graph.modules)
        edge_count = graph.graph.number_of_edges()
        shortcut_count = len(graph.get_shortcuts())

        metrics = evaluate_retrieval_batch(
            results=results,
            ground_truths=ground_truths,
            module_count=mod_count,
            edge_count=edge_count,
            shortcut_count=shortcut_count,
        )

        return metrics, results

