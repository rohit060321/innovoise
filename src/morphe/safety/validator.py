"""Safe graph-copy trial sandbox and metric-based validation gate."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time

from ..adaptation.operations import AdaptationOperation, apply_adaptation
from ..config import ThresholdsConfig
from ..embedding.embedder import BaseEmbedder
from ..evaluation.dataset import BenchmarkQuery
from ..evaluation.evaluator import SystemEvaluator
from ..evaluation.metrics import EvaluationMetrics
from ..graph.memory_graph import MemoryGraph
from ..retrieval.router import RoutedRouter


@dataclass
class ValidationResult:
    accepted: bool
    decision: str  # "ACCEPTED" or "REJECTED"
    reason: str
    operation_type: str
    target_modules: List[str]
    delta_recall: float
    delta_mrr: float
    delta_latency: float
    pre_metrics: EvaluationMetrics
    post_metrics: EvaluationMetrics
    trial_graph: Optional[MemoryGraph]
    record_reassignments: Dict[str, str]


class SandboxValidator:
    """
    Evaluates proposed topological adaptations inside an isolated sandbox graph copy.
    Guarantees that production memory is never corrupted by regressions.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        thresholds: ThresholdsConfig,
        top_k_centroids: int = 2,
        expand_shortcuts: bool = True,
    ) -> None:
        self.embedder = embedder
        self.thresholds = thresholds
        self.top_k_centroids = top_k_centroids
        self.expand_shortcuts = expand_shortcuts

    def validate_adaptation(
        self,
        prod_graph: MemoryGraph,
        operation: AdaptationOperation,
        tuning_queries: List[BenchmarkQuery],
        seed: int = 42,
    ) -> ValidationResult:
        # Step 1: Pre-evaluation on production graph
        prod_router = RoutedRouter(
            graph=prod_graph,
            embedder=self.embedder,
            top_k_centroids=self.top_k_centroids,
            expand_shortcuts=self.expand_shortcuts,
        )
        prod_evaluator = SystemEvaluator(router=prod_router)
        pre_metrics, _ = prod_evaluator.evaluate(queries=tuning_queries, log_coaccess=False)

        # Step 2: Create isolated sandbox copy of graph
        trial_graph = prod_graph.clone()

        # Step 3: Apply candidate adaptation to sandbox
        adapt_res = apply_adaptation(operation=operation, graph=trial_graph, seed=seed)

        # Step 4: Post-evaluation on trial sandbox graph
        trial_router = RoutedRouter(
            graph=trial_graph,
            embedder=self.embedder,
            top_k_centroids=self.top_k_centroids,
            expand_shortcuts=self.expand_shortcuts,
        )
        trial_evaluator = SystemEvaluator(router=trial_router)
        post_metrics, _ = trial_evaluator.evaluate(queries=tuning_queries, log_coaccess=False)

        # Step 5: Compute metric deltas
        # Focus on Recall@3 and MRR
        delta_recall = post_metrics.recall_at_3 - pre_metrics.recall_at_3
        delta_mrr = post_metrics.mrr - pre_metrics.mrr
        delta_latency = post_metrics.avg_latency_ms - pre_metrics.avg_latency_ms
        latency_ratio = post_metrics.avg_latency_ms / max(0.001, pre_metrics.avg_latency_ms)

        # Step 6: Acceptance Gate Logic
        accepted = True
        rejection_reasons = []

        if delta_recall < self.thresholds.min_delta_recall:
            accepted = False
            rejection_reasons.append(
                f"Recall regression: delta_recall={delta_recall:+.4f} < min_threshold={self.thresholds.min_delta_recall:+.4f}"
            )

        # Only flag latency surge if ratio is breached AND the absolute latency increase is meaningful (>1.5ms)
        if latency_ratio > self.thresholds.max_delta_latency_ratio and delta_latency > 1.5:
            accepted = False
            rejection_reasons.append(
                f"Latency surge: post/pre ratio={latency_ratio:.2f} > max_ratio={self.thresholds.max_delta_latency_ratio:.2f} (+{delta_latency:.2f}ms)"
            )

        if accepted:
            decision = "ACCEPTED"
            reason = f"Validated improvement: delta_recall={delta_recall:+.4f}, delta_mrr={delta_mrr:+.4f}, delta_latency={delta_latency:+.2f}ms"
            committed_trial_graph = trial_graph
        else:
            decision = "REJECTED"
            reason = "; ".join(rejection_reasons)
            committed_trial_graph = None

        return ValidationResult(
            accepted=accepted,
            decision=decision,
            reason=reason,
            operation_type=adapt_res.operation_type,
            target_modules=adapt_res.target_modules,
            delta_recall=round(delta_recall, 4),
            delta_mrr=round(delta_mrr, 4),
            delta_latency=round(delta_latency, 2),
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            trial_graph=committed_trial_graph,
            record_reassignments=adapt_res.record_reassignments if accepted else {},
        )

