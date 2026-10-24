"""Orchestrator for autonomous adaptation cycles and three-way benchmark comparisons."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import time

from ..adaptation.triggers import detect_adaptation_candidates, AdaptationCandidate
from ..config import MorpheConfig
from ..embedding.embedder import BaseEmbedder
from ..evaluation.dataset import BenchmarkDataset, BenchmarkQuery
from ..evaluation.evaluator import SystemEvaluator
from ..evaluation.metrics import EvaluationMetrics
from ..graph.memory_graph import MemoryGraph
from ..retrieval.router import RoutedRouter
from ..safety.validator import SandboxValidator, ValidationResult
from ..safety.versioning import VersionManager
from ..storage.db import SQLiteStorage
from .bandit import LinUCBPolicy, BanditAction, BanditStateContext


class BenchmarkMode(str, Enum):
    NO_ADAPTATION = "no_adaptation"
    HEURISTIC_ONLY = "heuristic_only"
    RL_GUIDED = "rl_guided"


@dataclass
class StepLog:
    step: int
    action_type: str
    target_modules: List[str]
    decision: str
    reason: str
    delta_recall: float
    delta_latency: float
    reward: Optional[float]
    metrics: Dict[str, Any]


@dataclass
class ExperimentResult:
    mode: BenchmarkMode
    initial_metrics: EvaluationMetrics
    final_metrics_tuning: EvaluationMetrics
    final_metrics_heldout: EvaluationMetrics
    steps_executed: int
    accepted_count: int
    rejected_count: int
    logs: List[StepLog]


class MemoryController:
    """Coordinates autonomous self-adaptation cycles under different operational modes."""

    def __init__(
        self,
        graph: MemoryGraph,
        embedder: BaseEmbedder,
        storage: SQLiteStorage,
        config: MorpheConfig,
        validator: Optional[SandboxValidator] = None,
        version_manager: Optional[VersionManager] = None,
        bandit_policy: Optional[LinUCBPolicy] = None,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.storage = storage
        self.config = config
        self.validator = validator or SandboxValidator(
            embedder=embedder,
            thresholds=config.thresholds,
            top_k_centroids=config.graph.top_k_centroids,
            expand_shortcuts=config.graph.expand_shortcuts,
        )
        self.version_manager = version_manager or VersionManager(storage=storage)
        self.bandit_policy = bandit_policy or LinUCBPolicy(seed=config.seed)

    def run_adaptation_cycle(
        self,
        tuning_queries: List[BenchmarkQuery],
        mode: BenchmarkMode = BenchmarkMode.RL_GUIDED,
        step_index: int = 1,
    ) -> Optional[StepLog]:
        """Execute a single adaptation cycle."""
        router = RoutedRouter(
            graph=self.graph,
            embedder=self.embedder,
            top_k_centroids=self.config.graph.top_k_centroids,
            expand_shortcuts=self.config.graph.expand_shortcuts,
        )
        evaluator = SystemEvaluator(router=router, storage=self.storage)
        current_metrics, _ = evaluator.evaluate(tuning_queries, log_coaccess=True)

        if mode == BenchmarkMode.NO_ADAPTATION:
            return StepLog(
                step=step_index,
                action_type="NO_OP",
                target_modules=[],
                decision="ACCEPTED",
                reason="No adaptation mode active.",
                delta_recall=0.0,
                delta_latency=0.0,
                reward=0.0,
                metrics=current_metrics.to_dict(),
            )

        # Detect candidate adaptations
        candidates = detect_adaptation_candidates(
            graph=self.graph,
            thresholds=self.config.thresholds,
            storage=self.storage,
        )

        chosen_candidate: Optional[AdaptationCandidate] = None
        action_chosen = BanditAction.NO_OP
        bandit_context: Optional[BanditStateContext] = None

        if mode == BenchmarkMode.HEURISTIC_ONLY:
            if candidates:
                chosen_candidate = candidates[0]  # Highest priority heuristic
            else:
                return StepLog(
                    step=step_index,
                    action_type="NO_OP",
                    target_modules=[],
                    decision="ACCEPTED",
                    reason="No heuristic candidates met threshold.",
                    delta_recall=0.0,
                    delta_latency=0.0,
                    reward=0.0,
                    metrics=current_metrics.to_dict(),
                )

        elif mode == BenchmarkMode.RL_GUIDED:
            bandit_context = self.bandit_policy.extract_context(
                graph=self.graph,
                recent_recall=current_metrics.recall_at_3,
                recent_latency_ms=current_metrics.avg_latency_ms,
                storage=self.storage,
            )

            # Map available candidates to arms
            candidate_map: Dict[BanditAction, AdaptationCandidate] = {}
            for c in candidates:
                act = BanditAction[c.operation_type.upper()]
                if act not in candidate_map:
                    candidate_map[act] = c

            available_actions = [BanditAction.NO_OP] + list(candidate_map.keys())
            action_chosen, ucb_scores = self.bandit_policy.select_action(
                context=bandit_context,
                available_actions=available_actions,
            )

            if action_chosen == BanditAction.NO_OP:
                reward = 0.0
                self.bandit_policy.update(action=action_chosen, context=bandit_context, reward=reward)
                return StepLog(
                    step=step_index,
                    action_type="NO_OP",
                    target_modules=[],
                    decision="ACCEPTED",
                    reason=f"Bandit chose NO_OP (scores={ucb_scores})",
                    delta_recall=0.0,
                    delta_latency=0.0,
                    reward=reward,
                    metrics=current_metrics.to_dict(),
                )
            else:
                chosen_candidate = candidate_map[action_chosen]

        if not chosen_candidate:
            return None

        # Execute safe sandbox trial validation
        val_result = self.validator.validate_adaptation(
            prod_graph=self.graph,
            operation=chosen_candidate.operation,
            tuning_queries=tuning_queries,
            seed=self.config.seed + step_index,
        )

        reward = 0.0
        if mode == BenchmarkMode.RL_GUIDED and bandit_context is not None:
            reward = self.bandit_policy.compute_reward(
                delta_recall=val_result.delta_recall,
                delta_latency=val_result.delta_latency,
                was_accepted=val_result.accepted,
            )
            self.bandit_policy.update(
                action=action_chosen,
                context=bandit_context,
                reward=reward,
            )

        if val_result.accepted and val_result.trial_graph is not None:
            # Commit adaptation to production graph and version history
            self.graph.modules = val_result.trial_graph.modules
            self.graph.graph = val_result.trial_graph.graph

            self.version_manager.commit_version(
                graph=self.graph,
                operation_name=val_result.operation_type,
                metrics_dict=val_result.post_metrics.to_dict(),
                reassignments=val_result.record_reassignments,
                target_modules=val_result.target_modules,
                reason=val_result.reason,
            )
        else:
            # Reject and log rollback
            self.version_manager.log_rollback(val_result)

        post_metric_dict = val_result.post_metrics.to_dict() if val_result.accepted else val_result.pre_metrics.to_dict()

        return StepLog(
            step=step_index,
            action_type=val_result.operation_type,
            target_modules=val_result.target_modules,
            decision=val_result.decision,
            reason=val_result.reason,
            delta_recall=val_result.delta_recall,
            delta_latency=val_result.delta_latency,
            reward=reward if mode == BenchmarkMode.RL_GUIDED else None,
            metrics=post_metric_dict,
        )

    def run_benchmark(
        self,
        dataset: BenchmarkDataset,
        mode: BenchmarkMode = BenchmarkMode.RL_GUIDED,
        num_cycles: int = 5,
    ) -> ExperimentResult:
        """Run complete multi-step experiment under specified mode."""
        # Baseline measurement on tuning queries
        initial_router = RoutedRouter(
            graph=self.graph,
            embedder=self.embedder,
            top_k_centroids=self.config.graph.top_k_centroids,
            expand_shortcuts=self.config.graph.expand_shortcuts,
        )
        init_eval = SystemEvaluator(router=initial_router, storage=self.storage)
        initial_metrics, _ = init_eval.evaluate(dataset.tuning_queries, log_coaccess=True)

        logs: List[StepLog] = []
        accepted = 0
        rejected = 0

        for step in range(1, num_cycles + 1):
            log = self.run_adaptation_cycle(
                tuning_queries=dataset.tuning_queries,
                mode=mode,
                step_index=step,
            )
            if log:
                logs.append(log)
                if log.decision == "ACCEPTED" and log.action_type != "NO_OP":
                    accepted += 1
                elif log.decision == "REJECTED":
                    rejected += 1

        # Final evaluation on tuning and held-out queries
        final_router = RoutedRouter(
            graph=self.graph,
            embedder=self.embedder,
            top_k_centroids=self.config.graph.top_k_centroids,
            expand_shortcuts=self.config.graph.expand_shortcuts,
        )
        final_eval = SystemEvaluator(router=final_router, storage=self.storage)
        final_metrics_tuning, _ = final_eval.evaluate(dataset.tuning_queries, log_coaccess=False)
        final_metrics_heldout, _ = final_eval.evaluate(dataset.heldout_queries, log_coaccess=False)

        return ExperimentResult(
            mode=mode,
            initial_metrics=initial_metrics,
            final_metrics_tuning=final_metrics_tuning,
            final_metrics_heldout=final_metrics_heldout,
            steps_executed=num_cycles,
            accepted_count=accepted,
            rejected_count=rejected,
            logs=logs,
        )

