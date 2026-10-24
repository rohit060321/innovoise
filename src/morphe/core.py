"""Unified facade and runtime engine for MORPHE."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .adaptation.triggers import detect_adaptation_candidates
from .config import MorpheConfig
from .embedding.embedder import BaseEmbedder, get_embedder
from .evaluation.dataset import BenchmarkDataset, get_standard_benchmark
from .evaluation.evaluator import SystemEvaluator
from .evaluation.metrics import EvaluationMetrics
from .graph.memory_graph import MemoryGraph
from .ingestion import ingest_documents, ingest_incremental_documents
from .policy.bandit import LinUCBPolicy
from .policy.controller import MemoryController, BenchmarkMode, ExperimentResult, StepLog
from .retrieval.router import RoutedRouter, RetrievalResult
from .safety.validator import SandboxValidator
from .safety.versioning import VersionManager
from .storage.db import SQLiteStorage, MemoryRecord


class MorpheSystem:
    """
    Main system interface for MORPHE (Memory-Oriented Reconfigurable Plasticity and Hierarchical Evolution).
    """

    def __init__(
        self,
        config: Optional[MorpheConfig] = None,
        config_path: Optional[str | Path] = None,
        thresholds_path: Optional[str | Path] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = MorpheConfig.load(
                default_yaml=config_path or "configs/default.yaml",
                thresholds_yaml=thresholds_path or "configs/thresholds.yaml",
            )

        self.storage = SQLiteStorage(db_path=self.config.storage.db_path)
        self.embedder: BaseEmbedder = get_embedder(
            provider=self.config.embedding.provider,
            model_name=self.config.embedding.model_name,
            dimension=self.config.embedding.dimension,
            seed=self.config.seed,
            cache_embeddings=self.config.embedding.cache_embeddings,
        )
        self.graph = MemoryGraph(dimension=self.embedder.dimension)
        self.router = RoutedRouter(
            graph=self.graph,
            embedder=self.embedder,
            top_k_centroids=self.config.graph.top_k_centroids,
            expand_shortcuts=self.config.graph.expand_shortcuts,
        )
        self.validator = SandboxValidator(
            embedder=self.embedder,
            thresholds=self.config.thresholds,
            top_k_centroids=self.config.graph.top_k_centroids,
            expand_shortcuts=self.config.graph.expand_shortcuts,
        )
        self.version_manager = VersionManager(storage=self.storage)
        self.bandit_policy = LinUCBPolicy(seed=self.config.seed)
        self.controller = MemoryController(
            graph=self.graph,
            embedder=self.embedder,
            storage=self.storage,
            config=self.config,
            validator=self.validator,
            version_manager=self.version_manager,
            bandit_policy=self.bandit_policy,
        )
        self.benchmark_dataset: BenchmarkDataset = get_standard_benchmark()
        self.is_initialized: bool = False

    def initialize_from_benchmark(self) -> None:
        """Ingest the standard multi-domain benchmark dataset into memory."""
        ingest_documents(
            docs=self.benchmark_dataset.documents,
            graph=self.graph,
            storage=self.storage,
            embedder=self.embedder,
            config=self.config,
        )
        # Commit initial version v0
        evaluator = SystemEvaluator(router=self.router, storage=self.storage)
        init_metrics, _ = evaluator.evaluate(self.benchmark_dataset.tuning_queries, log_coaccess=True)
        self.version_manager.commit_version(
            graph=self.graph,
            operation_name="initial_commit",
            metrics_dict=init_metrics.to_dict(),
            reason="Initial ingestion and hierarchical clustering",
        )
        self.is_initialized = True

    def ingest_custom_documents(self, docs: List[Dict[str, Any]]) -> List[MemoryRecord]:
        """Incrementally ingest user-supplied documents into active memory graph."""
        records = ingest_incremental_documents(
            docs=docs,
            graph=self.graph,
            storage=self.storage,
            embedder=self.embedder,
            config=self.config,
        )
        if records:
            evaluator = SystemEvaluator(router=self.router, storage=self.storage)
            metrics, _ = evaluator.evaluate(self.benchmark_dataset.tuning_queries, log_coaccess=False)
            self.version_manager.commit_version(
                graph=self.graph,
                operation_name="user_ingest",
                metrics_dict=metrics.to_dict(),
                reason=f"User ingested {len(docs)} documents ({len(records)} chunks)",
            )
        return records

    def query(self, text: str, top_k: int = 5) -> RetrievalResult:
        """Route a query through graph centroids and shortcuts."""
        res = self.router.retrieve(query=text, top_k_records=top_k)
        if len(res.queried_modules) > 1:
            self.storage.log_coaccess(text, res.queried_modules)
        return res

    def adapt_step(self, mode: str = "rl_guided") -> Optional[StepLog]:
        """Trigger an autonomous adaptation step."""
        benchmark_mode = BenchmarkMode(mode)
        step_log = self.controller.run_adaptation_cycle(
            tuning_queries=self.benchmark_dataset.tuning_queries,
            mode=benchmark_mode,
            step_index=self.version_manager.version_counter + 1,
        )
        # Update router with active graph instance
        self.router.graph = self.graph
        return step_log

    def restore_version(self, version_id: str) -> None:
        """Restore memory graph to a previous snapshot."""
        restored_graph = self.version_manager.restore_version(version_id)
        self.graph = restored_graph
        self.router.graph = restored_graph
        self.controller.graph = restored_graph

    def get_evaluation_metrics(self, use_heldout: bool = False) -> EvaluationMetrics:
        evaluator = SystemEvaluator(router=self.router, storage=self.storage)
        queries = self.benchmark_dataset.heldout_queries if use_heldout else self.benchmark_dataset.tuning_queries
        metrics, _ = evaluator.evaluate(queries=queries, log_coaccess=False)
        return metrics

    def run_three_way_comparison(self, num_cycles: int = 4) -> Dict[str, ExperimentResult]:
        """Run comparison between No-Adaptation, Heuristic, and RL-Guided modes."""
        results: Dict[str, ExperimentResult] = {}
        for mode in [BenchmarkMode.NO_ADAPTATION, BenchmarkMode.HEURISTIC_ONLY, BenchmarkMode.RL_GUIDED]:
            # Fork a clean replica of the system for each mode
            replica = MorpheSystem(config=self.config)
            replica.initialize_from_benchmark()
            res = replica.controller.run_benchmark(
                dataset=self.benchmark_dataset,
                mode=mode,
                num_cycles=num_cycles,
            )
            results[mode.value] = res
        return results


# Backward compatibility
MorpheCore = MorpheSystem
