"""Configuration manager for MORPHE."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


@dataclass
class EmbeddingConfig:
    provider: str = "auto"  # "sentence_transformers", "deterministic", or "auto"
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 128
    cache_embeddings: bool = True


@dataclass
class GraphConfig:
    initial_clusters: int = 4
    max_module_capacity: int = 15
    min_module_size: int = 3
    top_k_centroids: int = 2
    expand_shortcuts: bool = True
    similarity_metric: str = "cosine"


@dataclass
class StorageConfig:
    db_path: str = "morphe_memory.db"
    snapshot_dir: str = "snapshots"


@dataclass
class ThresholdsConfig:
    split_variance_threshold: float = 0.45
    split_size_threshold: int = 15
    merge_similarity_threshold: float = 0.82
    shortcut_coaccess_threshold: int = 2
    shortcut_max_distance: int = 1
    # Validation gates
    min_delta_recall: float = -0.01  # Allow neutral or improving recall
    max_delta_latency_ratio: float = 1.35  # At most 35% latency growth


@dataclass
class MorpheConfig:
    project_name: str = "morphe"
    seed: int = 42
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)

    @classmethod
    def load(
        cls,
        default_yaml: Optional[Path | str] = None,
        thresholds_yaml: Optional[Path | str] = None,
    ) -> "MorpheConfig":
        config = cls()
        if default_yaml and Path(default_yaml).exists():
            with open(default_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                if "project_name" in data:
                    config.project_name = data["project_name"]
                if "seed" in data:
                    config.seed = data["seed"]
                if "embedding" in data:
                    config.embedding = EmbeddingConfig(**data["embedding"])
                if "graph" in data:
                    config.graph = GraphConfig(**data["graph"])
                if "storage" in data:
                    config.storage = StorageConfig(**data["storage"])

        if thresholds_yaml and Path(thresholds_yaml).exists():
            with open(thresholds_yaml, "r", encoding="utf-8") as f:
                t_data = yaml.safe_load(f) or {}
                raw_thresholds = t_data.get("thresholds", {})
                validation = t_data.get("validation", {})
                config.thresholds = ThresholdsConfig(
                    split_variance_threshold=raw_thresholds.get("split_variance", 0.45),
                    split_size_threshold=raw_thresholds.get("split_size", 15),
                    merge_similarity_threshold=raw_thresholds.get("merge_similarity", 0.82),
                    shortcut_coaccess_threshold=raw_thresholds.get("shortcut_coaccess", 2),
                    shortcut_max_distance=raw_thresholds.get("shortcut_max_distance", 1),
                    min_delta_recall=validation.get("min_delta_recall", -0.01),
                    max_delta_latency_ratio=validation.get("max_delta_latency_ratio", 1.35),
                )
        return config

