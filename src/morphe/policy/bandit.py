"""Contextual bandit policy (LinUCB) for learning adaptive structural mutations."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..graph.memory_graph import MemoryGraph
from ..storage.db import SQLiteStorage


class BanditAction(IntEnum):
    NO_OP = 0
    SPLIT = 1
    MERGE = 2
    SHORTCUT = 3


@dataclass
class BanditStateContext:
    feature_vector: np.ndarray
    feature_names: List[str]


class LinUCBPolicy:
    """
    Linear Upper Confidence Bound (LinUCB) contextual bandit.
    Selects structural adaptation actions conditioned on memory graph topology
    and retrieval telemetry.
    """

    def __init__(self, feature_dim: int = 9, alpha: float = 0.5, seed: int = 42) -> None:
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.num_actions = len(BanditAction)
        # Initialize A_a as identity matrices and b_a as zero vectors
        self.A: Dict[int, np.ndarray] = {
            a: np.eye(feature_dim, dtype=np.float32) for a in range(self.num_actions)
        }
        self.b: Dict[int, np.ndarray] = {
            a: np.zeros((feature_dim, 1), dtype=np.float32) for a in range(self.num_actions)
        }
        self.history: List[Dict[str, Any]] = []

    def extract_context(
        self,
        graph: MemoryGraph,
        recent_recall: float = 0.8,
        recent_latency_ms: float = 10.0,
        storage: Optional[SQLiteStorage] = None,
    ) -> BanditStateContext:
        num_mods = len(graph.modules)
        num_edges = graph.graph.number_of_edges()
        num_shortcuts = len(graph.get_shortcuts())

        max_possible_edges = max(1.0, (num_mods * (num_mods - 1)) / 2.0)
        edge_density = float(num_edges / max_possible_edges)
        shortcut_ratio = float(num_shortcuts / max(1, num_edges))

        sizes = [len(m.local_index) for m in graph.get_all_modules()]
        avg_size = float(np.mean(sizes)) if sizes else 0.0
        max_size = float(np.max(sizes)) if sizes else 0.0

        variances = [m.local_index.compute_variance() for m in graph.get_all_modules()]
        max_variance = float(np.max(variances)) if variances else 0.0

        max_coaccess = 0.0
        if storage:
            coaccess = storage.get_coaccess_counts()
            if coaccess:
                max_coaccess = float(max(coaccess.values()))

        features = np.array([
            float(num_mods) / 20.0,
            edge_density,
            shortcut_ratio,
            avg_size / 20.0,
            max_size / 20.0,
            max_variance,
            recent_recall,
            recent_latency_ms / 50.0,
            max_coaccess / 10.0,
        ], dtype=np.float32)

        feature_names = [
            "num_modules_norm",
            "edge_density",
            "shortcut_ratio",
            "avg_module_size_norm",
            "max_module_size_norm",
            "max_variance",
            "recent_recall",
            "recent_latency_norm",
            "max_coaccess_norm",
        ]

        return BanditStateContext(feature_vector=features, feature_names=feature_names)

    def select_action(
        self,
        context: BanditStateContext,
        available_actions: Optional[List[BanditAction]] = None,
    ) -> Tuple[BanditAction, Dict[str, float]]:
        x = context.feature_vector.reshape(-1, 1)
        scores: Dict[str, float] = {}
        valid_actions = available_actions or list(BanditAction)

        best_action = valid_actions[0]
        max_ucb = -1e9

        for a_enum in valid_actions:
            a = int(a_enum)
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]
            mean_est = float((theta_a.T @ x)[0, 0])
            var_term = float(np.sqrt(x.T @ A_inv @ x)[0, 0])
            ucb = mean_est + self.alpha * var_term

            scores[a_enum.name] = round(ucb, 4)
            if ucb > max_ucb:
                max_ucb = ucb
                best_action = a_enum

        return best_action, scores

    def update(
        self,
        action: BanditAction,
        context: BanditStateContext,
        reward: float,
    ) -> None:
        a = int(action)
        x = context.feature_vector.reshape(-1, 1)
        self.A[a] += x @ x.T
        self.b[a] += reward * x

        self.history.append({
            "action": action.name,
            "reward": round(reward, 4),
            "features": context.feature_vector.tolist(),
        })

    def compute_reward(
        self,
        delta_recall: float,
        delta_latency: float,
        was_accepted: bool,
        delta_edges: int = 0,
        w_recall: float = 1.0,
        w_latency: float = 0.2,
        w_complexity: float = 0.05,
        rejection_penalty: float = 0.3,
    ) -> float:
        if not was_accepted:
            return -float(rejection_penalty)

        reward = (
            w_recall * delta_recall
            - w_latency * (delta_latency / 10.0)
            - w_complexity * (delta_edges / 5.0)
        )
        return float(reward)


ContextualBanditPolicy = LinUCBPolicy

