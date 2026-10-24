"""Topological adaptation operators: Split, Merge, and Shortcut."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans

from ..graph.memory_graph import MemoryGraph
from ..graph.index import LocalVectorIndex


@dataclass
class AdaptationResult:
    operation_type: str
    target_modules: List[str]
    new_modules: List[str]
    description: str
    record_reassignments: Dict[str, str]  # rec_id -> new_mod_id


class AdaptationOperation(ABC):
    """Abstract base class for memory graph mutations."""

    @abstractmethod
    def apply(self, graph: MemoryGraph, seed: int = 42) -> AdaptationResult:
        """Apply structural mutation to the given graph instance in-place."""
        pass


class SplitOperation(AdaptationOperation):
    """
    Splits an overloaded or high-variance module into two distinct child modules
    using 2-Means clustering, updating centroids and graph adjacency.
    """

    def __init__(self, module_id: str) -> None:
        self.module_id = module_id

    def apply(self, graph: MemoryGraph, seed: int = 42) -> AdaptationResult:
        target_mod = graph.get_module(self.module_id)
        if not target_mod or len(target_mod.local_index) < 2:
            return AdaptationResult(
                operation_type="split",
                target_modules=[self.module_id],
                new_modules=[],
                description=f"Split skipped: module {self.module_id} has fewer than 2 records.",
                record_reassignments={},
            )

        rec_ids, vectors = target_mod.local_index.get_all_vectors()
        kmeans = KMeans(n_clusters=2, random_state=seed, n_init=5)
        labels = kmeans.fit_predict(vectors)

        child_a_id = f"{self.module_id}_a"
        child_b_id = f"{self.module_id}_b"

        # Create child local indexes
        idx_a = LocalVectorIndex(dimension=graph.dimension)
        idx_b = LocalVectorIndex(dimension=graph.dimension)
        reassignments: Dict[str, str] = {}

        for rid, vec, label in zip(rec_ids, vectors, labels):
            if label == 0:
                idx_a.add(rid, vec)
                reassignments[rid] = child_a_id
            else:
                idx_b.add(rid, vec)
                reassignments[rid] = child_b_id

        # Guard: if one cluster is empty, force at least one element into each
        if len(idx_a) == 0:
            rid, vec = rec_ids[0], vectors[0]
            idx_b.remove(rid)
            idx_a.add(rid, vec)
            reassignments[rid] = child_a_id
        elif len(idx_b) == 0:
            rid, vec = rec_ids[0], vectors[0]
            idx_a.remove(rid)
            idx_b.add(rid, vec)
            reassignments[rid] = child_b_id

        # Preserve neighbors of target module
        old_neighbors = list(graph.graph.neighbors(self.module_id))
        old_edges = [(n, graph.graph.get_edge_data(self.module_id, n)) for n in old_neighbors]

        # Remove old module
        graph.remove_module(self.module_id)

        # Add child modules
        node_a = graph.add_module(
            module_id=child_a_id,
            capacity=target_mod.capacity,
            local_index=idx_a,
            metadata={"parent": self.module_id, "split_cluster": 0},
        )
        node_b = graph.add_module(
            module_id=child_b_id,
            capacity=target_mod.capacity,
            local_index=idx_b,
            metadata={"parent": self.module_id, "split_cluster": 1},
        )

        # Bridge child modules with structural edge
        sim_ab = float(np.dot(node_a.centroid, node_b.centroid))
        graph.add_edge(child_a_id, child_b_id, edge_type="structural", weight=round(max(0.1, sim_ab), 3))

        # Re-attach former neighbors to closest child
        for neighbor, edge_data in old_edges:
            if not graph.graph.has_node(neighbor):
                continue
            n_mod = graph.get_module(neighbor)
            if not n_mod:
                continue
            sim_to_a = float(np.dot(node_a.centroid, n_mod.centroid))
            sim_to_b = float(np.dot(node_b.centroid, n_mod.centroid))
            e_type = edge_data.get("edge_type", "structural")
            weight = edge_data.get("weight", 1.0)

            if sim_to_a >= sim_to_b:
                graph.add_edge(child_a_id, neighbor, edge_type=e_type, weight=weight)
            else:
                graph.add_edge(child_b_id, neighbor, edge_type=e_type, weight=weight)

        return AdaptationResult(
            operation_type="split",
            target_modules=[self.module_id],
            new_modules=[child_a_id, child_b_id],
            description=f"Split {self.module_id} into {child_a_id} ({len(idx_a)} items) and {child_b_id} ({len(idx_b)} items).",
            record_reassignments=reassignments,
        )


class MergeOperation(AdaptationOperation):
    """
    Fuses two semantically similar or underutilized modules into a single unified module,
    consolidating local vectors and merging incident graph edges.
    """

    def __init__(self, module_a: str, module_b: str) -> None:
        self.module_a = module_a
        self.module_b = module_b

    def apply(self, graph: MemoryGraph, seed: int = 42) -> AdaptationResult:
        mod_a = graph.get_module(self.module_a)
        mod_b = graph.get_module(self.module_b)
        if not mod_a or not mod_b:
            return AdaptationResult(
                operation_type="merge",
                target_modules=[self.module_a, self.module_b],
                new_modules=[],
                description=f"Merge failed: one or both modules ({self.module_a}, {self.module_b}) not found.",
                record_reassignments={},
            )

        merged_id = f"{self.module_a}_{self.module_b}_merged"
        merged_idx = LocalVectorIndex(dimension=graph.dimension)
        reassignments: Dict[str, str] = {}

        # Transfer records from mod_a
        ids_a, vecs_a = mod_a.local_index.get_all_vectors()
        for rid, vec in zip(ids_a, vecs_a):
            merged_idx.add(rid, vec)
            reassignments[rid] = merged_id

        # Transfer records from mod_b
        ids_b, vecs_b = mod_b.local_index.get_all_vectors()
        for rid, vec in zip(ids_b, vecs_b):
            merged_idx.add(rid, vec)
            reassignments[rid] = merged_id

        # Collect combined incident edges
        neighbors_a = [(n, graph.graph.get_edge_data(self.module_a, n)) for n in graph.graph.neighbors(self.module_a) if n != self.module_b]
        neighbors_b = [(n, graph.graph.get_edge_data(self.module_b, n)) for n in graph.graph.neighbors(self.module_b) if n != self.module_a]

        # Remove old modules
        graph.remove_module(self.module_a)
        graph.remove_module(self.module_b)

        # Add merged module
        new_capacity = max(mod_a.capacity, mod_b.capacity, len(merged_idx) + 5)
        graph.add_module(
            module_id=merged_id,
            capacity=new_capacity,
            local_index=merged_idx,
            metadata={"merged_from": [self.module_a, self.module_b]},
        )

        # Wire combined edges
        for neighbor, edge_data in neighbors_a + neighbors_b:
            if graph.graph.has_node(neighbor) and neighbor != merged_id:
                graph.add_edge(
                    merged_id,
                    neighbor,
                    edge_type=edge_data.get("edge_type", "structural"),
                    weight=edge_data.get("weight", 1.0),
                )

        return AdaptationResult(
            operation_type="merge",
            target_modules=[self.module_a, self.module_b],
            new_modules=[merged_id],
            description=f"Merged {self.module_a} and {self.module_b} into {merged_id} ({len(merged_idx)} records).",
            record_reassignments=reassignments,
        )


class ShortcutOperation(AdaptationOperation):
    """
    Synthesizes a 1-hop shortcut topological edge between two modules
    frequently co-accessed in multi-domain retrieval queries.
    """

    def __init__(self, module_a: str, module_b: str, weight: float = 1.0) -> None:
        self.module_a = module_a
        self.module_b = module_b
        self.weight = weight

    def apply(self, graph: MemoryGraph, seed: int = 42) -> AdaptationResult:
        if not graph.graph.has_node(self.module_a) or not graph.graph.has_node(self.module_b):
            return AdaptationResult(
                operation_type="shortcut",
                target_modules=[self.module_a, self.module_b],
                new_modules=[],
                description=f"Shortcut failed: missing nodes {self.module_a} or {self.module_b}.",
                record_reassignments={},
            )

        graph.add_edge(
            self.module_a,
            self.module_b,
            edge_type="shortcut",
            weight=round(self.weight, 3),
        )

        return AdaptationResult(
            operation_type="shortcut",
            target_modules=[self.module_a, self.module_b],
            new_modules=[],
            description=f"Created shortcut edge between {self.module_a} and {self.module_b} (weight={self.weight}).",
            record_reassignments={},
        )


def apply_adaptation(
    operation: AdaptationOperation,
    graph: MemoryGraph,
    seed: int = 42,
) -> AdaptationResult:
    return operation.apply(graph=graph, seed=seed)

