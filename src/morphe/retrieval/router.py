"""Routed semantic retrieval engine strictly traversing centroids and graph shortcuts."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from ..embedding.embedder import BaseEmbedder
from ..graph.memory_graph import MemoryGraph


@dataclass
class RetrievalResult:
    query: str
    records: List[Dict[str, Any]]  # [{"id": id, "score": float, "module_id": str}]
    queried_modules: List[str]
    centroid_ranked_modules: List[Tuple[str, float]]
    shortcut_hops: List[Tuple[str, str]]
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


class RoutedRouter:
    """
    Directs queries through graph centroids and learned topological shortcuts.
    Never falls back to a global scan across all records.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        embedder: BaseEmbedder,
        top_k_centroids: int = 2,
        expand_shortcuts: bool = True,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.top_k_centroids = top_k_centroids
        self.expand_shortcuts = expand_shortcuts

    def retrieve(
        self,
        query: str,
        top_k_records: int = 5,
        local_search_k: int = 5,
    ) -> RetrievalResult:
        start_time = time.perf_counter()

        # Step 1: Embed query
        query_vec = self.embedder.encode(query)

        all_modules = self.graph.get_all_modules()
        if not all_modules:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return RetrievalResult(
                query=query,
                records=[],
                queried_modules=[],
                centroid_ranked_modules=[],
                shortcut_hops=[],
                latency_ms=elapsed,
            )

        # Step 2: Route through module centroids
        centroid_scores: List[Tuple[str, float]] = []
        for mod in all_modules:
            if len(mod.local_index) == 0:
                continue
            norm_c = np.linalg.norm(mod.centroid)
            if norm_c > 1e-9:
                sim = float(np.dot(query_vec, mod.centroid / norm_c))
            else:
                sim = 0.0
            centroid_scores.append((mod.module_id, sim))

        centroid_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top-k centroids
        k_centroids = min(self.top_k_centroids, len(centroid_scores))
        chosen_centroid_mods = [mod_id for mod_id, _ in centroid_scores[:k_centroids]]

        # Step 3: Expand candidate modules along 1-hop shortcuts
        queried_modules_set: Set[str] = set(chosen_centroid_mods)
        shortcut_hops: List[Tuple[str, str]] = []

        if self.expand_shortcuts:
            for mod_id in chosen_centroid_mods:
                shortcut_neighbors = self.graph.get_neighbors(mod_id, edge_type="shortcut")
                for s_neighbor in shortcut_neighbors:
                    if s_neighbor not in queried_modules_set:
                        queried_modules_set.add(s_neighbor)
                        shortcut_hops.append((mod_id, s_neighbor))

        queried_modules = list(queried_modules_set)

        # Step 4: Search ONLY local vector indexes of queried modules
        candidates: List[Dict[str, Any]] = []
        for mod_id in queried_modules:
            mod = self.graph.get_module(mod_id)
            if not mod or len(mod.local_index) == 0:
                continue
            hits = mod.local_index.search(query_vec, top_k=local_search_k)
            for rec_id, score in hits:
                candidates.append({
                    "id": rec_id,
                    "score": score,
                    "module_id": mod_id,
                })

        # Step 5: Rank aggregated hits from traversed modules
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_records = candidates[:top_k_records]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResult(
            query=query,
            records=top_records,
            queried_modules=queried_modules,
            centroid_ranked_modules=centroid_scores,
            shortcut_hops=shortcut_hops,
            latency_ms=elapsed_ms,
        )

