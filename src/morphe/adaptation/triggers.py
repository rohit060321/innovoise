"""Trigger detectors that propose Split, Merge, and Shortcut candidates."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..config import ThresholdsConfig
from ..graph.memory_graph import MemoryGraph
from ..storage.db import SQLiteStorage
from .operations import AdaptationOperation, SplitOperation, MergeOperation, ShortcutOperation


@dataclass
class AdaptationCandidate:
    candidate_id: str
    operation_type: str  # "split", "merge", "shortcut"
    target_modules: List[str]
    priority_score: float
    reason: str
    operation: AdaptationOperation


def detect_adaptation_candidates(
    graph: MemoryGraph,
    thresholds: ThresholdsConfig,
    storage: Optional[SQLiteStorage] = None,
) -> List[AdaptationCandidate]:
    candidates: List[AdaptationCandidate] = []

    # 1. Check for SPLIT candidates
    for mod in graph.get_all_modules():
        mod_size = len(mod.local_index)
        if mod_size < 2:
            continue
        variance = mod.local_index.compute_variance()
        is_oversized = mod_size >= thresholds.split_size_threshold
        is_high_variance = variance >= thresholds.split_variance_threshold

        if is_oversized or is_high_variance:
            priority = (mod_size / max(1, thresholds.split_size_threshold)) + (variance / max(0.1, thresholds.split_variance_threshold))
            reason = f"Module {mod.module_id} triggered split: size={mod_size} (thresh={thresholds.split_size_threshold}), variance={variance:.3f} (thresh={thresholds.split_variance_threshold})"
            candidates.append(
                AdaptationCandidate(
                    candidate_id=f"split_{mod.module_id}",
                    operation_type="split",
                    target_modules=[mod.module_id],
                    priority_score=float(priority),
                    reason=reason,
                    operation=SplitOperation(module_id=mod.module_id),
                )
            )

    # 2. Check for MERGE candidates
    all_mods = graph.get_all_modules()
    for i in range(len(all_mods)):
        for j in range(i + 1, len(all_mods)):
            mod_a = all_mods[i]
            mod_b = all_mods[j]
            combined_size = len(mod_a.local_index) + len(mod_b.local_index)
            if combined_size > thresholds.split_size_threshold:
                continue

            sim = float(np.dot(mod_a.centroid, mod_b.centroid))
            if sim >= thresholds.merge_similarity_threshold:
                priority = sim + (1.0 / max(1, combined_size))
                reason = f"Modules {mod_a.module_id} and {mod_b.module_id} triggered merge: centroid cosine={sim:.3f} >= {thresholds.merge_similarity_threshold}"
                candidates.append(
                    AdaptationCandidate(
                        candidate_id=f"merge_{mod_a.module_id}_{mod_b.module_id}",
                        operation_type="merge",
                        target_modules=[mod_a.module_id, mod_b.module_id],
                        priority_score=float(priority),
                        reason=reason,
                        operation=MergeOperation(module_a=mod_a.module_id, module_b=mod_b.module_id),
                    )
                )

    # 3. Check for SHORTCUT candidates
    if storage:
        coaccess = storage.get_coaccess_counts()
        for (u, v), count in coaccess.items():
            if count >= thresholds.shortcut_coaccess_threshold:
                if graph.graph.has_node(u) and graph.graph.has_node(v):
                    dist = graph.shortest_path_length(u, v)
                    # Propose if disconnected or separated by distance > 1
                    if dist is None or dist > thresholds.shortcut_max_distance:
                        weight = float(min(1.0, 0.2 * count))
                        reason = f"Modules {u} and {v} co-accessed {count} times (thresh={thresholds.shortcut_coaccess_threshold}) with graph distance={dist}"
                        candidates.append(
                            AdaptationCandidate(
                                candidate_id=f"shortcut_{u}_{v}",
                                operation_type="shortcut",
                                target_modules=[u, v],
                                priority_score=float(count),
                                reason=reason,
                                operation=ShortcutOperation(module_a=u, module_b=v, weight=weight),
                            )
                        )

    # Sort descending by priority score
    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    return candidates

