"""Version management, snapshot persistence, and zero-risk point-in-time restore."""

from typing import Any, Dict, List, Optional
import time

from ..graph.memory_graph import MemoryGraph
from ..storage.db import SQLiteStorage
from .validator import ValidationResult


class VersionManager:
    """Manages version snapshots, historical checkpoints, and instant restores."""

    def __init__(self, storage: SQLiteStorage) -> None:
        self.storage = storage
        self.active_version_id: Optional[str] = None
        self.version_counter: int = 0

    def commit_version(
        self,
        graph: MemoryGraph,
        operation_name: str,
        metrics_dict: Dict[str, Any],
        reassignments: Optional[Dict[str, str]] = None,
        target_modules: Optional[List[str]] = None,
        reason: str = "Structural adaptation applied",
    ) -> str:
        parent_id = self.active_version_id
        self.version_counter += 1
        new_version_id = f"v{self.version_counter}_{operation_name}"

        snapshot_dict = graph.to_dict()
        self.storage.save_snapshot(
            version_id=new_version_id,
            parent_version_id=parent_id,
            operation=operation_name,
            snapshot_dict=snapshot_dict,
            metrics_dict=metrics_dict,
        )

        if reassignments:
            self.storage.update_record_modules(reassignments)

        self.storage.log_audit(
            action_type=operation_name,
            target_modules=target_modules or [],
            decision="ACCEPTED",
            reason=reason,
            delta_recall=metrics_dict.get("delta_recall", 0.0),
            delta_latency=metrics_dict.get("delta_latency", 0.0),
            metrics=metrics_dict,
        )

        self.active_version_id = new_version_id
        return new_version_id

    def log_rollback(self, rejected_result: ValidationResult) -> None:
        self.storage.log_audit(
            action_type=rejected_result.operation_type,
            target_modules=rejected_result.target_modules,
            decision="REJECTED",
            reason=rejected_result.reason,
            delta_recall=rejected_result.delta_recall,
            delta_latency=rejected_result.delta_latency,
            metrics={
                "pre": rejected_result.pre_metrics.to_dict(),
                "post": rejected_result.post_metrics.to_dict(),
            },
        )

    def restore_version(self, version_id: str) -> MemoryGraph:
        snapshot_entry = self.storage.get_snapshot(version_id)
        if not snapshot_entry:
            raise ValueError(f"Snapshot version '{version_id}' not found in storage.")

        restored_graph = MemoryGraph.from_dict(snapshot_entry["snapshot"])

        # Update record module assignments in database
        reassignments = {}
        for mod_id, mod in restored_graph.modules.items():
            for rid in mod.local_index.record_ids:
                reassignments[rid] = mod_id
        if reassignments:
            self.storage.update_record_modules(reassignments)

        self.active_version_id = version_id
        self.storage.log_audit(
            action_type="restore",
            target_modules=list(restored_graph.modules.keys()),
            decision="ACCEPTED",
            reason=f"Restored graph to historical snapshot {version_id}",
            metrics=snapshot_entry.get("metrics", {}),
        )
        return restored_graph

    def get_history(self) -> List[Dict[str, Any]]:
        return self.storage.get_all_snapshots()

    def get_audit_trail(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.storage.get_audit_logs(limit=limit)

