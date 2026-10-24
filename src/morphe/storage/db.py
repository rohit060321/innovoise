"""SQLite storage manager for records, modules, audit logs, and version snapshots."""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MemoryRecord:
    id: str
    text: str
    metadata: Dict[str, Any]
    module_id: str
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.time()


class SQLiteStorage:
    """Manages SQLite storage for MORPHE persistent state and audit history."""

    def __init__(self, db_path: str | Path = "morphe_memory.db") -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL,
                module_id TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS modules (
                module_id TEXT PRIMARY KEY,
                centroid_json TEXT NOT NULL,
                capacity INTEGER NOT NULL,
                created_at REAL NOT NULL
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (source_id, target_id)
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS version_snapshots (
                version_id TEXT PRIMARY KEY,
                parent_version_id TEXT,
                timestamp REAL NOT NULL,
                operation TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                action_type TEXT NOT NULL,
                target_modules TEXT NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                delta_recall REAL,
                delta_latency REAL,
                metrics_json TEXT
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS coaccess_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                query_text TEXT NOT NULL,
                module_a TEXT NOT NULL,
                module_b TEXT NOT NULL
            );
            """)
            conn.commit()

    def insert_records(self, records: List[MemoryRecord]) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO records (id, text, metadata, module_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.id,
                        r.text,
                        json.dumps(r.metadata),
                        r.module_id,
                        r.created_at,
                    )
                    for r in records
                ],
            )
            conn.commit()

    def update_record_modules(self, record_module_map: Dict[str, str]) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "UPDATE records SET module_id = ? WHERE id = ?",
                [(mod_id, rec_id) for rec_id, mod_id in record_module_map.items()],
            )
            conn.commit()

    def get_all_records(self) -> List[MemoryRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, text, metadata, module_id, created_at FROM records")
            rows = cursor.fetchall()
            return [
                MemoryRecord(
                    id=row["id"],
                    text=row["text"],
                    metadata=json.loads(row["metadata"]),
                    module_id=row["module_id"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def log_coaccess(self, query_text: str, modules: List[str]) -> None:
        """Log co-accessed modules for a query."""
        if len(modules) < 2:
            return
        now = time.time()
        pairs = []
        sorted_mods = sorted(list(set(modules)))
        for i in range(len(sorted_mods)):
            for j in range(i + 1, len(sorted_mods)):
                pairs.append((now, query_text, sorted_mods[i], sorted_mods[j]))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO coaccess_logs (timestamp, query_text, module_a, module_b) VALUES (?, ?, ?, ?)",
                pairs,
            )
            conn.commit()

    def get_coaccess_counts(self) -> Dict[Tuple[str, str], int]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT module_a, module_b, COUNT(*) as count FROM coaccess_logs GROUP BY module_a, module_b"
            )
            rows = cursor.fetchall()
            return {(row["module_a"], row["module_b"]): row["count"] for row in rows}

    def save_snapshot(
        self,
        version_id: str,
        parent_version_id: Optional[str],
        operation: str,
        snapshot_dict: Dict[str, Any],
        metrics_dict: Dict[str, Any],
    ) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO version_snapshots 
                (version_id, parent_version_id, timestamp, operation, snapshot_json, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    parent_version_id,
                    time.time(),
                    operation,
                    json.dumps(snapshot_dict),
                    json.dumps(metrics_dict),
                ),
            )
            conn.commit()

    def get_snapshot(self, version_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT version_id, parent_version_id, timestamp, operation, snapshot_json, metrics_json FROM version_snapshots WHERE version_id = ?",
                (version_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "version_id": row["version_id"],
                "parent_version_id": row["parent_version_id"],
                "timestamp": row["timestamp"],
                "operation": row["operation"],
                "snapshot": json.loads(row["snapshot_json"]),
                "metrics": json.loads(row["metrics_json"]),
            }

    def get_all_snapshots(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT version_id, parent_version_id, timestamp, operation, metrics_json FROM version_snapshots ORDER BY timestamp ASC"
            )
            rows = cursor.fetchall()
            return [
                {
                    "version_id": row["version_id"],
                    "parent_version_id": row["parent_version_id"],
                    "timestamp": row["timestamp"],
                    "operation": row["operation"],
                    "metrics": json.loads(row["metrics_json"]),
                }
                for row in rows
            ]

    def log_audit(
        self,
        action_type: str,
        target_modules: List[str] | str,
        decision: str,
        reason: str,
        delta_recall: Optional[float] = None,
        delta_latency: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        targets_str = ",".join(target_modules) if isinstance(target_modules, list) else str(target_modules)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO audit_log 
                (timestamp, action_type, target_modules, decision, reason, delta_recall, delta_latency, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    action_type,
                    targets_str,
                    decision,
                    reason,
                    delta_recall,
                    delta_latency,
                    json.dumps(metrics or {}),
                ),
            )
            conn.commit()

    def get_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, timestamp, action_type, target_modules, decision, reason, delta_recall, delta_latency, metrics_json FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "action_type": row["action_type"],
                    "target_modules": row["target_modules"],
                    "decision": row["decision"],
                    "reason": row["reason"],
                    "delta_recall": row["delta_recall"],
                    "delta_latency": row["delta_latency"],
                    "metrics": json.loads(row["metrics_json"]),
                }
                for row in rows
            ]

