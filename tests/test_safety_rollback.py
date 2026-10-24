"""Tests for safe validation, rollback, and point-in-time restore."""

import sys
from pathlib import Path
import unittest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from morphe.core import MorpheSystem
from morphe.adaptation.operations import AdaptationOperation, AdaptationResult


class HarmfulDestructiveOp(AdaptationOperation):
    """Synthetic operation that deletes all modules to cause catastrophic failure."""
    def apply(self, graph, seed=42):
        for m in list(graph.modules.keys()):
            graph.remove_module(m)
        return AdaptationResult(
            operation_type="destructive_purge",
            target_modules=["all"],
            new_modules=[],
            description="Destructive deletion",
            record_reassignments={},
        )


class TestSafetyAndRollback(unittest.TestCase):
    def setUp(self):
        self.system = MorpheSystem()
        self.system.initialize_from_benchmark()

    def test_harmful_change_rejected_and_rolled_back(self):
        initial_mod_count = len(self.system.graph.modules)
        initial_records = len(self.system.storage.get_all_records())

        val_result = self.system.validator.validate_adaptation(
            prod_graph=self.system.graph,
            operation=HarmfulDestructiveOp(),
            tuning_queries=self.system.benchmark_dataset.tuning_queries,
        )

        self.assertFalse(val_result.accepted)
        self.assertEqual(val_result.decision, "REJECTED")
        self.assertIn("Recall regression", val_result.reason)

        # Log rollback
        self.system.version_manager.log_rollback(val_result)

        # Ensure production graph was NEVER modified
        self.assertEqual(len(self.system.graph.modules), initial_mod_count)
        self.assertEqual(len(self.system.storage.get_all_records()), initial_records)

    def test_version_snapshot_and_restore(self):
        initial_ver = self.system.version_manager.active_version_id
        initial_mods = set(self.system.graph.modules.keys())

        # Trigger a valid adaptation
        step_log = self.system.adapt_step(mode="heuristic_only")
        second_ver = self.system.version_manager.active_version_id

        # Verify new version was committed
        self.assertNotEqual(initial_ver, second_ver)

        # Restore initial version
        self.system.restore_version(initial_ver)
        self.assertEqual(self.system.version_manager.active_version_id, initial_ver)
        self.assertEqual(set(self.system.graph.modules.keys()), initial_mods)


if __name__ == "__main__":
    unittest.main()

