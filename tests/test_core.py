"""Sanity and integration tests for MorpheSystem facade."""

import sys
from pathlib import Path
import unittest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from morphe.core import MorpheSystem


class TestMorpheCore(unittest.TestCase):
    def test_core_initialization(self) -> None:
        system = MorpheSystem()
        self.assertIsNotNone(system)
        self.assertIsNotNone(system.config)
        self.assertIsNotNone(system.graph)

    def test_core_benchmark_ingestion_and_metrics(self) -> None:
        system = MorpheSystem()
        system.initialize_from_benchmark()
        self.assertTrue(system.is_initialized)
        self.assertGreater(len(system.graph.modules), 0)

        metrics = system.get_evaluation_metrics(use_heldout=False)
        self.assertGreater(metrics.recall_at_5, 0.0)


if __name__ == "__main__":
    unittest.main()
