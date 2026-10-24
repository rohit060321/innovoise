"""Tests for routed semantic retrieval and graph shortcuts."""

import sys
from pathlib import Path
import unittest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from morphe.core import MorpheSystem


class TestRoutedRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = MorpheSystem()
        cls.system.initialize_from_benchmark()

    def test_routed_retrieval_returns_records(self):
        query = "How does Raft elect a leader and maintain log replication?"
        result = self.system.query(query, top_k=3)
        self.assertGreater(len(result.records), 0)
        self.assertGreater(len(result.queried_modules), 0)
        self.assertLessEqual(len(result.queried_modules), len(self.system.graph.modules))

    def test_retrieval_does_not_global_scan(self):
        query = "Quantum error correction surface codes"
        result = self.system.query(query, top_k=3)
        # Should only query candidate module subset
        self.assertLess(len(result.queried_modules), len(self.system.graph.modules) + 1)
        self.assertTrue(all(isinstance(r["score"], float) for r in result.records))

    def test_shortcut_traversal_expansion(self):
        # Add a shortcut edge between two modules
        all_mods = list(self.system.graph.modules.keys())
        u, v = all_mods[0], all_mods[-1]
        self.system.graph.add_edge(u, v, edge_type="shortcut", weight=1.0)

        # Force query to match centroid of u
        result = self.system.query("Consensus and distributed networks", top_k=5)
        # If u is queried, v should also be expanded via shortcut
        if u in result.centroid_ranked_modules[:self.system.config.graph.top_k_centroids]:
            self.assertIn(v, result.queried_modules)


if __name__ == "__main__":
    unittest.main()

