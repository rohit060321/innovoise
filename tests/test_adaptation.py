"""Tests for Split, Merge, and Shortcut adaptation operators."""

import sys
from pathlib import Path
import unittest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from morphe.core import MorpheSystem
from morphe.adaptation.operations import SplitOperation, MergeOperation, ShortcutOperation


class TestAdaptationOperations(unittest.TestCase):
    def setUp(self):
        self.system = MorpheSystem()
        self.system.initialize_from_benchmark()

    def test_split_operation(self):
        # Pick the module with the most records
        target_mod = max(self.system.graph.get_all_modules(), key=lambda m: len(m.local_index))
        initial_records = len(target_mod.local_index)
        initial_modules = len(self.system.graph.modules)
        target_id = target_mod.module_id

        split_op = SplitOperation(module_id=target_id)
        result = split_op.apply(self.system.graph)

        self.assertEqual(result.operation_type, "split")
        self.assertEqual(len(result.new_modules), 2)
        # Verify old module removed, new modules added
        self.assertNotIn(target_id, self.system.graph.modules)
        self.assertIn(result.new_modules[0], self.system.graph.modules)
        self.assertIn(result.new_modules[1], self.system.graph.modules)

        # Net modules increased by 1
        self.assertEqual(len(self.system.graph.modules), initial_modules + 1)
        # Sum of records in child modules equals original
        sum_records = len(self.system.graph.get_module(result.new_modules[0]).local_index) + \
                      len(self.system.graph.get_module(result.new_modules[1]).local_index)
        self.assertEqual(sum_records, initial_records)

    def test_merge_operation(self):
        all_mods = list(self.system.graph.modules.keys())
        u, v = all_mods[0], all_mods[1]
        rec_u = len(self.system.graph.get_module(u).local_index)
        rec_v = len(self.system.graph.get_module(v).local_index)
        initial_mod_count = len(self.system.graph.modules)

        merge_op = MergeOperation(module_a=u, module_b=v)
        result = merge_op.apply(self.system.graph)

        self.assertEqual(result.operation_type, "merge")
        self.assertEqual(len(result.new_modules), 1)
        merged_id = result.new_modules[0]
        self.assertNotIn(u, self.system.graph.modules)
        self.assertNotIn(v, self.system.graph.modules)
        self.assertIn(merged_id, self.system.graph.modules)
        # Net modules decreased by 1
        self.assertEqual(len(self.system.graph.modules), initial_mod_count - 1)
        # Combined record count matches
        self.assertEqual(len(self.system.graph.get_module(merged_id).local_index), rec_u + rec_v)

    def test_shortcut_operation(self):
        all_mods = list(self.system.graph.modules.keys())
        u, v = all_mods[0], all_mods[-1]

        shortcut_op = ShortcutOperation(module_a=u, module_b=v, weight=0.9)
        result = shortcut_op.apply(self.system.graph)

        self.assertEqual(result.operation_type, "shortcut")
        self.assertTrue(self.system.graph.graph.has_edge(u, v))
        edge_data = self.system.graph.graph.get_edge_data(u, v)
        self.assertEqual(edge_data.get("edge_type"), "shortcut")


if __name__ == "__main__":
    unittest.main()

