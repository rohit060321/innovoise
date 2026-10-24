"""Tests for LinUCB contextual bandit policy and reinforcement learning controller."""

import sys
from pathlib import Path
import unittest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from morphe.core import MorpheSystem
from morphe.policy.bandit import LinUCBPolicy, BanditAction
from morphe.policy.controller import BenchmarkMode


class TestBanditPolicy(unittest.TestCase):
    def setUp(self):
        self.system = MorpheSystem()
        self.system.initialize_from_benchmark()
        self.policy = LinUCBPolicy(seed=42)

    def test_extract_context(self):
        context = self.policy.extract_context(
            graph=self.system.graph,
            recent_recall=0.85,
            recent_latency_ms=12.0,
            storage=self.system.storage,
        )
        self.assertEqual(len(context.feature_vector), self.policy.feature_dim)
        self.assertEqual(len(context.feature_names), self.policy.feature_dim)

    def test_select_action_and_update(self):
        context = self.policy.extract_context(
            graph=self.system.graph,
            recent_recall=0.85,
            recent_latency_ms=12.0,
            storage=self.system.storage,
        )
        action, scores = self.policy.select_action(context=context)
        self.assertIn(action, list(BanditAction))
        self.assertEqual(len(scores), len(BanditAction))

        # Reward update
        reward = self.policy.compute_reward(
            delta_recall=0.08,
            delta_latency=-2.0,
            was_accepted=True,
        )
        self.assertGreater(reward, 0.0)

        initial_b_norm = float(sum(b.sum() for b in self.policy.b.values()))
        self.policy.update(action=action, context=context, reward=reward)
        post_b_norm = float(sum(b.sum() for b in self.policy.b.values()))
        self.assertGreater(post_b_norm, initial_b_norm)

    def test_three_way_comparison_execution(self):
        # Run a 2-cycle run across all three modes
        results = self.system.run_three_way_comparison(num_cycles=2)
        self.assertIn("no_adaptation", results)
        self.assertIn("heuristic_only", results)
        self.assertIn("rl_guided", results)

        for mode_name, exp_res in results.items():
            self.assertIsNotNone(exp_res.final_metrics_heldout)
            self.assertGreaterEqual(exp_res.final_metrics_heldout.recall_at_3, 0.0)


if __name__ == "__main__":
    unittest.main()

