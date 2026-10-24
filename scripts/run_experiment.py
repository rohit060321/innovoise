"""CLI script to run MORPHE experiments and three-way benchmark comparisons."""

import argparse
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from morphe.core import MorpheSystem
from morphe.policy.controller import BenchmarkMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MORPHE Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "default.yaml"),
        help="Path to default configuration YAML",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=str(project_root / "configs" / "thresholds.yaml"),
        help="Path to thresholds configuration YAML",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="three_way",
        choices=["three_way", "no_adaptation", "heuristic_only", "rl_guided"],
        help="Experiment evaluation mode",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=4,
        help="Number of adaptation cycles to execute",
    )
    return parser.parse_args()


def print_metrics_table(title: str, metrics_dict: dict) -> None:
    print(f"\n=== {title} ===")
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            print(f"  {k:<28}: {v:.4f}")
        else:
            print(f"  {k:<28}: {v}")


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("MORPHE - Reconfigurable Plasticity Benchmark")
    print(f"Config: {args.config}")
    print(f"Mode: {args.mode} | Cycles: {args.cycles}")
    print("=" * 60)

    system = MorpheSystem(config_path=args.config, thresholds_path=args.thresholds)
    system.initialize_from_benchmark()

    if args.mode == "three_way":
        print("\nExecuting Three-Way Comparative Benchmark...")
        results = system.run_three_way_comparison(num_cycles=args.cycles)

        print("\n" + "=" * 70)
        print(f"{'Mode':<18} | {'Held-Out R@3':<14} | {'Held-Out MRR':<14} | {'Avg Latency':<12}")
        print("-" * 70)
        for mode_name, exp_res in results.items():
            h = exp_res.final_metrics_heldout
            print(f"{mode_name:<18} | {h.recall_at_3:<14.4f} | {h.mrr:<14.4f} | {h.avg_latency_ms:<10.2f} ms")
        print("=" * 70)

        for mode_name, exp_res in results.items():
            print_metrics_table(f"Final Held-Out Metrics: {mode_name.upper()}", exp_res.final_metrics_heldout.to_dict())

    else:
        mode_enum = BenchmarkMode(args.mode)
        exp_res = system.controller.run_benchmark(
            dataset=system.benchmark_dataset,
            mode=mode_enum,
            num_cycles=args.cycles,
        )
        print_metrics_table(f"Baseline Metrics ({args.mode})", exp_res.initial_metrics.to_dict())
        print_metrics_table(f"Final Held-Out Metrics ({args.mode})", exp_res.final_metrics_heldout.to_dict())
        print(f"\nAccepted adaptations: {exp_res.accepted_count} | Rejected: {exp_res.rejected_count}")


if __name__ == "__main__":
    main()
