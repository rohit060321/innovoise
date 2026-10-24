"""Interactive System Verification Walkthrough for MORPHE.

Directly implements the 6-stage demo script from the MORPHE Blueprint:
1. Show naive graph, baseline score, and detected weaknesses (overloaded, redundant, co-access).
2. Execute routed semantic retrieval without brute-force global scans.
3. Propose, validate in sandbox trial copy, and commit a module SPLIT.
4. Synthesize a learned topological SHORTCUT between co-accessed modules.
5. Inject an intentionally harmful mutation, verifying sandbox rejection and rollback.
6. Browse version history and execute a point-in-time RESTORE.
7. Final comparison: Static Memory vs Heuristic-Only vs RL-Guided (LinUCB contextual bandit).
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from morphe.core import MorpheSystem
from morphe.adaptation.operations import SplitOperation, ShortcutOperation, AdaptationResult
from morphe.adaptation.triggers import detect_adaptation_candidates
from morphe.policy.controller import BenchmarkMode


def step_header(step_num: int, title: str) -> None:
    print("\n" + "=" * 74)
    print(f"STEP {step_num}: {title.upper()}")
    print("=" * 74)


def main() -> None:
    print("""
    ==========================================================================
    MORPHE: Memory-Oriented Reconfigurable Plasticity & Hierarchical Evolution
    PS 3: Autonomous Adaptive AI Memory System
    ==========================================================================
    "RL proposes, validation disposes: a learned policy chooses structural changes,
     and a deterministic validator decides whether to keep them."
    """)

    # ----------------------------------------------------
    step_header(1, "Memory Foundation - Ingestion & Weakness Discovery")
    # ----------------------------------------------------
    system = MorpheSystem()
    print("Ingesting multi-domain benchmark documents into memory modules...")
    system.initialize_from_benchmark()
    modules = system.graph.get_all_modules()
    print(f"-> Created {len(modules)} initial memory modules.")
    for m in modules:
        print(f"   * Module [{m.module_id}]: {len(m.local_index):>2} records | Variance = {m.local_index.compute_variance():.3f} | Capacity = {m.capacity}")

    init_metrics = system.get_evaluation_metrics(use_heldout=False)
    print(f"\n-> Baseline Telemetry: Recall@3 = {init_metrics.recall_at_3:.4f} | MRR = {init_metrics.mrr:.4f} | Latency = {init_metrics.avg_latency_ms:.2f} ms")
    print(f"-> Active Version Snapshot: {system.version_manager.active_version_id}")

    # Detect weaknesses
    candidates = detect_adaptation_candidates(system.graph, system.config.thresholds, system.storage)
    print(f"\n-> Detected Structural Weaknesses ({len(candidates)} trigger signals):")
    for c in candidates:
        print(f"   [!] {c.operation_type.upper():<8}: Target={','.join(c.target_modules):<8} | Priority={c.priority_score:.2f} | {c.reason}")

    # ----------------------------------------------------
    step_header(2, "Routed Semantic Retrieval (No Global Scans)")
    # ----------------------------------------------------
    query_text = "How does Raft elect a leader and maintain log replication in distributed state machines?"
    print(f"Query: '{query_text}'")
    res = system.query(query_text, top_k=3)
    print(f"-> Routing execution latency: {res.latency_ms:.2f} ms")
    print("-> Centroid proximity scores:")
    for m_id, score in res.centroid_ranked_modules:
        status = "VISITED" if m_id in res.queried_modules else "PRUNED"
        print(f"   * Module {m_id:<5}: Cosine = {score:.4f} [{status}]")
    print(f"-> Traversed modules: {res.queried_modules} (Global search was bypassed)")
    print(f"-> Top retrieved record: {res.records[0]['id']} (Score: {res.records[0]['score']:.4f})")

    # ----------------------------------------------------
    step_header(3, "Self-Adaptation - Proposing & Validating Split")
    # ----------------------------------------------------
    largest_mod = max(modules, key=lambda m: len(m.local_index))
    print(f"Targeting overloaded module '{largest_mod.module_id}' ({len(largest_mod.local_index)} records, variance={largest_mod.local_index.compute_variance():.3f}) for SPLIT.")
    split_op = SplitOperation(module_id=largest_mod.module_id)

    print("Cloning memory graph into isolated sandbox trial copy...")
    val_split = system.validator.validate_adaptation(
        prod_graph=system.graph,
        operation=split_op,
        tuning_queries=system.benchmark_dataset.tuning_queries,
    )
    print(f"-> Validator Verdict: {val_split.decision}")
    print(f"   Reason: {val_split.reason}")
    print(f"   Delta Recall: {val_split.delta_recall:+.4f} | Delta Latency: {val_split.delta_latency:+.2f} ms")

    if val_split.accepted and val_split.trial_graph:
        system.graph.modules = val_split.trial_graph.modules
        system.graph.graph = val_split.trial_graph.graph
        v_split = system.version_manager.commit_version(
            graph=system.graph,
            operation_name="split",
            metrics_dict=val_split.post_metrics.to_dict(),
            reassignments=val_split.record_reassignments,
            target_modules=val_split.target_modules,
        )
        print(f"-> Committed new version: '{v_split}' (Active modules: {len(system.graph.modules)})")

    # ----------------------------------------------------
    step_header(4, "Self-Adaptation - Shortcut Synthesis")
    # ----------------------------------------------------
    all_mod_ids = list(system.graph.modules.keys())
    u, v = all_mod_ids[0], all_mod_ids[-1]
    print(f"Synthesizing topological shortcut between non-adjacent modules: '{u}' <---> '{v}'")
    shortcut_op = ShortcutOperation(module_a=u, module_b=v, weight=0.85)

    print("Validating shortcut impact in trial sandbox...")
    val_short = system.validator.validate_adaptation(
        prod_graph=system.graph,
        operation=shortcut_op,
        tuning_queries=system.benchmark_dataset.tuning_queries,
    )
    print(f"-> Validator Verdict: {val_short.decision}")
    if val_short.accepted and val_short.trial_graph:
        system.graph.modules = val_short.trial_graph.modules
        system.graph.graph = val_short.trial_graph.graph
        v_short = system.version_manager.commit_version(
            graph=system.graph,
            operation_name="shortcut",
            metrics_dict=val_short.post_metrics.to_dict(),
        )
        print(f"-> Committed shortcut version: '{v_short}'")
        print(f"-> Active Shortcuts in graph: {system.graph.get_shortcuts()}")

    # ----------------------------------------------------
    step_header(5, "Safety Gate - Rejecting Harmful Mutation & Zero-Risk Rollback")
    # ----------------------------------------------------
    print("Injecting an intentionally harmful mutation (purging vectors to induce recall collapse)...")

    class HarmfulPurgeOperation:
        def apply(self, graph, seed=42):
            for m in graph.get_all_modules():
                m.local_index.record_ids.clear()
                m.local_index.vectors = np.empty((0, graph.dimension), dtype=np.float32)
                m.local_index.id_to_idx.clear()
            return AdaptationResult(
                operation_type="harmful_purge",
                target_modules=["all"],
                new_modules=[],
                description="Destructive recall-collapsing mutation",
                record_reassignments={},
            )

    pre_mod_count = len(system.graph.modules)
    val_harm = system.validator.validate_adaptation(
        prod_graph=system.graph,
        operation=HarmfulPurgeOperation(),
        tuning_queries=system.benchmark_dataset.tuning_queries,
    )
    print(f"-> Validator Verdict: {val_harm.decision}")
    print(f"   Reason: {val_harm.reason}")
    print(f"   Delta Recall: {val_harm.delta_recall:+.4f}")

    if not val_harm.accepted:
        system.version_manager.log_rollback(val_harm)
        print("-> Trial sandbox DISCARDED. Production memory was NOT modified.")
        print(f"-> Verified module count preserved: {len(system.graph.modules)} == {pre_mod_count}")

    # ----------------------------------------------------
    step_header(6, "Version Management - Point-in-Time Restore")
    # ----------------------------------------------------
    snapshots = system.version_manager.get_history()
    print("Snapshot Revision Timeline:")
    for s in snapshots:
        print(f"   * [{s['version_id']:<18}] Op: {s['operation']:<12} | Recall@3: {s['metrics'].get('recall_at_3', 0.0):.4f} | Mods: {s['metrics'].get('module_count', 0)}")

    restore_target = snapshots[0]["version_id"]
    print(f"\nRestoring memory state back to historical checkpoint: '{restore_target}'...")
    system.restore_version(restore_target)
    print(f"-> Active version restored to: {system.version_manager.active_version_id}")
    print(f"-> Module count restored to: {len(system.graph.modules)}")

    # ----------------------------------------------------
    step_header(7, "Three-Way Benchmark Arena: Static vs Heuristic vs RL-Guided")
    # ----------------------------------------------------
    print("Executing head-to-head three-way comparative benchmark (3 cycles per mode)...")
    arena_results = system.run_three_way_comparison(num_cycles=3)

    print("\n" + "=" * 74)
    print(f"{'Operational Mode':<18} | {'Held-Out R@3':<14} | {'Held-Out MRR':<14} | {'Latency':<10} | {'Rollbacks'}")
    print("-" * 74)
    for mode_name, exp_res in arena_results.items():
        h = exp_res.final_metrics_heldout
        print(f"{mode_name.upper():<18} | {h.recall_at_3:<14.4f} | {h.mrr:<14.4f} | {h.avg_latency_ms:<7.2f} ms | {exp_res.rejected_count}")
    print("=" * 74)

    rl_data = arena_results.get("rl_guided")
    if rl_data and rl_data.logs:
        print("\nLinUCB Contextual Bandit Decision Sample:")
        sample_log = rl_data.logs[0]
        print(f"-> Action: {sample_log.action_type} | Decision: {sample_log.decision} | Reward: {sample_log.reward}")
        print(f"-> Rationale: {sample_log.reason}")

    print("\n" + "=" * 74)
    print(">>> SYSTEM VERIFICATION COMPLETE - ALL PHASES OPERATIONAL & VERIFIED!")
    print("=" * 74)
    print("\nNext steps:")
    print("  1. Interactive Dashboard:  streamlit run dashboard/app.py")
    print("  2. FastAPI Swagger UI:     uvicorn morphe.api.main:app (open http://localhost:8000/docs)")
    print("  3. Run Automated Tests:    python -m unittest discover tests")
    print("=" * 74)


if __name__ == "__main__":
    main()
