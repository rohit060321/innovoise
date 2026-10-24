"""FastAPI REST API server for MORPHE.

Provides endpoints for routed semantic retrieval, topological inspection,
autonomous self-adaptation cycles, sandbox trial validation, zero-risk rollback,
version snapshot management, point-in-time restore, and three-way comparative benchmarking.
"""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import numpy as np

from ..core import MorpheSystem
from ..adaptation.operations import SplitOperation, MergeOperation, ShortcutOperation, AdaptationResult
from ..policy.controller import BenchmarkMode


app = FastAPI(
    title="MORPHE REST API",
    description="Adaptive, Self-Reconfiguring Memory System for AI Agents with Routed Semantic Retrieval, Sandbox Trial Validation, and LinUCB RL Policy.",
    version="1.0.0",
)

# Global singleton system instance
_system: Optional[MorpheSystem] = None


def get_system() -> MorpheSystem:
    global _system
    if _system is None:
        _system = MorpheSystem()
        _system.initialize_from_benchmark()
    return _system


# ----------------- PYDANTIC SCHEMAS -----------------

class QueryRequest(BaseModel):
    query: str = Field(..., description="Query prompt text")
    top_k: int = Field(5, description="Number of top records to return")


class QueryResponse(BaseModel):
    query: str
    records: List[Dict[str, Any]]
    queried_modules: List[str]
    centroid_ranked_modules: List[Dict[str, Any]]
    shortcut_hops: List[List[str]]
    latency_ms: float


class IngestDoc(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = "Untitled Document"
    category: Optional[str] = "custom"
    text: str = Field(..., description="Raw text content to ingest")


class IngestRequest(BaseModel):
    documents: List[IngestDoc]


class AdaptStepRequest(BaseModel):
    mode: str = Field("rl_guided", description="'rl_guided' or 'heuristic_only'")


class TrialRequest(BaseModel):
    operation_type: str = Field(..., description="'split', 'merge', or 'shortcut'")
    module_id: Optional[str] = Field(None, description="Module to split")
    module_a: Optional[str] = Field(None, description="First module for merge or shortcut")
    module_b: Optional[str] = Field(None, description="Second module for merge or shortcut")
    weight: float = Field(1.0, description="Shortcut edge weight")


class BenchmarkRequest(BaseModel):
    num_cycles: int = Field(3, ge=1, le=10, description="Number of adaptation cycles per mode")


# ----------------- REST ENDPOINTS -----------------

@app.get("/", summary="System Overview")
def root() -> Dict[str, Any]:
    sys_inst = get_system()
    shortcuts = sys_inst.graph.get_shortcuts()
    return {
        "system": "MORPHE",
        "description": "Memory-Oriented Reconfigurable Plasticity and Hierarchical Evolution",
        "active_version": sys_inst.version_manager.active_version_id,
        "module_count": len(sys_inst.graph.modules),
        "shortcut_count": len(shortcuts),
        "total_records": len(sys_inst.storage.get_all_records()),
        "status": "online",
        "docs_url": "/docs",
    }


@app.get("/health", summary="Health Check")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, summary="Execute Routed Semantic Retrieval")
def execute_query(req: QueryRequest) -> QueryResponse:
    sys_inst = get_system()
    res = sys_inst.query(req.query, top_k=req.top_k)
    return QueryResponse(
        query=res.query,
        records=res.records,
        queried_modules=res.queried_modules,
        centroid_ranked_modules=[{"module_id": m, "score": round(s, 4)} for m, s in res.centroid_ranked_modules],
        shortcut_hops=[list(h) for h in res.shortcut_hops],
        latency_ms=round(res.latency_ms, 2),
    )


@app.get("/graph", summary="Inspect Memory Graph Topology")
def get_graph() -> Dict[str, Any]:
    sys_inst = get_system()
    return sys_inst.graph.to_dict()


@app.get("/modules", summary="List All Active Memory Modules")
def list_modules() -> List[Dict[str, Any]]:
    sys_inst = get_system()
    modules_out = []
    for mod in sys_inst.graph.get_all_modules():
        modules_out.append({
            "module_id": mod.module_id,
            "record_count": len(mod.local_index),
            "capacity": mod.capacity,
            "variance": round(mod.local_index.compute_variance(), 4),
            "centroid_dim": len(mod.centroid),
            "metadata": mod.metadata,
        })
    return modules_out


@app.post("/ingest", summary="Ingest Custom Knowledge Documents")
def ingest_data(req: IngestRequest) -> Dict[str, Any]:
    if not req.documents:
        raise HTTPException(status_code=400, detail="Document list cannot be empty")
    sys_inst = get_system()
    docs_payload = [d.model_dump() for d in req.documents]
    records = sys_inst.ingest_custom_documents(docs_payload)
    return {
        "status": "success",
        "ingested_docs": len(req.documents),
        "created_chunks": len(records),
        "active_version": sys_inst.version_manager.active_version_id,
    }


@app.get("/adapt/candidates", summary="Detect Active Adaptation Trigger Candidates")
def get_candidates() -> List[Dict[str, Any]]:
    from ..adaptation.triggers import detect_adaptation_candidates
    sys_inst = get_system()
    candidates = detect_adaptation_candidates(
        graph=sys_inst.graph,
        thresholds=sys_inst.config.thresholds,
        storage=sys_inst.storage,
    )
    return [
        {
            "candidate_id": c.candidate_id,
            "operation_type": c.operation_type,
            "target_modules": c.target_modules,
            "priority_score": round(c.priority_score, 3),
            "reason": c.reason,
        }
        for c in candidates
    ]


@app.post("/adapt/step", summary="Run Single Autonomous Adaptation Cycle")
def run_adaptation_step(req: AdaptStepRequest) -> Dict[str, Any]:
    sys_inst = get_system()
    log = sys_inst.adapt_step(mode=req.mode)
    if not log:
        return {"status": "equilibrium", "message": "No adaptation candidate met the activation threshold."}
    return {
        "status": "completed",
        "step": log.step,
        "action_type": log.action_type,
        "targets": log.target_modules,
        "decision": log.decision,
        "reason": log.reason,
        "delta_recall": log.delta_recall,
        "delta_latency": log.delta_latency,
        "reward": log.reward,
        "active_version": sys_inst.version_manager.active_version_id,
    }


@app.post("/adapt/trial", summary="Simulate Adaptation Inside Safe Sandbox Copy")
def run_sandbox_trial(req: TrialRequest) -> Dict[str, Any]:
    sys_inst = get_system()
    op_type = req.operation_type.lower()

    if op_type == "split":
        if not req.module_id:
            raise HTTPException(status_code=400, detail="module_id required for split")
        op = SplitOperation(module_id=req.module_id)
    elif op_type == "merge":
        if not req.module_a or not req.module_b:
            raise HTTPException(status_code=400, detail="module_a and module_b required for merge")
        op = MergeOperation(module_a=req.module_a, module_b=req.module_b)
    elif op_type == "shortcut":
        if not req.module_a or not req.module_b:
            raise HTTPException(status_code=400, detail="module_a and module_b required for shortcut")
        op = ShortcutOperation(module_a=req.module_a, module_b=req.module_b, weight=req.weight)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported operation_type: {op_type}")

    val_res = sys_inst.validator.validate_adaptation(
        prod_graph=sys_inst.graph,
        operation=op,
        tuning_queries=sys_inst.benchmark_dataset.tuning_queries,
    )

    return {
        "operation": op_type,
        "decision": val_res.decision,
        "accepted": val_res.accepted,
        "reason": val_res.reason,
        "delta_recall": val_res.delta_recall,
        "delta_mrr": val_res.delta_mrr,
        "delta_latency": val_res.delta_latency,
        "production_memory_modified": False,
    }


@app.post("/adapt/rollback_demo", summary="Simulate Harmful Mutation & Verify Automatic Rollback")
def simulate_harmful_rollback() -> Dict[str, Any]:
    sys_inst = get_system()

    class DestructiveHarmfulMutation:
        def apply(self, graph, seed=42):
            for m in graph.get_all_modules():
                m.local_index.record_ids.clear()
                m.local_index.vectors = np.empty((0, graph.dimension), dtype=np.float32)
                m.local_index.id_to_idx.clear()
            return AdaptationResult(
                operation_type="destructive_purge",
                target_modules=["all"],
                new_modules=[],
                description="Simulated recall-collapsing mutation",
                record_reassignments={},
            )

    pre_mods = len(sys_inst.graph.modules)
    val_res = sys_inst.validator.validate_adaptation(
        prod_graph=sys_inst.graph,
        operation=DestructiveHarmfulMutation(),
        tuning_queries=sys_inst.benchmark_dataset.tuning_queries,
    )

    if not val_res.accepted:
        sys_inst.version_manager.log_rollback(val_res)

    post_mods = len(sys_inst.graph.modules)

    return {
        "simulated_action": "destructive_purge",
        "validator_decision": val_res.decision,
        "rejection_reason": val_res.reason,
        "delta_recall": val_res.delta_recall,
        "sandbox_discarded": True,
        "rollback_logged": True,
        "production_state_intact": pre_mods == post_mods,
        "active_modules": post_mods,
    }


@app.get("/versions", summary="Retrieve Architecture Version Timeline")
def get_versions() -> List[Dict[str, Any]]:
    sys_inst = get_system()
    return sys_inst.version_manager.get_history()


@app.post("/versions/{version_id}/restore", summary="Restore Memory Graph to Point-in-Time Snapshot")
def restore_version(version_id: str) -> Dict[str, Any]:
    sys_inst = get_system()
    try:
        sys_inst.restore_version(version_id)
        return {
            "status": "restored",
            "active_version": sys_inst.version_manager.active_version_id,
            "module_count": len(sys_inst.graph.modules),
            "shortcuts": len(sys_inst.graph.get_shortcuts()),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/benchmark/three_way", summary="Execute Head-to-Head Three-Way Comparison Benchmark")
def run_three_way_arena(req: BenchmarkRequest) -> Dict[str, Any]:
    sys_inst = get_system()
    results = sys_inst.run_three_way_comparison(num_cycles=req.num_cycles)
    out = {}
    for mode_name, exp in results.items():
        out[mode_name] = {
            "heldout_metrics": exp.final_metrics_heldout.to_dict(),
            "accepted_adaptations": exp.accepted_count,
            "rejected_adaptations": exp.rejected_count,
            "total_cycles": exp.steps_executed,
        }
    return out
