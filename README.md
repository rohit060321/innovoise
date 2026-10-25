# MORPHE: Memory-Oriented Reconfigurable Plasticity and Hierarchical Evolution

> **PS 3: An Adaptive, Self-Reconfiguring Memory System for AI Agents**  
> *"RL proposes, validation disposes: a learned policy chooses which structural change to try, and a deterministic validator decides whether to keep it."*

---

## Quick Start Guide

### 1. Run the Terminal Verification Walkthrough
Execute the self-contained live demonstration covering all blueprint phases:
```powershell
.venv\Scripts\python scripts/demo.py
```
This runs the entire adaptation loop end-to-end:
1. Ingestion of multi-domain benchmark documents and discovery of structural weaknesses.
2. Centroid-routed retrieval without brute-force global scans.
3. Proposing, sandbox-validating, and committing an overloaded module **Split**.
4. Synthesizing a learned topological **Shortcut** between co-accessed modules.
5. Injecting an intentionally harmful mutation, verifying sandbox **Rejection** and zero-risk **Rollback**.
6. Executing a point-in-time **Restore** back to a historical version snapshot.
7. LinUCB **Contextual Bandit** decision making and 3-way comparative benchmark.

---

### 2. Launch the Interactive Streamlit Dashboard
```powershell
.venv\Scripts\streamlit run dashboard/app.py
```
Features available in the dashboard:
- **Overview & Topology**: Interactive 2D graph network (Plotly) showing module clusters, variance, and high-visibility shortcuts.
- **Semantic Retrieval**: Visual trace of centroid proximity ranking, visited vs pruned modules, and ranked documents.
- **Knowledge Ingestion & Upload**: Multi-file uploader (.txt, .json, .csv, .md) and direct text composer.
- **Adaptation Cockpit**: Candidate trigger table, manual sandbox simulator, and audit log.
- **Version History & Restore**: Snapshot timeline with single-click restore to any past checkpoint.
- **Before & After Comparison**: Side-by-side metric tables and differential analysis.
- **Multi-Policy Benchmark Arena**: Head-to-head comparison between Static Memory, Heuristic-Only, and LinUCB RL-Guided.
- **Interactive System Walkthrough**: Step-by-step verification demonstrating structural plasticity, routed retrieval, and rollback safety.

---

### 3. Launch the FastAPI REST Server
```powershell
.venv\Scripts\uvicorn morphe.api.main:app --port 8000
```
Open **Interactive Swagger API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- `POST /query`: Routed retrieval with latency and visited module path.
- `GET /graph`: NetworkX graph structure in node-link JSON.
- `GET /modules`: Active memory module statistics and records.
- `POST /ingest`: Ingest custom text or documents.
- `GET /adapt/candidates`: Active candidate triggers.
- `POST /adapt/step`: Trigger autonomous adaptation step (heuristic or RL).
- `POST /adapt/trial`: Simulate operation in sandbox without modifying production.
- `POST /adapt/rollback_demo`: Simulate harmful mutation and prove rollback.
- `GET /versions`: Snapshot history.
- `POST /versions/{version_id}/restore`: Revert memory state.
- `POST /benchmark/three_way`: Run comparative benchmark.

---

### 4. Run the Automated Test Suite
```powershell
.venv\Scripts\python -m unittest discover tests
```
All 22 unit tests verify:
- Ingestion & hierarchical graph construction
- Routed semantic retrieval without global scans
- Structural operators (Split, Merge, Shortcut)
- Sandbox trial validation & rejection of regressions
- Rollback safety & point-in-time snapshot restore
- Contextual bandit (LinUCB) policy updates
- FastAPI REST endpoints

---

## System Architecture

```
                       +-----------------------------+
                       |     Incoming Query (q)      |
                       +--------------+--------------+
                                      |
                                      v
                       +--------------+--------------+
                       | Step 1: Top-k Centroids     |
                       | (Prune unpromising modules) |
                       +--------------+--------------+
                                      |
                                      v
                       +--------------+--------------+
                       | Step 2: 1-Hop Shortcuts     |
                       | (Traverse learned pathways) |
                       +--------------+--------------+
                                      |
                                      v
                       +--------------+--------------+
                       | Step 3: Local Sub-Indexes   |
                       | (Search only visited mods)  |
                       +--------------+--------------+
                                      |
                                      v
                       +--------------+--------------+
                       | Step 4: Merged Results Rank |
                       +--------------+--------------+
                                      |
        +-----------------------------+-----------------------------+
        | Telemetry: Recall@k, MRR, Latency, Co-Access Matrix       |
        +-----------------------------+-----------------------------+
                                      |
                       +--------------v--------------+
                       | Candidate Trigger Detectors |
                       | (Split, Merge, Shortcut)    |
                       +--------------+--------------+
                                      |
                       +--------------v--------------+
                       | LinUCB Contextual Bandit    |
                       | (Ranks candidate mutations) |
                       +--------------+--------------+
                                      |
                       +--------------v--------------+
                       | Isolated Sandbox Graph Copy |
                       +--------------+--------------+
                                      |
                             [Tuning Test Suite]
                                      |
                                      v
                       +--------------+--------------+
                       | Acceptance Gate Validator   |
                       +-------+--------------+------+
                               |              |
                Validated Improvement         Harmful Regression
                               |              |
                               v              v
                  +------------+----+    +----+------------+
                  | Commit Snapshot |    | Discard Sandbox |
                  | & Version DB    |    | & Log Rollback  |
                  +-----------------+    +-----------------+
```

---

## Key Design Innovations

### 1. Make Structure Affect Retrieval
If an agent searches all records globally across a monolithic vector index, restructuring changes nothing. In MORPHE, retrieval is **routed through the graph**:
1. Query is embedded and compared to module centroids.
2. Only top-k centroids are selected (fixed budget).
3. Shortcut edges are followed 1-hop from selected modules.
4. Only traversed modules' local vector indices are searched.
5. Hits are merged, deduplicated, and ranked.

This guarantees that each structural operation has a measurable, observable effect:
- **Overloaded Module**: Blurry centroid and slower local search. **Splitting** creates coherent child clusters with sharp centroids, boosting retrieval relevance.
- **Redundant Modules**: Near-duplicates consume multiple top-k routing slots on the same topic. **Merging** consolidates them, freeing routing capacity.
- **Co-Accessed Modules**: Relevant answers span across topics, but top-k centroid routing only visits one. A **Shortcut** allows 1-hop traversal to recover the missing documents without global search!

### 2. RL Proposes, Validation Disposes
- **Candidate Generator**: Predefined heuristics monitor module size, variance, centroid similarity, and co-access frequency.
- **Contextual Bandit (LinUCB)**: Learns which structural action (NO_OP, SPLIT, MERGE, SHORTCUT) yields the highest expected reward given the topological state.
- **Deterministic Sandbox Validator**: Evaluates the proposed mutation on an isolated copy (`trial_graph = prod_graph.clone()`).
- **Zero-Risk Rollback**: If a mutation regresses recall or surges latency, the sandbox is discarded, the rollback is logged to SQLite, and production memory remains 100% untouched.

---

## Project Directory Structure

```
morphe/
├── configs/
│   ├── default.yaml            # Embedding model, dimensions, top-k routing, seed
│   └── thresholds.yaml         # Split/merge/shortcut triggers, validation gates, RL rewards
├── src/morphe/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py             # FastAPI REST endpoints & Swagger UI
│   ├── config.py               # Hyperparameter configuration schema
│   ├── core.py                 # Unified MorpheSystem facade
│   ├── ingestion.py            # Chunking, embeddings, clustering, incremental updates
│   ├── storage/
│   │   └── db.py               # SQLite storage for records, snapshots, co-access, audit trail
│   ├── embedding/
│   │   └── embedder.py         # SentenceTransformers + Deterministic local embedder
│   ├── graph/
│   │   ├── index.py            # Per-module local vector index (exact cosine similarity)
│   │   └── memory_graph.py     # NetworkX topological memory graph & module nodes
│   ├── retrieval/
│   │   └── router.py           # Routed semantic retrieval & path tracing
│   ├── evaluation/
│   │   ├── dataset.py          # Benchmark documents, tuning, and held-out queries
│   │   ├── evaluator.py        # Evaluation harness
│   │   └── metrics.py          # Recall@k, MRR, nDCG, latency, utilization
│   ├── adaptation/
│   │   ├── operations.py       # Split, Merge, Shortcut operators
│   │   └── triggers.py         # Priority-based candidate trigger detectors
│   ├── safety/
│   │   ├── validator.py        # Sandbox trial evaluator & acceptance gate
│   │   └── versioning.py       # Snapshot persistence, rollback & point-in-time restore
│   └── policy/
│       ├── bandit.py           # LinUCB Contextual Bandit policy
│       └── controller.py       # Autonomous adaptation loop & 3-way arena
├── dashboard/
│   └── app.py                  # Streamlit visualizer & 5-minute Judge Showcase
├── scripts/
│   ├── demo.py                 # 5-minute terminal live demonstration
│   └── run_experiment.py       # CLI runner for 3-way benchmark experiments
├── tests/
│   ├── test_core.py            # Integration & initialization tests
│   ├── test_routed_retrieval.py# Routing & shortcut traversal tests
│   ├── test_adaptation.py      # Split, Merge, Shortcut unit tests
│   ├── test_safety_rollback.py # Sandbox trial, rejection & restore tests
│   ├── test_bandit.py          # LinUCB policy tests
│   └── test_api.py             # FastAPI REST endpoint tests
├── requirements.txt            # Python dependencies
└── README.md
```

---
   > *"Our reinforcement learning differentiator is a LinUCB contextual bandit. In our Three-Way Comparative Arena, Static Memory stays flat, Heuristics make greedy errors, while our RL-guided policy achieves superior held-out retrieval quality with minimal rollbacks."*
