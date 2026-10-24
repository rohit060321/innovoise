"""Standardized benchmark dataset with multi-domain documents, tuning, and held-out queries."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class BenchmarkQuery:
    query_id: str
    text: str
    ground_truth_doc_ids: List[str]
    category: str


@dataclass
class BenchmarkDataset:
    documents: List[Dict[str, Any]]
    tuning_queries: List[BenchmarkQuery]
    heldout_queries: List[BenchmarkQuery]


def get_standard_benchmark() -> BenchmarkDataset:
    """
    Returns a curated multi-domain benchmark dataset with realistic cross-thematic
    relationships ideal for testing split, merge, shortcut, and routed retrieval.
    """
    documents = [
        # Domain 1: Distributed Systems & Consensus
        {
            "id": "dist_sys_1",
            "title": "Raft Consensus Algorithm",
            "category": "distributed_systems",
            "text": "Raft is a consensus algorithm designed to be easy to understand. It decomposes consensus into leader election, log replication, and safety guarantees across replicated state machines.",
        },
        {
            "id": "dist_sys_2",
            "title": "Paxos Consensus Architecture",
            "category": "distributed_systems",
            "text": "The Paxos protocol ensures distributed state agreement across asynchronous networks subject to packet delay and server failure using proposal numbers, prepare requests, and accept phases.",
        },
        {
            "id": "dist_sys_3",
            "title": "Vector Clocks and Causality",
            "category": "distributed_systems",
            "text": "Vector clocks track causal relationships between concurrent events in distributed databases, enabling detection of concurrent updates and network partition reconciliation.",
        },
        # Domain 2: Deep Learning & Neural Architectures
        {
            "id": "deep_learn_1",
            "title": "Transformer Attention Mechanisms",
            "category": "deep_learning",
            "text": "Transformer architectures rely on scaled dot-product multi-head attention to model long-range token dependencies in natural language without sequential recurrence.",
        },
        {
            "id": "deep_learn_2",
            "title": "Diffusion Probabilistic Models",
            "category": "deep_learning",
            "text": "Denoising diffusion models synthesize high-fidelity image samples by progressively reversing a Markov forward diffusion chain with score matching objectives.",
        },
        {
            "id": "deep_learn_3",
            "title": "Neuromorphic Spiking Networks",
            "category": "deep_learning",
            "text": "Spiking neural networks simulate discrete temporal action potentials on event-driven neuromorphic silicon hardware, drastically lowering inference power consumption.",
        },
        # Domain 3: Neuroscience & Cognitive Memory
        {
            "id": "neuro_1",
            "title": "Hippocampal Indexing Theory",
            "category": "neuroscience",
            "text": "Hippocampal indexing posits that episodic memory traces in the cerebral cortex are reactivated through sparse hippocampal neural indices and memory consolidation during sleep.",
        },
        {
            "id": "neuro_2",
            "title": "Synaptic Plasticity and Long-Term Potentiation",
            "category": "neuroscience",
            "text": "Long-term potentiation induces persistent strengthening of synapse efficacy based on correlated pre- and post-synaptic firing rates, embodying Hebbian plasticity.",
        },
        {
            "id": "neuro_3",
            "title": "Working Memory Gating in Prefrontal Cortex",
            "category": "neuroscience",
            "text": "Prefrontal dopaminergic gating regulates working memory updating and maintenance, shielding active mental representations from sensory distraction.",
        },
        # Domain 4: Quantum Computing & Physics
        {
            "id": "quantum_1",
            "title": "Quantum Teleportation Protocol",
            "category": "quantum_computing",
            "text": "Quantum teleportation transmits an unknown qubit state across space using a shared Bell entangled pair and two bits of classical communication without transmitting physical matter.",
        },
        {
            "id": "quantum_2",
            "title": "Surface Codes and Quantum Error Correction",
            "category": "quantum_computing",
            "text": "Surface code topologies arrange physical qubits on a 2D planar grid to detect bit-flip and phase-flip errors via local stabilizer syndrome measurements.",
        },
        {
            "id": "quantum_3",
            "title": "Variational Quantum Eigensolver",
            "category": "quantum_computing",
            "text": "The Variational Quantum Eigensolver algorithm calculates ground state molecular energies on noisy intermediate-scale quantum devices using parameterized trial quantum circuits.",
        },
        # Domain 5: Cross-Domain Intersections (Bridging categories)
        {
            "id": "bridge_1",
            "title": "Neuromorphic Hardware for Edge Robotics",
            "category": "bridge_tech",
            "text": "Deploying event-driven spiking neural models on neuromorphic chips enables real-time edge robotics processing with microsecond temporal resolution and low thermal signature.",
        },
        {
            "id": "bridge_2",
            "title": "Quantum Neural Networks for Material Discovery",
            "category": "bridge_tech",
            "text": "Quantum neural networks parameterize variational circuits to accelerate molecular energy simulation and crystal structure generation for battery materials.",
        },
    ]

    # Tuning Queries (Used to validate candidate adaptations)
    tuning_queries = [
        BenchmarkQuery(
            query_id="tune_1",
            text="How does Raft elect a leader and maintain log replication in distributed state machines?",
            ground_truth_doc_ids=["rec_dist_sys_1_0"],
            category="distributed_systems",
        ),
        BenchmarkQuery(
            query_id="tune_2",
            text="How does multi-head attention work in Transformer language models?",
            ground_truth_doc_ids=["rec_deep_learn_1_0"],
            category="deep_learning",
        ),
        BenchmarkQuery(
            query_id="tune_3",
            text="Explain hippocampal memory indexing and consolidation into the cortex.",
            ground_truth_doc_ids=["rec_neuro_1_0"],
            category="neuroscience",
        ),
        BenchmarkQuery(
            query_id="tune_4",
            text="How does quantum teleportation transfer quantum state using entangled Bell pairs?",
            ground_truth_doc_ids=["rec_quantum_1_0"],
            category="quantum_computing",
        ),
        BenchmarkQuery(
            query_id="tune_5",
            text="Explain event-driven spike processing on neuromorphic silicon chips for neural computation.",
            ground_truth_doc_ids=["rec_deep_learn_3_0", "rec_bridge_1_0"],
            category="bridge_tech",
        ),
    ]

    # Held-Out Queries (Strictly preserved for unbiased evaluation and 3-way RL comparison)
    heldout_queries = [
        BenchmarkQuery(
            query_id="test_1",
            text="How do Paxos prepare and accept phases prevent conflicting updates during network failures?",
            ground_truth_doc_ids=["rec_dist_sys_2_0"],
            category="distributed_systems",
        ),
        BenchmarkQuery(
            query_id="test_2",
            text="What is vector clock causality tracking in distributed partitions?",
            ground_truth_doc_ids=["rec_dist_sys_3_0"],
            category="distributed_systems",
        ),
        BenchmarkQuery(
            query_id="test_3",
            text="How does reverse diffusion step-by-step generate images from noise?",
            ground_truth_doc_ids=["rec_deep_learn_2_0"],
            category="deep_learning",
        ),
        BenchmarkQuery(
            query_id="test_4",
            text="How does synaptic long-term potentiation strengthen neuronal connections in Hebbian learning?",
            ground_truth_doc_ids=["rec_neuro_2_0"],
            category="neuroscience",
        ),
        BenchmarkQuery(
            query_id="test_5",
            text="How do 2D surface codes identify stabilizer error syndromes on physical qubits?",
            ground_truth_doc_ids=["rec_quantum_2_0"],
            category="quantum_computing",
        ),
        BenchmarkQuery(
            query_id="test_6",
            text="How do variational quantum eigensolvers compute molecular ground states?",
            ground_truth_doc_ids=["rec_quantum_3_0", "rec_bridge_2_0"],
            category="quantum_computing",
        ),
    ]

    return BenchmarkDataset(
        documents=documents,
        tuning_queries=tuning_queries,
        heldout_queries=heldout_queries,
    )

