"""NetworkX-based hierarchical memory graph containing modules and shortcuts."""

from dataclasses import dataclass, field
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np

from .index import LocalVectorIndex


@dataclass
class ModuleNode:
    module_id: str
    centroid: np.ndarray
    capacity: int = 15
    local_index: LocalVectorIndex = field(default_factory=lambda: LocalVectorIndex())
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.local_index)

    def update_centroid(self) -> None:
        if len(self.local_index) > 0:
            self.centroid = self.local_index.compute_centroid()


class MemoryGraph:
    """Graph structure managing semantic modules, structural relationships, and shortcuts."""

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension
        self.graph: nx.Graph = nx.Graph()
        self.modules: Dict[str, ModuleNode] = {}

    def add_module(
        self,
        module_id: str,
        centroid: Optional[np.ndarray] = None,
        capacity: int = 15,
        local_index: Optional[LocalVectorIndex] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModuleNode:
        if local_index is None:
            local_index = LocalVectorIndex(dimension=self.dimension)
        if centroid is None:
            centroid = local_index.compute_centroid() if len(local_index) > 0 else np.zeros(self.dimension, dtype=np.float32)

        node = ModuleNode(
            module_id=module_id,
            centroid=centroid,
            capacity=capacity,
            local_index=local_index,
            metadata=metadata or {},
        )
        self.modules[module_id] = node
        self.graph.add_node(module_id, capacity=capacity, size=len(local_index))
        return node

    def remove_module(self, module_id: str) -> None:
        if module_id in self.modules:
            del self.modules[module_id]
        if self.graph.has_node(module_id):
            self.graph.remove_node(module_id)

    def add_edge(
        self,
        u: str,
        v: str,
        edge_type: str = "structural",
        weight: float = 1.0,
    ) -> None:
        if u in self.modules and v in self.modules and u != v:
            self.graph.add_edge(u, v, edge_type=edge_type, weight=float(weight))

    def remove_edge(self, u: str, v: str) -> None:
        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)

    def get_module(self, module_id: str) -> Optional[ModuleNode]:
        return self.modules.get(module_id)

    def get_all_modules(self) -> List[ModuleNode]:
        return list(self.modules.values())

    def get_neighbors(self, module_id: str, edge_type: Optional[str] = None) -> List[str]:
        if not self.graph.has_node(module_id):
            return []
        neighbors = []
        for n in self.graph.neighbors(module_id):
            edge_data = self.graph.get_edge_data(module_id, n) or {}
            if edge_type is None or edge_data.get("edge_type") == edge_type:
                neighbors.append(n)
        return neighbors

    def get_shortcuts(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        shortcuts = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == "shortcut":
                shortcuts.append((u, v, data))
        return shortcuts

    def shortest_path_length(self, u: str, v: str) -> Optional[int]:
        if not self.graph.has_node(u) or not self.graph.has_node(v):
            return None
        try:
            return nx.shortest_path_length(self.graph, u, v)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def clone(self) -> "MemoryGraph":
        """Perform an isolated deep copy for sandbox trial adaptations."""
        cloned = MemoryGraph(dimension=self.dimension)
        for mod_id, mod in self.modules.items():
            cloned.add_module(
                module_id=mod_id,
                centroid=np.copy(mod.centroid),
                capacity=mod.capacity,
                local_index=mod.local_index.clone(),
                metadata=dict(mod.metadata),
            )
        for u, v, data in self.graph.edges(data=True):
            cloned.add_edge(u, v, edge_type=data.get("edge_type", "structural"), weight=data.get("weight", 1.0))
        return cloned

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete graph structure, records, centroids, and edges to dictionary."""
        modules_data = {}
        for mod_id, mod in self.modules.items():
            rec_ids, vecs = mod.local_index.get_all_vectors()
            modules_data[mod_id] = {
                "capacity": mod.capacity,
                "centroid": mod.centroid.tolist(),
                "created_at": mod.created_at,
                "metadata": mod.metadata,
                "record_ids": rec_ids,
                "vectors": vecs.tolist(),
            }

        edges_data = []
        for u, v, data in self.graph.edges(data=True):
            edges_data.append({
                "u": u,
                "v": v,
                "edge_type": data.get("edge_type", "structural"),
                "weight": data.get("weight", 1.0),
            })

        return {
            "dimension": self.dimension,
            "modules": modules_data,
            "edges": edges_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryGraph":
        """Reconstruct MemoryGraph from serialized dictionary."""
        mg = cls(dimension=data.get("dimension", 128))
        modules_data = data.get("modules", {})
        for mod_id, m_dict in modules_data.items():
            local_idx = LocalVectorIndex(dimension=mg.dimension)
            rec_ids = m_dict.get("record_ids", [])
            vecs = np.array(m_dict.get("vectors", []), dtype=np.float32)
            for rid, vec in zip(rec_ids, vecs):
                local_idx.add(rid, vec)

            centroid = np.array(m_dict["centroid"], dtype=np.float32)
            mg.add_module(
                module_id=mod_id,
                centroid=centroid,
                capacity=m_dict.get("capacity", 15),
                local_index=local_idx,
                metadata=m_dict.get("metadata", {}),
            )

        for e in data.get("edges", []):
            mg.add_edge(e["u"], e["v"], edge_type=e.get("edge_type", "structural"), weight=e.get("weight", 1.0))

        return mg

