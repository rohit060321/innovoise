"""Per-module local vector index with exact cosine search and statistical analysis."""

from typing import Dict, List, Optional, Tuple
import numpy as np


class LocalVectorIndex:
    """Stores vectors and computes vector similarity specifically inside one module."""

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension
        self.record_ids: List[str] = []
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self.id_to_idx: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.record_ids)

    def add(self, record_id: str, vector: np.ndarray) -> None:
        """Add or update record embedding in local index."""
        norm = np.linalg.norm(vector)
        norm_vec = (vector / norm if norm > 1e-9 else vector).astype(np.float32)

        if record_id in self.id_to_idx:
            idx = self.id_to_idx[record_id]
            self.vectors[idx] = norm_vec
        else:
            idx = len(self.record_ids)
            self.id_to_idx[record_id] = idx
            self.record_ids.append(record_id)
            if self.vectors.shape[0] == 0:
                self.vectors = np.expand_dims(norm_vec, axis=0)
            else:
                self.vectors = np.vstack([self.vectors, norm_vec])

    def remove(self, record_id: str) -> None:
        """Remove a record from the local index."""
        if record_id not in self.id_to_idx:
            return
        idx = self.id_to_idx[record_id]
        del self.id_to_idx[record_id]
        self.record_ids.pop(idx)
        self.vectors = np.delete(self.vectors, idx, axis=0)
        # Re-index mapping
        self.id_to_idx = {rid: i for i, rid in enumerate(self.record_ids)}

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search local records by cosine similarity."""
        if len(self.record_ids) == 0:
            return []

        norm = np.linalg.norm(query_vec)
        q = (query_vec / norm if norm > 1e-9 else query_vec).astype(np.float32)

        # Dot product of normalized vectors = cosine similarity
        scores = self.vectors @ q
        k = min(top_k, len(self.record_ids))
        top_indices = np.argpartition(-scores, k - 1)[:k]
        sorted_indices = top_indices[np.argsort(-scores[top_indices])]

        return [(self.record_ids[i], float(scores[i])) for i in sorted_indices]

    def compute_centroid(self) -> np.ndarray:
        """Compute the normalized mean vector of all records in this index."""
        if len(self.record_ids) == 0:
            return np.zeros(self.dimension, dtype=np.float32)
        mean_vec = np.mean(self.vectors, axis=0)
        norm = np.linalg.norm(mean_vec)
        return (mean_vec / norm if norm > 1e-9 else mean_vec).astype(np.float32)

    def compute_variance(self) -> float:
        """Compute mean squared Euclidean distance from the centroid."""
        if len(self.record_ids) <= 1:
            return 0.0
        centroid = self.compute_centroid()
        # Distance to centroid for each vector
        diffs = self.vectors - centroid
        squared_dists = np.sum(diffs ** 2, axis=1)
        return float(np.mean(squared_dists))

    def get_all_vectors(self) -> Tuple[List[str], np.ndarray]:
        """Return all record IDs and their corresponding vectors."""
        return list(self.record_ids), np.copy(self.vectors)

    def clone(self) -> "LocalVectorIndex":
        """Deep copy for sandbox trial mutations."""
        cloned = LocalVectorIndex(dimension=self.dimension)
        cloned.record_ids = list(self.record_ids)
        cloned.vectors = np.copy(self.vectors)
        cloned.id_to_idx = dict(self.id_to_idx)
        return cloned

