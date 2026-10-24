"""Embedding engines for MORPHE with automatic fallback and caching."""

from abc import ABC, abstractmethod
import hashlib
from typing import Dict, List, Optional
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for embedding generation."""

    def __init__(self, dimension: int = 128, cache_embeddings: bool = True) -> None:
        self.dimension = dimension
        self.cache_embeddings = cache_embeddings
        self._cache: Dict[str, np.ndarray] = {}

    @abstractmethod
    def _encode_raw(self, texts: List[str]) -> np.ndarray:
        """Encode raw texts into 2D float32 numpy array with normalized rows."""
        pass

    def encode(self, texts: List[str] | str) -> np.ndarray:
        """Encode single text or list of texts into normalized embedding vectors."""
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts)

        if not self.cache_embeddings:
            res = self._encode_raw(text_list)
            return res[0] if is_single else res

        results: List[Optional[np.ndarray]] = [None] * len(text_list)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for idx, t in enumerate(text_list):
            if t in self._cache:
                results[idx] = self._cache[t]
            else:
                missing_indices.append(idx)
                missing_texts.append(t)

        if missing_texts:
            encoded_missing = self._encode_raw(missing_texts)
            for idx, vec, t in zip(missing_indices, encoded_missing, missing_texts):
                self._cache[t] = vec
                results[idx] = vec

        res = np.array(results, dtype=np.float32)
        return res[0] if is_single else res


class DeterministicEmbedder(BaseEmbedder):
    """
    Fast, deterministic semantic embedder using token hashing and
    seeded random projection matrices. Produces stable, normalized embeddings
    without requiring external model downloads or PyTorch.
    """

    def __init__(self, dimension: int = 128, seed: int = 42, cache_embeddings: bool = True) -> None:
        super().__init__(dimension=dimension, cache_embeddings=cache_embeddings)
        self.seed = seed
        self.vocab_size = 8192
        rng = np.random.RandomState(seed)
        # Random projection matrix Gaussian distribution
        self.projection = rng.randn(self.vocab_size, dimension).astype(np.float32)
        # Orthogonalize / normalize columns
        self.projection /= np.linalg.norm(self.projection, axis=1, keepdims=True) + 1e-9

    def _hash_token(self, token: str) -> int:
        h = hashlib.md5(token.lower().encode("utf-8")).hexdigest()
        return int(h, 16) % self.vocab_size

    def _encode_raw(self, texts: List[str]) -> np.ndarray:
        batch_size = len(texts)
        embeddings = np.zeros((batch_size, self.dimension), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = [t.strip(".,!?:;\"'()[]{}") for t in text.lower().split() if t.strip()]
            if not tokens:
                vec = np.ones((self.dimension,), dtype=np.float32)
            else:
                bow = np.zeros(self.vocab_size, dtype=np.float32)
                for t in tokens:
                    idx = self._hash_token(t)
                    bow[idx] += 1.0
                # Project bag-of-words
                vec = bow @ self.projection
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            embeddings[i] = vec

        return embeddings


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder backed by HuggingFace / sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_embeddings: bool = True) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        dim = self.model.get_sentence_embedding_dimension()
        super().__init__(dimension=dim, cache_embeddings=cache_embeddings)

    def _encode_raw(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vectors, dtype=np.float32)


def get_embedder(
    provider: str = "auto",
    model_name: str = "all-MiniLM-L6-v2",
    dimension: int = 128,
    seed: int = 42,
    cache_embeddings: bool = True,
) -> BaseEmbedder:
    """Factory function to instantiate the preferred embedder."""
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name=model_name, cache_embeddings=cache_embeddings)
    elif provider == "deterministic":
        return DeterministicEmbedder(dimension=dimension, seed=seed, cache_embeddings=cache_embeddings)
    else:  # "auto"
        try:
            return SentenceTransformerEmbedder(model_name=model_name, cache_embeddings=cache_embeddings)
        except Exception:
            # Fallback seamlessly to deterministic embedder
            return DeterministicEmbedder(dimension=dimension, seed=seed, cache_embeddings=cache_embeddings)

