"""Embeddings package for MORPHE."""

from .embedder import (
    BaseEmbedder,
    DeterministicEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)

__all__ = [
    "BaseEmbedder",
    "DeterministicEmbedder",
    "SentenceTransformerEmbedder",
    "get_embedder",
]

