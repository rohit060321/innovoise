"""MORPHE: Memory-Oriented Reconfigurable Plasticity and Hierarchical Evolution."""

from .core import MorpheSystem, MorpheCore
from .config import MorpheConfig
from .graph.memory_graph import MemoryGraph
from .retrieval.router import RoutedRouter
from .evaluation.metrics import EvaluationMetrics

__version__ = "0.1.0"
__all__ = [
    "MorpheSystem",
    "MorpheCore",
    "MorpheConfig",
    "MemoryGraph",
    "RoutedRouter",
    "EvaluationMetrics",
]
