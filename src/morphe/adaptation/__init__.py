"""Self-adaptation operators and trigger detectors for MORPHE."""

from .operations import (
    AdaptationOperation,
    SplitOperation,
    MergeOperation,
    ShortcutOperation,
    apply_adaptation,
)
from .triggers import (
    AdaptationCandidate,
    detect_adaptation_candidates,
)

__all__ = [
    "AdaptationOperation",
    "SplitOperation",
    "MergeOperation",
    "ShortcutOperation",
    "apply_adaptation",
    "AdaptationCandidate",
    "detect_adaptation_candidates",
]

