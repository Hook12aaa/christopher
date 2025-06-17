"""
Emotional Dimension - Field Modulation (Section 3.1.3)

Reconceptualizes emotion as field-modulating forces that transform semantic
landscapes through amplitude modulation, phase shifts, and metric warping.

Components:
- EmotionalDimensionHelper.py: Core emotional field modulation implementation
- analytics/: Field-theoretic pattern discovery in BGE embeddings
"""

from .analytics import BGEEmotionalAnalyzer
from .EmotionalDimensionHelper import (
    EmotionalDimensionHelper,
    EmotionalFieldModulation,
)

__all__ = [
    "EmotionalDimensionHelper",
    "EmotionalFieldModulation",
    "BGEEmotionalAnalyzer",
]