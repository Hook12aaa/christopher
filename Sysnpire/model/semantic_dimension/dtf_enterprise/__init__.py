"""
DTF Enterprise - Dynamic Field Theory Mathematical Core

This module contains ONLY the core mathematical implementations for Dynamic Field Theory
as specified in the research paper. No placeholder values, no fake adapters, no simulated data.

IMPORTANT: The real DTF semantic basis functions are implemented in:
- semantic_dimension/semantic_basis_functions.py (DTFSemanticBasisExtractor)
- semantic_dimension/processing/field_pool.py (integration point)

This module provides supplementary DTF mathematical engines.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np

# Core DTF mathematics
from .core import (
    DTFMathematicalCore,
    FieldDynamicsEngine, 
    LateralInteractionEngine
)

__all__ = [
    # Core mathematical engines
    'DTFMathematicalCore',
    'FieldDynamicsEngine',
    'LateralInteractionEngine'
]