"""
Field Compute - Audio Optimization

Field calculations and evolution computations.
Handles the mathematical transformations and propagation mechanics.

Components:
- charge_to_field: ConceptualCharge → Field imprint transformations
- evolution_solver: ∂M/∂t field evolution calculations
- spectral_methods: O(n³)→O(n) optimizations using spectral techniques
- propagation: Field propagation mechanics with O(k) complexity bounds
"""

from .charge_to_field import ChargeToField
from .evolution_solver import EvolutionSolver
from .spectral_methods import SpectralMethods
from .propagation import FieldPropagation

__all__ = [
    'ChargeToField',
    'EvolutionSolver',
    'SpectralMethods',
    'FieldPropagation'
]