"""
DTF Core - Fundamental Dynamic Field Theory Mathematics

Pure mathematical implementation of DTF equations and field dynamics
independent of any specific embedding model.

MATHEMATICAL FOUNDATION:
- Neural Field Equation: τu̇(x,t) = -u(x,t) + h + S(x,t) + ∫w(x-x')f(u(x',t))dx'
- Lateral Interaction Kernels: w(x-x') defining excitation/inhibition patterns
- Activation Functions: f(u) for neural field dynamics
- Steady-State Solutions: For basis function computation

COMPONENTS:
- DTFMathematicalCore: Core equation implementations
- LateralInteractionEngine: Interaction kernel computation
- FieldDynamicsEngine: Field evolution and steady-state solving
"""

from .dtf_core import DTFMathematicalCore
from .lateral_interactions import LateralInteractionEngine
from .field_dynamics import FieldDynamicsEngine

__all__ = [
    'DTFMathematicalCore',
    'LateralInteractionEngine', 
    'FieldDynamicsEngine'
]