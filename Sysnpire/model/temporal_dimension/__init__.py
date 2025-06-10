"""
Temporal Dimension - Trajectory Operators (Section 3.1.4)

Transforms static positional encodings into dynamic trajectory operators that capture
how conceptual charges move through observational states.

Components:
- trajectory_operators.py: T_i(τ,s) trajectory integration implementation
- observational_persistence.py: Ψ_persistence layered memory functions
- phase_coordination.py: Temporal phase integration and interference
"""

from .trajectory_operators import TrajectoryOperatorEngine
from .observational_persistence import ObservationalPersistence
from .phase_coordination import TemporalPhaseCoordinator

__all__ = ['TrajectoryOperatorEngine', 'ObservationalPersistence', 'TemporalPhaseCoordinator']