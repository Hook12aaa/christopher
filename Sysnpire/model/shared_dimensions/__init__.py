"""
Shared Dimensions - Cross-Component Integration

This module contains shared mathematical components that integrate across
multiple dimensions (Semantic, Trajectory, Emotional) for the complete
Q(τ, C, s) conceptual charge formula.

Components:
- phase_dimension/: Complete phase integration e^(iθ_total(τ,C,s))
- persistence_dimension/: Observational persistence Ψ_persistence(s-s₀)
"""

from .phase_dimension import (
    compute_total_phase,
    PhaseIntegrator,
    PhaseComponents,
    analyze_phase_coherence,
    compute_phase_dynamics
)

__all__ = [
    'compute_total_phase',
    'PhaseIntegrator', 
    'PhaseComponents',
    'analyze_phase_coherence',
    'compute_phase_dynamics'
]