"""
Phase Dimension - Complete Phase Integration e^(iθ_total(τ,C,s))

This module implements the complete phase integration component of the Q(τ, C, s) 
conceptual charge formula, integrating phases from all dimensions:

θ_total(τ,C,s) = θ_semantic(τ,s) + θ_emotional(τ,s) + θ_temporal(τ,s) + θ_interaction(τ,C,s) + θ_field(s)

Components:
- phase_integrator.py: Core phase integration logic
- phase_extractors.py: Extract phases from each dimension
- interaction_phase.py: Context-dependent phase calculations
- field_phase.py: Background field phase contributions
- phase_analysis.py: Phase coherence and dynamics analysis
"""

from .phase_integrator import (
    PhaseIntegrator,
    PhaseComponents,
    compute_total_phase
)
from .phase_extractors import (
    PhaseExtractor,
    extract_semantic_phase,
    extract_emotional_phase,
    extract_temporal_phase
)
from .interaction_phase import (
    InteractionPhaseCalculator,
    compute_interaction_phase
)
from .field_phase import (
    FieldPhaseCalculator,
    compute_field_phase
)
from .phase_analysis import (
    PhaseAnalyzer,
    analyze_phase_coherence,
    compute_phase_dynamics
)
from .field_coherence import (
    FieldCoherenceEngine,
    FieldCoherenceMetrics,
    InterferencePattern
)
from .phase_evolution import (
    PhaseEvolutionEngine,
    PhaseEvolutionState,
    EvolutionDynamics
)

__all__ = [
    # Main integration interface
    'PhaseIntegrator',
    'PhaseComponents',
    'compute_total_phase',
    
    # Phase extraction
    'PhaseExtractor',
    'extract_semantic_phase',
    'extract_emotional_phase', 
    'extract_temporal_phase',
    
    # Interaction and field phases
    'InteractionPhaseCalculator',
    'compute_interaction_phase',
    'FieldPhaseCalculator',
    'compute_field_phase',
    
    # Analysis
    'PhaseAnalyzer',
    'analyze_phase_coherence',
    'compute_phase_dynamics',
    
    # Critical Requirements - Field Coherence
    'FieldCoherenceEngine',
    'FieldCoherenceMetrics', 
    'InterferencePattern',
    
    # Critical Requirements - Evolution Dynamics
    'PhaseEvolutionEngine',
    'PhaseEvolutionState',
    'EvolutionDynamics'
]