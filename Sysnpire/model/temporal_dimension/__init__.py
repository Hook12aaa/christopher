"""
Temporal Dimension - Pure Field Theory of Social Constructs

This module implements the complete field theory of social constructs using
cutting-edge computational field theory libraries (ComFiT, JAX, PyTorch Geometric).

NO transformer concepts - pure field physics with:
- Social construct field dynamics
- Topological defects in conceptual space  
- Collective response phenomena
- Inter-field coupling and orchestration
- Real PDE solvers and automatic differentiation

Mathematical Foundation:
∂φ/∂t = -δF/δφ + Γ(φ) + ξ(r,t)

Where φ represents social construct fields, F is the free energy functional,
Γ represents field coupling terms, and ξ represents stochastic social interactions.
"""

# Pure Field Theory Implementation - NO transformer concepts
from .social_construct_field import (
    SocialConstructField,
    SocialConstructFieldFactory
)

from .field_orchestrator import (
    FieldOrchestrator,
    CollectiveResponseOrchestrator
)

# Legacy modular architecture (for backward compatibility during transition)
# These will be deprecated in favor of pure field theory
from .trajectory_operators import (
    TrajectoryOperator,
    FrequencyEvolution, 
    PhaseAccumulator,
    ComplexIntegrator,
    TrajectoryOperatorEngine
)

from .observational_persistence import (
    PersistenceLayer,
    GaussianMemory,
    ExponentialCosineMemory,
    DualDecayPersistence,
    ObservationalPersistence
)

from .phase_coordination import (
    PhaseOrchestra,
    InterferenceManaager,  # Preserves intentional typo from README
    MemoryResonance,
    CrossDimensionalCoupling,
    TemporalPhaseCoordinator,
    EnhancedTemporalPhaseCoordinator
)

from .developmental_distance import (
    DevelopmentalDistanceCalculator
)

from .field_integration import (
    TemporalFieldIntegrator
)

from .field_coupling import (
    FieldCouplingIntegrator,
    TemporalFieldCoupler,
    BreathingPatternGenerator,
    SemanticTemporalSynchronizer
)

from .temporal_orchestrator import (
    TemporalOrchestrator
)

# Main exports - PURE FIELD THEORY (recommended)
__all__ = [
    # === PURE FIELD THEORY IMPLEMENTATION ===
    'SocialConstructField',
    'SocialConstructFieldFactory', 
    'FieldOrchestrator',
    'CollectiveResponseOrchestrator',
    
    # === LEGACY MODULAR ARCHITECTURE (transitional) ===
    # Trajectory operators
    'TrajectoryOperator',
    'FrequencyEvolution',
    'PhaseAccumulator', 
    'ComplexIntegrator',
    'TrajectoryOperatorEngine',
    
    # Observational persistence
    'PersistenceLayer',
    'GaussianMemory',
    'ExponentialCosineMemory',
    'DualDecayPersistence',
    'ObservationalPersistence',
    
    # Phase coordination
    'PhaseOrchestra',
    'InterferenceManaager',
    'MemoryResonance',
    'CrossDimensionalCoupling',
    'TemporalPhaseCoordinator',
    'EnhancedTemporalPhaseCoordinator',
    
    # Other components
    'DevelopmentalDistanceCalculator',
    'TemporalFieldIntegrator',
    'FieldCouplingIntegrator',
    'TemporalFieldCoupler',
    'BreathingPatternGenerator',
    'SemanticTemporalSynchronizer',
    'TemporalOrchestrator'
]

# Version and metadata
__version__ = "2.0.0"  # Major version bump for pure field theory implementation
__description__ = "Pure Field Theory of Social Constructs - NO transformer concepts"
__libraries__ = [
    "ComFiT 1.9.6 - Computational Field Theory with topological defects",
    "JAX - Automatic differentiation for complex manifolds",
    "PyTorch Geometric - Manifold operations and geometric learning"
]

# Recommended usage pattern
def create_social_construct_field_system(conceptual_dimensions: int = 3,
                                       domain_size: float = 10.0,
                                       coupling_strength: float = 0.1) -> FieldOrchestrator:
    """
    Create a complete social construct field system using pure field theory.
    
    This is the recommended entry point for field theory of social constructs.
    NO transformer concepts - pure field physics.
    
    Args:
        conceptual_dimensions: Dimensionality of conceptual space (NOT embedding dimension)
        domain_size: Size of observational domain in field space
        coupling_strength: Strength of inter-field coupling
        
    Returns:
        FieldOrchestrator managing multiple social construct field systems
    """
    orchestrator = FieldOrchestrator(
        conceptual_space_dimensions=conceptual_dimensions,
        observational_domain_size=domain_size,
        inter_field_coupling_strength=coupling_strength
    )
    
    return orchestrator

# Field theory factory function
def create_field_theory_system(field_contexts: list,
                             conceptual_dimensions: int = 3,
                             domain_size: float = 10.0) -> FieldOrchestrator:
    """
    Create field theory system with multiple conceptual contexts.
    
    Args:
        field_contexts: List of conceptual contexts to create fields for
        conceptual_dimensions: Dimensionality of conceptual space
        domain_size: Size of observational domain
        
    Returns:
        FieldOrchestrator with field systems for all contexts
    """
    orchestrator = create_social_construct_field_system(
        conceptual_dimensions=conceptual_dimensions,
        domain_size=domain_size
    )
    
    # Add field systems for each context
    for context in field_contexts:
        orchestrator.add_field_system(context)
    
    return orchestrator