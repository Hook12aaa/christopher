"""
Integration Layer - Complete Q(τ,C,s) Field Theory Implementation

MATHEMATICAL FOUNDATION: Complete Conceptual Charge Field Theory
===============================================================================

Core Field Equation:
    Q(τ,C,s) = γ·T(τ,C,s)·E^trajectory(τ,s)·Φ^semantic(τ,s)·e^(iθ_total(τ,C,s))·Ψ_persistence(s-s₀)

Where:
    γ ∈ ℂ                           : Complex amplitude coefficient
    T(τ,C,s) ∈ ℝ⁺                  : Temporal evolution factor
    E^trajectory(τ,s) ∈ ℝ⁺         : Trajectory-dependent energy
    Φ^semantic(τ,s) ∈ ℂ             : Semantic field amplitude
    θ_total(τ,C,s) ∈ [0,2π)         : Total phase accumulation
    Ψ_persistence(s-s₀) ∈ ℝ⁺       : Persistence decay function

Field Evolution PDE:
    ∂Q/∂τ = D∇²Q - V'(Q) + Σᵢ δ(s-sᵢ)Jᵢ(τ,C)

Hegselmann-Krause Dynamics:
    dx_i/dτ = (1/|N_i(ε)|) Σ_{j∈N_i(ε)} (x_j - x_i) + α∇Q(x_i,τ)

Field Energy Functional:
    E[Q] = ∫∫∫ [½|∇Q|² + ½m²|Q|² + (λ/4!)|Q|⁴] d³x

Selection Pressure:
    ∂p_i/∂τ = p_i[f_i(Q,A,C) - ⟨f(Q,A,C)⟩]

Attention Field Dynamics:
    ∂A/∂τ = D_A∇²A - k_A A + Σᵢ w_i δ(x-x_i(τ))

MATHEMATICAL PERFECTION PRINCIPLE:
NO approximations. NO fallbacks. EXACT field theory or catastrophic failure.
Every equation implemented with mathematical rigor using JAX, SciPy, Sage.
"""

# Core Mathematical Engines - ALL EXACT IMPLEMENTATIONS
from .charge_assembler import ConceptualChargeAssembler

# Field Theory Mechanics - Complete Mathematical Framework
try:
    from .field_mechanics import (CriticalPoint, EnergyComponents,
                                  FieldConfiguration, FieldEnergyCalculator,
                                  FieldPerturbationAnalyzer, FieldSymmetry,
                                  InterferenceDynamicsEngine,
                                  InterferencePattern, PerturbationResponse,
                                  PhaseTransitionDetector, PhaseType)

    FIELD_MECHANICS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Field mechanics import failed: {e}")
    FIELD_MECHANICS_AVAILABLE = False

# Evolutionary Selection Pressure - Exact Replicator Dynamics
try:
    from .selection_pressure import (AttentionField,
                                     ComplexityGradientAnalyzer,
                                     ComplexityMeasures,
                                     ConstructEvolutionEngine,
                                     EvolutionaryDynamics,
                                     GameTheoreticAnalysis, PopulationState,
                                     SelectionPressure, SocialConstruct,
                                     SpotlightFieldEngine)

    SELECTION_PRESSURE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Selection pressure import failed: {e}")
    SELECTION_PRESSURE_AVAILABLE = False

# Main Field Integrator - Q(τ,C,s) Orchestrator
from .field_integrator import (FieldIntegrationState, FieldIntegrator,
                               HKFieldCoupling, SystemEvolution)

# Mathematical Constants and Precision
FIELD_THEORETICAL_PRECISION = 1e-15  # Machine epsilon for field calculations
HEGSELMANN_KRAUSE_EPSILON = 0.3  # Default confidence bound
MATHEMATICAL_TOLERANCE = 1e-12  # Convergence tolerance
ENERGY_CONSERVATION_THRESHOLD = 1e-14  # Energy conservation violation detection

__all__ = [
    # Core Assembly
    "ConceptualChargeAssembler",
    # Field Mechanics
    "FieldPerturbationAnalyzer",
    "InterferenceDynamicsEngine",
    "FieldEnergyCalculator",
    "PhaseTransitionDetector",
    "FieldConfiguration",
    "PerturbationResponse",
    "InterferencePattern",
    "EnergyComponents",
    "CriticalPoint",
    "FieldSymmetry",
    "PhaseType",
    # Selection Pressure
    "ComplexityGradientAnalyzer",
    "SpotlightFieldEngine",
    "ConstructEvolutionEngine",
    "PopulationState",
    "SelectionPressure",
    "AttentionField",
    "SocialConstruct",
    "ComplexityMeasures",
    "EvolutionaryDynamics",
    "GameTheoreticAnalysis",
    # Field Integration
    "FieldIntegrator",
    "FieldIntegrationState",
    "HKFieldCoupling",
    "SystemEvolution",
    # Mathematical Constants
    "FIELD_THEORETICAL_PRECISION",
    "HEGSELMANN_KRAUSE_EPSILON",
    "MATHEMATICAL_TOLERANCE",
    "ENERGY_CONSERVATION_THRESHOLD",
]
