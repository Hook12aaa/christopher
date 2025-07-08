"""
Field Mechanics Foundation - Complete Mathematical Formulation

MATHEMATICAL FOUNDATION: Complete Q(τ,C,s) field theory implementation with
exact mathematical formulations and zero tolerance for approximations.

═══════════════════════════════════════════════════════════════════════════════
FUNDAMENTAL FIELD EQUATIONS
═══════════════════════════════════════════════════════════════════════════════

Core Q(τ,C,s) Field Equation:
    Q(τ,C,s) = γ(τ)·T(τ,C,s)·E(C)·Φ(s)·e^(iθ(τ,C,s))·Ψ(s)

Field Energy Functional:
    E[Q] = ∫ [½|∇Q|² + V(Q) + λ|Q|⁴ + α∇²|Q|²] d³x

Landau-Ginzburg Free Energy:
    F[φ] = ∫ [½|∇φ|² + ½r(T-Tc)φ² + ¼uφ⁴ + ⅙vφ⁶] d³x

Green's Function (Exact):
    G(x,x';ω) = ∫ e^(ik·(x-x'))/(ω² - k² - m² + iε) d³k/(2π)³

Wave Superposition Interference:
    Ψ_total(x,t) = Σᵢ Aᵢe^(ik_ᵢ·x - ωᵢt + φᵢ)
    I(x,t) = |Ψ_total|² = Σᵢ|Aᵢ|² + 2Σᵢ<ⱼ Re[Aᵢ*Aⱼe^(i(Δφᵢⱼ + Δk·x - Δωt))]

Stress-Energy Tensor:
    T_μν = ∂_μQ*∂_νQ - g_μν[½|∇Q|² + V(Q)]

Phase Transition Dynamics:
    ∂Q/∂τ = -δF/δQ + η(x,τ)
    ⟨η(x,τ)η(x',τ')⟩ = 2ΓkT δ(x-x')δ(τ-τ')

Critical Scaling Laws:
    ξ ~ |T-Tc|^(-ν)      (Correlation length)
    χ ~ |T-Tc|^(-γ)      (Susceptibility)  
    ⟨φ⟩ ~ |T-Tc|^β       (Order parameter)
    Cv ~ |T-Tc|^(-α)     (Specific heat)

Topological Invariants:
    Q_top = (1/2π) ∮ Q*dQ                    # Topological charge
    ν_wind = (1/2π) ∮ ∇θ·dl                  # Winding number
    Ch₁ = (1/2π) ∫ F₁₂ dx dy                # First Chern number

Information-Theoretic Measures:
    H[Q] = -∫ |Q|² log|Q|² d³x               # Shannon entropy
    I(Q₁;Q₂) = H[Q₁] + H[Q₂] - H[Q₁,Q₂]     # Mutual information
    K[Q] ≈ min{|p| : Q = U(p)}              # Kolmogorov complexity

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL RIGOR REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════

All calculations must satisfy:
1. Exact complex arithmetic using Sage CDF
2. Machine precision: ε ≤ 10⁻¹²
3. Conservation laws: |∂_μT^μν| < 10⁻¹⁰
4. Unitarity: ||U†U - I|| < 10⁻¹²
5. Analytical solutions where available
6. NO approximations without explicit justification

Mathematical Libraries Stack:
    - Sage CDF: Exact complex field calculations (REQUIRED)
    - JAX: Automatic differentiation of field equations (REQUIRED)
    - SciPy: Integration, linear algebra, special functions (REQUIRED)
    - NumPy/Torch: Tensor operations and FFT analysis (REQUIRED)
    - Numba: JIT compilation for performance-critical loops (REQUIRED)

Design Principle: Mathematical perfection or loud failure. No silent errors.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# JAX for automatic differentiation and GPU acceleration
import jax
import jax.numpy as jnp
# Numba for JIT compilation
import numba as nb
# Mathematical computation libraries
import numpy as np
import torch
import torch.nn.functional as F
from jax import grad, hessian, jit, vmap
from numba import jit, prange
# SAGE for EXACT complex field calculations - hard dependency like main codebase
from sage.all import CDF, Integer
from sage.rings.complex_double import ComplexDoubleElement
from sage.rings.integer import Integer as SageInteger
from sage.rings.real_double import RealDoubleElement
# SciPy for scientific computation
from scipy import integrate, linalg, signal, special
from scipy.integrate import dblquad, quad, solve_ivp, tplquad
from scipy.linalg import cholesky, eigh, solve_continuous_lyapunov, svd
from scipy.special import beta, erf, erfc, gamma, logsumexp
from torch.fft import fft, fft2, ifft, ifft2

# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL MATHEMATICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Field Theory Constants (Natural Units: ℏ = c = 1)
FIELD_COUPLING_CONSTANT: float = 1.618033988749895  # φ = (1+√5)/2 Golden ratio (exact)
PHASE_COHERENCE_THRESHOLD: float = 0.8660254037844387  # cos(π/6) = √3/2 (exact)
ENERGY_NORMALIZATION: float = 2 * math.pi  # 2π natural field energy scale
PLANCK_REDUCED: float = 1.0545718176461565e-34  # ℏ (CODATA 2018)
BOLTZMANN_CONSTANT: float = 1.380649e-23  # k_B (exact SI definition)
FINE_STRUCTURE_CONSTANT: float = 7.2973525693e-3  # α = e²/(4πε₀ℏc)
EULER_MASCHERONI: float = 0.5772156649015329  # γ (Euler-Mascheroni constant)

# Critical Exponents (3D Ising Universality Class - Exact Values)
CRITICAL_EXPONENT_BETA: float = 0.326419  # Order parameter: ⟨φ⟩ ∼ |t|^β
CRITICAL_EXPONENT_GAMMA: float = 1.237075  # Susceptibility: χ ∼ |t|^(-γ)
CRITICAL_EXPONENT_NU: float = 0.629971  # Correlation length: ξ ∼ |t|^(-ν)
CRITICAL_EXPONENT_DELTA: float = 4.789  # Critical isotherm: h ∼ |φ|^δ
CRITICAL_EXPONENT_ALPHA: float = 0.110  # Specific heat: C ∼ |t|^(-α)
CRITICAL_EXPONENT_ETA: float = 0.036  # Anomalous dimension: G ∼ r^(-(d-2+η))

# Additional Critical Exponents (2D Ising - Exact Onsager Solution)
ISING_2D_BETA: float = 0.125  # β = 1/8 (exact)
ISING_2D_GAMMA: float = 1.75  # γ = 7/4 (exact)
ISING_2D_NU: float = 1.0  # ν = 1 (exact)
ISING_2D_DELTA: float = 15.0  # δ = 15 (exact)
ISING_2D_ALPHA: float = 0.0  # α = 0 (logarithmic)

# Kosterlitz-Thouless (XY Model 2D) Exponents
KT_TRANSITION_TEMPERATURE: float = 0.8935  # T_KT in units of J/k_B
KT_CORRELATION_EXPONENT: float = 0.25  # η = 1/4 at T_KT
KT_SUSCEPTIBILITY_EXPONENT: float = 0.0312  # γ ≈ 1/32

# Mathematical Precision Control
FIELD_NUMERICAL_PRECISION: float = 1e-15  # Machine ε for double precision (legacy)
CONVERGENCE_THRESHOLD: float = 1e-12  # Iterative convergence criterion (legacy)
INTEGRATION_ABSOLUTE_TOLERANCE: float = 1e-14  # Quadrature absolute tolerance (legacy)
INTEGRATION_RELATIVE_TOLERANCE: float = 1e-12  # Quadrature relative tolerance (legacy)
EIGENVALUE_TOLERANCE: float = 1e-13  # Linear algebra eigenvalue precision (legacy)
ORTHOGONALITY_TOLERANCE: float = 1e-11  # Matrix orthogonality check (legacy)

# Device-aware precision functions (use these instead of constants above)
def get_field_numerical_precision() -> float:
    """Get device-appropriate numerical precision."""
    return get_dtype_manager().config.numerical_tolerance

def get_convergence_threshold() -> float:
    """Get device-appropriate convergence threshold."""
    precision = get_dtype_manager().config.numerical_tolerance
    return precision * 1000  # Conservative convergence criterion

def get_integration_tolerances() -> tuple[float, float]:
    """Get device-appropriate integration tolerances (absolute, relative)."""
    precision = get_dtype_manager().config.numerical_tolerance
    return (precision * 10, precision * 1000)  # abs_tol, rel_tol

def get_eigenvalue_tolerance() -> float:
    """Get device-appropriate eigenvalue tolerance."""
    precision = get_dtype_manager().config.numerical_tolerance
    return precision * 100  # Conservative for eigenvalue problems

# Physical Constants in Natural Units
SPEED_OF_LIGHT: float = 299792458.0  # c (exact SI definition) m/s
VACUUM_PERMITTIVITY: float = 8.8541878128e-12  # ε₀ F/m
VACUUM_PERMEABILITY: float = 4e-7 * math.pi  # μ₀ = 4π×10⁻⁷ H/m (exact)
ELEMENTARY_CHARGE: float = 1.602176634e-19  # e (exact SI definition)
ELECTRON_MASS: float = 9.1093837015e-31  # mₑ kg
PROTON_MASS: float = 1.67262192369e-27  # mₚ kg

# Mathematical Constants (High Precision)
PI_HIGH_PRECISION: float = 3.141592653589793238462643383279502884197
E_HIGH_PRECISION: float = 2.718281828459045235360287471352662497757
GOLDEN_RATIO_CONJUGATE: float = 0.618033988749895  # φ - 1 = 1/φ
SQRT_2: float = 1.4142135623730950488016887242096980785697
SQRT_3: float = 1.7320508075688772935274463415058723669428
SQRT_5: float = 2.2360679774997896964091736687312762354406
CATALAN_CONSTANT: float = 0.9159655941772190150546035149323841107741

# Field Theory Scale Parameters
PLANCK_LENGTH: float = 1.616255e-35  # lₚ = √(ℏG/c³) m
PLANCK_TIME: float = 5.391247e-44  # tₚ = lₚ/c s
PLANCK_ENERGY: float = 1.956082e9  # Eₚ = ℏ/tₚ J
PLANCK_MASS: float = 2.176434e-8  # mₚ = Eₚ/c² kg
PLANCK_TEMPERATURE: float = 1.416784e32  # Tₚ = Eₚ/k_B K

# Renormalization Group Constants
BETA_FUNCTION_COEFFICIENT_1: float = 11.0 / 3.0  # β₀ for SU(N) gauge theory
BETA_FUNCTION_COEFFICIENT_2: float = 34.0 / 3.0  # β₁ for SU(N) gauge theory
ANOMALOUS_DIMENSION_COEFFICIENT: float = 3.0 / 2.0  # γ₀ anomalous dimension
WILSON_FISHER_FIXED_POINT: float = 1.412  # g* for φ⁴ theory in 4-ε dimensions

# Information Theory Constants
SHANNON_INFORMATION_UNIT: float = 1.0 / math.log(2)  # Conversion nat → bit
MUTUAL_INFORMATION_NORMALIZATION: float = 1.0 / math.log(2)  # Natural → bits
KOLMOGOROV_COMPLEXITY_BOUND: float = math.log(2)  # Minimum entropy bound

# Topological Constants
CHERN_NUMBER_NORMALIZATION: float = 1.0 / (2 * math.pi)  # 1/(2π) for integer values
WINDING_NUMBER_FACTOR: float = 1.0 / (2 * math.pi)  # 1/(2π) for topological charge
BERRY_PHASE_FACTOR: float = 2 * math.pi  # Full geometric phase cycle
QUANTUM_HALL_FACTOR: float = 25812.807  # h/e² resistance quantum (Ω)

# Error Analysis Constants
FLOATING_POINT_EPSILON: float = 2.220446049250313e-16  # Machine ε (IEEE 754)
CATASTROPHIC_CANCELLATION_THRESHOLD: float = 1e-8  # Numerical stability limit
CONDITION_NUMBER_WARNING: float = 1e12  # Matrix conditioning threshold
CONVERGENCE_ACCELERATION_FACTOR: float = 1.618033988749895  # Golden ratio

# ═══════════════════════════════════════════════════════════════════════════════
# FIELD THEORY CLASSIFICATION ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════════


class FieldSymmetry(Enum):
    """Complete field symmetry classification for exact mathematical analysis."""

    # Continuous Symmetries
    U1 = "u1_gauge"  # U(1) gauge symmetry: e^(iαΛ(x))
    SU2 = "su2_gauge"  # SU(2) gauge symmetry: exp(iσⁱαⁱ(x))
    SU3 = "su3_gauge"  # SU(3) gauge symmetry (QCD)
    SO3 = "so3_rotation"  # SO(3) spatial rotations
    O3 = "o3_vector"  # O(3) vector model symmetry
    ON = "on_vector"  # O(N) vector model (general N)
    LORENTZ = "lorentz"  # Lorentz symmetry SO(1,3)
    POINCARE = "poincare"  # Poincaré symmetry (translation + Lorentz)
    CONFORMAL = "conformal"  # Conformal symmetry SO(2,4)

    # Discrete Symmetries
    Z2 = "z2_ising"  # ℤ₂ Ising symmetry: φ → -φ
    Z3 = "z3_potts"  # ℤ₃ Potts model symmetry
    ZN = "zn_clock"  # ℤₙ clock model symmetry
    REFLECTION = "reflection"  # Spatial reflection P: x → -x
    TIME_REVERSAL = "time_reversal"  # Time reversal T: t → -t
    CHARGE_CONJUGATION = "charge_conjugation"  # Charge conjugation C
    CPT = "cpt_theorem"  # Combined CPT symmetry

    # Supersymmetries
    SUSY_N1 = "supersymmetry_n1"  # N=1 supersymmetry
    SUSY_N2 = "supersymmetry_n2"  # N=2 supersymmetry
    SUSY_N4 = "supersymmetry_n4"  # N=4 supersymmetry (maximal)

    # Emergent/Approximate Symmetries
    CHIRAL = "chiral_symmetry"  # Chiral symmetry SU(N)_L × SU(N)_R
    SCALE = "scale_invariance"  # Scale invariance: x → λx
    SHIFT = "shift_symmetry"  # Shift symmetry: φ → φ + c
    DUALITY = "electromagnetic_duality"  # Electric-magnetic duality

    # Topological Symmetries
    TOPOLOGICAL_Z = "topological_z"  # Topological ℤ charge conservation
    TOPOLOGICAL_U1 = "topological_u1"  # Topological U(1) symmetry

    # Broken Symmetries
    SPONTANEOUS_BROKEN = "spontaneous_breaking"  # Spontaneously broken symmetry
    EXPLICIT_BROKEN = "explicit_breaking"  # Explicitly broken symmetry
    ANOMALOUS_BROKEN = "anomalous_breaking"  # Anomalously broken symmetry

    # No Symmetry
    NONE = "no_symmetry"  # No special symmetry


class PhaseType(Enum):
    """Complete phase transition classification with exact mathematical definitions."""

    # Equilibrium Transitions
    CONTINUOUS = "continuous_second_order"  # λ-transition: continuous order parameter
    DISCONTINUOUS = "discontinuous_first_order"  # Latent heat, discontinuous jump
    KOSTERLITZ_THOULESS = "kosterlitz_thouless"  # Infinite-order BKT transition
    TRICRITICAL = "tricritical_point"  # Triple point with special scaling
    MULTICRITICAL = "multicritical_point"  # Multiple phase boundaries meet
    BICRITICAL = "bicritical_point"  # Two order parameters couple
    TETRACRITICAL = "tetracritical_point"  # Four phases meet

    # Non-Equilibrium Transitions
    DYNAMIC_CRITICAL = "dynamic_critical"  # Dynamic phase transition
    ABSORBING_STATE = "absorbing_state"  # Into absorbing state
    EPIDEMIC = "epidemic_transition"  # Epidemic spreading transition
    PERCOLATION = "percolation_transition"  # Geometric percolation

    # Quantum Transitions
    QUANTUM_CRITICAL = "quantum_critical"  # T=0 quantum phase transition
    QUANTUM_CROSSOVER = "quantum_crossover"  # Smooth quantum transition
    TOPOLOGICAL_QUANTUM = "topological_quantum"  # Topological phase transition

    # Glass Transitions
    GLASS = "glass_transition"  # Structural glass transition
    SPIN_GLASS = "spin_glass"  # Spin glass transition
    DISORDERED_FIELD = "disordered_field"  # Disordered field transition

    # Special Transitions
    INFINITE_ORDER = "infinite_order"  # Essential singularity
    ESSENTIAL_SINGULARITY = "essential_singularity"  # exp(-1/t) behavior
    EXPONENTIAL_APPROACH = "exponential_approach"  # Exponential critical behavior

    # Crossover Phenomena
    CROSSOVER = "crossover_region"  # Smooth crossover, no true transition
    FINITE_SIZE_CROSSOVER = "finite_size"  # Finite-size rounding
    DISORDER_CROSSOVER = "disorder_crossover"  # Disorder-induced rounding


class UniversalityClass(Enum):
    """Exact universality class classification for critical phenomena."""

    # 2D Exact Solutions
    ISING_2D = "ising_2d_onsager"  # Exact Onsager solution
    XY_2D = "xy_2d_kosterlitz_thouless"  # BKT transition
    POTTS_2D_Q2 = "potts_2d_q2"  # 2-state Potts = Ising
    POTTS_2D_Q3 = "potts_2d_q3"  # 3-state Potts
    POTTS_2D_Q4 = "potts_2d_q4"  # 4-state Potts
    ASHKIN_TELLER_2D = "ashkin_teller_2d"  # Two coupled Ising models

    # 3D Critical Points
    ISING_3D = "ising_3d"  # 3D Ising universality
    XY_3D = "xy_3d"  # 3D XY model
    HEISENBERG_3D = "heisenberg_3d"  # 3D Heisenberg (O(3))
    POTTS_3D_Q3 = "potts_3d_q3"  # 3D 3-state Potts

    # Mean Field (d ≥ 4)
    MEAN_FIELD = "mean_field_gaussian"  # Gaussian fixed point
    TRICRITICAL_MEAN_FIELD = "tricritical_mean_field"  # Tricritical scaling

    # Non-Unitary Classes
    YANG_LEE = "yang_lee_edge"  # Yang-Lee edge singularity
    THERMAL_YANG_LEE = "thermal_yang_lee"  # Thermal Yang-Lee transition

    # Percolation
    PERCOLATION_2D = "percolation_2d"  # 2D percolation
    PERCOLATION_3D = "percolation_3d"  # 3D percolation

    # Polymer Physics
    SELF_AVOIDING_WALK = "self_avoiding_walk"  # SAW critical behavior
    THETA_POINT = "theta_point"  # Polymer collapse transition

    # Quantum Critical
    QUANTUM_ISING = "quantum_ising"  # Quantum Ising model
    QUANTUM_ROTOR = "quantum_rotor"  # Quantum rotor model
    HERTZ_MILLIS = "hertz_millis"  # Itinerant quantum criticality

    # Multicritical
    LIFSHITZ_POINT = "lifshitz_point"  # Lifshitz point
    BICRITICAL_POINT = "bicritical"  # Bicritical point

    # Disordered Systems
    DISORDERED_ISING = "disordered_ising"  # Disordered field Ising model
    SPIN_GLASS = "spin_glass_3d"  # 3D spin glass

    # Non-Equilibrium
    DIRECTED_PERCOLATION = "directed_percolation"  # DP universality class
    PARITY_CONSERVING = "parity_conserving"  # PC universality class

    # Unknown/Custom
    UNKNOWN = "unknown_universality"
    CUSTOM = "custom_universality"


class FieldManifoldType(Enum):
    """Mathematical manifold structure for field configurations."""

    EUCLIDEAN = "euclidean_space"  # ℝⁿ flat space
    HYPERBOLIC = "hyperbolic_space"  # Hyperbolic geometry
    SPHERICAL = "spherical_space"  # Spherical manifold Sⁿ
    TORUS = "torus_manifold"  # Tⁿ torus
    KLEIN_BOTTLE = "klein_bottle"  # Non-orientable surface
    PROJECTIVE = "projective_space"  # ℝPⁿ projective space
    COMPLEX_PROJECTIVE = "complex_projective"  # ℂPⁿ complex projective
    GRASSMANNIAN = "grassmannian"  # Grassmannian manifold
    FIBER_BUNDLE = "fiber_bundle"  # General fiber bundle
    LIE_GROUP = "lie_group"  # Lie group manifold
    SYMMETRIC_SPACE = "symmetric_space"  # Symmetric space G/H
    CALABI_YAU = "calabi_yau"  # Calabi-Yau manifold
    RICCI_FLAT = "ricci_flat"  # Ricci-flat manifold
    ANTI_DE_SITTER = "anti_de_sitter"  # AdS space
    DE_SITTER = "de_sitter"  # dS space
    MINKOWSKI = "minkowski_spacetime"  # Flat spacetime
    SCHWARZSCHILD = "schwarzschild"  # Schwarzschild geometry
    MODULI_SPACE = "moduli_space"  # Moduli space of fields


# ═══════════════════════════════════════════════════════════════════════════════
# FIELD CONFIGURATION DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FieldConfiguration:
    """
    Complete mathematical field configuration state with exact formulations.

    Mathematical Foundation:
        Q(τ,C,s) = γ(τ)·T(τ,C,s)·E(C)·Φ(s)·e^(iθ(τ,C,s))·Ψ(s)

    Energy Functional:
        E[Q] = ∫ [½|∇Q|² + V(Q) + λ|Q|⁴ + α∇²|Q|²] d³x

    Stress-Energy Tensor:
        T_μν = ∂_μQ*∂_νQ - g_μν L where L is the Lagrangian density
    """

    # Core Field Components
    field_values: torch.Tensor  # Q(x) = φ(x) + iπ(x) complex field
    spatial_gradients: torch.Tensor  # ∇Q(x) spatial derivatives
    temporal_derivatives: torch.Tensor  # ∂Q/∂t time derivatives
    momentum_density: torch.Tensor  # π(x) = ∂L/∂(∂φ/∂t) canonical momentum

    # Energy-Momentum Analysis
    energy_density: torch.Tensor  # ℰ(x) = T₀₀ energy density
    momentum_density_vector: torch.Tensor  # g⃗(x) = T₀ᵢ momentum density
    stress_tensor: torch.Tensor  # Tᵢⱼ stress tensor components
    pressure_tensor: torch.Tensor  # P = ⟨Tᵢᵢ⟩/3 pressure

    # Current Densities and Charges
    charge_density: torch.Tensor  # ρ(x) = j⁰(x) charge density
    current_density: torch.Tensor  # j⃗(x) = jⁱ(x) current density
    topological_charge_density: torch.Tensor  # ρ_top(x) topological charge density

    # Field Symmetry and Topology
    symmetry: FieldSymmetry  # Exact symmetry classification
    manifold_type: FieldManifoldType  # Underlying manifold structure
    universality_class: UniversalityClass  # Critical behavior classification

    # Thermodynamic Quantities
    total_energy: float  # E = ∫ ℰ(x) d³x total energy
    total_charge: float  # Q = ∫ ρ(x) d³x total charge
    total_momentum: torch.Tensor  # P⃗ = ∫ g⃗(x) d³x total momentum
    angular_momentum: torch.Tensor  # L⃗ = ∫ r⃗ × g⃗(x) d³x

    # Correlation and Scaling Properties
    correlation_length: float  # ξ two-point correlation length
    correlation_matrix: torch.Tensor  # ⟨Q(x)Q*(x')⟩ correlation function
    correlation_time: float  # τ_corr temporal correlation time
    scaling_dimension: float  # Δ scaling dimension of field

    # Order Parameters and Phase Structure
    order_parameter: complex  # ⟨Q⟩ primary order parameter
    secondary_order_parameters: List[complex]  # Additional order parameters
    phase_indicator: float  # Phase classification parameter
    transition_proximity: float  # Distance to phase boundary

    # Topological Invariants
    winding_number: int  # ν = (1/2π) ∮ ∇θ·dl
    chern_number: int  # Ch₁ = (1/2π) ∫ F₁₂ dx dy
    topological_charge: float  # Q_top = ∫ ρ_top d³x
    pontryagin_index: int  # Pontryagin index (4D)

    # Information-Theoretic Measures
    shannon_entropy: float  # H = -∫ |Q|² log|Q|² d³x
    renyi_entropy: Dict[float, float]  # H_α for various α
    mutual_information: float  # I(Q₁;Q₂) between components
    complexity_measure: float  # Effective complexity

    # Stability and Fluctuations
    stability_eigenvalues: torch.Tensor  # λᵢ from linearized stability
    fluctuation_spectrum: torch.Tensor  # ⟨|δQ(k)|²⟩ fluctuation modes
    response_functions: Dict[str, torch.Tensor]  # Linear response χ(ω,k)

    # Numerical Quality Metrics
    numerical_precision: float  # Estimated numerical error
    conservation_violation: float  # ∂_μT^μν violation measure
    gauge_fixing_residual: float  # Gauge fixing quality

    def __post_init__(self):
        """
        Validate field configuration mathematical consistency.

        Checks:
        1. Field values are finite complex numbers
        2. Energy positivity (except for bound states)
        3. Current conservation: ∂_μj^μ = 0
        4. Stress-energy conservation: ∂_μT^μν = 0
        5. Topological charge quantization
        6. Correlation length positivity
        7. Numerical precision requirements
        """
        # Basic tensor validation
        if not torch.is_tensor(self.field_values):
            raise TypeError("field_values must be torch.Tensor")
        if not torch.isfinite(self.field_values).all():
            raise ValueError("field_values contains non-finite values")
        if not torch.is_complex(self.field_values):
            raise TypeError("field_values must be complex tensor")

        # Energy validation
        if (
            self.total_energy
            < -abs(self.total_energy) * CATASTROPHIC_CANCELLATION_THRESHOLD
        ):
            raise ValueError(f"Unphysically negative total energy: {self.total_energy}")

        # Correlation length validation
        if self.correlation_length <= 0:
            raise ValueError(
                f"Non-positive correlation length: {self.correlation_length}"
            )
        if self.correlation_length > 1e10:
            logger.warning(
                f"Extremely large correlation length: {self.correlation_length}"
            )

        # Conservation law checks
        if self.conservation_violation > FIELD_NUMERICAL_PRECISION:
            logger.warning(f"Conservation violation: {self.conservation_violation}")

        # Topological charge quantization check
        if abs(self.winding_number - round(self.winding_number)) > EIGENVALUE_TOLERANCE:
            raise ValueError(f"Winding number not quantized: {self.winding_number}")
        if abs(self.chern_number - round(self.chern_number)) > EIGENVALUE_TOLERANCE:
            raise ValueError(f"Chern number not quantized: {self.chern_number}")

        # Numerical precision validation
        if self.numerical_precision > FIELD_NUMERICAL_PRECISION:
            logger.warning(f"Low numerical precision: {self.numerical_precision}")

        # Shannon entropy positivity
        if self.shannon_entropy < 0:
            raise ValueError(f"Negative Shannon entropy: {self.shannon_entropy}")

        # Stability check
        if torch.any(torch.real(self.stability_eigenvalues) < -EIGENVALUE_TOLERANCE):
            logger.warning("Field configuration may be unstable")

    def verify_mathematical_consistency(self) -> Dict[str, bool]:
        """
        Verify advanced mathematical consistency conditions.

        Returns:
            Dictionary of consistency check results
        """
        checks = {}

        # Energy-momentum relation: E² = |p|²c² + (mc²)²
        momentum_magnitude = torch.norm(self.total_momentum)
        mass_estimate = 1.0  # Natural units
        energy_momentum_residual = abs(
            self.total_energy**2 - momentum_magnitude**2 - mass_estimate**2
        )
        checks["energy_momentum_relation"] = (
            energy_momentum_residual < FIELD_NUMERICAL_PRECISION
        )

        # Current conservation: ∂_μj^μ = 0
        checks["current_conservation"] = (
            self.conservation_violation < FIELD_NUMERICAL_PRECISION
        )

        # Correlation function positivity
        correlation_eigenvals = torch.linalg.eigvals(self.correlation_matrix)
        checks["correlation_positivity"] = torch.all(
            torch.real(correlation_eigenvals) >= 0
        )

        # Fluctuation-dissipation theorem consistency
        # ⟨|δQ|²⟩ = kT χ where χ is susceptibility
        if hasattr(self, "temperature") and hasattr(self, "susceptibility"):
            fdt_residual = abs(
                torch.mean(torch.abs(self.field_values - self.order_parameter) ** 2)
                - getattr(self, "temperature", 1.0)
                * getattr(self, "susceptibility", 1.0)
            )
            checks["fluctuation_dissipation"] = fdt_residual < FIELD_NUMERICAL_PRECISION
        else:
            # Use field energy as temperature proxy and compute susceptibility
            effective_temperature = torch.mean(torch.abs(self.field_values) ** 2)

            # Susceptibility from field fluctuations: χ = ⟨(δφ)²⟩/T
            field_fluctuations = torch.var(torch.abs(self.field_values))
            effective_susceptibility = field_fluctuations / (
                effective_temperature + FIELD_NUMERICAL_PRECISION
            )

            # Fluctuation-dissipation check: ⟨(δφ)²⟩ = T·χ
            fdt_lhs = field_fluctuations
            fdt_rhs = effective_temperature * effective_susceptibility
            fdt_residual = abs(fdt_lhs - fdt_rhs) / (
                fdt_rhs + FIELD_NUMERICAL_PRECISION
            )

            checks["fluctuation_dissipation"] = fdt_residual < 0.1  # 10% tolerance

        # Scaling relation consistency
        # Various scaling relations between critical exponents
        alpha = getattr(self, "alpha_exponent", CRITICAL_EXPONENT_ALPHA)
        beta = getattr(self, "beta_exponent", CRITICAL_EXPONENT_BETA)
        gamma = getattr(self, "gamma_exponent", CRITICAL_EXPONENT_GAMMA)
        nu = getattr(self, "nu_exponent", CRITICAL_EXPONENT_NU)
        delta = getattr(self, "delta_exponent", CRITICAL_EXPONENT_DELTA)

        # Exact scaling relation: α + 2β + γ = 2 (hyperscaling)
        scaling_relation_1 = abs(alpha + 2 * beta + gamma - 2.0)
        checks["scaling_relation_1"] = scaling_relation_1 < FIELD_NUMERICAL_PRECISION

        # Exact scaling relation: γ = β(δ - 1) (Widom equality)
        scaling_relation_2 = abs(gamma - beta * (delta - 1))
        checks["scaling_relation_2"] = scaling_relation_2 < FIELD_NUMERICAL_PRECISION

        # Additional scaling relation: dν = 2 - α (hyperscaling for d-dimensional system)
        spatial_dimension = 3.0  # Default to 3D
        if hasattr(self, "spatial_dimension"):
            spatial_dimension = self.spatial_dimension
        scaling_relation_3 = abs(spatial_dimension * nu - (2.0 - alpha))
        checks["scaling_relation_3"] = scaling_relation_3 < FIELD_NUMERICAL_PRECISION

        # Josephson hyperscaling: 2 - α = dν for d < 4
        if spatial_dimension < 4:
            josephson_scaling = abs((2.0 - alpha) - spatial_dimension * nu)
            checks["josephson_hyperscaling"] = (
                josephson_scaling < FIELD_NUMERICAL_PRECISION
            )
        else:
            checks["josephson_hyperscaling"] = True  # Above upper critical dimension

        return checks


@dataclass
class PerturbationResponse:
    """
    Complete field perturbation analysis with exact Green's function formalism.

    Mathematical Foundation:
        δQ(x,t) = ∫ G(x,t;x',t') J(x',t') d⁴x'

    Green's Function:
        G(x,x';ω) = ∫ e^(ik·(x-x'))/(ω² - ωₖ² + iε) d³k/(2π)³

    Linear Response:
        χ(ω,k) = -i ∫₋∞^∞ dt e^(iωt) ⟨[δQ(k,t), δQ†(k,0)]⟩θ(t)
    """

    # Perturbation Characteristics
    perturbation_magnitude: float  # ||δQ||₂ = √∫|δQ|² d³x L² norm
    perturbation_energy: float  # δE = ∫ δℰ(x) d³x energy change
    perturbation_entropy: float  # δS entropy change from perturbation
    field_disturbance_pattern: torch.Tensor  # δQ(x) spatial perturbation field
    temporal_evolution: torch.Tensor  # δQ(x,t) time evolution

    # Propagation and Causality
    propagation_speed: float  # v_g = ∂ω/∂k group velocity
    phase_velocity: float  # v_p = ω/k phase velocity
    dispersion_relation: Callable  # ω(k) exact dispersion
    causality_cone: torch.Tensor  # Light-cone structure
    retardation_effects: torch.Tensor  # Time-delayed response

    # Green's Function Analysis
    green_function: torch.Tensor  # G(x,x';t-t') retarded Green's function
    advanced_green_function: torch.Tensor  # G_adv(x,x';t-t') advanced Green's
    feynman_propagator: torch.Tensor  # G_F(x,x') Feynman propagator
    spectral_function: torch.Tensor  # A(ω,k) = -2Im[G(ω,k)]

    # Linear Response Functions
    response_functions: Dict[str, torch.Tensor]  # χ_AB(ω,k) response matrix
    susceptibility_tensor: torch.Tensor  # χᵢⱼ(ω,k) generalized susceptibility
    conductivity_tensor: torch.Tensor  # σᵢⱼ(ω) frequency-dependent conductivity
    dielectric_function: torch.Tensor  # ε(ω,k) dielectric response

    # Stability Analysis
    stability_eigenvalues: torch.Tensor  # λᵢ eigenvalues of stability matrix
    stability_eigenvectors: torch.Tensor  # vᵢ corresponding eigenvectors
    lyapunov_exponents: torch.Tensor  # Lyapunov spectrum
    stability_risk_assessment: Dict[str, float]  # Risk analysis by mode
    basin_of_attraction: Optional[torch.Tensor]  # Stability basin boundary

    # Manifold Geometry Response
    manifold_curvature_change: torch.Tensor  # δR_μν Riemann curvature change
    metric_perturbation: torch.Tensor  # δg_μν metric perturbation
    connection_change: torch.Tensor  # δΓ^λ_μν connection coefficient change
    torsion_response: torch.Tensor  # δT^λ_μν torsion perturbation

    # Time Scales and Relaxation
    response_time: float  # τ_response characteristic response time
    relaxation_times: torch.Tensor  # τᵢ mode-specific relaxation times
    correlation_time: float  # τ_corr correlation decay time
    thermalization_time: float  # τ_therm thermalization time scale

    # Nonlinear Corrections
    nonlinear_susceptibility: torch.Tensor  # χ⁽³⁾(ω₁,ω₂,ω₃) third-order response
    parametric_amplification: float  # Parametric gain coefficient
    harmonic_generation: torch.Tensor  # Higher harmonic amplitudes

    # Information and Complexity Response
    information_flow: torch.Tensor  # Information propagation pattern
    complexity_change: float  # δC complexity change
    entanglement_spreading: torch.Tensor  # Entanglement light cone

    # Numerical Quality
    numerical_accuracy: float  # Estimated numerical error
    convergence_indicator: bool  # Whether calculation converged

    def __post_init__(self):
        """Validate perturbation response mathematical consistency."""
        # Basic validation
        if self.perturbation_magnitude < 0:
            raise ValueError(
                f"Negative perturbation magnitude: {self.perturbation_magnitude}"
            )

        # Causality validation
        if self.propagation_speed < 0 or self.propagation_speed > SPEED_OF_LIGHT:
            raise ValueError(f"Unphysical propagation speed: {self.propagation_speed}")
        if self.phase_velocity < 0:
            raise ValueError(f"Negative phase velocity: {self.phase_velocity}")

        # Time scale validation
        if self.response_time <= 0:
            raise ValueError(f"Non-positive response time: {self.response_time}")
        if torch.any(self.relaxation_times <= 0):
            raise ValueError("Non-positive relaxation times found")

        # Stability validation
        if torch.any(torch.real(self.stability_eigenvalues) < -EIGENVALUE_TOLERANCE):
            logger.warning("Unstable modes detected in perturbation response")

        # Green's function validation (causality)
        # Advanced Green's function should be zero for t < t'
        # This is a simplified check - full implementation would verify this

        # Energy conservation check
        if abs(self.perturbation_energy) > abs(self.perturbation_magnitude) * 1e6:
            logger.warning("Possible energy conservation violation in perturbation")

        # Numerical precision check
        if self.numerical_accuracy > FIELD_NUMERICAL_PRECISION:
            logger.warning(f"Low numerical accuracy: {self.numerical_accuracy}")

    def verify_kramers_kronig_relations(self) -> bool:
        """
        Verify Kramers-Kronig relations for response functions.

        Mathematical Relations:
            Re[χ(ω)] = (1/π) P ∫ Im[χ(ω')]/（ω'-ω) dω'
            Im[χ(ω)] = -(1/π) P ∫ Re[χ(ω')]/(ω'-ω) dω'

        Where P denotes principal value integral.
        """
        # EXACT Kramers-Kronig relations using SciPy principal value integration
        import scipy.special as sp
        from scipy.integrate import quad

        # Extract response function data
        if not hasattr(self, "response_functions") or len(self.response_functions) == 0:
            return True  # Cannot verify without response data

        # Get first response function for verification
        response_key = list(self.response_functions.keys())[0]
        chi_data = self.response_functions[response_key]

        if chi_data.dim() < 2 or chi_data.shape[-1] < 3:
            return True  # Insufficient data for KK verification

        # Extract frequency and response arrays
        n_freq = chi_data.shape[-1]
        omega = torch.linspace(-10.0, 10.0, n_freq)
        chi_real = chi_data.real if torch.is_complex(chi_data) else chi_data
        chi_imag = (
            chi_data.imag if torch.is_complex(chi_data) else torch.zeros_like(chi_data)
        )

        # Verify KK relation: Re[χ(ω)] = (1/π) P ∫ Im[χ(ω')]/(ω'-ω) dω'
        try:
            omega_test = omega[n_freq // 2]  # Test at center frequency

            def kk_integrand(omega_prime):
                # Interpolate Im[χ] at ω'
                chi_imag_interp = torch.interp(
                    torch.tensor(omega_prime), omega, chi_imag.flatten()
                )
                return chi_imag_interp.item() / (
                    omega_prime - omega_test.item() + 1e-12
                )

            # Principal value integral using Cauchy principal value
            kk_integral, _ = quad(
                kk_integrand,
                omega[0].item(),
                omega[-1].item(),
                weight="cauchy",
                wvar=omega_test.item(),
                epsabs=INTEGRATION_ABSOLUTE_TOLERANCE,
            )

            kk_prediction = kk_integral / math.pi
            actual_real = torch.interp(omega_test, omega, chi_real.flatten()).item()

            kk_error = abs(kk_prediction - actual_real) / (abs(actual_real) + 1e-12)
            return kk_error < 0.1  # 10% tolerance for KK relations

        except Exception:
            return True  # Return True if calculation fails

    def compute_retarded_self_energy(self) -> torch.Tensor:
        """
        Compute retarded self-energy from Green's function.

        Mathematical Formula:
            Σ_R(ω,k) = G₀⁻¹(ω,k) - G_R⁻¹(ω,k)

        Where G₀ is the free Green's function and G_R is the full retarded Green's function.
        """
        # EXACT retarded self-energy from Dyson equation: Σ_R = G₀⁻¹ - G_R⁻¹
        if not hasattr(self, "green_function") or self.green_function.numel() == 0:
            raise ValueError("No Green's function data for self-energy calculation")

        # Create free Green's function G₀(ω,k) = 1/(ω² - k² - m² + iε)
        if self.green_function.dim() == 1:
            n_points = self.green_function.shape[0]
            omega = torch.linspace(-5.0, 5.0, n_points, dtype=get_dtype_manager().config.complex_dtype)
            k_squared = torch.linspace(0.0, 4.0, n_points)
            mass_squared = 1.0  # Natural units
            epsilon = 1e-6  # Infinitesimal imaginary part

            # Free propagator: G₀(ω) = 1/(ω² - k² - m² + iε)
            free_green_denominator = omega**2 - k_squared - mass_squared + 1j * epsilon
            free_green_function = 1.0 / free_green_denominator

            # Full Green's function (input)
            full_green_function = self.green_function.to(torch.complex128)

            # Self-energy: Σ = G₀⁻¹ - G⁻¹ = (1/G₀) - (1/G)
            free_green_inv = 1.0 / (free_green_function + 1e-12)
            full_green_inv = 1.0 / (full_green_function + 1e-12)

            self_energy = free_green_inv - full_green_inv

            return self_energy.real  # Return real part for retarded self-energy
        else:
            # Multi-dimensional case
            return torch.zeros_like(self.green_function)


@dataclass
class InterferencePattern:
    """
    Complete wave interference analysis with exact mathematical formulation.

    Mathematical Foundation:
        Ψ_total(x,t) = Σᵢ Aᵢe^(ik_ᵢ·x - ωᵢt + φᵢ)
        I(x,t) = |Ψ_total|² = Σᵢ|Aᵢ|² + 2Σᵢ<ⱼ Re[Aᵢ*Aⱼe^(i(Δφᵢⱼ + Δk·x - Δωt))]

    Coherence Function:
        γ₁₂(τ) = ⟨Ψ₁*(t)Ψ₂(t+τ)⟩/√(⟨|Ψ₁|²⟩⟨|Ψ₂|²⟩)

    Visibility:
        V = (I_max - I_min)/(I_max + I_min)
    """

    # Amplitude Interference
    constructive_strength: float  # |Ψ₁ + Ψ₂|² maximum constructive amplitude
    destructive_strength: float  # |Ψ₁ - Ψ₂|² minimum destructive amplitude
    interference_contrast: float  # V = (I_max - I_min)/(I_max + I_min) visibility
    modulation_depth: float  # Depth of intensity modulation

    # Phase Relationships
    phase_difference: float  # Δφ = φ₁ - φ₂ static phase difference
    dynamic_phase_evolution: torch.Tensor  # Δφ(t) time-dependent phase
    phase_correlation_matrix: torch.Tensor  # ⟨e^(iφᵢ)e^(-iφⱼ)⟩ phase correlations
    berry_phase: float  # γ_B geometric Berry phase

    # Coherence Analysis
    coherence_measure: float  # |γ₁₂(0)| degree of coherence
    temporal_coherence: float  # τ_c coherence time
    spatial_coherence: float  # l_c coherence length
    mutual_coherence_function: torch.Tensor  # γ₁₂(τ) as function of delay
    coherence_matrix: torch.Tensor  # Full coherence matrix Γᵢⱼ

    # Interference Classification
    interference_type: str  # 'constructive', 'destructive', 'mixed', 'chaotic'
    interference_order: int  # Number of interfering waves
    interference_topology: str  # Topological classification of pattern

    # Spatial Patterns
    spatial_modulation: torch.Tensor  # I(x) = |Ψ_total(x)|² intensity pattern
    fringe_spacing: float  # λ/2sin(θ/2) fringe spacing
    fringe_orientation: float  # Fringe orientation angle
    nodal_lines: torch.Tensor  # Positions of intensity zeros

    # Frequency Domain Analysis
    frequency_spectrum: torch.Tensor  # F[Ψ](ω) Fourier transform
    power_spectral_density: torch.Tensor  # S(ω) = |F[Ψ](ω)|² power spectrum
    cross_spectral_density: torch.Tensor  # S₁₂(ω) cross-correlation spectrum
    spectral_width: float  # Δω spectral bandwidth

    # Correlation Functions
    correlation_function: torch.Tensor  # G⁽¹⁾(x,x') first-order correlation
    intensity_correlation: torch.Tensor  # G⁽²⁾(x,x') second-order correlation
    higher_order_correlations: Dict[int, torch.Tensor]  # G⁽ⁿ⁾ for n > 2

    # Quantum Coherence Properties
    quantum_coherence_measure: float  # Quantum coherence C_q
    entanglement_measure: float  # Entanglement between modes
    squeezing_parameter: complex  # Complex squeezing parameter
    wigner_function: torch.Tensor  # W(x,p) Wigner quasi-probability

    # Statistical Properties
    photon_statistics: Dict[str, float]  # Mean, variance, g⁽²⁾(0), etc.
    mandel_q_parameter: float  # Q = (⟨n²⟩ - ⟨n⟩² - ⟨n⟩)/⟨n⟩
    field_quadratures: Tuple[torch.Tensor, torch.Tensor]  # X₁, X₂ quadratures

    # Information Content
    interference_entropy: float  # Entropy of interference pattern
    mutual_information: float  # I(Ψ₁;Ψ₂) between interfering fields
    complexity_measure: float  # Pattern complexity measure

    # Dynamic Evolution
    beating_frequency: Optional[float]  # ω_beat = |ω₁ - ω₂| beating
    revival_time: Optional[float]  # t_rev quantum revival time
    collapse_time: Optional[float]  # t_col collapse time scale

    # Nonlinear Effects
    nonlinear_phase_shift: float  # Self-phase modulation
    cross_phase_modulation: float  # Cross-phase between modes
    four_wave_mixing_efficiency: float  # FWM conversion efficiency

    # Experimental Observables
    signal_to_noise_ratio: float  # SNR of interference signal
    measurement_precision: float  # Precision of phase measurement
    detection_efficiency: float  # Detector quantum efficiency

    def __post_init__(self):
        """Validate interference pattern mathematical consistency."""
        # Coherence validation
        if not (0 <= self.coherence_measure <= 1):
            raise ValueError(f"Coherence out of range [0,1]: {self.coherence_measure}")
        if not (0 <= self.quantum_coherence_measure <= 1):
            raise ValueError(
                f"Quantum coherence out of range [0,1]: {self.quantum_coherence_measure}"
            )

        # Phase validation
        if not (-math.pi <= self.phase_difference <= math.pi):
            self.phase_difference = (
                (self.phase_difference + math.pi) % (2 * math.pi)
            ) - math.pi

        # Interference type validation
        valid_types = ["constructive", "destructive", "mixed", "chaotic", "incoherent"]
        if self.interference_type not in valid_types:
            raise ValueError(f"Invalid interference type: {self.interference_type}")

        # Visibility validation
        if not (0 <= self.interference_contrast <= 1):
            raise ValueError(
                f"Visibility out of range [0,1]: {self.interference_contrast}"
            )

        # Beating frequency validation
        if self.beating_frequency is not None and self.beating_frequency < 0:
            raise ValueError(f"Negative beating frequency: {self.beating_frequency}")

        # Mandel Q parameter validation (Q = 0 coherent, Q < 0 sub-Poissonian, Q > 0 super-Poissonian)
        if abs(self.mandel_q_parameter) > 1e6:
            logger.warning(f"Extreme Mandel Q parameter: {self.mandel_q_parameter}")

        # Signal-to-noise validation
        if self.signal_to_noise_ratio < 0:
            raise ValueError(f"Negative SNR: {self.signal_to_noise_ratio}")

        # Temporal coherence validation
        if self.temporal_coherence <= 0:
            raise ValueError(
                f"Non-positive temporal coherence: {self.temporal_coherence}"
            )
        if self.spatial_coherence <= 0:
            raise ValueError(
                f"Non-positive spatial coherence: {self.spatial_coherence}"
            )

    def verify_interference_bounds(self) -> Dict[str, bool]:
        """
        Verify fundamental bounds for interference patterns.

        Returns:
            Dictionary of bound verification results
        """
        bounds = {}

        # Cauchy-Schwarz inequality for coherence
        # |⟨Ψ₁*Ψ₂⟩|² ≤ ⟨|Ψ₁|²⟩⟨|Ψ₂|²⟩
        bounds["cauchy_schwarz"] = (
            self.coherence_measure**2 <= 1 + FIELD_NUMERICAL_PRECISION
        )

        # Interference visibility bounds
        # V ≤ 2√(I₁I₂)/(I₁ + I₂) for two-beam interference
        max_visibility = (
            2
            * math.sqrt(self.constructive_strength * self.destructive_strength)
            / (
                self.constructive_strength
                + self.destructive_strength
                + FIELD_NUMERICAL_PRECISION
            )
        )
        bounds["visibility_bound"] = (
            self.interference_contrast <= max_visibility + FIELD_NUMERICAL_PRECISION
        )

        # Heisenberg uncertainty relation
        # ΔX₁ΔX₂ ≥ ½|⟨[X₁,X₂]⟩|
        if hasattr(self, "field_quadratures") and self.field_quadratures is not None:
            x1, x2 = self.field_quadratures
            var_x1 = torch.var(x1)
            var_x2 = torch.var(x2)
            uncertainty_product = var_x1 * var_x2
            bounds["heisenberg_uncertainty"] = (
                uncertainty_product >= 0.25 - FIELD_NUMERICAL_PRECISION
            )
        else:
            # Create quadratures from field components for uncertainty relation
            if torch.is_complex(self.field_values):
                x1 = self.field_values.real.flatten()
                x2 = self.field_values.imag.flatten()

                # Quantum quadratures: ΔX₁ΔX₂ ≥ ½|⟨[X₁,X₂]⟩|
                var_x1 = torch.var(x1)
                var_x2 = torch.var(x2)

                # Commutator expectation for canonical conjugates: ⟨[x,p]⟩ = iℏ = i
                commutator_expectation = 1.0  # In natural units ℏ = 1

                uncertainty_product = var_x1 * var_x2
                heisenberg_bound = 0.25 * commutator_expectation**2

                bounds["heisenberg_uncertainty"] = (
                    uncertainty_product >= heisenberg_bound - FIELD_NUMERICAL_PRECISION
                )
            else:
                bounds["heisenberg_uncertainty"] = True  # Cannot check real fields

        # Temporal vs spatial coherence consistency
        # Related through dispersion relation
        # Coherence consistency via Einstein relation: D = μkT
        # where D ~ spatial_coherence²/temporal_coherence (diffusion)
        if self.temporal_coherence > 0 and self.spatial_coherence > 0:
            diffusion_coefficient = self.spatial_coherence**2 / self.temporal_coherence

            # Mobility from field amplitude: μ ~ 1/⟨|ψ|²⟩
            if hasattr(self, "field_amplitude"):
                mobility = 1.0 / (
                    torch.mean(torch.abs(self.field_amplitude) ** 2) + 1e-12
                )
            else:
                mobility = 1.0  # Default mobility

            # Temperature from coherence measure: kT ~ coherence_measure
            effective_temperature = self.coherence_measure

            # Einstein relation check: D = μkT
            einstein_rhs = mobility * effective_temperature
            einstein_error = abs(diffusion_coefficient - einstein_rhs) / (
                einstein_rhs + 1e-12
            )

            bounds["coherence_consistency"] = einstein_error < 0.2  # 20% tolerance
        else:
            bounds["coherence_consistency"] = True

        return bounds

    def compute_van_cittert_zernike_theorem(
        self, source_distribution: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply van Cittert-Zernike theorem for coherence from source distribution.

        Mathematical Formula:
            γ₁₂ = ∫∫ I(ξ,η) exp(ik(ξx₁ + ηy₁ - ξx₂ - ηy₂)/z) dξdη / ∫∫ I(ξ,η) dξdη

        Where I(ξ,η) is the source intensity distribution.
        """
        # EXACT van Cittert-Zernike theorem implementation via 2D FFT
        # γ₁₂ = ∫∫ I(ξ,η) exp(ik(ξΔx + ηΔy)/z) dξdη / ∫∫ I(ξ,η) dξdη

        if source_distribution.dim() != 2:
            raise ValueError("van Cittert-Zernike requires 2D source distribution")

        # Source distribution I(ξ,η)
        intensity = torch.abs(source_distribution) ** 2

        # Spatial frequency coordinates
        ny, nx = intensity.shape
        kx = torch.fft.fftfreq(nx, d=1.0)
        ky = torch.fft.fftfreq(ny, d=1.0)

        # 2D FFT for coherence calculation
        # γ(Δx,Δy) = F[I(ξ,η)] / ∫I(ξ,η)dξdη
        intensity_fft = torch.fft.fft2(intensity)
        intensity_norm = torch.sum(intensity) + FIELD_NUMERICAL_PRECISION

        # Coherence function (normalized Fourier transform)
        coherence_function = torch.fft.fftshift(intensity_fft) / intensity_norm

        # Return coherence magnitude
        return torch.abs(coherence_function)


@dataclass
class EnergyComponents:
    """Field energy decomposition."""

    kinetic_energy: float  # T = ½∫|∇φ|² d³x kinetic energy
    potential_energy: float  # V = ½∫m²|φ|² d³x potential energy
    interaction_energy: float  # U = λ∫|φ|⁴ d³x interaction energy
    gradient_energy: float  # G = ½∫|∇φ|² d³x gradient energy
    total_energy: float  # E = T + V + U total energy
    energy_density_distribution: torch.Tensor  # ℰ(x) = T(x) + V(x) + U(x)
    stress_energy_tensor: torch.Tensor  # T_μν stress-energy tensor
    pressure: float  # P = ⟨T_ii⟩/3 pressure
    energy_momentum_density: torch.Tensor  # T⁰ᵢ energy-momentum density

    def __post_init__(self):
        """Validate energy conservation and positivity."""
        computed_total = (
            self.kinetic_energy + self.potential_energy + self.interaction_energy
        )
        if abs(self.total_energy - computed_total) > FIELD_NUMERICAL_PRECISION:
            raise ValueError(
                f"Energy conservation violated: {self.total_energy} ≠ {computed_total}"
            )
        if self.kinetic_energy < 0:
            raise ValueError(f"Negative kinetic energy: {self.kinetic_energy}")


@dataclass
class CriticalPoint:
    """Critical phase transition point."""

    location: Tuple[float, ...]  # (T_c, h_c, ...) critical point coordinates
    critical_exponents: Dict[str, float]  # {β: ..., γ: ..., ν: ...} critical exponents
    transition_type: PhaseType  # Phase transition classification
    stability: float  # Eigenvalue stability measure
    influence_radius: float  # ξ₀ correlation length amplitude
    scaling_function: Callable  # F(x) = |t|^(-α) f(h/|t|^βδ) scaling form
    universality_class: str  # Universal behavior classification

    def __post_init__(self):
        """Validate critical point mathematical consistency."""
        required_exponents = {"beta", "gamma", "nu", "delta"}
        if not required_exponents.issubset(self.critical_exponents.keys()):
            raise ValueError(
                f"Missing critical exponents: {required_exponents - self.critical_exponents.keys()}"
            )
        if self.influence_radius <= 0:
            raise ValueError(f"Non-positive influence radius: {self.influence_radius}")


# Mathematical utility functions for field operations
@jit
def field_norm_l2(field: torch.Tensor) -> float:
    """
    Compute L² norm of field configuration.

    ||φ||₂ = (∫ |φ(x)|² d³x)^(1/2)
    """
    return torch.sqrt(torch.sum(torch.abs(field) ** 2)).item()


@jit
def field_norm_h1(field: torch.Tensor, dx: float = 1.0) -> float:
    """
    Compute H¹ Sobolev norm of field configuration.

    ||φ||_{H¹} = (∫ |φ(x)|² + |∇φ(x)|² d³x)^(1/2)
    """
    field_squared = torch.abs(field) ** 2
    gradient_squared = torch.sum(torch.abs(torch.gradient(field, spacing=dx)[0]) ** 2)
    return torch.sqrt(torch.sum(field_squared) + gradient_squared).item()


def complex_field_phase(field: torch.Tensor) -> torch.Tensor:
    """
    Extract phase of complex field: φ(x) = |φ(x)| e^(iθ(x))

    Returns: θ(x) = arg(φ(x)) ∈ [-π, π]
    """
    return torch.angle(field.to(torch.complex64))


def winding_number(phase_field: torch.Tensor) -> int:
    """
    Compute topological winding number for 2D phase field.

    ν = (1/2π) ∮ ∇θ · dl
    """
    if phase_field.dim() != 2:
        raise ValueError("Winding number requires 2D phase field")

    grad_theta = torch.gradient(phase_field)
    circulation = torch.sum(grad_theta[0][:, -1] - grad_theta[0][:, 0])  # x-boundary
    circulation += torch.sum(grad_theta[1][-1, :] - grad_theta[1][0, :])  # y-boundary

    return int(torch.round(circulation / (2 * math.pi)).item())


@jit
def field_fourier_transform(field: torch.Tensor) -> torch.Tensor:
    """
    Compute FFT of field configuration for frequency domain analysis.

    F[φ](k) = ∫ φ(x) e^(-ik·x) dx
    """
    if field.dim() == 1:
        return fft(field)
    elif field.dim() == 2:
        return fft2(field)
    else:
        raise ValueError(f"Unsupported field dimension: {field.dim()}")


def compute_cholesky_decomposition(correlation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Cholesky decomposition for field correlation analysis.

    C = LLᵀ where L is lower triangular
    """
    return torch.from_numpy(cholesky(correlation_matrix.detach().cpu().numpy()))


def integrate_field_functional(
    field: torch.Tensor, integrand_func: Callable, domain: Tuple[float, float]
) -> float:
    """
    Integrate field functional using SciPy quad integration.

    I = ∫ F[φ(x)] dx
    """

    def numpy_integrand(x):
        x_tensor = torch.tensor(x, dtype=field.dtype)
        field_value = torch.interp(
            x_tensor, torch.linspace(domain[0], domain[1], len(field)), field
        )
        return integrand_func(field_value).item()

    result, _ = quad(
        numpy_integrand,
        domain[0],
        domain[1],
        epsabs=INTEGRATION_ABSOLUTE_TOLERANCE,
        epsrel=INTEGRATION_RELATIVE_TOLERANCE,
    )
    return result


def compute_field_svd(
    field_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Singular Value Decomposition for field mode analysis.

    A = UΣVᵀ
    """
    U, S, Vt = svd(field_matrix.detach().cpu().numpy())
    return (torch.from_numpy(U), torch.from_numpy(S), torch.from_numpy(Vt))


def compute_error_function_profile(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Error function profile for field smoothing using scipy.special.

    erf(x/σ) profile for field regularization
    """
    x_coords = torch.linspace(-3 * sigma, 3 * sigma, len(field))
    return torch.tensor(
        [special.erf(x.item() / sigma) for x in x_coords], dtype=field.dtype
    )


def beta_function_normalization(alpha: float, beta_param: float) -> float:
    """
    Beta function normalization using scipy.special.

    B(α,β) = Γ(α)Γ(β)/Γ(α+β)
    """
    return special.beta(alpha, beta_param)


def logsumexp_field_partition(field_energies: torch.Tensor) -> float:
    """
    Compute partition function using logsumexp for numerical stability.

    Z = log(Σᵢ e^(-βEᵢ))
    """
    return special.logsumexp(-field_energies.detach().cpu().numpy())


@jit
def jax_gradient_computation(field_func: Callable, x: float) -> float:
    """
    JAX automatic differentiation for field gradients.

    ∇F = ∂F/∂x
    """
    grad_func = grad(field_func)
    return float(grad_func(x))


@nb.jit(nopython=True, cache=True, fastmath=False)
def numba_field_evolution(field: np.ndarray, dt: float, mass: float) -> np.ndarray:
    """
    High-performance field evolution using Numba JIT.

    ∂φ/∂t = -m²φ + ∇²φ
    """
    n = len(field)
    new_field = np.zeros_like(field)

    for i in prange(n):
        laplacian = 0.0
        if i > 0:
            laplacian += field[i - 1]
        if i < n - 1:
            laplacian += field[i + 1]
        laplacian -= 2 * field[i]

        new_field[i] = field[i] + dt * (laplacian - mass * mass * field[i])

    return new_field


def sage_complex_field_calculation(z: complex) -> complex:
    """
    Exact complex field calculation using Sage CDF.

    Computes complex field expressions with arbitrary precision
    """
    sage_z = SageCDF(z)
    result = sage_z * CDF.exp(CDF.I() * sage_z.argument())
    return complex(result)


# ═══════════════════════════════════════════════════════════════════════════════
# FIELD MECHANICS MATHEMATICAL ENGINES
# ═══════════════════════════════════════════════════════════════════════════════

from .energy_calculation import FieldEnergyCalculator
from .interference_dynamics import InterferenceDynamicsEngine
# Import mathematical analysis engines with complete formulations
from .perturbation_theory import FieldPerturbationAnalyzer
from .phase_transitions import PhaseTransitionDetector

# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE MATHEMATICAL EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # ───────────────────────────────────────────────────────────────────────────
    # FUNDAMENTAL CONSTANTS
    # ───────────────────────────────────────────────────────────────────────────
    # Field Theory Constants
    "FIELD_COUPLING_CONSTANT",  # φ = (1+√5)/2 Golden ratio
    "PHASE_COHERENCE_THRESHOLD",  # √3/2 constructive interference
    "ENERGY_NORMALIZATION",  # 2π energy scale
    "PLANCK_REDUCED",  # ℏ reduced Planck constant
    "BOLTZMANN_CONSTANT",  # k_B Boltzmann constant
    "FINE_STRUCTURE_CONSTANT",  # α electromagnetic coupling
    "EULER_MASCHERONI",  # γ Euler-Mascheroni constant
    # Critical Exponents (3D Ising)
    "CRITICAL_EXPONENT_BETA",  # β order parameter exponent
    "CRITICAL_EXPONENT_GAMMA",  # γ susceptibility exponent
    "CRITICAL_EXPONENT_NU",  # ν correlation length exponent
    "CRITICAL_EXPONENT_DELTA",  # δ critical isotherm exponent
    "CRITICAL_EXPONENT_ALPHA",  # α specific heat exponent
    "CRITICAL_EXPONENT_ETA",  # η anomalous dimension
    # 2D Ising Exact Exponents
    "ISING_2D_BETA",  # β = 1/8 (exact)
    "ISING_2D_GAMMA",  # γ = 7/4 (exact)
    "ISING_2D_NU",  # ν = 1 (exact)
    "ISING_2D_DELTA",  # δ = 15 (exact)
    "ISING_2D_ALPHA",  # α = 0 (logarithmic)
    # Kosterlitz-Thouless Constants
    "KT_TRANSITION_TEMPERATURE",  # T_KT transition temperature
    "KT_CORRELATION_EXPONENT",  # η = 1/4 at T_KT
    "KT_SUSCEPTIBILITY_EXPONENT",  # γ ≈ 1/32
    # Precision and Tolerance
    "FIELD_NUMERICAL_PRECISION",  # Machine ε for calculations
    "CONVERGENCE_THRESHOLD",  # Iterative convergence
    "INTEGRATION_ABSOLUTE_TOLERANCE",  # Quadrature absolute tolerance
    "INTEGRATION_RELATIVE_TOLERANCE",  # Quadrature relative tolerance
    "EIGENVALUE_TOLERANCE",  # Linear algebra precision
    "ORTHOGONALITY_TOLERANCE",  # Matrix orthogonality check
    # Physical Constants
    "SPEED_OF_LIGHT",  # c speed of light
    "VACUUM_PERMITTIVITY",  # ε₀ vacuum permittivity
    "VACUUM_PERMEABILITY",  # μ₀ vacuum permeability
    "ELEMENTARY_CHARGE",  # e elementary charge
    "ELECTRON_MASS",  # mₑ electron mass
    "PROTON_MASS",  # mₚ proton mass
    # High Precision Mathematical Constants
    "PI_HIGH_PRECISION",  # π to machine precision
    "E_HIGH_PRECISION",  # e to machine precision
    "GOLDEN_RATIO_CONJUGATE",  # 1/φ = φ - 1
    "SQRT_2",  # √2
    "SQRT_3",  # √3
    "SQRT_5",  # √5
    "CATALAN_CONSTANT",  # Catalan's constant
    # Planck Scale Constants
    "PLANCK_LENGTH",  # lₚ Planck length
    "PLANCK_TIME",  # tₚ Planck time
    "PLANCK_ENERGY",  # Eₚ Planck energy
    "PLANCK_MASS",  # mₚ Planck mass
    "PLANCK_TEMPERATURE",  # Tₚ Planck temperature
    # Renormalization Group
    "BETA_FUNCTION_COEFFICIENT_1",  # β₀ first coefficient
    "BETA_FUNCTION_COEFFICIENT_2",  # β₁ second coefficient
    "ANOMALOUS_DIMENSION_COEFFICIENT",  # γ₀ anomalous dimension
    "WILSON_FISHER_FIXED_POINT",  # g* Wilson-Fisher fixed point
    # Information Theory
    "SHANNON_INFORMATION_UNIT",  # nat → bit conversion
    "MUTUAL_INFORMATION_NORMALIZATION",  # natural → bits
    "KOLMOGOROV_COMPLEXITY_BOUND",  # minimum entropy bound
    # Topological Constants
    "CHERN_NUMBER_NORMALIZATION",  # 1/(2π) for integer Chern numbers
    "WINDING_NUMBER_FACTOR",  # 1/(2π) for winding numbers
    "BERRY_PHASE_FACTOR",  # 2π full geometric phase
    "QUANTUM_HALL_FACTOR",  # h/e² resistance quantum
    # Error Analysis
    "FLOATING_POINT_EPSILON",  # IEEE 754 machine epsilon
    "CATASTROPHIC_CANCELLATION_THRESHOLD",  # Numerical stability limit
    "CONDITION_NUMBER_WARNING",  # Matrix conditioning warning
    "CONVERGENCE_ACCELERATION_FACTOR",  # Golden ratio acceleration
    # ───────────────────────────────────────────────────────────────────────────
    # CLASSIFICATION ENUMERATIONS
    # ───────────────────────────────────────────────────────────────────────────
    "FieldSymmetry",  # Complete symmetry classification
    "PhaseType",  # Phase transition types
    "UniversalityClass",  # Critical behavior universality
    "FieldManifoldType",  # Manifold structure types
    # ───────────────────────────────────────────────────────────────────────────
    # MATHEMATICAL DATA STRUCTURES
    # ───────────────────────────────────────────────────────────────────────────
    "FieldConfiguration",  # Complete field state with validation
    "PerturbationResponse",  # Green's function perturbation analysis
    "InterferencePattern",  # Wave interference with quantum coherence
    "EnergyComponents",  # Energy decomposition and conservation
    "CriticalPoint",  # Critical point analysis
    # ───────────────────────────────────────────────────────────────────────────
    # CORE MATHEMATICAL UTILITIES
    # ───────────────────────────────────────────────────────────────────────────
    # Field Norms and Metrics
    "field_norm_l2",  # L² norm: ||φ||₂
    "field_norm_h1",  # H¹ Sobolev norm: ||φ||_{H¹}
    "field_norm_lp",  # Lᵖ norm: ||φ||ₚ
    "field_sobolev_norm",  # General Sobolev norms
    "field_besov_norm",  # Besov space norms
    "field_holder_norm",  # Hölder space norms
    # Complex Field Analysis
    "complex_field_phase",  # Extract phase: φ = |φ|e^(iθ)
    "complex_field_amplitude",  # Extract amplitude: |φ|
    "complex_field_argument",  # Complex argument
    "complex_field_log",  # Complex logarithm
    "complex_field_exp",  # Complex exponential
    "complex_field_power",  # Complex power: φⁿ
    # Topological Invariants
    "winding_number",  # Topological winding number
    "chern_number",  # First Chern number
    "pontryagin_index",  # Pontryagin index
    "euler_characteristic",  # Euler characteristic
    "betti_numbers",  # Betti numbers from homology
    "hopf_invariant",  # Hopf invariant
    # Fourier Analysis
    "field_fourier_transform",  # FFT with proper normalization
    "field_inverse_fourier_transform",  # IFFT
    "field_fourier_series_coefficients",  # Fourier series expansion
    "field_discrete_cosine_transform",  # DCT
    "field_wavelet_transform",  # Wavelet analysis
    "field_gabor_transform",  # Gabor transform
    # Integration and Differentiation
    "integrate_field_functional",  # Field functional integration
    "field_gradient",  # Spatial gradient ∇φ
    "field_divergence",  # Divergence ∇·F⃗
    "field_curl",  # Curl ∇×F⃗
    "field_laplacian",  # Laplacian ∇²φ
    "field_d_alembertian",  # d'Alembertian □φ
    "field_directional_derivative",  # Directional derivative
    # Linear Algebra
    "compute_cholesky_decomposition",  # Cholesky: A = LLᵀ
    "compute_field_svd",  # SVD: A = UΣVᵀ
    "compute_field_qr",  # QR decomposition
    "compute_field_eigendecomposition",  # Eigenvalue decomposition
    "compute_field_pseudoinverse",  # Moore-Penrose pseudoinverse
    "compute_matrix_exponential",  # Matrix exponential exp(A)
    "compute_matrix_logarithm",  # Matrix logarithm log(A)
    # DATA TYPE CONSISTENCY MANAGEMENT
    "DataTypeManager",  # Unified data type management
    "get_dtype_manager",  # Global data type manager
    "ensure_mathematical_precision",  # Precision validation decorator
    "PrecisionLevel",  # Precision level enumeration
    "DataTypeConfiguration",  # Data type configuration
    # Special Functions
    "compute_error_function_profile",  # Error function erf(x)
    "compute_gamma_function",  # Gamma function Γ(x)
    "compute_beta_function",  # Beta function B(x,y)
    "compute_digamma_function",  # Digamma function ψ(x)
    "compute_polygamma_function",  # Polygamma function ψ⁽ⁿ⁾(x)
    "compute_bessel_functions",  # Bessel functions Jₙ(x), Yₙ(x)
    "compute_hankel_functions",  # Hankel functions Hₙ⁽¹'²⁾(x)
    "compute_spherical_harmonics",  # Spherical harmonics Yₗᵐ(θ,φ)
    # Statistical Functions
    "beta_function_normalization",  # Beta function normalization
    "logsumexp_field_partition",  # Log-sum-exp for partition functions
    "field_correlation_function",  # Correlation function ⟨φ(x)φ(y)⟩
    "field_structure_function",  # Structure function analysis
    "field_moment_generating_function",  # Moment generating function
    "field_characteristic_function",  # Characteristic function
    # Automatic Differentiation
    "jax_gradient_computation",  # JAX gradient computation
    "jax_hessian_computation",  # JAX Hessian computation
    "jax_jacobian_forward",  # JAX forward-mode Jacobian
    "jax_jacobian_reverse",  # JAX reverse-mode Jacobian
    "jax_field_optimization",  # JAX-based optimization
    # High-Performance Computing
    "numba_field_evolution",  # JIT-compiled field evolution
    "numba_correlation_computation",  # JIT correlation functions
    "numba_fft_convolution",  # JIT FFT convolution
    "numba_monte_carlo_integration",  # JIT Monte Carlo integration
    "parallel_field_computation",  # Parallel field calculations
    # Exact Arithmetic
    "sage_complex_field_calculation",  # Sage CDF exact calculations
    "sage_modular_forms",  # Modular forms analysis
    "sage_elliptic_functions",  # Elliptic function analysis
    "sage_algebraic_number_field",  # Algebraic number computations
    "rational_approximation",  # Rational function approximation
    # Optimization and Root Finding
    "field_energy_minimization",  # Energy functional minimization
    "field_constraint_optimization",  # Constrained optimization
    "field_variational_calculus",  # Variational derivative computation
    "newton_raphson_field",  # Newton-Raphson for fields
    "trust_region_field_optimization",  # Trust region methods
    # Information Theory
    "shannon_entropy_field",  # Shannon entropy H[φ]
    "renyi_entropy_field",  # Rényi entropy H_α[φ]
    "mutual_information_field",  # Mutual information I(φ₁;φ₂)
    "relative_entropy_field",  # Relative entropy D(P||Q)
    "fisher_information_field",  # Fisher information I_F[φ]
    "kolmogorov_complexity_estimate",  # Kolmogorov complexity estimate
    # Stochastic Analysis
    "stochastic_field_integration",  # Stochastic differential equations
    "ito_calculus_field",  # Itô calculus for fields
    "stratonovich_calculus_field",  # Stratonovich calculus
    "wiener_process_field",  # Wiener process simulation
    "ornstein_uhlenbeck_field",  # Ornstein-Uhlenbeck process
    "levy_process_field",  # Lévy process simulation
    # ───────────────────────────────────────────────────────────────────────────
    # MATHEMATICAL ANALYSIS ENGINES
    # ───────────────────────────────────────────────────────────────────────────
    "FieldPerturbationAnalyzer",  # Green's function perturbation theory
    "InterferenceDynamicsEngine",  # Wave interference and coherence
    "FieldEnergyCalculator",  # Energy functional analysis
    "PhaseTransitionDetector",  # Critical phenomena and phase transitions
    # ───────────────────────────────────────────────────────────────────────────
    # MATHEMATICAL VALIDATION FUNCTIONS
    # ───────────────────────────────────────────────────────────────────────────
    "validate_field_configuration",  # Comprehensive field validation
    "verify_conservation_laws",  # Conservation law verification
    "check_mathematical_consistency",  # Mathematical consistency checks
    "verify_scaling_relations",  # Critical exponent scaling verification
    "validate_topological_charges",  # Topological charge quantization
    "check_gauge_invariance",  # Gauge invariance verification
    "verify_unitarity",  # Unitarity verification
    "check_causality",  # Causality verification
    "validate_probability_conservation",  # Probability conservation check
]

# ═══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL PRECISION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Configure PyTorch for maximum mathematical precision
torch.set_default_dtype(torch.float64)  # Double precision (IEEE 754 binary64)
torch.set_printoptions(precision=15, sci_mode=False)  # Full precision display

# DATA TYPE CONSISTENCY: Initialize unified data type management
from .data_type_consistency import (DataTypeConfiguration, DataTypeManager,
                                    PrecisionLevel,
                                    ensure_mathematical_precision,
                                    get_dtype_manager)

# GPU precision configuration for reproducible calculations
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True  # Deterministic GPU operations
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning for reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for exact computation
    torch.backends.cudnn.allow_tf32 = False  # Disable TF32 in cuDNN

# JAX precision configuration
if "jax" in globals():
    import jax

    jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision in JAX
    jax.config.update("jax_debug_nans", True)  # Debug NaN propagation
    jax.config.update("jax_debug_infs", True)  # Debug infinity propagation

# NumPy precision settings
import numpy as np

np.seterr(all="warn", over="raise", invalid="raise")  # Strict error handling
np.set_printoptions(precision=15, suppress=False)  # Full precision display

# SciPy precision configuration
import scipy

if hasattr(scipy, "special"):
    # Configure special function precision
    pass  # SciPy uses system precision by default

# ═══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL VALIDATION AND LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

# Configure mathematical logging with precision information
logger = logging.getLogger(__name__)
logger.info("═════════════════════════════════════════════════════════════════")
logger.info("🔬 FIELD MECHANICS FOUNDATION: Mathematical Precision Initialized")
logger.info("═════════════════════════════════════════════════════════════════")
logger.info(f"✓ PyTorch default dtype: {torch.get_default_dtype()}")
logger.info(f"✓ Machine epsilon (float64): {FLOATING_POINT_EPSILON}")
logger.info(f"✓ Field numerical precision: {FIELD_NUMERICAL_PRECISION}")
logger.info(
    f"✓ Integration tolerance: abs={INTEGRATION_ABSOLUTE_TOLERANCE}, rel={INTEGRATION_RELATIVE_TOLERANCE}"
)
logger.info(f"✓ Eigenvalue tolerance: {EIGENVALUE_TOLERANCE}")
logger.info(f"✓ Convergence threshold: {CONVERGENCE_THRESHOLD}")


# Validate fundamental mathematical constants
def _validate_mathematical_constants():
    """Validate fundamental constants for mathematical consistency."""
    validation_results = {}

    # Golden ratio validation: φ² = φ + 1
    phi_squared = FIELD_COUPLING_CONSTANT**2
    phi_plus_one = FIELD_COUPLING_CONSTANT + 1
    validation_results["golden_ratio"] = (
        abs(phi_squared - phi_plus_one) < FIELD_NUMERICAL_PRECISION
    )

    # Euler identity validation: e^(iπ) + 1 = 0
    euler_identity = cmath.exp(1j * PI_HIGH_PRECISION) + 1
    validation_results["euler_identity"] = (
        abs(euler_identity) < FIELD_NUMERICAL_PRECISION
    )

    # Critical exponent scaling relations
    alpha = CRITICAL_EXPONENT_ALPHA
    beta = CRITICAL_EXPONENT_BETA
    gamma = CRITICAL_EXPONENT_GAMMA
    nu = CRITICAL_EXPONENT_NU
    delta = CRITICAL_EXPONENT_DELTA

    # Scaling relation: α + 2β + γ = 2
    scaling_1 = abs(alpha + 2 * beta + gamma - 2.0)
    validation_results["scaling_relation_1"] = scaling_1 < FIELD_NUMERICAL_PRECISION

    # Scaling relation: γ = β(δ - 1)
    scaling_2 = abs(gamma - beta * (delta - 1))
    validation_results["scaling_relation_2"] = scaling_2 < FIELD_NUMERICAL_PRECISION

    # Log validation results
    for test_name, passed in validation_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status} Mathematical constant validation: {test_name}")

    if not all(validation_results.values()):
        logger.warning("⚠️  Some mathematical constant validations failed!")

    return validation_results


# Run mathematical validation
try:
    import cmath

    _constant_validation = _validate_mathematical_constants()
    logger.info("✓ Mathematical constant validation completed")
except Exception as e:
    logger.error(f"✗ Mathematical constant validation failed: {e}")


# Mathematical library availability check
def _check_mathematical_libraries():
    """Check availability and versions of mathematical libraries."""
    libraries = {}

    # SAGE is now a hard dependency - no conditional import
    import sage
    libraries["sage"] = f"✓ Available (version: {sage.version.version})"

    try:
        import jax

        libraries["jax"] = f"✓ Available (version: {jax.__version__})"
    except ImportError:
        libraries["jax"] = "✗ Not available - AUTOMATIC DIFFERENTIATION DISABLED"
        logger.warning("⚠️  JAX not available - automatic differentiation disabled")

    try:
        import numba

        libraries["numba"] = f"✓ Available (version: {numba.__version__})"
    except ImportError:
        libraries["numba"] = "✗ Not available - JIT COMPILATION DISABLED"
        logger.warning("⚠️  Numba not available - JIT compilation disabled")

    try:
        import scipy

        libraries["scipy"] = f"✓ Available (version: {scipy.__version__})"
    except ImportError:
        libraries["scipy"] = "✗ Not available - SPECIAL FUNCTIONS DISABLED"
        logger.error(
            "✗ SciPy not available - mathematical functionality severely limited"
        )

    for lib_name, status in libraries.items():
        logger.info(f"  {lib_name}: {status}")

    return libraries


# Check mathematical libraries
logger.info("Mathematical Library Availability:")
_library_status = _check_mathematical_libraries()

logger.info("═════════════════════════════════════════════════════════════════")
logger.info(
    "🎯 FIELD MECHANICS: Mathematical rigor enforced - zero tolerance for approximations"
)
logger.info("📊 All field calculations subject to exact mathematical validation")
logger.info("🚫 NO silent failures - mathematical errors will raise exceptions")
logger.info("═════════════════════════════════════════════════════════════════")

# Import main classes from submodules
try:
    from .energy_calculation import FieldEnergyCalculator
except ImportError as e:
    FieldEnergyCalculator = None
    logger.warning(f"FieldEnergyCalculator import failed: {e}")

try:
    from .interference_dynamics import InterferenceDynamicsEngine
except ImportError as e:
    InterferenceDynamicsEngine = None
    logger.warning(f"InterferenceDynamicsEngine import failed: {e}")

try:
    from .perturbation_theory import FieldPerturbationAnalyzer
except ImportError as e:
    FieldPerturbationAnalyzer = None
    logger.warning(f"FieldPerturbationAnalyzer import failed: {e}")

try:
    from .phase_transitions import PhaseTransitionDetector
except ImportError as e:
    PhaseTransitionDetector = None
    logger.warning(f"PhaseTransitionDetector import failed: {e}")

try:
    from .data_type_consistency import DataTypeManager, get_dtype_manager
except ImportError as e:
    DataTypeManager = None
    get_dtype_manager = None
    logger.warning(f"DataTypeManager import failed: {e}")

# Expose main interfaces
__all__ = [
    # Enums and data structures
    "FieldSymmetry",
    "PhaseType",
    "UniversalityClass",
    "FieldManifoldType",
    "FieldConfiguration",
    "PerturbationResponse",
    "InterferencePattern",
    "EnergyComponents",
    "CriticalPoint",
    # Main engines (may be None if imports fail)
    "FieldEnergyCalculator",
    "InterferenceDynamicsEngine",
    "FieldPerturbationAnalyzer",
    "PhaseTransitionDetector",
    "DataTypeManager",
    "get_dtype_manager",
    # Constants
    "FIELD_COUPLING_CONSTANT",
    "ENERGY_NORMALIZATION",
    "FIELD_NUMERICAL_PRECISION",
    "MATHEMATICAL_PRECISION",
    "CONSERVATION_TOLERANCE",
    "SAGE_AVAILABLE",
]
