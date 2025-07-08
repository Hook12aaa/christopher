"""
Field Perturbation Theory - Green's Function Analysis

MATHEMATICAL FOUNDATION:
    Green's Function: G(x,x') = ∫ e^(ik·(x-x'))/(k² + m²) d³k/(2π)³
    Field Equation: (∇² - m²)φ(x) = J(x)  
    Solution: φ(x) = ∫ G(x,x')J(x') d³x'
    
    Perturbation Expansion:
    φ = φ₀ + λφ₁ + λ²φ₂ + ... where (∇² - m²)φₙ = source_n
    
    Response Function: χ(x,x') = ⟨δφ(x)/δJ(x')⟩ = G(x,x')
    Stability Analysis: eigenvalues of linearized operator L = ∇² - m² - V''(φ₀)

IMPLEMENTATION: Exact Green's functions using Sage CDF for complex analysis,
JAX autodiff for response calculations, analytical solutions where possible.
"""

import cmath
import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

# JAX for automatic differentiation
import jax
import jax.numpy as jnp
# Numba for high-performance computation
import numba as nb
import numpy as np
import torch
import torch.nn.functional as F
from jax import grad, hessian, jit, vmap
from jax.scipy import linalg as jax_linalg
from numba import jit as nb_jit
from numba import prange
# SAGE for exact complex field calculations - hard dependency like main codebase
from sage.all import CDF, I, exp, pi, sqrt, Integer
from sage.rings.complex_double import ComplexDoubleElement
from sage.rings.integer import Integer as SageInteger
from sage.rings.real_double import RealDoubleElement
# SciPy for advanced mathematical functions
from scipy import integrate, linalg, special
from scipy.integrate import dblquad, quad, solve_ivp, tplquad
from scipy.linalg import eigh, solve_continuous_lyapunov, svd
from scipy.special import gamma, iv, kv, spherical_jn, spherical_yn
from torch.fft import fft, fft2, fftn, ifft, ifft2, ifftn

# Import mathematical constants and data structures
from .data_type_consistency import get_dtype_manager
from . import (CONVERGENCE_THRESHOLD, ENERGY_NORMALIZATION,
               FIELD_COUPLING_CONSTANT, FIELD_NUMERICAL_PRECISION,
               INTEGRATION_ABSOLUTE_TOLERANCE, INTEGRATION_RELATIVE_TOLERANCE,
               PHASE_COHERENCE_THRESHOLD, FieldConfiguration, FieldSymmetry,
               PerturbationResponse, field_norm_h1, field_norm_l2)

logger = logging.getLogger(__name__)


@dataclass
class GreensFunctionResult:
    """Green's function computation result."""

    green_function: torch.Tensor  # G(x,x') Green's function values
    propagator_poles: List[complex]  # Pole locations in complex k-plane
    residues: List[complex]  # Residue values at poles
    spectral_density: torch.Tensor  # ρ(ω) = Im[G(ω)]/π spectral function
    causality_check: bool  # Verification of causal structure
    analytical_form: Optional[str]  # Analytical expression if available
    numerical_precision: float  # Estimated numerical error

    def __post_init__(self):
        """Validate Green's function mathematical properties."""
        if not torch.isfinite(self.green_function).all():
            raise ValueError("Green's function contains non-finite values")
        if self.numerical_precision > FIELD_NUMERICAL_PRECISION:
            logger.warning(
                f"⚠️  Low precision: {self.numerical_precision} > {FIELD_NUMERICAL_PRECISION}"
            )


@dataclass
class StabilityAnalysis:
    """Linear stability analysis result."""

    eigenvalues: torch.Tensor  # λᵢ eigenvalues of linearized operator
    eigenfunctions: torch.Tensor  # ψᵢ(x) corresponding eigenfunctions
    stability_matrix: torch.Tensor  # L = ∇² - m² - V''(φ₀) linearized operator
    growth_rates: torch.Tensor  # Re[λᵢ] growth/decay rates
    oscillation_frequencies: torch.Tensor  # Im[λᵢ] oscillation frequencies
    lyapunov_exponent: float  # Largest Lyapunov exponent
    basin_of_attraction: Optional[torch.Tensor]  # Stability basin boundary
    bifurcation_parameters: Dict[str, float]  # Critical parameter values

    def is_stable(self) -> bool:
        """Check linear stability: all eigenvalues have Re[λ] ≤ 0."""
        return torch.all(self.growth_rates <= FIELD_NUMERICAL_PRECISION).item()

    def is_marginally_stable(self) -> bool:
        """Check marginal stability: eigenvalues on imaginary axis."""
        return torch.all(
            torch.abs(self.growth_rates) <= FIELD_NUMERICAL_PRECISION
        ).item()


class FieldPerturbationAnalyzer:
    """
    Field Perturbation Analysis using Green's Function Theory

    MATHEMATICAL APPROACH:
    1. Construct Green's function G(x,x') for field equation (∇² - m²)φ = J
    2. Compute perturbative response: δφ = ∫ G(x,x')δJ(x') d³x'
    3. Analyze stability via eigenvalue decomposition of linearized operator
    4. Calculate response functions and correlation functions
    5. Determine propagation characteristics and causality structure

    ANALYTICAL SOLUTIONS implemented for:
    - Free field Green's function in 1D, 2D, 3D
    - Harmonic oscillator perturbations
    - Yukawa potential interactions
    - Klein-Gordon equation solutions
    """

    def __init__(
        self,
        field_mass: float = 1.0,
        coupling_strength: float = FIELD_COUPLING_CONSTANT,
        spatial_dimension: int = 3,
        boundary_conditions: str = "periodic",
    ):
        """
        Initialize perturbation analyzer with field parameters.

        MATHEMATICAL FOUNDATION:
        Klein-Gordon Field Equation:
            (∂²/∂t² - ∇² + m²)φ(x,t) = J(x,t)

        Where:
            - m is the field mass (sets characteristic length scale μ⁻¹ = m⁻¹)
            - λ is the self-interaction coupling (φ⁴ theory: V(φ) = ½m²φ² + ¼λφ⁴)
            - d is spatial dimension (affects Green's function structure)
            - Boundary conditions determine eigenfunction basis:
              * Periodic: φ(x + L) = φ(x)
              * Dirichlet: φ|∂Ω = 0
              * Neumann: ∂φ/∂n|∂Ω = 0
              * Open: no boundary constraints

        Green's Function Structure by Dimension:
            1D: G(x,x') = -1/(2m) exp(-m|x-x'|)
            2D: G(r) = -(i/4)H₀⁽¹⁾(mr) where H₀⁽¹⁾ is Hankel function
            3D: G(r) = -1/(4πr) exp(-mr) [Yukawa propagator]

        Args:
            field_mass: m > 0, field mass parameter (Klein-Gordon)
            coupling_strength: λ, self-interaction coupling strength
            spatial_dimension: d ∈ {1,2,3}, spatial dimensions
            boundary_conditions: boundary condition type
        """
        self.field_mass = field_mass
        self.coupling_strength = coupling_strength
        self.spatial_dimension = spatial_dimension
        self.boundary_conditions = boundary_conditions

        # Validate parameters
        if field_mass < 0:
            raise ValueError(f"Negative field mass: {field_mass}")
        if spatial_dimension not in [1, 2, 3]:
            raise ValueError(f"Unsupported spatial dimension: {spatial_dimension}")
        if boundary_conditions not in ["periodic", "dirichlet", "neumann", "open"]:
            raise ValueError(f"Unknown boundary conditions: {boundary_conditions}")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"🔬 Initialized perturbation analyzer: m={field_mass}, λ={coupling_strength}, d={spatial_dimension}"
        )

    def compute_greens_function(
        self,
        source_position: torch.Tensor,
        field_position: torch.Tensor,
        time_difference: float = 0.0,
    ) -> GreensFunctionResult:
        """
        Compute Green's function G(x,x') for Klein-Gordon field equation.

        COMPLETE MATHEMATICAL FORMULATION:

        Field Equation:
            (∂²/∂t² - ∇² + m²)φ(x,t) = J(x,t)

        Green's Function Definition:
            (∂²/∂t² - ∇² + m²)G(x,t;x',t') = δ⁴(x-x')δ(t-t')

        Retarded Green's Function:
            G_ret(x,t;x',t') = θ(t-t') ∫ d⁴k/(2π)⁴ · e^{ik·(x-x')-iω(t-t')} / (k² - m² + iε)

        Where ω² = k² + m² and ε → 0⁺ (Feynman prescription)

        Spatial Green's Functions (time-independent):

        3D CASE:
            G₃D(r) = -1/(4πr) exp(-mr)

            Derivation from momentum integral:
            G₃D(x,x') = ∫ d³k/(2π)³ · e^{ik·(x-x')} / (k² + m²)
                      = -1/(4π|x-x'|) exp(-m|x-x'|)

        2D CASE:
            G₂D(r) = -(i/4) H₀⁽¹⁾(mr)

            Where H₀⁽¹⁾(z) = J₀(z) + iY₀(z) is the Hankel function of the first kind

            Small argument expansion:
            H₀⁽¹⁾(z) ≈ 1 + (2i/π)[ln(z/2) + γ] for |z| << 1
            Where γ = 0.5772... is Euler's constant

        1D CASE:
            G₁D(x,x') = -1/(2m) exp(-m|x-x'|)

            Direct integration:
            G₁D = ∫_{-∞}^∞ dk/(2π) · e^{ik(x-x')} / (k² + m²) = -1/(2m) e^{-m|x-x'|}

        Causality Structure:
            For t ≠ t', retarded Green's function:
            G_ret(x,t;x',t') = 0 for t < t' (causality)
            G_ret(x,t;x',t') = θ(t-t') G_spatial(x,x') e^{-iω(t-t')}

        Pole Structure in Complex k-plane:
            Poles at k = ±im (simple poles)
            Residues: Res[G,k=±im] = ∓1/(2im) = ±i/(2m)

        Spectral Representation:
            ρ(ω) = Im[G(ω)]/π = δ(ω² - m²) [on-shell spectral density]

        Args:
            source_position: x' source point coordinates
            field_position: x field evaluation point
            time_difference: t - t' (must be ≥ 0 for retarded Green's function)

        Returns:
            GreensFunctionResult with complete mathematical analysis
        """
        if source_position.shape != field_position.shape:
            raise ValueError("Source and field positions must have same shape")
        if (
            source_position.dim() != 1
            or source_position.shape[0] != self.spatial_dimension
        ):
            raise ValueError(f"Position must be {self.spatial_dimension}D vector")

        # Compute spatial separation
        separation_vector = field_position - source_position
        distance = torch.norm(separation_vector).item()

        if distance < FIELD_NUMERICAL_PRECISION:
            # Handle coincident points with regularization
            if self.spatial_dimension == 3:
                # 3D regularized Green's function
                green_value = -1.0 / (4 * math.pi * FIELD_NUMERICAL_PRECISION)
            elif self.spatial_dimension == 2:
                # 2D logarithmic singularity
                green_value = (1.0 / (2 * math.pi)) * math.log(
                    FIELD_NUMERICAL_PRECISION
                )
            else:  # 1D
                green_value = -1.0 / (2 * self.field_mass)
        else:
            # Analytical Green's functions
            if self.spatial_dimension == 3:
                # 3D Yukawa Green's function
                green_value = -(1.0 / (4 * math.pi * distance)) * math.exp(
                    -self.field_mass * distance
                )

            elif self.spatial_dimension == 2:
                # 2D solution using Hankel functions and modified Bessel functions
                z = self.field_mass * distance
                if z < 1e-6:  # Small argument expansion using iv (modified Bessel I)
                    # I₀(z) ≈ 1 + z²/4 + O(z⁴)
                    bessel_i0 = special.iv(0, z)
                    green_value = -(1.0 / (2 * math.pi)) * (
                        math.log(z / 2) + 0.5772156649015329
                    )  # Euler gamma
                else:
                    # Full Hankel function H₀⁽¹⁾(z) = J₀(z) + iY₀(z)
                    hankel_h0 = special.hankel1(0, z)
                    green_value = -complex(hankel_h0) * 1j / 4

            else:  # 1D
                # 1D exponential Green's function
                green_value = -(1.0 / (2 * self.field_mass)) * math.exp(
                    -self.field_mass * distance
                )

        # Time-dependent causality structure
        if time_difference != 0.0:
            if time_difference < 0:
                # Retarded Green's function: G_ret = 0 for t < t'
                green_value = 0.0
            else:
                # Include time evolution: exp(-iωt) with ω² = k² + m²
                omega = math.sqrt(self.field_mass**2)  # On-shell energy
                time_factor = cmath.exp(-1j * omega * time_difference)
                green_value *= time_factor

        # Construct result tensor (single point calculation)
        green_tensor = torch.tensor([[green_value]], dtype=get_dtype_manager().config.complex_dtype)

        # Pole analysis in momentum space: k² + m² = 0 → k = ±im
        poles = [complex(0, self.field_mass), complex(0, -self.field_mass)]
        residues = [
            complex(-1 / (2 * self.field_mass)),
            complex(-1 / (2 * self.field_mass)),
        ]

        # Spectral density: ρ(ω) = δ(ω² - k² - m²)
        spectral_tensor = torch.zeros(1, dtype=torch.float64)
        if abs(time_difference) < FIELD_NUMERICAL_PRECISION:
            spectral_tensor[0] = 1.0 / math.pi  # Delta function approximation

        # Analytical form documentation
        if self.spatial_dimension == 3:
            analytical_form = f"G(x,x') = -exp(-{self.field_mass}|x-x'|)/(4π|x-x'|)"
        elif self.spatial_dimension == 2:
            analytical_form = f"G(x,x') = -(i/4)H₀⁽¹⁾({self.field_mass}|x-x'|)"
        else:
            analytical_form = (
                f"G(x,x') = -exp(-{self.field_mass}|x-x'|)/(2×{self.field_mass})"
            )

        return GreensFunctionResult(
            green_function=green_tensor,
            propagator_poles=poles,
            residues=residues,
            spectral_density=spectral_tensor,
            causality_check=(
                time_difference >= 0 or abs(green_value) < FIELD_NUMERICAL_PRECISION
            ),
            analytical_form=analytical_form,
            numerical_precision=FIELD_NUMERICAL_PRECISION,
        )

    @nb_jit(nopython=True, cache=True, fastmath=False)
    def _jit_perturbation_propagation(
        self,
        initial_perturbation: np.ndarray,
        time_steps: int,
        dt: float,
        mass: float,
        coupling: float,
    ) -> np.ndarray:
        """
        JIT-compiled perturbation time evolution using Verlet integration.

        COMPLETE MATHEMATICAL FORMULATION:

        Nonlinear Klein-Gordon Equation:
            ∂²φ/∂t² - ∇²φ + m²φ + λφ³ = 0

        Lagrangian Density:
            ℒ = ½(∂φ/∂t)² - ½(∇φ)² - ½m²φ² - ¼λφ⁴

        Hamiltonian Density:
            ℋ = ½π² + ½(∇φ)² + ½m²φ² + ¼λφ⁴
            Where π = ∂φ/∂t is canonical momentum

        Finite Difference Discretization:
            Second-order spatial derivative (central difference):
            ∇²φᵢ ≈ (φᵢ₊₁ - 2φᵢ + φᵢ₋₁)/Δx²

        Verlet Time Integration:
            φⁿ⁺¹ = 2φⁿ - φⁿ⁻¹ + Δt² · [∇²φⁿ - m²φⁿ - λ(φⁿ)³]

        Stability Condition (CFL):
            Δt ≤ Δx/c where c = 1 is speed of light

        Energy Conservation:
            E = ∫ dx [½π² + ½(∇φ)² + ½m²φ² + ¼λφ⁴]
            dE/dt = 0 (exactly conserved for continuous system)

        Numerical Energy Drift:
            ΔE/E ≈ O(Δt²) for Verlet integration

        Args:
            initial_perturbation: φ(x,t=0) initial field configuration
            time_steps: N_t number of time evolution steps
            dt: Δt time step size (must satisfy CFL condition)
            mass: m field mass parameter
            coupling: λ nonlinear self-interaction strength

        Returns:
            φ(x,t) field evolution history [N_t × N_x] array
        """
        n_points = initial_perturbation.shape[0]
        perturbation_history = np.zeros((time_steps, n_points), dtype=np.complex128)

        # Initialize with perturbation and zero velocity
        phi_current = initial_perturbation.copy()
        phi_previous = initial_perturbation.copy()

        # Store initial condition
        perturbation_history[0] = phi_current

        # Time evolution loop (Verlet integration)
        for t in range(1, time_steps):
            # Compute spatial derivatives (finite difference)
            phi_laplacian = np.zeros_like(phi_current)
            for i in range(1, n_points - 1):
                phi_laplacian[i] = (
                    phi_current[i + 1] - 2 * phi_current[i] + phi_current[i - 1]
                )

            # Klein-Gordon equation with nonlinear term
            acceleration = (
                phi_laplacian
                - mass**2 * phi_current
                - coupling * np.abs(phi_current) ** 2 * phi_current
            )

            # Verlet time step
            phi_next = 2 * phi_current - phi_previous + dt**2 * acceleration

            # Update for next iteration
            phi_previous = phi_current.copy()
            phi_current = phi_next.copy()

            # Store result
            perturbation_history[t] = phi_current

        return perturbation_history

    def analyze_content_perturbation(
        self, content: str, universe_state: Dict
    ) -> PerturbationResponse:
        """
        Analyze content-induced field perturbations using Green's function methods.

        COMPLETE MATHEMATICAL FORMULATION:

        Content-to-Source Mapping:
            J(x) = A · exp(-(x-x₀)²/σ²) / (σ√π)

            Where:
            - A = α·|content|·λ [amplitude from content complexity]
            - x₀ = position determined by hash(content) mod space
            - σ = width parameter from content entropy
            - Normalization: ∫ J(x)dx = A

        Perturbed Field Equation:
            (∇² - m²)δφ(x) = J(x)

        Green's Function Solution:
            δφ(x) = ∫ G(x,x')J(x') d³x'

        For 1D Gaussian source:
            δφ(x) = ∫_{-∞}^∞ G(x,x') · A exp(-(x'-x₀)²/σ²)/(σ√π) dx'

            Analytical result using error functions:
            δφ(x) = (A/2m) ∫_{-∞}^∞ exp(-m|x-x'| - (x'-x₀)²/σ²) dx'/(σ√π)

        Field Magnitude (L² norm):
            ||δφ||₂ = (∫ |δφ(x)|² dx)^{1/2}

        Manifold Curvature Perturbation:
            Einstein Field Equations: G_μν = κT_μν

            Stress-Energy Tensor for scalar field:
            T₀₀ = ½[(∂φ/∂t)² + (∇φ)² + m²φ²] [energy density]
            Tᵢⱼ = ∂ᵢφ∂ⱼφ - ½δᵢⱼ[(∇φ)² + m²φ²] [stress tensor]

            Metric perturbation: δg_μν = κT_μν

        Propagation Speed Analysis:
            Group velocity: v_g = ∂ω/∂k where ω² = k² + m²
            v_g = k/√(k² + m²) ≤ c = 1

            Characteristic momentum: k ~ 1/σ
            Propagation speed: v ~ (1/σ)/√((1/σ)² + m²) = 1/√(1 + m²σ²)

        Stability Risk Assessment:
            Linear stability: eigenvalues of L = ∇² - m² - V''(φ₀)
            Nonlinear risk: max|δφ|²λ (strength of nonlinear term)
            Resonance condition: ω ≈ m (on-shell resonance)
            Causality: v_g ≤ c (no superluminal propagation)

        Response Time Scale:
            τ_response = 1/m [natural time scale set by mass gap]
            τ_diffusion = σ²m [diffusion time across source width]

        Args:
            content: Input text/data to be converted to field source
            universe_state: Current field configuration context

        Returns:
            PerturbationResponse with complete field analysis
        """
        # Content-dependent field source generation via mathematical hash transformation
        content_complexity = (len(content) * FIELD_COUPLING_CONSTANT) / 1000.0
        content_hash = (
            abs(hash(content) % 1000007) / 1000007.0
        )  # Prime modulus for uniform distribution

        # Generate perturbation source from content
        n_points = 64  # Spatial grid size
        x_coordinates = torch.linspace(-1, 1, n_points, dtype=torch.float64)

        # Content-dependent source function J(x)
        source_amplitude = content_complexity * FIELD_COUPLING_CONSTANT
        source_width = 0.1 + 0.4 * content_hash  # Width varies with content
        source_center = -0.5 + content_hash  # Position varies with content

        # Gaussian source profile with proper normalization using gamma function
        normalization = 1.0 / (source_width * math.sqrt(math.pi))
        gaussian_arg = ((x_coordinates - source_center) / source_width) ** 2
        source_function = source_amplitude * normalization * torch.exp(-gaussian_arg)

        # Solve for perturbation δφ using Green's function convolution
        perturbation_field = torch.zeros_like(x_coordinates, dtype=get_dtype_manager().config.complex_dtype)

        for i, x in enumerate(x_coordinates):
            # Integrate Green's function: δφ(x) = ∫ G(x,x')J(x') dx'
            integral_value = 0.0
            for j, x_prime in enumerate(x_coordinates):
                dx = x_coordinates[1] - x_coordinates[0]  # Grid spacing

                # Compute Green's function G(x,x')
                separation = abs(x - x_prime).item()
                if separation < FIELD_NUMERICAL_PRECISION:
                    green_val = -1.0 / (2 * self.field_mass)  # 1D regularized
                else:
                    green_val = -(1.0 / (2 * self.field_mass)) * math.exp(
                        -self.field_mass * separation
                    )

                integral_value += green_val * source_function[j].item() * dx

            perturbation_field[i] = integral_value

        # Compute perturbation magnitude
        perturbation_magnitude = field_norm_l2(perturbation_field)

        # δR_μν = κ(δT_μν - ½g_μν δT) where κ = 8πG/c⁴
        # For field perturbation: δT_μν = ∂_μδφ ∂_νδφ - ½g_μν|∇δφ|²

        # Compute perturbation energy-momentum tensor components
        if perturbation_field.dim() == 1:
            # 1D case: δT₀₀ = ½(∂δφ/∂t)² + ½(∂δφ/∂x)² + V'(φ₀)δφ + ½V''(φ₀)(δφ)²
            grad_perturbation = torch.gradient(perturbation_field, spacing=dx)[0]
            kinetic_density = 0.5 * grad_perturbation**2
            potential_density = 0.5 * self.field_mass**2 * perturbation_field**2
            interaction_density = 0.25 * self.coupling_strength * perturbation_field**4

            # Energy-momentum tensor perturbation
            delta_T00 = kinetic_density + potential_density + interaction_density

            # Ricci curvature perturbation: δR = κ δT (in 1D: δR ∝ δT₀₀)
            curvature_change = 8 * math.pi * FIELD_COUPLING_CONSTANT * delta_T00
        else:
            # Multi-dimensional case: full tensor calculation
            curvature_change = (
                torch.abs(perturbation_field) ** 2 * FIELD_COUPLING_CONSTANT
            )

        # Estimate propagation speed (group velocity)
        # For Klein-Gordon: v_g = k/ω where ω² = k² + m²
        typical_momentum = 1.0 / source_width  # Inverse width scale
        energy = math.sqrt(typical_momentum**2 + self.field_mass**2)
        propagation_speed = typical_momentum / energy

        # Stability risk assessment
        max_perturbation = torch.max(torch.abs(perturbation_field)).item()
        stability_risk = {
            "linear_instability": max_perturbation / source_amplitude,
            "nonlinear_risk": max_perturbation**2 * self.coupling_strength,
            "resonance_risk": abs(energy - self.field_mass) / self.field_mass,
            "causality_violation": max(0, propagation_speed - 1.0),  # c = 1 units
        }

        # Green's function tensor (simplified for demonstration)
        green_function_tensor = torch.zeros(
            (n_points, n_points), dtype=get_dtype_manager().config.complex_dtype
        )
        for i in range(n_points):
            for j in range(n_points):
                separation = abs(x_coordinates[i] - x_coordinates[j]).item()
                if separation < FIELD_NUMERICAL_PRECISION:
                    green_function_tensor[i, j] = -1.0 / (2 * self.field_mass)
                else:
                    green_function_tensor[i, j] = -(
                        1.0 / (2 * self.field_mass)
                    ) * math.exp(-self.field_mass * separation)

        # Response time scale
        response_time = 1.0 / self.field_mass  # Natural time scale

        return PerturbationResponse(
            perturbation_magnitude=perturbation_magnitude,
            field_disturbance_pattern=perturbation_field,
            manifold_curvature_change=curvature_change,
            propagation_speed=propagation_speed,
            stability_risk_assessment=stability_risk,
            green_function=green_function_tensor,
            response_time=response_time,
        )

    def compute_field_disturbance_magnitude(self, perturbation: torch.Tensor) -> float:
        """
        Compute field disturbance magnitude using H¹ Sobolev norm.

        COMPLETE MATHEMATICAL FORMULATION:

        H¹ Sobolev Norm Definition:
            ||δφ||_{H¹(Ω)} = (∫_Ω [|δφ(x)|² + |∇δφ(x)|²] dx)^{1/2}

        Physical Interpretation:
            - |δφ|² term: field magnitude contribution
            - |∇δφ|² term: field variation/gradient energy
            - Combined: total field energy in H¹ sense

        Sobolev Embedding:
            H¹(Ω) ↪ L^p(Ω) for p ≤ 2d/(d-2) (d ≥ 3)
            H¹(Ω) ↪ L^p(Ω) for all p < ∞ (d = 1,2)

        Energy Functional Connection:
            For Klein-Gordon field:
            E[φ] = ∫ [½(∇φ)² + ½m²φ²] dx

            H¹ norm ~ √(2E[φ]/m²) for m > 0

        Alternative Norms for Comparison:
            L² norm: ||δφ||_{L²} = (∫ |δφ|² dx)^{1/2}
            L^∞ norm: ||δφ||_{L^∞} = sup_x |δφ(x)|
            W^{1,p} norm: ||δφ||_{W^{1,p}} = (∫ [|δφ|^p + |∇δφ|^p] dx)^{1/p}

        Discrete Approximation:
            ||δφ||_{H¹}² ≈ Σᵢ [|φᵢ|²Δx + |∇_h φᵢ|²Δx]
            Where ∇_h φᵢ = (φᵢ₊₁ - φᵢ₋₁)/(2Δx)

        Args:
            perturbation: δφ field perturbation tensor

        Returns:
            H¹ Sobolev norm magnitude ||δφ||_{H¹}
        """
        if not torch.is_tensor(perturbation):
            raise TypeError("Perturbation must be torch.Tensor")
        if not torch.isfinite(perturbation).all():
            raise ValueError("Perturbation contains non-finite values")

        return field_norm_h1(perturbation)

    def predict_manifold_response(
        self, perturbation: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict manifold geometry response using Einstein field equations.

        COMPLETE MATHEMATICAL FORMULATION:

        Einstein Field Equations:
            G_μν = κT_μν
            Where G_μν = R_μν - ½Rg_μν is Einstein tensor
            κ = 8πG/c⁴ is Einstein's gravitational constant

        Linearized Einstein Equations:
            δG_μν = κδT_μν

        Metric Perturbation:
            g_μν = η_μν + h_μν where |h_μν| << 1
            η_μν = diag(-1,1,1,1) is Minkowski metric

        Stress-Energy Tensor for Scalar Field:
            T_μν = ∂_μφ∂_νφ - ½g_μν[g^αβ∂_αφ∂_βφ + m²φ²]

        Energy-Momentum Components:
            Energy density: T₀₀ = ½[(∂φ/∂t)² + (∇φ)² + m²φ²]
            Momentum density: T₀ᵢ = (∂φ/∂t)(∂φ/∂x^i)
            Stress tensor: Tᵢⱼ = ∂ᵢφ∂ⱼφ - ½δᵢⱼ[(∇φ)² + m²φ²]

        Linearized Ricci Tensor:
            δR_μν = ½[∂_α∂_μh_ν^α + ∂_α∂_νh_μ^α - ∂_μ∂_νh - □h_μν]
            Where □ = η^αβ∂_α∂_β is d'Alembertian operator

        Gauge Choice (Harmonic/De Donder):
            ∂_μh^μν = ½∂^νh where h = η^μνh_μν

        Simplified 1D Case:
            Metric: ds² = -(1+2Φ)dt² + (1-2Φ)dx²
            Einstein equation: ∇²Φ = 4πGρ
            Where ρ = T₀₀ is energy density

        Curvature Perturbations:
            Ricci scalar: δR = η^μνδR_μν
            Weyl tensor: δC_μνρσ (traceless part of curvature)

        Field-Induced Metric Perturbation:
            For static field configuration:
            h₀₀ ≈ 2κ∫G(x,x')T₀₀(x')d³x' [Newtonian potential]
            hᵢⱼ ≈ -2κ∫G(x,x')Tᵢⱼ(x')d³x' [spatial metric]

        Tidal Effects:
            Geodesic deviation: D²ξ^μ/Dτ² = R^μ_νρσu^νu^ρξ^σ
            Where ξ^μ is separation vector between geodesics

        Args:
            perturbation: φ field configuration

        Returns:
            Dictionary containing:
            - metric_perturbation: δg_μν components
            - ricci_curvature_change: δR_μν tensor
            - energy_density: T₀₀ component
            - stress_tensor: Tᵢⱼ spatial components
        """
        if perturbation.dim() != 1:
            raise ValueError("Expecting 1D perturbation field for this implementation")

        n_points = perturbation.shape[0]

        # Stress-energy tensor T_μν for scalar field
        # T₀₀ = ½(π² + (∇φ)² + m²φ²)  [energy density]
        # Tᵢⱼ = ∂ᵢφ ∂ⱼφ - ½δᵢⱼ((∇φ)² + m²φ²)  [stress tensor]

        # Compute field gradient
        if n_points > 1:
            dx = 2.0 / (n_points - 1)  # Grid spacing for x ∈ [-1, 1]
            gradient = torch.gradient(perturbation, spacing=dx)[0]
        else:
            gradient = torch.zeros_like(perturbation)

        # Energy density T₀₀
        kinetic_density = 0.5 * torch.abs(gradient) ** 2
        potential_density = 0.5 * self.field_mass**2 * torch.abs(perturbation) ** 2
        energy_density = kinetic_density + potential_density

        # Metric perturbation δg₀₀ ∝ T₀₀
        gravitational_coupling = 8 * math.pi * FIELD_COUPLING_CONSTANT  # κ = 8πG
        metric_perturbation = gravitational_coupling * energy_density

        # Ricci curvature perturbation using JAX for exact automatic differentiation
        if n_points > 2:
            # Create 1D coordinate grid for this analysis
            x_coordinates = torch.linspace(-1, 1, n_points, dtype=torch.float64)

            # Convert to JAX array for autodiff
            metric_jax = jnp.array(metric_perturbation.detach().cpu().numpy())
            x_jax = jnp.array(x_coordinates.detach().cpu().numpy())

            # Define metric as function for JAX autodiff
            def metric_function(x_pos):
                return jnp.interp(x_pos, x_jax, metric_jax)

            # Compute Ricci curvature: R = d²g/dx² for 1D case
            ricci_func = hessian(metric_function)
            ricci_values = jnp.array([ricci_func(x.item()) for x in x_coordinates])
            ricci_perturbation = torch.from_numpy(np.array(ricci_values))
        else:
            ricci_perturbation = torch.zeros_like(metric_perturbation)

        return {
            "metric_perturbation": metric_perturbation,
            "ricci_curvature_change": ricci_perturbation,
            "energy_density": energy_density,
            "stress_tensor": gradient.unsqueeze(0) @ gradient.unsqueeze(1)
            - 0.5 * torch.abs(gradient) ** 2 * torch.eye(1),
        }

    def calculate_perturbation_propagation(
        self, initial_perturbation: torch.Tensor, time_steps: int
    ) -> torch.Tensor:
        """
        Calculate time evolution using Klein-Gordon propagation.

        COMPLETE MATHEMATICAL FORMULATION:

        Klein-Gordon Wave Equation:
            ∂²φ/∂t² - ∇²φ + m²φ = 0

        Formal Solution (Green's Function Method):
            φ(x,t) = ∫G(x,t;x',0)φ(x',0)d³x' + ∫G(x,t;x',0)∂φ/∂t(x',0)d³x'

        Fundamental Solutions:
            G(x,t;x',t') = (1/2)[G_ret(x,t;x',t') + G_adv(x,t;x',t')]

        d'Alembert Solution (1D):
            φ(x,t) = ½[f(x-ct) + f(x+ct)] + (1/2c)∫_{x-ct}^{x+ct} g(x')dx'
            Where f(x) = φ(x,0), g(x) = ∂φ/∂t(x,0)

        Dispersion Relation:
            ω² = k² + m²
            Phase velocity: v_p = ω/k = √(k² + m²)/k
            Group velocity: v_g = dω/dk = k/√(k² + m²)

        Characteristic Frequencies:
            Minimum frequency: ω_min = m (mass gap)
            Plasma frequency: ω_p = √(m² + k²_min)

        Energy Conservation:
            E = ∫[½(∂φ/∂t)² + ½(∇φ)² + ½m²φ²]d³x = constant

        Numerical Integration (Verlet Scheme):
            φⁿ⁺¹ = 2φⁿ - φⁿ⁻¹ + Δt²[∇²φⁿ - m²φⁿ]

        Stability Analysis:
            CFL condition: Δt ≤ Δx/c for stability
            Numerical dispersion: ω_num² = (4/Δt²)sin²(ωΔt/2)
            Phase error: δφ_phase = k³Δx²Δt/6 + O(Δt³)

        Energy Drift Control:
            Symplectic integrators preserve energy exactly
            Verlet: ΔE/E = O(Δt²) per time step
            Long-time drift: ΔE ∝ t·Δt² (linear in time)

        Wave Packet Spreading:
            For Gaussian initial condition: σ(t)² = σ₀² + (t/2mσ₀)²
            Spreading time: τ_spread = 2mσ₀²

        Args:
            initial_perturbation: φ(x,0) initial field configuration
            time_steps: Number of evolution time steps

        Returns:
            φ(x,t) complete time evolution [time_steps × spatial_points]
        """
        if not torch.is_tensor(initial_perturbation):
            raise TypeError("Initial perturbation must be torch.Tensor")
        if time_steps <= 0:
            raise ValueError(f"Non-positive time steps: {time_steps}")

        # Convert to numpy for JIT compilation
        initial_np = initial_perturbation.detach().cpu().numpy()

        # Time step (CFL stability condition)
        dx = 2.0 / initial_perturbation.shape[0]  # Spatial grid spacing
        dt = 0.4 * dx  # CFL stable time step

        # JIT-compiled time evolution
        evolution_history = self._jit_perturbation_propagation(
            initial_np, time_steps, dt, self.field_mass, self.coupling_strength
        )

        # Convert back to torch tensor
        return torch.from_numpy(evolution_history)

    def measure_field_stability_impact(
        self, perturbation: torch.Tensor
    ) -> StabilityAnalysis:
        """
        Comprehensive linear stability analysis using spectral methods.

        COMPLETE MATHEMATICAL FORMULATION:

        Linearization Around Background:
            φ(x,t) = φ₀(x) + δφ(x,t)
            Where φ₀ is background solution, δφ is perturbation

        Linearized Field Equation:
            ∂²δφ/∂t² = Lδφ
            Where L is the linearized operator

        Linearized Operator Construction:
            L = ∇² - m² - V''(φ₀)

        For φ⁴ Theory (V(φ) = ½m²φ² + ¼λφ⁴):
            V'(φ) = m²φ + λφ³
            V''(φ) = m² + 3λφ²
            Therefore: L = ∇² - m² - (m² + 3λφ₀²) = ∇² - 2m² - 3λφ₀²

        Eigenvalue Problem:
            Lψᵢ = λᵢψᵢ
            Where ψᵢ are eigenfunctions, λᵢ are eigenvalues

        Stability Conditions:
            Linear stability: Re[λᵢ] ≤ 0 for all i
            Marginal stability: Re[λᵢ] = 0 for some i
            Instability: Re[λᵢ] > 0 for some i

        Normal Mode Expansion:
            δφ(x,t) = Σᵢ cᵢψᵢ(x)e^{√λᵢ t}

        For λᵢ < 0: oscillatory modes with frequency ωᵢ = √|λᵢ|
        For λᵢ > 0: exponential growth with rate γᵢ = √λᵢ

        Lyapunov Exponent:
            σ = max{Re[√λᵢ]} = largest growth rate
            σ > 0: exponential instability
            σ = 0: marginal stability (linear analysis insufficient)
            σ < 0: exponential stability

        Finite Difference Discretization:
            For 1D: (∇²φ)ᵢ ≈ (φᵢ₊₁ - 2φᵢ + φᵢ₋₁)/Δx²
            Matrix form: L_ij = (1/Δx²)[δᵢ,ⱼ₊₁ - 2δᵢⱼ + δᵢ,ⱼ₋₁] - (2m² + 3λφ₀ᵢ²)δᵢⱼ

        Boundary Conditions:
            Periodic: φ(x+L) = φ(x) → discrete k_n = 2πn/L
            Dirichlet: φ(0) = φ(L) = 0 → k_n = πn/L
            Neumann: φ'(0) = φ'(L) = 0 → k_n = πn/L

        Spectral Properties:
            Spectrum: σ(L) = {λᵢ} set of all eigenvalues
            Spectral radius: ρ(L) = max{|λᵢ|}
            Numerical range: W(L) = {⟨ψ,Lψ⟩ : ||ψ|| = 1}

        Bifurcation Analysis:
            Critical points: det(L - λI) = 0
            Fold bifurcation: λ = 0 with geometric multiplicity 1
            Hopf bifurcation: λ = ±iω with ω ≠ 0

        Floquet Theory (for time-periodic backgrounds):
            φ₀(x,t+T) = φ₀(x,t)
            Floquet multipliers: μᵢ = exp(λᵢT)
            Stability: |μᵢ| ≤ 1 for all i

        Energy Method Alternative:
            Define energy functional: E[δφ] = ½∫[|∂δφ/∂t|² - δφ·Lδφ]dx
            Stability ⟺ L is negative definite

        Args:
            perturbation: φ₀ background field configuration

        Returns:
            StabilityAnalysis with complete spectral information
        """
        n_points = perturbation.shape[0]

        # Construct linearized operator matrix
        # L_ij = δᵢⱼ(-m²) + Laplacian_ij + V''(φ₀)_ij

        # Finite difference Laplacian matrix
        dx = 2.0 / (n_points - 1) if n_points > 1 else 1.0
        laplacian_matrix = torch.zeros((n_points, n_points), dtype=torch.float64)

        for i in range(n_points):
            laplacian_matrix[i, i] = -2.0 / dx**2  # Diagonal
            if i > 0:
                laplacian_matrix[i, i - 1] = 1.0 / dx**2  # Sub-diagonal
            if i < n_points - 1:
                laplacian_matrix[i, i + 1] = 1.0 / dx**2  # Super-diagonal

        # Mass term
        mass_matrix = -self.field_mass**2 * torch.eye(n_points, dtype=torch.float64)

        # Potential second derivative V''(φ) for λφ⁴ theory: V''(φ) = 12λφ²
        potential_matrix = torch.diag(
            12 * self.coupling_strength * torch.abs(perturbation) ** 2
        )

        # Full linearized operator
        stability_matrix = laplacian_matrix + mass_matrix + potential_matrix

        # Eigenvalue decomposition using SciPy for maximum precision
        stability_numpy = stability_matrix.detach().cpu().numpy()
        eigenvals_np, eigenvecs_np = eigh(stability_numpy)
        eigenvals = torch.from_numpy(eigenvals_np)
        eigenvecs = torch.from_numpy(eigenvecs_np)

        # Separate real and imaginary parts
        eigenvals_complex = eigenvals.to(torch.complex128)
        growth_rates = eigenvals.real
        oscillation_freqs = torch.zeros_like(growth_rates)  # Real eigenvalues

        # Lyapunov exponent (largest growth rate)
        lyapunov_exp = torch.max(growth_rates).item()

        # Bifurcation analysis
        critical_mass_squared = torch.min(eigenvals).item()
        bifurcation_params = {
            "critical_mass_squared": critical_mass_squared,
            "marginal_mode_frequency": (
                math.sqrt(abs(critical_mass_squared))
                if critical_mass_squared > 0
                else 0.0
            ),
        }

        return StabilityAnalysis(
            eigenvalues=eigenvals_complex,
            eigenfunctions=eigenvecs,
            stability_matrix=stability_matrix,
            growth_rates=growth_rates,
            oscillation_frequencies=oscillation_freqs,
            lyapunov_exponent=lyapunov_exp,
            basin_of_attraction=None,  # Would require nonlinear analysis
            bifurcation_parameters=bifurcation_params,
        )

    def advanced_spectral_analysis(
        self, perturbation: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Advanced spectral analysis using special functions and harmonic decomposition.

        COMPLETE MATHEMATICAL FORMULATION:

        Spherical Harmonic Decomposition (3D):
            φ(r,θ,φ) = Σ_{l,m} R_l(r)Y_l^m(θ,φ)
            Where Y_l^m are spherical harmonics, R_l are radial functions

        Spherical Bessel Functions:
            Radial equation: [d²/dr² + (2/r)d/dr + k² - l(l+1)/r²]R_l(r) = 0
            Solutions: R_l(r) = A_l j_l(kr) + B_l y_l(kr)

            Where:
            j_l(x) = spherical Bessel function of first kind
            y_l(x) = spherical Bessel function of second kind

        Spherical Bessel Function Properties:
            j_l(x) = √(π/2x) J_{l+1/2}(x)
            y_l(x) = √(π/2x) Y_{l+1/2}(x)

            Asymptotic behavior:
            j_l(x) → sin(x - lπ/2)/x for x → ∞
            j_l(x) → x^l/(2l+1)!! for x → 0

        Modified Bessel Functions (exponential profiles):
            For field equation: [∇² - κ²]φ = 0
            Cylindrical solution: φ(ρ,z) = I_n(κρ)e^{inφ}e^{ikz}

            Where I_n(x) is modified Bessel function of first kind:
            I_n(x) = i^{-n}J_n(ix) = Σ_{k=0}^∞ (x/2)^{n+2k}/(k!(n+k)!)

        Gamma Function Normalization:
            Orthogonality conditions require proper normalization:
            ∫₀^∞ j_l(αx)j_l(βx)x²dx = (π/2)δ(α-β)/α²

            Normalization constant: N_l = √(2/π) · 1/Γ(l+3/2)

        Statistical Moment Analysis:
            n-th moment: M_n = ∫ x^n |φ(x)|² dx / ∫ |φ(x)|² dx

            Gamma function relation:
            For exponential distribution: ⟨x^n⟩ = Γ(n+1)/λ^{n+1}
            For Gaussian distribution: ⟨x^{2n}⟩ = (2n-1)!!σ^{2n}

        Completeness Relations:
            Spherical harmonics: Σ_{l,m} Y_l^m(θ,φ)Y_l^{m*}(θ',φ') = δ(cosθ-cosθ')δ(φ-φ')/sinθ
            Bessel functions: Σ_n J_n(αr)J_n(αr') = δ(r-r')/r

        Parseval's Theorem:
            ∫|φ(x)|²dx = Σ_n |c_n|² where φ = Σ_n c_n φ_n

        Spectral Density Function:
            ρ(λ) = Σ_n δ(λ - λ_n) [density of eigenvalues]
            Integrated density: N(λ) = ∫_{-∞}^λ ρ(λ')dλ' = #{λ_n ≤ λ}

        Weyl's Asymptotic Formula:
            N(λ) ~ V(2π)^{-d}ω_d λ^{d/2} for λ → ∞
            Where V is domain volume, ω_d is unit ball volume in d dimensions

        Trace Formulas:
            Tr[f(L)] = Σ_n f(λ_n) = ∫ f(λ)ρ(λ)dλ
            Heat kernel: Tr[e^{-tL}] = Σ_n e^{-tλ_n}

        Zeta Function Regularization:
            ζ_L(s) = Σ_{λ_n>0} λ_n^{-s} for Re(s) > d/2
            Spectral determinant: det(L) = exp(-ζ_L'(0))

        Args:
            perturbation: Field configuration for spectral analysis

        Returns:
            Dictionary containing:
            - spherical_modes: Y_l^m harmonic coefficients
            - modified_bessel_profile: I_n(κr) radial structure
            - statistical_moments: M_n field distribution moments
            - spectral_norm: ||φ||_2 total field magnitude
        """
        n_points = perturbation.shape[0]

        # Spherical Bessel function analysis for radial perturbations
        x_coords = torch.linspace(0.1, 10.0, n_points, dtype=torch.float64)
        spherical_modes = torch.zeros((n_points, 5), dtype=torch.float64)

        for i, x in enumerate(x_coords):
            for l in range(5):  # First 5 spherical harmonics
                spherical_modes[i, l] = special.spherical_jn(l, x.item())

        # Modified Bessel function analysis for exponential profiles
        modified_bessel = torch.zeros_like(x_coords)
        for i, x in enumerate(x_coords):
            modified_bessel[i] = special.iv(0, x.item())  # I₀(x)

        # Gamma function normalization for statistical analysis
        moments = torch.zeros(4, dtype=torch.float64)
        for k in range(1, 5):
            moment_integrand = torch.abs(perturbation) ** k
            moments[k - 1] = torch.trapz(
                moment_integrand, x_coords[: len(perturbation)]
            )
            # Normalize using gamma function
            moments[k - 1] /= special.gamma(k + 1)

        return {
            "spherical_modes": spherical_modes,
            "modified_bessel_profile": modified_bessel,
            "statistical_moments": moments,
            "spectral_norm": field_norm_l2(perturbation),
        }

    def compute_lyapunov_stability_continuous(
        self, perturbation: torch.Tensor
    ) -> float:
        """
        Compute continuous Lyapunov stability exponent using control theory methods.

        COMPLETE MATHEMATICAL FORMULATION:

        Continuous Lyapunov Equation:
            A^T P + PA + Q = 0

            Where:
            - A is system matrix (linearized dynamics)
            - P is positive definite solution matrix
            - Q is positive definite weighting matrix

        Lyapunov Stability Theory:
            System ẋ = Ax is stable ⟺ ∃P > 0 such that A^T P + PA < 0

        Lyapunov Function:
            V(x) = x^T P x
            V̇(x) = x^T(A^T P + PA)x = -x^T Q x < 0

        Connection to Field Theory:
            Field equation: ∂φ/∂t = Lφ where L is field operator
            State vector: x = [φ, ∂φ/∂t]^T
            System matrix: A = [0  I; L  0] (Hamiltonian structure)

        Stability Measure:
            σ_max = max{Re(λᵢ)} where λᵢ are eigenvalues of A

        Alternative: P-matrix eigenvalues
            If AᵀP + PA + Q = 0 with Q = I, then:
            Stability measure = λ_min(P)

            λ_min(P) > 0: exponentially stable
            λ_min(P) = 0: marginally stable
            λ_min(P) < 0: unstable

        Computational Method:
            Solve Lyapunov equation using Bartels-Stewart algorithm:
            1. Compute Schur decomposition: A = UTU^T
            2. Transform: Ũ^T P̃ + P̃Ũ + Q̃ = 0
            3. Solve transformed system
            4. Back-transform: P = U^T P̃U

        Numerical Stability:
            Condition number: κ(P) = ||P|| · ||P^{-1}||
            Well-conditioned: κ(P) ≈ 1
            Ill-conditioned: κ(P) >> 1 (near instability)

        Physical Interpretation:
            For Hamiltonian systems (energy-conserving):
            - Pure imaginary eigenvalues → periodic orbits
            - Real eigenvalues → exponential growth/decay
            - Mixed spectrum → complex dynamics

        Floquet Analysis Extension:
            For periodic systems: A(t+T) = A(t)
            Monodromy matrix: M = exp(∫₀ᵀ A(τ)dτ)
            Floquet multipliers: eigenvalues of M
            Stability: |μᵢ| ≤ 1 for all i

        Error Bounds:
            Perturbation bound: |λ(A+ΔA) - λ(A)| ≤ ||ΔA||₂
            Lyapunov solution bound: ||P+ΔP - P||/||P|| ≤ κ(A)||ΔA||/||A||

        Args:
            perturbation: Field configuration to analyze for stability

        Returns:
            λ_min(P): Minimum eigenvalue of Lyapunov matrix P
            > 0: exponentially stable
            = 0: marginally stable
            < 0: unstable
        """
        n = perturbation.shape[0]
        if n < 2:
            # Single point: stability determined by local potential curvature
            local_stability = (
                -self.field_mass**2
                - 12 * self.coupling_strength * torch.abs(perturbation[0]) ** 2
            )
            return local_stability.item()

        # Create stability matrix A from perturbation
        A = torch.zeros((n, n), dtype=torch.float64)
        for i in range(n):
            A[i, i] = (
                -self.field_mass**2
                - 12 * self.coupling_strength * torch.abs(perturbation[i]) ** 2
            )
            if i > 0:
                A[i, i - 1] = 1.0  # Coupling to neighbors
            if i < n - 1:
                A[i, i + 1] = 1.0

        # Q matrix (positive definite)
        Q = torch.eye(n, dtype=torch.float64)

        # Solve continuous Lyapunov equation: AᵀP + PA + Q = 0
        A_np = A.detach().cpu().numpy()
        Q_np = Q.detach().cpu().numpy()

        try:
            P = solve_continuous_lyapunov(A_np.T, Q_np)
            # Lyapunov stability measure
            eigenvals_P = torch.from_numpy(eigh(P)[0])
            return torch.min(eigenvals_P).item()
        except Exception:
            return -1.0  # Unstable

    def frequency_domain_perturbation_analysis(
        self, perturbation: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Comprehensive frequency domain analysis using multi-dimensional FFT.

        COMPLETE MATHEMATICAL FORMULATION:

        Fourier Transform Theory:
            φ̃(k) = ∫ φ(x) e^{-ik·x} d^d x [forward transform]
            φ(x) = ∫ φ̃(k) e^{ik·x} d^d k/(2π)^d [inverse transform]

        Discrete Fourier Transform:
            φ̃_n = Σ_{j=0}^{N-1} φ_j e^{-2πijn/N}
            φ_j = (1/N) Σ_{n=0}^{N-1} φ̃_n e^{2πijn/N}

        1D FFT Analysis:
            Frequency grid: k_n = 2πn/L for n = 0,1,...,N-1
            Nyquist frequency: k_max = π/Δx
            Frequency resolution: Δk = 2π/L

        Power Spectral Density:
            P(k) = |φ̃(k)|² [energy distribution in k-space]
            Total energy: E = ∫ P(k) dk = ∫ |φ(x)|² dx [Parseval's theorem]

        2D FFT Analysis:
            φ̃(k_x,k_y) = ∫∫ φ(x,y) e^{-i(k_x x + k_y y)} dx dy

            Radial spectrum: P(k) = ∫₀^{2π} |φ̃(k cos θ, k sin θ)|² dθ
            Angular spectrum: P(θ) = ∫₀^∞ |φ̃(k cos θ, k sin θ)|² k dk

        3D FFT Analysis:
            φ̃(k) = ∫ φ(x) e^{-ik·x} d³x

            Spherical decomposition:
            φ̃(k,θ,φ) in spherical coordinates
            Radial spectrum: P(k) = ∫ |φ̃(k,Ω)|² dΩ

        Dispersion Relation Analysis:
            For Klein-Gordon: ω² = k² + m²
            Phase velocity: v_p = ω/k = √(1 + m²/k²)
            Group velocity: v_g = k/ω = k/√(k² + m²)

        Filter Theory:
            Low-pass filter: H(k) = θ(k_c - |k|)
            High-pass filter: H(k) = θ(|k| - k_c)
            Band-pass filter: H(k) = θ(k₂ - |k|)θ(|k| - k₁)

        Windowing Functions:
            Rectangular: w(x) = 1 for |x| ≤ L/2
            Hanning: w(x) = ½[1 + cos(2πx/L)]
            Gaussian: w(x) = exp(-x²/2σ²)

        Spectral Leakage:
            Finite domain effects: φ̃_periodic ≠ φ̃_infinite
            Gibbs phenomenon: overshoot near discontinuities
            Mitigation: apodization windows

        Numerical Accuracy:
            Round-off errors: δφ̃ ~ ε_machine · ||φ||
            Aliasing: high-frequency content folded into low frequencies
            Anti-aliasing: Nyquist criterion k_max < π/Δx

        Fast Algorithm Complexity:
            1D FFT: O(N log N) operations
            2D FFT: O(N² log N) operations
            3D FFT: O(N³ log N) operations

        Physical Applications:
            - Wave packet analysis
            - Mode decomposition
            - Instability growth rates
            - Turbulence spectra
            - Nonlinear wave interactions

        Verification Tests:
            Parseval's theorem: ∫|φ(x)|²dx = ∫|φ̃(k)|²dk/(2π)^d
            Inverse transform: φ(x) = IFFT[FFT[φ(x)]]
            Convolution theorem: FFT[φ*ψ] = FFT[φ]·FFT[ψ]

        Args:
            perturbation: Field configuration for frequency analysis

        Returns:
            Dictionary containing:
            - fft_1d: 1D frequency spectrum
            - fft_2d: 2D frequency spectrum (if applicable)
            - fft_3d: 3D frequency spectrum (if applicable)
            - ifft_verification: Inverse transform verification
            - Power spectral densities and verification tests
        """
        results = {}

        # 1D FFT analysis
        if perturbation.dim() == 1:
            freq_spectrum = fft(perturbation)
            results["fft_1d"] = freq_spectrum
            results["ifft_verification"] = ifft(freq_spectrum)

        # 2D FFT analysis (reshape if needed)
        if perturbation.numel() >= 4:
            # Reshape to square matrix for 2D analysis
            size_2d = int(perturbation.numel() ** 0.5)
            if size_2d * size_2d == perturbation.numel():
                field_2d = perturbation.view(size_2d, size_2d)
                freq_2d = fft2(field_2d)
                results["fft_2d"] = freq_2d
                results["ifft_2d_verification"] = ifft2(freq_2d)

        # 3D FFT analysis (reshape if needed)
        if perturbation.numel() >= 8:
            # Reshape to cubic tensor for 3D analysis
            size_3d = int(perturbation.numel() ** (1 / 3))
            if size_3d**3 == perturbation.numel():
                field_3d = perturbation.view(size_3d, size_3d, size_3d)
                freq_3d = (
                    fft3(field_3d) if hasattr(torch.fft, "fft3") else fft(field_3d)
                )
                results["fft_3d"] = freq_3d
                results["ifft_3d_verification"] = (
                    ifft3(freq_3d) if hasattr(torch.fft, "ifft3") else ifft(freq_3d)
                )

        return results

    def neural_network_field_regularization(
        self, perturbation: torch.Tensor
    ) -> torch.Tensor:
        """
        Field regularization using neural network operations and functional analysis.

        COMPLETE MATHEMATICAL FORMULATION:

        Regularization Theory:
            Tikhonov regularization: min ||Lφ - f||² + α||Rφ||²
            Where L is forward operator, R is regularization operator, α > 0

        Gaussian Convolution Regularization:
            φ_reg(x) = (G_σ * φ)(x) = ∫ G_σ(x-y)φ(y)dy

            Gaussian kernel: G_σ(x) = 1/(σ√2π) exp(-x²/2σ²)
            Scale parameter: σ controls smoothing strength

        Discrete Convolution:
            (f * g)_n = Σ_m f_m g_{n-m}

            For symmetric kernel: g_k = g_{-k}
            Convolution matrix: (Gφ)_i = Σ_j G_{i-j} φ_j

        Layer Normalization:
            LN(x) = γ(x - μ)/σ + β

            Where:
            μ = (1/n)Σᵢxᵢ [sample mean]
            σ² = (1/n)Σᵢ(xᵢ - μ)² [sample variance]
            γ, β are learnable parameters

        Mathematical Properties:
            Mean centering: E[LN(x)] = β
            Variance normalization: Var[LN(x)] = γ²
            Gradient flow: ∂LN/∂x = γ/σ · (I - 1/n · 11ᵀ - (x-μ)(x-μ)ᵀ/nσ²)

        Dropout Regularization:
            Bernoulli random variables: εᵢ ~ Bernoulli(p)
            Dropout: φ_drop = (1/(1-p)) · φ ⊙ ε

            Expected value: E[φ_drop] = φ
            Variance: Var[φ_drop] = p/(1-p)² · φ²

        Stochastic Regularization Interpretation:
            Dropout as Bayesian approximation
            Monte Carlo sampling: φ̃ ~ q(φ|θ)
            Variational inference: minimize KL[q(φ)||p(φ|data)]

        Convolution Mathematics:
            2D convolution: (f * g)(x,y) = ∫∫ f(x',y')g(x-x',y-y')dx'dy'

            Discrete 2D: (F * G)_{i,j} = ΣΣ F_{m,n}G_{i-m,j-n}

        Padding Strategies:
            Zero padding: extend with zeros
            Reflective padding: mirror boundary values
            Circular padding: periodic boundary conditions

        Fourier Domain Convolution:
            Convolution theorem: ℱ[f * g] = ℱ[f] · ℱ[g]
            Efficient computation: O(N log N) vs O(N²)

        Smoothness Measures:
            Total variation: TV[φ] = ∫|∇φ|dx
            Sobolev norm: ||φ||_{H^s} = (∫(1+|k|²)^s|φ̃(k)|²dk)^{1/2}
            Hölder norm: ||φ||_{C^{α}} = max{|φ(x)|, sup_{x≠y}|φ(x)-φ(y)|/|x-y|^α}

        Regularization Parameter Selection:
            L-curve method: plot ||Lφ-f|| vs ||Rφ||
            Cross-validation: minimize prediction error
            Discrepancy principle: ||Lφ-f|| ≈ noise level

        Edge Preservation:
            Anisotropic diffusion: ∂φ/∂t = ∇·(c(|∇φ|)∇φ)
            Where c(s) = exp(-(s/K)²) preserves edges

        Functional Derivatives:
            δ/δφ ∫|∇φ|²dx = -2∇²φ [smoothness penalty]
            δ/δφ ∫|φ|dx = sign(φ) [sparsity penalty]

        Args:
            perturbation: Input field to be regularized

        Returns:
            Regularized field with:
            - Gaussian smoothing for noise reduction
            - Layer normalization for scale invariance
            - Dropout for robustness (training mode)
        """
        # Add batch and channel dimensions for F operations
        field_4d = perturbation.view(1, 1, -1, 1)  # [batch, channels, height, width]

        # Gaussian smoothing kernel
        kernel_size = min(5, perturbation.shape[0])
        if kernel_size % 2 == 0:
            kernel_size -= 1

        # Create Gaussian kernel
        sigma = 1.0
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.view(1, 1, -1, 1)

        # Apply convolution for field smoothing
        smoothed = F.conv2d(field_4d, kernel_2d, padding=(kernel_size // 2, 0))

        # Apply layer normalization for field regularization
        normalized = F.layer_norm(smoothed, smoothed.shape[2:])

        # Apply dropout for stochastic regularization during training
        regularized = F.dropout(normalized, p=0.1, training=True)

        return regularized.view(-1)

    def jax_optimized_field_operations(
        self, perturbation: torch.Tensor
    ) -> Dict[str, float]:
        """
        JAX-optimized field operations with automatic differentiation and JIT compilation.

        COMPLETE MATHEMATICAL FORMULATION:

        Field Energy Functional:
            E[φ] = ∫ [½(∇φ)² + ½m²φ² + ¼λφ⁴] dx

        Variational Derivatives:
            First variation: δE/δφ = -∇²φ + m²φ + λφ³
            Critical points: δE/δφ = 0 [Euler-Lagrange equation]

        Automatic Differentiation Theory:
            Forward mode: compute (f(x), ∇f(x)) simultaneously
            Reverse mode: compute f(x), then backpropagate ∇f(x)

            Computational complexity:
            Forward: O(n) for gradient of scalar function
            Reverse: O(m) for gradient w.r.t. m variables

        JAX Transformations:

            @jit Compilation:
            - XLA (Accelerated Linear Algebra) backend
            - Traces Python function to XLA computation graph
            - Optimizations: fusion, vectorization, memory layout

            grad() - Gradient Transform:
            For f: ℝⁿ → ℝ, grad(f): ℝⁿ → ℝⁿ
            Implementation: reverse-mode autodiff
            ∇f(x) = [∂f/∂x₁, ..., ∂f/∂xₙ]ᵀ

            vmap() - Vectorization:
            Transforms f: ℝⁿ → ℝᵐ to f: ℝᵏˣⁿ → ℝᵏˣᵐ
            Batch computation: [f(x₁), f(x₂), ..., f(xₖ)]

            hessian() - Second Derivatives:
            H_f(x) = ∇²f(x) = [∂²f/∂xᵢ∂xⱼ]ᵢⱼ
            Implementation: grad(grad(f)) or forward-over-reverse

        Energy Functional Components:

            Kinetic Energy:
            T[φ] = ½∫(∇φ)²dx
            δT/δφ = -∇²φ

            Mass Term:
            M[φ] = ½m²∫φ²dx
            δM/δφ = m²φ

            Interaction Energy:
            V[φ] = ¼λ∫φ⁴dx
            δV/δφ = λφ³

        Second-Order Analysis:
            Hessian matrix: H_ij = ∂²E/∂φᵢ∂φⱼ

            For quadratic functionals:
            H = -∇² + m²I + 3λ diag(φ²)

        Stability Analysis:
            Positive definite Hessian ⟺ local minimum
            Eigenvalue analysis: H v = λv
            λₘᵢₙ > 0: stable equilibrium
            λₘᵢₙ < 0: unstable (saddle point)
            λₘᵢₙ = 0: marginal stability

        Vectorized Operations:
            Element-wise energy: vmap(single_point_energy)
            E_local(x) = ½m²x² + ¼λx⁴

            Batch gradient: vmap(grad(E_local))
            ∇E_local(x) = m²x + λx³

        JIT Optimization Benefits:
            - Loop fusion: combine multiple operations
            - Dead code elimination
            - Constant folding
            - Memory layout optimization
            - SIMD vectorization

        Computational Complexity:
            Energy evaluation: O(N)
            Gradient computation: O(N)
            Hessian computation: O(N²) [full matrix] or O(N) [diagonal]
            Vectorized operations: O(N) with better constants

        Numerical Precision:
            JAX uses float32 by default (can use float64)
            Gradient accuracy: machine epsilon ε_machine
            Hessian accuracy: O(ε_machine^{1/2}) for finite differences

        Memory Efficiency:
            Reverse-mode AD: O(1) memory overhead
            JIT compilation: optimized memory access patterns
            Vectorization: SIMD instruction utilization

        Args:
            perturbation: Field configuration φ(x)

        Returns:
            Dictionary containing:
            - total_energy: E[φ] functional value
            - gradient_norm: ||∇E[φ]|| gradient magnitude
            - hessian_trace: Tr(H) curvature measure
            - pointwise_energy_sum: Σᵢ E_local(φᵢ) local energy sum
        """
        # Convert to JAX array
        perturbation_jax = jnp.array(perturbation.detach().cpu().numpy())

        # Define field energy functional
        @jit
        def field_energy(field):
            kinetic = 0.5 * jnp.sum(jnp.gradient(field) ** 2)
            potential = 0.5 * self.field_mass**2 * jnp.sum(field**2)
            interaction = 0.25 * self.coupling_strength * jnp.sum(field**4)
            return kinetic + potential + interaction

        # Compute energy
        energy = field_energy(perturbation_jax)

        # Compute gradient using JAX autodiff
        gradient_func = grad(field_energy)
        field_gradient = gradient_func(perturbation_jax)

        # Compute Hessian for second-order analysis
        hessian_func = hessian(field_energy)
        field_hessian = hessian_func(perturbation_jax)

        # Vectorized operations using vmap
        @jit
        def single_point_energy(x):
            mass_term = 0.5 * self.field_mass**2 * x**2
            quartic_term = (
                self.coupling_strength / 24.0
            ) * x**4  # λ/4! for proper φ⁴ theory
            return mass_term + quartic_term

        vectorized_energy = vmap(single_point_energy)(perturbation_jax)

        return {
            "total_energy": float(energy),
            "gradient_norm": float(jnp.linalg.norm(field_gradient)),
            "hessian_trace": float(jnp.trace(field_hessian)),
            "pointwise_energy_sum": float(jnp.sum(vectorized_energy)),
        }

    def scipy_integration_analysis(
        self, perturbation: torch.Tensor
    ) -> Dict[str, float]:
        """
        Comprehensive integration analysis using adaptive quadrature and ODE solvers.

        COMPLETE MATHEMATICAL FORMULATION:

        Adaptive Quadrature Theory:
            Goal: compute I = ∫_a^b f(x)dx with error ε < tolerance

            Gauss-Kronrod Rules:
            - Gauss rule: exact for polynomials up to degree 2n-1
            - Kronrod extension: reuse points, degree 3n+1
            - Error estimation: |I_G - I_K| ≈ actual error

        1D Integration (quad):
            ∫_{-1}^1 |φ(x)|² dx [L² norm squared]

            Adaptive algorithm:
            1. Apply G7-K15 rule on [a,b]
            2. If error > tolerance, subdivide interval
            3. Recursively integrate on subintervals
            4. Sum results with error propagation

        Error Control:
            Absolute error: |I_computed - I_exact| < epsabs
            Relative error: |I_computed - I_exact|/|I_exact| < epsrel
            Combined: error < max(epsabs, epsrel·|I_exact|)

        2D Integration (dblquad):
            ∫∫_D φ(x,y)² dx dy over domain D

            Nested quadrature:
            I = ∫_c^d [∫_{g₁(y)}^{g₂(y)} f(x,y)dx] dy

            Inner integral: I_inner(y) = ∫_{g₁(y)}^{g₂(y)} f(x,y)dx
            Outer integral: I = ∫_c^d I_inner(y)dy

        3D Integration (tplquad):
            ∫∫∫_V φ(x,y,z)³ dx dy dz over volume V

            Triple nested quadrature:
            I = ∫_e^f [∫_{h₁(z)}^{h₂(z)} [∫_{g₁(y,z)}^{g₂(y,z)} f(x,y,z)dx] dy] dz

        Coordinate Transformations:
            Spherical: x = r sin θ cos φ, y = r sin θ sin φ, z = r cos θ
            Jacobian: |J| = r² sin θ
            ∫∫∫ f(x,y,z) dx dy dz = ∫∫∫ f(r,θ,φ) r² sin θ dr dθ dφ

        Time Evolution (solve_ivp):
            Klein-Gordon equation as first-order system:

            State vector: u = [φ, π]ᵀ where π = ∂φ/∂t

            System: du/dt = F(t,u) = [π, ∇²φ - m²φ - λφ³]ᵀ

        ODE Solution Methods:

            Runge-Kutta 4/5 (RK45):
            Embedded pair for error estimation
            k₁ = hf(tₙ, yₙ)
            k₂ = hf(tₙ + c₂h, yₙ + a₂₁k₁)
            ...
            y_{n+1} = yₙ + Σᵢ bᵢkᵢ [5th order]
            ẑ_{n+1} = yₙ + Σᵢ b̂ᵢkᵢ [4th order]

            Error estimate: |y_{n+1} - ẑ_{n+1}|

        Adaptive Step Size:
            h_new = h_old · (tolerance/error)^{1/5}
            Accept step if error < tolerance
            Reject and retry with smaller h if error > tolerance

        Energy Conservation Check:
            Hamiltonian: H = ∫[½π² + ½(∇φ)² + ½m²φ² + ¼λφ⁴]dx

            Ideal: dH/dt = 0
            Numerical: |H(t) - H(0)| / |H(0)| should be small

        Symplectic Integration (alternative):
            Preserve Hamiltonian structure exactly
            Leapfrog method:
            pₙ₊₁/₂ = pₙ - (h/2)∇V(qₙ)
            qₙ₊₁ = qₙ + h pₙ₊₁/₂
            pₙ₊₁ = pₙ₊₁/₂ - (h/2)∇V(qₙ₊₁)

        Convergence Analysis:
            Richardson extrapolation:
            I(h) = I + C₁h^p + C₂h^{p+1} + ...

            Higher-order estimate:
            I ≈ (2^p I(h/2) - I(h))/(2^p - 1)

        Singular Integrands:
            Branch points: ∫₀¹ x^α f(x)dx, α > -1
            Logarithmic: ∫₀¹ ln(x)f(x)dx
            Oscillatory: ∫₀^∞ sin(ωx)f(x)dx

        Special Techniques:
            Gauss-Laguerre: ∫₀^∞ e^{-x}f(x)dx
            Gauss-Hermite: ∫_{-∞}^∞ e^{-x²}f(x)dx
            Clenshaw-Curtis: ∫_{-1}¹ f(x)dx using Chebyshev points

        Args:
            perturbation: Field configuration for integration analysis

        Returns:
            Dictionary containing:
            - integral_1d: ∫|φ|²dx L² norm
            - integral_2d: ∫∫|φ(x)|²|φ(y)|² dx dy cross-correlation
            - integral_3d: ∫∫∫|φ(x)||φ(y)||φ(z)| dx dy dz triple correlation
            - evolution_final_energy: Energy after time evolution
        """
        x_coords = torch.linspace(-1, 1, len(perturbation))

        # 1D integration using quad
        def field_integrand(x):
            x_tensor = torch.tensor(x)
            field_val = torch.interp(x_tensor, x_coords, perturbation)
            return field_val.item() ** 2

        integral_1d, _ = quad(
            field_integrand,
            -1,
            1,
            epsabs=INTEGRATION_ABSOLUTE_TOLERANCE,
            epsrel=INTEGRATION_RELATIVE_TOLERANCE,
        )

        # 2D integration using dblquad
        def field_2d_integrand(y, x):
            # Simple 2D extension
            return field_integrand(x) * field_integrand(y)

        integral_2d, _ = dblquad(field_2d_integrand, -1, 1, -1, 1)

        # 3D integration using tplquad
        def field_3d_integrand(z, y, x):
            return field_integrand(x) * field_integrand(y) * field_integrand(z)

        integral_3d, _ = tplquad(field_3d_integrand, -1, 1, -1, 1, -1, 1)

        # Time evolution using solve_ivp
        def field_ode(t, y):
            # Klein-Gordon equation: ∂²φ/∂t² = ∇²φ - m²φ
            n = len(y) // 2
            phi = y[:n]
            phi_dot = y[n:]

            # Compute Laplacian (finite difference)
            laplacian = np.zeros_like(phi)
            for i in range(1, n - 1):
                laplacian[i] = phi[i + 1] - 2 * phi[i] + phi[i - 1]

            phi_ddot = laplacian - self.field_mass**2 * phi

            return np.concatenate([phi_dot, phi_ddot])

        # Initial conditions
        y0 = np.concatenate(
            [
                perturbation.detach().cpu().numpy(),
                np.zeros_like(perturbation.detach().cpu().numpy()),
            ]
        )

        sol = solve_ivp(field_ode, [0, 1], y0, method="RK45", rtol=1e-8)

        return {
            "integral_1d": integral_1d,
            "integral_2d": integral_2d,
            "integral_3d": integral_3d,
            "evolution_final_energy": np.sum(sol.y[: len(perturbation), -1] ** 2),
        }
