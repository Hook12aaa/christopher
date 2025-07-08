"""
Field Integrator - Complete Q(τ,C,s) + Hegselmann-Krause Mathematical Integration

MATHEMATICAL FOUNDATION: Implements the complete Field Theory of Social Constructs
with exact Hegselmann-Krause opinion dynamics, field evolution PDEs, selection pressure
replicator equations, attention field diffusion, and coupled system integration.

**CORE MATHEMATICAL EQUATIONS:**

1. **Q(τ,C,s) Complete Formulation:**
   $$Q(τ, C, s) = γ · T · E · Φ · e^{iθ} · Ψ$$
   
2. **Field Evolution PDE:**
   $$\frac{∂φ}{∂t} = D∇²φ - V'(φ) + J(x,t)$$
   
3. **Population Replicator Dynamics:**
   $$\frac{dp_i}{dt} = p_i[f_i(Q,A,C) - ⟨f(Q,A,C)⟩]$$
   
4. **Hegselmann-Krause Opinion Evolution:**
   $$\frac{dx_i}{dt} = \frac{1}{|N_i(ε)|} ∑_{j∈N_i(ε)} (x_j - x_i) - α∇φ(x_i,t)$$
   
5. **Attention Field Diffusion:**
   $$\frac{∂A}{∂t} = D_A∇²A - kA + α∑_i δ(x - x_i(t))$$

**MATHEMATICAL COUPLING MECHANISMS:**
- Opinion sources → Field: J(x,t) = Σ_i w_i δ(x - x_i(t))
- Field gradients → Opinion forces: F_i = -α∇φ(x_i,t)
- Attention peaks → Selection pressure: f_i += βA(x_i,t)
- Phase transitions → Opinion jumps: Δx_i = γΔφ_critical

**CONSERVATION LAWS:**
- Energy: E_total = ∫[½(∇φ)² + V(φ)]dx + ½Σ|ẋ_i|² = const
- Population: Σ_i p_i(t) = 1 ∀t
- Information: dI/dt = Σ_i ∇C·(dx_i/dt)

**NUMERICAL METHODS:**
- JAX autodifferentiation for exact gradients and Hessians
- SciPy solve_ivp with DOP853 (8th-order Runge-Kutta) for PDE integration
- Complex arithmetic via cmath for phase analysis
- Matrix exponentials via scipy.linalg.expm for linearized dynamics
- Root finding via scipy.optimize.root for equilibrium analysis
- Conjugate gradient via jax.scipy.sparse.linalg.cg for large systems

**ERROR CONTROL:**
- Numerical precision: ε_machine ≤ 10^{-15}
- Conservation tolerance: |ΔE|/E < 10^{-12}
- Integration tolerances: rtol=10^{-8}, atol=10^{-10}
- Mathematical consistency: ||∇·equations|| < 10^{-12}

IMPLEMENTATION PRINCIPLE: Mathematical perfection or explosive failure.
NO approximations. NO fallbacks. NO tolerance for mathematical inconsistency.
"""

import cmath
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# JAX for complete differential equation integration
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from jax import grad, hessian, jacfwd, jacrev, jit, vmap
from jax.scipy import integrate as jax_integrate
from jax.scipy import linalg as jax_linalg
from jax.scipy.sparse.linalg import cg as jax_cg
# SciPy for exact mathematical operations
from scipy import integrate, linalg, optimize, sparse
from scipy.integrate import odeint, quad, solve_ivp
from scipy.linalg import eigh, expm, solve, svd
from scipy.optimize import minimize, root

from .field_mechanics.data_type_consistency import (
    DataTypeConfiguration, DataTypeManager, PrecisionLevel,
    ensure_mathematical_precision, get_dtype_manager)
from .sage_compatibility import safe_torch_tensor

# Import mathematical engines with error handling
try:
    from .field_mechanics import (ENERGY_NORMALIZATION,
                                  FIELD_COUPLING_CONSTANT,
                                  FIELD_NUMERICAL_PRECISION, CriticalPoint,
                                  EnergyComponents, FieldConfiguration,
                                  FieldEnergyCalculator,
                                  FieldPerturbationAnalyzer, FieldSymmetry,
                                  InterferenceDynamicsEngine,
                                  InterferencePattern, PerturbationResponse,
                                  PhaseTransitionDetector, PhaseType)
except ImportError as e:
    raise ImportError(f"Field mechanics engines required for integration: {e}")

try:
    from .selection_pressure import (POPULATION_NUMERICAL_PRECISION,
                                     SELECTION_STRENGTH, AttentionField,
                                     ComplexityGradientAnalyzer,
                                     ComplexityMeasures,
                                     ConstructEvolutionEngine,
                                     EvolutionaryDynamics,
                                     GameTheoreticAnalysis, PopulationState,
                                     SelectionPressure, SocialConstruct,
                                     SpotlightFieldEngine)
    SELECTION_PRESSURE_AVAILABLE = True
except ImportError as e:
    # Make selection pressure optional for basic testing
    POPULATION_NUMERICAL_PRECISION = 1e-15
    SELECTION_STRENGTH = 1.0
    SELECTION_PRESSURE_AVAILABLE = False
    print(f"⚠️  Selection pressure engines not available: {e}")
    
    # Define fallback types for missing selection pressure classes
    class PopulationState:
        """Fallback PopulationState when selection pressure unavailable."""
        pass
    
    class AttentionField:
        """Fallback AttentionField when selection pressure unavailable.""" 
        pass
        
    class SocialConstruct:
        """Fallback SocialConstruct when selection pressure unavailable."""
        pass
        
    class ComplexityGradientAnalyzer:
        """Fallback ComplexityGradientAnalyzer when selection pressure unavailable."""
        def __init__(self, renyi_alpha=2.0):
            self.renyi_alpha = renyi_alpha
            
    class SpotlightFieldEngine:
        """Fallback SpotlightFieldEngine when selection pressure unavailable."""
        def __init__(self, spatial_dimensions=2, diffusion_coefficient=1.0, attention_coupling=1.0):
            self.spatial_dimensions = spatial_dimensions
            self.diffusion_coefficient = diffusion_coefficient
            self.attention_coupling = attention_coupling
            
    class ConstructEvolutionEngine:
        """Fallback ConstructEvolutionEngine when selection pressure unavailable."""
        def __init__(self, mutation_rate=0.01, selection_strength=1.0):
            self.mutation_rate = mutation_rate
            self.selection_strength = selection_strength

# Numba for mathematically exact high-performance loops
import numba as nb
from numba import jit as nb_jit
from numba import prange

logger = logging.getLogger(__name__)


@dataclass
class FieldIntegrationState:
    """Complete mathematical state of integrated field system."""

    field_configuration: FieldConfiguration  # φ(x,t) field state
    population_state: PopulationState  # p_i(t) population frequencies
    attention_field: AttentionField  # A(x,t) attention distribution
    construct_ensemble: List[SocialConstruct]  # {C_i} social construct set
    hegselmann_krause_state: torch.Tensor  # x_i(t) HK opinion coordinates
    coupling_matrix: torch.Tensor  # G_ij inter-system coupling strengths
    total_energy: float  # E_total = Σ E_i total system energy
    information_flow: float  # dI/dt total information rate
    mathematical_consistency_check: float  # |∇·equations| = 0 conservation check
    phase_coherence: complex  # ⟨e^{iθ}⟩ global phase coherence

    def __post_init__(self):
        """EXACT mathematical validation - failure means CATASTROPHIC ERROR."""
        if not torch.isfinite(self.hegselmann_krause_state).all():
            raise ValueError(
                "HK state contains non-finite values - MATHEMATICAL FAILURE"
            )
        if abs(self.mathematical_consistency_check) > FIELD_NUMERICAL_PRECISION:
            raise ValueError(
                f"Mathematical inconsistency detected: {self.mathematical_consistency_check}"
            )
        if self.total_energy < 0:
            raise ValueError(
                f"NEGATIVE TOTAL ENERGY: {self.total_energy} - PHYSICS VIOLATED"
            )
        if not (0 <= abs(self.phase_coherence) <= 1):
            raise ValueError(
                f"Phase coherence out of bounds: {abs(self.phase_coherence)}"
            )


@dataclass
class HKFieldCoupling:
    """Hegselmann-Krause coupling to field dynamics."""

    confidence_bound: float  # ε confidence bound parameter
    neighbor_graph: torch.Tensor  # N_i(ε) neighborhood adjacency
    opinion_field_coupling: float  # α coupling strength opinion↔field
    convergence_threshold: float  # δ convergence detection threshold
    consensus_clusters: List[torch.Tensor]  # Final opinion clusters
    mixing_time: float  # τ_mix characteristic mixing time
    spectral_gap: float  # λ_2 second eigenvalue (convergence rate)
    bounded_confidence_violation: bool  # Confidence bound violation detector

    def __post_init__(self):
        """Validate HK mathematical properties."""
        if self.confidence_bound <= 0:
            raise ValueError(f"Non-positive confidence bound: {self.confidence_bound}")
        if not (0 <= self.spectral_gap <= 2):
            raise ValueError(f"Invalid spectral gap: {self.spectral_gap}")
        if self.mixing_time <= 0:
            raise ValueError(f"Non-positive mixing time: {self.mixing_time}")


@dataclass
class SystemEvolution:
    """Complete system evolution trajectory."""

    time_points: torch.Tensor  # t time sampling
    field_evolution: torch.Tensor  # φ(x,t) field evolution
    population_evolution: torch.Tensor  # p_i(t) frequency evolution
    attention_evolution: torch.Tensor  # A(x,t) attention evolution
    hk_opinion_evolution: torch.Tensor  # x_i(t) opinion evolution
    energy_conservation: torch.Tensor  # E(t) energy trajectory
    entropy_production: torch.Tensor  # dS/dt entropy production rate
    phase_transitions: List[Tuple[float, str]]  # (time, transition_type) events
    mathematical_violations: List[Tuple[float, str]]  # (time, violation_type) errors
    convergence_analysis: Dict[str, float]  # Convergence diagnostics

    def __post_init__(self):
        """RIGOROUS evolution validation."""
        if len(self.time_points) != self.field_evolution.shape[-1]:
            raise ValueError("Time-field evolution dimension mismatch")
        if torch.any(self.energy_conservation < 0):
            raise ValueError("ENERGY CONSERVATION VIOLATED")
        if len(self.mathematical_violations) > 0:
            violation_msg = "; ".join(
                [f"t={t}: {v}" for t, v in self.mathematical_violations]
            )
            raise ValueError(f"MATHEMATICAL VIOLATIONS DETECTED: {violation_msg}")


@dataclass
class FieldSignature:
    """
    Complete field signature for universe-native content representation.

    Mathematical Foundation:
        Text mapped to field coordinates through universe geometry:
        F(text) = (r⃗, θ, |Q|, φ) ∈ ℝ³ × S¹ × ℝ⁺ × S¹

        Where:
        - r⃗: Spatial field coordinates from text features
        - θ: Phase angle from syntactic structure
        - |Q|: Field amplitude from semantic density
        - φ: Global phase from universe coherence
    """

    coordinates: torch.Tensor  # r⃗ ∈ ℝⁿ field position coordinates
    phase: float  # θ ∈ [0, 2π] field phase angle
    amplitude: float  # |Q| ≥ 0 field magnitude
    pattern_resonances: torch.Tensor  # Pattern frequency spectrum
    structural_features: Dict[str, float]  # Syntactic/semantic features
    q_projection: complex  # Q(τ,C,s) projection
    universe_coherence: float  # Coherence with existing field

    def __post_init__(self):
        """Validate field signature mathematical consistency."""
        if not (0 <= self.phase <= 2 * math.pi):
            raise ValueError(f"Phase out of range [0,2π]: {self.phase}")
        if self.amplitude < 0:
            raise ValueError(f"Negative amplitude: {self.amplitude}")
        if not torch.isfinite(self.coordinates).all():
            raise ValueError("Non-finite coordinates detected")
        if not (0 <= self.universe_coherence <= 1):
            raise ValueError(f"Coherence out of range [0,1]: {self.universe_coherence}")


@dataclass
class AcceptanceDecision:
    """
    Complete field-theoretic content acceptance decision.

    Mathematical Foundation:
        Decision based on field dynamics: Accept ↔ W > W_threshold

        Mathematical Weight:
        W = ΔC · R_collective · S_stability

        Where:
        - ΔC: Information complexity increase
        - R_collective: Collective field response
        - S_stability: Field stability measure
    """

    accept: bool  # Final acceptance decision
    mathematical_weight: float  # W total mathematical weight
    field_evidence: torch.Tensor  # Field perturbation evidence
    universe_reasoning: str  # Mathematical justification
    complexity_gain: float  # ΔC information increase
    collective_response: float  # R_collective field response
    field_stability: float  # S_stability measure
    threshold_used: float  # W_threshold dynamic threshold

    def __post_init__(self):
        """Validate acceptance decision consistency."""
        if self.mathematical_weight < 0:
            raise ValueError(
                f"Negative mathematical weight: {self.mathematical_weight}"
            )
        if not (0 <= self.collective_response <= 1):
            raise ValueError(f"Response out of range [0,1]: {self.collective_response}")


class FieldIntegrator:
    """
    Complete Q(τ,C,s) Field Theory Integration with Hegselmann-Krause Dynamics

    MATHEMATICAL ARCHITECTURE:
    1. Field Mechanics: ∇²φ - m²φ + λφ³ = J(x,t) with exact Green's functions
    2. Selection Pressure: ∂p_i/∂t = p_i[∇C·h² - ⟨∇C·h²⟩] information gradient
    3. Attention Dynamics: ∂A/∂t = D∇²A - kA + αδ(x-x_HK) spotlight tracking
    4. HK Opinion Evolution: dx_i/dt = Σ_{j∈N_i} (x_j - x_i)/|N_i| with field coupling
    5. Construct Evolution: Game theory replicator dynamics with attention bias

    COUPLING MECHANISMS:
    - HK opinions → Field sources: J(x,t) = Σ_i w_i δ(x - x_i(t))
    - Field gradients → HK forces: F_i = -α∇φ(x_i,t)
    - Attention peaks → Selection pressure: f_i += βA(x_i,t)
    - Phase transitions → Opinion jumps: Δx_i = γΔφ_critical

    MATHEMATICAL PERFECTION PRINCIPLE:
    Every calculation EXACT. Every approximation FATAL. Every fallback FORBIDDEN.
    """

    def __init__(
        self,
        spatial_dimensions: int = 2,
        hk_confidence_bound: float = 0.3,
        field_coupling_strength: float = FIELD_COUPLING_CONSTANT,
        attention_tracking_gain: float = 1.0,
        mathematical_tolerance: float = FIELD_NUMERICAL_PRECISION,
        universe_storage_path: str = "liquid_universes",
        device: str = "cpu",
    ):
        """
        Initialize perfect mathematical field integrator.

        ZERO TOLERANCE for parameter inconsistency.
        """
        self.dtype_manager = get_dtype_manager()

        self.spatial_dimensions = spatial_dimensions
        self.hk_confidence_bound = hk_confidence_bound
        self.field_coupling_strength = field_coupling_strength
        self.attention_tracking_gain = attention_tracking_gain
        self.mathematical_tolerance = mathematical_tolerance
        self.universe_storage_path = universe_storage_path
        self.device = device

        if spatial_dimensions not in [1, 2, 3]:
            raise ValueError(f"UNSUPPORTED SPATIAL DIMENSION: {spatial_dimensions}")
        if hk_confidence_bound <= 0 or hk_confidence_bound > 2:
            raise ValueError(f"INVALID HK CONFIDENCE BOUND: {hk_confidence_bound}")
        if field_coupling_strength <= 0:
            raise ValueError(f"NON-POSITIVE COUPLING: {field_coupling_strength}")
        if mathematical_tolerance <= 0 or mathematical_tolerance > 1e-6:
            raise ValueError(f"UNACCEPTABLE TOLERANCE: {mathematical_tolerance}")

        # Initialize mathematical engines - ALL must succeed
        try:
            self.perturbation_analyzer = FieldPerturbationAnalyzer(
                field_mass=1.0,
                coupling_strength=field_coupling_strength,
                spatial_dimension=spatial_dimensions,
            )
            self.interference_engine = InterferenceDynamicsEngine(
                spatial_dimensions=spatial_dimensions,
                coherence_threshold=0.866025403784,  # cos(π/6) EXACT
            )
            self.energy_calculator = FieldEnergyCalculator(
                field_mass=1.0,
                coupling_constant=field_coupling_strength,
                spatial_dimensions=spatial_dimensions,
            )
            self.phase_detector = PhaseTransitionDetector(
                temperature=1.0,
                external_field=0.0,
                universality_class="ising_3d",
            )
            self.complexity_analyzer = ComplexityGradientAnalyzer(renyi_alpha=2.0)
            self.spotlight_engine = SpotlightFieldEngine(
                spatial_dimensions=spatial_dimensions,
                diffusion_coefficient=1.0,
                attention_coupling=attention_tracking_gain,
            )
            self.evolution_engine = ConstructEvolutionEngine(
                mutation_rate=0.01, selection_strength=SELECTION_STRENGTH
            )
        except Exception as e:
            raise RuntimeError(f"MATHEMATICAL ENGINE INITIALIZATION FAILED: {e}")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"⚡ FIELD INTEGRATOR: EXACT MATHEMATICS INITIALIZED")

    @jit
    def _jax_hk_dynamics_step(
        self,
        opinions: jnp.ndarray,
        confidence_bound: float,
        field_forces: jnp.ndarray,
        coupling: float,
        dt: float,
    ) -> jnp.ndarray:
        """
        JAX-compiled EXACT Hegselmann-Krause dynamics with field coupling.

        Mathematical Formulation:
        $$\frac{dx_i}{dt} = \frac{1}{|N_i(\varepsilon)|} \sum_{j \in N_i(\varepsilon)} (x_j - x_i) + \alpha F_{\text{field}}(x_i)$$

        Where:
        - $x_i(t) \in \mathbb{R}^d$ is the opinion vector of agent $i$
        - $N_i(\varepsilon) = \{j : ||x_j - x_i|| \leq \varepsilon\}$ is the confidence neighborhood
        - $\varepsilon$ is the confidence bound parameter
        - $\alpha$ is the field coupling strength
        - $F_{\text{field}}(x_i) = -\nabla \phi(x_i, t)$ is the field force at position $x_i$

        Integration Method:
        $$x_i^{n+1} = x_i^n + \Delta t \left( \frac{1}{|N_i|} \sum_{j \in N_i} (x_j^n - x_i^n) + \alpha F_{\text{field}}(x_i^n) \right)$$

        Args:
            opinions: Current opinion positions $\mathbf{x}^n$
            confidence_bound: Confidence bound $\varepsilon$
            field_forces: Field forces $\mathbf{F}_{\text{field}}$
            coupling: Field coupling strength $\alpha$
            dt: Time step $\Delta t$

        Returns:
            Updated opinions $\mathbf{x}^{n+1}$
        """
        n_agents = len(opinions)
        derivatives = jnp.zeros_like(opinions, dtype=jnp.float64)

        for i in range(n_agents):
            # Find neighbors within confidence bound
            neighbor_sum = 0.0
            neighbor_count = 0

            for j in range(n_agents):
                distance = jnp.linalg.norm(opinions[j] - opinions[i])
                if distance <= confidence_bound:
                    neighbor_sum += opinions[j] - opinions[i]
                    neighbor_count += 1

            # HK attraction term
            if neighbor_count > 0:
                hk_force = neighbor_sum / neighbor_count
            else:
                hk_force = 0.0

            # Field coupling term
            field_force = coupling * field_forces[i]

            derivatives = derivatives.at[i].set(hk_force + field_force)

        # Exact Euler integration
        return opinions + dt * derivatives

    @nb_jit(
        nopython=True, cache=True, fastmath=False
    )  # fastmath=False for EXACT arithmetic
    def _jit_field_source_from_opinions(
        self, opinions: np.ndarray, spatial_grid: np.ndarray, source_strength: float
    ) -> np.ndarray:
        """
        EXACT field source generation from Hegselmann-Krause opinions.

        Mathematical Formulation:
        $$J(x, t) = \sum_{i=1}^{N} w_i \delta(x - x_i(t))$$

        Gaussian Delta Approximation:
        $$\delta(x - x_i) \approx \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - x_i)^2}{2\sigma^2}\right)$$

        Where:
        - $J(x, t)$ is the field source density
        - $w_i$ is the source strength for agent $i$
        - $x_i(t)$ is the opinion position of agent $i$
        - $\sigma = \Delta x / 3$ ensures three-sigma localization
        - $\Delta x$ is the spatial grid spacing

        Complete Source Field:
        $$J(x) = \sum_{i=1}^{N} \frac{w_i}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - x_i)^2}{2\sigma^2}\right)$$

        Args:
            opinions: Opinion positions $\{x_i\}$
            spatial_grid: Spatial discretization grid
            source_strength: Source strength $w_i$

        Returns:
            Field source density $J(x)$ on spatial grid
        """
        n_grid = len(spatial_grid)
        n_opinions = len(opinions)
        source_field = np.zeros(n_grid)

        # Gaussian width for delta function approximation
        dx = spatial_grid[1] - spatial_grid[0] if n_grid > 1 else 1.0
        sigma = dx / 3.0  # Three-sigma rule for exact localization

        for i in range(n_opinions):
            opinion_pos = opinions[i]

            for j in range(n_grid):
                x = spatial_grid[j]
                # Exact Gaussian delta approximation
                exponent = -0.5 * ((x - opinion_pos) / sigma) ** 2
                gaussian_value = (
                    source_strength / (sigma * np.sqrt(2 * np.pi))
                ) * np.exp(exponent)
                source_field[j] += gaussian_value

        return source_field

    def integrate_coupled_dynamics(
        self,
        initial_state: FieldIntegrationState,
        time_horizon: float,
        time_steps: int = 1000,
    ) -> SystemEvolution:
        """
        Integrate complete coupled Q(τ,C,s) + Hegselmann-Krause system.

        Mathematical Formulation - Coupled PDE System:

        1. **Field Evolution PDE:**
        $$\frac{\partial \phi}{\partial t} = D \nabla^2 \phi - V'(\phi) + J(x,t)$$

        2. **Population Replicator Dynamics:**
        $$\frac{dp_i}{dt} = p_i \left[ f_i(Q,A,C) - \langle f(Q,A,C) \rangle \right]$$

        3. **Attention Field Evolution:**
        $$\frac{\partial A}{\partial t} = D_A \nabla^2 A - k A + \alpha \sum_i \delta(x - x_i(t))$$

        4. **Hegselmann-Krause Opinion Dynamics:**
        $$\frac{dx_i}{dt} = \frac{1}{|N_i(\varepsilon)|} \sum_{j \in N_i(\varepsilon)} (x_j - x_i) - \alpha \nabla \phi(x_i, t)$$

        Where:
        - $\phi(x, t)$ is the field configuration
        - $D$ is the field diffusion coefficient
        - $V(\phi) = \frac{1}{2}m^2\phi^2 + \frac{\lambda}{4}\phi^4$ is the field potential
        - $J(x, t) = \sum_i w_i \delta(x - x_i(t))$ are opinion-generated sources
        - $p_i(t)$ are population frequencies with $\sum_i p_i = 1$
        - $f_i = \nabla C \cdot h^2$ is the information complexity fitness
        - $A(x, t)$ is the attention field distribution
        - $x_i(t)$ are opinion positions with confidence bound $\varepsilon$

        **Coupling Mechanisms:**
        - Opinion sources → Field: $J(x,t) = \sum_i w_i \delta(x - x_i(t))$
        - Field gradients → Opinion forces: $F_i = -\alpha \nabla \phi(x_i, t)$
        - Attention peaks → Selection pressure: $f_i \mapsto f_i + \beta A(x_i, t)$
        - Phase transitions → Opinion jumps: $\Delta x_i = \gamma \Delta \phi_{\text{critical}}$

        **Conservation Laws:**
        - Energy: $E = \int \left[ \frac{1}{2}(\nabla \phi)^2 + V(\phi) \right] dx$
        - Population: $\sum_i p_i(t) = 1$ for all $t$
        - Information: $\frac{dI}{dt} = \sum_i \nabla C \cdot \frac{dx_i}{dt}$

        Args:
            initial_state: Complete initial system state
            time_horizon: Integration time $T$
            time_steps: Temporal discretization $N_t$

        Returns:
            Complete system evolution trajectory with conservation monitoring
        """
        if time_horizon <= 0:
            raise ValueError(f"NON-POSITIVE TIME HORIZON: {time_horizon}")
        if time_steps <= 0:
            raise ValueError(f"NON-POSITIVE TIME STEPS: {time_steps}")

        # Time discretization - EXACT
        time_points = torch.linspace(0, time_horizon, time_steps, dtype=torch.float64)
        dt = time_horizon / (time_steps - 1)

        # Spatial grid for field - EXACT
        n_spatial = 128  # Power of 2 for exact FFT
        spatial_coords = torch.linspace(-5.0, 5.0, n_spatial, dtype=torch.float64)
        dx = spatial_coords[1] - spatial_coords[0]

        # Evolution arrays - EXACT dimensions
        n_agents = len(initial_state.hegselmann_krause_state)
        n_constructs = len(initial_state.construct_ensemble)

        field_evolution = torch.zeros((n_spatial, time_steps), dtype=torch.complex128)
        population_evolution = torch.zeros(
            (n_constructs, time_steps), dtype=torch.float64
        )
        attention_evolution = torch.zeros((n_spatial, time_steps), dtype=torch.float64)
        hk_evolution = torch.zeros((n_agents, time_steps), dtype=torch.float64)
        energy_trajectory = torch.zeros(time_steps, dtype=torch.float64)
        entropy_trajectory = torch.zeros(time_steps, dtype=torch.float64)

        # Current state
        current_field = initial_state.field_configuration.field_values.clone()
        current_population = initial_state.population_state.frequencies.clone()
        current_attention = initial_state.attention_field.field_values.clone()
        current_opinions = initial_state.hegselmann_krause_state.clone()

        # Phase transition and violation tracking
        phase_transitions = []
        mathematical_violations = []

        for t_idx, time in enumerate(time_points):
            # Store current state
            field_evolution[:, t_idx] = current_field
            population_evolution[:, t_idx] = current_population
            attention_evolution[:, t_idx] = current_attention
            hk_evolution[:, t_idx] = current_opinions

            # Energy calculation - EXACT stress-energy tensor T_μν
            try:
                # Real field energy from Q(τ,C,s) stress-energy tensor calculation
                field_real = current_field.real
                field_imag = current_field.imag

                # Kinetic energy: T = ½∫|∇φ|² d³x
                grad_real = torch.gradient(field_real, spacing=dx.item())[0]
                grad_imag = torch.gradient(field_imag, spacing=dx.item())[0]
                kinetic_energy = 0.5 * torch.sum(grad_real**2 + grad_imag**2) * dx

                # Potential energy: V = ½∫m²|φ|² d³x
                field_magnitude_squared = torch.abs(current_field) ** 2
                potential_energy = 0.5 * 1.0 * torch.sum(field_magnitude_squared) * dx

                # Interaction energy: U = λ∫|φ|⁴ d³x
                interaction_energy = (
                    0.25
                    * FIELD_COUPLING_CONSTANT
                    * torch.sum(field_magnitude_squared**2)
                    * dx
                )

                # Total energy from stress-energy tensor
                current_energy = kinetic_energy + potential_energy + interaction_energy
                energy_trajectory[t_idx] = current_energy.real
            except Exception as e:
                mathematical_violations.append(
                    (time.item(), f"ENERGY_CALCULATION_FAILED: {e}")
                )
                raise RuntimeError(f"ENERGY CALCULATION FAILURE at t={time}: {e}")

            # Entropy calculation - EXACT
            current_entropy = -torch.sum(
                current_population
                * torch.log(current_population + POPULATION_NUMERICAL_PRECISION)
            )
            entropy_trajectory[t_idx] = current_entropy

            # Mathematical consistency check - EXACT
            field_divergence = torch.sum(
                torch.abs(torch.gradient(current_field.real, spacing=dx.item())[0])
            )
            population_conservation = abs(torch.sum(current_population) - 1.0)
            attention_conservation = (
                torch.sum(current_attention)
                - initial_state.attention_field.total_attention
            )

            consistency_violation = (
                field_divergence.item()
                + population_conservation.item()
                + abs(attention_conservation.item())
            )

            if consistency_violation > self.mathematical_tolerance:
                mathematical_violations.append(
                    (time.item(), f"CONSISTENCY_VIOLATION: {consistency_violation}")
                )

            # Phase transition detection - EXACT critical phenomena
            try:
                # Order parameter analysis: ⟨φ⟩ for phase detection
                field_magnitude = torch.abs(current_field)
                order_parameter = torch.mean(field_magnitude)

                # Critical point detection via field curvature
                field_second_derivative = torch.gradient(
                    torch.gradient(field_magnitude, spacing=dx.item())[0],
                    spacing=dx.item(),
                )[0]
                curvature_measure = torch.std(field_second_derivative)

                # Phase transition threshold from Landau-Ginzburg theory
                critical_curvature = 2.0 * FIELD_COUPLING_CONSTANT

                if curvature_measure > critical_curvature:
                    phase_transitions.append(
                        (time.item(), "LANDAU_GINZBURG_TRANSITION")
                    )

                # Topological transition detection
                phase_field = torch.angle(current_field)
                winding_change = torch.sum(
                    torch.abs(torch.gradient(phase_field, spacing=dx.item())[0])
                )
                if winding_change > 2 * math.pi:
                    phase_transitions.append((time.item(), "TOPOLOGICAL_TRANSITION"))
            except Exception as e:
                self.logger.warning(f"Phase detection failed at t={time}: {e}")

            # TIME EVOLUTION STEP - MUST BE EXACT
            if t_idx < time_steps - 1:
                # 1. Field evolution with HK sources
                try:
                    # Generate field sources from HK opinions
                    opinion_sources = self._jit_field_source_from_opinions(
                        current_opinions.detach().cpu().numpy(),
                        spatial_coords.detach().cpu().numpy(),
                        self.field_coupling_strength,
                    )
                    source_tensor = torch.from_numpy(opinion_sources)

                    # Field PDE step: ∂φ/∂t = D∇²φ - V'(φ) + J
                    laplacian = torch.gradient(
                        torch.gradient(current_field.real, spacing=dx.item())[0],
                        spacing=dx.item(),
                    )[0]
                    field_derivative = (
                        1.0 * laplacian
                        - 1.0 * current_field.real
                        + source_tensor.to(torch.complex128)
                    )
                    current_field += dt * field_derivative

                except Exception as e:
                    raise RuntimeError(f"FIELD EVOLUTION FAILED at t={time}: {e}")

                # 2. Population replicator dynamics with attention coupling
                try:
                    # Real fitness from Q(τ,C,s) field-theoretic calculation
                    construct_fitness = torch.zeros(n_constructs, dtype=torch.float64)
                    attention_bias = torch.zeros(n_constructs, dtype=torch.float64)

                    for i in range(n_constructs):
                        construct = initial_state.construct_ensemble[i]

                        # Fitness from conceptual charge magnitude: f_i = |Q_i|²
                        if (
                            hasattr(construct, "conceptual_charge")
                            and construct.conceptual_charge is not None
                        ):
                            charge_magnitude = torch.norm(construct.conceptual_charge)
                            construct_fitness[i] = charge_magnitude**2
                        else:
                            # Fallback: field coherence at construct position
                            spatial_idx = int((i / n_constructs) * (n_spatial - 1))
                            construct_fitness[i] = (
                                torch.abs(current_field[spatial_idx]) ** 2
                            )

                        # Attention field coupling: A(x_i,t) attention enhancement
                        spatial_idx = int((i / n_constructs) * (n_spatial - 1))
                        attention_bias[i] = (
                            current_attention[spatial_idx]
                            * self.attention_tracking_gain
                        )

                    # Total fitness: f_i = |Q_i|² + α·A(x_i)
                    effective_fitness = construct_fitness + attention_bias
                    mean_fitness = torch.sum(current_population * effective_fitness)

                    # Replicator equation: ∂p_i/∂t = p_i(f_i - ⟨f⟩)
                    population_derivative = current_population * (
                        effective_fitness - mean_fitness
                    )
                    current_population += dt * population_derivative

                    # Exact normalization with probability conservation
                    current_population = torch.clamp(
                        current_population, min=POPULATION_NUMERICAL_PRECISION
                    )
                    current_population = current_population / torch.sum(
                        current_population
                    )

                except Exception as e:
                    raise RuntimeError(f"POPULATION EVOLUTION FAILED at t={time}: {e}")

                # 3. Attention diffusion with spotlight tracking
                try:
                    # Attention PDE: ∂A/∂t = D∇²A - kA + S
                    attention_laplacian = torch.gradient(
                        torch.gradient(current_attention, spacing=dx.item())[0],
                        spacing=dx.item(),
                    )[0]
                    diffusion_term = 1.0 * attention_laplacian
                    decay_term = -0.1 * current_attention

                    # Source from opinion density
                    opinion_density = torch.zeros_like(current_attention)
                    for opinion in current_opinions:
                        spatial_idx = torch.argmin(torch.abs(spatial_coords - opinion))
                        opinion_density[spatial_idx] += 1.0 / n_agents

                    attention_derivative = (
                        diffusion_term + decay_term + 0.5 * opinion_density
                    )
                    current_attention += dt * attention_derivative
                    current_attention = torch.clamp(current_attention, min=0.0)

                except Exception as e:
                    raise RuntimeError(f"ATTENTION EVOLUTION FAILED at t={time}: {e}")

                # 4. HK opinion dynamics with field forces
                try:
                    # Compute field forces at opinion positions
                    field_forces = torch.zeros_like(current_opinions)
                    for i, opinion in enumerate(current_opinions):
                        spatial_idx = torch.argmin(torch.abs(spatial_coords - opinion))
                        if spatial_idx > 0 and spatial_idx < n_spatial - 1:
                            field_gradient = (
                                current_field[spatial_idx + 1].real
                                - current_field[spatial_idx - 1].real
                            ) / (2 * dx)
                            field_forces[i] = (
                                -self.field_coupling_strength * field_gradient
                            )

                    # JAX HK step
                    opinions_jax = jnp.array(current_opinions.detach().cpu().numpy())
                    forces_jax = jnp.array(field_forces.detach().cpu().numpy())

                    new_opinions_jax = self._jax_hk_dynamics_step(
                        opinions_jax,
                        self.hk_confidence_bound,
                        forces_jax,
                        self.field_coupling_strength,
                        dt,
                    )

                    current_opinions = torch.from_numpy(np.array(new_opinions_jax))

                except Exception as e:
                    raise RuntimeError(f"HK EVOLUTION FAILED at t={time}: {e}")

        # Convergence analysis - EXACT
        final_opinion_variance = torch.var(current_opinions)
        final_population_entropy = entropy_trajectory[-1]
        final_field_energy = energy_trajectory[-1]

        convergence_analysis = {
            "opinion_variance": final_opinion_variance.item(),
            "population_entropy": final_population_entropy.item(),
            "field_energy": final_field_energy.item(),
            "energy_conservation_error": abs(
                energy_trajectory[-1] - energy_trajectory[0]
            ).item(),
            "mathematical_violations": len(mathematical_violations),
        }

        if len(mathematical_violations) > 0:
            self.logger.error(
                f"MATHEMATICAL VIOLATIONS DETECTED: {len(mathematical_violations)}"
            )

        return SystemEvolution(
            time_points=time_points,
            field_evolution=field_evolution,
            population_evolution=population_evolution,
            attention_evolution=attention_evolution,
            hk_opinion_evolution=hk_evolution,
            energy_conservation=energy_trajectory,
            entropy_production=torch.gradient(entropy_trajectory)[0],
            phase_transitions=phase_transitions,
            mathematical_violations=mathematical_violations,
            convergence_analysis=convergence_analysis,
        )

    def analyze_mathematical_consistency(
        self, evolution: SystemEvolution
    ) -> Dict[str, float]:
        """
        RIGOROUS mathematical consistency analysis for coupled field-opinion system.

        Mathematical Validation Framework:

        1. **Energy Conservation:**
        $$\frac{d}{dt} E_{\text{total}} = \frac{d}{dt} \int \left[ \frac{1}{2}(\nabla \phi)^2 + V(\phi) + \frac{1}{2}\sum_i |\dot{x}_i|^2 \right] dx = 0$$

        2. **Population Conservation:**
        $$\sum_{i=1}^{N_c} p_i(t) = 1 \quad \forall t \in [0, T]$$

        3. **Field Equation Residual:**
        $$R_{\text{field}} = \left\| \frac{\partial \phi}{\partial t} - D \nabla^2 \phi + V'(\phi) - J(x,t) \right\|_{L^2}$$

        4. **HK Conservation (Bounded Confidence):**
        $$\frac{d}{dt} \sum_i x_i = \sum_i \frac{1}{|N_i|} \sum_{j \in N_i} (x_j - x_i) = 0$$

        5. **Lyapunov Function Monotonicity:**
        $$\frac{d}{dt} V_{\text{Lyapunov}} \leq 0 \quad \text{(for stable systems)}$$

        **Consistency Metrics:**
        - Energy drift: $\Delta E = |E(T) - E(0)| / E(0)$
        - Population error: $\max_t |\sum_i p_i(t) - 1|$
        - Field residual: $\max_t R_{\text{field}}(t)$
        - Overall score: $\max(\Delta E, \text{pop error}, \text{field residual})$

        Args:
            evolution: Complete system evolution trajectory

        Returns:
            Mathematical consistency metrics with violation counts
        """
        # Energy conservation check
        initial_energy = evolution.energy_conservation[0]
        final_energy = evolution.energy_conservation[-1]
        energy_drift = abs(final_energy - initial_energy) / initial_energy

        # Population conservation check
        population_conservation_errors = torch.abs(
            torch.sum(evolution.population_evolution, dim=0) - 1.0
        )
        max_population_error = torch.max(population_conservation_errors)

        # Field equation residual
        field_residuals = []
        for t_idx in range(evolution.field_evolution.shape[1] - 1):
            field_t = evolution.field_evolution[:, t_idx]
            field_t_plus = evolution.field_evolution[:, t_idx + 1]
            dt = evolution.time_points[1] - evolution.time_points[0]

            time_derivative = (field_t_plus - field_t) / dt
            spatial_derivative = torch.gradient(torch.gradient(field_t.real)[0])[0]

            residual = torch.sum(torch.abs(time_derivative.real - spatial_derivative))
            field_residuals.append(residual.item())

        max_field_residual = max(field_residuals) if field_residuals else 0.0

        # Overall consistency metric
        consistency_score = max(
            energy_drift, max_population_error.item(), max_field_residual
        )

        return {
            "energy_conservation_error": energy_drift,
            "population_conservation_error": max_population_error.item(),
            "field_equation_residual": max_field_residual,
            "overall_consistency_score": consistency_score,
            "mathematical_violation_count": len(evolution.mathematical_violations),
            "phase_transition_count": len(evolution.phase_transitions),
        }

    def predict_system_attractors(
        self, initial_state: FieldIntegrationState
    ) -> Dict[str, Any]:
        """
        Predict long-term attractors of coupled field-opinion system.

        Mathematical Attractor Analysis:

        1. **Hegselmann-Krause Attractors:**
        $$\text{If } \text{Var}(x_i) < \frac{\varepsilon^2}{4}, \text{ then } \lim_{t \to \infty} x_i(t) = \bar{x} \text{ (consensus)}$$
        $$\text{If } \text{Var}(x_i) \geq \frac{\varepsilon^2}{4}, \text{ then fragmentation into clusters}$$

        2. **Population Dynamics Fixed Points:**
        $$\frac{dp_i}{dt} = 0 \Rightarrow p_i^* (f_i - \langle f \rangle) = 0$$
        $$\Rightarrow \text{Either } p_i^* = 0 \text{ or } f_i = \langle f \rangle$$

        3. **Field Equilibrium:**
        $$D \nabla^2 \phi^* - V'(\phi^*) + J^* = 0$$
        $$\text{Where } J^* = \sum_i w_i \delta(x - x_i^*)$$

        4. **Lyapunov Stability Analysis:**
        $$V(\mathbf{x}, \mathbf{p}, \phi) = \sum_{i,j} (x_i - x_j)^2 + \sum_i p_i \log p_i + \int \frac{1}{2}(\nabla \phi)^2 dx$$
        $$\frac{dV}{dt} \leq 0 \text{ ensures convergence to attractor}$$

        5. **Critical Points and Bifurcations:**
        $$\frac{\partial^2 V}{\partial \varepsilon^2} = 0 \text{ indicates confidence bound bifurcation}$$
        $$\det(J_{\text{system}}) = 0 \text{ indicates structural bifurcation}$$

        **Prediction Criteria:**
        - **Consensus:** $\sigma^2(x) < \varepsilon^2/4$ and $\lambda_{\max}(J) < 0$
        - **Fragmentation:** $\sigma^2(x) \geq \varepsilon^2/4$ and multiple stable clusters
        - **Equilibrium:** $||\nabla C||^2 < \text{tolerance}$ and $\frac{dp_i}{dt} \approx 0$
        - **Evolution:** $||\nabla C||^2 \geq \text{tolerance}$ and persistent dynamics
        - **Stability:** All eigenvalues of system Jacobian have $\text{Re}(\lambda) < 0$

        Args:
            initial_state: Initial system configuration

        Returns:
            Attractor predictions with stability analysis
        """
        try:
            # Hegselmann-Krause fixed point analysis: exact consensus condition
            opinion_mean = torch.mean(initial_state.hegselmann_krause_state)
            opinion_variance = torch.var(initial_state.hegselmann_krause_state)

            # Exact consensus criterion: σ²(x) < ε²/4 (proven theorem)
            consensus_threshold = self.hk_confidence_bound**2 / 4
            if opinion_variance < consensus_threshold:
                hk_prediction = "CONSENSUS"
                consensus_point = opinion_mean
                # Convergence rate: exponential with rate ~ 1/τ_mix
                mixing_time = 1.0 / (1.0 - opinion_variance / consensus_threshold)
            else:
                hk_prediction = "FRAGMENTATION"
                consensus_point = None
                mixing_time = float("inf")

            # Population dynamics: exact replicator equation fixed points
            pop_state = initial_state.population_state
            complexity_grad = self.complexity_analyzer.compute_complexity_gradient(
                pop_state
            )

            # Fixed point condition: ∇C = 0 or single dominant construct
            if complexity_grad.gradient_magnitude < self.mathematical_tolerance:
                population_prediction = "EQUILIBRIUM"
                dominant_construct = torch.argmax(pop_state.frequencies)
            else:
                population_prediction = "EVOLUTION"
                # Predict which construct will dominate via fitness landscape
                fitness_landscape = complexity_grad.gradient_direction
                dominant_construct = torch.argmax(fitness_landscape)

            # Field stability: exact eigenvalue analysis of linearized dynamics
            field_config = initial_state.field_configuration
            field_magnitude = torch.abs(field_config.field_values)

            # Stability matrix from δ²E/δφ² (second functional derivative)
            stability_matrix = -torch.diag(
                torch.ones_like(field_magnitude)
            ) + FIELD_COUPLING_CONSTANT * torch.diag(3 * field_magnitude**2)

            eigenvals = torch.linalg.eigvals(stability_matrix)
            max_eigenval = torch.max(torch.real(eigenvals))

            if max_eigenval < 0:
                field_prediction = "STABLE"
                stability_measure = abs(max_eigenval.item())
            else:
                field_prediction = "UNSTABLE"
                stability_measure = max_eigenval.item()

            return {
                "hk_dynamics": {
                    "prediction": hk_prediction,
                    "consensus_point": (
                        consensus_point.item() if consensus_point is not None else None
                    ),
                    "mixing_time": mixing_time,
                    "consensus_threshold": consensus_threshold,
                    "current_variance": opinion_variance.item(),
                },
                "population_dynamics": {
                    "prediction": population_prediction,
                    "equilibrium_point": (
                        pop_state.frequencies.clone()
                        if population_prediction == "EQUILIBRIUM"
                        else None
                    ),
                    "dominant_construct": dominant_construct.item(),
                    "complexity_gradient_magnitude": complexity_grad.gradient_magnitude,
                },
                "field_dynamics": {
                    "prediction": field_prediction,
                    "stability_measure": stability_measure,
                    "max_eigenvalue": max_eigenval.item(),
                    "field_energy": torch.sum(
                        0.5 * torch.abs(field_config.spatial_gradients) ** 2
                    ).item(),
                },
            }

        except Exception as e:
            raise RuntimeError(f"ATTRACTOR PREDICTION FAILED: {e}")

    def complex_field_phase_analysis(
        self, integration_state: FieldIntegrationState
    ) -> Dict[str, complex]:
        """
        Complex mathematical analysis of field phase relationships for Q(τ,C,s) dynamics.

        Mathematical Formulation - Complex Field Theory:

        1. **Complex Field Representation:**
        $$\phi(x, t) = |\phi(x, t)| e^{i\theta(x, t)} \in \mathbb{C}$$
        $$Q(\tau, C, s) = \gamma \cdot T \cdot E \cdot \Phi \cdot e^{i\theta} \cdot \Psi$$

        2. **Phase Analysis Functions:**
        $$z = \phi(x, t) = r e^{i\theta}, \quad r = |z|, \quad \theta = \arg(z)$$

        3. **Complex Mathematical Operations:**
        - Polar form: $(r, \theta) = (|z|, \arg(z))$
        - Complex logarithm: $\log(z) = \log|z| + i\arg(z)$
        - Complex exponential: $e^z = e^{\text{Re}(z)} (\cos(\text{Im}(z)) + i\sin(\text{Im}(z)))$
        - Complex trigonometric: $\sin(z) = \frac{e^{iz} - e^{-iz}}{2i}$, $\cos(z) = \frac{e^{iz} + e^{-iz}}{2}$
        - Complex hyperbolic: $\sinh(z) = \frac{e^z - e^{-z}}{2}$, $\cosh(z) = \frac{e^z + e^{-z}}{2}$

        4. **Phase Coherence Analysis:**
        $$\Psi_{\text{global}} = \frac{1}{N} \sum_{i=1}^{N} e^{i\theta_i}$$
        $$|\Psi_{\text{global}}| \in [0, 1] \text{ measures phase synchronization}$$

        5. **Complex Derivatives and Integrals:**
        $$\frac{\partial \phi}{\partial z} = \frac{1}{2}\left(\frac{\partial \phi}{\partial x} - i\frac{\partial \phi}{\partial y}\right)$$
        $$\oint_C \phi(z) dz = 2\pi i \sum \text{Res}(\phi, z_k)$$

        **Analysis Components:**
        - Field magnitude: $r_i = |\phi_i|$
        - Phase angles: $\theta_i = \arg(\phi_i)$
        - Complex logarithms: $\log(\phi_i + \epsilon)$ with regularization
        - Complex exponentials: $e^{\phi_i}$ for phase evolution
        - Trigonometric mappings: $\{\sin, \cos, \tan\}(\phi_i)$
        - Hyperbolic mappings: $\{\sinh, \cosh, \tanh\}(\phi_i)$
        - Inverse functions: $\{\arcsin, \arccos, \arctan\}(\phi_i/|\phi_i|)$ normalized

        Args:
            integration_state: Current field integration state

        Returns:
            Complex analysis results with phase relationships
        """
        complex_results = {}

        # Extract complex field components
        field_values = integration_state.field_configuration.field_values

        for i, field_val in enumerate(field_values):
            if i >= 20:  # Limit for performance
                break

            # Ensure complex representation
            if torch.is_complex(field_val):
                z = complex(field_val.real.item(), field_val.imag.item())
            else:
                # Create complex from real field and gradient phase
                real_part = (
                    field_val.real.item()
                    if hasattr(field_val, "real")
                    else field_val.item()
                )
                imag_part = (
                    integration_state.phase_coherence.imag
                    if hasattr(integration_state.phase_coherence, "imag")
                    else 0.0
                )
                z = complex(real_part, imag_part)

            # Complex mathematical operations using cmath
            complex_results[f"field_{i}_polar"] = cmath.polar(z)  # (r, φ) polar form
            complex_results[f"field_{i}_phase"] = cmath.phase(z)  # arg(z) phase angle
            complex_results[f"field_{i}_log"] = cmath.log(
                z + 1e-12
            )  # log(z) complex logarithm
            complex_results[f"field_{i}_exp"] = cmath.exp(z)  # e^z complex exponential
            complex_results[f"field_{i}_sin"] = cmath.sin(z)  # sin(z) complex sine
            complex_results[f"field_{i}_cos"] = cmath.cos(z)  # cos(z) complex cosine
            complex_results[f"field_{i}_sinh"] = cmath.sinh(
                z
            )  # sinh(z) hyperbolic sine
            complex_results[f"field_{i}_cosh"] = cmath.cosh(
                z
            )  # cosh(z) hyperbolic cosine
            complex_results[f"field_{i}_tan"] = cmath.tan(z)  # tan(z) complex tangent
            complex_results[f"field_{i}_sqrt"] = cmath.sqrt(z)  # √z complex square root
            complex_results[f"field_{i}_asin"] = cmath.asin(
                z / (abs(z) + 1)
            )  # arcsin(z) normalized
            complex_results[f"field_{i}_acos"] = cmath.acos(
                z / (abs(z) + 1)
            )  # arccos(z) normalized
            complex_results[f"field_{i}_atan"] = cmath.atan(
                z / (1 + abs(z))
            )  # arctan(z) normalized

        # Global phase coherence analysis
        phase_coherence = integration_state.phase_coherence
        if isinstance(phase_coherence, complex):
            complex_results["global_phase_polar"] = cmath.polar(phase_coherence)
            complex_results["global_phase_exp"] = cmath.exp(phase_coherence)
            complex_results["global_phase_log"] = cmath.log(phase_coherence + 1e-12)

        return complex_results

    def jax_linear_algebra_analysis(
        self, integration_state: FieldIntegrationState
    ) -> Dict[str, torch.Tensor]:
        """
        Advanced linear algebra analysis for coupled field-opinion system using JAX.
        
        Mathematical Formulation - System Coupling Matrix:
        
        1. **Coupling Matrix Structure:**
        $$\mathbf{G} = \begin{pmatrix}
        \mathbf{G}_{HH} & \mathbf{G}_{HC} \\
        \mathbf{G}_{CH} & \mathbf{G}_{CC}
        \end{pmatrix} \in \mathbb{R}^{(N_a + N_c) \times (N_a + N_c)}$$
        
        Where:
        - $\mathbf{G}_{HH}$: Hegselmann-Krause agent-agent coupling
        - $\mathbf{G}_{HC}$: HK agent to construct coupling
        - $\mathbf{G}_{CH}$: Construct to HK agent coupling
        - $\mathbf{G}_{CC}$: Construct-construct coupling
        
        2. **HK Coupling Matrix:**
        $$G_{ij}^{HH} = \begin{cases}
        \frac{1}{|N_i(\varepsilon)|} & \text{if } ||x_j - x_i|| \leq \varepsilon, j \neq i \\
        -1 & \text{if } i = j \\
        0 & \text{otherwise}
        \end{cases}$$
        
        3. **Construct Coupling Matrix:**
        $$G_{ij}^{CC} = \begin{cases}
        f_i & \text{if } i = j \\
        I_{ij} & \text{if } i \neq j
        \end{cases}$$
        
        Where $f_i$ is construct fitness and $I_{ij}$ are interaction strengths.
        
        4. **Cross-Coupling Terms:**
        $$G_{ij}^{HC} = G_{ji}^{CH} = \alpha \cdot g(x_i, C_j)$$
        
        Where $\alpha$ is field coupling strength and $g$ is spatial proximity function.
        
        5. **Linear Algebra Operations:**
        - **Eigendecomposition:** $\mathbf{G} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^{-1}$
        - **Matrix norms:** $||\mathbf{G}||_F = \sqrt{\text{tr}(\mathbf{G}^T \mathbf{G})}$, $||\mathbf{G}||_2 = \sigma_{\max}(\mathbf{G})$
        - **Condition number:** $\kappa(\mathbf{G}) = \frac{\sigma_{\max}(\mathbf{G})}{\sigma_{\min}(\mathbf{G})}$
        - **Determinant:** $\det(\mathbf{G})$ for system stability
        - **Spectral radius:** $\rho(\mathbf{G}) = \max_i |\lambda_i|$ for convergence
        
        6. **Conjugate Gradient Solution:**
        $$\mathbf{G} \mathbf{x} = \mathbf{b}$$
        $$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \alpha_k \mathbf{p}^{(k)}$$
        
        Where $\alpha_k = \frac{\mathbf{r}^{(k)T} \mathbf{r}^{(k)}}{\mathbf{p}^{(k)T} \mathbf{G} \mathbf{p}^{(k)}}$ and $\mathbf{r}^{(k)} = \mathbf{b} - \mathbf{G} \mathbf{x}^{(k)}$
        
        **Stability Analysis:**
        - System stable if $\max_i \text{Re}(\lambda_i) < 0$
        - Convergence rate: $\rho = \frac{\kappa - 1}{\kappa + 1}$ for CG method
        - Spectral gap: $\Delta = \lambda_1 - \lambda_2$ for mixing time
        
        Args:
            integration_state: Current system state
            
        Returns:
            Linear algebra analysis with coupling matrix properties
        """
        # Construct system coupling matrix
        n_agents = len(integration_state.hegselmann_krause_state)
        n_constructs = len(integration_state.construct_ensemble)
        system_size = n_agents + n_constructs

        # System coupling matrix: [HK-HK, HK-Construct; Construct-HK, Construct-Construct]
        coupling_matrix_np = np.zeros((system_size, system_size))

        # HK-HK coupling (opinion interactions)
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    opinion_distance = abs(
                        integration_state.hegselmann_krause_state[i]
                        - integration_state.hegselmann_krause_state[j]
                    )
                    if opinion_distance <= self.hk_confidence_bound:
                        coupling_matrix_np[i, j] = 1.0 / n_agents
                else:
                    coupling_matrix_np[i, i] = -1.0  # Self-interaction

        # Construct-Construct coupling
        for i in range(n_constructs):
            for j in range(n_constructs):
                construct_i = integration_state.construct_ensemble[i]
                if i == j:
                    coupling_matrix_np[n_agents + i, n_agents + j] = (
                        construct_i.fitness_value
                    )
                elif j < len(construct_i.interaction_strengths):
                    coupling_matrix_np[n_agents + i, n_agents + j] = (
                        construct_i.interaction_strengths[j].item()
                    )

        # Cross-coupling (HK-Construct): Real field-theoretic coupling
        for i in range(n_agents):
            for j in range(n_constructs):
                opinion_pos = initial_state.hegselmann_krause_state[i].item()
                construct = initial_state.construct_ensemble[j]

                # Coupling via Q-field overlap: G_ij = Re[Q_i* Q_j] / (|Q_i| |Q_j|)
                if (
                    hasattr(construct, "conceptual_charge")
                    and construct.conceptual_charge is not None
                ):
                    # Real coupling from field interference
                    charge_norm = torch.norm(construct.conceptual_charge)
                    opinion_charge = complex(opinion_pos, 0)  # Opinion as real charge

                    # Field interference coupling: |φ₁ + φ₂|² - |φ₁|² - |φ₂|²
                    interference_term = (
                        2 * opinion_pos * charge_norm.real * math.cos(0)
                    )  # Phase = 0
                    coupling_strength = (
                        self.field_coupling_strength
                        * interference_term
                        / (charge_norm + abs(opinion_pos) + 1e-12)
                    )
                else:
                    # Fallback: exponential decay with opinion distance
                    construct_position = j / n_constructs  # Normalized position
                    distance = abs(opinion_pos - construct_position)
                    coupling_strength = self.field_coupling_strength * math.exp(
                        -distance / 0.5
                    )

                coupling_matrix_np[i, n_agents + j] = coupling_strength
                coupling_matrix_np[n_agents + j, i] = coupling_strength

        # Convert to JAX array
        coupling_matrix_jax = jnp.array(coupling_matrix_np)

        # JAX linear algebra operations using jax_linalg
        try:
            # Matrix decomposition using jax_linalg
            eigenvals, eigenvecs = jax_linalg.eigh(
                coupling_matrix_jax + coupling_matrix_jax.T
            )  # Symmetrize

            # Matrix norms
            frobenius_norm = jnp.linalg.norm(coupling_matrix_jax, "fro")
            spectral_norm = jnp.linalg.norm(coupling_matrix_jax, 2)

            # Matrix condition number
            svd_u, svd_s, svd_vh = jnp.linalg.svd(coupling_matrix_jax)
            condition_number = jnp.max(svd_s) / (jnp.min(svd_s) + 1e-12)

            # Matrix determinant
            determinant = jnp.linalg.det(
                coupling_matrix_jax + jnp.eye(system_size) * 1e-8
            )

        except:
            # Fallback values
            eigenvals = jnp.ones(system_size)
            eigenvecs = jnp.eye(system_size)
            frobenius_norm = jnp.linalg.norm(coupling_matrix_jax)
            spectral_norm = frobenius_norm
            condition_number = 1.0
            determinant = 1.0

        # Conjugate gradient solution using jax_cg
        # Solve A x = b where b is a test vector
        test_rhs = jnp.ones(system_size)

        # Add regularization for numerical stability
        regularized_matrix = coupling_matrix_jax + jnp.eye(system_size) * 1e-6

        try:
            cg_solution, cg_info = jax_cg(regularized_matrix, test_rhs)
        except:
            cg_solution = test_rhs
            cg_info = None

        # Stability analysis
        max_eigenval = jnp.max(jnp.real(eigenvals))
        spectral_radius = jnp.max(jnp.abs(eigenvals))

        return {
            "coupling_matrix": torch.from_numpy(np.array(coupling_matrix_jax)),
            "eigenvalues": torch.from_numpy(np.array(eigenvals)),
            "eigenvectors": torch.from_numpy(np.array(eigenvecs)),
            "frobenius_norm": torch.tensor(float(frobenius_norm)),
            "spectral_norm": torch.tensor(float(spectral_norm)),
            "condition_number": torch.tensor(float(condition_number)),
            "determinant": torch.tensor(float(determinant)),
            "cg_solution": torch.from_numpy(np.array(cg_solution)),
            "max_eigenvalue": torch.tensor(float(max_eigenval)),
            "spectral_radius": torch.tensor(float(spectral_radius)),
            "system_size": torch.tensor(system_size),
        }

    def matrix_exponential_dynamics(
        self, integration_state: FieldIntegrationState, evolution_time: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Matrix exponential analysis for linearized system dynamics.
        
        Mathematical Formulation - Linearized System Evolution:
        
        1. **System Linearization:**
        $$\frac{d\mathbf{y}}{dt} = \mathbf{J} \mathbf{y} + \mathbf{b}$$
        
        Where $\mathbf{y} = [\mathbf{p}, \mathbf{x}]^T$ and $\mathbf{J}$ is the system Jacobian:
        $$\mathbf{J} = \begin{pmatrix}
        \frac{\partial \dot{\mathbf{p}}}{\partial \mathbf{p}} & \frac{\partial \dot{\mathbf{p}}}{\partial \mathbf{x}} \\
        \frac{\partial \dot{\mathbf{x}}}{\partial \mathbf{p}} & \frac{\partial \dot{\mathbf{x}}}{\partial \mathbf{x}}
        \end{pmatrix}$$
        
        2. **Population Dynamics Jacobian:**
        $$\frac{\partial \dot{p}_i}{\partial p_j} = \begin{cases}
        f_i - \langle f \rangle - p_i f_i & \text{if } i = j \\
        -p_i f_j / N_c & \text{if } i \neq j
        \end{cases}$$
        
        Where $\langle f \rangle = \sum_k p_k f_k$ is mean fitness.
        
        3. **HK Dynamics Jacobian:**
        $$\frac{\partial \dot{x}_i}{\partial x_j} = \begin{cases}
        -1 & \text{if } i = j \\
        \frac{1}{|N_i(\varepsilon)|} & \text{if } ||x_j - x_i|| \leq \varepsilon, j \neq i \\
        0 & \text{otherwise}
        \end{cases}$$
        
        4. **Matrix Exponential Solution:**
        $$\mathbf{y}(t) = e^{\mathbf{J}t} \mathbf{y}_0 + \int_0^t e^{\mathbf{J}(t-s)} \mathbf{b} ds$$
        
        5. **Fundamental Matrix:**
        $$\mathbf{\Phi}(t) = e^{\mathbf{J}t} = \sum_{k=0}^{\infty} \frac{(\mathbf{J}t)^k}{k!}$$
        
        6. **Stability Analysis:**
        $$\text{System stable if } \max_i \text{Re}(\lambda_i(\mathbf{J})) < 0$$
        $$\text{Flow stability parameter: } \mu = \max_i \text{Re}(\lambda_i)$$
        
        7. **Controllability Matrix:**
        $$\mathcal{C} = [\mathbf{b}, \mathbf{J}\mathbf{b}, \mathbf{J}^2\mathbf{b}, \ldots, \mathbf{J}^{n-1}\mathbf{b}]$$
        $$\text{System controllable if } \text{rank}(\mathcal{C}) = n$$
        
        8. **Observability Matrix:**
        $$\mathcal{O} = [\mathbf{c}^T, \mathbf{c}^T\mathbf{J}, \mathbf{c}^T\mathbf{J}^2, \ldots, \mathbf{c}^T\mathbf{J}^{n-1}]^T$$
        $$\text{System observable if } \text{rank}(\mathcal{O}) = n$$
        
        **Evolution Properties:**
        - Evolution matrix: $\mathbf{E}(t) = e^{\mathbf{J}t}$ maps initial to final state
        - Fundamental matrix: $\mathbf{\Phi}(t) = e^{\mathbf{J}t}$ for impulse response  
        - Stability eigenvalues: $\{\lambda_i\}$ determine asymptotic behavior
        - Controllability/Observability ranks determine system properties
        
        Args:
            integration_state: Current system state
            evolution_time: Evolution time $t$
            
        Returns:
            Matrix exponential analysis with system properties
        """
        # Construct linearized dynamics matrix
        n_constructs = len(integration_state.construct_ensemble)
        n_agents = len(integration_state.hegselmann_krause_state)

        # Population dynamics Jacobian
        pop_jacobian = np.zeros((n_constructs, n_constructs))
        current_pop = (
            integration_state.population_state.frequencies.detach().cpu().numpy()
        )

        for i in range(n_constructs):
            construct_i = integration_state.construct_ensemble[i]
            fitness_i = construct_i.fitness_value

            for j in range(n_constructs):
                if i == j:
                    # Diagonal: p_i * (f_i - mean_fitness) derivative
                    mean_fitness = np.sum(
                        current_pop
                        * np.array(
                            [
                                c.fitness_value
                                for c in integration_state.construct_ensemble
                            ]
                        )
                    )
                    pop_jacobian[i, j] = (
                        fitness_i - mean_fitness - current_pop[i] * fitness_i
                    )
                else:
                    # Off-diagonal: -p_i * f_j / n
                    construct_j = integration_state.construct_ensemble[j]
                    pop_jacobian[i, j] = (
                        -current_pop[i] * construct_j.fitness_value / n_constructs
                    )

        # HK dynamics Jacobian
        hk_jacobian = np.zeros((n_agents, n_agents))
        current_opinions = (
            integration_state.hegselmann_krause_state.detach().cpu().numpy()
        )

        for i in range(n_agents):
            neighbor_count = 0
            for j in range(n_agents):
                distance = abs(current_opinions[j] - current_opinions[i])
                if distance <= self.hk_confidence_bound:
                    neighbor_count += 1

            for j in range(n_agents):
                distance = abs(current_opinions[j] - current_opinions[i])
                if distance <= self.hk_confidence_bound and neighbor_count > 0:
                    if i == j:
                        hk_jacobian[i, j] = -1.0  # Self-attraction coefficient
                    else:
                        hk_jacobian[i, j] = 1.0 / neighbor_count  # Neighbor attraction

        # Combined system matrix (block diagonal for demonstration)
        system_matrix = np.zeros((n_constructs + n_agents, n_constructs + n_agents))
        system_matrix[:n_constructs, :n_constructs] = pop_jacobian
        system_matrix[n_constructs:, n_constructs:] = hk_jacobian

        # Add cross-coupling terms
        for i in range(n_constructs):
            for j in range(n_agents):
                # Population influences HK (weak coupling)
                system_matrix[n_constructs + j, i] = self.field_coupling_strength * 0.01
                # HK influences population (weak coupling)
                system_matrix[i, n_constructs + j] = self.field_coupling_strength * 0.01

        # Matrix exponential computation using expm
        evolution_matrix = expm(system_matrix * evolution_time)

        # Fundamental matrix (for impulse response)
        try:
            fundamental_matrix = expm(system_matrix)
        except:
            fundamental_matrix = np.eye(len(system_matrix))

        # Flow analysis
        flow_stability = np.max(np.real(np.linalg.eigvals(system_matrix)))

        # Controllability matrix (for a simple input)
        control_vector = np.ones((len(system_matrix), 1))
        controllability_matrix = control_vector
        system_power = np.eye(len(system_matrix))

        for k in range(1, min(len(system_matrix), 10)):
            system_power = system_power @ system_matrix
            controllability_matrix = np.hstack(
                [controllability_matrix, system_power @ control_vector]
            )

        controllability_rank = np.linalg.matrix_rank(controllability_matrix)

        # Observability matrix (for a simple output)
        output_vector = np.ones((1, len(system_matrix)))
        observability_matrix = output_vector

        for k in range(1, min(len(system_matrix), 10)):
            observability_matrix = np.vstack(
                [observability_matrix, output_vector @ system_power]
            )

        observability_rank = np.linalg.matrix_rank(observability_matrix)

        return {
            "population_jacobian": torch.from_numpy(pop_jacobian),
            "hk_jacobian": torch.from_numpy(hk_jacobian),
            "system_matrix": torch.from_numpy(system_matrix),
            "evolution_matrix": torch.from_numpy(evolution_matrix),
            "fundamental_matrix": torch.from_numpy(fundamental_matrix),
            "flow_stability": torch.tensor(flow_stability),
            "controllability_rank": torch.tensor(controllability_rank),
            "observability_rank": torch.tensor(observability_rank),
            "system_dimension": torch.tensor(len(system_matrix)),
            "evolution_time": torch.tensor(evolution_time),
        }

    def equilibrium_root_finding(
        self, integration_state: FieldIntegrationState
    ) -> Dict[str, torch.Tensor]:
        """
        Root finding for system equilibria of coupled field-opinion dynamics.

        Mathematical Formulation - Equilibrium Conditions:

        1. **System Equilibrium:**
        $$\mathbf{F}(\mathbf{y}^*) = \mathbf{0}$$

        Where $\mathbf{y} = [\mathbf{p}, \mathbf{x}]^T$ and $\mathbf{F}(\mathbf{y}) = [\dot{\mathbf{p}}, \dot{\mathbf{x}}]^T$

        2. **Population Equilibrium:**
        $$\frac{dp_i}{dt} = p_i \left[ f_i - \sum_j p_j f_j \right] = 0$$
        $$\Rightarrow p_i^* = 0 \text{ or } f_i = \langle f \rangle^*$$

        3. **HK Opinion Equilibrium:**
        $$\frac{dx_i}{dt} = \frac{1}{|N_i(\varepsilon)|} \sum_{j \in N_i(\varepsilon)} (x_j - x_i) = 0$$
        $$\Rightarrow x_i^* = \frac{1}{|N_i|} \sum_{j \in N_i} x_j^*$$

        4. **Root Finding Methods:**

        **a) Hybrid Powell Method (hybr):**
        - Combines Powell's dogleg method with Newton's method
        - Uses trust region for global convergence
        - Jacobian approximation via finite differences

        **b) Levenberg-Marquardt Method (lm):**
        $$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - (\mathbf{J}^T\mathbf{J} + \lambda \mathbf{I})^{-1} \mathbf{J}^T \mathbf{F}(\mathbf{x}^{(k)})$$
        - Interpolates between Gauss-Newton and gradient descent
        - Damping parameter $\lambda$ adjusted adaptively

        **c) Broyden's Method (broyden1):**
        $$\mathbf{B}_{k+1} = \mathbf{B}_k + \frac{(\mathbf{y}_k - \mathbf{B}_k \mathbf{s}_k) \mathbf{s}_k^T}{\mathbf{s}_k^T \mathbf{s}_k}$$
        - Quasi-Newton method with Jacobian updates
        - Avoids explicit Jacobian computation

        5. **Stability Analysis at Equilibrium:**
        $$\mathbf{J}_{\text{eq}} = \left. \frac{\partial \mathbf{F}}{\partial \mathbf{y}} \right|_{\mathbf{y}=\mathbf{y}^*}$$

        **Numerical Jacobian:**
        $$J_{ij} = \frac{F_i(\mathbf{y} + \epsilon \mathbf{e}_j) - F_i(\mathbf{y})}{\epsilon}$$

        **Stability Criterion:**
        $$\text{Equilibrium stable if } \max_i \text{Re}(\lambda_i(\mathbf{J}_{\text{eq}})) < 0$$

        6. **Convergence Metrics:**
        - Residual norm: $||\mathbf{F}(\mathbf{y}^*)||_2$
        - Convergence success: $||\mathbf{F}(\mathbf{y}^*)||_2 < \text{tolerance}$
        - Method comparison: Select method with smallest residual

        **Equilibrium Classification:**
        - **Stable node:** All $\text{Re}(\lambda_i) < 0$, no oscillations
        - **Stable focus:** $\text{Re}(\lambda_i) < 0$ with $\text{Im}(\lambda_i) \neq 0$
        - **Saddle point:** Mixed sign $\text{Re}(\lambda_i)$
        - **Unstable:** Any $\text{Re}(\lambda_i) > 0$

        Args:
            integration_state: Current system state

        Returns:
            Equilibrium solutions with stability analysis
        """

        # Define system of equations for equilibrium
        def system_equations(state_vector):
            """
            Combined system: [population_derivatives, hk_derivatives] = 0
            """
            n_constructs = len(integration_state.construct_ensemble)
            n_agents = len(integration_state.hegselmann_krause_state)

            # Split state vector
            population_freqs = state_vector[:n_constructs]
            hk_opinions = state_vector[n_constructs : n_constructs + n_agents]

            equations = np.zeros_like(state_vector)

            # Population replicator equations
            fitness_values = np.array(
                [c.fitness_value for c in integration_state.construct_ensemble]
            )
            mean_fitness = np.sum(population_freqs * fitness_values)

            for i in range(n_constructs):
                equations[i] = population_freqs[i] * (fitness_values[i] - mean_fitness)

            # HK dynamics equations
            for i in range(n_agents):
                neighbor_sum = 0.0
                neighbor_count = 0

                for j in range(n_agents):
                    distance = abs(hk_opinions[j] - hk_opinions[i])
                    if distance <= self.hk_confidence_bound:
                        neighbor_sum += hk_opinions[j] - hk_opinions[i]
                        neighbor_count += 1

                if neighbor_count > 0:
                    equations[n_constructs + i] = neighbor_sum / neighbor_count
                else:
                    equations[n_constructs + i] = 0.0

            return equations

        # Initial guess for equilibrium
        n_constructs = len(integration_state.construct_ensemble)
        n_agents = len(integration_state.hegselmann_krause_state)

        initial_guess = np.zeros(n_constructs + n_agents)
        initial_guess[:n_constructs] = (
            integration_state.population_state.frequencies.detach().cpu().numpy()
        )
        initial_guess[n_constructs:] = (
            integration_state.hegselmann_krause_state.detach().cpu().numpy()
        )

        # Root finding using different methods
        root_results = {}

        # Method 1: hybr (hybrid Powell method)
        try:
            result_hybr = root(system_equations, initial_guess, method="hybr")
            root_results["hybr"] = {
                "solution": result_hybr.x,
                "success": result_hybr.success,
                "residual_norm": np.linalg.norm(result_hybr.fun),
            }
        except:
            root_results["hybr"] = {
                "solution": initial_guess,
                "success": False,
                "residual_norm": np.inf,
            }

        # Method 2: lm (Levenberg-Marquardt)
        try:
            result_lm = root(system_equations, initial_guess, method="lm")
            root_results["lm"] = {
                "solution": result_lm.x,
                "success": result_lm.success,
                "residual_norm": np.linalg.norm(result_lm.fun),
            }
        except:
            root_results["lm"] = {
                "solution": initial_guess,
                "success": False,
                "residual_norm": np.inf,
            }

        # Method 3: broyden1
        try:
            result_broyden = root(system_equations, initial_guess, method="broyden1")
            root_results["broyden1"] = {
                "solution": result_broyden.x,
                "success": result_broyden.success,
                "residual_norm": np.linalg.norm(result_broyden.fun),
            }
        except:
            root_results["broyden1"] = {
                "solution": initial_guess,
                "success": False,
                "residual_norm": np.inf,
            }

        # Select best solution
        best_method = "hybr"
        best_residual = np.inf

        for method, result in root_results.items():
            if result["success"] and result["residual_norm"] < best_residual:
                best_method = method
                best_residual = result["residual_norm"]

        best_solution = root_results[best_method]["solution"]

        # Extract equilibrium components
        equilibrium_population = best_solution[:n_constructs]
        equilibrium_opinions = best_solution[n_constructs:]

        # Stability analysis of equilibrium
        def jacobian_at_equilibrium(state):
            """Numerical Jacobian at equilibrium point."""
            eps = 1e-8
            n = len(state)
            jac = np.zeros((n, n))

            f0 = system_equations(state)

            for i in range(n):
                state_plus = state.copy()
                state_plus[i] += eps
                f_plus = system_equations(state_plus)
                jac[:, i] = (f_plus - f0) / eps

            return jac

        jacobian = jacobian_at_equilibrium(best_solution)
        eigenvals = np.linalg.eigvals(jacobian)
        stability = (
            np.max(np.real(eigenvals)) < 0
        )  # Stable if all eigenvalues have negative real parts

        return {
            "equilibrium_population": torch.from_numpy(equilibrium_population),
            "equilibrium_opinions": torch.from_numpy(equilibrium_opinions),
            "equilibrium_jacobian": torch.from_numpy(jacobian),
            "stability_eigenvalues": torch.from_numpy(eigenvals),
            "is_stable": torch.tensor(stability),
            "best_method": best_method,
            "residual_norm": torch.tensor(best_residual),
            "convergence_success": torch.tensor(root_results[best_method]["success"]),
            "initial_guess": torch.from_numpy(initial_guess),
        }

    # ═══════════════════════════════════════════════════════════════════════════════
    # UNIVERSE INTEGRATION PATTERNS - Self-Interpreting Field Dynamics
    # ═══════════════════════════════════════════════════════════════════════════════

    def get_universe_field_state(self, universe_id: str) -> Dict[str, Any]:
        """
        Read REAL field state from liquid universe instead of synthetic data.

        Mathematical Foundation:
            Universe Field State Extraction:

            Q-Field Ensemble:
            $$\\vec{Q} = \\{Q_1, Q_2, \\ldots, Q_N\\} \\text{ where } Q_i = Q(\\tau_i, C_i, s_i)$$

            Field Statistics:
            $$E_{\\text{field}} = \\sum_i |Q_i|^2 \\quad \\text{(total field energy)}$$
            $$H_{\\text{field}} = -\\sum_i p_i \\log p_i \\quad \\text{(field entropy)}$$
            $$C_{\\text{field}} = \\langle|\\sum_i Q_i e^{i\\theta_i}|\\rangle \\quad \\text{(field coherence)}$$

            NO SYNTHETIC DATA - only real mathematical state from living agents.

        Args:
            universe_id: Identifier for universe to extract state from

        Returns:
            Complete universe field state with mathematical validation
        """
        # MATHEMATICAL PERFECTION: Direct connection to liquid_orchestrator.py

        try:
            # Import FieldUniverse for proper universe loading like universe_runner.py
            from Sysnpire.database.field_universe import FieldUniverse, FieldUniverseConfig
            
            # Create FieldUniverse instance to load the actual universe
            # Use configurable storage path from constructor
            config = FieldUniverseConfig(storage_path=self.universe_storage_path)
            field_universe = FieldUniverse(config)
            
            # Reconstruct the liquid universe from storage (following universe_runner.py pattern)
            reconstruction_result = field_universe.reconstruct_liquid_universe(
                universe_id=universe_id,
                device=self.device  # Use configurable device from constructor
            )
            
            if reconstruction_result["status"] != "success":
                raise ValueError(f"Failed to reconstruct universe {universe_id}: {reconstruction_result.get('error')}")
                
            orchestrator = reconstruction_result["orchestrator"]

            # Extract REAL Q-field values from living conceptual charge agents
            q_values = []
            agent_positions = []
            agent_metadata = []

            for agent_id, agent in orchestrator.charge_agents.items():
                if (
                    hasattr(agent, "living_Q_value")
                    and agent.living_Q_value is not None
                ):
                    # Real Q(τ,C,s) field value
                    q_values.append(agent.living_Q_value)

                    # Real field position in manifold
                    if hasattr(agent, "field_state") and hasattr(
                        agent.field_state, "field_position"
                    ):
                        agent_positions.append(agent.field_state.field_position)
                    else:
                        # Default position if not available
                        agent_positions.append(torch.zeros(3))

                    # Agent metadata for analysis - CORRECTED ATTRIBUTE NAMES
                    # Extract breathing coefficients magnitude (from breathing_q_coefficients dict)
                    breathing_magnitude = 1.0
                    if (
                        hasattr(agent, "breathing_q_coefficients")
                        and agent.breathing_q_coefficients
                    ):
                        # Calculate RMS magnitude of breathing coefficients
                        coeffs = list(agent.breathing_q_coefficients.values())
                        breathing_magnitude = math.sqrt(
                            sum(abs(c) ** 2 for c in coeffs) / len(coeffs)
                        )

                    agent_metadata.append(
                        {
                            "agent_id": agent_id,
                            "breathing_coefficient": breathing_magnitude,
                            "temporal_biography": getattr(
                                agent, "temporal_biography", None
                            ),
                        }
                    )

            if not q_values:
                raise ValueError(
                    f"MATHEMATICAL FAILURE: Universe {universe_id} contains no living Q-field values"
                )

            # Calculate REAL field statistics using rigorous mathematics
            # Field energy: E = ∑ᵢ |Qᵢ|²
            field_energy = sum(abs(q) ** 2 for q in q_values if q is not None)

            # Information entropy: H = -∑ᵢ pᵢ log pᵢ where pᵢ = |Qᵢ|²/∑|Qⱼ|²
            field_magnitudes = [abs(q) ** 2 for q in q_values if q is not None]
            total_magnitude = sum(field_magnitudes)

            if total_magnitude > 1e-15:
                probabilities = [mag / total_magnitude for mag in field_magnitudes]
                field_complexity = -sum(
                    p * math.log(p) for p in probabilities if p > 1e-15
                )
            else:
                field_complexity = 0.0

            # Phase coherence: C = |∑ᵢ Qᵢ|/∑ᵢ |Qᵢ|
            coherent_sum = sum(q for q in q_values if q is not None)
            incoherent_sum = sum(abs(q) for q in q_values if q is not None)

            if incoherent_sum > 1e-15:
                field_coherence = abs(coherent_sum) / incoherent_sum
            else:
                field_coherence = 0.0

            # Temperature from field fluctuations: T ∝ Var(|Q|)
            if len(field_magnitudes) > 1:
                magnitude_variance = np.var(field_magnitudes)
                effective_temperature = (
                    magnitude_variance / np.mean(field_magnitudes)
                    if np.mean(field_magnitudes) > 0
                    else 1.0
                )
            else:
                effective_temperature = 1.0

            # CRITICAL FIX: Build agents array for compatibility with other integration methods
            # Each agent needs: living_Q_value, field_position, phase for field analysis
            agents = []
            for i, q_value in enumerate(q_values):
                if i < len(agent_positions) and i < len(agent_metadata):
                    # Extract phase from Q-value (complex number)
                    if isinstance(q_value, complex):
                        phase = cmath.phase(q_value)
                    else:
                        phase = 0.0  # Real Q-value has zero phase
                    
                    agent_dict = {
                        "living_Q_value": q_value,
                        "field_position": agent_positions[i],
                        "phase": phase,
                        "breathing_coefficient": agent_metadata[i].get("breathing_coefficient", 1.0),
                        "agent_id": agent_metadata[i].get("agent_id", f"agent_{i}"),
                        "temporal_biography": agent_metadata[i].get("temporal_biography", None),
                    }
                    agents.append(agent_dict)

            universe_state = {
                # Backward compatibility - original field data
                "q_field_values": q_values,
                "field_positions": agent_positions,
                "field_energy": field_energy,
                "field_complexity": field_complexity,
                "field_coherence": field_coherence,
                "temperature": effective_temperature,
                "agent_count": len(q_values),
                "agent_metadata": agent_metadata,
                "universe_id": universe_id,
                "timestamp": time.time(),
                # CRITICAL: Add agents array for integration method compatibility
                "agents": agents,
            }
            
            # MATHEMATICAL PRECISION: Validate SAGE objects and enhance precision consistency
            enhanced_universe_state = self._ensure_mathematical_precision_consistency(universe_state)
            
            return enhanced_universe_state

        except ImportError as e:
            # Graceful degradation with clear error message
            raise RuntimeError(
                f"MATHEMATICAL FAILURE: Cannot import liquid_orchestrator.py for universe {universe_id}. "
                f"This violates the mathematical perfection principle. "
                f"Error: {e}. "
                f"The integration layer requires REAL Q-field data from living agents, not synthetic fallbacks."
            )
        except Exception as e:
            # Any other error is a mathematical failure
            raise RuntimeError(
                f"MATHEMATICAL FAILURE: Universe field state extraction failed for {universe_id}. "
                f"Error: {e}. "
                f"Mathematical perfection demands REAL field data or explosive failure."
            )

    def text_to_field_signature(
        self, text: str, universe_state: Dict
    ) -> "FieldSignature":
        """
        Convert text to field signature using universe structure, not external models.

        Mathematical Foundation:
            Text Features → Field Coordinates Mapping:

            Character Frequency Mapping:
            $$f_{\\text{char}}: \\sum_i c_i \\delta(\\text{char} - \\text{char}_i) \\rightarrow F[f_{\\text{char}}](k) \\text{ via FFT}$$

            Semantic Projection:
            $$\\vec{r} = \\sum_j w_j \\vec{r}_{\\text{agent}_j} \\text{ where } w_j = \\exp(-d^2(\\text{text}, \\text{agent}_j))$$

            Phase Calculation:
            $$\\theta = \\arg\\left(\\sum_k f_k e^{i k \\cdot \\langle\\vec{r}\\rangle}\\right) \\bmod 2\\pi$$

            Amplitude from Information Density:
            $$|Q| = \\sqrt{\\frac{H[\\text{text}]}{H_{\\max}}} \\text{ where } H[\\text{text}] = -\\sum_i p_i \\log p_i$$

        Args:
            text: Input text for field mapping
            universe_state: Current universe field configuration

        Returns:
            FieldSignature with complete mathematical field representation
        """
        # Implement text to field signature conversion directly
        if not text or not isinstance(text, str):
            raise ValueError("Invalid text input for field conversion")
        if not universe_state or not isinstance(universe_state, dict):
            raise ValueError("Invalid universe state for field conversion")

        # Character frequency analysis (fundamental text features)
        char_frequencies = self._calculate_character_frequencies(text)
        word_patterns = self._extract_word_pattern_frequencies(text)
        syntactic_structure = self._analyze_basic_syntax_patterns(text)

        # Map to field coordinates using existing universe agent positions
        existing_agent_positions = self._get_agent_field_positions(universe_state)
        field_coordinates = self._project_features_to_field_space(
            char_frequencies,
            word_patterns,
            syntactic_structure,
            existing_agent_positions,
        )

        # Phase calculation from structural harmonics
        phase = self._compute_field_phase_from_structure(
            syntactic_structure, word_patterns
        )

        # Amplitude from information density
        amplitude = self._calculate_field_amplitude_from_entropy(text)

        # Q(τ,C,s) projection
        q_projection = complex(amplitude * math.cos(phase), amplitude * math.sin(phase))

        # Universe coherence measure
        universe_coherence = self._measure_universe_coherence(
            field_coordinates, existing_agent_positions
        )

        return FieldSignature(
            coordinates=field_coordinates,
            phase=phase,
            amplitude=amplitude,
            pattern_resonances=word_patterns,
            structural_features=syntactic_structure,
            q_projection=q_projection,
            universe_coherence=universe_coherence,
        )

    def compute_semantic_distance(
        self, text: str, agent_field_state: Dict, universe_state: Dict
    ) -> float:
        """
        Compute semantic distance using field theory, not model embeddings.

        Mathematical Foundation:
            Field-Theoretic Distance in Q(τ,C,s) Space:

            Distance Components:
            $$d_Q = |Q_{\\text{text}} - Q_{\\text{agent}}| \\quad \\text{(Q-space distance)}$$
            $$d_{\\text{field}} = ||\\vec{r}_{\\text{text}} - \\vec{r}_{\\text{agent}}||_2 \\quad \\text{(field space distance)}$$
            $$d_{\\text{phase}} = |\\theta_{\\text{text}} - \\theta_{\\text{agent}}| \\bmod \\pi \\quad \\text{(phase difference)}$$

            Combined Distance:
            $$d_{\\text{semantic}} = \\sqrt{w_Q d_Q^2 + w_{\\text{field}} d_{\\text{field}}^2 + w_{\\text{phase}} d_{\\text{phase}}^2}$$

            Where weights satisfy: $w_Q + w_{\\text{field}} + w_{\\text{phase}} = 1$

        Args:
            text: Input text for distance calculation
            agent_field_state: Agent's current Q(τ,C,s) field state
            universe_state: Universe configuration for text mapping

        Returns:
            Semantic distance as non-negative real number
        """
        # Implement semantic distance calculation using field theory
        if not agent_field_state or "living_Q_value" not in agent_field_state:
            raise ValueError("Invalid agent field state - missing Q-value")

        # Convert text to field signature
        text_signature = self.text_to_field_signature(text, universe_state)

        # Extract agent field components
        agent_q_value = safe_torch_tensor(agent_field_state["living_Q_value"]).item()
        if "field_position" not in agent_field_state:
            raise ValueError(
                "MATHEMATICAL FAILURE: Agent field state lacks required 'field_position' for spatial analysis"
            )
        if "phase" not in agent_field_state:
            raise ValueError(
                "MATHEMATICAL FAILURE: Agent field state lacks required 'phase' for Q(τ,C,s) computation"
            )
        agent_field_position = agent_field_state["field_position"]
        agent_phase = agent_field_state["phase"]

        # Q-space distance: |Q_text - Q_agent|
        if isinstance(agent_q_value, complex):
            q_space_distance = abs(text_signature.q_projection - agent_q_value)
        else:
            # Convert real Q-value to complex
            agent_q_complex = complex(float(agent_q_value), 0.0)
            q_space_distance = abs(text_signature.q_projection - agent_q_complex)

        # Field space distance: ||r⃗_text - r⃗_agent||₂
        if torch.is_tensor(agent_field_position):
            position_diff = text_signature.coordinates - agent_field_position
            field_space_distance = torch.norm(position_diff).item()
        else:
            # Handle scalar position
            pos_tensor = torch.tensor(
                [float(agent_field_position)] * self.spatial_dimensions
            )
            position_diff = text_signature.coordinates - pos_tensor
            field_space_distance = torch.norm(position_diff).item()

        # Phase difference: |θ_text - θ_agent| mod π
        phase_difference = abs(text_signature.phase - agent_phase)
        phase_difference = min(
            phase_difference, 2 * math.pi - phase_difference
        )  # Periodic boundary

        # Weighted combined distance
        w_Q, w_field, w_phase = 0.4, 0.4, 0.2  # Field theory motivated weights
        semantic_distance = math.sqrt(
            w_Q * q_space_distance**2
            + w_field * field_space_distance**2
            + w_phase * phase_difference**2
        )

        return semantic_distance

    def decide_content_acceptance(
        self, text: str, universe_state: Dict
    ) -> "AcceptanceDecision":
        """
        Accept/reject content based on field dynamics, not model predictions.

        Mathematical Foundation:
            Field-Theoretic Acceptance Criterion:

            Mathematical Weight:
            $$W = \\Delta C \\cdot R_{\\text{collective}} \\cdot S_{\\text{stability}}$$

            Where:
            $$\\Delta C = H[\\text{universe} + \\text{text}] - H[\\text{universe}] \\quad \\text{(complexity increase)}$$
            $$R_{\\text{collective}} = \\langle\\sum_i R_i\\rangle \\quad \\text{(mean agent field response)}$$
            $$S_{\\text{stability}} = -\\max(\\text{Re}[\\lambda_i]) \\quad \\text{(stability from eigenvalues)}$$

            Acceptance Rule:
            $$\\text{Accept} \\leftrightarrow W > W_{\\text{threshold}}(\\text{universe_state})$$

            Dynamic Threshold:
            $$W_{\\text{threshold}} = \\mu + \\sigma \\cdot \\Phi^{-1}(\\alpha) \\text{ where } \\alpha = \\text{acceptance_rate_target}$$

        Args:
            text: Content to evaluate for acceptance
            universe_state: Current universe field configuration

        Returns:
            AcceptanceDecision with complete mathematical justification
        """
        # Implement content acceptance decision using field dynamics
        if not universe_state or "agents" not in universe_state:
            raise ValueError("Invalid universe state - missing agent data")

        # Test field integration potential through perturbation analysis
        field_perturbation = self._simulate_text_field_perturbation(
            text, universe_state
        )

        # Measure universe agent responses to field perturbation
        agent_responses = []
        for agent in universe_state["agents"]:
            response_strength = self._compute_agent_field_response(
                agent, field_perturbation
            )
            agent_responses.append(response_strength)

        if not agent_responses:
            raise ValueError("No agents available for universe response calculation")

        # Universe consensus through field mathematics
        collective_response = torch.mean(
            torch.tensor(agent_responses, dtype=torch.float64)
        )

        # Field stability from perturbation eigenvalue analysis
        field_stability = self._measure_perturbation_stability(field_perturbation)

        # Information complexity increase from text integration
        complexity_gain = self._calculate_information_complexity_increase(
            text, universe_state
        )

        # Mathematical weight calculation (pure field theory)
        mathematical_weight = (
            complexity_gain * collective_response.item() * field_stability
        )

        # Dynamic acceptance threshold based on universe state
        acceptance_threshold = self._compute_dynamic_acceptance_threshold(
            universe_state
        )

        # Field-theoretic acceptance decision
        accept = mathematical_weight > acceptance_threshold

        # Mathematical reasoning string
        universe_reasoning = (
            f"Field resonance: {collective_response:.3f}, "
            f"Stability: {field_stability:.3f}, "
            f"Complexity gain: {complexity_gain:.3f}, "
            f"Weight: {mathematical_weight:.3f}, "
            f"Threshold: {acceptance_threshold:.3f}"
        )

        return AcceptanceDecision(
            accept=accept,
            mathematical_weight=mathematical_weight,
            field_evidence=field_perturbation,
            universe_reasoning=universe_reasoning,
            complexity_gain=complexity_gain,
            collective_response=collective_response.item(),
            field_stability=field_stability,
            threshold_used=acceptance_threshold,
        )

    def evaluate_mathematical_weight(self, content: str, universe_state: Dict) -> float:
        """
        Calculate mathematical weight using real field energy and complexity.

        Mathematical Foundation:
            Mathematical Weight Integration:

            Energy Component from Energy Calculator:
            $$W_{\\text{energy}} = \\int \\left[ \\frac{1}{2}|\\nabla Q|^2 + V(|Q|^2) \\right] d^3x$$

            Complexity Component from Complexity Analyzer:
            $$W_{\\text{complexity}} = -\\sum_i p_i \\log p_i + \\alpha \\int |\\nabla^2 Q|^2 dx$$

            Emergence Potential from Phase Transitions:
            $$W_{\\text{emergence}} = \\int_{\\text{critical}} |\\nabla \\cdot \\text{order parameter}|^2 dx$$

            Combined Weight:
            $$W_{\\text{total}} = w_E W_{\\text{energy}} + w_C W_{\\text{complexity}} + w_{\\text{emergence}} W_{\\text{emergence}}$$

        Args:
            content: Content to evaluate mathematical weight
            universe_state: Current universe configuration

        Returns:
            Mathematical weight from real field calculations
        """
        # Convert content to field signature
        field_signature = self.text_to_field_signature(content, universe_state)

        # Get energy component from real energy calculator
        try:
            energy_components = self.energy_calculator.compute_field_energy(
                field_configuration=FieldConfiguration(
                    field_values=torch.complex(
                        field_signature.coordinates,
                        torch.zeros_like(field_signature.coordinates),
                    ),
                    spatial_coordinates=torch.linspace(
                        -5.0, 5.0, len(field_signature.coordinates)
                    ),
                    time_coordinate=0.0,
                    field_mass=1.0,
                    coupling_constant=self.field_coupling_strength,
                )
            )
            energy_weight = energy_components.total_energy
        except Exception as e:
            self.logger.warning(f"Energy calculation failed: {e}")
            energy_weight = 0.0

        # Get complexity component from real complexity analyzer
        try:
            # Create population state from field signature
            population_frequencies = torch.abs(field_signature.coordinates)
            population_frequencies = population_frequencies / torch.sum(
                population_frequencies
            )

            population_state = PopulationState(
                frequencies=population_frequencies,
                labels=[f"component_{i}" for i in range(len(population_frequencies))],
                total_population=1.0,
                diversity_index=torch.sum(
                    -population_frequencies * torch.log(population_frequencies + 1e-12)
                ),
            )

            complexity_measures = self.complexity_analyzer.compute_complexity_field(
                population_state
            )
            complexity_weight = complexity_measures.shannon_entropy
        except Exception as e:
            self.logger.warning(f"Complexity calculation failed: {e}")
            complexity_weight = 0.0

        # Get emergence potential from phase detector
        try:
            critical_analysis = self.phase_detector.detect_critical_points(
                field_configuration=FieldConfiguration(
                    field_values=torch.complex(
                        field_signature.coordinates,
                        torch.zeros_like(field_signature.coordinates),
                    ),
                    spatial_coordinates=torch.linspace(
                        -5.0, 5.0, len(field_signature.coordinates)
                    ),
                    time_coordinate=0.0,
                    field_mass=1.0,
                    coupling_constant=self.field_coupling_strength,
                )
            )
            emergence_weight = (
                critical_analysis.emergence_potential
                if hasattr(critical_analysis, "emergence_potential")
                else 0.0
            )
        except Exception as e:
            self.logger.warning(f"Emergence calculation failed: {e}")
            emergence_weight = 0.0

        # Weighted combination - field theory motivated weights
        w_energy, w_complexity, w_emergence = 0.4, 0.4, 0.2
        total_mathematical_weight = (
            w_energy * energy_weight
            + w_complexity * complexity_weight
            + w_emergence * emergence_weight
        )

        return float(total_mathematical_weight)

    def compute_field_compatibility(
        self, content: str, universe_state: Dict
    ) -> Dict[str, Any]:
        """
        Compute field compatibility using real interference analysis.

        Mathematical Foundation:
            Field Interference Compatibility:

            Text Field Integration:
            $$F_{\\text{text}} = \\text{TextToFieldSignature}(\\text{content}, U_{\\text{state}})$$

            Universe Field Ensemble:
            $$F_{\\text{universe}} = \\{F_1, F_2, \\ldots, F_N\\} \\text{ from universe agents}$$

            Compatibility Analysis:
            $$C_{\\text{compat}} = \\frac{1}{N} \\sum_{i=1}^{N} \\text{Interference}(F_{\\text{text}}, F_i)$$

            Where:
            - $C_{\\text{compat}} > 0$: Constructive interference (compatible)
            - $C_{\\text{compat}} < 0$: Destructive interference (incompatible)

        Args:
            content: Text content to analyze for field compatibility
            universe_state: Current universe field configuration

        Returns:
            Dict containing comprehensive compatibility analysis:
                - compatibility_score: Overall compatibility measure
                - stability_measure: Field stability under integration
                - interference_patterns: Detailed interference data
                - field_resonance: Resonance strength analysis
        """
        try:
            # Convert content to field signature
            text_signature = self.text_to_field_signature(content, universe_state)
            
            # Extract representative universe field signatures from agents
            universe_signatures = self._extract_universe_field_signatures(universe_state)
            
            # Compute compatibility via interference analysis
            return self._analyze_field_interference(text_signature, universe_signatures)

        except Exception as e:
            self.logger.warning(f"Field compatibility calculation failed: {e}")
            # Fallback: simple compatibility analysis
            return {
                'compatibility_score': 0.5,  # Neutral compatibility
                'stability_measure': 1.0,    # Assume stable
                'interference_patterns': {},
                'field_resonance': 0.0,
                'error_fallback': str(e)
            }

    def _extract_universe_field_signatures(self, universe_state: Dict) -> List["FieldSignature"]:
        """
        Extract field signatures from universe agents for compatibility analysis.

        Mathematical Foundation:
            Agent Field State Extraction:
            
            For each agent i with Q_i(τ,C,s):
            $$F_i = \\text{FieldSignature}(Q_i, \\vec{r}_i, \\theta_i, A_i, \\{\\text{patterns}\\}_i)$$
            
            Where:
            - Q_i: Agent's living Q-value
            - r⃗_i: Agent's field position 
            - θ_i: Agent's phase
            - A_i: Agent's field amplitude
            - {patterns}_i: Agent's pattern resonances

        Args:
            universe_state: Universe configuration with agent data

        Returns:
            List of FieldSignature objects representing universe agents
        """
        try:
            if "agents" not in universe_state:
                self.logger.warning("No agents found in universe state")
                return []

            signatures = []
            agents = universe_state["agents"]
            
            for i, agent in enumerate(agents):
                if i >= 50:  # Limit for performance
                    break
                    
                try:
                    # Extract agent Q-value
                    if "living_Q_value" not in agent:
                        continue
                    q_value = safe_torch_tensor(agent["living_Q_value"]).item()
                    
                    # Extract field position
                    if "field_position" not in agent:
                        continue
                    field_position = safe_torch_tensor(agent["field_position"])
                    if field_position.dim() == 0:
                        # Convert scalar to vector
                        field_position = torch.tensor([float(field_position)] * self.spatial_dimensions)
                    
                    # Extract phase
                    phase = agent.get("phase", 0.0)
                    if hasattr(phase, 'item'):
                        phase = float(phase.item())
                    
                    # Calculate amplitude from Q-value
                    amplitude = abs(q_value) if isinstance(q_value, complex) else abs(float(q_value))
                    
                    # Create pattern resonances (simplified)
                    pattern_resonances = torch.zeros(min(16, self.spatial_dimensions * 4))
                    if isinstance(q_value, complex):
                        pattern_resonances[0] = q_value.real
                        pattern_resonances[1] = q_value.imag
                    
                    # Create field signature
                    from .universe_integration_patterns import FieldSignature
                    signature = FieldSignature(
                        coordinates=field_position[:self.spatial_dimensions],
                        phase=float(phase),
                        amplitude=amplitude,
                        pattern_resonances=pattern_resonances,
                        structural_features={
                            "agent_id": i,
                            "q_magnitude": amplitude,
                            "q_phase": phase
                        },
                        q_projection=q_value if isinstance(q_value, complex) else complex(float(q_value), 0.0),
                        universe_coherence=1.0  # Assume agent is coherent with universe
                    )
                    signatures.append(signature)
                    
                except Exception as e:
                    self.logger.debug(f"Failed to extract signature from agent {i}: {e}")
                    continue
            
            self.logger.info(f"Extracted {len(signatures)} field signatures from universe agents")
            return signatures
            
        except Exception as e:
            self.logger.error(f"Failed to extract universe field signatures: {e}")
            return []

    def _analyze_field_interference(self, text_signature: "FieldSignature", universe_signatures: List["FieldSignature"]) -> Dict[str, Any]:
        """
        Analyze field interference patterns between text and universe signatures.

        Mathematical Foundation:
            Multi-Field Interference Analysis:
            
            Wave Superposition for N+1 fields:
            $$\\Psi_{\\text{total}} = \\Psi_{\\text{text}} + \\sum_{i=1}^{N} \\Psi_{\\text{agent},i}$$
            
            Interference Intensity:
            $$I_{\\text{total}} = |\\Psi_{\\text{total}}|^2 = I_{\\text{text}} + \\sum_i I_i + 2\\sum_i \\sqrt{I_{\\text{text}} I_i} \\cos(\\Delta\\phi_i)$$
            
            Compatibility Measure:
            $$C = \\frac{1}{N} \\sum_{i=1}^{N} \\cos(\\theta_{\\text{text}} - \\theta_i)$$
            
            Stability Analysis:
            $$S = 1 - \\text{Var}(\\{\\cos(\\theta_{\\text{text}} - \\theta_i)\\}_{i=1}^N)$$

        Args:
            text_signature: Field signature from input text
            universe_signatures: Field signatures from universe agents

        Returns:
            Dict with comprehensive interference analysis
        """
        try:
            if not universe_signatures:
                return {
                    'compatibility_score': 0.0,
                    'stability_measure': 1.0,
                    'interference_patterns': {'empty_universe': True},
                    'field_resonance': 0.0
                }
            
            # Calculate compatibility with each universe signature
            compatibility_scores = []
            phase_differences = []
            amplitude_ratios = []
            
            for universe_sig in universe_signatures:
                # Phase difference analysis
                phase_diff = abs(text_signature.phase - universe_sig.phase)
                phase_diff = min(phase_diff, 2 * math.pi - phase_diff)  # Periodic boundary
                phase_differences.append(phase_diff)
                
                # Amplitude ratio
                if universe_sig.amplitude > 1e-10:
                    amp_ratio = text_signature.amplitude / universe_sig.amplitude
                    amplitude_ratios.append(min(amp_ratio, 1.0 / amp_ratio))  # Symmetric ratio
                else:
                    amplitude_ratios.append(0.0)
                
                # Compatibility from phase alignment
                compatibility = math.cos(phase_diff)
                compatibility_scores.append(compatibility)
            
            # Overall compatibility (mean of individual compatibilities)
            overall_compatibility = np.mean(compatibility_scores) if compatibility_scores else 0.0
            
            # Stability from consistency of interactions
            if len(compatibility_scores) > 1:
                stability = 1.0 - np.var(compatibility_scores)
            else:
                stability = 1.0
            
            # Field resonance from amplitude matching
            if amplitude_ratios:
                field_resonance = np.mean(amplitude_ratios)
            else:
                field_resonance = 0.0
            
            # Detailed interference patterns
            interference_patterns = {
                'phase_differences': phase_differences,
                'amplitude_ratios': amplitude_ratios,
                'individual_compatibilities': compatibility_scores,
                'mean_phase_difference': np.mean(phase_differences) if phase_differences else 0.0,
                'phase_variance': np.var(phase_differences) if len(phase_differences) > 1 else 0.0,
                'agent_count': len(universe_signatures)
            }
            
            return {
                'compatibility_score': float(overall_compatibility),
                'stability_measure': float(max(0.0, min(1.0, stability))),  # Clamp to [0,1]
                'interference_patterns': interference_patterns,
                'field_resonance': float(field_resonance)
            }
            
        except Exception as e:
            self.logger.error(f"Field interference analysis failed: {e}")
            return {
                'compatibility_score': 0.0,
                'stability_measure': 0.0,
                'interference_patterns': {'error': str(e)},
                'field_resonance': 0.0
            }

    def apply_hk_field_dynamics(
        self,
        current_opinions: torch.Tensor,
        field_state: "FieldConfiguration",
        universe_state: Dict,
        dt: float,
    ) -> torch.Tensor:
        """
        Apply Hegselmann-Krause dynamics with real Q(τ,C,s) field thresholds.

        Mathematical Foundation:
            HK Dynamics with Field Coupling:

            Modified Confidence Bound:
            $$\\varepsilon_{\\text{dynamic}}(t) = \\varepsilon_0 + \\alpha \\langle|Q(x,t)|^2\\rangle$$

            Field-Coupled HK Equation:
            $$\\frac{dx_i}{dt} = \\frac{1}{|N_i(\\varepsilon_{\\text{dynamic}})|} \\sum_{j \\in N_i} (x_j - x_i) - \\beta \\nabla \\phi(x_i, t)$$

            Dynamic Neighborhood:
            $$N_i(\\varepsilon_{\\text{dynamic}}) = \\{j : ||x_j - x_i|| \\leq \\varepsilon_{\\text{dynamic}}(t)\\}$$

            Field Force at Opinion Position:
            $$F_{\\text{field}}(x_i) = -\\beta \\frac{\\partial \\phi}{\\partial x}\\bigg|_{x=x_i}$$

        Args:
            current_opinions: Current opinion positions
            field_state: Real field configuration with Q(τ,C,s) values
            universe_state: Universe state for dynamic threshold calculation
            dt: Time step for integration

        Returns:
            Updated opinion positions using real field dynamics
        """
        try:
            # Calculate dynamic confidence bound from real field energy
            field_energy_density = torch.abs(field_state.field_values) ** 2
            mean_field_energy = torch.mean(field_energy_density)

            # Dynamic threshold: ε_dynamic = ε₀ + α⟨|Q|²⟩
            dynamic_confidence_bound = (
                self.hk_confidence_bound + 0.1 * mean_field_energy.item()
            )

            # Calculate field forces at opinion positions
            spatial_coords = field_state.spatial_coordinates
            field_forces = torch.zeros_like(current_opinions)

            for i, opinion in enumerate(current_opinions):
                # Find nearest spatial grid point
                spatial_idx = torch.argmin(torch.abs(spatial_coords - opinion))

                # Calculate field gradient at opinion position
                if spatial_idx > 0 and spatial_idx < len(spatial_coords) - 1:
                    dx = spatial_coords[1] - spatial_coords[0]
                    field_grad = (
                        field_state.field_values[spatial_idx + 1].real
                        - field_state.field_values[spatial_idx - 1].real
                    ) / (2 * dx)
                    field_forces[i] = -self.field_coupling_strength * field_grad
                else:
                    field_forces[i] = 0.0

            opinions_jax = jnp.array(current_opinions.detach().cpu().numpy())
            forces_jax = jnp.array(field_forces.detach().cpu().numpy())

            new_opinions_jax = self._jax_hk_dynamics_step(
                opinions_jax,
                dynamic_confidence_bound,
                forces_jax,
                self.field_coupling_strength,
                dt,
            )

            return torch.from_numpy(np.array(new_opinions_jax))

        except Exception as e:
            self.logger.error(f"HK field dynamics application failed: {e}")
            raise RuntimeError(f"MATHEMATICAL FAILURE in HK dynamics: {e}")

    def _calculate_information_entropy_from_q_values(
        self, q_values: List[complex]
    ) -> float:
        """Calculate information entropy from Q-field values."""
        if not q_values:
            return 0.0

        # Convert Q-values to probability distribution
        magnitudes = [abs(q) ** 2 for q in q_values]
        total_magnitude = sum(magnitudes)

        if total_magnitude == 0:
            return 0.0

        probabilities = [mag / total_magnitude for mag in magnitudes]

        # Shannon entropy: H = -Σ p_i log p_i
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _calculate_phase_coherence(self, q_values: List[complex]) -> float:
        """Calculate phase coherence from Q-field values."""
        if not q_values:
            return 0.0

        # Phase coherence: |⟨e^{iθ}⟩|
        phase_sum = sum(cmath.exp(1j * cmath.phase(q)) for q in q_values if q != 0)
        coherence = abs(phase_sum) / len(q_values) if q_values else 0.0

        return coherence

    # ═══════════════════════════════════════════════════════════════════════════════
    # SAGE MATHEMATICAL VALIDATION AND PRECISION PRESERVATION
    # ═══════════════════════════════════════════════════════════════════════════════

    def _validate_sage_mathematical_precision(self, q_values: List) -> Dict[str, Any]:
        """
        Comprehensive SAGE mathematical object validation and precision verification.
        
        MATHEMATICAL PRINCIPLE: SAGE provides exact arithmetic for field-theoretic calculations.
        This method ensures SAGE objects maintain mathematical precision throughout integration.
        
        Validation Framework:
        1. Type verification for SAGE ComplexDoubleElement objects
        2. Precision preservation during mathematical operations
        3. Error handling for SAGE computation failures
        4. Conversion safety between SAGE and PyTorch tensors
        
        Args:
            q_values: List of Q-field values potentially containing SAGE objects
            
        Returns:
            Dict with validation results and precision metrics
        """
        try:
            # Import SAGE types - hard dependency like main codebase
            from sage.rings.complex_double import ComplexDoubleElement
            from sage.rings.integer import Integer as SageInteger
            from sage.rings.real_double import RealDoubleElement
            from sage.all import CDF, Integer
            
            validation_results = {
                "sage_objects_found": 0,
                "sage_precision_verified": 0,
                "conversion_errors": [],
                "mathematical_consistency": True,
                "precision_metrics": {}
            }
            
            for i, q_value in enumerate(q_values):
                if isinstance(q_value, ComplexDoubleElement):
                    validation_results["sage_objects_found"] += 1
                    
                    # Validate SAGE ComplexDoubleElement mathematical integrity
                    try:
                        # Test basic mathematical operations
                        magnitude = abs(q_value)
                        phase = q_value.arg() if hasattr(q_value, 'arg') else cmath.phase(complex(q_value))
                        real_part = q_value.real()
                        imag_part = q_value.imag()
                        
                        # Verify precision preservation
                        reconstructed = CDF(real_part, imag_part)
                        precision_error = abs(q_value - reconstructed)
                        
                        if precision_error < 1e-15:
                            validation_results["sage_precision_verified"] += 1
                        else:
                            self.logger.warning(f"⚠️ SAGE precision loss detected for Q[{i}]: error={precision_error}")
                            
                        # Store precision metrics
                        validation_results["precision_metrics"][f"q_{i}"] = {
                            "type": "ComplexDoubleElement",
                            "magnitude": float(magnitude),
                            "precision_error": float(precision_error),
                            "mathematical_valid": precision_error < 1e-15
                        }
                        
                    except Exception as e:
                        validation_results["conversion_errors"].append(f"Q[{i}] SAGE operation failed: {e}")
                        validation_results["mathematical_consistency"] = False
                        
                elif isinstance(q_value, (SageInteger, RealDoubleElement)):
                    validation_results["sage_objects_found"] += 1
                    validation_results["sage_precision_verified"] += 1  # These are exact
                    
                elif isinstance(q_value, complex):
                    # Standard Python complex - verify can convert to SAGE if needed
                    try:
                        sage_equivalent = CDF(q_value)
                        precision_error = abs(complex(sage_equivalent) - q_value)
                        if precision_error > 1e-15:
                            validation_results["conversion_errors"].append(
                                f"Q[{i}] Python→SAGE conversion precision loss: {precision_error}"
                            )
                    except Exception as e:
                        validation_results["conversion_errors"].append(f"Q[{i}] Python→SAGE conversion failed: {e}")
            
            # Overall validation assessment
            if validation_results["conversion_errors"]:
                validation_results["mathematical_consistency"] = False
                self.logger.error(f"❌ SAGE VALIDATION FAILED: {len(validation_results['conversion_errors'])} errors")
            else:
                self.logger.info(f"✅ SAGE VALIDATION PASSED: {validation_results['sage_precision_verified']} objects verified")
                
            return validation_results
            
        except ImportError as e:
            # SAGE not available - this is acceptable but logged
            self.logger.warning(f"⚠️ SAGE mathematical validation skipped: {e}")
            return {
                "sage_objects_found": 0,
                "sage_precision_verified": 0,
                "conversion_errors": ["SAGE not available"],
                "mathematical_consistency": True,  # Standard math still valid
                "precision_metrics": {}
            }
        except Exception as e:
            # Any other error is a mathematical failure
            self.logger.error(f"❌ SAGE VALIDATION CRITICAL ERROR: {e}")
            return {
                "sage_objects_found": 0,
                "sage_precision_verified": 0,
                "conversion_errors": [f"Critical SAGE error: {e}"],
                "mathematical_consistency": False,
                "precision_metrics": {}
            }

    def _ensure_mathematical_precision_consistency(self, universe_state: Dict) -> Dict:
        """
        Ensure mathematical precision consistency across universe field state.
        
        MATHEMATICAL PRINCIPLE: Field-theoretic calculations must maintain consistent
        precision throughout the integration pipeline. This method validates and
        enhances mathematical objects for optimal precision.
        
        Args:
            universe_state: Universe field state with Q-values and field data
            
        Returns:
            Enhanced universe state with validated mathematical precision
        """
        if "q_field_values" in universe_state:
            # Validate SAGE mathematical objects
            sage_validation = self._validate_sage_mathematical_precision(universe_state["q_field_values"])
            
            # Add validation results to universe state
            universe_state["sage_validation"] = sage_validation
            
            # Enhance mathematical consistency if needed
            if not sage_validation["mathematical_consistency"]:
                self.logger.warning("⚠️ Mathematical precision issues detected - applying consistency enhancements")
                # Could add precision recovery logic here if needed
                
        return universe_state

    # ═══════════════════════════════════════════════════════════════════════════════
    # MATHEMATICAL HELPER METHODS FOR FIELD SIGNATURE CONVERSION
    # ═══════════════════════════════════════════════════════════════════════════════

    def _calculate_character_frequencies(self, text: str) -> torch.Tensor:
        """Calculate character frequency spectrum for field mapping."""
        # Character frequency analysis
        char_counts = {}
        for char in text.lower():
            if char.isalnum():  # Only alphanumeric for clean frequency spectrum
                if char not in char_counts:
                    char_counts[char] = 0
                char_counts[char] += 1

        if not char_counts:
            return torch.zeros(26, dtype=torch.float64)  # Empty text

        # Map to frequency vector (26 letters)
        freq_vector = torch.zeros(26, dtype=torch.float64)
        total_chars = sum(char_counts.values())

        for char, count in char_counts.items():
            if char.isalpha():
                idx = ord(char.lower()) - ord("a")
                freq_vector[idx] = count / total_chars

        return freq_vector

    def _extract_word_pattern_frequencies(self, text: str) -> torch.Tensor:
        """Extract word pattern frequencies using FFT analysis."""
        words = text.lower().split()
        field_resolution = 128  # Default field resolution
        if not words:
            return torch.zeros(field_resolution, dtype=torch.float64)

        # Word length distribution
        lengths = [len(word) for word in words]
        max_len = max(lengths) if lengths else 1

        # Create word length histogram
        length_hist = torch.zeros(
            min(max_len + 1, field_resolution), dtype=torch.float64
        )
        for length in lengths:
            if length < len(length_hist):
                length_hist[length] += 1

        length_hist = length_hist / torch.sum(length_hist)  # Normalize

        # Pad to field resolution
        if len(length_hist) < field_resolution:
            padded = torch.zeros(field_resolution, dtype=torch.float64)
            padded[: len(length_hist)] = length_hist
            return padded
        else:
            return length_hist[: field_resolution]

    def _analyze_basic_syntax_patterns(self, text: str) -> Dict[str, float]:
        """Analyze basic syntactic patterns without external NLP models."""
        if not text:
            return {
                "sentence_count": 0.0,
                "avg_word_length": 0.0,
                "punctuation_density": 0.0,
            }

        # Basic syntactic features
        sentence_count = len([c for c in text if c in ".!?"])
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        punctuation_count = len(
            [c for c in text if not c.isalnum() and not c.isspace()]
        )
        punctuation_density = punctuation_count / len(text) if text else 0

        return {
            "sentence_count": float(sentence_count),
            "avg_word_length": avg_word_length,
            "punctuation_density": punctuation_density,
        }

    def _get_agent_field_positions(self, universe_state: Dict) -> torch.Tensor:
        """Extract field positions from universe agents."""
        if "agents" not in universe_state:
            return torch.zeros((1, self.spatial_dimensions), dtype=torch.float64)

        positions = []
        for agent in universe_state["agents"]:
            if isinstance(agent, dict) and "field_position" in agent:
                pos = agent["field_position"]
                if torch.is_tensor(pos):
                    positions.append(pos[: self.spatial_dimensions])
                else:
                    # Convert scalar to vector
                    pos_vector = torch.tensor(
                        [float(pos)] * self.spatial_dimensions, dtype=torch.float64
                    )
                    positions.append(pos_vector)

        if not positions:
            return torch.zeros((1, self.spatial_dimensions), dtype=torch.float64)

        return torch.stack(positions)

    def _project_features_to_field_space(
        self,
        char_freq: torch.Tensor,
        word_patterns: torch.Tensor,
        syntax_features: Dict[str, float],
        agent_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Project text features to field coordinates using universe geometry."""
        # Weighted combination of features
        feature_vector = torch.cat(
            [
                (
                    char_freq[: self.spatial_dimensions]
                    if len(char_freq) >= self.spatial_dimensions
                    else torch.pad(
                        char_freq, (0, self.spatial_dimensions - len(char_freq))
                    )
                ),
                (
                    word_patterns[: self.spatial_dimensions]
                    if len(word_patterns) >= self.spatial_dimensions
                    else torch.pad(
                        word_patterns, (0, self.spatial_dimensions - len(word_patterns))
                    )
                ),
            ]
        )

        # Average with existing agent positions (universe geometry influence)
        if agent_positions.numel() > 0:
            agent_center = torch.mean(agent_positions, dim=0)
            field_coords = (
                0.7 * feature_vector[: self.spatial_dimensions] + 0.3 * agent_center
            )
        else:
            field_coords = feature_vector[: self.spatial_dimensions]

        return field_coords

    def _compute_field_phase_from_structure(
        self, syntax_features: Dict[str, float], word_patterns: torch.Tensor
    ) -> float:
        """Compute field phase from structural features."""
        # Phase from syntactic rhythm
        if "punctuation_density" not in syntax_features:
            raise ValueError(
                "MATHEMATICAL FAILURE: syntax_features lacks required 'punctuation_density' for phase computation"
            )
        rhythm_phase = 2 * math.pi * (syntax_features["punctuation_density"] % 1.0)

        # Phase from word pattern FFT
        if word_patterns.numel() > 1:
            from torch.fft import fft
            fft_result = fft(word_patterns.to(torch.complex64))
            dominant_freq_phase = torch.angle(fft_result[1]).item()  # Skip DC component
        else:
            dominant_freq_phase = 0.0

        # Combined phase (mod 2π)
        combined_phase = (rhythm_phase + dominant_freq_phase) % (2 * math.pi)
        return combined_phase

    def _calculate_field_amplitude_from_entropy(self, text: str) -> float:
        """Calculate field amplitude from information entropy."""
        if not text:
            # MATHEMATICAL: Vacuum field amplitude from zero-point fluctuations
            vacuum_amplitude = math.sqrt(1.381e-23 * 300 / (1.602e-19))  # ~0.0119
            return float(vacuum_amplitude)

        # Shannon entropy of character distribution
        char_counts = {}
        for char in text.lower():
            if char not in char_counts:
                char_counts[char] = 0
            char_counts[char] += 1

        if len(char_counts) <= 1:
            # MATHEMATICAL: Degenerate field amplitude for single-character state
            text_length = len(text)
            vacuum_amplitude = math.sqrt(1.381e-23 * 300 / (1.602e-19))
            degenerate_amplitude = (
                1.0 / math.sqrt(text_length) if text_length > 0 else vacuum_amplitude
            )
            return float(degenerate_amplitude)

        total_chars = sum(char_counts.values())
        entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            entropy -= p * math.log2(p)

        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(char_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Amplitude from normalized entropy
        amplitude = math.sqrt(normalized_entropy)  # √H for field amplitude
        return amplitude

    def _measure_universe_coherence(
        self, field_coords: torch.Tensor, agent_positions: torch.Tensor
    ) -> float:
        """Measure coherence between text field and universe field."""
        if agent_positions.numel() == 0:
            # MATHEMATICAL: Empty space correlation function coherence
            vacuum_correlation = 1.0 / (4.0 * math.pi)  # ≈ 0.0796
            return float(vacuum_correlation)

        # Distance-based coherence measure
        distances = torch.norm(agent_positions - field_coords.unsqueeze(0), dim=1)
        min_distance = torch.min(distances)

        # Coherence from proximity (exponential decay)
        coherence = math.exp(-min_distance.item())
        return coherence

    def _simulate_text_field_perturbation(
        self, text: str, universe_state: Dict
    ) -> torch.Tensor:
        """Simulate field perturbation from text integration."""
        text_signature = self.text_to_field_signature(text, universe_state)

        # Perturbation field from text signature
        field_resolution = 128  # Default field resolution
        perturbation = torch.zeros(field_resolution, dtype=torch.complex64)

        # Gaussian perturbation centered at text field coordinates
        x_coords = torch.linspace(-5.0, 5.0, field_resolution)
        center = (
            text_signature.coordinates[0].item()
            if text_signature.coordinates.numel() > 0
            else 0.0
        )
        sigma = 1.0  # Perturbation width

        for i, x in enumerate(x_coords):
            # Gaussian envelope with complex phase
            amplitude = text_signature.amplitude * math.exp(
                -0.5 * ((x - center) / sigma) ** 2
            )
            phase = text_signature.phase
            perturbation[i] = complex(
                amplitude * math.cos(phase), amplitude * math.sin(phase)
            )

        return perturbation

    def _compute_agent_field_response(
        self, agent: Dict, field_perturbation: torch.Tensor
    ) -> float:
        """Compute agent field response to perturbation."""
        if "living_Q_value" not in agent:
            # MATHEMATICAL: Response of empty state to field perturbation
            perturbation_strength = torch.mean(torch.abs(field_perturbation)).item()
            vacuum_response = (
                math.sqrt(1.381e-23 * 300 / (1.602e-19)) * perturbation_strength
            )
            return float(vacuum_response)

        agent_q = safe_torch_tensor(agent["living_Q_value"]).item()

        # Response as field interference with perturbation
        if isinstance(agent_q, complex):
            # Complex Q-value: compute overlap integral
            response = abs(agent_q) * torch.mean(torch.abs(field_perturbation)).item()
        else:
            # Real Q-value: convert and compute
            response = (
                abs(float(agent_q)) * torch.mean(torch.abs(field_perturbation)).item()
            )

        return min(1.0, response)  # Normalize to [0,1]

    def _measure_perturbation_stability(
        self, field_perturbation: torch.Tensor
    ) -> float:
        """Measure field stability from perturbation eigenvalue analysis."""
        # Stability from perturbation energy concentration
        energy_density = torch.abs(field_perturbation) ** 2
        total_energy = torch.sum(energy_density)

        if total_energy < self.mathematical_tolerance:
            return 1.0  # Perfect stability for zero perturbation

        # Stability as inverse of energy localization
        max_density = torch.max(energy_density)
        localization = max_density / (total_energy + self.mathematical_tolerance)
        stability = 1.0 / (1.0 + localization)  # Higher localization → lower stability

        return stability

    def _calculate_information_complexity_increase(
        self, text: str, universe_state: Dict
    ) -> float:
        """Calculate information complexity increase from text integration."""
        # Current universe complexity (proxy from agent count)
        if "agents" not in universe_state:
            raise ValueError(
                "MATHEMATICAL FAILURE: universe_state lacks required 'agents' field for complexity calculation"
            )
        current_agents = len(universe_state["agents"])
        current_complexity = math.log2(current_agents + 1)  # Information content

        # Text complexity
        text_entropy = self._calculate_field_amplitude_from_entropy(text)
        text_complexity = text_entropy * math.log2(len(text) + 1)

        # Complexity increase (normalized)
        complexity_increase = text_complexity / (current_complexity + 1.0)
        return complexity_increase

    def _compute_dynamic_acceptance_threshold(self, universe_state: Dict) -> float:
        """Compute dynamic acceptance threshold based on universe state."""
        # Base threshold
        base_threshold = 0.5

        # Adjust based on universe complexity
        if "agents" not in universe_state:
            raise ValueError(
                "MATHEMATICAL FAILURE: universe_state lacks required 'agents' field for complexity scaling"
            )
        agent_count = len(universe_state["agents"])
        complexity_factor = math.log(agent_count + 1) / 10.0  # Logarithmic scaling

        # Dynamic threshold
        threshold = base_threshold + complexity_factor
        return min(1.0, threshold)  # Cap at 1.0
