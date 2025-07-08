"""
Selection Pressure Mechanisms - Evolutionary Field Dynamics

MATHEMATICAL FOUNDATION: Replicator equations extended to Q(τ,C,s) field space
for modeling evolutionary pressures in conceptual charge systems.

Core Equations:
    ∂pᵢ/∂t = pᵢ(fᵢ(p) - ⟨f(p)⟩)                    # Replicator dynamics
    fᵢ(p) = ∫ Q(τ,C,s)ᵢ · A(s,C,τ) · ρ(s) ds       # Fitness function
    ⟨f(p)⟩ = Σⱼ pⱼ fⱼ(p)                             # Average fitness
    
Selection Pressure Types:
    - Stability Selection: P_s ∝ -∇²E[φ]           # Energy minimization pressure
    - Diversity Selection: P_d ∝ ∇H[p]             # Entropy maximization pressure  
    - Coherence Selection: P_c ∝ |⟨Ψ₁*Ψ₂⟩|        # Phase alignment pressure
    - Complexity Selection: P_k ∝ ∂C/∂t            # Information growth pressure

Mathematical Libraries:
    - JAX: Automatic differentiation for gradient-based selection
    - SciPy: ODE solvers for replicator equation integration
    - NumPy/Torch: Population dynamics and fitness landscapes
    - Numba: JIT compilation for population evolution loops

Design Principle: Pure mathematical evolutionary dynamics without approximation.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# JAX for automatic differentiation and optimization
import jax
import jax.numpy as jnp
# Numba for high-performance population dynamics
import numba as nb
# Mathematical computation libraries
import numpy as np
import torch
import torch.nn.functional as F
from jax import grad, hessian, jacfwd, jacrev, jit, vmap
from jax.scipy import optimize as jax_optimize
from numba import jit as nb_jit
from numba import prange
# SciPy for differential equations and optimization
from scipy import integrate, linalg, optimize, special
from scipy.integrate import odeint, quad_vec, solve_ivp
from scipy.linalg import eigh, solve_continuous_lyapunov, svd
from scipy.optimize import basinhopping, differential_evolution, minimize
from scipy.special import beta, gamma, logsumexp, polygamma, softmax
from torch.distributions import Beta, Categorical, Normal

# Mathematical Constants for Evolutionary Dynamics
SELECTION_STRENGTH: float = 2.718281828  # e Natural selection intensity
MUTATION_RATE: float = 0.01  # Base mutation rate per generation
GENETIC_DRIFT_COEFFICIENT: float = 1.0 / math.sqrt(2 * math.pi)  # √(2π)⁻¹
CARRYING_CAPACITY: float = 1000.0  # Population carrying capacity
EXTINCTION_THRESHOLD: float = 1e-6  # Population extinction cutoff

# Evolutionary Time Scales
SELECTION_TIME_SCALE: float = 1.0  # τ_s = 1/s selection time
MUTATION_TIME_SCALE: float = 100.0  # τ_μ = 1/μ mutation time
DRIFT_TIME_SCALE: float = 1000.0  # τ_d = N effective population time
INVASION_TIME_SCALE: float = 10.0  # τ_inv invasion time scale

# Information Theory Constants
SHANNON_ENTROPY_BASE: float = 2.0  # Base for information measurement
FISHER_INFORMATION_REGULARIZATION: float = 1e-8  # Numerical stability
KULLBACK_LEIBLER_THRESHOLD: float = 1e-10  # KL divergence precision

# Mathematical Precision for Evolution (legacy constants)
POPULATION_NUMERICAL_PRECISION: float = 1e-12  # Population frequency precision (legacy)
EVOLUTIONARY_CONVERGENCE_THRESHOLD: float = 1e-10  # Equilibrium detection (legacy)
REPLICATOR_INTEGRATION_TOLERANCE: float = 1e-12  # ODE solver tolerance (legacy)

# Device-aware precision functions
def get_population_precision() -> float:
    """Get device-appropriate population precision."""
    try:
        from .complexity_gradient import get_dtype_manager
        return get_dtype_manager().config.numerical_tolerance
    except ImportError:
        return 1e-6 if torch.backends.mps.is_available() else 1e-12

def get_evolutionary_convergence_threshold() -> float:
    """Get device-appropriate convergence threshold."""
    try:
        from .complexity_gradient import get_dtype_manager
        precision = get_dtype_manager().config.numerical_tolerance
        return precision * 100  # More lenient for evolutionary convergence
    except ImportError:
        return 1e-5 if torch.backends.mps.is_available() else 1e-10


class SelectionType(Enum):
    """
    Types of evolutionary selection pressures.

    Mathematical Foundation:
        Each selection type corresponds to different fitness function curvatures:

        Stabilizing Selection:
        $$f(x) = f_0 - \frac{s}{2}(x - \theta)^2$$
        where $s > 0$ is selection strength, $\theta$ is optimal phenotype

        Directional Selection:
        $$f(x) = f_0 + sx$$
        where $s \neq 0$ is directional selection coefficient

        Disruptive Selection:
        $$f(x) = f_0 + \frac{s}{2}(x - \theta)^2$$
        where $s > 0$ increases variance

        Balancing Selection:
        $$f_i(p) = f_0 + s_i \left(1 - \frac{p_i}{p_0}\right)$$
        where $p_i$ is frequency of type $i$

        Neutral Selection:
        $$f_i = f_0 \quad \forall i$$
        where all fitness values are equal
    """

    STABILIZING = "stabilizing"  # Variance-reducing selection
    DIRECTIONAL = "directional"  # Mean-shifting selection
    DISRUPTIVE = "disruptive"  # Variance-increasing selection
    BALANCING = "balancing"  # Frequency-dependent selection
    NEUTRAL = "neutral"  # No selective advantage


class EvolutionaryRegime(Enum):
    """
    Evolutionary dynamics regimes based on relative parameter strengths.

    Mathematical Foundation:
        Regimes determined by dimensionless parameter ratios:

        Mutation-Selection Balance:
        $$\frac{\mu}{s} \sim 1 \quad \text{where} \quad \Delta p = \frac{\mu}{s}$$

        Drift-Selection Balance:
        $$N_e s \sim 1 \quad \text{where} \quad \pi = \frac{1 - e^{-2N_e s}}{1 - e^{-4N_e s}}$$

        Neutral Drift:
        $$s = 0 \quad \text{and} \quad \frac{dp_i}{dt} = \sum_j (M_{ji} p_j - M_{ij} p_i)$$

        Deterministic Limit:
        $$N_e \to \infty \quad \text{and} \quad \frac{dp_i}{dt} = p_i[f_i(p) - \langle f \rangle]$$

        Quasi-Neutral:
        $$N_e s \ll 1 \quad \text{and} \quad \pi \approx \frac{2s}{1-e^{-4N_e\mu}}$$
    """

    MUTATION_SELECTION = "mutation_selection"  # μ-s balance
    DRIFT_SELECTION = "drift_selection"  # N-s balance
    NEUTRAL_DRIFT = "neutral_drift"  # Pure drift dynamics
    DETERMINISTIC = "deterministic"  # Large population limit
    QUASI_NEUTRAL = "quasi_neutral"  # Ns ≪ 1 regime


@dataclass
class PopulationState:
    """Population state in evolutionary dynamics."""

    frequencies: torch.Tensor  # pᵢ frequency distribution
    fitness_values: torch.Tensor  # fᵢ individual fitness values
    population_size: float  # N effective population size
    generation_time: float  # τ generation time scale
    mutation_matrix: torch.Tensor  # Mᵢⱼ mutation transition matrix
    selection_coefficients: torch.Tensor  # sᵢ selection coefficients
    genetic_variance: float  # σ²_G genetic variance
    phenotypic_variance: float  # σ²_P phenotypic variance

    def __post_init__(self):
        """
        Validate population state consistency.

        Mathematical Constraints:
            Population frequencies must satisfy:
            $$\sum_{i=1}^n p_i = 1, \quad p_i \geq 0 \quad \forall i$$

            Mutation matrix must be stochastic:
            $$\sum_{j=1}^n M_{ij} = 1 \quad \forall i$$

            Population size and generation time positivity:
            $$N > 0, \quad \tau > 0$$

            Variance relationship:
            $$\sigma^2_P = \sigma^2_G + \sigma^2_E \quad \text{(total variance decomposition)}$$
        """
        if not torch.is_tensor(self.frequencies):
            raise TypeError("frequencies must be torch.Tensor")
        if not torch.allclose(torch.sum(self.frequencies), torch.tensor(1.0)):
            raise ValueError(
                f"Frequencies must sum to 1: {torch.sum(self.frequencies)}"
            )
        if torch.any(self.frequencies < 0):
            raise ValueError("Frequencies must be non-negative")
        if self.population_size <= 0:
            raise ValueError(
                f"Population size must be positive: {self.population_size}"
            )
        if self.generation_time <= 0:
            raise ValueError(
                f"Generation time must be positive: {self.generation_time}"
            )


@dataclass
class SelectionPressure:
    """Selection pressure analysis result."""

    pressure_magnitude: float  # |∇f| selection pressure strength
    pressure_direction: torch.Tensor  # ∇f/|∇f| selection gradient direction
    selection_type: SelectionType  # Classification of selection
    target_phenotype: torch.Tensor  # Optimal phenotype under selection
    selection_intensity: float  # s = (f_max - f_min)/f_mean intensity
    heritability: float  # h² = σ²_G/σ²_P heritability
    response_prediction: torch.Tensor  # Δz⃗ = h²β⃗ breeder's equation
    evolutionary_time_scale: float  # τ_evo = 1/(h²s) response time

    def __post_init__(self):
        """
        Validate selection pressure mathematical properties.

        Mathematical Constraints:
            Selection pressure magnitude:
            $$|\nabla f| \geq 0$$

            Heritability bounds:
            $$0 \leq h^2 = \frac{\sigma^2_G}{\sigma^2_P} \leq 1$$

            Evolutionary time scale:
            $$\tau_{evo} = \frac{1}{h^2 s} > 0$$

            Response to selection (Breeder's equation):
            $$\Delta \bar{z} = h^2 S = h^2 i \sigma_P$$
            where $S$ is selection differential, $i$ is selection intensity
        """
        if self.pressure_magnitude < 0:
            raise ValueError(f"Negative pressure magnitude: {self.pressure_magnitude}")
        if not (0 <= self.heritability <= 1):
            raise ValueError(f"Heritability out of range [0,1]: {self.heritability}")
        if self.evolutionary_time_scale <= 0:
            raise ValueError(f"Non-positive time scale: {self.evolutionary_time_scale}")


@dataclass
class FitnessLandscape:
    """Fitness landscape characterization."""

    fitness_function: Callable  # f(x): ℝⁿ → ℝ fitness function
    local_optima: List[torch.Tensor]  # {x*ᵢ} local fitness maxima
    global_optimum: torch.Tensor  # x* global fitness maximum
    fitness_variance: float  # Var[f] fitness variance
    ruggedness_measure: float  # R roughness/epistasis measure
    neutrality_fraction: float  # ν fraction of neutral mutations
    accessibility_graph: torch.Tensor  # Aᵢⱼ mutational accessibility
    escape_barriers: torch.Tensor  # ΔE escape energy barriers

    def __post_init__(self):
        """
        Validate fitness landscape properties.

        Mathematical Constraints:
            Fitness variance non-negativity:
            $$\text{Var}[f] = \langle f^2 \rangle - \langle f \rangle^2 \geq 0$$

            Neutrality fraction bounds:
            $$0 \leq \nu = \frac{|\{i : |f_i - f_j| < \epsilon\}|}{|\{\text{all mutations}\}|} \leq 1$$

            Ruggedness measure (epistasis):
            $$R = \frac{1}{N} \sum_{i \sim j} |f_i - f_j|^2$$
            where $i \sim j$ denotes adjacent genotypes

            Escape barriers:
            $$\Delta E_{i \to j} = \max_{k \in \text{path}(i,j)} f_k - f_i$$
        """
        if not (0 <= self.neutrality_fraction <= 1):
            raise ValueError(
                f"Neutrality fraction out of range: {self.neutrality_fraction}"
            )
        if self.fitness_variance < 0:
            raise ValueError(f"Negative fitness variance: {self.fitness_variance}")


@dataclass
class EvolutionaryDynamics:
    """Evolutionary trajectory analysis."""

    time_points: torch.Tensor  # t time sampling points
    population_trajectory: torch.Tensor  # p(t) population frequency evolution
    fitness_trajectory: torch.Tensor  # f(t) mean fitness evolution
    diversity_trajectory: torch.Tensor  # H(t) genetic diversity evolution
    fixation_probabilities: torch.Tensor  # π_i fixation probability for each type
    invasion_fitness: torch.Tensor  # r_i invasion fitness (rare type)
    coalescence_times: torch.Tensor  # T_c coalescence time distribution
    effective_population_size: torch.Tensor  # N_e(t) effective size evolution

    def __post_init__(self):
        """
        Validate evolutionary dynamics consistency.

        Mathematical Constraints:
            Fixation probability bounds:
            $$0 \leq \pi_i \leq 1 \quad \forall i$$

            Probability conservation:
            $$\sum_{i=1}^n \pi_i = 1$$

            Invasion fitness (rare mutant):
            $$r_i = f_i(p^*) - \langle f(p^*) \rangle$$
            where $p^*$ is resident equilibrium

            Coalescence time expectation:
            $$E[T_c] = 2N_e \left(1 - \frac{1}{n}\right)$$
            for neutral alleles
        """
        if len(self.time_points) != self.population_trajectory.shape[-1]:
            raise ValueError("Time points and trajectory length mismatch")
        if torch.any(self.fixation_probabilities < 0) or torch.any(
            self.fixation_probabilities > 1
        ):
            raise ValueError("Fixation probabilities must be in [0,1]")


@dataclass
class InformationFlow:
    """Information-theoretic evolution analysis."""

    shannon_entropy: float  # H = -Σᵢ pᵢ log pᵢ population entropy
    fisher_information: float  # I = E[(∂ log L/∂θ)²] Fisher information
    kullback_leibler_divergence: float  # D_KL = Σᵢ pᵢ log(pᵢ/qᵢ) KL divergence
    mutual_information: float  # I(X;Y) = H(X) - H(X|Y) mutual information
    information_geometry_curvature: float  # κ Riemannian curvature of model manifold
    model_complexity: float  # K Kolmogorov complexity estimate
    minimum_description_length: float  # MDL = -log L + (k/2)log n model selection
    akaike_information_criterion: float  # AIC = -2 log L + 2k information criterion

    def __post_init__(self):
        """
        Validate information measures.

        Mathematical Constraints:
            Shannon entropy non-negativity:
            $$H = -\sum_{i=1}^n p_i \log p_i \geq 0$$

            Fisher information positivity:
            $$I_{ij} = E\left[\frac{\partial^2 (-\log L)}{\partial \theta_i \partial \theta_j}\right] \geq 0$$
            (positive semi-definite matrix)

            KL divergence properties:
            $$D_{KL}(P||Q) = \sum_i p_i \log \frac{p_i}{q_i} \geq 0$$
            with equality iff $P = Q$

            Information geometry curvature:
            $$\kappa = \text{Riemann curvature of statistical manifold}$$

            Model selection criteria:
            $$\text{AIC} = -2\log L + 2k$$
            $$\text{MDL} = -\log L + \frac{k}{2}\log n$$
        """
        if self.shannon_entropy < 0:
            raise ValueError(f"Negative entropy: {self.shannon_entropy}")
        if self.fisher_information < 0:
            raise ValueError(f"Negative Fisher information: {self.fisher_information}")


# Mathematical utility functions for evolutionary dynamics
@nb_jit(nopython=True, cache=True, fastmath=False)
def _jit_replicator_dynamics(
    frequencies: np.ndarray, fitness_matrix: np.ndarray, dt: float
) -> np.ndarray:
    """
    JIT-compiled replicator equation integration.

    Mathematical Foundation:
        Replicator Dynamics:
        $$\frac{dp_i}{dt} = p_i [f_i(p) - \langle f(p) \rangle]$$

        where:
        $$f_i(p) = \sum_{j=1}^n A_{ij} p_j \quad \text{(fitness function)}$$
        $$\langle f(p) \rangle = \sum_{i=1}^n p_i f_i(p) \quad \text{(mean fitness)}$$

        Frequency constraint maintained:
        $$\sum_{i=1}^n p_i = 1 \quad \text{and} \quad \frac{d}{dt}\sum_{i=1}^n p_i = 0$$

        Euler integration:
        $$p_i(t + \Delta t) = p_i(t) + \Delta t \cdot p_i(t)[f_i(p(t)) - \langle f(p(t)) \rangle]$$

        Simplex projection:
        $$p_i^{new} = \max(0, p_i^{temp}) / \sum_j \max(0, p_j^{temp})$$

    Args:
        frequencies: Population frequency vector $p \in \Delta^{n-1}$
        fitness_matrix: Payoff matrix $A \in \mathbb{R}^{n \times n}$
        dt: Integration time step $\Delta t > 0$

    Returns:
        Updated frequency vector after one integration step
    """
    n_types = len(frequencies)

    # Compute fitness values fᵢ = Σⱼ Aᵢⱼ pⱼ
    fitness_values = np.zeros(n_types)
    for i in range(n_types):
        for j in range(n_types):
            fitness_values[i] += fitness_matrix[i, j] * frequencies[j]

    # Mean fitness ⟨f⟩ = Σᵢ pᵢ fᵢ
    mean_fitness = 0.0
    for i in range(n_types):
        mean_fitness += frequencies[i] * fitness_values[i]

    # Replicator dynamics dpᵢ/dt = pᵢ(fᵢ - ⟨f⟩)
    derivatives = np.zeros(n_types)
    for i in range(n_types):
        derivatives[i] = frequencies[i] * (fitness_values[i] - mean_fitness)

    # Euler integration step
    new_frequencies = frequencies + dt * derivatives

    # Ensure non-negativity and normalization
    new_frequencies = np.maximum(new_frequencies, 0.0)
    total = np.sum(new_frequencies)
    if total > 0:
        new_frequencies = new_frequencies / total

    return new_frequencies


@nb_jit(nopython=True, cache=True, fastmath=False)
def _jit_wright_fisher_step(
    frequencies: np.ndarray,
    fitness_values: np.ndarray,
    population_size: float,
    mutation_matrix: np.ndarray,
) -> np.ndarray:
    """
    JIT-compiled Wright-Fisher population genetics step.

    Mathematical Foundation:
        Wright-Fisher Model with Selection and Mutation:

        Selection Phase:
        $$P(\text{type } i \text{ chosen}) = \frac{p_i f_i}{\sum_j p_j f_j}$$

        Multinomial Sampling:
        $$\mathbf{n} \sim \text{Multinomial}\left(N, \left(\frac{p_1 f_1}{\bar{f}}, \ldots, \frac{p_n f_n}{\bar{f}}\right)\right)$$

        where $\bar{f} = \sum_i p_i f_i$ is mean fitness

        Poisson Approximation (large $N$):
        $$n_i \sim \text{Poisson}(N p_i f_i / \bar{f})$$

        Normal Approximation (large $\lambda$):
        $$n_i \sim \mathcal{N}(\lambda_i, \lambda_i) \quad \text{where } \lambda_i = N p_i f_i / \bar{f}$$

        Mutation Phase:
        $$n_j^{\text{mut}} = \sum_{i=1}^n n_i M_{ij}$$

        where $M_{ij}$ is mutation probability from type $i$ to type $j$

        New Frequencies:
        $$p_i^{new} = \frac{n_i^{\text{mut}}}{N}$$

    Args:
        frequencies: Current population frequencies $p \in \Delta^{n-1}$
        fitness_values: Absolute fitness values $f_i > 0$
        population_size: Effective population size $N_e$
        mutation_matrix: Mutation transition matrix $M$ (row stochastic)

    Returns:
        New population frequencies after one generation
    """
    n_types = len(frequencies)

    # Selection: weighted sampling probabilities
    weighted_frequencies = frequencies * fitness_values
    total_weight = np.sum(weighted_frequencies)
    if total_weight > 0:
        selection_probs = weighted_frequencies / total_weight
    else:
        selection_probs = frequencies

    # Deterministic expectation value calculation (no random sampling)
    offspring_counts = np.zeros(n_types)
    for i in range(n_types):
        # Deterministic expected offspring count: E[nᵢ] = N * pᵢ
        lambda_i = population_size * selection_probs[i]
        if lambda_i > 0:
            # Use exact expectation value (deterministic)
            offspring_counts[i] = lambda_i

    # Renormalize to population size
    total_offspring = np.sum(offspring_counts)
    if total_offspring > 0:
        offspring_counts = offspring_counts * (population_size / total_offspring)

    # Mutation step: apply mutation matrix
    mutated_counts = np.zeros(n_types)
    for i in range(n_types):
        for j in range(n_types):
            mutated_counts[j] += offspring_counts[i] * mutation_matrix[i, j]

    # Convert back to frequencies
    new_frequencies = (
        mutated_counts / population_size if population_size > 0 else frequencies
    )

    return new_frequencies


def shannon_entropy(
    frequencies: torch.Tensor, base: float = SHANNON_ENTROPY_BASE
) -> float:
    """
    Compute Shannon entropy of frequency distribution.

    Mathematical Foundation:
        Shannon Entropy:
        $$H(p) = -\sum_{i=1}^n p_i \log_b(p_i)$$

        Properties:
        - $H(p) \geq 0$ with equality iff distribution is degenerate
        - $H(p) \leq \log_b(n)$ with equality iff $p_i = 1/n \forall i$ (uniform)
        - $H(p)$ is concave in $p$
        - $H(p)$ is continuous in $p$

        Information Content:
        $$I_i = -\log_b(p_i) \quad \text{(surprisal of event } i\text{)}$$

        Expected Information:
        $$H(p) = E[I] = \sum_{i=1}^n p_i I_i$$

        Relation to Genetic Diversity:
        $$H_e = -\sum_{i=1}^k p_i \ln p_i \quad \text{(expected heterozygosity)}$$

        Base Conversion:
        $$H_b(p) = \frac{H_e(p)}{\ln b} = H_e(p) \log_e(b)$$

    Args:
        frequencies: Probability distribution $p \in \Delta^{n-1}$
        base: Logarithm base for entropy units (2=bits, e=nats, 10=dits)

    Returns:
        Shannon entropy $H(p)$ in specified base units
    """
    # Handle zero frequencies
    nonzero_mask = frequencies > POPULATION_NUMERICAL_PRECISION
    if not torch.any(nonzero_mask):
        return 0.0

    log_probs = torch.log(frequencies[nonzero_mask]) / math.log(base)
    entropy = -torch.sum(frequencies[nonzero_mask] * log_probs)

    return entropy.item()


def fisher_information_matrix(
    log_likelihood_func: Callable, parameters: torch.Tensor
) -> torch.Tensor:
    """
    Compute Fisher information matrix using JAX autodiff.

    Mathematical Foundation:
        Fisher Information Matrix:
        $$I_{ij}(\theta) = E\left[\frac{\partial^2 (-\ell(\theta))}{\partial \theta_i \partial \theta_j}\right]$$

        where $\ell(\theta) = \log L(\theta|\mathbf{x})$ is log-likelihood

        Alternative Form (Score Function):
        $$I_{ij}(\theta) = E\left[\frac{\partial \ell(\theta)}{\partial \theta_i} \frac{\partial \ell(\theta)}{\partial \theta_j}\right]$$

        Properties:
        - $I(\theta) \succeq 0$ (positive semi-definite)
        - $I(\theta)$ is symmetric: $I_{ij} = I_{ji}$
        - Related to Cramér-Rao bound: $\text{Var}[\hat{\theta}] \succeq I(\theta)^{-1}$
        - Reparameterization: $I_{\phi}(\phi) = J^T I_{\theta}(\theta) J$ where $J = \partial\theta/\partial\phi$

        Information Geometry:
        $$ds^2 = \sum_{i,j} I_{ij}(\theta) d\theta_i d\theta_j \quad \text{(Fisher-Rao metric)}$$

        Asymptotic Normality:
        $$\sqrt{n}(\hat{\theta}_n - \theta_0) \rightarrow^{d} \mathcal{N}(0, I(\theta_0)^{-1})$$

    Args:
        log_likelihood_func: Log-likelihood function $\ell: \Theta \to \mathbb{R}$
        parameters: Parameter vector $\theta \in \Theta$

    Returns:
        Fisher information matrix $I(\theta) \in \mathbb{R}^{p \times p}$
    """
    # Convert to JAX arrays
    params_jax = jnp.array(parameters.detach().cpu().numpy())

    # Compute Hessian of negative log-likelihood
    hessian_func = hessian(log_likelihood_func)
    fisher_matrix_jax = -hessian_func(params_jax)

    # Convert back to torch
    return torch.from_numpy(np.array(fisher_matrix_jax))


def effective_population_size(
    variance_frequency_change: float, drift_variance: float
) -> float:
    """
    Estimate effective population size from frequency variance.

    Mathematical Foundation:
        Effective Population Size from Drift Variance:
        $$N_e = \frac{\sigma^2_{\text{drift}}}{\sigma^2_{\text{observed}}}$$

        Wright-Fisher Drift Variance:
        $$\sigma^2_{\text{drift}} = \frac{p(1-p)}{2N_e}$$

        Therefore:
        $$N_e = \frac{p(1-p)}{2\sigma^2_{\text{observed}}}$$

        Alternative Estimators:

        Temporal Method:
        $$N_e = \frac{1}{2} \cdot \frac{1}{F_t - F_{t-1}/S}$$
        where $F_t$ is inbreeding coefficient, $S$ is number of samples

        Linkage Disequilibrium Method:
        $$N_e = \frac{1}{3} \left(\frac{1}{r^2} - 1\right)$$
        where $r^2$ is linkage disequilibrium measure

        Variance Effective Size:
        $$N_e = \frac{4N_m N_f}{N_m + N_f}$$
        where $N_m$, $N_f$ are numbers of reproducing males and females

        Inbreeding Effective Size:
        $$N_e = \frac{1}{2f}$$
        where $f$ is inbreeding coefficient per generation

    Args:
        variance_frequency_change: Observed variance in allele frequency change
        drift_variance: Expected drift variance under neutrality

    Returns:
        Effective population size estimate $N_e$
    """
    if variance_frequency_change <= 0 or drift_variance <= 0:
        return float("inf")

    return drift_variance / variance_frequency_change


# Export selection pressure classes and functions
from .complexity_gradient import ComplexityGradientAnalyzer
from .construct_evolution import ConstructEvolutionEngine
from .spotlight_field import SpotlightFieldEngine

__all__ = [
    # Mathematical constants
    "SELECTION_STRENGTH",
    "MUTATION_RATE",
    "GENETIC_DRIFT_COEFFICIENT",
    "CARRYING_CAPACITY",
    "EXTINCTION_THRESHOLD",
    "SELECTION_TIME_SCALE",
    "MUTATION_TIME_SCALE",
    "DRIFT_TIME_SCALE",
    "INVASION_TIME_SCALE",
    "SHANNON_ENTROPY_BASE",
    "FISHER_INFORMATION_REGULARIZATION",
    "KULLBACK_LEIBLER_THRESHOLD",
    "POPULATION_NUMERICAL_PRECISION",
    "EVOLUTIONARY_CONVERGENCE_THRESHOLD",
    "REPLICATOR_INTEGRATION_TOLERANCE",
    # Enumerations
    "SelectionType",
    "EvolutionaryRegime",
    # Data structures
    "PopulationState",
    "SelectionPressure",
    "FitnessLandscape",
    "EvolutionaryDynamics",
    "InformationFlow",
    # Mathematical utilities
    "_jit_replicator_dynamics",
    "_jit_wright_fisher_step",
    "shannon_entropy",
    "fisher_information_matrix",
    "effective_population_size",
    # Core evolutionary engines
    "ComplexityGradientAnalyzer",
    "SpotlightFieldEngine",
    "ConstructEvolutionEngine",
    # Additional utilities
    "evolutionary_neural_processing",
    "stochastic_population_sampling",
    "jax_evolutionary_optimization",
    "scipy_evolutionary_integration",
    "scipy_optimization_evolution",
    "scipy_special_population_genetics",
    "scipy_linalg_population_analysis",
]

# Numerical precision now handled by DataTypeManager (device-aware)
# Using deterministic population dynamics - no random seed needed

# Mathematical logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="🧬 EVOLUTION: %(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
logger.info("Selection Pressure Foundation initialized with mathematical rigor")


def evolutionary_neural_processing(frequencies: torch.Tensor) -> torch.Tensor:
    """
    Process population frequencies using torch.nn.functional operations.
    
    Mathematical Foundation:
        Neural Network Processing of Evolutionary Dynamics:
        
        Convolution (Selection Smoothing):
        $$f^{(\text{conv})}[i] = \sum_{j} k[j] \cdot p[i + j]$$
        where $k = [0.25, 0.5, 0.25]$ is smoothing kernel
        
        Sigmoid Activation (Selection Boundaries):
        $$\sigma(x) = \frac{1}{1 + e^{-\beta x}}$$
        where $\beta = 10$ controls selection sharpness
        
        Max Pooling (Trait Grouping):
        $$p^{(\text{pool})}[i] = \max_{j \in \text{window}} p^{(\text{conv})}[j]$$
        
        Dropout (Genetic Drift Simulation):
        $$p^{(\text{drift})}[i] = \begin{cases}
        p^{(\text{pool})}[i] / (1-\mu) & \text{with prob } 1-\mu \\
        0 & \text{with prob } \mu
        \end{cases}$$
        where $\mu$ is mutation/drift rate
        
        Layer Normalization (Frequency Constraint):
        $$p^{(\text{norm})}[i] = \frac{p^{(\text{drift})}[i] - \mu_{\text{layer}}}{\sigma_{\text{layer}}}$$
        
        Final Projection to Simplex:
        $$p^{(\text{final})} = \text{softmax}(p^{(\text{norm})})$$
    
    Args:
        frequencies: Population frequency vector $p \in \Delta^{n-1}$
        
    Returns:
        Processed frequencies after neural evolutionary operations
    """
    # Prepare for neural operations
    freq_4d = frequencies.view(1, 1, -1, 1)  # [batch, channels, height, width]

    # Selection smoothing convolution
    selection_kernel = torch.tensor([[[[0.25, 0.5, 0.25]]]], dtype=frequencies.dtype)
    smoothed = F.conv2d(freq_4d, selection_kernel, padding=(1, 0))

    # Activation for selection boundaries
    selected = F.sigmoid(smoothed * 10)  # Sigmoid selection

    # Pooling for trait grouping
    grouped = F.max_pool2d(selected, kernel_size=(2, 1), stride=1, padding=(1, 0))

    # Dropout for genetic drift simulation
    with_drift = F.dropout(grouped, p=MUTATION_RATE, training=True)

    # Layer norm for frequency normalization
    normalized = F.layer_norm(with_drift, with_drift.shape[2:])

    return normalized.view(-1)


def stochastic_population_sampling(
    population_state: PopulationState,
) -> Dict[str, torch.Tensor]:
    """
    Stochastic population sampling using torch.distributions.

    Mathematical Foundation:
        Population Genetic Sampling Distributions:

        Categorical Distribution (Discrete Traits):
        $$P(X = i) = p_i \quad \sum_{i=1}^n p_i = 1$$

        Sampling $N$ individuals:
        $$\mathbf{n} \sim \text{Multinomial}(N, \mathbf{p})$$

        Normal Distribution (Continuous Traits):
        $$z \sim \mathcal{N}(\mu_z, \sigma^2_P)$$
        where $\mu_z = \sum_i p_i z_i$ is population mean trait value

        Beta Distribution (Allele Frequencies):
        $$p \sim \text{Beta}(\alpha, \beta)$$
        where $\alpha = N p_0 + 1$, $\beta = N(1-p_0) + 1$

        Dirichlet Distribution (Multiple Alleles):
        $$\mathbf{p} \sim \text{Dirichlet}(\alpha_1, \ldots, \alpha_k)$$
        where $\alpha_i = N p_i + 1$

        Information Measures:
        - Categorical Entropy: $H = -\sum_i p_i \log p_i$
        - Normal Variance: $\text{Var}[z] = \sigma^2_P$
        - Beta Mean: $E[p] = \frac{\alpha}{\alpha + \beta}$
        - Beta Variance: $\text{Var}[p] = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$

    Args:
        population_state: Current population state with frequencies and parameters

    Returns:
        Dictionary of sampled values and statistical measures
    """
    # Categorical distribution for discrete trait sampling
    trait_distribution = Categorical(probs=population_state.frequencies)
    trait_samples = trait_distribution.sample((int(population_state.population_size),))

    # Normal distribution for continuous trait variation
    mean_trait = torch.sum(
        population_state.frequencies * torch.arange(len(population_state.frequencies))
    )
    trait_variance = population_state.phenotypic_variance
    continuous_traits = Normal(mean_trait, torch.sqrt(trait_variance))
    continuous_samples = continuous_traits.sample((100,))

    # Beta distribution for allele frequency modeling
    alpha = population_state.population_size * population_state.frequencies[0]
    beta_param = population_state.population_size * (
        1 - population_state.frequencies[0]
    )
    allele_freq_dist = Beta(alpha + 1, beta_param + 1)
    allele_freq_samples = allele_freq_dist.sample((100,))

    return {
        "trait_samples": trait_samples,
        "continuous_trait_samples": continuous_samples,
        "allele_frequency_samples": allele_freq_samples,
        "trait_entropy": trait_distribution.entropy(),
        "continuous_variance": continuous_traits.variance,
        "allele_mean": allele_freq_dist.mean,
    }


def jax_evolutionary_optimization(fitness_landscape: torch.Tensor) -> Dict[str, Any]:
    """
    Evolutionary optimization using JAX operations.

    Mathematical Foundation:
        JAX-Based Evolutionary Optimization:

        Fitness Functional:
        $$F(\mathbf{x}) = \sum_{i=1}^n x_i \cdot L_i$$
        where $L_i$ is fitness landscape value at position $i$

        Selection Gradient (Automatic Differentiation):
        $$\nabla F = \frac{\partial F}{\partial \mathbf{x}} = \mathbf{L}$$

        Vectorized Fitness Evaluation:
        $$\mathbf{F} = \text{vmap}(f)(\mathbf{X}) = [f(\mathbf{x}_1), \ldots, f(\mathbf{x}_n)]$$

        Forward-Mode Differentiation:
        $$J_{\text{fwd}} = \frac{\partial F}{\partial \mathbf{x}} \bigg|_{\mathbf{x}=\mathbf{x}_0}$$

        Reverse-Mode Differentiation:
        $$J_{\text{rev}} = \nabla_{\mathbf{x}} F(\mathbf{x}) \bigg|_{\mathbf{x}=\mathbf{x}_0}$$

        Hessian Matrix (Second-Order Optimization):
        $$H_{ij} = \frac{\partial^2 F}{\partial x_i \partial x_j}$$

        JIT Compilation for Performance:
        $$\text{compile}(f) : \mathbb{R}^n \to \mathbb{R}^m \text{ (optimized)}$$

    Args:
        fitness_landscape: Fitness values across trait space

    Returns:
        Dictionary containing gradients, Jacobians, and vectorized evaluations
    """
    landscape_jax = jnp.array(fitness_landscape.detach().cpu().numpy())

    # Define fitness functional
    @jit
    def fitness_functional(traits):
        return jnp.sum(traits * landscape_jax)

    # JAX autodiff for selection gradients
    fitness_grad = grad(fitness_functional)
    selection_gradient = fitness_grad(jnp.ones_like(landscape_jax))

    # JAX vectorized operations
    vmap_fitness = vmap(lambda x: x * landscape_jax)
    vectorized_selection = vmap_fitness(jnp.arange(len(landscape_jax)))

    # Forward and reverse mode differentiation
    jacfwd_fitness = jacfwd(fitness_functional)
    jacrev_fitness = jacrev(fitness_functional)

    forward_jacobian = jacfwd_fitness(jnp.ones_like(landscape_jax))
    reverse_jacobian = jacrev_fitness(jnp.ones_like(landscape_jax))

    return {
        "selection_gradient": torch.from_numpy(np.array(selection_gradient)),
        "vectorized_selection": torch.from_numpy(np.array(vectorized_selection)),
        "forward_jacobian": torch.from_numpy(np.array(forward_jacobian)),
        "reverse_jacobian": torch.from_numpy(np.array(reverse_jacobian)),
    }


def scipy_evolutionary_integration(
    initial_state: PopulationState, time_span: Tuple[float, float]
) -> EvolutionaryDynamics:
    """
    Integrate evolutionary dynamics using scipy methods.

    Mathematical Foundation:
        Ordinary Differential Equation Integration:

        Replicator Equation System:
        $$\frac{dp_i}{dt} = p_i [f_i(p) - \langle f(p) \rangle]$$

        where $f_i(p) = \sum_j A_{ij} p_j$ and $\langle f(p) \rangle = \sum_i p_i f_i(p)$

        Runge-Kutta 4th Order Method:
        $$p_{n+1} = p_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

        where:
        $$k_1 = f(t_n, p_n)$$
        $$k_2 = f(t_n + h/2, p_n + hk_1/2)$$
        $$k_3 = f(t_n + h/2, p_n + hk_2/2)$$
        $$k_4 = f(t_n + h, p_n + hk_3)$$

        Adaptive Step Size Control:
        $$\epsilon_{\text{local}} \leq \text{rtol} \cdot |p| + \text{atol}$$

        Vectorized Quadrature Integration:
        $$I = \int_a^b \mathbf{f}(t) dt \approx \sum_{i=1}^n w_i \mathbf{f}(t_i)$$

        Fitness Trajectory:
        $$\bar{f}(t) = \sum_{i=1}^n p_i(t) f_i$$

        Diversity Trajectory:
        $$H(t) = -\sum_{i=1}^n p_i(t) \log p_i(t)$$

        Fixation Probability (Kimura's Formula):
        $$\pi_i = \frac{1 - e^{-2N_e s_i}}{1 - e^{-2N_e s_i n_i}}$$

    Args:
        initial_state: Initial population state at t=0
        time_span: Integration interval [t_start, t_end]

    Returns:
        Complete evolutionary trajectory with fitness and diversity dynamics
    """
    # Convert to numpy
    initial_freqs = initial_state.frequencies.detach().cpu().numpy()
    fitness_values = initial_state.fitness_values.detach().cpu().numpy()

    # Define replicator ODE system
    def replicator_ode(t, y):
        frequencies = y
        mean_fitness = np.sum(frequencies * fitness_values)
        dydt = frequencies * (fitness_values - mean_fitness)
        return dydt

    # Solve using solve_ivp
    sol = solve_ivp(
        replicator_ode,
        time_span,
        initial_freqs,
        method="RK45",
        rtol=REPLICATOR_INTEGRATION_TOLERANCE,
    )

    # Alternative solution using odeint
    time_points = np.linspace(time_span[0], time_span[1], 100)
    sol_odeint = odeint(lambda y, t: replicator_ode(t, y), initial_freqs, time_points)

    # Vectorized quadrature for fitness integrals
    def fitness_integrand(freqs):
        return freqs * fitness_values

    fitness_integrals = quad_vec(fitness_integrand, 0, 1)[0]

    # Convert results
    return EvolutionaryDynamics(
        time_points=torch.from_numpy(sol.t),
        population_trajectory=torch.from_numpy(sol.y),
        fitness_trajectory=torch.from_numpy(
            np.array(
                [np.sum(sol.y[:, i] * fitness_values) for i in range(sol.y.shape[1])]
            )
        ),
        diversity_trajectory=torch.from_numpy(
            np.array(
                [
                    shannon_entropy(torch.from_numpy(sol.y[:, i]))
                    for i in range(sol.y.shape[1])
                ]
            )
        ),
        fixation_probabilities=torch.from_numpy(sol.y[:, -1]),
        invasion_fitness=torch.from_numpy(fitness_values - np.mean(fitness_values)),
        coalescence_times=torch.ones(len(fitness_values)),
        effective_population_size=torch.tensor(
            [initial_state.population_size] * len(sol.t)
        ),
    )


def scipy_optimization_evolution(
    fitness_landscape_func: Callable, initial_params: torch.Tensor
) -> Dict[str, Union[torch.Tensor, float]]:
    """
    Evolutionary optimization using scipy.optimize methods.
    
    Mathematical Foundation:
        Multi-Method Evolutionary Optimization:
        
        Local Optimization (BFGS):
        $$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k B_k^{-1} \nabla f(\mathbf{x}_k)$$
        where $B_k$ is BFGS approximation to Hessian:
        $$B_{k+1} = B_k + \frac{\mathbf{y}_k \mathbf{y}_k^T}{\mathbf{y}_k^T \mathbf{s}_k} - \frac{B_k \mathbf{s}_k \mathbf{s}_k^T B_k}{\mathbf{s}_k^T B_k \mathbf{s}_k}$$
        
        Global Optimization (Differential Evolution):
        $$\mathbf{v}_i = \mathbf{x}_{r1} + F(\mathbf{x}_{r2} - \mathbf{x}_{r3})$$
        $$u_{ij} = \begin{cases}
        v_{ij} & \text{if } \text{rand}() \leq CR \text{ or } j = j_{\text{rand}} \\
        x_{ij} & \text{otherwise}
        \end{cases}$$
        
        Basin Hopping (Global + Local):
        $$\mathbf{x}_{\text{new}} = \begin{cases}
        \mathbf{x}_{\text{local}} & \text{if } f(\mathbf{x}_{\text{local}}) < f(\mathbf{x}_{\text{current}}) \\
        \mathbf{x}_{\text{local}} & \text{if } \exp\left(-\frac{\Delta f}{T}\right) > \text{rand}() \\
        \mathbf{x}_{\text{current}} & \text{otherwise}
        \end{cases}$$
        
        Acceptance Probability (Metropolis Criterion):
        $$P_{\text{accept}} = \min\left(1, \exp\left(-\frac{f(\mathbf{x}_{\text{new}}) - f(\mathbf{x}_{\text{old}})}{T}\right)\right)$$
        
        Convergence Criteria:
        - Gradient norm: $\|\nabla f\| < \epsilon_g$
        - Function tolerance: $|f_{k+1} - f_k| < \epsilon_f$
        - Parameter tolerance: $\|\mathbf{x}_{k+1} - \mathbf{x}_k\| < \epsilon_x$
    
    Args:
        fitness_landscape_func: Fitness function $f: \mathbb{R}^n \to \mathbb{R}$
        initial_params: Starting point $\mathbf{x}_0 \in \mathbb{R}^n$
        
    Returns:
        Dictionary with optimal solutions from different methods
    """
    initial_np = initial_params.detach().cpu().numpy()

    # Standard minimization
    result_minimize = minimize(
        lambda x: -fitness_landscape_func(x),  # Maximize fitness
        initial_np,
        method="BFGS",
    )

    # Global optimization via differential evolution
    bounds = [(x - 2, x + 2) for x in initial_np]
    result_de = differential_evolution(
        lambda x: -fitness_landscape_func(x), bounds, seed=42
    )

    # Basin hopping for rugged landscapes
    result_basin = basinhopping(
        lambda x: -fitness_landscape_func(x), initial_np, niter=100, seed=42
    )

    return {
        "optimal_traits_local": torch.from_numpy(result_minimize.x),
        "optimal_fitness_local": -result_minimize.fun,
        "optimal_traits_global": torch.from_numpy(result_de.x),
        "optimal_fitness_global": -result_de.fun,
        "optimal_traits_basin": torch.from_numpy(result_basin.x),
        "optimal_fitness_basin": -result_basin.fun,
        "convergence_local": result_minimize.success,
        "convergence_global": result_de.success,
    }


def scipy_special_population_genetics(
    frequencies: torch.Tensor, selection_coefficients: torch.Tensor
) -> Dict[str, float]:
    """
    Population genetics calculations using scipy.special functions.

    Mathematical Foundation:
        Special Functions in Population Genetics:

        Softmax Normalization:
        $$p_i = \frac{e^{s_i}}{\sum_{j=1}^n e^{s_j}} = \text{softmax}(\mathbf{s})_i$$

        LogSumExp (Numerical Stability):
        $$\log\left(\sum_{i=1}^n e^{x_i}\right) = x_{\max} + \log\left(\sum_{i=1}^n e^{x_i - x_{\max}}\right)$$

        Gamma Function in Wright's Formula:
        $$\pi = \frac{\Gamma(2N_e s)}{\Gamma(2N_e s + 1)} \cdot \frac{1 - e^{-2s}}{1 - e^{-4N_e s}}$$

        where $\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt$

        Beta Function for Allele Frequencies:
        $$P(p) = \frac{1}{B(\alpha, \beta)} p^{\alpha-1} (1-p)^{\beta-1}$$

        where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$

        Polygamma Function (Fisher Information):
        $$\psi^{(n)}(z) = \frac{d^{n+1}}{dz^{n+1}} \log \Gamma(z)$$

        Trigamma Function ($n=1$):
        $$\psi^{(1)}(z) = \sum_{k=0}^\infty \frac{1}{(z+k)^2} = \frac{d^2}{dz^2} \log \Gamma(z)$$

        Homozygosity (Inbreeding Coefficient):
        $$F = \sum_{i=1}^n p_i^2 = 1 - H_e$$
        where $H_e$ is expected heterozygosity

        Wright's F-Statistics:
        $$F_{ST} = \frac{\sigma^2_p}{p(1-p)} \quad \text{(population differentiation)}$$

    Args:
        frequencies: Allele/type frequencies $\mathbf{p} \in \Delta^{n-1}$
        selection_coefficients: Selection coefficients $\mathbf{s} \in \mathbb{R}^n$

    Returns:
        Dictionary of population genetic statistics
    """
    freqs_np = frequencies.detach().cpu().numpy()
    sel_coef_np = selection_coefficients.detach().cpu().numpy()

    # Softmax for frequency normalization
    normalized_freqs = special.softmax(freqs_np)

    # LogSumExp for numerical stability in likelihood calculations
    log_likelihood = special.logsumexp(np.log(freqs_np) + sel_coef_np)

    # Gamma function for Wright's formula
    # Γ(Ns) appears in fixation probability formulas
    Ne_s_product = CARRYING_CAPACITY * np.mean(sel_coef_np)
    if Ne_s_product > 0:
        wright_gamma = special.gamma(Ne_s_product)
    else:
        wright_gamma = 1.0

    # Beta function for allele frequency distributions
    # B(Na, N(1-a)) for binomial sampling
    alpha_allele = CARRYING_CAPACITY * freqs_np[0]
    beta_allele = CARRYING_CAPACITY * (1 - freqs_np[0])
    if alpha_allele > 0 and beta_allele > 0:
        allele_beta = special.beta(alpha_allele, beta_allele)
    else:
        allele_beta = 1.0

    # Polygamma for Fisher information
    if Ne_s_product > 1:
        fisher_info_component = special.polygamma(1, Ne_s_product)  # Trigamma
    else:
        fisher_info_component = 1.0

    return {
        "normalized_frequencies": np.sum(normalized_freqs**2),  # Homozygosity
        "log_likelihood": log_likelihood,
        "wright_gamma_factor": wright_gamma,
        "allele_beta_function": allele_beta,
        "fisher_information": fisher_info_component,
    }


def scipy_linalg_population_analysis(
    transition_matrix: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Population transition matrix analysis using scipy.linalg.

    Mathematical Foundation:
        Linear Algebra in Population Genetics:

        Eigenvalue Decomposition:
        $$\mathbf{M} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{-1}$$
        where $\mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_n)$

        Stationary Distribution:
        $$\mathbf{\pi} \mathbf{M} = \mathbf{\pi} \quad \Rightarrow \quad \mathbf{\pi} = \mathbf{q}_1$$
        where $\mathbf{q}_1$ is eigenvector for $\lambda_1 = 1$

        Convergence Rate:
        $$\|\mathbf{p}(t) - \mathbf{\pi}\| \leq C |\lambda_2|^t$$
        where $\lambda_2$ is second-largest eigenvalue

        Singular Value Decomposition:
        $$\mathbf{M} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
        where $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$

        Effective Dimensionality:
        $$d_{\text{eff}} = \frac{(\sum_i \sigma_i)^2}{\sum_i \sigma_i^2} \quad \text{(participation ratio)}$$

        Continuous-Time Lyapunov Equation:
        $$\mathbf{A} \mathbf{X} + \mathbf{X} \mathbf{A}^T = -\mathbf{Q}$$

        For variance dynamics:
        $$\frac{d\mathbf{V}}{dt} = \mathbf{A} \mathbf{V} + \mathbf{V} \mathbf{A}^T + \mathbf{Q}$$

        where $\mathbf{V}$ is covariance matrix, $\mathbf{Q}$ is noise covariance

        Spectral Radius (Stability):
        $$\rho(\mathbf{M}) = \max_i |\lambda_i| \quad \text{(largest eigenvalue magnitude)}$$

        Condition Number:
        $$\kappa(\mathbf{M}) = \frac{\sigma_{\max}}{\sigma_{\min}} \quad \text{(numerical stability)}$$

    Args:
        transition_matrix: Population transition/mutation matrix $\mathbf{M}$

    Returns:
        Dictionary with eigenvalues, eigenvectors, and matrix analysis
    """
    matrix_np = transition_matrix.detach().cpu().numpy()

    # Eigenvalue decomposition for equilibrium
    eigenvals, eigenvecs = eigh(matrix_np + matrix_np.T)  # Symmetrize

    # SVD for dimensionality analysis
    U, S, Vt = svd(matrix_np)

    # Lyapunov equation for variance dynamics
    # AX + XA^T = -Q where Q is covariance source
    Q = np.eye(len(matrix_np))
    try:
        variance_matrix = solve_continuous_lyapunov(matrix_np, -Q)
    except:
        variance_matrix = np.eye(len(matrix_np))

    return {
        "eigenvalues": torch.from_numpy(eigenvals),
        "dominant_eigenvector": torch.from_numpy(eigenvecs[:, -1]),
        "singular_values": torch.from_numpy(S),
        "effective_dimensions": torch.sum(torch.from_numpy(S) > 0.01),
        "variance_predictions": torch.from_numpy(variance_matrix),
    }


# Import main classes from submodules
try:
    from .complexity_gradient import ComplexityGradientAnalyzer
except ImportError as e:
    ComplexityGradientAnalyzer = None
    logging.warning(f"ComplexityGradientAnalyzer import failed: {e}")

try:
    from .spotlight_field import AttentionField, SpotlightFieldEngine
except ImportError as e:
    SpotlightFieldEngine = None
    AttentionField = None
    logging.warning(f"SpotlightFieldEngine import failed: {e}")

try:
    from .construct_evolution import ConstructEvolutionEngine, SocialConstruct
except ImportError as e:
    ConstructEvolutionEngine = None
    SocialConstruct = None
    logging.warning(f"ConstructEvolutionEngine import failed: {e}")

# Expose main interfaces
__all__ = [
    # Enums and data structures
    "SelectionType",
    "EvolutionaryRegime",
    "PopulationState",
    "SelectionPressure",
    "FitnessLandscape",
    "EvolutionaryDynamics",
    "InformationFlow",
    # Mathematical functions
    "replicator_dynamics_step",
    "wright_fisher_step",
    "evolutionary_analysis",
    # Main engines (may be None if imports fail)
    "ComplexityGradientAnalyzer",
    "SpotlightFieldEngine",
    "ConstructEvolutionEngine",
    "AttentionField",
    "SocialConstruct",
    # Constants
    "SELECTION_STRENGTH",
    "MUTATION_RATE",
    "GENETIC_DRIFT_COEFFICIENT",
    "POPULATION_NUMERICAL_PRECISION",
    "EVOLUTIONARY_CONVERGENCE_THRESHOLD",
]
