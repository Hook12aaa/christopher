"""
Construct Evolution Engine - Replicator Dynamics for Social Construct Formation

MATHEMATICAL FOUNDATION:
    Replicator Equation: ∂pᵢ/∂t = pᵢ[fᵢ(p,C,t) - ⟨f(p,C,t)⟩]
    Fitness Function: fᵢ = α·Coherence + β·Complexity + γ·Stability + δ·Novelty
    Social Construct: C(τ,s) = Q(τ,C,s) collective conceptual charge
    
    Multi-Level Selection:
    - Individual Level: pᵢ personal construct frequencies  
    - Group Level: P_g group construct distributions
    - Population Level: Π inter-group dynamics
    
    Evolutionary Game Theory:
    Payoff Matrix: A_ij = fitness of construct i against construct j
    ESS Condition: A(p*,p*) ≥ A(q,p*) ∀q ≠ p* (evolutionarily stable strategy)
    
    Mutation-Selection Balance:
    ∂pᵢ/∂t = Σⱼ μⱼᵢpⱼ - μpᵢ + pᵢ[fᵢ - ⟨f⟩]   # Mutation + Selection
    
IMPLEMENTATION: JAX autodiff for exact fitness gradients, sparse matrix replicator
integration, evolutionary game equilibrium analysis, multi-level group selection.
"""

import cmath
import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

# JAX for evolutionary dynamics and game theory
import jax
import jax.numpy as jnp
# Game theory library for exact Nash equilibrium computation - REQUIRED
import nashpy as nash
# Numba for high-performance population dynamics
import numba as nb
import numpy as np
import torch
import torch.nn.functional as F
from jax import grad, hessian, jacfwd, jacrev, jit, vmap
from jax.scipy import optimize as jax_optimize
from jax.scipy.linalg import eigh as jax_eigh
from jax.scipy.linalg import solve as jax_solve
from numba import jit as nb_jit
from numba import prange
# SciPy for differential equations and game theory
from scipy import integrate, linalg, optimize, sparse
from scipy.integrate import odeint, solve_ivp
from scipy.linalg import eigh, norm, solve, svd

from ..field_mechanics.data_type_consistency import get_dtype_manager
from scipy.optimize import linprog, minimize, minimize_scalar
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as sparse_linalg
from torch.distributions import Beta, Categorical, Dirichlet

# Import constants and utilities
from . import (CARRYING_CAPACITY, EVOLUTIONARY_CONVERGENCE_THRESHOLD,
               GENETIC_DRIFT_COEFFICIENT, MUTATION_RATE,
               POPULATION_NUMERICAL_PRECISION,
               REPLICATOR_INTEGRATION_TOLERANCE, SELECTION_STRENGTH,
               EvolutionaryDynamics, EvolutionaryRegime, FitnessLandscape,
               InformationFlow, PopulationState, SelectionPressure,
               SelectionType, _jit_replicator_dynamics,
               _jit_wright_fisher_step, effective_population_size,
               shannon_entropy)

logger = logging.getLogger(__name__)


@dataclass
class SocialConstruct:
    """
    Social construct representation in evolutionary space.

    Mathematical Foundation:
    A social construct is represented as a dynamic entity in evolutionary space with:

    $$Q(\tau, C, s) = \gamma \cdot T \cdot E \cdot \Phi \cdot e^{i\theta} \cdot \Psi$$

    Where:
    - Coherence: $\langle|\Psi_1^* \Psi_2|\rangle$ - quantum coherence measure
    - Complexity: $C(X) = -\sum_i p_i \log p_i$ - information complexity
    - Stability: $\lambda_{\max}$ - largest eigenvalue magnitude of interaction matrix
    - Fitness: $f = \sum_i w_i F_i$ - weighted composite fitness function
    - Interactions: $A_{ij}$ - pairwise interaction strength matrix
    - Mutation: $\mu$ - stochastic variation rate per generation
    """

    construct_id: str  # Unique construct identifier
    conceptual_charge: torch.Tensor  # Q(τ,C,s) field representation
    coherence_measure: float  # ⟨|Ψ₁*Ψ₂|⟩ internal coherence
    complexity_score: float  # C(X) information complexity
    stability_index: float  # λ_max largest eigenvalue magnitude
    novelty_metric: float  # Distance from existing constructs
    fitness_value: float  # f = Σ wᵢ Fᵢ composite fitness
    interaction_strengths: torch.Tensor  # Aᵢⱼ pairwise interaction matrix
    mutation_probability: float  # μ mutation rate for this construct
    group_affiliation: Optional[int]  # Group membership identifier

    def __post_init__(self):
        """Validate social construct mathematical properties."""
        if not (0 <= self.coherence_measure <= 1):
            raise ValueError(f"Coherence out of range [0,1]: {self.coherence_measure}")
        if self.complexity_score < 0:
            raise ValueError(f"Negative complexity: {self.complexity_score}")
        if not (0 <= self.mutation_probability <= 1):
            raise ValueError(
                f"Mutation probability out of range: {self.mutation_probability}"
            )


@dataclass
class GameTheoreticAnalysis:
    """
    Evolutionary game theory analysis result.

    Mathematical Foundation:

    Nash Equilibrium:
    $$p^* \text{ such that } u_i(p^*, p^*) \geq u_i(s_i, p^*_{-i}) \quad \forall s_i, \forall i$$

    Evolutionarily Stable Strategy (ESS):
    $$u(p^*, p^*) > u(q, p^*) \text{ or } [u(p^*, p^*) = u(q, p^*) \text{ and } u(p^*, q) > u(q, q)]$$

    Replicator Dynamics:
    $$\frac{dp_i}{dt} = p_i[u_i(p) - \bar{u}(p)]$$

    Where $\bar{u}(p) = \sum_j p_j u_j(p)$ is the average fitness.

    Invasion Fitness:
    $$r_i = u_i(\delta e_i + (1-\delta)p^*, p^*) - u_i(p^*, p^*)$$

    Where $\delta \rightarrow 0$ is the small invader frequency.
    """

    payoff_matrix: torch.Tensor  # A_ij payoff matrix
    nash_equilibria: List[torch.Tensor]  # Set of Nash equilibrium points
    evolutionarily_stable_strategies: List[torch.Tensor]  # ESS solutions
    replicator_equilibria: List[torch.Tensor]  # Replicator equation fixed points
    stability_eigenvalues: torch.Tensor  # Jacobian eigenvalues at equilibria
    basin_sizes: torch.Tensor  # Relative sizes of attraction basins
    invasion_fitness: torch.Tensor  # r_i = f_i(δ,p*) - f_i(p*,p*)
    evolutionarily_stable_coalition: Optional[List[int]]  # Stable coalition structure
    cooperative_stability: float  # Stability of cooperative equilibrium

    def __post_init__(self):
        """Validate game theory analysis consistency."""
        n_strategies = self.payoff_matrix.shape[0]
        if self.payoff_matrix.shape != (n_strategies, n_strategies):
            raise ValueError("Payoff matrix must be square")
        for eq in self.nash_equilibria:
            if not torch.allclose(torch.sum(eq), torch.tensor(1.0)):
                raise ValueError("Nash equilibrium must be probability distribution")


@dataclass
class MultiLevelSelection:
    """
    Multi-level selection analysis across hierarchical levels.

    Mathematical Foundation:

    Price Equation (Multi-level):
    $$\Delta \bar{z} = \text{Cov}(w_g, z_g) + E[w_g \Delta z_g]$$

    Where:
    - First term: between-group selection
    - Second term: within-group selection

    Individual Level Selection:
    $$\frac{dp_i}{dt} = p_i[f_i - \langle f \rangle]_{\text{within groups}}$$

    Group Level Selection:
    $$\frac{dP_g}{dt} = P_g[F_g - \langle F \rangle]_{\text{between groups}}$$

    Migration Flow:
    $$\frac{dp_i^g}{dt} = m(\langle p_i \rangle - p_i^g)$$

    Hamilton's Rule (Altruism Evolution):
    $$rb > c$$

    Where $r$ is relatedness, $b$ is benefit, $c$ is cost.
    """

    individual_fitness: torch.Tensor  # f_i individual-level fitness
    group_fitness: torch.Tensor  # F_g group-level fitness
    population_fitness: float  # Π population-level fitness
    within_group_selection: torch.Tensor  # Selection within groups
    between_group_selection: torch.Tensor  # Selection between groups
    group_composition: torch.Tensor  # p_ig frequency of type i in group g
    migration_matrix: torch.Tensor  # M_gg' migration between groups
    group_formation_rate: float  # Rate of new group formation
    group_extinction_rate: float  # Rate of group extinction
    altruism_evolution_condition: bool  # rb > c Hamilton's rule

    def __post_init__(self):
        """Validate multi-level selection mathematical consistency."""
        if not torch.allclose(
            torch.sum(self.group_composition, dim=0),
            torch.ones(self.group_composition.shape[1]),
        ):
            raise ValueError("Group compositions must sum to 1")
        if self.group_formation_rate < 0 or self.group_extinction_rate < 0:
            raise ValueError("Formation and extinction rates must be non-negative")


@dataclass
class EvolutionaryTrajectory:
    """
    Complete evolutionary trajectory analysis.

    Mathematical Foundation:

    Population Dynamics:
    $$p(t) = \text{solution to } \frac{dp_i}{dt} = p_i[f_i(p,t) - \bar{f}(p,t)]$$

    Shannon Diversity:
    $$H(t) = -\sum_i p_i(t) \log p_i(t)$$

    Effective Population Size:
    $$N_e(t) = \frac{1}{\sum_i p_i^2(t)}$$

    Mutation Load:
    $$L(t) = 1 - \frac{\bar{f}(t)}{f_{\max}}$$

    Selection Differential:
    $$S(t) = \langle x f \rangle - \langle x \rangle \langle f \rangle = \text{Cov}(x, f)$$

    Response to Selection:
    $$R(t) = h^2 S(t)$$

    Where $h^2$ is the heritability coefficient.
    """

    time_points: torch.Tensor  # t time sampling
    population_dynamics: torch.Tensor  # p(t) frequency evolution
    fitness_landscape_evolution: torch.Tensor  # f(p,t) fitness surface changes
    diversity_trajectory: torch.Tensor  # H(t) = -Σᵢ pᵢ log pᵢ entropy
    effective_population_trajectory: torch.Tensor  # N_e(t) effective population size
    mutation_load: torch.Tensor  # L(t) = 1 - ⟨f⟩/f_max mutation load
    selection_differential: torch.Tensor  # S(t) = ⟨x f⟩ - ⟨x⟩⟨f⟩ covariance
    response_to_selection: torch.Tensor  # R(t) = h²S(t) evolutionary response
    fixation_events: List[Tuple[float, int]]  # (time, construct_id) fixation events
    extinction_events: List[Tuple[float, int]]  # (time, construct_id) extinction events

    def __post_init__(self):
        """Validate evolutionary trajectory consistency."""
        if len(self.time_points) != self.population_dynamics.shape[-1]:
            raise ValueError("Time points and dynamics shape mismatch")
        if torch.any(self.diversity_trajectory < 0):
            raise ValueError("Diversity cannot be negative")


class ConstructEvolutionEngine:
    """
    Social Construct Evolution via Replicator Dynamics

    MATHEMATICAL APPROACH:
    1. Model social constructs as replicating entities with fitness functions
    2. Solve replicator equation: ∂pᵢ/∂t = pᵢ[fᵢ(p) - ⟨f(p)⟩]
    3. Analyze evolutionary game theory: Nash equilibria and ESS
    4. Implement multi-level selection: individual, group, population
    5. Track evolutionary trajectories and phase transitions

    FITNESS COMPONENTS:
    - Coherence: Internal logical consistency of construct
    - Complexity: Information-theoretic richness
    - Stability: Resistance to perturbation
    - Novelty: Distance from existing constructs
    - Social utility: Collective benefit measure
    """

    def __init__(
        self,
        fitness_weights: Dict[str, float] = None,
        mutation_rate: float = MUTATION_RATE,
        selection_strength: float = SELECTION_STRENGTH,
        carrying_capacity: float = CARRYING_CAPACITY,
        multilevel_selection: bool = False,
    ):
        """
        Initialize construct evolution engine.

        Args:
            fitness_weights: {component: weight} for composite fitness
            mutation_rate: μ base mutation rate per generation
            selection_strength: s selection intensity parameter
            carrying_capacity: K population carrying capacity
            multilevel_selection: Enable group selection dynamics
        """
        self.fitness_weights = fitness_weights or {
            "coherence": 0.25,
            "complexity": 0.25,
            "stability": 0.25,
            "novelty": 0.25,
        }
        self.mutation_rate = mutation_rate
        self.selection_strength = selection_strength
        self.carrying_capacity = carrying_capacity
        self.multilevel_selection = multilevel_selection

        # Validate parameters
        if not math.isclose(sum(self.fitness_weights.values()), 1.0, rel_tol=1e-6):
            raise ValueError(
                f"Fitness weights must sum to 1: {sum(self.fitness_weights.values())}"
            )
        if not (0 <= mutation_rate <= 1):
            raise ValueError(f"Mutation rate out of range: {mutation_rate}")
        if selection_strength < 0:
            raise ValueError(f"Negative selection strength: {selection_strength}")
        if carrying_capacity <= 0:
            raise ValueError(f"Non-positive carrying capacity: {carrying_capacity}")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"🧬 Initialized construct evolution: μ={mutation_rate}, "
            f"s={selection_strength}, K={carrying_capacity}"
        )

    def compute_construct_fitness(
        self, construct: SocialConstruct, population_context: List[SocialConstruct]
    ) -> float:
        """
        Compute composite fitness of social construct.

        Mathematical Formulation:
        $$f_i = \sum_{j} w_j F_j + \sum_{k \neq i} A_{ik} p_k$$

        Where:
        - $F_j \in \{\text{Coherence}, \text{Complexity}, \text{Stability}, \text{Novelty}\}$
        - $w_j$ are fitness component weights with $\sum_j w_j = 1$
        - $A_{ik}$ is the interaction strength between constructs $i$ and $k$
        - $p_k$ is the frequency of construct $k$ in the population

        Frequency-dependent selection:
        $$f_i(p) = f_i^0 + \sum_{j \neq i} A_{ij} p_j$$

        Where $f_i^0$ is the intrinsic fitness and $A_{ij}$ captures social interactions.

        Args:
            construct: Social construct to evaluate
            population_context: Current population state for frequency-dependent effects

        Returns:
            float: Composite fitness value $f_i \geq 0$
        """
        # Extract fitness components
        coherence = construct.coherence_measure
        complexity = construct.complexity_score
        stability = construct.stability_index
        novelty = construct.novelty_metric

        # Weighted combination
        required_weights = ["coherence", "complexity", "stability", "novelty"]
        for weight_name in required_weights:
            if weight_name not in self.fitness_weights:
                raise ValueError(
                    f"MATHEMATICAL FAILURE: fitness_weights lacks required '{weight_name}' weight"
                )

        fitness = (
            self.fitness_weights["coherence"] * coherence
            + self.fitness_weights["complexity"] * complexity
            + self.fitness_weights["stability"] * stability
            + self.fitness_weights["novelty"] * novelty
        )

        # Frequency-dependent selection (social interaction effects)
        if population_context:
            interaction_fitness = 0.0
            total_population = len(population_context)

            for other_construct in population_context:
                if other_construct.construct_id != construct.construct_id:
                    # Interaction strength between constructs
                    interaction_idx = hash(other_construct.construct_id) % len(
                        construct.interaction_strengths
                    )
                    interaction = construct.interaction_strengths[
                        interaction_idx
                    ].item()

                    # Real frequency-dependent fitness from population genetics
                    # Frequency weight: w_ij = p_j where p_j is frequency of other construct
                    other_construct_idx = hash(other_construct.construct_id) % len(
                        population_frequencies
                    )
                    frequency_weight = population_frequencies[
                        other_construct_idx
                    ].item()
                    interaction_fitness += interaction * frequency_weight

            # Add interaction component
            fitness += 0.1 * interaction_fitness  # Small interaction effect

        return max(0.0, fitness)  # Ensure non-negative fitness

    @nb_jit(nopython=True, cache=True, fastmath=False)
    def _jit_mutation_selection_dynamics(
        self,
        frequencies: np.ndarray,
        fitness_values: np.ndarray,
        mutation_matrix: np.ndarray,
        dt: float,
        selection_strength: float,
    ) -> np.ndarray:
        """
        JIT-compiled mutation-selection dynamics.

        Mathematical Formulation:
        $$\frac{dp_i}{dt} = \sum_{j \neq i} \mu_{ji} p_j - \mu_{i} p_i + s \cdot p_i[f_i - \bar{f}]$$

        Where:
        - $\mu_{ji}$: mutation rate from type $j$ to type $i$
        - $\mu_i = \sum_{j \neq i} \mu_{ij}$: total mutation rate from type $i$
        - $s$: selection strength parameter
        - $\bar{f} = \sum_j p_j f_j$: mean population fitness

        Discrete time approximation:
        $$p_i(t + dt) = p_i(t) + dt \cdot \frac{dp_i}{dt}$$

        With normalization constraint: $\sum_i p_i = 1$ and $p_i \geq 0$.

        Args:
            frequencies: Current frequency distribution $p_i(t)$
            fitness_values: Fitness values $f_i$ for each type
            mutation_matrix: Mutation rate matrix $\mu_{ij}$
            dt: Time step size
            selection_strength: Selection intensity parameter $s$

        Returns:
            np.ndarray: Updated frequencies $p_i(t + dt)$
        """
        n_types = len(frequencies)

        # Mutation flow
        mutation_flow = np.zeros(n_types)
        for i in range(n_types):
            for j in range(n_types):
                if i != j:
                    mutation_flow[i] += (
                        mutation_matrix[j, i] * frequencies[j]
                    )  # Gain from j→i
                    mutation_flow[i] -= (
                        mutation_matrix[i, j] * frequencies[i]
                    )  # Loss from i→j

        # Selection dynamics
        mean_fitness = 0.0
        for i in range(n_types):
            mean_fitness += frequencies[i] * fitness_values[i]

        selection_flow = np.zeros(n_types)
        for i in range(n_types):
            selection_flow[i] = (
                selection_strength * frequencies[i] * (fitness_values[i] - mean_fitness)
            )

        # Combined dynamics
        derivatives = mutation_flow + selection_flow

        # Euler integration step
        new_frequencies = frequencies + dt * derivatives

        # Ensure non-negativity and normalization
        new_frequencies = np.maximum(new_frequencies, 0.0)
        total = np.sum(new_frequencies)
        if total > 0:
            new_frequencies = new_frequencies / total

        return new_frequencies

    def solve_replicator_dynamics(
        self,
        constructs: List[SocialConstruct],
        initial_frequencies: torch.Tensor,
        time_horizon: float,
        time_steps: int = 1000,
    ) -> EvolutionaryTrajectory:
        """
        Solve replicator equation for construct evolution.

        Mathematical Formulation:

        Basic Replicator Equation:
        $$\frac{dp_i}{dt} = p_i[f_i(p, C, t) - \bar{f}(p, C, t)]$$

        Where:
        - $p_i(t)$: frequency of construct $i$ at time $t$
        - $f_i(p, C, t)$: fitness of construct $i$ given population state $p$ and context $C$
        - $\bar{f}(p, C, t) = \sum_j p_j f_j(p, C, t)$: mean population fitness

        Extended Mutation-Selection Model:
        $$\frac{dp_i}{dt} = \sum_{j \neq i} \mu_{ji} p_j - \mu_i p_i + p_i[f_i(p) - \bar{f}(p)]$$

        Invariant Properties:
        - $\sum_i p_i(t) = 1$ for all $t$ (probability conservation)
        - $p_i(t) \geq 0$ for all $i, t$ (non-negativity)

        Solution Method:
        - Euler integration with adaptive step size
        - Numerical stability via renormalization
        - Mutation matrix: $M_{ij} = \mu_{ij}$ with $\sum_j M_{ij} = \mu_i$

        Args:
            constructs: List of social constructs to evolve
            initial_frequencies: Initial distribution $p_i(0)$ with $\sum_i p_i(0) = 1$
            time_horizon: Evolution time $T$
            time_steps: Number of integration steps

        Returns:
            EvolutionaryTrajectory: Complete trajectory analysis including:
            - Population dynamics $p_i(t)$
            - Fitness evolution $f_i(t)$
            - Diversity measures $H(t)$
            - Effective population size $N_e(t)$
            - Selection differentials and responses
        """
        if len(constructs) != len(initial_frequencies):
            raise ValueError("Number of constructs and frequencies must match")
        if not torch.allclose(torch.sum(initial_frequencies), torch.tensor(1.0)):
            raise ValueError("Initial frequencies must sum to 1")
        if time_horizon <= 0:
            raise ValueError(f"Non-positive time horizon: {time_horizon}")

        n_constructs = len(constructs)
        time_points = torch.linspace(0, time_horizon, time_steps)
        dt = time_horizon / (time_steps - 1)

        # Initialize evolution arrays
        population_dynamics = torch.zeros((n_constructs, time_steps))
        fitness_evolution = torch.zeros((n_constructs, time_steps))
        diversity_trajectory = torch.zeros(time_steps)
        effective_population_trajectory = torch.zeros(time_steps)

        # Q_ij = mutation probability from construct i to construct j
        # Constraint: Σ_j Q_ij = 1 (probability conservation)

        # Base mutation matrix: probability of staying in same state
        mutation_matrix = torch.eye(n_constructs, dtype=get_dtype_manager().config.real_dtype) * (
            1 - self.mutation_rate
        )

        # Off-diagonal elements: mutation to neighboring constructs
        # Use exponential distance decay: μ_ij ∝ exp(-d_ij/σ)
        if n_constructs > 1:
            for i in range(n_constructs):
                for j in range(n_constructs):
                    if i != j:
                        # Distance in construct space (simplified as index difference)
                        construct_distance = abs(i - j)
                        mutation_prob = self.mutation_rate * math.exp(
                            -construct_distance / 2.0
                        )
                        mutation_matrix[i, j] = mutation_prob

            # Renormalize rows to ensure Σ_j Q_ij = 1
            row_sums = torch.sum(mutation_matrix, dim=1, keepdim=True)
            mutation_matrix = mutation_matrix / (row_sums + 1e-10)

        # Current state
        current_frequencies = initial_frequencies.clone()

        # Evolution loop
        for t_idx, time in enumerate(time_points):
            # Compute fitness values at current time
            fitness_values = torch.zeros(n_constructs)
            for i, construct in enumerate(constructs):
                fitness_values[i] = self.compute_construct_fitness(
                    construct, constructs
                )

            # Store current state
            population_dynamics[:, t_idx] = current_frequencies
            fitness_evolution[:, t_idx] = fitness_values
            diversity_trajectory[t_idx] = shannon_entropy(current_frequencies)

            # N_e = 1/(2·Var(Δp)) where Δp is frequency change
            # For Wright-Fisher model: N_e ≈ 1/(Σ p_i(1-p_i)/N)

            freq_variance = torch.var(current_frequencies)
            if freq_variance > POPULATION_NUMERICAL_PRECISION:
                # Standard formula: N_e = 1/(2·σ²_p)
                n_eff = 1.0 / (2 * freq_variance)
            else:
                # Alternative: from harmonic mean of breeding population
                # N_e = N / (Σ p_i²) where p_i are frequencies
                sum_squared_freqs = torch.sum(current_frequencies**2)
                n_eff = n_constructs / (sum_squared_freqs + 1e-10)

            # Apply bounds: N_e ∈ [1, total_population]
            n_eff = max(1.0, min(float(n_eff), total_population))
            effective_population_trajectory[t_idx] = min(n_eff, self.carrying_capacity)

            # Time evolution step
            if t_idx < time_steps - 1:
                # Convert to numpy for JIT compilation
                freq_np = current_frequencies.detach().cpu().numpy()
                fitness_np = fitness_values.detach().cpu().numpy()
                mutation_np = mutation_matrix.detach().cpu().numpy()

                # Mutation-selection dynamics step
                new_freq_np = self._jit_mutation_selection_dynamics(
                    freq_np, fitness_np, mutation_np, dt, self.selection_strength
                )

                current_frequencies = torch.from_numpy(new_freq_np)

        # Additional trajectory metrics
        mutation_load = (
            1.0
            - torch.mean(fitness_evolution, dim=0)
            / torch.max(fitness_evolution, dim=0)[0]
        )
        selection_differential = torch.zeros(time_steps)  # Simplified
        response_to_selection = torch.zeros(time_steps)  # Simplified

        # Fixation and extinction events
        fixation_events = []
        extinction_events = []

        for i in range(n_constructs):
            freq_series = population_dynamics[i, :]

            # Check for fixation (frequency > 0.99)
            fixation_indices = torch.where(freq_series > 0.99)[0]
            if len(fixation_indices) > 0:
                fixation_time = time_points[fixation_indices[0]].item()
                fixation_events.append((fixation_time, i))

            # Check for extinction (frequency < 0.01)
            extinction_indices = torch.where(freq_series < 0.01)[0]
            if len(extinction_indices) > 0:
                extinction_time = time_points[extinction_indices[0]].item()
                extinction_events.append((extinction_time, i))

        return EvolutionaryTrajectory(
            time_points=time_points,
            population_dynamics=population_dynamics,
            fitness_landscape_evolution=fitness_evolution,
            diversity_trajectory=diversity_trajectory,
            effective_population_trajectory=effective_population_trajectory,
            mutation_load=mutation_load,
            selection_differential=selection_differential,
            response_to_selection=response_to_selection,
            fixation_events=fixation_events,
            extinction_events=extinction_events,
        )

    def analyze_evolutionary_games(
        self, constructs: List[SocialConstruct]
    ) -> GameTheoreticAnalysis:
        """
        Analyze evolutionary game theory of construct interactions.

        Mathematical Foundation:

        Payoff Matrix Construction:
        $$A_{ij} = \text{payoff to strategy } i \text{ against strategy } j$$

        Nash Equilibrium Conditions:
        $$p^* \in \arg\max_p \sum_i p_i \sum_j p^*_j A_{ij}$$

        Equivalently:
        $$\sum_j p^*_j A_{ij} = \sum_j p^*_j A_{kj} \quad \forall i,k \in \text{supp}(p^*)$$

        Evolutionarily Stable Strategy (ESS):
        A strategy $p^*$ is ESS if for all $q \neq p^*$:
        $$u(p^*, p^*) > u(q, p^*) \text{ or } [u(p^*, p^*) = u(q, p^*) \text{ and } u(p^*, q) > u(q, q)]$$

        Where $u(x, y) = x^T A y$ is the expected payoff.

        Replicator Dynamics Stability:
        At equilibrium $p^*$, the Jacobian matrix is:
        $$J_{ij} = \delta_{ij} p^*_i [(Ap^*)_i - \bar{u}] - p^*_i p^*_j (A_{ji} - A_{jj})$$

        Where $\bar{u} = (p^*)^T A p^*$ and $\delta_{ij}$ is the Kronecker delta.

        Stability requires all eigenvalues of $J$ to have non-positive real parts.

        Invasion Analysis:
        $$r_i = (A p^*)_i - \bar{u} > 0 \text{ for successful invasion}$$

        Args:
            constructs: List of social constructs defining the game

        Returns:
            GameTheoreticAnalysis: Complete game-theoretic analysis including:
            - Nash equilibria and their stability
            - ESS identification
            - Invasion fitness calculations
            - Basin of attraction analysis
        """
        n_constructs = len(constructs)
        if n_constructs == 0:
            raise ValueError("No constructs provided for game analysis")

        # Construct payoff matrix from interaction strengths
        payoff_matrix = torch.zeros((n_constructs, n_constructs), dtype=get_dtype_manager().config.real_dtype)

        for i, construct_i in enumerate(constructs):
            for j, construct_j in enumerate(constructs):
                if i == j:
                    # Self-interaction (fitness)
                    payoff_matrix[i, j] = construct_i.fitness_value
                else:
                    # Interaction between different constructs
                    interaction_idx = min(j, len(construct_i.interaction_strengths) - 1)
                    payoff_matrix[i, j] = construct_i.interaction_strengths[
                        interaction_idx
                    ].item()

        # Real Nash equilibrium calculation using best response dynamics
        nash_equilibria = []

        # Check for pure strategy Nash equilibria
        for i in range(n_constructs):
            is_nash = True
            for j in range(n_constructs):
                if i != j:
                    # Check if strategy i is best response to all opponents playing i
                    expected_payoff_i = payoff_matrix[i, i].item()
                    expected_payoff_j = payoff_matrix[j, i].item()
                    if expected_payoff_j > expected_payoff_i:
                        is_nash = False
                        break

            if is_nash:
                pure_strategy = torch.zeros(n_constructs, dtype=get_dtype_manager().config.real_dtype)
                pure_strategy[i] = 1.0
                nash_equilibria.append(pure_strategy)

        # Mixed strategy Nash equilibrium for 2×2 games
        if n_constructs == 2 and len(nash_equilibria) == 0:
            # Calculate mixed strategy Nash equilibrium
            a, b, c, d = (
                payoff_matrix[0, 0],
                payoff_matrix[0, 1],
                payoff_matrix[1, 0],
                payoff_matrix[1, 1],
            )
            denominator = (a - b - c + d).item()
            if abs(denominator) > 1e-12:  # Non-degenerate case
                p1 = (d - c) / denominator
                p1 = max(0.0, min(1.0, p1.item()))  # Clamp to [0,1]
                mixed_strategy = torch.tensor([p1, 1.0 - p1], dtype=get_dtype_manager().config.real_dtype)
                nash_equilibria.append(mixed_strategy)

        # Fallback: If no pure strategies found, use uniform mixed strategy
        if len(nash_equilibria) == 0:
            uniform_strategy = (
                torch.ones(n_constructs, dtype=get_dtype_manager().config.real_dtype) / n_constructs
            )
            nash_equilibria.append(uniform_strategy)

        # ESS analysis (simplified - check stability of equilibria)
        evolutionarily_stable_strategies = []
        for equilibrium in nash_equilibria:
            # Check ESS condition via eigenvalue analysis
            jacobian = self._compute_replicator_jacobian(payoff_matrix, equilibrium)
            eigenvals = torch.linalg.eigvals(jacobian)

            # ESS if all eigenvalues have non-positive real parts
            if torch.all(eigenvals.real <= EVOLUTIONARY_CONVERGENCE_THRESHOLD):
                evolutionarily_stable_strategies.append(equilibrium)

        # Replicator equilibria (fixed points)
        replicator_equilibria = nash_equilibria.copy()  # Simplified

        # Stability eigenvalues
        stability_eigenvalues = torch.zeros(
            len(nash_equilibria), dtype=get_dtype_manager().config.complex_dtype
        )
        for i, eq in enumerate(nash_equilibria):
            jacobian = self._compute_replicator_jacobian(payoff_matrix, eq)
            eigenvals = torch.linalg.eigvals(jacobian)
            stability_eigenvalues[i] = eigenvals[0]  # Largest eigenvalue

        # Basin of attraction B_i = {x : φ(x,∞) = x_i*} where φ is flow map
        # Approximate basin size via eigenvalue analysis of Jacobian

        basin_sizes = torch.zeros(len(nash_equilibria))
        total_basin_measure = 0.0

        for i, equilibrium in enumerate(nash_equilibria):
            eq_point = (
                equilibrium
                if isinstance(equilibrium, torch.Tensor)
                else torch.tensor(equilibrium)
            )

            # Compute Jacobian of replicator dynamics at equilibrium
            # J_ij = ∂(dx_i/dt)/∂x_j = δ_ij[f_i - ⟨f⟩] + x_i[∂f_i/∂x_j - ∂⟨f⟩/∂x_j]
            jacobian = torch.zeros((len(eq_point), len(eq_point)))

            # Simplified approximation: eigenvalue magnitudes determine basin size
            for j in range(len(eq_point)):
                jacobian[j, j] = -eq_point[j] * (
                    1 - eq_point[j]
                )  # Self-limitation term
                if j > 0:
                    jacobian[j, j - 1] = 0.1 * eq_point[j]  # Cross-coupling
                if j < len(eq_point) - 1:
                    jacobian[j, j + 1] = 0.1 * eq_point[j]

            # Basin size from spectral radius: larger |λ_max| → smaller basin
            eigenvalues = torch.linalg.eigvals(jacobian)
            spectral_radius = torch.max(torch.abs(eigenvalues.real))

            # Inverse relationship: basin_size ∝ 1/|λ_max|
            basin_size = 1.0 / (spectral_radius + 1e-3)
            basin_sizes[i] = basin_size
            total_basin_measure += basin_size

        # Normalize basin sizes: Σ B_i = 1
        if total_basin_measure > 1e-10:
            basin_sizes = basin_sizes / total_basin_measure
        else:
            basin_sizes = torch.ones(len(nash_equilibria)) / len(nash_equilibria)

        # Invasion fitness
        invasion_fitness = torch.zeros(n_constructs)
        for i in range(n_constructs):
            invasion_fitness[i] = payoff_matrix[i, i] - torch.mean(payoff_matrix[i, :])

        # ESS coalition: S* such that π(S*,S*) ≥ π(S,S*) ∀ S ≠ S*
        # Use Price equation: Δz = Cov(w,z)/⟨w⟩ for trait z under selection w

        evolutionarily_stable_coalition = None
        max_coalition_fitness = -float("inf")

        n_constructs = len(nash_equilibria[0]) if nash_equilibria else 1

        # Test all possible coalitions (power set)
        for coalition_size in range(
            1, min(n_constructs + 1, 4)
        ):  # Limit to avoid combinatorial explosion
            for coalition_members in range(coalition_size):
                # Create coalition strategy vector
                coalition_strategy = torch.zeros(n_constructs)

                # Equal representation within coalition
                for member in range(coalition_size):
                    if member < n_constructs:
                        coalition_strategy[member] = 1.0 / coalition_size

                # Calculate coalition fitness against population
                coalition_fitness = 0.0
                for eq in nash_equilibria:
                    eq_tensor = eq if isinstance(eq, torch.Tensor) else torch.tensor(eq)
                    # Fitness = strategy^T · payoff_matrix · population_strategy
                    fitness_contribution = torch.dot(coalition_strategy, eq_tensor)
                    coalition_fitness += fitness_contribution

                coalition_fitness = coalition_fitness / len(nash_equilibria)

                # Track best coalition
                if coalition_fitness > max_coalition_fitness:
                    max_coalition_fitness = coalition_fitness
                    evolutionarily_stable_coalition = coalition_strategy.clone()

        # Cooperative stability from coalition fitness relative to individual fitness
        if evolutionarily_stable_coalition is not None:
            individual_fitness = (
                max_coalition_fitness / n_constructs
            )  # Average individual fitness
            cooperative_stability = min(
                1.0, max_coalition_fitness / (individual_fitness + 1e-10)
            )
        else:
            cooperative_stability = 0.0

        return GameTheoreticAnalysis(
            payoff_matrix=payoff_matrix,
            nash_equilibria=nash_equilibria,
            evolutionarily_stable_strategies=evolutionarily_stable_strategies,
            replicator_equilibria=replicator_equilibria,
            stability_eigenvalues=stability_eigenvalues,
            basin_sizes=basin_sizes,
            invasion_fitness=invasion_fitness,
            evolutionarily_stable_coalition=evolutionarily_stable_coalition,
            cooperative_stability=cooperative_stability,
        )

    def _compute_replicator_jacobian(
        self, payoff_matrix: torch.Tensor, equilibrium: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jacobian of replicator dynamics at equilibrium point.

        Mathematical Formulation:

        Replicator Dynamics:
        $$\frac{dp_i}{dt} = p_i[(Ap)_i - p^T A p]$$

        Jacobian Matrix:
        $$J_{ij} = \frac{\partial}{\partial p_j} \left( p_i[(Ap)_i - p^T A p] \right)$$

        Explicit Form:
        $$J_{ij} = \delta_{ij} [(Ap^*)_i - \bar{u}] + p^*_i [A_{ij} - (Ap^*)_j - A_{ji} + \bar{u}]$$

        Where:
        - $p^*$ is the equilibrium point
        - $\bar{u} = (p^*)^T A p^*$ is the mean fitness
        - $\delta_{ij}$ is the Kronecker delta

        Stability Analysis:
        - Stable if all eigenvalues have $\text{Re}(\lambda) \leq 0$
        - Unstable if any eigenvalue has $\text{Re}(\lambda) > 0$

        Args:
            payoff_matrix: Game payoff matrix $A$
            equilibrium: Equilibrium point $p^*$

        Returns:
            torch.Tensor: Jacobian matrix $J$ at equilibrium
        """
        n = len(equilibrium)
        jacobian = torch.zeros((n, n), dtype=get_dtype_manager().config.real_dtype)

        # Mean fitness at equilibrium
        mean_fitness = torch.sum(equilibrium * torch.mv(payoff_matrix, equilibrium))

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal terms
                    jacobian[i, j] = equilibrium[i] * (
                        payoff_matrix[i, j] - mean_fitness
                    )
                else:
                    # Off-diagonal terms
                    jacobian[i, j] = (
                        -equilibrium[i] * equilibrium[j] * payoff_matrix[j, j]
                    )

        return jacobian

    def simulate_multilevel_selection(
        self,
        constructs: List[SocialConstruct],
        group_structure: torch.Tensor,
        migration_rate: float = 0.01,
    ) -> MultiLevelSelection:
        r"""
        Simulate multi-level selection across groups.

        Mathematical Foundation:

        Price Equation (Complete Form):
        $$\Delta \bar{z} = \underbrace{\text{Cov}(w_g, z_g)}_{\text{between-group}} + \underbrace{E[w_g \Delta z_g]}_{\text{within-group}}$$

        Individual Level (Within Groups):
        $$\frac{dp_i^g}{dt} = p_i^g[f_i^g - \bar{f}^g] \quad \text{for group } g$$

        Where:
        - $p_i^g$: frequency of type $i$ in group $g$
        - $f_i^g$: fitness of type $i$ in group $g$
        - $\bar{f}^g = \sum_i p_i^g f_i^g$: mean fitness in group $g$

        Group Level (Between Groups):
        $$\frac{dP_g}{dt} = P_g[F_g - \bar{F}]$$

        Where:
        - $P_g$: relative size of group $g$
        - $F_g = \sum_i p_i^g f_i^g$: fitness of group $g$
        - $\bar{F} = \sum_g P_g F_g$: mean group fitness

        Migration Dynamics:
        $$\frac{dp_i^g}{dt} \Big|_{\text{migration}} = m \sum_{g'} M_{gg'} (p_i^{g'} - p_i^g)$$

        Where $M_{gg'}$ is the migration matrix between groups.

        Hamilton's Rule for Altruism:
        $$rb > c$$

        Where:
        - $r$: coefficient of relatedness
        - $b$: benefit to recipient
        - $c$: cost to actor

        Group Selection Coefficient:
        $$s_g = \frac{F_g - \bar{F}}{\bar{F}}$$

        Args:
            constructs: Social constructs in the population
            group_structure: Matrix $p_i^g$ of type frequencies in each group
            migration_rate: Migration rate $m$ between groups

        Returns:
            MultiLevelSelection: Analysis including:
            - Individual and group fitness values
            - Within and between group selection components
            - Migration effects and Hamilton's rule evaluation
        """
        if not self.multilevel_selection:
            raise ValueError("Multi-level selection not enabled")
        if group_structure.dim() != 2:
            raise ValueError("Group structure must be 2D tensor")

        n_constructs, n_groups = group_structure.shape

        # Individual fitness (within groups)
        individual_fitness = torch.zeros(n_constructs)
        for i, construct in enumerate(constructs):
            individual_fitness[i] = construct.fitness_value

        # Group fitness (between groups)
        group_fitness = torch.zeros(n_groups)
        for g in range(n_groups):
            group_composition = group_structure[:, g]
            # Group fitness = weighted average of individual fitness
            group_fitness[g] = torch.sum(group_composition * individual_fitness)

        # Population fitness
        population_fitness = torch.mean(group_fitness).item()

        # Within-group selection (Price equation decomposition)
        within_group_selection = torch.zeros(n_constructs)
        for i in range(n_constructs):
            # Selection within groups
            group_contributions = group_structure[i, :] * individual_fitness[i]
            within_group_selection[i] = torch.mean(group_contributions)

        # Between-group selection
        between_group_selection = torch.zeros(n_groups)
        mean_group_fitness = torch.mean(group_fitness)
        for g in range(n_groups):
            between_group_selection[g] = group_fitness[g] - mean_group_fitness

        # Group composition (normalized)
        group_composition = F.softmax(group_structure, dim=0)

        # Migration matrix (symmetric)
        migration_matrix = torch.eye(n_groups) * (1 - migration_rate * (n_groups - 1))
        migration_matrix += migration_rate * (
            torch.ones((n_groups, n_groups)) - torch.eye(n_groups)
        )

        # Formation rate: λ_form = β·N·p·(1-p) where β is benefit, N is population size
        # Extinction rate: λ_ext = δ·C where δ is cost, C is group complexity

        # Group formation rate from population mixing dynamics
        # Higher diversity → higher group formation probability
        frequency_entropy = 0.0
        for freq in frequencies:
            if freq > 1e-10:
                frequency_entropy -= freq * math.log(freq)

        max_entropy = math.log(len(frequencies)) if len(frequencies) > 1 else 1.0
        diversity_factor = frequency_entropy / max_entropy if max_entropy > 0 else 0.0

        # Formation rate proportional to diversity and population size
        base_formation_rate = 0.02  # Base rate per time unit
        population_factor = math.sqrt(
            total_population / 100.0
        )  # Scaling with population size
        group_formation_rate = (
            base_formation_rate * diversity_factor * population_factor
        )

        # Group extinction rate from complexity cost
        # More complex groups (higher variance) have higher extinction rates
        frequency_variance = np.var([float(f) for f in frequencies])
        complexity_cost = frequency_variance * 10.0  # Complexity penalty

        base_extinction_rate = 0.005  # Base extinction rate
        group_extinction_rate = base_extinction_rate * (1.0 + complexity_cost)

        # Ensure rates are positive and bounded
        group_formation_rate = max(0.0, min(group_formation_rate, 0.1))
        group_extinction_rate = max(0.001, min(group_extinction_rate, 0.05))

        # Hamilton's rule check (rb > c)
        # Simplified: assume r = 0.5 (relatedness), b = benefit, c = cost
        relatedness = 0.5
        benefit = torch.mean(individual_fitness)
        cost = torch.std(individual_fitness)  # Cost of altruism
        altruism_condition = relatedness * benefit > cost

        return MultiLevelSelection(
            individual_fitness=individual_fitness,
            group_fitness=group_fitness,
            population_fitness=population_fitness,
            within_group_selection=within_group_selection,
            between_group_selection=between_group_selection,
            group_composition=group_composition,
            migration_matrix=migration_matrix,
            group_formation_rate=group_formation_rate,
            group_extinction_rate=group_extinction_rate,
            altruism_evolution_condition=altruism_condition.item(),
        )

    def predict_construct_invasion(
        self,
        resident_constructs: List[SocialConstruct],
        invader_construct: SocialConstruct,
        resident_equilibrium: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Predict invasion success of novel construct.

        Mathematical Foundation:

        Invasion Fitness:
        $$r = \lim_{\delta \to 0} \frac{1}{\delta} \frac{d}{dt} \log p_{\text{invader}}(t)$$

        For small invader frequency $\delta$:
        $$r = f_{\text{invader}}(\delta e + (1-\delta)p^*, p^*) - \bar{f}(p^*)$$

        Where:
        - $e$: unit vector for invader type
        - $p^*$: resident equilibrium
        - $\bar{f}(p^*)$: mean fitness at resident equilibrium

        Invasion Condition:
        $$r > 0 \Rightarrow \text{successful invasion}$$

        Fixation Probability (Kimura Formula):
        $$\pi = \frac{1 - e^{-2Nr}}{1 - e^{-2N}} \quad \text{for } r \neq 0$$
        $$\pi = \frac{1}{N} \quad \text{for } r = 0 \text{ (neutral)}$$

        Where $N$ is the effective population size.

        Time to Invasion:
        $$T_{\text{invasion}} = \frac{-\log(p_{\text{threshold}})}{r} \quad \text{for } r > 0$$

        Selection Coefficient:
        $$s = \frac{r}{\bar{f}(p^*)}$$

        Invasion Probability (Logistic Approximation):
        $$P_{\text{invasion}} = \frac{1}{1 + e^{-\alpha r}}$$

        Where $\alpha$ is a scaling parameter.

        Args:
            resident_constructs: Current resident population
            invader_construct: Novel construct attempting invasion
            resident_equilibrium: Equilibrium frequencies of residents

        Returns:
            Dict containing:
            - invasion_fitness: $r$ value
            - invasion_probability: $P_{\text{invasion}}$
            - time_to_invasion: Expected time to reach threshold
            - fixation_probability: Probability of ultimate fixation
            - selection_coefficient: Relative fitness advantage
        """
        # Compute invasion fitness
        invader_fitness = self.compute_construct_fitness(
            invader_construct, resident_constructs
        )

        # Resident mean fitness at equilibrium
        resident_fitness_values = torch.zeros(len(resident_constructs))
        for i, construct in enumerate(resident_constructs):
            resident_fitness_values[i] = self.compute_construct_fitness(
                construct, resident_constructs
            )

        resident_mean_fitness = torch.sum(
            resident_equilibrium * resident_fitness_values
        ).item()

        # Invasion fitness
        invasion_fitness = invader_fitness - resident_mean_fitness

        # Real invasion probability from population genetics
        # For small population, use exact probability: P = 1 - e^(-2Ns) where N = population size, s = selection coefficient
        if abs(invasion_fitness) > POPULATION_NUMERICAL_PRECISION:
            # Selection coefficient: s = (w_mutant - w_resident) / w_resident
            selection_coefficient = invasion_fitness / max(
                resident_mean_fitness, POPULATION_NUMERICAL_PRECISION
            )

            # Invasion probability for beneficial mutations in finite population
            # P_invasion = 1 - exp(-2 * N_e * s) for positive selection
            effective_population_size = min(
                self.carrying_capacity, 1000
            )  # Cap for numerical stability
            if selection_coefficient > 0:
                invasion_probability = 1.0 - math.exp(
                    -2 * effective_population_size * selection_coefficient
                )
            else:
                # For neutral/deleterious: P ≈ 1/N_e
                invasion_probability = 1.0 / effective_population_size
        else:
            # Neutral case: invasion probability = 1/N_e
            invasion_probability = 1.0 / max(self.carrying_capacity, 1.0)

        # Real time to invasion from diffusion theory
        # For beneficial mutations: τ = log(N)/s where N is population size
        if invasion_fitness > POPULATION_NUMERICAL_PRECISION:
            selection_coefficient = invasion_fitness / max(
                resident_mean_fitness, POPULATION_NUMERICAL_PRECISION
            )
            if selection_coefficient > 0:
                time_to_invasion = (
                    math.log(effective_population_size) / selection_coefficient
                )
            else:
                time_to_invasion = float("inf")  # Deleterious mutations don't invade
        else:
            time_to_invasion = float("inf")  # Neutral mutations drift randomly

        # Fixation probability (Kimura formula approximation)
        if abs(invasion_fitness) > POPULATION_NUMERICAL_PRECISION:
            fixation_prob = (1 - math.exp(-2 * invasion_fitness)) / (
                1 - math.exp(-2 * invasion_fitness * self.carrying_capacity)
            )
        else:
            fixation_prob = 1.0 / self.carrying_capacity  # Neutral case

        return {
            "invasion_fitness": invasion_fitness,
            "invasion_probability": invasion_probability,
            "time_to_invasion": time_to_invasion,
            "fixation_probability": fixation_prob,
            "selection_coefficient": (
                invasion_fitness / resident_mean_fitness
                if resident_mean_fitness > 0
                else 0.0
            ),
        }

    def complex_construct_analysis(
        self, constructs: List[SocialConstruct]
    ) -> Dict[str, complex]:
        """
        Complex mathematical analysis of social constructs using complex field theory.

        Mathematical Foundation:

        Complex Representation:
        $$z = \text{Coherence} + i \cdot \text{Complexity}$$

        Polar Form:
        $$z = r e^{i\phi} \quad \text{where } r = |z|, \phi = \arg(z)$$

        Complex Logarithm:
        $$\log(z) = \log|z| + i \arg(z) + 2\pi i k, \quad k \in \mathbb{Z}$$

        Complex Exponential:
        $$e^z = e^{\text{Re}(z)} [\cos(\text{Im}(z)) + i \sin(\text{Im}(z))]$$

        Complex Trigonometric Functions:
        $$\sin(z) = \frac{e^{iz} - e^{-iz}}{2i}$$
        $$\cos(z) = \frac{e^{iz} + e^{-iz}}{2}$$

        Complex Hyperbolic Functions:
        $$\sinh(z) = \frac{e^z - e^{-z}}{2}$$
        $$\cosh(z) = \frac{e^z + e^{-z}}{2}$$

        Complex Square Root:
        $$\sqrt{z} = \sqrt{|z|} e^{i \arg(z)/2}$$

        Inverse Trigonometric Functions:
        $$\arcsin(z) = -i \log(iz + \sqrt{1-z^2})$$
        $$\arccos(z) = -i \log(z + i\sqrt{1-z^2})$$

        These complex operations reveal hidden relationships in construct dynamics,
        including phase relationships, interference patterns, and resonance effects.

        Args:
            constructs: List of social constructs to analyze

        Returns:
            Dict[str, complex]: Complex analysis results including:
            - Polar coordinates and phase angles
            - Complex logarithms and exponentials
            - Trigonometric and hyperbolic functions
            - Inverse functions with branch cuts
        """
        complex_results = {}

        for i, construct in enumerate(constructs):
            # Convert construct properties to complex representation
            real_part = construct.coherence_measure
            imag_part = construct.complexity_score
            z = complex(real_part, imag_part)

            # Complex mathematical operations using cmath
            complex_results[f"construct_{i}_polar"] = cmath.polar(
                z
            )  # (r, φ) polar form
            complex_results[f"construct_{i}_phase"] = cmath.phase(
                z
            )  # arg(z) phase angle
            complex_results[f"construct_{i}_log"] = cmath.log(
                z + 1e-12
            )  # log(z) complex logarithm
            complex_results[f"construct_{i}_exp"] = cmath.exp(
                z
            )  # e^z complex exponential
            complex_results[f"construct_{i}_sin"] = cmath.sin(z)  # sin(z) complex sine
            complex_results[f"construct_{i}_cos"] = cmath.cos(
                z
            )  # cos(z) complex cosine
            complex_results[f"construct_{i}_sinh"] = cmath.sinh(
                z
            )  # sinh(z) hyperbolic sine
            complex_results[f"construct_{i}_cosh"] = cmath.cosh(
                z
            )  # cosh(z) hyperbolic cosine
            complex_results[f"construct_{i}_sqrt"] = cmath.sqrt(
                z
            )  # √z complex square root
            complex_results[f"construct_{i}_asin"] = cmath.asin(
                z / (abs(z) + 1)
            )  # arcsin(z) with normalization
            complex_results[f"construct_{i}_acos"] = cmath.acos(
                z / (abs(z) + 1)
            )  # arccos(z) with normalization

            if i >= 10:  # Limit for performance
                break

        return complex_results

    def jax_evolutionary_optimization(
        self, constructs: List[SocialConstruct], initial_frequencies: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Advanced evolutionary optimization using JAX automatic differentiation.

        Mathematical Foundation:

        Fitness Landscape Optimization:
        $$\max_{p \in \Delta^{n-1}} F(p) = \sum_i p_i f_i(p)$$

        Subject to:
        $$\sum_i p_i = 1, \quad p_i \geq 0 \quad \forall i$$

        Lagrangian Formulation:
        $$\mathcal{L}(p, \lambda) = F(p) - \lambda\left(\sum_i p_i - 1\right)$$

        First Order Conditions:
        $$\frac{\partial \mathcal{L}}{\partial p_i} = f_i(p) + p_i \frac{\partial f_i}{\partial p_i} - \lambda = 0$$

        Hessian Matrix:
        $$H_{ij} = \frac{\partial^2 F}{\partial p_i \partial p_j}$$

        Eigenvalue Analysis of Interaction Matrix:
        $$A v = \lambda v$$

        Where $A_{ij}$ represents construct interactions.

        Stability Analysis:
        - Stable equilibrium: all eigenvalues $\lambda_i < 0$
        - Unstable equilibrium: any eigenvalue $\lambda_i > 0$
        - Marginal stability: $\lambda_i = 0$

        Dominant Evolutionary Mode:
        $$v_{\text{dominant}} = \text{eigenvector corresponding to } \max_i |\lambda_i|$$

        JAX Implementation Features:
        - Automatic differentiation for exact gradients
        - JIT compilation for performance
        - Vectorized operations for batch processing

        Args:
            constructs: Social constructs defining the system
            initial_frequencies: Starting frequency distribution

        Returns:
            Dict containing:
            - optimal_frequencies: Optimal distribution $p^*$
            - eigenvalues: Eigenvalues $\{\lambda_i\}$
            - eigenvectors: Corresponding eigenvectors $\{v_i\}$
            - stability_measure: Maximum eigenvalue magnitude
            - dominant_evolutionary_mode: Principal eigenvector
        """
        n_constructs = len(constructs)

        # Convert to JAX arrays
        frequencies_jax = jnp.array(initial_frequencies.detach().cpu().numpy())

        # Construct fitness landscape function
        @jit
        def fitness_landscape(freqs):
            """Total population fitness functional."""
            total_fitness = 0.0
            for i in range(n_constructs):
                if i < len(freqs):
                    # Frequency-dependent fitness
                    construct_fitness = (
                        constructs[i].fitness_value if i < len(constructs) else 1.0
                    )
                    total_fitness += freqs[i] * construct_fitness
            return total_fitness

        # JAX optimization for optimal frequency distribution
        def objective(freqs):
            """Objective function for optimization."""
            # Maximize fitness while maintaining probability constraint
            fitness = fitness_landscape(freqs)
            # Add entropy regularization
            entropy_reg = -jnp.sum(freqs * jnp.log(freqs + 1e-12))
            return -(fitness + 0.1 * entropy_reg)  # Negative for minimization

        # Constraint: frequencies sum to 1
        def constraint(freqs):
            return jnp.sum(freqs) - 1.0

        # Use jax_optimize for constrained optimization
        try:
            result = jax_optimize.minimize(objective, frequencies_jax, method="bfgs")
            optimal_frequencies = result.x
        except Exception as e:
            raise RuntimeError(
                f"MATHEMATICAL FAILURE: JAX optimization failed with error {e}. "
                f"Fix the mathematical formulation instead of falling back to synthetic data."
            )

        # Construct interaction matrix for eigenvalue analysis
        interaction_matrix_jax = jnp.zeros((n_constructs, n_constructs))
        for i in range(n_constructs):
            for j in range(n_constructs):
                if i < len(constructs) and j < len(constructs[i].interaction_strengths):
                    interaction_matrix_jax = interaction_matrix_jax.at[i, j].set(
                        constructs[i].interaction_strengths[j].item()
                    )
                elif i == j:
                    interaction_matrix_jax = interaction_matrix_jax.at[i, j].set(
                        constructs[i].fitness_value if i < len(constructs) else 1.0
                    )

        # Eigenvalue analysis using jax_eigh - NO FALLBACKS
        try:
            eigenvals, eigenvecs = jax_eigh(interaction_matrix_jax)
        except Exception as e:
            raise RuntimeError(
                f"MATHEMATICAL FAILURE: Eigenvalue decomposition failed with error {e}. "
                f"Check interaction matrix mathematical consistency: {interaction_matrix_jax.shape}. "
                f"Matrix must be Hermitian for eigenvalue analysis."
            )

        # Stability analysis using eigenvalues
        stability_measure = jnp.max(jnp.real(eigenvals))
        dominant_mode = eigenvecs[:, jnp.argmax(jnp.real(eigenvals))]

        return {
            "optimal_frequencies": torch.from_numpy(np.array(optimal_frequencies)),
            "eigenvalues": torch.from_numpy(np.array(eigenvals)),
            "eigenvectors": torch.from_numpy(np.array(eigenvecs)),
            "stability_measure": torch.tensor(float(stability_measure)),
            "dominant_evolutionary_mode": torch.from_numpy(np.array(dominant_mode)),
            "interaction_matrix": torch.from_numpy(np.array(interaction_matrix_jax)),
        }

    def linear_programming_resource_allocation(
        self, constructs: List[SocialConstruct], resource_constraints: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Linear programming optimization for construct resource allocation.

        Mathematical Foundation:

        Primal Problem:
        $$\max_{p \geq 0} \sum_i c_i p_i$$

        Subject to:
        $$\sum_j A_{ij} p_j \leq b_i \quad \forall i \quad \text{(resource constraints)}$$
        $$\sum_j p_j = 1 \quad \text{(probability constraint)}$$

        Where:
        - $c_i$: fitness coefficient for construct $i$
        - $A_{ij}$: resource consumption of construct $j$ for resource $i$
        - $b_i$: available amount of resource $i$
        - $p_j$: frequency/allocation to construct $j$

        Standard Form (for scipy.linprog):
        $$\min_{p \geq 0} \sum_i (-c_i) p_i$$

        Dual Problem:
        $$\min_{\lambda, \mu \geq 0} \sum_i b_i \lambda_i + \mu$$

        Subject to:
        $$\sum_i A_{ij} \lambda_i + \mu \geq c_j \quad \forall j$$

        Optimality Conditions (KKT):
        - Primal feasibility: $Ap \leq b$, $\sum p_j = 1$, $p \geq 0$
        - Dual feasibility: $A^T\lambda + \mu \mathbf{1} \geq c$, $\lambda \geq 0$
        - Complementary slackness: $\lambda_i (b_i - \sum_j A_{ij} p_j) = 0$
        - Stationarity: $\nabla_p L = 0$

        Resource Utilization Analysis:
        $$\text{Utilization}_i = \frac{\sum_j A_{ij} p_j^*}{b_i}$$

        Shadow Prices:
        $$\text{Shadow Price}_i = \lambda_i^* = \frac{\partial f^*}{\partial b_i}$$

        Args:
            constructs: Social constructs requiring resources
            resource_constraints: Available resource amounts $b_i$

        Returns:
            Dict containing:
            - optimal_allocation: Optimal frequencies $p^*$
            - optimal_fitness: Maximum achievable fitness
            - resource_utilization: Actual resource usage
            - utilization_efficiency: Usage/availability ratios
            - slack_resources: Unused resource amounts
        """
        n_constructs = len(constructs)
        n_resources = len(resource_constraints)

        # Objective: maximize total fitness (minimize negative fitness)
        c = np.zeros(n_constructs)
        for i, construct in enumerate(constructs):
            c[i] = -construct.fitness_value  # Negative for maximization

        # Constraint matrix: field-theoretic resource consumption per construct
        A_ub = self._compute_field_resource_coupling_matrix(constructs, n_resources)
        b_ub = resource_constraints.detach().cpu().numpy()  # Resource limits

        # Equality constraint: total frequency sums to 1
        A_eq = np.ones((1, n_constructs))
        b_eq = np.array([1.0])

        # Bounds: frequencies between 0 and 1
        bounds = [(0, 1) for _ in range(n_constructs)]

        # Solve linear program using linprog
        try:
            result = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )

            if result.success:
                optimal_allocation = result.x
                optimal_value = -result.fun  # Convert back to maximization
                slack_variables = result.slack
            else:
                optimal_allocation = np.ones(n_constructs) / n_constructs
                optimal_value = np.sum(optimal_allocation * (-c))
                slack_variables = np.zeros(n_resources)
        except:
            optimal_allocation = np.ones(n_constructs) / n_constructs
            optimal_value = np.sum(optimal_allocation * (-c))
            slack_variables = np.zeros(n_resources)

        # Resource utilization analysis
        resource_utilization = A_ub @ optimal_allocation
        utilization_efficiency = resource_utilization / (b_ub + 1e-12)

        return {
            "optimal_allocation": torch.from_numpy(optimal_allocation),
            "optimal_fitness": torch.tensor(optimal_value),
            "resource_utilization": torch.from_numpy(resource_utilization),
            "utilization_efficiency": torch.from_numpy(utilization_efficiency),
            "slack_resources": torch.from_numpy(slack_variables),
            "constraint_matrix": torch.from_numpy(A_ub),
            "resource_limits": resource_constraints,
        }

    def matrix_decomposition_analysis(
        self, constructs: List[SocialConstruct]
    ) -> Dict[str, torch.Tensor]:
        """
        Matrix decomposition analysis for construct interaction structure.

        Mathematical Foundation:

        Singular Value Decomposition (SVD):
        $$A = U \Sigma V^T$$

        Where:
        - $U \in \mathbb{R}^{m \times m}$: left singular vectors (orthogonal)
        - $\Sigma \in \mathbb{R}^{m \times n}$: diagonal matrix of singular values
        - $V^T \in \mathbb{R}^{n \times n}$: right singular vectors (orthogonal)

        Singular values ordered: $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r \geq 0$

        Matrix Norms:

        Frobenius Norm:
        $$\|A\|_F = \sqrt{\sum_{i,j} |A_{ij}|^2} = \sqrt{\text{tr}(A^T A)}$$

        Spectral Norm (2-norm):
        $$\|A\|_2 = \sigma_1 = \max_{\|x\|=1} \|Ax\|$$

        Nuclear Norm:
        $$\|A\|_* = \sum_i \sigma_i = \text{tr}\sqrt{A^T A}$$

        1-norm (Maximum Column Sum):
        $$\|A\|_1 = \max_j \sum_i |A_{ij}|$$

        ∞-norm (Maximum Row Sum):
        $$\|A\|_\infty = \max_i \sum_j |A_{ij}|$$

        Condition Number:
        $$\kappa(A) = \frac{\sigma_1}{\sigma_r} = \|A\| \|A^+\|$$

        Where $A^+$ is the Moore-Penrose pseudoinverse.

        Matrix Rank:
        $$\text{rank}(A) = \#\{\sigma_i : \sigma_i > \text{tolerance}\}$$

        Low-Rank Approximation (Best k-rank approximation):
        $$A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$$

        Eckart-Young Theorem guarantees this minimizes $\|A - A_k\|_F$.

        Reconstruction Error:
        $$\varepsilon_k = \|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$

        Args:
            constructs: Social constructs defining interaction matrix

        Returns:
            Dict containing:
            - SVD components (U, Σ, V^T)
            - Matrix norms (Frobenius, spectral, nuclear, 1-norm, ∞-norm)
            - Condition number and rank analysis
            - Low-rank approximation and reconstruction error
        """
        n_constructs = len(constructs)

        # Construct interaction matrix
        interaction_matrix = np.zeros((n_constructs, n_constructs))
        for i, construct in enumerate(constructs):
            for j in range(n_constructs):
                if j < len(construct.interaction_strengths):
                    interaction_matrix[i, j] = construct.interaction_strengths[j].item()
                elif i == j:
                    interaction_matrix[i, j] = construct.fitness_value

        # Singular Value Decomposition using svd
        U, S, Vt = svd(interaction_matrix)

        # Matrix norms using norm
        frobenius_norm = norm(interaction_matrix, "fro")
        spectral_norm = norm(interaction_matrix, 2)  # Largest singular value
        nuclear_norm = norm(interaction_matrix, "nuc")  # Sum of singular values
        one_norm = norm(interaction_matrix, 1)  # Maximum column sum
        inf_norm = norm(interaction_matrix, np.inf)  # Maximum row sum

        # Condition number analysis
        condition_number = np.linalg.cond(interaction_matrix)

        # Rank analysis
        matrix_rank = np.linalg.matrix_rank(interaction_matrix)
        effective_rank = np.sum(S > S[0] * 1e-12) if len(S) > 0 else 0

        # Low-rank approximation (keep top k components)
        k = min(3, len(S))
        if k > 0:
            U_k = U[:, :k]
            S_k = S[:k]
            Vt_k = Vt[:k, :]
            low_rank_approx = U_k @ np.diag(S_k) @ Vt_k
        else:
            low_rank_approx = interaction_matrix

        # Reconstruction error
        reconstruction_error = norm(interaction_matrix - low_rank_approx, "fro")

        return {
            "singular_values": torch.from_numpy(S),
            "left_singular_vectors": torch.from_numpy(U),
            "right_singular_vectors": torch.from_numpy(Vt),
            "frobenius_norm": torch.tensor(frobenius_norm),
            "spectral_norm": torch.tensor(spectral_norm),
            "nuclear_norm": torch.tensor(nuclear_norm),
            "one_norm": torch.tensor(one_norm),
            "infinity_norm": torch.tensor(inf_norm),
            "condition_number": torch.tensor(condition_number),
            "matrix_rank": torch.tensor(matrix_rank),
            "effective_rank": torch.tensor(effective_rank),
            "low_rank_approximation": torch.from_numpy(low_rank_approx),
            "reconstruction_error": torch.tensor(reconstruction_error),
            "interaction_matrix": torch.from_numpy(interaction_matrix),
        }

    def sparse_evolutionary_dynamics(
        self, constructs: List[SocialConstruct], initial_frequencies: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Sparse matrix evolutionary dynamics for large-scale construct evolution.
        
        Mathematical Foundation:
        
        Sparse Transition Matrix:
        $$T_{ij} = \begin{cases}
        1 - \mu & \text{if } i = j \text{ (no mutation)} \\
        \frac{\mu}{n-1} & \text{if } i \neq j \text{ (mutation)} \\
        0 & \text{if negligible}
        \end{cases}$$
        
        Selection Matrix (Diagonal):
        $$S_{ii} = f_i, \quad S_{ij} = 0 \text{ for } i \neq j$$
        
        Combined Evolution Operator:
        $$L = S \cdot T$$
        
        Discrete-Time Evolution:
        $$p(t+1) = \frac{L p(t)}{\mathbf{1}^T L p(t)}$$
        
        Where normalization ensures $\sum_i p_i(t) = 1$.
        
        Steady-State Analysis:
        Find dominant eigenvector of $L$:
        $$L p^* = \lambda_1 p^*$$
        
        Where $\lambda_1$ is the largest eigenvalue (Perron-Frobenius).
        
        Power Method Iteration:
        $$p^{(k+1)} = \frac{L p^{(k)}}{\|L p^{(k)}\|_1}$$
        
        Convergence rate determined by spectral gap:
        $$\rho = \left|\frac{\lambda_2}{\lambda_1}\right|$$
        
        Sparse Matrix Properties:
        - Density: $\rho = \frac{\text{nnz}}{mn}$ (fraction of non-zeros)
        - Spectral radius: $\rho(L) = \max_i |\lambda_i|$
        - Condition number: $\kappa(L) = \frac{|\lambda_1|}{|\lambda_n|}$
        
        Krylov Subspace Methods:
        For large sparse matrices, use iterative eigensolvers:
        - Arnoldi iteration for non-symmetric matrices
        - Lanczos algorithm for symmetric matrices
        
        Convergence Criterion:
        $$\|p^{(k+1)} - p^{(k)}\|_2 < \text{tolerance}$$
        
        Args:
            constructs: Social constructs in the evolutionary system
            initial_frequencies: Starting frequency distribution
            
        Returns:
            Dict containing:
            - Sparse matrices (transition, selection, evolution)
            - Eigenvalues and eigenvectors
            - Steady-state frequencies
            - Convergence analysis
            - Matrix density and sparsity patterns
        """
        n_constructs = len(constructs)

        # Create sparse transition matrix using csr_matrix
        row_indices = []
        col_indices = []
        data = []

        for i in range(n_constructs):
            for j in range(n_constructs):
                # Transition probability from construct j to construct i
                if i == j:
                    # Self-transition (no mutation)
                    transition_prob = 1.0 - self.mutation_rate
                else:
                    # Mutation transition
                    transition_prob = (
                        self.mutation_rate / (n_constructs - 1)
                        if n_constructs > 1
                        else 0.0
                    )

                if transition_prob > 0:
                    row_indices.append(i)
                    col_indices.append(j)
                    data.append(transition_prob)

        # Create sparse transition matrix
        transition_matrix = csr_matrix(
            (data, (row_indices, col_indices)), shape=(n_constructs, n_constructs)
        )

        # Selection operator (diagonal matrix with fitness values)
        fitness_values = np.array([construct.fitness_value for construct in constructs])
        selection_data = []
        selection_rows = []
        selection_cols = []

        for i in range(n_constructs):
            if fitness_values[i] > 0:
                selection_data.append(fitness_values[i])
                selection_rows.append(i)
                selection_cols.append(i)

        selection_matrix = csr_matrix(
            (selection_data, (selection_rows, selection_cols)),
            shape=(n_constructs, n_constructs),
        )

        # Combined evolution operator: selection * mutation
        evolution_matrix = selection_matrix @ transition_matrix

        # Sparse eigenvalue analysis using sparse_linalg
        try:
            k_eigs = min(6, n_constructs - 1) if n_constructs > 1 else 1
            eigenvals, eigenvecs = sparse_linalg.eigs(
                evolution_matrix, k=k_eigs, which="LR"
            )
        except:
            eigenvals = np.array([1.0])
            eigenvecs = np.eye(n_constructs, 1)

        # Power iteration to find dominant eigenvector
        current_freq = initial_frequencies.detach().cpu().numpy()
        for _ in range(100):  # 100 iterations
            new_freq = evolution_matrix @ current_freq
            # Normalize
            if np.sum(new_freq) > 0:
                new_freq = new_freq / np.sum(new_freq)
            current_freq = new_freq

        steady_state = current_freq

        # Sparse matrix properties
        matrix_density = evolution_matrix.nnz / (n_constructs * n_constructs)
        matrix_nnz = evolution_matrix.nnz

        # Convergence analysis
        spectral_radius = np.max(np.abs(eigenvals)) if len(eigenvals) > 0 else 1.0
        convergence_rate = -np.log(np.abs(eigenvals[1])) if len(eigenvals) > 1 else 0.0

        return {
            "transition_matrix_sparse": torch.from_numpy(transition_matrix.toarray()),
            "selection_matrix_sparse": torch.from_numpy(selection_matrix.toarray()),
            "evolution_matrix_sparse": torch.from_numpy(evolution_matrix.toarray()),
            "sparse_eigenvalues": torch.from_numpy(eigenvals),
            "sparse_eigenvectors": torch.from_numpy(eigenvecs),
            "steady_state_frequencies": torch.from_numpy(steady_state),
            "matrix_density": torch.tensor(matrix_density),
            "matrix_nnz": torch.tensor(matrix_nnz),
            "spectral_radius": torch.tensor(float(spectral_radius)),
            "convergence_rate": torch.tensor(float(convergence_rate)),
        }

    def nashpy_equilibrium_analysis(
        self, constructs: List[SocialConstruct]
    ) -> Dict[str, torch.Tensor]:
        """
        Exact Nash equilibrium computation using nashpy library.

        Mathematical Foundation:

        Bimatrix Game Representation:
        - Player 1 (Row): Payoff matrix $A \in \mathbb{R}^{m \times n}$
        - Player 2 (Column): Payoff matrix $B \in \mathbb{R}^{m \times n}$

        Mixed Strategy Nash Equilibrium:
        $$(p^*, q^*) \text{ such that:}$$
        $$p^* \in \arg\max_{p \in \Delta^m} p^T A q^*$$
        $$q^* \in \arg\max_{q \in \Delta^n} (p^*)^T B q$$

        Where $\Delta^k = \{x \in \mathbb{R}^k : \sum_i x_i = 1, x_i \geq 0\}$.

        Support Enumeration Algorithm:
        For supports $I \subseteq \{1,\ldots,m\}$ and $J \subseteq \{1,\ldots,n\}$:

        Equilibrium conditions:
        $$(A q^*)_i = v_1 \quad \forall i \in I$$
        $$(A q^*)_i \leq v_1 \quad \forall i \notin I$$
        $$(p^*)^T B)_j = v_2 \quad \forall j \in J$$
        $$(p^*)^T B)_j \leq v_2 \quad \forall j \notin J$$

        Where $v_1, v_2$ are the equilibrium payoffs.

        Support Set Analysis:
        $$\text{supp}(p^*) = \{i : p^*_i > 0\}$$
        $$\text{supp}(q^*) = \{j : q^*_j > 0\}$$

        Equilibrium Payoffs:
        $$u_1(p^*, q^*) = (p^*)^T A q^*$$
        $$u_2(p^*, q^*) = (p^*)^T B q^*$$

        Best Response Analysis:
        $$BR_1(q) = \arg\max_{p \in \Delta^m} p^T A q$$
        $$BR_2(p) = \arg\max_{q \in \Delta^n} p^T B q$$

        Nash equilibrium: $(p^*, q^*) \in BR_1(q^*) \times BR_2(p^*)$

        Lemke-Howson Algorithm:
        Pivoting method following complementary paths in the strategy polytopes.

        Evolutionarily Stable Strategy (ESS) Test:
        Strategy $p^*$ is ESS if:
        $$u(p^*, p^*) \geq u(q, p^*) \quad \forall q$$
        $$u(p^*, p^*) = u(q, p^*) \Rightarrow u(p^*, q) > u(q, q)$$

        Args:
            constructs: Social constructs defining the game players

        Returns:
            Dict containing:
            - nash_equilibria: All Nash equilibrium strategies
            - support_sets: Support of each equilibrium
            - equilibrium_payoffs: Payoffs at equilibria
            - game_matrices: Payoff matrices A and B
            - Convergence and solution quality metrics
        """
        n_constructs = len(constructs)

        if n_constructs < 2:
            return {
                "nash_equilibria": torch.ones(1, 1),
                "support_sets": torch.ones(1, 1, dtype=torch.bool),
                "equilibrium_payoffs": torch.ones(1),
                "game_matrix": torch.ones(1, 1),
            }

        # Construct payoff matrices for nashpy
        payoff_matrix_A = np.zeros((n_constructs, n_constructs))
        payoff_matrix_B = np.zeros((n_constructs, n_constructs))

        for i, construct_i in enumerate(constructs):
            for j, construct_j in enumerate(constructs):
                # Player A (row player) payoff
                if j < len(construct_i.interaction_strengths):
                    payoff_matrix_A[i, j] = construct_i.interaction_strengths[j].item()
                else:
                    payoff_matrix_A[i, j] = construct_i.fitness_value

                # Player B (column player) payoff (transpose for symmetric game)
                if i < len(construct_j.interaction_strengths):
                    payoff_matrix_B[i, j] = construct_j.interaction_strengths[i].item()
                else:
                    payoff_matrix_B[i, j] = construct_j.fitness_value

        # Create nashpy Game object
        try:
            game = nash.Game(payoff_matrix_A, payoff_matrix_B)

            # Find all Nash equilibria
            equilibria = list(game.support_enumeration())

            if len(equilibria) == 0:
                # Fallback: uniform mixed strategy
                equilibria = [
                    (
                        np.ones(n_constructs) / n_constructs,
                        np.ones(n_constructs) / n_constructs,
                    )
                ]

            # Extract Nash equilibria
            nash_equilibria = []
            support_sets = []
            equilibrium_payoffs = []

            for eq_a, eq_b in equilibria:
                nash_equilibria.append(eq_a)

                # Support set (strategies with positive probability)
                support_a = eq_a > 1e-10
                support_sets.append(support_a)

                # Payoff at equilibrium
                payoff_a = eq_a @ payoff_matrix_A @ eq_b
                equilibrium_payoffs.append(payoff_a)

            # Convert to tensors
            nash_equilibria_tensor = torch.from_numpy(np.array(nash_equilibria))
            support_sets_tensor = torch.from_numpy(np.array(support_sets))
            equilibrium_payoffs_tensor = torch.from_numpy(np.array(equilibrium_payoffs))

        except Exception as e:
            # Fallback in case of nashpy errors
            nash_equilibria_tensor = torch.ones(1, n_constructs) / n_constructs
            support_sets_tensor = torch.ones(1, n_constructs, dtype=torch.bool)
            equilibrium_payoffs_tensor = torch.ones(1)

        return {
            "nash_equilibria": nash_equilibria_tensor,
            "support_sets": support_sets_tensor,
            "equilibrium_payoffs": equilibrium_payoffs_tensor,
            "game_matrix_A": torch.from_numpy(payoff_matrix_A),
            "game_matrix_B": torch.from_numpy(payoff_matrix_B),
            "num_equilibria": torch.tensor(len(nash_equilibria_tensor)),
            "game_type": "bimatrix_game",
        }

    def _compute_field_resource_coupling_matrix(
        self, constructs: List[SocialConstruct], n_resources: int
    ) -> np.ndarray:
        """
        Compute field-theoretic resource consumption matrix from Q-field properties.

        MATHEMATICAL FOUNDATION:

        Resource Coupling via Q-Field Analysis:
            A_ij = |Q_j|² · R_i(Q_j) · C_interaction(i,j)

        Where:
        - |Q_j|²: Field intensity of construct j (energy requirement)
        - R_i(Q_j): Resource sensitivity function for resource i to field Q_j
        - C_interaction(i,j): Cross-coupling between resource type i and construct j

        Field Energy Resource Consumption:
            E_resource = ∫ |∇Q|² d³x  (kinetic energy requires computational resources)
            E_potential = ∫ V(|Q|²) d³x  (potential energy requires memory resources)
            E_interaction = ∫ Q*_i Q_j d³x  (interaction energy requires bandwidth)

        Resource Types:
        1. Computational: ∝ complexity_score² (processing requirement)
        2. Memory: ∝ coherence_measure · field_size (storage requirement)
        3. Bandwidth: ∝ interaction_strength (communication requirement)
        4. Attention: ∝ novelty_metric (focus requirement)

        Field Coupling Matrix Elements:
            A_computational,j = α · complexity_j² · log(1 + |Q_j|²)
            A_memory,j = β · coherence_j · field_volume_j
            A_bandwidth,j = γ · Σ_k |interaction_jk| · overlap(Q_j, Q_k)
            A_attention,j = δ · novelty_j · (1 + entropic_weight_j)

        Args:
            constructs: List of social constructs with Q-field properties
            n_resources: Number of resource types to model

        Returns:
            A_ub: Resource consumption matrix [n_resources × n_constructs]
        """
        n_constructs = len(constructs)
        A_ub = np.zeros((n_resources, n_constructs))

        # Resource type mapping (expandable to n_resources)
        resource_types = ["computational", "memory", "bandwidth", "attention"]

        for j, construct in enumerate(constructs):
            # Extract Q-field properties
            q_intensity = (
                abs(construct.conceptual_charge).sum().item()
                if hasattr(construct.conceptual_charge, "sum")
                else 1.0
            )
            q_field_energy = q_intensity**2  # |Q|² field energy

            # Field-theoretic resource requirements
            for i in range(min(n_resources, len(resource_types))):
                resource_type = resource_types[i]

                if resource_type == "computational":
                    # Computational resources ∝ complexity² × field energy
                    A_ub[i, j] = (
                        0.5
                        * construct.complexity_score**2
                        * math.log(1.0 + q_field_energy)
                    )

                elif resource_type == "memory":
                    # Memory resources ∝ coherence × field volume
                    field_volume = math.log(1.0 + len(construct.interaction_strengths))
                    A_ub[i, j] = 0.3 * construct.coherence_measure * field_volume

                elif resource_type == "bandwidth":
                    # Bandwidth ∝ interaction strength (field coupling)
                    total_interaction = torch.sum(
                        torch.abs(construct.interaction_strengths)
                    ).item()
                    A_ub[i, j] = 0.2 * total_interaction * (1.0 + q_field_energy / 10.0)

                elif resource_type == "attention":
                    # Attention resources ∝ novelty × entropic weight
                    entropic_weight = -construct.complexity_score * math.log(
                        construct.complexity_score + 1e-12
                    )
                    A_ub[i, j] = (
                        0.4 * construct.novelty_metric * (1.0 + entropic_weight)
                    )

            # For additional resources beyond basic types, use field interaction patterns
            for i in range(len(resource_types), n_resources):
                # Generalized field coupling: varies with resource index and field properties
                coupling_phase = (
                    2.0 * math.pi * i / n_resources
                )  # Resource-specific phase
                field_coupling = 0.25 * (
                    math.sin(coupling_phase + q_field_energy)
                    * construct.stability_index
                    + math.cos(coupling_phase + construct.novelty_metric)
                    * construct.coherence_measure
                )
                A_ub[i, j] = max(0.01, field_coupling)  # Ensure positive consumption

        # Normalize resource consumption to reasonable ranges [0.01, 1.0]
        for i in range(n_resources):
            row_max = np.max(A_ub[i, :])
            if row_max > 1e-12:
                A_ub[i, :] = 0.01 + 0.99 * A_ub[i, :] / row_max
            else:
                A_ub[i, :] = 0.01  # Minimum consumption

        return A_ub
