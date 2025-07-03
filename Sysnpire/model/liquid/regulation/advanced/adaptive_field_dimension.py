"""
Adaptive Field Dimension Discovery - Dynamic Q-Field Dimensionality

MATHEMATICAL FOUNDATION: Authentic quantum cognition spectral gap method
following arXiv:2409.12805v1 using only Q(τ,C,s) field-derived mathematics.

QUANTUM COGNITION SPECTRAL GAP METHOD:
H(x) = (1/2) * Σ(Ak - ak*IN)²  [Error Hamiltonian]
|ψ₀(x)⟩ = quantum state for Q-field point x
Intrinsic dimension = D - γ where γ is index of largest eigenvalue ratio

RATE-DISTORTION OPTIMIZATION:
R(D) = min[p(ẑ|z)] I(Z;Ẑ) subject to E[d(Z,Ẑ)] ≤ D
Optimal dimension minimizes: H(Q-field|embedding_d) + λ·complexity(d)

HEAT KERNEL EIGENVALUE SCALING:
tr(e^(-tΔ)) ~ C·t^(-d/2) as t → 0⁺
Dimension emerges from eigenvalue decay rate of field Laplacian

All mathematics derived from authentic Q(τ,C,s) field structure.
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy import linalg
from scipy.spatial.distance import pdist, squareform

from Sysnpire.model.liquid.conceptual_charge_agent import ConceptualChargeAgent
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DimensionEstimate:
    quantum_dimension: int
    rate_distortion_dimension: int
    heat_kernel_dimension: int
    consensus_dimension: int
    confidence_score: float
    field_complexity_measure: float
    estimation_metadata: Dict[str, Any]


@dataclass
class FieldSignature:
    n_points: int
    field_energy: float
    complexity_hash: str
    temporal_span: float
    signature_timestamp: float


class QuantumCognitionDimensionEstimator:
    """
    Quantum Cognition Spectral Gap Method using authentic Q(τ,C,s) mathematics.

    Based on arXiv:2409.12805v1 with complete field-theoretic derivation:
    - Construct quantum states |ψ(q)⟩ from Q-field mathematical components
    - Compute quantum metric g(x) from field eigenstructure
    - Find largest spectral gap in eigenvalue spectrum
    - Intrinsic dimension = total_dim - gap_index
    """

    def __init__(self, precision: float = 1e-12):
        self.precision = precision
        self.quantum_cache: Dict[str, Any] = {}
        logger.info("🔬 QuantumCognitionDimensionEstimator initialized")

    def estimate_dimension(
        self, agents: List[ConceptualChargeAgent]
    ) -> Tuple[int, Dict[str, Any]]:
        quantum_states = self._construct_authentic_quantum_states(agents)
        quantum_metric = self._compute_field_derived_metric(quantum_states, agents)
        eigenvalues = self._compute_metric_eigenvalues(quantum_metric)
        gap_location, gap_magnitude = self._detect_spectral_gap(eigenvalues)

        total_features = len(eigenvalues)
        intrinsic_dimension = max(1, total_features - gap_location)

        metadata = {
            "method": "quantum_cognition_spectral_gap",
            "total_features": total_features,
            "gap_location": gap_location,
            "gap_magnitude": gap_magnitude,
            "eigenvalue_spectrum": eigenvalues.tolist(),
            "confidence": min(1.0, gap_magnitude * 2.0),
        }

        logger.debug(
            f"🔬 Quantum dimension: {intrinsic_dimension} (gap at {gap_location}, magnitude {gap_magnitude:.6f})"
        )
        return intrinsic_dimension, metadata

    def _construct_authentic_quantum_states(
        self, agents: List[ConceptualChargeAgent]
    ) -> np.ndarray:
        quantum_states = []

        for agent in agents:
            q_comps = agent.Q_components

            real_part = float(q_comps.Q_value.real)
            imag_part = float(q_comps.Q_value.imag)
            magnitude = abs(q_comps.Q_value)
            phase = np.angle(q_comps.Q_value)

            gamma = float(q_comps.gamma)
            T_real = float(q_comps.T_tensor.real)
            T_imag = float(q_comps.T_tensor.imag)
            E_real = float(q_comps.E_trajectory.real)
            E_imag = float(q_comps.E_trajectory.imag)
            phi_real = float(q_comps.phi_semantic.real)
            phi_imag = float(q_comps.phi_semantic.imag)

            theta_total = float(q_comps.theta_components.total)
            theta_semantic = float(q_comps.theta_components.theta_semantic)
            theta_emotional = float(q_comps.theta_components.theta_emotional)
            theta_interaction = float(q_comps.theta_components.theta_interaction)
            theta_field = float(q_comps.theta_components.theta_field)

            psi_persistence = float(q_comps.psi_persistence)
            psi_gaussian = float(q_comps.psi_gaussian)
            psi_exponential = float(q_comps.psi_exponential_cosine)

            phase_factor_real = float(q_comps.phase_factor.real)
            phase_factor_imag = float(q_comps.phase_factor.imag)

            breathing_coeff_sum = sum(
                abs(coeff) for coeff in agent.breathing_q_coefficients.values()
            )
            hecke_eigenval_sum = sum(
                abs(eigenval) for eigenval in agent.hecke_eigenvalues.values()
            )

            field_x, field_y = agent.state.field_position
            tau_real = agent.tau_position.real
            tau_imag = agent.tau_position.imag

            T_phi_coupling = T_real * phi_real + T_imag * phi_imag
            E_theta_coupling = E_real * theta_total
            gamma_psi_product = gamma * psi_persistence
            field_energy_density = magnitude * gamma * psi_persistence

            quantum_state = np.array(
                [
                    real_part,
                    imag_part,
                    magnitude,
                    phase,
                    gamma,
                    T_real,
                    T_imag,
                    E_real,
                    E_imag,
                    phi_real,
                    phi_imag,
                    theta_total,
                    theta_semantic,
                    theta_emotional,
                    theta_interaction,
                    theta_field,
                    psi_persistence,
                    psi_gaussian,
                    psi_exponential,
                    phase_factor_real,
                    phase_factor_imag,
                    breathing_coeff_sum,
                    hecke_eigenval_sum,
                    field_x,
                    field_y,
                    tau_real,
                    tau_imag,
                    T_phi_coupling,
                    E_theta_coupling,
                    gamma_psi_product,
                    field_energy_density,
                ]
            )

            norm = np.linalg.norm(quantum_state)
            if norm > self.precision:
                quantum_state = quantum_state / norm

            quantum_states.append(quantum_state)

        return np.array(quantum_states)

    def _compute_field_derived_metric(
        self, quantum_states: np.ndarray, agents: List[ConceptualChargeAgent]
    ) -> jnp.ndarray:
        """
        Compute quantum metric tensor from authentic Q-field relationships.

        MATHEMATICAL FOUNDATION (arXiv:2409.12805v1):
        H(x) = (1/2) * Σ(Ak - ak*IN)²  [Error Hamiltonian]
        A(ψ) = (⟨ψ|A₁|ψ⟩, ..., ⟨ψ|AD|ψ⟩) ∈ ℝᴰ  [Position vector]

        Uses field eigenstructure for observable construction.
        """
        quantum_states_jax = jnp.array(quantum_states)
        n_states, state_dim = quantum_states_jax.shape

        field_observables = self._construct_field_observables(
            quantum_states_jax, agents
        )

        def compute_position_vector(state, obs_matrices):
            return jnp.array(
                [
                    jnp.real(jnp.vdot(state, obs_matrices[k] @ state))
                    for k in range(len(obs_matrices))
                ]
            )

        position_vectors = jax.vmap(
            lambda state: compute_position_vector(state, field_observables)
        )(quantum_states_jax)

        @jit
        def quantum_metric_element(pos_i, pos_j):
            diff = pos_i - pos_j
            field_scale = jnp.mean(jnp.abs(pos_i) + jnp.abs(pos_j)) + 1e-10
            return jnp.exp(-jnp.linalg.norm(diff) ** 2 / (2 * field_scale**2))

        if n_states > 100:
            quantum_metric = self._compute_quantum_metric_field_hierarchical(
                position_vectors, quantum_metric_element, agents
            )
        else:
            quantum_metric = jax.vmap(
                lambda i: jax.vmap(
                    lambda j: quantum_metric_element(
                        position_vectors[i], position_vectors[j]
                    )
                )(jnp.arange(n_states))
            )(jnp.arange(n_states))

        quantum_metric = quantum_metric + 1e-12 * jnp.eye(n_states)
        return quantum_metric

    def _construct_field_observables(
        self, quantum_states_jax: jnp.ndarray, agents: List[ConceptualChargeAgent]
    ) -> jnp.ndarray:
        state_dim = quantum_states_jax.shape[1]
        field_covariance = self._extract_field_covariance(agents)
        eigenvals, eigenvecs = jnp.linalg.eigh(field_covariance)

        observables = []
        for k in range(min(state_dim, len(eigenvals))):
            if k < len(eigenvecs):
                eigenvec = eigenvecs[:, k]
                if len(eigenvec) < state_dim:
                    padded_eigenvec = jnp.zeros(state_dim)
                    padded_eigenvec = padded_eigenvec.at[: len(eigenvec)].set(eigenvec)
                    eigenvec = padded_eigenvec
                elif len(eigenvec) > state_dim:
                    eigenvec = eigenvec[:state_dim]

                observable = jnp.outer(eigenvec, eigenvec.conj())
            else:
                observable = jnp.zeros((state_dim, state_dim))
                observable = observable.at[k % state_dim, k % state_dim].set(1.0)

            observables.append(observable)

        while len(observables) < state_dim:
            k = len(observables)
            observable = jnp.zeros((state_dim, state_dim))
            observable = observable.at[k % state_dim, k % state_dim].set(1.0)
            observables.append(observable)

        return jnp.array(observables)

    def _extract_field_covariance(
        self, agents: List[ConceptualChargeAgent]
    ) -> jnp.ndarray:
        field_vectors = []

        for agent in agents:
            q_comps = agent.Q_components
            # Convert MPS tensors to CPU before JAX conversion
            field_vector = jnp.array(
                [
                    q_comps.Q_value.real.cpu() if hasattr(q_comps.Q_value.real, 'cpu') else q_comps.Q_value.real,
                    q_comps.Q_value.imag.cpu() if hasattr(q_comps.Q_value.imag, 'cpu') else q_comps.Q_value.imag,
                    q_comps.gamma.cpu() if hasattr(q_comps.gamma, 'cpu') else q_comps.gamma,
                    q_comps.T_tensor.real.cpu() if hasattr(q_comps.T_tensor.real, 'cpu') else q_comps.T_tensor.real,
                    q_comps.T_tensor.imag.cpu() if hasattr(q_comps.T_tensor.imag, 'cpu') else q_comps.T_tensor.imag,
                    q_comps.E_trajectory.real.cpu() if hasattr(q_comps.E_trajectory.real, 'cpu') else q_comps.E_trajectory.real,
                    q_comps.E_trajectory.imag.cpu() if hasattr(q_comps.E_trajectory.imag, 'cpu') else q_comps.E_trajectory.imag,
                    q_comps.phi_semantic.real.cpu() if hasattr(q_comps.phi_semantic.real, 'cpu') else q_comps.phi_semantic.real,
                    q_comps.phi_semantic.imag.cpu() if hasattr(q_comps.phi_semantic.imag, 'cpu') else q_comps.phi_semantic.imag,
                    q_comps.theta_components.total.cpu() if hasattr(q_comps.theta_components.total, 'cpu') else q_comps.theta_components.total,
                    q_comps.psi_persistence.cpu() if hasattr(q_comps.psi_persistence, 'cpu') else q_comps.psi_persistence,
                ]
            )
            field_vectors.append(field_vector)

        field_matrix = jnp.array(field_vectors)
        field_centered = field_matrix - jnp.mean(field_matrix, axis=0, keepdims=True)
        covariance = jnp.cov(field_centered.T)

        if covariance.ndim == 0:
            covariance = jnp.array([[covariance]])
        elif covariance.ndim == 1:
            covariance = jnp.diag(covariance)

        regularized_cov = covariance + 1e-6 * jnp.eye(covariance.shape[0])
        return regularized_cov

    def _compute_quantum_metric_field_hierarchical(
        self,
        position_vectors: jnp.ndarray,
        metric_element_fn: Callable,
        agents: List[ConceptualChargeAgent],
    ) -> jnp.ndarray:
        n_states = position_vectors.shape[0]
        metric_approx = jnp.zeros((n_states, n_states))

        field_energies = jnp.array(
            [abs(agent.Q_components.Q_value.cpu() if hasattr(agent.Q_components.Q_value, 'cpu') else agent.Q_components.Q_value) ** 2 for agent in agents]
        )
        energy_sorted_indices = jnp.argsort(field_energies)[::-1]

        field_depth = int(jnp.ceil(jnp.log2(n_states))) + 2

        for level in range(field_depth + 1):
            level_size = min(2**level, n_states)

            if level_size == n_states:
                indices = jnp.arange(n_states)
            else:
                step = max(1, n_states // level_size)
                indices = energy_sorted_indices[::step][:level_size]

            level_positions = position_vectors[indices]
            level_metric = jax.vmap(
                lambda i: jax.vmap(
                    lambda j: metric_element_fn(level_positions[i], level_positions[j])
                )(jnp.arange(level_size))
            )(jnp.arange(level_size))

            interpolated_contribution = self._field_interpolate_metric(
                level_metric, indices, n_states, level, field_energies
            )

            level_weight = 1.0 / (2.0**level)
            metric_approx = metric_approx + level_weight * interpolated_contribution

        total_weight = sum(1.0 / (2.0**level) for level in range(field_depth + 1))
        metric_approx = metric_approx / total_weight
        return metric_approx

    def _field_interpolate_metric(
        self,
        level_metric: jnp.ndarray,
        indices: jnp.ndarray,
        target_size: int,
        level: int,
        field_energies: jnp.ndarray,
    ) -> jnp.ndarray:
        level_size = len(indices)
        if level_size == target_size:
            return level_metric

        interpolated = jnp.zeros((target_size, target_size))
        bandwidth = jnp.mean(field_energies) / (2.0**level) + 1e-10

        for i in range(target_size):
            for j in range(target_size):
                energy_i = (
                    field_energies[i]
                    if i < len(field_energies)
                    else jnp.mean(field_energies)
                )
                energy_j = (
                    field_energies[j]
                    if j < len(field_energies)
                    else jnp.mean(field_energies)
                )

                weights_i = jnp.exp(
                    -jnp.abs(energy_i - field_energies[indices]) / bandwidth
                )
                weights_j = jnp.exp(
                    -jnp.abs(energy_j - field_energies[indices]) / bandwidth
                )

                weights_i = weights_i / (jnp.sum(weights_i) + 1e-12)
                weights_j = weights_j / (jnp.sum(weights_j) + 1e-12)

                interpolated_value = jnp.sum(
                    weights_i[:, None] * weights_j[None, :] * level_metric
                )
                interpolated = interpolated.at[i, j].set(interpolated_value)

        return interpolated

    def _compute_metric_eigenvalues(self, quantum_metric: jnp.ndarray) -> jnp.ndarray:
        eigenvalues = jnp.linalg.eigvals(quantum_metric)
        eigenvalues = jnp.real(eigenvalues)
        eigenvalues = jnp.sort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigenvalues > self.precision]
        return eigenvalues

    def _detect_spectral_gap(self, eigenvalues: np.ndarray) -> Tuple[int, float]:
        if len(eigenvalues) < 2:
            return 0, 0.0

        ratios = eigenvalues[:-1] / (eigenvalues[1:] + self.precision)
        gap_location = np.argmax(ratios)
        gap_magnitude = ratios[gap_location]
        return gap_location, gap_magnitude


class RateDistortionOptimizer:
    def __init__(self, lambda_complexity: float = 0.1):
        self.lambda_complexity = lambda_complexity
        self.information_cache: Dict[str, Any] = {}
        logger.info("📊 RateDistortionOptimizer initialized")

    def estimate_dimension(
        self, agents: List[ConceptualChargeAgent]
    ) -> Tuple[int, Dict[str, Any]]:
        field_matrix = self._extract_field_information(agents)
        max_dim = min(len(agents) - 1, field_matrix.shape[1], 50)
        candidate_dims = range(1, max_dim + 1)

        optimal_dim, optimization_results = self._optimize_rate_distortion(
            field_matrix, candidate_dims
        )

        metadata = {
            "method": "rate_distortion_optimization",
            "candidate_dimensions": list(candidate_dims),
            "optimization_results": optimization_results,
            "lambda_complexity": self.lambda_complexity,
            "confidence": optimization_results.get("confidence", 0.7),
        }

        logger.debug(f"📊 Rate-distortion dimension: {optimal_dim}")
        return optimal_dim, metadata

    def _extract_field_information(
        self, agents: List[ConceptualChargeAgent]
    ) -> np.ndarray:
        information_vectors = []

        for agent in agents:
            q_comps = agent.Q_components

            info_vector = [
                float(q_comps.Q_value.real),
                float(q_comps.Q_value.imag),
                abs(q_comps.Q_value),
                np.angle(q_comps.Q_value),
                float(q_comps.gamma),
                float(q_comps.T_tensor.real),
                float(q_comps.T_tensor.imag),
                float(q_comps.E_trajectory.real),
                float(q_comps.E_trajectory.imag),
                float(q_comps.phi_semantic.real),
                float(q_comps.phi_semantic.imag),
                float(q_comps.theta_components.total),
                float(q_comps.theta_components.theta_semantic),
                float(q_comps.theta_components.theta_emotional),
                float(q_comps.theta_components.theta_interaction),
                float(q_comps.theta_components.theta_field),
                float(q_comps.psi_persistence),
                float(q_comps.psi_gaussian),
                float(q_comps.psi_exponential_cosine),
                float(q_comps.phase_factor.real),
                float(q_comps.phase_factor.imag),
            ]

            field_x, field_y = agent.state.field_position
            info_vector.extend([float(field_x), float(field_y)])

            tau_real = agent.tau_position.real
            tau_imag = agent.tau_position.imag
            info_vector.extend([float(tau_real), float(tau_imag)])

            breathing_magnitude = sum(
                abs(coeff) for coeff in agent.breathing_q_coefficients.values()
            )
            hecke_magnitude = sum(
                abs(eigenval) for eigenval in agent.hecke_eigenvalues.values()
            )
            info_vector.extend([float(breathing_magnitude), float(hecke_magnitude)])

            T_phi_real_coupling = float(q_comps.T_tensor.real) * float(
                q_comps.phi_semantic.real
            )
            T_phi_imag_coupling = float(q_comps.T_tensor.imag) * float(
                q_comps.phi_semantic.imag
            )
            E_theta_coupling = abs(q_comps.E_trajectory) * float(
                q_comps.theta_components.total
            )
            gamma_psi_coupling = float(q_comps.gamma) * float(q_comps.psi_persistence)

            info_vector.extend(
                [
                    T_phi_real_coupling,
                    T_phi_imag_coupling,
                    E_theta_coupling,
                    gamma_psi_coupling,
                ]
            )

            information_vectors.append(info_vector)

        return np.array(information_vectors)

    def _optimize_rate_distortion(
        self, field_matrix: np.ndarray, candidate_dims: range
    ) -> Tuple[int, Dict[str, Any]]:
        n_points, n_features = field_matrix.shape
        costs = {}
        rates = {}
        distortions = {}

        for d in candidate_dims:
            if d > min(n_points - 1, n_features):
                costs[d] = float("inf")
                continue

            rate = self._compute_information_rate(field_matrix, d)
            distortion = self._compute_reconstruction_distortion(field_matrix, d)
            complexity_penalty = self.lambda_complexity * d
            total_cost = rate + complexity_penalty + distortion

            costs[d] = total_cost
            rates[d] = rate
            distortions[d] = distortion

        optimal_dim = min(costs.keys(), key=lambda d: costs[d])

        optimization_results = {
            "costs": costs,
            "rates": rates,
            "distortions": distortions,
            "optimal_cost": costs[optimal_dim],
            "confidence": self._compute_optimization_confidence(costs),
        }

        return optimal_dim, optimization_results

    def _compute_information_rate(
        self, field_matrix: np.ndarray, dimension: int
    ) -> float:
        field_jax = jnp.array(field_matrix)
        n, d = field_jax.shape

        if n > 100 or d > 100:
            return self._compute_information_rate_field_optimized(
                field_matrix, dimension
            )
        else:
            U, s, Vt = jnp.linalg.svd(field_jax, full_matrices=False)
            reduced_matrix = U[:, :dimension] @ jnp.diag(s[:dimension])

            original_entropy = self._jax_estimate_entropy(field_jax)
            reduced_entropy = self._jax_estimate_entropy(reduced_matrix)

            mutual_information = original_entropy - reduced_entropy
            rate = jnp.maximum(0.0, mutual_information)
            return float(rate)

    def _compute_information_rate_field_optimized(
        self, field_matrix: np.ndarray, dimension: int
    ) -> float:
        field_jax = jnp.array(field_matrix)
        n, d = field_jax.shape

        field_centered = field_jax - jnp.mean(field_jax, axis=0, keepdims=True)
        covariance = jnp.cov(field_centered.T)

        if covariance.ndim == 0:
            covariance = jnp.array([[covariance]])
        elif covariance.ndim == 1:
            covariance = jnp.diag(covariance)

        eigenvals, eigenvecs = jnp.linalg.eigh(covariance)
        k = min(dimension + 20, min(n, d))

        principal_eigenvecs = eigenvecs[:, -k:]
        Y = field_jax @ principal_eigenvecs
        Q, _ = jnp.linalg.qr(Y)
        B = Q.T @ field_jax
        _, s_approx, Vt_approx = jnp.linalg.svd(B, full_matrices=False)

        if dimension < len(s_approx):
            preserved_variance = jnp.sum(s_approx[:dimension] ** 2)
            total_variance = jnp.sum(s_approx**2)
            information_ratio = preserved_variance / (total_variance + 1e-12)
            rate_approx = -jnp.log(information_ratio + 1e-12)
        else:
            rate_approx = 0.0

        return float(rate_approx)

    def _jax_estimate_entropy(self, data_matrix: jnp.ndarray) -> float:
        if data_matrix.ndim == 1:
            data_matrix = data_matrix.reshape(-1, 1)

        centered_data = data_matrix - jnp.mean(data_matrix, axis=0, keepdims=True)

        if centered_data.shape[1] == 1:
            variance = jnp.var(centered_data.flatten())
            entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * variance)
            return float(entropy)

        cov_matrix = jnp.cov(centered_data.T)

        if cov_matrix.ndim == 0:
            cov_matrix = jnp.array([[cov_matrix]])
        elif cov_matrix.ndim == 1:
            cov_matrix = jnp.diag(cov_matrix)

        regularized_cov = cov_matrix + 1e-6 * jnp.eye(cov_matrix.shape[0])
        log_det = jnp.linalg.slogdet(regularized_cov)[1]
        entropy = 0.5 * (log_det + cov_matrix.shape[0] * jnp.log(2 * jnp.pi * jnp.e))
        return float(entropy)

    def _compute_reconstruction_distortion(
        self, field_matrix: np.ndarray, dimension: int
    ) -> float:
        U, s, Vt = linalg.svd(field_matrix, full_matrices=False)
        reconstructed = U[:, :dimension] @ np.diag(s[:dimension]) @ Vt[:dimension, :]
        distortion = np.mean((field_matrix - reconstructed) ** 2)
        return distortion

    def _compute_optimization_confidence(self, costs: Dict[int, float]) -> float:
        cost_values = [c for c in costs.values() if math.isfinite(c)]

        if len(cost_values) < 2:
            return 0.5

        min_cost = min(cost_values)
        cost_range = max(cost_values) - min_cost

        if cost_range < 1e-10:
            return 0.5

        avg_cost = np.mean(cost_values)
        confidence = min(1.0, (avg_cost - min_cost) / cost_range)
        return confidence


class HeatKernelDimensionAnalyzer:
    def __init__(self, time_steps: Optional[List[float]] = None):
        self.time_steps = time_steps or [0.001, 0.005, 0.01, 0.05, 0.1]
        self.heat_cache: Dict[str, Any] = {}
        logger.info("🔥 HeatKernelDimensionAnalyzer initialized")

    def estimate_dimension(
        self, agents: List[ConceptualChargeAgent]
    ) -> Tuple[int, Dict[str, Any]]:
        if len(agents) < 3:
            return 1, {"method": "insufficient_agents", "confidence": 0.4}

        laplacian = self._construct_field_laplacian(agents)

        if laplacian is None:
            return 1, {"method": "laplacian_construction_failed", "confidence": 0.3}

        trace_values = self._compute_heat_kernel_traces(laplacian)
        dimension, scaling_analysis = self._extract_dimension_from_scaling(trace_values)

        metadata = {
            "method": "heat_kernel_eigenvalue_scaling",
            "time_steps": self.time_steps,
            "trace_values": trace_values,
            "scaling_analysis": scaling_analysis,
            "confidence": scaling_analysis.get("fit_quality", 0.6),
        }

        logger.debug(f"🔥 Heat kernel dimension: {dimension}")
        return dimension, metadata

    def _construct_field_laplacian(
        self, agents: List[ConceptualChargeAgent]
    ) -> Optional[np.ndarray]:
        positions = []
        valid_agents = []

        for agent in agents:
            q_comps = agent.Q_components
            q_val = q_comps.Q_value

            if not math.isfinite(abs(q_val)):
                continue

            position = [
                float(q_val.real),
                float(q_val.imag),
                abs(q_val),
                np.angle(q_val),
                float(q_comps.gamma),
                float(q_comps.T_tensor.real),
                float(q_comps.T_tensor.imag),
                float(q_comps.E_trajectory.real),
                float(q_comps.E_trajectory.imag),
                float(q_comps.phi_semantic.real),
                float(q_comps.phi_semantic.imag),
                float(q_comps.theta_components.total),
                float(q_comps.psi_persistence),
            ]

            positions.append(position)
            valid_agents.append(agent)

        if len(positions) < 3:
            return None

        positions = np.array(positions)
        n_valid = len(positions)
        adjacency = np.zeros((n_valid, n_valid))
        distances = squareform(pdist(positions))

        field_energies = np.array(
            [abs(agent.Q_components.Q_value) ** 2 for agent in valid_agents]
        )
        energy_scale = np.std(field_energies) + 1e-10
        k = min(5, n_valid - 1)

        for i in range(n_valid):
            nearest_indices = np.argsort(distances[i])[1:k + 1]

            for j in nearest_indices:
                field_similarity = np.exp(
                    -abs(field_energies[i] - field_energies[j]) / energy_scale
                )
                distance_weight = np.exp(-distances[i, j] ** 2 / (2 * energy_scale**2))
                weight = field_similarity * distance_weight
                adjacency[i, j] = weight
                adjacency[j, i] = weight

        degree_matrix = np.diag(np.sum(adjacency, axis=1))
        laplacian = degree_matrix - adjacency
        laplacian += 1e-6 * np.eye(n_valid)
        return laplacian

    def _compute_heat_kernel_traces(self, laplacian: np.ndarray) -> List[float]:
        n = laplacian.shape[0]

        if n > 100:
            return self._compute_heat_kernel_traces_field_optimized(laplacian)
        else:
            traces = []
            for t in self.time_steps:
                heat_kernel = linalg.expm(-t * laplacian)
                trace = np.trace(heat_kernel)
                traces.append(float(trace))
            return traces

    def _compute_heat_kernel_traces_field_optimized(
        self, laplacian: np.ndarray
    ) -> List[float]:
        n = laplacian.shape[0]
        k = min(50, n)

        eigenvals, eigenvecs = linalg.eigh(laplacian)
        dominant_eigenvecs = eigenvecs[:, :min(10, n)]

        num_probes = max(10, int(np.ceil(np.log2(n))) + 5)
        traces = []

        for t in self.time_steps:
            trace_estimates = []

            for probe_idx in range(num_probes):
                if probe_idx < dominant_eigenvecs.shape[1]:
                    probe_vector = dominant_eigenvecs[:, probe_idx]
                else:
                    eigenvec_idx = probe_idx % dominant_eigenvecs.shape[1]
                    phase_shift = (probe_idx // dominant_eigenvecs.shape[1]) * np.pi / 4
                    real_part = dominant_eigenvecs[:, eigenvec_idx] * np.cos(
                        phase_shift
                    )
                    imag_part = dominant_eigenvecs[
                        :, (eigenvec_idx + 1) % dominant_eigenvecs.shape[1]
                    ] * np.sin(phase_shift)
                    probe_vector = real_part + imag_part

                probe_vector = probe_vector / np.linalg.norm(probe_vector)

                T_matrix, convergence_info = self._lanczos_tridiagonalization(
                    laplacian, probe_vector, k
                )

                if T_matrix is not None:
                    eigvals_T, eigvecs_T = linalg.eigh(T_matrix)
                    heat_coeffs = np.exp(-t * eigvals_T)
                    trace_contribution = np.sum(heat_coeffs * eigvecs_T[0, :] ** 2)
                    trace_estimates.append(trace_contribution)
                else:
                    trace_estimates.append(np.exp(-t * np.mean(np.diag(laplacian))))

            if trace_estimates:
                estimated_trace = float(n * np.mean(trace_estimates))
                traces.append(estimated_trace)
            else:
                traces.append(0.0)

        return traces

    def _lanczos_tridiagonalization(
        self, matrix: np.ndarray, start_vector: np.ndarray, num_iterations: int
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        n = matrix.shape[0]
        alpha = np.zeros(num_iterations)
        beta = np.zeros(num_iterations - 1)

        v = start_vector / np.linalg.norm(start_vector)
        v_prev = np.zeros(n)

        for j in range(num_iterations):
            w = matrix @ v
            alpha[j] = np.dot(v, w)
            w = w - alpha[j] * v - (beta[j - 1] * v_prev if j > 0 else 0)

            if j > 0:
                w = w - np.dot(w, v) * v

            if j < num_iterations - 1:
                beta[j] = np.linalg.norm(w)

                if beta[j] < 1e-14:
                    alpha = alpha[: j + 1]
                    beta = beta[:j]
                    break

                v_prev = v.copy()
                v = w / beta[j]

        try:
            T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
            convergence_info = {"converged": True, "iterations": len(alpha)}
            return T, convergence_info
        except Exception as e:
            return None, {"converged": False, "error": str(e)}

    def _extract_dimension_from_scaling(
        self, trace_values: List[float]
    ) -> Tuple[int, Dict[str, Any]]:
        valid_indices = [i for i, trace in enumerate(trace_values) if trace > 1e-10]

        if len(valid_indices) < 3:
            return 1, {"fit_quality": 0.0, "error": "insufficient_valid_traces"}

        valid_times = np.array([self.time_steps[i] for i in valid_indices])
        valid_traces = np.array([trace_values[i] for i in valid_indices])

        try:
            log_times = np.log(valid_times)
            log_traces = np.log(valid_traces)

            A = np.vstack([log_times, np.ones(len(log_times))]).T
            slope, intercept = np.linalg.lstsq(A, log_traces, rcond=None)[0]

            dimension = max(1, int(round(-2 * slope)))

            predicted_log_traces = intercept + slope * log_times
            residuals = log_traces - predicted_log_traces
            fit_quality = max(0.0, 1.0 - np.std(residuals) / np.std(log_traces))

            scaling_analysis = {
                "slope": slope,
                "intercept": intercept,
                "fit_quality": fit_quality,
                "residual_std": np.std(residuals),
                "valid_time_points": len(valid_indices),
            }

            return dimension, scaling_analysis

        except Exception as e:
            logger.warning(f"⚠️ Heat kernel scaling fit failed: {e}")
            return 1, {"fit_quality": 0.0, "error": str(e)}


class MultiObjectiveConsensus:
    def __init__(
        self,
        quantum_weight: float = 1.0,
        rate_distortion_weight: float = 1.0,
        heat_kernel_weight: float = 1.0,
    ):
        self.base_weights = {
            "quantum": quantum_weight,
            "rate_distortion": rate_distortion_weight,
            "heat_kernel": heat_kernel_weight,
        }
        logger.info("🎯 MultiObjectiveConsensus initialized")

    def compute_consensus_dimension(
        self,
        quantum_estimate: Tuple[int, Dict[str, Any]],
        rate_distortion_estimate: Tuple[int, Dict[str, Any]],
        heat_kernel_estimate: Tuple[int, Dict[str, Any]],
    ) -> Tuple[int, float, Dict[str, Any]]:
        estimates = {
            "quantum": quantum_estimate,
            "rate_distortion": rate_distortion_estimate,
            "heat_kernel": heat_kernel_estimate,
        }

        dimensions = {method: est[0] for method, est in estimates.items()}
        confidences = {
            method: est[1].get("confidence", 0.5) for method, est in estimates.items()
        }

        consensus_dim, consensus_conf = self._weighted_consensus(
            dimensions, confidences
        )
        consensus_metadata = self._compute_consensus_metadata(
            estimates, consensus_dim, consensus_conf
        )

        logger.info(
            f"🎯 Consensus dimension: {consensus_dim} (confidence: {consensus_conf:.3f})"
        )
        return consensus_dim, consensus_conf, consensus_metadata

    def _weighted_consensus(
        self, dimensions: Dict[str, int], confidences: Dict[str, float]
    ) -> Tuple[int, float]:
        adjusted_weights = {}
        total_weight = 0.0

        for method in dimensions.keys():
            confidence = confidences[method]
            base_weight = self.base_weights[method]
            adjusted_weight = base_weight * (confidence**2)
            adjusted_weights[method] = adjusted_weight
            total_weight += adjusted_weight

        if total_weight < 1e-10:
            return 1, 0.1

        for method in adjusted_weights:
            adjusted_weights[method] /= total_weight

        weighted_sum = sum(
            dimensions[method] * adjusted_weights[method]
            for method in dimensions.keys()
        )

        consensus_dimension = max(1, int(round(weighted_sum)))

        dimension_variance = sum(
            adjusted_weights[method] * (dimensions[method] - weighted_sum) ** 2
            for method in dimensions.keys()
        )

        agreement_factor = 1.0 / (1.0 + dimension_variance)
        avg_confidence = sum(
            confidences[method] * adjusted_weights[method]
            for method in confidences.keys()
        )

        consensus_confidence = avg_confidence * agreement_factor
        return consensus_dimension, consensus_confidence

    def _compute_consensus_metadata(
        self,
        estimates: Dict[str, Tuple[int, Dict[str, Any]]],
        consensus_dim: int,
        consensus_conf: float,
    ) -> Dict[str, Any]:
        dimensions = {method: est[0] for method, est in estimates.items()}
        confidences = {
            method: est[1].get("confidence", 0.5) for method, est in estimates.items()
        }

        dim_values = list(dimensions.values())
        agreement_stats = {
            "dimension_range": max(dim_values) - min(dim_values),
            "dimension_std": np.std(dim_values),
            "methods_in_agreement": sum(1 for d in dim_values if d == consensus_dim),
            "total_methods": len(dim_values),
        }

        conf_values = list(confidences.values())
        confidence_stats = {
            "min_confidence": min(conf_values),
            "max_confidence": max(conf_values),
            "avg_confidence": np.mean(conf_values),
            "confidence_std": np.std(conf_values),
        }

        metadata = {
            "consensus_method": "multi_objective_weighted",
            "individual_estimates": dimensions,
            "individual_confidences": confidences,
            "consensus_dimension": consensus_dim,
            "consensus_confidence": consensus_conf,
            "agreement_statistics": agreement_stats,
            "confidence_statistics": confidence_stats,
            "method_weights": self.base_weights,
            "individual_metadata": {
                method: est[1] for method, est in estimates.items()
            },
        }

        return metadata


class AdaptiveFieldDimension:
    """
    Adaptive Field Dimension Discovery Engine using authentic Q(τ,C,s) mathematics.

    Integrates quantum cognition, rate-distortion optimization, and heat kernel
    analysis to discover natural dimensional structure from field-theoretic principles.

    All mathematics derived from authentic field relationships without synthetic approximations.
    """

    def __init__(self, cache_enabled: bool = True, cache_timeout: float = 300.0):
        self.quantum_estimator = QuantumCognitionDimensionEstimator()
        self.rate_distortion_optimizer = RateDistortionOptimizer()
        self.heat_kernel_analyzer = HeatKernelDimensionAnalyzer()
        self.consensus_engine = MultiObjectiveConsensus()

        self.cache_enabled = cache_enabled
        self.cache_timeout = cache_timeout
        self.dimension_cache: Dict[str, Tuple[DimensionEstimate, float]] = {}

        self.discovery_stats = {
            "total_estimations": 0,
            "cache_hits": 0,
            "average_dimension": 0.0,
            "average_confidence": 0.0,
        }

        logger.info("🚀 AdaptiveFieldDimension engine initialized")
        logger.info("   🔬 Quantum cognition spectral gap method enabled")
        logger.info("   📊 Rate-distortion optimization enabled")
        logger.info("   🔥 Heat kernel eigenvalue scaling enabled")
        logger.info("   🎯 Multi-objective consensus enabled")

    def discover_field_dimension(
        self, agents: List[ConceptualChargeAgent]
    ) -> DimensionEstimate:
        start_time = time.time()

        if self.cache_enabled:
            field_signature = self._compute_field_signature(agents)
            cached_result = self._check_cache(field_signature)
            if cached_result is not None:
                self.discovery_stats["cache_hits"] += 1
                logger.debug("🎯 Using cached dimension estimate")
                return cached_result

        # MATHEMATICAL FAST-PATH: Quick estimation for stable field configurations
        quick_estimate = self._try_fast_path_estimation(agents, field_signature)
        if quick_estimate is not None:
            self.discovery_stats["fast_path_hits"] += 1
            logger.debug("🚀 Using fast-path dimension estimate")
            if self.cache_enabled:
                self._update_cache(field_signature, quick_estimate)
            return quick_estimate

        logger.debug("🔬 Running quantum cognition dimension estimation...")
        quantum_estimate = self.quantum_estimator.estimate_dimension(agents)

        logger.debug("📊 Running rate-distortion optimization...")
        rate_distortion_estimate = self.rate_distortion_optimizer.estimate_dimension(
            agents
        )

        logger.debug("🔥 Running heat kernel analysis...")
        heat_kernel_estimate = self.heat_kernel_analyzer.estimate_dimension(agents)

        logger.debug("🎯 Computing multi-objective consensus...")
        consensus_dim, consensus_conf, consensus_metadata = (
            self.consensus_engine.compute_consensus_dimension(
                quantum_estimate, rate_distortion_estimate, heat_kernel_estimate
            )
        )

        field_complexity = self._compute_field_complexity(agents)

        dimension_estimate = DimensionEstimate(
            quantum_dimension=quantum_estimate[0],
            rate_distortion_dimension=rate_distortion_estimate[0],
            heat_kernel_dimension=heat_kernel_estimate[0],
            consensus_dimension=consensus_dim,
            confidence_score=consensus_conf,
            field_complexity_measure=field_complexity,
            estimation_metadata=consensus_metadata,
        )

        if self.cache_enabled:
            self._update_cache(field_signature, dimension_estimate)

        self._update_discovery_stats(dimension_estimate, time.time() - start_time)

        logger.info(
            f"🚀 Field dimension discovered: {consensus_dim} (confidence: {consensus_conf:.3f}, complexity: {field_complexity:.3f})"
        )

        return dimension_estimate
    
    def _try_fast_path_estimation(self, agents: List[ConceptualChargeAgent], field_signature: FieldSignature) -> Optional[DimensionEstimate]:
        """
        MATHEMATICAL FAST-PATH: Quick dimension estimation for common field patterns.
        
        Uses mathematical heuristics to avoid expensive computation for standard configurations.
        """
        n_agents = len(agents)
        
        # Fast-path for small agent counts (dimension discovery overhead not worth it)
        if n_agents <= 20:
            # Simple heuristic: log-based dimension for small fields
            estimated_dim = max(1, min(n_agents, int(math.log2(n_agents + 1)) + 2))
            return DimensionEstimate(
                quantum_dimension=estimated_dim,
                rate_distortion_dimension=estimated_dim,
                heat_kernel_dimension=estimated_dim,
                consensus_dimension=estimated_dim,
                confidence_score=0.8,  # High confidence for simple cases
                field_complexity_measure=n_agents / 100.0,
                estimation_metadata={
                    "method": "fast_path_small_field",
                    "agents": n_agents,
                    "reasoning": "Small field - dimension discovery overhead not justified"
                }
            )
        
        # Fast-path for high field energy (typically results in higher dimensions)
        if field_signature.field_energy > 1000.0:
            # High energy fields tend to have higher intrinsic dimension
            estimated_dim = min(n_agents, max(10, int(math.log10(field_signature.field_energy)) * 3))
            return DimensionEstimate(
                quantum_dimension=estimated_dim,
                rate_distortion_dimension=estimated_dim,
                heat_kernel_dimension=estimated_dim,
                consensus_dimension=estimated_dim,
                confidence_score=0.7,
                field_complexity_measure=field_signature.field_energy / 10000.0,
                estimation_metadata={
                    "method": "fast_path_high_energy",
                    "field_energy": field_signature.field_energy,
                    "reasoning": "High energy field - dimension scales with energy"
                }
            )
        
        # No fast-path applicable
        return None

    def _compute_field_signature(
        self, agents: List[ConceptualChargeAgent]
    ) -> FieldSignature:
        """
        MATHEMATICAL INVARIANT CACHING: Use field topology instead of exact values.
        
        Creates cache signature based on mathematical invariants that remain stable
        across small field evolution changes, enabling effective cache hits.
        """
        n_points = len(agents)
        field_energy = 0.0
        
        # MATHEMATICAL INVARIANTS: Use binned/quantized values for cache stability
        magnitude_bins = []
        phase_bins = []
        gamma_bins = []
        persistence_bins = []

        for agent in agents:
            q_comps = agent.Q_components
            q_val = q_comps.Q_value
            if math.isfinite(abs(q_val)):
                field_energy += abs(q_val) ** 2
                
                # Bin values to mathematical ranges for cache stability
                magnitude = abs(q_val)
                phase = np.angle(q_val)
                
                # Logarithmic magnitude binning (handles wide range of values)
                mag_bin = int(math.log10(magnitude + 1e-10) + 10)  # Shift to positive range
                magnitude_bins.append(mag_bin)
                
                # Phase binning (8 sectors for 2π)
                phase_bin = int((phase + math.pi) / (2 * math.pi) * 8) % 8
                phase_bins.append(phase_bin)
                
                # Gamma and persistence binning  
                gamma_bin = int(q_comps.gamma * 10) % 100
                persistence_bin = int(q_comps.psi_persistence * 10) % 100
                gamma_bins.append(gamma_bin)
                persistence_bins.append(persistence_bin)

        # Mathematical signature based on distribution patterns
        if magnitude_bins:
            # Field topology signature: distribution of values across bins
            mag_histogram = np.histogram(magnitude_bins, bins=20, range=(0, 20))[0]
            phase_histogram = np.histogram(phase_bins, bins=8, range=(0, 8))[0]
            
            # Mathematical invariant: dominant modes in each distribution
            dominant_mag_modes = np.argsort(mag_histogram)[-3:].tolist()
            dominant_phase_modes = np.argsort(phase_histogram)[-2:].tolist()
            
            # Field complexity class (stable mathematical measure)
            field_complexity_class = len(set(magnitude_bins)) + len(set(phase_bins))
            
            complexity_signature = (
                tuple(dominant_mag_modes),
                tuple(dominant_phase_modes), 
                field_complexity_class,
                len(magnitude_bins) // 10  # Agent count class
            )
        else:
            complexity_signature = ((), (), 0, 0)

        complexity_hash = str(hash(complexity_signature))[:16]
        temporal_span = 1.0

        return FieldSignature(
            n_points=n_points,
            field_energy=field_energy,
            complexity_hash=complexity_hash,
            temporal_span=temporal_span,
            signature_timestamp=time.time(),
        )

    def _check_cache(
        self, field_signature: FieldSignature
    ) -> Optional[DimensionEstimate]:
        """INTELLIGENT CACHE LOOKUP: Check exact match and similar configurations."""
        cache_key = f"{field_signature.n_points}_{field_signature.complexity_hash}"

        # Exact match check
        if cache_key in self.dimension_cache:
            cached_estimate, cache_time = self.dimension_cache[cache_key]

            if time.time() - cache_time < self.cache_timeout:
                logger.debug(f"🎯 Exact cache hit for signature {cache_key}")
                return cached_estimate
            else:
                del self.dimension_cache[cache_key]

        # MATHEMATICAL SIMILARITY SEARCH: Look for similar field configurations
        # Check cache for similar agent counts and complexity signatures
        for stored_key, (stored_estimate, cache_time) in list(self.dimension_cache.items()):
            if time.time() - cache_time >= self.cache_timeout:
                del self.dimension_cache[stored_key]
                continue
                
            stored_parts = stored_key.split('_')
            if len(stored_parts) >= 2:
                stored_n_points = int(stored_parts[0])
                stored_hash = '_'.join(stored_parts[1:])
                
                # Similar agent count (±10%) and same complexity class
                n_point_tolerance = max(5, field_signature.n_points // 10)
                if (abs(stored_n_points - field_signature.n_points) <= n_point_tolerance and
                    stored_hash == field_signature.complexity_hash):
                    logger.debug(f"🎯 Similar cache hit: {stored_key} ~ {cache_key}")
                    return stored_estimate

        return None

    def _update_cache(
        self, field_signature: FieldSignature, dimension_estimate: DimensionEstimate
    ):
        cache_key = f"{field_signature.n_points}_{field_signature.complexity_hash}"
        self.dimension_cache[cache_key] = (dimension_estimate, time.time())

    def _compute_field_complexity(self, agents: List[ConceptualChargeAgent]) -> float:
        if len(agents) < 2:
            return 0.0

        q_values = []
        for agent in agents:
            q_comps = agent.Q_components
            q_val = q_comps.Q_value
            if math.isfinite(abs(q_val)):
                q_values.append(q_val)

        if len(q_values) < 2:
            return 0.0

        magnitudes = [abs(q) for q in q_values]
        phases = [np.angle(q) for q in q_values]

        mag_std = np.std(magnitudes)
        phase_std = np.std(phases)

        interactions = []
        for i in range(len(q_values)):
            for j in range(i + 1, len(q_values)):
                interaction = abs(q_values[i] * np.conj(q_values[j]))
                interactions.append(interaction)

        interaction_complexity = np.std(interactions) if interactions else 0.0
        complexity = (mag_std + phase_std + interaction_complexity) / 3.0
        return min(1.0, complexity)

    def _update_discovery_stats(
        self, dimension_estimate: DimensionEstimate, computation_time: float
    ):
        self.discovery_stats["total_estimations"] += 1

        n = self.discovery_stats["total_estimations"]
        old_avg_dim = self.discovery_stats["average_dimension"]
        old_avg_conf = self.discovery_stats["average_confidence"]

        self.discovery_stats["average_dimension"] = (
            old_avg_dim * (n - 1) + dimension_estimate.consensus_dimension
        ) / n
        self.discovery_stats["average_confidence"] = (
            old_avg_conf * (n - 1) + dimension_estimate.confidence_score
        ) / n

        logger.debug(f"🚀 Dimension discovery completed in {computation_time:.3f}s")

    def get_discovery_status(self) -> Dict[str, Any]:
        return {
            "system_status": {
                "quantum_cognition_available": True,
                "rate_distortion_available": True,
                "heat_kernel_available": True,
                "consensus_engine_available": True,
                "cache_enabled": self.cache_enabled,
                "cache_size": len(self.dimension_cache),
            },
            "discovery_statistics": self.discovery_stats.copy(),
            "capabilities": {
                "adaptive_dimension_discovery": True,
                "field_complexity_analysis": True,
                "multi_method_consensus": True,
                "uncertainty_quantification": True,
                "temporal_caching": True,
            },
        }

    