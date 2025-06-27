"""
Variational Regulation - JAX-Based Field Energy Optimization

MATHEMATICAL FOUNDATION: Implements variational regulation using JAX automatic 
differentiation to optimize field energy functionals. All regulation emerges from
minimizing the total field energy functional:

E_total = ∫ (field_instability + λ·regulation_cost) dx

CORE PHILOSOPHY: Regulation parameters are determined through gradient descent
on energy functionals rather than heuristic rules. This ensures all regulation
decisions emerge from mathematical optimization principles.

JAX INTEGRATION: Uses JAX for:
- Just-in-time compilation for performance
- Automatic differentiation for gradient computation
- Vectorized operations for efficient field computations
- Functional programming paradigm for mathematical clarity

FIELD THEORY COMPLIANCE: All optimization respects the Q(τ, C, s) formula
structure and maintains field-theoretic consistency.
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, jit
import optax

from Sysnpire.model.liquid.conceptual_charge_agent import ConceptualChargeAgent
from Sysnpire.model.liquid.regulation.listeners import InformationMetrics, RegulationSuggestion
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VariationalRegulationParams:
    """Parameters for variational regulation optimization."""

    lambda_coupling: float = 0.1  # Coupling strength between Q-field and regulation
    lambda_cost: float = 0.01  # Regulation energy cost weighting
    learning_rate: float = 0.01  # Optimization learning rate
    max_iterations: int = 100  # Maximum optimization iterations
    convergence_tolerance: float = 1e-6  # Convergence threshold
    stability_penalty: float = 10.0  # Penalty for unstable configurations


@dataclass
class EnergyFunctionalComponents:
    """Components of the field energy functional."""

    field_instability: float  # Instability measure
    regulation_cost: float  # Cost of applying regulation
    total_energy: float  # Total energy functional value
    gradient_norm: float  # Norm of energy gradient
    optimization_step: int  # Current optimization step


class VariationalRegulation:
    """
    JAX-Based Variational Regulation System

    Implements field energy optimization using variational principles.
    All regulation decisions emerge from minimizing total field energy.
    """

    def __init__(self, params: Optional[VariationalRegulationParams] = None):
        """
        Initialize variational regulation system.

        Args:
            params: Regulation parameters (uses defaults if None)
        """
        self.params = params or VariationalRegulationParams()

        self.optimizer = optax.adam(learning_rate=self.params.learning_rate)
        self.opt_state = None
        logger.info("🔧 JAX variational regulation initialized")

        self.energy_history: List[EnergyFunctionalComponents] = []
        self.regulation_parameters_cache: Optional[jnp.ndarray] = None

        self.optimization_stats = {
            "total_optimizations": 0,
            "average_iterations": 0.0,
            "convergence_rate": 0.0,
            "energy_reduction_ratio": 0.0,
        }

        logger.info("🌊 VariationalRegulation initialized with JAX")
        logger.info(f"   Lambda coupling: {self.params.lambda_coupling}")
        logger.info(f"   Lambda cost: {self.params.lambda_cost}")
        logger.info(f"   Learning rate: {self.params.learning_rate}")

    def _compile_energy_functional(self) -> Callable:
        """
        Compile JAX energy functional for performance.

        Returns:
            JIT-compiled energy functional
        """

        @jit
        def energy_functional(
            regulation_params: jnp.ndarray, field_state: jnp.ndarray, agent_states: jnp.ndarray
        ) -> float:
            """
            Total field energy functional:
            E = field_instability + λ_coupling * regulation_coupling + λ_cost * regulation_cost
            """
            field_instability = self._compute_field_instability_jax(field_state, agent_states)

            regulation_coupling = self._compute_regulation_coupling_jax(regulation_params, field_state, agent_states)

            regulation_cost = self._compute_regulation_cost_jax(regulation_params)

            total_energy = (
                field_instability
                + self.params.lambda_coupling * regulation_coupling
                + self.params.lambda_cost * regulation_cost
            )

            return total_energy

        return energy_functional

    def _compute_field_instability_jax(self, field_state: jnp.ndarray, agent_states: jnp.ndarray) -> float:
        """
        Compute field instability measure using JAX.

        Instability comes from:
        1. High field energy concentration
        2. Phase decoherence
        3. Rapid field variations
        """
        q_magnitudes = jnp.abs(field_state)

        energy_density = q_magnitudes**2
        concentration_instability = jnp.var(energy_density) / (jnp.mean(energy_density) + 1e-12)

        q_phases = jnp.angle(field_state)
        phase_coherence = jnp.abs(jnp.mean(jnp.exp(1j * q_phases)))
        phase_instability = 1.0 - phase_coherence

        if len(field_state) > 1:
            field_gradients = jnp.diff(field_state)
            variation_instability = jnp.mean(jnp.abs(field_gradients))
        else:
            variation_instability = 0.0

        total_instability = concentration_instability + phase_instability + 0.1 * variation_instability

        return total_instability

    def _compute_regulation_coupling_jax(
        self, regulation_params: jnp.ndarray, field_state: jnp.ndarray, agent_states: jnp.ndarray
    ) -> float:
        """
        Compute coupling energy between regulation and Q-field.

        This measures how well regulation parameters couple with field state.
        """
        regulation_modulation = regulation_params * jnp.abs(field_state)

        coupling_energy = jnp.mean(regulation_modulation**2)

        over_regulation_penalty = jnp.mean(jnp.maximum(0, regulation_params - 1.0) ** 2)

        return coupling_energy + self.params.stability_penalty * over_regulation_penalty

    def _compute_regulation_cost_jax(self, regulation_params: jnp.ndarray) -> float:
        """
        Compute cost of applying regulation.

        Regulation should be minimal but effective.
        """
        l2_cost = jnp.sum(regulation_params**2)

        if len(regulation_params) > 1:
            smoothness_penalty = jnp.sum(jnp.diff(regulation_params) ** 2)
        else:
            smoothness_penalty = 0.0

        return l2_cost + 0.1 * smoothness_penalty

    def _numpy_energy_functional(
        self, regulation_params: np.ndarray, field_state: np.ndarray, agent_states: np.ndarray
    ) -> float:
        """
        NumPy fallback implementation of energy functional.
        """
        reg_params = np.array(regulation_params)
        field = np.array(field_state)
        agents = np.array(agent_states)

        q_magnitudes = np.abs(field)
        energy_density = q_magnitudes**2
        concentration_instability = np.var(energy_density) / (np.mean(energy_density) + 1e-12)

        q_phases = np.angle(field)
        phase_coherence = np.abs(np.mean(np.exp(1j * q_phases)))
        phase_instability = 1.0 - phase_coherence

        field_instability = concentration_instability + phase_instability

        regulation_modulation = reg_params * q_magnitudes
        coupling_energy = np.mean(regulation_modulation**2)

        regulation_cost = np.sum(reg_params**2)

        total_energy = (
            field_instability
            + self.params.lambda_coupling * coupling_energy
            + self.params.lambda_cost * regulation_cost
        )

        return float(total_energy)

    def optimize_regulation_parameters(
        self, agents: List[ConceptualChargeAgent]
    ) -> Tuple[jnp.ndarray, EnergyFunctionalComponents]:
        """
        Optimize regulation parameters using variational principles.

        Args:
            agents: List of conceptual charge agents

        Returns:
            Tuple of (optimal_regulation_params, energy_components)
        """
        optimization_start = time.time()

        field_state, agent_states = self._extract_field_state(agents)

        if len(field_state) == 0:
            logger.warning("⚠️ No valid field state for optimization")
            return jnp.array([]), EnergyFunctionalComponents(0, 0, 0, 0, 0)

        if self.regulation_parameters_cache is None:
            initial_params = jnp.ones(len(field_state)) * 0.1  # Small initial regulation
        else:
            initial_params = self.regulation_parameters_cache

        energy_func = self._compile_energy_functional()

        optimal_params, energy_components = self._jax_optimize(
            energy_func, initial_params, field_state, agent_states
        )

        self.regulation_parameters_cache = optimal_params

        optimization_time = time.time() - optimization_start
        self._update_optimization_stats(energy_components, optimization_time)

        logger.info(f"🔧 Variational optimization completed in {optimization_time:.4f}s")
        logger.info(f"   Final energy: {energy_components.total_energy:.6f}")
        logger.info(f"   Gradient norm: {energy_components.gradient_norm:.6f}")
        logger.info(f"   Optimization steps: {energy_components.optimization_step}")

        return optimal_params, energy_components

    def _jax_optimize(
        self, energy_func: Callable, initial_params: jnp.ndarray, field_state: jnp.ndarray, agent_states: jnp.ndarray
    ) -> Tuple[jnp.ndarray, EnergyFunctionalComponents]:
        """
        Perform JAX-based optimization using automatic differentiation.
        """
        if self.opt_state is None:
            self.opt_state = self.optimizer.init(initial_params)

        params = initial_params

        grad_func = jit(grad(energy_func))

        for step in range(self.params.max_iterations):
            energy_value = energy_func(params, field_state, agent_states)
            grads = grad_func(params, field_state, agent_states)

            gradient_norm = jnp.linalg.norm(grads)
            if gradient_norm < self.params.convergence_tolerance:
                logger.debug(f"🎯 Converged at step {step}, gradient norm: {gradient_norm:.8f}")
                break

            updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
            params = optax.apply_updates(params, updates)

            if step % 20 == 0:
                logger.debug(f"   Step {step}: energy={energy_value:.6f}, |grad|={gradient_norm:.6f}")

        final_energy = energy_func(params, field_state, agent_states)
        final_grads = grad_func(params, field_state, agent_states)
        final_gradient_norm = jnp.linalg.norm(final_grads)

        field_instability = self._compute_field_instability_jax(field_state, agent_states)
        regulation_cost = self._compute_regulation_cost_jax(params)

        energy_components = EnergyFunctionalComponents(
            field_instability=float(field_instability),
            regulation_cost=float(regulation_cost),
            total_energy=float(final_energy),
            gradient_norm=float(final_gradient_norm),
            optimization_step=step + 1,
        )

        return params, energy_components

    def _numpy_optimize(
        self, energy_func: Callable, initial_params: np.ndarray, field_state: np.ndarray, agent_states: np.ndarray
    ) -> Tuple[np.ndarray, EnergyFunctionalComponents]:
        """
        NumPy fallback optimization using simple gradient descent.
        """
        params = initial_params.copy()

        for step in range(self.params.max_iterations):
            energy_value = energy_func(params, field_state, agent_states)
            grads = self._compute_numerical_gradient(energy_func, params, field_state, agent_states)

            gradient_norm = np.linalg.norm(grads)
            if gradient_norm < self.params.convergence_tolerance:
                logger.debug(f"🎯 NumPy converged at step {step}, gradient norm: {gradient_norm:.8f}")
                break

            params -= self.params.learning_rate * grads

            if step % 20 == 0:
                logger.debug(f"   NumPy Step {step}: energy={energy_value:.6f}, |grad|={gradient_norm:.6f}")

        final_energy = energy_func(params, field_state, agent_states)

        energy_components = EnergyFunctionalComponents(
            field_instability=final_energy * 0.7,  # Approximate breakdown
            regulation_cost=final_energy * 0.3,
            total_energy=final_energy,
            gradient_norm=gradient_norm,
            optimization_step=step + 1,
        )

        return params, energy_components

    def _compute_numerical_gradient(
        self,
        func: Callable,
        params: np.ndarray,
        field_state: np.ndarray,
        agent_states: np.ndarray,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute numerical gradient using finite differences.
        """
        grads = np.zeros_like(params)

        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon

            f_plus = func(params_plus, field_state, agent_states)
            f_minus = func(params_minus, field_state, agent_states)

            grads[i] = (f_plus - f_minus) / (2 * epsilon)

        return grads

    def _extract_field_state(self, agents: List[ConceptualChargeAgent]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Extract field state arrays from conceptual charge agents.

        Returns:
            Tuple of (Q_field_values, agent_state_features)
        """
        q_values = []
        agent_features = []

        for agent in agents:
            if hasattr(agent, "Q_components") and agent.Q_components is not None:
                q_val = agent.Q_components.Q_value
                if q_val is not None and math.isfinite(abs(q_val)):
                    q_values.append(complex(q_val))

                    features = [
                        abs(q_val),  # Q magnitude
                        np.angle(q_val),  # Q phase
                        agent.field_state.field_position[0] if hasattr(agent, "field_state") else 0.0,
                        agent.field_state.field_position[1] if hasattr(agent, "field_state") else 0.0,
                    ]
                    agent_features.append(features)

        if not q_values:
            return jnp.array([]), jnp.array([])

        field_state = jnp.array(q_values)
        agent_state = jnp.array(agent_features)

        return field_state, agent_state

    def _update_optimization_stats(self, energy_components: EnergyFunctionalComponents, optimization_time: float):
        """Update optimization performance statistics."""
        self.optimization_stats["total_optimizations"] += 1

        total_opts = self.optimization_stats["total_optimizations"]
        current_avg = self.optimization_stats["average_iterations"]
        new_iterations = energy_components.optimization_step
        self.optimization_stats["average_iterations"] = (current_avg * (total_opts - 1) + new_iterations) / total_opts

        converged = energy_components.gradient_norm < self.params.convergence_tolerance
        current_rate = self.optimization_stats["convergence_rate"]
        self.optimization_stats["convergence_rate"] = (
            current_rate * (total_opts - 1) + (1.0 if converged else 0.0)
        ) / total_opts

        self.energy_history.append(energy_components)
        if len(self.energy_history) > 2:
            initial_energy = self.energy_history[-2].total_energy
            final_energy = energy_components.total_energy
            reduction_ratio = (initial_energy - final_energy) / (initial_energy + 1e-12)
            self.optimization_stats["energy_reduction_ratio"] = reduction_ratio

    def apply_variational_regulation(
        self, agents: List[ConceptualChargeAgent], current_interaction_strength: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Apply variational regulation to field interaction strength.

        Args:
            agents: List of conceptual charge agents
            current_interaction_strength: Current field interaction strength

        Returns:
            Tuple of (regulated_strength, regulation_metrics)
        """
        start_time = time.time()

        optimal_params, energy_components = self.optimize_regulation_parameters(agents)

        if len(optimal_params) == 0:
            return current_interaction_strength, {
                "variational_regulation_applied": False,
                "reason": "No valid field state",
            }

        regulation_strength = float(jnp.mean(optimal_params))
        regulation_factor = 1.0 - regulation_strength

        regulation_factor = max(0.01, min(1.0, regulation_factor))

        regulated_strength = current_interaction_strength * regulation_factor

        regulation_time = time.time() - start_time

        regulation_metrics = {
            "variational_regulation_applied": True,
            "regulation_factor": regulation_factor,
            "regulation_strength": regulation_strength,
            "energy_functional": {
                "total_energy": energy_components.total_energy,
                "field_instability": energy_components.field_instability,
                "regulation_cost": energy_components.regulation_cost,
                "gradient_norm": energy_components.gradient_norm,
                "optimization_steps": energy_components.optimization_step,
            },
            "optimization_stats": self.optimization_stats.copy(),
            "regulation_time": regulation_time,
            "jax_enabled": True,
            "parameters_optimized": len(optimal_params),
        }

        logger.info(f"🔧 Variational regulation applied in {regulation_time:.4f}s")
        logger.info(f"   Regulation factor: {regulation_factor:.3f}")
        logger.info(f"   Energy functional: {energy_components.total_energy:.6f}")
        logger.info(f"   Optimization steps: {energy_components.optimization_step}")

        return regulated_strength, regulation_metrics

    def get_regulation_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of variational regulation system.

        Returns:
            Dictionary with system status and performance metrics
        """
        return {
            "system_status": {
                "jax_available": True,
                "parameters": {
                    "lambda_coupling": self.params.lambda_coupling,
                    "lambda_cost": self.params.lambda_cost,
                    "learning_rate": self.params.learning_rate,
                    "max_iterations": self.params.max_iterations,
                    "convergence_tolerance": self.params.convergence_tolerance,
                },
                "cache_status": {
                    "parameters_cached": self.regulation_parameters_cache is not None,
                    "energy_history_length": len(self.energy_history),
                },
            },
            "performance_stats": self.optimization_stats.copy(),
            "recent_energy_history": [
                {
                    "total_energy": comp.total_energy,
                    "field_instability": comp.field_instability,
                    "regulation_cost": comp.regulation_cost,
                    "optimization_steps": comp.optimization_step,
                }
                for comp in self.energy_history[-5:]  # Last 5 optimizations
            ],
        }
