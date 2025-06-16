"""
Temporal Trajectory Operators - Dynamic Movement Through Observational States

Mathematical Reference: Section 3.1.4.3 - Reconstruction of Temporal Positional Encoding
Key Formula: Tᵢ(τ,s) = ∫₀ˢ ωᵢ(τ,s')·e^(iφᵢ(τ,s')) ds'

Documentation: See /models/temporal_dimension/README.md
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from numbers import Complex
from scipy.integrate import quad, quad_vec, complex_ode
from scipy.special import expit
import logging

logger = logging.getLogger(__name__)


class TrajectoryOperator:
    """
    Base class for trajectory integration T_i(τ,C,s) = ∫₀ˢ ω_i(τ,s')·e^(iφ_i(τ,s')) ds'
    
    Mathematical Foundation:
    - From static sinusoidal PE to dynamic trajectory integration (Section 3.1.4.3.2)
    - Observational persistence and layered memory (Section 3.1.4.3.3)
    - Trajectory-semantic field coupling (Section 3.1.4.3.4)
    """
    
    def __init__(self, embedding_dimension: int):
        """
        Initialize trajectory operator base class.
        
        Args:
            embedding_dimension: Dimension of embedding space (1024 for BGE, 768 for MPNet)
        """
        self.embedding_dimension = embedding_dimension
        logger.info(f"Initialized TrajectoryOperator for {embedding_dimension}D space")
    
    def compute_trajectory_integral(self,
                                  token: str,
                                  context: str,
                                  observational_state: Union[float, complex],
                                  semantic_embedding: Optional[np.ndarray] = None) -> complex:
        """
        Base method for trajectory integration - to be implemented by subclasses.
        
        Mathematical Formula: T_i(τ,C,s) = ∫₀ˢ ω_i(τ,s')·e^(iφ_i(τ,s')) ds'
        
        Args:
            token: Token τ for trajectory computation
            context: Context C for trajectory computation
            observational_state: Current observational state s
            semantic_embedding: Optional semantic vector for modulation
            
        Returns:
            Complex trajectory integral value
        """
        raise NotImplementedError("Subclasses must implement compute_trajectory_integral")


class FrequencyEvolution:
    """
    Manages ω_i(τ,s') evolution functions for dynamic frequency modulation.
    
    Mathematical Foundation: ω_i(τ,s') = ω_base,i + semantic_modulation(τ,s') + context_influence(C,s')
    """
    
    def __init__(self, embedding_dimension: int):
        """Initialize frequency evolution manager."""
        self.embedding_dimension = embedding_dimension
        self.base_frequencies = self._compute_adaptive_frequencies_from_theory(embedding_dimension)
        self.frequency_evolution_history = []
        self.context_frequency_adaptations = {}
        self.experience_modulation_factors = np.ones(embedding_dimension, dtype=complex)
    
    def _compute_adaptive_frequencies_from_theory(self, embedding_dimension: int) -> np.ndarray:
        """
        Compute adaptive frequencies from mathematical formulations.
        
        Uses harmonic series, golden ratio, and mathematical constants
        for sophisticated frequency patterns as required by CLAUDE.md.
        """
        frequencies = np.zeros(embedding_dimension, dtype=complex)
        
        for i in range(embedding_dimension):
            # Base frequency from transformer-inspired pattern (but made complex)
            base_freq = 1.0 / (10000.0 ** (2.0 * i / embedding_dimension))
            
            # Multi-scale frequency bands for different semantic timescales
            if i < embedding_dimension // 4:
                # Fast scale (immediate semantic processing)
                scale_factor = 2.0 + 0.5 * np.sin(2 * np.pi * i / (embedding_dimension // 4))
            elif i < embedding_dimension // 2:
                # Medium scale (contextual semantic integration)
                scale_factor = 1.0 + 0.3 * np.cos(2 * np.pi * i / (embedding_dimension // 2))
            else:
                # Slow scale (long-term semantic memory)
                scale_factor = 0.5 + 0.2 * np.sin(2 * np.pi * i / embedding_dimension)
            
            # Golden ratio harmonic relationships for natural resonance
            golden_ratio = (1 + np.sqrt(5)) / 2
            golden_modulation = 1 + 0.1 * np.sin(2 * np.pi * i / (golden_ratio * 100))
            
            # Mathematical constant modulation (π, e, √2) for rich harmonic content
            pi_modulation = 1 + 0.05 * np.cos(2 * np.pi * i / (np.pi * 50))
            e_modulation = 1 + 0.03 * np.sin(2 * np.pi * i / (np.e * 50))
            sqrt2_modulation = 1 + 0.02 * np.cos(2 * np.pi * i / (np.sqrt(2) * 50))
            
            # Complex frequency with imaginary component for phase coupling
            real_part = base_freq * scale_factor * golden_modulation * pi_modulation * e_modulation * sqrt2_modulation
            imag_part = 0.1 * base_freq * scale_factor * np.sin(2 * np.pi * i / embedding_dimension)
            
            frequencies[i] = complex(real_part, imag_part)
        
        return frequencies
    
    def evolve_frequency(self,
                        dimension: int,
                        token: str,
                        context: str,
                        observational_state: Union[float, complex],
                        semantic_embedding: Optional[np.ndarray] = None) -> complex:
        """
        Compute evolved frequency ω_i(τ,s') for specific dimension and state.
        
        Mathematical Formula: ω_i(τ,s') = ω_base,i + semantic_modulation(τ,s') + context_influence(C,s')
        
        Args:
            dimension: Dimension index i
            token: Token τ for semantic modulation
            context: Context C for context influence
            observational_state: Current state s'
            semantic_embedding: Optional semantic vector
            
        Returns:
            Evolved frequency for this dimension
        """
        if dimension >= self.embedding_dimension:
            raise ValueError(f"Dimension {dimension} exceeds embedding_dimension {self.embedding_dimension}. CLAUDE.md prohibits fallback values.")
        
        # Base frequency for this dimension
        base_freq = self.base_frequencies[dimension]
        
        # Semantic modulation based on token content
        semantic_mod = self._compute_semantic_modulation(token, dimension, semantic_embedding)
        
        # Context influence
        context_influence = self._compute_context_influence(context, dimension, observational_state)
        
        # Experience-based adaptation
        experience_factor = self.experience_modulation_factors[dimension]
        
        # Combined frequency evolution
        evolved_frequency = base_freq + semantic_mod + context_influence
        evolved_frequency *= experience_factor
        
        return evolved_frequency
    
    def _compute_semantic_modulation(self, token: str, dimension: int, semantic_embedding: Optional[np.ndarray]) -> complex:
        """Compute semantic modulation component."""
        # Hash-based deterministic modulation
        token_hash = hash(token) % 1000 / 1000.0
        
        # Semantic embedding influence
        embedding_influence = complex(0, 0)
        if semantic_embedding is not None and dimension < len(semantic_embedding):
            embedding_strength = semantic_embedding[dimension]
            embedding_influence = 0.1 * embedding_strength * np.exp(1j * token_hash * np.pi)
        
        # Token-specific frequency shift
        token_modulation = 0.05 * np.sin(2 * np.pi * token_hash + dimension * np.pi / self.embedding_dimension)
        
        return complex(token_modulation, 0) + embedding_influence
    
    def _compute_context_influence(self, context: str, dimension: int, observational_state: Union[float, complex]) -> complex:
        """Compute context influence component."""
        context_hash = hash(context) % 1000 / 1000.0
        
        # State-dependent context modulation
        if np.iscomplexobj(observational_state):
            state_magnitude = np.abs(observational_state)
            state_phase = np.angle(observational_state)
        else:
            state_magnitude = abs(observational_state)
            state_phase = 0.0
        
        # Context influence with state dependency
        context_strength = 0.1 * np.sin(2 * np.pi * context_hash * state_magnitude)
        context_phase = context_hash * np.pi + state_phase
        
        return context_strength * np.exp(1j * context_phase)


class PhaseAccumulator:
    """
    Handles φ_i(τ,s') accumulation for complex phase evolution.
    
    Mathematical Foundation: φ_i(τ,s') = φ_initial,i + ∫₀ˢ' Δφ_i(τ,u) du
    """
    
    def __init__(self, embedding_dimension: int):
        """Initialize phase accumulator."""
        self.embedding_dimension = embedding_dimension
        self.phase_accumulation = np.zeros(embedding_dimension, dtype=complex)
        self.phase_history = []
    
    def accumulate_phase(self,
                        dimension: int,
                        token: str,
                        context: str,
                        observational_state: Union[float, complex],
                        frequency: complex) -> complex:
        """
        Accumulate phase for dimension i: φ_i(τ,s') = φ_initial,i + ∫₀ˢ' Δφ_i(τ,u) du
        
        Args:
            dimension: Dimension index i
            token: Token τ for phase computation
            context: Context C for phase modulation
            observational_state: Current state s'
            frequency: Current frequency ω_i for phase evolution
            
        Returns:
            Accumulated phase for this dimension
        """
        if dimension >= self.embedding_dimension:
            raise ValueError(f"Dimension {dimension} exceeds embedding_dimension {self.embedding_dimension}. CLAUDE.md prohibits fallback values.")
        
        # Phase change based on frequency and state
        if np.iscomplexobj(observational_state):
            state_real = np.real(observational_state)
        else:
            state_real = float(observational_state)
        
        # Phase accumulation: Δφ = ω * Δs
        phase_delta = frequency * state_real
        
        # Content-dependent phase shifts
        token_phase_shift = self._compute_token_phase_shift(token, dimension)
        context_phase_shift = self._compute_context_phase_shift(context, dimension, observational_state)
        
        # Update accumulated phase
        total_phase_delta = phase_delta + token_phase_shift + context_phase_shift
        self.phase_accumulation[dimension] += total_phase_delta
        
        return self.phase_accumulation[dimension]
    
    def _compute_token_phase_shift(self, token: str, dimension: int) -> complex:
        """Compute token-dependent phase shift."""
        token_hash = hash(token) % 1000 / 1000.0
        phase_shift = 0.1 * np.sin(2 * np.pi * token_hash + dimension * np.pi / self.embedding_dimension)
        return complex(0, phase_shift)
    
    def _compute_context_phase_shift(self, context: str, dimension: int, observational_state: Union[float, complex]) -> complex:
        """Compute context-dependent phase shift."""
        context_hash = hash(context) % 1000 / 1000.0
        
        if np.iscomplexobj(observational_state):
            state_influence = np.abs(observational_state)
        else:
            state_influence = abs(observational_state)
        
        phase_shift = 0.05 * np.cos(2 * np.pi * context_hash * state_influence)
        return complex(0, phase_shift)


class ComplexIntegrator:
    """
    Performs ∫₀ˢ complex exponential integration for trajectory operators.
    
    Mathematical Foundation: T_i(τ,C,s) = ∫₀ˢ ω_i(τ,s')·e^(iφ_i(τ,s')) ds'
    """
    
    def __init__(self, integration_method: str):
        """
        Initialize complex integrator.
        
        Args:
            integration_method: Method for numerical integration ("adaptive_quad", "fixed_quad", etc.)
        """
        self.integration_method = integration_method
        self.integration_cache = {}
    
    def integrate_complex_trajectory(self,
                                   frequency_func: Callable[[float], complex],
                                   phase_func: Callable[[float], complex],
                                   observational_state: Union[float, complex]) -> complex:
        """
        Perform complex integration: ∫₀ˢ ω_i(τ,s')·e^(iφ_i(τ,s')) ds'
        
        Args:
            frequency_func: Function ω_i(τ,s') returning frequency at state s'
            phase_func: Function φ_i(τ,s') returning phase at state s'
            observational_state: Upper limit s for integration
            
        Returns:
            Complex trajectory integral T_i(τ,C,s)
        """
        # Use real part of observational_state for integration limits
        if np.iscomplexobj(observational_state):
            upper_limit = np.real(observational_state)
        else:
            upper_limit = float(observational_state)
        
        if upper_limit <= 0:
            return complex(0, 0)
        
        # Define complex integrand: ω_i(τ,s') * e^(iφ_i(τ,s'))
        def complex_integrand(s_prime):
            freq = frequency_func(s_prime)
            phase = phase_func(s_prime)
            return freq * np.exp(1j * phase)
        
        # Separate real and imaginary parts for scipy integration
        def real_integrand(s_prime):
            return np.real(complex_integrand(s_prime))
        
        def imag_integrand(s_prime):
            return np.imag(complex_integrand(s_prime))
        
        try:
            # Integrate real and imaginary parts separately
            real_result, _ = quad(real_integrand, 0, upper_limit)
            imag_result, _ = quad(imag_integrand, 0, upper_limit)
            
            trajectory_integral = complex(real_result, imag_result)
            
            # Validate result
            if not np.isfinite(np.real(trajectory_integral)) or not np.isfinite(np.imag(trajectory_integral)):
                raise ValueError(f"Non-finite trajectory integral. CLAUDE.md prohibits fallback values.")
            
            return trajectory_integral
            
        except Exception as e:
            logger.error(f"Complex integration failed: {e}")
            raise RuntimeError(f"Complex trajectory integration required. CLAUDE.md prohibits fallback values. Error: {e}")


class TrajectoryOperatorEngine(TrajectoryOperator):
    """
    Complete trajectory operator engine combining all components.
    
    Reuses existing sophisticated implementations while following modular architecture.
    """
    
    def __init__(self,
                 embedding_dimension: int,
                 base_frequencies: np.ndarray,
                 integration_method: str):
        """
        Initialize complete trajectory operator engine.
        
        Args:
            embedding_dimension: Dimension of embedding space
            base_frequencies: Base frequencies for each dimension (computed from theory)
            integration_method: Method for numerical integration
        """
        super().__init__(embedding_dimension)
        
        # Initialize modular components
        self.frequency_evolution = FrequencyEvolution(embedding_dimension)
        self.frequency_evolution.base_frequencies = np.array(base_frequencies, dtype=complex)
        self.phase_accumulator = PhaseAccumulator(embedding_dimension)
        self.complex_integrator = ComplexIntegrator(integration_method)
        
        # Coupling matrix for cross-dimensional interactions
        self.coupling_matrix = self._initialize_coupling_matrix()
        
        logger.info(f"Initialized TrajectoryOperatorEngine with modular architecture")
    
    def _initialize_coupling_matrix(self) -> np.ndarray:
        """Initialize coupling matrix for cross-dimensional phase coordination."""
        # Start with identity (no coupling) and add small off-diagonal terms
        coupling = np.eye(self.embedding_dimension)
        
        # Add nearest-neighbor coupling (inspired by physical systems)
        for i in range(self.embedding_dimension - 1):
            coupling[i, i+1] = 0.1
            coupling[i+1, i] = 0.1
            
        return coupling
    
    def compute_trajectory_integral(self,
                                  token: str,
                                  context: str,
                                  observational_state: Union[float, complex],
                                  semantic_embedding: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute complete trajectory integration using modular components.
        
        Returns comprehensive trajectory data for all dimensions.
        """
        trajectory_operators = np.zeros(self.embedding_dimension, dtype=complex)
        phase_accumulation = np.zeros(self.embedding_dimension, dtype=complex)
        frequency_evolution = np.zeros(self.embedding_dimension, dtype=complex)
        
        for dimension in range(self.embedding_dimension):
            # Get evolved frequency for this dimension
            freq_func = lambda s_prime: self.frequency_evolution.evolve_frequency(
                dimension, token, context, s_prime, semantic_embedding
            )
            
            # Accumulate phase for this dimension
            current_freq = self.frequency_evolution.evolve_frequency(
                dimension, token, context, observational_state, semantic_embedding
            )
            accumulated_phase = self.phase_accumulator.accumulate_phase(
                dimension, token, context, observational_state, current_freq
            )
            
            # Define phase function
            phase_func = lambda s_prime: accumulated_phase * (s_prime / np.real(observational_state) if np.real(observational_state) != 0 else 0)
            
            # Perform complex integration
            trajectory_integral = self.complex_integrator.integrate_complex_trajectory(
                freq_func, phase_func, observational_state
            )
            
            trajectory_operators[dimension] = trajectory_integral
            phase_accumulation[dimension] = accumulated_phase
            frequency_evolution[dimension] = current_freq
        
        # Compute additional trajectory metrics
        transformative_potential = np.mean(np.abs(trajectory_operators))
        transformative_potential_tensor = trajectory_operators.reshape((-1, 1)) @ trajectory_operators.reshape((1, -1))
        
        return {
            'trajectory_operators': trajectory_operators,
            'phase_accumulation': phase_accumulation,
            'frequency_evolution': frequency_evolution,
            'transformative_potential': transformative_potential,
            'transformative_potential_tensor': transformative_potential_tensor,
            'phase_coordination': {
                'coherence': np.abs(np.mean(np.exp(1j * np.angle(trajectory_operators)))),
                'mean_phase': np.mean(np.angle(trajectory_operators))
            }
        }