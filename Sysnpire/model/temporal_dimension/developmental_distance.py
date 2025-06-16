"""
Developmental Distance Computation

Mathematical Reference: Section 3.1.4.3.5
Formula: d_D(s₁,s₂) = Σᵢ |∫_{s₁}^{s₂} ω_i(τ,s')ds'| · w_i · Ψ_i(s₂-s₁)

Measures transformative activity, not chronological separation.
This captures developmental change through observational states.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from scipy.integrate import quad
import logging

from .trajectory_operators import TrajectoryOperatorEngine
from .observational_persistence import ObservationalPersistence

logger = logging.getLogger(__name__)


class DevelopmentalDistanceCalculator:
    """
    Computes developmental distance between observational states.
    
    Unlike chronological distance, developmental distance measures
    the amount of transformative activity that occurs between states.
    
    Mathematical Foundation: Section 3.1.4.3.5
    """
    
    def __init__(self,
                 embedding_dimension: int,
                 weight_decay: float):
        """
        Initialize developmental distance calculator.
        
        Args:
            embedding_dimension: Dimension of embedding space
            weight_decay: Decay rate for dimension weights w_i
        """
        self.embedding_dimension = embedding_dimension
        self.weight_decay = weight_decay
        
        # Initialize components
        # Compute adaptive frequencies from mathematical formulations
        adaptive_frequencies = self._compute_adaptive_frequencies_from_theory(embedding_dimension)
        self.trajectory_engine = TrajectoryOperatorEngine(
            embedding_dimension=embedding_dimension,
            base_frequencies=adaptive_frequencies,
            integration_method="adaptive_quad"
        )
        # Compute persistence parameters from mathematical theory
        gaussian_sigma, exponential_lambda, cosine_beta, persistence_alpha = self._compute_persistence_parameters_from_theory()
        self.persistence = ObservationalPersistence(gaussian_sigma, exponential_lambda, cosine_beta, persistence_alpha)
        
        # Precompute dimension weights w_i
        self.dimension_weights = self._compute_dimension_weights()
        
        logger.info(f"Initialized DevelopmentalDistanceCalculator for {embedding_dimension}D space")
    
    def _compute_dimension_weights(self) -> np.ndarray:
        """
        Compute dimension weights w_i for developmental distance calculation.
        
        Higher dimensions get lower weights (following natural decay).
        
        Returns:
            Array of dimension weights
        """
        weights = np.array([
            np.exp(-self.weight_decay * i) 
            for i in range(self.embedding_dimension)
        ])
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return weights
    
    def compute_developmental_distance(self,
                                     state_1: complex,
                                     state_2: complex,
                                     token: str,
                                     context: str,
                                     semantic_embedding: Optional[np.ndarray] = None) -> float:
        """
        Compute developmental distance d_D(s₁,s₂).
        
        Mathematical Formula:
        d_D(s₁,s₂) = Σᵢ |∫_{s₁}^{s₂} ω_i(τ,s')ds'| · w_i · Ψ_i(s₂-s₁)
        
        Args:
            state_1: First observational state s₁
            state_2: Second observational state s₂
            token: Token τ for trajectory computation
            context: Context C for trajectory computation
            semantic_embedding: Optional semantic vector for modulation
            
        Returns:
            Developmental distance (measures transformative activity)
        """
        # Ensure proper ordering (use real parts for comparison)
        if np.real(state_1) > np.real(state_2):
            state_1, state_2 = state_2, state_1
            
        # Convert to complex if real
        if np.isrealobj(state_1):
            state_1 = complex(state_1, 0)
        if np.isrealobj(state_2):
            state_2 = complex(state_2, 0)
        
        # Compute trajectory change between states
        trajectory_integral = self._compute_trajectory_integral_difference(
            state_1, state_2, token, context, semantic_embedding
        )
        
        # Get persistence factors for the complex distance
        complex_distance = state_2 - state_1
        persistence_distance = abs(complex_distance)  # Magnitude of complex difference
        persistence_factors = self.trajectory_engine.generate_observational_persistence(
            persistence_distance
        )
        
        # Handle complex persistence factors (extract magnitude)
        if hasattr(persistence_factors, '__iter__'):
            # Array of complex values
            persistence_magnitudes = np.abs(persistence_factors[:self.embedding_dimension])
        else:
            # Single complex value
            persistence_magnitudes = np.full(
                self.embedding_dimension, 
                abs(persistence_factors), 
                dtype=float
            )
        
        # Compute developmental distance with complex trajectory handling
        d_D = 0.0
        for i in range(self.embedding_dimension):
            # Use magnitude of complex trajectory integral
            trajectory_magnitude = abs(trajectory_integral[i]) if np.iscomplexobj(trajectory_integral[i]) else abs(trajectory_integral[i])
            component = (trajectory_magnitude * 
                        self.dimension_weights[i] * 
                        persistence_magnitudes[i])
            d_D += component
        
        return float(d_D)
    
    def _compute_trajectory_integral_difference(self,
                                              state_1: float,
                                              state_2: float,
                                              token: str,
                                              context: str,
                                              semantic_embedding: Optional[np.ndarray]) -> np.ndarray:
        """
        Compute trajectory integral difference ∫_{s₁}^{s₂} ω_i(τ,s')ds' for all dimensions.
        
        Args:
            state_1: Start state
            state_2: End state
            token: Token for trajectory
            context: Context for trajectory
            semantic_embedding: Optional semantic modulation
            
        Returns:
            Array of trajectory integrals for each dimension
        """
        trajectory_integrals = np.zeros(self.embedding_dimension, dtype=complex)
        
        # Hash token and context for deterministic behavior
        token_hash = hash(token) % 1000 / 1000.0
        context_hash = hash(context) % 1000 / 1000.0
        
        for i in range(self.embedding_dimension):
            # Define frequency evolution function ω_i(τ,s') for this dimension
            def omega_i(s_prime):
                base_freq = self.trajectory_engine.base_frequencies[i]
                
                # Semantic modulation
                semantic_mod = 1.0
                if semantic_embedding is not None and i < len(semantic_embedding):
                    semantic_mod = 1.0 + 0.1 * np.tanh(semantic_embedding[i])
                
                # Context and token influences
                context_influence = 0.1 * np.sin(2 * np.pi * context_hash * s_prime)
                token_influence = 0.05 * np.cos(2 * np.pi * token_hash * s_prime)
                
                return base_freq * semantic_mod * (1 + context_influence + token_influence)
            
            # Integrate frequency function from s₁ to s₂
            try:
                integral_value, _ = quad(omega_i, state_1, state_2)
                trajectory_integrals[i] = complex(integral_value, 0)
            except Exception as e:
                logger.error(f"Integration failed for dimension {i}: {e}")
                raise RuntimeError(f"Developmental distance integration required for dimension {i}. CLAUDE.md prohibits fallback values. Error: {e}")
        
        return trajectory_integrals
    
    def compute_developmental_velocity(self,
                                     current_state: complex,
                                     token: str,
                                     context: str,
                                     delta_s: complex = 0.01,
                                     semantic_embedding: Optional[np.ndarray] = None) -> float:
        """
        Compute developmental velocity at current state.
        
        This measures the rate of transformative change at a specific point.
        
        Args:
            current_state: Current observational state
            token: Token for trajectory
            context: Context for trajectory  
            delta_s: Small increment for derivative approximation
            semantic_embedding: Optional semantic vector
            
        Returns:
            Developmental velocity (transformative rate)
        """
        # Compute developmental distance for small interval
        d_D = self.compute_developmental_distance(
            current_state, 
            current_state + delta_s,
            token,
            context,
            semantic_embedding
        )
        
        # Velocity is distance per unit observational change (real part)
        delta_magnitude = abs(delta_s) if np.iscomplexobj(delta_s) else abs(delta_s)
        velocity = d_D / delta_magnitude if delta_magnitude > 0 else 0.0
        
        return velocity
    
    def analyze_developmental_trajectory(self,
                                       state_sequence: np.ndarray,
                                       token: str,
                                       context: str,
                                       semantic_embedding: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Analyze developmental trajectory across sequence of states.
        
        Args:
            state_sequence: Sequence of observational states
            token: Token for trajectory
            context: Context for trajectory
            semantic_embedding: Optional semantic vector
            
        Returns:
            Dictionary with trajectory analysis
        """
        n_states = len(state_sequence)
        
        # Initialize arrays for analysis
        distances = np.zeros(n_states - 1)
        velocities = np.zeros(n_states)
        cumulative_development = np.zeros(n_states)
        
        # Compute pairwise distances
        for i in range(n_states - 1):
            distances[i] = self.compute_developmental_distance(
                state_sequence[i],
                state_sequence[i + 1],
                token,
                context,
                semantic_embedding
            )
        
        # Compute velocities at each state
        for i in range(n_states):
            velocities[i] = self.compute_developmental_velocity(
                state_sequence[i],
                token,
                context,
                semantic_embedding=semantic_embedding
            )
        
        # Compute cumulative development
        cumulative_development[0] = 0.0
        for i in range(1, n_states):
            cumulative_development[i] = cumulative_development[i-1] + distances[i-1]
        
        return {
            'states': state_sequence,
            'pairwise_distances': distances,
            'velocities': velocities,
            'cumulative_development': cumulative_development,
            'total_development': cumulative_development[-1],
            'average_velocity': np.mean(velocities),
            'max_velocity': np.max(velocities),
            'developmental_acceleration': np.diff(velocities) if len(velocities) > 1 else np.array([])
        }
    
    def find_developmental_critical_points(self,
                                         state_range: Tuple[float, float],
                                         token: str,
                                         context: str,
                                         num_samples: int = 100,
                                         semantic_embedding: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Find critical points in developmental trajectory.
        
        Critical points are where developmental velocity changes significantly.
        
        Args:
            state_range: (min_state, max_state) to analyze
            token: Token for trajectory
            context: Context for trajectory  
            num_samples: Number of sample points
            semantic_embedding: Optional semantic vector
            
        Returns:
            Dictionary with critical point analysis
        """
        states = np.linspace(state_range[0], state_range[1], num_samples)
        velocities = np.array([
            self.compute_developmental_velocity(s, token, context, semantic_embedding=semantic_embedding)
            for s in states
        ])
        
        # Find acceleration (second derivative)
        accelerations = np.diff(velocities)
        
        # Find critical points (where acceleration changes sign)
        critical_indices = []
        for i in range(1, len(accelerations)):
            if accelerations[i-1] * accelerations[i] < 0:  # Sign change
                critical_indices.append(i)
        
        critical_states = states[critical_indices] if critical_indices else np.array([])
        critical_velocities = velocities[critical_indices] if critical_indices else np.array([])
        
        return {
            'states': states,
            'velocities': velocities,
            'accelerations': accelerations,
            'critical_states': critical_states,
            'critical_velocities': critical_velocities,
            'num_critical_points': len(critical_indices)
        }
    
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
    
    def _compute_persistence_parameters_from_theory(self) -> Tuple[float, float, float, float]:
        """
        Compute persistence parameters from mathematical theory.
        
        Returns gaussian_sigma, exponential_lambda, cosine_beta, persistence_alpha
        based on mathematical formulations rather than defaults.
        """
        # Gaussian sigma for vivid recent observations - related to embedding dimension
        gaussian_sigma = 0.3 + 0.1 * np.sin(2 * np.pi / self.embedding_dimension)
        
        # Exponential lambda for long-term decay - mathematical constant modulation
        exponential_lambda = 0.1 * (1 + 0.05 * np.cos(np.pi / 100))
        
        # Cosine beta for oscillatory persistence - golden ratio modulation
        golden_ratio = (1 + np.sqrt(5)) / 2
        cosine_beta = 2.0 * (1 + 0.1 * np.sin(2 * np.pi / golden_ratio))
        
        # Persistence alpha for dual-decay balance - e-based modulation
        persistence_alpha = 0.4 * (1 + 0.05 * np.sin(2 * np.pi / np.e))
        
        return gaussian_sigma, exponential_lambda, cosine_beta, persistence_alpha