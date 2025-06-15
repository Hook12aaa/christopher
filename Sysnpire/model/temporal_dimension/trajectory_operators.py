"""
Temporal Trajectory Operators - Dynamic Movement Through Observational States

Mathematical Reference: Section 3.1.4.3 - Reconstruction of Temporal Positional Encoding
Key Formula: Tᵢ(τ,s) = ∫₀ˢ ωᵢ(τ,s')·e^(iφᵢ(τ,s')) ds'

Documentation: See /models/temporal_dimension/README.md
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.integrate import quad, quad_vec
from scipy.special import expit
import logging

logger = logging.getLogger(__name__)


class TrajectoryOperatorEngine:
    """
    Transforms static positional encodings into dynamic trajectory operators.
    
    Mathematical Foundation:
    - From static sinusoidal PE to dynamic trajectory integration (Section 3.1.4.3.2)
    - Observational persistence and layered memory (Section 3.1.4.3.3)
    - Trajectory-semantic field coupling (Section 3.1.4.3.4)
    - Developmental distance and interpolation (Section 3.1.4.3.9)
    """
    
    def __init__(self, embedding_dimension: int = 1024, 
                 base_frequencies: Optional[np.ndarray] = None,
                 integration_method: str = "adaptive_quad"):
        """
        Initialize trajectory operator engine.
        
        Mathematical Components:
        - ωᵢ(τ,s'): Instantaneous frequency functions
        - φᵢ(τ,s'): Accumulated phase relationships  
        - Ψ_persistence: Dual-decay persistence functions
        - Coupling tensors for semantic integration
        
        Args:
            embedding_dimension: Dimension of embedding space (1024 for BGE, 768 for MPNet)
            base_frequencies: Base frequencies for each dimension (auto-generated if None)
            integration_method: Method for numerical integration
        """
        self.embedding_dimension = embedding_dimension
        self.integration_method = integration_method
        
        # Initialize base frequencies if not provided
        if base_frequencies is None:
            # Create frequencies inspired by original transformer PE
            # but adapted for dynamic evolution
            self.base_frequencies = np.array([
                1.0 / (10000.0 ** (2.0 * i / embedding_dimension))
                for i in range(embedding_dimension)
            ])
        else:
            self.base_frequencies = base_frequencies
            
        # Initialize phase accumulation tracking
        self.phase_accumulation = np.zeros(embedding_dimension)
        
        # Persistence parameters (from Section 3.1.4.3.3)
        self.gaussian_sigma = 0.3
        self.exponential_lambda = 0.1
        self.cosine_beta = 2.0
        self.persistence_alpha = 0.4
        
        # Coupling matrix for cross-dimensional interactions
        self.coupling_matrix = self._initialize_coupling_matrix()
        
        logger.info(f"Initialized TrajectoryOperatorEngine with {embedding_dimension}D space")
    
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
                                  observational_state: float,
                                  semantic_embedding: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute trajectory operator Tᵢ(τ,s) for all dimensions.
        
        Mathematical Reference: Section 3.1.4.3.2
        Formula: Tᵢ(τ,s) = ∫₀ˢ ωᵢ(τ,s')·e^(iφᵢ(τ,s')) ds'
        
        Process:
        1. Define frequency evolution ωᵢ(τ,s') for each dimension
        2. Track phase accumulation φᵢ(τ,s') across trajectory
        3. Integrate complex exponential across observational path
        4. Handle numerical integration for continuous evolution
        
        Key Insight: Captures movement patterns, not just position!
        
        Args:
            token: Token τ for which to compute trajectory
            context: Context C influencing trajectory evolution
            observational_state: Current observational state s
            semantic_embedding: Optional semantic vector for modulation
            
        Returns:
            Dictionary containing trajectory operators and transformative analysis:
            - 'trajectory_operators': Complex T_i(τ,s) array for all dimensions
            - 'transformative_magnitude': |T_i(τ,s)| - strength of transformative potential
            - 'frequency_evolution': ω_i(τ,s') patterns across trajectory
            - 'phase_accumulation': φ_i(τ,s') evolution through observational states
            - 'semantic_modulation': How semantic content shapes trajectory
            - 'context_influence': How context C affects transformative potential
        """
        T_operators = np.zeros(self.embedding_dimension, dtype=complex)
        frequency_patterns = np.zeros(self.embedding_dimension)
        phase_patterns = np.zeros(self.embedding_dimension)
        semantic_modulation_factors = np.zeros(self.embedding_dimension)
        
        # Hash token and context for deterministic but varied behavior
        token_hash = hash(token) % 1000 / 1000.0
        context_hash = hash(context) % 1000 / 1000.0
        
        for i in range(self.embedding_dimension):
            # Define frequency evolution function ω_i(τ,s')
            def omega_i(s_prime):
                base_freq = self.base_frequencies[i]
                
                # Semantic modulation (if embedding provided)
                semantic_mod = 1.0
                if semantic_embedding is not None and i < len(semantic_embedding):
                    semantic_mod = 1.0 + 0.1 * np.tanh(semantic_embedding[i])
                
                # Context influence
                context_influence = 0.1 * np.sin(2 * np.pi * context_hash * s_prime)
                
                # Token-specific modulation
                token_influence = 0.05 * np.cos(2 * np.pi * token_hash * s_prime)
                
                return base_freq * semantic_mod * (1 + context_influence + token_influence)
            
            # Define phase evolution function φ_i(τ,s')
            def phi_i(s_prime):
                # Initial phase based on token/dimension
                initial_phase = 2 * np.pi * (token_hash + i / self.embedding_dimension)
                
                # Accumulated phase changes
                phase_drift = 0.1 * s_prime * (1 + 0.1 * np.sin(s_prime))
                
                return initial_phase + phase_drift
            
            # Complex integrand: ω_i(τ,s') * e^(iφ_i(τ,s'))
            def integrand_real(s_prime):
                return omega_i(s_prime) * np.cos(phi_i(s_prime))
            
            def integrand_imag(s_prime):
                return omega_i(s_prime) * np.sin(phi_i(s_prime))
            
            # Perform integration from 0 to s
            try:
                real_part, _ = quad(integrand_real, 0, observational_state)
                imag_part, _ = quad(integrand_imag, 0, observational_state)
                T_operators[i] = complex(real_part, imag_part)
                
                # Capture transformative characteristics
                frequency_patterns[i] = omega_i(observational_state)  # Current frequency
                phase_patterns[i] = phi_i(observational_state)        # Current phase
                
                # Semantic modulation strength
                if semantic_embedding is not None and i < len(semantic_embedding):
                    semantic_modulation_factors[i] = 0.1 * np.abs(np.tanh(semantic_embedding[i]))
                else:
                    semantic_modulation_factors[i] = 0.0
                    
            except Exception as e:
                logger.warning(f"Integration failed for dimension {i}: {e}")
                # Fallback to simple approximation
                T_operators[i] = complex(
                    observational_state * self.base_frequencies[i],
                    0.1 * observational_state
                )
                frequency_patterns[i] = self.base_frequencies[i]
                phase_patterns[i] = 0.0
                semantic_modulation_factors[i] = 0.0
        
        # Compute overall transformative characteristics
        transformative_magnitudes = np.abs(T_operators)
        context_influence = 0.1 * context_hash * np.sum(transformative_magnitudes)
        
        return {
            'trajectory_operators': T_operators,
            'transformative_magnitude': transformative_magnitudes,
            'frequency_evolution': frequency_patterns,
            'phase_accumulation': phase_patterns,
            'semantic_modulation': semantic_modulation_factors,
            'context_influence': context_influence,
            'total_transformative_potential': np.mean(transformative_magnitudes)
        }
    
    def generate_observational_persistence(self, 
                                         observational_distance: float,
                                         dimension: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Generate layered persistence function Ψ_persistence(s-s₀).
        
        Mathematical Reference: Section 3.1.4.3.3
        Formula: Ψ = exp(-(s-s₀)²/2σ²) + α·exp(-λ(s-s₀))·cos(β(s-s₀))
        
        Dual-decay structure:
        1. Gaussian: "vivid recent chapters" (sharp, immediate memory)
        2. Exponential-cosine: "persistent character traits" (rhythmic reinforcement)
        
        This creates "layered narrative memory" - different persistence timescales
        
        Args:
            observational_distance: Distance s-s₀ in observational space
            dimension: Optional specific dimension (returns array if None)
            
        Returns:
            Dual-decay persistence value(s)
        """
        # Ensure non-negative distance
        distance = abs(observational_distance)
        
        # Gaussian component - vivid recent memory
        gaussian_component = np.exp(-(distance**2) / (2 * self.gaussian_sigma**2))
        
        # Exponential-cosine component - persistent traits
        exp_cos_component = (self.persistence_alpha * 
                           np.exp(-self.exponential_lambda * distance) * 
                           np.cos(self.cosine_beta * distance))
        
        # Combined dual-decay persistence
        persistence = gaussian_component + exp_cos_component
        
        if dimension is not None:
            # Modulate by dimension-specific factors
            dim_factor = 1.0 + 0.1 * np.sin(dimension * np.pi / self.embedding_dimension)
            return persistence * dim_factor
        else:
            # Return array for all dimensions with slight variations
            dim_factors = np.array([
                1.0 + 0.1 * np.sin(i * np.pi / self.embedding_dimension)
                for i in range(self.embedding_dimension)
            ])
            return persistence * dim_factors
    
    def couple_with_semantic_fields(self,
                                   trajectory_operators: np.ndarray,
                                   semantic_field: np.ndarray,
                                   breathing_modulation: float = 1.0) -> np.ndarray:
        """
        Couple trajectory operators with semantic fields.
        
        Mathematical Reference: Section 3.1.4.3.4
        Formula: Φ^semantic(τ,s) = w_i * T_i(τ,s) * x[i] * breathing_modulation * e^(iθ)
        
        Args:
            trajectory_operators: T_i(τ,s) complex trajectory operators
            semantic_field: Semantic embedding or field values
            breathing_modulation: Rhythmic expansion/contraction factor
            
        Returns:
            Coupled semantic-temporal field
        """
        # Ensure arrays have compatible shapes
        if len(trajectory_operators) != len(semantic_field):
            min_len = min(len(trajectory_operators), len(semantic_field))
            trajectory_operators = trajectory_operators[:min_len]
            semantic_field = semantic_field[:min_len]
        
        # Apply coupling with breathing modulation
        coupled_field = np.zeros_like(trajectory_operators, dtype=complex)
        
        for i in range(len(coupled_field)):
            # Weight factors (could be learned)
            w_i = 1.0 / (1.0 + i)  # Decreasing importance with dimension
            
            # Complex coupling with semantic field
            coupled_field[i] = (w_i * trajectory_operators[i] * 
                              semantic_field[i] * breathing_modulation)
        
        # Apply coupling matrix for cross-dimensional interactions
        if len(coupled_field) == self.embedding_dimension:
            coupled_field = self.coupling_matrix @ coupled_field
        
        return coupled_field
    
    def compute_developmental_distance(self,
                                     state_1: float,
                                     state_2: float,
                                     token: str,
                                     context: str,
                                     semantic_embedding: Optional[np.ndarray] = None) -> float:
        """
        Compute developmental distance between observational states.
        
        Mathematical Reference: Section 3.1.4.3.5
        Formula: d_D(s₁,s₂) = Σᵢ |∫_{s₁}^{s₂} ω_i(τ,s')ds'| · w_i · Ψ_i(s₂-s₁)
        
        Measures transformative activity, not chronological separation!
        
        Args:
            state_1: First observational state
            state_2: Second observational state
            token: Token for trajectory computation
            context: Context for trajectory computation
            semantic_embedding: Optional semantic vector
            
        Returns:
            Developmental distance (transformative activity measure)
        """
        # Compute trajectory change between states
        T_1 = self.compute_trajectory_integral(token, context, state_1, semantic_embedding)
        T_2 = self.compute_trajectory_integral(token, context, state_2, semantic_embedding)
        
        # Trajectory difference represents accumulated transformation
        trajectory_diff = np.abs(T_2 - T_1)
        
        # Get persistence factors for the distance
        persistence_factors = self.generate_observational_persistence(state_2 - state_1)
        
        # Weight factors (decreasing with dimension)
        weights = np.array([1.0 / (1.0 + i) for i in range(len(trajectory_diff))])
        
        # Developmental distance: weighted sum of trajectory changes
        # modulated by persistence factors
        d_D = np.sum(trajectory_diff * weights * persistence_factors[:len(trajectory_diff)])
        
        return float(np.real(d_D))
    
    def generate_breathing_pattern(self,
                                  observational_state: float,
                                  base_frequency: float = 0.1,
                                  depth: float = 0.2) -> float:
        """
        Generate breathing modulation pattern for semantic field coupling.
        
        Creates rhythmic expansion/contraction based on observational state.
        
        Args:
            observational_state: Current observational state
            base_frequency: Base breathing frequency
            depth: Modulation depth (0-1)
            
        Returns:
            Breathing modulation factor
        """
        # Multiple frequency components for complex breathing
        primary = np.cos(base_frequency * observational_state)
        harmonic = 0.3 * np.cos(3 * base_frequency * observational_state)
        subharmonic = 0.2 * np.sin(0.5 * base_frequency * observational_state)
        
        # Combined breathing pattern
        breathing = 1.0 + depth * (primary + harmonic + subharmonic)
        
        return breathing
    
    def compute_phase_coordination(self,
                                  trajectory_operators: np.ndarray,
                                  observational_state: float) -> Dict[str, np.ndarray]:
        """
        Compute orchestral phase coordination across dimensions.
        
        Mathematical Reference: Section 3.1.4.3.8
        Formula: θ_orchestral,i(s) = ∫₀ˢ ω_i(τ,s') ds' + Σⱼ coupling_ij · θⱼ(s')
        
        Args:
            trajectory_operators: Complex trajectory operators
            observational_state: Current observational state
            
        Returns:
            Dictionary with phase angles, coherence measures, and interference patterns
        """
        # Extract phases from trajectory operators
        phases = np.angle(trajectory_operators)
        
        # Apply coupling matrix for cross-dimensional coordination
        if len(phases) == self.embedding_dimension:
            coupled_phases = self.coupling_matrix @ phases
        else:
            coupled_phases = phases
        
        # Compute phase coherence (how aligned are the phases)
        mean_phase = np.mean(np.exp(1j * phases))
        coherence = np.abs(mean_phase)  # 0 = random, 1 = perfectly aligned
        
        # Identify constructive/destructive interference regions
        phase_diffs = np.outer(phases, np.ones_like(phases)) - np.outer(np.ones_like(phases), phases)
        constructive_pairs = np.sum(np.abs(np.cos(phase_diffs)) > 0.8) / 2
        destructive_pairs = np.sum(np.abs(np.cos(phase_diffs)) < 0.2) / 2
        
        return {
            'phases': phases,
            'coupled_phases': coupled_phases,
            'coherence': coherence,
            'mean_phase': np.angle(mean_phase),
            'constructive_pairs': int(constructive_pairs),
            'destructive_pairs': int(destructive_pairs),
            'phase_variance': np.var(phases)
        }