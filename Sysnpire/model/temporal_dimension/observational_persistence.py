"""
Observational Persistence - Layered Memory Structure

Mathematical Reference: Section 3.1.4.3.3
Formula: Ψ_persistence(s-s₀) = exp(-(s-s₀)²/2σ²) + α·exp(-λ(s-s₀))·cos(β(s-s₀))

This implements the dual-decay memory structure that captures both:
- Vivid recent observations (Gaussian decay)
- Persistent semantic traits (Exponential-cosine decay)
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class PersistenceLayer(ABC):
    """
    Base class for memory decay functions implementing different persistence mechanisms.
    
    Mathematical Foundation: Abstract layer for implementing different memory decay patterns
    that contribute to the overall observational persistence Ψ_persistence(s-s₀).
    """
    
    def __init__(self, embedding_dimension: int):
        """
        Initialize persistence layer base class.
        
        Args:
            embedding_dimension: Dimension of embedding space for multi-dimensional persistence
        """
        self.embedding_dimension = embedding_dimension
        logger.info(f"Initialized {self.__class__.__name__} for {embedding_dimension}D space")
    
    @abstractmethod
    def compute_persistence(self,
                          observational_distance: Union[float, complex],
                          **kwargs) -> Union[float, complex]:
        """
        Compute persistence value for given observational distance.
        
        Args:
            observational_distance: Distance s-s₀ in observational space
            **kwargs: Additional parameters specific to persistence type
            
        Returns:
            Persistence value for this layer
        """
        pass


class GaussianMemory(PersistenceLayer):
    """
    Implements vivid recent memory component: exp(-(s-s₀)²/2σ²)
    
    Mathematical Foundation: Gaussian decay for sharp, detailed memory of recent observations
    with fast decay for immediate episodic memory and high precision for current context.
    """
    
    def __init__(self, embedding_dimension: int, gaussian_sigma: float):
        """
        Initialize Gaussian memory layer.
        
        Args:
            embedding_dimension: Dimension of embedding space
            gaussian_sigma: Standard deviation σ for Gaussian decay
        """
        super().__init__(embedding_dimension)
        self.gaussian_sigma = gaussian_sigma
        
        # Dimension-specific sigma variations for rich memory patterns
        self.sigma_variations = self._compute_sigma_variations()
    
    def _compute_sigma_variations(self) -> np.ndarray:
        """
        Compute dimension-specific sigma variations for multi-dimensional persistence.
        
        Different dimensions may have different memory characteristics based on
        their semantic roles (fast vs slow timescales).
        """
        variations = np.zeros(self.embedding_dimension)
        
        for i in range(self.embedding_dimension):
            # Multi-scale memory variations
            if i < self.embedding_dimension // 4:
                # Fast scale - sharper memory (smaller sigma)
                scale_factor = 0.8
            elif i < self.embedding_dimension // 2:
                # Medium scale - balanced memory
                scale_factor = 1.0
            else:
                # Slow scale - broader memory (larger sigma)
                scale_factor = 1.2
            
            # Mathematical constant modulation for natural patterns
            golden_ratio = (1 + np.sqrt(5)) / 2
            modulation = 1 + 0.1 * np.sin(2 * np.pi * i / (golden_ratio * 100))
            
            variations[i] = self.gaussian_sigma * scale_factor * modulation
        
        return variations
    
    def compute_persistence(self,
                          observational_distance: Union[float, complex],
                          dimension: Optional[int] = None) -> Union[float, complex]:
        """
        Compute Gaussian persistence: exp(-(s-s₀)²/2σ²)
        
        Args:
            observational_distance: Distance s-s₀ in observational space
            dimension: Optional specific dimension for dimension-aware persistence
            
        Returns:
            Gaussian persistence value
        """
        # Handle complex observational distances
        if np.iscomplexobj(observational_distance):
            distance_magnitude = np.abs(observational_distance)
            distance_phase = np.angle(observational_distance)
        else:
            distance_magnitude = abs(observational_distance)
            distance_phase = 0.0
        
        # Use dimension-specific sigma if provided
        if dimension is not None and dimension < self.embedding_dimension:
            sigma = self.sigma_variations[dimension]
        else:
            sigma = self.gaussian_sigma
        
        # Gaussian decay computation
        gaussian_magnitude = np.exp(-(distance_magnitude**2) / (2 * sigma**2))
        
        # For complex distances, include phase information
        if np.iscomplexobj(observational_distance):
            # Phase modulation for complex Gaussian persistence
            phase_modulation = np.exp(1j * 0.1 * distance_phase)
            return gaussian_magnitude * phase_modulation
        else:
            return gaussian_magnitude
    
    def compute_multi_dimensional_persistence(self, observational_distance: Union[float, complex]) -> np.ndarray:
        """
        Compute Gaussian persistence for all dimensions simultaneously.
        
        Args:
            observational_distance: Distance s-s₀ in observational space
            
        Returns:
            Array of persistence values for each dimension
        """
        persistence_values = np.zeros(self.embedding_dimension, dtype=complex)
        
        for i in range(self.embedding_dimension):
            persistence_values[i] = self.compute_persistence(observational_distance, dimension=i)
        
        return persistence_values


class ExponentialCosineMemory(PersistenceLayer):
    """
    Implements persistent traits component: α·exp(-λ(s-s₀))·cos(β(s-s₀))
    
    Mathematical Foundation: Exponential-cosine decay for long-term semantic impressions
    with oscillatory memory and slow decay that captures enduring semantic relationships.
    """
    
    def __init__(self,
                 embedding_dimension: int,
                 exponential_lambda: float,
                 cosine_beta: float,
                 persistence_alpha: float):
        """
        Initialize exponential-cosine memory layer.
        
        Args:
            embedding_dimension: Dimension of embedding space
            exponential_lambda: Decay rate λ for exponential component
            cosine_beta: Frequency β for cosine oscillations
            persistence_alpha: Weight α for persistent component
        """
        super().__init__(embedding_dimension)
        self.exponential_lambda = exponential_lambda
        self.cosine_beta = cosine_beta
        self.persistence_alpha = persistence_alpha
        
        # Dimension-specific parameter variations
        self.lambda_variations = self._compute_lambda_variations()
        self.beta_variations = self._compute_beta_variations()
        self.alpha_variations = self._compute_alpha_variations()
    
    def _compute_lambda_variations(self) -> np.ndarray:
        """Compute dimension-specific exponential decay rates."""
        variations = np.zeros(self.embedding_dimension)
        
        for i in range(self.embedding_dimension):
            # Different decay rates for different semantic scales
            if i < self.embedding_dimension // 4:
                # Fast scale - faster decay
                scale_factor = 1.5
            elif i < self.embedding_dimension // 2:
                # Medium scale - standard decay
                scale_factor = 1.0
            else:
                # Slow scale - slower decay (more persistent)
                scale_factor = 0.7
            
            variations[i] = self.exponential_lambda * scale_factor
        
        return variations
    
    def _compute_beta_variations(self) -> np.ndarray:
        """Compute dimension-specific oscillation frequencies."""
        variations = np.zeros(self.embedding_dimension)
        
        for i in range(self.embedding_dimension):
            # Harmonic frequency relationships
            base_freq = self.cosine_beta
            harmonic_factor = 1 + 0.1 * np.sin(2 * np.pi * i / self.embedding_dimension)
            
            # Mathematical constant modulation
            pi_factor = 1 + 0.05 * np.cos(2 * np.pi * i / (np.pi * 50))
            
            variations[i] = base_freq * harmonic_factor * pi_factor
        
        return variations
    
    def _compute_alpha_variations(self) -> np.ndarray:
        """Compute dimension-specific persistence weights."""
        variations = np.zeros(self.embedding_dimension)
        
        for i in range(self.embedding_dimension):
            # Weight distribution across dimensions
            weight_factor = 1.0 / (1.0 + 0.1 * i)  # Gradual decay with dimension
            
            # Golden ratio modulation
            golden_ratio = (1 + np.sqrt(5)) / 2
            golden_factor = 1 + 0.1 * np.sin(2 * np.pi * i / (golden_ratio * 100))
            
            variations[i] = self.persistence_alpha * weight_factor * golden_factor
        
        return variations
    
    def compute_persistence(self,
                          observational_distance: Union[float, complex],
                          dimension: Optional[int] = None) -> Union[float, complex]:
        """
        Compute exponential-cosine persistence: α·exp(-λ(s-s₀))·cos(β(s-s₀))
        
        Args:
            observational_distance: Distance s-s₀ in observational space
            dimension: Optional specific dimension for dimension-aware persistence
            
        Returns:
            Exponential-cosine persistence value
        """
        # Handle complex observational distances
        if np.iscomplexobj(observational_distance):
            distance_magnitude = np.abs(observational_distance)
            distance_phase = np.angle(observational_distance)
        else:
            distance_magnitude = abs(observational_distance)
            distance_phase = 0.0
        
        # Use dimension-specific parameters if provided
        if dimension is not None and dimension < self.embedding_dimension:
            lambda_val = self.lambda_variations[dimension]
            beta_val = self.beta_variations[dimension]
            alpha_val = self.alpha_variations[dimension]
        else:
            lambda_val = self.exponential_lambda
            beta_val = self.cosine_beta
            alpha_val = self.persistence_alpha
        
        # Exponential-cosine computation
        exponential_component = np.exp(-lambda_val * distance_magnitude)
        cosine_component = np.cos(beta_val * distance_magnitude)
        persistence_magnitude = alpha_val * exponential_component * cosine_component
        
        # For complex distances, include phase information
        if np.iscomplexobj(observational_distance):
            # Complex oscillatory persistence
            complex_oscillation = np.exp(1j * beta_val * distance_phase)
            return persistence_magnitude * complex_oscillation
        else:
            return persistence_magnitude
    
    def compute_multi_dimensional_persistence(self, observational_distance: Union[float, complex]) -> np.ndarray:
        """
        Compute exponential-cosine persistence for all dimensions simultaneously.
        
        Args:
            observational_distance: Distance s-s₀ in observational space
            
        Returns:
            Array of persistence values for each dimension
        """
        persistence_values = np.zeros(self.embedding_dimension, dtype=complex)
        
        for i in range(self.embedding_dimension):
            persistence_values[i] = self.compute_persistence(observational_distance, dimension=i)
        
        return persistence_values


class DualDecayPersistence:
    """
    Combines both memory layers implementing complete dual-decay structure.
    
    Mathematical Foundation: 
    Ψ_persistence(s-s₀) = [Vivid Recent] + [Persistent Traits]
                         = exp(-(s-s₀)²/2σ²) + α·exp(-λ(s-s₀))·cos(β(s-s₀))
    """
    
    def __init__(self,
                 embedding_dimension: int,
                 gaussian_sigma: float,
                 exponential_lambda: float,
                 cosine_beta: float,
                 persistence_alpha: float):
        """
        Initialize complete dual-decay persistence structure.
        
        Args:
            embedding_dimension: Dimension of embedding space
            gaussian_sigma: Standard deviation for Gaussian component
            exponential_lambda: Decay rate for exponential component
            cosine_beta: Frequency for cosine oscillations
            persistence_alpha: Weight for persistent component
        """
        self.embedding_dimension = embedding_dimension
        
        # Initialize memory layers
        self.gaussian_memory = GaussianMemory(embedding_dimension, gaussian_sigma)
        self.exponential_cosine_memory = ExponentialCosineMemory(
            embedding_dimension, exponential_lambda, cosine_beta, persistence_alpha
        )
        
        logger.info(f"Initialized DualDecayPersistence with Gaussian and ExponentialCosine layers")
    
    def compute_persistence(self,
                          observational_distance: Union[float, complex],
                          dimension: Optional[int] = None) -> Union[float, complex]:
        """
        Compute complete dual-decay persistence combining both layers.
        
        Mathematical Formula: Ψ = Gaussian + ExponentialCosine
        
        Args:
            observational_distance: Distance s-s₀ in observational space
            dimension: Optional specific dimension for dimension-aware persistence
            
        Returns:
            Combined persistence value
        """
        # Compute Gaussian (vivid recent) component
        gaussian_component = self.gaussian_memory.compute_persistence(
            observational_distance, dimension
        )
        
        # Compute exponential-cosine (persistent traits) component
        exp_cos_component = self.exponential_cosine_memory.compute_persistence(
            observational_distance, dimension
        )
        
        # Combine components
        total_persistence = gaussian_component + exp_cos_component
        
        return total_persistence
    
    def compute_multi_dimensional_persistence(self, observational_distance: Union[float, complex]) -> np.ndarray:
        """
        Compute dual-decay persistence for all dimensions simultaneously.
        
        Args:
            observational_distance: Distance s-s₀ in observational space
            
        Returns:
            Array of combined persistence values for each dimension
        """
        # Get persistence from both layers
        gaussian_persistence = self.gaussian_memory.compute_multi_dimensional_persistence(observational_distance)
        exp_cos_persistence = self.exponential_cosine_memory.compute_multi_dimensional_persistence(observational_distance)
        
        # Combine layers
        total_persistence = gaussian_persistence + exp_cos_persistence
        
        return total_persistence
    
    def analyze_memory_components(self, observational_distance: Union[float, complex]) -> Dict[str, Any]:
        """
        Analyze the contribution of different memory components.
        
        Args:
            observational_distance: Distance s-s₀ in observational space
            
        Returns:
            Dictionary with detailed memory analysis
        """
        # Compute individual components
        gaussian_component = self.gaussian_memory.compute_persistence(observational_distance)
        exp_cos_component = self.exponential_cosine_memory.compute_persistence(observational_distance)
        total_persistence = gaussian_component + exp_cos_component
        
        # Analyze component contributions
        if np.abs(total_persistence) > 1e-10:
            gaussian_contribution = np.abs(gaussian_component) / np.abs(total_persistence)
            exp_cos_contribution = np.abs(exp_cos_component) / np.abs(total_persistence)
        else:
            gaussian_contribution = 0.0
            exp_cos_contribution = 0.0
        
        return {
            'total_persistence': total_persistence,
            'gaussian_component': gaussian_component,
            'exponential_cosine_component': exp_cos_component,
            'gaussian_contribution_ratio': float(gaussian_contribution),
            'exponential_cosine_contribution_ratio': float(exp_cos_contribution),
            'memory_balance': 'gaussian_dominant' if gaussian_contribution > 0.6 else 
                           'exp_cos_dominant' if exp_cos_contribution > 0.6 else 'balanced',
            'observational_distance': observational_distance
        }


# Main interface class for external usage
class ObservationalPersistence:
    """
    Main interface for observational persistence implementing the complete dual-decay structure.
    
    This class provides the external interface while internally using the modular architecture
    with PersistenceLayer, GaussianMemory, ExponentialCosineMemory, and DualDecayPersistence.
    """
    
    def __init__(self,
                 gaussian_sigma: float,
                 exponential_lambda: float,
                 cosine_beta: float,
                 persistence_alpha: float):
        """
        Initialize observational persistence with dual-decay parameters.
        
        Args:
            gaussian_sigma: Standard deviation for vivid recent memory
            exponential_lambda: Decay rate for persistent traits
            cosine_beta: Oscillatory frequency for persistent traits
            persistence_alpha: Weight for persistent component
        """
        # Set default embedding dimension (will be overridden when used with specific dimensions)
        self.default_embedding_dimension = 1024
        
        # Store parameters for later initialization
        self.gaussian_sigma = gaussian_sigma
        self.exponential_lambda = exponential_lambda
        self.cosine_beta = cosine_beta
        self.persistence_alpha = persistence_alpha
        
        # Initialize with default dimension (can be recreated with specific dimension)
        self.dual_decay = DualDecayPersistence(
            self.default_embedding_dimension,
            gaussian_sigma,
            exponential_lambda,
            cosine_beta,
            persistence_alpha
        )
        
        logger.info("Initialized ObservationalPersistence with dual-decay structure")
    
    def compute_persistence(self,
                          current_state: Union[float, complex],
                          reference_state: Union[float, complex]) -> complex:
        """
        Compute observational persistence Ψ_persistence(s-s₀).
        
        Args:
            current_state: Current observational state s
            reference_state: Reference observational state s₀
            
        Returns:
            Complex persistence value
        """
        # Compute observational distance
        observational_distance = current_state - reference_state
        
        # Use dual-decay persistence
        persistence = self.dual_decay.compute_persistence(observational_distance)
        
        # Ensure complex return type for field theory compatibility
        if not np.iscomplexobj(persistence):
            persistence = complex(persistence, 0.1 * abs(persistence))
        
        return persistence
    
    def compute_multi_dimensional_persistence(self,
                                            current_state: Union[float, complex],
                                            reference_state: Union[float, complex],
                                            embedding_dimension: int) -> np.ndarray:
        """
        Compute persistence for specific embedding dimension.
        
        Args:
            current_state: Current observational state s
            reference_state: Reference observational state s₀
            embedding_dimension: Specific embedding dimension to use
            
        Returns:
            Array of persistence values for each dimension
        """
        # Create dimension-specific dual-decay if needed
        if embedding_dimension != self.default_embedding_dimension:
            dual_decay = DualDecayPersistence(
                embedding_dimension,
                self.gaussian_sigma,
                self.exponential_lambda,
                self.cosine_beta,
                self.persistence_alpha
            )
        else:
            dual_decay = self.dual_decay
        
        # Compute observational distance
        observational_distance = current_state - reference_state
        
        # Get multi-dimensional persistence
        persistence_array = dual_decay.compute_multi_dimensional_persistence(observational_distance)
        
        return persistence_array