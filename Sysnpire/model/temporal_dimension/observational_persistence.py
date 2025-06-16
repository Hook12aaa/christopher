"""
Observational Persistence - Layered Memory Structure

Mathematical Reference: Section 3.1.4.3.3
Formula: Ψ_persistence(s-s₀) = exp(-(s-s₀)²/2σ²) + α·exp(-λ(s-s₀))·cos(β(s-s₀))

Implements dual-decay memory structure:
1. Gaussian component: Vivid recent memory (like sharp recall of recent chapters)
2. Exponential-cosine: Persistent traits (like enduring character impressions)
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class ObservationalPersistence:
    """
    Implements the dual-decay observational persistence function Ψ_persistence(s-s₀).
    
    This creates a layered memory structure that captures both:
    - Vivid recent observations (Gaussian decay)
    - Persistent semantic traits (Exponential-cosine decay)
    
    Mathematical Foundation: Section 3.1.4.3.3
    """
    
    def __init__(self,
                 gaussian_sigma: float = 0.3,
                 exponential_lambda: float = 0.1,
                 cosine_beta: float = 2.0,
                 persistence_alpha: float = 0.4):
        """
        Initialize observational persistence parameters.
        
        Args:
            gaussian_sigma: Width of Gaussian decay (vivid memory window)
            exponential_lambda: Decay rate for persistent component
            cosine_beta: Oscillation frequency for rhythmic reinforcement
            persistence_alpha: Weight of persistent component
        """
        self.gaussian_sigma = gaussian_sigma
        self.exponential_lambda = exponential_lambda
        self.cosine_beta = cosine_beta
        self.persistence_alpha = persistence_alpha
        
        logger.info(f"Initialized ObservationalPersistence with σ={gaussian_sigma}, λ={exponential_lambda}")
    
    def compute_persistence(self, 
                          current_state: float,
                          reference_state: float = 0.0) -> complex:
        """
        Compute persistence value Ψ_persistence(s-s₀).
        
        Args:
            current_state: Current observational state s
            reference_state: Reference state s₀ (default: 0)
            
        Returns:
            Persistence value combining both memory components
        """
        distance = abs(current_state - reference_state)
        
        # Gaussian component - vivid recent memory (complex field effect)
        gaussian_component = np.exp(-(distance**2) / (2 * self.gaussian_sigma**2))
        
        # Complex exponential component with phase coupling - persistent traits
        exp_complex_component = (self.persistence_alpha * 
                               np.exp(-self.exponential_lambda * distance + 1j * self.cosine_beta * distance))
        
        return complex(gaussian_component) + exp_complex_component
    
    def compute_memory_components(self,
                                current_state: float,
                                reference_state: float = 0.0) -> Dict[str, complex]:
        """
        Compute individual memory components for analysis.
        
        Args:
            current_state: Current observational state
            reference_state: Reference state
            
        Returns:
            Dictionary with individual component values
        """
        distance = abs(current_state - reference_state)
        
        # Complex field components
        gaussian = complex(np.exp(-(distance**2) / (2 * self.gaussian_sigma**2)))
        exp_complex = self.persistence_alpha * np.exp(-self.exponential_lambda * distance + 1j * self.cosine_beta * distance)
        
        return {
            'distance': distance,
            'gaussian_component': gaussian,
            'exp_complex_component': exp_complex,
            'total_persistence': gaussian + exp_complex,
            'magnitude': abs(gaussian + exp_complex),
            'phase': np.angle(gaussian + exp_complex)
        }
    
    def find_memory_horizon(self, 
                          threshold: float = 0.1,
                          max_distance: float = 20.0) -> Tuple[float, float]:
        """
        Find distances where memory drops below threshold.
        
        Args:
            threshold: Persistence threshold
            max_distance: Maximum distance to search
            
        Returns:
            Tuple of (gaussian_horizon, persistent_horizon)
        """
        # Find Gaussian horizon
        gaussian_horizon = np.sqrt(-2 * self.gaussian_sigma**2 * np.log(threshold))
        
        # Find persistent horizon (approximate)
        distances = np.linspace(0, max_distance, 1000)
        persistences = [self.compute_persistence(d) for d in distances]
        
        persistent_horizon = max_distance
        for d, p in zip(distances, persistences):
            if p < threshold and d > gaussian_horizon:
                persistent_horizon = d
                break
        
        return gaussian_horizon, persistent_horizon
    
    def modulate_by_context(self,
                          base_persistence: complex,
                          context_relevance: float = 1.0,
                          semantic_similarity: float = 1.0) -> complex:
        """
        Modulate persistence based on contextual factors.
        
        Args:
            base_persistence: Base persistence value
            context_relevance: How relevant the context is (0-1)
            semantic_similarity: Semantic similarity measure (0-1)
            
        Returns:
            Context-modulated persistence
        """
        # Context enhances memory retention with phase modulation
        context_boost = (1.0 + 0.5 * context_relevance) * np.exp(1j * 0.1 * context_relevance)
        
        # Semantic similarity creates complex resonance
        semantic_resonance = (1.0 + 0.3 * semantic_similarity) * np.exp(1j * 0.2 * semantic_similarity)
        
        return base_persistence * context_boost * semantic_resonance
    
    def generate_persistence_profile(self,
                                   state_range: Tuple[float, float],
                                   num_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Generate persistence profile over a range of states.
        
        Args:
            state_range: (min_state, max_state) tuple
            num_points: Number of points to sample
            
        Returns:
            Dictionary with state values and persistence profiles
        """
        states = np.linspace(state_range[0], state_range[1], num_points)
        
        profiles = {
            'states': states,
            'total_persistence': np.zeros(num_points, dtype=complex),
            'gaussian_component': np.zeros(num_points, dtype=complex),
            'exp_complex_component': np.zeros(num_points, dtype=complex),
            'magnitude': np.zeros(num_points),
            'phase': np.zeros(num_points)
        }
        
        for i, state in enumerate(states):
            components = self.compute_memory_components(state)
            profiles['total_persistence'][i] = components['total_persistence']
            profiles['gaussian_component'][i] = components['gaussian_component']
            profiles['exp_complex_component'][i] = components['exp_complex_component']
            profiles['magnitude'][i] = components['magnitude']
            profiles['phase'][i] = components['phase']
        
        return profiles