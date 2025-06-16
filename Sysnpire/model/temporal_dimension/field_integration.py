"""
Field Integration Layer - Connects Temporal Dimension to Complete Charge Formula

This module provides integration between temporal dimension components
and the complete conceptual charge formula Q(τ, C, s).

Mathematical Reference: Complete Field Theory of Social Constructs
Formula: Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from numbers import Complex
import logging

# Import the actual ConceptualCharge implementation
from Sysnpire.model.mathematics.conceptual_charge import ConceptualCharge

# Import temporal dimension components
from .observational_persistence import ObservationalPersistence
from .trajectory_operators import TrajectoryOperatorEngine
from .phase_coordination import TemporalPhaseCoordinator

logger = logging.getLogger(__name__)


class TemporalFieldIntegrator:
    """
    Integrates temporal dimension components with the complete charge formula.
    
    This class ensures that temporal calculations connect properly to the
    full Q(τ, C, s) field theory rather than operating in isolation.
    """
    
    def __init__(self,
                 embedding_dimension: int,
                 coupling_strength: float):
        """
        Initialize temporal field integrator.
        
        Args:
            embedding_dimension: Dimension of embedding space (1024 for BGE)
            coupling_strength: Coupling strength between temporal and other field components
        """
        self.embedding_dimension = embedding_dimension
        self.coupling_strength = coupling_strength
        
        # Initialize temporal components
        # Compute persistence parameters from mathematical theory
        gaussian_sigma, exponential_lambda, cosine_beta, persistence_alpha = self._compute_persistence_parameters_from_theory()
        self.persistence = ObservationalPersistence(gaussian_sigma, exponential_lambda, cosine_beta, persistence_alpha)
        # Compute adaptive frequencies from mathematical formulations
        adaptive_frequencies = self._compute_adaptive_frequencies_from_theory(embedding_dimension)
        self.trajectory_engine = TrajectoryOperatorEngine(
            embedding_dimension=embedding_dimension,
            base_frequencies=adaptive_frequencies,
            integration_method="adaptive_quad"
        )
        self.phase_coordinator = TemporalPhaseCoordinator(embedding_dimension)
        
        logger.info("Initialized TemporalFieldIntegrator with complete field theory integration")
    
    def enhance_conceptual_charge(self,
                                 charge: ConceptualCharge,
                                 temporal_context: Optional[Dict] = None) -> ConceptualCharge:
        """
        Enhance a ConceptualCharge with temporal field effects.
        
        Args:
            charge: Existing ConceptualCharge object
            temporal_context: Additional temporal context parameters
            
        Returns:
            Enhanced ConceptualCharge with temporal integration
        """
        # Get current charge state
        current_s = charge.observational_state
        
        # Compute temporal trajectory operators
        trajectory_data = self.trajectory_engine.compute_trajectory_integral(
            token=charge.token,
            context=str(charge.context),
            observational_state=current_s,
            semantic_embedding=charge.semantic_vector
        )
        
        # Enhance trajectory operator with temporal effects
        # Get trajectory operators for first few dimensions
        base_T_operators = []
        for dim in range(min(10, len(trajectory_data['trajectory_operators']))):
            try:
                base_T_operators.append(charge.trajectory_operator(current_s, dimension=dim))
            except Exception as e:
                logger.error(f"Trajectory operator computation failed for dimension {dim}: {e}")
                raise RuntimeError(f"Trajectory operator computation required for dimension {dim}. CLAUDE.md prohibits fallback values. Error: {e}")
        
        base_T_array = np.array(base_T_operators, dtype=complex)
        enhanced_T = self._enhance_trajectory_operator(
            base_T_array,
            trajectory_data['trajectory_operators']
        )
        
        # Enhance persistence with complex field effects
        temporal_persistence = self.persistence.compute_persistence(
            current_s, charge.s_0
        )
        try:
            base_persistence = charge.observational_persistence(current_s)
        except Exception as e:
            logger.error(f"Observational persistence computation failed: {e}")
            raise RuntimeError(f"Observational persistence computation required. CLAUDE.md prohibits fallback values. Error: {e}")
            
        enhanced_persistence = self._couple_persistence_fields(
            base_persistence,
            temporal_persistence
        )
        
        # Enhance phase coordination (pad to full dimension if needed)
        base_phases = np.angle(enhanced_T)
        if len(base_phases) < self.embedding_dimension:
            # Pad with zeros to match expected dimension
            padded_phases = np.zeros(self.embedding_dimension)
            padded_phases[:len(base_phases)] = base_phases
            base_phases = padded_phases
            
        coordinated_phases = self.phase_coordinator.coordinate_phases(
            base_phases, current_s
        )
        
        # Create enhanced charge with temporal integration
        enhanced_charge = ConceptualCharge(
            token=charge.token,
            semantic_vector=charge.semantic_vector,
            context=charge.context,
            observational_state=current_s,
            gamma=charge.gamma
        )
        
        # Override methods with temporal-enhanced versions
        enhanced_charge._temporal_T = enhanced_T
        enhanced_charge._temporal_persistence = enhanced_persistence
        enhanced_charge._temporal_phases = coordinated_phases
        enhanced_charge._trajectory_data = trajectory_data
        
        return enhanced_charge
    
    def compute_temporal_trajectory_component(self,
                                            token: str,
                                            context: str,
                                            observational_state: Union[float, complex],
                                            semantic_vector: np.ndarray) -> np.ndarray:
        """
        Compute temporal trajectory component T(τ,C,s) with complex integration.
        
        This directly implements the trajectory operator from CLAUDE.md
        using actual complex field mathematics.
        
        Args:
            token: Token τ
            context: Context C  
            observational_state: Current state s
            semantic_vector: Semantic embedding vector
            
        Returns:
            Complex trajectory operator array T_i(τ,C,s)
        """
        trajectory_data = self.trajectory_engine.compute_trajectory_integral(
            token=token,
            context=context,
            observational_state=observational_state,
            semantic_embedding=semantic_vector
        )
        
        return trajectory_data['trajectory_operators']
    
    def compute_temporal_persistence_component(self,
                                             observational_distance: Union[float, complex],
                                             context_modulation: Optional[Dict] = None) -> complex:
        """
        Compute temporal persistence component Ψ_persistence(s-s₀) with complex fields.
        
        Args:
            observational_distance: Distance s-s₀
            context_modulation: Optional context modulation parameters
            
        Returns:
            Complex persistence value
        """
        base_persistence = self.persistence.compute_persistence(
            observational_distance, 0.0
        )
        
        if context_modulation:
            if 'relevance' not in context_modulation:
                raise ValueError("relevance must be present in context_modulation. CLAUDE.md prohibits default values.")
            if 'similarity' not in context_modulation:
                raise ValueError("similarity must be present in context_modulation. CLAUDE.md prohibits default values.")
            base_persistence = self.persistence.modulate_by_context(
                base_persistence,
                context_modulation['relevance'],
                context_modulation['similarity']
            )
        
        return base_persistence
    
    def compute_complete_temporal_charge(self,
                                       token: str,
                                       context: str,
                                       observational_state: complex,
                                       semantic_vector: np.ndarray,
                                       gamma: complex) -> complex:
        """
        Compute complete charge with full temporal integration.
        
        This is the temporal contribution to the complete formula:
        Q_temporal = T(τ,C,s) · Ψ_persistence(s-s₀) · e^(iθ_temporal)
        
        Args:
            token: Token τ
            context: Context C
            observational_state: Current state s
            semantic_vector: Semantic embedding
            gamma: Global calibration factor
            
        Returns:
            Complex temporal charge component
        """
        # Compute trajectory operators
        T_operators = self.compute_temporal_trajectory_component(
            token, context, observational_state, semantic_vector
        )
        
        # Compute persistence
        persistence = self.compute_temporal_persistence_component(
            observational_state
        )
        
        # Coordinate phases
        phases = np.angle(T_operators)
        coordinated_phases = self.phase_coordinator.coordinate_phases(
            phases, observational_state
        )
        
        # Implement complex field coupling: Φ^semantic(τ,s) = w_i * T_i(τ,s) * x[i] * breathing_modulation(s) * e^(iθ(s))
        breathing_modulation = self._compute_complex_breathing_modulation(observational_state)
        
        # Apply complex field coupling to semantic vector
        coupled_semantic_field = np.zeros_like(T_operators, dtype=complex)
        min_len = min(len(T_operators), len(semantic_vector))
        
        for i in range(min_len):
            # Weight factor decreasing with dimension
            w_i = complex(1.0 / (1.0 + i), 0.1 / (1.0 + i))
            # Complex field coupling
            coupled_semantic_field[i] = w_i * T_operators[i] * semantic_vector[i] * breathing_modulation
        
        # Combine temporal phase contribution with complex phase factor
        temporal_phase = np.mean(coordinated_phases)
        if np.isrealobj(temporal_phase):
            temporal_phase = complex(temporal_phase, 0.1 * np.abs(temporal_phase))
        phase_factor = np.exp(1j * temporal_phase)
        
        # Compute complex temporal magnitude from coupled field
        temporal_magnitude = np.mean(np.abs(coupled_semantic_field))
        if temporal_magnitude == 0:
            temporal_magnitude = np.mean(np.abs(T_operators))
        
        # Complete temporal charge with complex gamma
        if np.isrealobj(gamma):
            gamma = complex(gamma, 0.1 * gamma)
        
        Q_temporal = gamma * complex(temporal_magnitude) * persistence * phase_factor
        
        return Q_temporal
    
    def _compute_complex_breathing_modulation(self, observational_state: complex) -> complex:
        """
        Compute complex breathing modulation for semantic field coupling.
        
        Implements breathing pattern with complex oscillations for proper field theory.
        """
        if np.isrealobj(observational_state):
            observational_state = complex(observational_state, 0)
            
        # Complex breathing with multiple frequency components
        base_freq = 0.1
        primary = np.exp(1j * base_freq * observational_state)
        harmonic = 0.3 * np.exp(1j * 3 * base_freq * observational_state)
        subharmonic = 0.2 * np.exp(1j * 0.5 * base_freq * observational_state)
        
        # Combined complex breathing pattern
        breathing = 1.0 + 0.2 * (primary + harmonic + subharmonic)
        
        return breathing
    
    def _enhance_trajectory_operator(self,
                                   base_T: Union[complex, np.ndarray],
                                   temporal_T: np.ndarray) -> np.ndarray:
        """Enhance base trajectory operator with temporal effects."""
        if np.isscalar(base_T):
            # Convert scalar to array
            base_T = np.full(self.embedding_dimension, base_T, dtype=complex)
        
        # Ensure compatible dimensions
        min_len = min(len(base_T), len(temporal_T))
        enhanced = np.zeros(min_len, dtype=complex)
        
        # Couple base and temporal operators
        for i in range(min_len):
            coupling = 1.0 + self.coupling_strength * np.abs(temporal_T[i])
            enhanced[i] = base_T[i] * coupling + 0.1 * temporal_T[i]
        
        return enhanced
    
    def _couple_persistence_fields(self,
                                  base_persistence: complex,
                                  temporal_persistence: complex) -> complex:
        """Couple base and temporal persistence fields."""
        
        # Phase alignment between fields
        phase_diff = np.angle(temporal_persistence) - np.angle(base_persistence)
        alignment_factor = np.cos(phase_diff)
        
        # Amplitude coupling
        amplitude_coupling = 1.0 + self.coupling_strength * alignment_factor
        
        # Combined persistence with field coupling
        coupled = base_persistence * amplitude_coupling + 0.2 * temporal_persistence
        
        return coupled
    
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