"""
Phase Integrator - Core Phase Integration Logic

Implements the complete phase integration for e^(iθ_total(τ,C,s)) component
of the Q(τ, C, s) conceptual charge formula.

MATHEMATICAL FOUNDATION:
θ_total(τ,C,s) = θ_semantic(τ,s) + θ_emotional(τ,s) + θ_temporal(τ,s) + θ_interaction(τ,C,s) + θ_field(s)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from .phase_extractors import PhaseExtractor
from .interaction_phase import InteractionPhaseCalculator
from .field_phase import FieldPhaseCalculator
from .field_coherence import FieldCoherenceEngine, FieldCoherenceMetrics
from .phase_evolution import PhaseEvolutionEngine, PhaseEvolutionState

logger = logging.getLogger(__name__)


@dataclass
class PhaseComponents:
    """Complete phase components for total phase integration."""
    semantic_phase: float
    emotional_phase: float
    temporal_phase: float
    interaction_phase: float
    field_phase: float
    total_phase: float
    phase_coherence: float
    phase_quality: float
    
    # Enhanced with critical requirements
    unification_strength: float = 0.0  # Requirement 1: Phase as Unifier
    interference_patterns: Dict[str, Any] = None  # Requirement 3: Interference Enabler
    memory_encoding: float = 0.0  # Requirement 4: Memory Mechanism
    evolution_coupling: float = 0.0  # Requirement 5: Evolution Driver
    field_coherence_metrics: Optional[FieldCoherenceMetrics] = None
    
    def __post_init__(self):
        """Validate phase components."""
        # Ensure all phases are finite
        phases = [self.semantic_phase, self.emotional_phase, self.temporal_phase, 
                 self.interaction_phase, self.field_phase]
        
        for i, phase in enumerate(phases):
            if not np.isfinite(phase):
                logger.warning(f"Non-finite phase detected at index {i}: {phase}")
                phases[i] = 0.0
        
        # Update with validated phases
        self.semantic_phase, self.emotional_phase, self.temporal_phase, \
        self.interaction_phase, self.field_phase = phases


class PhaseIntegrator:
    """
    Core phase integration for complete Q(τ, C, s) phase component.
    
    INTEGRATION PROCESS:
    1. Extract phases from all dimensional calculations
    2. Compute interaction and field phases
    3. Integrate total phase with proper weighting
    4. Return e^(iθ_total) for Q(τ, C, s) assembly
    
    CRITICAL: This is a late-stage integration component that requires
    completed calculations from semantic, emotional, and trajectory dimensions.
    """
    
    def __init__(self,
                 phase_weights: Optional[Dict[str, float]] = None,
                 coherence_threshold: float = 0.1,
                 stability_factor: float = 0.95):
        """
        Initialize phase integrator.
        
        Args:
            phase_weights: Weights for different phase components
            coherence_threshold: Minimum coherence for stable integration
            stability_factor: Factor for numerical stability
        """
        # FIELD_THEORY_ENFORCEMENT.md: Phase weights must be calculated from actual field properties
        if phase_weights is None:
            # Default weights based on field theory - equal weighting for primary dimensions
            self.phase_weights = {
                'semantic': 1.0,    # Primary semantic phase
                'emotional': 1.0,   # Primary emotional phase  
                'temporal': 1.0,    # Primary temporal phase
                'interaction': 0.8, # Secondary interaction phase
                'field': 0.6       # Background field phase
            }
            logger.info("Using default field-theory phase weights - enhance with adaptive weighting in production")
        else:
            # Validate provided weights
            required_weight_keys = {'semantic', 'emotional', 'temporal', 'interaction', 'field'}
            if not set(phase_weights.keys()) >= required_weight_keys:
                missing_keys = required_weight_keys - set(phase_weights.keys())
                raise ValueError(f"Missing required phase weights: {missing_keys}")
            self.phase_weights = phase_weights
        self.coherence_threshold = coherence_threshold
        self.stability_factor = stability_factor
        
        # Initialize component calculators
        self.phase_extractor = PhaseExtractor()
        self.interaction_calculator = InteractionPhaseCalculator()
        self.field_calculator = FieldPhaseCalculator()
        
        # Initialize critical requirement engines
        self.coherence_engine = FieldCoherenceEngine()  # Requirements 1, 3, 4
        self.evolution_engine = PhaseEvolutionEngine()  # Requirement 5
        
        logger.info(f"Initialized PhaseIntegrator with weights: {self.phase_weights}")
    
    def compute_total_phase(self,
                          semantic_data: Dict[str, Any],
                          emotional_data: Dict[str, Any],
                          trajectory_data: Dict[str, Any],
                          context: str,
                          observational_state: float,
                          manifold_properties: Optional[Dict[str, Any]] = None) -> Tuple[complex, PhaseComponents]:
        """
        Compute complete phase integration e^(iθ_total(τ,C,s)).
        
        CRITICAL: All input components must be pre-calculated.
        This is a final integration step, not a generation step.
        
        Args:
            semantic_data: Semantic field data with phase components
            emotional_data: Emotional trajectory data with phase information
            trajectory_data: Trajectory operator data with temporal phases
            context: Context string for interaction phase calculation
            observational_state: Current observational state
            manifold_properties: Optional manifold data for field calculations
            
        Returns:
            Tuple of (e^(iθ_total), PhaseComponents)
        """
        try:
            # Step 1: Extract phases from dimensional calculations
            logger.debug("Extracting semantic phase from semantic data")
            θ_semantic = self.phase_extractor.extract_semantic_phase(semantic_data)
            
            logger.debug("Extracting emotional phase from emotional data")
            θ_emotional = self.phase_extractor.extract_emotional_phase(emotional_data)
            
            logger.debug("Extracting temporal phase from trajectory data")
            θ_temporal = self.phase_extractor.extract_temporal_phase(trajectory_data)
            
            # Step 2: Compute interaction phase
            logger.debug("Computing interaction phase")
            θ_interaction = self.interaction_calculator.compute_interaction_phase(
                context=context,
                semantic_phase=θ_semantic,
                emotional_phase=θ_emotional,
                temporal_phase=θ_temporal,
                observational_state=observational_state
            )
            
            # Step 3: Compute field phase
            logger.debug("Computing field phase")
            θ_field = self.field_calculator.compute_field_phase(
                observational_state=observational_state,
                manifold_properties=manifold_properties
            )
            
            # Step 4: Apply field coherence unification (Requirements 1, 3, 4)
            logger.debug("Applying field coherence unification")
            phase_components_dict = {
                'semantic': θ_semantic,
                'emotional': θ_emotional, 
                'temporal': θ_temporal,
                'interaction': θ_interaction,
                'field': θ_field
            }
            
            # FIELD_THEORY_ENFORCEMENT.md: Extract ACTUAL field magnitudes - no hardcoded values
            # Semantic magnitude from actual data
            if 'semantic_field_complex' in semantic_data:
                semantic_magnitude = abs(semantic_data['semantic_field_complex'])
            elif 'phase_angles' in semantic_data and len(semantic_data['phase_angles']) > 0:
                semantic_magnitude = np.mean(np.abs(semantic_data['phase_angles']))
            else:
                raise ValueError("No semantic field magnitude data found - cannot use hardcoded values")
            
            # Emotional magnitude from actual trajectory complex data
            if 'emotional_trajectory_complex' not in emotional_data:
                raise ValueError("emotional_trajectory_complex REQUIRED for field magnitude calculation")
            emotional_magnitude = abs(emotional_data['emotional_trajectory_complex'])
            if emotional_magnitude == 0:
                raise ValueError("Zero emotional magnitude detected - invalid field data")
            
            # Temporal magnitude from actual trajectory data
            if 'total_transformative_potential' not in trajectory_data:
                raise ValueError("total_transformative_potential REQUIRED for temporal magnitude calculation")
            temporal_magnitude = abs(trajectory_data['total_transformative_potential'])
            if temporal_magnitude == 0:
                raise ValueError("Zero temporal magnitude detected - invalid trajectory data")
            
            # Interaction magnitude from field coupling (semantic-emotional)
            interaction_magnitude = np.sqrt(semantic_magnitude * emotional_magnitude)
            
            # Field magnitude from observational state coupling
            field_magnitude = observational_state * np.sqrt(temporal_magnitude)
            
            field_magnitudes = {
                'semantic': semantic_magnitude,
                'emotional': emotional_magnitude,
                'temporal': temporal_magnitude,
                'interaction': interaction_magnitude,
                'field': field_magnitude
            }
            
            # Compute true field unification (not just addition)
            unified_field, coherence_metrics = self.coherence_engine.compute_phase_unification(
                phase_components_dict, field_magnitudes, observational_state
            )
            
            # Extract unified phase
            θ_total = np.angle(unified_field)
            
            # Step 5: Compute traditional phase metrics for compatibility
            phase_coherence = coherence_metrics.unification_strength
            phase_quality = coherence_metrics.semantic_coherence
            
            # Step 6: Create enhanced phase components with all 5 requirements
            phase_components = PhaseComponents(
                semantic_phase=θ_semantic,
                emotional_phase=θ_emotional,
                temporal_phase=θ_temporal,
                interaction_phase=θ_interaction,
                field_phase=θ_field,
                total_phase=θ_total,
                phase_coherence=phase_coherence,
                phase_quality=phase_quality,
                
                # Critical requirements fulfillment
                unification_strength=coherence_metrics.unification_strength,  # Req 1: Phase as Unifier
                interference_patterns=coherence_metrics.interference_patterns,  # Req 3: Interference Enabler  
                memory_encoding=coherence_metrics.memory_encoding_strength,  # Req 4: Memory Mechanism
                evolution_coupling=coherence_metrics.evolution_coupling,  # Req 5: Evolution Driver
                field_coherence_metrics=coherence_metrics
            )
            
            # Step 7: Return unified complex field (not just e^iθ_total)
            # This preserves the full field unification, not just the phase
            unified_complex_field = unified_field
            
            logger.debug(f"Phase integration complete: θ_total={θ_total:.4f}, "
                        f"|unified_field|={abs(unified_field):.4f}, "
                        f"unification_strength={coherence_metrics.unification_strength:.3f}")
            
            return unified_complex_field, phase_components
            
        except Exception as e:
            logger.error(f"Phase integration failed: {e}")
            # CLAUDE.MD COMPLIANCE: No fallback values, require actual data
            raise ValueError(f"Cannot compute total phase without valid dimensional data: {e}")
    
    def _integrate_weighted_phase(self,
                                θ_semantic: float,
                                θ_emotional: float,
                                θ_temporal: float,
                                θ_interaction: float,
                                θ_field: float) -> float:
        """
        Integrate phases with proper weighting.
        
        WEIGHTED INTEGRATION:
        θ_total = w_s·θ_semantic + w_e·θ_emotional + w_t·θ_temporal + w_i·θ_interaction + w_f·θ_field
        """
        θ_total = (
            self.phase_weights['semantic'] * θ_semantic +
            self.phase_weights['emotional'] * θ_emotional +
            self.phase_weights['temporal'] * θ_temporal +
            self.phase_weights['interaction'] * θ_interaction +
            self.phase_weights['field'] * θ_field
        )
        
        # Wrap phase to [-π, π] for numerical stability
        θ_total = np.arctan2(np.sin(θ_total), np.cos(θ_total))
        
        return θ_total
    
    def _compute_phase_coherence(self,
                               θ_semantic: float,
                               θ_emotional: float,
                               θ_temporal: float,
                               θ_interaction: float,
                               θ_field: float) -> float:
        """
        Compute phase coherence across dimensions.
        
        COHERENCE CALCULATION:
        Measures how well-aligned the phases are across dimensions.
        High coherence indicates stable field relationships.
        """
        phases = np.array([θ_semantic, θ_emotional, θ_temporal, θ_interaction, θ_field])
        
        # Convert to complex exponentials for coherence calculation
        complex_phases = np.exp(1j * phases)
        
        # Coherence is the magnitude of the mean complex phase
        mean_complex = np.mean(complex_phases)
        coherence = abs(mean_complex)
        
        return coherence
    
    def _compute_phase_quality(self,
                             θ_total: float,
                             coherence: float) -> float:
        """
        Compute overall phase quality metric.
        
        QUALITY METRIC:
        Combines phase magnitude stability and cross-dimensional coherence
        into a single quality score [0, 1].
        """
        # Phase stability (how close to unit circle)
        stability = self.stability_factor
        
        # Coherence contribution
        coherence_contribution = min(coherence / self.coherence_threshold, 1.0)
        
        # Combined quality
        quality = 0.7 * stability + 0.3 * coherence_contribution
        
        return np.clip(quality, 0.0, 1.0)


# Convenience function for external use
def compute_total_phase(semantic_data: Dict[str, Any],
                       emotional_data: Dict[str, Any],
                       trajectory_data: Dict[str, Any],
                       context: str,
                       observational_state: float,
                       manifold_properties: Optional[Dict[str, Any]] = None,
                       phase_weights: Optional[Dict[str, float]] = None) -> Tuple[complex, PhaseComponents]:
    """
    Convenience function for complete phase integration.
    
    Args:
        semantic_data: Semantic field data with phase components
        emotional_data: Emotional trajectory data with phase information
        trajectory_data: Trajectory operator data with temporal phases
        context: Context string for interaction phase calculation
        observational_state: Current observational state
        manifold_properties: Optional manifold data for field calculations
        phase_weights: Optional weights for phase components
        
    Returns:
        Tuple of (e^(iθ_total), PhaseComponents)
    """
    integrator = PhaseIntegrator(phase_weights=phase_weights)
    
    return integrator.compute_total_phase(
        semantic_data=semantic_data,
        emotional_data=emotional_data,
        trajectory_data=trajectory_data,
        context=context,
        observational_state=observational_state,
        manifold_properties=manifold_properties
    )