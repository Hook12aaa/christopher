"""
Emotional-Phase Integration Bridge
 
Bridges emotional trajectory calculations to PhaseIntegrator for complete
θ_total(τ,C,s) phase integration in the Q(τ, C, s) conceptual charge formula.

MATHEMATICAL FOUNDATION:
This module facilitates the integration of emotional phase θ_emotional(τ,s) 
into the complete phase integration: θ_total = θ_semantic + θ_emotional + θ_temporal + θ_interaction + θ_field

INTEGRATION APPROACH:
1. Extract emotional phase from complex emotional trajectory E^trajectory(τ, s)
2. Format emotional data for phase dimension consumption
3. Bridge to PhaseIntegrator for complete phase synthesis
4. Return unified complex field with complete phase integration
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EmotionalPhaseIntegrationBridge:
    """
    Bridge emotional calculations to PhaseIntegrator for complete phase integration.
    
    BRIDGE FUNCTIONALITY:
    - Extracts emotional phase from trajectory results
    - Formats data for phase dimension integration
    - Coordinates with semantic and temporal dimensions
    - Returns unified complex field results
    """
    
    def __init__(self):
        """Initialize emotional-phase integration bridge."""
        logger.info("Initialized EmotionalPhaseIntegrationBridge for cross-dimensional phase coordination")
    
    def integrate_emotional_trajectory_with_complete_phase(self,
                                                          emotional_results: Dict[str, Any],
                                                          semantic_data: Dict[str, Any],
                                                          trajectory_data: Dict[str, Any],
                                                          context: str,
                                                          observational_state: float,
                                                          manifold_properties: Dict[str, Any]) -> Tuple[complex, Dict[str, Any]]:
        """
        Integrate emotional trajectory with complete phase calculation.
        
        MATHEMATICAL PROCESS:
        1. Extract emotional data formatted for phase integration
        2. Coordinate with semantic and temporal phase data
        3. Use PhaseIntegrator for complete θ_total(τ,C,s) calculation
        4. Return unified complex field e^(iθ_total) and analysis
        
        Args:
            emotional_results: Results from compute_emotional_trajectory()
            semantic_data: Semantic field data with phase information
            trajectory_data: Temporal trajectory data with phase information
            context: Processing context for phase integration
            observational_state: Current observational state s
            manifold_properties: Manifold properties for field phase calculation
            
        Returns:
            Tuple of (unified_complex_field, complete_phase_integration_data)
        """
        try:
            logger.debug(f"Starting emotional-phase integration for context: {context}")
            
            # Extract emotional data formatted for phase integration
            emotional_data_for_phase = emotional_results.get('emotional_data_for_phase', {})
            
            if not emotional_data_for_phase:
                logger.warning("No emotional_data_for_phase found - extracting from emotional_results")
                emotional_data_for_phase = self._extract_emotional_phase_data(emotional_results)
            
            # Validate emotional phase data
            if not self._validate_emotional_phase_data(emotional_data_for_phase):
                raise ValueError("Invalid emotional phase data for integration")
            
            # Import phase dimension for complete integration
            from Sysnpire.model.shared_dimensions.phase_dimension import compute_total_phase
            
            # Perform complete phase integration
            logger.debug("Computing complete phase integration via phase dimension")
            
            unified_complex_field, phase_components = compute_total_phase(
                semantic_data=semantic_data,
                emotional_data=emotional_data_for_phase,
                trajectory_data=trajectory_data,
                context=context,
                observational_state=observational_state,
                manifold_properties=manifold_properties
            )
            
            # Create comprehensive phase integration analysis
            phase_integration_data = {
                'unified_complex_field': unified_complex_field,
                'total_phase': np.angle(unified_complex_field),
                'total_magnitude': abs(unified_complex_field),
                
                'phase_components': {
                    'semantic_phase': phase_components.semantic_phase,
                    'emotional_phase': phase_components.emotional_phase,
                    'temporal_phase': phase_components.temporal_phase,
                    'interaction_phase': phase_components.interaction_phase,
                    'field_phase': phase_components.field_phase,
                    'total_phase': phase_components.total_phase
                },
                
                'field_quality_metrics': {
                    'phase_coherence': phase_components.phase_coherence,
                    'phase_quality': phase_components.phase_quality,
                    'unification_strength': phase_components.unification_strength,
                    'interference_patterns': phase_components.interference_patterns,
                    'memory_encoding': phase_components.memory_encoding,
                    'evolution_coupling': phase_components.evolution_coupling
                },
                
                'emotional_contribution': {
                    'emotional_phase_magnitude': abs(emotional_data_for_phase.get('emotional_trajectory_complex', 0)),
                    'emotional_phase_angle': emotional_data_for_phase.get('emotional_phase', 0.0),
                    'emotional_field_strength': emotional_data_for_phase.get('emotional_magnitude', 0.0)
                },
                
                'integration_status': 'complete',
                'bridge_method': 'emotional_phase_integration_bridge'
            }
            
            logger.info(f"Emotional-phase integration complete: "
                       f"θ_total={np.angle(unified_complex_field):.4f}, "
                       f"|e^(iθ)|={abs(unified_complex_field):.4f}, "
                       f"coherence={phase_components.phase_coherence:.3f}")
            
            return unified_complex_field, phase_integration_data
            
        except Exception as e:
            logger.error(f"Emotional-phase integration failed: {e}")
            raise RuntimeError(f"Cannot integrate emotional trajectory with phase dimension: {e}")
    
    def _extract_emotional_phase_data(self, emotional_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and format emotional phase data from emotional results.
        
        FALLBACK EXTRACTION:
        When emotional_data_for_phase is not present, extract from raw results.
        """
        emotional_trajectory_complex = emotional_results.get('emotional_trajectory_complex', complex(0))
        emotional_magnitude = emotional_results.get('emotional_trajectory_magnitude', 0.0)
        emotional_phase = emotional_results.get('emotional_phase', 0.0)
        
        # If no proper emotional data, cannot proceed per CLAUDE.md
        if emotional_trajectory_complex == 0 and emotional_magnitude == 0:
            raise ValueError("No valid emotional trajectory data found for phase integration")
        
        return {
            'emotional_trajectory_complex': emotional_trajectory_complex,
            'emotional_phase': emotional_phase,
            'emotional_magnitude': emotional_magnitude,
            'complex_field_data': {
                'magnitude': emotional_magnitude,
                'phase': emotional_phase,
                'real': emotional_trajectory_complex.real,
                'imag': emotional_trajectory_complex.imag
            },
            'phase_components': [emotional_phase],
            'field_magnitudes': [emotional_magnitude]
        }
    
    def _validate_emotional_phase_data(self, emotional_data: Dict[str, Any]) -> bool:
        """
        Validate emotional phase data for integration.
        
        VALIDATION CRITERIA:
        - Must have emotional_trajectory_complex
        - Must have emotional_phase as finite number
        - Must have emotional_magnitude as positive number
        """
        try:
            # Check for required keys
            required_keys = ['emotional_trajectory_complex', 'emotional_phase', 'emotional_magnitude']
            for key in required_keys:
                if key not in emotional_data:
                    logger.error(f"Missing required key: {key}")
                    return False
            
            # Validate complex trajectory
            complex_trajectory = emotional_data['emotional_trajectory_complex']
            if not isinstance(complex_trajectory, complex):
                logger.error(f"emotional_trajectory_complex must be complex, got {type(complex_trajectory)}")
                return False
            
            # Validate phase
            emotional_phase = emotional_data['emotional_phase']
            if not isinstance(emotional_phase, (int, float)) or not np.isfinite(emotional_phase):
                logger.error(f"emotional_phase must be finite number, got {emotional_phase}")
                return False
            
            # Validate magnitude
            emotional_magnitude = emotional_data['emotional_magnitude']
            if not isinstance(emotional_magnitude, (int, float)) or emotional_magnitude < 0:
                logger.error(f"emotional_magnitude must be non-negative, got {emotional_magnitude}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Emotional phase data validation failed: {e}")
            return False
    
    def extract_complete_phase_contribution(self, emotional_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract complete emotional phase contribution for external integration.
        
        EXTRACTION PURPOSE:
        Provides emotional phase data in the exact format expected by
        phase dimension integration without requiring full bridge integration.
        
        Args:
            emotional_results: Results from compute_emotional_trajectory()
            
        Returns:
            Dict containing complete emotional phase contribution data
        """
        try:
            # Get emotional data for phase integration
            emotional_data = emotional_results.get('emotional_data_for_phase', {})
            
            if not emotional_data:
                emotional_data = self._extract_emotional_phase_data(emotional_results)
            
            # Extract phase contribution components
            emotional_phase_contribution = {
                'primary_emotional_phase': emotional_data.get('emotional_phase', 0.0),
                'emotional_complex_field': emotional_data.get('emotional_trajectory_complex', complex(0)),
                'emotional_field_magnitude': emotional_data.get('emotional_magnitude', 0.0),
                
                'phase_evolution_data': {
                    'phase_trajectory': [emotional_data.get('emotional_phase', 0.0)],
                    'magnitude_trajectory': [emotional_data.get('emotional_magnitude', 0.0)],
                    'complex_trajectory': [emotional_data.get('emotional_trajectory_complex', complex(0))]
                },
                
                'field_coupling_data': {
                    'emotional_field_strength': emotional_data.get('emotional_magnitude', 0.0),
                    'phase_coupling_factor': np.exp(1j * emotional_data.get('emotional_phase', 0.0)),
                    'complex_field_data': emotional_data.get('complex_field_data', {})
                },
                
                'integration_metadata': {
                    'extraction_method': 'emotional_phase_bridge',
                    'field_theory_compliant': True,
                    'complex_valued': True,
                    'trajectory_dependent': True
                }
            }
            
            logger.debug(f"Extracted emotional phase contribution: "
                        f"φ={emotional_phase_contribution['primary_emotional_phase']:.4f}, "
                        f"|E|={emotional_phase_contribution['emotional_field_magnitude']:.4f}")
            
            return emotional_phase_contribution
            
        except Exception as e:
            logger.error(f"Emotional phase contribution extraction failed: {e}")
            raise RuntimeError(f"Cannot extract emotional phase contribution: {e}")


# Convenience functions for easy integration

def integrate_emotional_with_complete_phase(emotional_results: Dict[str, Any],
                                          semantic_data: Dict[str, Any],
                                          trajectory_data: Dict[str, Any],
                                          context: str,
                                          observational_state: float,
                                          manifold_properties: Dict[str, Any]) -> Tuple[complex, Dict[str, Any]]:
    """
    Convenience function for emotional-phase integration.
    
    QUICK INTEGRATION:
    Single function call to integrate emotional trajectory results with
    complete phase calculation from phase dimension.
    
    Args:
        emotional_results: Results from compute_emotional_trajectory()
        semantic_data: Semantic field data with phase information
        trajectory_data: Temporal trajectory data with phase information  
        context: Processing context
        observational_state: Current observational state s
        manifold_properties: Manifold properties for field phase
        
    Returns:
        Tuple of (unified_complex_field, phase_integration_analysis)
    """
    bridge = EmotionalPhaseIntegrationBridge()
    return bridge.integrate_emotional_trajectory_with_complete_phase(
        emotional_results=emotional_results,
        semantic_data=semantic_data,
        trajectory_data=trajectory_data,
        context=context,
        observational_state=observational_state,
        manifold_properties=manifold_properties
    )


def extract_emotional_phase_for_integration(emotional_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to extract emotional phase data for external integration.
    
    SIMPLE EXTRACTION:
    Extracts emotional phase contribution in format suitable for
    direct use by phase dimension or other integration modules.
    
    Args:
        emotional_results: Results from compute_emotional_trajectory()
        
    Returns:
        Dict containing emotional phase contribution data
    """
    bridge = EmotionalPhaseIntegrationBridge()
    return bridge.extract_complete_phase_contribution(emotional_results)