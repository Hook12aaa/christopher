"""
Phase Extractors - Extract Phases from Each Dimension

Extracts phase information from semantic, emotional, and trajectory calculations
for integration into the complete phase component e^(iθ_total(τ,C,s)).

PHASE EXTRACTION METHODS:
- Semantic Phase: From semantic field Φ^semantic(τ, s)
- Emotional Phase: From emotional trajectory E^trajectory(τ, s)
- Temporal Phase: From trajectory operator T(τ, C, s)
"""

import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class PhaseExtractor:
    """
    Extract phase components from dimensional calculations.
    
    EXTRACTION PRINCIPLES:
    - Use np.angle() for complex number phase extraction
    - Handle arrays by averaging or selecting representative phases
    - Validate phase ranges and handle edge cases
    - Maintain numerical stability throughout extraction
    """
    
    def __init__(self):
        """Initialize phase extractor."""
        logger.info("Initialized PhaseExtractor for dimensional phase extraction")
    
    def extract_semantic_phase(self, semantic_data: Dict[str, Any]) -> float:
        """
        Extract semantic phase θ_semantic(τ,s) from semantic field data.
        
        SEMANTIC PHASE SOURCES:
        1. Phase angles from manifold properties
        2. Complex semantic field components
        3. Semantic modulation phases
        
        Args:
            semantic_data: Semantic field data containing phase information
            
        Returns:
            Extracted semantic phase in radians
        """
        try:
            # Priority 1: Direct phase angles from manifold properties
            if 'phase_angles' in semantic_data:
                phase_angles = semantic_data['phase_angles']
                if isinstance(phase_angles, (list, np.ndarray)) and len(phase_angles) > 0:
                    # Use mean phase for representative semantic phase
                    semantic_phase = np.mean(phase_angles)
                    logger.debug(f"Extracted semantic phase from phase_angles: {semantic_phase:.4f}")
                    return self._normalize_phase(semantic_phase)
            
            # Priority 2: Complex semantic field components
            if 'semantic_field_complex' in semantic_data:
                complex_field = semantic_data['semantic_field_complex']
                if isinstance(complex_field, complex):
                    semantic_phase = np.angle(complex_field)
                    logger.debug(f"Extracted semantic phase from complex field: {semantic_phase:.4f}")
                    return semantic_phase
                elif isinstance(complex_field, (list, np.ndarray)):
                    # Extract phases from complex array
                    phases = [np.angle(z) for z in complex_field if isinstance(z, complex)]
                    if phases:
                        semantic_phase = np.mean(phases)
                        logger.debug(f"Extracted semantic phase from complex array: {semantic_phase:.4f}")
                        return self._normalize_phase(semantic_phase)
            
            # Priority 3: Semantic modulation data
            if 'semantic_modulation' in semantic_data:
                modulation = semantic_data['semantic_modulation']
                if isinstance(modulation, (list, np.ndarray)) and len(modulation) > 0:
                    # Use phase derived from modulation patterns
                    modulation_array = np.array(modulation)
                    if np.any(modulation_array != 0):
                        # Compute phase from modulation using FFT
                        fft_result = np.fft.fft(modulation_array[:min(64, len(modulation_array))])
                        dominant_component = fft_result[np.argmax(np.abs(fft_result))]
                        semantic_phase = np.angle(dominant_component)
                        logger.debug(f"Extracted semantic phase from modulation: {semantic_phase:.4f}")
                        return semantic_phase
            
            # Priority 4: Gradient-based phase
            if 'gradient' in semantic_data:
                gradient = semantic_data['gradient']
                if isinstance(gradient, (list, np.ndarray)) and len(gradient) > 0:
                    gradient_array = np.array(gradient)
                    # Compute phase from gradient direction
                    gradient_angle = np.arctan2(gradient_array[1] if len(gradient_array) > 1 else 0,
                                              gradient_array[0] if len(gradient_array) > 0 else 1)
                    semantic_phase = gradient_angle
                    logger.debug(f"Extracted semantic phase from gradient: {semantic_phase:.4f}")
                    return semantic_phase
            
            # CLAUDE.MD COMPLIANCE: No default values allowed
            logger.error("No semantic phase data found - cannot extract phase without actual data")
            raise ValueError("Semantic phase extraction requires actual semantic data with phase components")
            
        except Exception as e:
            logger.error(f"Semantic phase extraction failed: {e}")
            raise ValueError(f"Cannot extract semantic phase without valid data: {e}")
    
    def extract_emotional_phase(self, emotional_data: Dict[str, Any]) -> float:
        """
        Extract emotional phase θ_emotional(τ,s) from emotional trajectory data.
        
        EMOTIONAL PHASE SOURCES:
        1. Direct emotional phase from trajectory computation
        2. Complex emotional trajectory phase
        3. Emotional field modulation phase
        
        Args:
            emotional_data: Emotional trajectory data containing phase information
            
        Returns:
            Extracted emotional phase in radians
        """
        try:
            # Priority 1: Direct emotional phase
            if 'emotional_phase' in emotional_data:
                emotional_phase = emotional_data['emotional_phase']
                if isinstance(emotional_phase, (int, float)) and np.isfinite(emotional_phase):
                    logger.debug(f"Extracted direct emotional phase: {emotional_phase:.4f}")
                    return self._normalize_phase(emotional_phase)
            
            # Priority 2: Complex emotional trajectory
            if 'emotional_trajectory_complex' in emotional_data:
                complex_trajectory = emotional_data['emotional_trajectory_complex']
                if isinstance(complex_trajectory, complex):
                    emotional_phase = np.angle(complex_trajectory)
                    logger.debug(f"Extracted emotional phase from complex trajectory: {emotional_phase:.4f}")
                    return emotional_phase
            
            # Priority 3: Emotional field modulation
            if 'field_modulation' in emotional_data:
                modulation = emotional_data['field_modulation']
                if isinstance(modulation, dict) and 'emotional_phase' in modulation:
                    emotional_phase = modulation['emotional_phase']
                    if isinstance(emotional_phase, (int, float)) and np.isfinite(emotional_phase):
                        logger.debug(f"Extracted emotional phase from field modulation: {emotional_phase:.4f}")
                        return self._normalize_phase(emotional_phase)
            
            # Priority 4: Gaussian alignment phase
            if 'gaussian_alignment' in emotional_data:
                alignment = emotional_data['gaussian_alignment']
                if isinstance(alignment, (int, float)) and alignment != 0:
                    # Convert alignment to phase representation
                    emotional_phase = np.arccos(np.clip(alignment, -1, 1))
                    logger.debug(f"Extracted emotional phase from gaussian alignment: {emotional_phase:.4f}")
                    return emotional_phase
            
            # CLAUDE.MD COMPLIANCE: No default values allowed
            logger.error("No emotional phase data found - cannot extract phase without actual data")
            raise ValueError("Emotional phase extraction requires actual emotional trajectory data with phase components")
            
        except Exception as e:
            logger.error(f"Emotional phase extraction failed: {e}")
            raise ValueError(f"Cannot extract emotional phase without valid data: {e}")
    
    def extract_temporal_phase(self, trajectory_data: Dict[str, Any]) -> float:
        """
        Extract temporal phase θ_temporal(τ,s) from trajectory operator data.
        
        TEMPORAL PHASE SOURCES:
        1. Phase accumulation from trajectory evolution
        2. Frequency evolution phase components
        3. Transformative potential phase
        
        Args:
            trajectory_data: Trajectory operator data containing temporal phase information
            
        Returns:
            Extracted temporal phase in radians
        """
        try:
            # Priority 1: Direct phase accumulation
            if 'phase_accumulation' in trajectory_data:
                phase_accumulation = trajectory_data['phase_accumulation']
                if isinstance(phase_accumulation, (list, np.ndarray)) and len(phase_accumulation) > 0:
                    # Use final accumulated phase
                    temporal_phase = phase_accumulation[-1] if hasattr(phase_accumulation, '__getitem__') else phase_accumulation
                    if np.isfinite(temporal_phase):
                        logger.debug(f"Extracted temporal phase from phase accumulation: {temporal_phase:.4f}")
                        return self._normalize_phase(temporal_phase)
            
            # Priority 2: Frequency evolution phase
            if 'frequency_evolution' in trajectory_data:
                freq_evolution = trajectory_data['frequency_evolution']
                if isinstance(freq_evolution, (list, np.ndarray)) and len(freq_evolution) > 0:
                    freq_array = np.array(freq_evolution)
                    # Integrate frequency to get phase
                    if len(freq_array) > 1:
                        dt = 1.0  # Assume unit time steps
                        temporal_phase = np.sum(freq_array) * dt
                        logger.debug(f"Extracted temporal phase from frequency evolution: {temporal_phase:.4f}")
                        return self._normalize_phase(temporal_phase)
            
            # Priority 3: Transformative magnitude phase representation
            if 'transformative_magnitude' in trajectory_data:
                transform_mag = trajectory_data['transformative_magnitude']
                if isinstance(transform_mag, (list, np.ndarray)) and len(transform_mag) > 0:
                    transform_array = np.array(transform_mag)
                    # Compute phase from magnitude evolution
                    if len(transform_array) > 1:
                        phase_derivative = np.diff(transform_array)
                        temporal_phase = np.sum(phase_derivative)
                        logger.debug(f"Extracted temporal phase from transformative magnitude: {temporal_phase:.4f}")
                        return self._normalize_phase(temporal_phase)
            
            # Priority 4: Total transformative potential
            if 'total_transformative_potential' in trajectory_data:
                total_potential = trajectory_data['total_transformative_potential']
                if isinstance(total_potential, (int, float)) and total_potential != 0:
                    # Convert potential to phase representation
                    temporal_phase = np.arctan(total_potential)
                    logger.debug(f"Extracted temporal phase from total potential: {temporal_phase:.4f}")
                    return temporal_phase
            
            # CLAUDE.MD COMPLIANCE: No default values allowed
            logger.error("No temporal phase data found - cannot extract phase without actual data")
            raise ValueError("Temporal phase extraction requires actual trajectory data with phase components")
            
        except Exception as e:
            logger.error(f"Temporal phase extraction failed: {e}")
            raise ValueError(f"Cannot extract temporal phase without valid data: {e}")
    
    def _normalize_phase(self, phase: float) -> float:
        """
        Normalize phase to [-π, π] range.
        
        Args:
            phase: Phase value to normalize
            
        Returns:
            Normalized phase in [-π, π]
        """
        if not np.isfinite(phase):
            return 0.0
        
        # Wrap phase to [-π, π]
        normalized_phase = np.arctan2(np.sin(phase), np.cos(phase))
        return normalized_phase


# Convenience functions for external use
def extract_semantic_phase(semantic_data: Dict[str, Any]) -> float:
    """Extract semantic phase from semantic field data."""
    extractor = PhaseExtractor()
    return extractor.extract_semantic_phase(semantic_data)


def extract_emotional_phase(emotional_data: Dict[str, Any]) -> float:
    """Extract emotional phase from emotional trajectory data."""
    extractor = PhaseExtractor()
    return extractor.extract_emotional_phase(emotional_data)


def extract_temporal_phase(trajectory_data: Dict[str, Any]) -> float:
    """Extract temporal phase from trajectory operator data."""
    extractor = PhaseExtractor()
    return extractor.extract_temporal_phase(trajectory_data)