"""
Phase Analysis - Phase Coherence and Dynamics Analysis

Provides analysis tools for phase coherence, phase dynamics, and phase quality
assessment in the complete phase integration system.

ANALYSIS CAPABILITIES:
- Phase coherence metrics across dimensions
- Phase stability analysis over observational states
- Phase dynamics and evolution patterns
- Quality assessment and validation
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhaseCoherenceMetrics:
    """Metrics for phase coherence analysis."""
    coherence_magnitude: float
    coherence_phase: float
    stability_index: float
    synchronization_strength: float
    phase_spread: float
    quality_score: float


@dataclass
class PhaseDynamics:
    """Phase dynamics analysis results."""
    phase_velocity: float
    phase_acceleration: float
    group_velocity: float
    phase_dispersion: float
    stability_eigenvalue: float
    attractor_strength: float


class PhaseAnalyzer:
    """
    Analyze phase coherence, dynamics, and quality metrics.
    
    ANALYSIS METHODS:
    1. Cross-dimensional phase coherence
    2. Phase stability over time/state evolution
    3. Phase dynamics and group velocity
    4. Quality assessment and validation
    """
    
    def __init__(self,
                 coherence_window: int = 10,
                 stability_threshold: float = 0.1,
                 dynamics_resolution: float = 0.01):
        """
        Initialize phase analyzer.
        
        Args:
            coherence_window: Window size for coherence analysis
            stability_threshold: Threshold for stability classification
            dynamics_resolution: Resolution for dynamics calculation
        """
        self.coherence_window = coherence_window
        self.stability_threshold = stability_threshold
        self.dynamics_resolution = dynamics_resolution
        
        logger.info(f"Initialized PhaseAnalyzer: window={coherence_window}, "
                   f"threshold={stability_threshold}")
    
    def analyze_phase_coherence(self,
                              semantic_phases: List[float],
                              emotional_phases: List[float],
                              temporal_phases: List[float],
                              interaction_phases: List[float],
                              field_phases: List[float]) -> PhaseCoherenceMetrics:
        """
        Analyze phase coherence across all dimensions.
        
        COHERENCE ANALYSIS:
        1. Compute complex coherence from phase alignment
        2. Measure phase synchronization strength
        3. Calculate phase spread and stability
        4. Generate overall quality score
        
        Args:
            semantic_phases: Sequence of semantic phases
            emotional_phases: Sequence of emotional phases
            temporal_phases: Sequence of temporal phases
            interaction_phases: Sequence of interaction phases
            field_phases: Sequence of field phases
            
        Returns:
            PhaseCoherenceMetrics with analysis results
        """
        try:
            # Convert phases to complex exponentials
            sem_complex = np.exp(1j * np.array(semantic_phases))
            emo_complex = np.exp(1j * np.array(emotional_phases))
            temp_complex = np.exp(1j * np.array(temporal_phases))
            inter_complex = np.exp(1j * np.array(interaction_phases))
            field_complex = np.exp(1j * np.array(field_phases))
            
            # Step 1: Compute coherence magnitude
            mean_complex = np.mean([
                np.mean(sem_complex),
                np.mean(emo_complex),
                np.mean(temp_complex),
                np.mean(inter_complex),
                np.mean(field_complex)
            ])
            coherence_magnitude = abs(mean_complex)
            coherence_phase = np.angle(mean_complex)
            
            # Step 2: Phase synchronization strength
            all_phases = np.array([semantic_phases, emotional_phases, temporal_phases,
                                 interaction_phases, field_phases])
            phase_diff_matrix = np.abs(all_phases[:, np.newaxis] - all_phases[np.newaxis, :])
            phase_diff_matrix = np.minimum(phase_diff_matrix, 2*np.pi - phase_diff_matrix)
            synchronization_strength = 1.0 - np.mean(phase_diff_matrix) / np.pi
            
            # Step 3: Phase spread calculation
            all_phases_flat = np.concatenate([semantic_phases, emotional_phases,
                                            temporal_phases, interaction_phases, field_phases])
            phase_spread = np.std(all_phases_flat)
            
            # Step 4: Stability index
            if len(semantic_phases) > 1:
                phase_derivatives = []
                for phases in [semantic_phases, emotional_phases, temporal_phases,
                             interaction_phases, field_phases]:
                    if len(phases) > 1:
                        derivatives = np.diff(phases)
                        phase_derivatives.extend(derivatives)
                
                stability_index = 1.0 / (1.0 + np.std(phase_derivatives)) if phase_derivatives else 1.0
            else:
                stability_index = 1.0
            
            # Step 5: Quality score
            quality_score = self._compute_coherence_quality(
                coherence_magnitude, synchronization_strength, phase_spread, stability_index
            )
            
            logger.debug(f"Phase coherence analysis: magnitude={coherence_magnitude:.3f}, "
                        f"sync={synchronization_strength:.3f}, quality={quality_score:.3f}")
            
            return PhaseCoherenceMetrics(
                coherence_magnitude=coherence_magnitude,
                coherence_phase=coherence_phase,
                stability_index=stability_index,
                synchronization_strength=synchronization_strength,
                phase_spread=phase_spread,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Phase coherence analysis failed: {e}")
            return PhaseCoherenceMetrics(
                coherence_magnitude=0.0,
                coherence_phase=0.0,
                stability_index=0.0,
                synchronization_strength=0.0,
                phase_spread=np.pi,
                quality_score=0.0
            )
    
    def compute_phase_dynamics(self,
                             phase_sequence: List[float],
                             state_sequence: List[float]) -> PhaseDynamics:
        """
        Compute phase dynamics from phase evolution sequence.
        
        DYNAMICS ANALYSIS:
        1. Phase velocity and acceleration
        2. Group velocity calculation
        3. Phase dispersion analysis
        4. Stability and attractor analysis
        
        Args:
            phase_sequence: Sequence of phase values
            state_sequence: Corresponding observational states
            
        Returns:
            PhaseDynamics with analysis results
        """
        try:
            if len(phase_sequence) < 3 or len(state_sequence) < 3:
                logger.warning("Insufficient data for dynamics analysis")
                return PhaseDynamics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            phases = np.array(phase_sequence)
            states = np.array(state_sequence)
            
            # Step 1: Phase velocity (dφ/ds)
            phase_velocity = np.mean(np.diff(phases) / np.diff(states))
            
            # Step 2: Phase acceleration (d²φ/ds²)
            if len(phases) > 2:
                phase_accel_sequence = np.diff(phases, n=2) / np.diff(states[:-1])**2
                phase_acceleration = np.mean(phase_accel_sequence)
            else:
                phase_acceleration = 0.0
            
            # Step 3: Group velocity (analytical approximation)
            # Group velocity = dω/dk ≈ d(dφ/dt)/d(k)
            if len(phases) > 4:
                phase_freq = np.diff(phases)
                freq_gradient = np.gradient(phase_freq)
                group_velocity = np.mean(freq_gradient)
            else:
                group_velocity = phase_velocity
            
            # Step 4: Phase dispersion
            phase_spread_evolution = []
            window_size = min(5, len(phases)//2)
            for i in range(len(phases) - window_size + 1):
                window_phases = phases[i:i+window_size]
                window_spread = np.std(window_phases)
                phase_spread_evolution.append(window_spread)
            
            phase_dispersion = np.mean(phase_spread_evolution) if phase_spread_evolution else 0.0
            
            # Step 5: Stability eigenvalue (linearized stability analysis)
            if len(phases) > 3:
                # Approximate Jacobian from phase evolution
                jacobian_trace = np.mean(np.diff(phase_velocity * np.ones_like(phases[:-1])))
                stability_eigenvalue = -abs(jacobian_trace)  # Negative for stability
            else:
                stability_eigenvalue = -1.0  # Assume stable
            
            # Step 6: Attractor strength
            # Measure tendency to return to mean phase
            mean_phase = np.mean(phases)
            phase_deviations = phases - mean_phase
            restoring_force = -np.mean(phase_deviations * np.diff(np.concatenate([[0], phase_deviations])))
            attractor_strength = max(0.0, restoring_force)
            
            logger.debug(f"Phase dynamics: velocity={phase_velocity:.3f}, "
                        f"dispersion={phase_dispersion:.3f}, stability={stability_eigenvalue:.3f}")
            
            return PhaseDynamics(
                phase_velocity=phase_velocity,
                phase_acceleration=phase_acceleration,
                group_velocity=group_velocity,
                phase_dispersion=phase_dispersion,
                stability_eigenvalue=stability_eigenvalue,
                attractor_strength=attractor_strength
            )
            
        except Exception as e:
            logger.error(f"Phase dynamics computation failed: {e}")
            return PhaseDynamics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def validate_phase_quality(self,
                             phase_components: Dict[str, float],
                             expected_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Validate phase quality and detect anomalies.
        
        QUALITY VALIDATION:
        1. Range validation for each phase component
        2. Finite value checking
        3. Coherence consistency validation
        4. Anomaly detection
        
        Args:
            phase_components: Dictionary of phase components
            expected_ranges: Optional expected ranges for validation
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'quality_metrics': {}
        }
        
        default_ranges = {
            'semantic_phase': (-np.pi, np.pi),
            'emotional_phase': (-np.pi, np.pi),
            'temporal_phase': (-np.pi, np.pi),
            'interaction_phase': (-np.pi, np.pi),
            'field_phase': (-np.pi, np.pi),
            'total_phase': (-np.pi, np.pi)
        }
        
        ranges = expected_ranges or default_ranges
        
        try:
            for component, value in phase_components.items():
                # Check if value is finite
                if not np.isfinite(value):
                    validation_results['errors'].append(f"Non-finite {component}: {value}")
                    validation_results['valid'] = False
                    continue
                
                # Check range if specified
                if component in ranges:
                    min_val, max_val = ranges[component]
                    if not (min_val <= value <= max_val):
                        validation_results['warnings'].append(
                            f"{component} outside expected range [{min_val:.2f}, {max_val:.2f}]: {value:.4f}"
                        )
                
                # Compute quality metric for this component
                if component in ranges:
                    min_val, max_val = ranges[component]
                    range_size = max_val - min_val
                    normalized_distance = abs(value - (min_val + max_val) / 2) / (range_size / 2)
                    quality = max(0.0, 1.0 - normalized_distance)
                    validation_results['quality_metrics'][component] = quality
            
            # Overall quality score
            if validation_results['quality_metrics']:
                overall_quality = np.mean(list(validation_results['quality_metrics'].values()))
                validation_results['overall_quality'] = overall_quality
            else:
                validation_results['overall_quality'] = 0.0
            
            logger.debug(f"Phase validation: valid={validation_results['valid']}, "
                        f"quality={validation_results['overall_quality']:.3f}")
            
        except Exception as e:
            logger.error(f"Phase validation failed: {e}")
            validation_results['errors'].append(f"Validation error: {e}")
            validation_results['valid'] = False
        
        return validation_results
    
    def _compute_coherence_quality(self,
                                 coherence_magnitude: float,
                                 synchronization_strength: float,
                                 phase_spread: float,
                                 stability_index: float) -> float:
        """Compute overall coherence quality score."""
        # Normalize components to [0, 1]
        norm_coherence = np.clip(coherence_magnitude, 0, 1)
        norm_sync = np.clip(synchronization_strength, 0, 1)
        norm_spread = np.clip(1.0 - phase_spread / np.pi, 0, 1)  # Lower spread is better
        norm_stability = np.clip(stability_index, 0, 1)
        
        # Weighted combination
        quality = (
            0.3 * norm_coherence +
            0.3 * norm_sync +
            0.2 * norm_spread +
            0.2 * norm_stability
        )
        
        return quality


# Convenience functions for external use
def analyze_phase_coherence(semantic_phases: List[float],
                          emotional_phases: List[float],
                          temporal_phases: List[float],
                          interaction_phases: List[float],
                          field_phases: List[float]) -> PhaseCoherenceMetrics:
    """Analyze phase coherence across dimensions."""
    analyzer = PhaseAnalyzer()
    return analyzer.analyze_phase_coherence(
        semantic_phases, emotional_phases, temporal_phases,
        interaction_phases, field_phases
    )


def compute_phase_dynamics(phase_sequence: List[float],
                         state_sequence: List[float]) -> PhaseDynamics:
    """Compute phase dynamics from evolution sequence."""
    analyzer = PhaseAnalyzer()
    return analyzer.compute_phase_dynamics(phase_sequence, state_sequence)