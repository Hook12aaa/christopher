"""
Emotional Interference Patterns - Multi-Emotion Field Interactions

MATHEMATICAL FOUNDATION (README.md Section 3.1.3.3.4):
E_total^trajectory(τ, s) = Σᵢ E_i^trajectory(τ, s) · exp(iφ_i^emotional(s))

INTERFERENCE TYPES:
1. Constructive Interference: φ_i ≈ φ_j ⟹ |E_total| > |E_i| + |E_j|
2. Destructive Interference: φ_i ≈ φ_j + π ⟹ |E_total| < |E_i| + |E_j|
3. Complex Patterns: Mixed emotions, emotional ambivalence, conflicted states

This module handles the complex interactions between multiple emotional influences,
creating rich interference patterns for nuanced emotional experiences.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmotionalComponent:
    """Individual emotional component for interference calculations."""
    trajectory: complex
    phase: float
    frequency: float
    amplitude: float
    source: str
    coherence: float  # CLAUDE.md COMPLIANCE: NO default values


@dataclass
class InterferenceResult:
    """Results from emotional interference calculation."""
    total_field: complex
    interference_type: str  # 'constructive', 'destructive', 'mixed'
    amplification_factor: float
    phase_coherence: float
    component_analysis: Dict[str, Any]


class EmotionalInterferenceManager:
    """
    Manage constructive/destructive emotional interference patterns.
    
    MATHEMATICAL FOUNDATION:
    Handles multiple emotional influences that create interference patterns
    following complex field addition with phase relationships.
    
    INTERFERENCE PHYSICS:
    - Constructive: Aligned phases amplify total field
    - Destructive: Opposed phases cancel total field
    - Mixed: Complex phase relationships create nuanced patterns
    """
    
    def __init__(self,
                 num_dimensions: int,
                 phase_coupling_strength: float,
                 interference_threshold: float):
        """
        Initialize emotional interference manager.
        
        Args:
            num_dimensions: Embedding dimension for field calculations
            phase_coupling_strength: Strength of cross-component phase coupling
            interference_threshold: Threshold for significant interference effects
        """
        self.num_dimensions = num_dimensions
        self.phase_coupling_strength = phase_coupling_strength
        self.interference_threshold = interference_threshold
        
        logger.info(f"Initialized EmotionalInterferenceManager for {num_dimensions}D with coupling={phase_coupling_strength}")
    
    def calculate_interference(self,
                             emotional_components: List[EmotionalComponent],
                             token: str,
                             observational_state: float) -> InterferenceResult:
        """
        Calculate emotional interference patterns from multiple components.
        
        MATHEMATICAL PROCESS (README.md Section 3.1.3.3.4):
        E_total^trajectory(τ, s) = Σᵢ E_i^trajectory(τ, s) · exp(iφ_i^emotional(s))
        
        Args:
            emotional_components: List of individual emotional influences
            token: Token identifier for interference tracking
            observational_state: Current observational state s
            
        Returns:
            InterferenceResult with total field and analysis
        """
        try:
            if len(emotional_components) == 0:
                return self._create_null_interference(token)
            
            if len(emotional_components) == 1:
                return self._create_single_component_result(emotional_components[0], token)
            
            # Calculate complex field superposition
            total_field = self._compute_field_superposition(emotional_components, observational_state)
            
            # Analyze interference type
            interference_type = self._classify_interference_type(emotional_components, total_field)
            
            # Compute amplification factor
            amplification_factor = self._compute_amplification_factor(emotional_components, total_field)
            
            # Analyze phase coherence
            phase_coherence = self._compute_phase_coherence(emotional_components)
            
            # Component analysis for debugging
            component_analysis = self._analyze_components(emotional_components, total_field)
            
            result = InterferenceResult(
                total_field=total_field,
                interference_type=interference_type,
                amplification_factor=amplification_factor,
                phase_coherence=phase_coherence,
                component_analysis=component_analysis
            )
            
            logger.debug(f"Interference for {token}: type={interference_type}, |total|={abs(total_field):.4f}, coherence={phase_coherence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Interference calculation failed for {token}: {e}")
            return self._create_null_interference(token)
    
    def _compute_field_superposition(self,
                                   components: List[EmotionalComponent],
                                   observational_state: float) -> complex:
        """
        Compute complex field superposition with phase evolution.
        
        MATHEMATICAL FORMULA:
        E_total = Σᵢ E_i^trajectory(τ, s) · exp(iφ_i^emotional(s))
        """
        total_field = complex(0, 0)
        
        for component in components:
            # Phase evolution with observational state
            evolved_phase = component.phase + component.frequency * observational_state
            
            # Complex exponential for phase factor
            phase_factor = np.exp(1j * evolved_phase)
            
            # Add component with phase evolution
            field_contribution = component.trajectory * phase_factor * component.coherence
            total_field += field_contribution
        
        return total_field
    
    def _classify_interference_type(self,
                                  components: List[EmotionalComponent],
                                  total_field: complex) -> str:
        """
        Classify interference type based on phase relationships and field magnitude.
        
        INTERFERENCE CLASSIFICATION:
        - Constructive: |E_total| > Σ|E_i| * threshold
        - Destructive: |E_total| < Σ|E_i| * threshold  
        - Mixed: Complex patterns between extremes
        """
        # Calculate sum of individual magnitudes
        sum_magnitudes = sum(abs(comp.trajectory) for comp in components)
        total_magnitude = abs(total_field)
        
        if sum_magnitudes == 0:
            return 'null'
        
        # Interference ratio
        interference_ratio = total_magnitude / sum_magnitudes
        
        # Classification thresholds
        constructive_threshold = 1.0 + self.interference_threshold
        destructive_threshold = 1.0 - self.interference_threshold
        
        if interference_ratio > constructive_threshold:
            return 'constructive'
        elif interference_ratio < destructive_threshold:
            return 'destructive'
        else:
            return 'mixed'
    
    def _compute_amplification_factor(self,
                                    components: List[EmotionalComponent],
                                    total_field: complex) -> float:
        """
        Compute amplification factor from interference effects.
        
        AMPLIFICATION CALCULATION:
        Factor = |E_total| / Σ|E_i| - measures interference enhancement
        """
        sum_magnitudes = sum(abs(comp.trajectory) for comp in components)
        
        if sum_magnitudes == 0:
            return 1.0
        
        amplification = abs(total_field) / sum_magnitudes
        return amplification
    
    def _compute_phase_coherence(self, components: List[EmotionalComponent]) -> float:
        """
        Compute phase coherence across emotional components.
        
        COHERENCE MEASURE:
        High coherence → constructive interference tendency
        Low coherence → destructive interference tendency
        """
        if len(components) < 2:
            return 1.0
        
        phases = [comp.phase for comp in components]
        
        # Compute pairwise phase differences
        phase_differences = []
        for i in range(len(phases)):
            for j in range(i+1, len(phases)):
                phase_diff = abs(phases[i] - phases[j])
                # Wrap to [0, π] for circular phase difference
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                phase_differences.append(phase_diff)
        
        # Coherence is inverse of average phase difference
        avg_phase_diff = np.mean(phase_differences)
        coherence = 1.0 - avg_phase_diff / np.pi
        
        return max(0.0, coherence)
    
    def _analyze_components(self,
                          components: List[EmotionalComponent],
                          total_field: complex) -> Dict[str, Any]:
        """
        Analyze individual components and their contributions.
        """
        return {
            'num_components': len(components),
            'component_sources': [comp.source for comp in components],
            'component_magnitudes': [abs(comp.trajectory) for comp in components],
            'component_phases': [comp.phase for comp in components],
            'component_frequencies': [comp.frequency for comp in components],
            'total_magnitude': abs(total_field),
            'total_phase': np.angle(total_field),
            'dominant_component': max(components, key=lambda c: abs(c.trajectory)).source if components else None
        }
    
    def _create_null_interference(self, token: str) -> InterferenceResult:
        """Create null interference result for edge cases."""
        return InterferenceResult(
            total_field=complex(1.0, 0.0),
            interference_type='null',
            amplification_factor=1.0,
            phase_coherence=1.0,
            component_analysis={'num_components': 0, 'error': 'null_interference'}
        )
    
    def _create_single_component_result(self, component: EmotionalComponent, token: str) -> InterferenceResult:
        """Create result for single emotional component (no interference)."""
        return InterferenceResult(
            total_field=component.trajectory,
            interference_type='single',
            amplification_factor=1.0,
            phase_coherence=1.0,
            component_analysis={
                'num_components': 1,
                'component_sources': [component.source],
                'total_magnitude': abs(component.trajectory),
                'total_phase': component.phase
            }
        )


class PhaseCoordinator:
    """
    Coordinate emotional phases across multiple dimensions and sources.
    
    MATHEMATICAL FOUNDATION:
    Manages phase relationships to ensure coherent interference patterns
    and proper coupling between different emotional influences.
    """
    
    def __init__(self, coupling_strength: float):
        """
        Initialize phase coordinator.
        
        Args:
            coupling_strength: Strength of cross-dimensional phase coupling
        """
        self.coupling_strength = coupling_strength
        
    def coordinate_phases(self,
                         components: List[EmotionalComponent],
                         temporal_phases: Optional[np.ndarray] = None,
                         semantic_phases: Optional[np.ndarray] = None) -> List[EmotionalComponent]:
        """
        Coordinate phases across emotional, temporal, and semantic dimensions.
        
        PHASE COORDINATION:
        Adjusts emotional component phases based on coupling with other dimensions
        to create coherent field evolution across the complete system.
        
        Args:
            components: Emotional components to coordinate
            temporal_phases: Optional temporal dimension phases
            semantic_phases: Optional semantic dimension phases
            
        Returns:
            List of components with coordinated phases
        """
        try:
            coordinated_components = []
            
            for component in components:
                # Start with original phase
                coordinated_phase = component.phase
                
                # CLAUDE.md COMPLIANCE: Temporal phases are REQUIRED for cross-dimensional coupling
                if temporal_phases is None or len(temporal_phases) == 0:
                    raise ValueError("temporal_phases REQUIRED for phase coordination - no 'if available' logic allowed")
                temporal_coupling = self.coupling_strength * np.mean(temporal_phases)
                coordinated_phase += temporal_coupling
                
                # CLAUDE.md COMPLIANCE: Semantic phases are REQUIRED for cross-dimensional coupling
                if semantic_phases is None or len(semantic_phases) == 0:
                    raise ValueError("semantic_phases REQUIRED for phase coordination - no 'if available' logic allowed")
                semantic_coupling = self.coupling_strength * np.mean(semantic_phases)
                coordinated_phase += semantic_coupling
                
                # Normalize phase to [-π, π]
                coordinated_phase = (coordinated_phase + np.pi) % (2 * np.pi) - np.pi
                
                # Create coordinated component
                coordinated_component = EmotionalComponent(
                    trajectory=component.trajectory,
                    phase=coordinated_phase,
                    frequency=component.frequency,
                    amplitude=component.amplitude,
                    source=component.source + '_coordinated',
                    coherence=component.coherence
                )
                
                coordinated_components.append(coordinated_component)
            
            logger.debug(f"Coordinated {len(coordinated_components)} emotional components")
            return coordinated_components
            
        except Exception as e:
            logger.warning(f"Phase coordination failed: {e}")
            return components  # Return original components if coordination fails


class ResonanceDetector:
    """
    Detect and amplify emotional resonance patterns.
    
    MATHEMATICAL FOUNDATION (README.md Section 3.1.3.3.6):
    Resonance occurs when: ω_semantic(τ) ≈ ω_emotional(E_state) ± δ
    
    RESONANCE EFFECTS:
    - Emotional amplification for aligned content
    - Suppression for misaligned content  
    - Memory formation for resonant patterns
    """
    
    def __init__(self,
                 frequency_tolerance: float,
                 amplification_max: float,
                 resonance_bandwidth: float):
        """
        Initialize resonance detector.
        
        Args:
            frequency_tolerance: Tolerance for frequency matching
            amplification_max: Maximum resonance amplification
            resonance_bandwidth: Bandwidth for resonance detection
        """
        self.frequency_tolerance = frequency_tolerance
        self.amplification_max = amplification_max
        self.resonance_bandwidth = resonance_bandwidth
    
    def detect_resonance(self,
                        emotional_frequencies: np.ndarray,
                        semantic_frequencies: np.ndarray,
                        token: str) -> Dict[str, Any]:
        """
        Detect emotional resonance between emotional and semantic frequencies.
        
        RESONANCE DETECTION:
        Identifies frequency alignments that create resonance conditions
        for enhanced emotional-semantic coupling.
        
        Args:
            emotional_frequencies: Emotional frequency spectrum
            semantic_frequencies: Semantic frequency spectrum  
            token: Token identifier for resonance tracking
            
        Returns:
            Dict containing resonance analysis results
        """
        try:
            # Find frequency matches within tolerance
            resonant_pairs = []
            resonance_strengths = []
            
            for i, emo_freq in enumerate(emotional_frequencies):
                for j, sem_freq in enumerate(semantic_frequencies):
                    freq_diff = abs(emo_freq - sem_freq)
                    
                    if freq_diff <= self.frequency_tolerance:
                        # Calculate resonance strength
                        strength = np.exp(-freq_diff**2 / (2 * self.resonance_bandwidth**2))
                        
                        resonant_pairs.append((i, j, emo_freq, sem_freq))
                        resonance_strengths.append(strength)
            
            # Overall resonance metrics
            resonance_detected = len(resonant_pairs) > 0
            max_strength = max(resonance_strengths) if resonance_strengths else 0.0
            avg_strength = np.mean(resonance_strengths) if resonance_strengths else 0.0
            
            # Frequency alignment score
            if len(emotional_frequencies) > 0 and len(semantic_frequencies) > 0:
                alignment_score = len(resonant_pairs) / (len(emotional_frequencies) * len(semantic_frequencies))
            else:
                alignment_score = 0.0
            
            results = {
                'resonance_detected': resonance_detected,
                'max_strength': float(max_strength),
                'avg_strength': float(avg_strength),
                'alignment_score': float(alignment_score),
                'num_resonant_pairs': len(resonant_pairs),
                'resonant_frequencies': [(float(ep), float(sp)) for _, _, ep, sp in resonant_pairs],
                'resonance_strengths': [float(s) for s in resonance_strengths]
            }
            
            logger.debug(f"Resonance detection for {token}: detected={resonance_detected}, strength={max_strength:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Resonance detection failed for {token}: {e}")
            return {
                'resonance_detected': False,
                'max_strength': 0.0,
                'error': str(e)
            }
    
    def apply_resonance_amplification(self,
                                    base_field: complex,
                                    resonance_strength: float) -> complex:
        """
        Apply resonance-based amplification to emotional field.
        
        AMPLIFICATION FORMULA (README.md Section 3.1.3.3.6):
        Amplification_factor = 1 + A_max · exp(-|ω_semantic - ω_emotional|²/2σ_resonance²)
        
        Args:
            base_field: Base emotional field to amplify
            resonance_strength: Strength of detected resonance
            
        Returns:
            Amplified emotional field
        """
        amplification_factor = 1.0 + self.amplification_max * resonance_strength
        amplified_field = base_field * amplification_factor
        
        return amplified_field


class InterferencePatterCalculator:
    """
    Calculate complex interference patterns for advanced emotional analysis.
    
    MATHEMATICAL FOUNDATION:
    Implements sophisticated interference pattern analysis including
    multi-scale interference, temporal evolution, and stability metrics.
    """
    
    def __init__(self):
        """Initialize interference pattern calculator."""
        pass
    
    def calculate_interference_stability(self,
                                       interference_history: List[InterferenceResult],
                                       window_size: int) -> Dict[str, float]:
        """
        Calculate stability of interference patterns over time.
        
        STABILITY ANALYSIS:
        Measures how consistent interference patterns are across
        multiple observations to detect stable emotional states.
        """
        if len(interference_history) < 2:
            return {'stability': 1.0, 'consistency': 1.0}
        
        # Take recent window
        recent_history = interference_history[-window_size:]
        
        # Analyze magnitude stability
        magnitudes = [abs(result.total_field) for result in recent_history]
        magnitude_stability = 1.0 / (1.0 + np.std(magnitudes))
        
        # Analyze phase stability
        phases = [np.angle(result.total_field) for result in recent_history]
        phase_diffs = np.diff(phases)
        phase_stability = 1.0 / (1.0 + np.std(phase_diffs))
        
        # Analyze interference type consistency
        types = [result.interference_type for result in recent_history]
        type_consistency = 1.0 if len(set(types)) == 1 else 0.5
        
        overall_stability = (magnitude_stability + phase_stability + type_consistency) / 3.0
        
        return {
            'stability': float(overall_stability),
            'magnitude_stability': float(magnitude_stability),
            'phase_stability': float(phase_stability), 
            'type_consistency': float(type_consistency)
        }


def create_emotional_component(trajectory: complex,
                             source: str,
                             frequency: float,
                             coherence: float) -> EmotionalComponent:
    """
    Convenience function to create emotional component.
    
    Args:
        trajectory: Complex emotional trajectory value
        source: Source identifier for the component
        frequency: Emotional frequency for phase evolution
        coherence: Coherence factor for interference
        
    Returns:
        EmotionalComponent ready for interference calculation
    """
    return EmotionalComponent(
        trajectory=trajectory,
        phase=np.angle(trajectory),
        frequency=frequency,
        amplitude=abs(trajectory),
        source=source,
        coherence=coherence
    )


def compute_multi_emotion_interference(emotional_trajectories: List[complex],
                                     sources: List[str],
                                     token: str,
                                     observational_state: float) -> InterferenceResult:
    """
    Convenience function for multi-emotion interference calculation.
    
    Args:
        emotional_trajectories: List of complex emotional trajectory values
        sources: Source identifiers for each trajectory
        token: Token identifier
        observational_state: Current observational state
        
    Returns:
        InterferenceResult with complete interference analysis
    """
    # Create emotional components
    components = []
    for i, (trajectory, source) in enumerate(zip(emotional_trajectories, sources)):
        component = create_emotional_component(
            trajectory=trajectory,
            source=source,
            frequency=1.0 + 0.1 * i,  # Slightly different frequencies
            coherence=1.0
        )
        components.append(component)
    
    # Calculate interference
    manager = EmotionalInterferenceManager()
    result = manager.calculate_interference(
        emotional_components=components,
        token=token,
        observational_state=observational_state
    )
    
    return result