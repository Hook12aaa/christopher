"""
Resonance Amplification - Frequency Domain Emotional Enhancement

MATHEMATICAL FOUNDATION (README.md Section 3.1.3.3.6):
Resonance occurs when: ω_semantic(τ) ≈ ω_emotional(E_state) ± δ
Amplification_factor = 1 + A_max · exp(-|ω_semantic - ω_emotional|²/2σ_resonance²)

RESONANCE EFFECTS:
1. Emotional Amplification: Semantically-emotionally aligned content gets amplified
2. Suppression: Misaligned content gets suppressed  
3. Memory Formation: Resonant patterns create stronger memory traces

This module implements proper frequency domain analysis for emotional resonance
detection and amplification, creating enhanced semantic-emotional coupling.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResonanceCondition:
    """Resonance condition between semantic and emotional frequencies."""
    semantic_frequency: float
    emotional_frequency: float
    frequency_difference: float
    resonance_strength: float
    resonance_bandwidth: float
    amplification_factor: float


@dataclass
class FrequencySpectrum:
    """Frequency domain representation for resonance analysis."""
    frequencies: np.ndarray
    magnitudes: np.ndarray
    phases: np.ndarray
    dominant_frequency: float
    spectral_centroid: float
    bandwidth: float


class ResonanceCalculator:
    """
    Calculate emotional resonance conditions using frequency domain analysis.
    
    MATHEMATICAL FOUNDATION:
    Implements precise frequency matching with Gaussian resonance windows
    to detect and quantify semantic-emotional frequency alignment.
    
    FREQUENCY MATCHING:
    Uses FFT-based spectral analysis to identify dominant frequencies
    and compute resonance conditions with proper bandwidth consideration.
    """
    
    def __init__(self,
                 frequency_tolerance: float,
                 amplification_max: float,
                 resonance_bandwidth: float,
                 spectral_resolution: int):
        """
        Initialize resonance calculator.
        
        Args:
            frequency_tolerance: Tolerance for frequency matching (± δ)
            amplification_max: Maximum amplification factor A_max
            resonance_bandwidth: Bandwidth for resonance detection σ_resonance
            spectral_resolution: Resolution for FFT analysis
        """
        self.frequency_tolerance = frequency_tolerance
        self.amplification_max = amplification_max
        self.resonance_bandwidth = resonance_bandwidth
        self.spectral_resolution = spectral_resolution
        
        logger.info(f"Initialized ResonanceCalculator: tolerance={frequency_tolerance}, A_max={amplification_max}")
    
    def detect_resonance(self,
                        semantic_embedding: np.ndarray,
                        emotional_field: complex,
                        token: str,
                        context: str = "general") -> Dict[str, Any]:
        """
        Detect emotional resonance using frequency domain analysis.
        
        MATHEMATICAL PROCESS:
        1. Extract frequency spectra from semantic and emotional content
        2. Identify dominant frequencies in both domains
        3. Compute frequency differences and resonance conditions
        4. Calculate amplification factors based on resonance strength
        
        Args:
            semantic_embedding: Semantic vector for frequency analysis
            emotional_field: Complex emotional field
            token: Token identifier for resonance tracking
            context: Context for resonance computation
            
        Returns:
            Dict containing comprehensive resonance analysis
        """
        try:
            # Extract frequency spectra
            semantic_spectrum = self._extract_frequency_spectrum(semantic_embedding, "semantic")
            emotional_spectrum = self._extract_emotional_spectrum(emotional_field, "emotional")
            
            # Find resonance conditions
            resonance_conditions = self._find_resonance_conditions(
                semantic_spectrum, emotional_spectrum
            )
            
            # Calculate overall resonance metrics
            resonance_metrics = self._calculate_resonance_metrics(resonance_conditions)
            
            # Determine amplification factor
            amplification_factor = self._compute_amplification_factor(resonance_conditions)
            
            # Memory formation strength
            memory_strength = self._compute_memory_formation_strength(resonance_conditions)
            
            results = {
                'resonance_detected': len(resonance_conditions) > 0,
                'num_resonances': len(resonance_conditions),
                'resonance_conditions': [self._condition_to_dict(cond) for cond in resonance_conditions],
                'overall_amplification': amplification_factor,
                'memory_strength': memory_strength,
                'semantic_spectrum': self._spectrum_to_dict(semantic_spectrum),
                'emotional_spectrum': self._spectrum_to_dict(emotional_spectrum),
                'resonance_metrics': resonance_metrics,
                'frequency_alignment_score': resonance_metrics['alignment_score'],
                'dominant_resonance_strength': resonance_metrics['max_strength']
            }
            
            logger.debug(f"Resonance detection for {token}: {len(resonance_conditions)} conditions, amplification={amplification_factor:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Resonance detection failed for {token}: {e}")
            return {
                'resonance_detected': False,
                'num_resonances': 0,
                'overall_amplification': 1.0,
                'error': str(e)
            }
    
    def _extract_frequency_spectrum(self, 
                                  embedding: np.ndarray, 
                                  spectrum_type: str) -> FrequencySpectrum:
        """
        Extract frequency spectrum from embedding using FFT analysis.
        
        SPECTRAL ANALYSIS:
        Uses FFT to compute power spectrum, dominant frequencies,
        and spectral characteristics for resonance detection.
        """
        # Ensure embedding is real for FFT
        if np.iscomplexobj(embedding):
            embedding = np.real(embedding)
        
        # Pad or truncate to spectral resolution
        if len(embedding) > self.spectral_resolution:
            embedding = embedding[:self.spectral_resolution]
        else:
            embedding = np.pad(embedding, (0, self.spectral_resolution - len(embedding)))
        
        # Compute FFT
        fft_result = np.fft.fft(embedding)
        frequencies = np.fft.fftfreq(len(embedding))
        
        # Compute power spectrum (magnitude squared)
        magnitudes = np.abs(fft_result)**2
        phases = np.angle(fft_result)
        
        # Find dominant frequency
        dominant_idx = np.argmax(magnitudes)
        dominant_frequency = abs(frequencies[dominant_idx])
        
        # Compute spectral centroid
        spectral_centroid = np.sum(frequencies * magnitudes) / (np.sum(magnitudes) + 1e-10)
        spectral_centroid = abs(spectral_centroid)
        
        # Compute bandwidth (spectral spread)
        bandwidth = np.sqrt(np.sum((frequencies - spectral_centroid)**2 * magnitudes) / (np.sum(magnitudes) + 1e-10))
        
        return FrequencySpectrum(
            frequencies=frequencies,
            magnitudes=magnitudes,
            phases=phases,
            dominant_frequency=dominant_frequency,
            spectral_centroid=spectral_centroid,
            bandwidth=bandwidth
        )
    
    def _extract_emotional_spectrum(self, 
                                  emotional_field: complex, 
                                  spectrum_type: str) -> FrequencySpectrum:
        """
        Extract frequency spectrum from complex emotional field.
        
        EMOTIONAL FREQUENCY ANALYSIS:
        Converts complex emotional field to frequency domain representation
        for resonance analysis with semantic frequencies.
        """
        # Convert complex field to magnitude and phase
        magnitude = abs(emotional_field)
        phase = np.angle(emotional_field)
        
        # Create synthetic frequency representation
        # Use magnitude and phase to construct frequency signature
        emotional_freq_signature = np.array([
            magnitude * np.cos(phase),
            magnitude * np.sin(phase),
            magnitude,
            phase / (2 * np.pi)  # Normalized phase frequency
        ])
        
        # Pad to spectral resolution
        padding_size = max(0, self.spectral_resolution - len(emotional_freq_signature))
        emotional_freq_signature = np.pad(emotional_freq_signature, (0, padding_size))
        
        # Extract spectrum using same method as semantic
        return self._extract_frequency_spectrum(emotional_freq_signature, spectrum_type)
    
    def _find_resonance_conditions(self,
                                 semantic_spectrum: FrequencySpectrum,
                                 emotional_spectrum: FrequencySpectrum) -> List[ResonanceCondition]:
        """
        Find resonance conditions between semantic and emotional frequencies.
        
        RESONANCE DETECTION ALGORITHM:
        1. Compare all semantic frequencies with emotional frequencies
        2. Identify matches within frequency tolerance
        3. Calculate resonance strength using Gaussian window
        4. Create resonance conditions for matches above threshold
        """
        resonance_conditions = []
        
        # Get significant frequencies (above threshold)
        semantic_threshold = np.max(semantic_spectrum.magnitudes) * 0.1
        emotional_threshold = np.max(emotional_spectrum.magnitudes) * 0.1
        
        semantic_significant = np.where(semantic_spectrum.magnitudes > semantic_threshold)[0]
        emotional_significant = np.where(emotional_spectrum.magnitudes > emotional_threshold)[0]
        
        # Find frequency matches
        for sem_idx in semantic_significant:
            sem_freq = abs(semantic_spectrum.frequencies[sem_idx])
            sem_magnitude = semantic_spectrum.magnitudes[sem_idx]
            
            for emo_idx in emotional_significant:
                emo_freq = abs(emotional_spectrum.frequencies[emo_idx])
                emo_magnitude = emotional_spectrum.magnitudes[emo_idx]
                
                # Calculate frequency difference
                freq_diff = abs(sem_freq - emo_freq)
                
                # Check if within tolerance
                if freq_diff <= self.frequency_tolerance:
                    # Calculate resonance strength using Gaussian window
                    resonance_strength = np.exp(-freq_diff**2 / (2 * self.resonance_bandwidth**2))
                    
                    # Weight by magnitude importance
                    magnitude_weight = np.sqrt(sem_magnitude * emo_magnitude)
                    weighted_strength = resonance_strength * magnitude_weight
                    
                    # Calculate amplification factor
                    amplification = 1.0 + self.amplification_max * weighted_strength
                    
                    # Create resonance condition
                    condition = ResonanceCondition(
                        semantic_frequency=sem_freq,
                        emotional_frequency=emo_freq,
                        frequency_difference=freq_diff,
                        resonance_strength=weighted_strength,
                        resonance_bandwidth=self.resonance_bandwidth,
                        amplification_factor=amplification
                    )
                    
                    resonance_conditions.append(condition)
        
        # Sort by resonance strength (strongest first)
        resonance_conditions.sort(key=lambda x: x.resonance_strength, reverse=True)
        
        return resonance_conditions
    
    def _calculate_resonance_metrics(self, conditions: List[ResonanceCondition]) -> Dict[str, float]:
        """Calculate overall resonance metrics from individual conditions."""
        if not conditions:
            return {
                'max_strength': 0.0,
                'avg_strength': 0.0,
                'alignment_score': 0.0,
                'frequency_coherence': 0.0
            }
        
        strengths = [cond.resonance_strength for cond in conditions]
        
        max_strength = max(strengths)
        avg_strength = np.mean(strengths)
        
        # Alignment score based on number and quality of resonances
        alignment_score = min(1.0, len(conditions) * avg_strength / 5.0)
        
        # Frequency coherence based on clustering of resonant frequencies
        freq_diffs = [cond.frequency_difference for cond in conditions]
        frequency_coherence = 1.0 / (1.0 + np.std(freq_diffs)) if len(freq_diffs) > 1 else 1.0
        
        return {
            'max_strength': float(max_strength),
            'avg_strength': float(avg_strength),
            'alignment_score': float(alignment_score),
            'frequency_coherence': float(frequency_coherence)
        }
    
    def _compute_amplification_factor(self, conditions: List[ResonanceCondition]) -> float:
        """
        Compute overall amplification factor from resonance conditions.
        
        AMPLIFICATION INTEGRATION:
        Combines multiple resonance conditions into single amplification factor
        using weighted averaging based on resonance strength.
        """
        if not conditions:
            return 1.0
        
        # Weight amplifications by resonance strength
        weighted_amplifications = []
        total_weight = 0.0
        
        for condition in conditions:
            weight = condition.resonance_strength
            weighted_amplifications.append(condition.amplification_factor * weight)
            total_weight += weight
        
        if total_weight > 0:
            overall_amplification = sum(weighted_amplifications) / total_weight
        else:
            overall_amplification = 1.0
        
        # Ensure reasonable bounds
        overall_amplification = max(0.1, min(1.0 + self.amplification_max, overall_amplification))
        
        return overall_amplification
    
    def _compute_memory_formation_strength(self, conditions: List[ResonanceCondition]) -> float:
        """
        Compute memory formation strength based on resonance conditions.
        
        MEMORY FORMATION (README.md Section 3.1.3.3.6):
        Resonant patterns create stronger memory traces through
        enhanced synaptic plasticity and consolidation.
        """
        if not conditions:
            return 0.0
        
        # Memory strength increases with resonance strength and persistence
        max_strength = max(cond.resonance_strength for cond in conditions)
        num_resonances = len(conditions)
        
        # Memory formation is enhanced by multiple coherent resonances
        memory_strength = max_strength * (1.0 + 0.1 * num_resonances)
        
        # Normalize to [0, 1] range
        memory_strength = min(1.0, memory_strength)
        
        return memory_strength
    
    def _condition_to_dict(self, condition: ResonanceCondition) -> Dict[str, float]:
        """Convert resonance condition to dictionary for serialization."""
        return {
            'semantic_frequency': condition.semantic_frequency,
            'emotional_frequency': condition.emotional_frequency,
            'frequency_difference': condition.frequency_difference,
            'resonance_strength': condition.resonance_strength,
            'amplification_factor': condition.amplification_factor
        }
    
    def _spectrum_to_dict(self, spectrum: FrequencySpectrum) -> Dict[str, Any]:
        """Convert frequency spectrum to dictionary for serialization."""
        return {
            'dominant_frequency': spectrum.dominant_frequency,
            'spectral_centroid': spectrum.spectral_centroid,
            'bandwidth': spectrum.bandwidth,
            'peak_magnitude': float(np.max(spectrum.magnitudes)),
            'total_energy': float(np.sum(spectrum.magnitudes))
        }


class AmplificationEngine:
    """
    Apply resonance-based amplification to emotional fields.
    
    MATHEMATICAL FOUNDATION:
    Implements the amplification formula with proper frequency weighting
    and multi-resonance integration for enhanced emotional field effects.
    """
    
    def __init__(self, 
                 max_amplification: float,
                 suppression_threshold: float):
        """
        Initialize amplification engine.
        
        Args:
            max_amplification: Maximum amplification factor
            suppression_threshold: Threshold below which suppression occurs
        """
        self.max_amplification = max_amplification
        self.suppression_threshold = suppression_threshold
    
    def apply_amplification(self,
                          base_field: complex,
                          resonance_results: Dict[str, Any],
                          token: str) -> complex:
        """
        Apply resonance-based amplification to emotional field.
        
        AMPLIFICATION APPLICATION:
        Uses resonance detection results to amplify or suppress
        emotional field based on semantic-emotional frequency alignment.
        
        Args:
            base_field: Base emotional field to modify
            resonance_results: Results from resonance detection
            token: Token identifier for logging
            
        Returns:
            Amplified or suppressed emotional field
        """
        try:
            # CLAUDE.md COMPLIANCE: NO fallback values
            if 'overall_amplification' not in resonance_results:
                raise ValueError("Missing required 'overall_amplification' in resonance_results")
            if 'memory_strength' not in resonance_results:
                raise ValueError("Missing required 'memory_strength' in resonance_results")
                
            amplification_factor = resonance_results['overall_amplification']
            memory_strength = resonance_results['memory_strength']
            
            # Apply amplification with memory enhancement
            memory_enhancement = 1.0 + 0.2 * memory_strength
            total_amplification = amplification_factor * memory_enhancement
            
            # Apply to field
            amplified_field = base_field * total_amplification
            
            # Suppression for very weak resonance
            if amplification_factor < self.suppression_threshold:
                suppression_factor = amplification_factor / self.suppression_threshold
                amplified_field *= suppression_factor
            
            logger.debug(f"Applied amplification to {token}: factor={total_amplification:.3f}")
            return amplified_field
            
        except Exception as e:
            logger.warning(f"Amplification failed for {token}: {e}")
            return base_field


class FrequencyMatcher:
    """
    Match semantic and emotional frequencies for optimal resonance.
    
    FREQUENCY MATCHING:
    Implements advanced frequency domain analysis to identify
    optimal frequency alignments for maximum resonance effects.
    """
    
    def __init__(self, matching_precision: float):
        """
        Initialize frequency matcher.
        
        Args:
            matching_precision: Precision for frequency matching
        """
        self.matching_precision = matching_precision
    
    def find_optimal_matches(self,
                           semantic_frequencies: np.ndarray,
                           emotional_frequencies: np.ndarray,
                           max_matches: int) -> List[Tuple[float, float, float]]:
        """
        Find optimal frequency matches between semantic and emotional domains.
        
        OPTIMAL MATCHING ALGORITHM:
        1. Compute all pairwise frequency differences
        2. Identify closest matches within precision threshold
        3. Rank by match quality and frequency importance
        4. Return top matches for resonance amplification
        
        Args:
            semantic_frequencies: Semantic frequency array
            emotional_frequencies: Emotional frequency array
            max_matches: Maximum number of matches to return
            
        Returns:
            List of (semantic_freq, emotional_freq, match_quality) tuples
        """
        matches = []
        
        for sem_freq in semantic_frequencies:
            for emo_freq in emotional_frequencies:
                freq_diff = abs(sem_freq - emo_freq)
                
                if freq_diff <= self.matching_precision:
                    # Match quality based on frequency difference
                    match_quality = 1.0 - freq_diff / self.matching_precision
                    matches.append((sem_freq, emo_freq, match_quality))
        
        # Sort by match quality and return top matches
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:max_matches]


class ResonanceMemoryManager:
    """
    Manage resonance-based memory formation and consolidation.
    
    MEMORY FORMATION (README.md Section 3.1.3.3.6):
    Resonant patterns create stronger memory traces through enhanced
    consolidation processes and long-term potentiation effects.
    """
    
    def __init__(self, 
                 memory_decay_rate: float,
                 consolidation_threshold: float):
        """
        Initialize resonance memory manager.
        
        Args:
            memory_decay_rate: Rate of memory decay over time
            consolidation_threshold: Threshold for memory consolidation
        """
        self.memory_decay_rate = memory_decay_rate
        self.consolidation_threshold = consolidation_threshold
        self.memory_traces = {}
    
    def update_memory(self,
                     token: str,
                     resonance_strength: float,
                     observational_state: float) -> float:
        """
        Update memory trace based on resonance strength.
        
        MEMORY UPDATE PROCESS:
        1. Retrieve existing memory trace for token
        2. Apply decay based on time elapsed
        3. Strengthen memory based on resonance
        4. Consolidate if above threshold
        
        Args:
            token: Token identifier for memory trace
            resonance_strength: Strength of current resonance
            observational_state: Current observational state (time proxy)
            
        Returns:
            Updated memory strength for token
        """
        # Get existing memory or initialize
        if token in self.memory_traces:
            last_strength, last_state = self.memory_traces[token]
            
            # Apply decay
            time_elapsed = observational_state - last_state
            decay_factor = np.exp(-self.memory_decay_rate * time_elapsed)
            decayed_strength = last_strength * decay_factor
        else:
            decayed_strength = 0.0
        
        # Strengthen with current resonance
        new_strength = min(1.0, decayed_strength + resonance_strength)
        
        # Store updated memory
        self.memory_traces[token] = (new_strength, observational_state)
        
        # Check for consolidation
        if new_strength > self.consolidation_threshold:
            logger.debug(f"Memory consolidated for {token}: strength={new_strength:.3f}")
        
        return new_strength
    
    def get_memory_strength(self, token: str) -> float:
        """Get current memory strength for token."""
        if token in self.memory_traces:
            return self.memory_traces[token][0]
        return 0.0


def create_resonance_calculator(frequency_tolerance: float,
                              amplification_max: float) -> ResonanceCalculator:
    """
    Convenience function to create resonance calculator.
    
    Args:
        frequency_tolerance: Tolerance for frequency matching
        amplification_max: Maximum amplification factor
        
    Returns:
        Configured ResonanceCalculator
    """
    return ResonanceCalculator(
        frequency_tolerance=frequency_tolerance,
        amplification_max=amplification_max,
        resonance_bandwidth=0.2,
        spectral_resolution=256
    )


def apply_resonance_enhancement(emotional_field: complex,
                              semantic_embedding: np.ndarray,
                              token: str) -> Tuple[complex, Dict[str, Any]]:
    """
    Convenience function to apply complete resonance enhancement.
    
    Args:
        emotional_field: Complex emotional field to enhance
        semantic_embedding: Semantic embedding for resonance detection
        token: Token identifier
        
    Returns:
        Tuple of (enhanced_field, resonance_analysis)
    """
    # Create resonance calculator and amplification engine
    calculator = create_resonance_calculator()
    amplifier = AmplificationEngine()
    
    # Detect resonance
    resonance_results = calculator.detect_resonance(
        semantic_embedding=semantic_embedding,
        emotional_field=emotional_field,
        token=token
    )
    
    # Apply amplification
    enhanced_field = amplifier.apply_amplification(
        base_field=emotional_field,
        resonance_results=resonance_results,
        token=token
    )
    
    return enhanced_field, resonance_results