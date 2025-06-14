"""
Frequency Analysis Engine

High-performance spectral analysis for phase integration components
in field theory applications. Implements FFT-based computations for
e^(iθ_total(τ,C,s)) phase integration in conceptual charge formulas.
"""

import numpy as np
import numba as nb
from typing import Tuple, Dict, Any


class FrequencyAnalyzer:
    """
    Enterprise-grade frequency domain analysis for field theory applications.
    
    MATHEMATICAL PURPOSE: Computes spectral properties essential for phase
    integration e^(iθ_total(τ,C,s)) in the complete Q(τ, C, s) conceptual
    charge formula. Provides frequency-domain field characteristics.
    
    PERFORMANCE: Uses actual FFT computations for accurate spectral analysis.
    Spectral functions require complex number operations not compatible with
    numba nopython mode, but provide exact frequency domain calculations.
    """
    
    @staticmethod
    def compute_spectral_features(embedding: np.ndarray) -> Tuple[np.ndarray, 
                                                                 np.ndarray, 
                                                                 np.ndarray]:
        """
        Extract spectral features for phase integration analysis.
        
        FIELD THEORY APPLICATION: Computes frequency domain properties needed
        for e^(iθ_total(τ,C,s)) phase integration in conceptual charge formula.
        Provides dominant frequencies and phase angle distributions.
        
        MATHEMATICAL IMPLEMENTATION:
        - FFT analysis for power spectrum computation
        - Dominant frequency identification for phase integration
        - Complex phase angle extraction for field coherence
        
        Args:
            embedding: Input embedding vector [D]
            
        Returns:
            Tuple of spectral features:
            - dominant_frequencies: Normalized dominant frequency indices
            - frequency_magnitudes: Power spectrum magnitudes at dominant frequencies  
            - phase_angles: Complex phase angles from real/imaginary decomposition
        """
        # FFT-based power spectrum analysis - actual computation
        fft_result = np.fft.fft(embedding)
        power_spectrum = np.abs(fft_result)**2
        
        # Identify dominant frequencies (top 10)
        top_indices = np.argsort(power_spectrum)[-10:]
        dominant_frequencies = top_indices.astype(np.float64) / len(embedding)
        frequency_magnitudes = power_spectrum[top_indices]
        
        # Phase angle analysis for complex field representation
        half_len = len(embedding) // 2
        real_part = embedding[:half_len]
        imag_part = embedding[half_len:half_len+len(real_part)]
        complex_embedding = real_part + 1j * imag_part
        phase_angles = np.angle(complex_embedding)
        
        return dominant_frequencies, frequency_magnitudes, phase_angles
    
    @staticmethod
    def compute_phase_coherence(embeddings: np.ndarray) -> float:
        """
        Compute phase coherence across multiple embeddings.
        
        FIELD THEORY APPLICATION: Measures field coherence for stable
        phase relationships in conceptual charge calculations.
        
        Args:
            embeddings: Matrix of embedding vectors [N, D]
            
        Returns:
            Phase coherence measure [0, 1] - higher indicates better coherence
        """
        if len(embeddings) < 2:
            return 1.0
        
        # Compute phase angles for each embedding
        total_coherence = 0.0
        comparisons = 0
        
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                # Extract phase information
                half_len = embeddings.shape[1] // 2
                
                # Embedding i phase
                real_i = embeddings[i, :half_len]
                imag_i = embeddings[i, half_len:half_len+len(real_i)]
                phase_i = np.angle(real_i + 1j * imag_i)
                
                # Embedding j phase  
                real_j = embeddings[j, :half_len]
                imag_j = embeddings[j, half_len:half_len+len(real_j)]
                phase_j = np.angle(real_j + 1j * imag_j)
                
                # Phase coherence via circular correlation
                phase_diff = np.abs(phase_i - phase_j)
                # Wrap to [0, π] for circular distance
                phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)
                coherence = 1.0 - np.mean(phase_diff) / np.pi
                
                total_coherence += coherence
                comparisons += 1
        
        return total_coherence / comparisons if comparisons > 0 else 1.0
    
    @staticmethod
    def compute_spectral_entropy(power_spectrum: np.ndarray) -> float:
        """
        Compute spectral entropy for frequency distribution analysis.
        
        FIELD THEORY APPLICATION: Measures spectral complexity for
        field entropy calculations in thermodynamic field theory.
        
        Args:
            power_spectrum: Power spectrum from FFT analysis
            
        Returns:
            Spectral entropy measure - higher indicates more distributed spectrum
        """
        # Normalize power spectrum to probability distribution
        total_power = np.sum(power_spectrum) + 1e-10
        prob_distribution = power_spectrum / total_power
        
        # Shannon entropy calculation
        entropy = 0.0
        for p in prob_distribution:
            if p > 1e-10:
                entropy -= p * np.log(p)
        
        return entropy
    
    @staticmethod
    def analyze_spectral_properties(embedding: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive spectral analysis for field theory applications.
        
        ENTERPRISE INTERFACE: Provides complete frequency-domain analysis
        including power spectrum, phase relationships, and spectral entropy
        for advanced field theory calculations.
        
        Args:
            embedding: Input embedding vector for spectral analysis
            
        Returns:
            Dictionary of comprehensive spectral properties
        """
        # Core spectral feature extraction
        dominant_frequencies, frequency_magnitudes, phase_angles = \
            FrequencyAnalyzer.compute_spectral_features(embedding)
        
        # Power spectrum analysis
        fft_result = np.fft.fft(embedding)
        power_spectrum = np.abs(fft_result)**2
        
        # Spectral statistics
        spectral_entropy = FrequencyAnalyzer.compute_spectral_entropy(power_spectrum)
        spectral_centroid = np.sum(
            np.arange(len(power_spectrum)) * power_spectrum
        ) / (np.sum(power_spectrum) + 1e-10)
        
        # Phase statistics
        phase_variance = np.var(phase_angles)
        phase_mean = np.mean(phase_angles)
        
        # Frequency bandwidth analysis
        peak_frequency = dominant_frequencies[np.argmax(frequency_magnitudes)]
        frequency_bandwidth = np.std(dominant_frequencies)
        
        return {
            # Core spectral features
            'dominant_frequencies': dominant_frequencies.tolist(),
            'frequency_magnitudes': frequency_magnitudes.tolist(),
            'phase_angles': phase_angles.tolist(),
            
            # Spectral statistics
            'spectral_entropy': float(spectral_entropy),
            'spectral_centroid': float(spectral_centroid),
            'peak_frequency': float(peak_frequency),
            'frequency_bandwidth': float(frequency_bandwidth),
            
            # Phase analysis
            'phase_variance': float(phase_variance),
            'phase_mean': float(phase_mean),
            'phase_complexity': float(phase_variance * spectral_entropy),
            
            # Field theory parameters
            'phase_integration_factor': float(np.exp(-phase_variance)),
            'spectral_field_strength': float(np.max(frequency_magnitudes)),
            'frequency_distribution_width': float(frequency_bandwidth)
        }
    
    @staticmethod
    def compute_batch_spectral_analysis(embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Batch spectral analysis for multiple embeddings.
        
        ENTERPRISE FEATURE: Processes multiple embeddings simultaneously
        for efficient spectral analysis in production systems.
        
        Args:
            embeddings: Matrix of embedding vectors [N, D]
            
        Returns:
            Aggregated spectral analysis results
        """
        n_embeddings = len(embeddings)
        
        # Aggregate spectral features
        all_dominant_freqs = []
        all_freq_magnitudes = []
        all_phase_angles = []
        
        for i in range(n_embeddings):
            dom_freqs, freq_mags, phase_angles = \
                FrequencyAnalyzer.compute_spectral_features(embeddings[i])
            
            all_dominant_freqs.append(dom_freqs)
            all_freq_magnitudes.append(freq_mags) 
            all_phase_angles.append(phase_angles)
        
        # Compute batch statistics
        phase_coherence = FrequencyAnalyzer.compute_phase_coherence(embeddings)
        
        # Aggregate metrics
        mean_dominant_freqs = np.mean(all_dominant_freqs, axis=0)
        mean_freq_magnitudes = np.mean(all_freq_magnitudes, axis=0)
        freq_stability = 1.0 / (1.0 + np.std(all_dominant_freqs, axis=0))
        
        return {
            'batch_size': n_embeddings,
            'mean_dominant_frequencies': mean_dominant_freqs.tolist(),
            'mean_frequency_magnitudes': mean_freq_magnitudes.tolist(),
            'frequency_stability': freq_stability.tolist(),
            'phase_coherence': float(phase_coherence),
            'spectral_consistency': float(np.mean(freq_stability))
        }