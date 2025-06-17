"""
Frequency Resonance Analysis - Discovering Emotional Patterns through Spectral Properties

KEY INSIGHT: Emotional content creates distinctive frequency signatures in embedding space.
We discover these natural resonance patterns without categorizing them as specific emotions.

APPROACH:
- FFT analysis to extract frequency components
- Wavelet decomposition for multi-scale patterns
- Spectral energy distribution analysis
- Phase relationship discovery
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.fft import fft, fftfreq, fft2, ifft
from scipy.signal import find_peaks, welch, spectrogram, hilbert
import pywt  # PyWavelets for wavelet decomposition
from dataclasses import dataclass

from Sysnpire.utils.logger import get_logger
logger = get_logger(__name__)


@dataclass
class SpectralEmotionalPattern:
    """
    Discovered spectral pattern that may indicate emotional resonance.
    NO emotion labels - only frequency domain properties.
    """
    dominant_frequencies: np.ndarray     # Primary frequency components
    frequency_amplitudes: np.ndarray     # Amplitude at each frequency
    phase_spectrum: np.ndarray          # Phase relationships
    wavelet_coefficients: Dict          # Multi-scale wavelet patterns
    spectral_energy: float              # Total energy in frequency domain
    resonance_peaks: np.ndarray         # Frequency peaks (potential resonances)
    coherence_profile: np.ndarray       # Phase coherence across dimensions


class FrequencyResonanceAnalyzer:
    """
    Discovers intrinsic emotional patterns through frequency domain analysis.
    
    PHILOSOPHY: Emotions create characteristic "rhythms" or frequency patterns
    in embedding space. We discover these without labeling them.
    """
    
    def __init__(self, embedding_dim: int = 1024, sample_rate: float = 1.0):
        """
        Initialize frequency analyzer.
        
        Args:
            embedding_dim: Dimension of embeddings
            sample_rate: Conceptual "sampling rate" for frequency analysis
        """
        self.embedding_dim = embedding_dim
        self.sample_rate = sample_rate
        self.wavelet = 'db4'  # Daubechies wavelet for multi-scale analysis
        
    def discover_frequency_patterns(self, embeddings: np.ndarray) -> List[SpectralEmotionalPattern]:
        """
        Discover natural frequency patterns that may indicate emotional resonance.
        
        Args:
            embeddings: Array of BGE embeddings to analyze
            
        Returns:
            List of discovered spectral patterns (no emotion labels)
        """
        logger.info(f"🎵 Discovering frequency patterns in {len(embeddings)} embeddings")
        
        patterns = []
        
        # Analyze each embedding's spectral properties
        for i, embedding in enumerate(embeddings):
            # 1. FFT analysis
            fft_pattern = self._analyze_fft_spectrum(embedding)
            
            # 2. Wavelet decomposition
            wavelet_pattern = self._wavelet_analysis(embedding)
            
            # 3. Phase analysis
            phase_pattern = self._analyze_phase_relationships(embedding)
            
            # 4. Detect resonance peaks
            resonance_peaks = self._detect_resonance_peaks(fft_pattern)
            
            # Create spectral pattern
            pattern = SpectralEmotionalPattern(
                dominant_frequencies=fft_pattern['dominant_freqs'],
                frequency_amplitudes=fft_pattern['amplitudes'],
                phase_spectrum=phase_pattern['phases'],
                wavelet_coefficients=wavelet_pattern,
                spectral_energy=fft_pattern['total_energy'],
                resonance_peaks=resonance_peaks,
                coherence_profile=phase_pattern['coherence']
            )
            
            # Only keep patterns with significant spectral structure
            if pattern.spectral_energy > self._energy_threshold(embeddings):
                patterns.append(pattern)
        
        logger.info(f"🎼 Discovered {len(patterns)} significant spectral patterns")
        return patterns
    
    def _analyze_fft_spectrum(self, embedding: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze frequency spectrum using FFT.
        
        Theory: Emotional content creates characteristic frequency signatures
        that we can discover without labeling.
        """
        # Apply FFT to embedding vector
        fft_result = fft(embedding)
        frequencies = fftfreq(len(embedding), d=1/self.sample_rate)
        
        # Get magnitude spectrum (ignore negative frequencies)
        positive_freq_idx = frequencies > 0
        positive_freqs = frequencies[positive_freq_idx]
        amplitudes = np.abs(fft_result[positive_freq_idx])
        
        # Find dominant frequencies
        top_k = 10
        dominant_idx = np.argsort(amplitudes)[-top_k:]
        dominant_freqs = positive_freqs[dominant_idx]
        
        # Calculate total spectral energy
        total_energy = np.sum(amplitudes**2)
        
        return {
            'frequencies': positive_freqs,
            'amplitudes': amplitudes,
            'dominant_freqs': dominant_freqs,
            'dominant_amps': amplitudes[dominant_idx],
            'total_energy': total_energy,
            'full_spectrum': fft_result
        }
    
    def _wavelet_analysis(self, embedding: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Multi-scale analysis using wavelet decomposition.
        
        Theory: Emotions may manifest at different scales - wavelets help
        discover these multi-scale patterns.
        """
        # Perform wavelet decomposition
        max_level = pywt.dwt_max_level(len(embedding), self.wavelet)
        coeffs = pywt.wavedec(embedding, self.wavelet, level=min(max_level, 5))
        
        # Extract features from each scale
        wavelet_features = {}
        for i, coeff in enumerate(coeffs):
            level_name = f'level_{i}'
            wavelet_features[level_name] = {
                'coefficients': coeff,
                'energy': np.sum(coeff**2),
                'mean_magnitude': np.mean(np.abs(coeff)),
                'sparsity': np.sum(np.abs(coeff) > 0.1) / len(coeff)
            }
        
        return wavelet_features
    
    def _analyze_phase_relationships(self, embedding: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze phase relationships using Hilbert transform.
        
        Theory: Phase coherence patterns may indicate emotional field effects.
        """
        # Apply Hilbert transform to get analytic signal
        analytic_signal = hilbert(embedding)
        instantaneous_phase = np.angle(analytic_signal)
        
        # Analyze phase coherence across different segments
        segment_size = len(embedding) // 8
        phase_segments = []
        
        for i in range(0, len(embedding) - segment_size, segment_size // 2):
            segment_phase = instantaneous_phase[i:i+segment_size]
            phase_segments.append(np.mean(segment_phase))
        
        # Calculate phase coherence (how consistent phases are)
        phase_coherence = 1 - np.std(phase_segments) / np.pi
        
        # Create coherence profile across dimensions
        coherence_profile = self._calculate_coherence_profile(embedding, instantaneous_phase)
        
        return {
            'phases': instantaneous_phase,
            'phase_segments': np.array(phase_segments),
            'coherence': coherence_profile,
            'global_coherence': phase_coherence
        }
    
    def _detect_resonance_peaks(self, fft_pattern: Dict) -> np.ndarray:
        """
        Detect potential resonance peaks in frequency spectrum.
        
        Theory: Emotional resonance creates peaks at specific frequencies.
        """
        amplitudes = fft_pattern['amplitudes']
        frequencies = fft_pattern['frequencies']
        
        # Find peaks with prominence
        peaks, properties = find_peaks(
            amplitudes,
            prominence=np.std(amplitudes),
            distance=5
        )
        
        # Extract peak frequencies and strengths
        if len(peaks) > 0:
            peak_freqs = frequencies[peaks]
            peak_strengths = amplitudes[peaks]
            
            # Sort by strength
            sorted_idx = np.argsort(peak_strengths)[::-1]
            return np.column_stack([
                peak_freqs[sorted_idx],
                peak_strengths[sorted_idx]
            ])
        else:
            return np.array([])
    
    def _calculate_coherence_profile(self, embedding: np.ndarray, 
                                   phase: np.ndarray) -> np.ndarray:
        """
        Calculate phase coherence profile across embedding dimensions.
        """
        # Divide embedding into chunks and calculate phase coherence
        chunk_size = 32
        coherence_values = []
        
        for i in range(0, len(embedding) - chunk_size, chunk_size):
            chunk_phase = phase[i:i+chunk_size]
            # Coherence as circular variance
            mean_phase = np.angle(np.mean(np.exp(1j * chunk_phase)))
            coherence = np.abs(np.mean(np.exp(1j * (chunk_phase - mean_phase))))
            coherence_values.append(coherence)
        
        return np.array(coherence_values)
    
    def _energy_threshold(self, embeddings: np.ndarray) -> float:
        """
        Calculate energy threshold for significant patterns.
        """
        # Calculate energy for all embeddings
        energies = []
        for emb in embeddings:
            fft_result = fft(emb)
            energy = np.sum(np.abs(fft_result)**2)
            energies.append(energy)
        
        # Threshold at mean + 1 std
        return np.mean(energies) + np.std(energies)
    
    def analyze_collective_resonance(self, patterns: List[SpectralEmotionalPattern]) -> Dict:
        """
        Analyze collective resonance patterns across multiple embeddings.
        
        Theory: Emotional fields create collective resonance effects.
        """
        if not patterns:
            return {}
        
        # Extract all resonance peaks
        all_peaks = []
        for pattern in patterns:
            if len(pattern.resonance_peaks) > 0:
                all_peaks.extend(pattern.resonance_peaks[:, 0])  # Frequencies only
        
        if not all_peaks:
            return {'collective_resonance': 'none_detected'}
        
        all_peaks = np.array(all_peaks)
        
        # Find frequency clusters (potential collective resonances)
        from sklearn.cluster import KMeans
        n_clusters = min(5, len(all_peaks) // 3)
        
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(all_peaks.reshape(-1, 1))
            
            collective_frequencies = kmeans.cluster_centers_.flatten()
            cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
            
            return {
                'collective_frequencies': collective_frequencies,
                'resonance_strengths': cluster_sizes,
                'dominant_resonance': collective_frequencies[np.argmax(cluster_sizes)]
            }
        else:
            return {
                'collective_frequencies': [np.mean(all_peaks)],
                'resonance_strengths': [len(all_peaks)],
                'dominant_resonance': np.mean(all_peaks)
            }


if __name__ == "__main__":
    """Test with real BGE embeddings following FoundationManifoldBuilder pattern."""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from Sysnpire.model.initial.bge_ingestion import BGEIngestion
    
    # Follow the same pattern as FoundationManifoldBuilder
    bge_model = BGEIngestion(model_name="BAAI/bge-large-en-v1.5", random_seed=42)
    model_loaded = bge_model.load_total_embeddings()
    
    if model_loaded is None:
        print("❌ Could not load model embeddings")
        exit(1)
    
    # Use same slice as FoundationManifoldBuilder for consistency  
    test_embeddings = model_loaded['embeddings'][550:560]
    embeddings = np.array(test_embeddings)
    
    print(f"Testing with {len(embeddings)} real BGE embeddings (indices 550-560)")
    
    analyzer = FrequencyResonanceAnalyzer()
    patterns = analyzer.discover_frequency_patterns(embeddings)
    print(f"Discovered {len(patterns)} spectral patterns")
    
    # Analyze collective resonance
    collective = analyzer.analyze_collective_resonance(patterns)
    if 'dominant_resonance' in collective:
        print(f"Dominant collective resonance at: {collective['dominant_resonance']:.3f}")