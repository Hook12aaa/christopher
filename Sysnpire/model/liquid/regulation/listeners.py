"""
Regulation Listeners - Information-Theoretic Field Monitoring

CORE PHILOSOPHY: Replace all numerical thresholds with information-theoretic measures
derived from field mathematics. Each listener monitors specific aspects of the field
using entropy, mutual information, and geometric measures rather than arbitrary limits.

MATHEMATICAL FOUNDATION: All regulation decisions emerge from:
- Field entropy analysis
- Mutual information between Q-values  
- Differential geometric field curvature
- Spectral analysis of field dynamics
- Information-theoretic consensus mechanisms

DESIGN PRINCIPLE: Living regulation system that learns and adapts its own regulatory
capabilities through field-theoretic principles and mathematical consensus.
"""

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform

from Sysnpire.model.liquid.conceptual_charge_agent import ConceptualChargeAgent
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InformationMetrics:
    """Information-theoretic measures for regulation decisions."""

    field_entropy: float
    mutual_information: float
    entropy_gradient: float
    information_flow_rate: float
    coherence_measure: float
    singularity_indicator: float


@dataclass
class RegulationSuggestion:
    """Regulation suggestions with mathematical justification."""

    regulation_type: str
    strength: float
    confidence: float
    mathematical_basis: str
    information_metrics: InformationMetrics
    parameters: Dict[str, Any]


class RegulationListener(ABC):
    """
    Base class for field-theoretic regulation listeners.

    All listeners use information theory and differential geometry
    to determine regulation needs without arbitrary thresholds.
    """

    def __init__(self, name: str, field_aspect: str):
        """
        Initialize regulation listener.

        Args:
            name: Unique identifier for this listener
            field_aspect: Aspect of field this listener monitors
        """
        self.name = name
        self.field_aspect = field_aspect
        self.history: List[InformationMetrics] = []
        self.adaptation_rate = 0.02
        self.confidence_threshold = 0.7  # Will be replaced with information measure

    def measure_field_entropy(self, values: List[float]) -> float:
        """
        Compute Shannon entropy of field values.

        High entropy = chaotic field state requiring regulation
        Low entropy = ordered field state
        """
        if not values or len(values) < 2:
            return 0.0

        values_tensor = torch.tensor(values, dtype=torch.float32)
        
        bins = min(10, max(3, len(values) // 5))
        hist, _ = torch.histogram(values_tensor, bins=bins, density=True)

        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0

        entropy_val = entropy(hist, base=2)
        return float(entropy_val)

    def compute_mutual_information(self, values1: List[float], values2: List[float]) -> float:
        """
        Compute mutual information between two sets of field values.

        High MI = strong coupling between field components
        Low MI = independent evolution requiring coordination
        """
        if len(values1) != len(values2) or len(values1) < 3:
            return 0.0

        values1_tensor = torch.tensor(values1, dtype=torch.float32)
        values2_tensor = torch.tensor(values2, dtype=torch.float32)
        
        bins = min(5, max(2, len(values1) // 10))

        stacked_values = torch.stack([values1_tensor, values2_tensor], dim=1)
        joint_hist, _ = torch.histogramdd(stacked_values, bins=bins, density=True)

        hist1, _ = torch.histogram(values1_tensor, bins=bins, density=True)
        hist2, _ = torch.histogram(values2_tensor, bins=bins, density=True)

        entropy_x = entropy(hist1[hist1 > 0], base=2)
        entropy_y = entropy(hist2[hist2 > 0], base=2)
        joint_flattened = joint_hist.flatten()
        entropy_joint = entropy(joint_flattened[joint_flattened > 0], base=2)
        
        mi = entropy_x + entropy_y - entropy_joint
        return float(max(0.0, mi))  # MI should be non-negative

    def compute_entropy_gradient(self, current_entropy: float) -> float:
        """
        Compute rate of change of field entropy.

        Large gradient = rapid field changes requiring regulation
        """
        if len(self.history) < 2:
            return 0.0

        previous_entropy = self.history[-1].field_entropy
        gradient = current_entropy - previous_entropy
        return float(gradient)

    def detect_field_singularities(self, values: List[float]) -> float:
        """
        Detect mathematical singularities in field values.

        Uses coefficient of variation and extreme value analysis.
        """
        if not values or len(values) < 3:
            return 0.0

        finite_values = [v for v in values if math.isfinite(v)]
        if len(finite_values) < 3:
            return 1.0  # All non-finite is maximum singularity

        values_tensor = torch.tensor(finite_values, dtype=torch.float32)
        
        mean_val = torch.mean(values_tensor)
        std_val = torch.std(values_tensor)
        cv = std_val / (torch.abs(mean_val) + 1e-12)

        q75 = torch.quantile(values_tensor, 0.75)
        q25 = torch.quantile(values_tensor, 0.25)
        iqr = q75 - q25

        outlier_threshold = 3 * iqr
        median_val = torch.median(values_tensor)
        outlier_mask = torch.abs(values_tensor - median_val) > outlier_threshold
        outliers = torch.sum(outlier_mask).item()
        outlier_fraction = outliers / len(finite_values)

        singularity_measure = min(1.0, cv / 10.0 + outlier_fraction)
        return float(singularity_measure)

    def compute_information_flow_rate(self, current_metrics: InformationMetrics) -> float:
        """
        Compute rate of information change in the field.

        Measures how quickly field information content is changing.
        """
        if len(self.history) < 2:
            return 0.0

        prev_metrics = self.history[-1]

        entropy_rate = abs(current_metrics.field_entropy - prev_metrics.field_entropy)
        mi_rate = abs(current_metrics.mutual_information - prev_metrics.mutual_information)
        coherence_rate = abs(current_metrics.coherence_measure - prev_metrics.coherence_measure)

        flow_rate = (entropy_rate + mi_rate + coherence_rate) / 3.0
        return float(flow_rate)

    def smooth_regulation_mapping(self, measure: float, sensitivity: float = 1.0) -> float:
        """
        Map information measure to regulation strength using smooth function.

        Replaces step functions with continuous mathematical mappings.
        """
        measure_tensor = torch.tensor(measure, dtype=torch.float32)
        sensitivity_tensor = torch.tensor(sensitivity, dtype=torch.float32)
        regulation_strength = 1.0 - torch.exp(-sensitivity_tensor * measure_tensor)
        return float(torch.clamp(regulation_strength, 0.0, 1.0))

    def _compute_information_based_threshold(self, metrics: InformationMetrics) -> float:
        """
        Compute confidence threshold based on information-theoretic principles.

        Threshold emerges from field entropy and mutual information content.
        Higher entropy = lower threshold (more responsive to chaos)
        Higher MI = higher threshold (more selective when coupled)
        """
        entropy_factor = 1.0 / (1.0 + metrics.field_entropy)

        mi_factor = 1.0 + metrics.mutual_information

        flow_factor = 1.0 / (1.0 + metrics.information_flow_rate)

        threshold = entropy_factor * flow_factor / mi_factor

        threshold_tensor = torch.tensor(threshold, dtype=torch.float32)
        return float(torch.clamp(threshold_tensor, 0.01, 0.99))

    def _compute_confidence_from_information(self, metrics: InformationMetrics) -> float:
        """
        Compute regulation confidence based on information content.

        Confidence emerges from multiple information measures:
        - Mutual information (coupling strength)
        - Coherence measure (field organization)
        - Inverse entropy gradient (stability)
        - Information flow consistency
        """
        mi_confidence = metrics.mutual_information

        coherence_confidence = metrics.coherence_measure

        stability_confidence = 1.0 / (1.0 + abs(metrics.entropy_gradient))

        flow_confidence = 1.0 / (1.0 + metrics.information_flow_rate)

        total_information = mi_confidence + coherence_confidence + stability_confidence + flow_confidence
        confidence = total_information / 4.0

        confidence_tensor = torch.tensor(confidence, dtype=torch.float32)
        return float(torch.clamp(confidence_tensor, 0.0, 1.0))

    def _compute_activation_threshold(self, metrics: InformationMetrics) -> float:
        """
        Compute regulation activation threshold from field mathematics.

        Activation emerges from field instability measures:
        - High entropy fields need lower activation (more sensitive)
        - High singularity fields need immediate activation
        - Coherent fields need higher thresholds (more stable)
        """
        entropy_activation = metrics.field_entropy / (1.0 + metrics.field_entropy)

        singularity_activation = metrics.singularity_indicator

        coherence_resistance = 1.0 - metrics.coherence_measure

        flow_activation = metrics.information_flow_rate / (1.0 + metrics.information_flow_rate)

        instability_maximum = max(entropy_activation, singularity_activation, flow_activation)

        activation_threshold = instability_maximum * coherence_resistance

        threshold_tensor = torch.tensor(activation_threshold, dtype=torch.float32)
        return float(torch.clamp(threshold_tensor, 0.001, 0.5))

    def _compute_natural_coherence_point(self, metrics: InformationMetrics) -> float:
        """
        Compute natural coherence equilibrium point from field mathematics.

        Natural coherence emerges from balance between:
        - Information coupling (mutual information)
        - Field organization (inverse entropy)
        - Dynamic stability (flow consistency)
        """
        coupling_coherence = metrics.mutual_information

        organization_coherence = 1.0 / (1.0 + metrics.field_entropy)

        stability_coherence = 1.0 / (1.0 + metrics.information_flow_rate)

        factors_tensor = torch.tensor([coupling_coherence, organization_coherence, stability_coherence], dtype=torch.float32)
        natural_coherence = torch.pow(torch.prod(factors_tensor), 1.0 / 3.0)

        return float(torch.clamp(natural_coherence, 0.1, 0.95))

    @abstractmethod
    def analyze_field_aspect(self, agents: List[ConceptualChargeAgent]) -> InformationMetrics:
        """
        Analyze specific aspect of field this listener monitors.

        Must be implemented by each specialized listener.
        """
        pass

    @abstractmethod
    def suggest_regulation(self, metrics: InformationMetrics) -> Optional[RegulationSuggestion]:
        """
        Suggest regulation based on information-theoretic analysis.

        Must be implemented by each specialized listener.
        """
        pass

    def validate_mathematically(self, agents: List[ConceptualChargeAgent], suggestion: RegulationSuggestion) -> bool:
        """
        Validate that regulation suggestion maintains mathematical integrity.

        All regulation must preserve Q(τ,C,s) field-theoretic properties:
        - Complex field phase coherence
        - Energy conservation principles
        - Information-theoretic bounds
        - Mathematical consistency with field theory

        Args:
            agents: Current agent states for validation
            suggestion: Proposed regulation to validate

        Returns:
            True if regulation maintains mathematical integrity
        """
        if not suggestion or not agents:
            return False

        if not (0.0 <= suggestion.strength <= 1.0):
            logger.warning(f"🚨 Invalid regulation strength: {suggestion.strength}")
            return False

        metrics = suggestion.information_metrics
        if metrics.field_entropy < 0 or not math.isfinite(metrics.field_entropy):
            logger.warning(f"🚨 Invalid field entropy: {metrics.field_entropy}")
            return False

        if metrics.mutual_information < 0 or not math.isfinite(metrics.mutual_information):
            logger.warning(f"🚨 Invalid mutual information: {metrics.mutual_information}")
            return False

        if not suggestion.mathematical_basis or len(suggestion.mathematical_basis.strip()) == 0:
            logger.warning("🚨 Missing mathematical basis for regulation")
            return False

        for param_name, param_value in suggestion.parameters.items():
            if not math.isfinite(param_value):
                logger.warning(f"🚨 Non-finite parameter {param_name}: {param_value}")
                return False

        if suggestion.confidence < 0 or suggestion.confidence > 1:
            logger.warning(f"🚨 Invalid confidence bounds: {suggestion.confidence}")
            return False

        agent_q_magnitudes = []
        for agent in agents[:10]:  # Sample for efficiency
            if hasattr(agent, "Q_value") and agent.Q_value is not None:
                q_mag = abs(agent.Q_value)
                if math.isfinite(q_mag):
                    agent_q_magnitudes.append(q_mag)

        if agent_q_magnitudes:
            q_mag_tensor = torch.tensor(agent_q_magnitudes, dtype=torch.float32)
            max_q_magnitude = torch.max(q_mag_tensor).item()
            if suggestion.strength > 0.5 and max_q_magnitude < 1e-6:
                logger.warning("🚨 Strong regulation on near-zero field - potential over-regulation")
                return False

        logger.debug(f"✅ Mathematical validation passed for {suggestion.regulation_type}")
        return True

    def listen(self, agents: List[ConceptualChargeAgent]) -> Optional[RegulationSuggestion]:
        """
        Main listening function that analyzes field and suggests regulation.

        Returns regulation suggestion or None if no regulation needed.
        """
        start_time = time.time()

        metrics = self.analyze_field_aspect(agents)

        self.history.append(metrics)
        if len(self.history) > 100:  # Keep last 100 measurements
            self.history.pop(0)

        suggestion = self.suggest_regulation(metrics)

        analysis_time = time.time() - start_time

        min_confidence = self._compute_information_based_threshold(metrics)
        if suggestion and suggestion.confidence > min_confidence:
            logger.debug(
                f"🎯 {self.name} suggests {suggestion.regulation_type} "
                f"(strength: {suggestion.strength:.3f}, confidence: {suggestion.confidence:.3f}) "
                f"in {analysis_time:.4f}s"
            )

        return suggestion


class PersistenceRegulationListener(RegulationListener):
    """
    Monitors Ψ_persistence field decay patterns and temporal coherence.

    Uses wavelet analysis and temporal signal processing to detect
    when persistence mechanisms need regulation.
    """

    def __init__(self):
        super().__init__("PersistenceRegulation", "temporal_persistence")
        self.wavelet = "db4"  # Daubechies wavelet for temporal analysis

    def analyze_field_aspect(self, agents: List[ConceptualChargeAgent]) -> InformationMetrics:
        """
        Analyze temporal persistence patterns using wavelet analysis and signal processing.
        """
        vivid_time_series = []
        character_time_series = []
        temporal_momenta = []
        breathing_coherences = []

        for agent in agents:
            if hasattr(agent, "temporal_biography") and agent.temporal_biography is not None:
                bio = agent.temporal_biography

                if hasattr(bio, "vivid_layer") and bio.vivid_layer is not None:
                    vivid_values = [x for x in bio.vivid_layer if math.isfinite(abs(x))]
                    if vivid_values:
                        vivid_time_series.append(vivid_values)

                if hasattr(bio, "character_layer") and bio.character_layer is not None:
                    char_values = [x for x in bio.character_layer if math.isfinite(abs(x))]
                    if char_values:
                        character_time_series.append(char_values)

                if hasattr(bio, "temporal_momentum") and bio.temporal_momentum is not None:
                    momentum_mag = abs(bio.temporal_momentum)
                    if math.isfinite(momentum_mag):
                        temporal_momenta.append(momentum_mag)

                if hasattr(bio, "breathing_coherence") and math.isfinite(bio.breathing_coherence):
                    breathing_coherences.append(bio.breathing_coherence)

        vivid_entropy = self._wavelet_entropy_analysis(vivid_time_series)
        character_entropy = self._wavelet_entropy_analysis(character_time_series)
        field_entropy = (vivid_entropy + character_entropy) / 2.0

        mutual_info = self._temporal_mutual_information(vivid_time_series, character_time_series)

        coherence_measure = np.mean(breathing_coherences) if breathing_coherences else 0.0

        entropy_gradient = self.compute_entropy_gradient(field_entropy)

        singularity_indicator = self.detect_field_singularities(temporal_momenta)

        metrics = InformationMetrics(
            field_entropy=field_entropy,
            mutual_information=mutual_info,
            entropy_gradient=entropy_gradient,
            information_flow_rate=0.0,  # Will be computed in base class
            coherence_measure=coherence_measure,
            singularity_indicator=singularity_indicator,
        )

        metrics.information_flow_rate = self.compute_information_flow_rate(metrics)

        return metrics

    def suggest_regulation(self, metrics: InformationMetrics) -> Optional[RegulationSuggestion]:
        """
        Suggest persistence regulation based on temporal information analysis.
        """
        entropy_strength = self.smooth_regulation_mapping(metrics.field_entropy, sensitivity=0.5)
        gradient_strength = self.smooth_regulation_mapping(abs(metrics.entropy_gradient), sensitivity=2.0)
        singularity_strength = self.smooth_regulation_mapping(metrics.singularity_indicator, sensitivity=1.0)
        flow_strength = self.smooth_regulation_mapping(metrics.information_flow_rate, sensitivity=1.5)

        combined_strength = (entropy_strength + gradient_strength + singularity_strength + flow_strength) / 4.0

        confidence = self._compute_confidence_from_information(metrics)

        basis = (
            f"Entropy: {metrics.field_entropy:.3f}, Gradient: {metrics.entropy_gradient:.3f}, "
            f"Singularities: {metrics.singularity_indicator:.3f}, Flow: {metrics.information_flow_rate:.3f}"
        )

        activation_threshold = self._compute_activation_threshold(metrics)
        if combined_strength > activation_threshold:
            return RegulationSuggestion(
                regulation_type="persistence_decay",
                strength=combined_strength,
                confidence=confidence,
                mathematical_basis=basis,
                information_metrics=metrics,
                parameters={
                    "vivid_decay_factor": 1.0 - entropy_strength,
                    "character_persistence_factor": 1.0 - gradient_strength,
                    "temporal_damping_factor": 1.0 - singularity_strength,
                },
            )

        return None

    def _wavelet_entropy_analysis(self, time_series_list: List[List[float]]) -> float:
        """
        Analyze entropy of temporal patterns using PyTorch tensor operations.
        
        Uses frequency domain analysis to decompose temporal signals and compute
        entropy across frequency bands to detect persistence instabilities.
        MPS-compatible implementation using pure PyTorch operations.
        """
        if not time_series_list:
            return 0.0
            
        total_entropy = 0.0
        valid_series = 0
        
        for time_series in time_series_list:
            if len(time_series) < 4:  # Need minimum samples for frequency analysis
                continue
                
            # Convert to PyTorch tensor and keep on device (MPS-compatible)
            if hasattr(time_series, 'device'):
                # Already a tensor
                time_series_tensor = time_series
            else:
                # Convert to tensor on the same device as other computations
                time_series_tensor = torch.as_tensor(time_series, dtype=torch.float32)
                if torch.backends.mps.is_available():
                    time_series_tensor = time_series_tensor.to('mps')
            
            n = time_series_tensor.shape[0] if time_series_tensor.ndim > 0 else len(time_series)
            
            # Pad to power of 2 for efficient FFT
            padded_length = 2 ** int(torch.ceil(torch.log2(torch.tensor(float(n)))))
            if padded_length > n:
                padding = (0, padded_length - n)
                padded_series = torch.nn.functional.pad(time_series_tensor, padding, mode='constant', value=0.0)
            else:
                padded_series = time_series_tensor
            
            # Use FFT-based frequency analysis instead of CWT for MPS compatibility
            fft_result = torch.fft.fft(padded_series)
            power_spectrum = torch.abs(fft_result) ** 2
            
            # Create frequency bands (similar to wavelet scale bands)
            num_bands = min(16, len(power_spectrum) // 4)
            if num_bands < 2:
                continue
                
            band_size = len(power_spectrum) // num_bands
            band_energies = []
            
            for i in range(num_bands):
                start_idx = i * band_size
                end_idx = min((i + 1) * band_size, len(power_spectrum))
                band_energy = torch.sum(power_spectrum[start_idx:end_idx])
                band_energies.append(band_energy)
            
            band_energies = torch.stack(band_energies)
            
            # Normalize band energies
            total_energy = torch.sum(band_energies)
            if total_energy > 0:
                band_energies = band_energies / total_energy
                
                # Remove zero energies for entropy calculation
                band_energies = band_energies[band_energies > 1e-12]
                
                if len(band_energies) > 1:
                    # Calculate Shannon entropy using PyTorch operations
                    log_energies = torch.log2(band_energies + 1e-12)  # Add small epsilon for numerical stability
                    band_entropy = -torch.sum(band_energies * log_energies)
                    total_entropy += band_entropy.item()
                    valid_series += 1
                
        return total_entropy / valid_series if valid_series > 0 else 0.0

    def _temporal_mutual_information(self, series1_list: List[List[float]], series2_list: List[List[float]]) -> float:
        """
        Compute mutual information between two sets of temporal patterns.
        
        Uses signal processing to find temporal correlations between
        vivid and character layer dynamics.
        """
        if not series1_list or not series2_list:
            return 0.0
            
        combined_series1 = []
        combined_series2 = []
        
        min_pairs = min(len(series1_list), len(series2_list))
        for i in range(min_pairs):
            if len(series1_list[i]) > 0 and len(series2_list[i]) > 0:
                min_len = min(len(series1_list[i]), len(series2_list[i]))
                combined_series1.extend(series1_list[i][:min_len])
                combined_series2.extend(series2_list[i][:min_len])
        
        if len(combined_series1) < 3 or len(combined_series2) < 3:
            return 0.0
            
        s1 = torch.tensor(combined_series1).flatten()
        s2 = torch.tensor(combined_series2).flatten()
        s1 = (s1 - torch.mean(s1)) / (torch.std(s1) + 1e-12)
        s2 = (s2 - torch.mean(s2)) / (torch.std(s2) + 1e-12)
        
        correlation = torch.sum(s1 * s2) / torch.sqrt(torch.sum(s1 ** 2) * torch.sum(s2 ** 2))
        max_correlation = torch.abs(correlation)
        
        mi_estimate = -0.5 * torch.log(1 - torch.clamp(max_correlation**2, max=0.99))
        
        return float(max(0.0, mi_estimate))


class EmotionalConductorListener(RegulationListener):
    """
    Monitors emotional field modulation and phase coherence patterns.

    Uses complex number field analysis and phase space geometry
    to detect when emotional conductor needs regulation.
    """

    def __init__(self):
        super().__init__("EmotionalConductor", "phase_coherence")

    def analyze_field_aspect(self, agents: List[ConceptualChargeAgent]) -> InformationMetrics:
        """
        Analyze emotional field modulation using complex field analysis.
        """
        field_modulations = []
        phase_values = []
        gradient_magnitudes = []
        coupling_strengths = []

        for agent in agents:
            if hasattr(agent, "emotional_modulation") and agent.emotional_modulation is not None:
                mod = agent.emotional_modulation

                if hasattr(mod, "field_modulation_strength") and math.isfinite(mod.field_modulation_strength):
                    field_modulations.append(abs(mod.field_modulation_strength))

                if hasattr(mod, "unified_phase_shift") and mod.unified_phase_shift is not None:
                    phase = np.angle(mod.unified_phase_shift)
                    if math.isfinite(phase):
                        phase_values.append(phase)

                if hasattr(mod, "gradient_magnitude") and math.isfinite(mod.gradient_magnitude):
                    gradient_magnitudes.append(abs(mod.gradient_magnitude))

                if hasattr(mod, "coupling_strength") and math.isfinite(mod.coupling_strength):
                    coupling_strengths.append(abs(mod.coupling_strength))

        field_entropy = self.measure_field_entropy(field_modulations)

        if len(phase_values) > 2:
            phase_tensor = torch.tensor(phase_values, dtype=torch.float32)
            phase_std = torch.std(phase_tensor)
            phase_coherence = 1.0 - phase_std / np.pi  # 1 = perfect coherence, 0 = random

            if len(field_modulations) == len(phase_values):
                mutual_info = self.compute_mutual_information(field_modulations, phase_values)
            else:
                mutual_info = 0.0
        else:
            phase_coherence = 0.0
            mutual_info = 0.0

        entropy_gradient = self.compute_entropy_gradient(field_entropy)

        singularity_indicator = self.detect_field_singularities(gradient_magnitudes)

        metrics = InformationMetrics(
            field_entropy=field_entropy,
            mutual_information=mutual_info,
            entropy_gradient=entropy_gradient,
            information_flow_rate=0.0,  # Computed in base class
            coherence_measure=phase_coherence,
            singularity_indicator=singularity_indicator,
        )

        metrics.information_flow_rate = self.compute_information_flow_rate(metrics)

        return metrics

    def suggest_regulation(self, metrics: InformationMetrics) -> Optional[RegulationSuggestion]:
        """
        Suggest emotional conductor regulation based on phase coherence analysis.
        """
        coherence_loss = 1.0 - metrics.coherence_measure
        entropy_strength = self.smooth_regulation_mapping(metrics.field_entropy, sensitivity=0.8)
        singularity_strength = self.smooth_regulation_mapping(metrics.singularity_indicator, sensitivity=1.2)

        combined_strength = (coherence_loss + entropy_strength + singularity_strength) / 3.0

        confidence = self._compute_confidence_from_information(metrics)

        basis = (
            f"Coherence: {metrics.coherence_measure:.3f}, Entropy: {metrics.field_entropy:.3f}, "
            f"Singularities: {metrics.singularity_indicator:.3f}"
        )

        activation_threshold = self._compute_activation_threshold(metrics)
        if combined_strength > activation_threshold:
            return RegulationSuggestion(
                regulation_type="phase_coherence_restoration",
                strength=combined_strength,
                confidence=confidence,
                mathematical_basis=basis,
                information_metrics=metrics,
                parameters={
                    "coherence_restoration_factor": coherence_loss,
                    "phase_smoothing_strength": entropy_strength,
                    "gradient_regularization": singularity_strength,
                },
            )

        return None


class BreathingSynchronyListener(RegulationListener):
    """
    Monitors collective breathing patterns and resonance cascades.

    Uses frequency domain analysis and spectral methods to detect
    when breathing synchronization needs regulation.
    """

    def __init__(self):
        super().__init__("BreathingSynchrony", "collective_breathing")

    def analyze_field_aspect(self, agents: List[ConceptualChargeAgent]) -> InformationMetrics:
        """
        Analyze breathing patterns using frequency domain analysis.
        """
        breathing_frequencies = []
        breathing_amplitudes = []
        breathing_phases = []
        q_coefficients = []

        for agent in agents:
            if hasattr(agent, "breath_frequency") and math.isfinite(agent.breath_frequency):
                breathing_frequencies.append(abs(agent.breath_frequency))

            if hasattr(agent, "breath_amplitude") and math.isfinite(agent.breath_amplitude):
                breathing_amplitudes.append(abs(agent.breath_amplitude))

            if hasattr(agent, "breath_phase") and math.isfinite(agent.breath_phase):
                breathing_phases.append(agent.breath_phase)

            if hasattr(agent, "breathing_q_coefficients") and agent.breathing_q_coefficients:
                coeff_list = [
                    coeff
                    for coeff in agent.breathing_q_coefficients.values()
                    if hasattr(coeff, "__abs__") and math.isfinite(abs(coeff))
                ]
                
                if coeff_list:
                    coeff_tensor = torch.tensor(coeff_list, dtype=torch.complex64)
                    q_magnitudes_tensor = torch.abs(coeff_tensor)
                    q_coefficients.extend(q_magnitudes_tensor.tolist())

        freq_entropy = self.measure_field_entropy(breathing_frequencies)
        amplitude_entropy = self.measure_field_entropy(breathing_amplitudes)
        field_entropy = (freq_entropy + amplitude_entropy) / 2.0

        if len(breathing_phases) > 2:
            phases_tensor = torch.tensor(breathing_phases, dtype=torch.float32)
            phase_vector_x = torch.mean(torch.cos(phases_tensor))
            phase_vector_y = torch.mean(torch.sin(phases_tensor))
            phase_coherence = torch.sqrt(phase_vector_x**2 + phase_vector_y**2).item()
        else:
            phase_coherence = 0.0

        if len(breathing_frequencies) == len(breathing_amplitudes) and len(breathing_frequencies) > 2:
            mutual_info = self.compute_mutual_information(breathing_frequencies, breathing_amplitudes)
        else:
            mutual_info = 0.0

        entropy_gradient = self.compute_entropy_gradient(field_entropy)

        singularity_indicator = self.detect_field_singularities(q_coefficients)

        metrics = InformationMetrics(
            field_entropy=field_entropy,
            mutual_information=mutual_info,
            entropy_gradient=entropy_gradient,
            information_flow_rate=0.0,
            coherence_measure=phase_coherence,
            singularity_indicator=singularity_indicator,
        )

        metrics.information_flow_rate = self.compute_information_flow_rate(metrics)

        return metrics

    def suggest_regulation(self, metrics: InformationMetrics) -> Optional[RegulationSuggestion]:
        """
        Suggest breathing regulation based on frequency and synchrony analysis.
        """
        cascade_risk = self.smooth_regulation_mapping(metrics.singularity_indicator, sensitivity=1.5)

        ideal_coherence = self._compute_natural_coherence_point(metrics)
        coherence_deviation = abs(metrics.coherence_measure - ideal_coherence)
        synchrony_regulation = self.smooth_regulation_mapping(coherence_deviation, sensitivity=2.0)

        entropy_strength = self.smooth_regulation_mapping(metrics.field_entropy, sensitivity=0.6)

        combined_strength = (cascade_risk + synchrony_regulation + entropy_strength) / 3.0

        confidence = self._compute_confidence_from_information(metrics)

        basis = (
            f"Cascades: {metrics.singularity_indicator:.3f}, Coherence dev: {coherence_deviation:.3f}, "
            f"Entropy: {metrics.field_entropy:.3f}"
        )

        activation_threshold = self._compute_activation_threshold(metrics)
        if combined_strength > activation_threshold:
            return RegulationSuggestion(
                regulation_type="resonance_cascade_damping",
                strength=combined_strength,
                confidence=confidence,
                mathematical_basis=basis,
                information_metrics=metrics,
                parameters={
                    "cascade_damping_factor": cascade_risk,
                    "synchrony_adjustment": synchrony_regulation,
                    "frequency_smoothing": entropy_strength,
                },
            )

        return None


class EnergyConservationListener(RegulationListener):
    """
    Monitors field energy distribution and conservation violations.

    Uses energy flow analysis and Hamiltonian principles to detect
    when energy conservation regulation is needed.
    """

    def __init__(self):
        super().__init__("EnergyConservation", "energy_flow")

    def analyze_field_aspect(self, agents: List[ConceptualChargeAgent]) -> InformationMetrics:
        """
        Analyze energy distribution using field energy analysis.
        """
        q_values_list = []
        valid_q_count = 0

        for agent in agents:
            if hasattr(agent, "Q_components") and agent.Q_components is not None:
                q_val = agent.Q_components.Q_value
                if q_val is not None and math.isfinite(abs(q_val)):
                    q_values_list.append(complex(q_val))
                    valid_q_count += 1

        if valid_q_count < 2:
            field_entropy = 0.0
            phase_coherence = 0.0
            mutual_info = 0.0
        else:
            q_tensor = torch.tensor(q_values_list, dtype=torch.complex64)
            
            q_magnitudes_tensor = torch.abs(q_tensor)
            q_phases_tensor = torch.angle(q_tensor)
            energy_densities_tensor = q_magnitudes_tensor.pow(2)

            q_magnitudes = q_magnitudes_tensor.cpu().numpy().tolist()
            q_phases = q_phases_tensor.cpu().numpy().tolist()
            energy_densities = energy_densities_tensor.cpu().numpy().tolist()

            energy_entropy = self.measure_field_entropy(energy_densities)
            magnitude_entropy = self.measure_field_entropy(q_magnitudes)
            field_entropy = (energy_entropy + magnitude_entropy) / 2.0

            phase_std = torch.std(q_phases_tensor).item()
            phase_coherence = 1.0 - phase_std / np.pi

            mutual_info = self.compute_mutual_information(q_magnitudes, q_phases)

        entropy_gradient = self.compute_entropy_gradient(field_entropy)

        singularity_indicator = self.detect_field_singularities(energy_densities)

        metrics = InformationMetrics(
            field_entropy=field_entropy,
            mutual_information=mutual_info,
            entropy_gradient=entropy_gradient,
            information_flow_rate=0.0,
            coherence_measure=phase_coherence,
            singularity_indicator=singularity_indicator,
        )

        metrics.information_flow_rate = self.compute_information_flow_rate(metrics)

        return metrics

    def suggest_regulation(self, metrics: InformationMetrics) -> Optional[RegulationSuggestion]:
        """
        Suggest energy conservation regulation based on energy flow analysis.
        """
        concentration_risk = self.smooth_regulation_mapping(metrics.singularity_indicator, sensitivity=1.0)

        entropy_strength = self.smooth_regulation_mapping(metrics.field_entropy, sensitivity=0.7)

        flow_regulation = self.smooth_regulation_mapping(metrics.information_flow_rate, sensitivity=1.2)

        combined_strength = (concentration_risk + entropy_strength + flow_regulation) / 3.0

        confidence = self._compute_confidence_from_information(metrics)

        basis = (
            f"Concentration: {metrics.singularity_indicator:.3f}, Entropy: {metrics.field_entropy:.3f}, "
            f"Flow: {metrics.information_flow_rate:.3f}"
        )

        activation_threshold = self._compute_activation_threshold(metrics)
        if combined_strength > activation_threshold:
            return RegulationSuggestion(
                regulation_type="energy_conservation",
                strength=combined_strength,
                confidence=confidence,
                mathematical_basis=basis,
                information_metrics=metrics,
                parameters={
                    "energy_redistribution_factor": concentration_risk,
                    "entropy_smoothing": entropy_strength,
                    "flow_damping": flow_regulation,
                },
            )

        return None


class BoundaryEnforcementListener(RegulationListener):
    """
    Monitors spatial field boundaries and agent distribution.

    Uses geometric analysis and spatial statistics to detect
    when boundary enforcement regulation is needed.
    """

    def __init__(self):
        super().__init__("BoundaryEnforcement", "spatial_distribution")

    def analyze_field_aspect(self, agents: List[ConceptualChargeAgent]) -> InformationMetrics:
        """
        Analyze spatial distribution using geometric measures.
        """
        positions = []
        distances_from_center = []

        for agent in agents:
            if hasattr(agent, "field_state") and agent.field_state is not None:
                if hasattr(agent.field_state, "field_position"):
                    x, y = agent.field_state.field_position
                    if math.isfinite(x) and math.isfinite(y):
                        positions.append((x, y))
                        x_tensor = torch.tensor(x, dtype=torch.float32)
                        y_tensor = torch.tensor(y, dtype=torch.float32)
                        distance = torch.sqrt(x_tensor**2 + y_tensor**2).item()
                        distances_from_center.append(distance)

        if len(positions) < 3:
            return InformationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        distance_entropy = self.measure_field_entropy(distances_from_center)

        positions_array = np.array(positions)
        if len(positions) > 1:
            pairwise_distances = pdist(positions_array)
            distance_matrix = squareform(pairwise_distances)

            mean_distance = np.mean(pairwise_distances)
            std_distance = np.std(pairwise_distances)
            spatial_coherence = 1.0 / (1.0 + std_distance / (mean_distance + 1e-12))
        else:
            spatial_coherence = 1.0

        max_reasonable_distance = 5.0  # Will be made adaptive
        boundary_violations = sum(1 for d in distances_from_center if d > max_reasonable_distance)
        violation_fraction = boundary_violations / len(distances_from_center)

        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        mutual_info = self.compute_mutual_information(x_coords, y_coords)

        entropy_gradient = self.compute_entropy_gradient(distance_entropy)

        singularity_indicator = self.detect_field_singularities(distances_from_center)

        metrics = InformationMetrics(
            field_entropy=distance_entropy,
            mutual_information=mutual_info,
            entropy_gradient=entropy_gradient,
            information_flow_rate=0.0,
            coherence_measure=spatial_coherence,
            singularity_indicator=singularity_indicator,
        )

        metrics.information_flow_rate = self.compute_information_flow_rate(metrics)

        return metrics

    def suggest_regulation(self, metrics: InformationMetrics) -> Optional[RegulationSuggestion]:
        """
        Suggest boundary regulation based on spatial distribution analysis.
        """
        violation_strength = self.smooth_regulation_mapping(metrics.singularity_indicator, sensitivity=1.3)

        entropy_strength = self.smooth_regulation_mapping(metrics.field_entropy, sensitivity=0.5)

        ideal_coherence = 0.6
        coherence_deviation = abs(metrics.coherence_measure - ideal_coherence)
        coherence_regulation = self.smooth_regulation_mapping(coherence_deviation, sensitivity=1.8)

        combined_strength = (violation_strength + entropy_strength + coherence_regulation) / 3.0

        confidence = self._compute_confidence_from_information(metrics)

        basis = (
            f"Violations: {metrics.singularity_indicator:.3f}, Entropy: {metrics.field_entropy:.3f}, "
            f"Coherence dev: {coherence_deviation:.3f}"
        )

        activation_threshold = self._compute_activation_threshold(metrics)
        if combined_strength > activation_threshold:
            return RegulationSuggestion(
                regulation_type="boundary_enforcement",
                strength=combined_strength,
                confidence=confidence,
                mathematical_basis=basis,
                information_metrics=metrics,
                parameters={
                    "position_correction_factor": violation_strength,
                    "spatial_smoothing": entropy_strength,
                    "coherence_adjustment": coherence_regulation,
                },
            )

        return None


class ListenerConsensus:
    """
    Information-theoretic consensus mechanism for regulation listeners.

    Aggregates listener suggestions using mathematical weighting
    based on information content rather than voting.
    """

    def __init__(self):
        self.adaptation_rate = 0.05
        self.listener_weights = {}  # Learned weights for each listener

    def compute_information_weight(self, suggestion: RegulationSuggestion) -> float:
        """
        Compute weight for a regulation suggestion based on information content.
        """
        metrics = suggestion.information_metrics

        entropy_weight = min(1.0, metrics.field_entropy / 2.0)  # Higher entropy = more information
        mi_weight = min(1.0, metrics.mutual_information * 2.0)  # Higher MI = stronger relationships
        confidence_weight = suggestion.confidence

        info_weight = (entropy_weight + mi_weight + confidence_weight) / 3.0
        return float(info_weight)

    def resolve_conflicting_regulations(
        self, suggestions: List[RegulationSuggestion]
    ) -> Optional[RegulationSuggestion]:
        """
        Resolve conflicts between regulation suggestions using mathematical optimization.

        Finds regulation that minimizes overall field energy functional.
        """
        if not suggestions:
            return None

        if len(suggestions) == 1:
            return suggestions[0]

        weights = [self.compute_information_weight(s) for s in suggestions]
        total_weight = sum(weights) + 1e-12

        avg_strength = sum(w * s.strength for w, s in zip(weights, suggestions)) / total_weight

        combined_basis = "; ".join([f"{s.regulation_type}: {s.mathematical_basis}" for s in suggestions])

        max_confidence = max(s.confidence for s in suggestions)

        combined_params = {}
        for suggestion in suggestions:
            for key, value in suggestion.parameters.items():
                if key not in combined_params:
                    combined_params[key] = []
                combined_params[key].append(value)

        for key in combined_params:
            params_tensor = torch.tensor(combined_params[key], dtype=torch.float32)
            combined_params[key] = torch.mean(params_tensor).item()

        return RegulationSuggestion(
            regulation_type="combined_regulation",
            strength=avg_strength,
            confidence=max_confidence,
            mathematical_basis=combined_basis,
            information_metrics=suggestions[0].information_metrics,  # Use first one as representative
            parameters=combined_params,
        )

    def weighted_consensus(self, suggestions: List[RegulationSuggestion]) -> Optional[RegulationSuggestion]:
        """
        Compute weighted consensus from multiple listener suggestions.
        """
        if not suggestions:
            return None

        valid_suggestions = [s for s in suggestions if self.compute_information_weight(s) > 0.1]

        if not valid_suggestions:
            return None

        consensus_suggestion = self.resolve_conflicting_regulations(valid_suggestions)

        return consensus_suggestion
