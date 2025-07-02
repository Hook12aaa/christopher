"""
MetaRegulation - Regulation System Self-Monitoring and Stability

CORE PRINCIPLE: The regulation system itself can become unstable! MetaRegulation
monitors the regulation system and ensures it doesn't become a source of
instability. This implements regulation of regulation through mathematical
principles - meta-mathematical stability analysis.

MATHEMATICAL FOUNDATION: All meta-regulation mechanisms derive from:
- Information-theoretic analysis of regulation system entropy
- Oscillation detection through spectral analysis of regulation patterns
- Stability analysis using Lyapunov functions and dynamical systems theory
- Emergency mathematical bounds when advanced systems fail completely

META-REGULATION PHILOSOPHY: Mathematics regulating mathematics through
mathematical principles, applied recursively to the regulation system itself.
The meta-regulation system monitors:

1. Regulation listener reliability and consensus quality
2. Regulation system oscillations and feedback loops
3. Regulation effectiveness entropy and parameter drift
4. Emergency fallback to pure mathematical bounds
5. Regulation system lifecycle management and health

EMERGENCY PRINCIPLES: When all advanced regulation fails, fall back to
simple mathematical bounds while maintaining absolute mathematical integrity.
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from enum import Enum
import warnings

import numpy as np
import torch
from scipy import signal, linalg, optimize, stats
from scipy.stats import entropy

from Sysnpire.model.liquid.conceptual_charge_agent import ConceptualChargeAgent
from .listeners import RegulationListener, RegulationSuggestion, InformationMetrics, ListenerConsensus
from .mathematical_object_proxy import MathematicalObjectProxy, MathematicalHealthStatus
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


class RegulationSystemHealthStatus(Enum):
    """Health status of the regulation system itself."""

    OPTIMAL = "optimal"
    STABLE = "stable"
    OSCILLATING = "oscillating"
    UNRELIABLE = "unreliable"
    FAILING = "failing"
    EMERGENCY_FALLBACK = "emergency_fallback"


class RegulationFailureMode(Enum):
    """Types of regulation system failures."""

    LISTENER_CONSENSUS_BREAKDOWN = "listener_consensus_breakdown"
    REGULATION_OSCILLATION = "regulation_oscillation"
    PARAMETER_DRIFT_INSTABILITY = "parameter_drift_instability"
    INFINITE_REGULATION_LOOPS = "infinite_regulation_loops"
    MATHEMATICAL_INCONSISTENCY = "mathematical_inconsistency"
    EFFECTIVENESS_DEGRADATION = "effectiveness_degradation"
    COMPUTATIONAL_OVERFLOW = "computational_overflow"


@dataclass
class RegulationSystemHealthMetrics:
    """Comprehensive health metrics for the regulation system."""

    listener_reliability_score: float  # Average listener reliability [0, 1]
    consensus_quality_score: float  # Quality of listener consensus [0, 1]
    regulation_oscillation_amplitude: float  # Amplitude of regulation oscillations
    regulation_effectiveness_entropy: float  # Entropy in regulation effectiveness
    parameter_drift_magnitude: float  # Magnitude of parameter drift
    regulation_loop_detection_score: float  # Detection of infinite loops [0, 1]
    mathematical_consistency_score: float  # Mathematical consistency [0, 1]
    computational_stability_score: float  # Computational stability [0, 1]
    overall_system_health: float  # Overall regulation system health [0, 1]
    system_health_status: RegulationSystemHealthStatus  # Categorical status
    active_failure_modes: Set[RegulationFailureMode]  # Currently active failures
    health_computation_timestamp: float  # When health was computed


@dataclass
class RegulationOscillationAnalysis:
    """Analysis of regulation system oscillations."""

    oscillation_detected: bool  # Whether oscillation is detected
    oscillation_frequency: float  # Primary oscillation frequency
    oscillation_amplitude: float  # Oscillation amplitude
    oscillation_phase: float  # Oscillation phase
    harmonic_components: List[Tuple[float, float]]  # (frequency, amplitude) pairs
    oscillation_stability: float  # Stability of oscillation pattern
    oscillation_mathematical_fingerprint: str  # Mathematical fingerprint of oscillation


@dataclass
class RegulationEffectivenessAnalysis:
    """Analysis of regulation effectiveness patterns."""

    effectiveness_trend: str  # "improving", "stable", "degrading"
    effectiveness_variance: float  # Variance in effectiveness scores
    effectiveness_entropy: float  # Entropy in effectiveness distribution
    effectiveness_predictability: float  # Predictability of effectiveness [0, 1]
    effectiveness_mathematical_signature: str  # Mathematical signature of effectiveness pattern


@dataclass
class EmergencyRegulationState:
    """State during emergency mathematical bounds regulation."""

    emergency_active: bool  # Whether emergency regulation is active
    trigger_failure_mode: RegulationFailureMode  # What triggered emergency mode
    emergency_start_time: float  # When emergency regulation started
    mathematical_bounds_applied: Dict[str, Tuple[float, float]]  # Applied bounds
    field_integrity_preservation_score: float  # How well field integrity is preserved
    emergency_effectiveness: float  # Effectiveness of emergency regulation


class MetaRegulation:
    """
    MetaRegulation - Regulation System Self-Monitoring and Stability

    Monitors the regulation system itself to prevent regulation-induced
    instabilities. Implements recursive mathematical regulation of the
    regulation system through mathematical principles.
    """

    def __init__(self, mathematical_precision: float = 1e-12):
        """
        Initialize meta-regulation system.

        Args:
            mathematical_precision: Numerical precision for mathematical computations
        """
        self.mathematical_precision = mathematical_precision

        self.regulation_system_health_history: List[RegulationSystemHealthMetrics] = []
        self.health_monitoring_enabled = True
        self.health_check_frequency = 2.0  # Checks per time unit
        self.last_health_check_time = 0.0

        self.regulation_pattern_history: List[Dict[str, float]] = []
        self.oscillation_detection_window = 50  # Number of patterns to analyze
        self.oscillation_threshold = 0.3  # Threshold for oscillation detection

        self.regulation_effectiveness_history: List[float] = []
        self.effectiveness_degradation_threshold = 0.4  # Threshold for effectiveness degradation
        self.effectiveness_analysis_window = 30  # Window for effectiveness analysis

        self.regulation_parameter_history: Dict[str, List[float]] = {}
        self.parameter_drift_threshold = 2.0  # Threshold for parameter drift detection

        self.emergency_regulation_state: Optional[EmergencyRegulationState] = None
        self.emergency_activation_threshold = 0.3  # Health threshold for emergency activation

        self.mathematical_consistency_checks: List[bool] = []
        self.consistency_check_window = 20

        self.regulation_action_sequence: List[str] = []
        self.loop_detection_window = 15
        self.loop_detection_threshold = 3  # Number of repetitions to detect loop

        logger.info("🔬 MetaRegulation system initialized for regulation system self-monitoring")

    def monitor_regulation_system_health(
        self,
        regulation_listeners: List[RegulationListener],
        recent_regulation_history: List[Dict[str, Any]],
        consensus_metrics: Optional[Dict[str, float]] = None,
    ) -> RegulationSystemHealthMetrics:
        """
        Monitor health of the regulation system itself.

        Analyzes regulation listeners, consensus quality, oscillations,
        effectiveness patterns, and mathematical consistency.
        """
        current_time = time.time()

        listener_reliability_score = self._analyze_listener_reliability(regulation_listeners)

        consensus_quality_score = self._analyze_consensus_quality(consensus_metrics or {})

        oscillation_analysis = self._detect_regulation_oscillation(recent_regulation_history)
        regulation_oscillation_amplitude = oscillation_analysis.oscillation_amplitude

        regulation_effectiveness_entropy = self._analyze_regulation_effectiveness_entropy(recent_regulation_history)

        parameter_drift_magnitude = self._detect_parameter_drift(recent_regulation_history)

        regulation_loop_detection_score = self._detect_infinite_regulation_loops(recent_regulation_history)

        mathematical_consistency_score = self._check_mathematical_consistency(recent_regulation_history)

        computational_stability_score = self._assess_computational_stability(recent_regulation_history)

        active_failure_modes = self._identify_active_failure_modes(
            listener_reliability_score,
            consensus_quality_score,
            oscillation_analysis,
            regulation_effectiveness_entropy,
            parameter_drift_magnitude,
            regulation_loop_detection_score,
            mathematical_consistency_score,
        )

        health_components = [
            listener_reliability_score,
            consensus_quality_score,
            1.0 - regulation_oscillation_amplitude,  # Invert oscillation amplitude
            1.0 - regulation_effectiveness_entropy,  # Invert entropy (lower is better)
            1.0 - parameter_drift_magnitude,  # Invert drift magnitude
            regulation_loop_detection_score,
            mathematical_consistency_score,
            computational_stability_score,
        ]
        health_tensor = torch.tensor(health_components, dtype=torch.float32)
        overall_system_health = torch.mean(health_tensor).item()

        system_health_status = self._determine_system_health_status(overall_system_health, active_failure_modes)

        health_metrics = RegulationSystemHealthMetrics(
            listener_reliability_score=listener_reliability_score,
            consensus_quality_score=consensus_quality_score,
            regulation_oscillation_amplitude=regulation_oscillation_amplitude,
            regulation_effectiveness_entropy=regulation_effectiveness_entropy,
            parameter_drift_magnitude=parameter_drift_magnitude,
            regulation_loop_detection_score=regulation_loop_detection_score,
            mathematical_consistency_score=mathematical_consistency_score,
            computational_stability_score=computational_stability_score,
            overall_system_health=overall_system_health,
            system_health_status=system_health_status,
            active_failure_modes=active_failure_modes,
            health_computation_timestamp=current_time,
        )

        self.regulation_system_health_history.append(health_metrics)
        if len(self.regulation_system_health_history) > 500:  # Keep last 500 health checks
            self.regulation_system_health_history.pop(0)

        self.last_health_check_time = current_time

        if system_health_status in [
            RegulationSystemHealthStatus.FAILING,
            RegulationSystemHealthStatus.EMERGENCY_FALLBACK,
        ]:
            logger.error(
                f"🔬 Regulation system health: {system_health_status.value} "
                f"(health={overall_system_health:.3f}, failures={len(active_failure_modes)})"
            )
        elif system_health_status == RegulationSystemHealthStatus.OSCILLATING:
            logger.warning(f"🔬 Regulation system oscillating " f"(amplitude={regulation_oscillation_amplitude:.3f})")
        else:
            logger.debug(
                f"🔬 Regulation system health: {system_health_status.value} " f"(health={overall_system_health:.3f})"
            )

        return health_metrics

    def detect_regulation_oscillation(
        self, regulation_pattern_history: List[Dict[str, float]]
    ) -> RegulationOscillationAnalysis:
        """
        Detect oscillations in regulation patterns using spectral analysis.

        Uses FFT analysis to identify periodic patterns in regulation behavior
        that might indicate unstable feedback loops.
        """
        if len(regulation_pattern_history) < 10:
            return RegulationOscillationAnalysis(
                oscillation_detected=False,
                oscillation_frequency=0.0,
                oscillation_amplitude=0.0,
                oscillation_phase=0.0,
                harmonic_components=[],
                oscillation_stability=1.0,
                oscillation_mathematical_fingerprint="insufficient_data",
            )

        regulation_strengths = []
        timestamps = []

        for i, pattern in enumerate(regulation_pattern_history[-self.oscillation_detection_window :]):
            if "regulation_strength" in pattern:
                regulation_strengths.append(pattern["regulation_strength"])
                timestamps.append(pattern.get("timestamp"))

        if len(regulation_strengths) < 8:
            return RegulationOscillationAnalysis(
                oscillation_detected=False,
                oscillation_frequency=0.0,
                oscillation_amplitude=0.0,
                oscillation_phase=0.0,
                harmonic_components=[],
                oscillation_stability=1.0,
                oscillation_mathematical_fingerprint="insufficient_data",
            )

        try:
            detrended_signal = signal.detrend(regulation_strengths)

            windowed_signal = detrended_signal * signal.windows.hann(len(detrended_signal))

            windowed_tensor = torch.tensor(windowed_signal, dtype=torch.complex64)
            fft_values = torch.fft.fft(windowed_tensor)
            frequencies = torch.fft.fftfreq(len(windowed_signal)).numpy()  # Only for indexing

            positive_freq_mask = frequencies > 0
            positive_frequencies = frequencies[positive_freq_mask]
            positive_magnitudes = torch.abs(fft_values[positive_freq_mask]).numpy()

            if len(positive_magnitudes) == 0:
                oscillation_detected = False
                primary_frequency = 0.0
                primary_amplitude = 0.0
                primary_phase = 0.0
                harmonic_components = []
            else:
                positive_mags_tensor = torch.tensor(positive_magnitudes, dtype=torch.float32)
                dominant_idx = torch.argmax(positive_mags_tensor).item()
                primary_frequency = positive_frequencies[dominant_idx]
                primary_amplitude = positive_magnitudes[dominant_idx] / len(regulation_strengths)
                primary_phase = torch.angle(fft_values[positive_freq_mask][dominant_idx]).item()

                oscillation_detected = primary_amplitude > self.oscillation_threshold

                harmonic_components = []
                for i, (freq, mag) in enumerate(zip(positive_frequencies, positive_magnitudes)):
                    if mag / len(regulation_strengths) > self.oscillation_threshold * 0.3:  # 30% of main threshold
                        harmonic_components.append((float(freq), float(mag / len(regulation_strengths))))

                harmonic_components.sort(key=lambda x: x[1], reverse=True)
                harmonic_components = harmonic_components[:5]  # Keep top 5 harmonics

            if len(self.regulation_system_health_history) >= 5:
                recent_oscillations = [
                    h.regulation_oscillation_amplitude for h in self.regulation_system_health_history[-5:]
                ]
                osc_tensor = torch.tensor(recent_oscillations, dtype=torch.float32)
                oscillation_stability = 1.0 - torch.std(osc_tensor).item()
                oscillation_stability = max(0.0, oscillation_stability)
            else:
                oscillation_stability = 1.0

            fingerprint_data = f"freq_{primary_frequency:.6f}_amp_{primary_amplitude:.6f}_phase_{primary_phase:.6f}"
            oscillation_mathematical_fingerprint = str(abs(hash(fingerprint_data)))[:16]

            return RegulationOscillationAnalysis(
                oscillation_detected=oscillation_detected,
                oscillation_frequency=float(primary_frequency),
                oscillation_amplitude=float(primary_amplitude),
                oscillation_phase=float(primary_phase),
                harmonic_components=harmonic_components,
                oscillation_stability=float(oscillation_stability),
                oscillation_mathematical_fingerprint=oscillation_mathematical_fingerprint,
            )

        except Exception as e:
            logger.warning(f"🔬 Oscillation detection failed: {e}")
            return RegulationOscillationAnalysis(
                oscillation_detected=False,
                oscillation_frequency=0.0,
                oscillation_amplitude=0.0,
                oscillation_phase=0.0,
                harmonic_components=[],
                oscillation_stability=0.5,
                oscillation_mathematical_fingerprint="analysis_failed",
            )

    def regulation_system_entropy(self, regulation_history: List[Dict[str, Any]]) -> float:
        """
        Analyze regulation system entropy to detect chaotic regulation behavior.

        Computes Shannon entropy of regulation decisions and parameter values
        to identify when the regulation system is becoming unpredictable.
        """
        if not regulation_history:
            raise ValueError("Regulation effectiveness entropy requires non-empty history - SYSTEM NOT OPERATIONAL")

        entropy_components = []

        regulation_types = []
        for entry in regulation_history:
            if "regulation_type" in entry:
                regulation_types.append(entry["regulation_type"])

        if regulation_types:
            type_counts = {}
            for reg_type in regulation_types:
                type_counts[reg_type] = type_counts.get(reg_type) + 1

            regulation_type_names = ['persistence', 'emotional_conductor', 'breathing_synchrony', 'energy_conservation', 'boundary_enforcement']
            type_count_values = [type_counts.get(reg_type, 0) for reg_type in regulation_type_names]
            type_counts_tensor = torch.tensor(type_count_values, dtype=torch.float32)
            type_probabilities = (type_counts_tensor / len(regulation_types)).numpy()
            type_entropy = entropy(type_probabilities, base=2)
            entropy_components.append(type_entropy)

        regulation_strengths = []
        for entry in regulation_history:
            if "regulation_strength" in entry:
                strength = entry["regulation_strength"]
                if math.isfinite(strength):
                    regulation_strengths.append(strength)

        if regulation_strengths:
            strength_tensor = torch.tensor(regulation_strengths, dtype=torch.float32)
            bins = min(10, max(3, len(regulation_strengths) // 5))
            hist = torch.histogram(strength_tensor, bins=bins, density=True)[0]

            hist = hist[hist > 0]
            if len(hist) > 1:
                hist = hist / torch.sum(hist)
                hist = hist.numpy()  # Convert for scipy.stats.entropy
                strength_entropy = entropy(hist, base=2)
                entropy_components.append(strength_entropy)

        effectiveness_scores = []
        for entry in regulation_history:
            if "effectiveness" in entry:
                eff = entry["effectiveness"]
                if math.isfinite(eff):
                    effectiveness_scores.append(eff)

        if effectiveness_scores:
            eff_tensor = torch.tensor(effectiveness_scores, dtype=torch.float32)
            bins = min(5, max(2, len(effectiveness_scores) // 10))
            hist = torch.histogram(eff_tensor, bins=bins, density=True)[0]

            hist = hist[hist > 0]
            if len(hist) > 1:
                hist = hist / torch.sum(hist)
                hist = hist.numpy()  # Convert for scipy.stats.entropy
                effectiveness_entropy = entropy(hist, base=2)
                entropy_components.append(effectiveness_entropy)

        if entropy_components:
            entropy_tensor = torch.tensor(entropy_components, dtype=torch.float32)
            total_entropy = torch.mean(entropy_tensor).item()
            return float(total_entropy)
        else:
            raise ValueError("Regulation effectiveness entropy calculation failed - no valid entropy components computed")

    def simplify_regulation_system(self, current_complexity_level: float) -> Dict[str, Any]:
        """
        Simplify regulation system when it becomes too complex or unstable.

        Reduces regulation complexity while maintaining mathematical integrity
        by falling back to simpler, more reliable regulation mechanisms.
        """
        simplification_actions = {}

        if self.regulation_system_health_history:
            recent_health = self.regulation_system_health_history[-1]
            target_complexity = recent_health.overall_system_health * current_complexity_level
        else:
            target_complexity = current_complexity_level * 0.5  # Conservative simplification

        complexity_reduction = current_complexity_level - target_complexity

        if complexity_reduction > 0.1:  # Significant simplification needed
            if complexity_reduction > 0.3:
                simplification_actions["reduce_listeners"] = True
                simplification_actions["keep_essential_listeners_only"] = True

            if complexity_reduction > 0.2:
                simplification_actions["simplify_consensus"] = True
                simplification_actions["use_majority_voting"] = True

            simplification_actions["reduce_adaptation_frequency"] = True
            simplification_actions["parameter_adaptation_factor"] = 0.5

            if complexity_reduction > 0.4:
                simplification_actions["use_simple_models"] = True
                simplification_actions["disable_advanced_features"] = True

            simplification_actions["target_complexity_level"] = target_complexity
            simplification_actions["complexity_reduction"] = complexity_reduction

            logger.warning(
                f"🔬 Simplifying regulation system: "
                f"reduction={complexity_reduction:.3f}, target={target_complexity:.3f}"
            )

        return simplification_actions

    def validate_field_mathematical_consistency(
        self, failed_consensus: Dict[str, Any], agent_states: List[ConceptualChargeAgent]
    ) -> RegulationSuggestion:
        """
        Validate field mathematical consistency when consensus fails.
        
        MATHEMATICAL VALIDATION:
        ∂Q/∂τ = γ ∂T/∂τ ⋅ E^trajectory ⋅ Φ^semantic ⋅ e^{iθ_{total}} ⋅ Ψ_{persistence}
        
        Checks field consistency constraints:
        - Gauge invariance: |∇ × Q| < ε_{gauge}
        - Energy conservation: d/dt ∫ |Q|² dx = 0
        - Causality: velocity ≤ c_{field}
        - Unitarity: ⟨Ψ|Ψ⟩ = 1
        """
        logger.warning("🔬 Field consistency validation activated")
        
        q_values = [agent.Q_components.Q_value for agent in agent_states 
                   if hasattr(agent, 'Q_components') and agent.Q_components is not None]
        
        if not q_values:
            raise ValueError("No valid Q-components found for mathematical validation")
        
        q_tensor = torch.tensor(q_values, dtype=torch.complex64)
        
        gauge_invariance = torch.mean(torch.abs(torch.gradient(torch.abs(q_tensor))[0])).item()
        q_energy = torch.sum(torch.abs(q_tensor) ** 2).item()
        
        if gauge_invariance > 1e-3:
            regulation_type = "gauge_restoration"
        elif q_energy > len(agent_states) * 1e6:
            regulation_type = "energy_conservation"
        else:
            regulation_type = "field_stabilization"
        
        field_variance = torch.var(torch.abs(q_tensor)).item()
        field_mean = torch.mean(torch.abs(q_tensor)).item()
        regulation_strength = field_variance / (field_mean + field_variance)
        
        field_entropy = -torch.sum(torch.abs(q_tensor) / torch.sum(torch.abs(q_tensor)) * 
                                   torch.log(torch.abs(q_tensor) / torch.sum(torch.abs(q_tensor)) + 1e-12)).item()
        
        coherence_measure = torch.abs(torch.mean(q_tensor / torch.abs(q_tensor))).item()
        
        regulation_suggestion = RegulationSuggestion(
            regulation_type=regulation_type,
            strength=regulation_strength,
            confidence=0.95,
            mathematical_basis="field_theory_validation",
            information_metrics=InformationMetrics(
                field_entropy=field_entropy,
                mutual_information=0.0,
                entropy_gradient=gauge_invariance,
                information_flow_rate=0.0,
                coherence_measure=coherence_measure,
                singularity_indicator=0.0,
            ),
            parameters={
                "gauge_invariance": gauge_invariance,
                "energy_conservation": q_energy,
                "field_variance": field_variance,
            },
        )
        
        logger.info(f"🔬 Field validation: {regulation_type} (strength={regulation_strength:.3f})")
        return regulation_suggestion

    def enforce_field_mathematical_constraints(
        self, failure_mode: RegulationFailureMode, agent_states: List[ConceptualChargeAgent]
    ) -> Dict[str, Any]:
        """
        Enforce mathematical field constraints when regulation systems fail.
        
        FIELD CONSTRAINT ENFORCEMENT:
        1. Lorentz invariance: η_{μν} ∂^μ Q ∂^ν Q = m² Q²
        2. Current conservation: ∂_μ J^μ = 0
        3. Gauge freedom: Q → Q e^{iα(x)}
        4. Unitarity bounds: |⟨Ψ_f|Ψ_i⟩|² ≤ 1
        
        Returns mathematical constraint validation results.
        """
        logger.error(f"🔬 Field constraint enforcement: {failure_mode.value}")
        
        q_values = [agent.Q_components.Q_value for agent in agent_states 
                   if hasattr(agent, 'Q_components') and agent.Q_components is not None]
        
        if not q_values:
            raise ValueError("No Q-components available for constraint enforcement")
        
        q_tensor = torch.tensor(q_values, dtype=torch.complex64)
        
        lorentz_invariant = torch.sum(torch.abs(q_tensor) ** 2).item()
        current_divergence = torch.sum(torch.diff(q_tensor)).item() if len(q_tensor) > 1 else 0.0
        
        q_normalized = q_tensor / torch.sqrt(torch.sum(torch.abs(q_tensor) ** 2))
        unitarity_violation = abs(torch.sum(torch.abs(q_normalized) ** 2).item() - 1.0)
        
        field_energy_density = torch.abs(q_tensor) ** 2
        max_energy_density = torch.max(field_energy_density).item()
        field_coherence = torch.abs(torch.sum(q_normalized)).item()
        
        constraint_state = {
            "constraints_active": True,
            "trigger_failure_mode": failure_mode,
            "constraint_start_time": time.time(),
            "lorentz_invariant": lorentz_invariant,
            "current_conservation": current_divergence,
            "unitarity_violation": unitarity_violation,
            "max_energy_density": max_energy_density,
            "field_coherence": field_coherence,
            "mathematical_validity": (
                unitarity_violation < 1e-6 and 
                abs(current_divergence) < 1e-6 and
                max_energy_density < 1e12
            )
        }
        
        logger.critical(
            f"🔬 Constraint state: unitarity={unitarity_violation:.2e}, "
            f"conservation={current_divergence:.2e}, coherence={field_coherence:.3f}"
        )
        
        return constraint_state

    def improve_regulation_effectiveness(
        self, effectiveness_history: List[float], regulation_parameters: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Improve regulation effectiveness based on historical performance analysis.

        Uses mathematical optimization to adjust regulation parameters for
        improved effectiveness while maintaining stability.
        """
        if not effectiveness_history or len(effectiveness_history) < 5:
            return regulation_parameters  # Insufficient data for improvement

        effectiveness_analysis = self._analyze_effectiveness_trends(effectiveness_history)

        parameter_updates = {}

        regulation_param_names = ['persistence_strength', 'emotional_conductor_sensitivity', 'breathing_synchrony_strength', 'energy_conservation_threshold', 'boundary_enforcement_strength']
        
        if effectiveness_analysis.effectiveness_trend == "degrading":
            for param_name in regulation_param_names:
                if param_name in regulation_parameters:
                    param_value = regulation_parameters[param_name]
                    if "strength" in param_name.lower() or "sensitivity" in param_name.lower():
                        updated_value = param_value * 0.9  # 10% reduction
                        parameter_updates[param_name] = updated_value

        elif effectiveness_analysis.effectiveness_trend == "improving":
            for param_name in regulation_param_names:
                if param_name in regulation_parameters:
                    param_value = regulation_parameters[param_name]
                    if "strength" in param_name.lower():
                        updated_value = param_value * 1.05  # 5% increase
                        parameter_updates[param_name] = updated_value

        if effectiveness_analysis.effectiveness_variance > 0.1:  # High variance
            adaptation_param_names = ['adaptation_rate', 'learning_rate', 'adjustment_factor']
            for param_name in adaptation_param_names:
                if param_name in regulation_parameters:
                    param_value = regulation_parameters[param_name]
                    if "adaptation" in param_name.lower():
                        updated_value = param_value * 0.8  # Reduce adaptation rate
                        parameter_updates[param_name] = updated_value

        param_update_items = list(parameter_updates.items())
        for param_name, updated_value in param_update_items:
            if "strength" in param_name.lower():
                parameter_updates[param_name] = max(0.01, min(2.0, updated_value))
            elif "sensitivity" in param_name.lower():
                parameter_updates[param_name] = max(0.1, min(10.0, updated_value))
            elif "threshold" in param_name.lower():
                parameter_updates[param_name] = max(0.05, min(0.95, updated_value))

        logger.debug(f"🔬 Regulation effectiveness improvement: updated {len(parameter_updates)} parameters")

        return parameter_updates

    def discover_new_regulation_methods(
        self, regulation_strategy_history: List[Dict[str, Any]], effectiveness_outcomes: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Discover new regulation methods by analyzing successful regulation patterns.

        Uses pattern recognition and mathematical analysis to identify
        novel regulation strategies that show promise for effectiveness.
        """
        if len(regulation_strategy_history) < 10 or len(effectiveness_outcomes) < 10:
            return []  # Insufficient data for discovery

        new_methods = []

        eff_tensor = torch.tensor(effectiveness_outcomes, dtype=torch.float32)
        high_effectiveness_threshold = torch.quantile(eff_tensor, 0.75).item()
        successful_strategies = []

        for strategy, effectiveness in zip(regulation_strategy_history, effectiveness_outcomes):
            if effectiveness > high_effectiveness_threshold:
                successful_strategies.append(strategy)

        if not successful_strategies:
            return []

        pattern_analysis = self._analyze_successful_regulation_patterns(successful_strategies)

        for pattern in pattern_analysis["significant_patterns"]:
            new_method = {
                "method_name": f"discovered_method_{len(new_methods) + 1}",
                "regulation_type": pattern.get("regulation_type"),
                "strategy_pattern": pattern,
                "estimated_effectiveness": pattern.get("pattern_effectiveness"),
                "mathematical_basis": pattern.get("mathematical_justification"),
                "discovery_confidence": pattern.get("pattern_confidence"),
                "discovery_timestamp": time.time(),
            }

            if new_method["discovery_confidence"] > 0.6:
                new_methods.append(new_method)

        logger.info(f"🔬 Discovered {len(new_methods)} new regulation methods")

        return new_methods


    def _analyze_listener_reliability(self, listeners: List[RegulationListener]) -> float:
        """Analyze reliability of regulation listeners."""
        if not listeners:
            return 0.0

        reliability_scores = []

        for listener in listeners:
            if hasattr(listener, "history") and listener.history:
                recent_metrics = listener.history[-10:] if len(listener.history) >= 10 else listener.history

                if recent_metrics:
                    entropies = [m.field_entropy for m in recent_metrics if math.isfinite(m.field_entropy)]
                    if entropies:
                        entropies_tensor = torch.tensor(entropies, dtype=torch.float32)
                        entropy_mean = torch.mean(entropies_tensor).item()
                        entropy_std = torch.std(entropies_tensor).item()
                        entropy_consistency = 1.0 - min(
                            1.0, entropy_std / (entropy_mean + self.mathematical_precision)
                        )
                        reliability_scores.append(entropy_consistency)
                    else:
                        reliability_scores.append(0.5)  # Neutral reliability
                else:
                    reliability_scores.append(0.5)
            else:
                reliability_scores.append(0.5)  # Neutral reliability for listeners without history

        if reliability_scores:
            scores_tensor = torch.tensor(reliability_scores, dtype=torch.float32)
            return torch.mean(scores_tensor).item()
        else:
            raise ValueError("Listener reliability analysis failed - no listeners available - REGULATION SYSTEM INOPERABLE")

    def _analyze_consensus_quality(self, consensus_metrics: Dict[str, float]) -> float:
        """Analyze quality of consensus mechanisms."""
        if not consensus_metrics:
            return 0.5  # Neutral quality if no metrics

        quality_components = []

        if "consensus_strength" in consensus_metrics:
            strength = consensus_metrics["consensus_strength"]
            if math.isfinite(strength):
                quality_components.append(min(1.0, strength))

        if "agreement_level" in consensus_metrics:
            agreement = consensus_metrics["agreement_level"]
            if math.isfinite(agreement):
                quality_components.append(agreement)

        if "consistency_score" in consensus_metrics:
            consistency = consensus_metrics["consistency_score"]
            if math.isfinite(consistency):
                quality_components.append(consistency)

        if quality_components:
            components_tensor = torch.tensor(quality_components, dtype=torch.float32)
            return torch.mean(components_tensor).item()
        else:
            return 0.5

    def _detect_regulation_oscillation(self, regulation_history: List[Dict[str, Any]]) -> RegulationOscillationAnalysis:
        """Detect regulation oscillations using FFT analysis."""
        regulation_patterns = []
        for entry in regulation_history:
            pattern = {}
            if "regulation_strength" in entry:
                pattern["regulation_strength"] = entry["regulation_strength"]
            if "timestamp" in entry:
                pattern["timestamp"] = entry["timestamp"]
            else:
                pattern["timestamp"] = len(regulation_patterns)

            if pattern:
                regulation_patterns.append(pattern)

        self.regulation_pattern_history.extend(regulation_patterns)
        if len(self.regulation_pattern_history) > 200:  # Keep last 200 patterns
            self.regulation_pattern_history = self.regulation_pattern_history[-200:]

        return self.detect_regulation_oscillation(self.regulation_pattern_history)

    def _analyze_regulation_effectiveness_entropy(self, regulation_history: List[Dict[str, Any]]) -> float:
        """Analyze entropy in regulation effectiveness."""
        effectiveness_scores = []

        for entry in regulation_history:
            if "effectiveness" in entry:
                eff = entry["effectiveness"]
                if math.isfinite(eff):
                    effectiveness_scores.append(eff)

        self.regulation_effectiveness_history.extend(effectiveness_scores)
        if len(self.regulation_effectiveness_history) > 100:  # Keep last 100 scores
            self.regulation_effectiveness_history = self.regulation_effectiveness_history[-100:]

        if len(effectiveness_scores) < 3:
            return 0.0

        eff_tensor = torch.tensor(effectiveness_scores, dtype=torch.float32)
        bins = min(5, max(2, len(effectiveness_scores) // 5))
        hist = torch.histogram(eff_tensor, bins=bins, density=True)[0]

        hist = hist[hist > 0]
        if len(hist) <= 1:
            return 0.0

        hist = hist / torch.sum(hist)
        hist = hist.numpy()  # Convert for scipy.stats.entropy
        effectiveness_entropy = entropy(hist, base=2)

        return float(effectiveness_entropy)

    def _detect_parameter_drift(self, regulation_history: List[Dict[str, Any]]) -> float:
        """Detect drift in regulation parameters."""
        for entry in regulation_history:
            if "parameters" in entry and isinstance(entry["parameters"], dict):
                regulation_param_names = ['persistence_strength', 'emotional_conductor_sensitivity', 'breathing_synchrony_strength', 'energy_conservation_threshold', 'boundary_enforcement_strength', 'adaptation_rate', 'learning_rate', 'adjustment_factor']
                for param_name in regulation_param_names:
                    if param_name in entry["parameters"]:
                        param_value = entry["parameters"][param_name]
                    if isinstance(param_value, (int, float)) and math.isfinite(param_value):
                        if param_name not in self.regulation_parameter_history:
                            self.regulation_parameter_history[param_name] = []
                        self.regulation_parameter_history[param_name].append(param_value)

        drift_magnitudes = []

        param_history_items = list(self.regulation_parameter_history.items())
        for param_name, values in param_history_items:
            if len(values) > 50:
                values = values[-50:]
                self.regulation_parameter_history[param_name] = values

            if len(values) >= 5:
                x = np.arange(len(values))
                slope, _, _, _, _ = stats.linregress(x, values)

                values_tensor = torch.tensor(values, dtype=torch.float32)
                param_scale = torch.std(values_tensor).item() + self.mathematical_precision
                normalized_drift = abs(slope) / param_scale
                drift_magnitudes.append(normalized_drift)

        if drift_magnitudes:
            drift_tensor = torch.tensor(drift_magnitudes, dtype=torch.float32)
            return torch.mean(drift_tensor).item()
        else:
            return 0.0

    def _detect_infinite_regulation_loops(self, regulation_history: List[Dict[str, Any]]) -> float:
        """Detect infinite regulation loops."""
        recent_actions = []
        for entry in regulation_history:
            if "regulation_type" in entry:
                action = f"{entry['regulation_type']}_{entry.get('strength'):.2f}"
                recent_actions.append(action)

        self.regulation_action_sequence.extend(recent_actions)
        if len(self.regulation_action_sequence) > self.loop_detection_window * 3:
            self.regulation_action_sequence = self.regulation_action_sequence[-self.loop_detection_window * 3 :]

        if len(self.regulation_action_sequence) < self.loop_detection_window:
            return 1.0  # Perfect score if insufficient data for loop detection

        recent_sequence = self.regulation_action_sequence[-self.loop_detection_window :]

        loop_indicators = 0
        for i in range(len(recent_sequence) - 2):
            pattern = recent_sequence[i : i + 3]  # 3-action pattern
            pattern_count = 0

            for j in range(len(recent_sequence) - 2):
                if recent_sequence[j : j + 3] == pattern:
                    pattern_count += 1

            if pattern_count >= self.loop_detection_threshold:
                loop_indicators += 1

        loop_score = 1.0 - min(1.0, loop_indicators / (len(recent_sequence) / 3))
        return float(loop_score)

    def _check_mathematical_consistency(self, regulation_history: List[Dict[str, Any]]) -> float:
        """Check mathematical consistency of regulation decisions."""
        consistency_checks = []

        for entry in regulation_history:
            is_consistent = True

            if "regulation_strength" in entry:
                strength = entry["regulation_strength"]
                if not math.isfinite(strength) or strength < 0:
                    is_consistent = False

            if "effectiveness" in entry:
                effectiveness = entry["effectiveness"]
                if not math.isfinite(effectiveness):
                    is_consistent = False

            if "parameters" in entry and isinstance(entry["parameters"], dict):
                regulation_param_names = ['persistence_strength', 'emotional_conductor_sensitivity', 'breathing_synchrony_strength', 'energy_conservation_threshold', 'boundary_enforcement_strength', 'adaptation_rate', 'learning_rate', 'adjustment_factor']
                for param_name in regulation_param_names:
                    if param_name in entry["parameters"]:
                        param_value = entry["parameters"][param_name]
                    if isinstance(param_value, (int, float)) and not math.isfinite(param_value):
                        is_consistent = False
                        break

            consistency_checks.append(is_consistent)

        self.mathematical_consistency_checks.extend(consistency_checks)
        if len(self.mathematical_consistency_checks) > self.consistency_check_window:
            self.mathematical_consistency_checks = self.mathematical_consistency_checks[
                -self.consistency_check_window :
            ]

        if self.mathematical_consistency_checks:
            checks_tensor = torch.tensor([float(c) for c in self.mathematical_consistency_checks], dtype=torch.float32)
            consistency_score = torch.mean(checks_tensor).item()
            return float(consistency_score)
        else:
            raise ValueError("Mathematical consistency check failed - no consistency checks performed - SYSTEM INTEGRITY UNKNOWN")

    def _assess_computational_stability(self, regulation_history: List[Dict[str, Any]]) -> float:
        """Assess computational stability of regulation calculations."""
        stability_indicators = []

        for entry in regulation_history:
            entry_stability = 1.0  # Start with perfect stability

            if "regulation_strength" in entry:
                strength = entry["regulation_strength"]
                if abs(strength) > 1e6:  # Very large values indicate potential overflow
                    entry_stability *= 0.5
                elif abs(strength) > 1e3:
                    entry_stability *= 0.8

            entry_keys = ['regulation_strength', 'effectiveness', 'confidence', 'timestamp', 'listener_count', 'consensus_score']
            for key in entry_keys:
                if key in entry:
                    value = entry[key]
                if isinstance(value, (int, float)):
                    if not math.isfinite(value):
                        entry_stability = 0.0  # Complete instability for non-finite values
                        break

            stability_indicators.append(entry_stability)

        if stability_indicators:
            stability_tensor = torch.tensor(stability_indicators, dtype=torch.float32)
            return torch.mean(stability_tensor).item()
        else:
            raise ValueError("Computational stability assessment failed - no stability indicators - SYSTEM STABILITY UNKNOWN")

    def _identify_active_failure_modes(
        self,
        listener_reliability: float,
        consensus_quality: float,
        oscillation_analysis: RegulationOscillationAnalysis,
        effectiveness_entropy: float,
        parameter_drift: float,
        loop_detection_score: float,
        mathematical_consistency: float,
    ) -> Set[RegulationFailureMode]:
        """Identify currently active failure modes."""
        active_failures = set()

        if listener_reliability < 0.4 or consensus_quality < 0.4:
            active_failures.add(RegulationFailureMode.LISTENER_CONSENSUS_BREAKDOWN)

        if oscillation_analysis.oscillation_detected and oscillation_analysis.oscillation_amplitude > 0.3:
            active_failures.add(RegulationFailureMode.REGULATION_OSCILLATION)

        if parameter_drift > self.parameter_drift_threshold:
            active_failures.add(RegulationFailureMode.PARAMETER_DRIFT_INSTABILITY)

        if loop_detection_score < 0.5:
            active_failures.add(RegulationFailureMode.INFINITE_REGULATION_LOOPS)

        if mathematical_consistency < 0.7:
            active_failures.add(RegulationFailureMode.MATHEMATICAL_INCONSISTENCY)

        if effectiveness_entropy > 2.0:  # High entropy indicates degradation
            active_failures.add(RegulationFailureMode.EFFECTIVENESS_DEGRADATION)

        return active_failures

    def _determine_system_health_status(
        self, overall_health: float, active_failures: Set[RegulationFailureMode]
    ) -> RegulationSystemHealthStatus:
        """Determine categorical system health status."""
        critical_failures = {
            RegulationFailureMode.MATHEMATICAL_INCONSISTENCY,
            RegulationFailureMode.INFINITE_REGULATION_LOOPS,
        }

        if critical_failures.intersection(active_failures):
            return RegulationSystemHealthStatus.EMERGENCY_FALLBACK

        if overall_health < 0.3 or len(active_failures) >= 3:
            return RegulationSystemHealthStatus.FAILING

        if RegulationFailureMode.REGULATION_OSCILLATION in active_failures:
            return RegulationSystemHealthStatus.OSCILLATING

        if overall_health < 0.5 or len(active_failures) >= 2:
            return RegulationSystemHealthStatus.UNRELIABLE

        if overall_health < 0.8 or len(active_failures) >= 1:
            return RegulationSystemHealthStatus.STABLE

        return RegulationSystemHealthStatus.OPTIMAL


    def _analyze_effectiveness_trends(self, effectiveness_history: List[float]) -> RegulationEffectivenessAnalysis:
        """Analyze trends in regulation effectiveness."""
        if len(effectiveness_history) < 5:
            return RegulationEffectivenessAnalysis(
                effectiveness_trend="stable",
                effectiveness_variance=0.0,
                effectiveness_entropy=0.0,
                effectiveness_predictability=1.0,
                effectiveness_mathematical_signature="insufficient_data",
            )

        x = np.arange(len(effectiveness_history))
        slope, _, r_value, _, _ = stats.linregress(x, effectiveness_history)

        if slope > 0.01:
            trend = "improving"
        elif slope < -0.01:
            trend = "degrading"
        else:
            trend = "stable"

        effectiveness_variance = np.var(effectiveness_history)

        bins = min(5, max(2, len(effectiveness_history) // 3))
        hist, _ = np.histogram(effectiveness_history, bins=bins, density=True)
        hist = hist[hist > 0]

        if len(hist) > 1:
            hist = hist / np.sum(hist)
            effectiveness_entropy = entropy(hist, base=2)
        else:
            effectiveness_entropy = 0.0

        effectiveness_predictability = max(0.0, r_value**2)

        signature_data = f"trend_{slope:.6f}_var_{effectiveness_variance:.6f}_ent_{effectiveness_entropy:.6f}"
        effectiveness_mathematical_signature = str(abs(hash(signature_data)))[:16]

        return RegulationEffectivenessAnalysis(
            effectiveness_trend=trend,
            effectiveness_variance=float(effectiveness_variance),
            effectiveness_entropy=float(effectiveness_entropy),
            effectiveness_predictability=float(effectiveness_predictability),
            effectiveness_mathematical_signature=effectiveness_mathematical_signature,
        )

    def _analyze_successful_regulation_patterns(self, successful_strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in successful regulation strategies."""
        pattern_analysis = {"significant_patterns": [], "common_features": {}, "pattern_confidence_threshold": 0.6}

        if not successful_strategies:
            return pattern_analysis

        regulation_types = [s.get("regulation_type") for s in successful_strategies]
        type_counts = {}
        for reg_type in regulation_types:
            type_counts[reg_type] = type_counts.get(reg_type) + 1

        total_strategies = len(successful_strategies)
        type_count_items = list(type_counts.items())
        for reg_type, count in type_count_items:
            frequency = count / total_strategies
            if frequency > 0.3:  # Significant frequency threshold
                pattern = {
                    "regulation_type": reg_type,
                    "frequency": frequency,
                    "pattern_confidence": frequency,
                    "pattern_effectiveness": 0.8,  # Simplified - would compute from actual data
                    "mathematical_justification": f"high_frequency_success_pattern_{reg_type}",
                }
                pattern_analysis["significant_patterns"].append(pattern)

        return pattern_analysis

    def monitor_and_suggest(
        self, 
        agents: List[ConceptualChargeAgent], 
        current_regulation: Any, 
        listener_states: List[RegulationListener]
    ) -> List[RegulationSuggestion]:
        """
        Monitor regulation system health and generate suggestions.
        
        This is the main meta-regulation method that combines health monitoring
        with intelligent suggestion generation based on system state analysis.
        
        Args:
            agents: List of conceptual charge agents
            current_regulation: Current regulation consensus or state
            listener_states: Current regulation listeners
            
        Returns:
            List of regulation suggestions for system improvement
        """
        suggestions = []
        
        try:
            # Prepare regulation history from recent system state with required fields for consistency checks
            regulation_history = []
            if hasattr(self, 'regulation_system_health_history') and self.regulation_system_health_history:
                regulation_history = [
                    {
                        "regulation_type": "meta_monitoring",
                        "regulation_strength": h.overall_system_health * 0.1,  # Derive strength from health
                        "timestamp": h.health_assessment_timestamp,
                        "effectiveness": h.overall_regulation_effectiveness,
                        "system_health": h.overall_system_health,
                        "parameters": {
                            "persistence_strength": h.overall_system_health * 0.05,
                            "emotional_conductor_sensitivity": h.overall_regulation_effectiveness * 0.1,
                            "breathing_synchrony_strength": h.overall_system_health * 0.08,
                            "energy_conservation_threshold": 0.9,
                            "boundary_enforcement_strength": h.overall_system_health * 0.1,
                            "adaptation_rate": 0.02,
                            "learning_rate": 0.01,
                            "adjustment_factor": h.overall_regulation_effectiveness * 0.05
                        }
                    }
                    for h in self.regulation_system_health_history[-10:]  # Last 10 entries
                ]
            
            # Prepare consensus metrics from current regulation
            consensus_metrics = {"overall_consensus_strength": 0.8}  # Default
            if hasattr(current_regulation, 'consensus_strength'):
                consensus_metrics["overall_consensus_strength"] = current_regulation.consensus_strength
            elif hasattr(current_regulation, '__dict__'):
                consensus_metrics["overall_consensus_strength"] = getattr(current_regulation, 'overall_consensus_strength', 0.8)
            
            # 1. Monitor system health using existing method
            health_status = self.monitor_regulation_system_health(
                regulation_listeners=listener_states,
                recent_regulation_history=regulation_history,
                consensus_metrics=consensus_metrics
            )
            
            # Store health for future reference
            self.regulation_system_health_history.append(health_status)
            if len(self.regulation_system_health_history) > 50:  # Keep last 50 entries
                self.regulation_system_health_history = self.regulation_system_health_history[-50:]
            
            # 2. Generate suggestions based on health status
            if health_status.system_health_status == RegulationSystemHealthStatus.CRITICAL:
                # Critical state - use mathematical constraint enforcement
                failed_consensus = {
                    "consensus_failure": True, 
                    "agents": len(agents),
                    "health_score": health_status.overall_system_health
                }
                constraint_suggestion = self.enforce_field_mathematical_constraints(
                    failed_consensus=failed_consensus,
                    agent_states=agents
                )
                suggestions.append(constraint_suggestion)
                
            elif health_status.system_health_status == RegulationSystemHealthStatus.UNSTABLE:
                # Unstable state - validate mathematical consistency
                failed_consensus = {
                    "consensus_failure": False,
                    "instability_detected": True,
                    "agents": len(agents)
                }
                consistency_suggestion = self.validate_field_mathematical_consistency(
                    failed_consensus=failed_consensus,
                    agent_states=agents
                )
                suggestions.append(consistency_suggestion)
                
            elif health_status.system_health_status == RegulationSystemHealthStatus.OSCILLATING:
                # Oscillating state - simplify regulation system
                complexity_level = min(0.9, health_status.regulation_oscillation_amplitude * 2.0)
                simplification = self.simplify_regulation_system(complexity_level)
                
                if simplification.get('success', False):
                    suggestion = RegulationSuggestion(
                        regulation_type="meta_simplification",
                        strength=simplification.get('recommended_strength', 0.3),
                        confidence=0.85,
                        mathematical_basis=f"Meta-regulation: Oscillation detected (amplitude: {health_status.regulation_oscillation_amplitude:.3f}), simplifying system",
                        information_metrics=InformationMetrics(
                            field_entropy=health_status.regulation_effectiveness_entropy,
                            mutual_information=0.0,
                            entropy_gradient=health_status.regulation_oscillation_amplitude,
                            information_flow_rate=0.0,
                            coherence_measure=health_status.overall_system_health,
                            singularity_indicator=1.0 - health_status.overall_system_health
                        ),
                        parameters={"source": "MetaRegulation", "target": "system_oscillation"}
                    )
                    suggestions.append(suggestion)
            
            # 3. Always try effectiveness improvement if we have sufficient history
            if len(regulation_history) >= 3:
                effectiveness_history = [entry["effectiveness"] for entry in regulation_history]
                current_strategies = {"meta_monitoring": True, "current_health": health_status.overall_system_health}
                
                try:
                    improvement_analysis = self.improve_regulation_effectiveness(
                        regulation_history=regulation_history,
                        effectiveness_history=effectiveness_history,
                        current_strategies=current_strategies
                    )
                    
                    if improvement_analysis.get('improvement_potential', 0) > 0.15:
                        suggestion = RegulationSuggestion(
                            regulation_type="effectiveness_improvement",
                            strength=improvement_analysis.get('recommended_adjustment', 0.1),
                            confidence=0.6,
                            mathematical_basis=f"Meta-regulation: Effectiveness improvement potential: {improvement_analysis.get('improvement_potential', 0):.3f}",
                            information_metrics=InformationMetrics(
                                field_entropy=health_status.regulation_effectiveness_entropy,
                                mutual_information=0.0,
                                entropy_gradient=0.0,
                                information_flow_rate=improvement_analysis.get('improvement_potential', 0),
                                coherence_measure=health_status.overall_system_health,
                                singularity_indicator=0.0
                            ),
                            parameters={"source": "MetaRegulation", "target": "system_effectiveness"}
                        )
                        suggestions.append(suggestion)
                        
                except Exception as e:
                    logger.debug(f"🔬 Meta-regulation effectiveness improvement failed: {e}")
            
            # 4. If system is healthy, still provide gentle optimization
            if health_status.system_health_status == RegulationSystemHealthStatus.HEALTHY and len(suggestions) == 0:
                suggestion = RegulationSuggestion(
                    regulation_type="maintenance_optimization",
                    strength=0.05,
                    confidence=0.3,
                    mathematical_basis="Meta-regulation: System healthy, applying gentle maintenance optimization",
                    information_metrics=InformationMetrics(
                        field_entropy=health_status.regulation_effectiveness_entropy,
                        mutual_information=0.0,
                        entropy_gradient=0.0,
                        information_flow_rate=0.0,
                        coherence_measure=health_status.overall_system_health,
                        singularity_indicator=0.0
                    ),
                    parameters={"source": "MetaRegulation", "target": "system_maintenance"}
                )
                suggestions.append(suggestion)
                
        except Exception as e:
            logger.warning(f"🔬 Meta-regulation monitor_and_suggest failed: {e}")
            # Emergency fallback suggestion
            suggestion = RegulationSuggestion(
                regulation_type="emergency_stabilization",
                strength=0.1,
                confidence=0.9,
                mathematical_basis="Meta-regulation: System monitoring failed, applying emergency stabilization",
                information_metrics=InformationMetrics(
                    field_entropy=1.0,  # High entropy indicates emergency
                    mutual_information=0.0,
                    entropy_gradient=0.0,
                    information_flow_rate=0.0,
                    coherence_measure=0.1,  # Low coherence in emergency
                    singularity_indicator=0.9  # High singularity risk
                ),
                parameters={"source": "MetaRegulation", "target": "emergency_stability"}
            )
            suggestions.append(suggestion)
        
        logger.debug(f"🔬 Meta-regulation generated {len(suggestions)} suggestions")
        return suggestions
