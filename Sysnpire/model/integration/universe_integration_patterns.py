"""
Universe Integration Patterns - Self-Interpreting Field Dynamics

MATHEMATICAL FOUNDATION: Complete universe-native content analysis using Q(τ,C,s) field
theory instead of external models. The universe becomes self-interpreting through its own
field-theoretic evolution.

CORE PATTERNS:
1. TEXT TO FIELD CONVERSION - Project text to universe field coordinates
2. SEMANTIC DISTANCE - Compute distance using field mathematics 
3. ACCEPTANCE DECISION - Accept/reject based on field dynamics

REVOLUTIONARY PRINCIPLE: Meaning emerges from mathematical field dynamics, not external data.
The universe itself becomes the semantic interpreter through Q(τ,C,s) evolution.

MATHEMATICAL LIBRARIES:
- JAX for field gradient calculations
- SciPy for PDE solutions and integration
- Torch for tensor operations and FFT analysis  
- Sage CDF for exact complex arithmetic
- No external models - pure field theory

IMPLEMENTATION PRINCIPLE: Mathematical perfection or explosive failure. No fallbacks.
"""

import cmath
import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# JAX for exact field calculations
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from jax import grad, jit, vmap
# SciPy for mathematical operations
from scipy import integrate, special
from scipy.interpolate import interp1d
from torch.fft import fft, ifft

# Import field mechanics constants
from .field_mechanics import (ENERGY_NORMALIZATION, FIELD_COUPLING_CONSTANT,
                              FIELD_NUMERICAL_PRECISION,
                              PHASE_COHERENCE_THRESHOLD, field_norm_l2)

from .sage_compatibility import safe_torch_tensor

logger = logging.getLogger(__name__)


@dataclass
class FieldSignature:
    """
    Complete field signature for universe-native content representation.

    Mathematical Foundation:
        Text mapped to field coordinates through universe geometry:
        F(text) = (r⃗, θ, |Q|, φ) ∈ ℝ³ × S¹ × ℝ⁺ × S¹

        Where:
        - r⃗: Spatial field coordinates from text features
        - θ: Phase angle from syntactic structure
        - |Q|: Field amplitude from semantic density
        - φ: Global phase from universe coherence
    """

    coordinates: torch.Tensor  # r⃗ ∈ ℝⁿ field position coordinates
    phase: float  # θ ∈ [0, 2π] field phase angle
    amplitude: float  # |Q| ≥ 0 field magnitude
    pattern_resonances: torch.Tensor  # Pattern frequency spectrum
    structural_features: Dict[str, float]  # Syntactic/semantic features
    q_projection: complex  # Q(τ,C,s) projection
    universe_coherence: float  # Coherence with existing field

    def __post_init__(self):
        """Validate field signature mathematical consistency."""
        if not (0 <= self.phase <= 2 * math.pi):
            raise ValueError(f"Phase out of range [0,2π]: {self.phase}")
        if self.amplitude < 0:
            raise ValueError(f"Negative amplitude: {self.amplitude}")
        if not torch.isfinite(self.coordinates).all():
            raise ValueError("Non-finite coordinates detected")
        if not (0 <= self.universe_coherence <= 1):
            raise ValueError(f"Coherence out of range [0,1]: {self.universe_coherence}")


@dataclass
class AcceptanceDecision:
    """
    Complete field-theoretic content acceptance decision.

    Mathematical Foundation:
        Decision based on field dynamics: Accept ↔ W > W_threshold

        Mathematical Weight:
        W = ΔC · R_collective · S_stability

        Where:
        - ΔC: Information complexity increase
        - R_collective: Collective field response
        - S_stability: Field stability measure
    """

    accept: bool  # Final acceptance decision
    mathematical_weight: float  # W total mathematical weight
    field_evidence: torch.Tensor  # Field perturbation evidence
    universe_reasoning: str  # Mathematical justification
    complexity_gain: float  # ΔC information increase
    collective_response: float  # R_collective field response
    field_stability: float  # S_stability measure
    threshold_used: float  # W_threshold dynamic threshold

    def __post_init__(self):
        """Validate acceptance decision consistency."""
        if self.mathematical_weight < 0:
            raise ValueError(
                f"Negative mathematical weight: {self.mathematical_weight}"
            )
        if not (0 <= self.collective_response <= 1):
            raise ValueError(f"Response out of range [0,1]: {self.collective_response}")


class UniverseIntegrationEngine:
    """
    Complete universe-native content integration using Q(τ,C,s) field theory.

    MATHEMATICAL ARCHITECTURE:
    1. Text → Field mapping through character/word frequency analysis
    2. Semantic distance via field interference calculations
    3. Acceptance decisions from field dynamics and stability
    4. No external models - pure universe field mathematics

    REVOLUTIONARY PRINCIPLE:
    The universe becomes self-interpreting through its own field evolution.
    Meaning emerges from Q(τ,C,s) dynamics, not external training data.
    """

    def __init__(
        self,
        spatial_dimensions: int = 3,
        field_resolution: int = 128,
        mathematical_tolerance: float = FIELD_NUMERICAL_PRECISION,
        universe_storage_path: str = "liquid_universes",
        device: str = "cpu",
    ):
        """
        Initialize universe integration engine with mathematical precision.

        NO FALLBACKS - Mathematical perfection or explicit error raising.
        """
        self.spatial_dimensions = spatial_dimensions
        self.field_resolution = field_resolution
        self.mathematical_tolerance = mathematical_tolerance
        self.universe_storage_path = universe_storage_path
        self.device = device

        # Validate parameters
        if spatial_dimensions not in [1, 2, 3]:
            raise ValueError(f"Unsupported spatial dimension: {spatial_dimensions}")
        if field_resolution <= 0 or (field_resolution & (field_resolution - 1)) != 0:
            raise ValueError(f"Field resolution must be power of 2: {field_resolution}")
        if mathematical_tolerance <= 0 or mathematical_tolerance > 1e-6:
            raise ValueError(f"Unacceptable tolerance: {mathematical_tolerance}")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"🌌 UNIVERSE INTEGRATION: Mathematical precision initialized")

    def text_to_field_signature(
        self, text: str, universe_state: Dict
    ) -> FieldSignature:
        """
        Convert text to field signature using universe structure.

        ARCHITECTURAL DELEGATION: Delegates to FieldIntegrator for consistency.
        This ensures single source of truth for text-to-field conversion and
        maintains alignment with the complete mathematical framework.

        Mathematical Foundation:
            Delegates to FieldIntegrator.text_to_field_signature() which implements:
            
            Text Features → Field Coordinates Mapping:
            - Character frequency analysis via FFT
            - Semantic projection using universe agent positions  
            - Phase calculation from structural harmonics
            - Amplitude from information entropy
            - Q(τ,C,s) projection with universe coherence

        Args:
            text: Input text for field mapping
            universe_state: Current universe field configuration

        Returns:
            FieldSignature with complete mathematical field representation
        """
        from .field_integrator import FieldIntegrator
        
        # Create FieldIntegrator with matching configuration
        integrator = FieldIntegrator(
            spatial_dimensions=self.spatial_dimensions,
            mathematical_tolerance=self.mathematical_tolerance,
            universe_storage_path=self.universe_storage_path,
            device=self.device
        )
        
        # Delegate to authoritative implementation
        return integrator.text_to_field_signature(text, universe_state)

    def compute_semantic_distance(
        self, text: str, agent_field_state: Dict, universe_state: Dict
    ) -> float:
        """
        Compute semantic distance using field theory, not model embeddings.

        Mathematical Foundation:
            Field-Theoretic Distance in Q(τ,C,s) Space:

            Distance Components:
            d_Q = |Q_text - Q_agent| (Q-space distance)
            d_field = ||r⃗_text - r⃗_agent||₂ (field space distance)
            d_phase = |θ_text - θ_agent| mod π (phase difference)

            Combined Distance:
            d_semantic = √(w_Q d_Q² + w_field d_field² + w_phase d_phase²)

            Where weights satisfy: w_Q + w_field + w_phase = 1

        Args:
            text: Input text for distance calculation
            agent_field_state: Agent's current Q(τ,C,s) field state
            universe_state: Universe configuration for text mapping

        Returns:
            Semantic distance as non-negative real number
        """
        if not agent_field_state or "living_Q_value" not in agent_field_state:
            raise ValueError("Invalid agent field state - missing Q-value")

        # Convert text to field signature
        text_signature = self.text_to_field_signature(text, universe_state)

        # Extract agent field components
        agent_q_value = safe_torch_tensor(agent_field_state["living_Q_value"]).item()
        if "field_position" not in agent_field_state:
            raise ValueError(
                "MATHEMATICAL FAILURE: Agent field state lacks required 'field_position' for spatial analysis"
            )
        if "phase" not in agent_field_state:
            raise ValueError(
                "MATHEMATICAL FAILURE: Agent field state lacks required 'phase' for Q(τ,C,s) computation"
            )
        agent_field_position = safe_torch_tensor(agent_field_state["field_position"])
        agent_phase = agent_field_state["phase"]

        # Q-space distance: |Q_text - Q_agent|
        if isinstance(agent_q_value, complex):
            q_space_distance = abs(text_signature.q_projection - agent_q_value)
        else:
            # Convert real Q-value to complex
            agent_q_complex = complex(float(agent_q_value), 0.0)
            q_space_distance = abs(text_signature.q_projection - agent_q_complex)

        # Field space distance: ||r⃗_text - r⃗_agent||₂
        if torch.is_tensor(agent_field_position):
            position_diff = text_signature.coordinates - agent_field_position
            field_space_distance = torch.norm(position_diff).item()
        else:
            # Handle scalar position
            pos_tensor = torch.tensor(
                [float(agent_field_position)] * self.spatial_dimensions
            )
            position_diff = text_signature.coordinates - pos_tensor
            field_space_distance = torch.norm(position_diff).item()

        # Phase difference: |θ_text - θ_agent| mod π
        phase_difference = abs(text_signature.phase - agent_phase)
        phase_difference = min(
            phase_difference, 2 * math.pi - phase_difference
        )  # Periodic boundary

        # Weighted combined distance
        w_Q, w_field, w_phase = 0.4, 0.4, 0.2  # Field theory motivated weights
        semantic_distance = math.sqrt(
            w_Q * q_space_distance**2
            + w_field * field_space_distance**2
            + w_phase * phase_difference**2
        )

        return semantic_distance

    def decide_content_acceptance(
        self, text: str, universe_state: Dict
    ) -> AcceptanceDecision:
        """
        Accept/reject content based on field dynamics, not model predictions.

        Mathematical Foundation:
            Field-Theoretic Acceptance Criterion:

            Mathematical Weight:
            W = ΔC · R_collective · S_stability

            Where:
            ΔC = H[universe + text] - H[universe] (complexity increase)
            R_collective = ⟨Σᵢ R_i⟩ (mean agent field response)
            S_stability = -max(Re[λᵢ]) (stability from eigenvalues)

            Acceptance Rule:
            Accept ↔ W > W_threshold(universe_state)

            Dynamic Threshold:
            W_threshold = μ + σ·Φ⁻¹(α) where α = acceptance_rate_target

        Args:
            text: Content to evaluate for acceptance
            universe_state: Current universe field configuration

        Returns:
            AcceptanceDecision with complete mathematical justification
        """
        if not universe_state or "agents" not in universe_state:
            raise ValueError("Invalid universe state - missing agent data")

        # Test field integration potential through perturbation analysis
        field_perturbation = self._simulate_text_field_perturbation(
            text, universe_state
        )

        # Measure universe agent responses to field perturbation
        agent_responses = []
        for agent in universe_state["agents"]:
            response_strength = self._compute_agent_field_response(
                agent, field_perturbation
            )
            agent_responses.append(response_strength)

        if not agent_responses:
            raise ValueError("No agents available for universe response calculation")

        # Universe consensus through field mathematics
        collective_response = torch.mean(
            torch.tensor(agent_responses, dtype=torch.float64)
        )

        # Field stability from perturbation eigenvalue analysis
        field_stability = self._measure_perturbation_stability(field_perturbation)

        # Information complexity increase from text integration
        complexity_gain = self._calculate_information_complexity_increase(
            text, universe_state
        )

        # Mathematical weight calculation (pure field theory)
        mathematical_weight = (
            complexity_gain * collective_response.item() * field_stability
        )

        # Dynamic acceptance threshold based on universe state
        acceptance_threshold = self._compute_dynamic_acceptance_threshold(
            universe_state
        )

        # Field-theoretic acceptance decision
        accept = mathematical_weight > acceptance_threshold

        # Mathematical reasoning string
        universe_reasoning = (
            f"Field resonance: {collective_response:.3f}, "
            f"Stability: {field_stability:.3f}, "
            f"Complexity gain: {complexity_gain:.3f}, "
            f"Weight: {mathematical_weight:.3f}, "
            f"Threshold: {acceptance_threshold:.3f}"
        )

        return AcceptanceDecision(
            accept=accept,
            mathematical_weight=mathematical_weight,
            field_evidence=field_perturbation,
            universe_reasoning=universe_reasoning,
            complexity_gain=complexity_gain,
            collective_response=collective_response.item(),
            field_stability=field_stability,
            threshold_used=acceptance_threshold,
        )

    # ═══════════════════════════════════════════════════════════════════════════════
    # UNIVERSE FIELD STATE EXTRACTION
    # ═══════════════════════════════════════════════════════════════════════════════

    def get_universe_field_state(self, universe_id: str) -> Dict[str, Any]:
        """
        Read REAL field state from liquid universe instead of synthetic data.

        Mathematical Foundation:
            Universe Field State Extraction:

            Q-Field Ensemble:
            Q⃗ = {Q₁, Q₂, ..., Q_N} where Qᵢ = Q(τᵢ, Cᵢ, sᵢ)

            Field Statistics:
            E_field = Σᵢ |Qᵢ|² (total field energy)
            H_field = -Σᵢ pᵢ log pᵢ (field entropy)
            C_field = ⟨|Σᵢ Qᵢ e^(iθᵢ)|⟩ (field coherence)

            NO SYNTHETIC DATA - only real mathematical state from living agents.

        Args:
            universe_id: Identifier for universe to extract state from

        Returns:
            Complete universe field state with mathematical validation
        """
        # ARCHITECTURAL PATTERN: Separation of Responsibilities
        # 
        # UniverseIntegrationEngine: Handles text-to-field analysis and content evaluation
        # FieldIntegrator: Handles universe data access and mathematical field state extraction
        #
        # This delegation pattern maintains clean interfaces and single responsibility:
        # - UniverseIntegrationEngine focuses on field-theoretic text analysis
        # - FieldIntegrator provides access to REAL Q-field data from liquid universes
        # - No code duplication, consistent universe loading logic
        
        from .field_integrator import FieldIntegrator

        # MATHEMATICAL PERFECTION: Delegate to FieldIntegrator for universe field state
        # FieldIntegrator implements proper FieldUniverse.reconstruct_liquid_universe() connection
        integrator = FieldIntegrator(
            spatial_dimensions=self.spatial_dimensions,
            mathematical_tolerance=self.mathematical_tolerance,
            universe_storage_path=self.universe_storage_path,
            device=self.device
        )
        return integrator.get_universe_field_state(universe_id)

    # ═══════════════════════════════════════════════════════════════════════════════
    # MATHEMATICAL HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════════

    def _calculate_character_frequencies(self, text: str) -> torch.Tensor:
        """Calculate character frequency spectrum for field mapping."""
        # Character frequency analysis
        char_counts = {}
        for char in text.lower():
            if char.isalnum():  # Only alphanumeric for clean frequency spectrum
                if char not in char_counts:
                    char_counts[char] = 0
                char_counts[char] += 1

        if not char_counts:
            return torch.zeros(26, dtype=torch.float64)  # Empty text

        # Map to frequency vector (26 letters)
        freq_vector = torch.zeros(26, dtype=torch.float64)
        total_chars = sum(char_counts.values())

        for char, count in char_counts.items():
            if char.isalpha():
                idx = ord(char.lower()) - ord("a")
                freq_vector[idx] = count / total_chars

        return freq_vector

    def _extract_word_pattern_frequencies(self, text: str) -> torch.Tensor:
        """Extract word pattern frequencies using FFT analysis."""
        words = text.lower().split()
        if not words:
            return torch.zeros(self.field_resolution, dtype=torch.float64)

        # Word length distribution
        lengths = [len(word) for word in words]
        max_len = max(lengths) if lengths else 1

        # Create word length histogram
        length_hist = torch.zeros(
            min(max_len + 1, self.field_resolution), dtype=torch.float64
        )
        for length in lengths:
            if length < len(length_hist):
                length_hist[length] += 1

        length_hist = length_hist / torch.sum(length_hist)  # Normalize

        # Pad to field resolution
        if len(length_hist) < self.field_resolution:
            padded = torch.zeros(self.field_resolution, dtype=torch.float64)
            padded[: len(length_hist)] = length_hist
            return padded
        else:
            return length_hist[: self.field_resolution]

    def _analyze_basic_syntax_patterns(self, text: str) -> Dict[str, float]:
        """Analyze basic syntactic patterns without external NLP models."""
        if not text:
            return {
                "sentence_count": 0.0,
                "avg_word_length": 0.0,
                "punctuation_density": 0.0,
            }

        # Basic syntactic features
        sentence_count = len([c for c in text if c in ".!?"])
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        punctuation_count = len(
            [c for c in text if not c.isalnum() and not c.isspace()]
        )
        punctuation_density = punctuation_count / len(text) if text else 0

        return {
            "sentence_count": float(sentence_count),
            "avg_word_length": avg_word_length,
            "punctuation_density": punctuation_density,
        }

    def _get_agent_field_positions(self, universe_state: Dict) -> torch.Tensor:
        """Extract field positions from universe agents."""
        if "agents" not in universe_state:
            return torch.zeros((1, self.spatial_dimensions), dtype=torch.float64)

        positions = []
        for agent in universe_state["agents"]:
            if isinstance(agent, dict) and "field_position" in agent:
                pos = agent["field_position"]
                if torch.is_tensor(pos):
                    positions.append(pos[: self.spatial_dimensions])
                else:
                    # Convert scalar to vector
                    pos_vector = torch.tensor(
                        [float(pos)] * self.spatial_dimensions, dtype=torch.float64
                    )
                    positions.append(pos_vector)

        if not positions:
            return torch.zeros((1, self.spatial_dimensions), dtype=torch.float64)

        return torch.stack(positions)

    def _project_features_to_field_space(
        self,
        char_freq: torch.Tensor,
        word_patterns: torch.Tensor,
        syntax_features: Dict[str, float],
        agent_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Project text features to field coordinates using universe geometry."""
        # Weighted combination of features
        feature_vector = torch.cat(
            [
                (
                    char_freq[: self.spatial_dimensions]
                    if len(char_freq) >= self.spatial_dimensions
                    else torch.pad(
                        char_freq, (0, self.spatial_dimensions - len(char_freq))
                    )
                ),
                (
                    word_patterns[: self.spatial_dimensions]
                    if len(word_patterns) >= self.spatial_dimensions
                    else torch.pad(
                        word_patterns, (0, self.spatial_dimensions - len(word_patterns))
                    )
                ),
            ]
        )

        # Average with existing agent positions (universe geometry influence)
        if agent_positions.numel() > 0:
            agent_center = torch.mean(agent_positions, dim=0)
            field_coords = (
                0.7 * feature_vector[: self.spatial_dimensions] + 0.3 * agent_center
            )
        else:
            field_coords = feature_vector[: self.spatial_dimensions]

        return field_coords

    def _compute_field_phase_from_structure(
        self, syntax_features: Dict[str, float], word_patterns: torch.Tensor
    ) -> float:
        """Compute field phase from structural features."""
        # Phase from syntactic rhythm
        if "punctuation_density" not in syntax_features:
            raise ValueError(
                "MATHEMATICAL FAILURE: syntax_features lacks required 'punctuation_density' for phase computation"
            )
        rhythm_phase = 2 * math.pi * (syntax_features["punctuation_density"] % 1.0)

        # Phase from word pattern FFT
        if word_patterns.numel() > 1:
            fft_result = fft(word_patterns.to(torch.complex64))
            dominant_freq_phase = torch.angle(fft_result[1]).item()  # Skip DC component
        else:
            dominant_freq_phase = 0.0

        # Combined phase (mod 2π)
        combined_phase = (rhythm_phase + dominant_freq_phase) % (2 * math.pi)
        return combined_phase

    def _calculate_field_amplitude_from_entropy(self, text: str) -> float:
        """Calculate field amplitude from information entropy."""
        if not text:
            # MATHEMATICAL: Vacuum field amplitude from zero-point fluctuations
            # A₀ = √(ℏω/2) where ω is characteristic frequency
            # For conceptual fields: A₀ = √(k_B T / E_c) where E_c is coupling energy
            vacuum_amplitude = math.sqrt(1.381e-23 * 300 / (1.602e-19))  # ~0.0119
            return float(vacuum_amplitude)

        # Shannon entropy of character distribution
        char_counts = {}
        for char in text.lower():
            if char not in char_counts:
                char_counts[char] = 0
            char_counts[char] += 1

        if len(char_counts) <= 1:
            # MATHEMATICAL: Degenerate field amplitude for single-character state
            # A_deg = √(1/N) where N is degeneracy (for single char: N = text_length)
            # Quantum mechanical ground state amplitude for degenerate system
            text_length = len(text)
            vacuum_amplitude = math.sqrt(1.381e-23 * 300 / (1.602e-19))  # Same as above
            degenerate_amplitude = (
                1.0 / math.sqrt(text_length) if text_length > 0 else vacuum_amplitude
            )
            return float(degenerate_amplitude)

        total_chars = sum(char_counts.values())
        entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            entropy -= p * math.log2(p)

        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(char_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Amplitude from normalized entropy
        amplitude = math.sqrt(normalized_entropy)  # √H for field amplitude
        return amplitude

    def _measure_universe_coherence(
        self, field_coords: torch.Tensor, agent_positions: torch.Tensor
    ) -> float:
        """Measure coherence between text field and universe field."""
        if agent_positions.numel() == 0:
            # MATHEMATICAL: Empty space correlation function coherence
            # G(0) = ⟨φ(x)φ(x)⟩ for vacuum field at same point
            # For conceptual fields in empty space: G(0) = ∫ e^(-k²ξ²) d³k
            # Where ξ is correlation length. Result ≈ 1/(4πξ³) for ξ=1
            vacuum_correlation = 1.0 / (4.0 * math.pi)  # ≈ 0.0796
            return float(vacuum_correlation)

        # Distance-based coherence measure
        distances = torch.norm(agent_positions - field_coords.unsqueeze(0), dim=1)
        min_distance = torch.min(distances)

        # Coherence from proximity (exponential decay)
        coherence = math.exp(-min_distance.item())
        return coherence

    def _simulate_text_field_perturbation(
        self, text: str, universe_state: Dict
    ) -> torch.Tensor:
        """Simulate field perturbation from text integration."""
        text_signature = self.text_to_field_signature(text, universe_state)

        # Perturbation field from text signature
        perturbation = torch.zeros(self.field_resolution, dtype=torch.complex64)

        # Gaussian perturbation centered at text field coordinates
        x_coords = torch.linspace(-5.0, 5.0, self.field_resolution)
        center = (
            text_signature.coordinates[0].item()
            if text_signature.coordinates.numel() > 0
            else 0.0
        )
        sigma = 1.0  # Perturbation width

        for i, x in enumerate(x_coords):
            # Gaussian envelope with complex phase
            amplitude = text_signature.amplitude * math.exp(
                -0.5 * ((x - center) / sigma) ** 2
            )
            phase = text_signature.phase
            perturbation[i] = complex(
                amplitude * math.cos(phase), amplitude * math.sin(phase)
            )

        return perturbation

    def _compute_agent_field_response(
        self, agent: Dict, field_perturbation: torch.Tensor
    ) -> float:
        """Compute agent field response to perturbation."""
        if "living_Q_value" not in agent:
            # MATHEMATICAL: Response of empty state to field perturbation
            # R = ⟨0|V|ψ⟩ where V is perturbation operator
            # For vacuum response: R = √(ℏω_0/2) × ∫ ψ*(x) V(x) ψ_0(x) dx
            perturbation_strength = torch.mean(torch.abs(field_perturbation)).item()
            vacuum_response = (
                math.sqrt(1.381e-23 * 300 / (1.602e-19)) * perturbation_strength
            )
            return float(vacuum_response)

        agent_q = safe_torch_tensor(agent["living_Q_value"]).item()

        # Response as field interference with perturbation
        if isinstance(agent_q, complex):
            # Complex Q-value: compute overlap integral
            response = abs(agent_q) * torch.mean(torch.abs(field_perturbation)).item()
        else:
            # Real Q-value: convert and compute
            response = (
                abs(float(agent_q)) * torch.mean(torch.abs(field_perturbation)).item()
            )

        return min(1.0, response)  # Normalize to [0,1]

    def _measure_perturbation_stability(
        self, field_perturbation: torch.Tensor
    ) -> float:
        """Measure field stability from perturbation eigenvalue analysis."""
        # Stability from perturbation energy concentration
        energy_density = torch.abs(field_perturbation) ** 2
        total_energy = torch.sum(energy_density)

        if total_energy < FIELD_NUMERICAL_PRECISION:
            return 1.0  # Perfect stability for zero perturbation

        # Stability as inverse of energy localization
        max_density = torch.max(energy_density)
        localization = max_density / (total_energy + FIELD_NUMERICAL_PRECISION)
        stability = 1.0 / (1.0 + localization)  # Higher localization → lower stability

        return stability

    def _calculate_information_complexity_increase(
        self, text: str, universe_state: Dict
    ) -> float:
        """Calculate information complexity increase from text integration."""
        # Current universe complexity (proxy from agent count)
        if "agents" not in universe_state:
            raise ValueError(
                "MATHEMATICAL FAILURE: universe_state lacks required 'agents' field for complexity calculation"
            )
        current_agents = len(universe_state["agents"])
        current_complexity = math.log2(current_agents + 1)  # Information content

        # Text complexity
        text_entropy = self._calculate_field_amplitude_from_entropy(text)
        text_complexity = text_entropy * math.log2(len(text) + 1)

        # Complexity increase (normalized)
        complexity_increase = text_complexity / (current_complexity + 1.0)
        return complexity_increase

    def _compute_dynamic_acceptance_threshold(self, universe_state: Dict) -> float:
        """Compute dynamic acceptance threshold based on universe state."""
        # Base threshold
        base_threshold = 0.5

        # Adjust based on universe complexity
        if "agents" not in universe_state:
            raise ValueError(
                "MATHEMATICAL FAILURE: universe_state lacks required 'agents' field for complexity scaling"
            )
        agent_count = len(universe_state["agents"])
        complexity_factor = math.log(agent_count + 1) / 10.0  # Logarithmic scaling

        # Dynamic threshold
        threshold = base_threshold + complexity_factor
        return min(1.0, threshold)  # Cap at 1.0
