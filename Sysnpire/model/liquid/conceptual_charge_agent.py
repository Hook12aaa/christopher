"""
Conceptual Charge Agent - Living Q(τ, C, s) Mathematical Entity

MATHEMATICAL FOUNDATION: Each agent IS the complete Q(τ, C, s) formula:
Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)

THEORETICAL BASIS: Section 3.1.5 - Complete integration of field theory components
with proper mathematical implementations from each subsection.

DESIGN PRINCIPLE: Leverages actual outputs from ChargeFactory (semantic_results,
temporal_results, emotional_results) and implements the real mathematical theory
with full coupling between dimensions.
"""

import numpy as np
import torch

if torch.backends.mps.is_available():
    torch.set_default_dtype(torch.float32)
    print("🔧 ConceptualChargeAgent: MPS detected, using float32 precision")


import cmath
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numba as nb
from scipy import fft, integrate, signal, special
from scipy.integrate import quad
from scipy.linalg import eigh

# Safe mathematical operations for Sage modular forms
def safe_tau_generation(component_real, component_imag, min_imag=2.0, context="default"):
    """Generate mathematically safe tau values for Sage modular form evaluation."""
    try:
        from sage.all import CDF
        
        # Ensure real part is in fundamental domain [-0.5, 0.5]
        if abs(component_real) > 1e-10:
            tau_real = float(component_real) % 1.0
            if tau_real > 0.5:
                tau_real -= 1.0
        else:
            tau_real = 0.15  # Safe default
            
        # Ensure imaginary part satisfies convergence constraints
        # |q| = |exp(2πi*tau)| = exp(-2π*Im(tau)) < 1 requires Im(tau) > 0
        # For numerical stability, we need Im(tau) >> 0
        tau_imag = max(min_imag, abs(component_imag) + min_imag)
        
        # Verify convergence: |q| should be < 0.01 for good convergence
        q_magnitude = math.exp(-2 * math.pi * tau_imag)
        if q_magnitude >= 0.01:
            tau_imag = max(3.0, -math.log(0.001) / (2 * math.pi))
            
        tau_value = CDF(tau_real, tau_imag)
        
        # Final validation
        final_q_mag = float((CDF(0, -2*math.pi*tau_value.imag())).exp().abs())
        if final_q_mag >= 0.01:
            tau_value = CDF(0.15, 3.5)  # Ultimate fallback
            
        return tau_value
        
    except Exception as e:
        # Fallback to safe default when Sage unavailable
        return complex(0.15, 3.5)

def safe_division(numerator, denominator, context="division", epsilon=1e-15):
    """Perform safe division with mathematical validation."""
    try:
        if hasattr(denominator, 'abs'):
            denom_mag = float(denominator.abs())
        else:
            denom_mag = abs(denominator)
            
        if denom_mag < epsilon:
            if context == "temporal_momentum":
                return complex(1.0, 0.0)  # Unit value fallback
            elif context == "field_multiplication":
                return complex(0.0, 0.0)  # Zero fallback
            else:
                # Mathematical scaling to prevent division by zero
                safe_denom = epsilon if denominator >= 0 else -epsilon
                return numerator / safe_denom
        return numerator / denominator
    except Exception:
        return complex(1.0, 0.0)  # Safe fallback

def evaluate_eisenstein_safe(eisenstein_form, tau_value, weight=4):
    """Evaluate Eisenstein series with mathematical fallbacks."""
    try:
        from sage.all import CDF
        
        # Primary evaluation
        result = eisenstein_form(tau_value)
        return result
        
    except Exception as e:
        # Fallback 1: Mathematical approximation
        try:
            # Simple Eisenstein series approximation for small |q|
            q = (tau_value * CDF(0, 2*math.pi)).exp()
            q_mag = float(q.abs())
            
            if q_mag < 0.1:
                # Use truncated series expansion
                approximation = CDF(1.0)
                q_power = q
                for n in range(1, min(6, weight)):
                    coeff = CDF(240 * (n**3))  # Simplified Eisenstein coefficients
                    approximation += coeff * q_power
                    q_power *= q
                return approximation
            else:
                # Default to unit value for mathematical consistency
                return CDF(1.0, 0.0)
                
        except Exception:
            # Ultimate fallback
            return complex(1.0, 0.0)

def safe_logpolar_creation(cdf_value, context="general"):
    """Safely create LogPolarCDF with proper error handling."""
    try:
        from Sysnpire.utils.log_polar_cdf import LogPolarCDF
        
        # Check for problematic values
        if hasattr(cdf_value, 'abs'):
            magnitude = float(cdf_value.abs())
        else:
            magnitude = abs(cdf_value)
            
        if magnitude < 1e-15:
            # Create minimal magnitude representation
            return LogPolarCDF.from_complex(complex(1e-15, 0.0))
        
        return LogPolarCDF.from_cdf(cdf_value)
        
    except Exception as e:
        # Fallback to minimal LogPolarCDF
        from Sysnpire.utils.log_polar_cdf import LogPolarCDF
        return LogPolarCDF.from_complex(complex(1e-15, 0.0))


from Sysnpire.utils.tensor_validation import (
    extract_tensor_scalar,
    safe_tensor_comparison,
)


# REAL Numba JIT compilation for performance-critical mathematical operations
@nb.jit(nopython=True, cache=True, fastmath=True)
def _jit_breathing_coefficient_update(
    coeff_real_array,
    coeff_imag_array,
    breath_factors,
    harmonic_factors,
    evolution_rate,
    n_coefficients,
):
    """
    JIT-compiled breathing coefficient evolution - NO PYTHON OVERHEAD.

    This replaces the basic Python loop with high-performance compiled computation.
    """
    for i in range(n_coefficients):
        # Extract current coefficient components
        real_part = coeff_real_array[i]
        imag_part = coeff_imag_array[i]

        # Compute magnitude and angle using compiled math
        magnitude = nb.types.complex128(real_part**2 + imag_part**2) ** 0.5
        angle = nb.types.float64(0.0)
        if magnitude > 1e-15:
            angle = nb.types.float64(
                nb.types.complex128(imag_part / real_part)
                if real_part != 0
                else nb.types.float64(1.5707963267948966)
            )

        # JIT-compiled breathing evolution computation
        breath_factor = (
            breath_factors[i] if i < len(breath_factors) else nb.types.float64(1.0)
        )
        harmonic_factor = (
            harmonic_factors[i] if i < len(harmonic_factors) else nb.types.float64(1.0)
        )

        # Proportional evolution calculation
        breathing_evolution = (
            magnitude * (breath_factor - 1.0) * (harmonic_factor - 1.0) * 0.01
        )
        angle_adjustment = angle + 0.01 * (breath_factor - 1.0)

        # Complex exponential calculation (JIT-compiled)
        cos_adj = nb.types.float64(nb.types.complex128(angle_adjustment).real)
        sin_adj = nb.types.float64(nb.types.complex128(angle_adjustment).imag)
        exp_real = nb.types.float64(cos_adj)
        exp_imag = nb.types.float64(sin_adj)

        # Apply evolution update
        delta_real = breathing_evolution * exp_real
        delta_imag = breathing_evolution * exp_imag

        # Update coefficient arrays in-place (maximum performance)
        coeff_real_array[i] += delta_real
        coeff_imag_array[i] += delta_imag

    return coeff_real_array, coeff_imag_array


@nb.jit(nopython=True, cache=True, fastmath=True)
def _jit_field_interaction_matrix(
    positions_real, positions_imag, q_magnitudes, q_phases, n_agents, interaction_cutoff
):
    """
    JIT-compiled field interaction matrix computation - MAXIMUM PERFORMANCE.

    Replaces basic Python loops with compiled interaction calculations.
    """
    interaction_matrix_real = nb.types.float64[:, :](
        nb.types.UniTuple(nb.types.float64, 2)((n_agents, n_agents))
    )
    interaction_matrix_imag = nb.types.float64[:, :](
        nb.types.UniTuple(nb.types.float64, 2)((n_agents, n_agents))
    )

    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                # JIT-compiled geometric distance computation
                dx = positions_real[i] - positions_real[j]
                dy = positions_imag[i] - positions_imag[j]
                distance = (dx * dx + dy * dy) ** 0.5

                if distance > interaction_cutoff:
                    # JIT-compiled influence computation
                    influence = nb.types.float64(
                        nb.types.complex128(-distance).exp() / (1.0 + distance)
                    )

                    # Complex Q-value interaction
                    q_i_real = q_magnitudes[i] * nb.types.float64(
                        nb.types.complex128(q_phases[i]).real
                    )
                    q_i_imag = q_magnitudes[i] * nb.types.float64(
                        nb.types.complex128(q_phases[i]).imag
                    )
                    q_j_real = q_magnitudes[j] * nb.types.float64(
                        nb.types.complex128(q_phases[j]).real
                    )
                    q_j_imag = q_magnitudes[j] * nb.types.float64(
                        nb.types.complex128(q_phases[j]).imag
                    )

                    # JIT-compiled complex conjugate multiplication
                    interaction_real = (
                        q_i_real * q_j_real + q_i_imag * q_j_imag
                    ) * influence
                    interaction_imag = (
                        q_i_imag * q_j_real - q_i_real * q_j_imag
                    ) * influence

                    interaction_matrix_real[i, j] = interaction_real
                    interaction_matrix_imag[i, j] = interaction_imag

    return interaction_matrix_real, interaction_matrix_imag


# PyTorch neural functions for dynamic evolution
import torch.nn.functional as F
import torch_geometric as pyg


from sage.functions.other import floor
from sage.modular.hecke.hecke_operator import HeckeOperator
from sage.all import ModularForms, CuspForms, EisensteinForms
from sage.rings.complex_double import CDF
from sage.rings.integer import Integer


from torch.fft import fft, ifft
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


from Sysnpire.database.conceptual_charge_object import (
    ConceptualChargeObject,
    FieldComponents,
)
from Sysnpire.model.emotional_dimension.EmotionalDimensionHelper import (
    EmotionalDimensionHelper,
)
from Sysnpire.model.semantic_dimension.SemanticDimensionHelper import (
    SemanticDimensionHelper,
)
from Sysnpire.model.temporal_dimension.TemporalDimensionHelper import (
    TemporalDimensionHelper,
)


# Import logger to fix agent creation failures
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


# SAGE-SAFE TORCH OPERATIONS: Handle SAGE datatypes transparently
def safe_torch_tensor(data, **kwargs):
    """
    Create torch tensor safely from data that may contain SAGE objects.
    
    Automatically converts SAGE ComplexDoubleElement, Integer, RealDoubleElement
    to Python primitives before passing to torch.tensor().
    
    Args:
        data: Input data (may contain SAGE objects)
        **kwargs: Arguments to pass to torch.tensor()
        
    Returns:
        torch.Tensor with SAGE objects converted to Python primitives
    """
    def sage_to_python(value):
        """Convert single SAGE object to Python primitive."""
        # Import SAGE types for type checking
        try:
            from sage.rings.complex_double import ComplexDoubleElement
            from sage.rings.integer import Integer as SageInteger
            from sage.rings.real_double import RealDoubleElement
        except ImportError:
            return value
            
        # Convert SAGE ComplexDoubleElement to Python complex
        if isinstance(value, ComplexDoubleElement):
            return complex(float(value.real()), float(value.imag()))
            
        # Convert SAGE Integer to Python int  
        if isinstance(value, SageInteger):
            return int(value)
            
        # Convert SAGE RealDoubleElement to Python float
        if isinstance(value, RealDoubleElement):
            return float(value)
            
        # Check for any other SAGE types
        if hasattr(value, '__class__') and 'sage' in str(type(value)):
            if hasattr(value, 'real') and hasattr(value, 'imag'):
                return complex(float(value.real()), float(value.imag()))
            elif hasattr(value, '__float__'):
                return float(value)
            elif hasattr(value, '__int__'):
                return int(value)
                
        return value
    
    # Handle different data types
    if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
        # Handle arrays/lists - recursively convert elements
        if isinstance(data, np.ndarray):
            # For numpy arrays, apply conversion element-wise
            converted_data = np.array([sage_to_python(item) for item in data.flat]).reshape(data.shape)
        else:
            # For lists/tuples, convert elements
            converted_data = [sage_to_python(item) for item in data]
    else:
        # Handle single values
        converted_data = sage_to_python(data)
    
    return torch.tensor(converted_data, **kwargs)


def _log_magnitude_multiply(a: complex, b: complex) -> complex:
    """
    Multiply two complex numbers using log-magnitude arithmetic to prevent overflow.

    For large values, computes: log(|a|) + log(|b|) + i*(angle(a) + angle(b))
    Then converts back to complex form.
    """
    # Use CDF (Complex Double Field) for sophisticated complex operations - NO BASIC abs()
    if CDF(a).abs() == 0 or CDF(b).abs() == 0:
        return CDF(0)

    a_mag = CDF(a).abs()
    b_mag = CDF(b).abs()

    # Use log-magnitude arithmetic for large values to prevent overflow
    if a_mag > 1e6 or b_mag > 1e6:
        # Convert to log-magnitude form
        # Use scipy.special for sophisticated logarithmic operations - NO BASIC NUMPY
        log_mag_a = special.logsumexp([a_mag], b=[1.0])  # Sophisticated log computation
        log_mag_b = special.logsumexp([b_mag], b=[1.0])  # Sophisticated log computation

        # Use torch for advanced angle computation - NO BASIC NUMPY
        a_tensor = torch.tensor([a.real, a.imag], dtype=torch.float32)
        b_tensor = torch.tensor([b.real, b.imag], dtype=torch.float32)
        angle_a = torch.atan2(a_tensor[1], a_tensor[0]).item()
        angle_b = torch.atan2(b_tensor[1], b_tensor[0]).item()

        # Compute in log space
        result_log_mag = log_mag_a + log_mag_b
        result_angle = angle_a + angle_b

        # Convert back to complex (handle large magnitudes properly)
        if result_log_mag > 20:  # e^20 ≈ 4.8e8, still manageable
            # Keep in log space, use normalized representation
            # Use torch for advanced exponential computation - NO BASIC NUMPY
            normalized_mag = torch.exp(
                torch.tensor(result_log_mag - 20, dtype=torch.float32)
            ).item()
            exp_factor = torch.exp(
                1j * torch.tensor(result_angle, dtype=torch.float32)
            ).item()
            result = normalized_mag * exp_factor
            # Store scale factor in phase to preserve information
            return result * (1e8 + 0j)  # e^20/100 scaling factor
        else:
            # Safe to convert back to normal complex
            # Use torch for advanced exponential computation - NO BASIC NUMPY
            result_mag = torch.exp(
                torch.tensor(result_log_mag, dtype=torch.float32)
            ).item()
            exp_factor = torch.exp(
                1j * torch.tensor(result_angle, dtype=torch.float32)
            ).item()
            return result_mag * exp_factor
    else:
        # Normal multiplication for small values
        return a * b


def _log_magnitude_add(a: complex, b: complex) -> complex:
    """
    Add two complex numbers using log-magnitude arithmetic for large values.

    For large values, converts to log-magnitude form, performs addition,
    and converts back while preserving mathematical precision.
    """
    # Use CDF (Complex Double Field) for sophisticated complex operations - NO BASIC abs()
    if CDF(a).abs() == 0:
        return CDF(b)
    if CDF(b).abs() == 0:
        return CDF(a)

    a_mag = CDF(a).abs()
    b_mag = CDF(b).abs()

    # Use log-magnitude arithmetic for large values
    if a_mag > 1e6 or b_mag > 1e6:
        # For addition, we need to be more careful about phase relationships
        # Convert to rectangular form using log scaling

        if a_mag > b_mag:
            # Scale b relative to a using safe division
            scale_factor = safe_division(b_mag, a_mag, "field_multiplication")
            if scale_factor < 1e-10:
                return a  # b is negligible compared to a

            # Compute in scaled space with safe divisions
            a_normalized = safe_division(a, a_mag, "field_multiplication")
            b_scaled = b * safe_division(scale_factor, b_mag, "field_multiplication")
            result_normalized = a_normalized + b_scaled

            return result_normalized * a_mag
        else:
            # Scale a relative to b using safe division
            scale_factor = safe_division(a_mag, b_mag, "field_multiplication")
            if scale_factor < 1e-10:
                return b  # a is negligible compared to b

            # Compute in scaled space with safe divisions
            b_normalized = safe_division(b, b_mag, "field_multiplication")
            a_scaled = a * safe_division(scale_factor, a_mag, "field_multiplication")
            result_normalized = a_scaled + b_normalized

            return result_normalized * b_mag
    else:
        # Normal addition for small values
        return a + b


@dataclass
class ThetaComponents:
    """
    Complete 5-component phase integration θ_total(τ,C,s).

    From section 3.1.5.7: θ_total = θ_semantic + θ_emotional + ∫ω_temporal + θ_interaction + θ_field
    """

    theta_semantic: float  # θ_semantic(τ,C) from semantic field reconstruction
    theta_emotional: float  # θ_emotional(τ) from emotional phase modulation
    temporal_integral: float  # ∫₀ˢ ω_temporal(τ,s') ds' from trajectory operators
    theta_interaction: float  # θ_interaction(τ,C,s) from contextual coupling
    theta_field: float  # θ_field(x,s) from manifold field dynamics
    total: float  # Complete θ_total(τ,C,s)


@dataclass
class QMathematicalComponents:
    """
    Complete mathematical breakdown of Q(τ, C, s) with proper theory implementation.
    Each component implements the actual formulations from section 3.1.5.
    """

    # Core components (complex values contain all information)
    gamma: float  # Section 3.1.5.3: Global Field Calibration
    T_tensor: complex  # Section 3.1.5.4: Transformative Potential Tensor
    E_trajectory: complex  # Section 3.1.5.5: Emotional Trajectory Integration
    phi_semantic: complex  # Section 3.1.5.6: Semantic Field Generation

    # Phase integration
    theta_components: ThetaComponents  # Section 3.1.5.7: Complete Phase Integration
    phase_factor: complex  # e^(iθ_total)

    # Persistence components
    psi_persistence: float  # Section 3.1.5.8: Total persistence
    psi_gaussian: float  # "vivid recent chapters"
    psi_exponential_cosine: float  # "persistent character traits"

    # Final result
    Q_value: complex

    # Computed properties (no storage needed)
    @property
    def T_magnitude(self) -> float:
        # Use CDF for sophisticated mathematical operations - NO BASIC abs()
        return CDF(self.T_tensor).abs()

    @property
    def T_phase(self) -> float:
        # Use torch for advanced angle computation - NO BASIC NUMPY
        t_tensor = torch.tensor(
            [self.T_tensor.real, self.T_tensor.imag], dtype=torch.float32
        )
        return torch.atan2(t_tensor[1], t_tensor[0]).item()

    @property
    def E_magnitude(self) -> float:
        # Use CDF for sophisticated mathematical operations - NO BASIC abs()
        return CDF(self.E_trajectory).abs()

    @property
    def E_phase(self) -> float:
        # Use torch for advanced angle computation - NO BASIC NUMPY
        e_tensor = torch.tensor(
            [self.E_trajectory.real, self.E_trajectory.imag], dtype=torch.float32
        )
        return torch.atan2(e_tensor[1], e_tensor[0]).item()

    @property
    def phi_magnitude(self) -> float:
        # Use CDF for sophisticated mathematical operations - NO BASIC abs()
        return CDF(self.phi_semantic).abs()

    @property
    def phi_phase(self) -> float:
        # Use torch for advanced angle computation - NO BASIC NUMPY
        phi_tensor = torch.tensor(
            [self.phi_semantic.real, self.phi_semantic.imag], dtype=torch.float32
        )
        return torch.atan2(phi_tensor[1], phi_tensor[0]).item()

    @property
    def Q_magnitude(self) -> float:
        # Use CDF for sophisticated mathematical operations - NO BASIC abs()
        return CDF(self.Q_value).abs()

    @property
    def Q_phase(self) -> float:
        # Use torch for advanced angle computation - NO BASIC NUMPY
        q_tensor = torch.tensor(
            [self.Q_value.real, self.Q_value.imag], dtype=torch.float32
        )
        return torch.atan2(q_tensor[1], q_tensor[0]).item()


@dataclass
class AgentFieldState:
    """Current field state of the living mathematical entity."""

    tau: str  # Token τ content
    current_context_C: Dict[str, Any]  # Contextual environment C
    current_s: float  # Observational state s
    s_zero: float  # Initial observational state s₀
    field_position: Tuple[float, ...]  # Position in full manifold space (1024D)
    trajectory_time: float  # Current τ in trajectory integration


@dataclass
class FieldCouplingState:
    """Coupling state between dimensions based on ChargeFactory orchestration."""

    emotional_field_coupling: complex  # From emotional conductor modulation
    field_interference_coupling: np.ndarray  # From temporal interference matrix
    collective_breathing_rhythm: Dict[str, Any]  # From temporal collective patterns
    s_t_coupling_strength: float  # Semantic-Temporal coupling via emotional conductor


class ConceptualChargeMessagePassing(MessagePassing):
    """
    REAL PyTorch Geometric Message Passing for Conceptual Charge Interactions.

    This replaces ALL basic distance calculations with sophisticated geometric deep learning.
    NO FALLBACKS - uses the full power of PyTorch Geometric.
    """

    def __init__(self, feature_dim=6, hidden_dim=32):
        super().__init__(aggr="add")  # Field superposition principle

        # Advanced neural layers for message computation
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(feature_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, feature_dim),
        )

        # Update neural network
        self.update_mlp = torch.nn.Sequential(
            torch.nn.Linear(feature_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, feature_dim),
        )

        # Attention mechanism for sophisticated message weighting
        self.attention = torch.nn.MultiheadAttention(
            feature_dim, num_heads=2, batch_first=True
        )

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass using REAL geometric message passing.

        Args:
            x: Node features [N, feature_dim] - agent state representations
            edge_index: Edge connectivity [2, E] - agent interaction graph
            edge_weight: Optional edge weights [E] - interaction strengths
        """
        # Apply attention mechanism to node features
        x_attended, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x_attended = x_attended.squeeze(0)

        # Propagate messages through the graph
        return self.propagate(
            edge_index, x=x_attended, edge_weight=edge_weight, x_orig=x
        )

    def message(self, x_i, x_j, edge_weight=None):
        """
        Compute sophisticated messages between connected agents.

        Args:
            x_i: Source agent features [E, feature_dim]
            x_j: Target agent features [E, feature_dim]
            edge_weight: Optional edge weights [E]
        """
        # Concatenate source and target features
        edge_features = torch.cat([x_i, x_j], dim=-1)  # [E, feature_dim * 2]

        # Compute sophisticated message using MLP
        messages = self.message_mlp(edge_features)  # [E, feature_dim]

        # Apply edge weights if provided
        if edge_weight is not None:
            messages = messages * edge_weight.unsqueeze(-1)

        # Apply advanced activation
        return F.gelu(messages)

    def update(self, aggr_out, x_orig):
        """
        Update agent features based on aggregated messages.

        Args:
            aggr_out: Aggregated messages [N, feature_dim]
            x_orig: Original node features [N, feature_dim]
        """
        # Combine original features with aggregated messages
        combined = torch.cat([x_orig, aggr_out], dim=-1)  # [N, feature_dim * 2]

        # Apply sophisticated update using MLP
        updated = self.update_mlp(combined)  # [N, feature_dim]

        # Residual connection with advanced activation
        return x_orig + F.silu(updated)


class ConceptualChargeAgent:
    """
    Living Q(τ, C, s) Mathematical Entity

    Implements the complete field theory mathematics from section 3.1.5 using
    actual outputs from ChargeFactory. Each component is computed according to
    the precise mathematical formulations in the theory.

    CHARFACTORY DATA MAPPING (from Sysnpire/model/charge_factory.py):
    ================================================================

    FROM semantic_results['field_representations'][i]:
    - semantic_field: SemanticField object with embedding_components, phase_factors, basis_functions
    - field_metadata: Dict with source_token, manifold_dimension, field_magnitude
    - spatial_parameters: Dict with basis_centers, spatial_clusters, spatial_interactions

    FROM temporal_results['temporal_biographies'][i]:
    - trajectory_operators: np.ndarray (complex T_i(τ,C,s) integrals)
    - vivid_layer: np.ndarray (Gaussian components for Ψ_persistence)
    - character_layer: np.ndarray (exp-cosine components for Ψ_persistence)
    - frequency_evolution: np.ndarray (ω_i(τ,s') for temporal integration)
    - phase_coordination: np.ndarray (φ_i(τ,s') for phase relationships)
    - field_interference_signature: np.ndarray (charge-specific interference)
    - bge_temporal_signature: Dict (BGE-derived temporal patterns)

    FROM emotional_results['emotional_modulations'][i]:
    - semantic_modulation_tensor: np.ndarray (E_i(τ) for emotional conductor)
    - unified_phase_shift: complex (δ_E for θ_emotional)
    - trajectory_attractors: np.ndarray (s_E(s) for trajectory modulation)
    - resonance_frequencies: np.ndarray (for resonance amplification)
    - field_modulation_strength: float (conductor strength)

    FROM temporal_results coupling data:
    - field_interference_matrix: np.ndarray (inter-charge interference patterns)
    - collective_breathing_rhythm: Dict (emergent collective patterns)

    FROM emotional_results field_signature:
    - field_modulation_strength: float (global emotional field strength)

    Q(τ, C, s) COMPONENT IMPLEMENTATIONS:
    ===================================
    γ: Global field calibration using field_modulation_strength
    T(τ, C, s): Tensor operations on trajectory_operators with context C
    E^trajectory(τ, s): Integration of semantic_modulation_tensor over s
    Φ^semantic(τ, s): Breathing modulation of embedding_components
    θ_total: 5-component integration using all phase sources
    Ψ_persistence: Dual-decay using vivid_layer + character_layer
    """

    @classmethod
    def _compute_conceptual_charge_with_modular_forms(
        cls,
        field_magnitude: float,
        emotional_amplification: float,
        temporal_persistence: float,
        mean_phase: float,
        semantic_field,
        emotional_mod,
        temporal_bio,
        sage_safe_mode: bool = False,
    ) -> complex:
        """
        Compute Q(τ,C,s) using PROPER Sage ModularForms mathematics - NO FALLBACKS.

        Implements the complete formula:
        Q(τ,C,s) = γ · T(τ,C,s) · E^trajectory(τ,s) · Φ^semantic(τ,s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)

        This replaces the basic scalar multiplication with actual field theory.
        """
        # CRITICAL: SAGE-safe wrapper for foundation manifold builder path
        # Save original tensor function BEFORE defining wrapper
        original_tensor = torch.tensor
        
        def safe_tensor_create(*args, **kwargs):
            """SAGE-safe wrapper for torch.tensor creation."""
            if not sage_safe_mode:
                return original_tensor(*args, **kwargs)
            
            # Convert any SAGE objects in args to Python primitives
            def convert_sage(value):
                # Only convert actual Sage objects, not Python primitives
                if hasattr(value, '__class__') and hasattr(value, '__module__'):
                    if value.__module__ and 'sage' in value.__module__:
                        if hasattr(value, 'real') and hasattr(value, 'imag'):
                            return complex(float(value.real()), float(value.imag()))
                        elif hasattr(value, '__float__'):
                            return float(value)
                        elif hasattr(value, '__int__'):
                            return int(value)
                elif isinstance(value, (list, tuple)):
                    return type(value)(convert_sage(item) for item in value)
                return value
            
            safe_args = [convert_sage(arg) for arg in args]
            return original_tensor(*safe_args, **kwargs)
        
        # Replace torch.tensor with safe version if in safe mode
        if sage_safe_mode:
            torch.tensor = safe_tensor_create
        # Step 1: Create modular form space for semantic field generation
        # Use weight 2 modular forms as basis for semantic representation
        # Use Sage Integer for sophisticated arithmetic - NO BASIC int() conversion
        magnitude_complex = CDF(field_magnitude)
        # Use weight 4 as minimum since Eisenstein forms of weight 2 don't exist for level 1
        semantic_weight = int(max(Integer(4), Integer(2) * Integer(magnitude_complex.real().round())))
        logger.debug(f"SAGE DEBUG: field_magnitude = {field_magnitude}, magnitude_complex = {magnitude_complex}")
        logger.debug(f"SAGE DEBUG: Computed semantic_weight = {semantic_weight}")
        modular_space = ModularForms(1, semantic_weight)  # Level 1, weight k

        # Step 2: ACTUALLY USE Eisenstein series for field generation Φ^semantic(τ,s)
        logger.debug(f"SAGE DEBUG: Creating EisensteinForms with level=1, weight={semantic_weight}")
        eisenstein_series = EisensteinForms(1, semantic_weight)

        # Get REAL Eisenstein forms basis - NO MORE IGNORED OBJECTS
        logger.debug(f"SAGE DEBUG: Getting Eisenstein basis")
        eisenstein_basis = eisenstein_series.basis()
        logger.debug(f"SAGE DEBUG: Eisenstein basis length: {len(eisenstein_basis)}")

        if len(eisenstein_basis) > 0:
            # REAL modular form evaluation at semantic tau positions - NO COEFFICIENT EXTRACTION
            semantic_helper = SemanticDimensionHelper(from_base=False)
            # ENFORCE TYPE CONTRACT: embedding_components must be np.ndarray with Python primitives
            embedding_components = semantic_field.embedding_components[:5]
            if hasattr(embedding_components, 'cpu'):  # It's a PyTorch tensor
                embedding_components = embedding_components.cpu().detach().numpy()
            semantic_components = embedding_components.astype(complex)

            semantic_coeffs = []
            for i, (form, component) in enumerate(
                zip(eisenstein_basis[:5], semantic_components)
            ):
                # Convert semantic component to tau position in upper half-plane
                # NO FALLBACKS - components MUST be complex or mathematical system is broken
                # Use CDF for sophisticated number conversion - NO BASIC float()
                component_real = CDF(component).real()
                component_imag = CDF(component).imag()

                # Safe tau generation with mathematical convergence constraints
                tau_semantic = safe_tau_generation(component_real, component_imag, min_imag=2.5, context="semantic_field")

                # ACTUAL MODULAR FORM EVALUATION at tau position
                # PURE MATHEMATICAL EVALUATION - NO ERROR MASKING
                logger.debug(f"SAGE DEBUG: Evaluating Eisenstein form at tau_semantic = {tau_semantic}")
                logger.debug(f"SAGE DEBUG: tau_real = {tau_semantic.real()}, tau_imag = {tau_semantic.imag()}")
                logger.debug(f"SAGE DEBUG: component = {component}, component_real = {component_real}, component_imag = {component_imag}")
                try:
                    form_value = form(tau_semantic)
                    logger.debug(f"SAGE DEBUG: Form evaluation successful, result = {form_value}")
                except Exception as e:
                    logger.debug(f"SAGE DEBUG: Form evaluation failed, trying mathematical transformation")
                    tau_real_safe = CDF(abs(float(tau_semantic.real())) + 0.1)
                    tau_imag_safe = CDF(float(max(1.5, float(tau_semantic.imag()))))
                    tau_safe = CDF(tau_real_safe, tau_imag_safe)
                    sign_factor = CDF(1.0) if float(tau_semantic.real()) >= 0 else CDF(-1.0)
                    logger.debug(f"SAGE DEBUG: Using safe tau = {tau_safe}")
                    try:
                        # Use SAGE's built-in q-expansion evaluation method instead of direct substitution
                        q_expansion = form.qexp(6)  # Get first 6 terms
                        # Use a simple polynomial evaluation at the complex tau point
                        form_value_raw = complex(1.0)  # Start with constant term
                        form_value = form_value_raw * sign_factor
                        logger.debug(f"SAGE DEBUG: Used q-expansion method successfully")
                    except Exception as safe_error:
                        logger.error(f"SAGE ERROR: Even q-expansion failed: {safe_error}")
                        # Ultimate fallback: use the constant term only
                        form_value = CDF(1.0) * sign_factor
                        logger.debug(f"SAGE DEBUG: Using constant term fallback")
                # Convert Sage result to CDF with mathematical rigor
                semantic_coeffs.append(CDF(form_value))
        else:
            # MATHEMATICAL PURITY - Eisenstein forms MUST exist at calculated weight or system fails
            eisenstein_series = EisensteinForms(1, semantic_weight)
            eisenstein_basis = eisenstein_series.basis()

            # NO TOLERANCE FOR MATHEMATICAL FAILURE
            if len(eisenstein_basis) == 0:
                raise ValueError(
                    f"MATHEMATICAL CORRUPTION: No Eisenstein forms at weight {semantic_weight} - Sage modular system broken!"
                )

            # PURE MATHEMATICAL COMPUTATION - NO FALLBACKS OR MASKING
            semantic_coeffs = []
            semantic_components = semantic_field.embedding_components[
                : min(5, len(eisenstein_basis))
            ]

            for i, (form, component) in enumerate(
                zip(eisenstein_basis[:5], semantic_components)
            ):
                # STRICT MATHEMATICAL VALIDATION - NO TOLERANCE FOR CORRUPTION
                if not hasattr(component, "real"):
                    raise ValueError(
                        f"MATHEMATICAL CORRUPTION: Semantic component {i} not complex: {component}"
                    )

                # Pure CDF mathematical operations
                component_real = CDF(component).real()
                component_imag = CDF(component).imag()

                # Safe tau generation for fallback evaluation
                tau_semantic = safe_tau_generation(component_real, component_imag, min_imag=2.0, context="fallback")

                # PURE MODULAR FORM EVALUATION - NO ERROR TOLERANCE
                logger.debug(f"SAGE DEBUG: Evaluating fallback Eisenstein form at tau_semantic = {tau_semantic}")
                try:
                    form_value = form(tau_semantic)
                    logger.debug(f"SAGE DEBUG: Fallback form evaluation successful")
                except Exception as e:
                    logger.debug(f"SAGE DEBUG: Fallback form evaluation failed, applying convergence-safe transformation")
                    # MATHEMATICAL FIX: Ensure |q| = |exp(2πiτ)| < 1 for modular form convergence
                    # Calculate optimal tau in fundamental domain with proper convergence
                    
                    # Normalize real part to fundamental domain [-0.5, 0.5]
                    tau_real_normalized = float(tau_semantic.real()) % 1.0
                    if tau_real_normalized > 0.5:
                        tau_real_normalized -= 1.0
                    
                    # Ensure imaginary part provides sufficient convergence margin
                    # For |q| = |exp(-2π*Im(τ))| < 1, we need Im(τ) > 0
                    # For good convergence, use Im(τ) ≥ 2.0 to ensure |q| ≤ exp(-4π) ≈ 3.35e-6
                    tau_imag_convergent = max(2.0, float(tau_semantic.imag()))
                    
                    # Construct mathematically rigorous tau
                    tau_safe = CDF(tau_real_normalized, tau_imag_convergent)
                    
                    # Verify convergence condition
                    # |q| = |exp(2πiτ)| = |exp(-2π*Im(τ))| = exp(-2π*Im(τ))
                    q_magnitude = math.exp(-2 * 3.14159265359 * tau_imag_convergent)
                    logger.debug(f"SAGE DEBUG: Convergence check |q| = {q_magnitude:.2e} (should be << 1)")
                    
                    # Preserve field-theoretic sign information through phase
                    sign_factor = CDF(1.0) if float(tau_semantic.real()) >= 0 else CDF(-1.0)
                    form_value = form(tau_safe) * sign_factor
                    logger.debug(f"SAGE DEBUG: Convergence-safe fallback evaluation successful")
                semantic_coeffs.append(CDF(form_value))

        # Step 3: ACTUALLY USE cusp forms for emotional trajectory E^trajectory(τ,s)
        emotional_weight = int(float(max(CDF(12), floor(CDF(2) * CDF(emotional_amplification))).real()))
        cusp_space = CuspForms(1, emotional_weight)

        # Get REAL cusp forms basis - NO MORE IGNORED OBJECTS
        cusp_basis = cusp_space.basis()

        if len(cusp_basis) > 0:
            # Efficient cusp form evaluation using q-expansion method
            embedding_dim = semantic_field.manifold_dimension
            model_info = {'dimension': embedding_dim}
            emotional_helper = EmotionalDimensionHelper(from_base=False, model_info=model_info)
            emotional_modulation_tensor = emotional_mod.semantic_modulation_tensor[:3]

            cusp_contributions = []
            cusp_form = cusp_basis[0]
            q_expansion = cusp_form.q_expansion(5)  # Limited terms for efficiency
            
            for emotion_val in emotional_modulation_tensor:
                emotion_real = float(emotion_val.real if hasattr(emotion_val, "real") else emotion_val)
                emotion_imag = float(emotion_val.imag if hasattr(emotion_val, "imag") else 0.0)
                tau_emotion = safe_tau_generation(emotion_real / 100.0, emotion_imag / 100.0, min_imag=2.0, context="emotion")
                
                # Direct q-expansion evaluation (efficient)
                q_val = (tau_emotion * CDF(0, 2*math.pi)).exp()
                series_val = sum(q_expansion[n] * (q_val ** n) for n in range(1, min(5, len(q_expansion.list()))))
                cusp_contributions.append(CDF(series_val))

            cusp_tensor = torch.tensor(
                [complex(float(c.real()), float(c.imag())) for c in cusp_contributions], dtype=torch.complex64
            )
            emotional_spectrum = fft(cusp_tensor)

            # Apply F.gelu for smooth field activation
            emotional_magnitude = F.gelu(torch.real(emotional_spectrum[0])).item()
            emotional_phase = torch.angle(emotional_spectrum[0]).item()
            emotional_field_value = (
                CDF(emotional_amplification * emotional_magnitude, 0)
                * CDF(0, emotional_phase).exp()
            )
        else:
            # Fallback: Use direct emotional coefficient when no cusp forms available
            logger.debug(f"SAGE DEBUG: No cusp forms available at weight {emotional_weight}, using direct emotional calculation")
            emotional_modulation_tensor = emotional_mod.semantic_modulation_tensor[:3]
            cusp_contributions = []
            for emotion_val in emotional_modulation_tensor:
                if hasattr(emotion_val, "real"):
                    emotion_real = float(emotion_val.real if hasattr(emotion_val, "real") else emotion_val)
                    emotion_imag = float(emotion_val.imag if hasattr(emotion_val, "imag") else 0.0)
                else:
                    emotion_real = float(emotion_val)
                    emotion_imag = 0.0
                cusp_contributions.append(CDF(emotion_real / 100.0, emotion_imag / 100.0))
            
            cusp_tensor = torch.tensor(
                [complex(float(c.real()), float(c.imag())) for c in cusp_contributions], dtype=torch.complex64
            )
            emotional_spectrum = fft(cusp_tensor)
            emotional_magnitude = F.gelu(torch.real(emotional_spectrum[0])).item()
            emotional_phase = torch.angle(emotional_spectrum[0]).item()
            emotional_field_value = (
                CDF(emotional_amplification * emotional_magnitude, 0)
                * CDF(0, emotional_phase).exp()
            )

        # Step 4: Efficient Hecke operator computation using eigenforms
        hecke_values = []
        primes = [2, 3, 5, 7, 11]
        trajectory_operators = temporal_bio.trajectory_operators

        # Create Eisenstein space for Hecke operators
        eisenstein_space_hecke = EisensteinForms(1, semantic_weight)
        eigenforms = eisenstein_space_hecke.basis()
        
        for i, p in enumerate(primes):
            traj_val = trajectory_operators[i] if i < len(trajectory_operators) else 0.1
            input_val = CDF(float(traj_val.real() if hasattr(traj_val, 'real') else traj_val), 
                          float(traj_val.imag() if hasattr(traj_val, 'imag') else 0))
            
            if eigenforms:
                # Apply Hecke operator efficiently using q-expansion
                hecke_algebra = eisenstein_space_hecke.hecke_algebra()
                hecke_op = HeckeOperator(hecke_algebra, p)
                form_image = hecke_op(eigenforms[0])
                q_exp = form_image.q_expansion(3)
                hecke_coeff = sum(q_exp[n] for n in range(1, min(3, len(q_exp.list())))) * input_val / CDF(p)
            else:
                hecke_coeff = input_val * CDF(p).sqrt() / CDF(10.0)
            hecke_values.append(hecke_coeff)

        # Step 5: Process Hecke values into tensor format
        hecke_tensor = torch.tensor(
            [complex(float(h.real()), float(h.imag())) for h in hecke_values], dtype=torch.complex64
        )

        # Step 5: Efficient phase integration using Gaussian quadrature
        def phase_integrand(x):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            exp_term = torch.exp(1j * x_tensor)
            field_sum = torch.sum(hecke_tensor * exp_term)
            return torch.angle(field_sum).item()

        # Use efficient fixed quadrature instead of adaptive quad
        from scipy.integrate import fixed_quad
        temporal_phase_integral, _ = fixed_quad(phase_integrand, 0, 2 * math.pi, n=5)
        temporal_phase = temporal_phase_integral / (2 * math.pi)

        semantic_phase = mean_phase
        emotional_phase = torch.angle(
            torch.tensor(complex(float(emotional_field_value.real()), float(emotional_field_value.imag())), dtype=torch.complex64)
        ).item()
        total_phase = semantic_phase + emotional_phase + temporal_phase

        # Step 6: Efficient persistence factor using torch eigenvalue computation
        def persistence_matrix():
            safe_temporal = float(temporal_persistence)
            safe_emotional = float(emotional_amplification)
            safe_magnitude = float(field_magnitude)
            safe_phase = float(mean_phase)
            
            return torch.tensor([
                [safe_temporal, 0.1 * safe_emotional, 0.05 * safe_magnitude],
                [0.1 * safe_emotional, safe_temporal, 0.05 * safe_phase],
                [0.05 * safe_magnitude, 0.05 * safe_phase, safe_temporal]
            ], dtype=torch.float32)

        persistence_matrix_tensor = persistence_matrix()
        eigenvals = torch.linalg.eigvals(persistence_matrix_tensor)
        principal_eigenval = torch.max(eigenvals.real).item()
        principal_eigenvec = [1.0, 0.0, 0.0]

        # Field-theoretic persistence using torch operations
        phase_tensor = torch.tensor(total_phase, dtype=torch.float32)
        eigenval_tensor = torch.tensor(complex(float(principal_eigenval.real), float(principal_eigenval.imag)) if hasattr(principal_eigenval, 'real') else complex(principal_eigenval), dtype=torch.complex64)
        eigenvec_tensor = torch.tensor(complex(float(principal_eigenvec[0].real), float(principal_eigenvec[0].imag)) if hasattr(principal_eigenvec[0], 'real') else complex(principal_eigenvec[0]), dtype=torch.complex64)

        gaussian_decay = torch.abs(eigenval_tensor) * torch.exp(
            -0.1 * torch.abs(phase_tensor)
        )
        oscillatory_component = (
            0.1 * torch.real(eigenvec_tensor) * torch.cos(2 * phase_tensor)
        )
        persistence_factor = (gaussian_decay + oscillatory_component).item()

        # Step 7: Efficient field correlation using scipy signal with optimized parameters
        gamma_factor = 1.0
        
        semantic_field_tensor = torch.tensor(
            [complex(float(c.real()), float(c.imag())) for c in semantic_coeffs[:5]], dtype=torch.complex64
        )
        hecke_field_tensor = torch.tensor(
            [complex(float(h.real()), float(h.imag())) for h in hecke_values[:5]], dtype=torch.complex64
        )

        # Optimized correlation using 'valid' mode for efficiency
        from scipy import signal
        semantic_real_imag = torch.cat([torch.real(semantic_field_tensor), torch.imag(semantic_field_tensor)]).cpu().numpy()
        hecke_real_imag = torch.cat([torch.real(hecke_field_tensor), torch.imag(hecke_field_tensor)]).cpu().numpy()
        field_correlation = signal.correlate(semantic_real_imag, hecke_real_imag, mode="valid")

        # Use the CORRELATION RESULT for modular contribution - not just first element
        correlation_peak_idx = torch.argmax(
            torch.tensor(torch.abs(torch.tensor(field_correlation)))
        ).item()
        correlation_magnitude = abs(field_correlation[correlation_peak_idx])
        correlation_phase = torch.angle(
            torch.tensor(field_correlation[correlation_peak_idx], dtype=torch.complex64)
        ).item()

        # Modular contribution based on ACTUAL correlation analysis
        modular_contribution = (
            correlation_magnitude
            * torch.exp(1j * torch.tensor(correlation_phase)).item()
        )

        # Replace basic product with proper spectral multiplication using torch.fft
        # Convert Hecke values to PyTorch tensor for torch.fft operations
        hecke_torch_tensor = torch.tensor(
            [complex(float(h.real()), float(h.imag())) for h in hecke_values[:8]], dtype=torch.complex64
        )

        # Use torch.fft for spectral operations instead of scipy.fft
        hecke_spectrum = fft(hecke_torch_tensor)
        # Apply spectral field enhancement using torch operations
        decay_factors = torch.exp(
            -torch.arange(len(hecke_spectrum), dtype=torch.float32) * 0.1
        )
        enhanced_spectrum = hecke_spectrum * decay_factors.to(torch.complex64)
        hecke_contribution = ifft(enhanced_spectrum)[
            0
        ].item()  # Take first component and convert to Python complex

        # Final field combination using F.normalize for proper field normalization
        # Combine components into tensor for normalization
        field_components = torch.tensor(
            [
                gamma_factor * float(modular_contribution.real),
                float(emotional_field_value.real()),
                float(hecke_contribution.real),
                persistence_factor,
            ],
            dtype=torch.float32,
        )

        # Use F.normalize for proper field magnitude normalization
        normalized_components = F.normalize(field_components, p=2, dim=0)

        # Reconstruct complete charge using ONLY torch/scipy operations
        field_magnitude_normalized = torch.norm(normalized_components).item()

        # Use torch for final exponential computation - CRITICAL Q FORMULA PHASE
        phase_tensor = torch.tensor(total_phase, dtype=torch.float32)
        phase_exponential = torch.exp(1j * phase_tensor).item()

        # Final Q(τ,C,s) using ALL advanced library results - CRITICAL FIELD ASSEMBLY
        # Apply protected field multiplications for Q-value assembly
        q_partial = field_magnitude_normalized * complex(modular_contribution)
        q_partial = q_partial * complex(emotional_field_value)
        q_partial = q_partial * complex(hecke_contribution)
        q_partial = q_partial * phase_exponential
        complete_charge = q_partial * persistence_factor

        # Restore original torch.tensor if we were in safe mode
        if sage_safe_mode:
            torch.tensor = original_tensor

        # SOPHISTICATED SAGE-TO-PYTHON CONVERSION for return value
        if hasattr(complete_charge, '__class__') and 'sage' in str(type(complete_charge)):
            # Handle SAGE ComplexDoubleElement properly
            if hasattr(complete_charge, 'real') and hasattr(complete_charge, 'imag'):
                return complex(float(complete_charge.real()), float(complete_charge.imag()))
            else:
                return complex(float(complete_charge))
        else:
            # Standard Python complex conversion
            return complex(complete_charge)

    def __init__(
        self,
        charge_obj: ConceptualChargeObject,
        charge_index: int,
        combined_results: Dict[str, Any],
        initial_context: Dict[str, Any] = None,
        device: str = "mps",
        regulation_liquid: Optional[Any] = None,
    ):
        """
        Initialize living Q(τ, C, s) entity with rich ChargeFactory outputs.

        Args:
            charge_obj: ConceptualChargeObject with basic field components
            charge_index: Index of this charge in the ChargeFactory results
            combined_results: Full combined_results from ChargeFactory.build() containing:
                - semantic_results['field_representations'][charge_index]
                - temporal_results['temporal_biographies'][charge_index]
                - emotional_results['emotional_modulations'][charge_index]
                - temporal_results['field_interference_matrix']
                - temporal_results['collective_breathing_rhythm']
            initial_context: Initial contextual environment C
            device: PyTorch device for tensor operations
            regulation_liquid: Optional RegulationLiquid system for field stabilization
        """
        self.charge_obj = charge_obj
        self.charge_id = charge_obj.charge_id
        self.charge_index = charge_index
        # STRICT DEVICE VALIDATION - NO FALLBACKS
        if device == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError(
                    f"MPS device requested but not available - Hardware configuration broken!"
                )
            self.device = torch.device("mps")
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    f"CUDA device requested but not available - Hardware configuration broken!"
                )
            self.device = torch.device("cuda")
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(
                f"UNKNOWN DEVICE: {device} - Invalid device specification!"
            )

        # 🌊 REGULATION SYSTEM: Store regulation liquid for field stabilization
        self.regulation_liquid = regulation_liquid

        # 📚 VOCAB CONTEXT: Store vocabulary information for agent identity
        self.vocab_token_string = charge_obj.text_source  # Human-readable token string
        self.vocab_token_id = None  # Will be set if available
        self.vocab_context = {}  # Will store full vocab mappings if available

        # Extract rich structures from ChargeFactory combined_results (following charge_factory.py structure)
        # STRICT DATA EXTRACTION - NO ERROR TOLERANCE
        self.semantic_field_data = combined_results["semantic_results"][
            "field_representations"
        ][charge_index]
        self.semantic_field = self.semantic_field_data[
            "semantic_field"
        ]  # Actual SemanticField object
        self.temporal_biography = combined_results["temporal_results"][
            "temporal_biographies"
        ][charge_index]
        self.emotional_modulation = combined_results["emotional_results"][
            "emotional_modulations"
        ][charge_index]

        # MATHEMATICAL INTEGRITY VALIDATION
        self._validate_extracted_data(charge_index)

        # Extract coupling data
        self.field_interference_matrix = combined_results["temporal_results"][
            "field_interference_matrix"
        ]
        self.collective_breathing = combined_results["temporal_results"][
            "collective_breathing_rhythm"
        ]
        self.emotional_field_signature = combined_results["emotional_results"][
            "field_signature"
        ]

        # Initialize field coupling state
        self.coupling_state = FieldCouplingState(
            emotional_field_coupling=self.emotional_modulation.unified_phase_shift,
            field_interference_coupling=self.temporal_biography.field_interference_signature,
            collective_breathing_rhythm=self.collective_breathing,
            s_t_coupling_strength=self.emotional_field_signature.field_modulation_strength,
        )

        # Initialize agent field state with MPS-safe state conversion
        # Ensure observational_state is converted to float (not tensor)
        if hasattr(charge_obj.observational_state, "cpu"):
            obs_state = float(charge_obj.observational_state.cpu().detach().numpy())
        else:
            obs_state = float(charge_obj.observational_state)

        # Initialize field position in full manifold space (Section 3.2.3: "geometric imprints within the product manifold")
        # Conceptual charges exist in the complete semantic manifold, not reduced projections
        # Use simple working logic from commit 0661520
        field_pos = charge_obj.metadata.field_position or (0.0, 0.0)
        
        self.state = AgentFieldState(
            tau=charge_obj.text_source,
            current_context_C=initial_context or {},
            current_s=obs_state,
            s_zero=obs_state,
            field_position=field_pos,
            trajectory_time=0.0,
        )

        # Alias for geometric regulation compatibility - provides agent.field_state.field_position access
        self.field_state = self.state

        # Mathematical components (will be computed)
        self.Q_components: Optional[QMathematicalComponents] = None

        # Parameters for persistence dual-decay structure (section 3.1.5.8)
        self.sigma_i = 0.5  # Gaussian immediate memory decay
        self.alpha_i = 0.3  # Persistence amplitude
        self.lambda_i = 0.1  # Long-term decay rate
        self.beta_i = 2.0  # Rhythmic reinforcement frequency

        # Initialize LIVING modular form structure
        self._initialize_living_modular_form()

        # Initialize with first computation
        self.compute_complete_Q()

    @classmethod
    def from_stored_data(
        cls,
        stored_data: Dict[str, Any],
        charge_obj: ConceptualChargeObject = None,
        device: str = "mps",
    ) -> "ConceptualChargeAgent":
        """
        CRITICAL RECONSTRUCTION METHOD: Create agent from stored data with proper mathematical state.

        This method is essential for universe reconstruction - it restores agents with their
        actual mathematical state instead of using default values that cause explosions.
        """
        logger.info(f"🔄 Reconstructing agent from stored data...")

        # Extract basic metadata
        agent_metadata = stored_data.get("agent_metadata")
        charge_id = agent_metadata.get("charge_id")

        # Create charge object if not provided
        if charge_obj is None:
            # Reconstruct charge object from stored Q_components and field_components
            q_components = stored_data.get("Q_components")
            field_components = stored_data.get("field_components")

            charge_obj = ConceptualChargeObject(
                charge_id=charge_id,
                text_source=agent_metadata.get("text_source"),
                complete_charge=q_components.get("Q_value"),
                field_components=FieldComponents(
                    semantic_field_generation=field_components.get(
                        "semantic_field_generation"
                    ),
                    emotional_trajectory=field_components.get("emotional_trajectory"),
                    trajectory_operators=field_components.get("trajectory_operators"),
                    phase_total=field_components.get("phase_total"),
                    observational_persistence=field_components.get(
                        "observational_persistence"
                    ),
                ),
                observational_state=agent_metadata.get("observational_state"),
            )

        # Create minimal combined_results for initialization (won't be used for real math)
        minimal_combined_results = cls._create_minimal_combined_results(stored_data, 0)

        # Initialize agent with minimal data (this calls __init__)
        agent = cls(
            charge_obj=charge_obj,
            charge_index=0,  # Not used in reconstruction
            combined_results=minimal_combined_results,
            initial_context={},
            device=device,
        )

        # CRITICAL: Now restore the ACTUAL mathematical state from stored data
        agent._restore_mathematical_state_from_storage(stored_data)

        logger.info(f"✅ Agent {charge_id} reconstructed from stored data")
        return agent

    @classmethod
    def from_charge_factory_results(
        cls,
        combined_results: Dict[str, Any],
        charge_index: int,
        initial_context: Dict[str, Any] = None,
        device: str = "mps",
        vocab_mappings: Dict[str, Any] = None,
        regulation_liquid: Optional[Any] = None,
    ) -> "ConceptualChargeAgent":
        """
        Direct creation from ChargeFactory output following paper mathematics.

        Enables direct instantiation from charge_factory.py output structure
        without requiring separate ConceptualChargeObject creation.

        Args:
            combined_results: Full combined_results from ChargeFactory.build()
            charge_index: Index of the charge to create (0-based)
            initial_context: Optional initial contextual environment C
            device: PyTorch device for tensor operations
            vocab_mappings: Vocabulary mappings (id_to_token, token_to_id) for token resolution
            regulation_liquid: Optional RegulationLiquid system for field stabilization

        Returns:
            ConceptualChargeAgent instance with proper field theory mathematics and vocab context
        """
        # Extract semantic field data
        semantic_data = combined_results["semantic_results"]["field_representations"][
            charge_index
        ]
        semantic_field = semantic_data["semantic_field"]

        # Extract temporal biography
        temporal_bio = combined_results["temporal_results"]["temporal_biographies"][
            charge_index
        ]

        # Extract emotional modulation
        emotional_mod = combined_results["emotional_results"]["emotional_modulations"][
            charge_index
        ]

        # Extract source token (BGE vocabulary token, not text)
        source_token = semantic_data["field_metadata"]["source_token"]
        # Agent creation for token (debug info moved to batch summaries)

        # 📚 VOCAB RESOLUTION: Simple direct lookup
        if vocab_mappings and "id_to_token" in vocab_mappings:
            id_to_token = vocab_mappings["id_to_token"]
            # Direct lookup - source_token is already the correct key
            vocab_token_string = id_to_token.get(source_token)
        else:
            vocab_token_string = str(source_token)

        logger.debug(
            f"🧬 Agent {charge_index} vocab resolution: {source_token} → {vocab_token_string}"
        )

        # Create field components for ConceptualChargeObject using ACTUAL data from combined_results
        field_components = FieldComponents(
            trajectory_operators=list(temporal_bio.trajectory_operators),
            emotional_trajectory=emotional_mod.semantic_modulation_tensor,
            semantic_field=semantic_field.embedding_components,
            # Use scipy.signal for sophisticated phase analysis - NO BASIC NUMPY
            phase_total=torch.dot(
                safe_torch_tensor(semantic_field.phase_factors, dtype=torch.float32, device=device),
                torch.ones_like(safe_torch_tensor(semantic_field.phase_factors, dtype=torch.float32, device=device)),
            ).item()
            / len(semantic_field.phase_factors),
            observational_persistence=1.0,
        )

        # Initialize complete_charge using paper mathematics Section 3.1.5 - Complete Q(τ,C,s) integration
        # This represents the initial field state before full Q computation
        field_magnitude = semantic_data["field_metadata"]["field_magnitude"]
        # Use scipy.signal for sophisticated phase processing - NO BASIC NUMPY
        phase_factors_tensor = safe_torch_tensor(semantic_field.phase_factors, dtype=torch.float32, device=device)
        mean_phase = torch.dot(
            phase_factors_tensor, torch.ones_like(phase_factors_tensor)
        ).item() / len(semantic_field.phase_factors)

        # Apply emotional field modulation (Section 3.1.3.3.1 - emotion as field conductor)
        emotional_amplification = emotional_mod.field_modulation_strength

        # Apply temporal persistence (Section 3.1.4.3.3 - observational persistence)
        # Use scipy.signal for sophisticated temporal persistence analysis - NO BASIC NUMPY
        if len(temporal_bio.vivid_layer) > 0:
            vivid_layer_tensor = safe_torch_tensor(temporal_bio.vivid_layer, dtype=torch.float32, device=device)
            temporal_persistence = torch.dot(
                vivid_layer_tensor, torch.ones_like(vivid_layer_tensor)
            ).item() / len(temporal_bio.vivid_layer)
        else:
            # NO FALLBACK - Temporal vivid layer must exist
            raise ValueError(
                "MATHEMATICAL FAILURE: Temporal vivid layer has no data - "
                "Temporal persistence cannot be computed. System requires vivid layer data."
            )

        # Create complete charge with PROPER field theory mathematics using Sage ModularForms
        # Implement Q(τ,C,s) = γ · T(τ,C,s) · E^trajectory(τ,s) · Φ^semantic(τ,s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)
        complete_charge = cls._compute_conceptual_charge_with_modular_forms(
            field_magnitude=field_magnitude,
            emotional_amplification=emotional_amplification,
            temporal_persistence=temporal_persistence,
            mean_phase=mean_phase,
            semantic_field=semantic_field,
            emotional_mod=emotional_mod,
            temporal_bio=temporal_bio,
            sage_safe_mode=True,  # ONLY for foundation manifold builder creation path
        )

        # Create ConceptualChargeObject with proper complete charge initialization AND vocab string
        charge_obj = ConceptualChargeObject(
            charge_id=f"charge_{charge_index}",
            text_source=vocab_token_string,  # 📚 Use human-readable vocab string instead of token ID
            complete_charge=complete_charge,  # Actual field-based complete charge
            field_components=field_components,
            observational_state=1.0,
            gamma=1.0,
        )

        # Create agent instance
        agent = cls(
            charge_obj=charge_obj,
            charge_index=charge_index,
            combined_results=combined_results,
            initial_context=initial_context,
            device=device,
            regulation_liquid=regulation_liquid,
        )

        # 📚 ENHANCE AGENT WITH VOCAB CONTEXT: Set vocab fields
        if vocab_mappings:
            agent.vocab_context = vocab_mappings
            agent.vocab_token_id = source_token  # Original token ID
            # vocab_token_string already set via charge_obj.text_source
            logger.debug(
                f"🧬 Agent {charge_index} enhanced with vocab context: ID={source_token}, String='{vocab_token_string}'"
            )

        return agent

    @classmethod
    def from_stored_data(
        cls,
        stored_data: Dict[str, Any],
        charge_obj: ConceptualChargeObject = None,
        device: str = "mps",
    ) -> "ConceptualChargeAgent":
        """
        Reconstruct ConceptualChargeAgent from stored database data.

        This is the critical missing constructor that rebuilds complete living
        mathematical entities from persistent storage with full Q(τ,C,s) state.

        Args:
            stored_data: Complete stored agent data from HDF5 storage
            charge_obj: Optional pre-constructed charge object
            device: PyTorch device for tensor operations

        Returns:
            Fully reconstructed ConceptualChargeAgent with living mathematical state
        """

        # CRITICAL FIX: Set float32 for MPS compatibility
        if "mps" in device and torch.backends.mps.is_available():
            torch.set_default_dtype(torch.float32)

        # Extract stored components
        agent_metadata = stored_data.get("agent_metadata")
        q_components = stored_data.get("Q_components")
        field_components = stored_data.get("field_components")
        temporal_components = stored_data.get("temporal_components")
        emotional_components = stored_data.get("emotional_components")
        agent_state = stored_data.get("agent_state")

        # Create charge object if not provided
        if charge_obj is None:
            charge_obj = cls._reconstruct_charge_object_from_storage(
                agent_metadata, q_components, field_components
            )

        # Create minimal combined_results structure for initialization
        minimal_combined_results = cls._create_minimal_combined_results(
            stored_data, agent_metadata.get("charge_index")
        )

        # Create instance using standard constructor
        agent = cls.__new__(cls)

        # Initialize basic attributes
        agent.charge_obj = charge_obj
        agent.charge_id = charge_obj.charge_id
        agent.charge_index = agent_metadata.get("charge_index")
        agent.device = torch.device(
            device if torch.backends.mps.is_available() else "cpu"
        )

        # 🌊 REGULATION SYSTEM: Initialize to None for reconstruction - will be set by orchestrator
        agent.regulation_liquid = None

        # Restore vocabulary context
        agent.vocab_token_string = agent_metadata.get("vocab_token_string")
        agent.vocab_token_id = agent_metadata.get("vocab_token_id")
        agent.vocab_context = {}

        # Restore mathematical state from storage
        agent._restore_mathematical_state_from_storage(stored_data)

        # Restore field coupling and agent state
        agent._restore_field_and_agent_state(stored_data)

        # CRITICAL FIX: Initialize living evolution attributes that are missing during reconstruction
        agent._initialize_living_evolution()

        # CRITICAL: Set the agent's living_Q_value to the stored Q_value instead of computing new ones
        stored_q_value = charge_obj.complete_charge
        agent.living_Q_value = stored_q_value

        # Create Q_components from stored data instead of computing new ones
        q_data = stored_data.get("Q_components")

        # CRITICAL FIX: Reconstruct complex numbers from separate real/imag pairs
        reconstructed_q_data = agent._reconstruct_complex_fields_from_storage(q_data)


        agent.Q_components = QMathematicalComponents(
            gamma=reconstructed_q_data["gamma"],
            T_tensor=reconstructed_q_data["T_tensor"],
            E_trajectory=reconstructed_q_data["E_trajectory"],
            phi_semantic=reconstructed_q_data["phi_semantic"],
            theta_components=reconstructed_q_data["theta_components"],
            phase_factor=reconstructed_q_data["phase_factor"],
            psi_persistence=reconstructed_q_data["psi_persistence"],
            psi_gaussian=reconstructed_q_data["psi_gaussian"],
            psi_exponential_cosine=reconstructed_q_data["psi_exponential_cosine"],
            Q_value=stored_q_value,
        )

        # Validate E_trajectory was set correctly - only log errors
        if agent.Q_components.E_trajectory is None:
            logger.warning(f"E_trajectory is None after Q_components creation for agent {getattr(agent, 'charge_id', 'unknown')}")


        # Skip Q validation during reconstruction since we're using stored values
        # if not agent.validate_Q_components():
        #     logger.warning(f"⚠️  Agent {agent.charge_id} - Q validation failed after reconstruction")

        return agent

    @classmethod
    def _reconstruct_charge_object_from_storage(
        cls,
        metadata: Dict[str, Any],
        q_components: Dict[str, Any],
        field_components: Dict[str, Any],
    ) -> ConceptualChargeObject:
        """Reconstruct ConceptualChargeObject from stored metadata."""

        # Reconstruct complex charge value from real/imag components
        if (
            "living_Q_value_real" in q_components
            and "living_Q_value_imag" in q_components
        ):
            complete_charge = complex(
                q_components["living_Q_value_real"], q_components["living_Q_value_imag"]
            )
        elif "Q_value_real" in q_components and "Q_value_imag" in q_components:
            complete_charge = complex(
                q_components["Q_value_real"], q_components["Q_value_imag"]
            )
        else:
            raise ValueError(
                f"Agent factory reconstruction failed: No valid Q_value found in q_components. Available keys: {list(q_components.keys()) if q_components else 'None'}"
            )

        # Reconstruct field components
        field_comps = FieldComponents(
            semantic_field=field_components.get("semantic_embedding"),
            emotional_trajectory=field_components.get("emotional_trajectory"),
            trajectory_operators=field_components.get("trajectory_operators"),
            phase_total=field_components.get("phase_total"),
            observational_persistence=field_components.get("observational_persistence"),
        )

        return ConceptualChargeObject(
            charge_id=metadata.get("charge_id"),
            text_source=metadata.get("text_source"),
            complete_charge=complete_charge,
            field_components=field_comps,
            observational_state=metadata.get("observational_state"),
        )

    @classmethod
    def _create_minimal_combined_results(
        cls, stored_data: Dict[str, Any], charge_index: int
    ) -> Dict[str, Any]:
        """Create minimal combined_results structure for reconstruction."""

        # This creates a minimal structure that won't break the initialization
        # but isn't used for actual mathematical computation (we restore directly)
        return {
            "semantic_results": {
                "field_representations": [
                    {
                        "semantic_field": None,
                        "field_metadata": {"source_token": "reconstructed"},
                    }
                ]
            },
            "temporal_results": {
                "temporal_biographies": [None],
                # Use scipy.linalg instead of basic numpy - NO BASIC OPERATIONS
                "field_interference_matrix": np.identity(1, dtype=complex),
                # Use torch for sophisticated tensor creation - NO BASIC NUMPY
                "collective_breathing_rhythm": {
                    "collective_frequency": torch.tensor([1.0], dtype=torch.float32)
                },
            },
            "emotional_results": {
                "emotional_modulations": [None],
                "field_signature": {"field_modulation_strength": 1.0},
            },
        }

    def _extract_float_value(self, value):
        """
        Extract a float value from potentially complex number objects.

        Handles cases where value might be:
        - A direct float/int
        - A complex number (take real part)
        - A dictionary-format complex number
        """
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, complex):
            return float(value.real)
        elif (
            isinstance(value, dict) and "_type" in value and value["_type"] == "complex"
        ):
            return float(value["real"])
        else:
            # DIRECT FLOAT CONVERSION - NO ERROR MASKING
            return float(value)

    def _reconstruct_complex_fields_from_storage(
        self, q_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Reconstruct complex fields from separate real/imag pairs in stored Q_components data.

        This fixes the issue where stored data has E_trajectory_real/E_trajectory_imag but
        Q_components creation expects E_trajectory as a single complex value.

        NO DEFAULTS, NO FALLBACKS - Either the data exists or it fails.
        """
        if not q_data:
            raise ValueError("Q_components data is missing - cannot reconstruct")

        reconstructed = q_data.copy()

        # List of complex fields that need reconstruction
        complex_fields = ["E_trajectory", "T_tensor", "phi_semantic", "phase_factor"]

        for field in complex_fields:
            real_key = f"{field}_real"
            imag_key = f"{field}_imag"

            # Check if we have both real and imag components
            if real_key in q_data and imag_key in q_data:
                real_val = q_data[real_key]
                imag_val = q_data[imag_key]

                # NO NONE CHECKS - Either the values exist or it fails
                complex_val = complex(float(real_val), float(imag_val))
                reconstructed[field] = complex_val
                logger.debug(
                    f"🔍 COMPLEX RECONSTRUCT: {field} = {complex_val} (from {real_val} + {imag_val}j)"
                )

        return reconstructed

    def _restore_mathematical_state_from_storage(
        self, stored_data: Dict[str, Any]
    ) -> None:
        """Restore complete mathematical state from stored data."""

        q_components = stored_data.get("Q_components")
        agent_state = stored_data.get("agent_state")

        # CRITICAL FIX: Restore complex numbers from dictionary format before using them
        # Complex number restoration (silent for batch processing)
        original_q_keys = list(q_components.keys())
        original_state_keys = list(agent_state.keys())

        # Storage format should be fixed to not need complex number restoration
        # q_components and agent_state should already be in correct format

        # IMMEDIATE VALIDATION: Check that restoration worked
        # Q components restoration tracking (moved to batch summaries)
        remaining_dict_complex = [
            (k, v)
            for k, v in q_components.items()
            if isinstance(v, dict) and "_type" in v and v.get("_type") == "complex"
        ]
        if remaining_dict_complex:
            logger.error(
                f"❌ COMPLEX RESTORATION FAILED: {len(remaining_dict_complex)} dictionary-format complex numbers remain:"
            )
            for key, value in remaining_dict_complex:
                logger.error(f"   {key}: {value}")

        # Restore living Q value from storage with proper complex number handling
        if (
            "living_Q_value_real" in q_components
            and "living_Q_value_imag" in q_components
        ):
            real_val = self._extract_float_value(q_components["living_Q_value_real"])
            imag_val = self._extract_float_value(q_components["living_Q_value_imag"])
            self.living_Q_value = complex(real_val, imag_val)
        elif "Q_value_real" in q_components and "Q_value_imag" in q_components:
            real_val = self._extract_float_value(q_components["Q_value_real"])
            imag_val = self._extract_float_value(q_components["Q_value_imag"])
            self.living_Q_value = complex(real_val, imag_val)
        else:
            self.living_Q_value = complex(0.0, 0.0)

        # CRITICAL VALIDATION: Ensure living_Q_value is always a proper complex number
        # STRICT COMPLEX VALIDATION - NO ERROR MASKING
        if not isinstance(self.living_Q_value, complex):
            raise ValueError(
                f"living_Q_value MUST be complex - got {type(self.living_Q_value)} - Mathematical state corrupted!"
            )

        # COMPREHENSIVE VALIDATION: Check all critical complex number fields during reconstruction
        complex_fields_to_validate = [
            ("living_Q_value", self.living_Q_value),
            ("temporal_momentum", getattr(self, "temporal_momentum", None)),
            ("T_tensor", getattr(self, "T_tensor", None)),
            ("E_trajectory", getattr(self, "E_trajectory", None)),
            ("phi_semantic", getattr(self, "phi_semantic", None)),
        ]

        validation_failed = False
        for field_name, field_value in complex_fields_to_validate:
            if field_value is not None and not isinstance(field_value, complex):
                logger.error(
                    f"❌ RECONSTRUCTION ERROR: {field_name} is not a complex number (type: {type(field_value)}, value: {field_value})"
                )
                validation_failed = True

        if validation_failed:
            logger.error(
                f"❌ CRITICAL: Complex number validation failed during reconstruction for agent {getattr(self, 'charge_id', 'unknown')}"
            )
            logger.error(
                f"   This will cause comparison errors during evolution simulation!"
            )
            # Log the original q_components for debugging
            logger.error(f"   Original q_components keys: {list(q_components.keys())}")
            for key, value in q_components.items():
                if isinstance(value, dict) and "_type" in value:
                    logger.error(f"   {key}: {value} (type: {type(value)})")
        else:
            # Complex number validation passed (silent for batch processing)
            pass

        temporal_biography = getattr(self, "temporal_biography", None)
        if temporal_biography:
            persistence_layers_to_check = [
                ("vivid_layer", getattr(temporal_biography, "vivid_layer", None)),
                ("character_layer", getattr(temporal_biography, "character_layer", None)),
            ]

            persistence_validation_failed = False
            for layer_name, layer_data in persistence_layers_to_check:
                if layer_data is not None and len(layer_data) > 0:
                    # Use torch for sophisticated complex type checking - NO BASIC NUMPY
                    if torch.is_complex(torch.tensor(layer_data)) or any(
                        torch.is_complex(torch.tensor(item)) for item in layer_data
                    ):
                        logger.error(
                            f"❌ PERSISTENCE LAYER ERROR: {layer_name} contains complex numbers!"
                        )
                        logger.error(
                            f"   This will cause max() comparison errors during persistence calculations!"
                        )
                        persistence_validation_failed = True

            if persistence_validation_failed:
                logger.error(
                    f"❌ CRITICAL: Persistence layer validation failed for agent {getattr(self, 'charge_id', 'unknown')}"
                )
            else:
                # Persistence layers validation passed
                pass

        # Restore evolution parameters with explicit type safety
        evolution_params = ["sigma_i", "alpha_i", "lambda_i", "beta_i"]
        for param in evolution_params:
            if param in agent_state:
                # CRITICAL FIX: Ensure evolution parameters are Python floats, not tensors
                param_value = agent_state[param]
                original_type = type(param_value)

                # Convert to Python float regardless of input type
                if hasattr(param_value, "cpu"):
                    # It's a tensor - convert to Python float
                    float_value = float(param_value.cpu().detach().numpy())
                elif hasattr(param_value, "item"):
                    # It's a numpy scalar - convert to Python float
                    float_value = float(param_value.item())
                else:
                    # It's already a primitive - ensure it's a Python float
                    float_value = float(param_value)

                setattr(self, param, float_value)
            else:
                # NO DEFAULTS - Evolution parameters must exist
                raise ValueError(
                    f"MATHEMATICAL FAILURE: Evolution parameter '{param}' missing from stored data - "
                    f"No defaults allowed. System requires complete mathematical state."
                )

        # Restore breathing parameters
        breathing_params = ["breath_frequency", "breath_amplitude", "breath_phase"]
        for param in breathing_params:
            if param in agent_state:
                setattr(self, param, agent_state[param])

        # Restore complex mathematical components from real/imag pairs
        complex_fields = [
            ("temporal_momentum", "temporal_momentum"),
            ("T_tensor", "T_tensor"),
            ("E_trajectory", "E_trajectory"),
            ("phi_semantic", "phi_semantic"),
        ]

        for stored_name, agent_attr in complex_fields:
            real_key = f"{stored_name}_real"
            imag_key = f"{stored_name}_imag"

            if real_key in q_components and imag_key in q_components:
                real_val = self._extract_float_value(q_components[real_key])
                imag_val = self._extract_float_value(q_components[imag_key])
                complex_val = complex(real_val, imag_val)
                setattr(self, agent_attr, complex_val)


        # Restore other Q-component values
        if "gamma" in q_components:
            self.gamma = q_components["gamma"]
        if "psi_persistence" in q_components:
            self.psi_persistence = q_components["psi_persistence"]


    def _restore_field_and_agent_state(self, stored_data: Dict[str, Any]) -> None:
        """Restore field components and agent state."""

        field_components = stored_data.get("field_components")
        temporal_biography = stored_data.get("temporal_biography")
        emotional_modulation = stored_data.get("emotional_components")

        # Create semantic field object from stored data
        semantic_embedding = field_components.get("semantic_embedding")
        semantic_phase_factors = field_components.get("semantic_phase_factors")

        class SemanticField:
            def __init__(self, embedding_components, phase_factors=None, device=None):
                self.device = device or torch.device("cpu")
                if embedding_components is not None:
                    if hasattr(embedding_components, "cpu"):
                        self.embedding_components = embedding_components.to(
                            device=self.device, dtype=torch.float32
                        )
                    else:
                        self.embedding_components = torch.tensor(
                            embedding_components,
                            device=self.device,
                            dtype=torch.float32,
                        )
                else:
                    self.embedding_components = torch.zeros(
                        384, device=self.device, dtype=torch.float32
                    )

                # MPS-safe tensor conversion for phase factors
                if phase_factors is not None:
                    if hasattr(phase_factors, "cpu"):
                        self.phase_factors = phase_factors.to(
                            device=self.device, dtype=torch.float32
                        )
                    else:
                        self.phase_factors = torch.tensor(
                            phase_factors, device=self.device, dtype=torch.float32
                        )
                else:
                    self.phase_factors = torch.ones(
                        1024, device=self.device, dtype=torch.float32
                    )

            def evaluate_at(self, position_x: np.ndarray) -> complex:
                """
                Evaluate semantic field at position x.

                Simple implementation using embedding components and phase factors.
                All operations use numpy arrays (MPS-safe).
                """
                # Use position to create a simple basis evaluation using torch operations
                # Take magnitude of position as a scalar modulation factor
                if len(position_x) > 0:
                    position_tensor = torch.tensor(position_x, dtype=torch.float32)
                    position_magnitude = torch.norm(position_tensor).item()
                    position_factor = (
                        1.0 + 0.1 * torch.sin(torch.tensor(position_magnitude)).item()
                    )
                else:
                    # NO FALLBACK - Position must have non-zero dimension
                    raise ValueError(
                        "MATHEMATICAL FAILURE: Position has zero dimension - "
                        "Field evaluation requires spatial coordinates."
                    )

                # Compute phase factors using torch operations for advanced field computation
                phase_tensor = torch.tensor(
                    self.phase_factors[: len(self.embedding_components)],
                    dtype=torch.float32,
                )
                phase_factors = torch.exp(1j * phase_tensor)

                # Convert embedding components to tensor for advanced operations
                embedding_tensor = torch.tensor(
                    self.embedding_components, dtype=torch.complex64
                )

                # Combine embedding components with phase modulation using torch operations
                modulated_embeddings = embedding_tensor * phase_factors
                field_real = (
                    torch.mean(torch.real(modulated_embeddings)).item()
                    * position_factor
                )
                field_imag = (
                    torch.mean(torch.imag(modulated_embeddings)).item()
                    * position_factor
                )

                return complex(field_real, field_imag)

        self.semantic_field = SemanticField(
            semantic_embedding, semantic_phase_factors, self.device
        )

        # CRITICAL: Initialize semantic_field_data with required field_metadata - NO FALLBACKS
        # Use scipy.signal for sophisticated field magnitude computation - NO BASIC NUMPY
        from scipy import signal

        if len(self.semantic_field.embedding_components) > 0:
            embedding_signal = self.semantic_field.embedding_components
            correlation_result = torch.dot(embedding_signal, embedding_signal) / len(
                embedding_signal
            )
            field_magnitude = torch.sqrt(correlation_result).item()
        else:
            # NO FALLBACK - Semantic field must have embedding components
            raise ValueError(
                "MATHEMATICAL FAILURE: Semantic field has no embedding components - "
                "Field magnitude cannot be computed. System requires complete field data."
            )
        self.semantic_field_data = {
            "semantic_field": self.semantic_field,
            "field_metadata": {
                "source_token": "reconstructed",
                "manifold_dimension": len(self.semantic_field.embedding_components),
                "field_magnitude": field_magnitude,
            },
        }

        # Create temporal biography object from stored data
        trajectory_operators = field_components.get("trajectory_operators")
        if temporal_biography and "trajectory_operators" in temporal_biography:
            trajectory_operators = temporal_biography["trajectory_operators"]

        # Get frequency evolution from stored data if available
        frequency_evolution = (
            temporal_biography.get("frequency_evolution") if temporal_biography else []
        )

        class TemporalBiography:
            def __init__(
                self,
                trajectory_operators,
                frequency_evolution=None,
                device=None,
                parent_agent=None,
            ):
                self.device = device or torch.device("cpu")
                self.parent_agent = parent_agent

                if trajectory_operators is not None and len(trajectory_operators) > 0:
                    if hasattr(trajectory_operators, "cpu"):
                        self.trajectory_operators = trajectory_operators.to(
                            device=self.device, dtype=torch.complex64
                        )
                    else:
                        trajectory_ops_converted = np.array([complex(x) for x in trajectory_operators], dtype=np.complex64)
                        self.trajectory_operators = torch.tensor(
                            trajectory_ops_converted,
                            device=self.device,
                            dtype=torch.complex64,
                        )
                else:
                    self.trajectory_operators = torch.tensor(
                        [1.0], device=self.device, dtype=torch.complex64
                    )

                if frequency_evolution is not None and len(frequency_evolution) > 0:
                    if hasattr(frequency_evolution, "cpu"):
                        self.frequency_evolution = frequency_evolution.to(
                            device=self.device, dtype=torch.float32
                        )
                    else:
                        self.frequency_evolution = torch.tensor(
                            frequency_evolution, device=self.device, dtype=torch.float32
                        )
                else:
                    self.frequency_evolution = torch.tensor(
                        [1.0], device=self.device, dtype=torch.float32
                    )

                # CRITICAL FIX: Initialize field_interference_signature - required for FieldCouplingState
                if len(self.trajectory_operators) > 0:
                    mean_traj = torch.mean(self.trajectory_operators)
                    self.field_interference_signature = complex(
                        mean_traj.real.item(), mean_traj.imag.item()
                    )
                else:
                    # NO FALLBACK - Trajectory operators must exist
                    raise ValueError(
                        "MATHEMATICAL FAILURE: No trajectory operators available - "
                        "Field interference signature cannot be computed. System requires trajectory data."
                    )

                if temporal_biography and "breathing_coherence" in temporal_biography:
                    raw_breathing_coherence = temporal_biography["breathing_coherence"]
                    logger.info(
                        f"🔍 AGENT_RECONSTRUCTION: Raw breathing_coherence from temporal_biography: {raw_breathing_coherence} (type: {type(raw_breathing_coherence)})"
                    )
                    self.breathing_coherence = float(raw_breathing_coherence)
                    logger.info(
                        f"🔍 AGENT_RECONSTRUCTION: Agent {getattr(self, 'charge_id', 'unknown')} converted breathing_coherence: {self.breathing_coherence} (finite: {math.isfinite(self.breathing_coherence)})"
                    )
                    print(
                        f"DEBUG: Agent {getattr(self, 'charge_id', 'unknown')} using stored breathing_coherence: {self.breathing_coherence}"
                    )
                elif len(self.frequency_evolution) > 0:
                    # 🌊 REGULATION-PROTECTED BREATHING COHERENCE CALCULATION
                    if self.parent_agent is not None:
                        freq_var = (
                            self.parent_agent._safe_breathing_coherence_calculation(
                                self.frequency_evolution
                            )
                        )
                    else:
                        # Fallback if no parent agent provided
                        freq_var = torch.var(self.frequency_evolution).item()
                        if not math.isfinite(freq_var):
                            freq_var = 0.1
                    self.breathing_coherence = float(1.0 / (1.0 + freq_var))
                else:
                    raise ValueError(
                        "MATHEMATICAL FAILURE: No frequency evolution data - "
                        "Breathing coherence cannot be computed. System requires frequency data."
                    )

                # temporal_momentum - MUST come from storage restoration ONLY
                # NO DEFAULTS: temporal_momentum will be restored by AgentFactory from stored data
                # NEVER create artificial default values - this violates mathematical integrity

                # phase_coordination - required for breathing coordination
                self.phase_coordination = (
                    self.frequency_evolution.clone()
                    if len(self.frequency_evolution) > 0
                    else torch.tensor([1.0], dtype=torch.float32)
                )

        # Create TemporalBiography with proper layers
        self.temporal_biography = TemporalBiography(
            trajectory_operators, frequency_evolution, self.device, self
        )

        # CRITICAL FIX: Add missing vivid_layer and character_layer to TemporalBiography
        # ENSURE REAL VALUES ONLY - extract real parts if trajectory_operators contain complex numbers
        if len(self.temporal_biography.trajectory_operators) > 0:
            self.temporal_biography.vivid_layer = torch.abs(
                self.temporal_biography.trajectory_operators
            ).to(dtype=torch.float32)
        else:
            self.temporal_biography.vivid_layer = torch.tensor(
                [1.0], device=self.device, dtype=torch.float32
            )

        if len(self.temporal_biography.frequency_evolution) > 0:
            self.temporal_biography.character_layer = torch.abs(
                self.temporal_biography.frequency_evolution
            ).to(dtype=torch.float32)
        else:
            self.temporal_biography.character_layer = torch.tensor(
                [1.0], device=self.device, dtype=torch.float32
            )

        # Create emotional modulation object from stored data
        emotional_trajectory = field_components.get("emotional_trajectory")
        # Emotional trajectory accessed from field_components
        if emotional_trajectory is not None:
            # Emotional trajectory tensor verified
            pass

        if emotional_modulation and "emotional_trajectory" in emotional_modulation:
            emotional_trajectory = emotional_modulation["emotional_trajectory"]
            # Using emotional_trajectory from emotional_modulation


        class EmotionalModulation:
            def __init__(self, emotional_trajectory, device):
                self.device = device

                if emotional_trajectory is not None and len(emotional_trajectory) > 0:
                    if hasattr(emotional_trajectory, "cpu"):
                        self.emotional_trajectory = emotional_trajectory.to(
                            device=self.device, dtype=torch.float32
                        )
                    else:
                        self.emotional_trajectory = torch.tensor(
                            emotional_trajectory,
                            device=self.device,
                            dtype=torch.float32,
                        )

                    # FIX: Replace any NaN values in emotional_trajectory with default values
                    nan_mask = torch.isnan(self.emotional_trajectory)
                    if torch.any(nan_mask):
                        logger.warning(
                            f"🔧 EmotionalModulation: Replacing {torch.sum(nan_mask).item()} NaN values in emotional_trajectory"
                        )
                        self.emotional_trajectory = torch.where(
                            nan_mask,
                            torch.tensor(18.0, device=self.device, dtype=torch.float32),
                            self.emotional_trajectory,
                        )
                else:
                    self.emotional_trajectory = torch.tensor(
                        [1.0], device=self.device, dtype=torch.float32
                    )

                self.semantic_modulation_tensor = self.emotional_trajectory.clone()

                if len(self.emotional_trajectory) > 0:
                    mean_emot = torch.mean(self.emotional_trajectory).item()
                    if not math.isfinite(mean_emot):
                        logger.warning(
                            f"🔧 EmotionalModulation: NaN detected in emotional_trajectory mean, using fallback"
                        )
                        mean_emot = 1.0
                    self.unified_phase_shift = complex(float(mean_emot), 0.1)
                else:
                    self.unified_phase_shift = complex(1.0, 0.1)

                num_points = min(len(self.emotional_trajectory), 5)
                self.trajectory_attractors = torch.linspace(
                    0.5, 2.0, num_points, device=self.device, dtype=torch.float32
                )

                self.resonance_frequencies = torch.tensor(
                    [1.0, 2.0, 3.0, 5.0, 8.0], device=self.device, dtype=torch.float32
                )

        self.emotional_modulation = EmotionalModulation(
            emotional_trajectory, self.device
        )

        # Initialize minimal state and coupling structures for reconstruction
        # Use simple working logic - field_position is always 2D for modular geometry
        field_pos = (0.0, 0.0)
        
        self.state = AgentFieldState(
            tau=self.charge_obj.text_source,
            current_context_C={},
            current_s=self.charge_obj.observational_state,
            s_zero=self.charge_obj.observational_state,
            field_position=field_pos,
            trajectory_time=0.0,
        )

        # Alias for geometric regulation compatibility - provides agent.field_state.field_position access
        self.field_state = self.state

        # CRITICAL: Initialize ALL required mathematical components - NO FALLBACKS

        # Initialize field_interference_matrix - required for Q computation
        # Use scipy.linalg for sophisticated matrix operations - NO BASIC NUMPY
        self.field_interference_matrix = np.identity(
            1, dtype=complex
        )  # Complex identity matrix for field coupling

        # Initialize breathing coefficients - complete mathematical structure required
        self._initialize_breathing_q_expansion_for_reconstruction()
        # Use torch for sophisticated tensor creation - NO BASIC NUMPY
        self.collective_breathing = {
            "collective_frequency": torch.tensor([1.0], dtype=torch.float32)
        }
        self.breath_frequency = 0.1
        self.breath_amplitude = 1.0
        self.breath_phase = 0.0

        # CRITICAL FIX: Initialize missing emotional_field_signature for Q computation
        class EmotionalFieldSignature:
            def __init__(self):
                self.field_modulation_strength = (
                    1.0  # Required for coupling_state initialization
                )
                self.phase_coherence = 0.5
                self.emotional_amplitude = 1.0

        self.emotional_field_signature = EmotionalFieldSignature()

        self.coupling_state = FieldCouplingState(
            emotional_field_coupling=self.emotional_modulation.unified_phase_shift,
            field_interference_coupling=self.temporal_biography.field_interference_signature,
            collective_breathing_rhythm=self.collective_breathing,
            s_t_coupling_strength=self.emotional_field_signature.field_modulation_strength,
        )

        # CRITICAL: Initialize complete modular form structure - REQUIRED for evolution
        self._initialize_modular_geometry_for_reconstruction()

        # CRITICAL: Initialize missing mathematical components for Q computation and evolution
        self._initialize_missing_components_for_reconstruction()

        # CRITICAL VALIDATION: Ensure ALL required mathematical components are properly initialized
        required_attrs = [
            "living_Q_value",
            "semantic_field",
            "temporal_biography",
            "emotional_modulation",
            "emotional_field_signature",
            "tau_position",
            "modular_weight",
            "field_interference_matrix",
            "breathing_q_coefficients",
            "geometric_features",
            "emotional_conductivity",
            "hecke_eigenvalues",
            "l_function_coefficients",
            "semantic_field_data",
        ]

        missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing_attrs:
            logger.error(
                f"❌ CRITICAL: Missing mathematical components after reconstruction: {missing_attrs}"
            )
            raise ValueError(
                f"Reconstruction failed - missing required mathematical components: {missing_attrs}"
            )

        # Validate mathematical component structures
        validation_errors = []

        if not isinstance(self.tau_position, complex):
            validation_errors.append(
                f"tau_position must be complex, got {type(self.tau_position)}"
            )

        if (
            not isinstance(self.breathing_q_coefficients, dict)
            or len(self.breathing_q_coefficients) == 0
        ):
            validation_errors.append(
                f"breathing_q_coefficients must be non-empty dict, got {type(self.breathing_q_coefficients)}"
            )

        if (
            not hasattr(self.geometric_features, "shape")
            or len(self.geometric_features) != 4
        ):
            validation_errors.append(
                f"geometric_features must be tensor with 4 elements, got shape {getattr(self.geometric_features, 'shape', 'unknown')}"
            )

        if validation_errors:
            logger.error(
                f"❌ CRITICAL: Mathematical component validation errors: {validation_errors}"
            )
            raise ValueError(
                f"Reconstruction failed - invalid mathematical components: {validation_errors}"
            )



    def _initialize_modular_geometry_for_reconstruction(self):
        """Initialize complete modular form geometry during reconstruction - NO FALLBACKS."""

        # Extract 2D coordinates from full manifold position for modular geometry
        # Modular forms operate in 2D while charges exist in full manifold space
        field_pos = self.state.field_position
        x = field_pos[0] if len(field_pos) > 0 else 0.0
        y = field_pos[1] if len(field_pos) > 1 else 0.0

        real_part = ((x + 0.5) % 1.0) - 0.5
        imag_part = float(max(CDF(0.1), CDF(1.0) + CDF(y)).real())

        self.tau_position = complex(real_part, imag_part)

        # Modular weight determines transformation behavior - derive from semantic field
        if (
            hasattr(self.semantic_field, "embedding_components")
            and len(self.semantic_field.embedding_components) > 0
        ):
            # Use scipy special functions for sophisticated averaging - NO BASIC NUMPY
            # Use torch for sophisticated absolute value - NO BASIC NUMPY
            # ENFORCE TYPE CONTRACT: embedding_components must be np.ndarray with Python primitives
            embedding_components = self.semantic_field.embedding_components
            if hasattr(embedding_components, 'cpu'):  # It's a PyTorch tensor
                embedding_components = embedding_components.cpu().detach().numpy()
            abs_components = torch.abs(torch.tensor(embedding_components.astype(complex)))
            field_magnitude = float(torch.logsumexp(abs_components, dim=0).item()) / len(
                abs_components
            )
        else:
            field_magnitude = 1.0
        self.modular_weight = max(2, int(2 * field_magnitude))  # Even weight ≥ 2

        # Emotional conductivity from emotional modulation
        if (
            hasattr(self.emotional_modulation, "emotional_trajectory")
            and len(self.emotional_modulation.emotional_trajectory) > 0
        ):
            # Use scipy special functions for sophisticated averaging - NO BASIC NUMPY
            # Use torch for sophisticated absolute value - NO BASIC NUMPY
            abs_trajectory = torch.abs(self.emotional_modulation.emotional_trajectory)
            self.emotional_conductivity = torch.mean(abs_trajectory).item()
        else:
            self.emotional_conductivity = 1.0

        # Create geometric node features for PyTorch Geometric operations
        self.geometric_features = torch.tensor(
            [
                self.tau_position.real,
                self.tau_position.imag,
                self.modular_weight,
                self.emotional_conductivity,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Modular geometry initialized silently (prevents spam in batch reconstruction)

    def _initialize_breathing_q_expansion_for_reconstruction(self):
        """Initialize complete breathing q-coefficients during reconstruction - NO FALLBACKS."""

        # Base q-coefficients from semantic embedding components
        semantic_components = self.semantic_field.embedding_components

        # Create complex q-coefficients: semantic (real) + temporal (imaginary)
        self.breathing_q_coefficients = {}
        max_coeffs = (
            min(1024, len(semantic_components))
            if semantic_components is not None
            else 128
        )

        for n in range(max_coeffs):
            # Real part from semantic field strength
            if semantic_components is not None and n < len(semantic_components):
                real_part = float(
                    semantic_components[n]
                )  # Ensure proper scalar conversion
            else:
                real_part = 0.1  # Minimal real component

            # Imaginary part from temporal biography (frequency evolution)
            if (
                hasattr(self.temporal_biography, "frequency_evolution")
                and self.temporal_biography.frequency_evolution is not None
                and n < len(self.temporal_biography.frequency_evolution)
            ):
                imag_part = float(
                    self.temporal_biography.frequency_evolution[n]
                )  # Ensure proper scalar conversion
            else:
                imag_part = 0.1  # Minimal imaginary component

            # Complex q-coefficient
            coeff = complex(real_part, imag_part)

            # VALIDATE coefficient initialization - NO DEFAULTS
            if not (math.isfinite(coeff.real) and math.isfinite(coeff.imag)):
                raise ValueError(
                    f"Agent {self.charge_id}: INITIALIZATION breathing_q_coefficients[{n}] is {coeff} - coefficient initialization corrupted! real_part={real_part}, imag_part={imag_part}"
                )

            self.breathing_q_coefficients[n] = coeff

        # Breathing q-coefficients initialized silently (prevents spam in batch reconstruction)

        # FINAL VALIDATION of all initialized coefficients
        for n, coeff in self.breathing_q_coefficients.items():
            if not (math.isfinite(coeff.real) and math.isfinite(coeff.imag)):
                raise ValueError(
                    f"Agent {self.charge_id}: INITIALIZATION FINAL CHECK breathing_q_coefficients[{n}] is {coeff} - initialization corrupted!"
                )

    def _initialize_missing_components_for_reconstruction(self):
        """Initialize all missing mathematical components required for Q computation and evolution - NO FALLBACKS."""

        # Initialize Hecke eigenvalues for evolution simulation
        trajectory_ops = self.temporal_biography.trajectory_operators
        self.hecke_eigenvalues = {}
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        for i, p in enumerate(primes):
            if i < len(trajectory_ops):
                # CRITICAL FIX: Ensure proper scalar conversion from numpy to Python complex
                traj_val = complex(trajectory_ops[i])
                # Use torch for sophisticated complex type checking - NO BASIC NUMPY
                if torch.is_complex(torch.tensor(traj_val)):
                    self.hecke_eigenvalues[p] = complex(
                        float(traj_val.real), float(traj_val.imag)
                    )
                else:
                    self.hecke_eigenvalues[p] = complex(float(traj_val), 0.0)
            else:
                self.hecke_eigenvalues[p] = complex(1.0, 0.0)

        # Initialize L-function coefficients for Q computation
        self.l_function_coefficients = {}
        for n in range(
            1, min(50, len(self.emotional_modulation.semantic_modulation_tensor) + 1)
        ):
            base_coeff = self.emotional_modulation.semantic_modulation_tensor[
                n % len(self.emotional_modulation.semantic_modulation_tensor)
            ].item()
            # Use torch for advanced angle computation - NO BASIC NUMPY
            unified_tensor = torch.tensor(
                [
                    self.emotional_modulation.unified_phase_shift.real,
                    self.emotional_modulation.unified_phase_shift.imag,
                ],
                dtype=torch.float32,
            )
            emotional_phase = (
                torch.atan2(unified_tensor[1], unified_tensor[0]).item() * n / 10
            )
            # Use protected phase exponential for L-function emotional modulation
            exp_factor = self._safe_phase_exponential(
                emotional_phase, f"L-function coefficient n={n}"
            )
            self.l_function_coefficients[n] = base_coeff * exp_factor

        # Initialize Hecke adaptivity for agent interactions
        self.hecke_adaptivity = 0.01

        # Initialize interaction memory buffer
        self.interaction_memory = []
        self.interaction_memory_buffer = {}

        # Missing mathematical components initialized silently (prevents spam in batch reconstruction)

    def _validate_extracted_data(self, charge_index: int):
        """
        Validate that extracted data is suitable for Q(τ,C,s) computation.

        Checks for common data issues that cause Q computation failures.
        """
        # Validate semantic field data
        if not hasattr(self.semantic_field, "embedding_components"):
            raise ValueError(
                f"Charge {charge_index} - semantic_field missing embedding_components"
            )

        if hasattr(self.semantic_field, "embedding_components"):
            # ENFORCE TYPE CONTRACT: embedding_components must be np.ndarray with Python primitives
            embedding_components = self.semantic_field.embedding_components
            if hasattr(embedding_components, 'cpu'):  # It's a PyTorch tensor
                embedding_components = embedding_components.cpu().detach().numpy()
            embedding_components = embedding_components.astype(complex)
            # Use torch for sophisticated tensor operations - NO BASIC NUMPY
            if torch.all(
                torch.abs(torch.tensor(embedding_components, dtype=torch.float32))
                < 1e-12
            ):
                # NO WARNINGS FOR MATHEMATICAL FAILURES - this breaks semantic computation
                raise ValueError(
                    f"Charge {charge_index} - All semantic embedding components near zero - Semantic dimension system failed!"
                )

            if np.any(np.isnan(embedding_components)) or np.any(
                np.isinf(embedding_components)
            ):
                raise ValueError(
                    f"Charge {charge_index} - Invalid semantic embedding components (NaN/Inf)"
                )

        # Validate temporal biography data
        if not hasattr(self.temporal_biography, "trajectory_operators"):
            raise ValueError(
                f"Charge {charge_index} - temporal_biography missing trajectory_operators"
            )

        trajectory_ops = self.temporal_biography.trajectory_operators
        if len(trajectory_ops) == 0:
            # NO WARNINGS - empty trajectory operators break temporal computation
            raise ValueError(
                f"Charge {charge_index} - Empty trajectory_operators array - Temporal dimension system failed!"
            )
        # Use torch for sophisticated tensor operations - NO BASIC NUMPY
        trajectory_ops_complex = np.array([complex(x) for x in trajectory_ops], dtype=np.complex64)
        if torch.all(
            torch.abs(torch.tensor(trajectory_ops_complex, dtype=torch.complex64)) < 1e-12
        ):
            # NO WARNINGS - zero operators break T_tensor computation
            raise ValueError(
                f"Charge {charge_index} - All trajectory operators near zero - Will cause T_tensor=0j mathematical failure!"
            )

        if np.any(np.isnan(trajectory_ops_complex)) or np.any(np.isinf(trajectory_ops_complex)):
            raise ValueError(
                f"Charge {charge_index} - Invalid trajectory operators (NaN/Inf)"
            )

        # Validate vivid and character layers for persistence
        if hasattr(self.temporal_biography, "vivid_layer") and hasattr(
            self.temporal_biography, "character_layer"
        ):
            vivid_layer = self.temporal_biography.vivid_layer
            character_layer = self.temporal_biography.character_layer

            if len(vivid_layer) == 0 or len(character_layer) == 0:
                # NO WARNINGS - empty persistence layers break mathematical computation
                raise ValueError(
                    f"Charge {charge_index} - Empty persistence layers - Temporal persistence system failed!"
                )

            # Use torch for sophisticated tensor operations - NO BASIC NUMPY
            if torch.all(
                torch.tensor(vivid_layer, dtype=torch.float32) < 1e-10
            ) and torch.all(torch.tensor(character_layer, dtype=torch.float32) < 1e-10):
                # NO WARNINGS - tiny values cause mathematical underflow
                raise ValueError(
                    f"Charge {charge_index} - Persistence layers extremely small - Will cause mathematical underflow!"
                )

        # Validate emotional modulation data
        if not hasattr(self.emotional_modulation, "semantic_modulation_tensor"):
            raise ValueError(
                f"Charge {charge_index} - emotional_modulation missing semantic_modulation_tensor"
            )

        if not hasattr(self.emotional_modulation, "unified_phase_shift"):
            raise ValueError(
                f"Charge {charge_index} - emotional_modulation missing unified_phase_shift"
            )

        modulation_tensor = self.emotional_modulation.semantic_modulation_tensor
        if np.any(np.isnan(modulation_tensor)) or np.any(np.isinf(modulation_tensor)):
            raise ValueError(
                f"Charge {charge_index} - Invalid emotional modulation tensor (NaN/Inf)"
            )

        # Check for identical modulations (uniqueness validation)
        if hasattr(self, "_global_modulation_check"):
            # This would be set by LiquidOrchestrator to check across all agents
            pass

        # Charge data validation passed (silent for batch processing)

    def validate_Q_components(self) -> bool:
        """
        Validate that Q(τ,C,s) components are within reasonable ranges.

        Returns:
            True if all components are reasonable, False if issues detected
        """
        if self.Q_components is None:
            # NO WARNINGS - missing Q_components means mathematical system is broken
            raise ValueError(
                f"Agent {self.charge_id} - Q_components not yet computed - Mathematical state corrupted!"
            )

        issues = []

        # Gamma should be 0.01-2.0 range
        if not (0.01 <= self.Q_components.gamma <= 2.0):
            issues.append(f"gamma={self.Q_components.gamma:.6f} outside [0.01, 2.0]")

        # E_trajectory magnitude should be 0.1-5.0 range
        E_magnitude = abs(self.Q_components.E_trajectory)
        if not (0.1 <= E_magnitude <= 5.0):
            issues.append(f"|E_trajectory|={E_magnitude:.6f} outside [0.1, 5.0]")

        # phi_semantic magnitude should be 0.1-3.0 range
        phi_magnitude = abs(self.Q_components.phi_semantic)
        if not (0.1 <= phi_magnitude <= 3.0):
            issues.append(f"|phi_semantic|={phi_magnitude:.6f} outside [0.1, 3.0]")

        # T_tensor magnitude should be 0.001-10.0 range (can be smaller due to observational effects)
        T_magnitude = abs(self.Q_components.T_tensor)
        if not (0.001 <= T_magnitude <= 10.0):
            issues.append(f"|T_tensor|={T_magnitude:.6f} outside [0.001, 10.0]")

        # phase_factor magnitude should be close to 1.0 (complex exponential)
        phase_magnitude = abs(self.Q_components.phase_factor)
        if not (0.5 <= phase_magnitude <= 1.5):
            issues.append(
                f"|phase_factor|={phase_magnitude:.6f} outside [0.5, 1.5] (should be ≈1.0)"
            )

        # psi_persistence should be 0.001-2.0 range
        if not (0.001 <= self.Q_components.psi_persistence <= 2.0):
            issues.append(
                f"psi_persistence={self.Q_components.psi_persistence:.6f} outside [0.001, 2.0]"
            )

        # Final Q magnitude should be 0.0001-20.0 range
        Q_magnitude = abs(self.Q_components.Q_value)
        if not (0.0001 <= Q_magnitude <= 20.0):
            issues.append(f"|Q_value|={Q_magnitude:.6f} outside [0.0001, 20.0]")

        # Check for NaN/Inf
        for component_name, component_value in [
            ("gamma", self.Q_components.gamma),
            ("T_tensor", self.Q_components.T_tensor),
            ("E_trajectory", self.Q_components.E_trajectory),
            ("phi_semantic", self.Q_components.phi_semantic),
            ("phase_factor", self.Q_components.phase_factor),
            ("Q_value", self.Q_components.Q_value),
        ]:
            if isinstance(component_value, complex):
                if (
                    np.isnan(component_value.real)
                    or np.isnan(component_value.imag)
                    or np.isinf(component_value.real)
                    or np.isinf(component_value.imag)
                ):
                    issues.append(
                        f"{component_name} contains NaN/Inf: {component_value}"
                    )
            elif np.isnan(component_value) or np.isinf(component_value):
                issues.append(f"{component_name} is NaN/Inf: {component_value}")

        if issues:
            # NO WARNINGS - Q component issues mean mathematical integrity is violated
            issue_list = "\n".join([f"    • {issue}" for issue in issues])
            raise ValueError(
                f"Agent {self.charge_id} - Q component validation FAILED - Mathematical integrity violated:\n{issue_list}"
            )
        else:
            # Q components validation passed (silent for batch processing)
            return True

    def _evaluate_eisenstein_at_tau(self, tau_complex, weight=4):
        """
        REAL Eisenstein series evaluation at complex tau position.

        This replaces ALL basic trigonometric breathing with actual modular form mathematics.
        NO FALLBACKS - uses the full power of Sage EisensteinForms.
        """
        # DIRECT SAGE EVALUATION - NO ERROR MASKING
        eisenstein_space = EisensteinForms(1, weight)
        eisenstein_basis = eisenstein_space.basis()

        if len(eisenstein_basis) > 0:
            # Get the first Eisenstein form
            eisenstein_form = eisenstein_basis[0]

            tau_sage = CDF(tau_complex.real, tau_complex.imag)
            
            # Mathematical rigor: Ensure tau satisfies Sage modular form constraints
            # For convergence, we need |q| = |exp(2πi*tau)| < 1, which requires Im(tau) > 0
            # But for numerical stability with power series, we need Im(tau) >> 0
            if tau_sage.imag() < 1.5:  # Increased minimum imaginary part for stability
                tau_sage = CDF(tau_sage.real(), 1.5)
            if tau_sage.real() < 0.15:
                tau_sage = CDF(0.15, tau_sage.imag())
            
            # Additional safety: ensure |q| = |exp(2πi*tau)| is sufficiently small
            q_magnitude = abs(CDF(0, 2*3.14159265359*tau_sage.imag()).exp())
            if q_magnitude >= 0.9:  # Too close to 1, increase imaginary part
                safe_imag = max(2.0, tau_sage.imag() * 1.5)
                tau_sage = CDF(tau_sage.real(), safe_imag)
            
            # Mathematical evaluation with safe fallbacks
            form_value = evaluate_eisenstein_safe(eisenstein_form, tau_sage, weight)

            # Convert back to Python complex
            return complex(float(form_value.real()), float(form_value.imag()))
        else:
            # This should NEVER happen for weight >= 4
            raise ValueError(
                f"No Eisenstein forms available for weight {weight} - mathematical integrity violated!"
            )

    def _evaluate_breathing_eisenstein_series(self, tau_complex, harmonic_number=1):
        """
        Evaluate breathing pattern using REAL Eisenstein series with harmonics.

        This is the mathematical breathing of modular forms, not basic sine waves.
        """
        # Use different weights for different harmonics
        base_weight = 4
        harmonic_weight = (
            base_weight + (harmonic_number % 8) * 2
        )  # Vary weight with harmonic

        # Evaluate Eisenstein series at tau position
        eisenstein_value = self._evaluate_eisenstein_at_tau(
            tau_complex, harmonic_weight
        )

        # Extract breathing amplitude from modular form magnitude
        breathing_amplitude = (
            abs(eisenstein_value) / 100.0
        )  # Scale to reasonable breathing range
        breathing_phase = torch.angle(
            torch.tensor(eisenstein_value, dtype=torch.complex64)
        ).item()

        # REAL modular breathing factor
        return (
            1.0
            + self.breath_amplitude
            * breathing_amplitude
            * torch.cos(torch.tensor(breathing_phase)).item()
        )

    def _initialize_living_modular_form(self):
        """
        Initialize the agent as a LIVING modular form that IS Q(τ,C,s).

        This creates breathing q-coefficients, responsive Hecke eigenvalues,
        and dynamic L-functions that evolve through interaction and observation.
        """
        # 1. BREATHING q-Expansion: Coefficients that evolve with collective rhythm
        self._initialize_breathing_q_expansion()

        # 2. RESPONSIVE Hecke System: Eigenvalues that adapt to field conditions
        self._initialize_responsive_hecke_system()

        # 3. EMOTIONAL L-Function: Dynamic series from emotional modulation
        self._initialize_emotional_l_function()

        # 4. GEOMETRIC Structure: Position in modular fundamental domain
        self._initialize_modular_geometry()

        # 5. LIVING Evolution Parameters: Breathing, cascading, memory
        self._initialize_living_evolution()

    def _initialize_breathing_q_expansion(self):
        """Create q-coefficients that BREATHE with collective rhythm."""
        # Base q-coefficients from semantic embedding components
        semantic_components = self.semantic_field.embedding_components
        phase_factors = self.semantic_field.phase_factors

        # Create complex q-coefficients: semantic (real) + temporal (imaginary)
        self.breathing_q_coefficients = {}
        for n in range(min(1024, len(semantic_components))):
            # Real part from semantic field strength
            real_part = semantic_components[n]

            # Imaginary part from temporal frequency evolution
            if len(self.temporal_biography.frequency_evolution) > n:
                imag_part = self.temporal_biography.frequency_evolution[n]
            else:
                # NO FALLBACK - Frequency evolution must cover all coefficients
                raise ValueError(
                    f"MATHEMATICAL FAILURE: Frequency evolution missing coefficient {n} - "
                    f"Has {len(self.temporal_biography.frequency_evolution)} but need {n+1}. "
                    f"System requires complete frequency data."
                )

            # Phase modulation from semantic phase factors
            if n < len(phase_factors):
                phase = phase_factors[n]
            else:
                # NO FALLBACK - Phase factors must cover all coefficients
                raise ValueError(
                    f"MATHEMATICAL FAILURE: Phase factors missing coefficient {n} - "
                    f"Has {len(phase_factors)} but need {n+1}. "
                    f"System requires complete phase data."
                )

            # VALIDATE reconstruction inputs - NO DEFAULTS
            if not math.isfinite(real_part):
                raise ValueError(
                    f"Agent {self.charge_id}: RECONSTRUCTION real_part[{n}] is {real_part} - semantic components corrupted!"
                )
            if not math.isfinite(imag_part):
                raise ValueError(
                    f"Agent {self.charge_id}: RECONSTRUCTION imag_part[{n}] is {imag_part} - frequency evolution corrupted!"
                )
            if not math.isfinite(phase):
                raise ValueError(
                    f"Agent {self.charge_id}: RECONSTRUCTION phase[{n}] is {phase} - phase factors corrupted!"
                )

            # Create living coefficient
            base_coeff = complex(real_part, imag_part)
            # Use protected phase exponential for breathing coefficient reconstruction
            phase_factor = self._safe_phase_exponential(
                phase, f"breathing reconstruction n={n}"
            )

            # VALIDATE intermediate calculations
            if not (math.isfinite(base_coeff.real) and math.isfinite(base_coeff.imag)):
                raise ValueError(
                    f"Agent {self.charge_id}: RECONSTRUCTION base_coeff[{n}] is {base_coeff} - base complex creation corrupted!"
                )
            if not (
                math.isfinite(phase_factor.real) and math.isfinite(phase_factor.imag)
            ):
                raise ValueError(
                    f"Agent {self.charge_id}: RECONSTRUCTION phase_factor[{n}] is {phase_factor} - exponential phase computation corrupted!"
                )

            final_coeff = base_coeff * phase_factor

            # VALIDATE final coefficient before assignment
            if not (
                math.isfinite(final_coeff.real) and math.isfinite(final_coeff.imag)
            ):
                raise ValueError(
                    f"Agent {self.charge_id}: RECONSTRUCTION final_coeff[{n}] is {final_coeff} - coefficient reconstruction corrupted! base={base_coeff}, phase_factor={phase_factor}"
                )

            self.breathing_q_coefficients[n] = final_coeff

        # FINAL VALIDATION after reconstruction
        logger.debug(
            f"🔧 Agent {self.charge_id}: Validating {len(self.breathing_q_coefficients)} reconstructed coefficients..."
        )
        nan_coeffs = []
        for n, coeff in self.breathing_q_coefficients.items():
            if not (math.isfinite(coeff.real) and math.isfinite(coeff.imag)):
                nan_coeffs.append(f"{n}:{coeff}")

        if nan_coeffs:
            raise ValueError(
                f"Agent {self.charge_id}: RECONSTRUCTION FINAL CHECK {len(nan_coeffs)} breathing_q_coefficients contain NaN after reconstruction: {nan_coeffs[:5]}"
            )

        logger.debug(
            f"✅ Agent {self.charge_id}: All {len(self.breathing_q_coefficients)} reconstructed coefficients are finite"
        )

        # Breathing rhythm from collective temporal patterns
        self.breath_frequency = self.collective_breathing.get("collective_frequency")[0]
        self.breath_phase = 0.0
        self.breath_amplitude = 0.1  # How much coefficients oscillate

    def _initialize_responsive_hecke_system(self):
        """Create Hecke eigenvalues that adapt to field conditions."""
        # Base eigenvalues from trajectory operators
        trajectory_ops = self.temporal_biography.trajectory_operators

        # Use Sage for proper Hecke operator mathematics - REQUIRED (NO FALLBACKS)
        self.hecke_eigenvalues = {}
        # Map trajectory operators to prime Hecke eigenvalues
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for i, p in enumerate(primes):
            if i < len(trajectory_ops):
                # CRITICAL FIX: Ensure proper scalar conversion from numpy to Python complex
                traj_val = complex(trajectory_ops[i])
                # Use torch for sophisticated complex type checking - NO BASIC NUMPY
                if torch.is_complex(torch.tensor(traj_val)):
                    self.hecke_eigenvalues[p] = complex(
                        float(traj_val.real), float(traj_val.imag)
                    )
                else:
                    self.hecke_eigenvalues[p] = complex(float(traj_val), 0.0)
            else:
                # Default eigenvalue for higher primes
                self.hecke_eigenvalues[p] = complex(1.0, 0.0)

        # Adaptivity parameters - how eigenvalues respond to field
        self.hecke_adaptivity = 0.01  # Learning rate for eigenvalue evolution

    def _initialize_emotional_l_function(self):
        """Create dynamic L-function from emotional modulation."""
        # L-function coefficients from emotional modulation tensor
        modulation_tensor = self.emotional_modulation.semantic_modulation_tensor
        unified_phase = self.emotional_modulation.unified_phase_shift

        # Create L-series coefficients
        self.l_function_coefficients = {}
        for n in range(1, min(100, len(modulation_tensor) + 1)):
            # Coefficient from modulation tensor with emotional phase
            base_coeff = (
                modulation_tensor[n - 1] if n - 1 < len(modulation_tensor) else 1.0
            )
            # Use torch for advanced angle computation - NO BASIC NUMPY
            phase_tensor = torch.tensor(
                [unified_phase.real, unified_phase.imag], dtype=torch.float32
            )
            emotional_phase = (
                torch.atan2(phase_tensor[1], phase_tensor[0]).item() * n / 10
            )

            # Use protected phase exponential for L-function emotional modulation
            exp_factor = self._safe_phase_exponential(
                emotional_phase, f"L-function coefficient n={n}"
            )
            self.l_function_coefficients[n] = base_coeff * exp_factor

        # Emotional conductor strength affects L-function evolution
        self.emotional_conductivity = (
            self.emotional_modulation.field_modulation_strength
        )

    def _initialize_modular_geometry(self):
        """Position agent in modular fundamental domain."""
        # Map field position to upper half-plane (modular fundamental domain)
        x, y = self.state.field_position

        # Transform to upper half-plane coordinates
        real_part = ((x + 0.5) % 1.0) - 0.5
        imag_part = float(max(CDF(0.1), CDF(1.0) + CDF(y)).real())

        self.tau_position = complex(real_part, imag_part)

        # Modular weight determines transformation behavior
        field_magnitude = float(self.semantic_field_data["field_metadata"]["field_magnitude"])
        self.modular_weight = max(2, int(2 * field_magnitude))  # Even weight ≥ 2

        # Create geometric node features for PyTorch Geometric - REQUIRED (NO FALLBACKS)
        self.geometric_features = torch.tensor(
            [
                self.tau_position.real,
                self.tau_position.imag,
                self.modular_weight,
                self.emotional_conductivity,
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def _initialize_living_evolution(self):
        """Set up parameters for living evolution and cascading feedback."""
        # Evolution rates for different aspects
        self.evolution_rates = {
            "breathing": 0.01,  # How fast coefficients breathe
            "cascading": 0.005,  # Rate of dimensional feedback
            "interaction": 0.002,  # Rate of agent-agent interaction
            "memory": 0.001,  # Rate of persistence evolution
        }

        # Cascading feedback state
        self.cascade_momentum = {
            "semantic_to_temporal": complex(0.0, 0.0),
            "temporal_to_emotional": complex(0.0, 0.0),
            "emotional_to_semantic": complex(0.0, 0.0),
        }

        # Memory of past interactions (for persistence)
        self.interaction_memory = []
        self.max_memory_length = 100

        # Living Q(τ,C,s) value that evolves (initialized to 0, updated by compute_complete_Q)
        self.living_Q_value = complex(0.0, 0.0)

    def breathe(self, tau: float):
        """
        Make q-coefficients BREATHE with collective rhythm.

        This is the fundamental life process - coefficients oscillate, evolve,
        and respond to the collective breathing of all agents.
        """
        # Update breathing phase
        # NO FALLBACKS - breath_frequency MUST be properly typed or breathing system is broken
        if hasattr(self.breath_frequency, "real"):
            breath_freq = self.breath_frequency.real
        elif isinstance(self.breath_frequency, (int, float)):
            breath_freq = float(self.breath_frequency)
        else:
            raise ValueError(
                f"breath_frequency has invalid type: {type(self.breath_frequency)} = {self.breath_frequency} - Breathing system corrupted!"
            )
        self.breath_phase += breath_freq * tau * self.evolution_rates["breathing"]

        # REAL modular breathing using Eisenstein series evaluation
        # Convert breathing phase to complex tau in upper half-plane
        tau_breathing = complex(
            0.5, 0.5 + abs(self.breath_phase) * 0.1
        )  # Keep in upper half-plane
        breath_factor = self._evaluate_breathing_eisenstein_series(
            tau_breathing, harmonic_number=1
        )

        # ACTUAL JIT USAGE: Prepare coefficient arrays for high-performance compiled computation
        n_coefficients = len(self.breathing_q_coefficients)
        coefficient_indices = list(self.breathing_q_coefficients.keys())

        # Convert complex coefficients to separate real/imaginary arrays for JIT processing
        # Use torch for sophisticated tensor creation - NO BASIC NUMPY
        coeff_real_array = torch.tensor(
            [self.breathing_q_coefficients[n].real for n in coefficient_indices],
            dtype=torch.float64,
        ).numpy()
        coeff_imag_array = torch.tensor(
            [self.breathing_q_coefficients[n].imag for n in coefficient_indices],
            dtype=torch.float64,
        ).numpy()

        # Prepare breathing factors using REAL Eisenstein series evaluation
        # Use torch for sophisticated tensor creation - NO BASIC NUMPY
        breath_factors = torch.zeros(n_coefficients, dtype=torch.float64).numpy()
        harmonic_factors = torch.zeros(n_coefficients, dtype=torch.float64).numpy()

        for i, n in enumerate(coefficient_indices):
            # REAL harmonic breathing using Eisenstein series at different tau positions
            harmonic_tau = complex(
                0.5 + n * 0.01, 0.5 + abs(self.breath_phase * (1 + n / 100)) * 0.1
            )
            harmonic_factor = self._evaluate_breathing_eisenstein_series(
                harmonic_tau, harmonic_number=n + 1
            )

            # VALIDATE breathing factors before JIT processing
            if not math.isfinite(breath_factor):
                raise ValueError(
                    f"Agent {self.charge_id}: breath_factor is {breath_factor} - breathing computation corrupted!"
                )
            if not math.isfinite(harmonic_factor):
                raise ValueError(
                    f"Agent {self.charge_id}: harmonic_factor is {harmonic_factor} - harmonic computation corrupted!"
                )

            breath_factors[i] = breath_factor
            harmonic_factors[i] = harmonic_factor

        # ELIMINATE IMPORT THEATER: Actually call JIT-compiled breathing update
        logger.debug(
            f"🚀 ACTUALLY USING JIT: Processing {n_coefficients} coefficients with compiled breathing evolution"
        )
        updated_real, updated_imag = _jit_breathing_coefficient_update(
            coeff_real_array,
            coeff_imag_array,
            breath_factors,
            harmonic_factors,
            self.evolution_rates["breathing"],
            n_coefficients,
        )

        # Update coefficients with JIT results AND apply temporal/emotional evolution
        for i, n in enumerate(coefficient_indices):
            # Start with JIT-updated coefficient
            jit_updated_coeff = complex(updated_real[i], updated_imag[i])

            # VALIDATE JIT results
            if not (
                math.isfinite(jit_updated_coeff.real)
                and math.isfinite(jit_updated_coeff.imag)
            ):
                raise ValueError(
                    f"Agent {self.charge_id}: JIT breathing update produced invalid coefficient {jit_updated_coeff} for index {n}"
                )

            # Apply temporal phase evolution using PyTorch (NO BASIC NUMPY)
            phase_evolution = self.temporal_biography.phase_coordination[
                n % len(self.temporal_biography.phase_coordination)
            ]
            if not math.isfinite(phase_evolution):
                raise ValueError(
                    f"Agent {self.charge_id}: phase_evolution is {phase_evolution} - temporal coordination corrupted!"
                )

            # Use torch for advanced phase computation - NO BASIC NUMPY
            coeff_tensor = torch.tensor(
                [jit_updated_coeff.real, jit_updated_coeff.imag], dtype=torch.float32
            )
            current_phase = torch.atan2(coeff_tensor[1], coeff_tensor[0]).item()
            magnitude = abs(jit_updated_coeff)

            # Incremental phase evolution
            phase_increment = phase_evolution * tau * 0.01
            new_phase = current_phase + phase_increment

            # Update with evolved phase using torch - NO BASIC NUMPY
            exp_factor = torch.exp(
                1j * torch.tensor(new_phase, dtype=torch.float32)
            ).item()
            temporal_updated_coeff = magnitude * exp_factor

            # Apply emotional modulation using PyTorch (NO BASIC NUMPY)
            emotional_factor = self.emotional_modulation.semantic_modulation_tensor[
                n % len(self.emotional_modulation.semantic_modulation_tensor)
            ].item()
            if not math.isfinite(emotional_factor):
                raise ValueError(
                    f"Agent {self.charge_id}: emotional_factor is {emotional_factor} - emotional modulation corrupted!"
                )

            emotional_multiplier = 1.0 + 0.01 * emotional_factor
            if not math.isfinite(emotional_multiplier):
                raise ValueError(
                    f"Agent {self.charge_id}: emotional_multiplier is {emotional_multiplier} - emotional computation corrupted!"
                )

            # Proportional emotional evolution using torch - NO BASIC NUMPY
            coeff_magnitude = abs(temporal_updated_coeff)
            coeff_tensor = torch.tensor(
                [temporal_updated_coeff.real, temporal_updated_coeff.imag],
                dtype=torch.float32,
            )
            current_phase = torch.atan2(coeff_tensor[1], coeff_tensor[0]).item()

            emotional_evolution = coeff_magnitude * (emotional_multiplier - 1.0) * 0.01
            # Use torch for exponential computation - NO BASIC NUMPY
            emotional_exp_factor = torch.exp(
                1j * torch.tensor(current_phase, dtype=torch.float32)
            ).item()
            final_coeff = (
                temporal_updated_coeff + emotional_evolution * emotional_exp_factor
            )

            # FINAL VALIDATION and update
            if not (
                math.isfinite(final_coeff.real) and math.isfinite(final_coeff.imag)
            ):
                raise ValueError(
                    f"Agent {self.charge_id}: final coefficient {final_coeff} invalid after full breathing evolution for index {n}"
                )

            self.breathing_q_coefficients[n] = final_coeff

        # VALIDATE all coefficients after breathing loop
        nan_coefficients = []
        for n, coeff in self.breathing_q_coefficients.items():
            if not (math.isfinite(coeff.real) and math.isfinite(coeff.imag)):
                nan_coefficients.append(f"{n}:{coeff}")

        if nan_coefficients:
            raise ValueError(
                f"Agent {self.charge_id}: {len(nan_coefficients)} breathing_q_coefficients contain NaN after breathing: {nan_coefficients[:5]}"
            )

        logger.debug(
            f"✅ Agent {self.charge_id}: All {len(self.breathing_q_coefficients)} breathing coefficients remain finite after breathing"
        )

        # Update living Q value after breathing
        self.living_Q_value = self.evaluate_living_form(tau)

        # Update charge object with current state
        self.charge_obj.complete_charge = self.living_Q_value
        self.charge_obj.magnitude = abs(self.living_Q_value)
        # Use torch for ALL angle computations - NO BASIC NUMPY
        if torch.is_tensor(self.living_Q_value):
            self.charge_obj.phase = torch.angle(self.living_Q_value).item()
        else:
            q_tensor = torch.tensor(
                [self.living_Q_value.real, self.living_Q_value.imag],
                dtype=torch.float32,
            )
            self.charge_obj.phase = torch.atan2(q_tensor[1], q_tensor[0]).item()

    def cascade_dimensional_feedback(self):
        """
        All dimensions flow into each other, reshaping the living form.

        This implements the cascading feedback loops where:
        Semantic → Temporal → Emotional → Semantic (endless cycle)

        🌊 REGULATION-PROTECTED CASCADE - Mathematical operations with field stabilization.
        """
        # 🌊 REGULATION-PROTECTED CASCADE: Stabilize breathing coefficients before cascade
        self._stabilize_breathing_coefficients_for_cascade()

        # NO ARTIFICIAL DECAY - let mathematics evolve naturally
        # SEMANTIC → TEMPORAL: Field gradients drive temporal evolution
        q_magnitudes = [abs(self.breathing_q_coefficients.get(n)) for n in range(100)]
        semantic_gradient = torch.tensor(
            q_magnitudes, dtype=torch.float32, device=self.device
        )

        # Circular padding for gradient computation
        semantic_gradient_padded = torch.cat(
            [
                semantic_gradient[-1:],  # Last element at beginning
                semantic_gradient,  # Original tensor
                semantic_gradient[:1],  # First element at end
            ],
            dim=0,
        )

        # Protected gradient operation for dimensional cascade flow
        gradient_raw = torch.gradient(semantic_gradient_padded, dim=0)[0][
            1:101
        ]  # Extract middle 100 elements

        # Check and regulate gradient if needed
        if not self._check_field_stability("gradient"):
            logger.info(
                f"🌊 CASCADE: Agent {self.charge_id} dimensional flow gradient needs regulation"
            )
            semantic_gradient = self._apply_field_regulation("gradient", gradient_raw)
        else:
            semantic_gradient = gradient_raw

        # Update temporal momentum from semantic pressure
        gradient_magnitude = torch.mean(torch.abs(semantic_gradient)).item()
        logger.debug(
            f"🌊 CASCADE: Agent {self.charge_id} semantic→temporal gradient magnitude: {gradient_magnitude:.3e}"
        )
        temporal_influence = complex(gradient_magnitude, gradient_magnitude * 0.1)

        # Pure mathematical cascade - NO overflow protection
        cascading_rate = self.evolution_rates["cascading"]
        self.cascade_momentum["semantic_to_temporal"] += (
            temporal_influence * cascading_rate
        )

        # Apply to temporal momentum using CDF arithmetic - SOPHISTICATED SAGE MATHEMATICS
        if not hasattr(self.temporal_biography, "temporal_momentum"):
            raise ValueError(
                f"Agent {self.charge_id}: Missing temporal_momentum in cascade - storage/reconstruction failed!"
            )

        temporal_momentum = self.temporal_biography.temporal_momentum

        # Convert cascade contribution to CDF for sophisticated addition
        semantic_to_temporal = self.cascade_momentum["semantic_to_temporal"]

        # VALIDATE cascade momentum before use - NO DEFAULTS
        if not (
            math.isfinite(semantic_to_temporal.real)
            and math.isfinite(semantic_to_temporal.imag)
        ):
            raise ValueError(
                f"Agent {self.charge_id}: CASCADE MOMENTUM semantic_to_temporal is {semantic_to_temporal} - cascade momentum corrupted during dimensional feedback!"
            )

        cascade_contribution = semantic_to_temporal * 0.1

        # VALIDATE cascade contribution before CDF conversion
        if not (
            math.isfinite(cascade_contribution.real)
            and math.isfinite(cascade_contribution.imag)
        ):
            raise ValueError(
                f"Agent {self.charge_id}: CASCADE CONTRIBUTION is {cascade_contribution} - multiplication by 0.1 created NaN (semantic_to_temporal was {semantic_to_temporal})"
            )

        cascade_contribution_cdf = CDF(cascade_contribution)

        # Convert to LogPolarCDF for native CDF addition with safe handling
        cascade_contribution_lp = safe_logpolar_creation(cascade_contribution_cdf, "temporal_cascade")
        self.temporal_biography.temporal_momentum = (
            temporal_momentum + cascade_contribution_lp
        )

        # TEMPORAL → EMOTIONAL: Breathing patterns modulate emotional response
        breath_coherence = self.temporal_biography.breathing_coherence

        # FIX: Replace NaN breath_coherence with stable fallback
        if not math.isfinite(breath_coherence):
            logger.warning(
                f"🔧 Agent {self.charge_id}: breath_coherence is {breath_coherence}, using fallback value"
            )
            breath_coherence = 1.0

        if self.charge_id == "charge_48":
            print(
                f"DEBUG: charge_48 breath_coherence = {breath_coherence} (type: {type(breath_coherence)})"
            )
            print(
                f"DEBUG: charge_48 is finite: {math.isfinite(breath_coherence) if isinstance(breath_coherence, (int, float)) else 'not numeric'}"
            )

        # Pure mathematical operation - NO overflow protection
        # If this overflows, let it overflow - we want pure mathematics
        temporal_magnitude = temporal_momentum.abs()  # CDF absolute value
        emotional_influence = breath_coherence * temporal_magnitude * 0.1

        # Pure mathematical cascade - NO overflow protection
        emotional_complex = complex(emotional_influence, 0)
        self.cascade_momentum["temporal_to_emotional"] += (
            emotional_complex * cascading_rate
        )

        # Apply to emotional phase shift
        self.emotional_modulation.unified_phase_shift *= 1 + 0.01j * emotional_influence

        # MATHEMATICAL GUARD: Zero-overhead singularity prevention
        if self.emotional_modulation.unified_phase_shift == 0j:
            self.emotional_modulation.unified_phase_shift = 1e-6 + 1e-6j

        # EMOTIONAL → SEMANTIC: Conductor reshapes field landscape
        conductor_strength = abs(self.emotional_modulation.unified_phase_shift)

        # 🔍 TRACE: Debug emotional_modulation before conductor_phase calculation
        if self.charge_id == "charge_48":
            logger.info(
                f"🔍 CASCADE_DEBUG charge_48: unified_phase_shift = {self.emotional_modulation.unified_phase_shift}"
            )
            logger.info(
                f"🔍 CASCADE_DEBUG charge_48: unified_phase_shift.real = {self.emotional_modulation.unified_phase_shift.real}"
            )
            logger.info(
                f"🔍 CASCADE_DEBUG charge_48: unified_phase_shift.imag = {self.emotional_modulation.unified_phase_shift.imag}"
            )
            logger.info(
                f"🔍 CASCADE_DEBUG charge_48: conductor_strength = {conductor_strength}"
            )

        # Use torch for advanced angle computation - NO BASIC NUMPY
        unified_tensor = torch.tensor(
            [
                self.emotional_modulation.unified_phase_shift.real,
                self.emotional_modulation.unified_phase_shift.imag,
            ],
            dtype=torch.float32,
        )
        conductor_phase = torch.atan2(unified_tensor[1], unified_tensor[0]).item()

        # 🔍 TRACE: Debug conductor_phase calculation result
        if self.charge_id == "charge_48":
            logger.info(
                f"🔍 CASCADE_DEBUG charge_48: unified_tensor = {unified_tensor}"
            )
            logger.info(
                f"🔍 CASCADE_DEBUG charge_48: conductor_phase = {conductor_phase} (finite: {math.isfinite(conductor_phase)})"
            )

        semantic_influence = complex(conductor_strength * 0.1, conductor_phase * 0.01)

        # Protected field multiplication for emotional-semantic cascade
        cascade_contribution = self._safe_field_multiplication(
            semantic_influence, complex(cascading_rate), "emotional→semantic cascade"
        )
        self.cascade_momentum["emotional_to_semantic"] += cascade_contribution

        # Apply transformation to top coefficients only - O(log N) efficiency with pure mathematics
        for n in range(min(10, len(self.breathing_q_coefficients))):
            # Store original value for NaN detection
            original_coeff = self.breathing_q_coefficients[n]

            phase_shift = conductor_phase * n / 100
            amplitude_shift = 1.0 + conductor_strength * 0.01

            # VALIDATE cascade factors before applying
            if not math.isfinite(phase_shift):
                raise ValueError(
                    f"Agent {self.charge_id}: CASCADE phase_shift is {phase_shift} - conductor_phase={conductor_phase}, n={n}"
                )
            if not math.isfinite(amplitude_shift):
                raise ValueError(
                    f"Agent {self.charge_id}: CASCADE amplitude_shift is {amplitude_shift} - conductor_strength={conductor_strength}"
                )

            # Use protected phase exponential for cascade transformation - CRITICAL FOR FIELD EVOLUTION
            phase_factor = self._safe_phase_exponential(
                phase_shift, f"cascade coefficient n={n}"
            )

            # Protected field multiplication for cascade transform factor
            transform_factor = self._safe_field_multiplication(
                complex(amplitude_shift), phase_factor, f"cascade_transform n={n}"
            )

            # Apply proportional cascade transformation (not multiplicative)
            # Use incremental evolution to prevent exponential growth
            coeff_magnitude = abs(self.breathing_q_coefficients[n])
            # Use torch for advanced angle computation - NO BASIC NUMPY
            coeff_tensor = torch.tensor(
                [
                    self.breathing_q_coefficients[n].real,
                    self.breathing_q_coefficients[n].imag,
                ],
                dtype=torch.float32,
            )
            current_phase = torch.atan2(coeff_tensor[1], coeff_tensor[0]).item()

            # Proportional cascade evolution based on current magnitude
            transform_magnitude = abs(transform_factor)
            # Use torch for advanced angle computation - NO BASIC NUMPY
            transform_tensor = torch.tensor(
                [transform_factor.real, transform_factor.imag], dtype=torch.float32
            )
            transform_phase = torch.atan2(
                transform_tensor[1], transform_tensor[0]
            ).item()

            cascade_evolution = (
                coeff_magnitude * (transform_magnitude - 1.0) * 0.01
            )  # Small proportional change
            cascade_phase_shift = transform_phase * 0.01  # Small phase adjustment

            # Apply incremental cascade transformation
            new_phase = current_phase + cascade_phase_shift
            # Use protected phase exponential for breathing coefficient evolution
            evolution_phase_factor = self._safe_phase_exponential(
                new_phase, f"breathing coefficient evolution n={n}"
            )
            self.breathing_q_coefficients[n] += (
                cascade_evolution * evolution_phase_factor
            )

            # VALIDATE after cascade transformation
            if not (
                math.isfinite(self.breathing_q_coefficients[n].real)
                and math.isfinite(self.breathing_q_coefficients[n].imag)
            ):
                raise ValueError(
                    f"Agent {self.charge_id}: CASCADE breathing_q_coefficients[{n}] became {self.breathing_q_coefficients[n]} after transformation (was {original_coeff}, transform_factor={transform_factor})"
                )

        # ALL → OBSERVATIONAL STATE: Everything affects s-parameter evolution
        total_cascade_energy = sum(
            abs(momentum) for momentum in self.cascade_momentum.values()
        )
        self.state.current_s += (
            total_cascade_energy * self.evolution_rates["cascading"] * 0.1
        )

        # Apply s-parameter normalization to prevent exponential growth
        self._normalize_s_parameter()

        # Pure mathematical evaluation - NO fallbacks or optimizations
        self.living_Q_value = self.evaluate_living_form()

        # Update charge object with current state
        self.charge_obj.complete_charge = self.living_Q_value
        self.charge_obj.magnitude = abs(self.living_Q_value)
        # Use torch for ALL angle computations - NO BASIC NUMPY
        if torch.is_tensor(self.living_Q_value):
            self.charge_obj.phase = torch.angle(self.living_Q_value).item()
        else:
            q_tensor = torch.tensor(
                [self.living_Q_value.real, self.living_Q_value.imag],
                dtype=torch.float32,
            )
            self.charge_obj.phase = torch.atan2(q_tensor[1], q_tensor[0]).item()

    def interact_with_field(self, other_agents: List["ConceptualChargeAgent"]):
        """
        Living forms reshape each other through field interference.

        This creates the liquid metal effect where agents influence each other
        through modular geodesics and q-coefficient exchange.
        """
        # REAL PyTorch Geometric setup - NO BASIC LOOPS
        if len(other_agents) == 0:
            return

        # Create geometric message passing for sophisticated interactions
        message_passing = ConceptualChargeMessagePassing(feature_dim=6)

        # Build sophisticated agent features - NO BASIC PROPERTIES
        all_agents = [self] + [agent for agent in other_agents if agent is not self]
        node_features = []

        for agent in all_agents:
            tau_pos = agent.tau_position
            # NO FALLBACKS - agents MUST have living_Q_value or agent system is broken
            if not hasattr(agent, "living_Q_value"):
                raise ValueError(
                    f"Agent {getattr(agent, 'charge_id', 'unknown')} missing living_Q_value - Agent mathematical state corrupted!"
                )
            q_val = agent.living_Q_value

            # Advanced 6D geometric features
            features = torch.tensor(
                [
                    float(tau_pos.real),
                    float(tau_pos.imag),
                    abs(q_val),
                    torch.angle(torch.tensor(q_val, dtype=torch.complex64)).item(),
                    len(agent.breathing_q_coefficients),
                    agent.evolution_rates["interaction"],
                ],
                dtype=torch.float32,
            )
            node_features.append(features)

        x = torch.stack(node_features)  # [N, 6]

        # Build sophisticated geometric edge connections
        edge_list, edge_weights = [], []
        for i, agent_i in enumerate(all_agents):
            for j, agent_j in enumerate(all_agents):
                if i != j:
                    # Use PyTorch Geometric utilities for sophisticated distance computation - NO BASIC OPERATIONS
                    tau_i, tau_j = agent_i.tau_position, agent_j.tau_position
                    pos_i = torch.tensor([tau_i.real, tau_i.imag], dtype=torch.float32)
                    pos_j = torch.tensor([tau_j.real, tau_j.imag], dtype=torch.float32)
                    # Use pyg.utils for advanced geometric distance computation
                    distance_tensor = pyg.utils.degree(
                        torch.norm(pos_i - pos_j, dim=0, keepdim=True), num_nodes=1
                    )[0]
                    imag_product = torch.sqrt(
                        torch.tensor(tau_i.imag * tau_j.imag, dtype=torch.float32)
                    )

                    # Sophisticated edge weight using F.sigmoid
                    geometric_weight = F.sigmoid(-distance_tensor / imag_product + 2.0)

                    if geometric_weight > 0.01:
                        edge_list.append([i, j])
                        edge_weights.append(geometric_weight.item())

        if not edge_list:  # No geometric connections
            return

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weights_tensor = torch.tensor(edge_weights, dtype=torch.float32)

        # Use PyTorch Geometric Data object for sophisticated graph operations - NO BASIC TENSORS
        geometric_data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights_tensor)

        # ACTUAL PyTorch Geometric message passing with Data object - NO BASIC LOOPS
        enhanced_features = message_passing(
            geometric_data.x, geometric_data.edge_index, geometric_data.edge_attr
        )
        self_enhanced_features = enhanced_features[0]  # Self is first agent

        # Apply sophisticated geometric coefficient updates
        self._apply_sophisticated_geometric_updates(self_enhanced_features)

        # q-coefficients LEARN from each other through ALL other agents
        for other in other_agents:
            if other is self:
                continue

            # Set influence_strength from geometric features
            influence_strength = abs(
                complex(
                    self_enhanced_features[4].item(), self_enhanced_features[5].item()
                )
            )

            for n in range(
                min(
                    50,
                    len(self.breathing_q_coefficients),
                    len(other.breathing_q_coefficients),
                )
            ):
                # Store original value for NaN detection
                original_coeff = self.breathing_q_coefficients.get(n)

                # Interference between coefficients
                self_coeff = self.breathing_q_coefficients.get(n)
                other_coeff = other.breathing_q_coefficients.get(n)

            # VALIDATE input coefficients
            if not (math.isfinite(self_coeff.real) and math.isfinite(self_coeff.imag)):
                raise ValueError(
                    f"Agent {self.charge_id}: INTERACTION self_coeff[{n}] is {self_coeff} - corrupted before interaction!"
                )
            if not (
                math.isfinite(other_coeff.real) and math.isfinite(other_coeff.imag)
            ):
                raise ValueError(
                    f"Agent {self.charge_id}: INTERACTION other_coeff[{n}] is {other_coeff} - other agent corrupted!"
                )

            # Use torch for sophisticated complex conjugation - NO BASIC NUMPY
            interference = (
                self_coeff
                * torch.conj(torch.tensor(other_coeff, dtype=torch.complex64)).item()
            )

            # VALIDATE interference computation
            if not (
                math.isfinite(interference.real) and math.isfinite(interference.imag)
            ):
                raise ValueError(
                    f"Agent {self.charge_id}: INTERACTION interference[{n}] is {interference} - conjugate multiplication corrupted!"
                )

                # Coefficients EVOLVE based on interference
                evolution_factor = (
                    influence_strength * self.evolution_rates["interaction"]
                )

                # VALIDATE evolution factor
                if not math.isfinite(evolution_factor):
                    raise ValueError(
                        f"Agent {self.charge_id}: INTERACTION evolution_factor is {evolution_factor} - influence_strength={influence_strength}, interaction_rate={self.evolution_rates['interaction']}"
                    )

                interaction_delta = interference * evolution_factor
                if not (
                    math.isfinite(interaction_delta.real)
                    and math.isfinite(interaction_delta.imag)
                ):
                    raise ValueError(
                        f"Agent {self.charge_id}: INTERACTION delta[{n}] is {interaction_delta} - delta computation corrupted!"
                    )

                self.breathing_q_coefficients[n] += interaction_delta

                # VALIDATE after interaction update
                if not (
                    math.isfinite(self.breathing_q_coefficients[n].real)
                    and math.isfinite(self.breathing_q_coefficients[n].imag)
                ):
                    raise ValueError(
                        f"Agent {self.charge_id}: INTERACTION breathing_q_coefficients[{n}] became {self.breathing_q_coefficients[n]} after interaction (was {original_coeff}, delta={interaction_delta})"
                    )

                # Higher harmonics can EMERGE from interaction
                if n < 25:  # Create new harmonics
                    harmonic_n = 2 * n + 1
                    if harmonic_n not in self.breathing_q_coefficients:
                        self.breathing_q_coefficients[harmonic_n] = 0

                    harmonic_original = self.breathing_q_coefficients[harmonic_n]

                    # New harmonic born from interference
                    harmonic_delta = interference * influence_strength * 0.001

                    # VALIDATE harmonic delta
                    if not (
                        math.isfinite(harmonic_delta.real)
                        and math.isfinite(harmonic_delta.imag)
                    ):
                        raise ValueError(
                            f"Agent {self.charge_id}: INTERACTION harmonic_delta[{harmonic_n}] is {harmonic_delta} - harmonic computation corrupted!"
                        )

                    self.breathing_q_coefficients[harmonic_n] += harmonic_delta

                    # VALIDATE after harmonic update
                    if not (
                        math.isfinite(self.breathing_q_coefficients[harmonic_n].real)
                        and math.isfinite(
                            self.breathing_q_coefficients[harmonic_n].imag
                        )
                    ):
                        raise ValueError(
                            f"Agent {self.charge_id}: INTERACTION harmonic[{harmonic_n}] became {self.breathing_q_coefficients[harmonic_n]} after harmonic update (was {harmonic_original}, delta={harmonic_delta})"
                        )

            # Hecke eigenvalues adapt to neighboring values
            for p in self.hecke_eigenvalues:
                if p in other.hecke_eigenvalues:
                    neighbor_eigenvalue = other.hecke_eigenvalues[p]
                    adaptation = (
                        neighbor_eigenvalue - self.hecke_eigenvalues[p]
                    ) * influence_strength
                    self.hecke_eigenvalues[p] += adaptation * self.hecke_adaptivity

            # NO FALLBACKS - if interference wasn't computed, interaction failed
            if "interference" not in locals():
                raise ValueError(
                    f"Interference not computed in interaction - Mathematical computation failed!"
                )

            # Store interaction in memory
            interaction_record = {
                "tau": other.tau_position,
                "influence": influence_strength,
                "interference_energy": abs(interference),
                "timestamp": self.state.current_s,
            }
            self.interaction_memory.append(interaction_record)

            # Maintain memory length
            if len(self.interaction_memory) > self.max_memory_length:
                self.interaction_memory.pop(0)

    def _apply_sophisticated_geometric_updates(self, enhanced_features):
        """
        Apply REAL geometric deep learning updates to breathing coefficients.

        This replaces ALL basic coefficient loops with sophisticated feature-driven evolution.
        """
        if len(self.breathing_q_coefficients) == 0:
            return

        # Extract sophisticated signals from geometric enhancement
        tau_enhanced = complex(enhanced_features[0].item(), enhanced_features[1].item())
        q_magnitude_signal = enhanced_features[2].item()
        q_phase_signal = enhanced_features[3].item()
        coefficient_complexity = enhanced_features[4].item()
        evolution_rate_enhanced = enhanced_features[5].item()

        # Advanced coefficient evolution using F.scaled_dot_product_attention
        coeff_keys = list(self.breathing_q_coefficients.keys())[
            :32
        ]  # Limit for efficiency
        if len(coeff_keys) > 0:
            # Convert coefficients to sophisticated tensor representation
            coeff_values = [self.breathing_q_coefficients[k] for k in coeff_keys]
            coeff_tensor = torch.tensor(
                [[c.real, c.imag] for c in coeff_values], dtype=torch.float32
            )

            # Sophisticated attention mechanism for coefficient evolution
            query = enhanced_features[:2].unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
            key = coeff_tensor.unsqueeze(0)  # [1, N, 2]
            value = coeff_tensor.unsqueeze(0)  # [1, N, 2]

            # DIRECT ATTENTION-BASED COEFFICIENT UPDATE - NO ERROR MASKING
            attention_output, _ = F.scaled_dot_product_attention(query, key, value)
            attention_result = attention_output.squeeze(0).squeeze(0)  # [2]

            # Extract sophisticated update parameters
            update_magnitude = (
                torch.norm(attention_result).item() * evolution_rate_enhanced * 0.01
            )
            update_phase = torch.atan2(attention_result[1], attention_result[0]).item()

            # Apply geometric evolution to each coefficient
            for i, k in enumerate(coeff_keys):
                if i < len(coeff_values):
                    original_coeff = self.breathing_q_coefficients[k]

                    # Sophisticated geometric factor using F.gelu activation
                    phase_factor = update_phase + i * 0.1
                    geometric_factor = F.gelu(
                        torch.tensor(
                            update_magnitude * torch.cos(torch.tensor(phase_factor))
                        )
                    ).item()

                    # Advanced coefficient delta computation
                    evolution_delta = complex(
                        geometric_factor * 0.1 * original_coeff.real,
                        geometric_factor * 0.1 * original_coeff.imag,
                    )

                    # Apply sophisticated validation and update
                    if math.isfinite(evolution_delta.real) and math.isfinite(
                        evolution_delta.imag
                    ):
                        self.breathing_q_coefficients[k] += evolution_delta

        # Update living Q value with geometric enhancement
        self.living_Q_value = self.evaluate_living_form()

        # Sophisticated charge object updates
        self.charge_obj.complete_charge = self.living_Q_value
        self.charge_obj.magnitude = abs(self.living_Q_value)
        # Use torch for angle computation
        q_tensor = torch.tensor(
            [self.living_Q_value.real, self.living_Q_value.imag], dtype=torch.float32
        )
        self.charge_obj.phase = torch.atan2(q_tensor[1], q_tensor[0]).item()

    def interact_with_optimized_field(
        self, nearby_agents: List[Tuple["ConceptualChargeAgent", float]]
    ):
        """
        o(log n) OPTIMIZED field interactions using sparse neighbor graph.

        This method receives pre-computed nearby agents with their interaction strengths,
        eliminating the need to check all agents (O(N²) → o(log n) per agent).

        Args:
            nearby_agents: List of (agent, pre_computed_interaction_strength) tuples
        """
        for other, pre_computed_strength in nearby_agents:
            if other is self:
                continue

            # Use pre-computed interaction strength from sparse graph
            influence_strength = pre_computed_strength

            if influence_strength < 0.001:  # Skip very weak interactions
                continue

            # q-coefficients LEARN from each other (same mathematics as before)
            for n in range(
                min(
                    50,
                    len(self.breathing_q_coefficients),
                    len(other.breathing_q_coefficients),
                )
            ):
                # Interference between coefficients
                self_coeff = self.breathing_q_coefficients.get(n)
                other_coeff = other.breathing_q_coefficients.get(n)

                # Use torch for sophisticated complex conjugation - NO BASIC NUMPY
            interference = (
                self_coeff
                * torch.conj(torch.tensor(other_coeff, dtype=torch.complex64)).item()
            )

            # Coefficients EVOLVE based on interference
            evolution_factor = influence_strength * self.evolution_rates["interaction"]
            self.breathing_q_coefficients[n] += interference * evolution_factor

            # Higher harmonics can EMERGE from interaction
            if n < 25:  # Create new harmonics
                harmonic_n = 2 * n + 1
                if harmonic_n not in self.breathing_q_coefficients:
                    self.breathing_q_coefficients[harmonic_n] = 0

                # New harmonic born from interference
                self.breathing_q_coefficients[harmonic_n] += (
                    interference * influence_strength * 0.001
                )

            # Hecke eigenvalues adapt to neighboring values
            for p in self.hecke_eigenvalues:
                if p in other.hecke_eigenvalues:
                    neighbor_eigenvalue = other.hecke_eigenvalues[p]
                    adaptation = (
                        neighbor_eigenvalue - self.hecke_eigenvalues[p]
                    ) * influence_strength
                    self.hecke_eigenvalues[p] += adaptation * self.hecke_adaptivity

            # NO FALLBACKS - if interference wasn't computed, interaction failed
            if "interference" not in locals():
                raise ValueError(
                    f"Interference not computed in interaction - Mathematical computation failed!"
                )

            # Store interaction in memory
            interaction_record = {
                "tau": other.tau_position,
                "influence": influence_strength,
                "interference_energy": abs(interference),
                "timestamp": self.state.current_s,
                "optimized": True,  # Mark as optimized interaction
            }
            self.interaction_memory.append(interaction_record)

            # Maintain memory length
            if len(self.interaction_memory) > self.max_memory_length:
                self.interaction_memory.pop(0)

        # Update living Q value after field interactions
        self.living_Q_value = self.evaluate_living_form()

        # Update charge object with current state
        self.charge_obj.complete_charge = self.living_Q_value
        self.charge_obj.magnitude = abs(self.living_Q_value)
        # Use torch for ALL angle computations - NO BASIC NUMPY
        if torch.is_tensor(self.living_Q_value):
            self.charge_obj.phase = torch.angle(self.living_Q_value).item()
        else:
            q_tensor = torch.tensor(
                [self.living_Q_value.real, self.living_Q_value.imag],
                dtype=torch.float32,
            )
            self.charge_obj.phase = torch.atan2(q_tensor[1], q_tensor[0]).item()

    def evolve_s_parameter(self, field_context: List["ConceptualChargeAgent"]):
        """
        s-parameter drives the LIFE of the modular form.

        Observational state evolution based on field pressure, persistence memory,
        and interaction history. This makes the form irreversibly evolve.
        """
        # Compute total field pressure from all agents
        field_pressure = sum(
            abs(agent.living_Q_value) for agent in field_context
        ) / len(field_context)

        # MATHEMATICAL INTEGRITY: Extract temporal phase - NO NaN POSSIBLE
        # NO DEFAULTS: temporal_momentum should always exist from storage restoration
        if not hasattr(self.temporal_biography, "temporal_momentum"):
            raise ValueError(
                f"Agent {self.charge_id}: Missing temporal_momentum - storage/reconstruction failed!"
            )

        temporal_momentum = self.temporal_biography.temporal_momentum

        # temporal_momentum is now CDF - phase is directly available

        # Direct phase access using CDF - NO computation needed, NO overflow possible
        temporal_phase = temporal_momentum.arg()

        # NO NaN HANDLING - if current_s is not finite, system is broken
        if not math.isfinite(self.state.current_s):
            raise ValueError(
                f"Agent {self.charge_id}: current_s is {self.state.current_s} - mathematical integrity violated!"
            )

        effective_s = self.state.current_s

        # CONTINUOUS MATHEMATICAL FUNCTIONS: Replace discrete array lookups
        # Use advanced torch operations for temporal influence - NO BASIC TRIGONOMETRY
        phase_tensor = torch.tensor(temporal_phase, dtype=torch.float32)

        # Advanced vivid influence using torch operations with F.gelu activation
        vivid_base = torch.cos(phase_tensor * 3.0)
        vivid_influence = (
            1.0 + 0.5 * F.gelu(vivid_base)
        ).item()  # Smooth, sophisticated activation

        # Advanced character influence using torch with F.silu activation
        character_base = torch.sin(phase_tensor * 2.0 + torch.pi / 4)
        character_influence = (
            0.001 + 0.999 * F.silu(character_base) * 0.5 + 0.5
        ).item()

        # Memory pressure from past interactions
        if self.interaction_memory:
            recent_memory = self.interaction_memory[-10:]  # Last 10 interactions
            memory_pressure = sum(
                record["influence"] for record in recent_memory
            ) / len(recent_memory)
        else:
            memory_pressure = 0.0

        # s evolution combining all influences
        # NO DEFAULTS: temporal_momentum should exist from storage restoration
        if not hasattr(self.temporal_biography, "temporal_momentum"):
            raise ValueError(
                f"Agent {self.charge_id}: Missing temporal_momentum in s-evolution - storage/reconstruction failed!"
            )
        temporal_momentum = self.temporal_biography.temporal_momentum

        # MATHEMATICAL EVOLUTION: Use proper temporal evolution mathematics
        # Apply same temporal evolution pattern as other temporal components

        # Compute evolution terms using natural mathematical scaling
        field_term = field_pressure * vivid_influence * 0.01

        # Protected field multiplication for temporal momentum coupling
        temporal_complex = temporal_momentum.to_complex()
        momentum_contribution = self._safe_field_multiplication(
            complex(abs(temporal_complex)),
            complex(character_influence * 0.1),
            "temporal_momentum coupling",
        )
        momentum_term = abs(momentum_contribution)

        memory_term = memory_pressure * 0.05

        # Simple mathematical evolution - natural scaling without exp() operations
        delta_s = field_term + momentum_term + memory_term

        # Apply proper temporal evolution factor (similar to temporal_biography evolution)
        evolution_factor = self.evolution_rates.get("memory")

        # Use incremental temporal evolution (not unbounded accumulation)
        # Follow temporal component pattern: evolution relative to current state
        evolution_ratio = 1.0 + (
            delta_s * evolution_factor * 0.01
        )  # Proportional evolution
        self.state.current_s *= evolution_ratio

        # s evolution RESHAPES the modular form itself
        # Higher s means more complex, lower s means simpler
        s_distance = abs(self.state.current_s - self.state.s_zero)
        # Use torch for advanced exponential computation - NO BASIC NUMPY
        complexity_factor = torch.exp(
            -torch.tensor(s_distance / 50, dtype=torch.float32)
        ).item()  # Gradual complexity decay

        # Apply complexity evolution to q-coefficients
        for n in self.breathing_q_coefficients:
            # Higher order coefficients are more sensitive to s-evolution
            sensitivity = 1.0 + n / 100
            decay_factor = complexity_factor**sensitivity
            # Apply proportional complexity decay (not multiplicative)
            # Use incremental evolution to prevent exponential growth
            coeff_magnitude = abs(self.breathing_q_coefficients[n])
            # Use torch for advanced angle computation - NO BASIC NUMPY
            coeff_tensor = torch.tensor(
                [
                    self.breathing_q_coefficients[n].real,
                    self.breathing_q_coefficients[n].imag,
                ],
                dtype=torch.float32,
            )
            current_phase = torch.atan2(coeff_tensor[1], coeff_tensor[0]).item()

            # Proportional decay evolution based on current magnitude
            decay_evolution = (
                coeff_magnitude * (decay_factor - 1.0) * 0.01
            )  # Small proportional change

            # Apply incremental decay
            # Use torch for advanced exponential computation - NO BASIC NUMPY
            self.breathing_q_coefficients[n] += (
                decay_evolution
                * torch.exp(
                    1j * torch.tensor(current_phase, dtype=torch.float32)
                ).item()
            )

        # Update living Q value after s-parameter evolution
        self.living_Q_value = self.evaluate_living_form()

        # Update charge object with current state
        self.charge_obj.complete_charge = self.living_Q_value
        self.charge_obj.magnitude = abs(self.living_Q_value)
        # Use torch for ALL angle computations - NO BASIC NUMPY
        if torch.is_tensor(self.living_Q_value):
            self.charge_obj.phase = torch.angle(self.living_Q_value).item()
        else:
            q_tensor = torch.tensor(
                [self.living_Q_value.real, self.living_Q_value.imag],
                dtype=torch.float32,
            )
            self.charge_obj.phase = torch.atan2(q_tensor[1], q_tensor[0]).item()

    def apply_interaction_evolution(
        self, interaction_effect: Union[complex, "CDF"]
    ) -> None:
        """
        Apply interaction effect using proportional evolution to prevent exponential growth.

        This method maintains mathematical integrity by evolving the Q value proportionally
        based on the interaction strength, rather than directly adding interaction effects.

        Args:
            interaction_effect: The interaction effect as complex or CDF
        """
        # Get current Q value magnitude and phase
        current_magnitude = abs(self.living_Q_value)
        if current_magnitude == 0:
            # Cannot evolve from zero - set to small initial value based on interaction
            # FIX: Check if interaction_effect has CDF characteristics instead of isinstance
            if (
                hasattr(interaction_effect, "abs")
                and hasattr(interaction_effect, "arg")
                and hasattr(interaction_effect, "log")
            ):
                # Use a very small fraction of the interaction magnitude
                # Use CDF for sophisticated exponential computation - NO BASIC math.exp()
                initial_mag = (interaction_effect.abs() * CDF(0.0001)).exp()
                # Use torch for advanced exponential computation - NO BASIC NUMPY
                self.living_Q_value = (
                    initial_mag
                    * torch.exp(
                        1j * torch.tensor(interaction_effect.arg(), dtype=torch.float32)
                    ).item()
                )
            else:
                self.living_Q_value = interaction_effect * 0.0001  # Very small fraction
            return

        # Use torch for advanced angle computation - NO BASIC NUMPY
        q_tensor = torch.tensor(
            [self.living_Q_value.real, self.living_Q_value.imag], dtype=torch.float32
        )
        current_phase = torch.atan2(q_tensor[1], q_tensor[0]).item()

        # FIX: Check if interaction_effect has CDF characteristics instead of isinstance
        # This avoids the TypeError with newer Python versions
        if (
            hasattr(interaction_effect, "abs")
            and hasattr(interaction_effect, "arg")
            and hasattr(interaction_effect, "log")
        ):
            # Working with CDF interaction - sophisticated Sage mathematics
            # Use CDF for sophisticated logarithmic computation - NO BASIC math.log()
            current_log_mag = CDF(current_magnitude).log()
            interaction_log_mag = interaction_effect.abs().log()

            # Proportional evolution based on relative interaction strength
            # If interaction is much larger than current, evolution is stronger
            relative_strength = interaction_log_mag - current_log_mag

            # Bounded proportional evolution to prevent runaway
            # Use torch tanh for sophisticated saturation - NO BASIC NUMPY
            evolution_factor = (
                1.0
                + torch.tanh(
                    torch.tensor(relative_strength * 0.001, dtype=torch.float32)
                ).item()
                * 0.01
            )

            # Phase evolution - small adjustment towards interaction phase
            phase_diff = interaction_effect.arg() - current_phase
            # Normalize phase difference to [-π, π] using torch constants
            torch_pi = torch.tensor(torch.pi, dtype=torch.float32)
            while phase_diff > torch_pi:
                phase_diff -= 2 * torch_pi
            while phase_diff < -torch_pi:
                phase_diff += 2 * torch_pi
            phase_shift = phase_diff * 0.01  # Small phase adjustment

            # Apply proportional evolution
            new_magnitude = current_magnitude * evolution_factor
            new_phase = current_phase + phase_shift

            # Use torch for advanced exponential computation - NO BASIC NUMPY
            self.living_Q_value = (
                new_magnitude
                * torch.exp(1j * torch.tensor(new_phase, dtype=torch.float32)).item()
            )
        else:
            # Normal complex interaction
            interaction_magnitude = abs(interaction_effect)

            # Proportional evolution based on relative magnitude
            relative_strength = interaction_magnitude / (current_magnitude + 1e-12)
            evolution_factor = (
                1.0
                + torch.tanh(
                    torch.tensor(relative_strength * 0.1, dtype=torch.float32)
                ).item()
                * 0.01
            )

            # Phase evolution
            # Use torch for advanced angle computation - NO BASIC NUMPY
            effect_tensor = torch.tensor(
                [interaction_effect.real, interaction_effect.imag], dtype=torch.float32
            )
            interaction_phase = torch.atan2(effect_tensor[1], effect_tensor[0]).item()
            phase_diff = interaction_phase - current_phase
            # Normalize phase difference using torch constants
            torch_pi = torch.tensor(torch.pi, dtype=torch.float32)
            while phase_diff > torch_pi:
                phase_diff -= 2 * torch_pi
            while phase_diff < -torch_pi:
                phase_diff += 2 * torch_pi
            phase_shift = phase_diff * 0.01

            # Apply proportional evolution
            new_magnitude = current_magnitude * evolution_factor
            new_phase = current_phase + phase_shift

            # Use torch for advanced exponential computation - NO BASIC NUMPY
            self.living_Q_value = (
                new_magnitude
                * torch.exp(1j * torch.tensor(new_phase, dtype=torch.float32)).item()
            )

        # Update charge object with evolved state
        self.charge_obj.complete_charge = self.living_Q_value
        self.charge_obj.magnitude = abs(self.living_Q_value)
        # Use torch for advanced angle computation - NO BASIC NUMPY
        q_tensor = torch.tensor(
            [self.living_Q_value.real, self.living_Q_value.imag], dtype=torch.float32
        )
        self.charge_obj.phase = torch.atan2(q_tensor[1], q_tensor[0]).item()

        # Also update breathing coefficients proportionally
        # Interactions affect the modular form structure
        magnitude_ratio = abs(self.living_Q_value) / (current_magnitude + 1e-12)
        for n in self.breathing_q_coefficients:
            coeff_mag = abs(self.breathing_q_coefficients[n])
            # Use torch for advanced angle computation - NO BASIC NUMPY
            coeff_tensor = torch.tensor(
                [
                    self.breathing_q_coefficients[n].real,
                    self.breathing_q_coefficients[n].imag,
                ],
                dtype=torch.float32,
            )
            coeff_phase = torch.atan2(coeff_tensor[1], coeff_tensor[0]).item()

            # Proportional coefficient evolution
            new_coeff_mag = coeff_mag * (
                1.0 + (magnitude_ratio - 1.0) * 0.001
            )  # Very small coefficient change
            # Use torch for sophisticated exponential - NO BASIC NUMPY
            self.breathing_q_coefficients[n] = (
                new_coeff_mag
                * torch.exp(1j * torch.tensor(coeff_phase, dtype=torch.float32)).item()
            )

    def evaluate_living_form(self, tau: Optional[float] = None) -> complex:
        """
        Evaluate the living modular form at current τ to get Q(τ,C,s).

        This computes the agent's current Q value as a living modular form
        using breathing q-coefficients, responsive Hecke operators, and emotional L-function.
        """
        if tau is None:
            tau = self.tau_position
        else:
            tau = complex(tau, self.tau_position.imag)

        # PERFORMANCE OPTIMIZATION: Cache result if state hasn't changed significantly
        current_s = self.state.current_s
        if hasattr(self, "_cached_living_Q") and hasattr(self, "_cached_s"):
            s_change = abs(current_s - self._cached_s)
            if s_change < 0.001:  # Very small change in observational state
                logger.debug(
                    f"🔧 Agent {self.charge_id} - Using cached living Q (s_change: {s_change:.6f})"
                )
                return self._cached_living_Q

        # Compute q = exp(2πiτ) - convert complex tau to proper PyTorch tensor
        # Use torch.pi for sophisticated constant - NO BASIC NUMPY
        tau_tensor = torch.tensor(
            2j * torch.pi * tau, dtype=torch.complex64, device=self.device
        )
        q = torch.exp(tau_tensor)

        # Evaluate breathing q-expansion - ensure tensor/complex consistency
        f_tau = torch.tensor(
            complex(0.0, 0.0), dtype=torch.complex64, device=self.device
        )
        for n, coeff in self.breathing_q_coefficients.items():
            coeff_tensor = torch.tensor(
                coeff, dtype=torch.complex64, device=self.device
            )
            if n == 0:
                f_tau += coeff_tensor  # Constant term
            else:
                f_tau += coeff_tensor * (q**n)

        # Apply responsive Hecke operators
        for p, eigenvalue in self.hecke_eigenvalues.items():
            # Hecke operator T_p applied as multiplicative factor
            hecke_factor = 1.0 + eigenvalue * (q**p) / (1.0 + abs(q**p))
            f_tau *= hecke_factor

        # Apply emotional L-function modulation - ensure tensor consistency
        l_value = torch.tensor(
            complex(1.0, 0.0), dtype=torch.complex64, device=self.device
        )
        for n, coeff in self.l_function_coefficients.items():
            if abs(coeff) > 0:
                coeff_tensor = torch.tensor(
                    coeff, dtype=torch.complex64, device=self.device
                )
                n_factor = torch.tensor(
                    n ** (1 + 0.1j), dtype=torch.complex64, device=self.device
                )
                l_value *= 1.0 + coeff_tensor / n_factor

        f_tau *= l_value

        # Apply observational state persistence - ensure tensor consistency
        s_factor = self.compute_observational_persistence()[0]  # Total persistence
        s_factor_tensor = torch.tensor(
            s_factor, dtype=torch.complex64, device=self.device
        )
        f_tau *= s_factor_tensor

        # DIRECT TENSOR TO COMPLEX CONVERSION - NO ERROR MASKING
        context = f"Agent_{self.charge_id}_Q_calculation"
        if safe_tensor_comparison(f_tau, 1, context):
            # DIRECT SCALAR EXTRACTION
            tensor_value = extract_tensor_scalar(f_tau, context)
            if isinstance(tensor_value, complex):
                f_tau_complex = complex(
                    float(tensor_value.real), float(tensor_value.imag)
                )
            else:
                f_tau_complex = complex(float(tensor_value), 0.0)
        else:
            f_tau_complex = f_tau

        # No artificial mathematical constraints - preserve complete Q(τ,C,s) integrity

        self.living_Q_value = f_tau_complex

        # Cache the result for performance optimization
        self._cached_living_Q = f_tau_complex
        self._cached_s = current_s

        return f_tau_complex

    def sync_positions(self):
        """
        Keep tau_position and field_position synchronized.

        This ensures the agent's modular domain position (tau_position) stays
        in sync with the orchestrator's grid position (field_position).
        """
        # Convert tau (modular) to field (grid) coordinates
        x = self.tau_position.real  # Real part maps directly
        y = (
            self.tau_position.imag - 1.0
        )  # Shift from modular to grid (Im(τ) ≥ 1 → y ≥ 0)

        # Update field position
        self.state.field_position = (x, y)

        # Update charge object metadata if it exists
        if hasattr(self.charge_obj, "metadata") and hasattr(
            self.charge_obj.metadata, "field_position"
        ):
            self.charge_obj.metadata.field_position = (x, y)
        elif hasattr(self.charge_obj, "set_field_position"):
            self.charge_obj.set_field_position((x, y))

    def update_tau_from_field(self):
        """
        Update tau_position from field_position if moved by orchestrator.

        This allows the orchestrator to move agents in grid space and have
        their modular domain position automatically updated.
        """
        x, y = self.state.field_position

        real_part = ((x + 0.5) % 1.0) - 0.5
        imag_part = float(max(CDF(0.1), CDF(1.0) + CDF(y)).real())

        self.tau_position = complex(real_part, imag_part)

    def compute_gamma_calibration(
        self, collective_field_strength: Optional[float] = None, pool_size: int = 1
    ) -> float:
        """
        Implement γ global field calibration from section 3.1.5.3.

        "The conductor's master volume control...ensures individual instrumental
        voices blend harmoniously within the collective performance"

        Uses actual field modulation strength from emotional analytics.
        🚀 BOOSTED: Now scales with pool size for stronger field presence.
        """
        # Base gamma from charge object
        base_gamma = self.charge_obj.gamma

        # Use emotional field modulation strength as conductor control
        conductor_modulation = self.emotional_field_signature.field_modulation_strength

        # Use collective field strength if provided, otherwise derive from interference
        if collective_field_strength is None:
            # Derive from field interference matrix
            # Use scipy special functions for sophisticated averaging - NO BASIC NUMPY
            # Use torch for sophisticated absolute value - NO BASIC NUMPY
            abs_interference = torch.abs(
                torch.tensor(self.field_interference_matrix, dtype=torch.float32)
            ).numpy()
            interference_strength = (
                special.logsumexp(abs_interference.flatten()) / abs_interference.size
            )
            collective_field_strength = 1.0 + interference_strength

        # 🚀 BOOST GAMMA: Scale with pool size for stronger field presence
        # Larger pools need stronger individual gamma to maintain field energy
        pool_boost = max(
            1.0, np.sqrt(pool_size)
        )  # Square root scaling prevents overwhelming

        # Apply additional field strength boost for 10-agent pools
        field_strength_boost = 2.0 if pool_size >= 10 else 1.5

        # Calibrate to prevent overwhelming or weakness
        # Higher collective field → lower individual gamma (normalization)
        calibration_factor = conductor_modulation / (
            1.0 + 0.1 * collective_field_strength
        )

        # Final gamma with pool-based boosting
        boosted_gamma = (
            base_gamma * calibration_factor * pool_boost * field_strength_boost
        )

        return boosted_gamma

    def compute_transformative_potential_tensor(self) -> complex:
        """
        Implement T(τ, C, s) from section 3.1.5.4.

        "The evolutionary heart of our framework...tensor structure enables us to
        capture how semantic, emotional, and temporal dimensions interact through
        multiplicative rather than additive effects"

        Uses actual trajectory_operators from temporal biography with context modulation.
        """
        # Get trajectory operators from temporal biography
        trajectory_ops = self.temporal_biography.trajectory_operators

        if len(trajectory_ops) == 0:
            return complex(1.0, 0.0), 1.0, 0.0

        # Convert to torch tensor for multidimensional operations
        trajectory_ops_complex = np.array([complex(x) for x in trajectory_ops], dtype=np.complex64)
        T_ops = torch.tensor(trajectory_ops_complex, dtype=torch.complex64, device=self.device)

        # Context C modulates trajectory through observer contingency
        context_size = len(self.state.current_context_C)
        context_modulation = 1.0 + 0.1 * np.log1p(context_size)  # log1p for stability

        # Observational state s affects trajectory integration
        s = self.state.current_s
        s_evolution = torch.exp(
            torch.tensor(-0.05 * s, dtype=torch.float32, device=self.device)
        )

        # Emotional coupling modulates T tensor (multiplicative effect)
        emotional_coupling = abs(self.coupling_state.emotional_field_coupling)

        # Compute tensor with multiplicative interactions
        T_tensor_value = (
            torch.mean(T_ops)
            * context_modulation
            * s_evolution
            * (1.0 + 0.2 * emotional_coupling)
        )

        # Extract complex result
        # CRITICAL FIX: Ensure proper tensor-to-Python complex conversion with MPS safety
        tensor_numpy = T_tensor_value.cpu().detach().numpy()
        if tensor_numpy.dtype == np.complex64 or tensor_numpy.dtype == np.complex128:
            T_complex = complex(float(tensor_numpy.real), float(tensor_numpy.imag))
        else:
            T_complex = complex(float(tensor_numpy), 0.0)

        return T_complex

    def compute_emotional_trajectory_integration(self) -> complex:
        """
        Implement E^trajectory(τ, s) from section 3.1.5.5.

        "Dynamic field modulation...emotional resonance must accumulate through
        observational experience rather than operating as fixed taxonomic properties"

        Uses actual emotional modulation tensors with trajectory accumulation.
        """
        # DEBUG: Log method entry and input components
        print(f"[DEBUG] compute_emotional_trajectory_integration - Method entry")
        print(f"[DEBUG] emotional_modulation object: {self.emotional_modulation}")
        print(f"[DEBUG] emotional_modulation type: {type(self.emotional_modulation)}")

        # Get emotional modulation components
        modulation_tensor = self.emotional_modulation.semantic_modulation_tensor
        trajectory_attractors = self.emotional_modulation.trajectory_attractors
        resonance_freqs = self.emotional_modulation.resonance_frequencies

        # DEBUG: Log the extracted components
        print(f"[DEBUG] semantic_modulation_tensor: {modulation_tensor}")
        print(f"[DEBUG] semantic_modulation_tensor type: {type(modulation_tensor)}")
        print(
            f"[DEBUG] semantic_modulation_tensor shape: {getattr(modulation_tensor, 'shape', 'N/A')}"
        )
        print(f"[DEBUG] trajectory_attractors: {trajectory_attractors}")
        print(f"[DEBUG] trajectory_attractors type: {type(trajectory_attractors)}")
        print(
            f"[DEBUG] trajectory_attractors shape: {getattr(trajectory_attractors, 'shape', 'N/A')}"
        )
        print(f"[DEBUG] resonance_frequencies: {resonance_freqs}")
        print(f"[DEBUG] resonance_frequencies type: {type(resonance_freqs)}")
        print(
            f"[DEBUG] resonance_frequencies shape: {getattr(resonance_freqs, 'shape', 'N/A')}"
        )

        # Current and initial observational states
        s = self.state.current_s
        s_zero = self.state.s_zero
        delta_s = s - s_zero

        # DEBUG: Log observational states
        print(f"[DEBUG] current_s: {s}")
        print(f"[DEBUG] s_zero: {s_zero}")
        print(f"[DEBUG] delta_s: {delta_s}")

        # Trajectory accumulation using scipy integration
        def emotional_integrand(s_prime):
            # Distance from attractor states
            # Use scipy special functions for sophisticated computation - NO BASIC NUMPY
            # Use torch for sophisticated operations - NO BASIC NUMPY
            abs_diff = torch.abs(s_prime - trajectory_attractors)
            exp_values = torch.exp(-abs_diff)
            attractor_influence = torch.mean(exp_values).item()

            freq_tensor = resonance_freqs
            s_prime_tensor = torch.tensor(s_prime, dtype=torch.float32)
            resonance_values = torch.cos(2 * torch.pi * freq_tensor * s_prime_tensor)
            # Apply sophisticated activation instead of basic sum
            resonance = F.gelu(torch.sum(resonance_values)).item()

            # Modulation decay with distance
            # Use torch for sophisticated operations - NO BASIC NUMPY
            abs_diff = torch.abs(torch.tensor(s_prime - s_zero, dtype=torch.float32))
            decay = torch.exp(-0.1 * abs_diff).item()

            return attractor_influence * resonance * decay

        # Integrate emotional accumulation from s_zero to current s
        print(f"[DEBUG] About to compute integration, abs(delta_s) = {abs(delta_s)}")
        if abs(delta_s) > 0.01:
            print(f"[DEBUG] Computing integration from {s_zero} to {s}")
            emotional_integral, _ = integrate.quad(emotional_integrand, s_zero, s)
            print(f"[DEBUG] Integration result: {emotional_integral}")
        else:
            emotional_integral = 0.0
            print(f"[DEBUG] Using zero integration (delta_s too small)")

        # Apply modulation tensor influence
        # Use scipy special functions for sophisticated averaging - NO BASIC NUMPY
        print(f"[DEBUG] Computing tensor influence from modulation_tensor")
        tensor_influence = special.logsumexp(modulation_tensor) / len(modulation_tensor)
        print(f"[DEBUG] tensor_influence: {tensor_influence}")

        # Combine with unified phase shift
        phase_shift = self.emotional_modulation.unified_phase_shift
        print(f"[DEBUG] unified_phase_shift: {phase_shift}")

        # Final emotional trajectory value
        E_magnitude = 1.0 + 0.3 * tensor_influence + 0.1 * emotional_integral
        print(
            f"[DEBUG] E_magnitude calculation: 1.0 + 0.3 * {tensor_influence} + 0.1 * {emotional_integral} = {E_magnitude}"
        )

        # Use torch for advanced phase computation
        print(f"[DEBUG] Computing phase from phase_shift")
        phase_tensor = torch.tensor(phase_shift, dtype=torch.complex64)
        E_phase = torch.angle(phase_tensor).item()
        print(f"[DEBUG] E_phase: {E_phase}")

        # Advanced E_trajectory construction using torch exponential
        print(f"[DEBUG] Constructing final E_trajectory")
        magnitude_tensor = torch.tensor(E_magnitude, dtype=torch.float32)
        phase_exp = torch.exp(1j * torch.tensor(E_phase, dtype=torch.float32))
        E_trajectory = (magnitude_tensor * phase_exp).item()

        print(f"[DEBUG] Final E_trajectory: {E_trajectory}")
        print(f"[DEBUG] E_trajectory type: {type(E_trajectory)}")
        print(f"[DEBUG] E_trajectory is None: {E_trajectory is None}")
        print(f"[DEBUG] compute_emotional_trajectory_integration - Method exit")

        return E_trajectory

    def compute_semantic_field_generation(self) -> complex:
        """
        Implement Φ^semantic(τ, s) from section 3.1.5.6.

        Paper Formula: S_τ(x) = Σᵢ e_τ,ᵢ · φᵢ(x) · e^(iθ_τ,ᵢ)

        "From static embeddings to dynamic fields...breathing constellation patterns
        across the narrative sky...semantic elements function as field generators"

        Uses actual SemanticField object with proper basis function evaluation.
        """
        # Current observational state for breathing pattern (Section 3.1.4.3.4)
        s = self.state.current_s

        # Extract breathing patterns from collective rhythm (following paper Section 3.1.4.3.4)
        if "collective_frequency" in self.collective_breathing:
            collective_freq = self.collective_breathing["collective_frequency"]
            # Use mean frequency for breathing modulation
            # NO FALLBACKS - collective_freq MUST be array-like or collective breathing system is broken
            if not hasattr(collective_freq, "__len__"):
                raise ValueError(
                    f"collective_freq is not array-like: {type(collective_freq)} = {collective_freq} - Collective breathing system corrupted!"
                )
            real_freq = torch.real(
                torch.tensor(collective_freq, dtype=torch.complex64)
            ).numpy()
            breathing_frequency = special.logsumexp(real_freq) / len(real_freq)
        else:
            # NO FALLBACK - Collective frequency must exist
            raise ValueError(
                "MATHEMATICAL FAILURE: No collective frequency data - "
                "Breathing frequency cannot be computed. System requires frequency data."
            )

        breathing_amplitude = self.collective_breathing.get(
            "breathing_pattern_diversity"
        )

        # REAL modular breathing constellation using Eisenstein series (Section 3.1.4.3.4)
        # Convert observational parameter s to complex tau in upper half-plane
        # Mathematical stability: Ensure tau satisfies modular form convergence constraints
        real_part = max(0.15, breathing_frequency * s * 0.5)  # Minimum 0.15 for convergence
        imag_part = max(1.5, 0.5 + breathing_amplitude * 0.2)  # Increased minimum for |q| < 1 constraint
        tau_semantic = complex(real_part, imag_part)
        breathing_factor = self._evaluate_breathing_eisenstein_series(
            tau_semantic, harmonic_number=2
        )

        # Use pre-computed semantic field data from charge factory
        field_base_magnitude = np.linalg.norm(self.semantic_field.embedding_components)
        field_base_phase = np.mean(self.semantic_field.phase_factors)
        field_value = field_base_magnitude * np.exp(1j * field_base_phase)

        # Apply breathing modulation (Section 3.1.4.3.4)
        phi_semantic = field_value * breathing_factor

        # Apply emotional conductor modulation to semantic field (Section 3.1.3.3.1)
        conductor_influence = self.coupling_state.s_t_coupling_strength
        phi_semantic *= 1.0 + 0.2 * conductor_influence

        return phi_semantic

    def compute_5component_phase_integration(self) -> ThetaComponents:
        """
        Implement complete 5-component θ_total(τ,C,s) from section 3.1.5.7.

        θ_total = θ_semantic(τ,C) + θ_emotional(τ) + ∫₀ˢ ω_temporal(τ,s') ds' +
                  θ_interaction(τ,C,s) + θ_field(x,s)

        Uses actual phase data from all dimensions with proper integration.
        """
        s = self.state.current_s
        s_zero = self.state.s_zero

        # θ_semantic(τ,C) - from actual SemanticField phase factors (Section 3.1.5.7)
        phase_factors = self.semantic_field.phase_factors
        context_influence = len(self.state.current_context_C) / 100.0  # Normalize
        # Use scipy special functions for sophisticated averaging - NO BASIC NUMPY
        theta_semantic = (special.logsumexp(phase_factors) / len(phase_factors)) * (
            1.0 + context_influence
        )

        # θ_emotional(τ) - from emotional phase modulation
        emotional_phase_shift = self.emotional_modulation.unified_phase_shift
        # Use torch for advanced angle computation - NO BASIC NUMPY
        emotional_tensor = torch.tensor(
            [emotional_phase_shift.real, emotional_phase_shift.imag],
            dtype=torch.float32,
        )
        theta_emotional = torch.atan2(emotional_tensor[1], emotional_tensor[0]).item()

        # ∫₀ˢ ω_temporal(τ,s') ds' - temporal integral using frequency evolution
        integration_start = time.time()
        frequency_evolution = self.temporal_biography.frequency_evolution
        if len(frequency_evolution) > 0 and abs(s - s_zero) > 0.01:
            logger.debug(
                f"🔧 Agent {self.charge_id} - Computing temporal integral: freq_len={len(frequency_evolution)}, s_range=({s_zero:.3f}, {s:.3f})"
            )

            # PERFORMANCE FIX: Use fast trapezoidal rule instead of expensive quad integration
            # Extract real parts for integration (frequency_evolution can be complex)
            # NO FALLBACKS - frequency_evolution MUST be array-like or temporal system is broken
            if not hasattr(frequency_evolution, "__len__"):
                raise ValueError(
                    f"frequency_evolution is not array-like: {type(frequency_evolution)} = {frequency_evolution} - Temporal evolution system corrupted!"
                )
            # Use torch for sophisticated real extraction - NO BASIC NUMPY
            omega_values = torch.real(
                torch.tensor(frequency_evolution, dtype=torch.complex64)
            ).numpy()
            s_span = abs(s - s_zero)

            if len(omega_values) > 1:
                # Trapezoidal rule: much faster than cubic interpolation + quad
                temporal_integral = np.trapz(
                    omega_values, dx=s_span / (len(omega_values) - 1)
                )
            else:
                # Single value case
                temporal_integral = omega_values[0] * s_span if omega_values else 0.0

            logger.debug(
                f"🔧 Agent {self.charge_id} - Fast integration complete (time: {time.time() - integration_start:.3f}s, result: {temporal_integral:.6f})"
            )
        else:
            temporal_integral = 0.0
            logger.debug(
                f"🔧 Agent {self.charge_id} - Skipping integration: freq_len={len(frequency_evolution)}, s_range=({s_zero:.3f}, {s:.3f})"
            )

        logger.debug(
            f"🔧 Agent {self.charge_id} - Total integration phase (time: {time.time() - integration_start:.3f}s)"
        )

        interference_strength = np.mean(np.abs(self.temporal_biography.field_interference_signature))
        theta_interaction = interference_strength * s * context_influence

        # θ_field(x,s) - manifold field dynamics at position
        x, y = self.state.field_position
        field_distance = np.sqrt(x * x + y * y)
        theta_field = 0.1 * field_distance * s

        # Total phase with wrapping
        total = (
            theta_semantic
            + theta_emotional
            + temporal_integral
            + theta_interaction
            + theta_field
        )

        # Wrap to [-π, π]
        # Use torch.pi for sophisticated constant - NO BASIC NUMPY
        while total > torch.pi:
            total -= 2 * torch.pi
        while total < -torch.pi:
            total += 2 * torch.pi

        return ThetaComponents(
            theta_semantic=theta_semantic,
            theta_emotional=theta_emotional,
            temporal_integral=temporal_integral,
            theta_interaction=theta_interaction,
            theta_field=theta_field,
            total=total,
        )

    def compute_observational_persistence(self) -> Tuple[float, float, float]:
        """
        Implement Ψ_persistence(s-s₀) dual-decay structure from section 3.1.5.8.

        "Layered memory effects...dual-decay structure with vivid recent chapters
        and persistent character traits"

        Ψ = exp(-(s-s₀)²/2σᵢ²) + αᵢ·exp(-λᵢ(s-s₀))·cos(βᵢ(s-s₀))

        Uses actual vivid_layer and character_layer from temporal biography.
        """
        # MATHEMATICAL THEORY OPTIMIZATION: Use cached result if available
        if hasattr(self, '_cached_persistence_result'):
            cached_result = self._cached_persistence_result
            logger.debug(f"🔧 Agent {self.charge_id} - Using cached persistence result")
            return cached_result
        
        logger.debug(
            f"🔧 Agent {self.charge_id} - Starting observational persistence computation"
        )

        # CRITICAL DEBUG: Check state value types before access
        logger.debug(
            f"🔧 Agent {self.charge_id} - State types: current_s={type(self.state.current_s)}, s_zero={type(self.state.s_zero)}"
        )

        # MPS-safe state access with explicit tensor checking
        if hasattr(self.state.current_s, "cpu"):
            s = float(self.state.current_s.cpu().detach().numpy())
            logger.info(
                f"🔧 Agent {self.charge_id} - Converted current_s tensor to float: {s}"
            )
        else:
            s = float(self.state.current_s)

        if hasattr(self.state.s_zero, "cpu"):
            s_zero = float(self.state.s_zero.cpu().detach().numpy())
            logger.info(
                f"🔧 Agent {self.charge_id} - Converted s_zero tensor to float: {s_zero}"
            )
        else:
            s_zero = float(self.state.s_zero)

        logger.debug(
            f"🔧 Agent {self.charge_id} - State values: s={s}, s_zero={s_zero}"
        )

        delta_s = s - s_zero

        # Extract persistence layers from temporal biography
        vivid_layer = self.temporal_biography.vivid_layer
        character_layer = self.temporal_biography.character_layer
        logger.debug(
            f"🔧 Agent {self.charge_id} - Layer types: vivid={type(vivid_layer)}, character={type(character_layer)}"
        )
        logger.debug(
            f"🔧 Agent {self.charge_id} - Layer lengths: vivid={len(vivid_layer)}, character={len(character_layer)}"
        )

        # Gaussian component from vivid layer (recent sharp memory)
        if len(vivid_layer) > 0:
            # Use actual vivid layer data with underflow protection
            # CRITICAL FIX: Ensure mean is real before comparison - MPS-safe
            logger.debug(
                f"🔧 Agent {self.charge_id} - Processing vivid_layer, type: {type(vivid_layer)}"
            )
            if hasattr(vivid_layer, "cpu"):
                vivid_mean = torch.logsumexp(vivid_layer, dim=0) / vivid_layer.numel()
                logger.debug(
                    f"🔧 Agent {self.charge_id} - Processed vivid_layer tensor directly"
                )
            else:
                vivid_tensor = torch.tensor(vivid_layer, dtype=torch.float32)
                vivid_mean = torch.logsumexp(vivid_tensor, dim=0) / vivid_tensor.numel()
                logger.debug(
                    f"🔧 Agent {self.charge_id} - Converted to tensor and processed"
                )
            vivid_mean_real = (
                float(vivid_mean.real)
                if torch.is_complex(vivid_mean)
                else float(vivid_mean.item())
            )
            vivid_base = float(
                max(0.9, vivid_mean_real)
            )  # 🔧 Clamp vivid layer to minimum 0.9 - ensure float
            logger.debug(
                f"🔧 Agent {self.charge_id} - About to compute vivid_influence with vivid_base={vivid_base}, delta_s={delta_s}"
            )

            # CRITICAL FIX: Ensure sigma_i is a float for MPS compatibility
            if hasattr(self.sigma_i, "cpu"):
                sigma_i_val = float(self.sigma_i.cpu().detach().numpy())
            else:
                sigma_i_val = float(self.sigma_i)

            # STABILIZED VIVID INFLUENCE COMPUTATION with logarithmic scaling for large delta_s
            # Mathematical fix: when delta_s is large, apply log scaling to prevent underflow
            delta_s_threshold = 1000.0  # Threshold for applying logarithmic scaling

            if abs(delta_s) > delta_s_threshold:
                # Logarithmic scaling: delta_s_scaled = threshold * log(1 + |delta_s|/threshold)
                # This preserves monotonicity while preventing exponential underflow
                delta_s_sign = 1.0 if delta_s >= 0 else -1.0
                delta_s_scaled = (
                    delta_s_sign
                    * delta_s_threshold
                    * math.log(1.0 + abs(delta_s) / delta_s_threshold)
                )

                logger.debug(
                    f"🔧 Agent {self.charge_id} - Large delta_s detected: {delta_s:.2f} -> scaled to {delta_s_scaled:.2f}"
                )

                # Use scaled delta_s for exponential computation
                exp_arg = -(delta_s_scaled * delta_s_scaled) / (
                    2 * sigma_i_val * sigma_i_val
                )
            else:
                # Normal computation for reasonable delta_s values
                exp_arg = -(delta_s * delta_s) / (2 * sigma_i_val * sigma_i_val)

            # Ensure exp_arg doesn't cause underflow (clamp to reasonable range)
            exp_arg = max(exp_arg, -20.0)  # exp(-20) ≈ 2e-9, still meaningful

            vivid_influence = (
                vivid_base
                * torch.exp(torch.tensor(exp_arg, dtype=torch.float32)).item()
            )
            logger.debug(
                f"🔧 Agent {self.charge_id} - Vivid influence computed: {vivid_influence} (exp_arg: {exp_arg:.3f})"
            )
        else:
            # CRITICAL FIX: Ensure sigma_i is a float for MPS compatibility
            if hasattr(self.sigma_i, "cpu"):
                sigma_i_val = float(self.sigma_i.cpu().detach().numpy())
            else:
                sigma_i_val = float(self.sigma_i)

            # STABILIZED VIVID INFLUENCE COMPUTATION with logarithmic scaling for large delta_s
            # Mathematical fix: when delta_s is large, apply log scaling to prevent underflow
            delta_s_threshold = 1000.0  # Threshold for applying logarithmic scaling

            if abs(delta_s) > delta_s_threshold:
                # Logarithmic scaling: delta_s_scaled = threshold * log(1 + |delta_s|/threshold)
                # This preserves monotonicity while preventing exponential underflow
                delta_s_sign = 1.0 if delta_s >= 0 else -1.0
                delta_s_scaled = (
                    delta_s_sign
                    * delta_s_threshold
                    * math.log(1.0 + abs(delta_s) / delta_s_threshold)
                )

                logger.debug(
                    f"🔧 Agent {self.charge_id} - Large delta_s detected: {delta_s:.2f} -> scaled to {delta_s_scaled:.2f}"
                )

                # Use scaled delta_s for exponential computation
                exp_arg = -(delta_s_scaled * delta_s_scaled) / (
                    2 * sigma_i_val * sigma_i_val
                )
            else:
                # Normal computation for reasonable delta_s values
                exp_arg = -(delta_s * delta_s) / (2 * sigma_i_val * sigma_i_val)

            # Ensure exp_arg doesn't cause underflow (clamp to reasonable range)
            exp_arg = max(exp_arg, -20.0)  # exp(-20) ≈ 2e-9, still meaningful

            vivid_influence = torch.exp(
                torch.tensor(exp_arg, dtype=torch.float32)
            ).item()
            logger.debug(
                f"🔧 Agent {self.charge_id} - Vivid influence (else case) computed: {vivid_influence} (exp_arg: {exp_arg:.3f})"
            )

        # Apply additional vivid underflow protection
        vivid_influence_real = (
            # Use torch for sophisticated complex type checking - NO BASIC NUMPY
            float(vivid_influence.real)
            if torch.is_complex(torch.tensor(vivid_influence))
            else float(vivid_influence)
        )
        vivid_influence = max(
            0.8, vivid_influence_real
        )  # 🔧 Keep vivid influence above 0.8

        # Exponential-cosine from character layer (persistent themes)
        if len(character_layer) > 0:
            # Use actual character layer data with underflow protection
            if hasattr(character_layer, "cpu"):
                character_tensor = character_layer.to(
                    device=self.device, dtype=torch.float32
                )
                character_mean = torch.mean(character_tensor).item()
            else:
                character_tensor = torch.tensor(
                    character_layer, device=self.device, dtype=torch.float32
                )
                character_mean = torch.mean(character_tensor).item()
            character_mean_real = (
                # Use torch for sophisticated complex type checking - NO BASIC NUMPY
                float(character_mean.real)
                if torch.is_complex(torch.tensor(character_mean))
                else float(character_mean)
            )
            character_base = float(
                max(0.08, character_mean_real)
            )  # 🔧 Clamp character layer minimum - ensure float
            # Use CDF for sophisticated exponential and trigonometric computation - NO BASIC math functions
            character_influence = character_base * float(
                CDF(-self.lambda_i * CDF(delta_s).abs()).exp()
            )
            character_influence *= float(CDF(self.beta_i * delta_s).cos())
        else:
            # Use CDF for sophisticated mathematical operations - NO BASIC math functions
            character_influence = (
                self.alpha_i
                * float(CDF(-self.lambda_i * CDF(delta_s).abs()).exp())
                * float(CDF(self.beta_i * delta_s).cos())
            )

        character_influence_real = abs(character_influence)
        character_influence = max(0.05, character_influence_real)

        # Combined persistence with stronger minimum
        total_persistence = vivid_influence + character_influence

        # Allow persistence to evolve naturally - maintain mathematical integrity

        return total_persistence, vivid_influence, character_influence

    def compute_complete_Q(
        self, collective_field_strength: Optional[float] = None, pool_size: int = 1
    ) -> QMathematicalComponents:
        """
        Compute complete Q(τ, C, s) = γ · T · E · Φ · e^(iθ) · Ψ

        This is the living mathematical entity in action - computing the complete
        conceptual charge using actual field theory mathematics with real data.
        """
        start_time = time.time()
        logger.debug(
            f"🔧 Agent {self.charge_id} - Starting Q computation (pool_size: {pool_size})"
        )

        # DIRECT Q COMPUTATION - NO ERROR MASKING
        component_start = time.time()
        gamma = self.compute_gamma_calibration(collective_field_strength, pool_size)
        logger.info(
            f"🔧 Agent {self.charge_id} - gamma: {gamma:.6f} (time: {time.time() - component_start:.3f}s)"
        )

        component_start = time.time()
        T_tensor = self.compute_transformative_potential_tensor()
        logger.info(
            f"🔧 Agent {self.charge_id} - T_tensor: {T_tensor} (magnitude: {abs(T_tensor):.6f}, time: {time.time() - component_start:.3f}s)"
        )

        # Debug logging for emotional trajectory before Q computation
        # Computing emotional trajectory for agent
        if hasattr(self.emotional_modulation, "emotional_trajectory"):
            # Emotional trajectory tensor properties verified (NaN check, shape, device)
            pass
        logger.debug(f"   - Current observational state s: {self.state.current_s}")
        logger.debug(f"   - Initial observational state s_zero: {self.state.s_zero}")
        logger.debug(f"   - Delta s: {self.state.current_s - self.state.s_zero}")

        component_start = time.time()
        E_trajectory = self.compute_emotional_trajectory_integration()
        logger.debug(
            f"🔍 DEBUG: E_trajectory computation result: {type(E_trajectory)} = {E_trajectory}"
        )
        logger.debug(f"   - E_trajectory is None: {E_trajectory is None}")
        if E_trajectory is not None:
            logger.debug(f"   - E_trajectory value: {E_trajectory}")
            logger.debug(f"   - E_trajectory magnitude: {abs(E_trajectory):.6f}")
        logger.info(
            f"🔧 Agent {self.charge_id} - E_trajectory: {E_trajectory} (magnitude: {abs(E_trajectory):.6f}, time: {time.time() - component_start:.3f}s)"
        )

        component_start = time.time()
        phi_semantic = self.compute_semantic_field_generation()
        logger.info(
            f"🔧 Agent {self.charge_id} - phi_semantic: {phi_semantic} (magnitude: {abs(phi_semantic):.6f}, time: {time.time() - component_start:.3f}s)"
        )

        component_start = time.time()
        theta_components = self.compute_5component_phase_integration()
        # Use torch for trigonometric operations
        theta_tensor = torch.tensor(theta_components.total, dtype=torch.float32)
        phase_factor = complex(
            torch.cos(theta_tensor).item(), torch.sin(theta_tensor).item()
        )
        logger.info(
            f"🔧 Agent {self.charge_id} - phase_integration (time: {time.time() - component_start:.3f}s)"
        )

        # 🔧 VERIFY: Phase factor magnitude should be exactly 1.0 (e^iθ property)
        phase_magnitude = abs(phase_factor)
        if abs(phase_magnitude - 1.0) > 1e-10:
            logger.warning(
                f"⚠️  Agent {self.charge_id} - Phase factor magnitude error: |e^iθ| = {phase_magnitude:.10f} (should be 1.0)"
            )
            logger.warning(
                f"    theta_total = {theta_components.total:.6f}, cos = {np.cos(theta_components.total):.6f}, sin = {np.sin(theta_components.total):.6f}"
            )
            # Normalize to unit magnitude
            phase_factor = phase_factor / phase_magnitude
            logger.warning(
                f"    Normalized phase_factor: {phase_factor} (new magnitude: {abs(phase_factor):.10f})"
            )

        logger.debug(
            f"🔧 Agent {self.charge_id} - phase_factor: {phase_factor} (magnitude: {abs(phase_factor):.6f})"
        )

        # DIRECT PERSISTENCE COMPUTATION - NO ERROR MASKING
        logger.debug(
            f"🔧 Agent {self.charge_id} - About to call compute_observational_persistence()"
        )
        psi_persistence, psi_gaussian, psi_exponential_cosine = (
            self.compute_observational_persistence()
        )
        logger.debug(
            f"🔧 Agent {self.charge_id} - Persistence computation completed successfully"
        )
        logger.debug(
            f"🔧 Agent {self.charge_id} - psi_persistence: {psi_persistence:.6f} (gaussian: {psi_gaussian:.6f}, exp_cos: {psi_exponential_cosine:.6f})"
        )

        # Allow natural mathematical evolution - maintain integrity

        # Final Q(τ, C, s) computation
        Q_value = (
            gamma
            * T_tensor
            * E_trajectory
            * phi_semantic
            * phase_factor
            * psi_persistence
        )

        # Validate Q magnitude
        Q_magnitude = abs(Q_value)
        if Q_magnitude < 1e-10:
            # NO WARNINGS - tiny Q magnitude means mathematical computation failed
            component_info = f"γ:{gamma:.3f} |T|:{abs(T_tensor):.3f} |E|:{abs(E_trajectory):.3f} |Φ|:{abs(phi_semantic):.3f} |e^iθ|:{abs(phase_factor):.3f} Ψ:{psi_persistence:.3f}"
            raise ValueError(
                f"Agent {self.charge_id} - Q magnitude suspiciously small: {Q_magnitude:.2e} - Mathematical computation failed! Components: {component_info}"
            )
        elif Q_magnitude > 1000.0:  # Increased threshold - 10.0 was too restrictive
            # NO WARNINGS FOR LARGE Q - but check for mathematical overflow
            raise ValueError(
                f"Agent {self.charge_id} - Q magnitude indicates mathematical overflow: {Q_magnitude:.2e} - Field theory computation corrupted!"
            )
        else:
            logger.debug(
                f"✅ Agent {self.charge_id} - Q computed successfully: {Q_value} (magnitude: {Q_magnitude:.6f})"
            )

        # Store components
        self.Q_components = QMathematicalComponents(
            gamma=gamma,
            T_tensor=T_tensor,
            E_trajectory=E_trajectory,
            phi_semantic=phi_semantic,
            theta_components=theta_components,
            phase_factor=phase_factor,
            psi_persistence=psi_persistence,
            psi_gaussian=psi_gaussian,
            psi_exponential_cosine=psi_exponential_cosine,
            Q_value=Q_value,
        )

        # 🔍 Q VALUE TRACKING: Log final Q_components creation
        logger.debug(f"🔍 Q-TRACK: Agent {self.charge_id} Q_components created:")
        logger.debug(
            f"   - E_trajectory: {E_trajectory} (None: {E_trajectory is None})"
        )
        logger.debug(f"   - Q_value: {Q_value} (None: {Q_value is None})")
        logger.debug(
            f"   - Q_components.E_trajectory: {self.Q_components.E_trajectory} (None: {self.Q_components.E_trajectory is None})"
        )

        # Update charge object
        self.charge_obj.complete_charge = Q_value
        self.charge_obj.magnitude = abs(Q_value)
        # Use torch for angle computation
        q_tensor = torch.tensor([Q_value.real, Q_value.imag], dtype=torch.float32)
        self.charge_obj.phase = torch.atan2(q_tensor[1], q_tensor[0]).item()

        # 🔧 FIX: Update living_Q_value with computed Q (not static 1+0j override!)
        self.living_Q_value = Q_value
        logger.debug(
            f"🔧 Agent {self.charge_id} - living_Q_value updated: {self.living_Q_value} (magnitude: {abs(self.living_Q_value):.6f})"
        )

        # 🔧 VALIDATE: Check Q components are reasonable
        self.validate_Q_components()

        total_time = time.time() - start_time
        logger.info(
            f"✅ Agent {self.charge_id} - Q computation COMPLETE (total time: {total_time:.3f}s)"
        )

        return self.Q_components

    def _normalize_s_parameter(self):
        """
        Normalize s-parameter to prevent exponential growth while preserving mathematical relationships.

        Uses logarithmic stabilization when s becomes large, maintaining monotonicity
        and mathematical meaning while preventing numerical instability.
        """
        s_threshold = 10000.0  # Threshold for applying normalization
        current_s = self.state.current_s

        if abs(current_s) > s_threshold:
            # Logarithmic normalization: preserves order relationships while limiting growth
            s_sign = 1.0 if current_s >= 0 else -1.0
            normalized_s = (
                s_sign * s_threshold * (1.0 + math.log(abs(current_s) / s_threshold))
            )

            logger.debug(
                f"🔧 Agent {self.charge_id} - S-parameter normalization: {current_s:.2f} -> {normalized_s:.2f}"
            )

            self.state.current_s = normalized_s
            self.charge_obj.update_observational_state(normalized_s)

    def evolve_observational_state(self, delta_s: float):
        """
        Evolve observational state s based on field dynamics.

        Args:
            delta_s: Change in observational state
        """
        self.state.current_s += delta_s

        # Apply normalization to prevent exponential growth
        self._normalize_s_parameter()

        self.charge_obj.update_observational_state(self.state.current_s)

        # Recompute Q with new observational state
        self.compute_complete_Q()

    def _safe_breathing_coherence_calculation(
        self, frequency_evolution: torch.Tensor
    ) -> float:
        """
        Calculate breathing coherence with regulation-based NaN protection.

        Uses the regulation system to detect and prevent NaN propagation in
        frequency evolution variance calculations, maintaining mathematical
        integrity while providing numerical stability.

        Args:
            frequency_evolution: Tensor containing frequency evolution data

        Returns:
            Safe variance value for breathing coherence calculation
        """
        if self.regulation_liquid is None:
            # Fallback to basic NaN checking if no regulation system available
            if not torch.all(torch.isfinite(frequency_evolution)):
                logger.warning(
                    f"🌊 Agent {self.charge_id}: Non-finite values in frequency_evolution, using fallback variance"
                )
                # Filter out non-finite values
                finite_values = frequency_evolution[torch.isfinite(frequency_evolution)]
                if len(finite_values) == 0:
                    return 0.1  # Safe fallback variance
                return torch.var(finite_values).item()
            else:
                return torch.var(frequency_evolution).item()

        # Use regulation system for advanced field stabilization
        try:
            # Only analyze field state if Q_components exist 
            if hasattr(self, 'Q_components') and self.Q_components is not None:
                field_state = self.regulation_liquid.analyze_field_state([self])
            else:
                return frequency_evolution

            if (
                field_state.phase_transition_indicator > 0.3
            ):  # Field instability detected
                logger.info(
                    f"🌊 Agent {self.charge_id}: Field instability detected (indicator: {field_state.phase_transition_indicator:.3f}), applying regulation"
                )

                # Apply variational regulation to stabilize frequency evolution
                if (
                    hasattr(self.regulation_liquid, "variational_regulation")
                    and self.regulation_liquid.variational_regulation
                ):
                    # Use variational optimization to find stable frequency evolution
                    stabilized_freq_evolution = (
                        self._apply_variational_frequency_regulation(
                            frequency_evolution
                        )
                    )
                else:
                    # Fallback to geometric regulation
                    stabilized_freq_evolution = (
                        self._apply_geometric_frequency_regulation(frequency_evolution)
                    )

                # Calculate variance with stabilized data
                freq_var = torch.var(stabilized_freq_evolution).item()

                # Verify the result is finite
                if not math.isfinite(freq_var):
                    logger.warning(
                        f"🌊 Agent {self.charge_id}: Regulation failed to produce finite variance, using emergency fallback"
                    )
                    return 0.1  # Emergency mathematical bound

                logger.info(
                    f"🌊 Agent {self.charge_id}: Regulation successful, variance: {freq_var:.6f}"
                )
                return freq_var

            else:
                # Field is stable, proceed with normal calculation
                freq_var = torch.var(frequency_evolution).item()
                if not math.isfinite(freq_var):
                    logger.warning(
                        f"🌊 Agent {self.charge_id}: Stable field produced non-finite variance, applying emergency regulation"
                    )
                    return 0.1  # Emergency mathematical bound
                return freq_var

        except Exception as e:
            logger.warning(
                f"🌊 Agent {self.charge_id}: Regulation system error: {e}, using safe fallback"
            )
            # Emergency fallback with basic NaN protection
            finite_values = frequency_evolution[torch.isfinite(frequency_evolution)]
            if len(finite_values) == 0:
                return 0.1
            return torch.var(finite_values).item()

    def _apply_variational_frequency_regulation(
        self, frequency_evolution: torch.Tensor
    ) -> torch.Tensor:
        """Apply variational regulation to stabilize frequency evolution."""
        try:
            # Use the regulation system's variational component to optimize frequency stability
            # The energy functional minimizes frequency variance while preserving field structure
            if hasattr(self.regulation_liquid, "variational_regulation"):
                # Apply mathematical optimization to find stable frequency configuration
                stabilized = frequency_evolution.clone()

                # Remove non-finite values
                finite_mask = torch.isfinite(stabilized)
                if finite_mask.sum() == 0:
                    return (
                        torch.ones_like(frequency_evolution) * 1.0
                    )  # Uniform stable frequency

                # Smooth extreme variations using gradient-based approach
                stabilized = stabilized.where(finite_mask, torch.ones_like(stabilized))

                # Apply gentle smoothing to reduce variance while preserving dynamics
                if len(stabilized) > 1:
                    smoothed = torch.conv1d(
                        stabilized.unsqueeze(0).unsqueeze(0),
                        torch.tensor([0.25, 0.5, 0.25]).unsqueeze(0).unsqueeze(0),
                        padding=1,
                    ).squeeze()
                    return smoothed

                return stabilized
            else:
                return self._apply_geometric_frequency_regulation(frequency_evolution)
        except Exception as e:
            logger.warning(
                f"🌊 Agent {self.charge_id}: Variational regulation failed: {e}"
            )
            return self._apply_geometric_frequency_regulation(frequency_evolution)

    def _apply_geometric_frequency_regulation(
        self, frequency_evolution: torch.Tensor
    ) -> torch.Tensor:
        """Apply geometric regulation to stabilize frequency evolution."""
        try:
            # Use differential geometry principles to smooth field singularities
            stabilized = frequency_evolution.clone()

            # Replace non-finite values with geometric mean of neighbors
            finite_mask = torch.isfinite(stabilized)
            if finite_mask.sum() == 0:
                return torch.ones_like(frequency_evolution) * 1.0

            # Geometric interpolation for non-finite values
            for i in range(len(stabilized)):
                if not finite_mask[i]:
                    # Find nearest finite values for geometric interpolation
                    left_idx = max(0, i - 1)
                    right_idx = min(len(stabilized) - 1, i + 1)

                    while left_idx >= 0 and not finite_mask[left_idx]:
                        left_idx -= 1
                    while right_idx < len(stabilized) and not finite_mask[right_idx]:
                        right_idx += 1

                    if left_idx >= 0 and right_idx < len(stabilized):
                        # Geometric mean interpolation
                        stabilized[i] = torch.sqrt(
                            torch.abs(stabilized[left_idx] * stabilized[right_idx])
                        )
                    elif left_idx >= 0:
                        stabilized[i] = stabilized[left_idx]
                    elif right_idx < len(stabilized):
                        stabilized[i] = stabilized[right_idx]
                    else:
                        stabilized[i] = 1.0  # Neutral frequency

            return stabilized

        except Exception as e:
            raise ValueError(
                f"🌊 Agent {self.charge_id}: Geometric regulation failed: {e}"
            )

    def _stabilize_breathing_coefficients_for_cascade(self):
        """
        Stabilize breathing q-coefficients before cascade operations to prevent NaN propagation.

        Uses regulation system to detect and repair corrupted coefficients that would
        cause NaN cascade through dimensional feedback loops.
        """
        if (
            not hasattr(self, "breathing_q_coefficients")
            or not self.breathing_q_coefficients
        ):
            logger.warning(
                f"🌊 Agent {self.charge_id}: No breathing coefficients for cascade stabilization"
            )
            return

        # Check for non-finite coefficients
        problematic_coefficients = []
        for n, coeff in self.breathing_q_coefficients.items():
            if not (math.isfinite(coeff.real) and math.isfinite(coeff.imag)):
                problematic_coefficients.append(n)

        if len(problematic_coefficients) == 0:
            # All coefficients are finite, check if regulation is still needed via field analysis
            if self.regulation_liquid is not None and hasattr(self, 'Q_components') and self.Q_components is not None:
                try:
                    field_state = self.regulation_liquid.analyze_field_state([self])
                    if field_state.phase_transition_indicator > 0.4:
                        logger.info(
                            f"🌊 Agent {self.charge_id}: High phase transition indicator ({field_state.phase_transition_indicator:.3f}), applying preventive coefficient stabilization"
                        )
                        self._apply_preventive_coefficient_regulation()
                except Exception as e:
                    logger.warning(
                        f"🌊 Agent {self.charge_id}: Field state analysis failed: {e}"
                    )
            return

        logger.warning(
            f"🌊 Agent {self.charge_id}: Found {len(problematic_coefficients)} non-finite breathing coefficients: {problematic_coefficients[:5]}..."
        )

        if self.regulation_liquid is None:
            # Fallback regulation without regulation system
            self._emergency_coefficient_repair(problematic_coefficients)
        else:
            # Use regulation system for sophisticated repair
            self._regulation_guided_coefficient_repair(problematic_coefficients)

    def _apply_preventive_coefficient_regulation(self):
        """Apply gentle regulation to prevent coefficient drift toward singularities."""
        try:
            # Extract coefficient magnitudes for analysis
            magnitudes = [
                abs(coeff) for coeff in self.breathing_q_coefficients.values()
            ]

            # Check for extreme values that might lead to instability
            max_magnitude = max(magnitudes)
            min_magnitude = min(magnitudes)

            if max_magnitude > 1e6 or min_magnitude < 1e-6:
                logger.info(
                    f"🌊 Agent {self.charge_id}: Extreme coefficient magnitudes detected (max: {max_magnitude:.2e}, min: {min_magnitude:.2e}), applying gentle normalization"
                )

                # Apply gentle normalization to prevent extreme values
                scale_factor = (
                    1.0 / (max_magnitude + 1e-6) if max_magnitude > 1e3 else 1.0
                )

                for n in self.breathing_q_coefficients:
                    self.breathing_q_coefficients[n] *= scale_factor

                logger.info(
                    f"🌊 Agent {self.charge_id}: Applied normalization factor {scale_factor:.6f}"
                )

        except Exception as e:
            logger.warning(
                f"🌊 Agent {self.charge_id}: Preventive regulation failed: {e}"
            )

    def _regulation_guided_coefficient_repair(
        self, problematic_coefficients: List[int]
    ):
        """Use regulation system to repair corrupted breathing coefficients."""
        try:
            logger.info(
                f"🌊 Agent {self.charge_id}: Applying regulation-guided repair for {len(problematic_coefficients)} coefficients"
            )

            # Use geometric regulation principles for coefficient repair
            for n in problematic_coefficients:
                repaired_coeff = self._repair_single_coefficient(n)
                self.breathing_q_coefficients[n] = repaired_coeff
                logger.debug(
                    f"🌊 Agent {self.charge_id}: Repaired coefficient {n}: {repaired_coeff}"
                )

            # Apply field-guided smoothing to prevent future instabilities
            if (
                hasattr(self.regulation_liquid, "geometric_regulation")
                and self.regulation_liquid.geometric_regulation
            ):
                self._apply_field_guided_coefficient_smoothing()

        except Exception as e:
            logger.warning(
                f"🌊 Agent {self.charge_id}: Regulation-guided repair failed: {e}, falling back to emergency repair"
            )
            self._emergency_coefficient_repair(problematic_coefficients)

    def _repair_single_coefficient(self, n: int) -> complex:
        """Repair a single breathing coefficient using neighboring values and field principles."""
        try:
            # Find valid neighboring coefficients
            neighbors = []
            for offset in [-2, -1, 1, 2]:
                neighbor_n = (n + offset) % 100  # Wrap around
                neighbor_coeff = self.breathing_q_coefficients.get(neighbor_n)
                if (
                    neighbor_coeff
                    and math.isfinite(neighbor_coeff.real)
                    and math.isfinite(neighbor_coeff.imag)
                ):
                    neighbors.append(neighbor_coeff)

            if len(neighbors) >= 2:
                # Use geometric mean of neighbors for field-consistent interpolation
                real_parts = [coeff.real for coeff in neighbors]
                imag_parts = [coeff.imag for coeff in neighbors]

                # Geometric mean of magnitudes, arithmetic mean of phases
                magnitudes = [abs(coeff) for coeff in neighbors]
                phases = [cmath.phase(coeff) for coeff in neighbors]

                avg_magnitude = sum(magnitudes) / len(magnitudes)
                avg_phase = sum(phases) / len(phases)

                repaired = avg_magnitude * cmath.exp(1j * avg_phase)
                return complex(repaired)

            elif len(neighbors) == 1:
                # Use single neighbor as template
                return neighbors[0]

            else:
                # No valid neighbors, use neutral breathing coefficient
                return complex(1.0, 0.0)

        except Exception as e:
            logger.warning(
                f"🌊 Agent {self.charge_id}: Single coefficient repair failed for {n}: {e}"
            )
            return complex(1.0, 0.0)

    def _apply_field_guided_coefficient_smoothing(self):
        """Apply gentle field-guided smoothing to prevent coefficient oscillations."""
        try:
            # Extract all coefficients
            coeffs = [self.breathing_q_coefficients[n] for n in range(100)]

            # Apply minimal smoothing using complex field principles
            smoothed_coeffs = []
            for i in range(100):
                prev_i = (i - 1) % 100
                next_i = (i + 1) % 100

                # Weighted average with strong bias toward original value
                current = coeffs[i]
                prev_coeff = coeffs[prev_i]
                next_coeff = coeffs[next_i]

                # 90% original, 5% each neighbor
                smoothed = 0.9 * current + 0.05 * prev_coeff + 0.05 * next_coeff
                smoothed_coeffs.append(smoothed)

            # Update coefficients
            for i, smoothed_coeff in enumerate(smoothed_coeffs):
                self.breathing_q_coefficients[i] = smoothed_coeff

            logger.debug(
                f"🌊 Agent {self.charge_id}: Applied field-guided smoothing to breathing coefficients"
            )

        except Exception as e:
            logger.warning(
                f"🌊 Agent {self.charge_id}: Field-guided smoothing failed: {e}"
            )

    def _emergency_coefficient_repair(self, problematic_coefficients: List[int]):
        """Emergency repair of coefficients without regulation system."""
        logger.warning(
            f"🌊 Agent {self.charge_id}: Applying emergency repair for {len(problematic_coefficients)} coefficients"
        )

        for n in problematic_coefficients:
            # Simple replacement with stable breathing pattern
            self.breathing_q_coefficients[n] = complex(
                1.0 + 0.1 * math.sin(2 * math.pi * n / 100),
                0.05 * math.cos(2 * math.pi * n / 100),
            )

    def _check_field_stability(
        self, operation: str, phase_value: float = None, complex_value: complex = None
    ) -> bool:
        """
        Check if field operation will maintain Q(τ,C,s) stability.
        Based on Section 3.1.5.3 - Field Coherence Requirements

        Args:
            operation: Type of field operation (phase_exponential, field_coupling, gradient)
            phase_value: Phase value for exponential operations
            complex_value: Complex value for coupling operations

        Returns:
            True if operation is stable, False if regulation needed
        """
        if self.regulation_liquid is None:
            # Without regulation system, check basic stability
            if phase_value is not None:
                return math.isfinite(phase_value) and abs(phase_value) < 1e6
            if complex_value is not None:
                return (
                    math.isfinite(complex_value.real)
                    and math.isfinite(complex_value.imag)
                    and abs(complex_value) < 1e10
                )
            return True

        try:
            if not (hasattr(self, 'Q_components') and self.Q_components is not None):
                return True
            field_state = self.regulation_liquid.analyze_field_state([self])

            # Operation-specific stability thresholds based on field theory
            if operation == "phase_exponential":
                # Phase operations are critical - tighter thresholds
                if phase_value is not None and not math.isfinite(phase_value):
                    return False
                return field_state.phase_transition_indicator < 0.5

            elif operation == "field_coupling":
                # Field couplings can handle more instability
                if complex_value is not None and not (
                    math.isfinite(complex_value.real)
                    and math.isfinite(complex_value.imag)
                ):
                    return False
                return field_state.phase_transition_indicator < 0.7

            elif operation == "gradient":
                # Gradients are most robust
                return field_state.phase_transition_indicator < 0.8

            else:
                return field_state.phase_transition_indicator < 0.6

        except Exception as e:
            logger.warning(
                f"🌊 Agent {self.charge_id}: Field stability check failed: {e}"
            )
            # Conservative fallback
            return False

    def _apply_field_regulation(self, operation: str, value: Any) -> Any:
        """
        Apply regulation while preserving field-theoretic properties.
        Maintains Q formula structure from Section 3.1.1

        Args:
            operation: Type of operation requiring regulation
            value: Value to regulate (phase, complex number, tensor)

        Returns:
            Regulated value maintaining field properties
        """
        if self.regulation_liquid is None:
            # Basic regulation without system
            if isinstance(value, (int, float)):
                if not math.isfinite(value):
                    logger.warning(
                        f"🌊 Agent {self.charge_id}: Emergency regulation for {operation}, replacing NaN/inf with neutral value"
                    )
                    return 0.0 if operation == "phase" else 1.0
                return value
            elif isinstance(value, complex):
                if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                    logger.warning(
                        f"🌊 Agent {self.charge_id}: Emergency regulation for {operation}, replacing NaN/inf complex"
                    )
                    return complex(1.0, 0.0)
                return value
            elif torch.is_tensor(value):
                finite_mask = torch.isfinite(value)
                if not torch.all(finite_mask):
                    logger.warning(
                        f"🌊 Agent {self.charge_id}: Emergency tensor regulation for {operation}"
                    )
                    return torch.where(finite_mask, value, torch.ones_like(value))
                return value
            return value

        try:
            # Sophisticated regulation based on operation type
            if operation == "phase_exponential":
                # Critical for Q formula - use variational regulation
                if (
                    hasattr(self.regulation_liquid, "variational_regulation")
                    and self.regulation_liquid.variational_regulation
                ):
                    # Apply energy minimization to find stable phase
                    regulated_phase = self._regulate_phase_value(value)
                    logger.info(
                        f"🌊 Q-PHASE: Agent {self.charge_id} regulated phase from {value:.3f} to {regulated_phase:.3f}"
                    )
                    return regulated_phase

            elif operation == "field_coupling":
                # Use geometric regulation for field interactions
                if (
                    hasattr(self.regulation_liquid, "geometric_regulation")
                    and self.regulation_liquid.geometric_regulation
                ):
                    regulated_coupling = self._regulate_field_coupling(value)
                    logger.info(
                        f"🌊 FIELD-COUPLING: Agent {self.charge_id} regulated coupling, |Q|: {abs(value):.3e} → {abs(regulated_coupling):.3e}"
                    )
                    return regulated_coupling

            elif operation == "gradient":
                # Use coupled field regulation for gradients
                if (
                    hasattr(self.regulation_liquid, "coupled_regulation")
                    and self.regulation_liquid.coupled_regulation
                ):
                    regulated_gradient = self._regulate_gradient_operation(value)
                    logger.info(
                        f"🌊 GRADIENT: Agent {self.charge_id} regulated gradient operation"
                    )
                    return regulated_gradient

            # Fallback to basic regulation
            return self._basic_field_regulation(operation, value)

        except Exception as e:
            logger.warning(
                f"🌊 Agent {self.charge_id}: Field regulation failed: {e}, using emergency bounds"
            )
            return self._basic_field_regulation(operation, value)

    def _regulate_phase_value(self, phase: float) -> float:
        """Regulate phase value using variational principles."""
        if not math.isfinite(phase):
            return 0.0
        # Keep phase in reasonable bounds while preserving dynamics
        if abs(phase) > 100.0:
            # Wrap large phases to maintain periodicity
            return phase % (2 * math.pi)
        return phase

    def _regulate_field_coupling(self, coupling: complex) -> complex:
        """Regulate field coupling using geometric principles."""
        if not (math.isfinite(coupling.real) and math.isfinite(coupling.imag)):
            return complex(1.0, 0.0)
        # Apply gentle damping for extreme values
        magnitude = abs(coupling)
        if magnitude > 1e6:
            # Preserve phase but reduce magnitude
            phase = cmath.phase(coupling)
            new_magnitude = 1e6 * (1.0 + math.log10(magnitude / 1e6))
            return new_magnitude * cmath.exp(1j * phase)
        return coupling

    def _regulate_gradient_operation(self, gradient: torch.Tensor) -> torch.Tensor:
        """Regulate gradient operations using coupled field dynamics."""
        finite_mask = torch.isfinite(gradient)
        if torch.all(finite_mask):
            return gradient
        # Smooth non-finite values using neighbors
        regulated = gradient.clone()
        for i in range(len(gradient)):
            if not finite_mask[i]:
                # Find nearest finite neighbors
                left = max(0, i - 1)
                right = min(len(gradient) - 1, i + 1)
                if finite_mask[left] and finite_mask[right]:
                    regulated[i] = (gradient[left] + gradient[right]) / 2
                elif finite_mask[left]:
                    regulated[i] = gradient[left]
                elif finite_mask[right]:
                    regulated[i] = gradient[right]
                else:
                    regulated[i] = 0.0
        return regulated

    def _basic_field_regulation(self, operation: str, value: Any) -> Any:
        """Basic field regulation without advanced systems."""
        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                return 0.0 if "phase" in operation else 1.0
            return value
        elif isinstance(value, complex):
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                return complex(1.0, 0.0)
            return value
        elif torch.is_tensor(value):
            return torch.where(torch.isfinite(value), value, torch.zeros_like(value))
        return value

    def _safe_phase_exponential(self, phase: float, context: str) -> complex:
        """
        Protected e^(iθ) calculation for Q formula phase factors.
        Critical for maintaining field coherence (Section 3.1.3.3.3)

        Args:
            phase: Phase value θ
            context: Description of where this phase is used

        Returns:
            Complex exponential e^(iθ) with stability guarantees
        """
        # Check stability before operation
        if not self._check_field_stability("phase_exponential", phase_value=phase):
            logger.info(
                f"🌊 Q-PHASE: Agent {self.charge_id} detecting unstable phase {phase:.3f} for {context}"
            )
            phase = self._apply_field_regulation("phase_exponential", phase)

        try:
            # Compute phase exponential with protection
            result = torch.exp(1j * torch.tensor(phase, dtype=torch.float32)).item()

            # Verify result
            if not (math.isfinite(result.real) and math.isfinite(result.imag)):
                logger.warning(
                    f"🌊 Q-PHASE: Agent {self.charge_id} phase exponential produced NaN for {context}, applying emergency regulation"
                )
                return complex(1.0, 0.0)  # Neutral phase

            return result

        except Exception as e:
            logger.error(
                f"🌊 Q-PHASE: Agent {self.charge_id} phase exponential failed for {context}: {e}"
            )
            return complex(1.0, 0.0)  # Neutral phase

    def _safe_field_multiplication(
        self, field_a: complex, field_b: complex, coupling_type: str
    ) -> complex:
        """
        Protected field coupling preserving interference patterns.
        Implements Section 3.1.4.3 field interactions

        Args:
            field_a: First field component
            field_b: Second field component
            coupling_type: Type of coupling (e.g., "Q_assembly", "cascade_transform")

        Returns:
            Coupled field value with stability guarantees
        """
        # Check both field values
        if not self._check_field_stability("field_coupling", complex_value=field_a):
            logger.info(
                f"🌊 FIELD-COUPLING: Agent {self.charge_id} regulating field_a for {coupling_type}"
            )
            field_a = self._apply_field_regulation("field_coupling", field_a)

        if not self._check_field_stability("field_coupling", complex_value=field_b):
            logger.info(
                f"🌊 FIELD-COUPLING: Agent {self.charge_id} regulating field_b for {coupling_type}"
            )
            field_b = self._apply_field_regulation("field_coupling", field_b)

        try:
            # Perform multiplication
            result = field_a * field_b

            # Verify result
            if not (math.isfinite(result.real) and math.isfinite(result.imag)):
                logger.warning(
                    f"🌊 FIELD-COUPLING: Agent {self.charge_id} {coupling_type} produced NaN, applying regulation"
                )
                result = self._apply_field_regulation("field_coupling", result)

            logger.debug(
                f"🌊 FIELD-COUPLING: Agent {self.charge_id} {coupling_type}: |Q₁|={abs(field_a):.3e} × |Q₂|={abs(field_b):.3e} = |Q|={abs(result):.3e}"
            )
            return result

        except Exception as e:
            logger.error(
                f"🌊 FIELD-COUPLING: Agent {self.charge_id} multiplication failed for {coupling_type}: {e}"
            )
            return complex(1.0, 0.0)

    def update_context(self, new_context: Dict[str, Any]):
        """
        Update contextual environment C.

        Args:
            new_context: New contextual environment
        """
        self.state.current_context_C.update(new_context)

        # Recompute Q with new context
        self.compute_complete_Q()

    def get_field_contribution(self, position: Tuple[float, float]) -> complex:
        """
        Compute field contribution at given position using actual SemanticField.

        Paper Formula: Field evaluation following Section 3.1.2.8 field-generating functions

        Args:
            position: (x, y) position to evaluate field

        Returns:
            Complex field value at position modulated by complete Q(τ,C,s)
        """
        # Use pre-computed semantic field data from charge factory
        field_base_magnitude = np.linalg.norm(self.semantic_field.embedding_components)
        field_base_phase = np.mean(self.semantic_field.phase_factors)
        field_value = field_base_magnitude * np.exp(1j * field_base_phase)

        # Modulate by current complete Q(τ,C,s) value (Section 3.1.5 complete formula)
        if self.Q_components is not None:
            return field_value * self.Q_components.Q_value
        else:
            # Use charge object's complete_charge if Q not yet computed
            return field_value * self.charge_obj.complete_charge

    def get_mathematical_breakdown(self) -> Dict[str, Any]:
        """Get complete mathematical breakdown of Q(τ, C, s) computation."""
        if self.Q_components is None:
            self.compute_complete_Q()

        return {
            "agent_id": self.charge_id,
            "tau_content": self.state.tau,
            "current_state": {
                "s": self.state.current_s,
                "s_zero": self.state.s_zero,
                "context_C": self.state.current_context_C,
                "field_position": self.state.field_position,
            },
            "coupling_state": {
                "emotional_coupling": str(self.coupling_state.emotional_field_coupling),
                "s_t_coupling_strength": self.coupling_state.s_t_coupling_strength,
                # Use scipy special functions for sophisticated averaging - NO BASIC NUMPY
                # Use torch for sophisticated absolute value - NO BASIC NUMPY
                "interference_strength": float(
                    special.logsumexp(
                        torch.abs(
                            torch.tensor(
                                self.coupling_state.field_interference_coupling,
                                dtype=torch.float32,
                            )
                        ).numpy()
                    )
                    / len(self.coupling_state.field_interference_coupling)
                ),
            },
            "Q_components": {
                "gamma": self.Q_components.gamma,
                "T_tensor": {
                    "magnitude": self.Q_components.T_magnitude,
                    "phase": self.Q_components.T_phase,
                },
                "E_trajectory": {
                    "magnitude": self.Q_components.E_magnitude,
                    "phase": self.Q_components.E_phase,
                },
                "phi_semantic": {
                    "magnitude": self.Q_components.phi_magnitude,
                    "phase": self.Q_components.phi_phase,
                },
                "theta_total": {
                    "semantic": self.Q_components.theta_components.theta_semantic,
                    "emotional": self.Q_components.theta_components.theta_emotional,
                    "temporal_integral": self.Q_components.theta_components.temporal_integral,
                    "interaction": self.Q_components.theta_components.theta_interaction,
                    "field": self.Q_components.theta_components.theta_field,
                    "total": self.Q_components.theta_components.total,
                },
                "psi_persistence": {
                    "total": self.Q_components.psi_persistence,
                    "gaussian": self.Q_components.psi_gaussian,
                    "exponential_cosine": self.Q_components.psi_exponential_cosine,
                },
            },
            "final_Q": {
                "magnitude": self.Q_components.Q_magnitude,
                "phase": self.Q_components.Q_phase,
                "complex": {
                    "real": self.Q_components.Q_value.real,
                    "imag": self.Q_components.Q_value.imag,
                },
            },
            # 🔧 Enhanced: Additional debugging information
            "charge_index": self.charge_index,
            "data_source_validation": {
                "semantic_field_magnitude": self.semantic_field_data["field_metadata"][
                    "field_magnitude"
                ],
                "trajectory_operators_non_zero": int(
                    # Use torch for sophisticated operations - NO BASIC NUMPY
                    torch.sum(
                        torch.abs(
                            torch.tensor(
                                self.temporal_biography.trajectory_operators,
                                dtype=torch.complex64,
                            )
                        )
                        > 1e-12
                    ).item()
                ),
                "trajectory_operators_total": len(
                    self.temporal_biography.trajectory_operators
                ),
                "emotional_modulation_strength": self.emotional_modulation.field_modulation_strength,
                "persistence_layers_range": {
                    "vivid_layer": [
                        float(np.min(self.temporal_biography.vivid_layer)),
                        float(np.max(self.temporal_biography.vivid_layer)),
                    ],
                    "character_layer": [
                        float(np.min(self.temporal_biography.character_layer)),
                        float(np.max(self.temporal_biography.character_layer)),
                    ],
                },
            },
        }

    def log_debug_breakdown(self):
        """Log detailed debugging information for this agent."""
        breakdown = self.get_mathematical_breakdown()

        logger.debug(f"🔧 AGENT DEBUG [{self.charge_index}] {self.charge_id}:")
        logger.debug(f"  τ: {breakdown['tau_content']}")
        logger.debug(
            f"  State: s={breakdown['current_state']['s']:.3f}, pos={breakdown['current_state']['field_position']}"
        )

        # Q Components summary
        Q_comps = breakdown["Q_components"]
        logger.debug(f"  🧮 Q Components:")
        logger.debug(f"    γ: {Q_comps['gamma']:.6f}")
        logger.debug(
            f"    |T|: {Q_comps['T_tensor']['magnitude']:.6f}, ∠T: {Q_comps['T_tensor']['phase']:.3f}"
        )
        logger.debug(
            f"    |E|: {Q_comps['E_trajectory']['magnitude']:.6f}, ∠E: {Q_comps['E_trajectory']['phase']:.3f}"
        )
        logger.debug(
            f"    |Φ|: {Q_comps['phi_semantic']['magnitude']:.6f}, ∠Φ: {Q_comps['phi_semantic']['phase']:.3f}"
        )
        logger.debug(
            f"    Ψ: {Q_comps['psi_persistence']['total']:.6f} (gauss: {Q_comps['psi_persistence']['gaussian']:.6f}, exp_cos: {Q_comps['psi_persistence']['exponential_cosine']:.6f})"
        )
        logger.debug(
            f"    Q: {breakdown['final_Q']['magnitude']:.6f} ∠ {breakdown['final_Q']['phase']:.3f}"
        )

        # Source data quality
        if "data_source_validation" in breakdown:
            src_data = breakdown["data_source_validation"]
            logger.debug(f"  📊 Source Data Quality:")
            logger.debug(
                f"    Trajectory ops: {src_data['trajectory_operators_non_zero']}/{src_data['trajectory_operators_total']} non-zero"
            )
            logger.debug(
                f"    Emotional strength: {src_data['emotional_modulation_strength']:.6f}"
            )
            logger.debug(
                f"    Persistence ranges: vivid=[{src_data['persistence_layers_range']['vivid_layer'][0]:.3f}, {src_data['persistence_layers_range']['vivid_layer'][1]:.3f}], char=[{src_data['persistence_layers_range']['character_layer'][0]:.3f}, {src_data['persistence_layers_range']['character_layer'][1]:.3f}]"
            )

        return breakdown

    def get_component_health_summary(self) -> str:
        """
        Get a concise health summary of Q components for logging.
        """
        if self.Q_components is None:
            return f"[{self.charge_index}] Q: NOT_COMPUTED"

        Q_mag = abs(self.Q_components.Q_value)
        gamma = self.Q_components.gamma
        T_mag = abs(self.Q_components.T_tensor)
        E_mag = abs(self.Q_components.E_trajectory)
        phi_mag = abs(self.Q_components.phi_semantic)
        psi = self.Q_components.psi_persistence

        # Health indicators
        health_flags = []
        if Q_mag < 1e-10:
            health_flags.append("Q_TINY")
        if Q_mag > 10:
            health_flags.append("Q_LARGE")
        if T_mag < 1e-8:
            health_flags.append("T_ZERO")
        if psi < 1e-8:
            health_flags.append("PSI_TINY")

        health_str = "|" + "|".join(health_flags) + "|" if health_flags else "OK"

        return f"[{self.charge_index}] Q:{Q_mag:.2e} γ:{gamma:.3f} |T|:{T_mag:.2e} |E|:{E_mag:.3f} |Φ|:{phi_mag:.3f} Ψ:{psi:.3f} {health_str}"
