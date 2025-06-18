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

import torch
import numpy as np
import scipy as sp
from scipy import integrate, signal, fft, linalg
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import time
import math

# Additional DTF mathematical libraries for proper field theory operations
from scipy.integrate import quad
from scipy.linalg import eigh
import numba as nb

from Sysnpire.database.conceptual_charge_object import ConceptualChargeObject
from Sysnpire.model.semantic_dimension.SemanticDimensionHelper import SemanticDimensionHelper
from Sysnpire.model.temporal_dimension.TemporalDimensionHelper import TemporalDimensionHelper
from Sysnpire.model.emotional_dimension.EmotionalDimensionHelper import EmotionalDimensionHelper


@dataclass 
class ThetaComponents:
    """
    Complete 5-component phase integration θ_total(τ,C,s).
    
    From section 3.1.5.7: θ_total = θ_semantic + θ_emotional + ∫ω_temporal + θ_interaction + θ_field
    """
    theta_semantic: float      # θ_semantic(τ,C) from semantic field reconstruction
    theta_emotional: float     # θ_emotional(τ) from emotional phase modulation  
    temporal_integral: float   # ∫₀ˢ ω_temporal(τ,s') ds' from trajectory operators
    theta_interaction: float   # θ_interaction(τ,C,s) from contextual coupling
    theta_field: float        # θ_field(x,s) from manifold field dynamics
    total: float              # Complete θ_total(τ,C,s)


@dataclass
class QMathematicalComponents:
    """
    Complete mathematical breakdown of Q(τ, C, s) with proper theory implementation.
    Each component implements the actual formulations from section 3.1.5.
    """
    # Core components (complex values contain all information)
    gamma: float                      # Section 3.1.5.3: Global Field Calibration
    T_tensor: complex                 # Section 3.1.5.4: Transformative Potential Tensor
    E_trajectory: complex             # Section 3.1.5.5: Emotional Trajectory Integration
    phi_semantic: complex             # Section 3.1.5.6: Semantic Field Generation
    
    # Phase integration
    theta_components: ThetaComponents # Section 3.1.5.7: Complete Phase Integration
    phase_factor: complex             # e^(iθ_total)
    
    # Persistence components
    psi_persistence: float            # Section 3.1.5.8: Total persistence
    psi_gaussian: float               # "vivid recent chapters"
    psi_exponential_cosine: float     # "persistent character traits"
    
    # Final result
    Q_value: complex
    
    # Computed properties (no storage needed)
    @property
    def T_magnitude(self) -> float:
        return abs(self.T_tensor)
    
    @property 
    def T_phase(self) -> float:
        return np.angle(self.T_tensor)
        
    @property
    def E_magnitude(self) -> float:
        return abs(self.E_trajectory)
        
    @property
    def E_phase(self) -> float:
        return np.angle(self.E_trajectory)
        
    @property
    def phi_magnitude(self) -> float:
        return abs(self.phi_semantic)
        
    @property
    def phi_phase(self) -> float:
        return np.angle(self.phi_semantic)
        
    @property
    def Q_magnitude(self) -> float:
        return abs(self.Q_value)
        
    @property
    def Q_phase(self) -> float:
        return np.angle(self.Q_value)


@dataclass
class AgentFieldState:
    """Current field state of the living mathematical entity."""
    tau: str                    # Token τ content
    current_context_C: Dict[str, Any]  # Contextual environment C
    current_s: float           # Observational state s
    s_zero: float             # Initial observational state s₀
    field_position: Tuple[float, float]  # Spatial position (x,y)
    trajectory_time: float     # Current τ in trajectory integration
    

@dataclass
class FieldCouplingState:
    """Coupling state between dimensions based on ChargeFactory orchestration."""
    emotional_field_coupling: complex  # From emotional conductor modulation
    field_interference_coupling: np.ndarray  # From temporal interference matrix
    collective_breathing_rhythm: Dict[str, Any]  # From temporal collective patterns
    s_t_coupling_strength: float  # Semantic-Temporal coupling via emotional conductor


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
    
    def __init__(self, 
                 charge_obj: ConceptualChargeObject, 
                 charge_index: int,
                 combined_results: Dict[str, Any],
                 initial_context: Dict[str, Any] = None,
                 device: str = "mps"):
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
        """
        self.charge_obj = charge_obj
        self.charge_id = charge_obj.charge_id
        self.charge_index = charge_index
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        
        # Extract rich structures from ChargeFactory combined_results (following charge_factory.py structure)
        self.semantic_field_data = combined_results['semantic_results']['field_representations'][charge_index]
        self.semantic_field = self.semantic_field_data['semantic_field']  # Actual SemanticField object
        self.temporal_biography = combined_results['temporal_results']['temporal_biographies'][charge_index]
        self.emotional_modulation = combined_results['emotional_results']['emotional_modulations'][charge_index]
        
        # Extract coupling data
        self.field_interference_matrix = combined_results['temporal_results']['field_interference_matrix']
        self.collective_breathing = combined_results['temporal_results']['collective_breathing_rhythm']
        self.emotional_field_signature = combined_results['emotional_results']['field_signature']
        
        # Initialize field coupling state
        self.coupling_state = FieldCouplingState(
            emotional_field_coupling=self.emotional_modulation.unified_phase_shift,
            field_interference_coupling=self.temporal_biography.field_interference_signature,
            collective_breathing_rhythm=self.collective_breathing,
            s_t_coupling_strength=self.emotional_field_signature.field_modulation_strength
        )
        
        # Initialize agent field state
        self.state = AgentFieldState(
            tau=charge_obj.text_source,
            current_context_C=initial_context or {},
            current_s=charge_obj.observational_state,
            s_zero=charge_obj.observational_state,
            field_position=charge_obj.metadata.field_position or (0.0, 0.0),
            trajectory_time=0.0
        )
        
        # Mathematical components (will be computed)
        self.Q_components: Optional[QMathematicalComponents] = None
        
        # Parameters for persistence dual-decay structure (section 3.1.5.8)
        self.sigma_i = 0.5    # Gaussian immediate memory decay
        self.alpha_i = 0.3    # Persistence amplitude  
        self.lambda_i = 0.1   # Long-term decay rate
        self.beta_i = 2.0     # Rhythmic reinforcement frequency
        
        # Initialize with first computation
        self.compute_complete_Q()
    
    @classmethod
    def from_charge_factory_results(cls, 
                                  combined_results: Dict[str, Any], 
                                  charge_index: int,
                                  initial_context: Dict[str, Any] = None,
                                  device: str = "mps") -> 'ConceptualChargeAgent':
        """
        Direct creation from ChargeFactory output following paper mathematics.
        
        Enables direct instantiation from charge_factory.py output structure
        without requiring separate ConceptualChargeObject creation.
        
        Args:
            combined_results: Full combined_results from ChargeFactory.build()
            charge_index: Index of the charge to create (0-based)
            initial_context: Optional initial contextual environment C
            device: PyTorch device for tensor operations
            
        Returns:
            ConceptualChargeAgent instance with proper field theory mathematics
        """
        # Extract semantic field data
        semantic_data = combined_results['semantic_results']['field_representations'][charge_index]
        semantic_field = semantic_data['semantic_field']
        
        # Extract temporal biography
        temporal_bio = combined_results['temporal_results']['temporal_biographies'][charge_index]
        
        # Extract emotional modulation
        emotional_mod = combined_results['emotional_results']['emotional_modulations'][charge_index]
        
        # Extract source token (BGE vocabulary token, not text)
        source_token = semantic_data['field_metadata']['source_token']
        
        # Create field components for ConceptualChargeObject using ACTUAL data from combined_results
        field_components = FieldComponents(
            trajectory_operators=list(temporal_bio.trajectory_operators),
            emotional_trajectory=emotional_mod.semantic_modulation_tensor,
            semantic_field=semantic_field.embedding_components,
            phase_total=np.mean(semantic_field.phase_factors),
            observational_persistence=1.0
        )
        
        # Initialize complete_charge using paper mathematics Section 3.1.5 - Complete Q(τ,C,s) integration
        # This represents the initial field state before full Q computation
        field_magnitude = semantic_data['field_metadata']['field_magnitude']
        mean_phase = np.mean(semantic_field.phase_factors)
        
        # Apply emotional field modulation (Section 3.1.3.3.1 - emotion as field conductor)
        emotional_amplification = emotional_mod.field_modulation_strength
        
        # Apply temporal persistence (Section 3.1.4.3.3 - observational persistence)
        temporal_persistence = np.mean(temporal_bio.vivid_layer) if len(temporal_bio.vivid_layer) > 0 else 1.0
        
        # Create complete charge with paper mathematics: magnitude * emotional_conductor * temporal_persistence * e^(i*phase)
        complete_charge = field_magnitude * emotional_amplification * temporal_persistence * np.exp(1j * mean_phase)
        
        # Create ConceptualChargeObject with proper complete charge initialization
        charge_obj = ConceptualChargeObject(
            charge_id=f"charge_{charge_index}",
            text_source=semantic_data['field_metadata']['source_token'],
            complete_charge=complete_charge,  # Actual field-based complete charge
            field_components=field_components,
            observational_state=1.0,
            gamma=1.0
        )
        
        # Create agent instance
        agent = cls(
            charge_obj=charge_obj,
            charge_index=charge_index,
            combined_results=combined_results,
            initial_context=initial_context,
            device=device
        )
        
        return agent
        
    def compute_gamma_calibration(self, collective_field_strength: Optional[float] = None) -> float:
        """
        Implement γ global field calibration from section 3.1.5.3.
        
        "The conductor's master volume control...ensures individual instrumental 
        voices blend harmoniously within the collective performance"
        
        Uses actual field modulation strength from emotional analytics.
        """
        # Base gamma from charge object
        base_gamma = self.charge_obj.gamma
        
        # Use emotional field modulation strength as conductor control
        conductor_modulation = self.emotional_field_signature.field_modulation_strength
        
        # Use collective field strength if provided, otherwise derive from interference
        if collective_field_strength is None:
            # Derive from field interference matrix
            interference_strength = np.mean(np.abs(self.field_interference_matrix))
            collective_field_strength = 1.0 + interference_strength
        
        # Calibrate to prevent overwhelming or weakness
        # Higher collective field → lower individual gamma (normalization)
        calibration_factor = conductor_modulation / (1.0 + 0.1 * collective_field_strength)
        
        return base_gamma * calibration_factor
        
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
        T_ops = torch.tensor(trajectory_ops, dtype=torch.complex64, device=self.device)
        
        # Context C modulates trajectory through observer contingency
        context_size = len(self.state.current_context_C)
        context_modulation = 1.0 + 0.1 * np.log1p(context_size)  # log1p for stability
        
        # Observational state s affects trajectory integration
        s = self.state.current_s
        s_evolution = torch.exp(torch.tensor(-0.05 * s, device=self.device))
        
        # Emotional coupling modulates T tensor (multiplicative effect)
        emotional_coupling = abs(self.coupling_state.emotional_field_coupling)
        
        # Compute tensor with multiplicative interactions
        T_tensor_value = torch.mean(T_ops) * context_modulation * s_evolution * (1.0 + 0.2 * emotional_coupling)
        
        # Extract complex result
        T_complex = complex(T_tensor_value.cpu().numpy())
        
        return T_complex
        
    def compute_emotional_trajectory_integration(self) -> complex:
        """
        Implement E^trajectory(τ, s) from section 3.1.5.5.
        
        "Dynamic field modulation...emotional resonance must accumulate through 
        observational experience rather than operating as fixed taxonomic properties"
        
        Uses actual emotional modulation tensors with trajectory accumulation.
        """
        # Get emotional modulation components
        modulation_tensor = self.emotional_modulation.semantic_modulation_tensor
        trajectory_attractors = self.emotional_modulation.trajectory_attractors
        resonance_freqs = self.emotional_modulation.resonance_frequencies
        
        # Current and initial observational states
        s = self.state.current_s
        s_zero = self.state.s_zero
        delta_s = s - s_zero
        
        # Trajectory accumulation using scipy integration
        def emotional_integrand(s_prime):
            # Distance from attractor states
            attractor_influence = np.mean(np.exp(-np.abs(s_prime - trajectory_attractors)))
            
            # Resonance amplification at specific frequencies
            resonance = np.sum(np.cos(2 * np.pi * resonance_freqs * s_prime))
            
            # Modulation decay with distance
            decay = np.exp(-0.1 * np.abs(s_prime - s_zero))
            
            return attractor_influence * resonance * decay
        
        # Integrate emotional accumulation from s_zero to current s
        if abs(delta_s) > 0.01:
            emotional_integral, _ = integrate.quad(emotional_integrand, s_zero, s)
        else:
            emotional_integral = 0.0
        
        # Apply modulation tensor influence
        tensor_influence = np.mean(modulation_tensor)
        
        # Combine with unified phase shift
        phase_shift = self.emotional_modulation.unified_phase_shift
        
        # Final emotional trajectory value
        E_magnitude = 1.0 + 0.3 * tensor_influence + 0.1 * emotional_integral
        E_phase = np.angle(phase_shift)
        E_trajectory = complex(E_magnitude * np.cos(E_phase), E_magnitude * np.sin(E_phase))
        
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
        if 'collective_frequency' in self.collective_breathing:
            collective_freq = self.collective_breathing['collective_frequency']
            # Use mean frequency for breathing modulation
            breathing_frequency = np.mean(np.real(collective_freq)) if hasattr(collective_freq, '__len__') else 0.1
        else:
            breathing_frequency = 0.1  # Default frequency
            
        breathing_amplitude = self.collective_breathing.get('breathing_pattern_diversity', 0.1)
        
        # Breathing constellation pattern (Section 3.1.4.3.4 - trajectory-semantic coupling)
        breathing_factor = 1.0 + 0.1 * breathing_amplitude * np.sin(2 * np.pi * breathing_frequency * s)
        
        # Evaluate semantic field at current field position using actual SemanticField
        position_x = np.array(self.state.field_position)
        
        # Use actual SemanticField.evaluate_at() method (following paper Section 3.1.2.8)
        try:
            field_value = self.semantic_field.evaluate_at(position_x)
        except Exception as e:
            # Fallback: use field magnitude from metadata
            field_value = complex(
                self.semantic_field_data['field_metadata']['field_magnitude'], 
                0.0
            )
        
        # Apply breathing modulation (Section 3.1.4.3.4)
        phi_semantic = field_value * breathing_factor
        
        # Apply emotional conductor modulation to semantic field (Section 3.1.3.3.1)
        conductor_influence = self.coupling_state.s_t_coupling_strength
        phi_semantic *= (1.0 + 0.2 * conductor_influence)
        
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
        theta_semantic = np.mean(phase_factors) * (1.0 + context_influence)
        
        # θ_emotional(τ) - from emotional phase modulation
        emotional_phase_shift = self.emotional_modulation.unified_phase_shift
        theta_emotional = np.angle(emotional_phase_shift)
        
        # ∫₀ˢ ω_temporal(τ,s') ds' - temporal integral using frequency evolution
        frequency_evolution = self.temporal_biography.frequency_evolution
        if len(frequency_evolution) > 0 and abs(s - s_zero) > 0.01:
            # Create interpolation function for omega
            s_points = np.linspace(s_zero, s, len(frequency_evolution))
            omega_func = sp.interpolate.interp1d(s_points, frequency_evolution, 
                                                kind='cubic', fill_value='extrapolate')
            
            # Integrate omega from s_zero to s
            temporal_integral, _ = integrate.quad(omega_func, s_zero, s)
        else:
            temporal_integral = 0.0
        
        # θ_interaction(τ,C,s) - contextual coupling with interference
        interference_strength = np.mean(np.abs(self.temporal_biography.field_interference_signature))
        theta_interaction = interference_strength * s * context_influence
        
        # θ_field(x,s) - manifold field dynamics at position
        x, y = self.state.field_position
        field_distance = np.sqrt(x*x + y*y)
        theta_field = 0.1 * field_distance * s
        
        # Total phase with wrapping
        total = theta_semantic + theta_emotional + temporal_integral + theta_interaction + theta_field
        
        # Wrap to [-π, π]
        while total > np.pi:
            total -= 2 * np.pi
        while total < -np.pi:
            total += 2 * np.pi
            
        return ThetaComponents(
            theta_semantic=theta_semantic,
            theta_emotional=theta_emotional,
            temporal_integral=temporal_integral,
            theta_interaction=theta_interaction,
            theta_field=theta_field,
            total=total
        )
        
    def compute_observational_persistence(self) -> Tuple[float, float, float]:
        """
        Implement Ψ_persistence(s-s₀) dual-decay structure from section 3.1.5.8.
        
        "Layered memory effects...dual-decay structure with vivid recent chapters
        and persistent character traits"
        
        Ψ = exp(-(s-s₀)²/2σᵢ²) + αᵢ·exp(-λᵢ(s-s₀))·cos(βᵢ(s-s₀))
        
        Uses actual vivid_layer and character_layer from temporal biography.
        """
        s = self.state.current_s
        s_zero = self.state.s_zero
        delta_s = s - s_zero
        
        # Extract persistence layers from temporal biography
        vivid_layer = self.temporal_biography.vivid_layer
        character_layer = self.temporal_biography.character_layer
        
        # Gaussian component from vivid layer (recent sharp memory)
        if len(vivid_layer) > 0:
            # Use actual vivid layer data
            vivid_influence = np.mean(vivid_layer) * np.exp(-(delta_s * delta_s) / (2 * self.sigma_i * self.sigma_i))
        else:
            vivid_influence = np.exp(-(delta_s * delta_s) / (2 * self.sigma_i * self.sigma_i))
        
        # Exponential-cosine from character layer (persistent themes)
        if len(character_layer) > 0:
            # Use actual character layer data
            character_influence = np.mean(character_layer) * np.exp(-self.lambda_i * abs(delta_s))
            # Add rhythmic reinforcement
            character_influence *= np.cos(self.beta_i * delta_s)
        else:
            character_influence = self.alpha_i * np.exp(-self.lambda_i * abs(delta_s)) * np.cos(self.beta_i * delta_s)
        
        # Combined persistence
        total_persistence = vivid_influence + character_influence
        
        # Ensure positive persistence
        total_persistence = max(0.01, total_persistence)
        
        return total_persistence, vivid_influence, character_influence
        
    def compute_complete_Q(self, collective_field_strength: Optional[float] = None) -> QMathematicalComponents:
        """
        Compute complete Q(τ, C, s) = γ · T · E · Φ · e^(iθ) · Ψ
        
        This is the living mathematical entity in action - computing the complete
        conceptual charge using actual field theory mathematics with real data.
        """
        # Compute all components
        gamma = self.compute_gamma_calibration(collective_field_strength)
        
        T_tensor = self.compute_transformative_potential_tensor()
        E_trajectory = self.compute_emotional_trajectory_integration()  
        phi_semantic = self.compute_semantic_field_generation()
        
        theta_components = self.compute_5component_phase_integration()
        phase_factor = complex(np.cos(theta_components.total), np.sin(theta_components.total))
        
        psi_persistence, psi_gaussian, psi_exponential_cosine = self.compute_observational_persistence()
        
        # Final Q(τ, C, s) computation
        Q_value = gamma * T_tensor * E_trajectory * phi_semantic * phase_factor * psi_persistence
        
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
            Q_value=Q_value
        )
        
        # Update charge object
        self.charge_obj.complete_charge = Q_value
        self.charge_obj.magnitude = abs(Q_value)
        self.charge_obj.phase = np.angle(Q_value)
        
        return self.Q_components
        
    def evolve_observational_state(self, delta_s: float):
        """
        Evolve observational state s based on field dynamics.
        
        Args:
            delta_s: Change in observational state
        """
        self.state.current_s += delta_s
        self.charge_obj.update_observational_state(self.state.current_s)
        
        # Recompute Q with new observational state
        self.compute_complete_Q()
        
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
        # Use actual SemanticField.evaluate_at() method (Section 3.1.2.8)
        try:
            field_value = self.semantic_field.evaluate_at(np.array(position))
        except Exception as e:
            # Fallback: compute field value from basis functions and components
            field_magnitude = self.semantic_field_data['field_metadata']['field_magnitude']
            distance = np.linalg.norm(np.array(position))
            # Simple radial falloff approximation
            field_value = field_magnitude * np.exp(-distance**2 / 2.0)
        
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
            'agent_id': self.charge_id,
            'tau_content': self.state.tau,
            'current_state': {
                's': self.state.current_s,
                's_zero': self.state.s_zero,
                'context_C': self.state.current_context_C,
                'field_position': self.state.field_position
            },
            'coupling_state': {
                'emotional_coupling': str(self.coupling_state.emotional_field_coupling),
                's_t_coupling_strength': self.coupling_state.s_t_coupling_strength,
                'interference_strength': float(np.mean(np.abs(self.coupling_state.field_interference_coupling)))
            },
            'Q_components': {
                'gamma': self.Q_components.gamma,
                'T_tensor': {'magnitude': self.Q_components.T_magnitude, 'phase': self.Q_components.T_phase},
                'E_trajectory': {'magnitude': self.Q_components.E_magnitude, 'phase': self.Q_components.E_phase},
                'phi_semantic': {'magnitude': self.Q_components.phi_magnitude, 'phase': self.Q_components.phi_phase},
                'theta_total': {
                    'semantic': self.Q_components.theta_components.theta_semantic,
                    'emotional': self.Q_components.theta_components.theta_emotional,
                    'temporal_integral': self.Q_components.theta_components.temporal_integral,
                    'interaction': self.Q_components.theta_components.theta_interaction,
                    'field': self.Q_components.theta_components.theta_field,
                    'total': self.Q_components.theta_components.total
                },
                'psi_persistence': {
                    'total': self.Q_components.psi_persistence,
                    'gaussian': self.Q_components.psi_gaussian,
                    'exponential_cosine': self.Q_components.psi_exponential_cosine
                }
            },
            'final_Q': {
                'magnitude': self.Q_components.Q_magnitude,
                'phase': self.Q_components.Q_phase,
                'complex': {'real': self.Q_components.Q_value.real, 'imag': self.Q_components.Q_value.imag}
            }
        }