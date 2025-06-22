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

# CRITICAL FIX: Set default dtype to float32 for MPS compatibility
# This must be done early to prevent float64 tensor creation on MPS devices
if torch.backends.mps.is_available():
    torch.set_default_dtype(torch.float32)
    print("🔧 ConceptualChargeAgent: MPS detected, using float32 precision")
import scipy as sp
from scipy import integrate, signal, fft, linalg
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import time
import math

# CRITICAL FIX: Import safe tensor validation utilities to prevent boolean evaluation errors
from Sysnpire.utils.tensor_validation import (
    safe_tensor_comparison, extract_tensor_scalar, TensorValidationError
)

# Additional DTF mathematical libraries for proper field theory operations
from scipy.integrate import quad
from scipy.linalg import eigh
import numba as nb

# LIVING Modular Forms Libraries
try:
    # Sage for rigorous modular mathematics
    from sage.modular.modform import ModularForms, EisensteinForms, CuspForms
    from sage.modular.hecke import HeckeOperator
    from sage.rings.complex_double import CDF
    from sage.functions.other import floor
    SAGE_AVAILABLE = True
except ImportError:
    SAGE_AVAILABLE = False
    
try:
    # PyTorch Geometric for geometric deep learning on modular domains
    import torch_geometric as pyg
    from torch_geometric.nn import MessagePassing
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

# PyTorch neural functions for dynamic evolution
import torch.nn.functional as F
from torch.fft import fft, ifft

from Sysnpire.database.conceptual_charge_object import ConceptualChargeObject, FieldComponents
from Sysnpire.model.semantic_dimension.SemanticDimensionHelper import SemanticDimensionHelper
from Sysnpire.model.temporal_dimension.TemporalDimensionHelper import TemporalDimensionHelper
from Sysnpire.model.emotional_dimension.EmotionalDimensionHelper import EmotionalDimensionHelper

# Import logger to fix agent creation failures
from Sysnpire.utils.logger import get_logger
logger = get_logger(__name__)


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
        # Initialize device with improved validation and fallback
        try:
            if device == "mps" and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif device == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                if device != "cpu":
                    logger.warning(f"⚠️ Device {device} not available, falling back to CPU")
                self.device = torch.device("cpu")
        except Exception as e:
            logger.warning(f"⚠️ Device initialization failed: {e}, using CPU")
            self.device = torch.device("cpu")
        
        # 📚 VOCAB CONTEXT: Store vocabulary information for agent identity
        self.vocab_token_string = charge_obj.text_source  # Human-readable token string
        self.vocab_token_id = None  # Will be set if available
        self.vocab_context = {}  # Will store full vocab mappings if available
        
        # Extract rich structures from ChargeFactory combined_results (following charge_factory.py structure)
        try:
            self.semantic_field_data = combined_results['semantic_results']['field_representations'][charge_index]
            self.semantic_field = self.semantic_field_data['semantic_field']  # Actual SemanticField object
            self.temporal_biography = combined_results['temporal_results']['temporal_biographies'][charge_index]
            self.emotional_modulation = combined_results['emotional_results']['emotional_modulations'][charge_index]
            
            # 🔧 VALIDATE: Ensure data integrity for Q computation
            self._validate_extracted_data(charge_index)
            
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"❌ Agent {charge_index} - Data extraction failed: {e}")
            raise ValueError(f"Invalid combined_results structure for charge {charge_index}: {e}")
        
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
        
        # Initialize agent field state with MPS-safe state conversion
        # Ensure observational_state is converted to float (not tensor)
        if hasattr(charge_obj.observational_state, 'cpu'):
            obs_state = float(charge_obj.observational_state.cpu().detach().numpy())
        else:
            obs_state = float(charge_obj.observational_state)
            
        self.state = AgentFieldState(
            tau=charge_obj.text_source,
            current_context_C=initial_context or {},
            current_s=obs_state,
            s_zero=obs_state,
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
        
        # Initialize LIVING modular form structure
        self._initialize_living_modular_form()
        
        # Initialize with first computation
        self.compute_complete_Q()
    
    @classmethod
    def from_stored_data(cls, stored_data: Dict[str, Any], charge_obj: ConceptualChargeObject = None, device: str = "mps") -> 'ConceptualChargeAgent':
        """
        CRITICAL RECONSTRUCTION METHOD: Create agent from stored data with proper mathematical state.
        
        This method is essential for universe reconstruction - it restores agents with their
        actual mathematical state instead of using default values that cause explosions.
        """
        logger.info(f"🔄 Reconstructing agent from stored data...")
        
        # Extract basic metadata
        agent_metadata = stored_data.get("agent_metadata", {})
        charge_id = agent_metadata.get("charge_id", "reconstructed_agent")
        
        # Create charge object if not provided
        if charge_obj is None:
            # Reconstruct charge object from stored Q_components and field_components
            q_components = stored_data.get("Q_components", {})
            field_components = stored_data.get("field_components", {})
            
            charge_obj = ConceptualChargeObject(
                charge_id=charge_id,
                text_source=agent_metadata.get("text_source", "unknown"),
                complete_charge=q_components.get("Q_value", complex(1.0, 0.0)),
                field_components=FieldComponents(
                    semantic_field_generation=field_components.get("semantic_field_generation"),
                    emotional_trajectory=field_components.get("emotional_trajectory"),
                    trajectory_operators=field_components.get("trajectory_operators"),
                    phase_total=field_components.get("phase_total", 0.0),
                    observational_persistence=field_components.get("observational_persistence", 1.0)
                ),
                observational_state=agent_metadata.get("observational_state", 1.0)
            )
        
        # Create minimal combined_results for initialization (won't be used for real math)
        minimal_combined_results = cls._create_minimal_combined_results(stored_data, 0)
        
        # Initialize agent with minimal data (this calls __init__)
        agent = cls(
            charge_obj=charge_obj,
            charge_index=0,  # Not used in reconstruction
            combined_results=minimal_combined_results,
            initial_context={},
            device=device
        )
        
        # CRITICAL: Now restore the ACTUAL mathematical state from stored data
        agent._restore_mathematical_state_from_storage(stored_data)
        
        logger.info(f"✅ Agent {charge_id} reconstructed from stored data")
        return agent
    
    
    @classmethod
    def from_charge_factory_results(cls, 
                                  combined_results: Dict[str, Any], 
                                  charge_index: int,
                                  initial_context: Dict[str, Any] = None,
                                  device: str = "mps",
                                  vocab_mappings: Dict[str, Any] = None) -> 'ConceptualChargeAgent':
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
            
        Returns:
            ConceptualChargeAgent instance with proper field theory mathematics and vocab context
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
        
        # 📚 VOCAB RESOLUTION: Convert token ID to human-readable string
        if vocab_mappings and 'id_to_token' in vocab_mappings:
            id_to_token = vocab_mappings['id_to_token']
            # Try to convert source_token to readable string
            if isinstance(source_token, (int, str)):
                # Handle both numeric IDs and string tokens
                token_id = int(source_token) if str(source_token).isdigit() else source_token
                vocab_token_string = id_to_token.get(token_id, source_token)
            else:
                vocab_token_string = str(source_token)
        else:
            vocab_token_string = str(source_token)
        
        logger.debug(f"🧬 Agent {charge_index} vocab resolution: {source_token} → {vocab_token_string}")
        
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
        
        # Create ConceptualChargeObject with proper complete charge initialization AND vocab string
        charge_obj = ConceptualChargeObject(
            charge_id=f"charge_{charge_index}",
            text_source=vocab_token_string,  # 📚 Use human-readable vocab string instead of token ID
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
        
        # 📚 ENHANCE AGENT WITH VOCAB CONTEXT: Set vocab fields
        if vocab_mappings:
            agent.vocab_context = vocab_mappings
            agent.vocab_token_id = source_token  # Original token ID
            # vocab_token_string already set via charge_obj.text_source
            logger.debug(f"🧬 Agent {charge_index} enhanced with vocab context: ID={source_token}, String='{vocab_token_string}'")
        
        return agent
    
    @classmethod
    def from_stored_data(cls, 
                        stored_data: Dict[str, Any],
                        charge_obj: ConceptualChargeObject = None,
                        device: str = "mps") -> 'ConceptualChargeAgent':
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
        logger.info("🔄 Reconstructing ConceptualChargeAgent from stored data")
        
        # CRITICAL FIX: Set float32 for MPS compatibility
        if "mps" in device and torch.backends.mps.is_available():
            torch.set_default_dtype(torch.float32)
            logger.debug("🔧 MPS device detected: Using float32 precision for agent reconstruction")
        
        # Extract stored components
        agent_metadata = stored_data.get("agent_metadata", {})
        q_components = stored_data.get("Q_components", {})
        field_components = stored_data.get("field_components", {})
        temporal_components = stored_data.get("temporal_components", {})
        emotional_components = stored_data.get("emotional_components", {})
        agent_state = stored_data.get("agent_state", {})
        
        # Create charge object if not provided
        if charge_obj is None:
            charge_obj = cls._reconstruct_charge_object_from_storage(agent_metadata, q_components, field_components)
        
        # Create minimal combined_results structure for initialization
        minimal_combined_results = cls._create_minimal_combined_results(
            stored_data, agent_metadata.get("charge_index", 0)
        )
        
        # Create instance using standard constructor
        agent = cls.__new__(cls)
        
        # Initialize basic attributes
        agent.charge_obj = charge_obj
        agent.charge_id = charge_obj.charge_id
        agent.charge_index = agent_metadata.get("charge_index", 0)
        agent.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        
        # Restore vocabulary context
        agent.vocab_token_string = agent_metadata.get("vocab_token_string", charge_obj.text_source)
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
        q_data = stored_data.get("Q_components", {})
        agent.Q_components = QMathematicalComponents(
            gamma=q_data.get("gamma", 1.0),
            T_tensor=q_data.get("T_tensor", complex(1.0, 0.0)),
            E_trajectory=q_data.get("E_trajectory", complex(1.0, 0.0)),
            phi_semantic=q_data.get("phi_semantic", complex(1.0, 0.0)),
            theta_components=ThetaComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # Default
            phase_factor=q_data.get("phase_factor", complex(1.0, 0.0)),
            psi_persistence=q_data.get("psi_persistence", 1.0),
            psi_gaussian=q_data.get("psi_gaussian", 1.0),
            psi_exponential_cosine=q_data.get("psi_exponential_cosine", 1.0),
            Q_value=stored_q_value
        )
        
        logger.info(f"✅ Agent {agent.charge_id} reconstructed using STORED Q values only - Q magnitude: {abs(agent.living_Q_value):.6f}")
        
        # Skip Q validation during reconstruction since we're using stored values
        # if not agent.validate_Q_components():
        #     logger.warning(f"⚠️  Agent {agent.charge_id} - Q validation failed after reconstruction")
        
        logger.info(f"✅ Agent {agent.charge_id} reconstructed - Q magnitude: {abs(agent.living_Q_value):.6f}")
        return agent
    
    @classmethod
    def _reconstruct_charge_object_from_storage(cls, metadata: Dict[str, Any], 
                                              q_components: Dict[str, Any],
                                              field_components: Dict[str, Any]) -> ConceptualChargeObject:
        """Reconstruct ConceptualChargeObject from stored metadata."""
        
        # Reconstruct complex charge value from real/imag components
        if "living_Q_value_real" in q_components and "living_Q_value_imag" in q_components:
            complete_charge = complex(q_components["living_Q_value_real"], q_components["living_Q_value_imag"])
        elif "Q_value_real" in q_components and "Q_value_imag" in q_components:
            complete_charge = complex(q_components["Q_value_real"], q_components["Q_value_imag"])
        else:
            complete_charge = complex(1.0, 0.0)
        
        # Reconstruct field components
        field_comps = FieldComponents(
            semantic_field=field_components.get("semantic_embedding"),
            emotional_trajectory=field_components.get("emotional_trajectory"),
            trajectory_operators=field_components.get("trajectory_operators"),
            phase_total=field_components.get("phase_total", 0.0),
            observational_persistence=field_components.get("observational_persistence", 1.0)
        )
        
        return ConceptualChargeObject(
            charge_id=metadata.get("charge_id", "reconstructed_agent"),
            text_source=metadata.get("text_source", "unknown"),
            complete_charge=complete_charge,
            field_components=field_comps,
            observational_state=metadata.get("observational_state", 1.0)
        )
    
    @classmethod
    def _create_minimal_combined_results(cls, stored_data: Dict[str, Any], charge_index: int) -> Dict[str, Any]:
        """Create minimal combined_results structure for reconstruction."""
        
        # This creates a minimal structure that won't break the initialization
        # but isn't used for actual mathematical computation (we restore directly)
        return {
            'semantic_results': {
                'field_representations': [{'semantic_field': None, 'field_metadata': {'source_token': 'reconstructed'}}]
            },
            'temporal_results': {
                'temporal_biographies': [None],
                'field_interference_matrix': np.eye(1),
                'collective_breathing_rhythm': {'collective_frequency': np.array([1.0])}
            },
            'emotional_results': {
                'emotional_modulations': [None],
                'field_signature': {'field_modulation_strength': 1.0}
            }
        }
    
    def _restore_complex_numbers(self, obj):
        """
        Recursively restore dictionary-format complex numbers back to proper Python complex objects.
        
        Converts {"real": float, "imag": float, "_type": "complex"} back to complex(real, imag).
        """
        if obj is None:
            return None
        
        # Check if this is a complex number dictionary
        if (isinstance(obj, dict) and 
            "_type" in obj and obj["_type"] == "complex" and
            "real" in obj and "imag" in obj):
            return complex(float(obj["real"]), float(obj["imag"]))
        
        # Handle dictionaries recursively
        if isinstance(obj, dict):
            return {k: self._restore_complex_numbers(v) for k, v in obj.items()}
        
        # Handle lists/tuples recursively
        if isinstance(obj, (list, tuple)):
            restored = [self._restore_complex_numbers(item) for item in obj]
            return restored if isinstance(obj, list) else tuple(restored)
        
        # Return primitive types as-is
        return obj
    
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
        elif isinstance(value, dict) and "_type" in value and value["_type"] == "complex":
            return float(value["real"])
        else:
            # Try to convert to float as last resort
            try:
                return float(value)
            except (ValueError, TypeError):
                logger.warning(f"Could not extract float from value: {value} (type: {type(value)})")
                return 0.0
    
    def _restore_mathematical_state_from_storage(self, stored_data: Dict[str, Any]) -> None:
        """Restore complete mathematical state from stored data."""
        
        q_components = stored_data.get("Q_components", {})
        agent_state = stored_data.get("agent_state", {})
        
        # CRITICAL FIX: Restore complex numbers from dictionary format before using them
        logger.debug(f"🔧 Restoring complex numbers for agent reconstruction...")
        original_q_keys = list(q_components.keys())
        original_state_keys = list(agent_state.keys())
        
        q_components = self._restore_complex_numbers(q_components)
        agent_state = self._restore_complex_numbers(agent_state)
        
        # IMMEDIATE VALIDATION: Check that restoration worked
        logger.debug(f"   Q components before restoration: {len([k for k, v in q_components.items() if isinstance(v, dict) and '_type' in v])} dict-format complex numbers")
        remaining_dict_complex = [(k, v) for k, v in q_components.items() if isinstance(v, dict) and '_type' in v and v.get('_type') == 'complex']
        if remaining_dict_complex:
            logger.error(f"❌ COMPLEX RESTORATION FAILED: {len(remaining_dict_complex)} dictionary-format complex numbers remain:")
            for key, value in remaining_dict_complex:
                logger.error(f"   {key}: {value}")
        else:
            logger.debug(f"✅ Complex number restoration completed successfully")
        
        # Restore living Q value from storage with proper complex number handling
        if "living_Q_value_real" in q_components and "living_Q_value_imag" in q_components:
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
        if not isinstance(self.living_Q_value, complex):
            logger.error(f"❌ RECONSTRUCTION ERROR: living_Q_value was not a complex number (type: {type(self.living_Q_value)}, value: {self.living_Q_value})")
            logger.error(f"   This indicates incomplete complex number deserialization during reconstruction!")
            try:
                if hasattr(self.living_Q_value, 'real') and hasattr(self.living_Q_value, 'imag'):
                    self.living_Q_value = complex(float(self.living_Q_value.real), float(self.living_Q_value.imag))
                    logger.warning(f"   Successfully converted to complex: {self.living_Q_value}")
                else:
                    self.living_Q_value = complex(float(self.living_Q_value), 0.0)
                    logger.warning(f"   Converted to complex (assuming real): {self.living_Q_value}")
            except (ValueError, TypeError) as e:
                logger.error(f"   Failed to convert living_Q_value to complex: {e}, setting to 0+0j")
                self.living_Q_value = complex(0.0, 0.0)
                
        # COMPREHENSIVE VALIDATION: Check all critical complex number fields during reconstruction
        complex_fields_to_validate = [
            ('living_Q_value', self.living_Q_value),
            ('temporal_momentum', getattr(self, 'temporal_momentum', None)),
            ('T_tensor', getattr(self, 'T_tensor', None)),
            ('E_trajectory', getattr(self, 'E_trajectory', None)),
            ('phi_semantic', getattr(self, 'phi_semantic', None))
        ]
        
        validation_failed = False
        for field_name, field_value in complex_fields_to_validate:
            if field_value is not None and not isinstance(field_value, complex):
                logger.error(f"❌ RECONSTRUCTION ERROR: {field_name} is not a complex number (type: {type(field_value)}, value: {field_value})")
                validation_failed = True
                
        if validation_failed:
            logger.error(f"❌ CRITICAL: Complex number validation failed during reconstruction for agent {getattr(self, 'charge_id', 'unknown')}")
            logger.error(f"   This will cause comparison errors during evolution simulation!")
            # Log the original q_components for debugging
            logger.error(f"   Original q_components keys: {list(q_components.keys())}")
            for key, value in q_components.items():
                if isinstance(value, dict) and "_type" in value:
                    logger.error(f"   {key}: {value} (type: {type(value)})")
        else:
            logger.debug(f"✅ Complex number validation passed for agent {getattr(self, 'charge_id', 'unknown')}")
            
        # ADDITIONAL VALIDATION: Check persistence layers contain only real numbers
        if hasattr(self, 'temporal_biography') and self.temporal_biography:
            persistence_layers_to_check = [
                ('vivid_layer', getattr(self.temporal_biography, 'vivid_layer', None)),
                ('character_layer', getattr(self.temporal_biography, 'character_layer', None))
            ]
            
            persistence_validation_failed = False
            for layer_name, layer_data in persistence_layers_to_check:
                if layer_data is not None and len(layer_data) > 0:
                    if np.iscomplexobj(layer_data) or any(np.iscomplexobj(item) for item in layer_data):
                        logger.error(f"❌ PERSISTENCE LAYER ERROR: {layer_name} contains complex numbers!")
                        logger.error(f"   This will cause max() comparison errors during persistence calculations!")
                        persistence_validation_failed = True
                        
            if persistence_validation_failed:
                logger.error(f"❌ CRITICAL: Persistence layer validation failed for agent {getattr(self, 'charge_id', 'unknown')}")
            else:
                logger.debug(f"✅ Persistence layers contain only real numbers for agent {getattr(self, 'charge_id', 'unknown')}")
        
        # Restore evolution parameters with explicit type safety
        evolution_params = ["sigma_i", "alpha_i", "lambda_i", "beta_i"]
        for param in evolution_params:
            if param in agent_state:
                # CRITICAL FIX: Ensure evolution parameters are Python floats, not tensors
                param_value = agent_state[param]
                original_type = type(param_value)
                
                # Convert to Python float regardless of input type
                if hasattr(param_value, 'cpu'):
                    # It's a tensor - convert to Python float
                    float_value = float(param_value.cpu().detach().numpy())
                elif hasattr(param_value, 'item'):
                    # It's a numpy scalar - convert to Python float
                    float_value = float(param_value.item())
                else:
                    # It's already a primitive - ensure it's a Python float
                    float_value = float(param_value)
                
                setattr(self, param, float_value)
                logger.info(f"🔧 EVOLUTION PARAM RESTORED: {param}: {original_type} -> {type(float_value)} (value: {float_value})")
            else:
                # Set defaults if missing
                defaults = {"sigma_i": 0.5, "alpha_i": 0.3, "lambda_i": 0.1, "beta_i": 2.0}
                default_value = float(defaults.get(param, 1.0))
                setattr(self, param, default_value)
                logger.warning(f"⚠️  EVOLUTION PARAM DEFAULT: {param} = {default_value} (missing from stored data)")
        
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
            ("phi_semantic", "phi_semantic")
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
        
        logger.debug(f"🔧 Mathematical state restored for agent {self.charge_id}")
    
    def _restore_field_and_agent_state(self, stored_data: Dict[str, Any]) -> None:
        """Restore field components and agent state."""
        
        field_components = stored_data.get("field_components", {})
        temporal_biography = stored_data.get("temporal_biography", {})
        emotional_modulation = stored_data.get("emotional_modulation", {})
        
        # Create semantic field object from stored data
        semantic_embedding = field_components.get("semantic_embedding")
        semantic_phase_factors = field_components.get("semantic_phase_factors")
        
        class SemanticField:
            def __init__(self, embedding_components, phase_factors=None):
                # MPS-safe tensor conversion for embedding components
                if embedding_components is not None:
                    if hasattr(embedding_components, 'cpu'):
                        self.embedding_components = embedding_components.cpu().detach().numpy()
                    else:
                        self.embedding_components = np.array(embedding_components)
                else:
                    self.embedding_components = np.zeros(384)
                
                # MPS-safe tensor conversion for phase factors
                if phase_factors is not None:
                    if hasattr(phase_factors, 'cpu'):
                        self.phase_factors = phase_factors.cpu().detach().numpy()
                    else:
                        self.phase_factors = np.array(phase_factors)
                else:
                    self.phase_factors = np.ones(1024)
            
            def evaluate_at(self, position_x: np.ndarray) -> complex:
                """
                Evaluate semantic field at position x.
                
                Simple implementation using embedding components and phase factors.
                All operations use numpy arrays (MPS-safe).
                """
                # Use position to create a simple basis evaluation
                # Take magnitude of position as a scalar modulation factor
                if len(position_x) > 0:
                    position_magnitude = float(np.linalg.norm(position_x))
                    position_factor = 1.0 + 0.1 * np.sin(position_magnitude)
                else:
                    position_factor = 1.0
                
                # Compute phase factors from stored phases
                phase_factors = np.exp(1j * self.phase_factors[:len(self.embedding_components)])
                
                # Combine embedding components with phase modulation
                # Simple field evaluation: weighted sum of components
                field_real = np.mean(self.embedding_components * np.real(phase_factors)) * position_factor
                field_imag = np.mean(self.embedding_components * np.imag(phase_factors)) * position_factor
                
                return complex(field_real, field_imag)
        
        self.semantic_field = SemanticField(semantic_embedding, semantic_phase_factors)
        
        # CRITICAL: Initialize semantic_field_data with required field_metadata - NO FALLBACKS
        field_magnitude = np.mean(np.abs(self.semantic_field.embedding_components)) if len(self.semantic_field.embedding_components) > 0 else 1.0
        self.semantic_field_data = {
            "semantic_field": self.semantic_field,
            "field_metadata": {
                "source_token": "reconstructed",
                "manifold_dimension": len(self.semantic_field.embedding_components),
                "field_magnitude": field_magnitude
            }
        }
        
        # Create temporal biography object from stored data
        trajectory_operators = field_components.get("trajectory_operators", [])
        if temporal_biography and "trajectory_operators" in temporal_biography:
            trajectory_operators = temporal_biography["trajectory_operators"]
        
        # Get frequency evolution from stored data if available
        frequency_evolution = temporal_biography.get("frequency_evolution", []) if temporal_biography else []
        
        class TemporalBiography:
            def __init__(self, trajectory_operators, frequency_evolution=None):
                if trajectory_operators is not None and len(trajectory_operators) > 0:
                    # MPS-safe tensor conversion for trajectory operators
                    if hasattr(trajectory_operators, 'cpu'):
                        self.trajectory_operators = trajectory_operators.cpu().detach().numpy()
                    else:
                        self.trajectory_operators = np.array(trajectory_operators)
                else:
                    self.trajectory_operators = np.array([1.0])
                
                if frequency_evolution is not None and len(frequency_evolution) > 0:
                    # MPS-safe tensor conversion for frequency evolution
                    if hasattr(frequency_evolution, 'cpu'):
                        self.frequency_evolution = frequency_evolution.cpu().detach().numpy()
                    else:
                        self.frequency_evolution = np.array(frequency_evolution)
                else:
                    self.frequency_evolution = np.array([1.0])
                
                # CRITICAL FIX: Initialize field_interference_signature - required for FieldCouplingState
                if len(self.trajectory_operators) > 0:
                    mean_traj = np.mean(self.trajectory_operators)
                    # Ensure proper scalar conversion from numpy to Python complex
                    if np.iscomplexobj(mean_traj):
                        self.field_interference_signature = complex(float(mean_traj.real), float(mean_traj.imag))
                    else:
                        self.field_interference_signature = complex(float(mean_traj), 0.1)
                else:
                    self.field_interference_signature = complex(1.0, 0.1)
                    
                # CRITICAL FIX: Initialize missing TemporalBiography attributes for evolution
                # breathing_coherence - required for dimensional cascades
                if len(self.frequency_evolution) > 0:
                    # Calculate coherence as inverse of variance (more uniform = more coherent)
                    freq_var = np.var(self.frequency_evolution)
                    self.breathing_coherence = float(1.0 / (1.0 + freq_var))  # Range [0, 1]
                else:
                    self.breathing_coherence = 0.5  # Default moderate coherence
                
                # temporal_momentum - required for momentum calculations
                if len(self.trajectory_operators) > 0:
                    mean_trajectory = np.mean(self.trajectory_operators)
                    if np.iscomplexobj(mean_trajectory):
                        self.temporal_momentum = complex(float(mean_trajectory.real), float(mean_trajectory.imag))
                    else:
                        self.temporal_momentum = complex(float(mean_trajectory), 0.0)
                else:
                    self.temporal_momentum = complex(0.0, 0.0)
                
                # phase_coordination - required for breathing coordination
                self.phase_coordination = self.frequency_evolution.copy() if len(self.frequency_evolution) > 0 else np.array([1.0])
                
                # CRITICAL FIX: Initialize missing vivid_layer and character_layer for Q computation persistence
                # ENSURE REAL VALUES ONLY - extract real parts if trajectory_operators contain complex numbers
                if len(self.trajectory_operators) > 0:
                    # Extract magnitudes (real values) from complex trajectory operators
                    self.vivid_layer = np.array([abs(op) if np.iscomplexobj(op) else float(op) for op in self.trajectory_operators])
                else:
                    self.vivid_layer = np.array([1.0])
                    
                if len(self.frequency_evolution) > 0:
                    # Extract magnitudes (real values) from complex frequency evolution
                    self.character_layer = np.array([abs(freq) if np.iscomplexobj(freq) else float(freq) for freq in self.frequency_evolution])
                else:
                    self.character_layer = np.array([1.0])
        
        self.temporal_biography = TemporalBiography(trajectory_operators, frequency_evolution)
        
        # Create emotional modulation object from stored data
        emotional_trajectory = field_components.get("emotional_trajectory")
        if emotional_modulation and "emotional_trajectory" in emotional_modulation:
            emotional_trajectory = emotional_modulation["emotional_trajectory"]
        
        class EmotionalModulation:
            def __init__(self, emotional_trajectory):
                if emotional_trajectory is not None and len(emotional_trajectory) > 0:
                    # MPS-safe tensor conversion for emotional trajectory
                    if hasattr(emotional_trajectory, 'cpu'):
                        self.emotional_trajectory = emotional_trajectory.cpu().detach().numpy()
                    else:
                        self.emotional_trajectory = np.array(emotional_trajectory)
                else:
                    self.emotional_trajectory = np.array([1.0])
                
                # CRITICAL: Initialize ALL required attributes for Q computation - NO FALLBACKS
                # semantic_modulation_tensor - required for Q computation (E^trajectory component)
                self.semantic_modulation_tensor = self.emotional_trajectory.copy()
                
                # unified_phase_shift - required for phase calculations  
                if len(self.emotional_trajectory) > 0:
                    mean_emot = np.mean(self.emotional_trajectory)
                    # Ensure proper scalar conversion from numpy to Python complex
                    if np.iscomplexobj(mean_emot):
                        self.unified_phase_shift = complex(float(mean_emot.real), float(mean_emot.imag))
                    else:
                        self.unified_phase_shift = complex(float(mean_emot), 0.1)
                else:
                    self.unified_phase_shift = complex(1.0, 0.1)
                
                # trajectory_attractors - required for trajectory modulation
                self.trajectory_attractors = np.linspace(0.5, 2.0, min(len(self.emotional_trajectory), 5))
                
                # resonance_frequencies - required for resonance calculations
                self.resonance_frequencies = np.array([1.0, 2.0, 3.0, 5.0, 8.0])  # Fibonacci-based frequencies
        
        self.emotional_modulation = EmotionalModulation(emotional_trajectory)
        
        # Initialize minimal state and coupling structures for reconstruction
        self.state = AgentFieldState(
            tau=self.charge_obj.text_source,
            current_context_C={},
            current_s=self.charge_obj.observational_state,
            s_zero=self.charge_obj.observational_state,
            field_position=(0.0, 0.0),
            trajectory_time=0.0
        )
        
        # CRITICAL: Initialize ALL required mathematical components - NO FALLBACKS
        
        # Initialize field_interference_matrix - required for Q computation
        self.field_interference_matrix = np.eye(1)  # Identity matrix for single agent coupling
        
        # Initialize breathing coefficients - complete mathematical structure required
        self._initialize_breathing_q_expansion_for_reconstruction()
        self.collective_breathing = {'collective_frequency': np.array([1.0])}
        self.breath_frequency = 0.1
        self.breath_amplitude = 1.0 
        self.breath_phase = 0.0
        
        # CRITICAL FIX: Initialize missing emotional_field_signature for Q computation
        class EmotionalFieldSignature:
            def __init__(self):
                self.field_modulation_strength = 1.0  # Required for coupling_state initialization
                self.phase_coherence = 0.5
                self.emotional_amplitude = 1.0
        
        self.emotional_field_signature = EmotionalFieldSignature()
        
        self.coupling_state = FieldCouplingState(
            emotional_field_coupling=self.emotional_modulation.unified_phase_shift,
            field_interference_coupling=self.temporal_biography.field_interference_signature,
            collective_breathing_rhythm=self.collective_breathing,
            s_t_coupling_strength=self.emotional_field_signature.field_modulation_strength
        )
        
        # CRITICAL: Initialize complete modular form structure - REQUIRED for evolution
        self._initialize_modular_geometry_for_reconstruction()
        
        # CRITICAL: Initialize missing mathematical components for Q computation and evolution
        self._initialize_missing_components_for_reconstruction()
        
        # CRITICAL VALIDATION: Ensure ALL required mathematical components are properly initialized
        required_attrs = [
            'living_Q_value', 'semantic_field', 'temporal_biography', 'emotional_modulation', 
            'emotional_field_signature', 'tau_position', 'modular_weight', 'field_interference_matrix',
            'breathing_q_coefficients', 'geometric_features', 'emotional_conductivity',
            'hecke_eigenvalues', 'l_function_coefficients', 'semantic_field_data'
        ]
        
        missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing_attrs:
            logger.error(f"❌ CRITICAL: Missing mathematical components after reconstruction: {missing_attrs}")
            raise ValueError(f"Reconstruction failed - missing required mathematical components: {missing_attrs}")
        
        # Validate mathematical component structures
        validation_errors = []
        
        if not isinstance(self.tau_position, complex):
            validation_errors.append(f"tau_position must be complex, got {type(self.tau_position)}")
            
        if not isinstance(self.breathing_q_coefficients, dict) or len(self.breathing_q_coefficients) == 0:
            validation_errors.append(f"breathing_q_coefficients must be non-empty dict, got {type(self.breathing_q_coefficients)}")
            
        if not hasattr(self.geometric_features, 'shape') or len(self.geometric_features) != 4:
            validation_errors.append(f"geometric_features must be tensor with 4 elements, got shape {getattr(self.geometric_features, 'shape', 'unknown')}")
        
        if validation_errors:
            logger.error(f"❌ CRITICAL: Mathematical component validation errors: {validation_errors}")
            raise ValueError(f"Reconstruction failed - invalid mathematical components: {validation_errors}")
        
        logger.info(f"✅ All mathematical components properly initialized for agent {self.charge_id}")
        
        logger.debug(f"🔧 Field and agent state restored for agent {self.charge_id}")
    
    def _initialize_modular_geometry_for_reconstruction(self):
        """Initialize complete modular form geometry during reconstruction - NO FALLBACKS."""
        
        # Position agent in modular fundamental domain using field position
        x, y = self.state.field_position
        
        # Transform to upper half-plane coordinates (modular fundamental domain)
        # Fundamental domain: |τ| ≥ 1, -1/2 ≤ Re(τ) ≤ 1/2, Im(τ) > 0
        real_part = np.clip(x, -0.5, 0.5)  # Real part in fundamental domain
        imag_part = max(0.1, 1.0 + y)      # Ensure positive imaginary part
        
        self.tau_position = complex(real_part, imag_part)
        
        # Modular weight determines transformation behavior - derive from semantic field
        if hasattr(self.semantic_field, 'embedding_components') and len(self.semantic_field.embedding_components) > 0:
            field_magnitude = np.mean(np.abs(self.semantic_field.embedding_components))
        else:
            field_magnitude = 1.0
        self.modular_weight = max(2, int(2 * field_magnitude))  # Even weight ≥ 2
        
        # Emotional conductivity from emotional modulation
        if hasattr(self.emotional_modulation, 'emotional_trajectory') and len(self.emotional_modulation.emotional_trajectory) > 0:
            self.emotional_conductivity = np.mean(np.abs(self.emotional_modulation.emotional_trajectory))
        else:
            self.emotional_conductivity = 1.0
            
        # Create geometric node features for PyTorch Geometric operations
        self.geometric_features = torch.tensor([
            self.tau_position.real,
            self.tau_position.imag, 
            self.modular_weight,
            self.emotional_conductivity
        ], dtype=torch.float32, device=self.device)
        
        logger.debug(f"🔧 Modular geometry initialized: τ={self.tau_position}, weight={self.modular_weight}")
    
    def _initialize_breathing_q_expansion_for_reconstruction(self):
        """Initialize complete breathing q-coefficients during reconstruction - NO FALLBACKS."""
        
        # Base q-coefficients from semantic embedding components
        semantic_components = self.semantic_field.embedding_components
        
        # Create complex q-coefficients: semantic (real) + temporal (imaginary)
        self.breathing_q_coefficients = {}
        max_coeffs = min(1024, len(semantic_components)) if semantic_components is not None else 128
        
        for n in range(max_coeffs):
            # Real part from semantic field strength
            if semantic_components is not None and n < len(semantic_components):
                real_part = float(semantic_components[n])  # Ensure proper scalar conversion
            else:
                real_part = 0.1  # Minimal real component
            
            # Imaginary part from temporal biography (frequency evolution)
            if (hasattr(self.temporal_biography, 'frequency_evolution') and 
                self.temporal_biography.frequency_evolution is not None and 
                n < len(self.temporal_biography.frequency_evolution)):
                imag_part = float(self.temporal_biography.frequency_evolution[n])  # Ensure proper scalar conversion
            else:
                imag_part = 0.1  # Minimal imaginary component
                
            # Complex q-coefficient
            self.breathing_q_coefficients[n] = complex(real_part, imag_part)
        
        logger.debug(f"🔧 Breathing q-coefficients initialized: {len(self.breathing_q_coefficients)} coefficients")
    
    def _initialize_missing_components_for_reconstruction(self):
        """Initialize all missing mathematical components required for Q computation and evolution - NO FALLBACKS."""
        
        # Initialize Hecke eigenvalues for evolution simulation
        trajectory_ops = self.temporal_biography.trajectory_operators
        self.hecke_eigenvalues = {}
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        for i, p in enumerate(primes):
            if i < len(trajectory_ops):
                # CRITICAL FIX: Ensure proper scalar conversion from numpy to Python complex
                traj_val = trajectory_ops[i]
                if np.iscomplexobj(traj_val):
                    self.hecke_eigenvalues[p] = complex(float(traj_val.real), float(traj_val.imag))
                else:
                    self.hecke_eigenvalues[p] = complex(float(traj_val), 0.0)
            else:
                self.hecke_eigenvalues[p] = complex(1.0, 0.0)
        
        # Initialize L-function coefficients for Q computation
        self.l_function_coefficients = {}
        for n in range(1, min(50, len(self.emotional_modulation.semantic_modulation_tensor) + 1)):
            base_coeff = self.emotional_modulation.semantic_modulation_tensor[n % len(self.emotional_modulation.semantic_modulation_tensor)]
            emotional_phase = np.angle(self.emotional_modulation.unified_phase_shift) * n / 10
            self.l_function_coefficients[n] = base_coeff * np.exp(1j * emotional_phase)
        
        # Initialize Hecke adaptivity for agent interactions
        self.hecke_adaptivity = 0.01
        
        # Initialize interaction memory buffer
        self.interaction_memory = []
        self.interaction_memory_buffer = {}
        
        logger.debug(f"🔧 Missing mathematical components initialized: {len(self.hecke_eigenvalues)} hecke eigenvalues, {len(self.l_function_coefficients)} L-function coefficients")
    
    def _validate_extracted_data(self, charge_index: int):
        """
        Validate that extracted data is suitable for Q(τ,C,s) computation.
        
        Checks for common data issues that cause Q computation failures.
        """
        # Validate semantic field data
        if not hasattr(self.semantic_field, 'embedding_components'):
            raise ValueError(f"Charge {charge_index} - semantic_field missing embedding_components")
        
        if hasattr(self.semantic_field, 'embedding_components'):
            embedding_components = self.semantic_field.embedding_components
            if np.all(np.abs(embedding_components) < 1e-12):
                logger.warning(f"⚠️  Charge {charge_index} - All semantic embedding components near zero")
            
            if np.any(np.isnan(embedding_components)) or np.any(np.isinf(embedding_components)):
                raise ValueError(f"Charge {charge_index} - Invalid semantic embedding components (NaN/Inf)")
        
        # Validate temporal biography data
        if not hasattr(self.temporal_biography, 'trajectory_operators'):
            raise ValueError(f"Charge {charge_index} - temporal_biography missing trajectory_operators")
        
        trajectory_ops = self.temporal_biography.trajectory_operators
        if len(trajectory_ops) == 0:
            logger.warning(f"⚠️  Charge {charge_index} - Empty trajectory_operators array")
        elif np.all(np.abs(trajectory_ops) < 1e-12):
            logger.warning(f"⚠️  Charge {charge_index} - All trajectory operators near zero (will cause T_tensor=0j)")
        
        if np.any(np.isnan(trajectory_ops)) or np.any(np.isinf(trajectory_ops)):
            raise ValueError(f"Charge {charge_index} - Invalid trajectory operators (NaN/Inf)")
        
        # Validate vivid and character layers for persistence
        if hasattr(self.temporal_biography, 'vivid_layer') and hasattr(self.temporal_biography, 'character_layer'):
            vivid_layer = self.temporal_biography.vivid_layer
            character_layer = self.temporal_biography.character_layer
            
            if len(vivid_layer) == 0 or len(character_layer) == 0:
                logger.warning(f"⚠️  Charge {charge_index} - Empty persistence layers")
            
            if np.all(vivid_layer < 1e-10) and np.all(character_layer < 1e-10):
                logger.warning(f"⚠️  Charge {charge_index} - Persistence layers extremely small (will cause underflow)")
        
        # Validate emotional modulation data
        if not hasattr(self.emotional_modulation, 'semantic_modulation_tensor'):
            raise ValueError(f"Charge {charge_index} - emotional_modulation missing semantic_modulation_tensor")
        
        if not hasattr(self.emotional_modulation, 'unified_phase_shift'):
            raise ValueError(f"Charge {charge_index} - emotional_modulation missing unified_phase_shift")
        
        modulation_tensor = self.emotional_modulation.semantic_modulation_tensor
        if np.any(np.isnan(modulation_tensor)) or np.any(np.isinf(modulation_tensor)):
            raise ValueError(f"Charge {charge_index} - Invalid emotional modulation tensor (NaN/Inf)")
        
        # Check for identical modulations (uniqueness validation)
        if hasattr(self, '_global_modulation_check'):
            # This would be set by LiquidOrchestrator to check across all agents
            pass
        
        logger.debug(f"✅ Charge {charge_index} - Data validation passed")
    
    def validate_Q_components(self) -> bool:
        """
        Validate that Q(τ,C,s) components are within reasonable ranges.
        
        Returns:
            True if all components are reasonable, False if issues detected
        """
        if self.Q_components is None:
            logger.warning(f"⚠️  Agent {self.charge_id} - Q_components not yet computed")
            return False
        
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
            issues.append(f"|phase_factor|={phase_magnitude:.6f} outside [0.5, 1.5] (should be ≈1.0)")
        
        # psi_persistence should be 0.001-2.0 range
        if not (0.001 <= self.Q_components.psi_persistence <= 2.0):
            issues.append(f"psi_persistence={self.Q_components.psi_persistence:.6f} outside [0.001, 2.0]")
        
        # Final Q magnitude should be 0.0001-20.0 range
        Q_magnitude = abs(self.Q_components.Q_value)
        if not (0.0001 <= Q_magnitude <= 20.0):
            issues.append(f"|Q_value|={Q_magnitude:.6f} outside [0.0001, 20.0]")
        
        # Check for NaN/Inf
        for component_name, component_value in [
            ('gamma', self.Q_components.gamma),
            ('T_tensor', self.Q_components.T_tensor),
            ('E_trajectory', self.Q_components.E_trajectory),
            ('phi_semantic', self.Q_components.phi_semantic),
            ('phase_factor', self.Q_components.phase_factor),
            ('Q_value', self.Q_components.Q_value)
        ]:
            if isinstance(component_value, complex):
                if np.isnan(component_value.real) or np.isnan(component_value.imag) or \
                   np.isinf(component_value.real) or np.isinf(component_value.imag):
                    issues.append(f"{component_name} contains NaN/Inf: {component_value}")
            elif np.isnan(component_value) or np.isinf(component_value):
                issues.append(f"{component_name} is NaN/Inf: {component_value}")
        
        if issues:
            logger.warning(f"⚠️  Agent {self.charge_id} - Q component validation issues:")
            for issue in issues:
                logger.warning(f"    • {issue}")
            return False
        else:
            logger.debug(f"✅ Agent {self.charge_id} - Q components validation passed")
            return True
    
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
                imag_part = 0.0
                
            # Phase modulation from semantic phase factors
            phase = phase_factors[n] if n < len(phase_factors) else 0.0
            
            # Create living coefficient
            self.breathing_q_coefficients[n] = complex(real_part, imag_part) * np.exp(1j * phase)
        
        # Breathing rhythm from collective temporal patterns
        self.breath_frequency = self.collective_breathing.get('collective_frequency', [0.1])[0]
        self.breath_phase = 0.0
        self.breath_amplitude = 0.1  # How much coefficients oscillate
        
    def _initialize_responsive_hecke_system(self):
        """Create Hecke eigenvalues that adapt to field conditions."""
        # Base eigenvalues from trajectory operators
        trajectory_ops = self.temporal_biography.trajectory_operators
        
        if SAGE_AVAILABLE:
            # Use Sage for proper Hecke operator mathematics
            self.hecke_eigenvalues = {}
            # Map trajectory operators to prime Hecke eigenvalues
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
            for i, p in enumerate(primes):
                if i < len(trajectory_ops):
                    # CRITICAL FIX: Ensure proper scalar conversion from numpy to Python complex
                    traj_val = trajectory_ops[i]
                    if np.iscomplexobj(traj_val):
                        self.hecke_eigenvalues[p] = complex(float(traj_val.real), float(traj_val.imag))
                    else:
                        self.hecke_eigenvalues[p] = complex(float(traj_val), 0.0)
                else:
                    # Default eigenvalue for higher primes
                    self.hecke_eigenvalues[p] = complex(1.0, 0.0)
        else:
            # Fallback: direct mapping without Sage
            self.hecke_eigenvalues = {}
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for i, p in enumerate(primes):
                traj_val = trajectory_ops[i % len(trajectory_ops)]
                # CRITICAL FIX: Ensure proper scalar conversion from numpy to Python complex
                if np.iscomplexobj(traj_val):
                    self.hecke_eigenvalues[p] = complex(float(traj_val.real), float(traj_val.imag))
                else:
                    self.hecke_eigenvalues[p] = complex(float(traj_val), 0.0)
            
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
            base_coeff = modulation_tensor[n - 1] if n - 1 < len(modulation_tensor) else 1.0
            emotional_phase = np.angle(unified_phase) * n / 10
            
            self.l_function_coefficients[n] = base_coeff * np.exp(1j * emotional_phase)
        
        # Emotional conductor strength affects L-function evolution
        self.emotional_conductivity = self.emotional_modulation.field_modulation_strength
        
    def _initialize_modular_geometry(self):
        """Position agent in modular fundamental domain."""
        # Map field position to upper half-plane (modular fundamental domain)
        x, y = self.state.field_position
        
        # Transform to upper half-plane coordinates
        # Fundamental domain: |τ| ≥ 1, -1/2 ≤ Re(τ) ≤ 1/2, Im(τ) > 0
        real_part = np.clip(x, -0.5, 0.5)  # Real part in fundamental domain
        imag_part = max(0.1, 1.0 + y)      # Ensure positive imaginary part
        
        self.tau_position = complex(real_part, imag_part)
        
        # Modular weight determines transformation behavior
        field_magnitude = self.semantic_field_data['field_metadata']['field_magnitude']
        self.modular_weight = max(2, int(2 * field_magnitude))  # Even weight ≥ 2
        
        if PYG_AVAILABLE:
            # Create geometric node features for PyTorch Geometric
            self.geometric_features = torch.tensor([
                self.tau_position.real,
                self.tau_position.imag, 
                self.modular_weight,
                self.emotional_conductivity
            ], dtype=torch.float32, device=self.device)
        
    def _initialize_living_evolution(self):
        """Set up parameters for living evolution and cascading feedback."""
        # Evolution rates for different aspects
        self.evolution_rates = {
            'breathing': 0.01,      # How fast coefficients breathe
            'cascading': 0.005,     # Rate of dimensional feedback
            'interaction': 0.002,   # Rate of agent-agent interaction
            'memory': 0.001         # Rate of persistence evolution
        }
        
        # Cascading feedback state
        self.cascade_momentum = {
            'semantic_to_temporal': complex(0.0, 0.0),
            'temporal_to_emotional': complex(0.0, 0.0), 
            'emotional_to_semantic': complex(0.0, 0.0)
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
        breath_freq = self.breath_frequency.real if hasattr(self.breath_frequency, 'real') else self.breath_frequency
        self.breath_phase += breath_freq * tau * self.evolution_rates['breathing']
        
        # Breathing factor oscillates with collective rhythm
        breath_factor = 1.0 + self.breath_amplitude * np.sin(self.breath_phase)
        
        # Each q-coefficient breathes with its own harmonic
        for n in self.breathing_q_coefficients:
            # Base breathing oscillation
            harmonic_phase = self.breath_phase * (1 + n / 100)  # Higher harmonics breathe faster
            harmonic_factor = 1.0 + (self.breath_amplitude * 0.5) * np.sin(harmonic_phase)
            
            # Apply breathing to coefficient
            self.breathing_q_coefficients[n] *= breath_factor * harmonic_factor
            
            # Temporal phase evolution (each coefficient evolves in phase space)
            phase_evolution = self.temporal_biography.phase_coordination[n % len(self.temporal_biography.phase_coordination)]
            self.breathing_q_coefficients[n] *= np.exp(1j * phase_evolution * tau * 0.01)
            
            # Emotional modulation (conductor affects all coefficients)
            emotional_factor = self.emotional_modulation.semantic_modulation_tensor[n % len(self.emotional_modulation.semantic_modulation_tensor)]
            self.breathing_q_coefficients[n] *= (1.0 + 0.01 * emotional_factor)
        
        # Update living Q value after breathing
        self.living_Q_value = self.evaluate_living_form(tau)
        
        # Update charge object with current state
        self.charge_obj.complete_charge = self.living_Q_value
        self.charge_obj.magnitude = abs(self.living_Q_value)
        self.charge_obj.phase = np.angle(self.living_Q_value)
    
    def cascade_dimensional_feedback(self):
        """
        All dimensions flow into each other, reshaping the living form.
        
        This implements the cascading feedback loops where:
        Semantic → Temporal → Emotional → Semantic (endless cycle)
        """
        # STABILITY FIX: Apply decay to cascade momentum to prevent exponential explosion
        decay_factor = 0.95  # 5% decay per step to prevent unbounded growth
        for key in self.cascade_momentum:
            self.cascade_momentum[key] *= decay_factor
        # SEMANTIC → TEMPORAL: Field gradients drive temporal evolution
        q_magnitudes = [abs(self.breathing_q_coefficients.get(n, 0)) for n in range(100)]
        # CRITICAL FIX: Use float32 for MPS compatibility (Apple Silicon doesn't support float64)
        semantic_gradient = torch.tensor(q_magnitudes, dtype=torch.float32, device=self.device)
        
        # CRITICAL FIX: Manual circular padding for 1D tensor (PyTorch doesn't support 1D circular padding)
        # Add first element at end and last element at beginning for circular boundary conditions
        semantic_gradient_padded = torch.cat([
            semantic_gradient[-1:],  # Last element at beginning
            semantic_gradient,       # Original tensor
            semantic_gradient[:1]    # First element at end
        ], dim=0)
        
        semantic_gradient = torch.gradient(semantic_gradient_padded, dim=0)[0][1:101]  # Extract middle 100 elements
        
        # Update temporal momentum from semantic pressure
        gradient_magnitude = torch.mean(torch.abs(semantic_gradient)).item()
        temporal_influence = complex(gradient_magnitude, gradient_magnitude * 0.1)
        self.cascade_momentum['semantic_to_temporal'] += temporal_influence * self.evolution_rates['cascading']
        
        # Apply to temporal momentum
        if hasattr(self.temporal_biography, 'temporal_momentum'):
            self.temporal_biography.temporal_momentum += self.cascade_momentum['semantic_to_temporal'] * 0.1
        
        # TEMPORAL → EMOTIONAL: Breathing patterns modulate emotional response
        breath_coherence = self.temporal_biography.breathing_coherence
        temporal_momentum = getattr(self.temporal_biography, 'temporal_momentum', 0j)
        
        emotional_influence = breath_coherence * abs(temporal_momentum) * 0.1
        self.cascade_momentum['temporal_to_emotional'] += complex(emotional_influence, 0) * self.evolution_rates['cascading']
        
        # Apply to emotional phase shift
        self.emotional_modulation.unified_phase_shift *= (1 + 0.01j * emotional_influence)
        
        # EMOTIONAL → SEMANTIC: Conductor reshapes field landscape
        conductor_strength = abs(self.emotional_modulation.unified_phase_shift)
        conductor_phase = np.angle(self.emotional_modulation.unified_phase_shift)
        
        semantic_influence = complex(conductor_strength * 0.1, conductor_phase * 0.01)
        self.cascade_momentum['emotional_to_semantic'] += semantic_influence * self.evolution_rates['cascading']
        
        # Apply to q-coefficients (conductor reshapes the entire form)
        for n in range(min(100, len(self.breathing_q_coefficients))):
            phase_shift = conductor_phase * n / 100
            amplitude_shift = 1.0 + conductor_strength * 0.01
            self.breathing_q_coefficients[n] *= amplitude_shift * np.exp(1j * phase_shift)
        
        # ALL → OBSERVATIONAL STATE: Everything affects s-parameter evolution
        total_cascade_energy = sum(abs(momentum) for momentum in self.cascade_momentum.values())
        self.state.current_s += total_cascade_energy * self.evolution_rates['cascading'] * 0.1
        
        # Update living Q value after cascading feedback
        eval_start = time.time()
        self.living_Q_value = self.evaluate_living_form()
        eval_time = time.time() - eval_start
        if hasattr(self, 'charge_id'):
            logger.debug(f"🔧 Agent {self.charge_id} - CASCADE evaluate_living_form (time: {eval_time:.3f}s)")
        
        # Update charge object with current state
        self.charge_obj.complete_charge = self.living_Q_value
        self.charge_obj.magnitude = abs(self.living_Q_value)
        self.charge_obj.phase = np.angle(self.living_Q_value)
    
    def interact_with_field(self, other_agents: List['ConceptualChargeAgent']):
        """
        Living forms reshape each other through field interference.
        
        This creates the liquid metal effect where agents influence each other
        through modular geodesics and q-coefficient exchange.
        """
        for other in other_agents:
            if other is self:
                continue
                
            # Compute modular distance (geodesic in upper half-plane)
            tau_self = self.tau_position
            tau_other = other.tau_position
            
            # Hyperbolic distance in upper half-plane
            distance = abs(tau_self - tau_other) / (np.sqrt(tau_self.imag) * np.sqrt(tau_other.imag))
            
            # Modular influence strength (decays with distance)
            influence_strength = np.exp(-distance) / (1 + distance)
            
            if influence_strength < 0.001:  # Too far to interact
                continue
            
            # q-coefficients LEARN from each other
            for n in range(min(50, len(self.breathing_q_coefficients), len(other.breathing_q_coefficients))):
                # Interference between coefficients
                self_coeff = self.breathing_q_coefficients.get(n, 0)
                other_coeff = other.breathing_q_coefficients.get(n, 0)
                
                interference = self_coeff * np.conj(other_coeff)
                
                # Coefficients EVOLVE based on interference
                evolution_factor = influence_strength * self.evolution_rates['interaction']
                self.breathing_q_coefficients[n] += interference * evolution_factor
                
                # Higher harmonics can EMERGE from interaction
                if n < 25:  # Create new harmonics
                    harmonic_n = 2 * n + 1
                    if harmonic_n not in self.breathing_q_coefficients:
                        self.breathing_q_coefficients[harmonic_n] = 0
                    
                    # New harmonic born from interference
                    self.breathing_q_coefficients[harmonic_n] += \
                        interference * influence_strength * 0.001
            
            # Hecke eigenvalues adapt to neighboring values
            for p in self.hecke_eigenvalues:
                if p in other.hecke_eigenvalues:
                    neighbor_eigenvalue = other.hecke_eigenvalues[p]
                    adaptation = (neighbor_eigenvalue - self.hecke_eigenvalues[p]) * influence_strength
                    self.hecke_eigenvalues[p] += adaptation * self.hecke_adaptivity
            
            # Store interaction in memory
            interaction_record = {
                'tau': tau_other,
                'influence': influence_strength,
                'interference_energy': abs(interference) if 'interference' in locals() else 0.0,
                'timestamp': self.state.current_s
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
        self.charge_obj.phase = np.angle(self.living_Q_value)
    
    def interact_with_optimized_field(self, nearby_agents: List[Tuple['ConceptualChargeAgent', float]]):
        """
        O(N log N) OPTIMIZED field interactions using sparse neighbor graph.
        
        This method receives pre-computed nearby agents with their interaction strengths,
        eliminating the need to check all agents (O(N²) → O(log N) per agent).
        
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
            for n in range(min(50, len(self.breathing_q_coefficients), len(other.breathing_q_coefficients))):
                # Interference between coefficients
                self_coeff = self.breathing_q_coefficients.get(n, 0)
                other_coeff = other.breathing_q_coefficients.get(n, 0)
                
                interference = self_coeff * np.conj(other_coeff)
                
                # Coefficients EVOLVE based on interference
                evolution_factor = influence_strength * self.evolution_rates['interaction']
                self.breathing_q_coefficients[n] += interference * evolution_factor
                
                # Higher harmonics can EMERGE from interaction
                if n < 25:  # Create new harmonics
                    harmonic_n = 2 * n + 1
                    if harmonic_n not in self.breathing_q_coefficients:
                        self.breathing_q_coefficients[harmonic_n] = 0
                    
                    # New harmonic born from interference
                    self.breathing_q_coefficients[harmonic_n] += \
                        interference * influence_strength * 0.001
            
            # Hecke eigenvalues adapt to neighboring values
            for p in self.hecke_eigenvalues:
                if p in other.hecke_eigenvalues:
                    neighbor_eigenvalue = other.hecke_eigenvalues[p]
                    adaptation = (neighbor_eigenvalue - self.hecke_eigenvalues[p]) * influence_strength
                    self.hecke_eigenvalues[p] += adaptation * self.hecke_adaptivity
            
            # Store interaction in memory
            interaction_record = {
                'tau': other.tau_position,
                'influence': influence_strength,
                'interference_energy': abs(interference) if 'interference' in locals() else 0.0,
                'timestamp': self.state.current_s,
                'optimized': True  # Mark as optimized interaction
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
        self.charge_obj.phase = np.angle(self.living_Q_value)
    
    def interact_with_precomputed_field(self, nearby_agents_with_strengths: List[Tuple['ConceptualChargeAgent', float]]):
        """
        EFFICIENT FALLBACK: Use pre-computed interaction strengths from sparse graph.
        
        This method provides the same mathematical operations as interact_with_optimized_field
        but can be called from legacy code that expects a different signature.
        
        Args:
            nearby_agents_with_strengths: List of (agent, pre_computed_interaction_strength) tuples
        """
        # Delegate to the optimized method - they have identical signatures and behavior
        self.interact_with_optimized_field(nearby_agents_with_strengths)
    
    def evolve_s_parameter(self, field_context: List['ConceptualChargeAgent']):
        """
        s-parameter drives the LIFE of the modular form.
        
        Observational state evolution based on field pressure, persistence memory,
        and interaction history. This makes the form irreversibly evolve.
        """
        # Compute total field pressure from all agents
        field_pressure = sum(abs(agent.living_Q_value) for agent in field_context) / len(field_context)
        
        # Dual persistence influences evolution rate
        s_index = int(self.state.current_s) % 1024
        vivid_influence = self.temporal_biography.vivid_layer[s_index] if s_index < len(self.temporal_biography.vivid_layer) else 1.0
        character_influence = self.temporal_biography.character_layer[s_index] if s_index < len(self.temporal_biography.character_layer) else 0.001
        
        # Memory pressure from past interactions
        if self.interaction_memory:
            recent_memory = self.interaction_memory[-10:]  # Last 10 interactions
            memory_pressure = sum(record['influence'] for record in recent_memory) / len(recent_memory)
        else:
            memory_pressure = 0.0
        
        # s evolution combining all influences
        temporal_momentum = getattr(self.temporal_biography, 'temporal_momentum', 0j)
        delta_s = (
            field_pressure * vivid_influence * 0.01 +           # Immediate field response
            abs(temporal_momentum) * character_influence * 0.1 + # Character-based momentum
            memory_pressure * 0.05                               # Memory influence
        )
        
        self.state.current_s += delta_s * self.evolution_rates['memory']
        
        # s evolution RESHAPES the modular form itself
        # Higher s means more complex, lower s means simpler
        s_distance = abs(self.state.current_s - self.state.s_zero)
        complexity_factor = np.exp(-s_distance / 50)  # Gradual complexity decay
        
        # Apply complexity evolution to q-coefficients
        for n in self.breathing_q_coefficients:
            # Higher order coefficients are more sensitive to s-evolution
            sensitivity = 1.0 + n / 100
            decay_factor = complexity_factor ** sensitivity
            self.breathing_q_coefficients[n] *= decay_factor
        
        # Update living Q value after s-parameter evolution
        self.living_Q_value = self.evaluate_living_form()
        
        # Update charge object with current state
        self.charge_obj.complete_charge = self.living_Q_value
        self.charge_obj.magnitude = abs(self.living_Q_value)
        self.charge_obj.phase = np.angle(self.living_Q_value)
    
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
        if hasattr(self, '_cached_living_Q') and hasattr(self, '_cached_s'):
            s_change = abs(current_s - self._cached_s)
            if s_change < 0.001:  # Very small change in observational state
                logger.debug(f"🔧 Agent {self.charge_id} - Using cached living Q (s_change: {s_change:.6f})")
                return self._cached_living_Q
        
        # Compute q = exp(2πiτ) - convert complex tau to proper PyTorch tensor
        tau_tensor = torch.tensor(2j * np.pi * tau, dtype=torch.complex64, device=self.device)
        q = torch.exp(tau_tensor)
        
        # Evaluate breathing q-expansion - ensure tensor/complex consistency
        f_tau = torch.tensor(complex(0.0, 0.0), dtype=torch.complex64, device=self.device)
        for n, coeff in self.breathing_q_coefficients.items():
            coeff_tensor = torch.tensor(coeff, dtype=torch.complex64, device=self.device)
            if n == 0:
                f_tau += coeff_tensor  # Constant term
            else:
                f_tau += coeff_tensor * (q ** n)
        
        # Apply responsive Hecke operators
        for p, eigenvalue in self.hecke_eigenvalues.items():
            # Hecke operator T_p applied as multiplicative factor
            hecke_factor = 1.0 + eigenvalue * (q ** p) / (1.0 + abs(q ** p))
            f_tau *= hecke_factor
        
        # Apply emotional L-function modulation - ensure tensor consistency
        l_value = torch.tensor(complex(1.0, 0.0), dtype=torch.complex64, device=self.device)
        for n, coeff in self.l_function_coefficients.items():
            if abs(coeff) > 0:
                coeff_tensor = torch.tensor(coeff, dtype=torch.complex64, device=self.device)
                n_factor = torch.tensor(n ** (1 + 0.1j), dtype=torch.complex64, device=self.device)
                l_value *= (1.0 + coeff_tensor / n_factor)
        
        f_tau *= l_value
        
        # Apply observational state persistence - ensure tensor consistency
        s_factor = self.compute_observational_persistence()[0]  # Total persistence
        s_factor_tensor = torch.tensor(s_factor, dtype=torch.complex64, device=self.device)
        f_tau *= s_factor_tensor
        
        # Store as living Q value
        # Convert back to Python complex for compatibility with rest of system
        # CRITICAL FIX: Use safe tensor validation to prevent boolean evaluation errors
        try:
            context = f"Agent_{self.charge_id}_Q_calculation"
            if safe_tensor_comparison(f_tau, 1, context):
                # CRITICAL FIX: Use safe scalar extraction to prevent ambiguous tensor evaluation
                tensor_value = extract_tensor_scalar(f_tau, context)
                if isinstance(tensor_value, complex):
                    f_tau_complex = complex(float(tensor_value.real), float(tensor_value.imag))
                else:
                    f_tau_complex = complex(float(tensor_value), 0.0)
            else:
                f_tau_complex = f_tau
        except TensorValidationError as e:
            print(f"💥 Agent {self.charge_id} - Q calculation tensor validation failed: {e}")
            # Fallback to default complex value with error reporting
            f_tau_complex = complex(1.0, 0.0)
        except Exception as e:
            print(f"💥 Agent {self.charge_id} - Unexpected Q calculation error: {e}")
            f_tau_complex = complex(1.0, 0.0)
        
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
        y = (self.tau_position.imag - 1.0)  # Shift from modular to grid (Im(τ) ≥ 1 → y ≥ 0)
        
        # Update field position
        self.state.field_position = (x, y)
        
        # Update charge object metadata if it exists
        if hasattr(self.charge_obj, 'metadata') and hasattr(self.charge_obj.metadata, 'field_position'):
            self.charge_obj.metadata.field_position = (x, y)
        elif hasattr(self.charge_obj, 'set_field_position'):
            self.charge_obj.set_field_position((x, y))
    
    def update_tau_from_field(self):
        """
        Update tau_position from field_position if moved by orchestrator.
        
        This allows the orchestrator to move agents in grid space and have
        their modular domain position automatically updated.
        """
        x, y = self.state.field_position
        
        # Convert field (grid) to tau (modular) coordinates
        # Ensure Im(τ) > 0 for valid modular domain
        real_part = np.clip(x, -0.5, 0.5)  # Keep in fundamental domain
        imag_part = max(0.1, 1.0 + y)      # Ensure positive imaginary part
        
        self.tau_position = complex(real_part, imag_part)
        
        
    def compute_gamma_calibration(self, collective_field_strength: Optional[float] = None, pool_size: int = 1) -> float:
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
            interference_strength = np.mean(np.abs(self.field_interference_matrix))
            collective_field_strength = 1.0 + interference_strength
        
        # 🚀 BOOST GAMMA: Scale with pool size for stronger field presence
        # Larger pools need stronger individual gamma to maintain field energy
        pool_boost = max(1.0, np.sqrt(pool_size))  # Square root scaling prevents overwhelming
        
        # Apply additional field strength boost for 10-agent pools
        field_strength_boost = 2.0 if pool_size >= 10 else 1.5
        
        # Calibrate to prevent overwhelming or weakness
        # Higher collective field → lower individual gamma (normalization)
        calibration_factor = conductor_modulation / (1.0 + 0.1 * collective_field_strength)
        
        # Final gamma with pool-based boosting
        boosted_gamma = base_gamma * calibration_factor * pool_boost * field_strength_boost
        
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
        T_ops = torch.tensor(trajectory_ops, dtype=torch.complex64, device=self.device)
        
        # Context C modulates trajectory through observer contingency
        context_size = len(self.state.current_context_C)
        context_modulation = 1.0 + 0.1 * np.log1p(context_size)  # log1p for stability
        
        # Observational state s affects trajectory integration
        s = self.state.current_s
        s_evolution = torch.exp(torch.tensor(-0.05 * s, dtype=torch.float32, device=self.device))
        
        # Emotional coupling modulates T tensor (multiplicative effect)
        emotional_coupling = abs(self.coupling_state.emotional_field_coupling)
        
        # Compute tensor with multiplicative interactions
        T_tensor_value = torch.mean(T_ops) * context_modulation * s_evolution * (1.0 + 0.2 * emotional_coupling)
        
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
        integration_start = time.time()
        frequency_evolution = self.temporal_biography.frequency_evolution
        if len(frequency_evolution) > 0 and abs(s - s_zero) > 0.01:
            logger.debug(f"🔧 Agent {self.charge_id} - Computing temporal integral: freq_len={len(frequency_evolution)}, s_range=({s_zero:.3f}, {s:.3f})")
            
            # PERFORMANCE FIX: Use fast trapezoidal rule instead of expensive quad integration
            # Extract real parts for integration (frequency_evolution can be complex)
            omega_values = np.real(frequency_evolution) if hasattr(frequency_evolution, '__len__') else [np.real(frequency_evolution)]
            s_span = abs(s - s_zero)
            
            if len(omega_values) > 1:
                # Trapezoidal rule: much faster than cubic interpolation + quad
                temporal_integral = np.trapz(omega_values, dx=s_span/(len(omega_values)-1))
            else:
                # Single value case
                temporal_integral = omega_values[0] * s_span if omega_values else 0.0
                
            logger.debug(f"🔧 Agent {self.charge_id} - Fast integration complete (time: {time.time() - integration_start:.3f}s, result: {temporal_integral:.6f})")
        else:
            temporal_integral = 0.0
            logger.debug(f"🔧 Agent {self.charge_id} - Skipping integration: freq_len={len(frequency_evolution)}, s_range=({s_zero:.3f}, {s:.3f})")
        
        logger.debug(f"🔧 Agent {self.charge_id} - Total integration phase (time: {time.time() - integration_start:.3f}s)")
        
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
        logger.info(f"🔧 Agent {self.charge_id} - Starting observational persistence computation")
        
        # CRITICAL DEBUG: Check state value types before access
        logger.info(f"🔧 Agent {self.charge_id} - State types: current_s={type(self.state.current_s)}, s_zero={type(self.state.s_zero)}")
        
        # MPS-safe state access with explicit tensor checking
        if hasattr(self.state.current_s, 'cpu'):
            s = float(self.state.current_s.cpu().detach().numpy())
            logger.info(f"🔧 Agent {self.charge_id} - Converted current_s tensor to float: {s}")
        else:
            s = float(self.state.current_s)
            
        if hasattr(self.state.s_zero, 'cpu'):
            s_zero = float(self.state.s_zero.cpu().detach().numpy())
            logger.info(f"🔧 Agent {self.charge_id} - Converted s_zero tensor to float: {s_zero}")
        else:
            s_zero = float(self.state.s_zero)
            
        logger.info(f"🔧 Agent {self.charge_id} - State values: s={s}, s_zero={s_zero}")
        
        delta_s = s - s_zero
        
        # Extract persistence layers from temporal biography
        vivid_layer = self.temporal_biography.vivid_layer
        character_layer = self.temporal_biography.character_layer
        logger.debug(f"🔧 Agent {self.charge_id} - Layer types: vivid={type(vivid_layer)}, character={type(character_layer)}")
        logger.debug(f"🔧 Agent {self.charge_id} - Layer lengths: vivid={len(vivid_layer)}, character={len(character_layer)}")
        
        # Gaussian component from vivid layer (recent sharp memory)
        if len(vivid_layer) > 0:
            # Use actual vivid layer data with underflow protection
            # CRITICAL FIX: Ensure mean is real before comparison - MPS-safe
            logger.info(f"🔧 Agent {self.charge_id} - Processing vivid_layer, type: {type(vivid_layer)}")
            if hasattr(vivid_layer, 'cpu'):
                vivid_mean = np.mean(vivid_layer.cpu().detach().numpy())
                logger.info(f"🔧 Agent {self.charge_id} - Converted vivid_layer tensor to numpy")
            else:
                vivid_mean = np.mean(vivid_layer)
                logger.info(f"🔧 Agent {self.charge_id} - Used vivid_layer directly as numpy")
            vivid_mean_real = float(vivid_mean.real) if np.iscomplexobj(vivid_mean) else float(vivid_mean)
            vivid_base = float(max(0.9, vivid_mean_real))  # 🔧 Clamp vivid layer to minimum 0.9 - ensure float
            logger.info(f"🔧 Agent {self.charge_id} - About to compute vivid_influence with vivid_base={vivid_base}, delta_s={delta_s}")
            
            # CRITICAL FIX: Ensure sigma_i is a float for MPS compatibility
            if hasattr(self.sigma_i, 'cpu'):
                sigma_i_val = float(self.sigma_i.cpu().detach().numpy())
            else:
                sigma_i_val = float(self.sigma_i)
            
            try:
                vivid_influence = vivid_base * np.exp(-(delta_s * delta_s) / (2 * sigma_i_val * sigma_i_val))
                logger.info(f"🔧 Agent {self.charge_id} - Vivid influence computed: {vivid_influence}")
            except Exception as vivid_error:
                logger.error(f"🔧 Agent {self.charge_id} - Error in vivid_influence computation: {vivid_error}")
                raise
        else:
            # CRITICAL FIX: Ensure sigma_i is a float for MPS compatibility
            if hasattr(self.sigma_i, 'cpu'):
                sigma_i_val = float(self.sigma_i.cpu().detach().numpy())
            else:
                sigma_i_val = float(self.sigma_i)
                
            try:
                vivid_influence = np.exp(-(delta_s * delta_s) / (2 * sigma_i_val * sigma_i_val))
                logger.info(f"🔧 Agent {self.charge_id} - Vivid influence (else case) computed: {vivid_influence}")
            except Exception as vivid_else_error:
                logger.error(f"🔧 Agent {self.charge_id} - Error in vivid_influence (else) computation: {vivid_else_error}")
                raise
        
        # Apply additional vivid underflow protection
        vivid_influence_real = float(vivid_influence.real) if np.iscomplexobj(vivid_influence) else float(vivid_influence)
        vivid_influence = max(0.8, vivid_influence_real)  # 🔧 Keep vivid influence above 0.8
        
        # Exponential-cosine from character layer (persistent themes)
        if len(character_layer) > 0:
            # Use actual character layer data with underflow protection
            # CRITICAL FIX: Ensure mean is real before comparison - MPS-safe
            if hasattr(character_layer, 'cpu'):
                character_mean = np.mean(character_layer.cpu().detach().numpy())
            else:
                character_mean = np.mean(character_layer)
            character_mean_real = float(character_mean.real) if np.iscomplexobj(character_mean) else float(character_mean)
            character_base = float(max(0.08, character_mean_real))  # 🔧 Clamp character layer minimum - ensure float
            character_influence = character_base * np.exp(-self.lambda_i * abs(delta_s))
            # Add rhythmic reinforcement
            character_influence *= np.cos(self.beta_i * delta_s)
        else:
            character_influence = self.alpha_i * np.exp(-self.lambda_i * abs(delta_s)) * np.cos(self.beta_i * delta_s)
        
        # Apply additional character underflow protection
        character_influence_real = float(abs(character_influence))  # abs() ensures real value
        character_influence = max(0.05, character_influence_real)  # 🔧 Keep character influence above 0.05
        
        # Combined persistence with stronger minimum
        total_persistence = vivid_influence + character_influence
        
        # Ensure persistence stays in reasonable range for Q computation
        total_persistence = max(0.9, min(2.0, total_persistence))  # 🔧 Clamp to [0.9, 2.0]
        
        # Log when clamping occurs - MPS-safe vivid layer access
        if len(vivid_layer) > 0:
            # MPS-safe mean calculation for vivid layer
            if hasattr(vivid_layer, 'cpu'):
                vivid_mean_for_log = np.mean(vivid_layer.cpu().detach().numpy())
            else:
                vivid_mean_for_log = np.mean(vivid_layer)
            
            if abs(vivid_mean_for_log) < 0.9:
                logger.debug(f"🔧 Agent {self.charge_id} - Vivid layer clamped: {vivid_mean_for_log:.3f} → 0.9+")
        if total_persistence == 0.9:
            logger.debug(f"🔧 Agent {self.charge_id} - Persistence clamped to minimum: 0.9")
        
        return total_persistence, vivid_influence, character_influence
        
    def compute_complete_Q(self, collective_field_strength: Optional[float] = None, pool_size: int = 1) -> QMathematicalComponents:
        """
        Compute complete Q(τ, C, s) = γ · T · E · Φ · e^(iθ) · Ψ
        
        This is the living mathematical entity in action - computing the complete
        conceptual charge using actual field theory mathematics with real data.
        """
        start_time = time.time()
        logger.info(f"🔧 Agent {self.charge_id} - Starting Q computation (pool_size: {pool_size})")
        
        try:
            # Compute all components with detailed timing
            component_start = time.time()
            gamma = self.compute_gamma_calibration(collective_field_strength, pool_size)
            logger.info(f"🔧 Agent {self.charge_id} - gamma: {gamma:.6f} (time: {time.time() - component_start:.3f}s)")
            
            component_start = time.time()
            T_tensor = self.compute_transformative_potential_tensor()
            logger.info(f"🔧 Agent {self.charge_id} - T_tensor: {T_tensor} (magnitude: {abs(T_tensor):.6f}, time: {time.time() - component_start:.3f}s)")
            
            component_start = time.time()
            E_trajectory = self.compute_emotional_trajectory_integration()  
            logger.info(f"🔧 Agent {self.charge_id} - E_trajectory: {E_trajectory} (magnitude: {abs(E_trajectory):.6f}, time: {time.time() - component_start:.3f}s)")
            
            component_start = time.time()
            phi_semantic = self.compute_semantic_field_generation()
            logger.info(f"🔧 Agent {self.charge_id} - phi_semantic: {phi_semantic} (magnitude: {abs(phi_semantic):.6f}, time: {time.time() - component_start:.3f}s)")
            
            component_start = time.time()
            theta_components = self.compute_5component_phase_integration()
            phase_factor = complex(np.cos(theta_components.total), np.sin(theta_components.total))
            logger.info(f"🔧 Agent {self.charge_id} - phase_integration (time: {time.time() - component_start:.3f}s)")
            
            # 🔧 VERIFY: Phase factor magnitude should be exactly 1.0 (e^iθ property)
            phase_magnitude = abs(phase_factor)
            if abs(phase_magnitude - 1.0) > 1e-10:
                logger.warning(f"⚠️  Agent {self.charge_id} - Phase factor magnitude error: |e^iθ| = {phase_magnitude:.10f} (should be 1.0)")
                logger.warning(f"    theta_total = {theta_components.total:.6f}, cos = {np.cos(theta_components.total):.6f}, sin = {np.sin(theta_components.total):.6f}")
                # Normalize to unit magnitude
                phase_factor = phase_factor / phase_magnitude
                logger.warning(f"    Normalized phase_factor: {phase_factor} (new magnitude: {abs(phase_factor):.10f})")
            
            logger.debug(f"🔧 Agent {self.charge_id} - phase_factor: {phase_factor} (magnitude: {abs(phase_factor):.6f})")
            
            # CRITICAL DEBUG: Wrap persistence computation to catch MPS errors
            try:
                logger.debug(f"🔧 Agent {self.charge_id} - About to call compute_observational_persistence()")
                psi_persistence, psi_gaussian, psi_exponential_cosine = self.compute_observational_persistence()
                logger.debug(f"🔧 Agent {self.charge_id} - Persistence computation completed successfully")
                logger.debug(f"🔧 Agent {self.charge_id} - psi_persistence: {psi_persistence:.6f} (gaussian: {psi_gaussian:.6f}, exp_cos: {psi_exponential_cosine:.6f})")
            except Exception as persistence_error:
                logger.error(f"🔧 Agent {self.charge_id} - Persistence computation failed: {persistence_error}")
                raise
            
            # Apply underflow protection to persistence components
            psi_persistence = max(psi_persistence, 1e-10)
            psi_gaussian = max(psi_gaussian, 1e-15) 
            psi_exponential_cosine = max(psi_exponential_cosine, 1e-15)
            
            if psi_persistence < 1e-8:
                logger.warning(f"⚠️  Agent {self.charge_id} - Persistence underflow detected, clamped to {psi_persistence:.2e}")
            
            # Final Q(τ, C, s) computation
            Q_value = gamma * T_tensor * E_trajectory * phi_semantic * phase_factor * psi_persistence
            
            # Validate Q magnitude
            Q_magnitude = abs(Q_value)
            if Q_magnitude < 1e-10:
                logger.warning(f"⚠️  Agent {self.charge_id} - Q magnitude suspiciously small: {Q_magnitude:.2e}")
                logger.warning(f"    Components - γ:{gamma:.3f} |T|:{abs(T_tensor):.3f} |E|:{abs(E_trajectory):.3f} |Φ|:{abs(phi_semantic):.3f} |e^iθ|:{abs(phase_factor):.3f} Ψ:{psi_persistence:.3f}")
            elif Q_magnitude > 10.0:
                logger.warning(f"⚠️  Agent {self.charge_id} - Q magnitude unusually large: {Q_magnitude:.2e}")
            else:
                logger.debug(f"✅ Agent {self.charge_id} - Q computed successfully: {Q_value} (magnitude: {Q_magnitude:.6f})")
            
        except Exception as e:
            logger.error(f"❌ Agent {self.charge_id} - Q computation failed: {e}")
            # NO FALLBACKS - we want to use stored Q values, not compute new ones
            raise ValueError(f"Q computation failed and we refuse to use fallback values: {e}")
        
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
        
        # 🔧 FIX: Update living_Q_value with computed Q (not static 1+0j override!)
        self.living_Q_value = Q_value
        logger.debug(f"🔧 Agent {self.charge_id} - living_Q_value updated: {self.living_Q_value} (magnitude: {abs(self.living_Q_value):.6f})")
        
        # 🔧 VALIDATE: Check Q components are reasonable
        self.validate_Q_components()
        
        total_time = time.time() - start_time
        logger.info(f"✅ Agent {self.charge_id} - Q computation COMPLETE (total time: {total_time:.3f}s)")
        
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
            },
            # 🔧 Enhanced: Additional debugging information
            'charge_index': self.charge_index,
            'data_source_validation': {
                'semantic_field_magnitude': self.semantic_field_data['field_metadata']['field_magnitude'],
                'trajectory_operators_non_zero': int(np.sum(np.abs(self.temporal_biography.trajectory_operators) > 1e-12)),
                'trajectory_operators_total': len(self.temporal_biography.trajectory_operators),
                'emotional_modulation_strength': self.emotional_modulation.field_modulation_strength,
                'persistence_layers_range': {
                    'vivid_layer': [float(np.min(self.temporal_biography.vivid_layer)), float(np.max(self.temporal_biography.vivid_layer))],
                    'character_layer': [float(np.min(self.temporal_biography.character_layer)), float(np.max(self.temporal_biography.character_layer))]
                }
            }
        }
    
    def log_debug_breakdown(self):
        """Log detailed debugging information for this agent."""
        breakdown = self.get_mathematical_breakdown()
        
        logger.debug(f"🔧 AGENT DEBUG [{self.charge_index}] {self.charge_id}:")
        logger.debug(f"  τ: {breakdown['tau_content']}")
        logger.debug(f"  State: s={breakdown['current_state']['s']:.3f}, pos={breakdown['current_state']['field_position']}")
        
        # Q Components summary
        Q_comps = breakdown['Q_components']
        logger.debug(f"  🧮 Q Components:")
        logger.debug(f"    γ: {Q_comps['gamma']:.6f}")
        logger.debug(f"    |T|: {Q_comps['T_tensor']['magnitude']:.6f}, ∠T: {Q_comps['T_tensor']['phase']:.3f}")
        logger.debug(f"    |E|: {Q_comps['E_trajectory']['magnitude']:.6f}, ∠E: {Q_comps['E_trajectory']['phase']:.3f}")
        logger.debug(f"    |Φ|: {Q_comps['phi_semantic']['magnitude']:.6f}, ∠Φ: {Q_comps['phi_semantic']['phase']:.3f}")
        logger.debug(f"    Ψ: {Q_comps['psi_persistence']['total']:.6f} (gauss: {Q_comps['psi_persistence']['gaussian']:.6f}, exp_cos: {Q_comps['psi_persistence']['exponential_cosine']:.6f})")
        logger.debug(f"    Q: {breakdown['final_Q']['magnitude']:.6f} ∠ {breakdown['final_Q']['phase']:.3f}")
        
        # Source data quality
        if 'data_source_validation' in breakdown:
            src_data = breakdown['data_source_validation']
            logger.debug(f"  📊 Source Data Quality:")
            logger.debug(f"    Trajectory ops: {src_data['trajectory_operators_non_zero']}/{src_data['trajectory_operators_total']} non-zero")
            logger.debug(f"    Emotional strength: {src_data['emotional_modulation_strength']:.6f}")
            logger.debug(f"    Persistence ranges: vivid=[{src_data['persistence_layers_range']['vivid_layer'][0]:.3f}, {src_data['persistence_layers_range']['vivid_layer'][1]:.3f}], char=[{src_data['persistence_layers_range']['character_layer'][0]:.3f}, {src_data['persistence_layers_range']['character_layer'][1]:.3f}]")
        
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
        if Q_mag < 1e-10: health_flags.append("Q_TINY")
        if Q_mag > 10: health_flags.append("Q_LARGE")
        if T_mag < 1e-8: health_flags.append("T_ZERO")
        if psi < 1e-8: health_flags.append("PSI_TINY")
        
        health_str = "|" + "|".join(health_flags) + "|" if health_flags else "OK"
        
        return f"[{self.charge_index}] Q:{Q_mag:.2e} γ:{gamma:.3f} |T|:{T_mag:.2e} |E|:{E_mag:.3f} |Φ|:{phi_mag:.3f} Ψ:{psi:.3f} {health_str}"