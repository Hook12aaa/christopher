"""
Temporal Dimension Helper - Breathing Patterns of Conceptual Charges

UNIFIED FIELD-MOVEMENT ARCHITECTURE: Following the weather map analogy in my paper, this module
treats temporal dimensions as unified field-movement entities where clouds (conceptual charges)
move in trajectories AND generate field effects simultaneously. They are not separate 
phenomena but unified entities that interfere with each other.

MATHEMATICAL FOUNDATION: Section 3.1.4 - Temporal Dimension
Key Formula: T_i(τ,C,s) = ∫₀ˢ ω_i(τ,s')·e^(iφ_i(τ,s')) ds'
Observational Persistence: Ψ_persistence(s-s₀) with dual-decay structure

BREATHING PATTERN PHILOSOPHY: The temporal dimension IS the breathing pattern of 
conceptual charge formation. It's not separate from transformation - it IS the 
breathing rhythm that modulates the formation process itself.

BGE TEMPORAL EXTRACTION: Leverages existing BGE heat kernel field evolution,
spatial spectral analysis, and trajectory persistence measurements as natural
temporal signatures embedded in the embedding space.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.utils.logger import get_logger
logger = get_logger(__name__)


@dataclass
class TemporalBiography:
    """
    The 'temporal DNA' of a conceptual charge - its complete breathing pattern.
    
    This is the unified field-movement entity that encodes how a charge moves
    through observational states while generating field effects.
    """
    # Core trajectory integration T_i(τ,C,s)
    trajectory_operators: np.ndarray          # Complex trajectory integrals
    
    # Observational persistence Ψ_persistence(s-s₀)
    vivid_layer: np.ndarray                  # Gaussian decay - recent sharp memory
    character_layer: np.ndarray              # Exponential-cosine - persistent themes
    
    # Breathing rhythm components
    frequency_evolution: np.ndarray          # ω_i(τ,s') - how fast this charge breathes
    phase_coordination: np.ndarray           # φ_i(τ,s') - interference patterns
    
    # Field-movement unity metrics
    temporal_momentum: complex               # Strength and direction of temporal movement
    breathing_coherence: float               # How synchronized the breathing pattern is
    field_interference_signature: np.ndarray # How this charge interferes with others
    
    # BGE-derived temporal signatures
    bge_temporal_signature: Dict[str, Any]   # Natural temporal patterns from BGE analysis


class TemporalDimensionHelper:
    """
    Temporal Dimension Helper - Breathing Pattern Generator
    
    ARCHITECTURE PARALLEL: Follows SemanticDimensionHelper pattern for consistent
    integration with ChargeFactory while implementing unified field-movement dynamics.
    
    BREATHING RHYTHM PHILOSOPHY: Extracts natural temporal signatures from BGE's
    mathematical structure and transforms them into breathing patterns that modulate
    conceptual charge formation.
    
    FIELD COUPLING READY: Prepares temporal components for integration with 
    emotional and manifold field effects (hooks for future coupling).
    """
    
    def __init__(self, from_base: bool = True, model_info: dict = None, helper = None):
        """
        Initialize temporal dimension helper.
        
        Args:
            from_base (bool): Whether building from base model (BGE/MPNet) or existing universe
            model_info (dict): Model configuration information
            helper: BGE ingestion model (if from_base=True) or universe (if from_base=False)
        """
        self.from_base = from_base
        self.model_info = model_info
        
        # Set helper for BGE temporal signature extraction
        if self.from_base:
            self.bge_model = helper
            logger.info("TemporalDimensionHelper initialized with BGE temporal signature extraction.")
        else:
            self.universe = helper
            logger.info("TemporalDimensionHelper initialized for universe integration mode.")
        
        # Initialize temporal breathing pattern generators
        self._init_breathing_generators()
        
        # Prepare coupling hooks for future emotional/manifold integration
        self._init_coupling_hooks()
    
    def _init_breathing_generators(self):
        """Initialize the core breathing pattern generation components."""
        self.trajectory_integrator = TrajectoryIntegrator()
        self.breathing_rhythm_extractor = BreathingRhythmExtractor()
        self.temporal_persistence_engine = TemporalPersistenceEngine()
        
        logger.info("✅ Initialized temporal breathing pattern generators")
    
    def _init_coupling_hooks(self):
        """
        Initialize coupling interfaces for future emotional and manifold integration.
        
        TODO: These will be expanded when emotional dimension is implemented.
        For now, they provide the structure for field coupling preparation.
        """
        self.coupling_interfaces = {
            'emotional_field_coupling': None,  # TODO: Hook for emotional field modulation
            'manifold_warping_coupling': None, # TODO: Hook for manifold geometry effects
            'field_interference_coupling': None, # TODO: Hook for multi-charge interference
        }
        
        logger.info("🔗 Prepared coupling hooks for future field integration")
    
    def convert_embedding_to_temporal_field(self, embedding_data: List[Dict], vocab_mappings: dict = None) -> Dict[str, Any]:
        """
        Core transformation: Convert embedding vectors to temporal field representations.
        
        UNIFIED APPROACH: Extracts temporal breathing patterns from BGE's natural
        temporal signatures (heat kernel evolution, spectral analysis, trajectory persistence)
        and transforms them into unified field-movement entities.
        
        Args:
            embedding_data: List of embedding dictionaries containing:
                - 'vector': The embedding vector
                - 'token': Source token
                - 'manifold_properties': Spatial properties from BGE analysis
                - 'index': Token index
        
        Returns:
            Dict containing temporal field representations with breathing patterns
        """
        logger.info(f"🌊 Converting {len(embedding_data)} embeddings to temporal breathing patterns")
        
        # Extract individual embeddings from BGE search results for biography generation
        individual_embeddings = []
        for item in embedding_data:
            if 'embeddings' in item and isinstance(item['embeddings'], list):
                # This is a BGE search result with embedded list
                individual_embeddings.extend(item['embeddings'])
            else:
                # This is already an individual embedding dictionary
                individual_embeddings.append(item)
        
        # STEP 1: Extract BGE temporal signatures for the entire batch
        bge_temporal_analysis = self._extract_bge_temporal_signatures(embedding_data)
        
        # STEP 2: Generate temporal biographies for each individual embedding
        temporal_biographies = []
        for i, embedding in enumerate(individual_embeddings):
            biography = self._generate_temporal_biography(
                embedding, 
                bge_temporal_analysis,
                observational_index=i
            )
            temporal_biographies.append(biography)
        
        # STEP 3: Compute field interference patterns across all charges
        interference_matrix = self._compute_field_interference_matrix(temporal_biographies)
        
        # STEP 4: Generate collective breathing rhythm
        collective_rhythm = self._compute_collective_breathing_rhythm(temporal_biographies)
        
        return {
            'temporal_biographies': temporal_biographies,
            'bge_temporal_analysis': bge_temporal_analysis,
            'field_interference_matrix': interference_matrix,
            'collective_breathing_rhythm': collective_rhythm,
            'temporal_field_ready': True,
            
            # Coupling preparation for future integration inc in TODO
            'coupling_readiness': {
                'emotional_coupling_prepared': True,
                'manifold_coupling_prepared': True,
                'field_interference_prepared': True
            }
        }
    
    def _extract_bge_temporal_signatures(self, embedding_data: List[Dict]) -> Dict[str, Any]:
        """
        Extract natural temporal signatures from BGE's mathematical structure.
        
        INNOVATION: Uses BGE's existing heat kernel field evolution, spatial spectral
        analysis, and trajectory persistence as natural temporal patterns rather than
        imposing external temporal structure.
        """
        if not self.from_base or not self.bge_model:
            raise ValueError("BGE temporal signature extraction requires from_base=True and BGE model")
        
        # Extract individual embeddings from BGE search results
        individual_embeddings = []
        for item in embedding_data:
            if 'embeddings' in item and isinstance(item['embeddings'], list):
                # This is a BGE search result with embedded list
                individual_embeddings.extend(item['embeddings'])
            else:
                # This is already an individual embedding dictionary
                individual_embeddings.append(item)
        
        # Extract embeddings for temporal analysis
        embeddings = [np.array(emb['vector']) for emb in individual_embeddings]
        embeddings_array = np.array(embeddings)
        

        logger.info("🔍 Extracting BGE temporal signatures through REAL temporal analysis")
        
        # Extract actual tokens from embedding data (no defaults)
        actual_tokens = []
        for emb in individual_embeddings:
            if 'token' in emb:
                actual_tokens.append(emb['token'])
            else:
                raise ValueError("Embedding data missing 'token' field. Check BGE search_embeddings output structure.")
        
        temporal_analysis = self.bge_model.extract_temporal_field_analysis(
            embeddings_array,
            sample_tokens=actual_tokens,
            num_samples=len(embeddings),
            return_full_details=False
        )
        temporal_signatures = temporal_analysis['temporal_field_parameters']
        
        logger.info("✅ Extracted BGE temporal signatures from natural field dynamics")
        return temporal_signatures
    
    def _generate_temporal_biography(self, 
                                   embedding: Dict, 
                                   bge_temporal_analysis: Dict,
                                   observational_index: int) -> TemporalBiography:
        """
        Generate the complete temporal biography for a single conceptual charge.
        
        WEATHER MAP PATTERN: Creates unified field-movement entity that encodes both
        trajectory movement and field generation in a single breathing pattern.
        """
        vector = np.array(embedding['vector'])
        token = embedding.get('token', '<UNK>')
        manifold_props = embedding.get('manifold_properties')
        
        logger.debug(f"🧬 Generating temporal biography for token '{token}'")
        
        # STEP 1: Generate trajectory operators T_i(τ,C,s) from BGE signatures
        trajectory_operators = self.trajectory_integrator.compute_trajectory_operators(
            vector, 
            token,
            bge_temporal_analysis['breathing_rhythm_spectrum'],
            observational_state=float(observational_index)
        )
        
        # STEP 2: Generate dual-decay observational persistence Ψ_persistence(s-s₀)
        vivid_layer, character_layer = self.temporal_persistence_engine.generate_persistence_layers(
            vector,
            bge_temporal_analysis['trajectory_persistence_patterns'],
            observational_index
        )
        
        # STEP 3: Extract breathing rhythm from BGE natural frequencies
        frequency_evolution, phase_coordination = self.breathing_rhythm_extractor.extract_breathing_pattern(
            vector,
            bge_temporal_analysis['natural_frequency_patterns'],
            manifold_props
        )
        
        # STEP 4: Compute unified field-movement metrics
        temporal_momentum = self._compute_temporal_momentum(trajectory_operators, frequency_evolution)
        breathing_coherence = self._compute_breathing_coherence(phase_coordination)
        field_interference_signature = self._compute_field_interference_signature(
            trajectory_operators, phase_coordination
        )
        
        return TemporalBiography(
            trajectory_operators=trajectory_operators,
            vivid_layer=vivid_layer,
            character_layer=character_layer,
            frequency_evolution=frequency_evolution,
            phase_coordination=phase_coordination,
            temporal_momentum=temporal_momentum,
            breathing_coherence=breathing_coherence,
            field_interference_signature=field_interference_signature,
            bge_temporal_signature=bge_temporal_analysis
        )
    
    def _compute_field_interference_matrix(self, temporal_biographies: List[TemporalBiography]) -> np.ndarray:
        """
        Compute how temporal biographies interfere with each other.
        
        WEATHER MAP INTERFERENCE: Like clouds affecting each other's movement patterns,
        conceptual charges create interference patterns in their temporal signatures.
        """
        n_charges = len(temporal_biographies)
        interference_matrix = np.zeros((n_charges, n_charges), dtype=complex)
        
        for i in range(n_charges):
            for j in range(n_charges):
                if i != j:
                    # Compute interference between charge i and charge j
                    bio_i = temporal_biographies[i]
                    bio_j = temporal_biographies[j]
                    
                    # Phase interference
                    phase_interference = np.mean(
                        np.exp(1j * (bio_i.phase_coordination - bio_j.phase_coordination))
                    )
                    
                    # Trajectory interference
                    trajectory_interference = np.mean(
                        bio_i.trajectory_operators * np.conj(bio_j.trajectory_operators)
                    )
                    
                    # Combined interference pattern
                    interference_matrix[i, j] = phase_interference * trajectory_interference
        
        return interference_matrix
    
    def _compute_collective_breathing_rhythm(self, temporal_biographies: List[TemporalBiography]) -> Dict[str, Any]:
        """
        Compute the collective breathing rhythm emerging from all temporal biographies.
        
        COLLECTIVE FIELD EFFECT: The unified breathing pattern that emerges when
        multiple conceptual charges interact through field interference.
        """
        # Combine all frequency evolutions
        all_frequencies = np.array([bio.frequency_evolution for bio in temporal_biographies])
        collective_frequency = np.mean(all_frequencies, axis=0)
        
        # Combine all phase coordinations
        all_phases = np.array([bio.phase_coordination for bio in temporal_biographies])
        collective_phase = np.angle(np.mean(np.exp(1j * all_phases), axis=0))
        
        # Compute collective coherence
        collective_coherence = np.mean([bio.breathing_coherence for bio in temporal_biographies])
        
        # Compute collective momentum
        all_momenta = [bio.temporal_momentum for bio in temporal_biographies]
        collective_momentum = np.mean(all_momenta)
        
        return {
            'collective_frequency': collective_frequency,
            'collective_phase': collective_phase,
            'collective_coherence': collective_coherence,
            'collective_momentum': collective_momentum,
            'synchronization_strength': float(np.abs(collective_momentum)),
            'breathing_pattern_diversity': float(np.std(all_frequencies.flatten()))
        }
    
    def _compute_temporal_momentum(self, trajectory_operators: np.ndarray, frequency_evolution: np.ndarray) -> complex:
        """Compute unified temporal momentum from trajectory and frequency with real field dynamics."""
        # Filter out zero trajectory operators for realistic momentum
        non_zero_operators = trajectory_operators[np.abs(trajectory_operators) > 1e-12]
        
        if len(non_zero_operators) > 0:
            # Use only active trajectory components for momentum
            magnitude = np.mean(np.abs(non_zero_operators))
            
            # Compute weighted phase from active frequency components
            active_frequencies = frequency_evolution[np.abs(trajectory_operators) > 1e-12]
            if len(active_frequencies) > 0:
                # Weight frequencies by trajectory strength for realistic phase
                weights = np.abs(non_zero_operators) / np.sum(np.abs(non_zero_operators))
                weighted_frequency = np.sum(active_frequencies * weights)
                phase = np.angle(weighted_frequency)
            else:
                # Fallback to mean frequency phase
                phase = np.angle(np.mean(frequency_evolution))
        else:
            # All operators are zero - minimal momentum
            magnitude = 1e-8
            phase = np.angle(np.mean(frequency_evolution)) if len(frequency_evolution) > 0 else 0.0
        
        # Add small random variation to prevent artificial patterns
        # Use trajectory operator statistics for deterministic but varying results
        variation_seed = np.sum(np.real(trajectory_operators)) % 1.0
        magnitude_variation = 1.0 + 0.1 * np.sin(2 * np.pi * variation_seed)
        phase_variation = 0.1 * np.cos(2 * np.pi * variation_seed)
        
        final_magnitude = magnitude * magnitude_variation
        final_phase = phase + phase_variation
        
        momentum = final_magnitude * np.exp(1j * final_phase)
        
        logger.debug(f"🌀 Temporal momentum - active ops: {len(non_zero_operators)}/{len(trajectory_operators)}, magnitude: {final_magnitude:.6f}, phase: {final_phase:.3f}, momentum: {momentum}")
        
        return momentum
    
    def _compute_breathing_coherence(self, phase_coordination: np.ndarray) -> float:
        """Compute how synchronized the breathing pattern is."""
        # Coherence measured as phase synchronization
        return float(np.abs(np.mean(np.exp(1j * phase_coordination))))
    
    def _compute_field_interference_signature(self, 
                                            trajectory_operators: np.ndarray, 
                                            phase_coordination: np.ndarray) -> np.ndarray:
        """Compute signature for how this charge interferes with others."""
        # Combine trajectory strength with phase patterns
        interference_signature = trajectory_operators * np.exp(1j * phase_coordination)
        return interference_signature


class TrajectoryIntegrator:
    """
    Computes trajectory operators T_i(τ,C,s) = ∫₀ˢ ω_i(τ,s')·e^(iφ_i(τ,s')) ds'
    
    UNIFIED APPROACH: Trajectory operators ARE the breathing pattern of field generation.
    Not separate from field effects - they ARE the field's movement signature.
    """
    
    def compute_trajectory_operators(self, 
                                   vector: np.ndarray,
                                   token: str,
                                   breathing_spectrum: List[float],
                                   observational_state: float) -> np.ndarray:
        """Compute trajectory operators from BGE breathing spectrum with underflow protection."""
        embedding_dim = len(vector)
        trajectory_operators = np.zeros(embedding_dim, dtype=complex)
        
        # Use BGE eigenfrequency spectrum as natural breathing rhythm
        base_frequencies = np.array(breathing_spectrum)
        if len(base_frequencies) < embedding_dim:
            # Extend spectrum using harmonic relationships
            base_frequencies = np.resize(base_frequencies, embedding_dim)
        
        # 🔧 FIX: Add minimum frequency to prevent zero trajectory operators
        base_frequencies = np.maximum(base_frequencies, 1e-6)  # Minimum frequency
        
        # Check for problematic spectrum
        if np.max(np.abs(base_frequencies)) < 1e-10:
            logger.warning(f"⚠️  Breathing spectrum too small for token '{token}', using fallback frequencies")
            # Create fallback spectrum based on vector properties
            base_frequencies = 0.1 * (1.0 + 0.5 * np.abs(vector[:len(base_frequencies)]))
        
        # Normalize observational_state to prevent underflow
        normalized_obs_state = max(0.1, min(10.0, observational_state + 1.0))  # Clamp to [0.1, 10]
        
        for i in range(embedding_dim):
            # Component-specific frequency modulated by embedding strength
            vector_modulation = 1 + 0.1 * vector[i]
            frequency = base_frequencies[i] * vector_modulation
            
            # Ensure minimum frequency to prevent zero operators
            frequency = max(abs(frequency), 1e-8) * np.sign(frequency) if frequency != 0 else 1e-8
            
            # Token-specific phase modulation with embedding component influence
            token_hash = hash(token) % 1000 / 1000.0
            phase = 2 * np.pi * token_hash * i / embedding_dim
            
            # Add vector-dependent phase shift for uniqueness
            vector_phase = 0.1 * vector[i] * (i + 1) / embedding_dim
            total_phase = phase + vector_phase
            
            # Trajectory integration with improved formula
            # T_i = ∫₀ˢ ω_i(s')·e^(iφ_i(s')) ds' ≈ ω_i * s * e^(iφ_i)
            trajectory_operators[i] = frequency * normalized_obs_state * np.exp(1j * total_phase)
        
        # Final validation
        if np.all(np.abs(trajectory_operators) < 1e-12):
            logger.warning(f"⚠️  All trajectory operators near zero for token '{token}', applying emergency boost")
            # Emergency fallback: use vector-based operators
            for i in range(embedding_dim):
                emergency_freq = 0.01 * (1.0 + abs(vector[i]))
                emergency_phase = np.pi * vector[i] * i / embedding_dim
                trajectory_operators[i] = emergency_freq * np.exp(1j * emergency_phase)
        
        logger.debug(f"🕰️ Trajectory operators for '{token}' - range: [{np.min(np.abs(trajectory_operators)):.2e}, {np.max(np.abs(trajectory_operators)):.2e}], non-zero: {np.sum(np.abs(trajectory_operators) > 1e-12)}/{embedding_dim}")
        
        return trajectory_operators


class BreathingRhythmExtractor:
    """
    Extracts natural breathing rhythms from BGE's frequency patterns.
    
    BREATHING PHILOSOPHY: The temporal dimension IS the breathing pattern of
    conceptual charge formation - not separate temporal coordinates.
    """
    
    def extract_breathing_pattern(self, 
                                vector: np.ndarray,
                                frequency_patterns: Dict[str, Any],
                                manifold_props: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract frequency evolution and phase coordination from BGE patterns."""
        embedding_dim = len(vector)
        
        # Extract natural frequencies from BGE spectral analysis
        base_entropy = frequency_patterns.get('mean_spectral_entropy')
        phase_variance = frequency_patterns.get('phase_variance_distribution',)
        
        # Ensure proper dimensionality
        if len(phase_variance) < embedding_dim:
            phase_variance = (phase_variance * (embedding_dim // len(phase_variance) + 1))[:embedding_dim]
        
        # Generate frequency evolution ω_i(τ,s')
        frequency_evolution = np.zeros(embedding_dim, dtype=complex)
        for i in range(embedding_dim):
            # Base frequency from spectral entropy
            base_freq = base_entropy * (i + 1) / embedding_dim
            
            # Modulation from embedding component
            freq_modulation = 1 + 0.1 * vector[i]
            
            # Complex frequency with phase coupling
            frequency_evolution[i] = base_freq * freq_modulation * (1 + 0.1j)
        
        # Generate phase coordination φ_i(τ,s')
        phase_coordination = np.zeros(embedding_dim)
        for i in range(embedding_dim):
            # Base phase from variance pattern
            base_phase = phase_variance[i] * 2 * np.pi
            
            # Manifold-influenced phase shifts
            curvature_influence = 0.0
            if manifold_props and 'local_curvature' in manifold_props:
                curvature_influence = 0.1 * manifold_props['local_curvature']
            
            phase_coordination[i] = base_phase + curvature_influence
        
        return frequency_evolution, phase_coordination


class TemporalPersistenceEngine:
    """
    Generates dual-decay observational persistence Ψ_persistence(s-s₀).
    
    LAYERED MEMORY: Implements vivid recent memory (Gaussian decay) and 
    persistent character themes (exponential-cosine decay) from Section 3.1.4.3.3.
    """
    
    def generate_persistence_layers(self, 
                                  vector: np.ndarray,
                                  persistence_patterns: Dict[str, Any],
                                  observational_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate dual-decay persistence layers with stability across indices."""
        embedding_dim = len(vector)
        
        # Extract persistence parameters from BGE coupling analysis
        trajectory_persistence = persistence_patterns.get('temporal_persistence', 0.1)
        coupling_strength = persistence_patterns.get('coupling_strength_distribution', [0.1] * embedding_dim)
        
        # Ensure proper dimensionality
        if len(coupling_strength) < embedding_dim:
            coupling_strength = (coupling_strength * (embedding_dim // len(coupling_strength) + 1))[:embedding_dim]
        
        # 🔧 FIX: Use normalized observational distance to prevent underflow
        # Instead of raw observational_index, use a normalized version
        normalized_s = observational_index / max(10.0, embedding_dim / 100.0)  # Normalize to prevent extreme decay
        
        # Vivid layer: Modified Gaussian decay with underflow protection
        vivid_layer = np.zeros(embedding_dim)
        for i in range(embedding_dim):
            # Adaptive sigma with minimum bound
            sigma = max(0.8, 0.5 + 0.3 * abs(vector[i]))  # Increased minimum sigma
            gaussian_decay = np.exp(-(normalized_s**2) / (2 * sigma**2))
            
            # Apply underflow protection and add base persistence
            vivid_layer[i] = max(0.01, gaussian_decay + 0.05 * abs(vector[i]))  # Minimum 0.01 + vector influence
        
        # Character layer: Modified exponential-cosine decay with underflow protection
        character_layer = np.zeros(embedding_dim)
        for i in range(embedding_dim):
            # Reduced decay rates to prevent underflow
            alpha = max(0.1, coupling_strength[i] * trajectory_persistence)
            lambda_decay = max(0.01, 0.05 + 0.02 * abs(vector[i]))  # Reduced decay rate
            beta_freq = 0.5 + 0.2 * vector[i]
            
            exp_decay = alpha * np.exp(-lambda_decay * normalized_s)
            cos_modulation = np.cos(beta_freq * normalized_s)
            
            # Apply underflow protection and vector-based persistence
            base_persistence = 0.005 * (1.0 + abs(vector[i]))  # Vector-dependent base
            character_layer[i] = max(base_persistence, exp_decay * cos_modulation)
        
        # Final underflow protection
        vivid_layer = np.maximum(vivid_layer, 1e-15)
        character_layer = np.maximum(character_layer, 1e-15)
        
        logger.debug(f"🧬 Persistence layers - obs_idx: {observational_index}, vivid range: [{np.min(vivid_layer):.6f}, {np.max(vivid_layer):.6f}], char range: [{np.min(character_layer):.6f}, {np.max(character_layer):.6f}]")
        
        return vivid_layer, character_layer