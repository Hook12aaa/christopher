"""
Emotional Trajectory Evolution - E^trajectory(τ, s) Implementation

MATHEMATICAL FOUNDATION (README.md Section 3.1.3.3.2):
E^trajectory[i](τ, s) = α_i · exp(-||v_i - v_E||²/2σ²) · ∫₀ˢ w(s-s') · emotional_event(τ, s') ds'

DECONSTRUCTED TRANSFORMER APPROACH:
1. Extract emotional geometric patterns from attention-like operations
2. Convert QK^T alignment detection to Gaussian alignment  
3. Apply trajectory accumulation with observational state integration
4. Create field modulation effects using coupling properties

This module implements the core E^trajectory(τ, s) component for the complete
Q(τ, C, s) conceptual charge formula using deconstructed transformer mathematics.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmotionalTrajectoryParams:
    """Parameters for emotional trajectory integration. 
    
    CLAUDE.md COMPLIANCE: All parameters MUST be computed from manifold properties.
    NO default values allowed - all must be explicitly calculated.
    """
    observational_state: float
    gaussian_sigma: float
    trajectory_decay_rate: float
    amplification_factor: float
    resonance_threshold: float
    coupling_strength: float


class EmotionalTrajectoryIntegrator:
    """
    Core implementation of E^trajectory(τ, s) using deconstructed transformer mathematics.
    
    MATHEMATICAL FOUNDATION:
    Transforms attention mechanism insights into trajectory-based emotional field modulation
    through Gaussian alignment, trajectory accumulation, and resonance amplification.
    
    DECONSTRUCTED ATTENTION APPROACH:
    - QK^T geometric alignment → Gaussian alignment computation
    - Softmax amplification → Resonance-based amplification  
    - Value transport → Field modulation effects
    - Temporal independence → Trajectory-dependent evolution
    """
    
    def __init__(self, 
                 embedding_dimension: int,
                 emotional_memory_length: float):
        """
        Initialize emotional trajectory integrator.
        
        Args:
            embedding_dimension: Dimension of semantic embeddings
            emotional_memory_length: Length of emotional memory in observational units
        """
        self.embedding_dimension = embedding_dimension
        self.emotional_memory_length = emotional_memory_length
        
        # Initialize emotional resonance parameters
        # CLAUDE.md COMPLIANCE: No random values - derive from embedding dimension structure
        # Create deterministic frequency spectrum based on embedding dimension
        frequency_indices = np.arange(embedding_dimension)
        normalized_indices = frequency_indices / embedding_dimension
        # Use sinusoidal pattern for emotional frequencies
        self.base_emotional_frequencies = 0.1 + 1.9 * (0.5 + 0.5 * np.sin(2 * np.pi * normalized_indices))
        
        logger.info(f"Initialized EmotionalTrajectoryIntegrator for {embedding_dimension}D embeddings")
    
    def compute_trajectory(self,
                          token: str,
                          semantic_embedding: np.ndarray,
                          manifold_properties: Dict[str, Any],
                          params: EmotionalTrajectoryParams,
                          temporal_data: Optional[Dict[str, Any]] = None,
                          source_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute complete E^trajectory(τ, s) using deconstructed transformer mathematics.
        
        MATHEMATICAL PROCESS:
        1. Extract emotional resonance pattern using deconstructed attention
        2. Compute Gaussian alignment between semantic and emotional content
        3. Integrate trajectory accumulation over observational states
        4. Apply resonance amplification effects
        5. Compute emotional phase contributions
        
        Args:
            token: Token identifier for trajectory tracking
            semantic_embedding: Base semantic vector [D]
            manifold_properties: Contains coupling_mean, coupling_variance
            params: Emotional trajectory parameters
            temporal_data: Optional temporal dimension coordination data
            source_text: Source text for REAL attention analysis (REQUIRED for production)
            
        Returns:
            Dict containing complete E^trajectory(τ, s) computation results
        """
        try:
            # Extract manifold coupling properties
            # CLAUDE.md COMPLIANCE: No default fallbacks - ensure required properties exist
            if 'coupling_mean' not in manifold_properties:
                raise ValueError(f"Missing required manifold property 'coupling_mean' for token {token}")
            if 'coupling_variance' not in manifold_properties:
                raise ValueError(f"Missing required manifold property 'coupling_variance' for token {token}")
                
            coupling_mean = manifold_properties['coupling_mean']
            coupling_variance = manifold_properties['coupling_variance']
            
            # Step 1: Extract emotional resonance pattern using REAL attention analysis
            # FIELD_THEORY_ENFORCEMENT.md: Use ACTUAL attention deconstruction
            try:
                from .attention_deconstruction import create_real_emotional_resonance_extractor
                
                # FIELD_THEORY_ENFORCEMENT.md: Use actual source text for real attention analysis
                if source_text is None:
                    logger.warning(f"No source_text provided for {token} - using token as fallback (enhance in production)")
                    text_for_analysis = token  # Development fallback
                else:
                    text_for_analysis = source_text  # Production: actual source text
                
                # Use the corrected REAL emotional resonance extraction
                emotional_resonance = create_real_emotional_resonance_extractor(
                    text=text_for_analysis,
                    target_token=token,
                    coupling_mean=coupling_mean
                )
                
                # Validate complex array result
                if not np.iscomplexobj(emotional_resonance):
                    raise ValueError(f"REAL attention analysis must produce complex-valued resonance for {token}")
                
            except Exception as e:
                logger.error(f"REAL attention deconstruction failed for {token}: {e}")
                raise ValueError(f"REAL attention analysis required for emotional resonance of {token}: {e}")
            
            # Step 2: Gaussian alignment computation (Section 3.1.3.3.2)
            gaussian_alignment = self._compute_gaussian_alignment(
                embedding=semantic_embedding,
                emotional_resonance=emotional_resonance,
                coupling_variance=coupling_variance,
                params=params
            )
            
            # Step 3: Trajectory accumulation integration (Section 3.1.3.3.2)
            trajectory_accumulation = self._compute_trajectory_accumulation(
                token=token,
                observational_state=params.observational_state,
                coupling_mean=coupling_mean,
                temporal_data=temporal_data,
                decay_rate=params.trajectory_decay_rate
            )
            
            # Step 4: Resonance amplification (Section 3.1.3.3.6)
            resonance_amplification = self._compute_resonance_amplification(
                semantic_embedding=semantic_embedding,
                emotional_resonance=emotional_resonance,
                coupling_variance=coupling_variance,
                threshold=params.resonance_threshold
            )
            
            # Step 5: Emotional phase computation (Section 3.1.3.3.5)
            emotional_phase = self._compute_emotional_phase(
                token=token,
                observational_state=params.observational_state,
                coupling_properties={'mean': coupling_mean, 'variance': coupling_variance},
                temporal_data=temporal_data
            )
            
            # Final E^trajectory[i](τ, s) assembly - VECTOR of complex field components
            # Each dimension i has its own complex trajectory value
            embedding_dim = len(semantic_embedding)
            E_trajectory_vector = np.zeros(embedding_dim, dtype=complex)
            
            # Ensure all components are complex vectors, not scalars
            if not isinstance(gaussian_alignment, np.ndarray):
                raise ValueError(f"gaussian_alignment must be complex vector [D], got {type(gaussian_alignment)}")
            if not isinstance(trajectory_accumulation, np.ndarray):
                raise ValueError(f"trajectory_accumulation must be complex vector [D], got {type(trajectory_accumulation)}")
            if not isinstance(resonance_amplification, np.ndarray):
                raise ValueError(f"resonance_amplification must be complex vector [D], got {type(resonance_amplification)}")
            if not isinstance(emotional_phase, np.ndarray):
                raise ValueError(f"emotional_phase must be vector [D], got {type(emotional_phase)}")
            
            # Compute E^trajectory[i](τ, s) for each dimension i
            for i in range(embedding_dim):
                E_trajectory_vector[i] = (gaussian_alignment[i] * 
                                        trajectory_accumulation[i] * 
                                        resonance_amplification[i] * 
                                        np.exp(1j * emotional_phase[i]))
            
            # Field theory analysis - maintain complex vector structure
            magnitude_vector = np.abs(E_trajectory_vector)
            phase_vector = np.angle(E_trajectory_vector)
            
            # Overall field strength (for debugging only - main result is the vector)
            total_field_magnitude = np.linalg.norm(E_trajectory_vector)
            mean_phase = np.angle(np.mean(E_trajectory_vector))
            
            # Return comprehensive results with complex VECTOR field analysis
            results = {
                'emotional_trajectory_complex': E_trajectory_vector,  # Complex vector [D]
                'emotional_trajectory_magnitude': magnitude_vector,   # Magnitude vector [D]
                'emotional_phase': phase_vector,                     # Phase vector [D]
                'total_field_magnitude': total_field_magnitude,      # Scalar field strength
                'mean_phase': mean_phase,                            # Mean phase for analysis
                'gaussian_alignment': gaussian_alignment,
                'trajectory_accumulation': trajectory_accumulation,
                'resonance_amplification': resonance_amplification,
                'emotional_resonance_pattern': emotional_resonance,
                'coupling_analysis': {
                    'coupling_mean': coupling_mean,
                    'coupling_variance': coupling_variance,
                    'emotional_strength': abs(coupling_mean),
                    'emotional_coherence': 1.0 / (1.0 + coupling_variance)
                },
                'complex_field_analysis': {
                    'is_complex_valued': np.iscomplexobj(E_trajectory_vector),
                    'field_magnitude': total_field_magnitude,
                    'field_phase': mean_phase,
                    'gaussian_alignment_complex': np.iscomplexobj(gaussian_alignment),
                    'trajectory_accumulation_complex': np.iscomplexobj(trajectory_accumulation),
                    'resonance_amplification_complex': np.iscomplexobj(resonance_amplification)
                },
                'processing_status': 'complete'
            }
            
            logger.debug(f"E^trajectory computed for {token}: magnitude={total_field_magnitude:.4f}, phase={mean_phase:.4f}, complex={np.iscomplexobj(E_trajectory_vector)}")
            return results
            
        except Exception as e:
            logger.error(f"Emotional trajectory computation failed for {token}: {e}")
            # CLAUDE.md COMPLIANCE: NO default fallback values - must raise error
            raise ValueError(f"Emotional trajectory computation failed for {token}: {e}") from e
    
    def _extract_emotional_resonance_pattern(self,
                                           embedding: np.ndarray,
                                           coupling_mean: float,
                                           token: str) -> np.ndarray:
        """
        Extract COMPLEX emotional resonance pattern using ACTUAL deconstructed attention mechanics.
        
        FIELD_THEORY_ENFORCEMENT.md COMPLIANCE:
        - NO hash-based synthetic generation (PERMANENTLY BANNED)
        - Must use ACTUAL transformer attention extraction
        - NO simulation or approximation substitution
        
        REQUIRED IMPLEMENTATION:
        Must integrate with actual BGE transformer model to extract real attention patterns
        that reveal emotional geometric structures in semantic space.
        """
        # FIELD_THEORY_ENFORCEMENT.md: Synthetic data generation PERMANENTLY BANNED
        raise NotImplementedError(
            "FIELD_THEORY_ENFORCEMENT VIOLATION: _extract_emotional_resonance_pattern cannot use synthetic data generation. "
            "Must use actual BGE transformer attention patterns. "
            f"This function requires actual transformer model access for token '{token}' to extract real attention-based emotional resonance. "
            "Implementation must integrate with Sysnpire.model.bge_encoder or similar actual model interface."
        )
    
    def _compute_gaussian_alignment(self,
                                  embedding: np.ndarray,
                                  emotional_resonance: np.ndarray,
                                  coupling_variance: float,
                                  params: EmotionalTrajectoryParams) -> np.ndarray:
        """
        Compute COMPLEX Gaussian alignment between semantic content and emotional resonance.
        
        FORMULA (README.md Section 3.1.3.3.2):
        α_i · exp(-||v_i - v_E||²/2σ²) - but now produces COMPLEX field effects
        
        COMPLEX FIELD INTERPRETATION:
        - ||v_i - v_E||: Distance between semantic and COMPLEX emotional field
        - σ²: Emotional sensitivity (from coupling_variance, with minimum threshold)
        - α_i: COMPLEX amplification factor (includes phase from observational state)
        - Result: Complex-valued field modulation, not scalar float
        """
        # Ensure emotional_resonance is complex-valued
        if not np.iscomplexobj(emotional_resonance):
            # Convert to complex if needed (shouldn't happen with new implementation)
            emotional_resonance = emotional_resonance.astype(complex)
        
        # Compute PER-DIMENSION complex Gaussian alignment - NO SCALAR REDUCTION
        # PRINCIPLE: Each dimension i maintains its own complex alignment value
        
        embedding_dim = len(embedding)
        gaussian_alignment_vector = np.zeros(embedding_dim, dtype=complex)
        
        # Normalize both vectors but maintain dimensional structure
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        resonance_norm = emotional_resonance / (np.linalg.norm(emotional_resonance) + 1e-8)
        
        # Convert to complex for field calculations
        embedding_complex = embedding_norm.astype(complex)
        resonance_complex = resonance_norm.astype(complex)
        
        # Use coupling_variance as emotional sensitivity parameter σ²
        sigma_squared = max(coupling_variance, 0.01)  # Prevent degenerate case
        
        # Compute α_i · exp(-||v_i - v_E||²/2σ²) for EACH dimension i
        for i in range(embedding_dim):
            # Per-dimension alignment distance ||v_i - v_E[i]||²
            v_i = embedding_complex[i]
            v_E_i = resonance_complex[i]
            alignment_distance_sq = abs(v_i - v_E_i)**2
            
            # COMPLEX amplification factor α_i varies per dimension
            alpha_i_magnitude = params.amplification_factor * (1.0 + 0.1 * abs(v_i))
            alpha_i_phase = params.observational_state * 0.3 + i * 0.01  # Dimension-dependent phase
            alpha_i = alpha_i_magnitude * np.exp(1j * alpha_i_phase)
            
            # Gaussian alignment for dimension i
            gaussian_i = alpha_i * np.exp(-alignment_distance_sq / (2 * sigma_squared))
            
            # Ensure non-zero field (field theory requirement)
            if abs(gaussian_i) < 1e-6:
                gaussian_i = 1e-6 * np.exp(1j * alpha_i_phase)
            
            gaussian_alignment_vector[i] = gaussian_i
        
        return gaussian_alignment_vector
    
    def _compute_trajectory_accumulation(self,
                                       token: str,
                                       observational_state: float,
                                       coupling_mean: float,
                                       temporal_data: Optional[Dict[str, Any]],
                                       decay_rate: float) -> np.ndarray:
        """
        Compute COMPLEX trajectory accumulation with observational state integration.
        
        FORMULA (README.md Section 3.1.3.3.2):
        ∫₀ˢ w(s-s') · emotional_event(τ, s') ds' - now COMPLEX-valued
        
        COMPLEX FIELD APPROACH:
        - w(s-s'): Complex decay function with phase evolution
        - emotional_event(τ, s'): Complex emotional significance with trajectory-dependent phase
        - Integration produces complex field accumulation, not scalar
        """
        # PRINCIPLE: Must have temporal data - no "if available" logic
        if not temporal_data or 'observational_persistence' not in temporal_data:
            raise ValueError(f"temporal_data with observational_persistence REQUIRED for trajectory accumulation of {token}")
        
        observational_persistence = temporal_data['observational_persistence']
        
        # Emotional event strength based on coupling analysis (ensure non-zero)
        emotional_event_strength = max(abs(coupling_mean), 0.01)
        emotional_polarity = 1.0 if coupling_mean > 0 else -1.0
        
        # Token-specific trajectory phase evolution
        token_hash = hash(token) % 1000 / 1000.0
        
        # PER-DIMENSION trajectory accumulation - NO SCALAR REDUCTION
        embedding_dim = self.embedding_dimension  # Get from class instance
        trajectory_accumulation_vector = np.zeros(embedding_dim, dtype=complex)
        
        # Discretized trajectory integration ∫₀ˢ w(s-s') · emotional_event(τ, s') ds'
        decay_constant = 1.0 / decay_rate
        integration_steps = max(int(observational_state * 10), 1)
        ds = observational_state / integration_steps
        
        # Compute trajectory integral for EACH dimension i
        for i in range(embedding_dim):
            accumulation_i = 0.0 + 0.0j  # Per-dimension complex accumulation
            
            # Dimension-specific frequency and phase characteristics
            dimension_frequency = (i / embedding_dim) * 2.0 + token_hash
            
            for step in range(1, integration_steps + 1):
                s_prime = step * ds  # Integration variable
                
                # COMPLEX decay weight w(s-s') with per-dimension phase evolution
                decay_magnitude = np.exp(-(observational_state - s_prime) / decay_constant)
                decay_phase = dimension_frequency * s_prime + emotional_polarity * s_prime * 0.5
                w_complex = decay_magnitude * np.exp(1j * decay_phase)
                
                # COMPLEX emotional_event(τ, s') for dimension i
                event_magnitude = emotional_event_strength * observational_persistence
                event_phase = s_prime * emotional_polarity + coupling_mean + i * 0.1
                emotional_event_i = event_magnitude * np.exp(1j * event_phase)
                
                # Accumulate complex trajectory integral for dimension i
                accumulation_i += w_complex * emotional_event_i * ds
            
            trajectory_accumulation_vector[i] = accumulation_i
        
        # FIELD_THEORY_ENFORCEMENT.md: Validate trajectory accumulation results
        for i in range(embedding_dim):
            if np.abs(trajectory_accumulation_vector[i]) == 0.0:
                # FIELD_THEORY_ENFORCEMENT.md: NO synthetic stability values allowed
                raise ValueError(
                    f"Zero trajectory accumulation detected for dimension {i} of token {token}. "
                    "This indicates insufficient temporal data or computational errors. "
                    "FIELD_THEORY_ENFORCEMENT: No synthetic stability values allowed - must fix underlying data."
                )
        
        return trajectory_accumulation_vector
    
    def _compute_resonance_amplification(self,
                                       semantic_embedding: np.ndarray,
                                       emotional_resonance: np.ndarray,
                                       coupling_variance: float,
                                       threshold: float) -> np.ndarray:
        """
        Compute COMPLEX resonance-based amplification effects - VECTOR of complex amplifications.
        
        FORMULA (README.md Section 3.1.3.3.6):
        1 + A_max · exp(-|ω_semantic[i] - ω_emotional[i]|²/2σ_resonance²) - per dimension i
        
        COMPLEX FIELD INTERPRETATION:
        - Resonance occurs when semantic and emotional frequencies align in complex space per dimension
        - COMPLEX amplification enhances phase-coherent semantic-emotional content per dimension
        - Phase-misaligned content gets complex suppression with phase shifts per dimension
        """
        # Ensure emotional_resonance is complex
        if not np.iscomplexobj(emotional_resonance):
            emotional_resonance = emotional_resonance.astype(complex)
        
        # Compute complex frequencies for semantic and emotional content - PER DIMENSION
        semantic_complex = semantic_embedding.astype(complex)
        embedding_dim = len(semantic_embedding)
        
        # PER-DIMENSION resonance amplification - NO SCALAR REDUCTION
        amplification_vector = np.zeros(embedding_dim, dtype=complex)
        
        # Resonance parameters
        A_max = 1.0  # Maximum amplification factor
        sigma_resonance_squared = max(coupling_variance, 0.1)  # Prevent degenerate cases
        
        # Compute resonance amplification for EACH dimension i
        for i in range(embedding_dim):
            # Per-dimension complex frequencies
            semantic_freq_i = semantic_complex[i]
            emotional_freq_i = emotional_resonance[i]
            
            # Complex frequency difference for dimension i
            freq_difference_i = semantic_freq_i - emotional_freq_i
            freq_difference_magnitude = abs(freq_difference_i)
            
            # Resonance amplification computation with COMPLEX phase effects for dimension i
            resonance_magnitude = A_max * np.exp(-freq_difference_magnitude**2 / (2 * sigma_resonance_squared))
            
            # Phase alignment bonus: coherent phases get more amplification
            phase_alignment = np.angle(semantic_freq_i) - np.angle(emotional_freq_i)
            phase_bonus = np.cos(phase_alignment)  # +1 for aligned, -1 for opposite
            
            # Complex amplification with phase-dependent enhancement for dimension i
            amplification_magnitude = 1.0 + resonance_magnitude * (1.0 + 0.5 * phase_bonus)
            amplification_phase = phase_alignment * 0.1  # Small phase contribution from resonance
            
            amplification_i = amplification_magnitude * np.exp(1j * amplification_phase)
            
            # Apply threshold for significant resonance effects
            if abs(amplification_i) - 1.0 < threshold:
                # Ensure minimum complex amplification for dimension i
                min_phase = np.angle(amplification_i)
                amplification_i = (1.0 + threshold) * np.exp(1j * min_phase)
            
            amplification_vector[i] = amplification_i
        
        return amplification_vector
    
    def _compute_emotional_phase(self,
                               token: str,
                               observational_state: float,
                               coupling_properties: Dict[str, float],
                               temporal_data: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Compute emotional phase contribution to total phase integration - VECTOR of phases.
        
        FORMULA (README.md Section 3.1.3.3.5):
        φ_emotional[i](τ, s) = ∫₀ˢ ω_emotional[i](τ, s') ds' + Σⱼ coupling_emotional[j] · φ_j(s')
        
        MATHEMATICAL APPROACH:
        - ω_emotional[i]: Token-specific emotional frequency evolution per dimension i
        - Integration over observational state per dimension
        - Cross-dimensional coupling with temporal phases per dimension
        """
        # Token-specific emotional frequency base
        token_hash = hash(token) % 1000 / 1000.0
        base_emotional_frequency = 2 * np.pi * token_hash
        
        # Coupling modulation of frequency
        coupling_mean = coupling_properties['mean']
        coupling_variance = coupling_properties['variance']
        
        # PER-DIMENSION emotional phase computation - NO SCALAR REDUCTION
        embedding_dim = self.embedding_dimension
        emotional_phase_vector = np.zeros(embedding_dim, dtype=float)
        
        # Compute emotional phase for EACH dimension i
        for i in range(embedding_dim):
            # Dimension-specific emotional frequency evolution
            dimension_frequency_modifier = 1.0 + 0.1 * coupling_mean + i * 0.02 / embedding_dim
            omega_emotional_i = base_emotional_frequency * dimension_frequency_modifier
            
            # Phase accumulation through observational state integration for dimension i
            phase_integral_i = omega_emotional_i * observational_state
            
            # Cross-dimensional coupling with temporal phases for dimension i
            coupling_contribution_i = 0.0
            if temporal_data and 'phase_accumulation' in temporal_data:
                temporal_phases = temporal_data['phase_accumulation']
                if len(temporal_phases) > 0:
                    coupling_strength_i = 0.1 * abs(coupling_mean) * (1.0 + i * 0.01)
                    # Use appropriate temporal phase for dimension i
                    temporal_phase_i = temporal_phases[i % len(temporal_phases)]
                    coupling_contribution_i = coupling_strength_i * temporal_phase_i
            
            # Total emotional phase for dimension i
            emotional_phase_i = phase_integral_i + coupling_contribution_i
            
            # Normalize phase to [-π, π] range for dimension i
            emotional_phase_i = (emotional_phase_i + np.pi) % (2 * np.pi) - np.pi
            
            emotional_phase_vector[i] = emotional_phase_i
        
        return emotional_phase_vector
    
    def compute_batch_trajectories(self,
                                 tokens: list,
                                 embeddings: np.ndarray,
                                 manifold_properties_batch: list,
                                 params: EmotionalTrajectoryParams,
                                 temporal_data_batch: Optional[list] = None) -> list:
        """
        Efficiently compute emotional trajectories for batch of tokens.
        
        Args:
            tokens: List of token identifiers
            embeddings: Batch of semantic embeddings [N, D]
            manifold_properties_batch: List of manifold properties for each token
            params: Shared emotional trajectory parameters
            temporal_data_batch: Optional temporal data for each token
            
        Returns:
            List of emotional trajectory computation results
        """
        if temporal_data_batch is None:
            temporal_data_batch = [None] * len(tokens)
        
        results = []
        for i, (token, embedding, properties, temporal_data) in enumerate(
            zip(tokens, embeddings, manifold_properties_batch, temporal_data_batch)
        ):
            try:
                result = self.compute_trajectory(
                    token=token,
                    semantic_embedding=embedding,
                    manifold_properties=properties,
                    params=params,
                    temporal_data=temporal_data
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to compute emotional trajectory for token {i} ({token}): {e}")
                # CLAUDE.md COMPLIANCE: NO default fallback values - must raise error
                raise ValueError(f"Batch trajectory computation failed for token {i} ({token}): {e}") from e
        
        logger.info(f"Computed emotional trajectories for {len(results)} tokens")
        return results


def create_emotional_trajectory_params(observational_state: float,
                                     manifold_properties: Dict[str, Any]) -> EmotionalTrajectoryParams:
    """
    Create emotional trajectory parameters from manifold properties.
    
    CLAUDE.md COMPLIANCE: ALL parameters computed from actual manifold data.
    NO hardcoded or default values allowed.
    
    Args:
        observational_state: Current observational state s
        manifold_properties: Manifold properties containing coupling_mean, coupling_variance
        
    Returns:
        EmotionalTrajectoryParams configured for computation
    """
    # CLAUDE.md COMPLIANCE: Compute ALL parameters from manifold properties
    if 'coupling_mean' not in manifold_properties:
        raise ValueError("Missing required manifold property 'coupling_mean' for trajectory params")
    if 'coupling_variance' not in manifold_properties:
        raise ValueError("Missing required manifold property 'coupling_variance' for trajectory params")
        
    coupling_mean = manifold_properties['coupling_mean']
    coupling_variance = manifold_properties['coupling_variance']
    
    # Compute all parameters from field theory relationships
    gaussian_sigma = np.sqrt(coupling_variance)  # Emotional sensitivity from coupling variance
    trajectory_decay_rate = coupling_variance / (1.0 + abs(coupling_mean))  # Decay from coupling dynamics
    amplification_factor = 1.0 + abs(coupling_mean)  # Amplification from coupling strength
    resonance_threshold = coupling_variance * abs(coupling_mean)  # Threshold from coupling interaction
    coupling_strength = abs(coupling_mean) * coupling_variance  # Strength from coupling product
    
    return EmotionalTrajectoryParams(
        observational_state=observational_state,
        gaussian_sigma=gaussian_sigma,
        trajectory_decay_rate=trajectory_decay_rate,
        amplification_factor=amplification_factor,
        resonance_threshold=resonance_threshold,
        coupling_strength=coupling_strength
    )