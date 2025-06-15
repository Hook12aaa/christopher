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
    """Parameters for emotional trajectory integration."""
    observational_state: float
    gaussian_sigma: float = 0.5
    trajectory_decay_rate: float = 0.1
    amplification_factor: float = 1.0
    resonance_threshold: float = 0.1
    coupling_strength: float = 0.2


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
                 embedding_dimension: int = 1024,
                 emotional_memory_length: float = 10.0):
        """
        Initialize emotional trajectory integrator.
        
        Args:
            embedding_dimension: Dimension of semantic embeddings
            emotional_memory_length: Length of emotional memory in observational units
        """
        self.embedding_dimension = embedding_dimension
        self.emotional_memory_length = emotional_memory_length
        
        # Initialize emotional resonance parameters
        self.base_emotional_frequencies = np.random.uniform(
            0.1, 2.0, size=embedding_dimension
        )
        
        logger.info(f"Initialized EmotionalTrajectoryIntegrator for {embedding_dimension}D embeddings")
    
    def compute_trajectory(self,
                          token: str,
                          semantic_embedding: np.ndarray,
                          manifold_properties: Dict[str, Any],
                          params: EmotionalTrajectoryParams,
                          temporal_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            
        Returns:
            Dict containing complete E^trajectory(τ, s) computation results
        """
        try:
            # Extract manifold coupling properties
            coupling_mean = manifold_properties.get('coupling_mean', 0.0)
            coupling_variance = manifold_properties.get('coupling_variance', 1.0)
            
            # Step 1: Extract emotional resonance pattern (Section 3.1.3.3.1)
            emotional_resonance = self._extract_emotional_resonance_pattern(
                embedding=semantic_embedding,
                coupling_mean=coupling_mean,
                token=token
            )
            
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
            
            # Final E^trajectory(τ, s) assembly with COMPLEX components
            # All components are now complex-valued field effects
            emotional_trajectory_complex = gaussian_alignment * trajectory_accumulation * resonance_amplification * np.exp(1j * emotional_phase)
            
            # Extract magnitude and phase for analysis
            magnitude = np.abs(emotional_trajectory_complex)
            final_phase = np.angle(emotional_trajectory_complex)
            
            # Return comprehensive results with complex field analysis
            results = {
                'emotional_trajectory_complex': emotional_trajectory_complex,
                'emotional_trajectory_magnitude': magnitude,
                'emotional_phase': final_phase,
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
                    'is_complex_valued': np.iscomplexobj(emotional_trajectory_complex),
                    'field_magnitude': magnitude,
                    'field_phase': final_phase,
                    'gaussian_alignment_complex': np.iscomplexobj(gaussian_alignment),
                    'trajectory_accumulation_complex': np.iscomplexobj(trajectory_accumulation),
                    'resonance_amplification_complex': np.iscomplexobj(resonance_amplification)
                },
                'processing_status': 'complete'
            }
            
            logger.debug(f"E^trajectory computed for {token}: magnitude={magnitude:.4f}, phase={final_phase:.4f}, complex={np.iscomplexobj(emotional_trajectory_complex)}")
            return results
            
        except Exception as e:
            logger.error(f"Emotional trajectory computation failed for {token}: {e}")
            return {
                'emotional_trajectory_complex': complex(1.0, 0.0),
                'emotional_trajectory_magnitude': 1.0,
                'emotional_phase': 0.0,
                'processing_status': 'failed',
                'error': str(e)
            }
    
    def _extract_emotional_resonance_pattern(self,
                                           embedding: np.ndarray,
                                           coupling_mean: float,
                                           token: str) -> np.ndarray:
        """
        Extract COMPLEX emotional resonance pattern using deconstructed attention mechanics.
        
        DECONSTRUCTED ATTENTION (README.md Section 3.1.3.2.1):
        - QK^T → geometric alignment detection producing COMPLEX field effects
        - Emotional content creates directional biases in embedding space  
        - Convert dot product alignment to COMPLEX-valued emotional field modulation
        
        FIELD THEORY APPROACH:
        1. Generate complex-valued emotional field vectors, not scalar floats
        2. Use embedding as Q (query) and create emotional K,V matrices
        3. Produce attention-like complex field patterns that modulate semantic geometry
        """
        # Token-specific emotional characteristics
        token_hash = hash(token) % 1000 / 1000.0
        
        # Emotional strength and polarity from coupling analysis (ensure non-zero)
        emotional_strength = max(abs(coupling_mean), 0.1)  # Prevent zero fields
        emotional_polarity = 1.0 if coupling_mean > 0 else -1.0
        
        # Create COMPLEX resonance pattern matching embedding dimension
        embedding_dim = len(embedding)
        
        # Generate emotional KEY matrix (deconstructed attention approach)
        emotional_keys = np.zeros(embedding_dim, dtype=complex)
        emotional_values = np.zeros(embedding_dim, dtype=complex)
        
        for i in range(embedding_dim):
            # Token-dependent emotional frequency (like attention heads)
            emotional_frequency = token_hash + i / embedding_dim
            phase_shift = 2 * np.pi * emotional_frequency + coupling_mean
            
            # COMPLEX emotional field generation (not just cosine)
            emotional_keys[i] = emotional_strength * emotional_polarity * np.exp(1j * phase_shift)
            
            # Emotional values with different phase characteristics
            value_phase = phase_shift + np.pi/4 + embedding[i] * 0.1  # Semantic influence
            emotional_values[i] = emotional_strength * np.exp(1j * value_phase)
        
        # Deconstructed attention: Q·K^T like operation creating field effects
        # embedding acts as Query, emotional_keys as Keys
        attention_weights = embedding @ emotional_keys.conj()  # Complex dot product
        
        # Softmax-like amplification for complex values (field amplification)
        magnitude = np.abs(attention_weights)
        if magnitude > 0:
            # Scale magnitude to reasonable range (prevent extreme values)
            scaled_magnitude = magnitude / (1.0 + magnitude)  # Keeps in [0,1] range
            amplified_magnitude = 0.1 + 0.9 * scaled_magnitude  # [0.1, 1.0] range
            phase = np.angle(attention_weights)
            attention_complex = amplified_magnitude * np.exp(1j * phase)
        else:
            attention_complex = 0.1 * np.exp(1j * token_hash * 2 * np.pi)
        
        # Generate final COMPLEX resonance pattern through normalized weighted transport
        # Scale emotional_values to be comparable to embedding magnitude
        embedding_scale = np.linalg.norm(embedding) / len(embedding)
        scaled_emotional_values = emotional_values * embedding_scale / np.mean(np.abs(emotional_values))
        
        # Final complex resonance pattern
        resonance_pattern = attention_complex * scaled_emotional_values
        
        return resonance_pattern
    
    def _compute_gaussian_alignment(self,
                                  embedding: np.ndarray,
                                  emotional_resonance: np.ndarray,
                                  coupling_variance: float,
                                  params: EmotionalTrajectoryParams) -> complex:
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
        
        # Compute semantic-emotional alignment distance in complex space
        # Normalize both vectors to compare directions rather than magnitudes
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        resonance_norm = emotional_resonance / (np.linalg.norm(emotional_resonance) + 1e-8)
        
        # Alignment distance in normalized space
        alignment_diff = embedding_norm.astype(complex) - resonance_norm
        alignment_distance = np.linalg.norm(alignment_diff)
        
        # Use coupling_variance as emotional sensitivity parameter σ²
        # Scale sigma appropriately for normalized space
        sigma_squared = max(coupling_variance, 0.1) * 2.0  # Scale for normalized vectors
        
        # COMPLEX amplification factor based on observational state and parameters
        alpha_magnitude = params.amplification_factor * (1.0 + 0.5 * params.observational_state)
        alpha_phase = params.observational_state * 0.5  # Phase from observational state
        alpha_complex = alpha_magnitude * np.exp(1j * alpha_phase)
        
        # Gaussian alignment computation with COMPLEX result
        gaussian_magnitude = np.exp(-alignment_distance**2 / (2 * sigma_squared))
        
        # Ensure minimum meaningful magnitude
        gaussian_magnitude = max(gaussian_magnitude, 0.1)
        
        # Add phase component from emotional field coherence
        emotional_coherence_phase = np.angle(np.mean(emotional_resonance))
        gaussian_phase = emotional_coherence_phase + alpha_phase
        
        # Final COMPLEX Gaussian alignment
        gaussian_alignment = alpha_complex * gaussian_magnitude * np.exp(1j * gaussian_phase)
        
        return gaussian_alignment
    
    def _compute_trajectory_accumulation(self,
                                       token: str,
                                       observational_state: float,
                                       coupling_mean: float,
                                       temporal_data: Optional[Dict[str, Any]],
                                       decay_rate: float) -> complex:
        """
        Compute COMPLEX trajectory accumulation with observational state integration.
        
        FORMULA (README.md Section 3.1.3.3.2):
        ∫₀ˢ w(s-s') · emotional_event(τ, s') ds' - now COMPLEX-valued
        
        COMPLEX FIELD APPROACH:
        - w(s-s'): Complex decay function with phase evolution
        - emotional_event(τ, s'): Complex emotional significance with trajectory-dependent phase
        - Integration produces complex field accumulation, not scalar
        """
        # Extract observational persistence from temporal coordination
        observational_persistence = 1.0
        if temporal_data and 'observational_persistence' in temporal_data:
            observational_persistence = temporal_data['observational_persistence']
        
        # Emotional event strength based on coupling analysis (ensure non-zero)
        emotional_event_strength = max(abs(coupling_mean), 0.1)
        emotional_polarity = 1.0 if coupling_mean > 0 else -1.0
        
        # Token-specific trajectory phase evolution
        token_hash = hash(token) % 1000 / 1000.0
        
        # Discretized trajectory integration with COMPLEX accumulation
        decay_constant = 1.0 / decay_rate
        integration_steps = max(int(observational_state * 10), 1)
        
        accumulation = 0.0 + 0.0j  # Complex accumulation
        ds = observational_state / integration_steps
        
        for step in range(1, integration_steps + 1):
            s_prime = step * ds  # Integration variable
            
            # COMPLEX decay weight with phase evolution
            decay_magnitude = np.exp(-(observational_state - s_prime) / decay_constant)
            # Phase evolves along trajectory based on token and observational state
            decay_phase = 2 * np.pi * token_hash * s_prime + observational_state * 0.1
            decay_weight_complex = decay_magnitude * np.exp(1j * decay_phase)
            
            # COMPLEX emotional event at s' with trajectory-dependent phase
            emotional_event_magnitude = emotional_event_strength * observational_persistence
            # Emotional phase evolution along trajectory
            emotional_phase = s_prime * emotional_polarity + coupling_mean
            emotional_event_complex = emotional_event_magnitude * np.exp(1j * emotional_phase)
            
            # COMPLEX trajectory integration
            accumulation += decay_weight_complex * emotional_event_complex * ds
        
        # Ensure minimum complex accumulation for numerical stability
        if np.abs(accumulation) == 0.0:
            # Create non-zero complex accumulation
            base_magnitude = observational_persistence * emotional_event_strength
            base_phase = token_hash * 2 * np.pi + observational_state * 0.5
            accumulation = base_magnitude * np.exp(1j * base_phase)
        
        return accumulation
    
    def _compute_resonance_amplification(self,
                                       semantic_embedding: np.ndarray,
                                       emotional_resonance: np.ndarray,
                                       coupling_variance: float,
                                       threshold: float) -> complex:
        """
        Compute COMPLEX resonance-based amplification effects.
        
        FORMULA (README.md Section 3.1.3.3.6):
        1 + A_max · exp(-|ω_semantic - ω_emotional|²/2σ_resonance²) - now COMPLEX
        
        COMPLEX FIELD INTERPRETATION:
        - Resonance occurs when semantic and emotional frequencies align in complex space
        - COMPLEX amplification enhances phase-coherent semantic-emotional content
        - Phase-misaligned content gets complex suppression with phase shifts
        """
        # Ensure emotional_resonance is complex
        if not np.iscomplexobj(emotional_resonance):
            emotional_resonance = emotional_resonance.astype(complex)
        
        # Compute complex frequencies for semantic and emotional content
        semantic_complex = semantic_embedding.astype(complex)
        
        # Complex frequency characteristics (magnitude and phase)
        semantic_freq_complex = np.mean(semantic_complex)  # Average complex "frequency"
        emotional_freq_complex = np.mean(emotional_resonance)  # Average complex "frequency"
        
        # Complex frequency difference for resonance detection
        freq_difference_complex = semantic_freq_complex - emotional_freq_complex
        freq_difference_magnitude = abs(freq_difference_complex)
        
        # Resonance parameters
        A_max = 1.0  # Maximum amplification factor
        sigma_resonance_squared = max(coupling_variance, 0.1)  # Prevent degenerate cases
        
        # Resonance amplification computation with COMPLEX phase effects
        resonance_magnitude = A_max * np.exp(-freq_difference_magnitude**2 / (2 * sigma_resonance_squared))
        
        # Phase alignment bonus: coherent phases get more amplification
        phase_alignment = np.angle(semantic_freq_complex) - np.angle(emotional_freq_complex)
        phase_bonus = np.cos(phase_alignment)  # +1 for aligned, -1 for opposite
        
        # Complex amplification with phase-dependent enhancement
        amplification_magnitude = 1.0 + resonance_magnitude * (1.0 + 0.5 * phase_bonus)
        amplification_phase = phase_alignment * 0.1  # Small phase contribution from resonance
        
        amplification_complex = amplification_magnitude * np.exp(1j * amplification_phase)
        
        # Apply threshold for significant resonance effects
        if abs(amplification_complex) - 1.0 < threshold:
            # Ensure minimum complex amplification
            min_phase = np.angle(amplification_complex)
            amplification_complex = (1.0 + threshold) * np.exp(1j * min_phase)
        
        return amplification_complex
    
    def _compute_emotional_phase(self,
                               token: str,
                               observational_state: float,
                               coupling_properties: Dict[str, float],
                               temporal_data: Optional[Dict[str, Any]]) -> float:
        """
        Compute emotional phase contribution to total phase integration.
        
        FORMULA (README.md Section 3.1.3.3.5):
        φ_emotional(τ, s) = ∫₀ˢ ω_emotional(τ, s') ds' + Σⱼ coupling_emotional[j] · φ_j(s')
        
        MATHEMATICAL APPROACH:
        - ω_emotional: Token-specific emotional frequency evolution
        - Integration over observational state
        - Cross-dimensional coupling with temporal phases
        """
        # Token-specific emotional frequency
        token_hash = hash(token) % 1000 / 1000.0
        base_emotional_frequency = 2 * np.pi * token_hash
        
        # Coupling modulation of frequency
        coupling_mean = coupling_properties['mean']
        coupling_variance = coupling_properties['variance']
        
        # Emotional frequency evolution with coupling effects
        omega_emotional = base_emotional_frequency * (1.0 + 0.1 * coupling_mean)
        
        # Phase accumulation through observational state integration
        phase_integral = omega_emotional * observational_state
        
        # Cross-dimensional coupling with temporal phases
        coupling_contribution = 0.0
        if temporal_data and 'phase_accumulation' in temporal_data:
            temporal_phases = temporal_data['phase_accumulation']
            if len(temporal_phases) > 0:
                coupling_strength = 0.1 * abs(coupling_mean)
                coupling_contribution = coupling_strength * np.mean(temporal_phases)
        
        # Total emotional phase
        emotional_phase = phase_integral + coupling_contribution
        
        # Normalize phase to [-π, π] range
        emotional_phase = (emotional_phase + np.pi) % (2 * np.pi) - np.pi
        
        return emotional_phase
    
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
                logger.warning(f"Failed to compute emotional trajectory for token {i} ({token}): {e}")
                results.append({
                    'emotional_trajectory_complex': complex(1.0, 0.0),
                    'processing_status': 'failed',
                    'error': str(e)
                })
        
        logger.info(f"Computed emotional trajectories for {len(results)} tokens")
        return results


def create_emotional_trajectory_params(observational_state: float,
                                     emotional_intensity: float = 1.0,
                                     memory_decay: float = 0.1) -> EmotionalTrajectoryParams:
    """
    Convenience function to create emotional trajectory parameters.
    
    Args:
        observational_state: Current observational state s
        emotional_intensity: Base emotional amplification factor
        memory_decay: Rate of emotional memory decay
        
    Returns:
        EmotionalTrajectoryParams configured for computation
    """
    return EmotionalTrajectoryParams(
        observational_state=observational_state,
        gaussian_sigma=0.5,
        trajectory_decay_rate=memory_decay,
        amplification_factor=emotional_intensity,
        resonance_threshold=0.1,
        coupling_strength=0.2
    )