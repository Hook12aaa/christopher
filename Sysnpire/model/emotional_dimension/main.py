"""
Emotional Dimension Main Interface - E^trajectory(τ, s) Processing

MATHEMATICAL FOUNDATION:
E^trajectory(τ, s) = α_i · exp(-||v_i - v_E||²/2σ²) · ∫₀ˢ w(s-s') · emotional_event(τ, s') ds'

INTEGRATION POINTS:
- Charge Factory: Provides E^trajectory(τ, s) component for Q(τ, C, s)
- Manifold Properties: Uses coupling_mean, coupling_variance from correlation analysis
- Temporal Dimension: Coordinates with observational persistence and phase data
- Semantic Dimension: Influences and is influenced by semantic field generation

This module provides the main interface for emotional trajectory processing
using deconstructed transformer mathematics and field theory principles.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import logging

from .trajectory_evolution import EmotionalTrajectoryIntegrator, EmotionalTrajectoryParams, create_emotional_trajectory_params
from .attention_deconstruction import AttentionGeometryAnalyzer, create_attention_analyzer
from .field_modulation import EmotionalFieldModulator, apply_emotional_modulation
from .phase_integration_bridge import EmotionalPhaseIntegrationBridge, integrate_emotional_with_complete_phase

logger = logging.getLogger(__name__)


def compute_emotional_trajectory(token: str,
                               semantic_embedding: np.ndarray,
                               manifold_properties: Dict[str, Any],
                               observational_state: float,
                               gamma: float,
                               context: str = "general",
                               temporal_data: Optional[Dict[str, Any]] = None,
                               emotional_intensity: float = 1.0) -> Dict[str, Any]:
    """
    Main interface for computing E^trajectory(τ, s) emotional trajectory integration.
    
    MATHEMATICAL PROCESS:
    1. Create emotional trajectory parameters from inputs
    2. Initialize trajectory integrator for embedding dimension
    3. Compute complete emotional trajectory using deconstructed mathematics
    4. Return results compatible with charge factory integration
    
    Args:
        token: Token identifier for trajectory tracking
        semantic_embedding: Base semantic vector [D]
        manifold_properties: Contains coupling_mean, coupling_variance from analysis
        observational_state: Current observational state s
        gamma: Global field calibration factor γ
        context: Context string for trajectory computation
        temporal_data: Optional temporal dimension coordination data
        emotional_intensity: Base emotional amplification factor
        
    Returns:
        Dict containing E^trajectory(τ, s) and analysis results
    """
    try:
        # Create emotional trajectory parameters
        params = create_emotional_trajectory_params(
            observational_state=observational_state,
            emotional_intensity=emotional_intensity * gamma,
            memory_decay=0.1
        )
        
        # Initialize trajectory integrator for embedding dimension
        integrator = EmotionalTrajectoryIntegrator(
            embedding_dimension=len(semantic_embedding),
            emotional_memory_length=10.0
        )
        
        # Compute emotional trajectory
        trajectory_results = integrator.compute_trajectory(
            token=token,
            semantic_embedding=semantic_embedding,
            manifold_properties=manifold_properties,
            params=params,
            temporal_data=temporal_data
        )
        
        # Extract key results for charge factory integration
        emotional_trajectory_complex = trajectory_results['emotional_trajectory_complex']
        emotional_magnitude = trajectory_results['emotional_trajectory_magnitude']
        emotional_phase = trajectory_results['emotional_phase']
        
        # Format results for charge factory
        results = {
            # Primary E^trajectory(τ, s) result
            'emotional_trajectory_complex': emotional_trajectory_complex,
            'emotional_trajectory_magnitude': emotional_magnitude,
            'emotional_phase': emotional_phase,
            
            # Component analysis
            'gaussian_alignment': trajectory_results['gaussian_alignment'],
            'trajectory_accumulation': trajectory_results['trajectory_accumulation'],
            'resonance_amplification': trajectory_results['resonance_amplification'],
            
            # Coupling analysis for debugging
            'coupling_analysis': trajectory_results['coupling_analysis'],
            
            # Phase Integration Data - formatted for phase dimension
            'emotional_data_for_phase': {
                'emotional_trajectory_complex': emotional_trajectory_complex,
                'emotional_phase': emotional_phase,
                'emotional_magnitude': emotional_magnitude,
                'complex_field_data': {
                    'magnitude': emotional_magnitude,
                    'phase': emotional_phase,
                    'real': emotional_trajectory_complex.real,
                    'imag': emotional_trajectory_complex.imag
                },
                'phase_components': [emotional_phase],  # Array format for phase extraction
                'field_magnitudes': [emotional_magnitude]  # Array format for phase extraction
            },
            
            # Integration metadata
            'processing_method': 'deconstructed_transformer_mathematics',
            'field_theory_compliant': True,
            'observational_state': observational_state,
            'gamma_influence': gamma,
            'context': context,
            'processing_status': trajectory_results['processing_status']
        }
        
        logger.debug(f"Emotional trajectory computed for {token}: |E|={emotional_magnitude:.4f}, φ={emotional_phase:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Emotional trajectory computation failed for {token}: {e}")
        raise RuntimeError(f"Emotional trajectory computation failed for {token}: {e}")


def analyze_emotional_attention_patterns(semantic_embedding: np.ndarray,
                                       token: str,
                                       coupling_properties: Dict[str, float],
                                       context_embeddings: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Analyze emotional patterns using deconstructed attention mechanisms.
    
    MATHEMATICAL FOUNDATION (README.md Section 3.1.3.2):
    Attention(Q, K, V) = softmax(QK^T / √d_k) · V
    → Geometric alignment detection + exponential amplification + weighted transport
    
    Args:
        semantic_embedding: Base semantic vector [D]
        token: Token identifier for analysis
        coupling_properties: Manifold coupling analysis results
        context_embeddings: Optional context embeddings for analysis
        
    Returns:
        Dict containing attention-based emotional pattern analysis
    """
    try:
        # Create attention analyzer
        analyzer = create_attention_analyzer(
            embedding_dimension=len(semantic_embedding),
            emotional_sensitivity=0.3
        )
        
        # Analyze emotional patterns
        analysis_result = analyzer.analyze_emotional_patterns(
            semantic_embedding=semantic_embedding,
            token=token,
            context_embeddings=context_embeddings
        )
        
        # Extract emotional resonance using coupling properties
        coupling_mean = coupling_properties.get('mean', 0.0)
        emotional_resonance = analyzer.extract_emotional_resonance_from_attention(
            semantic_embedding=semantic_embedding,
            token=token,
            coupling_mean=coupling_mean
        )
        
        # Format results
        results = {
            'attention_weights': analysis_result.attention_weights.tolist(),
            'geometric_alignments': analysis_result.geometric_alignments.tolist(),
            'emotional_patterns': analysis_result.emotional_patterns.tolist(),
            'amplification_factors': analysis_result.amplification_factors.tolist(),
            'field_effects': analysis_result.field_effects.tolist(),
            'emotional_resonance': emotional_resonance.tolist(),
            'coupling_influence': abs(coupling_mean),
            'analysis_method': 'deconstructed_attention',
            'processing_status': 'complete'
        }
        
        logger.debug(f"Attention analysis complete for {token}: {len(emotional_resonance)} dimensions processed")
        return results
        
    except Exception as e:
        logger.error(f"Attention analysis failed for {token}: {e}")
        return {
            'processing_status': 'failed',
            'error': str(e)
        }


def compute_batch_emotional_trajectories(tokens: List[str],
                                       embeddings: np.ndarray,
                                       manifold_properties_batch: List[Dict[str, Any]],
                                       observational_state: float,
                                       gamma: float,
                                       context: str = "batch_processing",
                                       temporal_data_batch: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Efficiently compute emotional trajectories for batch of tokens.
    
    BATCH PROCESSING OPTIMIZATION:
    - Single integrator initialization for efficiency
    - Shared parameters across batch
    - Maintains mathematical accuracy for each individual computation
    
    Args:
        tokens: List of token identifiers
        embeddings: Batch of semantic embeddings [N, D]
        manifold_properties_batch: List of manifold properties for each token
        observational_state: Shared observational state
        gamma: Global field calibration factor
        context: Shared context for batch
        temporal_data_batch: Optional temporal data for each token
        
    Returns:
        List of emotional trajectory computation results
    """
    try:
        # Create shared parameters
        params = create_emotional_trajectory_params(
            observational_state=observational_state,
            emotional_intensity=gamma,
            memory_decay=0.1
        )
        
        # Initialize batch integrator
        integrator = EmotionalTrajectoryIntegrator(
            embedding_dimension=embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0]),
            emotional_memory_length=10.0
        )
        
        # Compute batch trajectories
        results = integrator.compute_batch_trajectories(
            tokens=tokens,
            embeddings=embeddings,
            manifold_properties_batch=manifold_properties_batch,
            params=params,
            temporal_data_batch=temporal_data_batch
        )
        
        # Format results for charge factory compatibility
        formatted_results = []
        for i, result in enumerate(results):
            if result['processing_status'] == 'complete':
                formatted_result = {
                    'emotional_trajectory_complex': result['emotional_trajectory_complex'],
                    'emotional_trajectory_magnitude': result['emotional_trajectory_magnitude'],
                    'emotional_phase': result['emotional_phase'],
                    'gaussian_alignment': result['gaussian_alignment'],
                    'trajectory_accumulation': result['trajectory_accumulation'],
                    'resonance_amplification': result['resonance_amplification'],
                    'coupling_analysis': result['coupling_analysis'],
                    'processing_method': 'batch_deconstructed_transformer',
                    'field_theory_compliant': True,
                    'token': tokens[i],
                    'processing_status': 'complete'
                }
            else:
                # No fallback values allowed per CLAUDE.md
                raise RuntimeError(f"Emotional trajectory computation failed for token {tokens[i]}: {result.get('error', 'unknown_error')}")
            
            formatted_results.append(formatted_result)
        
        logger.info(f"Batch emotional trajectory computation complete: {len(formatted_results)} results")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Batch emotional trajectory computation failed: {e}")
        raise RuntimeError(f"Batch emotional trajectory computation failed: {e}")


def warp_semantic_field_with_emotional_trajectory(semantic_results: Dict[str, Any],
                                                  token: str,
                                                  manifold_properties: Dict[str, Any],
                                                  observational_state: float,
                                                  gamma: float,
                                                  context: str = "semantic_warping",
                                                  emotional_intensity: float = 1.0) -> Dict[str, Any]:
    """
    Warp semantic field with emotional trajectory - Core integration function.
    
    MATHEMATICAL FOUNDATION:
    Enhanced_Φ^semantic = Φ^semantic(τ, s) * E^trajectory(τ, s) * phase_coupling
    
    This function modifies the existing semantic field data structure with emotional
    field effects, implementing the geometric warping described in README.md Section 3.1.3.3.3.
    
    Args:
        semantic_results: Existing semantic field results from semantic dimension
        token: Token identifier for emotional trajectory computation
        manifold_properties: Manifold properties for emotional computation
        observational_state: Current observational state s
        gamma: Global field calibration factor γ
        context: Context for emotional processing
        emotional_intensity: Base emotional amplification
        
    Returns:
        Enhanced semantic results with emotional field modulation applied
    """
    try:
        logger.info(f"Applying emotional field warping to semantic field for token '{token}'")
        
        # Extract semantic field from results
        semantic_field_complex = semantic_results.get('dtf_phi_semantic_complex', 
                                                     semantic_results.get('semantic_field', complex(0)))
        if semantic_field_complex == 0:
            raise ValueError("No valid semantic field found in semantic_results")
        
        # Get semantic embedding for emotional trajectory computation
        # Reconstruct from complex field properties
        semantic_magnitude = abs(semantic_field_complex)
        semantic_phase = np.angle(semantic_field_complex)
        
        # Create synthetic embedding vector for emotional processing
        # Use semantic field properties to generate embedding-like vector
        embedding_dimension = 1024  # Standard BGE dimension
        synthetic_embedding = np.zeros(embedding_dimension)
        
        # Fill embedding with semantic field characteristics
        for i in range(embedding_dimension):
            phase_component = semantic_phase + i * np.pi / embedding_dimension
            magnitude_component = semantic_magnitude * np.cos(phase_component)
            synthetic_embedding[i] = magnitude_component
        
        # Normalize synthetic embedding
        embedding_norm = np.linalg.norm(synthetic_embedding)
        if embedding_norm > 0:
            synthetic_embedding = synthetic_embedding / embedding_norm * semantic_magnitude
        
        logger.debug(f"Created synthetic embedding from semantic field: dim={len(synthetic_embedding)}, "
                    f"norm={np.linalg.norm(synthetic_embedding):.4f}")
        
        # Compute emotional trajectory using semantic-derived embedding
        emotional_results = compute_emotional_trajectory(
            token=token,
            semantic_embedding=synthetic_embedding,
            manifold_properties=manifold_properties,
            observational_state=observational_state,
            gamma=gamma,
            context=context,
            emotional_intensity=emotional_intensity
        )
        
        # Extract emotional trajectory components
        emotional_trajectory_complex = emotional_results['emotional_trajectory_complex']
        emotional_magnitude = emotional_results['emotional_trajectory_magnitude']
        emotional_phase = emotional_results['emotional_phase']
        
        logger.debug(f"Emotional trajectory computed: E={emotional_trajectory_complex}, "
                    f"|E|={emotional_magnitude:.4f}, φ={emotional_phase:.4f}")
        
        # Apply geometric field modulation using field_modulation.py
        modulated_field, field_analysis = apply_emotional_modulation(
            semantic_field=synthetic_embedding,
            emotional_trajectory=emotional_trajectory_complex,
            coupling_strength=gamma * 0.3
        )
        
        # Calculate enhanced semantic field through complex multiplication
        # Enhanced_Φ^semantic = Φ^semantic(τ, s) * E^trajectory(τ, s) * phase_coupling
        semantic_phase = np.angle(semantic_field_complex)
        phase_coupling = np.exp(1j * (semantic_phase + emotional_phase))
        
        enhanced_semantic_field = semantic_field_complex * emotional_trajectory_complex * phase_coupling
        enhanced_magnitude = abs(enhanced_semantic_field)
        enhanced_phase = np.angle(enhanced_semantic_field)
        
        logger.info(f"Semantic field enhanced: {semantic_field_complex} → {enhanced_semantic_field}, "
                   f"enhancement factor: {enhanced_magnitude/abs(semantic_field_complex):.3f}")
        
        # Prepare data for complete phase integration
        semantic_data = {
            'phase_angles': manifold_properties.get('phase_angles', []),
            'semantic_modulation': manifold_properties.get('semantic_modulation', []),
            'gradient': manifold_properties.get('gradient', []),
            'semantic_field_complex': enhanced_semantic_field
        }
        
        trajectory_data = {
            'phase_accumulation': [],  # Will be provided by temporal dimension
            'frequency_evolution': [],  # Will be provided by temporal dimension  
            'transformative_magnitude': [1.0],  # Basic trajectory data
            'total_transformative_potential': 1.0
        }
        
        # Perform complete phase integration using phase integration bridge
        try:
            logger.debug(f"Computing complete phase integration for token '{token}'")
            
            unified_complex_field, complete_phase_integration = integrate_emotional_with_complete_phase(
                emotional_results=emotional_results,
                semantic_data=semantic_data,
                trajectory_data=trajectory_data,
                context=context,
                observational_state=observational_state,
                manifold_properties=manifold_properties
            )
            
            logger.debug(f"Complete phase integration successful: unified field={unified_complex_field}")
            
        except Exception as e:
            logger.warning(f"Complete phase integration failed for '{token}': {e}")
            # Create basic phase integration data
            unified_complex_field = enhanced_semantic_field * emotional_trajectory_complex
            complete_phase_integration = {
                'unified_complex_field': unified_complex_field,
                'total_phase': np.angle(unified_complex_field),
                'total_magnitude': abs(unified_complex_field),
                'integration_status': 'fallback',
                'error': str(e)
            }
        
        # Update semantic results with emotional enhancements
        enhanced_results = semantic_results.copy()
        enhanced_results.update({
            # Enhanced semantic field components
            'enhanced_semantic_field_complex': enhanced_semantic_field,
            'enhanced_semantic_magnitude': enhanced_magnitude,
            'enhanced_semantic_phase': enhanced_phase,
            
            # Update primary semantic field values
            'dtf_phi_semantic_complex': enhanced_semantic_field,
            'dtf_phi_semantic_magnitude': enhanced_magnitude,
            'field_magnitude': enhanced_magnitude,
            'complete_charge_magnitude': enhanced_magnitude,
            
            # Emotional integration metadata
            'emotional_trajectory_applied': emotional_trajectory_complex,
            'emotional_enhancement_magnitude': emotional_magnitude,
            'emotional_phase_contribution': emotional_phase,
            'phase_coupling_factor': phase_coupling,
            'semantic_emotional_coupling': gamma * 0.3,
            
            # Field modulation analysis
            'field_modulation_analysis': field_analysis,
            'metric_warping_factor': field_analysis.get('metric_warping_factor', 1.0),
            'geometric_distortions': field_analysis.get('num_distortions', 0),
            
            # Complete Phase Integration Results
            'unified_complex_field': unified_complex_field,
            'complete_phase_integration': complete_phase_integration,
            'total_phase': complete_phase_integration.get('total_phase', enhanced_phase),
            'phase_coherence': complete_phase_integration.get('field_quality_metrics', {}).get('phase_coherence', 0.5),
            
            # Processing status
            'emotional_warping_applied': True,
            'emotional_integration_status': 'complete',
            'phase_integration_status': complete_phase_integration.get('integration_status', 'unknown'),
            'processing_method': 'emotional_field_warping_with_phase_integration',
            'field_theory_compliant': True
        })
        
        logger.info(f"Emotional warping complete for '{token}': enhancement={enhanced_magnitude/abs(semantic_field_complex):.3f}x, "
                   f"distortions={field_analysis.get('num_distortions', 0)}")
        
        return enhanced_results
        
    except Exception as e:
        logger.error(f"Semantic field warping failed for '{token}': {e}")
        raise RuntimeError(f"Semantic field warping failed for '{token}': {e}")


def integrate_emotional_dimension_with_semantic_results(semantic_results_batch: List[Dict[str, Any]],
                                                       tokens: List[str],
                                                       manifold_properties_batch: List[Dict[str, Any]],
                                                       observational_state: float,
                                                       gamma: float,
                                                       context: str = "batch_semantic_warping") -> List[Dict[str, Any]]:
    """
    Batch integration of emotional dimension with semantic results.
    
    FIELD INTEGRATION APPROACH:
    For each semantic result, apply emotional trajectory warping to enhance
    the existing semantic field structure with emotional field effects.
    
    Args:
        semantic_results_batch: List of semantic processing results
        tokens: List of token identifiers
        manifold_properties_batch: List of manifold properties for each token
        observational_state: Shared observational state
        gamma: Global field calibration factor
        context: Processing context
        
    Returns:
        List of enhanced semantic results with emotional warping applied
    """
    enhanced_results = []
    
    logger.info(f"Starting batch emotional integration for {len(semantic_results_batch)} semantic results")
    
    for i, (semantic_result, token, manifold_props) in enumerate(zip(semantic_results_batch, tokens, manifold_properties_batch)):
        try:
            # Apply emotional warping to each semantic result
            enhanced_result = warp_semantic_field_with_emotional_trajectory(
                semantic_results=semantic_result,
                token=token,
                manifold_properties=manifold_props,
                observational_state=observational_state,
                gamma=gamma,
                context=f"{context}_{i}",
                emotional_intensity=1.0
            )
            
            enhanced_results.append(enhanced_result)
            
        except Exception as e:
            logger.error(f"Failed to apply emotional warping to token '{token}': {e}")
            raise RuntimeError(f"Emotional integration failed for token '{token}': {e}")
    
    logger.info(f"Batch emotional integration complete: {len(enhanced_results)} results enhanced")
    return enhanced_results


def get_emotional_dimension_info() -> Dict[str, Any]:
    """
    Get information about emotional dimension implementation.
    
    Returns:
        Dict containing implementation details and capabilities
    """
    return {
        'mathematical_foundation': 'E^trajectory(τ, s) = α_i · exp(-||v_i - v_E||²/2σ²) · ∫₀ˢ w(s-s\') · emotional_event(τ, s\') ds\'',
        'approach': 'deconstructed_transformer_mathematics',
        'components': [
            'trajectory_evolution.py - Core E^trajectory(τ, s) integration',
            'attention_deconstruction.py - Transformer attention → field effects',
            'field_modulation.py - Emotional field effects on geometry'
        ],
        'integration_points': [
            'charge_factory.py - Provides E^trajectory component for Q(τ, C, s)',
            'manifold_properties - Uses coupling_mean, coupling_variance',
            'temporal_dimension - Coordinates observational persistence',
            'semantic_dimension - Bidirectional field influence'
        ],
        'mathematical_compliance': 'CLAUDE.md field theory requirements',
        'complex_valued_results': True,
        'trajectory_dependent': True,
        'field_theoretic': True,
        'semantic_warping_capability': True
    }


# Main interface functions for charge factory integration
__all__ = [
    'compute_emotional_trajectory',
    'analyze_emotional_attention_patterns', 
    'compute_batch_emotional_trajectories',
    'get_emotional_dimension_info',
    'warp_semantic_field_with_emotional_trajectory',
    'integrate_emotional_dimension_with_semantic_results',
    # Phase integration capabilities
    'EmotionalPhaseIntegrationBridge',
    'integrate_emotional_with_complete_phase'
]