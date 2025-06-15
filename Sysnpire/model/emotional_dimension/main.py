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
        return {
            'emotional_trajectory_complex': complex(1.0, 0.0),
            'emotional_trajectory_magnitude': 1.0,
            'emotional_phase': 0.0,
            'processing_status': 'failed',
            'error': str(e)
        }


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
                formatted_result = {
                    'emotional_trajectory_complex': complex(1.0, 0.0),
                    'emotional_trajectory_magnitude': 1.0,
                    'emotional_phase': 0.0,
                    'processing_status': 'failed',
                    'token': tokens[i],
                    'error': result.get('error', 'unknown_error')
                }
            
            formatted_results.append(formatted_result)
        
        logger.info(f"Batch emotional trajectory computation complete: {len(formatted_results)} results")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Batch emotional trajectory computation failed: {e}")
        # Return fallback results
        return [
            {
                'emotional_trajectory_complex': complex(1.0, 0.0),
                'emotional_trajectory_magnitude': 1.0,
                'emotional_phase': 0.0,
                'processing_status': 'failed',
                'token': token,
                'error': str(e)
            }
            for token in tokens
        ]


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
        'field_theoretic': True
    }


# Main interface functions for charge factory integration
__all__ = [
    'compute_emotional_trajectory',
    'analyze_emotional_attention_patterns', 
    'compute_batch_emotional_trajectories',
    'get_emotional_dimension_info'
]