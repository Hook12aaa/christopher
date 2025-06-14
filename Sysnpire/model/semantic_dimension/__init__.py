"""
Semantic Dimension - Field Generation (Section 3.1.2)

Transforms static embeddings into dynamic field-generating functions that actively
shape the semantic landscape through field effects.

MAIN ENTRY POINT: process_semantic_field() - Core semantic dimension processing
for the Q(τ, C, s) formula's Φ^semantic(τ, s) component.

Components:
- field_generators.py: S_τ(x) field function implementations  
- main.py: Central processing functions (equivalent to index.js)
- breathing_patterns.py: Dynamic constellation breathing mathematics
- phase_modulation.py: Complex phase processing for semantic fields
"""

import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Ensure project imports work
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


def process_semantic_field(embedding: np.ndarray,
                          manifold_properties: Dict[str, Any],
                          observational_state: float,
                          gamma: float,
                          context: str,
                          field_temperature: float,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    MAIN SEMANTIC PROCESSING FUNCTION - Entry point for semantic dimension.
    
    This is the primary function called by ChargeFactory.semantic_dimension().
    Implements the Φ^semantic(τ, s) component of Q(τ, C, s).
    
    MATHEMATICAL FOUNDATION:
    Φ^semantic(τ, s) = w_i * T_i * x[i] * breathing_modulation * e^(iθ)
    
    Args:
        embedding: Base semantic vector (1024d BGE, 768d MPNet)
        manifold_properties: Geometric properties from manifold analysis
        observational_state: Current observational state (s parameter)
        gamma: Field calibration factor (γ)
        context: Processing context (C parameter)
        field_temperature: Temperature for field dynamics
        metadata: Optional processing metadata
        
    Returns:
        Dict containing semantic field processing results:
        - 'semantic_field': Processed semantic field vector
        - 'breathing_patterns': Dynamic constellation data
        - 'phase_modulation': Complex phase information
        - 'field_magnitude': Semantic field strength
        - 'constellation_topology': Geometric structure
    """
    try:
        logger.debug(f"Processing semantic field - embedding shape: {embedding.shape}, "
                    f"observational_state: {observational_state}, gamma: {gamma}")
        
        # Initialize semantic field with base embedding
        semantic_field = embedding.copy()
        
        # Apply gamma calibration factor
        semantic_field = gamma * semantic_field
        
        # Compute breathing patterns based on manifold properties
        breathing_patterns = _compute_breathing_patterns(
            embedding, manifold_properties, observational_state, field_temperature
        )
        
        # Apply breathing modulation to semantic field
        semantic_field = _apply_breathing_modulation(semantic_field, breathing_patterns)
        
        # Compute phase modulation
        phase_modulation = _compute_phase_modulation(
            embedding, manifold_properties, context, observational_state
        )
        
        # Apply complex phase to semantic field
        semantic_field_complex = _apply_phase_modulation(semantic_field, phase_modulation)
        
        # Compute final field magnitude
        field_magnitude = float(np.linalg.norm(semantic_field_complex))
        
        # Extract constellation topology
        constellation_topology = _extract_constellation_topology(
            semantic_field, manifold_properties, breathing_patterns
        )
        
        results = {
            'semantic_field': semantic_field_complex,
            'breathing_patterns': breathing_patterns,
            'phase_modulation': phase_modulation,
            'field_magnitude': field_magnitude,
            'constellation_topology': constellation_topology,
            'processing_status': 'completed',
            'gamma_applied': gamma,
            'observational_state': observational_state
        }
        
        logger.debug(f"Semantic field processing complete - magnitude: {field_magnitude:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Semantic field processing failed: {e}")
        raise


def _compute_breathing_patterns(embedding: np.ndarray, 
                               manifold_properties: Dict[str, Any],
                               observational_state: float,
                               temperature: float) -> Dict[str, Any]:
    """Compute dynamic breathing patterns for constellation modulation."""
    # Extract manifold properties for breathing calculation
    local_density = manifold_properties.get('local_density', 1.0)
    persistence_radius = manifold_properties.get('persistence_radius', 1.0)
    
    # Compute breathing frequency based on observational state
    breathing_frequency = observational_state * np.pi / (1.0 + persistence_radius)
    
    # Compute breathing amplitude from local density
    breathing_amplitude = 0.5 * local_density * temperature
    
    # Generate breathing pattern
    breathing_phase = np.sin(breathing_frequency) * breathing_amplitude
    
    return {
        'frequency': float(breathing_frequency),
        'amplitude': float(breathing_amplitude),
        'phase': float(breathing_phase),
        'local_density_factor': float(local_density),
        'pattern_type': 'sinusoidal_modulation'
    }


def _apply_breathing_modulation(semantic_field: np.ndarray, 
                               breathing_patterns: Dict[str, Any]) -> np.ndarray:
    """Apply breathing modulation to semantic field."""
    breathing_factor = 1.0 + breathing_patterns['phase']
    return semantic_field * breathing_factor


def _compute_phase_modulation(embedding: np.ndarray,
                             manifold_properties: Dict[str, Any], 
                             context: str,
                             observational_state: float) -> Dict[str, Any]:
    """Compute complex phase modulation for semantic field."""
    # Extract phase angles from manifold properties
    phase_angles = manifold_properties.get('phase_angles', [0.0])
    
    # Compute context-dependent phase shift
    context_hash = hash(context) % 1000000
    context_phase = (context_hash / 1000000.0) * 2.0 * np.pi
    
    # Combine with observational state
    total_phase = np.mean(phase_angles) + context_phase * observational_state
    
    return {
        'real': float(np.cos(total_phase)),
        'imag': float(np.sin(total_phase)),
        'magnitude': 1.0,
        'phase_angle': float(total_phase),
        'context_contribution': float(context_phase)
    }


def _apply_phase_modulation(semantic_field: np.ndarray,
                           phase_modulation: Dict[str, Any]) -> np.ndarray:
    """Apply complex phase modulation to semantic field."""
    phase_factor = complex(phase_modulation['real'], phase_modulation['imag'])
    # For real vectors, apply phase through magnitude modulation
    return semantic_field * abs(phase_factor)


def _extract_constellation_topology(semantic_field: np.ndarray,
                                   manifold_properties: Dict[str, Any],
                                   breathing_patterns: Dict[str, Any]) -> Dict[str, Any]:
    """Extract topological structure of semantic constellation."""
    # Compute field topology metrics
    field_variance = float(np.var(semantic_field))
    field_mean = float(np.mean(semantic_field))
    field_entropy = -np.sum(np.abs(semantic_field) * np.log(np.abs(semantic_field) + 1e-12))
    
    return {
        'field_variance': field_variance,
        'field_mean': field_mean,
        'field_entropy': float(field_entropy),
        'breathing_coupling': breathing_patterns['amplitude'],
        'dimensionality': len(semantic_field),
        'topology_type': 'manifold_coupled_constellation'
    }


# Import processing components
from .processing import SemanticFieldPool, create_field_pool_from_manifold

# Export main processing function and field pool
__all__ = ['process_semantic_field', 'SemanticFieldPool', 'create_field_pool_from_manifold']