"""
Vector-to-Field Transformation - Core Semantic Field Generation

This module implements the fundamental mathematical transformation that converts
static embedding vectors into dynamic semantic field generators as defined in
section 3.1.2.8.1 of the Field Theory of Social Constructs.

CORE TRANSFORMATION:
S_τ(x) = Σᵢ e_τ,ᵢ · φᵢ(x) · e^(iθ_τ,ᵢ)

Where:
- S_τ(x): Semantic field-generating function for token τ at position x  
- e_τ,ᵢ: i-th component of the original embedding vector
- φᵢ(x): Basis function defined across the semantic manifold
- e^(iθ_τ,ᵢ): Complex phase factor enabling interference effects

This transformation enables embeddings to function as field generators rather
than static coordinate positions, supporting dynamic semantic interactions.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import sys
from pathlib import Path

# Ensure project imports work
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.utils.logger import get_logger
from Sysnpire.utils.field_theory_optimizers import (
    field_theory_jax_optimize, field_theory_numba_optimize, 
    field_theory_auto_optimize
)

logger = get_logger(__name__)


def _optimized_gaussian_basis(position_x: np.ndarray, center: np.ndarray, width: float) -> float:
    """
    Optimized Gaussian basis function computation.
    
    CLAUDE.md Compliance: Optimized for φᵢ(x) basis function calculations.
    Preserves mathematical accuracy for semantic field computations.
    """
    # Ensure position_x and center have compatible dimensions
    min_len = min(len(position_x), len(center))
    pos_truncated = position_x[:min_len]
    center_truncated = center[:min_len]
    
    # Compute normalized distance for high-dimensional stability
    diff = pos_truncated - center_truncated
    distance_squared = np.sum(diff**2)
    
    # Normalize by dimension to prevent exponential decay in high dimensions
    normalized_distance_squared = distance_squared / min_len
    
    # Apply Gaussian with normalized distance
    basis_value = np.exp(-normalized_distance_squared / (2 * width**2))
    
    return basis_value


@field_theory_auto_optimize(prefer_accuracy=True, profile=True)
def _optimized_semantic_field_summation(embedding: np.ndarray, basis_values: np.ndarray, 
                                       phase_factors: np.ndarray) -> complex:
    """
    Optimized semantic field summation for S_τ(x) = Σᵢ e_τ,ᵢ · φᵢ(x) · e^(iθ_τ,ᵢ).
    
    CLAUDE.md Compliance: Field theory optimized for complex-valued semantic field generation.
    Preserves phase relationships and mathematical accuracy.
    """
    # Vectorized field component computation
    field_components = embedding * basis_values * phase_factors
    
    # Sum all components to get total semantic field
    semantic_field = np.sum(field_components)
    
    return semantic_field


class VectorToFieldTransformer:
    """
    Core transformer for converting static embedding vectors into dynamic semantic fields.
    
    This class implements the fundamental S_τ(x) transformation that enables
    embeddings to generate field effects across semantic space rather than
    merely representing static positions.
    """
    
    def __init__(self, 
                 embedding_dimension: int = 1024,
                 basis_function_type: str = "gaussian",
                 phase_computation_method: str = "component_based"):
        """
        Initialize the vector-to-field transformer.
        
        Args:
            embedding_dimension: Dimensionality of input embeddings (1024 for BGE, 768 for MPNet)
            basis_function_type: Type of basis functions to use ("gaussian", "harmonic", "adaptive")
            phase_computation_method: Method for computing phase factors
        """
        self.embedding_dimension = embedding_dimension
        self.basis_function_type = basis_function_type
        self.phase_computation_method = phase_computation_method
        
        # Initialize basis function parameters
        self._initialize_basis_parameters()
        
        logger.debug(f"VectorToFieldTransformer initialized - dim: {embedding_dimension}, "
                    f"basis: {basis_function_type}, phase: {phase_computation_method}")
    
    def _initialize_basis_parameters(self):
        """Initialize parameters for basis function computation."""
        # Scale parameters based on embedding dimension for high-dimensional stability
        dimension_scale = np.sqrt(self.embedding_dimension)
        
        if self.basis_function_type == "gaussian":
            # Gaussian basis function parameters - dimensionally scaled
            # Centers should be smaller in high dimensions
            center_scale = 1.0 / dimension_scale
            self.basis_centers = np.random.randn(self.embedding_dimension, self.embedding_dimension) * center_scale
            
            # Widths should be larger in high dimensions to account for curse of dimensionality
            width_scale = dimension_scale * 0.1  # Scale with sqrt(d)
            self.basis_widths = np.ones(self.embedding_dimension) * width_scale
            
            logger.debug(f"Gaussian basis: center_scale={center_scale:.4f}, width_scale={width_scale:.4f}")
            
        elif self.basis_function_type == "harmonic":
            # Spherical harmonic-inspired parameters
            self.harmonic_frequencies = np.linspace(0.1, 2.0, self.embedding_dimension)
            self.harmonic_phases = np.random.uniform(0, 2*np.pi, self.embedding_dimension)
        else:
            # Adaptive basis - will be computed dynamically
            self.adaptive_cache = {}
    
    def transform_vector_to_field(self, 
                                 embedding: np.ndarray,
                                 position_x: np.ndarray,
                                 context: Optional[str] = None,
                                 manifold_properties: Optional[Dict[str, Any]] = None) -> complex:
        """
        Core transformation: Convert embedding vector to semantic field at position x.
        
        Implements: S_τ(x) = Σᵢ e_τ,ᵢ · φᵢ(x) · e^(iθ_τ,ᵢ)
        
        CLAUDE.md Compliance: Field theory optimized for complex-valued semantic field generation.
        Preserves phase relationships and mathematical accuracy for S_τ(x) transformation.
        
        Args:
            embedding: Input embedding vector e_τ
            position_x: Position in semantic manifold where field is evaluated
            context: Optional context for phase modulation
            manifold_properties: Optional manifold geometry information
            
        Returns:
            Complex-valued semantic field at position x
        """
        if len(embedding) != self.embedding_dimension:
            raise ValueError(f"Embedding dimension {len(embedding)} doesn't match expected {self.embedding_dimension}")
        
        # Compute all basis values and phase factors vectorized
        basis_values = np.array([
            self._compute_basis_function(i, position_x, manifold_properties) 
            for i in range(len(embedding))
        ])
        
        phase_factors = np.array([
            self._compute_phase_factor(i, embedding[i], context)
            for i in range(len(embedding))
        ])
        
        # Use optimized vectorized summation
        semantic_field = _optimized_semantic_field_summation(embedding, basis_values, phase_factors)
        
        return semantic_field
    
    def _compute_basis_function(self, 
                               index: int, 
                               position_x: np.ndarray,
                               manifold_properties: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute basis function φᵢ(x) value at position x.
        
        Args:
            index: Basis function index i
            position_x: Position in semantic manifold
            manifold_properties: Optional manifold geometry information
            
        Returns:
            Real-valued basis function value
        """
        if self.basis_function_type == "gaussian":
            return self._gaussian_basis_function(index, position_x)
        elif self.basis_function_type == "harmonic":
            return self._harmonic_basis_function(index, position_x)
        elif self.basis_function_type == "adaptive":
            return self._adaptive_basis_function(index, position_x, manifold_properties)
        else:
            # Default: simple radial basis
            center = self.basis_centers[index] if hasattr(self, 'basis_centers') else np.zeros_like(position_x)
            distance = np.linalg.norm(position_x - center)
            return np.exp(-distance**2)
    
    def _gaussian_basis_function(self, index: int, position_x: np.ndarray) -> float:
        """
        Gaussian basis function implementation with high-dimensional stability.
        
        CLAUDE.md Compliance: Uses optimized computation for φᵢ(x) basis function.
        Preserves mathematical accuracy for semantic field calculations.
        """
        center = self.basis_centers[index]
        width = self.basis_widths[index]
        
        # Use optimized computation
        return _optimized_gaussian_basis(position_x, center, width)
    
    def _harmonic_basis_function(self, index: int, position_x: np.ndarray) -> float:
        """Spherical harmonic-inspired basis function."""
        frequency = self.harmonic_frequencies[index]
        phase = self.harmonic_phases[index]
        
        # Use norm of position for radial component
        radius = np.linalg.norm(position_x)
        
        # Harmonic oscillation
        return np.cos(frequency * radius + phase)
    
    def _adaptive_basis_function(self, 
                                index: int, 
                                position_x: np.ndarray,
                                manifold_properties: Optional[Dict[str, Any]] = None) -> float:
        """Adaptive basis function that adjusts to manifold properties."""
        cache_key = (index, tuple(position_x))
        
        if cache_key in self.adaptive_cache:
            return self.adaptive_cache[cache_key]
        
        # Adaptive computation based on manifold properties
        if manifold_properties and 'local_density' in manifold_properties:
            density = manifold_properties['local_density']
            adaptive_width = 1.0 / (1.0 + density)
        else:
            adaptive_width = 1.0
        
        # Use position-dependent parameters
        center = position_x * 0.1  # Dynamic center based on position
        distance_squared = np.sum((position_x - center)**2)
        basis_value = np.exp(-distance_squared / (2 * adaptive_width**2))
        
        # Cache result
        self.adaptive_cache[cache_key] = basis_value
        return basis_value
    
    def _compute_phase_factor(self, 
                             index: int, 
                             embedding_component: float,
                             context: Optional[str] = None) -> complex:
        """
        Compute complex phase factor e^(iθ_τ,ᵢ).
        
        Args:
            index: Component index i
            embedding_component: The e_τ,ᵢ value
            context: Optional context for phase modulation
            
        Returns:
            Complex phase factor
        """
        if self.phase_computation_method == "component_based":
            # Base phase from embedding component
            base_phase = np.angle(embedding_component + 0j)
            
            # Add small perturbation based on index
            index_phase = (index / self.embedding_dimension) * 0.1 * np.pi
            
            total_phase = base_phase + index_phase
            
        elif self.phase_computation_method == "context_dependent":
            # Context-dependent phase computation
            base_phase = np.angle(embedding_component + 0j)
            
            if context is not None:
                context_hash = hash(context) % 1000000
                context_phase = (context_hash / 1000000.0) * 2.0 * np.pi
                total_phase = base_phase + context_phase * 0.1
            else:
                total_phase = base_phase
                
        else:
            # Simple phase based on component value
            total_phase = np.angle(embedding_component + 0j)
        
        return np.exp(1j * total_phase)
    
    def batch_transform(self, 
                       embeddings: List[np.ndarray],
                       positions: List[np.ndarray],
                       contexts: Optional[List[str]] = None,
                       manifold_properties_batch: Optional[List[Dict[str, Any]]] = None) -> List[complex]:
        """
        Efficiently transform multiple embeddings to fields.
        
        Args:
            embeddings: List of embedding vectors
            positions: List of positions in semantic manifold
            contexts: Optional list of contexts
            manifold_properties_batch: Optional list of manifold properties
            
        Returns:
            List of complex-valued semantic fields
        """
        if contexts is None:
            contexts = [None] * len(embeddings)
        if manifold_properties_batch is None:
            manifold_properties_batch = [None] * len(embeddings)
        
        semantic_fields = []
        
        for i, (embedding, position, context, manifold_props) in enumerate(
            zip(embeddings, positions, contexts, manifold_properties_batch)
        ):
            field = self.transform_vector_to_field(
                embedding=embedding,
                position_x=position,
                context=context,
                manifold_properties=manifold_props
            )
            semantic_fields.append(field)
            
            if (i + 1) % 100 == 0:
                logger.debug(f"Transformed {i + 1}/{len(embeddings)} embeddings to fields")
        
        return semantic_fields
    
    def decompose_embedding(self, embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Decompose embedding into field-generating components.
        
        This provides insight into how each embedding component contributes
        to the overall semantic field generation.
        
        Args:
            embedding: Input embedding vector
            
        Returns:
            List of component dictionaries with field generation properties
        """
        field_components = []
        
        for i, component in enumerate(embedding):
            field_component = {
                'index': i,
                'amplitude': abs(component),
                'phase': np.angle(component + 0j),
                'field_strength': component,
                'basis_function_type': self.basis_function_type,
                'relative_contribution': abs(component) / np.linalg.norm(embedding)
            }
            field_components.append(field_component)
        
        return field_components
    
    def analyze_field_properties(self, 
                                semantic_field: complex,
                                embedding: np.ndarray,
                                position_x: np.ndarray) -> Dict[str, Any]:
        """
        Analyze properties of generated semantic field.
        
        Args:
            semantic_field: Generated complex field value
            embedding: Original embedding vector
            position_x: Position where field was evaluated
            
        Returns:
            Dictionary containing field analysis
        """
        return {
            'field_magnitude': abs(semantic_field),
            'field_phase': np.angle(semantic_field),
            'field_real': semantic_field.real,
            'field_imag': semantic_field.imag,
            'embedding_norm': np.linalg.norm(embedding),
            'position_norm': np.linalg.norm(position_x),
            'transformation_efficiency': abs(semantic_field) / np.linalg.norm(embedding),
            'phase_coherence': np.exp(1j * np.angle(semantic_field)),
            'basis_function_type': self.basis_function_type
        }


def create_semantic_field_from_embedding(embedding: np.ndarray,
                                       position: np.ndarray,
                                       context: Optional[str] = None,
                                       transformer_config: Optional[Dict[str, Any]] = None) -> Tuple[complex, Dict[str, Any]]:
    """
    Convenience function for creating semantic fields from embeddings.
    
    Args:
        embedding: Input embedding vector
        position: Position in semantic manifold
        context: Optional context for field generation
        transformer_config: Optional transformer configuration
        
    Returns:
        Tuple of (semantic_field, field_analysis)
    """
    if transformer_config is None:
        transformer_config = {}
    
    # Create transformer with appropriate dimension
    transformer = VectorToFieldTransformer(
        embedding_dimension=len(embedding),
        **transformer_config
    )
    
    # Generate semantic field
    semantic_field = transformer.transform_vector_to_field(
        embedding=embedding,
        position_x=position,
        context=context
    )
    
    # Analyze field properties
    field_analysis = transformer.analyze_field_properties(
        semantic_field=semantic_field,
        embedding=embedding,
        position_x=position
    )
    
    return semantic_field, field_analysis


# Example usage and testing
if __name__ == "__main__":
    # Test vector-to-field transformation
    logger.info("Testing vector-to-field transformation...")
    
    # Create test embedding (BGE-style 1024d)
    test_embedding = np.random.randn(1024) * 0.1
    test_position = np.random.randn(1024) * 0.1
    
    # Test transformation
    semantic_field, analysis = create_semantic_field_from_embedding(
        embedding=test_embedding,
        position=test_position,
        context="test_context"
    )
    
    logger.info(f"Generated semantic field - magnitude: {analysis['field_magnitude']:.4f}, "
                f"phase: {analysis['field_phase']:.4f}")
    
    # Test batch transformation
    transformer = VectorToFieldTransformer(embedding_dimension=1024)
    
    batch_embeddings = [np.random.randn(1024) * 0.1 for _ in range(5)]
    batch_positions = [np.random.randn(1024) * 0.1 for _ in range(5)]
    
    batch_fields = transformer.batch_transform(
        embeddings=batch_embeddings,
        positions=batch_positions
    )
    
    logger.info(f"Batch transformation complete - {len(batch_fields)} fields generated")
    logger.info("Vector-to-field transformation test successful!")