"""
Semantic Field Pool - Embedding Processing Pipeline

This module implements a field pool where embeddings can be added, transformed
through semantic field operations, and then sent back to the manifold. It serves
as an intermediate processing layer between the raw manifold and the final
conceptual charges.

PIPELINE FLOW:
Manifold → Field-Pool → S_τ(x) Transform → Field Effects → Back to Manifold

The field pool allows for:
- Batch processing of embeddings
- Field interaction and interference 
- Dynamic field evolution
- Efficient memory management
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass
from collections import deque
import sys
from pathlib import Path

# Ensure project imports work
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.model.semantic_dimension.vector_transformation import VectorToFieldTransformer
from Sysnpire.model.semantic_dimension.semantic_basis_functions import DTFSemanticBasisExtractor, create_dtf_semantic_basis_from_bge
from Sysnpire.utils.logger import get_logger
from Sysnpire.utils.field_theory_optimizers import (
    field_theory_jax_optimize, field_theory_numba_optimize, 
    field_theory_auto_optimize
)

logger = get_logger(__name__)


@field_theory_auto_optimize(prefer_accuracy=True, profile=True)
def _optimized_dtf_field_computation(embedding_components: np.ndarray, 
                                   basis_values: np.ndarray, 
                                   phase_factors: np.ndarray) -> complex:
    """
    Optimized DTF semantic field computation for S_τ(x) = Σᵢ e_τ,ᵢ · φᵢ(x) · e^(iθ_τ,ᵢ).
    
    CLAUDE.md Compliance: Field theory optimized for DTF semantic field transformation.
    Preserves complex-valued mathematics and phase relationships.
    """
    # Vectorized DTF field component computation
    field_components = embedding_components * basis_values * phase_factors
    
    # Sum all components for complete semantic field
    semantic_field = np.sum(field_components)
    
    return semantic_field


@field_theory_numba_optimize(preserve_complex=False, profile=True)
def _optimized_field_magnitude_statistics(field_magnitudes: np.ndarray, 
                                         current_avg: float, 
                                         total_count: int) -> float:
    """
    Optimized running average computation for field magnitude statistics.
    
    CLAUDE.md Compliance: Field theory optimized statistical computation.
    Preserves mathematical accuracy for performance tracking.
    """
    if total_count <= 0:
        return 0.0
    
    # Compute new running average efficiently
    total_magnitude = 0.0
    for i in range(len(field_magnitudes)):
        total_magnitude += field_magnitudes[i]
    
    # Update running average
    if total_count == 1:
        return total_magnitude / len(field_magnitudes)
    else:
        new_avg = ((current_avg * (total_count - 1)) + total_magnitude) / total_count
        return new_avg


@field_theory_jax_optimize(preserve_complex=True, profile=True) 
def _optimized_batch_phase_computation(embedding_components: np.ndarray) -> np.ndarray:
    """
    Optimized batch phase factor computation for multiple embeddings.
    
    CLAUDE.md Compliance: Field theory optimized for complex phase relationships.
    Preserves phase information for field theory calculations.
    """
    # Compute phase factors for embedding components
    # Add small epsilon to avoid division by zero
    complex_components = embedding_components + 1e-12j
    phase_factors = np.exp(1j * np.angle(complex_components))
    
    return phase_factors


@dataclass
class FieldPoolEntry:
    """Individual entry in the field pool."""
    embedding: np.ndarray
    token: str
    index: int
    metadata: Dict[str, Any]
    position: Optional[np.ndarray] = None
    semantic_field: Optional[complex] = None
    processed: bool = False
    

class SemanticFieldPool:
    """
    Field pool for processing embeddings through semantic field transformations.
    
    This pool manages the pipeline from raw embeddings to transformed semantic
    fields, enabling batch processing and field interactions using DTF-based
    semantic basis functions derived from BGE's learned structure.
    """
    
    def __init__(self, 
                 pool_capacity: int = 1000,
                 embedding_dimension: int = 1024,
                 auto_process: bool = True,
                 use_dtf_basis: bool = True,
                 all_embeddings: Optional[np.ndarray] = None,
                 id_to_token: Optional[Dict[int, str]] = None):
        """
        Initialize the semantic field pool.
        
        Args:
            pool_capacity: Maximum number of entries in the pool
            embedding_dimension: Dimension of embeddings (1024 for BGE, 768 for MPNet)
            auto_process: Whether to automatically process entries as they're added
            use_dtf_basis: Whether to use DTF semantic basis functions
            all_embeddings: Complete embedding matrix for DTF basis extraction
            id_to_token: Token mapping for semantic analysis
        """
        self.pool_capacity = pool_capacity
        self.embedding_dimension = embedding_dimension
        self.auto_process = auto_process
        self.use_dtf_basis = use_dtf_basis
        
        # Pool storage
        self.pool: deque = deque(maxlen=pool_capacity)
        self.processed_count = 0
        self.total_added = 0
        
        # Pool statistics (initialize before transformers)
        self.stats = {
            'total_added': 0,
            'total_processed': 0,
            'total_returned': 0,
            'average_field_magnitude': 0.0,
            'processing_errors': 0,
            'dtf_basis_used': use_dtf_basis,
            'semantic_basis_functions': 0
        }
        
        # Initialize transformers and basis functions
        self._initialize_transformers(all_embeddings, id_to_token)
        
        logger.info(f"SemanticFieldPool initialized - capacity: {pool_capacity}, "
                   f"dimension: {embedding_dimension}, DTF basis: {use_dtf_basis}")
    
    def _initialize_transformers(self, 
                               all_embeddings: Optional[np.ndarray],
                               id_to_token: Optional[Dict[int, str]]):
        """Initialize field transformers and DTF semantic basis functions."""
        # Standard vector-to-field transformer
        self.transformer = VectorToFieldTransformer(
            embedding_dimension=self.embedding_dimension,
            basis_function_type="gaussian"
        )
        
        # DTF semantic basis extractor and functions
        self.dtf_extractor = None
        self.semantic_basis_set = None
        
        if self.use_dtf_basis and all_embeddings is not None and id_to_token is not None:
            try:
                logger.info("Initializing DTF semantic basis functions from BGE structure...")
                
                # Create DTF semantic basis extractor
                self.dtf_extractor = DTFSemanticBasisExtractor(
                    embedding_dimension=self.embedding_dimension
                )
                
                # Generate semantic basis set from BGE embeddings
                num_basis_functions = min(64, len(all_embeddings), self.embedding_dimension // 16)
                self.semantic_basis_set = self.dtf_extractor.generate_semantic_basis_set(
                    all_embeddings=all_embeddings,
                    id_to_token=id_to_token,
                    num_basis_functions=num_basis_functions
                )
                
                self.stats['semantic_basis_functions'] = self.semantic_basis_set['num_functions']
                logger.info(f"Generated {self.semantic_basis_set['num_functions']} DTF semantic basis functions")
                
            except Exception as e:
                logger.warning(f"Failed to initialize DTF basis functions: {e}")
                logger.info("Falling back to standard basis functions")
                self.use_dtf_basis = False
                self.stats['dtf_basis_used'] = False
        else:
            if self.use_dtf_basis:
                logger.warning("DTF basis requested but missing embeddings/tokens - using standard basis")
                self.use_dtf_basis = False
                self.stats['dtf_basis_used'] = False
    
    def add_embedding(self, 
                     embedding: np.ndarray,
                     token: str,
                     index: int,
                     metadata: Optional[Dict[str, Any]] = None,
                     position: Optional[np.ndarray] = None) -> bool:
        """
        Add an embedding to the field pool for processing.
        
        Args:
            embedding: The embedding vector to add
            token: Token/concept this embedding represents
            index: Original index in the manifold
            metadata: Optional metadata about the embedding
            position: Optional position in semantic space
            
        Returns:
            bool: True if successfully added, False if pool is full
        """
        if len(embedding) != self.embedding_dimension:
            logger.warning(f"Embedding dimension {len(embedding)} doesn't match pool dimension {self.embedding_dimension}")
            return False
        
        if metadata is None:
            metadata = {}
        
        # Create pool entry
        entry = FieldPoolEntry(
            embedding=embedding.copy(),
            token=token,
            index=index,
            metadata=metadata,
            position=position.copy() if position is not None else None
        )
        
        # Add to pool
        self.pool.append(entry)
        self.total_added += 1
        self.stats['total_added'] += 1
        
        logger.debug(f"Added embedding for '{token}' (index {index}) to field pool")
        
        # Auto-process if enabled
        if self.auto_process:
            self._process_latest_entry()
        
        return True
    
    def _process_latest_entry(self) -> bool:
        """Process the most recently added entry."""
        if not self.pool:
            return False
        
        entry = self.pool[-1]
        if entry.processed:
            return True
        
        return self._process_entry(entry)
    
    def _process_entry(self, entry: FieldPoolEntry) -> bool:
        """
        Process a single entry through semantic field transformation.
        
        Uses either DTF semantic basis functions or standard vector-to-field
        transformation depending on configuration.
        
        Args:
            entry: The pool entry to process
            
        Returns:
            bool: True if processing succeeded
        """
        try:
            # Generate position if not provided
            if entry.position is None:
                entry.position = self._generate_default_position()
            
            context = entry.metadata.get('context', entry.token)
            
            # Choose processing method based on available DTF basis functions
            if self.use_dtf_basis and self.semantic_basis_set is not None:
                entry.semantic_field = self._process_with_dtf_basis(entry)
            else:
                # Fallback to standard vector-to-field transformation
                entry.semantic_field = self.transformer.transform_vector_to_field(
                    embedding=entry.embedding,
                    position_x=entry.position,
                    context=context,
                    manifold_properties=entry.metadata.get('manifold_properties')
                )
            
            entry.processed = True
            self.processed_count += 1
            self.stats['total_processed'] += 1
            
            # Update statistics using optimized computation
            field_magnitude = abs(entry.semantic_field)
            current_avg = self.stats['average_field_magnitude']
            n = self.stats['total_processed']
            
            # Use optimized statistics computation for performance
            self.stats['average_field_magnitude'] = _optimized_field_magnitude_statistics(
                field_magnitudes=np.array([field_magnitude]),
                current_avg=current_avg,
                total_count=n
            )
            
            processing_method = "DTF" if self.use_dtf_basis else "standard"
            logger.debug(f"Processed '{entry.token}' ({processing_method}) → field magnitude: {field_magnitude:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process entry '{entry.token}': {e}")
            self.stats['processing_errors'] += 1
            return False
    
    def _process_with_dtf_basis(self, entry: FieldPoolEntry) -> complex:
        """
        Process entry using DTF semantic basis functions.
        
        This implements the enhanced S_τ(x) transformation using semantic basis
        functions φᵢ(x) derived from BGE's learned structure via DTF.
        
        Args:
            entry: Pool entry to process
            
        Returns:
            Complex semantic field value
        """
        try:
            basis_functions = self.semantic_basis_set['basis_functions']
            
            # Prepare vectorized computation for efficiency
            num_basis = min(len(basis_functions), len(entry.embedding))
            embedding_components = np.zeros(num_basis)
            basis_values = np.zeros(num_basis)
            
            # Collect data for vectorized computation
            basis_indices = list(basis_functions.keys())[:num_basis]
            for i, basis_idx in enumerate(basis_indices):
                if basis_idx >= len(entry.embedding):
                    break
                    
                # e_τ,ᵢ: embedding component
                embedding_components[i] = entry.embedding[basis_idx]
                
                # φᵢ(x): DTF semantic basis function value
                basis_function = basis_functions[basis_idx]['function']
                basis_values[i] = basis_function(entry.position)
            
            # Use optimized batch phase computation
            phase_factors = _optimized_batch_phase_computation(embedding_components)
            
            # Use optimized DTF field computation
            semantic_field = _optimized_dtf_field_computation(
                embedding_components=embedding_components,
                basis_values=basis_values,
                phase_factors=phase_factors
            )
            
            # Apply semantic field normalization
            if abs(semantic_field) > 0:
                # Normalize by number of basis functions used
                semantic_field = semantic_field / np.sqrt(num_basis)
            
            return semantic_field
            
        except Exception as e:
            logger.warning(f"DTF processing failed for '{entry.token}': {e}")
            # Fallback to standard transformation
            return self.transformer.transform_vector_to_field(
                embedding=entry.embedding,
                position_x=entry.position,
                context=entry.metadata.get('context', entry.token),
                manifold_properties=entry.metadata.get('manifold_properties')
            )
    
    def _generate_default_position(self) -> np.ndarray:
        """Generate a default position for field evaluation."""
        # Use origin as default position - no random data per CLAUDE.md
        return np.zeros(self.embedding_dimension)
    
    def process_all(self) -> int:
        """
        Process all unprocessed entries in the pool.
        
        Returns:
            int: Number of entries successfully processed
        """
        processed_count = 0
        
        for entry in self.pool:
            if not entry.processed:
                if self._process_entry(entry):
                    processed_count += 1
        
        logger.info(f"Processed {processed_count} entries in field pool")
        return processed_count
    
    def get_processed_fields(self, 
                           limit: Optional[int] = None,
                           include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Get processed semantic fields from the pool.
        
        Args:
            limit: Maximum number of fields to return
            include_metadata: Whether to include original metadata
            
        Returns:
            List of processed field data
        """
        processed_fields = []
        count = 0
        
        for entry in self.pool:
            if entry.processed and entry.semantic_field is not None:
                field_data = {
                    'token': entry.token,
                    'index': entry.index,
                    'semantic_field': entry.semantic_field,
                    'field_magnitude': abs(entry.semantic_field),
                    'field_phase': np.angle(entry.semantic_field),
                    'embedding': entry.embedding,
                    'position': entry.position
                }
                
                if include_metadata:
                    field_data['metadata'] = entry.metadata
                
                processed_fields.append(field_data)
                count += 1
                
                if limit is not None and count >= limit:
                    break
        
        return processed_fields
    
    def return_to_manifold(self) -> Iterator[Dict[str, Any]]:
        """
        Generator that yields processed fields for return to manifold.
        
        This removes processed entries from the pool and yields them
        for integration back into the main manifold system.
        
        Yields:
            Dict containing processed field data ready for manifold integration
        """
        returned_count = 0
        
        # Process from the left (oldest entries first)
        while self.pool:
            entry = self.pool.popleft()
            
            if entry.processed and entry.semantic_field is not None:
                returned_count += 1
                self.stats['total_returned'] += 1
                
                # Package for manifold return
                manifold_data = {
                    'token': entry.token,
                    'original_index': entry.index,
                    'original_embedding': entry.embedding,
                    'transformed_field': entry.semantic_field,
                    'field_magnitude': abs(entry.semantic_field),
                    'field_phase': np.angle(entry.semantic_field),
                    'processing_position': entry.position,
                    'metadata': entry.metadata,
                    'pool_processing': True
                }
                
                logger.debug(f"Returning '{entry.token}' to manifold - field magnitude: {abs(entry.semantic_field):.4f}")
                yield manifold_data
            else:
                # Re-add unprocessed entries to the end
                self.pool.append(entry)
                break
        
        if returned_count > 0:
            logger.info(f"Returned {returned_count} processed fields to manifold")
    
    def batch_add_from_manifold(self, 
                               manifold_data: Dict[str, Any],
                               indices: Optional[List[int]] = None,
                               limit: Optional[int] = None) -> int:
        """
        Add embeddings from manifold data in batch.
        
        Args:
            manifold_data: Manifold data containing embeddings and metadata
            indices: Specific indices to add (if None, adds all)
            limit: Maximum number to add
            
        Returns:
            int: Number of embeddings successfully added
        """
        embeddings = manifold_data.get('embeddings', [])
        id_to_token = manifold_data.get('id_to_token', {})
        
        if indices is None:
            indices = list(range(len(embeddings)))
        
        if limit is not None:
            indices = indices[:limit]
        
        added_count = 0
        
        for i in indices:
            if i >= len(embeddings):
                continue
            
            embedding = embeddings[i]
            token = id_to_token.get(i, f"token_{i}")
            
            metadata = {
                'source': 'manifold_batch',
                'manifold_index': i,
                'vocab_size': manifold_data.get('vocab_size'),
                'embedding_dim': manifold_data.get('embedding_dim')
            }
            
            if self.add_embedding(embedding, token, i, metadata):
                added_count += 1
            else:
                logger.warning(f"Failed to add embedding {i} to pool")
                break
        
        logger.info(f"Added {added_count} embeddings from manifold to field pool")
        return added_count
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        current_size = len(self.pool)
        processed_in_pool = sum(1 for entry in self.pool if entry.processed)
        
        return {
            **self.stats,
            'current_pool_size': current_size,
            'processed_in_pool': processed_in_pool,
            'unprocessed_in_pool': current_size - processed_in_pool,
            'pool_capacity': self.pool_capacity,
            'embedding_dimension': self.embedding_dimension,
            'processing_rate': self.stats['total_processed'] / max(1, self.stats['total_added']),
            'average_field_magnitude': self.stats['average_field_magnitude']
        }
    
    def clear_pool(self):
        """Clear all entries from the pool."""
        cleared_count = len(self.pool)
        self.pool.clear()
        self.processed_count = 0
        logger.info(f"Cleared {cleared_count} entries from field pool")


# Convenience functions for integration
def create_field_pool_from_manifold(manifold_data: Dict[str, Any],
                                   pool_config: Optional[Dict[str, Any]] = None,
                                   use_dtf_basis: bool = True) -> SemanticFieldPool:
    """
    Create and populate a field pool from manifold data with DTF semantic basis functions.
    
    Args:
        manifold_data: Manifold data containing embeddings
        pool_config: Optional pool configuration
        use_dtf_basis: Whether to use DTF semantic basis functions
        
    Returns:
        Populated SemanticFieldPool with DTF semantic basis functions
    """
    if pool_config is None:
        pool_config = {}
    
    # Determine embedding dimension
    embeddings = manifold_data.get('embeddings', [])
    if embeddings:
        embedding_dim = len(embeddings[0])
    else:
        embedding_dim = manifold_data.get('embedding_dim', 1024)
    
    # Extract embeddings and token mapping for DTF basis
    all_embeddings = manifold_data.get('embeddings')
    id_to_token = manifold_data.get('id_to_token', {})
    
    # Create pool with DTF capability
    pool = SemanticFieldPool(
        embedding_dimension=embedding_dim,
        use_dtf_basis=use_dtf_basis,
        all_embeddings=all_embeddings,
        id_to_token=id_to_token,
        **pool_config
    )
    
    # Add embeddings
    pool.batch_add_from_manifold(manifold_data)
    
    return pool


def create_dtf_field_pool_with_basis_extraction(manifold_data: Dict[str, Any],
                                               num_basis_functions: Optional[int] = None,
                                               pool_config: Optional[Dict[str, Any]] = None) -> SemanticFieldPool:
    """
    Create field pool with explicit DTF basis function extraction.
    
    This function ensures DTF semantic basis functions are generated from the
    complete BGE embedding manifold before processing begins.
    
    Args:
        manifold_data: Complete manifold data with embeddings and tokens
        num_basis_functions: Number of DTF basis functions to generate
        pool_config: Optional pool configuration
        
    Returns:
        Field pool with DTF semantic basis functions ready for processing
    """
    if pool_config is None:
        pool_config = {}
    
    # Validate manifold data
    embeddings = manifold_data.get('embeddings')
    id_to_token = manifold_data.get('id_to_token', {})
    
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Manifold data must contain embeddings for DTF basis extraction")
    
    embedding_dim = len(embeddings[0])
    
    logger.info(f"Creating DTF field pool from manifold with {len(embeddings)} embeddings")
    
    # Create pool with DTF enabled
    pool = SemanticFieldPool(
        embedding_dimension=embedding_dim,
        use_dtf_basis=True,
        all_embeddings=embeddings,
        id_to_token=id_to_token,
        **pool_config
    )
    
    # Verify DTF basis functions were created successfully
    if pool.semantic_basis_set is None:
        logger.warning("DTF basis function extraction failed - using standard basis")
    else:
        logger.info(f"DTF field pool ready with {pool.semantic_basis_set['num_functions']} basis functions")
    
    return pool


# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing Semantic Field Pool...")
    
    # Create test pool
    pool = SemanticFieldPool(pool_capacity=10, embedding_dimension=4)
    
    # Add test embeddings
    test_embeddings = [
        (np.array([1.0, 0.5, -0.3, 0.8]), "democracy", 0),
        (np.array([0.7, -0.2, 0.9, 0.1]), "freedom", 1),
        (np.array([-0.1, 0.8, 0.3, -0.6]), "justice", 2)
    ]
    
    for embedding, token, idx in test_embeddings:
        pool.add_embedding(embedding, token, idx)
    
    # Process all
    processed = pool.process_all()
    logger.info(f"Processed {processed} embeddings")
    
    # Get results
    fields = pool.get_processed_fields()
    for field in fields:
        logger.info(f"'{field['token']}' → magnitude: {field['field_magnitude']:.4f}")
    
    # Test return to manifold
    logger.info("Returning to manifold:")
    for manifold_data in pool.return_to_manifold():
        logger.info(f"Returned '{manifold_data['token']}' with field magnitude {manifold_data['field_magnitude']:.4f}")
    
    # Print statistics
    stats = pool.get_pool_statistics()
    logger.info(f"Pool statistics: {stats}")
    
    logger.info("Field Pool test complete!")