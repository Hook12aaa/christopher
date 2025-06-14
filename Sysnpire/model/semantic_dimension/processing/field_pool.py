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
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


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
    fields, enabling batch processing and field interactions.
    """
    
    def __init__(self, 
                 pool_capacity: int = 1000,
                 embedding_dimension: int = 1024,
                 auto_process: bool = True):
        """
        Initialize the semantic field pool.
        
        Args:
            pool_capacity: Maximum number of entries in the pool
            embedding_dimension: Dimension of embeddings (1024 for BGE, 768 for MPNet)
            auto_process: Whether to automatically process entries as they're added
        """
        self.pool_capacity = pool_capacity
        self.embedding_dimension = embedding_dimension
        self.auto_process = auto_process
        
        # Pool storage
        self.pool: deque = deque(maxlen=pool_capacity)
        self.processed_count = 0
        self.total_added = 0
        
        # Field transformer
        self.transformer = VectorToFieldTransformer(
            embedding_dimension=embedding_dimension,
            basis_function_type="gaussian"
        )
        
        # Pool statistics
        self.stats = {
            'total_added': 0,
            'total_processed': 0,
            'total_returned': 0,
            'average_field_magnitude': 0.0,
            'processing_errors': 0
        }
        
        logger.info(f"SemanticFieldPool initialized - capacity: {pool_capacity}, "
                   f"dimension: {embedding_dimension}, auto_process: {auto_process}")
    
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
        
        Args:
            entry: The pool entry to process
            
        Returns:
            bool: True if processing succeeded
        """
        try:
            # Generate position if not provided
            if entry.position is None:
                entry.position = self._generate_default_position()
            
            # Apply S_τ(x) transformation
            context = entry.metadata.get('context', entry.token)
            
            entry.semantic_field = self.transformer.transform_vector_to_field(
                embedding=entry.embedding,
                position_x=entry.position,
                context=context,
                manifold_properties=entry.metadata.get('manifold_properties')
            )
            
            entry.processed = True
            self.processed_count += 1
            self.stats['total_processed'] += 1
            
            # Update statistics
            field_magnitude = abs(entry.semantic_field)
            current_avg = self.stats['average_field_magnitude']
            n = self.stats['total_processed']
            self.stats['average_field_magnitude'] = ((current_avg * (n-1)) + field_magnitude) / n
            
            logger.debug(f"Processed '{entry.token}' → field magnitude: {field_magnitude:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process entry '{entry.token}': {e}")
            self.stats['processing_errors'] += 1
            return False
    
    def _generate_default_position(self) -> np.ndarray:
        """Generate a default position for field evaluation."""
        # Use small random position around origin
        return np.random.randn(self.embedding_dimension) * 0.1
    
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
                                   pool_config: Optional[Dict[str, Any]] = None) -> SemanticFieldPool:
    """
    Create and populate a field pool from manifold data.
    
    Args:
        manifold_data: Manifold data containing embeddings
        pool_config: Optional pool configuration
        
    Returns:
        Populated SemanticFieldPool
    """
    if pool_config is None:
        pool_config = {}
    
    # Determine embedding dimension
    embeddings = manifold_data.get('embeddings', [])
    if embeddings:
        embedding_dim = len(embeddings[0])
    else:
        embedding_dim = manifold_data.get('embedding_dim', 1024)
    
    # Create pool
    pool = SemanticFieldPool(
        embedding_dimension=embedding_dim,
        **pool_config
    )
    
    # Add embeddings
    pool.batch_add_from_manifold(manifold_data)
    
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