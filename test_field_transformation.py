#!/usr/bin/env python3
"""
Test script for the field transformation pipeline.
This tests the main components: vector transformation, semantic field generation, and field pool.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from Sysnpire.model.semantic_dimension.vector_transformation import VectorTransformation
from Sysnpire.model.semantic_dimension.SemanticDimensionHelper import SemanticDimensionHelper
from Sysnpire.model.semantic_dimension.processing.field_pool import SemanticFieldPool
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)

def test_vector_transformation():
    """Test the basic vector transformation functionality."""
    logger.info("Testing Vector Transformation...")
    
    # Create a simple test embedding
    test_embedding = np.array([0.5, -0.3, 0.8, -0.1, 0.2])
    
    # Create vector transformer
    transformer = VectorTransformation(
        from_base=True,
        embedding_dimension=5,
        helper=None,
        phase_computation_method="component_based"
    )
    
    # Test embedding data
    embedding_data = {
        'embedding_vector': test_embedding,
        'token': 'test_token',
        'similarity': 0.95
    }
    
    # Transform to field
    field_result = transformer.model_transform_to_field(embedding_data)
    
    logger.info(f"Field transformation successful: {field_result['transformation_method']}")
    logger.info(f"Field dimension: {field_result['field_dimension']}")
    logger.info(f"Number of field components: {len(field_result['field_components'])}")
    
    # Test field evaluation at a position
    position = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    field_value = transformer.evaluate_semantic_field_at_position(field_result, position)
    
    logger.info(f"Field value at position: {field_value} (magnitude: {abs(field_value):.4f})")
    
    return True

def test_semantic_helper():
    """Test the semantic dimension helper."""
    logger.info("Testing Semantic Dimension Helper...")
    
    # Mock model info
    model_info = {
        'dimension': 5,
        'model_name': 'test_model'
    }
    
    # Create helper
    helper = SemanticDimensionHelper(
        from_base=True,
        model_info=model_info,
        helper=None
    )
    
    # Test embedding data
    test_embeddings = [
        {
            'embedding_vector': np.array([0.5, -0.3, 0.8, -0.1, 0.2]),
            'token': 'democracy',
            'similarity': 0.95
        },
        {
            'embedding_vector': np.array([0.2, 0.7, -0.4, 0.6, -0.3]),
            'token': 'freedom',
            'similarity': 0.87
        }
    ]
    
    # Convert to field representation
    field_results = helper.convert_vector_to_field_respentation(test_embeddings)
    
    logger.info(f"Semantic helper processed {field_results['total_processed']} embeddings")
    logger.info(f"Success: {field_results['success']}")
    
    return True

def test_field_pool():
    """Test the semantic field pool."""
    logger.info("Testing Semantic Field Pool...")
    
    # Create field pool
    pool = SemanticFieldPool(
        pool_capacity=10,
        embedding_dimension=5,
        auto_process=True,
        use_dtf_basis=False
    )
    
    # Test embeddings
    test_embeddings = [
        (np.array([0.5, -0.3, 0.8, -0.1, 0.2]), "democracy", 0),
        (np.array([0.2, 0.7, -0.4, 0.6, -0.3]), "freedom", 1),
        (np.array([-0.1, 0.4, 0.9, -0.5, 0.1]), "justice", 2)
    ]
    
    # Add embeddings to pool
    for embedding, token, idx in test_embeddings:
        metadata = {'context': f'political_concept_{token}'}
        pool.add_embedding(embedding, token, idx, metadata)
    
    # Process all
    processed_count = pool.process_all()
    logger.info(f"Processed {processed_count} embeddings in pool")
    
    # Get results
    processed_fields = pool.get_processed_fields()
    for field in processed_fields:
        logger.info(f"'{field['token']}' → field magnitude: {field['field_magnitude']:.4f}")
    
    # Get statistics
    stats = pool.get_pool_statistics()
    logger.info(f"Pool statistics - avg magnitude: {stats['average_field_magnitude']:.4f}")
    
    return True

def main():
    """Main test function."""
    logger.info("Starting field transformation pipeline tests...")
    
    try:
        # Test individual components
        test_vector_transformation()
        test_semantic_helper()
        test_field_pool()
        
        logger.info("✅ All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()