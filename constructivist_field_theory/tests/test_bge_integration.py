"""
Simple test for the BGE model and basic hyperbolic projection.

This script tests the BGE model loading and basic embedding functionality
without the more complex visualization components.
"""

import os
import sys
import numpy as np
import mlx.core as mx

# Add the project root to the path
# Using the parent directory of the current script's directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embedding_engine.models.bge_model import BGEModel
from embedding_engine.transformations.hyperbolic_projector import HyperbolicProjector

def test_bge_model_connection():
    """Test the basic connection to the BGE model."""
    
    print("\n---- Testing BGE Model Connection ----")
    
    # Initialize the BGE model with explicit parameters
    print("Initializing BGE model...")
    bge_model = BGEModel(
        model_name="BAAI/bge-large-v1.5",
        device=None,  # Let it choose the best device automatically
        cache_dir="./model_cache" 
    )
    
    # Print model information
    model_info = bge_model.get_model_info()
    print(f"Model type: {model_info['model_type']}")
    print(f"Model name: {model_info['model_name']}")
    print(f"Embedding dimension: {model_info['embedding_dim']}")
    print(f"Device: {model_info['device']}")
    
    # Test encoding with simple texts
    sample_texts = ["This is a test sentence.", "Another example text."]
    print(f"\nEncoding {len(sample_texts)} sample texts...")
    
    embeddings = bge_model.encode(sample_texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test that embeddings are normalized (if using the proper model)
    if bge_model.model_type != "simple":
        norms = mx.sqrt(mx.sum(embeddings * embeddings, axis=1))
        print(f"Embedding norms: min={mx.min(norms).item():.4f}, max={mx.max(norms).item():.4f}")
        print("Embeddings are properly normalized." if mx.all(mx.abs(norms - 1.0) < 0.01) else 
              "Warning: Embeddings are not normalized as expected.")
    
    print("\nBGE model test completed successfully!\n")
    return bge_model

def test_hyperbolic_projection(bge_model):
    """Test basic hyperbolic projection with the BGE model."""
    
    print("---- Testing Hyperbolic Projection ----")
    
    # Create a small hyperbolic projector (reduced dimensions for quicker testing)
    input_dim = bge_model.get_embedding_dim()
    output_dim = 128  # Reduced for testing
    
    print(f"Creating hyperbolic projector: {input_dim} → {output_dim} dimensions")
    projector = HyperbolicProjector(
        input_dim=input_dim,
        output_dim=output_dim,
        curvature=-1.0,
        mixed_curvature=False
    )
    
    # Test with a few simple texts
    test_texts = [
        "Hyperbolic geometry",
        "Poincaré ball model", 
        "Möbius transformation"
    ]
    
    print(f"\nGenerating embeddings for {len(test_texts)} test texts...")
    embeddings = bge_model.encode(test_texts)
    
    print("Projecting embeddings to hyperbolic space...")
    hyperbolic_points = projector.project(embeddings)
    
    # Verify the points are in the Poincaré ball (norm < 1)
    norms = mx.sqrt(mx.sum(hyperbolic_points * hyperbolic_points, axis=1))
    print(f"Hyperbolic point norms: min={mx.min(norms).item():.4f}, max={mx.max(norms).item():.4f}")
    print("All points are inside the Poincaré ball." if mx.all(norms < 1.0) else
          "Warning: Some points are outside the Poincaré ball!")
    
    # Calculate distances between points
    print("\nCalculating hyperbolic distances between points:")
    for i in range(len(test_texts)):
        for j in range(i+1, len(test_texts)):
            dist = projector.compute_distance(hyperbolic_points[i], hyperbolic_points[j])
            print(f"Distance between '{test_texts[i]}' and '{test_texts[j]}': {dist.item():.4f}")
    
    print("\nHyperbolic projection test completed successfully!")
    
if __name__ == "__main__":
    # Run the tests
    try:
        bge_model = test_bge_model_connection()
        test_hyperbolic_projection(bge_model)
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()