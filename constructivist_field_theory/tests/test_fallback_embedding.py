"""
Minimal test for the embedding engine fallback methods.

This script tests the fallback functionality of the embedding engine
without requiring the full BGE model to load.
"""

import os
import sys
import numpy as np
import mlx.core as mx

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embedding_engine.models.bge_model import BGEModel
from embedding_engine.transformations.hyperbolic_projector import HyperbolicProjector

def test_simple_fallback():
    """Test the simple fallback encoding method."""
    
    print("\n---- Testing Simple Fallback Embedding ----")
    
    # Force using the simple fallback by setting an invalid model name
    print("Initializing BGE model with forced fallback...")
    bge_model = BGEModel(
        model_name="invalid_model_to_force_fallback",
        device="cpu"
    )
    
    # Print model information - should use the simple fallback
    model_info = bge_model.get_model_info()
    print(f"Model type: {model_info['model_type']}")
    print(f"Model name: {model_info['model_name']}")
    print(f"Embedding dimension: {model_info['embedding_dim']}")
    
    # Test encoding with simple texts
    sample_texts = ["This is a test sentence.", "Another example text."]
    print(f"\nEncoding {len(sample_texts)} sample texts using fallback...")
    
    embeddings = bge_model.encode(sample_texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Basic projection test
    print("\nTesting projection to hyperbolic space...")
    projector = HyperbolicProjector(
        input_dim=bge_model.get_embedding_dim(),
        output_dim=64,
        curvature=-1.0,
        mixed_curvature=False
    )
    
    hyperbolic_points = projector.project(embeddings)
    
    # Verify points are inside the ball
    norms = mx.sqrt(mx.sum(hyperbolic_points * hyperbolic_points, axis=1))
    print(f"Hyperbolic point norms: min={mx.min(norms).item():.4f}, max={mx.max(norms).item():.4f}")
    print("All points are inside the Poincaré ball." if mx.all(norms < 1.0) else
          "Warning: Some points are outside the Poincaré ball!")
    
    print("\nSimple fallback test completed successfully!\n")
    return bge_model, hyperbolic_points
    
if __name__ == "__main__":
    # Run the fallback test
    try:
        bge_model, points = test_simple_fallback()
        print("All tests completed successfully!")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()