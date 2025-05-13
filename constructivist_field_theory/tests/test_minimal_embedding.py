"""
Minimal test script that focuses solely on the basic embedding engine functionality.
This script deliberately avoids visualization to isolate any potential issues.
"""

import os
import sys
import numpy as np

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embedding_engine.models.bge_model import BGEModel
from embedding_engine.transformations.hyperbolic_projector import HyperbolicProjector

def test_minimal_embedding():
    """Test the most basic embedding functionality without visualization."""
    
    print("\n==== Starting Minimal Embedding Test ====")
    
    # Initialize BGE model - force fallback by using an invalid model name
    print("Initializing BGE model with fallback...")
    bge_model = BGEModel(
        model_name="invalid_model_to_force_fallback",
        device="cpu"
    )
    
    print(f"Model type: {bge_model.model_type}")
    print(f"Model embedding dimension: {bge_model.get_embedding_dim()}")
    
    # Test with just a few short texts
    test_texts = ["Simple test", "Another test"]
    print(f"\nGenerating embeddings for {len(test_texts)} test texts...")
    
    # Get embeddings
    embeddings = bge_model.encode(test_texts, convert_to_mlx=True)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Create a basic projector
    print("\nCreating hyperbolic projector...")
    projector = HyperbolicProjector(
        input_dim=bge_model.get_embedding_dim(),
        output_dim=32,  # Small dimension for quick testing
        curvature=-1.0,
        mixed_curvature=False
    )
    
    # Project to hyperbolic space
    print("Projecting embeddings to hyperbolic space...")
    hyperbolic_points = projector.project(embeddings)
    print(f"Hyperbolic points shape: {hyperbolic_points.shape}")
    
    # Verify they're in the Poincaré ball
    import mlx.core as mx
    norms = mx.sqrt(mx.sum(hyperbolic_points * hyperbolic_points, axis=1))
    print(f"Norms min: {mx.min(norms).item():.4f}, max: {mx.max(norms).item():.4f}")
    
    # Calculate distance between points
    distance = projector.compute_distance(hyperbolic_points[0], hyperbolic_points[1])
    print(f"Hyperbolic distance between points: {distance.item():.4f}")
    
    print("\n==== Minimal Embedding Test Completed Successfully ====")
    
if __name__ == "__main__":
    try:
        test_minimal_embedding()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()