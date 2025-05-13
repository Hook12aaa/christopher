"""
Simple test script for the hyperbolic embedding space.

This script provides a minimal demonstration of the embedding engine
without relying on external visualization libraries.
"""

import os
import sys
import numpy as np

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embedding_engine.models.bge_model import BGEModel
from embedding_engine.transformations.hyperbolic_projector import HyperbolicProjector
from embedding_engine.hyperbolic.poincare_ball import PoincareBall

def test_basic_hyperbolic_projection():
    """Test basic functionality of the hyperbolic projector."""
    
    print("Creating a simple test hyperbolic space...")
    
    # Create a small input dimension for testing
    input_dim = 4
    
    # Create a test projector with simple dimensions
    projector = HyperbolicProjector(
        input_dim=input_dim, 
        output_dim=2,  # Project to 2D for easy visualization
        curvature=-1.0,
        mixed_curvature=False
    )
    
    # Create some sample input vectors
    test_inputs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # Convert to MLX array
    import mlx.core as mx
    test_inputs_mlx = mx.array(test_inputs)
    
    # Project to hyperbolic space
    print("Projecting to Poincaré ball...")
    hyperbolic_points = projector.project(test_inputs_mlx)
    
    # Print the results
    print("\nInput points (Euclidean):")
    for i, point in enumerate(test_inputs):
        print(f"Point {i+1}: {point}")
    
    print("\nProjected points (Poincaré ball):")
    for i, point in enumerate(hyperbolic_points):
        print(f"Point {i+1}: {point.numpy()}")
    
    # Check that all points are inside the Poincaré ball
    norms = mx.sqrt(mx.sum(hyperbolic_points * hyperbolic_points, axis=-1))
    print("\nNorms of projected points:")
    for i, norm in enumerate(norms):
        print(f"Point {i+1} norm: {norm.item():.6f} {'✓' if norm.item() < 1.0 else 'X'}")
    
    # Calculate distances between points
    print("\nHyperbolic distances:")
    for i in range(len(hyperbolic_points)):
        for j in range(i+1, len(hyperbolic_points)):
            dist = projector.compute_distance(hyperbolic_points[i], hyperbolic_points[j])
            print(f"Distance between points {i+1} and {j+1}: {dist.item():.6f}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_basic_hyperbolic_projection()