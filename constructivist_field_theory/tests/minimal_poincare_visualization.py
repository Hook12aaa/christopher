"""
Minimal test script for visualizing the Poincaré ball model.

This script creates a simple visualization of points in the Poincaré ball
to verify that our implementation is working correctly.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import mlx.core as mx

# Import our Poincaré ball implementation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding_engine.hyperbolic.poincare_ball import PoincareBall

def visualize_poincare_ball():
    """Create a simple visualization of the Poincaré ball model."""
    
    print("Creating Poincaré ball model...")
    # Create a 2D Poincaré ball
    poincare = PoincareBall(dim=2, curvature=-1.0)
    
    # Create a grid of points in Euclidean space
    grid_size = 5
    x = np.linspace(-0.8, 0.8, grid_size)
    y = np.linspace(-0.8, 0.8, grid_size)
    points = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Create a point in the Poincaré ball
            point = mx.array([x[i], y[j]])
            points.append(point)
    
    # Create a figure for visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw the boundary of the Poincaré ball
    circle = Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    ax.add_patch(circle)
    
    # Draw the grid points
    for point in points:
        ax.scatter(point[0].item(), point[1].item(), color='blue', s=50)
    
    # Draw geodesics between some pairs of points
    center = mx.array([0.0, 0.0])
    
    for point in points:
        # Sample points along the geodesic from center to point
        t_values = np.linspace(0, 1, 20)
        try:
            geodesic_points = np.array([
                poincare.geodesic(center, point, t).numpy() 
                for t in t_values
            ])
            
            # Plot the geodesic
            ax.plot(
                geodesic_points[:, 0], 
                geodesic_points[:, 1], 
                color='red', 
                alpha=0.5
            )
        except Exception as e:
            print(f"Error computing geodesic: {e}")
    
    # Set plot properties
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Poincaré Ball Model with Geodesics")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Save the figure
    plt.savefig("output/poincare_visualization.png", dpi=300, bbox_inches='tight')
    print("Visualization saved to 'output/poincare_visualization.png'")
    
    print("Visualization complete!")

if __name__ == "__main__":
    visualize_poincare_ball()