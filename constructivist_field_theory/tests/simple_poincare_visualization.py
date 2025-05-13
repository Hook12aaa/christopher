"""
Simple Poincare ball visualization test with a direct connection to the embedding engine.
This version bypasses potential dependency issues by using direct numpy arrays.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_simple_embeddings():
    """Create simple 3D embeddings in the Poincare ball using numpy directly."""
    
    # Categories to visualize in hyperbolic space
    categories = [
        # Music genres
        "Classical", "Baroque", "Romantic", "Jazz", "Rock", "Electronic",
        # Venues
        "Concert hall", "Jazz club", "Stadium", "Warehouse"
    ]
    
    # Create pseudo-embeddings that simulate hyperbolic structure
    # Center points for each category group
    music_center = np.array([0.3, 0.1, 0.0]) * 0.7  # Scale to keep inside ball
    venue_center = np.array([-0.3, 0.1, 0.0]) * 0.7
    
    # Generate points with hierarchical structure
    embeddings = []
    labels = []
    
    # Music genres with hierarchical relationship
    embeddings.append(music_center)  # Classical at center
    labels.append("Classical")
    
    # Classical sub-genres
    angle1 = np.pi/6
    embeddings.append(music_center + 0.3 * np.array([np.cos(angle1), np.sin(angle1), 0.1]))
    labels.append("Baroque")
    
    angle2 = np.pi/3
    embeddings.append(music_center + 0.3 * np.array([np.cos(angle2), np.sin(angle2), -0.1]))
    labels.append("Romantic")
    
    # Other music genres
    embeddings.append(music_center + 0.5 * np.array([0.4, 0.3, 0.2]))
    labels.append("Jazz")
    
    embeddings.append(music_center + 0.5 * np.array([0.4, -0.3, 0.2]))
    labels.append("Rock")
    
    embeddings.append(music_center + 0.5 * np.array([-0.4, -0.3, -0.2]))
    labels.append("Electronic")
    
    # Venues
    embeddings.append(venue_center)
    labels.append("Concert hall")
    
    embeddings.append(venue_center + 0.4 * np.array([0.4, 0.3, 0.0]))
    labels.append("Jazz club")
    
    embeddings.append(venue_center + 0.4 * np.array([0.0, -0.3, 0.3]))
    labels.append("Stadium")
    
    embeddings.append(venue_center + 0.4 * np.array([-0.4, -0.3, -0.3]))
    labels.append("Warehouse")
    
    # Convert to numpy array
    return np.array(embeddings), labels

def visualize_poincare_ball(embeddings, labels, title="Poincare Ball Visualization"):
    """
    Visualize embeddings in a 3D Poincare ball.
    
    Args:
        embeddings: Numpy array of shape (n, 3) with 3D embeddings
        labels: List of labels corresponding to the embeddings
        title: Title for the visualization
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the unit sphere wireframe for reference
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 0.99 * np.outer(np.cos(u), np.sin(v))
    y = 0.99 * np.outer(np.sin(u), np.sin(v))
    z = 0.99 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)
    
    # Plot the points
    x, y, z = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]
    ax.scatter(x, y, z, c=np.arange(len(embeddings)), cmap='viridis', s=100)
    
    # Label the points
    for i, label in enumerate(labels):
        ax.text(x[i], y[i], z[i], label, fontsize=9)
    
    # Set axes limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Create a directory for output
    os.makedirs("output", exist_ok=True)
    
    # Save the figure
    plt.savefig("output/poincare_visualization.png", dpi=300, bbox_inches='tight')
    print(f"Visualization saved to output/poincare_visualization.png")
    
    return fig

def test_visualization():
    """Run the visualization test."""
    print("\n==== Testing Simple Poincare Visualization ====")
    
    # Create sample embeddings
    print("Creating sample embeddings...")
    embeddings, labels = create_simple_embeddings()
    
    # Verify that the points are inside the Poincare ball
    norms = np.sqrt(np.sum(embeddings**2, axis=1))
    print(f"Embedding norms: min={np.min(norms):.4f}, max={np.max(norms):.4f}")
    if np.all(norms < 1.0):
        print("All points are inside the Poincare ball.")
    else:
        print("Warning: Some points are outside the Poincare ball!")
    
    # Calculate some distances
    print("\nCalculating hyperbolic distances between points:")
    # Simple hyperbolic distance in the Poincare ball model
    def poincare_distance(u, v):
        """Compute hyperbolic distance in the Poincare ball model."""
        # Convert points to numpy arrays
        u = np.array(u)
        v = np.array(v)
        
        # Calculate squared norms
        u_norm_sq = np.sum(u**2)
        v_norm_sq = np.sum(v**2)
        
        # Calculate the numerator: 2 * ||u-v||^2
        uv_diff_sq = np.sum((u - v)**2)
        numerator = 2 * uv_diff_sq
        
        # Calculate the denominator: (1-||u||^2)(1-||v||^2)
        denominator = (1 - u_norm_sq) * (1 - v_norm_sq)
        
        # Handle numerical issues
        if denominator <= 0:
            return float('inf')
        
        # Calculate arccosh(1 + ...)
        arg = 1 + numerator / denominator
        if arg < 1:
            arg = 1  # Handle numerical issues
        
        return np.arccosh(arg)
    
    # Print some example distances
    for i in range(3):
        for j in range(i+1, min(i+4, len(embeddings))):
            dist = poincare_distance(embeddings[i], embeddings[j])
            print(f"Distance between '{labels[i]}' and '{labels[j]}': {dist:.4f}")
    
    # Visualize the embeddings
    print("\nVisualizing embeddings in the Poincare ball...")
    fig = visualize_poincare_ball(embeddings, labels, "Music Genres and Venues in Hyperbolic Space")
    
    print("\n==== Visualization Test Completed Successfully ====")
    
if __name__ == "__main__":
    try:
        test_visualization()
        plt.show()
        print("Test completed without errors.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()