"""
Text-based Poincaré visualization test that connects the embedding engine to visualization.
This script demonstrates how to use the embedding engine with simple or fallback modes
and properly visualize the results in hyperbolic space.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embedding_engine.models.bge_model import BGEModel
from embedding_engine.transformations.hyperbolic_projector import HyperbolicProjector, HyperbolicEmbeddingSpace

def test_text_based_visualization():
    """Test visualization of text embeddings in hyperbolic space."""
    
    print("\n==== Testing Text-Based Poincaré Visualization ====")
    
    # Initialize BGE model with fallback mode for reliable testing
    print("Initializing BGE model...")
    bge_model = BGEModel(
        model_name="invalid_model_to_force_fallback",  # Force using fallback for testing
        device="cpu"
    )
    print(f"BGE model initialized. Type: {bge_model.model_type}")
    
    # Create hyperbolic embedding space with reduced dimensions
    print("Creating hyperbolic embedding space...")
    hyperbolic_space = HyperbolicEmbeddingSpace(
        bge_model, 
        output_dim=3,  # Use 3D for direct visualization
        curvature=-1.0,
        mixed_curvature=False
    )
    
    # Sample texts to embed - same categories as our simple test
    texts = [
        # Music genres
        "Classical music", "Baroque music", "Romantic music", 
        "Jazz music", "Rock music", "Electronic music",
        # Venues
        "Concert hall", "Jazz club", "Stadium", "Warehouse venue"
    ]
    
    # Get simpler labels for visualization
    labels = [text.split()[0] for text in texts]
    
    # Generate hyperbolic embeddings
    print(f"Generating hyperbolic embeddings for {len(texts)} texts...")
    embeddings = hyperbolic_space.encode_text(texts)
    
    # Convert to numpy for visualization
    print("Converting embeddings for visualization...")
    if hasattr(embeddings, 'numpy'):
        embeddings_np = embeddings.numpy()
    else:
        embeddings_np = embeddings
    
    # Verify embeddings are inside the Poincaré ball
    norms = np.sqrt(np.sum(embeddings_np**2, axis=1))
    print(f"Embedding norms: min={np.min(norms):.4f}, max={np.max(norms):.4f}")
    if np.all(norms < 1.0):
        print("All embeddings are inside the Poincaré ball.")
    else:
        print("Warning: Some embeddings are outside the Poincaré ball!")
    
    # Calculate some semantic distances
    print("\nCalculating semantic distances in hyperbolic space:")
    examples = [
        ("Classical music", "Baroque music"),
        ("Classical music", "Jazz music"),
        ("Jazz music", "Jazz club"),
        ("Rock music", "Stadium"),
        ("Electronic music", "Warehouse venue")
    ]
    
    for text1, text2 in examples:
        distance = hyperbolic_space.semantic_distance(text1, text2).item()
        print(f"Distance between '{text1}' and '{text2}': {distance:.4f}")
    
    # Visualize the embeddings
    print("\nVisualizing embeddings in the Poincaré ball...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the unit sphere wireframe as a reference
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 0.99 * np.outer(np.cos(u), np.sin(v))
    y = 0.99 * np.outer(np.sin(u), np.sin(v))
    z = 0.99 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)
    
    # Plot the points
    x, y, z = embeddings_np[:, 0], embeddings_np[:, 1], embeddings_np[:, 2]
    ax.scatter(x, y, z, c=np.arange(len(embeddings_np)), cmap='viridis', s=100)
    
    # Label the points
    for i, label in enumerate(labels):
        ax.text(x[i], y[i], z[i], label, fontsize=9)
    
    # Set axes limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    # Set title and labels
    ax.set_title("Text Embeddings in Hyperbolic Space")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Create a directory for output files
    os.makedirs("output", exist_ok=True)
    
    # Save the visualization
    plt.savefig("output/text_based_poincare.png", dpi=300, bbox_inches='tight')
    print(f"Visualization saved to output/text_based_poincare.png")
    
    print("\n==== Text-Based Visualization Test Completed Successfully ====")
    
if __name__ == "__main__":
    try:
        test_text_based_visualization()
        plt.show()
        print("Test completed without errors.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()