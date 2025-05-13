"""
Test script for the hyperbolic embedding space.

This script demonstrates the use of the embedding engine to create hyperbolic
embeddings and visualize them in the Poincaré ball model.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embedding_engine.models.bge_model import BGEModel
from embedding_engine.transformations.hyperbolic_projector import HyperbolicEmbeddingSpace
from embedding_engine.hyperbolic.poincare_ball import PoincareBall
from visualization.utils.poincare_visualizer import (
    visualize_poincare_ball_2d, 
    visualize_poincare_ball_3d, 
    create_embeddings_visualization
)

def test_hyperbolic_embeddings():
    """Test the hyperbolic embedding space with sample texts and visualize the results."""
    
    print("Loading BGE model...")
    # Initialize the BGE model with explicit parameters
    bge_model = BGEModel(
        model_name="BAAI/bge-large-v1.5",  # Explicitly specify the model name
        device=None,  # Let it choose the best device automatically
        cache_dir="./model_cache"  # Cache models to avoid repeated downloads
    )
    print(f"BGE model loaded. Type: {bge_model.model_type}, Embedding dimension: {bge_model.get_embedding_dim()}")
    
    # Create hyperbolic embedding space with mixed curvature
    print("Creating hyperbolic embedding space...")
    hyperbolic_space = HyperbolicEmbeddingSpace(
        bge_model, 
        output_dim=64,  # Reduced dimension for visualization purposes
        curvature=-1.0,
        mixed_curvature=True
    )
    
    # Sample texts to embed
    texts = [
        # Music genres - hierarchical structure
        "Classical music", "Baroque music", "Romantic music", "Modern classical",
        "Jazz", "Bebop", "Swing jazz", "Fusion jazz",
        "Rock music", "Progressive rock", "Hard rock", "Punk rock",
        "Electronic music", "Techno", "House music", "Ambient",
        
        # Venues - different types
        "Concert hall", "Jazz club", "Stadium", "Warehouse",
        "Festival grounds", "Small theater", "Arena", "Bar venue"
    ]
    
    # Labels for visualization
    labels = [text.split(" ")[0] for text in texts]
    
    # Get hyperbolic embeddings
    print("Generating hyperbolic embeddings...")
    embeddings = hyperbolic_space.encode_text(texts)
    
    # Convert to numpy for visualization - handle both MLX array and numpy array types
    print("Converting embeddings for visualization...")
    if hasattr(embeddings, 'numpy'):
        embeddings_np = embeddings.numpy()
    else:
        # If it's already a numpy array, use it directly
        embeddings_np = embeddings
    
    # Create a directory for output files
    os.makedirs("output", exist_ok=True)
    
    # Visualize the embeddings
    print("Creating visualization...")
    fig_2d, fig_3d = create_embeddings_visualization(
        embeddings_np, 
        labels=texts, 
        title="Music Genres and Venues in Hyperbolic Space",
        save_path="output/hyperbolic_embeddings"
    )
    
    # Show the plots
    plt.show()
    
    # For 3D visualization, save the HTML file
    if fig_3d:
        fig_3d.write_html("output/hyperbolic_embeddings_3d.html")
        print("3D visualization saved to output/hyperbolic_embeddings_3d.html")
    
    # Calculate some distances to demonstrate the hyperbolic metric
    print("\nCalculating semantic distances in hyperbolic space...")
    
    # Function to print distances between pairs
    def print_distances(text1, text2):
        distance = hyperbolic_space.semantic_distance(text1, text2).item()
        print(f"Distance between '{text1}' and '{text2}': {distance:.4f}")
    
    # Calculate some interesting distances
    print_distances("Classical music", "Baroque music")
    print_distances("Classical music", "Jazz")
    print_distances("Classical music", "Electronic music")
    print_distances("Jazz", "Jazz club")
    print_distances("Rock music", "Stadium")
    print_distances("Electronic music", "Warehouse")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    try:
        # Set a timeout for the entire test
        import signal
        
        def timeout_handler(signum, frame):
            print("Test execution timed out. This may be due to model loading delays.")
            sys.exit(1)
        
        # Set 60 second timeout (only works on Unix-like systems)
        if sys.platform != 'win32':
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)
            
        print("Starting hyperbolic embeddings test...")
        test_hyperbolic_embeddings()
        print("Test completed without errors.")
        
        # Cancel the timeout alarm if test completed successfully
        if sys.platform != 'win32':
            signal.alarm(0)
            
    except Exception as e:
        print(f"\nError during test execution: {str(e)}")
        import traceback
        traceback.print_exc()