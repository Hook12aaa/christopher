"""
Minimal hyperbolic tree visualization that connects text embeddings directly to visualization.
This standalone script integrates embedding and visualization without complex dependencies.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import cm

# Create a directory for output
os.makedirs("output", exist_ok=True)

class SimpleWordEmbedder:
    """Simple deterministic word embedder for testing hyperbolic visualization."""
    
    def __init__(self, embedding_dim=128):
        """Initialize the embedder with specified dimension."""
        self.embedding_dim = embedding_dim
        self.word_vectors = {}  # Cache for word vectors
        
    def _hash_word(self, word, seed=0):
        """Create a deterministic hash for a word."""
        random.seed(hash(word.lower()) + seed)
        return np.array([random.uniform(-1, 1) for _ in range(self.embedding_dim)])
    
    def embed_word(self, word):
        """Generate embedding for a single word."""
        if word.lower() not in self.word_vectors:
            vec = self._hash_word(word)
            # Normalize vector
            vec = vec / np.sqrt(np.sum(vec**2))
            self.word_vectors[word.lower()] = vec
        return self.word_vectors[word.lower()]
    
    def embed_text(self, text):
        """Generate embedding for a text by averaging word embeddings."""
        words = text.lower().split()
        if not words:
            return np.zeros(self.embedding_dim)
            
        embeddings = [self.embed_word(word) for word in words]
        avg_embedding = np.mean(embeddings, axis=0)
        # Normalize the average
        return avg_embedding / np.sqrt(np.sum(avg_embedding**2))
    
    def batch_embed(self, texts):
        """Embed a batch of texts."""
        return np.array([self.embed_text(text) for text in texts])


class HyperbolicProjector:
    """Projects Euclidean embeddings to the Poincaré ball model of hyperbolic space."""
    
    def __init__(self, input_dim, output_dim=3, curvature=-1.0):
        """Initialize the projector."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.curvature = curvature
        
        # Create a simple linear projection matrix
        # For reproducibility
        np.random.seed(42)
        self.projection_matrix = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        
    def project(self, embeddings):
        """Project embeddings to hyperbolic space (Poincaré ball)."""
        # Apply linear projection
        projected = np.matmul(embeddings, self.projection_matrix)
        
        # Normalize to ensure points are inside the Poincaré ball
        norms = np.sqrt(np.sum(projected**2, axis=1, keepdims=True))
        # Apply tanh to ensure points are inside the ball with radius 0.9
        scale_factor = 0.9
        return scale_factor * np.tanh(norms) * projected / (norms + 1e-15)
    
    def hyperbolic_distance(self, u, v):
        """Compute the hyperbolic distance in the Poincaré ball model."""
        # Squared norms
        u_norm_sq = np.sum(u**2)
        v_norm_sq = np.sum(v**2)
        
        # Numerator: 2 * ||u-v||^2
        uv_diff_sq = np.sum((u - v)**2)
        numerator = 2 * uv_diff_sq
        
        # Denominator: (1-||u||^2)(1-||v||^2)
        denominator = (1 - u_norm_sq) * (1 - v_norm_sq)
        
        # Handle numerical issues
        if denominator <= 0:
            return float('inf')
        
        arg = 1 + numerator / denominator
        if arg < 1:
            arg = 1  # Handle numerical issues
        
        return np.arccosh(arg)


def create_hyperbolic_tree():
    """Create a hierarchical tree structure in hyperbolic space."""
    print("\n==== Creating Hyperbolic Tree Visualization ====")
    
    # Create a word embedder and projector
    print("Initializing word embedder and hyperbolic projector...")
    embedder = SimpleWordEmbedder(embedding_dim=64)
    projector = HyperbolicProjector(input_dim=64, output_dim=3)
    
    # Define hierarchical categories
    categories = {
        "Mathematics": [
            "Algebra", "Geometry", "Calculus", "Statistics"
        ],
        "Physics": [
            "Mechanics", "Electromagnetism", "Quantum", "Relativity"
        ],
        "Biology": [
            "Genetics", "Ecology", "Zoology", "Botany"
        ],
        "Computer Science": [
            "Algorithms", "Databases", "Graphics", "Networks"
        ]
    }
    
    # Flatten the structure for embedding
    all_texts = []
    all_labels = []
    category_indices = {}
    
    # Add root node
    all_texts.append("Knowledge")
    all_labels.append("Knowledge")
    
    # Add main categories and subcategories
    for i, (category, subcategories) in enumerate(categories.items()):
        category_idx = len(all_texts)
        category_indices[category] = category_idx
        all_texts.append(category)
        all_labels.append(category)
        
        for subcategory in subcategories:
            all_texts.append(f"{category} {subcategory}")
            all_labels.append(subcategory)
    
    # Generate Euclidean embeddings
    print(f"Generating embeddings for {len(all_texts)} concepts...")
    euclidean_embeddings = embedder.batch_embed(all_texts)
    
    # Project to hyperbolic space
    print("Projecting to hyperbolic space...")
    hyperbolic_embeddings = projector.project(euclidean_embeddings)
    
    # Verify embeddings are inside the Poincaré ball
    norms = np.sqrt(np.sum(hyperbolic_embeddings**2, axis=1))
    print(f"Embedding norms: min={np.min(norms):.4f}, max={np.max(norms):.4f}")
    
    # Calculate some distances to verify hyperbolic properties
    print("\nCalculating hyperbolic distances between concepts:")
    # Root to main categories
    root_idx = 0
    for category, idx in category_indices.items():
        dist = projector.hyperbolic_distance(
            hyperbolic_embeddings[root_idx], hyperbolic_embeddings[idx]
        )
        print(f"Distance from Knowledge to {category}: {dist:.4f}")
    
    # Distance between subcategories
    subcategory_pairs = [
        ("Mathematics Algebra", "Mathematics Geometry"),
        ("Physics Quantum", "Physics Relativity"),
        ("Mathematics Calculus", "Physics Mechanics"),
        ("Biology Genetics", "Computer Science Algorithms")
    ]
    
    for pair in subcategory_pairs:
        idx1 = all_texts.index(pair[0])
        idx2 = all_texts.index(pair[1])
        dist = projector.hyperbolic_distance(
            hyperbolic_embeddings[idx1], hyperbolic_embeddings[idx2]
        )
        print(f"Distance from {pair[0].split()[-1]} to {pair[1].split()[-1]}: {dist:.4f}")
    
    # Create visualization
    print("\nCreating hyperbolic tree visualization...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the unit sphere wireframe as reference
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = 0.99 * np.outer(np.cos(u), np.sin(v))
    y = 0.99 * np.outer(np.sin(u), np.sin(v))
    z = 0.99 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.1)
    
    # Plot edges from root to main categories
    root_pos = hyperbolic_embeddings[0]
    for category, idx in category_indices.items():
        category_pos = hyperbolic_embeddings[idx]
        ax.plot([root_pos[0], category_pos[0]], 
                [root_pos[1], category_pos[1]], 
                [root_pos[2], category_pos[2]], 
                'k-', alpha=0.3)
    
    # Plot edges from categories to subcategories
    colors = ['red', 'blue', 'green', 'purple']
    for i, (category, subcategories) in enumerate(categories.items()):
        category_idx = category_indices[category]
        category_pos = hyperbolic_embeddings[category_idx]
        color = colors[i % len(colors)]
        
        # Find indices for this category's subcategories
        for j, subcategory in enumerate(subcategories):
            subcategory_text = f"{category} {subcategory}"
            subcategory_idx = all_texts.index(subcategory_text)
            subcategory_pos = hyperbolic_embeddings[subcategory_idx]
            
            # Draw edge from category to subcategory
            ax.plot([category_pos[0], subcategory_pos[0]], 
                    [category_pos[1], subcategory_pos[1]], 
                    [category_pos[2], subcategory_pos[2]], 
                    '-', color=color, alpha=0.5)
    
    # Plot the points with different colors for hierarchy levels
    # Root node
    ax.scatter(hyperbolic_embeddings[0, 0], hyperbolic_embeddings[0, 1], hyperbolic_embeddings[0, 2], 
               c='black', s=100, label='Root')
    
    # Main categories
    category_indices_list = list(category_indices.values())
    ax.scatter(hyperbolic_embeddings[category_indices_list, 0], 
               hyperbolic_embeddings[category_indices_list, 1], 
               hyperbolic_embeddings[category_indices_list, 2], 
               c='red', s=80, label='Categories')
    
    # Subcategories
    subcategory_indices = [i for i in range(len(all_texts)) 
                          if i not in category_indices_list and i != 0]
    ax.scatter(hyperbolic_embeddings[subcategory_indices, 0], 
               hyperbolic_embeddings[subcategory_indices, 1], 
               hyperbolic_embeddings[subcategory_indices, 2], 
               c='blue', s=60, label='Subcategories')
    
    # Add labels
    for i, label in enumerate(all_labels):
        # Adjust text size by hierarchy level
        if i == 0:  # Root
            fontsize = 12
        elif i in category_indices_list:  # Main categories
            fontsize = 10
        else:  # Subcategories
            fontsize = 8
            
        ax.text(hyperbolic_embeddings[i, 0], 
                hyperbolic_embeddings[i, 1], 
                hyperbolic_embeddings[i, 2], 
                label, fontsize=fontsize)
    
    # Set axes limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    # Set title and labels
    ax.set_title("Knowledge Hierarchy in Hyperbolic Space")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Save the figure
    plt.savefig("output/hyperbolic_tree_embedding.png", dpi=300, bbox_inches='tight')
    print(f"Visualization saved to output/hyperbolic_tree_embedding.png")
    
    print("\n==== Hyperbolic Tree Visualization Completed ====")
    
    return fig

if __name__ == "__main__":
    try:
        fig = create_hyperbolic_tree()
        plt.show()
        print("Visualization completed without errors.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()