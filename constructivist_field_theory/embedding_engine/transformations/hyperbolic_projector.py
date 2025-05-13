"""
Hyperbolic projector for transforming Euclidean embeddings into hyperbolic space.

This module provides functionality to project standard Euclidean embeddings from 
the BGE model into hyperbolic space using the Poincaré ball model, creating the 
foundation for our constructivist mathematics neural manifold.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from ..hyperbolic.poincare_ball import PoincareBall
from ..hyperbolic.mixed_curvature import MixedCurvatureSpace

class HyperbolicProjector:
    """
    Projects Euclidean embeddings into hyperbolic space using the Poincaré ball model.
    
    This class transforms standard embeddings from the BGE model into hyperbolic
    space, allowing us to leverage the exponential capacity growth and hierarchical
    modeling capabilities of hyperbolic geometry for our constructivist framework.
    """
    
    def __init__(self, input_dim, output_dim=None, curvature=-1.0, mixed_curvature=False):
        """
        Initialize the hyperbolic projector.
        
        Args:
            input_dim (int): Dimension of input Euclidean embeddings.
            output_dim (int, optional): Dimension of output hyperbolic embeddings.
                If None, same as input_dim.
            curvature (float): Curvature parameter for hyperbolic space.
                Must be negative for hyperbolic geometry.
            mixed_curvature (bool): Whether to use mixed-curvature product space.
                If True, creates a space with both hyperbolic and spherical components.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim else input_dim
        self.curvature = curvature
        self.mixed_curvature = mixed_curvature
        self.epsilon = 1e-15  # Small value for numerical stability
        
        # Create projection layer
        # We use a linear layer to map from input_dim to output_dim
        # before projecting to the Poincaré ball
        self.projection = nn.Linear(input_dim, self.output_dim)
        
        # Initialize hyperbolic space
        if mixed_curvature:
            # For mixed-curvature space, split dimensions in half
            # First half will be hyperbolic, second half will be spherical
            half_dim = self.output_dim // 2
            # Handle odd dimensions by making hyperbolic part slightly larger
            second_half = self.output_dim - half_dim
            self.manifold = MixedCurvatureSpace(
                dims=[half_dim, second_half],
                curvatures=[curvature, 1.0],  # Hyperbolic and spherical
                hyperbolic_dims=[0]  # First component is hyperbolic
            )
        else:
            self.manifold = PoincareBall(self.output_dim, curvature=curvature)
    
    def project(self, embeddings):
        """
        Project Euclidean embeddings into hyperbolic space.
        
        Args:
            embeddings: Euclidean embeddings as MLX tensor.
            
        Returns:
            Hyperbolic embeddings in the Poincaré ball.
        """
        # Apply linear projection
        projected = self.projection(embeddings)
        
        # For standard hyperbolic projection:
        # 1. Apply tanh to constrain the norm to be < 1
        # 2. Scale by a factor to avoid bunching near the origin
        if not self.mixed_curvature:
            # Compute the norm
            norm = mx.sqrt(mx.sum(projected ** 2, axis=-1, keepdims=True) + self.epsilon)
            
            # Apply tanh scaling to ensure points are inside the Poincaré ball
            # Scale by 0.9 to avoid numerical issues near the boundary
            scale_factor = 0.9
            return scale_factor * mx.tanh(norm) * projected / (norm + self.epsilon)
        else:
            # For mixed-curvature space, use its projection method
            return self.manifold.project_to_manifold(projected)
    
    def compute_distance(self, x, y):
        """
        Compute distance between points in hyperbolic space.
        
        Args:
            x, y: Points in hyperbolic space as MLX tensors.
            
        Returns:
            Hyperbolic distance between x and y.
        """
        return self.manifold.distance(x, y)
    
    def compute_geodesic(self, x, y, t):
        """
        Compute points along the geodesic between x and y.
        
        Args:
            x, y: Points in hyperbolic space as MLX tensors.
            t: Parameter between 0 and 1 indicating position along the geodesic.
            
        Returns:
            Point at position t along the geodesic from x to y.
        """
        if not self.mixed_curvature:
            return self.manifold.geodesic(x, y, t)
        else:
            # For mixed-curvature space, use exponential map-based interpolation
            log_xy = self.manifold.log_map(x, y)
            return self.manifold.exp_map(x, t * log_xy)
    
    def batch_project(self, batch_embeddings, batch_size=64):
        """
        Project a large batch of embeddings in smaller chunks to save memory.
        
        Args:
            batch_embeddings: Batch of Euclidean embeddings as MLX tensor.
            batch_size: Maximum number of embeddings to process at once.
            
        Returns:
            Batch of hyperbolic embeddings.
        """
        total_size = batch_embeddings.shape[0]
        result_chunks = []
        
        for i in range(0, total_size, batch_size):
            end_idx = min(i + batch_size, total_size)
            chunk = batch_embeddings[i:end_idx]
            result_chunks.append(self.project(chunk))
        
        return mx.concatenate(result_chunks, axis=0)


class HyperbolicEmbeddingSpace:
    """
    Complete hyperbolic embedding space integrating BGE model with hyperbolic projection.
    
    This class combines the BGE semantic model with hyperbolic projection to create
    a unified embedding space for constructivist mathematics.
    """
    
    def __init__(self, bge_model, output_dim=None, curvature=-1.0, mixed_curvature=False):
        """
        Initialize the hyperbolic embedding space.
        
        Args:
            bge_model: Instance of BGEModel
            output_dim (int, optional): Dimension of output hyperbolic embeddings.
                If None, uses the same dimension as the BGE model.
            curvature (float): Curvature parameter for hyperbolic space.
            mixed_curvature (bool): Whether to use mixed-curvature product space.
        """
        self.bge_model = bge_model
        input_dim = bge_model.get_embedding_dim()
        output_dim = output_dim if output_dim else input_dim
        
        self.projector = HyperbolicProjector(
            input_dim=input_dim,
            output_dim=output_dim,
            curvature=curvature,
            mixed_curvature=mixed_curvature
        )
    
    def encode_text(self, texts, batch_size=32):
        """
        Encode text into hyperbolic embeddings.
        
        Args:
            texts: Text or list of texts to encode.
            batch_size: Batch size for encoding.
            
        Returns:
            Hyperbolic embeddings of the texts.
        """
        # Get Euclidean embeddings from BGE model
        euclidean_embeddings = self.bge_model.encode(
            texts, 
            batch_size=batch_size,
            normalize=True,  # Normalize for stability
            convert_to_mlx=True
        )
        
        # Project to hyperbolic space
        return self.projector.batch_project(euclidean_embeddings)
    
    def semantic_distance(self, texts1, texts2, batch_size=32):
        """
        Compute semantic distance between texts in hyperbolic space.
        
        Args:
            texts1: First text or list of texts.
            texts2: Second text or list of texts.
            batch_size: Batch size for encoding.
            
        Returns:
            Hyperbolic distances between texts.
        """
        # Encode both sets of texts
        embeddings1 = self.encode_text(texts1, batch_size)
        embeddings2 = self.encode_text(texts2, batch_size)
        
        # Ensure both have the same shape for comparison
        if len(embeddings1.shape) == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if len(embeddings2.shape) == 1:
            embeddings2 = embeddings2.reshape(1, -1)
        
        # Compute distances
        if embeddings1.shape[0] == embeddings2.shape[0]:
            # Compute pairwise distances
            distances = []
            for i in range(embeddings1.shape[0]):
                distances.append(self.projector.compute_distance(
                    embeddings1[i], embeddings2[i]
                ))
            return mx.array(distances)
        else:
            # Compute all-pairs distances
            distances = mx.zeros((embeddings1.shape[0], embeddings2.shape[0]))
            for i in range(embeddings1.shape[0]):
                for j in range(embeddings2.shape[0]):
                    distances[i, j] = self.projector.compute_distance(
                        embeddings1[i], embeddings2[j]
                    )
            return distances
        
    def semantic_search(self, query, corpus, top_k=5, batch_size=32):
        """
        Perform semantic search in hyperbolic space.
        
        Args:
            query: Query text or list of query texts.
            corpus: List of corpus texts to search within.
            top_k: Number of top results to return.
            batch_size: Batch size for encoding.
            
        Returns:
            List of dictionaries with search results including indices and scores.
        """
        # Encode query and corpus
        query_embeddings = self.encode_text(query, batch_size)
        corpus_embeddings = self.encode_text(corpus, batch_size)
        
        # Ensure query_embeddings has the right shape
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Compute all-pairs distances
        distances = mx.zeros((query_embeddings.shape[0], corpus_embeddings.shape[0]))
        for i in range(query_embeddings.shape[0]):
            for j in range(corpus_embeddings.shape[0]):
                distances[i, j] = self.projector.compute_distance(
                    query_embeddings[i], corpus_embeddings[j]
                )
        
        # Convert distances to numpy for processing
        distances_np = distances.numpy()
        
        # For each query, get the top-k most similar corpus embeddings (smallest distances)
        results = []
        for i in range(len(query_embeddings)):
            scores = []
            for j in range(len(corpus_embeddings)):
                scores.append({
                    'corpus_id': j,
                    'score': float(-distances_np[i, j])  # Negated distance as score
                })
            scores = sorted(scores, key=lambda x: x['score'], reverse=True)  # Higher score is better
            results.append(scores[:top_k])
            
        return results