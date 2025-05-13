"""
Hyperbolic projector for transforming Euclidean embeddings into hyperbolic space.

This module projects standard embeddings into hyperbolic space using the Poincaré ball model,
creating the foundation for our constructivist mathematics neural manifold.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from ..hyperbolic.poincare_ball import PoincareBall
from ..hyperbolic.mixed_curvature import MixedCurvatureSpace

class HyperbolicProjector:
    """
    Projects Euclidean embeddings into hyperbolic space using the Poincaré ball model.
    
    Transforms embeddings into hyperbolic space, leveraging exponential capacity
    growth and hierarchical modeling capabilities for our constructivist framework.
    """
    
    def __init__(self, input_dim, output_dim=None, curvature=-1.0, mixed_curvature=False):
        """Initialize the hyperbolic projector."""
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim else input_dim
        self.curvature = curvature
        self.mixed_curvature = mixed_curvature
        self.epsilon = 1e-15  # Small value for numerical stability
        
        self.projection = nn.Linear(input_dim, self.output_dim)
        
        if mixed_curvature:
            half_dim = self.output_dim // 2
            second_half = self.output_dim - half_dim
            self.manifold = MixedCurvatureSpace(
                dims=[half_dim, second_half],
                curvatures=[curvature, 1.0],
                hyperbolic_dims=[0]
            )
        else:
            self.manifold = PoincareBall(self.output_dim, curvature=curvature)
    
    def project(self, embeddings):
        """Project Euclidean embeddings into hyperbolic space."""
        projected = self.projection(embeddings)
        
        if not self.mixed_curvature:
            norm = mx.sqrt(mx.sum(projected ** 2, axis=-1, keepdims=True) + self.epsilon)
            scale_factor = 0.9
            return scale_factor * mx.tanh(norm) * projected / (norm + self.epsilon)
        else:
            return self.manifold.project_to_manifold(projected)
    
    def compute_distance(self, x, y):
        """Compute distance between points in hyperbolic space."""
        return self.manifold.distance(x, y)
    
    def compute_geodesic(self, x, y, t):
        """Compute points along the geodesic between x and y."""
        if not self.mixed_curvature:
            return self.manifold.geodesic(x, y, t)
        else:
            log_xy = self.manifold.log_map(x, y)
            return self.manifold.exp_map(x, t * log_xy)
    
    def batch_project(self, batch_embeddings, batch_size=64):
        """Project a large batch of embeddings in smaller chunks to save memory."""
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
    
    Combines the BGE semantic model with hyperbolic projection to create
    a unified embedding space for constructivist mathematics.
    """
    
    def __init__(self, bge_model, output_dim=None, curvature=-1.0, mixed_curvature=False):
        """Initialize the hyperbolic embedding space."""
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
        """Encode text into hyperbolic embeddings."""
        euclidean_embeddings = self.bge_model.encode(
            texts, 
            batch_size=batch_size,
            normalize=True,
            convert_to_mlx=True
        )
        
        return self.projector.batch_project(euclidean_embeddings)
    
    def semantic_distance(self, texts1, texts2, batch_size=32):
        """Compute semantic distance between texts in hyperbolic space."""
        embeddings1 = self.encode_text(texts1, batch_size)
        embeddings2 = self.encode_text(texts2, batch_size)
        
        if len(embeddings1.shape) == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if len(embeddings2.shape) == 1:
            embeddings2 = embeddings2.reshape(1, -1)
        
        if embeddings1.shape[0] == embeddings2.shape[0]:
            distances = []
            for i in range(embeddings1.shape[0]):
                distances.append(self.projector.compute_distance(
                    embeddings1[i], embeddings2[i]
                ))
            return mx.array(distances)
        else:
            distances = mx.zeros((embeddings1.shape[0], embeddings2.shape[0]))
            for i in range(embeddings1.shape[0]):
                for j in range(embeddings2.shape[0]):
                    distances[i, j] = self.projector.compute_distance(
                        embeddings1[i], embeddings2[j]
                    )
            return distances
        
    def semantic_search(self, query, corpus, top_k=5, batch_size=32):
        """Perform semantic search in hyperbolic space."""
        query_embeddings = self.encode_text(query, batch_size)
        corpus_embeddings = self.encode_text(corpus, batch_size)
        
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        distances = mx.zeros((query_embeddings.shape[0], corpus_embeddings.shape[0]))
        for i in range(query_embeddings.shape[0]):
            for j in range(corpus_embeddings.shape[0]):
                distances[i, j] = self.projector.compute_distance(
                    query_embeddings[i], corpus_embeddings[j]
                )
        
        distances_np = distances.numpy()
        
        results = []
        for i in range(len(query_embeddings)):
            scores = []
            for j in range(len(corpus_embeddings)):
                scores.append({
                    'corpus_id': j,
                    'score': float(-distances_np[i, j])  # Negated distance as score
                })
            scores = sorted(scores, key=lambda x: x['score'], reverse=True)
            results.append(scores[:top_k])
            
        return results