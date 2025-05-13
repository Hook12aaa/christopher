"""
Mixed-curvature product space implementation for neural manifold embeddings.

This module implements a product space model that combines hyperbolic and spherical
geometries with learnable curvature parameters, allowing for more expressive
representation of different aspects of social relationships through appropriate
geometric principles.
"""

import numpy as np
import mlx.core as mx
from ..hyperbolic.poincare_ball import PoincareBall

class MixedCurvatureSpace:
    """
    Implementation of a mixed-curvature product space combining hyperbolic and spherical
    geometries with learnable curvature parameters.
    
    This space allows for representation of both hierarchical (hyperbolic) and
    similarity-based (spherical) relationships in the same embedding space.
    """
    
    def __init__(self, dims, curvatures=None, hyperbolic_dims=None):
        """
        Initialize the mixed-curvature product space.
        
        Args:
            dims (list of int): Dimensions of each component space.
            curvatures (list of float): Curvature parameters for each component.
                Negative for hyperbolic, positive for spherical, zero for Euclidean.
                If None, initialized to -1.0 for hyperbolic dims and 1.0 for spherical.
            hyperbolic_dims (list of int): Indices of dimensions that should use hyperbolic geometry.
                If None, the first half of dimensions are assumed to be hyperbolic.
        """
        self.dims = dims
        self.total_dim = sum(dims)
        
        # Default initialization if hyperbolic_dims not provided
        if hyperbolic_dims is None:
            hyperbolic_dims = list(range(len(dims) // 2))
        self.hyperbolic_dims = hyperbolic_dims
        
        # Initialize curvatures
        if curvatures is None:
            curvatures = []
            for i in range(len(dims)):
                if i in hyperbolic_dims:
                    curvatures.append(-1.0)  # Hyperbolic
                else:
                    curvatures.append(1.0)   # Spherical
        
        self.curvatures = mx.array(curvatures)
        
        # Initialize component spaces
        self.spaces = []
        for i, (dim, c) in enumerate(zip(dims, curvatures)):
            if i in hyperbolic_dims:
                # Hyperbolic space (Poincaré ball)
                self.spaces.append(PoincareBall(dim, curvature=c))
            else:
                # We'll use simple spherical/Euclidean operations for now
                # In a more complete implementation, we would have separate classes
                # for spherical and Euclidean spaces
                self.spaces.append(None)
                
        self.epsilon = 1e-15  # Small value for numerical stability
    
    def _split_points(self, x):
        """Split points into component spaces."""
        components = []
        start_idx = 0
        for dim in self.dims:
            components.append(x[..., start_idx:start_idx + dim])
            start_idx += dim
        return components
    
    def _merge_points(self, components):
        """Merge component points back into the full space."""
        return mx.concatenate(components, axis=-1)
    
    def distance(self, x, y):
        """
        Compute the product space distance between points x and y.
        
        The distance in product space is computed as the L2 norm of the
        component distances.
        
        Args:
            x: First point tensor of shape (..., total_dim)
            y: Second point tensor of shape (..., total_dim)
            
        Returns:
            Tensor representing the product space distance
        """
        x_components = self._split_points(x)
        y_components = self._split_points(y)
        
        component_distances = []
        for i, (x_comp, y_comp) in enumerate(zip(x_components, y_components)):
            if i in self.hyperbolic_dims:
                # Use hyperbolic distance for hyperbolic components
                dist = self.spaces[i].distance(x_comp, y_comp)
            else:
                # Use spherical distance for spherical components
                # For spherical spaces, distance is arccos(dot(x, y))
                x_norm = mx.sqrt(mx.sum(x_comp * x_comp, axis=-1) + self.epsilon)
                y_norm = mx.sqrt(mx.sum(y_comp * y_comp, axis=-1) + self.epsilon)
                
                # Normalize to the unit sphere
                x_normalized = x_comp / x_norm.reshape(x_norm.shape[:-1] + (1,))
                y_normalized = y_comp / y_norm.reshape(y_norm.shape[:-1] + (1,))
                
                # Compute cosine similarity
                cos_sim = mx.sum(x_normalized * y_normalized, axis=-1)
                # Clamp to valid arccos range
                cos_sim = mx.clip(cos_sim, -1.0 + self.epsilon, 1.0 - self.epsilon)
                
                # Compute spherical distance
                dist = mx.arccos(cos_sim)
                
            component_distances.append(dist)
        
        # Combine using L2 norm
        component_distances = mx.stack(component_distances, axis=-1)
        return mx.sqrt(mx.sum(component_distances ** 2, axis=-1))
    
    def project_to_manifold(self, x):
        """
        Project points onto the product manifold.
        
        For hyperbolic components, ensure points are inside the Poincaré ball.
        For spherical components, project onto the unit sphere.
        
        Args:
            x: Point tensor of shape (..., total_dim)
            
        Returns:
            Tensor with points projected onto the manifold
        """
        x_components = self._split_points(x)
        projected_components = []
        
        for i, x_comp in enumerate(x_components):
            if i in self.hyperbolic_dims:
                # Project to Poincaré ball
                x_norm = mx.sqrt(mx.sum(x_comp * x_comp, axis=-1, keepdims=True))
                # If norm >= 1, rescale to be inside the ball
                scale = mx.where(
                    x_norm >= 1.0 - self.epsilon,
                    (1.0 - self.epsilon) / (x_norm + self.epsilon),
                    mx.ones_like(x_norm)
                )
                projected_components.append(x_comp * scale)
            else:
                # Project to unit sphere (for spherical components)
                x_norm = mx.sqrt(mx.sum(x_comp * x_comp, axis=-1, keepdims=True) + self.epsilon)
                projected_components.append(x_comp / x_norm)
        
        return self._merge_points(projected_components)
    
    def exp_map(self, x, v):
        """
        Compute the exponential map for the product space.
        
        Args:
            x: Base point tensor of shape (..., total_dim)
            v: Tangent vector at x, shape (..., total_dim)
            
        Returns:
            Tensor representing exp_x(v) in the product space
        """
        x_components = self._split_points(x)
        v_components = self._split_points(v)
        
        result_components = []
        for i, (x_comp, v_comp) in enumerate(zip(x_components, v_components)):
            if i in self.hyperbolic_dims:
                # Use hyperbolic exponential map
                result_components.append(self.spaces[i].exp_map(x_comp, v_comp))
            else:
                # For spherical spaces, use the spherical exponential map
                # First, ensure x is on the unit sphere
                x_norm = mx.sqrt(mx.sum(x_comp * x_comp, axis=-1, keepdims=True) + self.epsilon)
                x_normalized = x_comp / x_norm
                
                # Project v to be tangent to the sphere at x
                v_dot_x = mx.sum(v_comp * x_normalized, axis=-1, keepdims=True)
                v_tangent = v_comp - v_dot_x * x_normalized
                
                v_norm = mx.sqrt(mx.sum(v_tangent * v_tangent, axis=-1, keepdims=True) + self.epsilon)
                
                # Compute the spherical exponential map
                cos_theta = mx.cos(v_norm)
                sin_theta = mx.sin(v_norm)
                
                result = cos_theta * x_normalized + sin_theta * (v_tangent / v_norm)
                result_components.append(result)
        
        return self._merge_points(result_components)
    
    def log_map(self, x, y):
        """
        Compute the logarithmic map for the product space.
        
        Args:
            x: Base point tensor of shape (..., total_dim)
            y: Target point tensor of shape (..., total_dim)
            
        Returns:
            Tensor representing log_x(y) in the product space
        """
        x_components = self._split_points(x)
        y_components = self._split_points(y)
        
        result_components = []
        for i, (x_comp, y_comp) in enumerate(zip(x_components, y_components)):
            if i in self.hyperbolic_dims:
                # Use hyperbolic logarithmic map
                result_components.append(self.spaces[i].log_map(x_comp, y_comp))
            else:
                # For spherical spaces, use the spherical logarithmic map
                # Ensure x and y are on the unit sphere
                x_norm = mx.sqrt(mx.sum(x_comp * x_comp, axis=-1, keepdims=True) + self.epsilon)
                y_norm = mx.sqrt(mx.sum(y_comp * y_comp, axis=-1, keepdims=True) + self.epsilon)
                
                x_normalized = x_comp / x_norm
                y_normalized = y_comp / y_norm
                
                # Compute the inner product
                inner_prod = mx.sum(x_normalized * y_normalized, axis=-1, keepdims=True)
                inner_prod = mx.clip(inner_prod, -1.0 + self.epsilon, 1.0 - self.epsilon)
                
                # Compute the distance
                theta = mx.arccos(inner_prod)
                
                # Compute the direction
                y_perp = y_normalized - inner_prod * x_normalized
                y_perp_norm = mx.sqrt(mx.sum(y_perp * y_perp, axis=-1, keepdims=True) + self.epsilon)
                
                # Avoid division by zero
                direction = mx.where(
                    y_perp_norm > self.epsilon,
                    y_perp / y_perp_norm,
                    mx.zeros_like(y_perp)
                )
                
                # Compute the logarithmic map
                result = theta * direction
                result_components.append(result)
        
        return self._merge_points(result_components)
    
    def update_curvatures(self, new_curvatures):
        """
        Update the curvature parameters of the mixed-curvature space.
        
        This allows for learning the optimal curvature for each component.
        
        Args:
            new_curvatures: Tensor of new curvature values
        """
        self.curvatures = mx.array(new_curvatures)
        
        # Update the component spaces with new curvatures
        for i, c in enumerate(self.curvatures):
            if i in self.hyperbolic_dims:
                self.spaces[i] = PoincareBall(self.dims[i], curvature=c)