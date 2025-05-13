"""
Poincaré Ball model implementation for hyperbolic embeddings.

This module provides functionality for working with the Poincaré ball model of hyperbolic
space, including operations like Möbius addition, Möbius scalar multiplication,
parallel transport, and distance calculations, which are essential for implementing
the neural manifold for our constructivist mathematics framework.
"""

import numpy as np
import mlx.core as mx

class PoincareBall:
    """
    Implementation of the Poincaré Ball model for hyperbolic geometry.
    
    The Poincaré ball model represents hyperbolic space as the interior of the
    unit ball in Euclidean space, with a metric that distorts distances to
    capture the exponential capacity growth characteristic of hyperbolic space.
    """
    
    def __init__(self, dim, curvature=-1.0):
        """
        Initialize the Poincaré Ball model.
        
        Args:
            dim (int): Dimension of the ball
            curvature (float): Curvature parameter, must be negative for hyperbolic space
        """
        self.dim = dim
        self.c = -curvature  # Store as positive for computational convenience
        self.epsilon = 1e-15  # Small value for numerical stability
        
    def _check_points_in_ball(self, x):
        """Check if points are inside the Poincaré ball."""
        norm = mx.sum(x * x, axis=-1)
        if mx.any(norm >= 1.0 - self.epsilon):
            raise ValueError("Points must be inside the unit ball.")
            
    def mobius_addition(self, x, y):
        """
        Compute the Möbius addition of points x and y in the Poincaré ball.
        
        Args:
            x: First point tensor of shape (..., dim)
            y: Second point tensor of shape (..., dim)
            
        Returns:
            Tensor representing the Möbius addition x ⊕ y
        """
        self._check_points_in_ball(x)
        self._check_points_in_ball(y)
        
        # Compute squared norms
        x_norm_sq = mx.sum(x * x, axis=-1, keepdims=True)
        y_norm_sq = mx.sum(y * y, axis=-1, keepdims=True)
        
        # Compute the numerator: x + y + c(x * y)
        xy_dot = mx.sum(x * y, axis=-1, keepdims=True)
        numerator = (1 + 2 * self.c * xy_dot + self.c * y_norm_sq) * x + (1 - self.c * x_norm_sq) * y
        
        # Compute the denominator: 1 + 2c<x,y> + c²|x|²|y|²
        denominator = 1 + 2 * self.c * xy_dot + self.c**2 * x_norm_sq * y_norm_sq
        
        return numerator / denominator
    
    def mobius_scalar_mul(self, r, x):
        """
        Compute the Möbius scalar multiplication of point x by scalar r.
        
        Args:
            r: Scalar value
            x: Point tensor of shape (..., dim)
            
        Returns:
            Tensor representing the Möbius scalar multiplication r ⊗ x
        """
        self._check_points_in_ball(x)
        
        x_norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
        # Avoid division by zero
        safe_norm = mx.maximum(x_norm, self.epsilon)
        
        return mx.tanh(r * mx.atanh(mx.sqrt(self.c) * safe_norm)) / (mx.sqrt(self.c) * safe_norm) * x
    
    def distance(self, x, y):
        """
        Compute the hyperbolic distance between points x and y.
        
        Args:
            x: First point tensor of shape (..., dim)
            y: Second point tensor of shape (..., dim)
            
        Returns:
            Tensor representing the hyperbolic distance d_H(x, y)
        """
        self._check_points_in_ball(x)
        self._check_points_in_ball(y)
        
        # Compute the Möbius addition of x and -y
        minus_y = -y
        sum_sq = mx.sum((x - y) * (x - y), axis=-1)
        prod_sq = mx.sum(x * x, axis=-1) * mx.sum(y * y, axis=-1)
        
        # Compute the distance formula with numerical stability
        numer = 2 * sum_sq
        denom = (1 - self.c * prod_sq) + self.epsilon
        
        return (2 / mx.sqrt(self.c)) * mx.asinh(mx.sqrt(self.c * numer / denom) / 2)
    
    def exp_map(self, x, v):
        """
        Compute the exponential map at point x with tangent vector v.
        
        The exponential map takes a tangent vector at a point in the manifold
        and returns the point reached by following the geodesic in the direction
        of the tangent vector for a unit time.
        
        Args:
            x: Point tensor of shape (..., dim)
            v: Tangent vector at x, shape (..., dim)
            
        Returns:
            Tensor representing exp_x(v)
        """
        self._check_points_in_ball(x)
        
        v_norm = mx.sqrt(mx.sum(v * v, axis=-1, keepdims=True))
        # Avoid division by zero
        safe_norm = mx.maximum(v_norm, self.epsilon)
        
        second_term = mx.tanh(mx.sqrt(self.c) * safe_norm / 2) * v / (mx.sqrt(self.c) * safe_norm)
        
        return self.mobius_addition(x, second_term)
    
    def log_map(self, x, y):
        """
        Compute the logarithmic map at point x with destination point y.
        
        The logarithmic map is the inverse of the exponential map and returns
        the tangent vector at x that points toward y along the geodesic.
        
        Args:
            x: Point tensor of shape (..., dim)
            y: Destination point tensor of shape (..., dim)
            
        Returns:
            Tensor representing log_x(y)
        """
        self._check_points_in_ball(x)
        self._check_points_in_ball(y)
        
        # Compute the Möbius addition of -x and y
        minus_x = -x
        addition = self.mobius_addition(minus_x, y)
        
        addition_norm = mx.sqrt(mx.sum(addition * addition, axis=-1, keepdims=True))
        # Avoid division by zero
        safe_norm = mx.maximum(addition_norm, self.epsilon)
        
        return (2 / mx.sqrt(self.c)) * mx.atanh(mx.sqrt(self.c) * safe_norm) * addition / safe_norm
    
    def parallel_transport(self, x, y, v):
        """
        Parallel transport of tangent vector v from point x to point y.
        
        Parallel transport moves a tangent vector along a geodesic while
        preserving its length and angle with the geodesic.
        
        Args:
            x: Source point tensor of shape (..., dim)
            y: Destination point tensor of shape (..., dim)
            v: Tangent vector at x, shape (..., dim)
            
        Returns:
            Tensor representing the transported vector at y
        """
        self._check_points_in_ball(x)
        self._check_points_in_ball(y)
        
        x_y_dist = self.distance(x, y)
        log_xy = self.log_map(x, y)
        log_xy_norm = mx.sqrt(mx.sum(log_xy * log_xy, axis=-1, keepdims=True))
        
        # Avoid division by zero
        safe_norm = mx.maximum(log_xy_norm, self.epsilon)
        
        # Parallel transport formula for the Poincaré ball
        v_dot_log = mx.sum(v * log_xy, axis=-1, keepdims=True) / (safe_norm ** 2)
        return v - v_dot_log * (x + y)
    
    def to_klein(self, x):
        """
        Convert from Poincaré ball to Klein model.
        
        Args:
            x: Point tensor in Poincaré ball, shape (..., dim)
            
        Returns:
            Tensor representing the point in Klein model
        """
        self._check_points_in_ball(x)
        
        x_norm_sq = mx.sum(x * x, axis=-1, keepdims=True)
        return 2 * x / (1 + x_norm_sq)
    
    def from_klein(self, k):
        """
        Convert from Klein model to Poincaré ball.
        
        Args:
            k: Point tensor in Klein model, shape (..., dim)
            
        Returns:
            Tensor representing the point in Poincaré ball
        """
        k_norm_sq = mx.sum(k * k, axis=-1, keepdims=True)
        if mx.any(k_norm_sq >= 1.0):
            raise ValueError("Klein coordinates must be inside the unit ball.")
            
        return k / (1 + mx.sqrt(1 - k_norm_sq))
    
    def geodesic(self, x, y, t):
        """
        Compute points along the geodesic between x and y.
        
        Args:
            x: Start point tensor of shape (..., dim)
            y: End point tensor of shape (..., dim)
            t: Parameter controlling the position along the geodesic, 0 <= t <= 1
            
        Returns:
            Tensor representing the point at position t along the geodesic
        """
        self._check_points_in_ball(x)
        self._check_points_in_ball(y)
        
        # Compute the tangent vector from x to y
        v = self.log_map(x, y)
        
        # Scale the tangent vector by t
        tv = t * v
        
        # Apply the exponential map to get the point at parameter t
        return self.exp_map(x, tv)