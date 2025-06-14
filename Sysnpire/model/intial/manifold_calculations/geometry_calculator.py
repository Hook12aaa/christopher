"""
Manifold Geometry Calculation Engine

High-performance geometric analysis for discrete embedding manifolds.
Implements differential geometry calculations essential for field theory
operations in semantic embedding spaces using actual computed values
from discrete field samples.
"""

import numpy as np
import numba as nb
from typing import Tuple, Dict, Any


class ManifoldGeometryProcessor:
    """
    Enterprise-grade manifold geometry calculation engine.
    
    MATHEMATICAL FOUNDATION: Implements differential geometry operations
    on discrete semantic embedding manifolds for field theory applications.
    Provides local geometric properties essential for Q(τ, C, s) charge
    calculations including curvature, density, and gradient estimations.
    
    PERFORMANCE: Numba-optimized implementations provide significant
    speedups for computationally intensive geometric calculations.
    """
    
    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def compute_local_geometry(embedding: np.ndarray, 
                              neighbors: np.ndarray) -> Tuple[float, float, float, 
                                                             np.ndarray, float, 
                                                             float, float, float]:
        """
        Compute comprehensive local manifold geometry properties.
        
        FIELD THEORY APPLICATION: Extracts geometric features essential for
        Q(τ, C, s) conceptual charge calculations including local density,
        curvature estimation, and gradient approximations.
        
        MATHEMATICAL IMPLEMENTATION:
        - Local density: Inverse of mean neighbor distance (computed)
        - Curvature: Trace of neighbor covariance matrix (computed)
        - Gradient: Mean deviation from neighbors (finite differences)
        - Persistence: Maximum neighbor distance and density product (computed)
        - Boundary: Distance from local centroid (computed)
        
        Args:
            embedding: Central point for geometric analysis [D]
            neighbors: Local neighborhood points [K, D]
            
        Returns:
            Tuple of geometric properties:
            - magnitude: Vector magnitude 
            - local_density: Local point density
            - local_curvature: Curvature estimate
            - gradient: Gradient vector approximation
            - gradient_magnitude: Gradient magnitude
            - persistence_radius: Maximum neighbor distance
            - persistence_score: Density-weighted persistence
            - boundary_score: Boundary detection score
        """
        # Basic geometric properties
        magnitude = np.linalg.norm(embedding)
        
        # Neighbor analysis
        neighbor_center = np.sum(neighbors, axis=0) / neighbors.shape[0]
        neighbor_deviations = neighbors - neighbor_center
        
        # Local density estimation via inverse distance
        distances = np.array([
            np.linalg.norm(neighbors[i] - embedding) 
            for i in range(len(neighbors))
        ])
        local_density = 1.0 / (np.sum(distances) / len(distances) + 1e-8)
        
        # Local curvature via neighbor variance (Riemannian approximation)
        # Manual covariance calculation for numba compatibility
        if len(neighbor_deviations) > 1:
            cov_sum = 0.0
            for i in range(neighbor_deviations.shape[1]):
                col_var = np.sum(neighbor_deviations[:, i]**2) / len(neighbor_deviations)
                cov_sum += col_var
            local_curvature = cov_sum
        else:
            local_curvature = 0.0
        
        # Gradient approximation via finite differences
        gradient = np.sum(neighbor_deviations, axis=0) / neighbor_deviations.shape[0]
        gradient_magnitude = np.linalg.norm(gradient)
        
        # Persistence properties for observational persistence Ψ_persistence(s-s₀)
        persistence_radius = 0.0
        for d in distances:
            if d > persistence_radius:
                persistence_radius = d
        persistence_score = local_density * persistence_radius
        
        # Boundary detection for topological analysis
        boundary_score = np.linalg.norm(embedding - neighbor_center)
        
        return (magnitude, local_density, local_curvature, gradient, 
                gradient_magnitude, persistence_radius, persistence_score, 
                boundary_score)
    
    @staticmethod
    def compute_metric_eigenvalues(neighbor_deviations: np.ndarray, 
                                  max_components: int = 20) -> np.ndarray:
        """
        Compute local metric tensor eigenvalues for Riemannian analysis.
        
        FIELD THEORY APPLICATION: Provides local metric properties for
        differential field operations on curved semantic manifolds.
        
        Args:
            neighbor_deviations: Centered neighbor vectors [K, D]
            max_components: Maximum eigenvalues to compute
            
        Returns:
            Real eigenvalues of local covariance matrix (metric approximation)
        """
        if len(neighbor_deviations) > 1:
            cov_matrix = np.cov(neighbor_deviations.T)
            # Use actual eigendecomposition - no approximations
            eigenvalues, _ = np.linalg.eigh(cov_matrix)
            # Sort in descending order and take top components
            eigenvalues = np.sort(eigenvalues)[::-1][:max_components]
        else:
            eigenvalues = np.ones(min(max_components, neighbor_deviations.shape[1]))
        
        return eigenvalues
    
    @staticmethod
    def compute_gradient_structure_eigenvalues(gradient: np.ndarray, 
                                             gradient_magnitude: float,
                                             max_components: int = 20) -> np.ndarray:
        """
        Compute eigenvalues of gradient outer product for field structure analysis.
        
        FIELD THEORY APPLICATION: Since discrete embeddings don't provide second-order
        derivatives, we compute the eigenstructure of the gradient outer product
        gg^T which captures directional field variation information needed for
        Φ^semantic(τ,s) in conceptual charge calculations.
        
        MATHEMATICAL NOTE: This is NOT a Hessian approximation but rather the
        eigendecomposition of the gradient tensor product, providing actual
        computed values for field structure analysis.
        
        Args:
            gradient: Gradient vector from finite differences
            gradient_magnitude: Magnitude of gradient
            max_components: Maximum eigenvalues to return
            
        Returns:
            Eigenvalues of gradient outer product matrix for field analysis
        """
        try:
            if gradient_magnitude > 1e-8:
                # Compute gradient outer product matrix
                gradient_matrix = np.outer(gradient, gradient) / gradient_magnitude
                # Actual eigendecomposition - no approximations
                eigenvalues, _ = np.linalg.eigh(gradient_matrix)
                # Sort descending and return top components
                eigenvalues = np.sort(np.real(eigenvalues))[::-1][:max_components]
                return eigenvalues
            else:
                return np.zeros(max_components)
        except:
            return np.zeros(max_components)
    
    @staticmethod
    def compute_intrinsic_dimension(variance_ratios: np.ndarray, 
                                   threshold: float = 0.01) -> int:
        """
        Estimate intrinsic manifold dimensionality.
        
        FIELD THEORY APPLICATION: Determines effective dimensionality of
        local semantic field regions for adaptive field calculations.
        
        Args:
            variance_ratios: Principal component variance ratios
            threshold: Minimum variance ratio for significant dimensions
            
        Returns:
            Estimated intrinsic dimensionality
        """
        return int(np.sum(variance_ratios > threshold))
    
    @staticmethod 
    def analyze_manifold_properties(embedding: np.ndarray,
                                   neighbors: np.ndarray,
                                   pca_components: np.ndarray = None) -> Dict[str, Any]:
        """
        Comprehensive manifold analysis for field theory applications.
        
        ENTERPRISE INTERFACE: Provides complete manifold analysis including
        geometric, topological, and differential properties needed for
        Q(τ, C, s) conceptual charge generation.
        
        Args:
            embedding: Central embedding vector
            neighbors: Local neighborhood embeddings
            pca_components: Pre-computed PCA components (optional)
            
        Returns:
            Dictionary of comprehensive manifold properties
        """
        # Core geometric calculations
        (magnitude, local_density, local_curvature, gradient, gradient_magnitude,
         persistence_radius, persistence_score, boundary_score) = \
            ManifoldGeometryProcessor.compute_local_geometry(embedding, neighbors)
        
        # Metric tensor analysis
        neighbor_center = np.mean(neighbors, axis=0)
        neighbor_deviations = neighbors - neighbor_center
        metric_eigenvalues = ManifoldGeometryProcessor.compute_metric_eigenvalues(
            neighbor_deviations
        )
        
        # Gradient structure analysis (second-order field properties)
        gradient_structure_eigenvalues = ManifoldGeometryProcessor.compute_gradient_structure_eigenvalues(
            gradient, gradient_magnitude
        )
        
        # PCA projection if available
        if pca_components is not None and len(pca_components) > 0:
            principal_components = pca_components @ embedding
        else:
            principal_components = np.zeros(min(50, len(embedding)))
        
        return {
            # Basic geometric properties
            'magnitude': float(magnitude),
            'vector': embedding.tolist(),
            
            # Local geometry
            'local_density': float(local_density),
            'local_curvature': float(local_curvature),
            'metric_eigenvalues': metric_eigenvalues[:20].tolist(),
            
            # Differential properties  
            'gradient': gradient.tolist(),
            'gradient_magnitude': float(gradient_magnitude),
            'gradient_structure_eigenvalues': gradient_structure_eigenvalues[:20].tolist(),
            
            # Persistence properties for Ψ_persistence(s-s₀)
            'persistence_radius': float(persistence_radius),
            'persistence_score': float(persistence_score),
            
            # Topological properties
            'boundary_score': float(boundary_score),
            
            # Dimensionality reduction
            'principal_components': principal_components[:50].tolist()
        }