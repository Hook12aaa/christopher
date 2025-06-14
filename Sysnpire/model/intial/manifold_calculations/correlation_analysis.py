"""
Correlation Analysis Engine

High-performance correlation calculations for emotional trajectory integration
in field theory applications. Supports E^trajectory(τ,s) computations in 
the complete Q(τ, C, s) conceptual charge formula.
"""

import numpy as np
import numba as nb
from typing import Tuple, List, Dict, Any


class CorrelationAnalyzer:
    """
    Enterprise-grade correlation analysis for field theory applications.
    
    MATHEMATICAL PURPOSE: Computes coupling properties essential for
    emotional trajectory integration E^trajectory(τ,s) in the complete
    conceptual charge formula Q(τ, C, s).
    
    PERFORMANCE: Numba-optimized correlation calculations provide significant
    speedups over scipy.stats implementations while maintaining numerical accuracy.
    """
    
    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def compute_correlation_coefficients(embedding: np.ndarray, 
                                        neighbors: np.ndarray) -> Tuple[float, float]:
        """
        Compute correlation statistics for coupling property analysis.
        
        FIELD THEORY APPLICATION: Calculates coupling properties needed for
        emotional trajectory integration E^trajectory(τ,s) in conceptual
        charge formula. Provides correlation-based coupling measures.
        
        NUMERICAL STABILITY: Manual correlation computation ensures numerical
        stability and numba compatibility while avoiding NaN propagation.
        
        Args:
            embedding: Central embedding vector [D]
            neighbors: Neighborhood embedding vectors [K, D]
            
        Returns:
            Tuple of correlation statistics:
            - coupling_mean: Average correlation coefficient
            - coupling_variance: Variance of correlation coefficients
        """
        correlations = np.empty(len(neighbors), dtype=np.float64)
        
        for i in range(len(neighbors)):
            neighbor = neighbors[i]
            
            # Manual correlation coefficient computation for numba compatibility
            emb_mean = np.sum(embedding) / len(embedding)
            neigh_mean = np.sum(neighbor) / len(neighbor)
            emb_centered = embedding - emb_mean
            neigh_centered = neighbor - neigh_mean
            
            # Pearson correlation: r = Σ(x-x̄)(y-ȳ) / √[Σ(x-x̄)²Σ(y-ȳ)²]
            numerator = np.sum(emb_centered * neigh_centered)
            emb_var = np.sum(emb_centered**2)
            neigh_var = np.sum(neigh_centered**2)
            denominator = np.sqrt(emb_var * neigh_var)
            
            if denominator > 1e-10:
                correlations[i] = numerator / denominator
            else:
                correlations[i] = 0.0
        
        # Filter NaN values and compute statistics
        valid_correlations = correlations[~np.isnan(correlations)]
        if len(valid_correlations) > 0:
            coupling_mean = np.sum(valid_correlations) / len(valid_correlations)
            # Manual variance calculation
            mean_val = coupling_mean
            var_sum = 0.0
            for val in valid_correlations:
                var_sum += (val - mean_val)**2
            coupling_variance = var_sum / len(valid_correlations)
        else:
            coupling_mean = 0.0
            coupling_variance = 0.0
        
        return coupling_mean, coupling_variance
    
    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def compute_cross_correlations(embedding: np.ndarray,
                                  neighbors: np.ndarray,
                                  max_lag: int = 5) -> np.ndarray:
        """
        Compute cross-correlation functions for temporal dynamics analysis.
        
        FIELD THEORY APPLICATION: Provides temporal correlation structure
        for trajectory operators T(τ,C,s) in dynamic field evolution.
        
        Args:
            embedding: Central embedding vector [D]
            neighbors: Neighborhood vectors [K, D]  
            max_lag: Maximum lag for cross-correlation
            
        Returns:
            Cross-correlation coefficients [max_lag]
        """
        cross_corrs = np.zeros(max_lag, dtype=np.float64)
        
        if len(neighbors) > max_lag:
            for lag in range(max_lag):
                if lag < len(neighbors) - 1:
                    # Cross-correlation at lag
                    neighbor1 = neighbors[0]
                    neighbor2 = neighbors[lag + 1] if lag + 1 < len(neighbors) else neighbors[-1]
                    
                    # Simplified cross-correlation computation
                    corr = np.dot(neighbor1, neighbor2) / (
                        np.linalg.norm(neighbor1) * np.linalg.norm(neighbor2) + 1e-10
                    )
                    cross_corrs[lag] = corr
        
        return cross_corrs
    
    @staticmethod
    def compute_correlation_matrix(embeddings: np.ndarray) -> np.ndarray:
        """
        Compute full correlation matrix for manifold structure analysis.
        
        ENTERPRISE FEATURE: Provides complete correlation structure for
        advanced field theory analysis and system diagnostics.
        
        Args:
            embeddings: Matrix of embedding vectors [N, D]
            
        Returns:
            Correlation matrix [N, N]
        """
        n = len(embeddings)
        correlation_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                # Use optimized correlation computation
                corr_mean, _ = CorrelationAnalyzer.compute_correlation_coefficients(
                    embeddings[i], embeddings[j:j+1]
                )
                correlation_matrix[i, j] = corr_mean
                correlation_matrix[j, i] = corr_mean
        
        return correlation_matrix
    
    @staticmethod
    def analyze_coupling_properties(embedding: np.ndarray,
                                   neighbors: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive coupling analysis for emotional trajectory integration.
        
        FIELD THEORY INTEGRATION: Provides complete coupling property analysis
        for E^trajectory(τ,s) emotional trajectory integration in Q(τ, C, s)
        conceptual charge calculations.
        
        Args:
            embedding: Central embedding vector
            neighbors: Local neighborhood embeddings
            
        Returns:
            Dictionary of coupling properties and correlation statistics
        """
        # Primary correlation statistics
        coupling_mean, coupling_variance = \
            CorrelationAnalyzer.compute_correlation_coefficients(embedding, neighbors)
        
        # Cross-correlation analysis
        cross_correlations = CorrelationAnalyzer.compute_cross_correlations(
            embedding, neighbors
        )
        
        # Correlation strength metrics
        correlation_strength = abs(coupling_mean)
        correlation_consistency = 1.0 / (1.0 + coupling_variance)
        
        # Temporal correlation decay
        if len(cross_correlations) > 1:
            temporal_decay = np.mean(np.diff(cross_correlations))
        else:
            temporal_decay = 0.0
        
        return {
            # Primary coupling metrics
            'coupling_mean': float(coupling_mean),
            'coupling_variance': float(coupling_variance),
            
            # Correlation analysis
            'correlation_strength': float(correlation_strength),
            'correlation_consistency': float(correlation_consistency),
            
            # Temporal dynamics
            'cross_correlations': cross_correlations.tolist(),
            'temporal_decay': float(temporal_decay),
            
            # Field theory parameters
            'emotional_coupling_factor': float(coupling_mean * correlation_consistency),
            'trajectory_persistence': float(1.0 - abs(temporal_decay))
        }
    
    @staticmethod
    def benchmark_correlation_performance(embeddings: np.ndarray,
                                        num_runs: int = 5) -> Dict[str, float]:
        """
        Performance benchmarking for correlation calculations.
        
        ENTERPRISE DIAGNOSTICS: Validates optimization effectiveness and
        provides performance metrics for production system monitoring.
        
        Args:
            embeddings: Test embedding matrix
            num_runs: Number of benchmark iterations
            
        Returns:
            Performance metrics for correlation calculations
        """
        import time
        
        # Select test embeddings
        central = embeddings[0]
        neighbors = embeddings[1:min(21, len(embeddings))]
        
        # Benchmark optimized implementation
        numba_times = []
        for _ in range(num_runs):
            start = time.time()
            _ = CorrelationAnalyzer.compute_correlation_coefficients(central, neighbors)
            numba_times.append(time.time() - start)
        
        return {
            'avg_time': np.mean(numba_times),
            'std_time': np.std(numba_times),
            'min_time': np.min(numba_times),
            'max_time': np.max(numba_times),
            'neighbor_count': len(neighbors),
            'embedding_dimension': len(central)
        }