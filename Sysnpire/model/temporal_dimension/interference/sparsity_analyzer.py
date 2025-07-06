"""
Sparsity Analyzer for BGE-based Temporal Field Interference

Analyzes whether BGE embeddings create naturally sparse interference patterns
to determine optimal computation strategy.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from scipy.stats import pearsonr
from dataclasses import dataclass

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.utils.logger import get_logger
logger = get_logger(__name__)


@dataclass
class SparsityAnalysis:
    """Results from interference sparsity analysis."""
    sparsity_ratio: float           # Fraction of pairs with negligible interference
    correlation_strength: float     # Correlation between BGE distance and interference
    mean_neighbors: float          # Average meaningful neighbors per charge
    sparsity_threshold: float      # Threshold used for sparsity detection
    sample_size: int               # Number of charges analyzed
    computation_strategy: str      # Recommended strategy: 'sparse', 'chunked', 'hybrid'


class SparsityAnalyzer:
    """Analyze BGE semantic structure to determine optimal interference computation strategy."""
    
    def __init__(self, sparsity_threshold: float = 0.01):
        """
        Initialize sparsity analyzer.
        
        Args:
            sparsity_threshold: Interference values below this fraction of max are considered sparse
        """
        self.sparsity_threshold = sparsity_threshold
    
    def analyze_sample(self, temporal_biographies: List, sample_size: int = 100) -> SparsityAnalysis:
        """
        Analyze sparsity patterns in a sample of temporal biographies.
        
        Args:
            temporal_biographies: Full list of temporal biographies
            sample_size: Number of charges to sample for analysis
            
        Returns:
            SparsityAnalysis with recommended computation strategy
        """
        logger.info(f"🔍 Analyzing interference sparsity with {sample_size} charge sample")
        
        # Sample charges for analysis
        n_total = len(temporal_biographies)
        if sample_size >= n_total:
            sample_indices = list(range(n_total))
            sample_size = n_total
        else:
            sample_indices = np.random.choice(n_total, sample_size, replace=False)
        
        sample_biographies = [temporal_biographies[i] for i in sample_indices]
        
        # Compute full interference matrix for sample
        interference_matrix = self._compute_sample_interference_matrix(sample_biographies)
        
        # Compute BGE semantic distances
        semantic_distances = self._compute_semantic_distances(sample_biographies)
        
        # Analyze sparsity patterns
        sparsity_ratio = self._compute_sparsity_ratio(interference_matrix)
        correlation_strength = self._compute_distance_correlation(semantic_distances, interference_matrix)
        mean_neighbors = self._estimate_mean_neighbors(interference_matrix)
        
        # Determine recommended strategy
        strategy = self._recommend_strategy(sparsity_ratio, correlation_strength, mean_neighbors)
        
        logger.info(f"📊 Sparsity analysis complete:")
        logger.info(f"   Sparsity ratio: {sparsity_ratio:.1%}")
        logger.info(f"   Distance correlation: {correlation_strength:.3f}")
        logger.info(f"   Average neighbors: {mean_neighbors:.1f}")
        logger.info(f"   Recommended strategy: {strategy}")
        
        return SparsityAnalysis(
            sparsity_ratio=sparsity_ratio,
            correlation_strength=correlation_strength,
            mean_neighbors=mean_neighbors,
            sparsity_threshold=self.sparsity_threshold,
            sample_size=sample_size,
            computation_strategy=strategy
        )
    
    def _compute_sample_interference_matrix(self, sample_biographies: List) -> np.ndarray:
        """Compute exact interference matrix for sample using original algorithm."""
        n_sample = len(sample_biographies)
        interference_matrix = np.zeros((n_sample, n_sample), dtype=complex)
        
        for i in range(n_sample):
            for j in range(n_sample):
                if i != j:
                    bio_i = sample_biographies[i]
                    bio_j = sample_biographies[j]
                    
                    # Phase interference
                    phase_interference = np.mean(
                        np.exp(1j * (bio_i.phase_coordination - bio_j.phase_coordination))
                    )
                    
                    # Trajectory interference
                    trajectory_interference = np.mean(
                        bio_i.trajectory_operators * np.conj(bio_j.trajectory_operators)
                    )
                    
                    interference_matrix[i, j] = phase_interference * trajectory_interference
        
        return interference_matrix
    
    def _compute_semantic_distances(self, sample_biographies: List) -> np.ndarray:
        """Compute pairwise BGE semantic distances."""
        n_sample = len(sample_biographies)
        
        # Use trajectory operators as BGE semantic representation
        trajectories = np.array([bio.trajectory_operators for bio in sample_biographies])
        
        # Compute pairwise Euclidean distances
        distances = np.zeros((n_sample, n_sample))
        for i in range(n_sample):
            for j in range(n_sample):
                if i != j:
                    distances[i, j] = np.linalg.norm(trajectories[i] - trajectories[j])
        
        return distances
    
    def _compute_sparsity_ratio(self, interference_matrix: np.ndarray) -> float:
        """Compute fraction of interference values that are negligible."""
        # Get non-diagonal values
        mask = ~np.eye(interference_matrix.shape[0], dtype=bool)
        interference_values = np.abs(interference_matrix[mask])
        
        # Find threshold based on maximum value
        max_interference = np.max(interference_values)
        threshold = max_interference * self.sparsity_threshold
        
        # Count sparse values
        sparse_count = np.sum(interference_values < threshold)
        total_count = len(interference_values)
        
        return sparse_count / total_count
    
    def _compute_distance_correlation(self, distances: np.ndarray, interference_matrix: np.ndarray) -> float:
        """Compute correlation between semantic distance and interference strength."""
        # Get non-diagonal values
        mask = ~np.eye(distances.shape[0], dtype=bool)
        distance_values = distances[mask]
        interference_values = np.abs(interference_matrix[mask])
        
        # Compute Pearson correlation
        if len(distance_values) > 1 and np.std(distance_values) > 0 and np.std(interference_values) > 0:
            correlation, _ = pearsonr(distance_values, interference_values)
            return abs(correlation)  # Return absolute correlation strength
        else:
            return 0.0
    
    def _estimate_mean_neighbors(self, interference_matrix: np.ndarray) -> float:
        """Estimate average number of meaningful neighbors per charge."""
        # Get non-diagonal values
        interference_magnitudes = np.abs(interference_matrix)
        np.fill_diagonal(interference_magnitudes, 0)
        
        # Find threshold for meaningful interactions
        max_interference = np.max(interference_magnitudes)
        meaningful_threshold = max_interference * self.sparsity_threshold
        
        # Count meaningful neighbors per charge
        meaningful_neighbors = np.sum(interference_magnitudes > meaningful_threshold, axis=1)
        
        return np.mean(meaningful_neighbors)
    
    def _recommend_strategy(self, sparsity_ratio: float, correlation_strength: float, 
                           mean_neighbors: float) -> str:
        """Recommend computation strategy based on analysis results."""
        
        # Strong sparsity indicators
        if sparsity_ratio > 0.8 and correlation_strength > 0.5:
            return "sparse"
        
        # Weak sparsity - dense computation needed
        elif sparsity_ratio < 0.5:
            return "chunked"
        
        # Mixed case - adaptive approach
        else:
            return "hybrid"


if __name__ == "__main__":
    """Test sparsity analyzer with sample data."""
    print("SparsityAnalyzer ready for BGE interference pattern analysis")