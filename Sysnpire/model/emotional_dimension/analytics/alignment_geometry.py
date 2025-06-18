"""
Alignment Geometry Analysis - Discovering Emotional Patterns through Geometric Relationships

KEY INSIGHT: Emotional content creates distinctive geometric patterns in BGE space:
- Stronger dot product alignments between emotionally related concepts
- Larger vector magnitudes for emotionally charged content
- Consistent angular clustering of emotional patterns

NO EMOTION LABELS - only geometric pattern discovery.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from dataclasses import dataclass

from Sysnpire.utils.logger import get_logger
logger = get_logger(__name__)


@dataclass
class GeometricEmotionalPattern:
    """
    Discovered geometric pattern that may indicate emotional field effects.
    NO emotion labels - only mathematical properties.
    """
    cluster_centers: np.ndarray          # Centers of discovered clusters
    alignment_strengths: np.ndarray      # Dot product alignment patterns
    magnitude_profile: np.ndarray        # Vector magnitude distribution
    angular_relationships: np.ndarray    # Angular clustering patterns
    coherence_score: float              # How coherent this pattern is
    field_strength: float               # Estimated field modulation strength


class AlignmentGeometryAnalyzer:
    """
    Discovers intrinsic emotional patterns through geometric analysis of BGE embeddings.
    
    PHILOSOPHY: Emotions manifest as geometric distortions - we discover these
    patterns without imposing categorical labels.
    """
    
    def __init__(self, embedding_dim: int = 1024):
        """Initialize geometric pattern analyzer."""
        self.embedding_dim = embedding_dim
        self.discovered_patterns = []
        
    def discover_alignment_patterns(self, embeddings: np.ndarray) -> List[GeometricEmotionalPattern]:
        """
        Discover natural alignment patterns that may indicate emotional field effects.
        
        APPROACH:
        1. Analyze dot product distributions to find anomalous alignments
        2. Detect magnitude outliers that suggest field amplification
        3. Find angular clustering that indicates coherent field regions
        
        Args:
            embeddings: Array of BGE embeddings to analyze
            
        Returns:
            List of discovered geometric patterns (no emotion labels)
        """
        logger.info(f"🔍 Discovering geometric patterns in {len(embeddings)} embeddings")
        
        patterns = []
        
        # 1. Analyze alignment strength distributions
        alignment_patterns = self._analyze_alignment_distributions(embeddings)
        
        # 2. Detect magnitude anomalies
        magnitude_patterns = self._detect_magnitude_patterns(embeddings)
        
        # 3. Discover angular clustering
        angular_patterns = self._discover_angular_clusters(embeddings)
        
        # 4. Combine patterns to identify coherent field regions
        field_regions = self._identify_field_regions(
            alignment_patterns, magnitude_patterns, angular_patterns
        )
        
        for region in field_regions:
            pattern = GeometricEmotionalPattern(
                cluster_centers=region['centers'],
                alignment_strengths=region['alignments'],
                magnitude_profile=region['magnitudes'],
                angular_relationships=region['angles'],
                coherence_score=region['coherence'],
                field_strength=region['strength']
            )
            patterns.append(pattern)
            
        logger.info(f"✨ Discovered {len(patterns)} geometric field patterns")
        return patterns
    
    def _analyze_alignment_distributions(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze dot product alignment distributions to find anomalous patterns.
        
        Theory: Emotional content creates stronger alignment scores between
        related concepts - we detect these without labeling them.
        """
        # Compute pairwise cosine similarities
        similarities = cosine_similarity(embeddings)
        
        # Find anomalously high alignments (potential emotional coherence)
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Detect alignments with minimal threshold (more sensitive for small datasets)
        strong_alignments = similarities > (mean_sim + 0.1 * std_sim)
        
        # Extract alignment strength patterns
        alignment_strengths = np.sum(strong_alignments, axis=1)
        
        return {
            'similarity_matrix': similarities,
            'strong_alignments': strong_alignments,
            'alignment_strengths': alignment_strengths,
            'alignment_threshold': mean_sim + 2 * std_sim
        }
    
    def _detect_magnitude_patterns(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect magnitude patterns that suggest emotional field amplification.
        
        Theory: Emotional content typically has larger vector norms due to
        field amplification effects.
        """
        # Calculate vector magnitudes
        magnitudes = np.linalg.norm(embeddings, axis=1)
        
        # Find magnitude outliers
        mag_mean = np.mean(magnitudes)
        mag_std = np.std(magnitudes)
        
        # Identify high-magnitude vectors (potential emotional amplification) - lowered threshold
        high_magnitude_mask = magnitudes > (mag_mean + 0.5 * mag_std)
        
        # Analyze magnitude gradients
        magnitude_gradients = np.gradient(magnitudes)
        
        return {
            'magnitudes': magnitudes,
            'high_magnitude_indices': np.where(high_magnitude_mask)[0],
            'magnitude_gradients': magnitude_gradients,
            'amplification_threshold': mag_mean + 1.5 * mag_std
        }
    
    def _discover_angular_clusters(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Discover natural angular clustering patterns in embedding space.
        
        Theory: Emotions create consistent angular relationships - we find
        these clusters without labeling them as specific emotions.
        """
        # Normalize embeddings to unit sphere for angular analysis
        normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Use DBSCAN for density-based clustering in angular space
        # eps is in terms of angular distance (1-cosine_similarity)
        angular_distances = 1 - cosine_similarity(normalized)
        
        # Fix floating-point precision issues
        # 1. Clip negative values to zero
        angular_distances = np.clip(angular_distances, 0, None)
        
        # 2. Ensure matrix is symmetric (distance matrices should be symmetric)
        angular_distances = (angular_distances + angular_distances.T) / 2
        
        # 3. Ensure diagonal is exactly zero (distance from point to itself)
        np.fill_diagonal(angular_distances, 0)
        
        clustering = DBSCAN(eps=0.3, min_samples=1, metric='precomputed')  # Accept all points for small datasets
        cluster_labels = clustering.fit_predict(angular_distances)
        
        # Find cluster centers and properties
        unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
        cluster_info = []
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_vectors = normalized[cluster_mask]
            
            # Compute cluster center (average direction)
            center = np.mean(cluster_vectors, axis=0)
            center = center / (np.linalg.norm(center) + 1e-8)
            
            # Compute cluster coherence (how tightly packed)
            coherence = np.mean(cosine_similarity(cluster_vectors, center.reshape(1, -1)))
            
            cluster_info.append({
                'id': cluster_id,
                'center': center,
                'size': np.sum(cluster_mask),
                'coherence': coherence,
                'member_indices': np.where(cluster_mask)[0]
            })
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_info': cluster_info,
            'n_clusters': len(unique_clusters),
            'noise_points': np.sum(cluster_labels == -1)
        }
    
    def _identify_field_regions(self, alignment_patterns: Dict, magnitude_patterns: Dict, 
                               angular_patterns: Dict) -> List[Dict]:
        """
        Combine geometric patterns to identify coherent field regions.
        
        These regions represent areas where multiple geometric indicators
        suggest emotional field modulation.
        """
        field_regions = []
        
        # For each angular cluster, check if it also shows alignment/magnitude patterns
        for cluster_info in angular_patterns['cluster_info']:
            indices = cluster_info['member_indices']
            
            # Check alignment strength in this cluster
            cluster_alignment = np.mean(alignment_patterns['alignment_strengths'][indices])
            
            # Check magnitude profile
            cluster_magnitudes = magnitude_patterns['magnitudes'][indices]
            
            # Compute field strength estimate
            field_strength = self._estimate_field_strength(
                cluster_alignment,
                np.mean(cluster_magnitudes),
                cluster_info['coherence']
            )
            
            if field_strength > 0.1:  # Threshold for significant field effect (lowered to detect real patterns)
                field_regions.append({
                    'centers': cluster_info['center'].reshape(1, -1),
                    'alignments': alignment_patterns['alignment_strengths'][indices],
                    'magnitudes': cluster_magnitudes,
                    'angles': np.array([cluster_info['coherence']]),
                    'coherence': cluster_info['coherence'],
                    'strength': field_strength
                })
        
        return field_regions
    
    def _estimate_field_strength(self, alignment: float, magnitude: float, 
                                coherence: float) -> float:
        """
        Estimate emotional field strength from geometric indicators.
        
        This is a heuristic combination - no emotion labels, just field strength.
        """
        # Normalize inputs to [0, 1] range (assuming reasonable bounds)
        norm_alignment = min(alignment / 10.0, 1.0)  # Alignment strength scale
        norm_magnitude = min(magnitude / 5.0, 1.0)   # Magnitude scale
        
        # Combine indicators (weighted geometric mean)
        field_strength = (norm_alignment * norm_magnitude * coherence) ** (1/3)
        
        return field_strength


if __name__ == "__main__":
    """Test with real BGE embeddings following FoundationManifoldBuilder pattern."""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from Sysnpire.model.initial.bge_ingestion import BGEIngestion
    
    # Follow the same pattern as FoundationManifoldBuilder
    bge_model = BGEIngestion(model_name="BAAI/bge-large-en-v1.5", random_seed=42)
    model_loaded = bge_model.load_total_embeddings()
    
    if model_loaded is None:
        print("❌ Could not load model embeddings")
        exit(1)
    
    # Use same slice as FoundationManifoldBuilder for consistency  
    test_embeddings = model_loaded['embeddings'][550:560]
    embeddings = np.array(test_embeddings)
    
    print(f"Testing with {len(embeddings)} real BGE embeddings (indices 550-560)")
    
    analyzer = AlignmentGeometryAnalyzer()
    patterns = analyzer.discover_alignment_patterns(embeddings)
    
    print(f"Discovered {len(patterns)} geometric field patterns")
    for i, pattern in enumerate(patterns):
        print(f"Pattern {i}: field_strength={pattern.field_strength:.3f}, "
              f"coherence={pattern.coherence_score:.3f}")