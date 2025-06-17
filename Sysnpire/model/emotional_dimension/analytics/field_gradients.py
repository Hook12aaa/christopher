"""
Field Gradient Analysis - Discovering Emotional Attractors and Flow Patterns

KEY INSIGHT: Emotional content creates gradient fields where embeddings naturally
"flow" toward certain attractors. We discover these flow patterns without
categorizing them as specific emotions.

APPROACH:
- Gradient analysis to find flow directions
- Attractor detection through local minima/maxima
- Flow field visualization
- Curvature estimation for field warping
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_filter
from scipy.spatial import Voronoi, ConvexHull
from scipy.optimize import minimize_scalar, minimize
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass

from Sysnpire.utils.logger import get_logger
logger = get_logger(__name__)


@dataclass  
class EmotionalFieldFlow:
    """
    Discovered field flow pattern indicating emotional dynamics.
    NO emotion labels - only field dynamics.
    """
    attractor_positions: np.ndarray      # Positions of field attractors
    attractor_strengths: np.ndarray      # Strength of each attractor
    gradient_field: np.ndarray          # Local gradient directions
    flow_magnitude: np.ndarray          # Strength of flow at each point
    curvature_map: np.ndarray          # Field curvature (warping)
    basin_boundaries: List[np.ndarray]  # Boundaries between attractor basins
    field_coherence: float              # Overall coherence of flow field


class FieldGradientAnalyzer:
    """
    Discovers emotional field dynamics through gradient analysis.
    
    PHILOSOPHY: Emotions create "gravitational" effects in embedding space,
    pulling meanings toward emotional attractors. We discover these dynamics
    without labeling the attractors.
    """
    
    def __init__(self, embedding_dim: int = 1024):
        """Initialize field gradient analyzer."""
        self.embedding_dim = embedding_dim
        self.n_neighbors = 10  # For local gradient estimation
        
    def discover_field_flows(self, embeddings: np.ndarray) -> List[EmotionalFieldFlow]:
        """
        Discover natural flow patterns in embedding space.
        
        Args:
            embeddings: Array of BGE embeddings
            
        Returns:
            List of discovered field flow patterns
        """
        logger.info(f"🌊 Discovering field flows in {len(embeddings)} embeddings")
        
        # 1. Estimate local gradients
        gradient_field = self._estimate_gradient_field(embeddings)
        
        # 2. Find attractors (convergence points)
        attractors = self._find_attractors(embeddings, gradient_field)
        
        # 3. Estimate field curvature
        curvature = self._estimate_curvature(embeddings, gradient_field)
        
        # 4. Identify attractor basins
        basins = self._identify_attractor_basins(embeddings, attractors)
        
        # 5. Calculate flow coherence
        coherence = self._calculate_field_coherence(gradient_field)
        
        # Create field flow description
        flow = EmotionalFieldFlow(
            attractor_positions=attractors['positions'],
            attractor_strengths=attractors['strengths'],
            gradient_field=gradient_field,
            flow_magnitude=np.linalg.norm(gradient_field, axis=1),
            curvature_map=curvature,
            basin_boundaries=basins['boundaries'],
            field_coherence=coherence
        )
        
        logger.info(f"💫 Found {len(attractors['positions'])} field attractors")
        return [flow]
    
    def _estimate_gradient_field(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Estimate local gradient field using nearest neighbors.
        
        Theory: The gradient shows the direction of strongest change,
        revealing how embeddings "flow" through the space.
        """
        n_samples = len(embeddings)
        gradient_field = np.zeros_like(embeddings)
        
        # Build nearest neighbor structure
        # Ensure we don't request more neighbors than available samples
        n_neighbors_adjusted = min(self.n_neighbors + 1, n_samples)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors_adjusted)
        nbrs.fit(embeddings)
        
        for i in range(n_samples):
            # Find nearest neighbors
            distances, indices = nbrs.kneighbors([embeddings[i]])
            neighbor_indices = indices[0][1:]  # Exclude self
            
            # Estimate gradient as weighted average of directions to neighbors
            gradient = np.zeros(self.embedding_dim)
            weights = 1.0 / (distances[0][1:] + 1e-8)
            
            for j, neighbor_idx in enumerate(neighbor_indices):
                direction = embeddings[neighbor_idx] - embeddings[i]
                gradient += weights[j] * direction
            
            # Normalize by total weight
            gradient_field[i] = gradient / np.sum(weights)
        
        return gradient_field
    
    def _find_attractors(self, embeddings: np.ndarray, 
                        gradient_field: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Find attractor points where gradients converge.
        
        Theory: Emotional attractors are regions where nearby points
        tend to flow toward a common center.
        """
        # Detect convergence points by finding local minima of gradient magnitude
        gradient_magnitudes = np.linalg.norm(gradient_field, axis=1)
        
        # Find points with low gradient magnitude (potential attractors)
        threshold = np.percentile(gradient_magnitudes, 10)
        potential_attractors = embeddings[gradient_magnitudes < threshold]
        
        # Cluster potential attractors to find true centers
        if len(potential_attractors) > 5:
            from sklearn.cluster import MeanShift
            ms = MeanShift(bandwidth=None)  # Automatic bandwidth
            ms.fit(potential_attractors)
            
            attractor_positions = ms.cluster_centers_
            
            # Estimate attractor strength by counting basin size
            attractor_strengths = []
            for center in attractor_positions:
                # Count how many points flow toward this attractor
                distances = np.linalg.norm(embeddings - center, axis=1)
                basin_size = np.sum(distances < np.median(distances))
                attractor_strengths.append(basin_size)
            
            attractor_strengths = np.array(attractor_strengths)
        else:
            # Too few points, use centroids
            attractor_positions = np.mean(embeddings, axis=0, keepdims=True)
            attractor_strengths = np.array([len(embeddings)])
        
        return {
            'positions': attractor_positions,
            'strengths': attractor_strengths,
            'n_attractors': len(attractor_positions)
        }
    
    def _estimate_curvature(self, embeddings: np.ndarray, 
                           gradient_field: np.ndarray) -> np.ndarray:
        """
        Estimate field curvature using second derivatives.
        
        Theory: Emotional fields warp the metric space, creating
        curvature that we can measure.
        """
        n_samples = len(embeddings)
        curvature = np.zeros(n_samples)
        
        # For each point, estimate how gradient changes in neighborhood
        # Ensure we don't request more neighbors than available samples
        n_neighbors_adjusted = min(self.n_neighbors + 1, n_samples)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors_adjusted)
        nbrs.fit(embeddings)
        
        for i in range(n_samples):
            distances, indices = nbrs.kneighbors([embeddings[i]])
            neighbor_indices = indices[0][1:]
            
            # Get gradients of neighbors
            neighbor_gradients = gradient_field[neighbor_indices]
            center_gradient = gradient_field[i]
            
            # Estimate divergence (how gradients spread out)
            gradient_changes = neighbor_gradients - center_gradient
            divergence = np.mean(np.sum(gradient_changes * gradient_changes, axis=1))
            
            curvature[i] = divergence
        
        # Smooth curvature estimates
        from scipy.ndimage import gaussian_filter1d
        curvature = gaussian_filter1d(curvature, sigma=1.0)
        
        return curvature
    
    def _identify_attractor_basins(self, embeddings: np.ndarray, 
                                  attractors: Dict) -> Dict[str, List[np.ndarray]]:
        """
        Identify basin boundaries between attractors using Voronoi tessellation.
        
        Theory: Each attractor has a basin of attraction - the region
        from which points flow toward it.
        """
        if len(attractors['positions']) < 2:
            return {'boundaries': [], 'basin_assignments': np.zeros(len(embeddings))}
        
        # Assign each embedding to nearest attractor
        from sklearn.metrics.pairwise import euclidean_distances
        distances_to_attractors = euclidean_distances(embeddings, attractors['positions'])
        basin_assignments = np.argmin(distances_to_attractors, axis=1)
        
        # Find boundaries between basins
        boundaries = []
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors)
        nbrs.fit(embeddings)
        
        for i in range(len(embeddings)):
            # Check if this point is near a boundary
            distances, indices = nbrs.kneighbors([embeddings[i]])
            neighbor_basins = basin_assignments[indices[0]]
            
            # If neighbors belong to different basins, this is a boundary point
            if len(np.unique(neighbor_basins)) > 1:
                boundaries.append(embeddings[i])
        
        return {
            'boundaries': boundaries,
            'basin_assignments': basin_assignments,
            'n_basins': len(attractors['positions'])
        }
    
    def _calculate_field_coherence(self, gradient_field: np.ndarray) -> float:
        """
        Calculate overall coherence of the gradient field.
        
        Theory: Coherent fields indicate strong emotional modulation.
        """
        # Normalize gradients
        norms = np.linalg.norm(gradient_field, axis=1, keepdims=True)
        normalized_gradients = gradient_field / (norms + 1e-8)
        
        # Calculate pairwise alignment
        n_samples = len(gradient_field)
        total_alignment = 0
        count = 0
        
        # Sample random pairs to estimate coherence
        n_pairs = min(1000, n_samples * (n_samples - 1) // 2)
        for _ in range(n_pairs):
            i, j = np.random.choice(n_samples, size=2, replace=False)
            alignment = np.dot(normalized_gradients[i], normalized_gradients[j])
            total_alignment += alignment
            count += 1
        
        # Average alignment (ranges from -1 to 1)
        avg_alignment = total_alignment / count if count > 0 else 0
        
        # Convert to coherence score (0 to 1)
        coherence = (avg_alignment + 1) / 2
        
        return coherence
    
    def visualize_flow_field(self, embeddings: np.ndarray, 
                            flow: EmotionalFieldFlow, 
                            projection_dims: Tuple[int, int] = (0, 1)) -> Dict:
        """
        Create visualization data for flow field (2D projection).
        
        Returns data that can be used to visualize the flow patterns.
        """
        # Project to 2D for visualization
        dim1, dim2 = projection_dims
        positions_2d = embeddings[:, [dim1, dim2]]
        gradients_2d = flow.gradient_field[:, [dim1, dim2]]
        
        # Attractor positions in 2D
        attractors_2d = flow.attractor_positions[:, [dim1, dim2]]
        
        return {
            'positions': positions_2d,
            'velocities': gradients_2d,
            'attractors': attractors_2d,
            'attractor_strengths': flow.attractor_strengths,
            'flow_magnitude': flow.flow_magnitude,
            'curvature': flow.curvature_map
        }


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
    
    analyzer = FieldGradientAnalyzer()
    
    # Discover field flows
    flows = analyzer.discover_field_flows(embeddings)
    flow = flows[0]
    
    print(f"Discovered {len(flow.attractor_positions)} attractors")
    print(f"Field coherence: {flow.field_coherence:.3f}")
    print(f"Strongest attractor has strength: {np.max(flow.attractor_strengths)}")
    
    # Show attractor distribution
    for i, strength in enumerate(flow.attractor_strengths):
        print(f"Attractor {i}: strength={strength}")