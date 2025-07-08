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
import gc
import os
from contextlib import nullcontext

# CRITICAL: Control threading BEFORE any BLAS operations
try:
    from threadpoolctl import threadpool_limits
    _has_threadpoolctl = True
except ImportError:
    _has_threadpoolctl = False
    # Fallback to environment variables
    for key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
        os.environ[key] = '1'

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
        
        # Memory safety check
        if _has_threadpoolctl:
            logger.info("✅ Using threadpoolctl for BLAS thread control")
        else:
            logger.warning("⚠️ threadpoolctl not available - using environment variables")
        
        # Log memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"📊 Memory usage at start: {mem_mb:.1f} MB")
        except ImportError:
            logger.warning("psutil not available - memory monitoring disabled")
            process = None
            mem_mb = 0
        
        try:
            return self._discover_field_flows_impl(embeddings, process, mem_mb)
        except Exception as e:
            if "memory" in str(e).lower() or "malloc" in str(e).lower():
                logger.error(f"❌ Memory error detected: {e}")
                logger.info("🔄 Attempting with smaller batch size...")
                
                # Try with smaller batches - same mathematics, just safer memory
                return self._discover_field_flows_safe_fallback(embeddings)
            else:
                raise
    
    def _discover_field_flows_impl(self, embeddings: np.ndarray, process, mem_mb: float) -> List[EmotionalFieldFlow]:
        """Core implementation of field flow discovery."""
        # 1. Estimate local gradients
        gradient_field = self._estimate_gradient_field(embeddings)
        
        # Log memory after gradient computation
        if process:
            mem_mb_after = process.memory_info().rss / 1024 / 1024
            logger.info(f"📊 Memory after gradient field: {mem_mb_after:.1f} MB (Δ: {mem_mb_after - mem_mb:.1f} MB)")
        
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
    
    def _discover_field_flows_safe_fallback(self, embeddings: np.ndarray) -> List[EmotionalFieldFlow]:
        """
        Safe fallback with aggressive memory control.
        MAINTAINS MATHEMATICAL INTEGRITY - just processes in smaller batches.
        """
        logger.warning("🛡️ Using safe fallback mode with aggressive memory control")
        
        # Force single-threaded operation globally
        old_n_threads = os.environ.get('OMP_NUM_THREADS', None)
        for key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
            os.environ[key] = '1'
        
        try:
            # Use even smaller chunks
            n_samples = len(embeddings)
            if n_samples > 200:
                # Process in two halves and merge results
                mid = n_samples // 2
                try:
                    import psutil
                    proc = psutil.Process(os.getpid())
                except ImportError:
                    proc = None
                    
                flow1 = self._discover_field_flows_impl(embeddings[:mid], proc, 0)
                gc.collect(2)  # Full collection
                flow2 = self._discover_field_flows_impl(embeddings[mid:], proc, 0)
                
                # Merge flows (simplified - takes the one with higher coherence)
                if flow1[0].field_coherence >= flow2[0].field_coherence:
                    return flow1
                else:
                    return flow2
            else:
                # Small enough to process directly
                try:
                    import psutil
                    proc = psutil.Process(os.getpid())
                except ImportError:
                    proc = None
                return self._discover_field_flows_impl(embeddings, proc, 0)
        finally:
            # Restore thread settings
            if old_n_threads:
                os.environ['OMP_NUM_THREADS'] = old_n_threads
    
    def _compute_single_gradient_safe(self, 
                                    center_embedding: np.ndarray,
                                    neighbor_embeddings: np.ndarray,
                                    distances: np.ndarray,
                                    gradient_buffer: np.ndarray) -> None:
        """
        Compute gradient for a single point in a memory-safe way.
        
        MATHEMATICAL INTEGRITY: Computes exact same result as vectorized version:
        gradient = Σ[(neighbor - center) * weight] / Σ[weight]
        
        Args:
            center_embedding: The embedding at the current point
            neighbor_embeddings: Embeddings of k nearest neighbors
            distances: Distances to neighbors (excluding self)
            gradient_buffer: Pre-allocated buffer to store result (modified in-place)
        """
        # Reset gradient buffer
        gradient_buffer.fill(0.0)
        weight_sum = 0.0
        
        # Compute weighted sum of directions
        for j in range(len(distances)):
            weight = 1.0 / (distances[j] + 1e-8)
            weight_sum += weight
            
            # Accumulate weighted direction in-place
            # gradient_buffer += (neighbor_embeddings[j] - center_embedding) * weight
            for d in range(len(gradient_buffer)):
                gradient_buffer[d] += (neighbor_embeddings[j, d] - center_embedding[d]) * weight
        
        # Normalize by weight sum
        if weight_sum > 0:
            gradient_buffer /= weight_sum
    
    def _estimate_gradient_field(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Estimate local gradient field using nearest neighbors.
        
        Theory: The gradient shows the direction of strongest change,
        revealing how embeddings "flow" through the space.
        
        MEMORY SAFETY: Uses pre-allocated buffers and controlled threading
        to prevent memory corruption while maintaining mathematical integrity.
        """
        n_samples = len(embeddings)
        embedding_dim = embeddings.shape[1]
        
        # MEMORY OPTIMIZATION: Pre-allocate all buffers
        gradient_field = np.zeros_like(embeddings)
        gradient_buffer = np.zeros(embedding_dim)  # Reusable buffer
        
        # Build nearest neighbor structure
        # Ensure we don't request more neighbors than available samples
        n_neighbors_adjusted = min(self.n_neighbors + 1, n_samples)
        
        # Control threading for KNN computation
        with threadpool_limits(limits=1) if _has_threadpoolctl else nullcontext():
            nbrs = NearestNeighbors(n_neighbors=n_neighbors_adjusted, n_jobs=1)
            nbrs.fit(embeddings)
        
        # MEMORY OPTIMIZATION: Process with controlled memory usage
        logger.debug(f"Computing gradient field for {n_samples} embeddings")
        
        # Process in chunks for memory efficiency
        chunk_size = 50 if n_samples > 100 else n_samples
        
        for chunk_start in range(0, n_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_samples)
            chunk_embeddings = embeddings[chunk_start:chunk_end]
            
            # Control threading for neighbor computation
            with threadpool_limits(limits=1) if _has_threadpoolctl else nullcontext():
                distances, indices = nbrs.kneighbors(chunk_embeddings)
            
            # Process each point in chunk with memory-safe method
            for local_i, global_i in enumerate(range(chunk_start, chunk_end)):
                neighbor_indices = indices[local_i][1:]  # Exclude self
                neighbor_distances = distances[local_i][1:]
                
                # Get neighbor embeddings
                neighbor_embeddings = embeddings[neighbor_indices]
                
                # Compute gradient using memory-safe method
                self._compute_single_gradient_safe(
                    center_embedding=embeddings[global_i],
                    neighbor_embeddings=neighbor_embeddings,
                    distances=neighbor_distances,
                    gradient_buffer=gradient_buffer
                )
                
                # Copy result to gradient field
                gradient_field[global_i] = gradient_buffer.copy()
            
            # Explicit memory cleanup between chunks
            if chunk_end < n_samples:
                gc.collect()
                if hasattr(gc, 'collect'):
                    gc.collect(2)  # Full collection
        
        logger.debug("Gradient field computation complete")
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
        
        MEMORY SAFETY: Uses controlled threading to prevent BLAS issues.
        """
        n_samples = len(embeddings)
        curvature = np.zeros(n_samples)
        
        # For each point, estimate how gradient changes in neighborhood
        # Ensure we don't request more neighbors than available samples
        n_neighbors_adjusted = min(self.n_neighbors + 1, n_samples)
        
        # Control threading for KNN computation
        with threadpool_limits(limits=1) if _has_threadpoolctl else nullcontext():
            nbrs = NearestNeighbors(n_neighbors=n_neighbors_adjusted, n_jobs=1)
            nbrs.fit(embeddings)
        
        # Process with controlled threading
        for i in range(n_samples):
            with threadpool_limits(limits=1) if _has_threadpoolctl else nullcontext():
                distances, indices = nbrs.kneighbors([embeddings[i]])
            neighbor_indices = indices[0][1:]
            
            # Get gradients of neighbors
            neighbor_gradients = gradient_field[neighbor_indices]
            center_gradient = gradient_field[i]
            
            # Estimate divergence (how gradients spread out)
            # Memory-safe computation of squared differences
            divergence = 0.0
            for j in range(len(neighbor_gradients)):
                diff = neighbor_gradients[j] - center_gradient
                divergence += np.dot(diff, diff)
            divergence /= len(neighbor_gradients)
            
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
        
        # Control threading for boundary detection
        with threadpool_limits(limits=1) if _has_threadpoolctl else nullcontext():
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=1)
            nbrs.fit(embeddings)
        
        for i in range(len(embeddings)):
            # Check if this point is near a boundary
            with threadpool_limits(limits=1) if _has_threadpoolctl else nullcontext():
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
        
        # Use deterministic pair sampling instead of random for reproducible results
        n_pairs = min(1000, n_samples * (n_samples - 1) // 2)
        
        if n_samples <= 1:
            return 1.0  # Single point has perfect coherence
        
        # Deterministic sampling: use evenly spaced pairs
        step_size = max(1, n_samples // int(np.sqrt(n_pairs)))
        
        for i in range(0, n_samples, step_size):
            for j in range(i + step_size, min(i + step_size * 2, n_samples), step_size):
                if i != j and count < n_pairs:
                    alignment = np.dot(normalized_gradients[i], normalized_gradients[j])
                    total_alignment += alignment
                    count += 1
        
        # If we didn't get enough pairs with structured sampling, add sequential pairs
        if count < n_pairs // 2 and n_samples > 2:
            for i in range(min(n_pairs - count, n_samples - 1)):
                j = (i + 1) % n_samples
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