"""
Lateral Interaction Engine - DTF Interaction Kernel Management

Manages the extraction, creation, and optimization of lateral interaction kernels
w(x-x') for Dynamic Field Theory. This module bridges embedding model neighborhoods
with DTF mathematical requirements.

MATHEMATICAL FOUNDATION:
Lateral interaction term: ∫w(x-x')f(u(x',t))dx'

Where w(x-x') defines how field positions interact:
- Excitation: w > 0 for semantically similar regions
- Inhibition: w < 0 for semantically distant regions
- Mexican hat: Common DTF pattern with center excitation, surround inhibition
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
import sys
from pathlib import Path

# Ensure project imports work
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


class LateralInteractionEngine:
    """
    Engine for extracting and managing DTF lateral interaction kernels.
    
    This class handles the conversion from embedding model neighborhoods
    (BGE, MPNet) to DTF-compatible interaction functions w(x-x').
    """
    
    def __init__(self, 
                 interaction_type: str = "mexican_hat",
                 adaptive_parameters: bool = True):
        """
        Initialize lateral interaction engine.
        
        Args:
            interaction_type: Type of interaction pattern ("mexican_hat", "gaussian", "adaptive")
            adaptive_parameters: Whether to adapt parameters to embedding neighborhoods
        """
        self.interaction_type = interaction_type
        self.adaptive_parameters = adaptive_parameters
        self.kernel_cache = {}
        
        logger.debug(f"LateralInteractionEngine initialized - type: {interaction_type}, "
                    f"adaptive: {adaptive_parameters}")
    
    def extract_interaction_kernel_from_neighborhoods(self,
                                                    central_embedding: np.ndarray,
                                                    neighbor_embeddings: List[np.ndarray],
                                                    neighbor_similarities: List[float],
                                                    neighbor_tokens: List[str]) -> Dict[str, Any]:
        """
        Extract DTF interaction kernel from embedding model neighborhoods.
        
        This is the core method that bridges semantic similarity patterns
        from embedding models to DTF lateral interaction functions.
        
        Args:
            central_embedding: Central embedding around which to build kernel
            neighbor_embeddings: Neighboring embeddings from similarity search
            neighbor_similarities: Similarity scores to neighbors
            neighbor_tokens: Token names for semantic analysis
            
        Returns:
            Dict containing interaction kernel function and metadata
        """
        try:
            # Analyze neighborhood structure
            neighborhood_analysis = self._analyze_neighborhood_structure(
                neighbor_similarities, neighbor_tokens
            )
            
            # Determine optimal kernel parameters
            kernel_params = self._optimize_kernel_parameters(
                central_embedding, neighbor_embeddings, neighbor_similarities, neighborhood_analysis
            )
            
            # Create DTF interaction function
            interaction_function = self._build_interaction_function(kernel_params)
            
            # Validate kernel properties
            validation_results = self._validate_interaction_kernel(
                interaction_function, kernel_params
            )
            
            return {
                'interaction_function': interaction_function,
                'kernel_parameters': kernel_params,
                'neighborhood_analysis': neighborhood_analysis,
                'validation': validation_results,
                'kernel_type': 'extracted_from_embeddings',
                'source_model_neighborhoods': True
            }
            
        except Exception as e:
            logger.error(f"Failed to extract interaction kernel: {e}")
            # Return default kernel as fallback
            return self._create_default_kernel()
    
    def _analyze_neighborhood_structure(self,
                                      similarities: List[float],
                                      tokens: List[str]) -> Dict[str, Any]:
        """
        Analyze the structure of embedding neighborhoods for DTF parameters.
        
        Identifies patterns in similarity distributions that inform DTF kernel design.
        """
        similarities = np.array(similarities)
        
        # Basic statistical analysis
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        
        # Identify similarity clusters
        high_sim_threshold = mean_sim + 0.5 * std_sim
        low_sim_threshold = mean_sim - 0.5 * std_sim
        
        high_sim_count = np.sum(similarities >= high_sim_threshold)
        medium_sim_count = np.sum((similarities >= low_sim_threshold) & (similarities < high_sim_threshold))
        low_sim_count = np.sum(similarities < low_sim_threshold)
        
        # Semantic diversity analysis
        unique_prefixes = len(set(token[:3] for token in tokens if len(token) >= 3))
        semantic_diversity = unique_prefixes / len(tokens) if tokens else 0.0
        
        # Determine excitation/inhibition balance
        if high_sim_count > len(similarities) * 0.6:
            # Many high-similarity neighbors → broad excitation
            excitation_pattern = "broad"
        elif high_sim_count > len(similarities) * 0.3:
            # Moderate high-similarity → focused excitation
            excitation_pattern = "focused"
        else:
            # Few high-similarity → narrow excitation
            excitation_pattern = "narrow"
        
        return {
            'similarity_stats': {
                'mean': float(mean_sim),
                'std': float(std_sim),
                'min': float(min_sim),
                'max': float(max_sim),
                'range': float(max_sim - min_sim)
            },
            'similarity_clusters': {
                'high_count': int(high_sim_count),
                'medium_count': int(medium_sim_count),
                'low_count': int(low_sim_count),
                'high_ratio': float(high_sim_count / len(similarities))
            },
            'semantic_diversity': float(semantic_diversity),
            'excitation_pattern': excitation_pattern,
            'neighbor_count': len(similarities)
        }
    
    def _optimize_kernel_parameters(self,
                                   central_embedding: np.ndarray,
                                   neighbor_embeddings: List[np.ndarray],
                                   similarities: List[float],
                                   neighborhood_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize DTF kernel parameters based on neighborhood analysis.
        
        Uses the embedding neighborhood structure to determine optimal
        excitation radius, inhibition strength, etc.
        """
        similarity_stats = neighborhood_analysis['similarity_stats']
        clusters = neighborhood_analysis['similarity_clusters']
        excitation_pattern = neighborhood_analysis['excitation_pattern']
        
        if self.interaction_type == "mexican_hat":
            # Optimize Mexican hat parameters
            
            # Excitation radius based on high-similarity cluster
            if excitation_pattern == "broad":
                excitation_radius = 0.8  # Broad excitation
            elif excitation_pattern == "focused":
                excitation_radius = 0.5  # Moderate excitation
            else:  # narrow
                excitation_radius = 0.3  # Narrow excitation
            
            # Inhibition strength based on semantic diversity
            diversity = neighborhood_analysis['semantic_diversity']
            if diversity > 0.7:
                # High diversity → strong inhibition to maintain selectivity
                inhibition_strength = 0.6
            elif diversity > 0.4:
                # Medium diversity → moderate inhibition
                inhibition_strength = 0.4
            else:
                # Low diversity → weak inhibition
                inhibition_strength = 0.2
            
            # Excitation strength based on similarity range
            sim_range = similarity_stats['range']
            excitation_strength = 1.0 * (1.0 + sim_range)  # Stronger for wider range
            
            return {
                'type': 'mexican_hat',
                'excitation_radius': float(excitation_radius),
                'inhibition_strength': float(inhibition_strength),
                'excitation_strength': float(excitation_strength),
                'adaptive_source': 'embedding_neighborhoods'
            }
            
        elif self.interaction_type == "gaussian":
            # Optimize Gaussian parameters
            
            # Width based on similarity distribution
            width = similarity_stats['std'] * 2.0  # Scale with similarity spread
            strength = 1.0 + similarity_stats['mean']  # Scale with average similarity
            
            return {
                'type': 'gaussian',
                'width': float(width),
                'strength': float(strength),
                'adaptive_source': 'embedding_neighborhoods'
            }
            
        else:  # adaptive
            # Create fully adaptive kernel based on actual similarity distribution
            return {
                'type': 'adaptive',
                'similarity_profile': similarities,
                'embedding_distances': self._compute_embedding_distances(central_embedding, neighbor_embeddings),
                'adaptive_source': 'full_neighborhood_profile'
            }
    
    def _compute_embedding_distances(self,
                                   central_embedding: np.ndarray,
                                   neighbor_embeddings: List[np.ndarray]) -> List[float]:
        """Compute distances from central embedding to neighbors."""
        distances = []
        for neighbor in neighbor_embeddings:
            # Use cosine distance (1 - cosine_similarity)
            cosine_sim = np.dot(central_embedding, neighbor) / (
                np.linalg.norm(central_embedding) * np.linalg.norm(neighbor)
            )
            cosine_distance = 1.0 - cosine_sim
            distances.append(float(cosine_distance))
        
        return distances
    
    def _build_interaction_function(self, kernel_params: Dict[str, Any]) -> Callable[[np.ndarray], float]:
        """
        Build DTF interaction function from optimized parameters.
        
        Creates the actual w(x-x') function that will be used in DTF computations.
        """
        kernel_type = kernel_params['type']
        
        if kernel_type == "mexican_hat":
            excitation_radius = kernel_params['excitation_radius']
            inhibition_strength = kernel_params['inhibition_strength']
            excitation_strength = kernel_params['excitation_strength']
            
            def mexican_hat_interaction(distance_vector: np.ndarray) -> float:
                distance = np.linalg.norm(distance_vector)
                
                if distance <= excitation_radius:
                    # Excitatory center
                    return excitation_strength * np.exp(-distance**2 / (2 * excitation_radius**2))
                else:
                    # Inhibitory surround
                    surround_width = excitation_radius * 1.5
                    return -inhibition_strength * np.exp(-(distance - excitation_radius)**2 / (2 * surround_width**2))
            
            return mexican_hat_interaction
            
        elif kernel_type == "gaussian":
            width = kernel_params['width']
            strength = kernel_params['strength']
            
            def gaussian_interaction(distance_vector: np.ndarray) -> float:
                distance_squared = np.sum(distance_vector**2)
                return strength * np.exp(-distance_squared / (2 * width**2))
            
            return gaussian_interaction
            
        elif kernel_type == "adaptive":
            # Create adaptive kernel based on actual neighborhood profile
            similarities = kernel_params['similarity_profile']
            distances = kernel_params['embedding_distances']
            
            # Create interpolation-based interaction function
            def adaptive_interaction(distance_vector: np.ndarray) -> float:
                query_distance = np.linalg.norm(distance_vector)
                
                # Find closest known distance
                if not distances:
                    return 0.0
                
                closest_idx = np.argmin(np.abs(np.array(distances) - query_distance))
                closest_similarity = similarities[closest_idx]
                
                # Convert similarity to interaction strength
                # High similarity → positive interaction (excitation)
                # Low similarity → negative interaction (inhibition)
                interaction_strength = 2.0 * (closest_similarity - 0.5)  # Map [0,1] to [-1,1]
                
                return float(interaction_strength)
            
            return adaptive_interaction
        
        else:
            # Default to simple Gaussian
            def default_interaction(distance_vector: np.ndarray) -> float:
                distance_squared = np.sum(distance_vector**2)
                return np.exp(-distance_squared / 2.0)
            
            return default_interaction
    
    def _validate_interaction_kernel(self,
                                   interaction_function: Callable[[np.ndarray], float],
                                   kernel_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate DTF interaction kernel properties.
        
        Ensures the kernel has reasonable properties for DTF dynamics.
        """
        # Test kernel at various distances
        test_distances = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
        test_vectors = [[d, 0.0] for d in test_distances]  # 2D test vectors
        
        kernel_values = []
        for vec in test_vectors:
            try:
                value = interaction_function(np.array(vec))
                kernel_values.append(float(value))
            except Exception as e:
                logger.warning(f"Kernel evaluation failed at distance {vec}: {e}")
                kernel_values.append(0.0)
        
        kernel_values = np.array(kernel_values)
        
        # Check kernel properties
        has_positive_values = np.any(kernel_values > 0)
        has_negative_values = np.any(kernel_values < 0)
        max_value = np.max(kernel_values)
        min_value = np.min(kernel_values)
        
        # Check for reasonable decay
        if len(kernel_values) >= 2:
            is_decaying = kernel_values[0] >= kernel_values[-1]  # Should decay with distance
        else:
            is_decaying = True
        
        return {
            'has_excitation': bool(has_positive_values),
            'has_inhibition': bool(has_negative_values),
            'max_value': float(max_value),
            'min_value': float(min_value),
            'value_range': float(max_value - min_value),
            'is_decaying': bool(is_decaying),
            'test_distances': test_distances.tolist(),
            'test_values': kernel_values.tolist(),
            'is_valid': bool(has_positive_values and is_decaying)
        }
    
    def _create_default_kernel(self) -> Dict[str, Any]:
        """Create a default DTF interaction kernel as fallback."""
        def default_mexican_hat(distance_vector: np.ndarray) -> float:
            distance = np.linalg.norm(distance_vector)
            if distance <= 0.5:
                return np.exp(-distance**2 / (2 * 0.25**2))
            else:
                return -0.3 * np.exp(-(distance - 0.5)**2 / (2 * 0.5**2))
        
        return {
            'interaction_function': default_mexican_hat,
            'kernel_parameters': {
                'type': 'default_mexican_hat',
                'excitation_radius': 0.5,
                'inhibition_strength': 0.3,
                'excitation_strength': 1.0
            },
            'neighborhood_analysis': {},
            'validation': {'is_valid': True, 'has_excitation': True, 'has_inhibition': True},
            'kernel_type': 'default_fallback',
            'source_model_neighborhoods': False
        }
    
    def create_kernel_from_parameters(self, 
                                    kernel_type: str,
                                    **parameters) -> Callable[[np.ndarray], float]:
        """
        Create interaction kernel from explicit parameters.
        
        Args:
            kernel_type: Type of kernel ("mexican_hat", "gaussian", "exponential")
            **parameters: Kernel-specific parameters
            
        Returns:
            Interaction function w(x-x')
        """
        if kernel_type == "mexican_hat":
            excitation_radius = parameters.get('excitation_radius', 0.5)
            inhibition_strength = parameters.get('inhibition_strength', 0.3)
            excitation_strength = parameters.get('excitation_strength', 1.0)
            
            def mexican_hat(distance_vector: np.ndarray) -> float:
                distance = np.linalg.norm(distance_vector)
                if distance <= excitation_radius:
                    return excitation_strength * np.exp(-distance**2 / (2 * excitation_radius**2))
                else:
                    surround_width = excitation_radius * 1.5
                    return -inhibition_strength * np.exp(-(distance - excitation_radius)**2 / (2 * surround_width**2))
            
            return mexican_hat
            
        elif kernel_type == "gaussian":
            width = parameters.get('width', 1.0)
            strength = parameters.get('strength', 1.0)
            
            def gaussian(distance_vector: np.ndarray) -> float:
                distance_squared = np.sum(distance_vector**2)
                return strength * np.exp(-distance_squared / (2 * width**2))
            
            return gaussian
            
        elif kernel_type == "exponential":
            decay_rate = parameters.get('decay_rate', 1.0)
            strength = parameters.get('strength', 1.0)
            
            def exponential(distance_vector: np.ndarray) -> float:
                distance = np.linalg.norm(distance_vector)
                return strength * np.exp(-decay_rate * distance)
            
            return exponential
        
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    def analyze_kernel_stability(self,
                                interaction_function: Callable[[np.ndarray], float],
                                spatial_extent: float = 5.0,
                                n_samples: int = 100) -> Dict[str, Any]:
        """
        Analyze stability properties of interaction kernel.
        
        Tests kernel properties that affect DTF field stability.
        """
        # Sample kernel at various distances
        distances = np.linspace(0, spatial_extent, n_samples)
        kernel_values = []
        
        for d in distances:
            try:
                value = interaction_function(np.array([d, 0.0]))
                kernel_values.append(value)
            except:
                kernel_values.append(0.0)
        
        kernel_values = np.array(kernel_values)
        
        # Analyze kernel properties
        total_excitation = np.sum(kernel_values[kernel_values > 0])
        total_inhibition = np.sum(kernel_values[kernel_values < 0])
        net_interaction = total_excitation + total_inhibition
        
        # Find zero crossings (excitation→inhibition transitions)
        zero_crossings = []
        for i in range(len(kernel_values) - 1):
            if kernel_values[i] * kernel_values[i+1] < 0:  # Sign change
                zero_crossings.append(distances[i])
        
        return {
            'total_excitation': float(total_excitation),
            'total_inhibition': float(total_inhibition),
            'net_interaction': float(net_interaction),
            'excitation_inhibition_ratio': float(total_excitation / abs(total_inhibition)) if total_inhibition != 0 else float('inf'),
            'zero_crossings': zero_crossings,
            'max_excitation': float(np.max(kernel_values)),
            'max_inhibition': float(np.min(kernel_values)),
            'kernel_extent': float(spatial_extent),
            'is_balanced': abs(net_interaction) < 0.1 * total_excitation
        }


# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing Lateral Interaction Engine...")
    
    # Initialize engine
    engine = LateralInteractionEngine(interaction_type="mexican_hat", adaptive_parameters=True)
    
    # Create test neighborhood data
    central_embedding = np.random.randn(768) * 0.1  # MPNet-style
    neighbor_embeddings = [np.random.randn(768) * 0.1 + central_embedding * 0.5 for _ in range(20)]
    neighbor_similarities = np.random.beta(2, 2, 20).tolist()  # Realistic similarity distribution
    neighbor_tokens = [f"token_{i}" for i in range(20)]
    
    # Extract interaction kernel
    kernel_data = engine.extract_interaction_kernel_from_neighborhoods(
        central_embedding=central_embedding,
        neighbor_embeddings=neighbor_embeddings,
        neighbor_similarities=neighbor_similarities,
        neighbor_tokens=neighbor_tokens
    )
    
    logger.info(f"Extracted kernel type: {kernel_data['kernel_parameters']['type']}")
    logger.info(f"Kernel validation: {kernel_data['validation']['is_valid']}")
    
    # Test kernel function
    interaction_func = kernel_data['interaction_function']
    test_distances = [0.0, 0.5, 1.0, 2.0]
    
    logger.info("Kernel response at test distances:")
    for d in test_distances:
        response = interaction_func(np.array([d, 0.0]))
        logger.info(f"  Distance {d:.1f}: w = {response:.4f}")
    
    # Analyze kernel stability
    stability = engine.analyze_kernel_stability(interaction_func)
    logger.info(f"Kernel stability - balanced: {stability['is_balanced']}, "
               f"E/I ratio: {stability['excitation_inhibition_ratio']:.2f}")
    
    logger.info("Lateral Interaction Engine test complete!")