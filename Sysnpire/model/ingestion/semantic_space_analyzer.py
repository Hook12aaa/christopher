"""
Semantic Space Analyzer - Extract comprehensive characteristics from embedding spaces

This module provides deep analysis of semantic embedding spaces to extract
all key characteristics needed for converting embedding space into conceptual
charge universe representations. Designed to capture the full semantic
landscape for field-theoretic social modeling.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from .semantic_embedding import SemanticEmbedding
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticSpaceAnalyzer:
    """
    Comprehensive Semantic Space Analysis for Universe Conversion
    
    This class extracts all key characteristics from semantic embedding spaces
    to enable conversion into conceptual charge universes. It analyzes the
    geometric, topological, and semantic properties of embedding spaces to
    provide the foundation for field-theoretic social modeling.
    
    The analyzer captures:
    - Geometric properties of the embedding space
    - Topological characteristics and clustering patterns
    - Semantic density distributions and concentration areas
    - Dimensional importance and feature significance
    - Relational structures and neighborhood properties
    - Temporal and dynamic patterns in embedding evolution
    
    This comprehensive analysis enables accurate conversion of static embedding
    spaces into dynamic conceptual charge universes with proper field effects.
    
    Attributes:
        embeddings (List[SemanticEmbedding]): Collection of embeddings to analyze
        dimension (int): Dimensionality of the embedding space
        space_metrics (Dict): Computed geometric and topological metrics
        semantic_clusters (Dict): Identified semantic clusters and regions
        field_characteristics (Dict): Properties relevant for field theory
    """
    
    def __init__(self, embeddings: List[SemanticEmbedding]):
        """
        Initialize the Semantic Space Analyzer with a collection of embeddings.
        
        Args:
            embeddings (List[SemanticEmbedding]): Collection of semantic embeddings
                                                 to analyze for space characteristics
                                                 
        Raises:
            ValueError: If embeddings list is empty or contains incompatible dimensions
        """
        if not embeddings:
            raise ValueError("Cannot analyze empty embedding collection")
        
        self.embeddings = embeddings
        self.dimension = embeddings[0].dimension
        
        for emb in embeddings[1:]:
            if emb.dimension != self.dimension:
                raise ValueError(f"All embeddings must have same dimension. "
                               f"Expected {self.dimension}, got {emb.dimension}")
        
        self.space_metrics: Dict[str, Any] = {}
        self.semantic_clusters: Dict[str, Any] = {}
        self.field_characteristics: Dict[str, Any] = {}
        
        logger.info(f"Semantic Space Analyzer initialized with {len(embeddings)} "
                   f"embeddings in {self.dimension}D space")
    
    def analyze_complete_space(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the semantic embedding space.
        
        This method extracts all key characteristics needed for converting
        the embedding space into a conceptual charge universe. It combines
        geometric, topological, semantic, and field-theoretic analysis.
        
        Returns:
            Dict[str, Any]: Complete space analysis containing:
                - geometric_properties: Space geometry and metric characteristics
                - topological_features: Clustering and connectivity patterns
                - semantic_distributions: Content and meaning distributions
                - dimensional_analysis: Per-dimension importance and characteristics
                - field_preparation: Properties needed for field theory conversion
                - universe_parameters: Recommended parameters for universe creation
        """
        logger.info("Starting comprehensive semantic space analysis...")
        
        analysis_result = {
            'geometric_properties': self._analyze_geometric_properties(),
            'topological_features': self._analyze_topological_features(),
            'semantic_distributions': self._analyze_semantic_distributions(),
            'dimensional_analysis': self._analyze_dimensional_characteristics(),
            'field_preparation': self._prepare_field_characteristics(),
            'universe_parameters': self._recommend_universe_parameters()
        }
        
        self.space_metrics = analysis_result['geometric_properties']
        self.semantic_clusters = analysis_result['topological_features']
        self.field_characteristics = analysis_result['field_preparation']
        
        logger.info("Comprehensive space analysis completed")
        return analysis_result
    
    def _analyze_geometric_properties(self) -> Dict[str, Any]:
        """
        Analyze geometric properties of the embedding space.
        
        Extracts fundamental geometric characteristics including distances,
        angles, curvature, and metric properties that define the space
        structure for field theory applications.
        
        Returns:
            Dict[str, Any]: Geometric property analysis
        """
        embedding_matrix = np.array([emb.vector for emb in self.embeddings])
        
        pairwise_distances = pdist(embedding_matrix, metric='euclidean')
        distance_matrix = squareform(pairwise_distances)
        
        cosine_similarities = []
        for i in range(len(self.embeddings)):
            for j in range(i + 1, len(self.embeddings)):
                sim = self.embeddings[i].similarity(self.embeddings[j])
                cosine_similarities.append(sim)
        
        cosine_similarities = np.array(cosine_similarities)
        
        magnitudes = np.array([emb.magnitude for emb in self.embeddings])
        
        space_center = np.mean(embedding_matrix, axis=0)
        space_variance = np.var(embedding_matrix, axis=0)
        space_std = np.std(embedding_matrix, axis=0)
        
        geometric_props = {
            'space_dimensionality': self.dimension,
            'num_points': len(self.embeddings),
            'distance_statistics': {
                'mean_distance': float(np.mean(pairwise_distances)),
                'std_distance': float(np.std(pairwise_distances)),
                'min_distance': float(np.min(pairwise_distances)),
                'max_distance': float(np.max(pairwise_distances)),
                'distance_range': float(np.ptp(pairwise_distances))
            },
            'similarity_statistics': {
                'mean_similarity': float(np.mean(cosine_similarities)),
                'std_similarity': float(np.std(cosine_similarities)),
                'min_similarity': float(np.min(cosine_similarities)),
                'max_similarity': float(np.max(cosine_similarities))
            },
            'magnitude_statistics': {
                'mean_magnitude': float(np.mean(magnitudes)),
                'std_magnitude': float(np.std(magnitudes)),
                'magnitude_range': float(np.ptp(magnitudes))
            },
            'space_characteristics': {
                'center_point': space_center.tolist(),
                'variance_per_dimension': space_variance.tolist(),
                'std_per_dimension': space_std.tolist(),
                'effective_dimensionality': self._calculate_effective_dimensionality(embedding_matrix),
                'space_density': self._calculate_space_density(distance_matrix),
                'isotropy_measure': self._calculate_isotropy(space_variance)
            }
        }
        
        return geometric_props
    
    def _analyze_topological_features(self) -> Dict[str, Any]:
        """
        Analyze topological features and clustering patterns in embedding space.
        
        Identifies semantic clusters, connectivity patterns, and topological
        structures that are essential for understanding the semantic landscape
        and converting it to field-theoretic representation.
        
        Returns:
            Dict[str, Any]: Topological feature analysis
        """
        embedding_matrix = np.array([emb.vector for emb in self.embeddings])
        distance_matrix = squareform(pdist(embedding_matrix, metric='euclidean'))
        
        cluster_analysis = self._identify_semantic_clusters(embedding_matrix, distance_matrix)
        connectivity_analysis = self._analyze_connectivity_patterns(distance_matrix)
        neighborhood_analysis = self._analyze_neighborhood_structure(distance_matrix)
        
        topological_features = {
            'clustering_results': cluster_analysis,
            'connectivity_patterns': connectivity_analysis,
            'neighborhood_structure': neighborhood_analysis,
            'topological_invariants': self._compute_topological_invariants(distance_matrix),
            'semantic_regions': self._identify_semantic_regions()
        }
        
        return topological_features
    
    def _analyze_semantic_distributions(self) -> Dict[str, Any]:
        """
        Analyze semantic content distributions across the embedding space.
        
        Examines how semantic content is distributed in the space, identifying
        areas of high semantic density, content patterns, and meaning gradients
        that are crucial for field effect modeling.
        
        Returns:
            Dict[str, Any]: Semantic distribution analysis
        """
        text_lengths = [len(emb.text) for emb in self.embeddings]
        
        semantic_categories = self._extract_semantic_categories()
        content_distributions = self._analyze_content_distributions()
        meaning_gradients = self._analyze_meaning_gradients()
        
        semantic_analysis = {
            'text_statistics': {
                'mean_length': float(np.mean(text_lengths)),
                'std_length': float(np.std(text_lengths)),
                'length_range': float(np.ptp(text_lengths))
            },
            'semantic_categories': semantic_categories,
            'content_distributions': content_distributions,
            'meaning_gradients': meaning_gradients,
            'semantic_density_map': self._create_semantic_density_map(),
            'content_diversity_score': self._calculate_content_diversity()
        }
        
        return semantic_analysis
    
    def _analyze_dimensional_characteristics(self) -> Dict[str, Any]:
        """
        Analyze characteristics of individual dimensions in embedding space.
        
        Examines each dimension for its contribution to semantic representation,
        importance for clustering, and relevance for field theory applications.
        This analysis is crucial for proper tau vector construction.
        
        Returns:
            Dict[str, Any]: Per-dimension characteristic analysis
        """
        embedding_matrix = np.array([emb.vector for emb in self.embeddings])
        
        dimension_variances = np.var(embedding_matrix, axis=0)
        dimension_means = np.mean(embedding_matrix, axis=0)
        dimension_ranges = np.ptp(embedding_matrix, axis=0)
        
        dimension_importance = self._calculate_dimension_importance(embedding_matrix)
        correlation_structure = self._analyze_dimensional_correlations(embedding_matrix)
        
        dimensional_analysis = {
            'dimension_statistics': {
                'variances': dimension_variances.tolist(),
                'means': dimension_means.tolist(),
                'ranges': dimension_ranges.tolist(),
                'importance_scores': dimension_importance.tolist()
            },
            'correlation_structure': correlation_structure,
            'active_dimensions': self._identify_active_dimensions(dimension_variances),
            'semantic_dimension_groups': self._group_semantic_dimensions(embedding_matrix),
            'field_relevant_dimensions': self._identify_field_relevant_dimensions(dimension_importance)
        }
        
        return dimensional_analysis
    
    def _prepare_field_characteristics(self) -> Dict[str, Any]:
        """
        Prepare characteristics specifically needed for field theory conversion.
        
        Extracts and computes properties that are essential for converting
        the embedding space into a conceptual charge universe with proper
        field effects, phase relationships, and trajectory dependencies.
        
        Returns:
            Dict[str, Any]: Field theory preparation characteristics
        """
        embedding_matrix = np.array([emb.vector for emb in self.embeddings])
        
        field_strength_map = self._calculate_field_strength_distribution(embedding_matrix)
        phase_relationships = self._analyze_phase_relationships(embedding_matrix)
        trajectory_potentials = self._calculate_trajectory_potentials(embedding_matrix)
        
        field_characteristics = {
            'field_strength_distribution': field_strength_map,
            'phase_relationship_matrix': phase_relationships,
            'trajectory_potential_map': trajectory_potentials,
            'charge_placement_grid': self._create_charge_placement_grid(),
            'field_interaction_zones': self._identify_field_interaction_zones(),
            'universe_boundary_conditions': self._determine_boundary_conditions(),
            'recommended_gamma_range': self._recommend_gamma_calibration(),
            'optimal_observational_states': self._recommend_observational_states()
        }
        
        return field_characteristics
    
    def _recommend_universe_parameters(self) -> Dict[str, Any]:
        """
        Recommend optimal parameters for conceptual charge universe creation.
        
        Based on the complete space analysis, provides recommendations for
        universe construction parameters that will best preserve semantic
        relationships while enabling effective field theory modeling.
        
        Returns:
            Dict[str, Any]: Recommended universe parameters
        """
        num_embeddings = len(self.embeddings)
        space_density = self.space_metrics.get('space_characteristics', {}).get('space_density', 1.0)
        
        universe_params = {
            'recommended_universe_size': self._calculate_optimal_universe_size(),
            'charge_density_target': min(0.1, 1000 / num_embeddings),
            'field_calibration_gamma': self._recommend_optimal_gamma(),
            'observational_state_range': self._recommend_state_range(),
            'spatial_resolution': self._recommend_spatial_resolution(),
            'temporal_evolution_rate': self._recommend_evolution_rate(),
            'field_interaction_strength': self._recommend_interaction_strength(),
            'universe_initialization': {
                'charge_placement_strategy': 'semantic_clustering',
                'initial_field_configuration': 'gradient_based',
                'boundary_handling': 'periodic' if num_embeddings > 500 else 'reflective'
            }
        }
        
        return universe_params
    
    def get_universe_conversion_map(self) -> Dict[str, Any]:
        """
        Generate a complete mapping for converting embedding space to universe.
        
        Provides all necessary information for converting the analyzed semantic
        embedding space into a conceptual charge universe with proper field
        effects and semantic preservation.
        
        Returns:
            Dict[str, Any]: Complete universe conversion mapping
        """
        if not self.space_metrics:
            self.analyze_complete_space()
        
        conversion_map = {
            'embedding_to_position_mapping': self._create_position_mapping(),
            'semantic_to_charge_mapping': self._create_charge_mapping(),
            'field_effect_parameters': self._extract_field_parameters(),
            'universe_topology': self._define_universe_topology(),
            'evolution_dynamics': self._define_evolution_dynamics(),
            'validation_metrics': self._define_validation_metrics()
        }
        
        return conversion_map
    
    def _calculate_effective_dimensionality(self, embedding_matrix: np.ndarray) -> float:
        """Calculate effective dimensionality using explained variance."""
        try:
            from sklearn.decomposition import PCA
            pca = PCA()
            pca.fit(embedding_matrix)
            explained_variance_ratio = pca.explained_variance_ratio_
            
            cumsum = np.cumsum(explained_variance_ratio)
            effective_dims = np.sum(cumsum < 0.95) + 1
            return float(effective_dims)
        except ImportError:
            logger.warning("sklearn not available, using variance-based approximation")
            variances = np.var(embedding_matrix, axis=0)
            total_variance = np.sum(variances)
            sorted_variances = np.sort(variances)[::-1]
            cumsum = np.cumsum(sorted_variances) / total_variance
            effective_dims = np.sum(cumsum < 0.95) + 1
            return float(effective_dims)
    
    def _calculate_space_density(self, distance_matrix: np.ndarray) -> float:
        """Calculate overall density of the embedding space."""
        n_points = distance_matrix.shape[0]
        if n_points < 2:
            return 1.0
        
        mean_distance = np.mean(distance_matrix[np.triu_indices(n_points, k=1)])
        space_volume = mean_distance ** self.dimension
        density = n_points / max(space_volume, 1e-10)
        return float(density)
    
    def _calculate_isotropy(self, variance_per_dimension: np.ndarray) -> float:
        """Calculate isotropy measure of the space."""
        if len(variance_per_dimension) == 0:
            return 1.0
        
        variance_ratio = np.max(variance_per_dimension) / (np.min(variance_per_dimension) + 1e-10)
        isotropy = 1.0 / variance_ratio
        return float(isotropy)
    
    def _identify_semantic_clusters(self, 
                                  embedding_matrix: np.ndarray, 
                                  distance_matrix: np.ndarray) -> Dict[str, Any]:
        """Identify semantic clusters in the embedding space."""
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=2)
            cluster_labels = clustering.fit_predict(embedding_matrix)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            return {
                'num_clusters': int(n_clusters),
                'num_noise_points': int(n_noise),
                'cluster_labels': cluster_labels.tolist(),
                'clustering_quality': self._evaluate_clustering_quality(distance_matrix, cluster_labels)
            }
        except ImportError:
            logger.warning("sklearn not available, using distance-based clustering")
            return self._simple_distance_clustering(distance_matrix)
    
    def _simple_distance_clustering(self, distance_matrix: np.ndarray) -> Dict[str, Any]:
        """Simple distance-based clustering when sklearn is not available."""
        threshold = np.mean(distance_matrix) * 0.8
        n_points = distance_matrix.shape[0]
        cluster_labels = [-1] * n_points
        current_cluster = 0
        
        for i in range(n_points):
            if cluster_labels[i] == -1:
                cluster_labels[i] = current_cluster
                for j in range(i + 1, n_points):
                    if distance_matrix[i, j] < threshold:
                        cluster_labels[j] = current_cluster
                current_cluster += 1
        
        n_clusters = len(set(cluster_labels))
        return {
            'num_clusters': n_clusters,
            'num_noise_points': 0,
            'cluster_labels': cluster_labels,
            'clustering_quality': 0.5
        }
    
    def _analyze_connectivity_patterns(self, distance_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze connectivity patterns in the embedding space."""
        threshold = np.mean(distance_matrix) * 0.7
        adjacency_matrix = distance_matrix < threshold
        
        connectivity_analysis = {
            'average_degree': float(np.mean(np.sum(adjacency_matrix, axis=1))),
            'max_degree': int(np.max(np.sum(adjacency_matrix, axis=1))),
            'connectivity_density': float(np.sum(adjacency_matrix) / (adjacency_matrix.shape[0] ** 2)),
            'isolated_points': int(np.sum(np.sum(adjacency_matrix, axis=1) == 1))
        }
        
        return connectivity_analysis
    
    def _analyze_neighborhood_structure(self, distance_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze neighborhood structure and local density patterns."""
        k = min(5, distance_matrix.shape[0] - 1)
        
        neighborhood_densities = []
        for i in range(distance_matrix.shape[0]):
            neighbors = np.argsort(distance_matrix[i, :])[:k+1]
            neighbor_distances = distance_matrix[i, neighbors[1:]]
            local_density = k / (np.mean(neighbor_distances) ** self.dimension + 1e-10)
            neighborhood_densities.append(local_density)
        
        neighborhood_analysis = {
            'mean_neighborhood_density': float(np.mean(neighborhood_densities)),
            'std_neighborhood_density': float(np.std(neighborhood_densities)),
            'density_variation': float(np.ptp(neighborhood_densities)),
            'high_density_regions': int(np.sum(np.array(neighborhood_densities) > np.mean(neighborhood_densities) + np.std(neighborhood_densities)))
        }
        
        return neighborhood_analysis
    
    def _compute_topological_invariants(self, distance_matrix: np.ndarray) -> Dict[str, float]:
        """Compute basic topological invariants of the space."""
        n_points = distance_matrix.shape[0]
        
        mean_distance = float(np.mean(distance_matrix[np.triu_indices(n_points, k=1)]))
        diameter = float(np.max(distance_matrix))
        radius = float(np.min(np.max(distance_matrix, axis=1)))
        
        return {
            'mean_distance': mean_distance,
            'diameter': diameter,
            'radius': radius,
            'compactness': mean_distance / diameter if diameter > 0 else 1.0
        }
    
    def _identify_semantic_regions(self) -> Dict[str, Any]:
        """Identify semantically coherent regions in the space."""
        semantic_regions = {}
        
        for i, embedding in enumerate(self.embeddings):
            metadata = embedding.metadata or {}
            
            if 'social_score' in metadata:
                social_score = metadata['social_score']
                if social_score > 0.1:
                    region_key = f"high_social_{i//10}"
                    if region_key not in semantic_regions:
                        semantic_regions[region_key] = []
                    semantic_regions[region_key].append(i)
        
        return semantic_regions
    
    def _extract_semantic_categories(self) -> Dict[str, int]:
        """Extract semantic category distributions."""
        categories = {}
        
        for embedding in self.embeddings:
            text_lower = embedding.text.lower()
            
            if any(word in text_lower for word in ['social', 'community', 'group']):
                categories['social'] = categories.get('social', 0) + 1
            if any(word in text_lower for word in ['emotion', 'feel', 'sentiment']):
                categories['emotional'] = categories.get('emotional', 0) + 1
            if any(word in text_lower for word in ['technical', 'system', 'algorithm']):
                categories['technical'] = categories.get('technical', 0) + 1
        
        return categories
    
    def _analyze_content_distributions(self) -> Dict[str, float]:
        """Analyze content distributions across embeddings."""
        text_lengths = [len(emb.text) for emb in self.embeddings]
        word_counts = [len(emb.text.split()) for emb in self.embeddings]
        
        return {
            'mean_text_length': float(np.mean(text_lengths)),
            'mean_word_count': float(np.mean(word_counts)),
            'content_diversity': float(len(set(emb.text for emb in self.embeddings)) / len(self.embeddings))
        }
    
    def _analyze_meaning_gradients(self) -> Dict[str, float]:
        """Analyze meaning gradients and semantic flow in the space."""
        similarities = []
        for i in range(len(self.embeddings)):
            for j in range(i + 1, len(self.embeddings)):
                sim = self.embeddings[i].similarity(self.embeddings[j])
                similarities.append(sim)
        
        return {
            'mean_semantic_similarity': float(np.mean(similarities)),
            'semantic_coherence': float(np.std(similarities)),
            'meaning_gradient_strength': float(np.ptp(similarities))
        }
    
    def _create_semantic_density_map(self) -> Dict[str, Any]:
        """Create a semantic density map of the embedding space."""
        embedding_matrix = np.array([emb.vector for emb in self.embeddings])
        
        density_map = {
            'high_density_regions': [],
            'sparse_regions': [],
            'density_centers': []
        }
        
        for i, embedding_vector in enumerate(embedding_matrix):
            distances_to_others = [
                np.linalg.norm(embedding_vector - other_vector)
                for j, other_vector in enumerate(embedding_matrix) if i != j
            ]
            
            if distances_to_others:
                local_density = 1.0 / (np.mean(distances_to_others) + 1e-10)
                
                if local_density > np.mean([1.0 / (np.mean([np.linalg.norm(ev - ov) for j, ov in enumerate(embedding_matrix) if k != j]) + 1e-10) for k, ev in enumerate(embedding_matrix)]):
                    density_map['high_density_regions'].append(i)
        
        return density_map
    
    def _calculate_content_diversity(self) -> float:
        """Calculate content diversity score."""
        unique_texts = set(emb.text for emb in self.embeddings)
        return len(unique_texts) / len(self.embeddings)
    
    def _calculate_dimension_importance(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """Calculate importance score for each dimension."""
        variances = np.var(embedding_matrix, axis=0)
        max_variance = np.max(variances)
        importance_scores = variances / (max_variance + 1e-10)
        return importance_scores
    
    def _analyze_dimensional_correlations(self, embedding_matrix: np.ndarray) -> Dict[str, float]:
        """Analyze correlations between dimensions."""
        correlation_matrix = np.corrcoef(embedding_matrix.T)
        
        return {
            'mean_correlation': float(np.mean(np.abs(correlation_matrix))),
            'max_correlation': float(np.max(np.abs(correlation_matrix))),
            'correlation_sparsity': float(np.sum(np.abs(correlation_matrix) < 0.1) / correlation_matrix.size)
        }
    
    def _identify_active_dimensions(self, dimension_variances: np.ndarray) -> List[int]:
        """Identify dimensions with significant variance."""
        threshold = np.mean(dimension_variances) * 0.5
        active_dims = np.where(dimension_variances > threshold)[0]
        return active_dims.tolist()
    
    def _group_semantic_dimensions(self, embedding_matrix: np.ndarray) -> Dict[str, List[int]]:
        """Group dimensions by semantic similarity."""
        dimension_groups = {
            'high_variance': [],
            'medium_variance': [],
            'low_variance': []
        }
        
        variances = np.var(embedding_matrix, axis=0)
        variance_thresholds = np.percentile(variances, [33, 67])
        
        for i, var in enumerate(variances):
            if var > variance_thresholds[1]:
                dimension_groups['high_variance'].append(i)
            elif var > variance_thresholds[0]:
                dimension_groups['medium_variance'].append(i)
            else:
                dimension_groups['low_variance'].append(i)
        
        return dimension_groups
    
    def _identify_field_relevant_dimensions(self, importance_scores: np.ndarray) -> List[int]:
        """Identify dimensions most relevant for field theory."""
        threshold = np.mean(importance_scores) + np.std(importance_scores)
        field_relevant = np.where(importance_scores > threshold)[0]
        return field_relevant.tolist()
    
    def _calculate_field_strength_distribution(self, embedding_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate field strength distribution across the space."""
        magnitudes = np.linalg.norm(embedding_matrix, axis=1)
        
        return {
            'mean_field_strength': float(np.mean(magnitudes)),
            'field_strength_variance': float(np.var(magnitudes)),
            'strong_field_regions': np.where(magnitudes > np.mean(magnitudes) + np.std(magnitudes))[0].tolist(),
            'weak_field_regions': np.where(magnitudes < np.mean(magnitudes) - np.std(magnitudes))[0].tolist()
        }
    
    def _analyze_phase_relationships(self, embedding_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze phase relationships between embeddings."""
        n_embeddings = embedding_matrix.shape[0]
        phase_matrix = np.zeros((n_embeddings, n_embeddings))
        
        for i in range(n_embeddings):
            for j in range(n_embeddings):
                if i != j:
                    dot_product = np.dot(embedding_matrix[i], embedding_matrix[j])
                    magnitude_product = np.linalg.norm(embedding_matrix[i]) * np.linalg.norm(embedding_matrix[j])
                    phase_matrix[i, j] = np.arccos(np.clip(dot_product / (magnitude_product + 1e-10), -1, 1))
        
        return {
            'mean_phase_difference': float(np.mean(phase_matrix)),
            'phase_coherence': float(1.0 / (np.std(phase_matrix) + 1e-10)),
            'phase_distribution': np.histogram(phase_matrix.flatten(), bins=10)[0].tolist()
        }
    
    def _calculate_trajectory_potentials(self, embedding_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate trajectory potentials for conceptual charge movement."""
        gradients = np.gradient(embedding_matrix, axis=0)
        gradient_magnitudes = np.linalg.norm(gradients, axis=1)
        
        return {
            'mean_gradient_magnitude': float(np.mean(gradient_magnitudes)),
            'trajectory_complexity': float(np.std(gradient_magnitudes)),
            'potential_wells': np.where(gradient_magnitudes < np.mean(gradient_magnitudes) * 0.5)[0].tolist(),
            'potential_peaks': np.where(gradient_magnitudes > np.mean(gradient_magnitudes) * 1.5)[0].tolist()
        }
    
    def _create_charge_placement_grid(self) -> Dict[str, Any]:
        """Create optimal grid for charge placement in universe."""
        embedding_matrix = np.array([emb.vector for emb in self.embeddings])
        
        min_coords = np.min(embedding_matrix, axis=0)
        max_coords = np.max(embedding_matrix, axis=0)
        
        return {
            'grid_bounds': {
                'min_coordinates': min_coords.tolist(),
                'max_coordinates': max_coords.tolist(),
                'coordinate_ranges': (max_coords - min_coords).tolist()
            },
            'recommended_grid_resolution': max(10, int(np.sqrt(len(self.embeddings)))),
            'charge_density_target': len(self.embeddings) / np.prod(max_coords - min_coords + 1e-10)
        }
    
    def _identify_field_interaction_zones(self) -> Dict[str, Any]:
        """Identify zones where field interactions are strongest."""
        embedding_matrix = np.array([emb.vector for emb in self.embeddings])
        distance_matrix = squareform(pdist(embedding_matrix))
        
        interaction_threshold = np.mean(distance_matrix) * 0.3
        strong_interaction_pairs = np.where(distance_matrix < interaction_threshold)
        
        return {
            'strong_interaction_pairs': list(zip(strong_interaction_pairs[0].tolist(), 
                                               strong_interaction_pairs[1].tolist())),
            'interaction_zone_density': len(strong_interaction_pairs[0]) / len(self.embeddings),
            'mean_interaction_distance': float(np.mean(distance_matrix[strong_interaction_pairs]))
        }
    
    def _determine_boundary_conditions(self) -> Dict[str, str]:
        """Determine optimal boundary conditions for the universe."""
        n_embeddings = len(self.embeddings)
        
        if n_embeddings > 1000:
            boundary_type = 'periodic'
        elif n_embeddings > 100:
            boundary_type = 'reflective'
        else:
            boundary_type = 'absorptive'
        
        return {
            'boundary_type': boundary_type,
            'reasoning': f'Selected based on {n_embeddings} embeddings for optimal field effects'
        }
    
    def _recommend_gamma_calibration(self) -> Dict[str, float]:
        """Recommend gamma calibration range for field theory."""
        embedding_matrix = np.array([emb.vector for emb in self.embeddings])
        mean_magnitude = np.mean(np.linalg.norm(embedding_matrix, axis=1))
        
        return {
            'min_gamma': max(0.5, mean_magnitude * 0.5),
            'max_gamma': min(2.0, mean_magnitude * 2.0),
            'optimal_gamma': mean_magnitude
        }
    
    def _recommend_observational_states(self) -> Dict[str, float]:
        """Recommend observational state range for conceptual charges."""
        return {
            'min_state': 0.1,
            'max_state': 3.0,
            'default_state': 1.0,
            'exploration_range': 2.0
        }
    
    def _calculate_optimal_universe_size(self) -> Dict[str, float]:
        """Calculate optimal universe size based on embedding distribution."""
        embedding_matrix = np.array([emb.vector for emb in self.embeddings])
        
        coordinate_ranges = np.ptp(embedding_matrix, axis=0)
        mean_range = np.mean(coordinate_ranges)
        
        return {
            'recommended_size': float(mean_range * 1.5),
            'minimum_size': float(mean_range),
            'maximum_size': float(mean_range * 3.0)
        }
    
    def _recommend_optimal_gamma(self) -> float:
        """Recommend optimal gamma value for universe."""
        magnitudes = [emb.magnitude for emb in self.embeddings]
        return float(np.mean(magnitudes))
    
    def _recommend_state_range(self) -> Tuple[float, float]:
        """Recommend observational state range."""
        return (0.5, 2.5)
    
    def _recommend_spatial_resolution(self) -> float:
        """Recommend spatial resolution for universe grid."""
        n_embeddings = len(self.embeddings)
        return max(0.1, 1.0 / np.sqrt(n_embeddings))
    
    def _recommend_evolution_rate(self) -> float:
        """Recommend temporal evolution rate."""
        return 0.01
    
    def _recommend_interaction_strength(self) -> float:
        """Recommend field interaction strength."""
        return 1.0
    
    def _create_position_mapping(self) -> Dict[int, List[float]]:
        """Create mapping from embedding index to universe position."""
        embedding_matrix = np.array([emb.vector for emb in self.embeddings])
        
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            positions_3d = pca.fit_transform(embedding_matrix)
        except ImportError:
            positions_3d = embedding_matrix[:, :3] if embedding_matrix.shape[1] >= 3 else np.pad(embedding_matrix, ((0, 0), (0, 3 - embedding_matrix.shape[1])), 'constant')
        
        return {i: pos.tolist() for i, pos in enumerate(positions_3d)}
    
    def _create_charge_mapping(self) -> Dict[int, Dict[str, float]]:
        """Create mapping from embedding to charge parameters."""
        charge_mapping = {}
        
        for i, embedding in enumerate(self.embeddings):
            charge_mapping[i] = {
                'magnitude': embedding.magnitude,
                'gamma_modifier': 1.0,
                'observational_state': 1.0,
                'tau_vector': embedding.vector.tolist()
            }
        
        return charge_mapping
    
    def _extract_field_parameters(self) -> Dict[str, Any]:
        """Extract field effect parameters."""
        return {
            'field_strength_calibration': 1.0,
            'interaction_range': 1.0,
            'field_decay_rate': 0.1,
            'resonance_frequency': 1.0
        }
    
    def _define_universe_topology(self) -> Dict[str, str]:
        """Define universe topology based on embedding characteristics."""
        return {
            'topology_type': 'euclidean',
            'metric': 'minkowski',
            'curvature': 'flat'
        }
    
    def _define_evolution_dynamics(self) -> Dict[str, float]:
        """Define evolution dynamics for the universe."""
        return {
            'temporal_step_size': 0.01,
            'evolution_rate': 0.1,
            'damping_factor': 0.95
        }
    
    def _define_validation_metrics(self) -> List[str]:
        """Define metrics for validating universe conversion."""
        return [
            'semantic_preservation',
            'distance_conservation',
            'cluster_coherence',
            'field_effect_accuracy'
        ]
    
    def _evaluate_clustering_quality(self, distance_matrix: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Evaluate the quality of clustering results."""
        if len(set(cluster_labels)) < 2:
            return 0.0
        
        silhouette_scores = []
        for i, label in enumerate(cluster_labels):
            if label == -1:
                continue
            
            same_cluster = [j for j, l in enumerate(cluster_labels) if l == label and j != i]
            other_clusters = [j for j, l in enumerate(cluster_labels) if l != label and l != -1]
            
            if not same_cluster or not other_clusters:
                continue
            
            intra_distance = np.mean([distance_matrix[i, j] for j in same_cluster])
            inter_distance = np.mean([distance_matrix[i, j] for j in other_clusters])
            
            silhouette_score = (inter_distance - intra_distance) / max(inter_distance, intra_distance)
            silhouette_scores.append(silhouette_score)
        
        return float(np.mean(silhouette_scores)) if silhouette_scores else 0.0
    
    def __repr__(self) -> str:
        """String representation of the Semantic Space Analyzer."""
        return (f"SemanticSpaceAnalyzer(embeddings={len(self.embeddings)}, "
                f"dimension={self.dimension}, "
                f"analyzed={'Yes' if self.space_metrics else 'No'})")