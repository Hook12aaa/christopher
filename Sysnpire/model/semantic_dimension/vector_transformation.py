from Sysnpire.utils.logger import get_logger
import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Note: Using existing analysis infrastructure from intial/ modules

logger = get_logger(__name__)


@dataclass
class SemanticField:
    """
    Represents a semantic field S_τ(x) as a functional object.
    
    This is the core data structure representing the field transformation:
    S_τ(x) = Σᵢ e_τ,ᵢ · φᵢ(x) · e^(iθ_τ,ᵢ)
    """
    # Field parameters (from embedding)
    embedding_components: np.ndarray  # e_τ,ᵢ values
    phase_factors: np.ndarray        # θ_τ,ᵢ values
    
    # Basis function system
    basis_functions: 'BasisFunctionSet'  # φᵢ(x) implementations
    
    # Field metadata
    source_token: str
    manifold_dimension: int = 1024
    
    def evaluate_at(self, position_x: np.ndarray) -> complex:
        """
        Evaluate semantic field at position x in manifold.
        
        PERFORMANCE OPTIMIZATION: Vectorized implementation replaces Python loops
        with numpy operations for significant speedup.
        
        Implements: S_τ(x) = Σᵢ e_τ,ᵢ · φᵢ(x) · e^(iθ_τ,ᵢ)
        """
        # VECTORIZED: Evaluate all basis functions at once
        phi_values = self.basis_functions.evaluate_basis_vectorized(position_x)
        
        # VECTORIZED: Compute all phase factors at once
        phase_factors = np.exp(1j * self.phase_factors)
        
        # VECTORIZED: Compute field sum in one operation
        field_value = np.sum(self.embedding_components * phi_values * phase_factors)
        
        return complex(field_value)
    
    def evaluate_at_batch(self, positions: np.ndarray) -> np.ndarray:
        """
        PERFORMANCE OPTIMIZATION: Batch evaluation of semantic field.
        
        Evaluates semantic field at multiple positions simultaneously.
        Provides significant speedup when evaluating many positions.
        
        Args:
            positions: Array of positions [N, D] where N is number of positions
            
        Returns:
            Array of complex field values [N] for each position
        """
        # VECTORIZED: Compute phase factors once
        phase_factors = np.exp(1j * self.phase_factors)
        
        # Initialize result array
        field_values = np.zeros(len(positions), dtype=complex)
        
        # Evaluate field at each position (can be further optimized if needed)
        for i, position in enumerate(positions):
            phi_values = self.basis_functions.evaluate_basis_vectorized(position)
            field_values[i] = np.sum(self.embedding_components * phi_values * phase_factors)
        
        return field_values
    
    def get_field_metadata(self) -> Dict[str, Any]:
        """Get metadata about this semantic field."""
        return {
            'source_token': self.source_token,
            'manifold_dimension': self.manifold_dimension,
            'num_components': len(self.embedding_components),
            'field_magnitude': float(np.linalg.norm(self.embedding_components)),
            'phase_range': [float(np.min(self.phase_factors)), float(np.max(self.phase_factors))]
        }


class BasisFunctionSet:
    """
    Set of basis functions φᵢ(x) for semantic field generation.
    
    Uses analysis from EmbeddingSpaceAnalyzer to design basis functions
    that respect the geometric structure of the embedding space.
    """
    
    def __init__(self, design_params: Dict[str, Any], embedding_dimension: int = 1024):
        """
        Initialize basis function set from design parameters.
        
        Args:
            design_params: Parameters from EmbeddingSpaceAnalyzer
            embedding_dimension: Dimension of embedding space
        """
        self.design_params = design_params
        self.embedding_dimension = embedding_dimension
        self.basis_cache = {}
        
        # Extract key parameters
        self.basis_type = design_params.get('basis_type', 'radial_semantic')
        self.num_centers = design_params['basis_centers']['num_centers']
        self.coverage_radius = design_params['basis_centers']['coverage_radius']
        
        logger.info(f"Initialized {self.basis_type} basis functions with {self.num_centers} centers")
    
    def evaluate_basis(self, i: int, position_x: np.ndarray) -> float:
        """
        Evaluate basis function φᵢ(x) at position x.
        
        Args:
            i: Basis function index
            position_x: Position in semantic manifold
            
        Returns:
            Real-valued basis function value
        """
        cache_key = (i, tuple(position_x) if len(position_x) <= 10 else hash(position_x.tobytes()))
        
        if cache_key in self.basis_cache:
            return self.basis_cache[cache_key]
        
        # Compute basis function value - only radial_semantic supported
        if self.basis_type == 'radial_semantic':
            basis_value = self._radial_semantic_basis(i, position_x)
        else:
            raise ValueError(
                f"Unsupported basis function type: {self.basis_type}. "
                f"Only 'radial_semantic' is supported (uses BGE cluster analysis). "
                f"No default basis functions allowed."
            )
        
        self.basis_cache[cache_key] = basis_value
        return basis_value
    
    def evaluate_basis_vectorized(self, position_x: np.ndarray) -> np.ndarray:
        """
        PERFORMANCE OPTIMIZATION: Vectorized evaluation of all basis functions.
        
        Evaluates all basis functions φᵢ(x) at position x simultaneously.
        Provides significant speedup over individual evaluate_basis calls.
        
        Args:
            position_x: Position in semantic manifold
            
        Returns:
            Array of all basis function values [φ₀(x), φ₁(x), ..., φₙ(x)]
        """
        # Check if full vectorized result is cached
        cache_key = tuple(position_x) if len(position_x) <= 10 else hash(position_x.tobytes())
        full_cache_key = ('vectorized', cache_key)
        
        if full_cache_key in self.basis_cache:
            return self.basis_cache[full_cache_key]
        
        # Compute all basis values vectorized
        if self.basis_type == 'radial_semantic':
            basis_values = self._radial_semantic_basis_vectorized(position_x)
        else:
            raise ValueError(
                f"Unsupported basis function type: {self.basis_type}. "
                f"Only 'radial_semantic' is supported (uses BGE cluster analysis). "
                f"No default basis functions allowed."
            )
        
        # Cache the vectorized result
        self.basis_cache[full_cache_key] = basis_values
        return basis_values
    
    def _radial_semantic_basis(self, i: int, position_x: np.ndarray) -> float:
        """
        Radial basis function based on semantic clustering structure.
        
        Uses cluster centers from embedding space analysis as basis centers.
        """
        # Create basis center for index i
        # For now, use systematic placement across embedding dimensions
        center = self._get_basis_center(i)
        
        # Compute distance in embedding space
        distance = np.linalg.norm(position_x - center)
        
        # Gaussian basis function with adaptive width
        sigma = self.coverage_radius * 0.5  # Adaptive width based on analysis
        basis_value = np.exp(-distance**2 / (2 * sigma**2))
        
        return float(basis_value)
    
    def _get_basis_center(self, i: int) -> np.ndarray:
        """Get basis center for index i using actual cluster centers from spatial analysis."""
        # Use actual cluster centers from BGE spatial analysis
        basis_centers = self.design_params['basis_centers']
        cluster_centers = np.array(basis_centers['cluster_centers'])
        
        if len(cluster_centers) == 0:
            raise ValueError("No cluster centers available in spatial analysis - BGE analysis required")
        
        # Map basis function index to cluster center
        cluster_idx = i % len(cluster_centers)
        center = cluster_centers[cluster_idx]
        
        # Ensure center is normalized to unit sphere (BGE's natural geometry)
        center_norm = np.linalg.norm(center)
        if center_norm == 0:
            raise ValueError(f"Cluster center {cluster_idx} has zero norm - invalid spatial analysis")
            
        return center / center_norm
    
    def _radial_semantic_basis_vectorized(self, position_x: np.ndarray) -> np.ndarray:
        """
        PERFORMANCE OPTIMIZATION: Vectorized radial basis function computation.
        
        Computes all radial basis functions simultaneously using vectorized operations.
        Much faster than individual _radial_semantic_basis calls.
        
        Args:
            position_x: Position in semantic manifold
            
        Returns:
            Array of all basis function values
        """
        # Get all cluster centers at once
        basis_centers = self.design_params['basis_centers']
        cluster_centers = np.array(basis_centers['cluster_centers'])
        
        if len(cluster_centers) == 0:
            raise ValueError("No cluster centers available in spatial analysis - BGE analysis required")
        
        # Normalize all centers to unit sphere (vectorized)
        center_norms = np.linalg.norm(cluster_centers, axis=1, keepdims=True)
        center_norms[center_norms == 0] = 1e-10  # Avoid division by zero
        normalized_centers = cluster_centers / center_norms
        
        # Compute distances from position to all centers (vectorized)
        distances = np.linalg.norm(normalized_centers - position_x[np.newaxis, :], axis=1)
        
        # Apply Gaussian basis function to all distances (vectorized)
        sigma = self.coverage_radius * 0.5
        basis_values = np.exp(-distances**2 / (2 * sigma**2))
        
        # Handle case where we need more basis functions than cluster centers
        # Cycle through cluster centers for additional basis functions
        num_components = len(position_x)  # Use embedding dimension
        if len(basis_values) < num_components:
            # Repeat pattern to match embedding dimension
            repetitions = (num_components // len(basis_values)) + 1
            extended_values = np.tile(basis_values, repetitions)
            basis_values = extended_values[:num_components]
        elif len(basis_values) > num_components:
            # Truncate to embedding dimension
            basis_values = basis_values[:num_components]
        
        return basis_values
    
    # REMOVED: _default_gaussian_basis - No default basis functions allowed!
    # All basis functions must use actual BGE spatial cluster analysis


class VectorTransformation():
    """
    Core transformer for converting static embedding vectors into dynamic semantic fields.
    
    This class implements the fundamental S_τ(x) transformation that enables
    embeddings to generate field effects across semantic space rather than
    merely representing static positions.
    """

    def __init__(self, from_base: bool, embedding_dimension: int = None, helper = None, phase_computation_method: str = "component_based"):
        """
        Initialize the VectorTransformation for S_τ(x) field generation.
        
        Args:
            from_base (bool): Whether building from base model (BGE/MPNet) or existing universe
            embedding_dimension (int, optional): Dimension of embedding vectors (1024 for BGE)
            helper: BGE ingestion model (if from_base=True) or universe (if from_base=False)
            phase_computation_method (str): Method for computing e^(iθ_τ,ᵢ) factors
        """
        self.from_base = from_base
        self.embedding_dimension = embedding_dimension
        self.phase_computation_method = phase_computation_method

        if self.from_base: # if we are building from scratch, we need to access our model ingestion tools directly, user must of turned on from_base
            self.model = helper
        else:
            self.universe = helper

    


    def model_transform_to_field(self, embedding: dict, precomputed_spatial_analysis: dict = None) -> dict:
        """
        Core transformation: Convert embedding vector to semantic field S_τ(x).
        
        Implements: S_τ(x) = Σᵢ e_τ,ᵢ · φᵢ(x) · e^(iθ_τ,ᵢ)
        
        This method transforms a static BGE embedding into a dynamic semantic field
        using spatial analysis from the ingestion model to design basis functions
        and phase relationships.
        
        Args:
            embedding: Dict containing:
                - 'vector': The embedding vector e_τ [1024]
                - 'manifold_properties': Spatial properties from BGE analysis
                - 'token': Source token (optional)
                - 'index': Token index (optional)
            precomputed_spatial_analysis: Optional pre-computed spatial analysis to avoid redundant computation
            
        Returns:
            Dict containing SemanticField object and metadata
        """
        # Extract embedding components
        embedding_vector = np.array(embedding['vector'])
        token = embedding.get('token', '<UNK>')
        manifold_props = embedding.get('manifold_properties', {})
        
        logger.info(f"Transforming embedding for token '{token}' to semantic field")
        
        # Get spatial analysis - use precomputed if available to avoid redundant computation
        if precomputed_spatial_analysis is not None:
            # PERFORMANCE: Use pre-computed spatial analysis (saves 6s per embedding!)
            logger.debug(f"🚀 Using pre-computed spatial analysis for token '{token}'")
            spatial_analysis = precomputed_spatial_analysis
            spatial_params = spatial_analysis['spatial_field_parameters']
        elif self.from_base and hasattr(self.model, 'extract_spatial_field_analysis'):
            # Fallback: compute spatial analysis if not provided (original behavior)
            logger.warning(f"⚠️ Computing spatial analysis for token '{token}' (consider pre-computing)")
            spatial_analysis = self.model.extract_spatial_field_analysis(
                num_samples=500, 
                return_full_details=False
            )
            spatial_params = spatial_analysis['spatial_field_parameters']
            logger.info(f"Using BGE spatial analysis: {spatial_analysis.get('available_full_data', 'sampled data')}")
        else:
            # No fallback - require proper BGE spatial analysis
            raise ValueError(
                "BGE spatial analysis required for field transformation. "
                "Ensure from_base=True and BGE model has extract_spatial_field_analysis method. "
                "No default parameters - proper spatial analysis is mandatory for field theory."
            )
        
        # 1. COMPUTE PHASE FACTORS: e^(iθ_τ,ᵢ)
        phase_factors = self._compute_phase_factors(
            embedding_vector, 
            spatial_params,
            manifold_props
        )
        
        # 2. CREATE BASIS FUNCTION SET: φᵢ(x) 
        basis_functions = BasisFunctionSet(
            design_params=spatial_params,
            embedding_dimension=len(embedding_vector)
        )
        
        # 3. CREATE SEMANTIC FIELD OBJECT
        semantic_field = SemanticField(
            embedding_components=embedding_vector,  # e_τ,ᵢ
            phase_factors=phase_factors,           # θ_τ,ᵢ  
            basis_functions=basis_functions,       # φᵢ(x)
            source_token=token,
            manifold_dimension=len(embedding_vector)
        )
        
        # 4. VALIDATE FIELD PROPERTIES
        field_validation = self._validate_semantic_field(semantic_field, spatial_params)
        
        # 5. RETURN FIELD REPRESENTATION WITH ORIGINAL PROPERTIES
        return {
            'semantic_field': semantic_field,
            'field_metadata': semantic_field.get_field_metadata(),
            'spatial_parameters': spatial_params,
            'phase_computation': {
                'method': self.phase_computation_method,
                'phase_range': [float(np.min(phase_factors)), float(np.max(phase_factors))],
                'phase_variance': float(np.var(phase_factors))
            },
            'basis_function_info': {
                'type': basis_functions.basis_type,
                'num_centers': basis_functions.num_centers,
                'coverage_radius': basis_functions.coverage_radius
            },
            'field_validation': field_validation,
            'transformation_complete': True,
            
            # ORIGINAL EMBEDDING PROPERTIES - preserved for reference/debugging
            'original_embedding': {
                'vector': embedding_vector.tolist(),
                'token': token,
                'manifold_properties': manifold_props,
                'vector_norm': float(np.linalg.norm(embedding_vector)),
                'vector_dimension': len(embedding_vector),
                'index': embedding.get('index', None)
            }
        }
    
    def _compute_phase_factors(self, embedding_vector: np.ndarray, 
                              spatial_params: Dict[str, Any],
                              manifold_props: Dict[str, Any]) -> np.ndarray:
        """
        Compute phase factors e^(iθ_τ,ᵢ) from embedding components.
        
        Uses component-based method that extracts phase information from
        the embedding vector itself, informed by spatial field analysis.
        """
        if self.phase_computation_method == "component_based":
            # Extract phase from embedding components using spatial structure
            phase_factors = np.zeros(len(embedding_vector))
            
            # Get spatial phase properties from analysis
            spatial_phase = spatial_params.get('spatial_phase_properties')
            if not spatial_phase or 'mean_spectral_entropy' not in spatial_phase:
                raise ValueError("Spatial phase properties required from BGE analysis")
            base_phase_variance = spatial_phase['mean_spectral_entropy']
            
            for i, component in enumerate(embedding_vector):
                # Base phase from component value
                if component != 0:
                    base_phase = np.arctan2(0.1 * component, abs(component))  # Stable phase extraction
                else:
                    base_phase = 0.0
                
                # Modulate with spatial frequency information
                spatial_freq_mod = 0.1 * np.sin(2 * np.pi * i / len(embedding_vector))
                
                # Add manifold curvature influence from spatial analysis
                curvature_mod = 0.0
                if manifold_props and 'local_curvature' in manifold_props:
                    curvature_mod = 0.05 * manifold_props['local_curvature'] * np.cos(i)
                else:
                    # Use average curvature from spatial analysis
                    avg_curvature = spatial_params.get('spatial_interactions').get('curvature_distribution').get('mean')
                    curvature_mod = 0.05 * avg_curvature * np.cos(i)
                
                # Total phase
                total_phase = base_phase + spatial_freq_mod + curvature_mod
                phase_factors[i] = total_phase
                
            return phase_factors
        else:
            raise ValueError(
                f"Unsupported phase computation method: {self.phase_computation_method}. "
                f"Only 'component_based' is supported (uses embedding components + spatial analysis). "
                f"No default phase computation allowed."
            )
    

    def _validate_semantic_field(self, semantic_field: SemanticField, 
                                spatial_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate semantic field properties and mathematical consistency.
        """
        validation_results = {}
        
        try:
            # Test field evaluation at origin
            origin = np.zeros(semantic_field.manifold_dimension)
            field_at_origin = semantic_field.evaluate_at(origin)
            
            validation_results.update({
                'field_evaluates': True,
                'field_at_origin': {
                    'magnitude': float(abs(field_at_origin)),
                    'phase': float(np.angle(field_at_origin)),
                    'real_part': float(field_at_origin.real),
                    'imag_part': float(field_at_origin.imag)
                },
                'field_finite': bool(np.isfinite(field_at_origin)),
                'components_finite': bool(np.all(np.isfinite(semantic_field.embedding_components))),
                'phases_finite': bool(np.all(np.isfinite(semantic_field.phase_factors)))
            })
            
            validation_results['validation_passed'] = True
            
        except Exception as e:
            validation_results.update({
                'field_evaluates': False,
                'validation_error': str(e),
                'validation_passed': False
            })
        
        return validation_results
    
    def get_detailed_spatial_analysis(self, num_samples: int = 500) -> Optional[Dict[str, Any]]:
        """
        DEBUGGING UTILITY: Get full spatial analysis details for field investigation.
        
        Use this method when you need to debug field behavior, understand spatial
        patterns, or optimize basis function placement. Returns complete analysis
        with all embedding details (memory intensive!).
        
        Args:
            num_samples: Number of embeddings to analyze in detail
            
        Returns:
            Full spatial analysis with all per-embedding details, or None if unavailable
        """
        if self.from_base and hasattr(self.model, 'extract_spatial_field_analysis'):
            logger.info("🔍 Extracting FULL spatial analysis details for debugging...")
            detailed_analysis = self.model.extract_spatial_field_analysis(
                num_samples=num_samples,
                return_full_details=True  # Get everything!
            )
            
            logger.info(f"📊 Retrieved detailed analysis: {detailed_analysis.get('memory_usage_warning', 'unknown size')}")
            return detailed_analysis
        else:
            logger.warning("❌ No BGE model available for detailed spatial analysis")
            return None

