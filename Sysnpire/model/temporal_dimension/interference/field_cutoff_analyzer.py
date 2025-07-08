"""
Field Cutoff Analyzer - Discovering Natural Cutoff Radius for O(N) Complexity

MATHEMATICAL FOUNDATION:
This analyzer discovers the natural cutoff radius inherent in field theory calculations,
leveraging the dual-decay persistence structure Ψ_persistence(s-s₀) and field interference
signatures to identify where interactions become negligible.

KEY INSIGHT:
Your field theory's Gaussian and exponential-cosine decay terms create natural
locality in embedding space, enabling O(N) complexity through intelligent cutoff.

FIELD-THEORETIC JUSTIFICATION:
- Vivid layer: Gaussian decay σ²(s) creates exponential dropoff
- Character layer: Exponential-cosine decay with λ parameter
- Trajectory operators: Natural clustering through breathing patterns
- Phase coordination: Local coherence structures in embedding space

APPROACH:
1. Analyze interference vs embedding distance relationships
2. Fit decay curves to field theory predictions
3. Find 99th percentile cutoff radius for negligible interactions
4. Validate against dual-decay persistence properties
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy import optimize, stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass
import warnings

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.utils.logger import get_logger
from Sysnpire.utils.field_theory_optimizers import field_theory_auto_optimize

logger = get_logger(__name__)


@dataclass
class FieldCutoffAnalysis:
    """Results from field-theoretic cutoff analysis."""
    optimal_cutoff_radius: float           # Radius where 99% of interference < threshold
    decay_constant: float                  # Fitted exponential decay parameter
    field_locality_strength: float        # How well interference follows distance decay
    mean_significant_neighbors: float     # Average neighbors within cutoff per charge
    max_significant_neighbors: int        # Maximum neighbors any charge has within cutoff
    cutoff_justification: str             # Mathematical basis for cutoff choice
    embedding_effective_dimension: int    # Effective dimension for cell partitioning
    decay_curve_r_squared: float         # Quality of exponential fit to decay
    field_coherence_radius: float        # Radius of coherent field structures
    sparsity_at_cutoff: float            # Sparsity ratio achieved at chosen cutoff


@dataclass
class DecayFitResults:
    """Results from fitting field theory decay curves to interference data."""
    exponential_decay_const: float       # λ parameter in exp(-λr) 
    gaussian_decay_sigma: float          # σ parameter in exp(-r²/2σ²)
    combined_decay_amplitude: float      # Overall amplitude scaling
    exponential_r_squared: float         # Quality of exponential fit
    gaussian_r_squared: float            # Quality of Gaussian fit
    combined_r_squared: float            # Quality of combined fit
    best_model: str                      # 'exponential', 'gaussian', or 'combined'


class FieldCutoffAnalyzer:
    """
    Discover natural cutoff radius from field theory decay properties.
    
    FIELD-THEORETIC APPROACH:
    Uses the mathematical structure of your field theory to identify natural
    locality patterns that enable O(N) complexity without mathematical approximation.
    
    MATHEMATICAL INTEGRITY:
    - Preserves exact interference calculations within cutoff
    - Justifies neglect beyond cutoff using field decay theory
    - No artificial approximations - uses natural field properties
    """
    
    def __init__(self, 
                 negligible_threshold_percentile: float = 1.0,  # 99th percentile cutoff
                 max_sample_size: int = 500,
                 pca_dimensions: int = 10):
        """
        Initialize field cutoff analyzer.
        
        Args:
            negligible_threshold_percentile: Percentile for negligible interactions (1% = 99th percentile cutoff)
            max_sample_size: Maximum sample size for analysis (memory management)
            pca_dimensions: Dimensions for PCA projection to handle high-dimensional embeddings
        """
        self.negligible_threshold_percentile = negligible_threshold_percentile
        self.max_sample_size = max_sample_size
        self.pca_dimensions = pca_dimensions
        
        # Field theory validation parameters
        self.field_locality_threshold = 0.7  # R² threshold for good distance correlation
        self.min_significant_neighbors = 3   # Minimum neighbors for meaningful interaction
        
    @field_theory_auto_optimize(prefer_accuracy=True, profile=True)
    def analyze_field_cutoff(self, temporal_biographies: List) -> FieldCutoffAnalysis:
        """
        Discover optimal cutoff radius from field theory properties.
        
        MATHEMATICAL APPROACH:
        1. Sample temporal biographies for analysis
        2. Compute exact interference matrix
        3. Calculate embedding space distances  
        4. Fit field theory decay curves
        5. Find 99th percentile cutoff radius
        6. Validate against field locality properties
        
        Args:
            temporal_biographies: List of TemporalBiography objects
            
        Returns:
            FieldCutoffAnalysis with optimal cutoff and field properties
        """
        logger.info(f"🔍 Analyzing field cutoff radius for {len(temporal_biographies)} temporal biographies")
        
        # Step 1: Intelligent sampling for computational efficiency
        sample_biographies, sample_indices = self._sample_biographies_for_analysis(temporal_biographies)
        sample_size = len(sample_biographies)
        
        logger.info(f"📊 Using {sample_size} biographies for cutoff analysis")
        
        # Step 2: Compute exact interference matrix for sample
        logger.info("🧮 Computing exact interference matrix...")
        interference_matrix = self._compute_exact_interference_matrix(sample_biographies)
        
        # Step 3: Calculate embedding space distances using field signatures
        logger.info("📏 Computing field-theoretic distances...")
        embedding_distances = self._compute_field_distances(sample_biographies)
        
        # Step 4: Analyze effective dimensionality for cell partitioning
        logger.info("🔬 Analyzing embedding space dimensionality...")
        effective_dimension = self._analyze_effective_dimension(sample_biographies)
        
        # Step 5: Fit field theory decay curves to interference vs distance
        logger.info("📈 Fitting field theory decay curves...")
        decay_fit = self._fit_field_decay_curves(embedding_distances, interference_matrix)
        
        # Step 6: Find optimal cutoff radius based on negligible threshold
        logger.info("🎯 Finding optimal cutoff radius...")
        cutoff_radius = self._find_optimal_cutoff_radius(
            embedding_distances, interference_matrix, decay_fit
        )
        
        # Step 7: Validate cutoff against field locality properties
        logger.info("✅ Validating field locality properties...")
        locality_validation = self._validate_field_locality(
            embedding_distances, interference_matrix, cutoff_radius
        )
        
        # Step 8: Analyze neighbor statistics at cutoff
        neighbor_stats = self._analyze_neighbor_statistics(
            embedding_distances, interference_matrix, cutoff_radius
        )
        
        # Step 9: Compute field coherence radius (breathing pattern synchronization)
        coherence_radius = self._compute_field_coherence_radius(
            sample_biographies, embedding_distances
        )
        
        # Step 10: Calculate final sparsity at chosen cutoff
        sparsity_at_cutoff = self._compute_sparsity_at_cutoff(
            embedding_distances, cutoff_radius
        )
        
        # Compile comprehensive analysis results
        analysis = FieldCutoffAnalysis(
            optimal_cutoff_radius=cutoff_radius,
            decay_constant=decay_fit.exponential_decay_const,
            field_locality_strength=locality_validation['correlation_strength'],
            mean_significant_neighbors=neighbor_stats['mean_neighbors'],
            max_significant_neighbors=neighbor_stats['max_neighbors'],
            cutoff_justification=self._generate_cutoff_justification(decay_fit, locality_validation),
            embedding_effective_dimension=effective_dimension,
            decay_curve_r_squared=decay_fit.combined_r_squared,
            field_coherence_radius=coherence_radius,
            sparsity_at_cutoff=sparsity_at_cutoff
        )
        
        # Comprehensive logging of results
        self._log_analysis_results(analysis, decay_fit, locality_validation)
        
        return analysis
    
    def _sample_biographies_for_analysis(self, temporal_biographies: List) -> Tuple[List, np.ndarray]:
        """
        Intelligently sample temporal biographies for cutoff analysis.
        
        SAMPLING STRATEGY:
        - Use stratified sampling based on temporal momentum magnitude
        - Ensure diverse representation of breathing patterns
        - Preserve field coherence structures in sample
        """
        n_total = len(temporal_biographies)
        
        if n_total <= self.max_sample_size:
            return temporal_biographies, np.arange(n_total)
        
        # Extract temporal momentum for stratified sampling
        temporal_momenta = []
        for bio in temporal_biographies:
            momentum_magnitude = abs(bio.temporal_momentum)
            temporal_momenta.append(momentum_magnitude)
        
        temporal_momenta = np.array(temporal_momenta)
        
        # Create momentum-based strata for diverse sampling
        momentum_percentiles = np.percentile(temporal_momenta, [20, 40, 60, 80])
        strata_indices = [[] for _ in range(5)]
        
        for i, momentum in enumerate(temporal_momenta):
            if momentum <= momentum_percentiles[0]:
                strata_indices[0].append(i)
            elif momentum <= momentum_percentiles[1]:
                strata_indices[1].append(i)
            elif momentum <= momentum_percentiles[2]:
                strata_indices[2].append(i)
            elif momentum <= momentum_percentiles[3]:
                strata_indices[3].append(i)
            else:
                strata_indices[4].append(i)
        
        # Sample proportionally from each stratum
        samples_per_stratum = self.max_sample_size // 5
        remainder = self.max_sample_size % 5
        
        sampled_indices = []
        for i, stratum in enumerate(strata_indices):
            if len(stratum) == 0:
                continue
                
            stratum_sample_size = samples_per_stratum + (1 if i < remainder else 0)
            stratum_sample_size = min(stratum_sample_size, len(stratum))
            
            # Deterministic sampling within stratum
            if stratum_sample_size >= len(stratum):
                sampled_indices.extend(stratum)
            else:
                step = len(stratum) // stratum_sample_size
                selected = stratum[::step][:stratum_sample_size]
                sampled_indices.extend(selected)
        
        sampled_indices = np.array(sampled_indices)
        sampled_biographies = [temporal_biographies[i] for i in sampled_indices]
        
        logger.debug(f"Stratified sampling: {len(sampled_biographies)} from {n_total} biographies")
        return sampled_biographies, sampled_indices
    
    def _compute_exact_interference_matrix(self, sample_biographies: List) -> np.ndarray:
        """
        Compute exact interference matrix using field theory calculations.
        
        MATHEMATICAL INTEGRITY:
        Uses exact same calculation as TemporalDimensionHelper._exact_interference
        to ensure consistency with field theory implementation.
        """
        n_sample = len(sample_biographies)
        interference_matrix = np.zeros((n_sample, n_sample), dtype=complex)
        
        for i in range(n_sample):
            for j in range(n_sample):
                if i != j:
                    bio_i = sample_biographies[i]
                    bio_j = sample_biographies[j]
                    
                    # Exact field theory interference calculation
                    interference_matrix[i, j] = self._compute_exact_pair_interference(bio_i, bio_j)
        
        return interference_matrix
    
    def _compute_exact_pair_interference(self, bio_i, bio_j) -> complex:
        """
        Exact interference calculation between two temporal biographies.
        
        FIELD THEORY IMPLEMENTATION:
        phase_interference * trajectory_interference following the mathematical formulation
        in your field theory: phase relationships combined with trajectory operators.
        """
        # Handle SAGE objects in trajectory operators
        traj_i = self._extract_complex_trajectory(bio_i.trajectory_operators)
        traj_j = self._extract_complex_trajectory(bio_j.trajectory_operators)
        
        # Phase interference: mean of complex exponentials of phase differences
        phase_interference = np.mean(
            np.exp(1j * (bio_i.phase_coordination - bio_j.phase_coordination))
        )
        
        # Trajectory interference: mean of trajectory cross-correlation
        trajectory_interference = np.mean(traj_i * np.conj(traj_j))
        
        # Combined field interference
        return phase_interference * trajectory_interference
    
    def _extract_complex_trajectory(self, trajectory_operators: np.ndarray) -> np.ndarray:
        """
        Extract complex trajectory operators, handling SAGE mathematical objects.
        
        MATHEMATICAL PRECISION:
        Preserves SAGE precision when available, falls back to numpy complex types.
        """
        if hasattr(trajectory_operators, 'dtype') and trajectory_operators.dtype == object:
            # Handle SAGE objects
            traj_complex = np.zeros(len(trajectory_operators), dtype=complex)
            for i, op in enumerate(trajectory_operators):
                if hasattr(op, 'real') and hasattr(op, 'imag'):
                    traj_complex[i] = complex(float(op.real()), float(op.imag()))
                else:
                    traj_complex[i] = complex(op)
            return traj_complex
        else:
            # Standard numpy complex array
            return trajectory_operators.astype(complex)
    
    def _compute_field_distances(self, sample_biographies: List) -> np.ndarray:
        """
        Compute embedding space distances using field interference signatures.
        
        FIELD-THEORETIC DISTANCE:
        Uses field_interference_signature as the natural embedding representation,
        as this captures the essential field properties for spatial relationships.
        """
        n_sample = len(sample_biographies)
        
        # Extract field interference signatures
        field_signatures = []
        for bio in sample_biographies:
            # Use field interference signature as natural embedding
            signature = bio.field_interference_signature
            
            # Handle complex signatures by taking magnitude and phase components
            if np.iscomplexobj(signature):
                magnitude = np.abs(signature)
                phase = np.angle(signature)
                # Combine magnitude and phase as real-valued feature vector
                combined_signature = np.concatenate([magnitude, phase])
            else:
                combined_signature = np.real(signature)
            
            field_signatures.append(combined_signature)
        
        field_signatures = np.array(field_signatures)
        
        # Reduce dimensionality for efficiency while preserving local structure
        if field_signatures.shape[1] > self.pca_dimensions:
            pca = PCA(n_components=self.pca_dimensions, random_state=42)
            field_signatures = pca.fit_transform(field_signatures)
            logger.debug(f"PCA reduction: {field_signatures.shape[1]} dimensions, "
                        f"variance explained: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Compute pairwise Euclidean distances
        distances = squareform(pdist(field_signatures, metric='euclidean'))
        
        return distances
    
    def _analyze_effective_dimension(self, sample_biographies: List) -> int:
        """
        Analyze effective dimensionality of embedding space for cell partitioning.
        
        DIMENSIONAL ANALYSIS:
        Determines the intrinsic dimensionality of field structures to optimize
        cell list partitioning strategy.
        """
        # Extract field signatures
        field_signatures = []
        for bio in sample_biographies:
            signature = bio.field_interference_signature
            if np.iscomplexobj(signature):
                # Use magnitude as primary dimension indicator
                field_signatures.append(np.abs(signature))
            else:
                field_signatures.append(signature)
        
        field_signatures = np.array(field_signatures)
        
        # Use PCA to estimate intrinsic dimensionality
        # Find number of components explaining 95% of variance
        pca = PCA(random_state=42)
        pca.fit(field_signatures)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        effective_dim = np.argmax(cumulative_variance >= 0.95) + 1
        
        # Ensure reasonable bounds for cell partitioning
        effective_dim = max(3, min(effective_dim, 10))
        
        logger.debug(f"Effective embedding dimension: {effective_dim} "
                    f"(95% variance threshold)")
        
        return effective_dim
    
    def _fit_field_decay_curves(self, distances: np.ndarray, 
                               interference_matrix: np.ndarray) -> DecayFitResults:
        """
        Fit field theory decay curves to interference vs distance data.
        
        FIELD THEORY MODELS:
        1. Exponential: A * exp(-λr) - character layer decay
        2. Gaussian: A * exp(-r²/2σ²) - vivid layer decay  
        3. Combined: A * [α*exp(-λr) + (1-α)*exp(-r²/2σ²)]
        """
        # Get non-diagonal distance and interference pairs
        mask = ~np.eye(distances.shape[0], dtype=bool)
        distance_values = distances[mask]
        interference_magnitudes = np.abs(interference_matrix[mask])
        
        # Filter out zero distances to avoid fitting issues
        valid_mask = distance_values > 1e-10
        distance_values = distance_values[valid_mask]
        interference_magnitudes = interference_magnitudes[valid_mask]
        
        if len(distance_values) < 10:
            logger.warning("Insufficient data points for decay curve fitting")
            return DecayFitResults(
                exponential_decay_const=1.0, gaussian_decay_sigma=1.0,
                combined_decay_amplitude=1.0, exponential_r_squared=0.0,
                gaussian_r_squared=0.0, combined_r_squared=0.0,
                best_model='exponential'
            )
        
        # Sort by distance for smoother fitting
        sort_indices = np.argsort(distance_values)
        distance_values = distance_values[sort_indices]
        interference_magnitudes = interference_magnitudes[sort_indices]
        
        # Fit exponential decay: A * exp(-λr)
        exp_fit = self._fit_exponential_decay(distance_values, interference_magnitudes)
        
        # Fit Gaussian decay: A * exp(-r²/2σ²)
        gauss_fit = self._fit_gaussian_decay(distance_values, interference_magnitudes)
        
        # Fit combined model
        combined_fit = self._fit_combined_decay(distance_values, interference_magnitudes)
        
        # Determine best model
        r_squared_values = {
            'exponential': exp_fit['r_squared'],
            'gaussian': gauss_fit['r_squared'], 
            'combined': combined_fit['r_squared']
        }
        best_model = max(r_squared_values, key=r_squared_values.get)
        
        return DecayFitResults(
            exponential_decay_const=exp_fit['decay_const'],
            gaussian_decay_sigma=gauss_fit['sigma'],
            combined_decay_amplitude=combined_fit['amplitude'],
            exponential_r_squared=exp_fit['r_squared'],
            gaussian_r_squared=gauss_fit['r_squared'],
            combined_r_squared=combined_fit['r_squared'],
            best_model=best_model
        )
    
    def _fit_exponential_decay(self, distances: np.ndarray, 
                              magnitudes: np.ndarray) -> Dict[str, float]:
        """Fit exponential decay model: A * exp(-λr)"""
        def exponential_func(r, A, lam):
            return A * np.exp(-lam * r)
        
        try:
            # Initial guess based on data
            A_init = np.max(magnitudes)
            lam_init = -np.log(0.5) / np.median(distances)  # Half-life estimate
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = optimize.curve_fit(
                    exponential_func, distances, magnitudes,
                    p0=[A_init, lam_init],
                    bounds=([0, 0], [np.inf, np.inf]),
                    maxfev=1000
                )
            
            A_fit, lam_fit = popt
            
            # Calculate R²
            y_pred = exponential_func(distances, A_fit, lam_fit)
            r_squared = 1 - np.sum((magnitudes - y_pred)**2) / np.sum((magnitudes - np.mean(magnitudes))**2)
            r_squared = max(0, r_squared)  # Ensure non-negative
            
            return {'decay_const': lam_fit, 'amplitude': A_fit, 'r_squared': r_squared}
            
        except Exception:
            logger.debug("Exponential fit failed, using default parameters")
            return {'decay_const': 1.0, 'amplitude': 1.0, 'r_squared': 0.0}
    
    def _fit_gaussian_decay(self, distances: np.ndarray, 
                           magnitudes: np.ndarray) -> Dict[str, float]:
        """Fit Gaussian decay model: A * exp(-r²/2σ²)"""
        def gaussian_func(r, A, sigma):
            return A * np.exp(-r**2 / (2 * sigma**2))
        
        try:
            # Initial guess
            A_init = np.max(magnitudes)
            sigma_init = np.std(distances)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = optimize.curve_fit(
                    gaussian_func, distances, magnitudes,
                    p0=[A_init, sigma_init],
                    bounds=([0, 1e-6], [np.inf, np.inf]),
                    maxfev=1000
                )
            
            A_fit, sigma_fit = popt
            
            # Calculate R²
            y_pred = gaussian_func(distances, A_fit, sigma_fit)
            r_squared = 1 - np.sum((magnitudes - y_pred)**2) / np.sum((magnitudes - np.mean(magnitudes))**2)
            r_squared = max(0, r_squared)
            
            return {'sigma': sigma_fit, 'amplitude': A_fit, 'r_squared': r_squared}
            
        except Exception:
            logger.debug("Gaussian fit failed, using default parameters")
            return {'sigma': 1.0, 'amplitude': 1.0, 'r_squared': 0.0}
    
    def _fit_combined_decay(self, distances: np.ndarray, 
                           magnitudes: np.ndarray) -> Dict[str, float]:
        """Fit combined exponential + Gaussian decay model"""
        def combined_func(r, A, lam, sigma, alpha):
            exp_term = alpha * np.exp(-lam * r)
            gauss_term = (1 - alpha) * np.exp(-r**2 / (2 * sigma**2))
            return A * (exp_term + gauss_term)
        
        try:
            # Initial guess
            A_init = np.max(magnitudes)
            lam_init = 1.0 / np.median(distances)
            sigma_init = np.std(distances)
            alpha_init = 0.5
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = optimize.curve_fit(
                    combined_func, distances, magnitudes,
                    p0=[A_init, lam_init, sigma_init, alpha_init],
                    bounds=([0, 0, 1e-6, 0], [np.inf, np.inf, np.inf, 1]),
                    maxfev=2000
                )
            
            A_fit, lam_fit, sigma_fit, alpha_fit = popt
            
            # Calculate R²
            y_pred = combined_func(distances, A_fit, lam_fit, sigma_fit, alpha_fit)
            r_squared = 1 - np.sum((magnitudes - y_pred)**2) / np.sum((magnitudes - np.mean(magnitudes))**2)
            r_squared = max(0, r_squared)
            
            return {'amplitude': A_fit, 'r_squared': r_squared}
            
        except Exception:
            logger.debug("Combined fit failed, using default parameters")
            return {'amplitude': 1.0, 'r_squared': 0.0}
    
    def _find_optimal_cutoff_radius(self, distances: np.ndarray, 
                                   interference_matrix: np.ndarray,
                                   decay_fit: DecayFitResults) -> float:
        """
        Find optimal cutoff radius where interactions become negligible.
        
        CUTOFF STRATEGY:
        Uses the negligible_threshold_percentile (default 99th percentile) to find
        the radius beyond which only 1% of interactions are significant.
        """
        # Get non-diagonal interference magnitudes
        mask = ~np.eye(interference_matrix.shape[0], dtype=bool)
        interference_magnitudes = np.abs(interference_matrix[mask])
        distance_values = distances[mask]
        
        # Find threshold for negligible interactions (1st percentile = bottom 1%)
        negligible_threshold = np.percentile(
            interference_magnitudes, self.negligible_threshold_percentile
        )
        
        # Find distances where interference drops below threshold
        negligible_mask = interference_magnitudes <= negligible_threshold
        significant_mask = ~negligible_mask
        
        if np.sum(significant_mask) == 0:
            logger.warning("No significant interactions found, using conservative cutoff")
            return np.percentile(distance_values, 90)
        
        # Find 99th percentile of distances for significant interactions
        significant_distances = distance_values[significant_mask]
        cutoff_radius = np.percentile(significant_distances, 99)
        
        # Ensure minimum cutoff for numerical stability
        min_cutoff = np.percentile(distance_values, 10)
        cutoff_radius = max(cutoff_radius, min_cutoff)
        
        logger.debug(f"Negligible threshold: {negligible_threshold:.2e}")
        logger.debug(f"Significant interactions: {np.sum(significant_mask)}/{len(significant_mask)}")
        logger.debug(f"Cutoff radius (99th percentile of significant): {cutoff_radius:.4f}")
        
        return cutoff_radius
    
    def _validate_field_locality(self, distances: np.ndarray, 
                                interference_matrix: np.ndarray,
                                cutoff_radius: float) -> Dict[str, float]:
        """
        Validate that field interference follows local distance relationships.
        
        VALIDATION METRICS:
        - Correlation strength between distance and interference
        - Fraction of interactions within cutoff
        - Validation of field locality assumption
        """
        mask = ~np.eye(interference_matrix.shape[0], dtype=bool)
        distance_values = distances[mask]
        interference_magnitudes = np.abs(interference_matrix[mask])
        
        # Compute correlation between distance and interference
        valid_mask = (distance_values > 1e-10) & (interference_magnitudes > 1e-10)
        if np.sum(valid_mask) > 10:
            correlation_coef = np.corrcoef(
                distance_values[valid_mask], 
                interference_magnitudes[valid_mask]
            )[0, 1]
            correlation_strength = abs(correlation_coef) if not np.isnan(correlation_coef) else 0.0
        else:
            correlation_strength = 0.0
        
        # Fraction of interactions within cutoff
        within_cutoff_fraction = np.sum(distance_values <= cutoff_radius) / len(distance_values)
        
        # Validate field locality (strong correlation indicates good locality)
        locality_valid = correlation_strength >= self.field_locality_threshold
        
        return {
            'correlation_strength': correlation_strength,
            'within_cutoff_fraction': within_cutoff_fraction,
            'locality_valid': locality_valid
        }
    
    def _analyze_neighbor_statistics(self, distances: np.ndarray, 
                                   interference_matrix: np.ndarray,
                                   cutoff_radius: float) -> Dict[str, Any]:
        """Analyze neighbor statistics at the chosen cutoff radius."""
        n_charges = distances.shape[0]
        neighbors_per_charge = []
        
        for i in range(n_charges):
            # Count neighbors within cutoff (excluding self)
            neighbor_distances = distances[i, :]
            neighbors_in_cutoff = np.sum((neighbor_distances <= cutoff_radius) & 
                                        (neighbor_distances > 1e-10))
            neighbors_per_charge.append(neighbors_in_cutoff)
        
        neighbors_per_charge = np.array(neighbors_per_charge)
        
        return {
            'mean_neighbors': np.mean(neighbors_per_charge),
            'max_neighbors': np.max(neighbors_per_charge),
            'std_neighbors': np.std(neighbors_per_charge),
            'median_neighbors': np.median(neighbors_per_charge)
        }
    
    def _compute_field_coherence_radius(self, sample_biographies: List, 
                                       distances: np.ndarray) -> float:
        """
        Compute field coherence radius based on breathing pattern synchronization.
        
        COHERENCE ANALYSIS:
        Finds the radius within which charges show coherent breathing patterns,
        indicating natural field structures.
        """
        n_sample = len(sample_biographies)
        
        # Extract breathing coherence values
        coherence_values = []
        for bio in sample_biographies:
            coherence_values.append(bio.breathing_coherence)
        
        coherence_values = np.array(coherence_values)
        
        if len(coherence_values) < 2:
            return distances.max() * 0.5
        
        # Find radius where coherence correlation drops significantly
        coherence_correlations = []
        distance_thresholds = np.percentile(distances[distances > 0], 
                                          np.linspace(10, 90, 9))
        
        for threshold in distance_thresholds:
            # Find pairs within this distance threshold
            close_pairs = []
            for i in range(n_sample):
                for j in range(i + 1, n_sample):
                    if distances[i, j] <= threshold and distances[i, j] > 1e-10:
                        coherence_diff = abs(coherence_values[i] - coherence_values[j])
                        close_pairs.append(coherence_diff)
            
            if len(close_pairs) > 5:
                # Lower coherence difference indicates higher correlation
                avg_coherence_diff = np.mean(close_pairs)
                coherence_correlations.append(1.0 / (1.0 + avg_coherence_diff))
            else:
                coherence_correlations.append(0.0)
        
        # Find radius where coherence correlation drops below threshold
        coherence_correlations = np.array(coherence_correlations)
        high_coherence_mask = coherence_correlations >= 0.7
        
        if np.any(high_coherence_mask):
            coherence_radius = distance_thresholds[high_coherence_mask][-1]
        else:
            coherence_radius = np.median(distance_thresholds)
        
        return coherence_radius
    
    def _compute_sparsity_at_cutoff(self, distances: np.ndarray, 
                                   cutoff_radius: float) -> float:
        """Compute sparsity ratio achieved at the chosen cutoff radius."""
        mask = ~np.eye(distances.shape[0], dtype=bool)
        distance_values = distances[mask]
        
        # Fraction of interactions beyond cutoff (sparse)
        beyond_cutoff = np.sum(distance_values > cutoff_radius)
        total_interactions = len(distance_values)
        
        sparsity_ratio = beyond_cutoff / total_interactions if total_interactions > 0 else 0.0
        return sparsity_ratio
    
    def _generate_cutoff_justification(self, decay_fit: DecayFitResults, 
                                      locality_validation: Dict[str, Any]) -> str:
        """Generate mathematical justification for the chosen cutoff."""
        justification_parts = []
        
        # Field theory decay model
        if decay_fit.best_model == 'exponential':
            justification_parts.append(
                f"Exponential decay model (R²={decay_fit.exponential_r_squared:.3f}) "
                f"consistent with character layer persistence"
            )
        elif decay_fit.best_model == 'gaussian':
            justification_parts.append(
                f"Gaussian decay model (R²={decay_fit.gaussian_r_squared:.3f}) "
                f"consistent with vivid layer persistence"
            )
        else:
            justification_parts.append(
                f"Combined decay model (R²={decay_fit.combined_r_squared:.3f}) "
                f"consistent with dual-layer persistence structure"
            )
        
        # Distance correlation validation
        if locality_validation['locality_valid']:
            justification_parts.append(
                f"Strong distance-interference correlation (r={locality_validation['correlation_strength']:.3f}) "
                f"validates field locality assumption"
            )
        else:
            justification_parts.append(
                f"Moderate distance-interference correlation (r={locality_validation['correlation_strength']:.3f}) "
                f"indicates complex field structure"
            )
        
        # Negligible threshold
        justification_parts.append(
            f"99th percentile cutoff ensures <1% significant interactions neglected"
        )
        
        return "; ".join(justification_parts)
    
    def _log_analysis_results(self, analysis: FieldCutoffAnalysis, 
                             decay_fit: DecayFitResults,
                             locality_validation: Dict[str, Any]) -> None:
        """Comprehensive logging of cutoff analysis results."""
        logger.info(f"🎯 Field cutoff analysis complete:")
        logger.info(f"   Optimal cutoff radius: {analysis.optimal_cutoff_radius:.4f}")
        logger.info(f"   Effective embedding dimension: {analysis.embedding_effective_dimension}")
        logger.info(f"   Field locality strength: {analysis.field_locality_strength:.3f}")
        logger.info(f"   Mean neighbors per charge: {analysis.mean_significant_neighbors:.1f}")
        logger.info(f"   Max neighbors per charge: {analysis.max_significant_neighbors}")
        logger.info(f"   Sparsity at cutoff: {analysis.sparsity_at_cutoff:.1%}")
        
        logger.info(f"📈 Decay curve analysis:")
        logger.info(f"   Best model: {decay_fit.best_model}")
        logger.info(f"   Exponential R²: {decay_fit.exponential_r_squared:.3f}")
        logger.info(f"   Gaussian R²: {decay_fit.gaussian_r_squared:.3f}")
        logger.info(f"   Combined R²: {decay_fit.combined_r_squared:.3f}")
        
        logger.info(f"✅ Field locality validation:")
        logger.info(f"   Distance correlation: {locality_validation['correlation_strength']:.3f}")
        logger.info(f"   Locality valid: {locality_validation['locality_valid']}")
        logger.info(f"   Interactions within cutoff: {locality_validation['within_cutoff_fraction']:.1%}")
        
        logger.info(f"🧮 Mathematical justification:")
        logger.info(f"   {analysis.cutoff_justification}")


if __name__ == "__main__":
    """Test field cutoff analyzer with sample data."""
    print("FieldCutoffAnalyzer ready for O(N) cutoff radius discovery")
    print("Uses field theory decay properties to find natural cutoff for cell lists algorithm")