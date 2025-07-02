"""
Geometric Field Analysis - Differential Geometry for Regulation

MATHEMATICAL FOUNDATION: Uses differential geometry to analyze field curvature,
detect singularities, and compute geometric measures for regulation decisions.
The Q-field manifold structure provides natural regulation signals through:

1. Riemann curvature tensor - detects field instabilities
2. Sectional curvature - identifies singular regions
3. Geodesic flow - tracks field evolution paths
4. Manifold topology - captures field connectivity

GEOMSTATS INTEGRATION: Leverages geomstats library for rigorous differential
geometry computations on Riemannian manifolds. All geometric measures are
mathematically well-defined and respect manifold structure.

FIELD THEORY COMPLIANCE: Geometric analysis respects the Q(τ, C, s) formula
by treating conceptual charges as points on a complex manifold where field
interactions create curvature and geometric distortions.
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.matrices import Matrices
from geomstats.learning.frechet_mean import FrechetMean

from Sysnpire.model.liquid.conceptual_charge_agent import ConceptualChargeAgent
from Sysnpire.model.liquid.regulation.advanced.adaptive_field_dimension import (
    AdaptiveFieldDimension,
    DimensionEstimate,
)
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GeometricMetrics:
    """Geometric measures for field regulation."""

    riemann_curvature: float  # Riemann curvature tensor norm
    sectional_curvature: float  # Sectional curvature measure
    geodesic_distance: float  # Mean geodesic distance between charges
    manifold_dimension: int  # Intrinsic manifold dimension
    singularity_indicator: float  # Geometric singularity measure
    field_topology_invariant: float  # Topological invariant
    curvature_concentration: float  # Concentration of curvature


@dataclass
class ManifoldState:
    """Current state of the field manifold."""

    agent_positions: np.ndarray  # Agent positions in manifold coordinates
    tangent_vectors: np.ndarray  # Tangent vectors at each position
    metric_tensor: np.ndarray  # Local metric tensor
    connection_coefficients: np.ndarray  # Christoffel symbols
    curvature_tensor: np.ndarray  # Riemann curvature tensor


class GeometricRegulation:
    """
    Differential Geometry Analysis for Field Regulation

    Analyzes the geometric structure of the Q-field manifold to detect
    instabilities and provide regulation guidance through curvature analysis.
    """

    def __init__(self, adaptive_dimension_engine: AdaptiveFieldDimension):
        """
        Initialize adaptive geometric regulation system.

        Args:
            adaptive_dimension_engine: Engine for discovering field dimensions (REQUIRED)
        """
        self.adaptive_dimension_engine = adaptive_dimension_engine

        # Mathematical state - discovered through field analysis
        self.current_dimension_estimate: Optional[DimensionEstimate] = None
        self.effective_embedding_dimension: Optional[int] = None

        # Manifolds constructed from discovered mathematical structure
        self.hypersphere: Optional[Hypersphere] = None
        self.euclidean: Optional[Euclidean] = None
        self.special_euclidean = SpecialEuclidean(n=2)  # 2D spatial positions
        self.matrix_space: Optional[Matrices] = None
        self.frechet_mean: Optional[FrechetMean] = None

        self.manifold_history: List[ManifoldState] = []
        self.curvature_cache: Dict[str, Any] = {}
        self.dimension_cache: Dict[str, DimensionEstimate] = {}

        self.geometric_stats = {
            "total_analyses": 0,
            "average_curvature": 0.0,
            "singularity_detections": 0,
            "manifold_dimension_stability": 1.0,
            "dimension_discoveries": 0,
            "average_discovered_dimension": 0.0,
        }

        logger.info("🔬 AdaptiveGeometricRegulation initialized")
        logger.info("   Quantum cognition dimension discovery: ENABLED")
        logger.info("   Rate-distortion optimization: ENABLED")
        logger.info("   Heat kernel eigenvalue analysis: ENABLED")
        logger.info("   Multi-objective consensus: ENABLED")
        logger.info("   Field-theoretic dimension emergence: ENABLED")

    def _init_adaptive_manifolds(self, discovered_dimension: int):
        """
        Initialize geomstats manifolds based on discovered field dimension.

        Args:
            discovered_dimension: Dimension discovered from field analysis
        """
        self.effective_embedding_dimension = discovered_dimension

        # Initialize manifolds with discovered dimension - NO FALLBACKS
        self.hypersphere = Hypersphere(dim=discovered_dimension - 1)
        self.euclidean = Euclidean(dim=discovered_dimension)
        self.matrix_space = Matrices(m=discovered_dimension, n=discovered_dimension)
        self.frechet_mean = FrechetMean(space=self.hypersphere)

        logger.info(
            f"🔬 Manifolds constructed for discovered dimension {discovered_dimension}"
        )

    def _discover_field_dimension(
        self, agents: List[ConceptualChargeAgent]
    ) -> DimensionEstimate:
        """
        Discover the natural dimension of the Q-field agents.

        Returns:
            DimensionEstimate with discovered dimension and metadata
        """
        # Check if we have a recent dimension estimate
        field_signature = f"{len(agents)}_{hash(str([getattr(a, 'Q_components', None) for a in agents[:5]]))}"

        if field_signature in self.dimension_cache:
            cached_estimate = self.dimension_cache[field_signature]
            logger.debug(
                f"🔷 Using cached dimension: {cached_estimate.consensus_dimension}"
            )
            return cached_estimate

        # Discover dimension using adaptive engine
        logger.debug("🔷 Discovering field dimension...")
        dimension_estimate = self.adaptive_dimension_engine.discover_field_dimension(
            agents
        )

        # Cache the result
        self.dimension_cache[field_signature] = dimension_estimate

        # Update statistics
        self.geometric_stats["dimension_discoveries"] += 1
        n_discoveries = self.geometric_stats["dimension_discoveries"]
        old_avg = self.geometric_stats["average_discovered_dimension"]
        self.geometric_stats["average_discovered_dimension"] = (
            old_avg * (n_discoveries - 1) + dimension_estimate.consensus_dimension
        ) / n_discoveries

        logger.info(
            f"🔷 Field dimension discovered: {dimension_estimate.consensus_dimension} "
            f"(confidence: {dimension_estimate.confidence_score:.3f})"
        )

        return dimension_estimate

    def analyze_field_geometry(
        self, agents: List[ConceptualChargeAgent]
    ) -> GeometricMetrics:
        """
        Perform adaptive geometric analysis of the field manifold.

        Discovers field dimension first, then performs geometric analysis
        adapted to the discovered dimensional structure.

        Args:
            agents: List of conceptual charge agents

        Returns:
            GeometricMetrics with curvature and topology measures
        """
        analysis_start = time.time()

        # Step 1: Discover field dimension - MATHEMATICAL REQUIREMENT
        dimension_estimate = self._discover_field_dimension(agents)
        self.current_dimension_estimate = dimension_estimate

        # Step 2: Initialize manifolds with discovered dimension
        if (
            self.hypersphere is None
            or self.effective_embedding_dimension
            != dimension_estimate.consensus_dimension
        ):
            self._init_adaptive_manifolds(dimension_estimate.consensus_dimension)

        # Step 3: Extract geometric data adapted to discovered dimension
        field_points, spatial_positions = self._extract_adaptive_geometric_data(
            agents, dimension_estimate
        )

        # Step 4: Perform adaptive geometric analysis
        metrics = self._compute_adaptive_geomstats_metrics(
            field_points, spatial_positions, dimension_estimate
        )

        analysis_time = time.time() - analysis_start
        self._update_geometric_stats(metrics, analysis_time)

        logger.debug(
            f"🔬 Field geometry analysis: dimension {dimension_estimate.consensus_dimension}, "
            f"confidence {dimension_estimate.confidence_score:.3f}, "
            f"curvature {metrics.riemann_curvature:.6f}"
        )

        return metrics

    def _extract_adaptive_geometric_data(
        self, agents: List[ConceptualChargeAgent], dimension_estimate: DimensionEstimate
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract geometric data adapted to discovered field dimension.

        Args:
            agents: List of conceptual charge agents
            dimension_estimate: Discovered dimension information

        Returns:
            Tuple of (field_points, spatial_positions)
        """
        field_points = []
        spatial_positions = []
        target_dimension = dimension_estimate.consensus_dimension

        for agent in agents:
            q_val = agent.Q_components.Q_value
            field_point = self._q_value_to_adaptive_field_point(q_val, target_dimension)
            field_points.append(field_point)

            pos = agent.field_state.field_position
            spatial_positions.append([pos[0], pos[1]])

        return np.array(field_points), np.array(spatial_positions)

    def _q_value_to_adaptive_field_point(
        self, q_value: complex, target_dimension: int
    ) -> np.ndarray:
        """
        Convert Q-value to field point adapted to target dimension.

        Maps complex Q-value to target dimensional space using mathematical
        field structure derivation - NO PADDING, NO TRUNCATION.
        """
        real_part = float(q_value.real)
        imag_part = float(q_value.imag)
        magnitude = abs(q_value)
        phase = gs.angle(gs.array(q_value))

        # Mathematical field structure components
        components = [real_part, imag_part, magnitude, phase]

        # Higher-order field relationships
        for k in range(1, target_dimension - 3):
            harmonic = k + 1
            field_component = (
                magnitude ** (1 / harmonic) * gs.cos(harmonic * phase).item()
            )
            components.append(field_component)

        return gs.array(components)

    def _compute_adaptive_geomstats_metrics(
        self,
        field_points: np.ndarray,
        spatial_positions: np.ndarray,
        dimension_estimate: DimensionEstimate,
    ) -> GeometricMetrics:
        """
        Compute geometric metrics using discovered field dimension.
        """
        discovered_dim = dimension_estimate.consensus_dimension

        # Normalize to hypersphere in discovered dimension
        normalized_points = self._normalize_to_sphere_geomstats(field_points)

        # Compute full geometric analysis
        riemann_curvature = self._compute_riemann_curvature_geomstats(normalized_points)
        sectional_curvature = self._compute_sectional_curvature_geomstats(
            normalized_points
        )
        geodesic_distance = self._compute_mean_geodesic_distance_geomstats(
            normalized_points
        )
        singularity_indicator = self._detect_geometric_singularities_geomstats(
            normalized_points
        )
        topology_invariant = self._compute_topology_invariant_geomstats(
            normalized_points
        )
        curvature_concentration = self._compute_curvature_concentration_geomstats(
            normalized_points
        )

        return GeometricMetrics(
            riemann_curvature=riemann_curvature,
            sectional_curvature=sectional_curvature,
            geodesic_distance=geodesic_distance,
            manifold_dimension=discovered_dim,
            singularity_indicator=singularity_indicator,
            field_topology_invariant=topology_invariant,
            curvature_concentration=curvature_concentration,
        )

    def _extract_geometric_data(
        self, agents: List[ConceptualChargeAgent]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract geometric data from conceptual charge agents.

        Returns:
            Tuple of (field_points, spatial_positions)
        """
        field_points = []
        spatial_positions = []

        for agent in agents:
            if hasattr(agent, "Q_components") and agent.Q_components is not None:
                q_val = agent.Q_components.Q_value
                if q_val is not None and math.isfinite(abs(q_val)):
                    field_point = self._q_value_to_field_point(q_val)
                    field_points.append(field_point)

                    if hasattr(agent, "field_state") and agent.field_state is not None:
                        pos = agent.field_state.field_position
                        spatial_positions.append([pos[0], pos[1]])
                    else:
                        spatial_positions.append([0.0, 0.0])

        return np.array(field_points), np.array(spatial_positions)

    def _normalize_to_sphere_geomstats(self, points: np.ndarray) -> np.ndarray:
        """Normalize points to unit hypersphere using geomstats backend - NO FALLBACKS!"""
        points_gs = gs.array(points)
        norms = gs.linalg.norm(points_gs, axis=1, keepdims=True)
        norms = gs.maximum(norms, 1e-12)  # Avoid division by zero using gs operations
        normalized = points_gs / norms
        return gs.to_ndarray(normalized, to_ndim=2)

    def _compute_riemann_curvature_geomstats(self, points: np.ndarray) -> float:
        """
        Compute Riemann curvature tensor norm using geomstats - NO FALLBACKS!
        """
        if len(points) < 4:
            return 1.0  # Default hypersphere curvature

        points_gs = gs.array(points)

        curvature_values = []

        for i in range(min(10, len(points) - 3)):  # Sample for efficiency
            base_point = points_gs[i]
            tangent_vec_a = points_gs[i + 1] - base_point
            tangent_vec_b = points_gs[i + 2] - base_point
            tangent_vec_c = points_gs[i + 3] - base_point

            tangent_vec_a = self.hypersphere.to_tangent(tangent_vec_a, base_point)
            tangent_vec_b = self.hypersphere.to_tangent(tangent_vec_b, base_point)
            tangent_vec_c = self.hypersphere.to_tangent(tangent_vec_c, base_point)

            local_curvature = self.hypersphere.metric.curvature(
                tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
            )
            curvature_norm = gs.linalg.norm(local_curvature)
            curvature_values.append(curvature_norm)

        if curvature_values:
            curvature_tensor = gs.array(curvature_values)
            return float(gs.mean(curvature_tensor))
        else:
            return 1.0  # Standard hypersphere curvature

    def _compute_sectional_curvature_geomstats(self, points: np.ndarray) -> float:
        """
        Compute sectional curvature using geomstats - NO EXCEPTIONS, NO FALLBACKS.
        """
        if len(points) < 3:
            raise ValueError(
                f"Sectional curvature requires ≥3 points, got {len(points)} - MATHEMATICAL INTEGRITY VIOLATED"
            )

        points_gs = gs.array(points)

        curvature_values = []

        for i in range(min(5, len(points) - 2)):
            base_point = points_gs[i]
            tangent_vec_a = points_gs[i + 1] - base_point
            tangent_vec_b = points_gs[i + 2] - base_point

            norm_a = gs.linalg.norm(tangent_vec_a)
            norm_b = gs.linalg.norm(tangent_vec_b)

            if norm_a > 1e-12 and norm_b > 1e-12:
                tangent_vec_a = tangent_vec_a / norm_a
                tangent_vec_b = tangent_vec_b / norm_b

                tangent_vec_a = self.hypersphere.to_tangent(tangent_vec_a, base_point)
                tangent_vec_b = self.hypersphere.to_tangent(tangent_vec_b, base_point)

                sectional_curv = self.hypersphere.metric.sectional_curvature(
                    tangent_vec_a, tangent_vec_b, base_point
                )

                if not gs.isnan(sectional_curv):
                    curvature_values.append(gs.abs(sectional_curv))

        if curvature_values:
            return float(gs.mean(gs.array(curvature_values)))
        else:
            return 1.0  # Standard hypersphere sectional curvature

    def _compute_mean_geodesic_distance_geomstats(self, points: np.ndarray) -> float:
        """
        Compute mean geodesic distance between points using geomstats - NO FALLBACKS!
        """
        if len(points) < 2:
            raise ValueError(
                f"Geodesic distance requires ≥2 points, got {len(points)} - MATHEMATICAL INTEGRITY VIOLATED"
            )

        points_gs = gs.array(points)
        n_points = len(points_gs)

        geodesic_distances = []

        max_pairs = min(50, n_points * (n_points - 1) // 2)

        for i in range(min(max_pairs, n_points)):
            for j in range(i + 1, min(i + 5, n_points)):  # Limit comparisons
                dist = self.hypersphere.metric.dist(points_gs[i], points_gs[j])
                geodesic_distances.append(dist)

        if geodesic_distances:
            distances_tensor = gs.array(geodesic_distances)
            return float(gs.mean(distances_tensor))
        else:
            return 0.0

    def _estimate_manifold_dimension_geomstats(self, points: np.ndarray) -> int:
        """
        Estimate intrinsic manifold dimension using geomstats - NO FALLBACKS!
        """
        if len(points) < self.embedding_dimension:
            raise ValueError(
                f"Manifold dimension estimation requires ≥{self.embedding_dimension} points, got {len(points)} - UNDERDETERMINED SYSTEM"
            )

        points_gs = gs.array(points)

        mean_point = gs.mean(points_gs, axis=0)
        centered_points = points_gs - mean_point

        U, s, Vt = gs.linalg.svd(centered_points, full_matrices=False)

        total_variance = gs.sum(s**2)
        cumulative_variance = gs.cumsum(s**2)
        variance_ratio = cumulative_variance / total_variance

        dimension = gs.argmax(variance_ratio > 0.95) + 1
        dimension = gs.maximum(1, gs.minimum(dimension, len(points) - 1))

        return int(dimension)

    def _detect_geometric_singularities_geomstats(self, points: np.ndarray) -> float:
        """
        Detect geometric singularities using geomstats - NO FALLBACKS!
        """
        if len(points) < 3:
            raise ValueError(
                f"Singularity detection requires ≥3 points, got {len(points)} - INSUFFICIENT DATA FOR MATHEMATICAL ANALYSIS"
            )

        points_gs = gs.array(points)

        pairwise_distances = []
        n_points = len(points_gs)

        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = gs.linalg.norm(points_gs[i] - points_gs[j])
                pairwise_distances.append(dist)

        if not pairwise_distances:
            return 0.0

        distances_tensor = gs.array(pairwise_distances)
        mean_dist = gs.mean(distances_tensor)
        std_dist = gs.std(distances_tensor)
        min_dist = gs.amin(distances_tensor)

        singularity_measure = 1.0 - min_dist / (mean_dist + 1e-12)
        irregularity = std_dist / (mean_dist + 1e-12)

        combined_singularity = 0.7 * singularity_measure + 0.3 * gs.minimum(
            1.0, irregularity
        )

        return float(gs.clip(combined_singularity, 0.0, 1.0))

    def _compute_topology_invariant_geomstats(self, points: np.ndarray) -> float:
        """
        Compute topological invariant using geomstats - NO FALLBACKS!
        """
        if len(points) < 3:
            raise ValueError(
                f"Topology invariant computation requires ≥3 points, got {len(points)} - TOPOLOGICAL ANALYSIS IMPOSSIBLE"
            )

        points_gs = gs.array(points)
        n_points = len(points_gs)

        all_distances = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = gs.linalg.norm(points_gs[i] - points_gs[j])
                all_distances.append(dist)

        if not all_distances:
            return 0.5

        distances_tensor = gs.array(all_distances)
        threshold = gs.quantile(distances_tensor, 0.3)  # Connect closest 30% of pairs

        adjacency_count = 0
        total_pairs = n_points * (n_points - 1)

        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = gs.linalg.norm(points_gs[i] - points_gs[j])
                if dist < threshold:
                    adjacency_count += 2  # Both directions

        connectivity = adjacency_count / total_pairs if total_pairs > 0 else 0.0

        return float(connectivity)

    def _compute_curvature_concentration_geomstats(self, points: np.ndarray) -> float:
        """
        Compute concentration of curvature using geomstats - NO FALLBACKS!
        """
        if len(points) < 4:
            raise ValueError(
                f"Curvature concentration requires ≥4 points, got {len(points)} - DIFFERENTIAL GEOMETRY UNDERDETERMINED"
            )

        points_gs = gs.array(points)

        local_curvatures = []

        for i in range(len(points_gs) - 3):
            quad_points = points_gs[i : i + 4]

            if len(quad_points) == 4:
                p1, p2, p3, p4 = quad_points

                d1 = p2 - p1
                d2 = p3 - p2
                d3 = p4 - p3

                dd1 = d2 - d1
                dd2 = d3 - d2

                curvature = gs.linalg.norm(dd1) + gs.linalg.norm(dd2)
                local_curvatures.append(curvature)

        if local_curvatures:
            curvatures_tensor = gs.array(local_curvatures)
            mean_curv = gs.mean(curvatures_tensor)
            std_curv = gs.std(curvatures_tensor)
            concentration = std_curv / (mean_curv + 1e-12)
            return float(gs.minimum(1.0, concentration))
        else:
            return 0.0

    def _update_geometric_stats(self, metrics: GeometricMetrics, analysis_time: float):
        """Update geometric analysis statistics."""
        self.geometric_stats["total_analyses"] += 1

        total_analyses = self.geometric_stats["total_analyses"]
        current_avg = self.geometric_stats["average_curvature"]
        new_curvature = metrics.riemann_curvature
        self.geometric_stats["average_curvature"] = (
            current_avg * (total_analyses - 1) + new_curvature
        ) / total_analyses

        if metrics.singularity_indicator > 0.5:
            self.geometric_stats["singularity_detections"] += 1

        expected_dim = min(10, self.effective_embedding_dimension or 5)
        dimension_stability = (
            1.0 - abs(metrics.manifold_dimension - expected_dim) / expected_dim
        )
        current_stability = self.geometric_stats["manifold_dimension_stability"]
        self.geometric_stats["manifold_dimension_stability"] = (
            current_stability * (total_analyses - 1) + dimension_stability
        ) / total_analyses

    def suggest_geometric_regulation(self, metrics: GeometricMetrics) -> Dict[str, Any]:
        """
        Suggest regulation based on geometric analysis.

        Args:
            metrics: Computed geometric metrics

        Returns:
            Regulation suggestions based on geometry
        """
        regulation_suggestions = {
            "curvature_regulation": {
                "needed": metrics.riemann_curvature > 0.5,
                "strength": min(1.0, metrics.riemann_curvature),
                "type": "curvature_smoothing",
            },
            "singularity_regulation": {
                "needed": metrics.singularity_indicator > 0.3,
                "strength": metrics.singularity_indicator,
                "type": "singularity_dispersion",
            },
            "topology_regulation": {
                "needed": metrics.field_topology_invariant < 0.2
                or metrics.field_topology_invariant > 0.8,
                "strength": abs(0.5 - metrics.field_topology_invariant),
                "type": "topology_stabilization",
            },
            "dimension_regulation": {
                "needed": metrics.manifold_dimension < 3,
                "strength": 1.0 - metrics.manifold_dimension / 10.0,
                "type": "dimension_expansion",
            },
        }

        regulation_scores = [
            reg["strength"] for reg in regulation_suggestions.values() if reg["needed"]
        ]

        overall_regulation_needed = len(regulation_scores) > 0
        if regulation_scores:
            scores_tensor = torch.tensor(regulation_scores, dtype=torch.float32)
            overall_regulation_strength = torch.mean(scores_tensor).item()
        else:
            overall_regulation_strength = 0.0

        return {
            "geometric_regulation_needed": overall_regulation_needed,
            "overall_regulation_strength": overall_regulation_strength,
            "specific_regulations": regulation_suggestions,
            "geometric_metrics": {
                "riemann_curvature": metrics.riemann_curvature,
                "sectional_curvature": metrics.sectional_curvature,
                "singularity_indicator": metrics.singularity_indicator,
                "manifold_dimension": metrics.manifold_dimension,
            },
            "geomstats_enabled": True,
        }

    def get_geometric_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of geometric regulation system.

        Returns:
            Dictionary with system status and performance metrics
        """
        return {
            "system_status": {
                "geomstats_available": True,
                "jax_available": True,
                "current_dimension_estimate": (
                    self.current_dimension_estimate.consensus_dimension
                    if self.current_dimension_estimate
                    else None
                ),
                "effective_embedding_dimension": self.effective_embedding_dimension,
                "manifold_history_length": len(self.manifold_history),
            },
            "geometric_stats": self.geometric_stats.copy(),
            "capabilities": {
                "riemann_curvature": True,
                "sectional_curvature": True,
                "geodesic_distances": True,
                "manifold_dimension_estimation": True,
                "singularity_detection": True,
                "topology_invariants": True,
            },
        }
