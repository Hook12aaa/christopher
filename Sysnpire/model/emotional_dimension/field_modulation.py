"""
Emotional Field Modulation - Transform Emotion into Field Effects

MATHEMATICAL FOUNDATION (README.md Section 3.1.3.3.3):
g^E_μν(x) = g_μν(x) · [1 + κ_E · E^trajectory(x) · cos(θ_E,g)]

GEOMETRIC EFFECTS:
1. Path Distortion: Emotional content creates preferred pathways for meaning propagation
2. Distance Warping: Emotional similarity/dissimilarity affects semantic distances  
3. Curvature Modulation: Strong emotions create local curvature in semantic space

This module implements emotional field effects on semantic geometry through
metric tensor modulation, field effect calculation, and geometric distortion management.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MetricTensor:
    """Metric tensor representation for geometric calculations."""
    components: np.ndarray  # [μ, ν] metric tensor components
    dimension: int
    determinant: float
    is_riemannian: bool = True


@dataclass 
class GeometricDistortion:
    """Geometric distortion caused by emotional fields."""
    distortion_type: str  # 'curvature', 'distance', 'path'
    magnitude: float
    direction_vector: np.ndarray
    affected_region: Tuple[np.ndarray, float]  # (center, radius)
    emotional_source: complex


class EmotionalFieldModulator:
    """
    Apply emotional field effects to semantic fields through geometric modulation.
    
    MATHEMATICAL FOUNDATION (README.md Section 3.1.3.3.3):
    Implements metric warping g^E_μν = g_μν · [1 + κ_E · E^trajectory · cos(θ_E,g)]
    where emotional fields create geometric distortions in semantic space.
    
    FIELD EFFECTS:
    - Metric tensor modulation for distance warping
    - Curvature modification for path distortion
    - Local field effects for emotional hot spots
    """
    
    def __init__(self,
                 emotional_coupling_strength: float = 0.3,
                 distortion_radius: float = 1.0,
                 max_curvature: float = 2.0):
        """
        Initialize emotional field modulator.
        
        Args:
            emotional_coupling_strength: κ_E coupling strength parameter
            distortion_radius: Radius of emotional field effects
            max_curvature: Maximum allowed curvature modification
        """
        self.emotional_coupling_strength = emotional_coupling_strength
        self.distortion_radius = distortion_radius
        self.max_curvature = max_curvature
        
        logger.info(f"Initialized EmotionalFieldModulator: κ_E={emotional_coupling_strength}, radius={distortion_radius}")
    
    def apply_emotional_field_effects(self,
                                    semantic_field: np.ndarray,
                                    emotional_trajectory: complex,
                                    base_metric: Optional[MetricTensor] = None,
                                    position: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply emotional field effects to semantic field through geometric modulation.
        
        FIELD MODULATION PROCESS:
        1. Create base metric tensor if not provided
        2. Calculate emotional field gradient and direction
        3. Apply metric tensor modulation
        4. Compute geometric distortions
        5. Transform semantic field through warped geometry
        
        Args:
            semantic_field: Base semantic field vector
            emotional_trajectory: Complex emotional field E^trajectory(τ, s)
            base_metric: Optional base metric tensor
            position: Optional position in semantic space
            
        Returns:
            Tuple of (modulated_field, field_analysis)
        """
        try:
            # Create base metric if not provided
            if base_metric is None:
                base_metric = self._create_euclidean_metric(len(semantic_field))
            
            # Calculate emotional field properties
            emotional_magnitude = abs(emotional_trajectory)
            emotional_phase = np.angle(emotional_trajectory)
            
            # Compute emotional field gradient
            emotional_gradient = self._compute_emotional_gradient(
                emotional_trajectory, semantic_field, position
            )
            
            # Apply metric tensor modulation
            warped_metric = self._apply_metric_warping(
                base_metric, emotional_trajectory, emotional_gradient
            )
            
            # Calculate geometric distortions
            distortions = self._calculate_geometric_distortions(
                emotional_trajectory, semantic_field, warped_metric
            )
            
            # Apply field transformation
            modulated_field = self._apply_field_transformation(
                semantic_field, warped_metric, distortions
            )
            
            # Create field analysis
            field_analysis = {
                'emotional_magnitude': emotional_magnitude,
                'emotional_phase': emotional_phase,
                'metric_determinant': warped_metric.determinant,
                'metric_warping_factor': warped_metric.determinant / base_metric.determinant,
                'num_distortions': len(distortions),
                'distortion_types': [d.distortion_type for d in distortions],
                'max_distortion': max([d.magnitude for d in distortions]) if distortions else 0.0,
                'field_modification': np.linalg.norm(modulated_field - semantic_field),
                'geometric_effects': self._analyze_geometric_effects(distortions)
            }
            
            logger.debug(f"Applied emotional field effects: warping={field_analysis['metric_warping_factor']:.3f}, distortions={len(distortions)}")
            return modulated_field, field_analysis
            
        except Exception as e:
            logger.error(f"Emotional field modulation failed: {e}")
            return semantic_field, {'error': str(e), 'field_modification': 0.0}
    
    def _create_euclidean_metric(self, dimension: int) -> MetricTensor:
        """Create Euclidean metric tensor as baseline."""
        components = np.eye(dimension)
        return MetricTensor(
            components=components,
            dimension=dimension,
            determinant=1.0,
            is_riemannian=True
        )
    
    def _compute_emotional_gradient(self,
                                  emotional_trajectory: complex,
                                  semantic_field: np.ndarray,
                                  position: Optional[np.ndarray]) -> np.ndarray:
        """
        Compute emotional field gradient for geometric effects.
        
        GRADIENT CALCULATION:
        Emotional fields create gradients that influence geometric structure
        through direction-dependent metric modifications.
        """
        # Extract emotional characteristics
        emotional_magnitude = abs(emotional_trajectory)
        emotional_phase = np.angle(emotional_trajectory)
        
        # Create gradient based on emotional direction and semantic content
        dimension = len(semantic_field)
        gradient = np.zeros(dimension)
        
        for i in range(dimension):
            # Emotional influence on each dimension
            phase_factor = np.cos(emotional_phase + i * np.pi / dimension)
            semantic_factor = semantic_field[i] / (np.linalg.norm(semantic_field) + 1e-10)
            
            gradient[i] = emotional_magnitude * phase_factor * semantic_factor
        
        # Normalize gradient
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > 0:
            gradient = gradient / gradient_norm
        
        return gradient
    
    def _apply_metric_warping(self,
                            base_metric: MetricTensor,
                            emotional_trajectory: complex,
                            emotional_gradient: np.ndarray) -> MetricTensor:
        """
        Apply emotional warping to metric tensor.
        
        METRIC WARPING FORMULA (README.md Section 3.1.3.3.3):
        g^E_μν(x) = g_μν(x) · [1 + κ_E · E^trajectory(x) · cos(θ_E,g)]
        """
        emotional_magnitude = abs(emotional_trajectory)
        emotional_phase = np.angle(emotional_trajectory)
        
        # Calculate angle between emotional field and metric structure
        # Simplified: use emotional gradient direction
        theta_E_g = emotional_phase  # Simplified angle calculation
        
        # Compute warping factor
        warping_factor = 1.0 + self.emotional_coupling_strength * emotional_magnitude * np.cos(theta_E_g)
        
        # Apply warping to metric components
        warped_components = base_metric.components.copy()
        
        # Apply direction-dependent warping
        for i in range(base_metric.dimension):
            for j in range(base_metric.dimension):
                # Direction-dependent warping based on emotional gradient
                direction_factor = emotional_gradient[i] * emotional_gradient[j]
                local_warping = 1.0 + self.emotional_coupling_strength * emotional_magnitude * direction_factor
                warped_components[i, j] *= local_warping
        
        # Calculate new determinant
        new_determinant = np.linalg.det(warped_components)
        
        return MetricTensor(
            components=warped_components,
            dimension=base_metric.dimension,
            determinant=new_determinant,
            is_riemannian=new_determinant > 0
        )
    
    def _calculate_geometric_distortions(self,
                                       emotional_trajectory: complex,
                                       semantic_field: np.ndarray,
                                       warped_metric: MetricTensor) -> List[GeometricDistortion]:
        """
        Calculate geometric distortions caused by emotional fields.
        
        DISTORTION TYPES:
        1. Curvature distortions from emotional intensity
        2. Distance distortions from emotional valence  
        3. Path distortions from emotional directionality
        """
        distortions = []
        emotional_magnitude = abs(emotional_trajectory)
        emotional_phase = np.angle(emotional_trajectory)
        
        # Curvature distortion from emotional intensity
        if emotional_magnitude > 0.1:
            curvature_magnitude = min(self.max_curvature, emotional_magnitude * 2.0)
            
            # Curvature direction based on emotional gradient
            curvature_direction = np.array([
                np.cos(emotional_phase + i * np.pi / len(semantic_field))
                for i in range(len(semantic_field))
            ])
            curvature_direction = curvature_direction / (np.linalg.norm(curvature_direction) + 1e-10)
            
            curvature_distortion = GeometricDistortion(
                distortion_type='curvature',
                magnitude=curvature_magnitude,
                direction_vector=curvature_direction,
                affected_region=(semantic_field, self.distortion_radius),
                emotional_source=emotional_trajectory
            )
            distortions.append(curvature_distortion)
        
        # Distance distortion from emotional valence
        distance_magnitude = emotional_magnitude * 0.5
        distance_direction = semantic_field / (np.linalg.norm(semantic_field) + 1e-10)
        
        distance_distortion = GeometricDistortion(
            distortion_type='distance',
            magnitude=distance_magnitude,
            direction_vector=distance_direction,
            affected_region=(semantic_field, self.distortion_radius * 0.5),
            emotional_source=emotional_trajectory
        )
        distortions.append(distance_distortion)
        
        # Path distortion from emotional directionality
        if abs(np.sin(emotional_phase)) > 0.1:  # Significant phase component
            path_magnitude = emotional_magnitude * abs(np.sin(emotional_phase))
            
            path_direction = np.array([
                np.sin(emotional_phase + i * np.pi / len(semantic_field))
                for i in range(len(semantic_field))
            ])
            path_direction = path_direction / (np.linalg.norm(path_direction) + 1e-10)
            
            path_distortion = GeometricDistortion(
                distortion_type='path',
                magnitude=path_magnitude,
                direction_vector=path_direction,
                affected_region=(semantic_field, self.distortion_radius * 0.8),
                emotional_source=emotional_trajectory
            )
            distortions.append(path_distortion)
        
        return distortions
    
    def _apply_field_transformation(self,
                                  semantic_field: np.ndarray,
                                  warped_metric: MetricTensor,
                                  distortions: List[GeometricDistortion]) -> np.ndarray:
        """
        Apply geometric transformation to semantic field.
        
        FIELD TRANSFORMATION:
        Uses warped metric and geometric distortions to transform
        semantic field through emotionally-modified geometry.
        """
        transformed_field = semantic_field.copy()
        
        # Apply metric-based transformation
        # Transform field using inverse warped metric
        try:
            metric_inverse = np.linalg.inv(warped_metric.components)
            metric_factor = np.sqrt(abs(warped_metric.determinant))
            transformed_field = np.dot(metric_inverse, transformed_field) * metric_factor
        except np.linalg.LinAlgError:
            logger.warning("Singular metric tensor, skipping metric transformation")
        
        # Apply geometric distortions
        for distortion in distortions:
            distortion_effect = self._compute_distortion_effect(
                transformed_field, distortion
            )
            transformed_field += distortion_effect
        
        return transformed_field
    
    def _compute_distortion_effect(self,
                                 field: np.ndarray,
                                 distortion: GeometricDistortion) -> np.ndarray:
        """Compute effect of geometric distortion on field."""
        effect_strength = distortion.magnitude * 0.1  # Scale factor
        
        if distortion.distortion_type == 'curvature':
            # Curvature creates non-linear field modification
            effect = effect_strength * distortion.direction_vector * np.sin(
                np.dot(field, distortion.direction_vector)
            )
        elif distortion.distortion_type == 'distance':
            # Distance distortion creates radial field modification
            field_norm = np.linalg.norm(field)
            if field_norm > 0:
                effect = effect_strength * field * (1.0 - np.exp(-field_norm))
            else:
                effect = np.zeros_like(field)
        elif distortion.distortion_type == 'path':
            # Path distortion creates directional field modification
            effect = effect_strength * distortion.direction_vector * np.dot(
                field, distortion.direction_vector
            )
        else:
            effect = np.zeros_like(field)
        
        return effect
    
    def _analyze_geometric_effects(self, distortions: List[GeometricDistortion]) -> Dict[str, Any]:
        """Analyze geometric effects from distortions."""
        if not distortions:
            return {'total_effect': 0.0, 'dominant_type': 'none'}
        
        effects_by_type = {}
        for distortion in distortions:
            if distortion.distortion_type not in effects_by_type:
                effects_by_type[distortion.distortion_type] = []
            effects_by_type[distortion.distortion_type].append(distortion.magnitude)
        
        total_effect = sum(d.magnitude for d in distortions)
        dominant_type = max(effects_by_type.keys(), key=lambda t: sum(effects_by_type[t]))
        
        return {
            'total_effect': total_effect,
            'dominant_type': dominant_type,
            'effects_by_type': {t: sum(mags) for t, mags in effects_by_type.items()},
            'num_distortions_by_type': {t: len(mags) for t, mags in effects_by_type.items()}
        }


class MetricWarping:
    """
    Implement emotional warping of semantic metric tensor.
    
    METRIC TENSOR MODULATION:
    Handles the mathematical details of metric tensor modification
    under emotional field influences with proper geometric constraints.
    """
    
    def __init__(self,
                 base_metric: Optional[MetricTensor] = None,
                 max_warping: float = 2.0,
                 stability_threshold: float = 0.01):
        """
        Initialize metric warping handler.
        
        Args:
            base_metric: Base metric tensor (Euclidean if None)
            max_warping: Maximum allowed metric warping
            stability_threshold: Threshold for numerical stability
        """
        self.base_metric = base_metric
        self.max_warping = max_warping
        self.stability_threshold = stability_threshold
    
    def warp_metric(self,
                   emotional_field: complex,
                   position: np.ndarray,
                   field_gradient: np.ndarray) -> MetricTensor:
        """
        Warp metric tensor based on emotional field at position.
        
        WARPING COMPUTATION:
        Applies emotional field effects to metric tensor while
        maintaining mathematical consistency and numerical stability.
        """
        dimension = len(position)
        
        # Create base metric if not provided
        if self.base_metric is None:
            base_components = np.eye(dimension)
        else:
            base_components = self.base_metric.components.copy()
        
        # Calculate emotional warping
        emotional_magnitude = abs(emotional_field)
        emotional_phase = np.angle(emotional_field)
        
        # Apply warping with stability constraints
        warped_components = base_components.copy()
        
        for i in range(dimension):
            for j in range(dimension):
                # Emotional influence on metric component
                gradient_product = field_gradient[i] * field_gradient[j]
                phase_factor = np.cos(emotional_phase + i * np.pi / dimension)
                
                warping_amount = emotional_magnitude * gradient_product * phase_factor
                warping_amount = np.clip(warping_amount, -self.max_warping, self.max_warping)
                
                warped_components[i, j] *= (1.0 + 0.1 * warping_amount)
        
        # Ensure numerical stability
        eigenvals = np.linalg.eigvals(warped_components)
        if np.any(eigenvals < self.stability_threshold):
            # Regularize metric to maintain positive definiteness
            regularization = self.stability_threshold * np.eye(dimension)
            warped_components += regularization
        
        # Calculate determinant
        determinant = np.linalg.det(warped_components)
        
        return MetricTensor(
            components=warped_components,
            dimension=dimension,
            determinant=determinant,
            is_riemannian=determinant > 0
        )


class FieldEffectCalculator:
    """
    Calculate field effects from emotional content on semantic geometry.
    
    FIELD EFFECT COMPUTATION:
    Implements the mathematical framework for computing how emotional
    fields influence semantic field propagation and interaction.
    """
    
    def __init__(self, coupling_strength: float = 0.2):
        """Initialize field effect calculator."""
        self.coupling_strength = coupling_strength
    
    def calculate_field_effects(self,
                              emotional_field: complex,
                              semantic_field: np.ndarray,
                              interaction_type: str = 'multiplicative') -> Dict[str, Any]:
        """
        Calculate emotional field effects on semantic field.
        
        FIELD INTERACTION TYPES:
        - multiplicative: E_field * S_field (direct modulation)
        - additive: E_field + S_field (field superposition)
        - phase_coupled: Phase-dependent interaction
        """
        emotional_magnitude = abs(emotional_field)
        emotional_phase = np.angle(emotional_field)
        semantic_magnitude = np.linalg.norm(semantic_field)
        
        effects = {
            'interaction_type': interaction_type,
            'coupling_strength': self.coupling_strength,
            'emotional_magnitude': emotional_magnitude,
            'semantic_magnitude': semantic_magnitude
        }
        
        if interaction_type == 'multiplicative':
            # Direct field modulation
            modulation_factor = 1.0 + self.coupling_strength * emotional_magnitude
            effects['modulation_factor'] = modulation_factor
            effects['field_enhancement'] = modulation_factor - 1.0
            
        elif interaction_type == 'additive':
            # Field superposition
            emotional_contribution = emotional_magnitude * self.coupling_strength
            effects['emotional_contribution'] = emotional_contribution
            effects['field_enhancement'] = emotional_contribution / (semantic_magnitude + 1e-10)
            
        elif interaction_type == 'phase_coupled':
            # Phase-dependent coupling
            phase_alignment = np.cos(emotional_phase)
            coupling_efficiency = self.coupling_strength * phase_alignment
            effects['phase_alignment'] = phase_alignment
            effects['coupling_efficiency'] = coupling_efficiency
            effects['field_enhancement'] = coupling_efficiency * emotional_magnitude
        
        return effects


class GeometricDistortionManager:
    """
    Manage geometric distortions caused by emotional fields.
    
    DISTORTION MANAGEMENT:
    Coordinates multiple geometric distortions to create coherent
    emotional field effects while maintaining geometric consistency.
    """
    
    def __init__(self, max_distortions: int = 10):
        """Initialize geometric distortion manager."""
        self.max_distortions = max_distortions
        self.active_distortions = []
    
    def add_distortion(self, distortion: GeometricDistortion):
        """Add geometric distortion to active set."""
        self.active_distortions.append(distortion)
        
        # Remove oldest if exceeding limit
        if len(self.active_distortions) > self.max_distortions:
            self.active_distortions.pop(0)
    
    def compute_total_distortion(self,
                               position: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute total geometric distortion at position.
        
        DISTORTION SUPERPOSITION:
        Combines multiple geometric distortions using appropriate
        mathematical principles for field superposition.
        """
        total_distortion = np.zeros_like(position)
        total_magnitude = 0.0
        
        for distortion in self.active_distortions:
            # Calculate distance from distortion center
            center, radius = distortion.affected_region
            distance = np.linalg.norm(position - center)
            
            if distance <= radius:
                # Apply distance-based decay
                decay_factor = np.exp(-distance / radius)
                
                # Compute distortion contribution
                contribution = distortion.magnitude * decay_factor * distortion.direction_vector
                total_distortion += contribution
                total_magnitude += distortion.magnitude * decay_factor
        
        return total_distortion, total_magnitude
    
    def cleanup_distortions(self, current_time: float, max_age: float = 10.0):
        """Remove old distortions based on age."""
        # Simplified cleanup - in full implementation would track creation time
        if len(self.active_distortions) > self.max_distortions // 2:
            self.active_distortions = self.active_distortions[self.max_distortions // 4:]


# Convenience functions for field modulation

def apply_emotional_modulation(semantic_field: np.ndarray,
                             emotional_trajectory: complex,
                             coupling_strength: float = 0.3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to apply emotional field modulation.
    
    Args:
        semantic_field: Base semantic field vector
        emotional_trajectory: Complex emotional trajectory
        coupling_strength: Emotional coupling strength κ_E
        
    Returns:
        Tuple of (modulated_field, analysis)
    """
    modulator = EmotionalFieldModulator(
        emotional_coupling_strength=coupling_strength
    )
    
    return modulator.apply_emotional_field_effects(
        semantic_field=semantic_field,
        emotional_trajectory=emotional_trajectory
    )


def create_emotional_modulator(coupling_strength: float = 0.3,
                             distortion_radius: float = 1.0) -> EmotionalFieldModulator:
    """
    Convenience function to create emotional field modulator.
    
    Args:
        coupling_strength: Emotional coupling strength
        distortion_radius: Radius of field effects
        
    Returns:
        Configured EmotionalFieldModulator
    """
    return EmotionalFieldModulator(
        emotional_coupling_strength=coupling_strength,
        distortion_radius=distortion_radius,
        max_curvature=2.0
    )