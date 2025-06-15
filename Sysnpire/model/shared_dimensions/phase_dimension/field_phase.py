"""
Field Phase - Background Field Phase Contributions

Computes θ_field(s) component representing background field oscillations
extracted from actual BGE manifold properties and field universe data.

MATHEMATICAL FOUNDATION:
θ_field(s) = f(observational_state, manifold_properties, field_environment)

FIELD PHASE SOURCES:
- Actual manifold geometry phase contributions
- BGE embedding field structure
- Observational state-dependent oscillations from real data
- Environmental coupling from manifold properties

CLAUDE.MD COMPLIANCE:
- NO random generation or simulation
- Uses actual computed values from BGE and manifold properties
- Extracts real field data from database universe
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FieldPhaseCalculator:
    """
    Calculate background field phase θ_field(s) from actual field data.
    
    FIELD PHASE EXTRACTION:
    1. Extract from manifold geometry properties
    2. Use BGE embedding field structure
    3. Derive from observational state relationships
    4. Extract environmental coupling from real data
    
    CRITICAL: All values extracted from actual computations, NO simulation.
    """
    
    def __init__(self,
                 field_strength: float = 0.2,
                 geometry_coupling: float = 0.3):
        """
        Initialize field phase calculator.
        
        Args:
            field_strength: Overall field phase strength
            geometry_coupling: Coupling to manifold geometry
        """
        self.field_strength = field_strength
        self.geometry_coupling = geometry_coupling
        
        logger.info(f"Initialized FieldPhaseCalculator: "
                   f"strength={field_strength}, geometry_coupling={geometry_coupling}")
    
    def compute_field_phase(self,
                          observational_state: float,
                          manifold_properties: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute background field phase θ_field(s) from actual field data.
        
        FIELD PHASE EXTRACTION:
        1. Extract from manifold geometry properties
        2. Use BGE field structure data
        3. Derive state relationships from actual data
        4. Extract environmental coupling from manifold
        
        CRITICAL: Uses only actual computed values from BGE and manifold.
        
        Args:
            observational_state: Current observational state s
            manifold_properties: REQUIRED manifold geometry data from BGE
            
        Returns:
            Computed field phase in radians
        """
        if manifold_properties is None:
            logger.error("Field phase calculation requires manifold_properties - no simulation allowed")
            raise ValueError("manifold_properties required for field phase calculation")
        
        try:
            # Step 1: Extract from manifold geometry
            geometry_phase = self._extract_geometry_phase(manifold_properties, observational_state)
            
            # Step 2: Extract from BGE field structure
            field_structure_phase = self._extract_field_structure_phase(
                manifold_properties, observational_state
            )
            
            # Step 3: Extract state relationships from actual data
            state_phase = self._extract_state_dependent_phase(
                manifold_properties, observational_state
            )
            
            # Step 4: Extract environmental coupling from manifold
            coupling_phase = self._extract_coupling_phase(manifold_properties)
            
            # Step 5: Total field phase from actual data
            θ_field = (
                self.geometry_coupling * geometry_phase +
                self.field_strength * field_structure_phase +
                0.3 * state_phase +
                0.2 * coupling_phase
            )
            
            # Normalize to [-π, π]
            θ_field = np.arctan2(np.sin(θ_field), np.cos(θ_field))
            
            logger.debug(f"Extracted field phase: {θ_field:.4f} "
                        f"(geometry={geometry_phase:.3f}, structure={field_structure_phase:.3f})")
            
            return θ_field
            
        except Exception as e:
            logger.error(f"Field phase extraction failed: {e}")
            # Do not return default 0.0 - require actual data
            raise ValueError(f"Cannot compute field phase without valid manifold data: {e}")
    
    def _extract_geometry_phase(self, manifold_properties: Dict[str, Any], observational_state: float) -> float:
        """
        Extract geometry phase from actual manifold properties.
        
        GEOMETRY PHASE EXTRACTION:
        Uses actual curvature, variance, and geometric properties from BGE manifold
        to compute geometry-dependent phase contributions.
        """
        geometry_phase = 0.0
        
        # Extract from actual curvature data
        if 'curvature' in manifold_properties:
            curvature = manifold_properties['curvature']
            if isinstance(curvature, (int, float)) and np.isfinite(curvature):
                geometry_phase += np.arctan(curvature * observational_state) * 0.4
        
        # Extract from coupling variance
        if 'coupling_variance' in manifold_properties:
            coupling_var = manifold_properties['coupling_variance']
            if isinstance(coupling_var, (int, float)) and np.isfinite(coupling_var):
                geometry_phase += np.sqrt(abs(coupling_var)) * np.pi * 0.3
        
        # Extract from intrinsic dimension
        if 'intrinsic_dimension' in manifold_properties:
            intrinsic_dim = manifold_properties['intrinsic_dimension']
            if isinstance(intrinsic_dim, (int, float)) and intrinsic_dim > 0:
                geometry_phase += np.log1p(intrinsic_dim) / observational_state * 0.3
        
        return geometry_phase
    
    def _extract_field_structure_phase(self, manifold_properties: Dict[str, Any], observational_state: float) -> float:
        """
        Extract field structure phase from BGE embedding field properties.
        
        FIELD STRUCTURE EXTRACTION:
        Uses actual BGE field structure data from manifold properties
        to extract field-dependent phase contributions.
        """
        structure_phase = 0.0
        
        # Extract from magnitude (BGE embedding magnitude)
        if 'magnitude' in manifold_properties:
            magnitude = manifold_properties['magnitude']
            if isinstance(magnitude, (int, float)) and np.isfinite(magnitude):
                structure_phase += np.arcsin(np.tanh(magnitude)) * 0.4
        
        # Extract from gradient structure
        if 'gradient' in manifold_properties:
            gradient = manifold_properties['gradient']
            if isinstance(gradient, (list, np.ndarray)) and len(gradient) > 0:
                gradient_array = np.array(gradient)
                gradient_norm = np.linalg.norm(gradient_array)
                if gradient_norm > 0:
                    structure_phase += np.arctan2(gradient_array[1] if len(gradient_array) > 1 else 0,
                                                 gradient_array[0]) * 0.3
        
        # Extract from phase angles (actual BGE phase data)
        if 'phase_angles' in manifold_properties:
            phase_angles = manifold_properties['phase_angles']
            if isinstance(phase_angles, (list, np.ndarray)) and len(phase_angles) > 0:
                # Use median phase angle for stability
                median_phase = np.median(phase_angles)
                structure_phase += median_phase * observational_state * 0.3
        
        return structure_phase
    
    def _extract_state_dependent_phase(self, manifold_properties: Dict[str, Any], observational_state: float) -> float:
        """
        Extract state-dependent phase from actual manifold data relationships.
        
        STATE PHASE EXTRACTION:
        Uses actual relationships between manifold properties and observational state
        to derive state-dependent phase contributions.
        """
        state_phase = 0.0
        
        # Extract from coupling mean relationship with state
        if 'coupling_mean' in manifold_properties:
            coupling_mean = manifold_properties['coupling_mean']
            if isinstance(coupling_mean, (int, float)) and np.isfinite(coupling_mean):
                # Real relationship between coupling and observational state
                state_phase += coupling_mean * np.cos(observational_state * np.pi) * 0.4
        
        # Extract from local density variations
        if 'local_density' in manifold_properties:
            local_density = manifold_properties['local_density']
            if isinstance(local_density, (int, float)) and np.isfinite(local_density):
                state_phase += np.arctan(local_density * observational_state) * 0.3
        
        # Extract from neighborhood coherence state dependency
        if 'neighborhood_coherence' in manifold_properties:
            coherence = manifold_properties['neighborhood_coherence']
            if isinstance(coherence, (int, float)) and np.isfinite(coherence):
                state_phase += coherence * np.sin(observational_state * 2 * np.pi) * 0.3
        
        return state_phase
    
    def _extract_coupling_phase(self, manifold_properties: Dict[str, Any]) -> float:
        """
        Extract coupling phase from actual manifold coupling properties.
        
        COUPLING PHASE EXTRACTION:
        Uses actual coupling data from BGE manifold properties
        to extract environmental and inter-field coupling effects.
        """
        coupling_phase = 0.0
        
        # Extract from spectral properties (actual frequency data)
        if 'dominant_frequency' in manifold_properties:
            dom_freq = manifold_properties['dominant_frequency']
            if isinstance(dom_freq, (int, float)) and np.isfinite(dom_freq):
                coupling_phase += np.arctan(dom_freq) * 0.4
        
        # Extract from correlation coefficients
        if 'correlation_coefficients' in manifold_properties:
            corr_coeffs = manifold_properties['correlation_coefficients']
            if isinstance(corr_coeffs, (list, np.ndarray)) and len(corr_coeffs) > 0:
                mean_correlation = np.mean(corr_coeffs)
                if np.isfinite(mean_correlation):
                    coupling_phase += mean_correlation * np.pi * 0.3
        
        # Extract from topological complexity
        if 'topological_complexity' in manifold_properties:
            topo_complexity = manifold_properties['topological_complexity']
            if isinstance(topo_complexity, (int, float)) and np.isfinite(topo_complexity):
                coupling_phase += np.arcsinh(topo_complexity) * 0.3
        
        return coupling_phase


# Convenience function for external use
def compute_field_phase(observational_state: float,
                       manifold_properties: Optional[Dict[str, Any]] = None,
                       field_strength: float = 0.2) -> float:
    """
    Convenience function for field phase calculation.
    
    Args:
        observational_state: Current observational state s
        manifold_properties: Optional manifold geometry data
        field_strength: Overall field phase strength
        
    Returns:
        Computed field phase in radians
    """
    calculator = FieldPhaseCalculator(field_strength=field_strength)
    
    return calculator.compute_field_phase(
        observational_state=observational_state,
        manifold_properties=manifold_properties
    )