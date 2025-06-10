"""
Transformation Operator: T[Q] Implementation

Converts conceptual charges Q(τ, C, s) into geometric imprints on the product manifold.
Moved from product_manifold/ - consolidated into universe.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import scipy.spatial.distance as spatial_dist
from scipy.special import j0, j1  # Bessel functions

# Import conceptual charge from charge_pipeline mathematics
try:
    from ..model.mathematics.conceptual_charge import ConceptualCharge
except ImportError:
    # Fallback for development
    ConceptualCharge = Any


@dataclass
class TransformationParameters:
    """Parameters controlling charge-to-imprint transformation"""
    spatial_extent: float = 1.0  # Characteristic length scale of imprints
    phase_coupling_strength: float = 0.5  # How strongly phase modulates spatial profile
    amplitude_scaling: float = 1.0  # Global scaling for imprint magnitudes
    bessel_order: int = 0  # Order of Bessel function for radial profiles
    decay_rate: float = 2.0  # Exponential decay rate for spatial profiles
    interference_strength: float = 1.0  # Strength of phase interference effects


class TransformationOperator:
    """
    Transforms conceptual charges into geometric imprints on product manifold.
    
    Core mathematical operation: T[Q(τ,C,s)] → manifold contribution
    
    The transformation preserves:
    - Charge magnitude → imprint strength  
    - Complex phase → interference patterns
    - Semantic content → spatial distribution
    - Trajectory state → temporal evolution
    """
    
    def __init__(self, 
                 spatial_dimensions: Tuple[int, int] = (64, 64),
                 spatial_extent: Tuple[float, float] = (10.0, 10.0),
                 params: Optional[TransformationParameters] = None):
        """
        Initialize transformation operator.
        
        Args:
            spatial_dimensions: Grid resolution (Nx, Ny)
            spatial_extent: Physical size (Lx, Ly) 
            params: Transformation parameters
        """
        self.spatial_dimensions = spatial_dimensions
        self.spatial_extent = spatial_extent
        self.params = params or TransformationParameters()
        
        # Create spatial coordinate grids
        self._initialize_spatial_grids()
        
        # Precompute spatial profile kernels
        self._precompute_profile_kernels()
    
    def _initialize_spatial_grids(self):
        """Create spatial coordinate meshgrids"""
        Nx, Ny = self.spatial_dimensions
        Lx, Ly = self.spatial_extent
        
        x = np.linspace(-Lx/2, Lx/2, Nx)
        y = np.linspace(-Ly/2, Ly/2, Ny)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Radial distance from center
        self.R = np.sqrt(self.X**2 + self.Y**2)
        
        # Angular coordinate
        self.Theta = np.arctan2(self.Y, self.X)
    
    def _precompute_profile_kernels(self):
        """Precompute spatial profile basis functions"""
        # Radial Bessel function profiles
        r_normalized = self.R / self.params.spatial_extent
        
        # Primary radial profile
        self.radial_profile_0 = j0(r_normalized) * np.exp(-r_normalized / self.params.decay_rate)
        
        # Secondary radial profile for richer spatial structure
        self.radial_profile_1 = j1(r_normalized) * np.exp(-r_normalized / self.params.decay_rate)
        
        # Angular modulation patterns
        self.angular_modes = {
            0: np.ones_like(self.Theta),  # Isotropic
            1: np.cos(self.Theta),        # Dipole x
            2: np.sin(self.Theta),        # Dipole y  
            3: np.cos(2 * self.Theta),    # Quadrupole
            4: np.sin(2 * self.Theta),    # Quadrupole rotated
        }
    
    def charge_to_imprint(self, 
                         charge: ConceptualCharge, 
                         position: Optional[Tuple[float, float]] = None,
                         observational_state: Optional[float] = None) -> np.ndarray:
        """
        Convert single conceptual charge to geometric imprint.
        
        Mathematical Implementation:
        T[Q(τ,C,s)](x) = |Q|² · phase_modulation(x) · spatial_profile(x,τ)
        
        Args:
            charge: ConceptualCharge to transform
            position: Center position for imprint (default: center)
            observational_state: Override charge's observational state
            
        Returns:
            Complex-valued imprint on spatial grid
        """
        # Use provided observational state or charge's current state
        s = observational_state if observational_state is not None else charge.observational_state
        
        # Compute complete charge value
        Q_complex = charge.compute_complete_charge(s)
        charge_magnitude_sq = abs(Q_complex)**2
        charge_phase = np.angle(Q_complex)
        
        # Position for imprint (default to center)
        if position is None:
            center_x, center_y = 0.0, 0.0
        else:
            center_x, center_y = position
        
        # Compute spatial profile centered at position
        spatial_profile = self._compute_spatial_profile(charge, center_x, center_y)
        
        # Phase modulation based on charge phase and spatial coordinates
        phase_modulation = self._compute_phase_modulation(charge_phase, center_x, center_y)
        
        # Complete imprint: magnitude × spatial × phase
        imprint = (charge_magnitude_sq * self.params.amplitude_scaling * 
                  spatial_profile * phase_modulation)
        
        return imprint.astype(np.complex128)
    
    def _compute_spatial_profile(self, 
                                charge: ConceptualCharge, 
                                center_x: float, 
                                center_y: float) -> np.ndarray:
        """
        Compute token-dependent spatial distribution.
        
        Uses semantic vector components to determine spatial profile characteristics.
        """
        # Translate coordinate system to center position
        X_centered = self.X - center_x
        Y_centered = self.Y - center_y
        R_centered = np.sqrt(X_centered**2 + Y_centered**2)
        
        # Use semantic vector to determine profile characteristics
        semantic_mean = np.mean(charge.semantic_vector)
        semantic_std = np.std(charge.semantic_vector)
        
        # Radial scale determined by semantic content
        radial_scale = self.params.spatial_extent * (1.0 + 0.3 * semantic_mean)
        
        # Primary radial component
        r_norm = R_centered / radial_scale
        radial_component = j0(r_norm) * np.exp(-r_norm / self.params.decay_rate)
        
        # Angular modulation based on semantic vector components
        if len(charge.semantic_vector) >= 5:
            # Use first few semantic dimensions for angular structure
            angular_weights = charge.semantic_vector[:5] / np.max(np.abs(charge.semantic_vector[:5]))
            
            angular_modulation = np.zeros_like(self.Theta)
            for i, (mode_idx, angular_pattern) in enumerate(self.angular_modes.items()):
                if i < len(angular_weights):
                    angular_modulation += angular_weights[i] * angular_pattern
        else:
            # Fallback: isotropic profile
            angular_modulation = np.ones_like(self.Theta)
        
        # Combine radial and angular components
        spatial_profile = radial_component * (1.0 + 0.2 * angular_modulation)
        
        return np.real(spatial_profile)  # Spatial profile is real-valued
    
    def _compute_phase_modulation(self, 
                                 charge_phase: float, 
                                 center_x: float, 
                                 center_y: float) -> np.ndarray:
        """
        Compute complex phase modulation pattern.
        
        Creates interference patterns based on charge phase and spatial coordinates.
        """
        # Base phase from charge
        base_phase = charge_phase
        
        # Spatial phase variation for interference effects
        if self.params.phase_coupling_strength > 0:
            # Translate coordinates to center
            X_centered = self.X - center_x
            Y_centered = self.Y - center_y
            
            # Phase varies linearly with distance for wave-like interference
            spatial_phase_x = self.params.phase_coupling_strength * X_centered / self.params.spatial_extent
            spatial_phase_y = self.params.phase_coupling_strength * Y_centered / self.params.spatial_extent
            
            total_phase = base_phase + spatial_phase_x + spatial_phase_y
        else:
            # Uniform phase if no spatial coupling
            total_phase = base_phase * np.ones_like(self.X)
        
        return np.exp(1j * total_phase)
    
    def batch_transform(self, 
                       charges: List[ConceptualCharge],
                       positions: Optional[List[Tuple[float, float]]] = None,
                       observational_states: Optional[List[float]] = None) -> np.ndarray:
        """
        Transform multiple charges simultaneously for efficiency.
        
        Args:
            charges: List of conceptual charges
            positions: List of center positions (default: random)
            observational_states: List of observational states (default: use charge states)
            
        Returns:
            Combined imprint from all charges (superposition)
        """
        if not charges:
            return np.zeros(self.spatial_dimensions, dtype=np.complex128)
        
        # Generate random positions if not provided
        if positions is None:
            Lx, Ly = self.spatial_extent
            positions = [(np.random.uniform(-Lx/4, Lx/4), 
                         np.random.uniform(-Ly/4, Ly/4)) for _ in charges]
        
        # Use charge states if observational states not provided
        if observational_states is None:
            observational_states = [None] * len(charges)
        
        # Transform each charge and sum (superposition principle)
        total_imprint = np.zeros(self.spatial_dimensions, dtype=np.complex128)
        
        for charge, position, obs_state in zip(charges, positions, observational_states):
            imprint = self.charge_to_imprint(charge, position, obs_state)
            total_imprint += imprint
        
        return total_imprint