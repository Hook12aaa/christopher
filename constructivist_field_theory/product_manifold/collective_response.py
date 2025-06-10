"""
Collective Response Function Implementation

Implements collective response: R_collective = M · G[M] · M†

This computes observable collective response patterns from manifold field interactions.
The collective response captures:
- Path integration effects through geometric interaction kernel G[M]
- Interference patterns from complex field structure
- Observable quantities through Hermitian structure
- Multi-scale collective phenomena

Creates measurable signatures of sociological field dynamics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.special import hankel1, hankel2
from scipy.integrate import trapz, simps
import warnings

# Import manifold field equation if available
try:
    from .manifold_field_equation import ManifoldFieldEquation
except ImportError:
    ManifoldFieldEquation = Any


@dataclass
class CollectiveResponseParameters:
    """Parameters controlling collective response computation"""
    interaction_range: float = 2.0      # Characteristic interaction distance
    coupling_strength: float = 1.0      # G[M] coupling strength
    curvature_sensitivity: float = 0.5  # How much curvature affects interactions
    damping_length: float = 1.0         # Exponential damping for long-range interactions  
    spectral_cutoff: float = 10.0       # High-frequency cutoff in k-space
    hermitian_tolerance: float = 1e-10  # Tolerance for Hermiticity verification


class CollectiveResponseFunction:
    """
    Computes collective response function R_collective = M · G[M] · M†.
    
    The collective response provides:
    1. Observable quantities from manifold field dynamics
    2. Path integration through geometric interaction kernel
    3. Interference pattern analysis
    4. Multi-scale collective phenomena detection
    5. Real-valued response signatures for measurement
    """
    
    def __init__(self,
                 manifold_equation: ManifoldFieldEquation,
                 params: Optional[CollectiveResponseParameters] = None):
        """
        Initialize collective response function.
        
        Args:
            manifold_equation: Manifold field equation providing field M
            params: Collective response parameters
        """
        self.manifold_equation = manifold_equation
        self.params = params or CollectiveResponseParameters()
        
        # Extract spatial information from manifold equation
        self.spatial_dimensions = manifold_equation.spatial_dimensions
        self.spatial_extent = manifold_equation.spatial_extent
        self.X = manifold_equation.X
        self.Y = manifold_equation.Y
        self.dx = manifold_equation.dx
        self.dy = manifold_equation.dy
        
        # Precompute interaction kernels
        self._precompute_interaction_kernels()
        
        # Storage for response analysis
        self.response_history: List[Dict[str, Any]] = []
    
    def _precompute_interaction_kernels(self):
        """Precompute geometric interaction kernels G[M]"""
        # Distance matrix for all spatial points
        Nx, Ny = self.spatial_dimensions
        points = np.column_stack([self.X.ravel(), self.Y.ravel()])
        self.distance_matrix = cdist(points, points)
        
        # Reshape to spatial grid dimensions
        self.distance_matrix = self.distance_matrix.reshape(
            (Nx * Ny, Nx * Ny)
        )
        
        # Base interaction kernel (modified Bessel function)
        r_normalized = self.distance_matrix / self.params.interaction_range
        
        # Green's function kernel with exponential damping
        kernel_base = np.exp(-r_normalized / self.params.damping_length) / (
            1.0 + r_normalized
        )
        
        # Avoid self-interaction singularities
        np.fill_diagonal(kernel_base, 0.0)
        
        self.base_kernel = kernel_base
        
        # Precompute spectral kernels for efficient convolution
        self._precompute_spectral_kernels()
    
    def _precompute_spectral_kernels(self):
        """Precompute Fourier-space interaction kernels for efficiency"""
        Nx, Ny = self.spatial_dimensions
        
        # Wavenumber grids
        kx = 2 * np.pi * np.fft.fftfreq(Nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, self.dy)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K = np.sqrt(KX**2 + KY**2)
        
        # Spectral interaction kernel
        self.spectral_kernel = (
            self.params.coupling_strength * 
            np.exp(-K * self.params.interaction_range) /
            (1.0 + K * self.params.interaction_range)
        )
        
        # Apply spectral cutoff
        self.spectral_kernel[K > self.params.spectral_cutoff] = 0.0
    
    def compute_geometric_interaction_kernel(self, field: np.ndarray) -> np.ndarray:
        """
        Compute field-dependent geometric interaction kernel G[M].
        
        The kernel depends on local field curvature and magnitude.
        
        Args:
            field: Current manifold field M(x,t)
            
        Returns:
            Geometric interaction kernel G[M]
        """
        magnitude = np.abs(field)
        phase = np.angle(field)
        
        # Compute field curvature (Laplacian of magnitude)
        magnitude_laplacian = self.manifold_equation.laplacian(magnitude.astype(np.complex128)).real
        
        # Curvature modulation factor
        curvature_factor = 1.0 + self.params.curvature_sensitivity * magnitude_laplacian / (
            1.0 + np.max(np.abs(magnitude_laplacian))
        )
        
        # Field-dependent interaction strength
        field_modulation = 1.0 + 0.2 * magnitude / (1.0 + np.max(magnitude))
        
        # Combined modulation
        total_modulation = curvature_factor * field_modulation
        
        # Apply to base kernel (via spectral convolution for efficiency)
        modulation_fft = np.fft.fft2(total_modulation)
        kernel_fft = self.spectral_kernel * modulation_fft
        
        # Return spatial kernel
        return np.fft.ifft2(kernel_fft).real
    
    def compute_collective_response(self, 
                                  field: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute complete collective response R_collective = M · G[M] · M†.
        
        Args:
            field: Manifold field (default: use current field from manifold_equation)
            
        Returns:
            Dictionary with collective response analysis
        """
        # Use provided field or current manifold field
        if field is None:
            field = self.manifold_equation.field
        
        # Geometric interaction kernel
        G_M = self.compute_geometric_interaction_kernel(field)
        
        # Flatten field for matrix operations
        field_flat = field.flatten()
        
        # Collective response computation: R = M · G[M] · M†
        # Using efficient spectral convolution
        
        # Method 1: Direct matrix multiplication (for small grids)
        if np.prod(self.spatial_dimensions) < 1024:
            G_matrix = G_M.reshape(field_flat.shape[0], field_flat.shape[0])
            
            # R_collective = M† · G[M] · M (Hermitian form)
            response_matrix = np.outer(field_flat.conj(), field_flat)
            collective_response = G_matrix * response_matrix
            
        else:
            # Method 2: Spectral convolution (for large grids)
            collective_response = self._compute_spectral_collective_response(field, G_M)
        
        # Extract observable quantities
        response_analysis = self._analyze_collective_response(field, collective_response)
        
        return response_analysis
    
    def _compute_spectral_collective_response(self, 
                                            field: np.ndarray, 
                                            G_M: np.ndarray) -> np.ndarray:
        """Compute collective response using spectral methods for efficiency"""
        # Field magnitude and phase
        magnitude = np.abs(field)
        phase = np.angle(field)
        
        # Spectral representation of field components
        magnitude_fft = np.fft.fft2(magnitude)
        phase_cos_fft = np.fft.fft2(np.cos(phase))
        phase_sin_fft = np.fft.fft2(np.sin(phase))
        
        # Interaction kernel in spectral space
        G_fft = np.fft.fft2(G_M)
        
        # Convolution: G[M] * |M|²
        magnitude_sq = magnitude**2
        response_magnitude_fft = G_fft * np.fft.fft2(magnitude_sq)
        
        # Phase correlation contribution
        phase_correlation_real = np.fft.ifft2(G_fft * phase_cos_fft * magnitude_fft.conj()).real
        phase_correlation_imag = np.fft.ifft2(G_fft * phase_sin_fft * magnitude_fft.conj()).imag
        
        # Combined collective response
        response_magnitude = np.fft.ifft2(response_magnitude_fft).real
        phase_correlation = phase_correlation_real + 1j * phase_correlation_imag
        
        # Full collective response matrix (diagonal approximation for large grids)
        collective_response = response_magnitude + np.abs(phase_correlation)
        
        return collective_response
    
    def _analyze_collective_response(self, 
                                   field: np.ndarray, 
                                   collective_response: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze collective response patterns for observable signatures"""
        magnitude = np.abs(field)
        phase = np.angle(field)
        
        # Handle different response matrix shapes
        if collective_response.ndim == 2 and collective_response.shape[0] == collective_response.shape[1]:
            # Full matrix case
            response_magnitude = np.abs(collective_response)
            response_trace = np.trace(collective_response).real
            response_eigenvalues = np.linalg.eigvals(collective_response)
            response_spectral_radius = np.max(np.abs(response_eigenvalues))
            
            # Reshape for spatial analysis
            response_diagonal = np.diag(collective_response).reshape(self.spatial_dimensions)
            
        else:
            # Spatial field case
            response_diagonal = np.abs(collective_response)
            response_magnitude = response_diagonal
            response_trace = np.sum(response_diagonal) * self.dx * self.dy
            response_spectral_radius = np.max(response_magnitude)
        
        # Observable quantities
        total_collective_response = np.sum(response_magnitude) * self.dx * self.dy
        peak_response = np.max(response_magnitude)
        mean_response = np.mean(response_magnitude)
        
        # Spatial structure analysis
        # Center of mass of collective response
        total_response = np.sum(response_diagonal)
        if total_response > 0:
            center_x = np.sum(response_diagonal * self.X) / total_response
            center_y = np.sum(response_diagonal * self.Y) / total_response
        else:
            center_x = center_y = 0.0
        
        # Response spread (second moments)
        X_centered = self.X - center_x
        Y_centered = self.Y - center_y
        spread_x = np.sqrt(np.sum(response_diagonal * X_centered**2) / total_response) if total_response > 0 else 0.0
        spread_y = np.sqrt(np.sum(response_diagonal * Y_centered**2) / total_response) if total_response > 0 else 0.0
        
        # Coherence measures
        field_response_correlation = np.corrcoef(magnitude.flatten(), response_diagonal.flatten())[0, 1]
        if np.isnan(field_response_correlation):
            field_response_correlation = 0.0
        
        # Interference pattern analysis
        interference_analysis = self._analyze_interference_patterns(field, response_diagonal)
        
        return {
            'collective_response_field': response_diagonal,
            'total_response': total_collective_response,
            'peak_response': peak_response,
            'mean_response': mean_response,
            'response_center_x': center_x,
            'response_center_y': center_y,
            'response_spread_x': spread_x,
            'response_spread_y': spread_y,
            'field_response_correlation': field_response_correlation,
            'response_trace': response_trace,
            'spectral_radius': response_spectral_radius,
            **interference_analysis
        }
    
    def _analyze_interference_patterns(self, 
                                     field: np.ndarray, 
                                     response: np.ndarray) -> Dict[str, Any]:
        """Analyze interference patterns in collective response"""
        magnitude = np.abs(field)
        phase = np.angle(field)
        
        # Phase coherence across space
        phase_gradients = np.gradient(phase)
        phase_gradient_magnitude = np.sqrt(phase_gradients[0]**2 + phase_gradients[1]**2)
        phase_coherence = np.exp(-np.mean(phase_gradient_magnitude))
        
        # Constructive vs destructive interference regions
        # High response with high field magnitude → constructive
        # Low response with high field magnitude → destructive
        field_threshold = 0.3 * np.max(magnitude)
        response_threshold = 0.3 * np.max(response)
        
        high_field_mask = magnitude > field_threshold
        high_response_mask = response > response_threshold
        
        constructive_regions = high_field_mask & high_response_mask
        destructive_regions = high_field_mask & (~high_response_mask)
        
        constructive_fraction = np.sum(constructive_regions) / np.sum(high_field_mask) if np.sum(high_field_mask) > 0 else 0.0
        destructive_fraction = np.sum(destructive_regions) / np.sum(high_field_mask) if np.sum(high_field_mask) > 0 else 0.0
        
        # Path integration measure (response spread vs field spread)
        field_spread = np.sqrt(np.var(magnitude))
        response_spread = np.sqrt(np.var(response))
        path_integration_factor = response_spread / (field_spread + 1e-10)
        
        return {
            'phase_coherence': phase_coherence,
            'constructive_fraction': constructive_fraction,
            'destructive_fraction': destructive_fraction,
            'path_integration_factor': path_integration_factor,
            'constructive_regions': constructive_regions,
            'destructive_regions': destructive_regions
        }
    
    def temporal_response_analysis(self, 
                                 time_steps: int = 50,
                                 dt: float = 0.01,
                                 charges: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Analyze collective response evolution over time.
        
        Args:
            time_steps: Number of evolution steps
            dt: Time step size
            charges: Conceptual charges driving evolution
            
        Returns:
            Temporal analysis of collective response
        """
        if charges is None:
            charges = []
        
        # Store initial state
        initial_field = self.manifold_equation.field.copy()
        initial_time = self.manifold_equation.time
        
        # Evolve and analyze response at each step
        response_evolution = []
        times = []
        
        for step in range(time_steps):
            # Compute current collective response
            response_analysis = self.compute_collective_response()
            response_evolution.append(response_analysis)
            times.append(self.manifold_equation.time)
            
            # Evolve field for next step
            if step < time_steps - 1:  # Don't evolve on last step
                self.manifold_equation.evolve_field(dt, charges)
        
        # Extract time series data
        total_responses = [r['total_response'] for r in response_evolution]
        peak_responses = [r['peak_response'] for r in response_evolution]
        phase_coherences = [r['phase_coherence'] for r in response_evolution]
        constructive_fractions = [r['constructive_fraction'] for r in response_evolution]
        
        # Temporal analysis
        response_growth_rate = np.gradient(total_responses, dt) if len(total_responses) > 1 else [0.0]
        response_stability = np.std(total_responses[-10:]) / np.mean(total_responses[-10:]) if len(total_responses) >= 10 else 0.0
        
        # Restore initial state if desired (comment out to keep evolved state)
        # self.manifold_equation.field = initial_field
        # self.manifold_equation.time = initial_time
        
        return {
            'times': times,
            'total_responses': total_responses,
            'peak_responses': peak_responses,
            'phase_coherences': phase_coherences,
            'constructive_fractions': constructive_fractions,
            'response_growth_rate': response_growth_rate,
            'response_stability': response_stability,
            'final_response_analysis': response_evolution[-1]
        }
    
    def verify_hermiticity(self, collective_response: np.ndarray) -> Dict[str, float]:
        """
        Verify Hermitian properties of collective response matrix.
        
        Essential for ensuring real-valued observables.
        """
        if collective_response.ndim != 2 or collective_response.shape[0] != collective_response.shape[1]:
            return {'hermiticity_error': float('inf'), 'message': 'Not a square matrix'}
        
        # Hermiticity check: A = A†
        hermitian_error = np.max(np.abs(collective_response - collective_response.conj().T))
        
        # Real diagonal check
        diagonal_imag_error = np.max(np.abs(np.imag(np.diag(collective_response))))
        
        # Eigenvalue reality check  
        eigenvalues = np.linalg.eigvals(collective_response)
        eigenvalue_imag_error = np.max(np.abs(np.imag(eigenvalues)))
        
        return {
            'hermiticity_error': hermitian_error,
            'diagonal_imag_error': diagonal_imag_error,
            'eigenvalue_imag_error': eigenvalue_imag_error,
            'is_hermitian': hermitian_error < self.params.hermitian_tolerance
        }
    
    def multi_scale_response_analysis(self, scales: List[float] = None) -> Dict[str, Any]:
        """
        Analyze collective response at multiple interaction scales.
        
        Args:
            scales: List of interaction ranges to analyze
            
        Returns:
            Multi-scale response characteristics
        """
        if scales is None:
            scales = [0.5, 1.0, 2.0, 4.0]  # Default scales
        
        original_range = self.params.interaction_range
        scale_responses = {}
        
        for scale in scales:
            # Temporarily modify interaction range
            self.params.interaction_range = scale
            self._precompute_interaction_kernels()  # Recompute kernels
            
            # Compute response at this scale
            response_analysis = self.compute_collective_response()
            scale_responses[f'scale_{scale}'] = {
                'total_response': response_analysis['total_response'],
                'peak_response': response_analysis['peak_response'],
                'phase_coherence': response_analysis['phase_coherence'],
                'constructive_fraction': response_analysis['constructive_fraction']
            }
        
        # Restore original parameters
        self.params.interaction_range = original_range
        self._precompute_interaction_kernels()
        
        # Cross-scale analysis
        total_responses = [scale_responses[f'scale_{s}']['total_response'] for s in scales]
        scale_dependence = np.std(total_responses) / np.mean(total_responses) if np.mean(total_responses) > 0 else 0.0
        
        return {
            'scales': scales,
            'scale_responses': scale_responses,
            'scale_dependence': scale_dependence,
            'preferred_scale': scales[np.argmax(total_responses)]
        }


def create_test_collective_response(grid_size: int = 32) -> CollectiveResponseFunction:
    """Create collective response function for testing"""
    from .manifold_field_equation import create_test_manifold_equation
    
    manifold_eq = create_test_manifold_equation(grid_size)
    
    return CollectiveResponseFunction(
        manifold_equation=manifold_eq,
        params=CollectiveResponseParameters(
            interaction_range=1.5,
            coupling_strength=1.0,
            curvature_sensitivity=0.3
        )
    )