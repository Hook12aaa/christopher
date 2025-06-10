"""
Manifold Field Equation Implementation

Implements the evolution equation: ∂M/∂t = ∇²M + F[M] + Σᵢ T[Qᵢ]

This governs how the product manifold evolves through time based on:
- Diffusion (∇²M): Smoothing of field gradients
- Nonlinear self-interaction (F[M]): Manifold-dependent dynamics  
- Charge sources (Σᵢ T[Qᵢ]): External driving from conceptual charges

Creates emergent collective phenomena and stable geometric structures.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import warnings

# Import transformation operator and conceptual charge if available
try:
    from .transformation_operator import TransformationOperator
    from ..core_mathematics.conceptual_charge import ConceptualCharge
except ImportError:
    # Fallback for development
    TransformationOperator = Any
    ConceptualCharge = Any


@dataclass
class ManifoldParameters:
    """Parameters controlling manifold field evolution"""
    diffusion_coefficient: float = 0.1  # Strength of ∇²M term
    nonlinear_coupling: float = 0.5     # Strength of F[M] nonlinearity
    damping_coefficient: float = 0.01   # Energy dissipation rate
    source_coupling: float = 1.0        # Coupling strength to charge sources
    field_normalization: float = 1.0    # Global field normalization
    max_field_magnitude: float = 10.0   # Prevent numerical overflow


@dataclass  
class EvolutionState:
    """Complete state of manifold evolution"""
    field: np.ndarray              # Current manifold field M(x,t)
    time: float                    # Current time t
    energy: float                  # Total field energy ∫|M|² dx
    source_strength: float         # Total source contribution
    peak_magnitude: float          # Maximum field magnitude
    center_of_mass: Tuple[float, float]  # Field center of mass


class ManifoldFieldEquation:
    """
    Implements manifold field evolution equation.
    
    Core equation: ∂M/∂t = ∇²M + F[M] + Σᵢ T[Qᵢ]
    
    The manifold field M(x,t) evolves through:
    1. Diffusion: ∇²M spreads field gradients  
    2. Nonlinear dynamics: F[M] creates stable structures
    3. Charge sources: T[Qᵢ] provide external driving
    4. Damping: Energy dissipation for stability
    """
    
    def __init__(self,
                 spatial_dimensions: Tuple[int, int] = (64, 64),
                 spatial_extent: Tuple[float, float] = (10.0, 10.0),
                 transformation_operator: Optional[TransformationOperator] = None,
                 params: Optional[ManifoldParameters] = None):
        """
        Initialize manifold field equation solver.
        
        Args:
            spatial_dimensions: Grid resolution (Nx, Ny)
            spatial_extent: Physical domain size (Lx, Ly)
            transformation_operator: For converting charges to sources
            params: Evolution parameters
        """
        self.spatial_dimensions = spatial_dimensions
        self.spatial_extent = spatial_extent
        self.params = params or ManifoldParameters()
        
        # Transformation operator for charge sources
        if transformation_operator is not None:
            self.transformation_operator = transformation_operator
        else:
            # Create default transformation operator
            from .transformation_operator import TransformationOperator
            self.transformation_operator = TransformationOperator(
                spatial_dimensions, spatial_extent
            )
        
        # Initialize spatial grids and operators
        self._initialize_spatial_operators()
        
        # Current manifold state
        self.field = np.zeros(spatial_dimensions, dtype=np.complex128)
        self.time = 0.0
        
        # Evolution history for analysis
        self.evolution_history: List[EvolutionState] = []
    
    def _initialize_spatial_operators(self):
        """Initialize spatial derivative operators"""
        Nx, Ny = self.spatial_dimensions
        Lx, Ly = self.spatial_extent
        
        # Spatial grids
        x = np.linspace(-Lx/2, Lx/2, Nx)
        y = np.linspace(-Ly/2, Ly/2, Ny)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Grid spacing for finite differences
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        
        # Wavenumber grids for spectral derivatives (periodic boundaries)
        kx = 2 * np.pi * np.fft.fftfreq(Nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2  # Laplacian in Fourier space
    
    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian ∇²field using spectral method.
        
        More accurate than finite differences for smooth fields.
        """
        field_fft = np.fft.fft2(field)
        laplacian_fft = -self.K2 * field_fft
        return np.fft.ifft2(laplacian_fft)
    
    def nonlinear_interaction(self, field: np.ndarray) -> np.ndarray:
        """
        Compute nonlinear self-interaction F[M].
        
        Implements: F[M] = λ|M|²M - γM
        - λ|M|²M: Nonlinear coupling creates stable solitonic structures
        - γM: Linear damping prevents unlimited growth
        """
        magnitude_squared = np.abs(field)**2
        
        # Nonlinear term: λ|M|²M (creates stable structures)
        nonlinear_term = self.params.nonlinear_coupling * magnitude_squared * field
        
        # Damping term: -γM (energy dissipation)
        damping_term = -self.params.damping_coefficient * field
        
        return nonlinear_term + damping_term
    
    def compute_source_term(self, 
                           charges: List[ConceptualCharge],
                           positions: Optional[List[Tuple[float, float]]] = None,
                           observational_states: Optional[List[float]] = None) -> np.ndarray:
        """
        Compute source term Σᵢ T[Qᵢ] from conceptual charges.
        
        Args:
            charges: List of conceptual charges
            positions: Positions for charge imprints
            observational_states: Override charge observational states
            
        Returns:
            Combined source term for manifold evolution
        """
        if not charges:
            return np.zeros(self.spatial_dimensions, dtype=np.complex128)
        
        # Transform charges to imprints
        source_imprint = self.transformation_operator.batch_transform(
            charges, positions, observational_states
        )
        
        return self.params.source_coupling * source_imprint
    
    def field_evolution_function(self, 
                                field: np.ndarray,
                                charges: List[ConceptualCharge],
                                positions: Optional[List[Tuple[float, float]]] = None,
                                observational_states: Optional[List[float]] = None) -> np.ndarray:
        """
        Complete field evolution: ∂M/∂t = ∇²M + F[M] + Σᵢ T[Qᵢ]
        
        Args:
            field: Current manifold field M(x,t)
            charges: Conceptual charges providing sources
            positions: Positions for charge imprints
            observational_states: Override charge observational states
            
        Returns:
            Time derivative ∂M/∂t
        """
        # Diffusion term: D∇²M
        diffusion_term = self.params.diffusion_coefficient * self.laplacian(field)
        
        # Nonlinear self-interaction: F[M]
        nonlinear_term = self.nonlinear_interaction(field)
        
        # External sources: Σᵢ T[Qᵢ]
        source_term = self.compute_source_term(charges, positions, observational_states)
        
        # Complete evolution equation
        dM_dt = diffusion_term + nonlinear_term + source_term
        
        # Apply field normalization and clipping for numerical stability
        return self._apply_stability_constraints(dM_dt)
    
    def _apply_stability_constraints(self, dM_dt: np.ndarray) -> np.ndarray:
        """Apply numerical stability constraints"""
        # Clip excessive derivatives
        max_derivative = self.params.max_field_magnitude / 0.01  # Assume dt ~ 0.01
        magnitude = np.abs(dM_dt)
        mask = magnitude > max_derivative
        
        if np.any(mask):
            # Normalize excessive derivatives
            dM_dt[mask] = dM_dt[mask] * (max_derivative / magnitude[mask])
            warnings.warn("Applied derivative clipping for numerical stability")
        
        return dM_dt
    
    def evolve_field(self, 
                    dt: float,
                    charges: List[ConceptualCharge],
                    positions: Optional[List[Tuple[float, float]]] = None,
                    observational_states: Optional[List[float]] = None,
                    method: str = 'RK45') -> EvolutionState:
        """
        Evolve manifold field by time step dt.
        
        Args:
            dt: Time step size
            charges: Conceptual charges providing sources
            positions: Positions for charge imprints  
            observational_states: Override charge observational states
            method: Integration method ('euler', 'RK45')
            
        Returns:
            New evolution state
        """
        if method == 'euler':
            # Simple Euler integration
            dM_dt = self.field_evolution_function(
                self.field, charges, positions, observational_states
            )
            new_field = self.field + dt * dM_dt
            
        elif method == 'RK45':
            # Adaptive Runge-Kutta integration (more accurate)
            
            def evolution_ode(t, field_flat):
                field_2d = field_flat.reshape(self.spatial_dimensions).astype(np.complex128)
                dM_dt = self.field_evolution_function(
                    field_2d, charges, positions, observational_states  
                )
                return dM_dt.flatten()
            
            # Solve ODE for one time step
            field_flat = self.field.flatten().astype(np.complex128)
            sol = solve_ivp(
                evolution_ode, 
                [self.time, self.time + dt], 
                field_flat,
                method='RK45',
                rtol=1e-6,
                atol=1e-8
            )
            
            new_field = sol.y[:, -1].reshape(self.spatial_dimensions)
            
        else:
            raise ValueError(f"Unknown integration method: {method}")
        
        # Update state
        self.field = new_field.astype(np.complex128)
        self.time += dt
        
        # Record evolution state
        evolution_state = self._compute_evolution_state()
        self.evolution_history.append(evolution_state)
        
        return evolution_state
    
    def _compute_evolution_state(self) -> EvolutionState:
        """Compute current evolution state metrics"""
        magnitude = np.abs(self.field)
        
        # Total energy
        energy = np.sum(magnitude**2) * self.dx * self.dy
        
        # Peak magnitude
        peak_magnitude = np.max(magnitude)
        
        # Center of mass
        total_magnitude = np.sum(magnitude)
        if total_magnitude > 0:
            center_x = np.sum(magnitude * self.X) / total_magnitude
            center_y = np.sum(magnitude * self.Y) / total_magnitude
        else:
            center_x = center_y = 0.0
        
        # Source strength (for current charges - will be updated by caller)
        source_strength = 0.0  # Placeholder
        
        return EvolutionState(
            field=self.field.copy(),
            time=self.time,
            energy=energy,
            source_strength=source_strength,
            peak_magnitude=peak_magnitude,
            center_of_mass=(center_x, center_y)
        )
    
    def initialize_field(self, 
                        initial_condition: str = 'gaussian',
                        amplitude: float = 1.0,
                        noise_level: float = 0.1) -> np.ndarray:
        """
        Initialize manifold field with specified initial condition.
        
        Args:
            initial_condition: Type ('gaussian', 'random', 'zero', 'soliton')
            amplitude: Initial amplitude scale
            noise_level: Random noise amplitude
            
        Returns:
            Initialized field
        """
        if initial_condition == 'zero':
            self.field = np.zeros(self.spatial_dimensions, dtype=np.complex128)
            
        elif initial_condition == 'gaussian':
            # Gaussian initial condition
            sigma = min(self.spatial_extent) / 4
            gaussian = amplitude * np.exp(-(self.X**2 + self.Y**2) / (2 * sigma**2))
            noise = noise_level * (np.random.randn(*self.spatial_dimensions) + 
                                 1j * np.random.randn(*self.spatial_dimensions))
            self.field = (gaussian + noise).astype(np.complex128)
            
        elif initial_condition == 'random':
            # Random initial condition
            self.field = (amplitude * (np.random.randn(*self.spatial_dimensions) + 
                                     1j * np.random.randn(*self.spatial_dimensions))).astype(np.complex128)
            
        elif initial_condition == 'soliton':
            # Approximate soliton solution
            sigma = min(self.spatial_extent) / 6
            r = np.sqrt(self.X**2 + self.Y**2)
            soliton = amplitude / np.cosh(r / sigma) * np.exp(1j * self.X / sigma)
            noise = noise_level * (np.random.randn(*self.spatial_dimensions) + 
                                 1j * np.random.randn(*self.spatial_dimensions))
            self.field = (soliton + noise).astype(np.complex128)
            
        else:
            raise ValueError(f"Unknown initial condition: {initial_condition}")
        
        # Reset time and history
        self.time = 0.0
        self.evolution_history = []
        
        return self.field
    
    def analyze_field_properties(self) -> Dict[str, Any]:
        """
        Analyze current field properties for stability and structure detection.
        
        Returns:
            Dictionary with field analysis results
        """
        magnitude = np.abs(self.field)
        phase = np.angle(self.field)
        
        # Basic properties
        total_energy = np.sum(magnitude**2) * self.dx * self.dy
        peak_magnitude = np.max(magnitude)
        mean_magnitude = np.mean(magnitude)
        
        # Spatial structure analysis
        # Center of mass
        total_magnitude = np.sum(magnitude)
        if total_magnitude > 0:
            center_x = np.sum(magnitude * self.X) / total_magnitude
            center_y = np.sum(magnitude * self.Y) / total_magnitude
        else:
            center_x = center_y = 0.0
        
        # Spatial spread (second moments)
        X_centered = self.X - center_x
        Y_centered = self.Y - center_y
        spread_x = np.sqrt(np.sum(magnitude * X_centered**2) / total_magnitude) if total_magnitude > 0 else 0.0
        spread_y = np.sqrt(np.sum(magnitude * Y_centered**2) / total_magnitude) if total_magnitude > 0 else 0.0
        
        # Phase coherence
        phase_gradients = np.gradient(phase)
        phase_gradient_magnitude = np.sqrt(phase_gradients[0]**2 + phase_gradients[1]**2)
        phase_coherence = 1.0 / (1.0 + np.mean(phase_gradient_magnitude))
        
        # Detect stable structures (local maxima)
        from scipy.ndimage import maximum_filter
        local_maxima = (magnitude == maximum_filter(magnitude, size=3))
        num_stable_structures = np.sum(local_maxima & (magnitude > 0.1 * peak_magnitude))
        
        return {
            'total_energy': total_energy,
            'peak_magnitude': peak_magnitude,
            'mean_magnitude': mean_magnitude,
            'center_x': center_x,
            'center_y': center_y,
            'spread_x': spread_x,
            'spread_y': spread_y,
            'phase_coherence': phase_coherence,
            'num_stable_structures': num_stable_structures,
            'field_magnitude': magnitude,
            'field_phase': phase
        }
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution history"""
        if not self.evolution_history:
            return {'message': 'No evolution history available'}
        
        times = [state.time for state in self.evolution_history]
        energies = [state.energy for state in self.evolution_history]
        peak_magnitudes = [state.peak_magnitude for state in self.evolution_history]
        
        return {
            'evolution_duration': times[-1] - times[0] if len(times) > 1 else 0.0,
            'final_energy': energies[-1],
            'energy_change': energies[-1] - energies[0] if len(energies) > 1 else 0.0,
            'max_peak_magnitude': max(peak_magnitudes),
            'num_evolution_steps': len(self.evolution_history),
            'times': times,
            'energies': energies,
            'peak_magnitudes': peak_magnitudes
        }


def create_test_manifold_equation(grid_size: int = 32) -> ManifoldFieldEquation:
    """Create manifold field equation for testing"""
    return ManifoldFieldEquation(
        spatial_dimensions=(grid_size, grid_size),
        spatial_extent=(5.0, 5.0),
        params=ManifoldParameters(
            diffusion_coefficient=0.1,
            nonlinear_coupling=0.3,
            damping_coefficient=0.05,
            source_coupling=1.0
        )
    )