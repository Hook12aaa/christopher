"""
Social Construct Field Theory Implementation

This implements the actual field theory of social constructs using ComFiT computational 
field theory framework. NO transformer concepts - pure field dynamics.

Mathematical Foundation: Social constructs as dynamic fields in conceptual space
with topological defects, field coupling, and collective response phenomena.
"""

import comfit
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import numpy as np
from typing import Dict, Any, Optional, Union, Callable
import logging

logger = logging.getLogger(__name__)


class SocialConstructField(comfit.BaseSystem):
    """
    Social construct field system implementing field theory of conceptual dynamics.
    
    This is NOT a transformer replacement - this is pure field physics applied
    to social constructs with:
    - Field evolution equations
    - Topological defects in meaning space
    - Collective response dynamics
    - Field coupling phenomena
    
    Mathematical Foundation:
    ∂φ/∂t = -δF/δφ + ξ(r,t)
    
    Where φ(r,t) is the social construct field, F is the free energy functional,
    and ξ represents stochastic fluctuations from social interactions.
    """
    
    def __init__(self,
                 conceptual_space_dimensions: int,
                 observational_domain_size: float,
                 field_coupling_strength: float,
                 collective_response_damping: float):
        """
        Initialize social construct field system.
        
        Args:
            conceptual_space_dimensions: Dimensionality of conceptual space (NOT embedding dim)
            observational_domain_size: Size of observational space domain
            field_coupling_strength: Strength of field coupling interactions
            collective_response_damping: Damping for collective response dynamics
        """
        # Initialize field domain in observational space
        super().__init__(
            dim=conceptual_space_dimensions,
            xRes=64,  # Spatial resolution for field calculations
            xmin=-observational_domain_size/2,
            xmax=observational_domain_size/2
        )
        
        self.conceptual_space_dims = conceptual_space_dimensions
        self.field_coupling_strength = field_coupling_strength
        self.collective_response_damping = collective_response_damping
        
        # Social construct field variables
        self.construct_field = None  # Main social construct field φ(r,t)
        self.meaning_density = None  # Local meaning density ρ(r,t)
        self.collective_response = None  # Collective response field Ψ(r,t)
        
        # Field coupling parameters
        self.coupling_matrix = self._initialize_field_coupling_matrix()
        self.topological_charge_density = None
        
        # Initialize field configuration
        self._initialize_social_construct_fields()
        
        logger.info(f"Initialized SocialConstructField in {conceptual_space_dimensions}D conceptual space")
    
    def _initialize_field_coupling_matrix(self) -> jnp.ndarray:
        """
        Initialize field coupling matrix for multi-field interactions.
        
        This represents how different aspects of social constructs couple
        through field interactions (NOT attention mechanisms).
        """
        # Field coupling follows physical principles, not transformer attention
        coupling = jnp.eye(self.conceptual_space_dims, dtype=complex)
        
        # Add field interaction terms based on conceptual proximity
        for i in range(self.conceptual_space_dims):
            for j in range(i+1, self.conceptual_space_dims):
                # Field coupling strength decays with conceptual distance
                distance = abs(i - j)
                coupling_strength = self.field_coupling_strength * jnp.exp(-distance/10.0)
                
                # Complex coupling for field interference
                coupling = coupling.at[i, j].set(coupling_strength * jnp.exp(1j * 0.1 * distance))
                coupling = coupling.at[j, i].set(jnp.conj(coupling.at[i, j].get()))
        
        return coupling
    
    def _initialize_social_construct_fields(self):
        """
        Initialize social construct field configuration.
        
        Sets up initial field state with small random fluctuations
        representing the baseline conceptual landscape.
        """
        # Main construct field - complex field representing conceptual states
        field_shape = (self.xRes,) if self.dim == 1 else (self.xRes, self.yRes) if self.dim == 2 else (self.xRes, self.yRes, self.zRes)
        
        # Initialize with small random fluctuations around equilibrium
        self.construct_field = 0.1 * (np.random.randn(*field_shape) + 
                                    1j * np.random.randn(*field_shape))
        
        # Meaning density field - represents local conceptual density
        self.meaning_density = 0.5 + 0.1 * np.random.randn(*field_shape)
        
        # Collective response field - represents social interaction effects
        self.collective_response = 0.1 * (np.random.randn(*field_shape) + 
                                        1j * np.random.randn(*field_shape))
        
        # Initialize topological charge density
        self.topological_charge_density = np.zeros(field_shape)
    
    def evolve_fields(self, dt: float, num_steps: int = 1):
        """
        Evolve social construct fields according to field equations.
        
        Implements the field evolution equation:
        ∂φ/∂t = -δF/δφ + ξ(r,t) + collective_coupling
        
        Args:
            dt: Time step
            num_steps: Number of evolution steps
        """
        for step in range(num_steps):
            # Compute field derivatives using JAX automatic differentiation
            field_gradient = self._compute_field_gradient()
            
            # Compute collective response coupling
            collective_coupling = self._compute_collective_response_coupling()
            
            # Add stochastic fluctuations from social interactions
            stochastic_noise = self._generate_social_noise(dt)
            
            # Field evolution step
            self.construct_field += dt * (
                -field_gradient + 
                collective_coupling + 
                stochastic_noise
            )
            
            # Evolve meaning density field
            meaning_evolution = self._compute_meaning_density_evolution()
            self.meaning_density += dt * meaning_evolution
            
            # Evolve collective response field
            response_evolution = self._compute_collective_response_evolution()
            self.collective_response += dt * response_evolution
            
            # Update topological defects
            self._update_topological_defects()
    
    def _compute_field_gradient(self) -> jnp.ndarray:
        """
        Compute field gradient ∂F/∂φ using physical field theory principles.
        
        This implements the variational derivative of the free energy
        functional F[φ] with respect to the field φ.
        """
        # Convert to JAX arrays for computation
        field_jax = jnp.array(self.construct_field)
        meaning_jax = jnp.array(self.meaning_density)
        
        # Compute local potential derivative analytically
        phi_magnitude_sq = jnp.abs(field_jax)**2
        
        # Derivative of Landau-Ginzburg potential: d/dφ[-0.5|φ|² + 0.25|φ|⁴]
        # For complex field: ∂/∂φ* = -0.5φ + 0.25|φ|²φ
        potential_grad = -0.5 * field_jax + 0.25 * phi_magnitude_sq * field_jax
        
        # Coupling to meaning density: d/dφ[-φ*ρ] = -ρ
        meaning_coupling_grad = -meaning_jax
        
        # Add gradient energy terms (spatial derivatives)
        if self.dim >= 1:
            field_laplacian = self._compute_laplacian(field_jax)
            gradient_term = -0.1 * field_laplacian  # Gradient energy coefficient
        else:
            gradient_term = 0
        
        return potential_grad + meaning_coupling_grad + gradient_term
    
    def _compute_laplacian(self, field: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Laplacian of field using finite differences.
        
        This represents the gradient energy contribution to field dynamics.
        """
        if self.dim == 1:
            # 1D Laplacian
            laplacian = jnp.roll(field, 1) - 2*field + jnp.roll(field, -1)
            return laplacian / (self.dx**2)
        elif self.dim == 2:
            # 2D Laplacian
            laplacian = (jnp.roll(field, 1, axis=0) + jnp.roll(field, -1, axis=0) +
                        jnp.roll(field, 1, axis=1) + jnp.roll(field, -1, axis=1) - 4*field)
            return laplacian / (self.dx**2)
        else:
            # 3D Laplacian
            laplacian = (jnp.roll(field, 1, axis=0) + jnp.roll(field, -1, axis=0) +
                        jnp.roll(field, 1, axis=1) + jnp.roll(field, -1, axis=1) +
                        jnp.roll(field, 1, axis=2) + jnp.roll(field, -1, axis=2) - 6*field)
            return laplacian / (self.dx**2)
    
    def _compute_collective_response_coupling(self) -> jnp.ndarray:
        """
        Compute coupling between individual field and collective response.
        
        This represents how individual conceptual changes influence and are
        influenced by collective social dynamics.
        """
        # Collective response affects individual field evolution
        field_jax = jnp.array(self.construct_field)
        response_jax = jnp.array(self.collective_response)
        
        # Coupling strength depends on local meaning density
        meaning_jax = jnp.array(self.meaning_density)
        coupling_strength = 0.1 * meaning_jax
        
        # Non-linear coupling between fields
        coupling = coupling_strength * (response_jax * jnp.conj(field_jax) + 
                                      jnp.conj(response_jax) * field_jax)
        
        return coupling
    
    def _generate_social_noise(self, dt: float) -> jnp.ndarray:
        """
        Generate stochastic noise representing social interaction fluctuations.
        
        This implements the ξ(r,t) term representing random social interactions
        that drive field evolution.
        """
        field_shape = self.construct_field.shape
        
        # Correlated noise with appropriate strength
        noise_strength = jnp.sqrt(2 * 0.01 * dt)  # Fluctuation-dissipation theorem
        
        real_noise = noise_strength * np.random.randn(*field_shape)
        imag_noise = noise_strength * np.random.randn(*field_shape)
        
        return real_noise + 1j * imag_noise
    
    def _compute_meaning_density_evolution(self) -> jnp.ndarray:
        """
        Compute evolution of meaning density field.
        
        Meaning density evolves based on field gradients and social interactions.
        """
        field_magnitude_sq = jnp.abs(jnp.array(self.construct_field))**2
        meaning_jax = jnp.array(self.meaning_density)
        
        # Meaning density follows field strength but with diffusion
        evolution = 0.1 * (field_magnitude_sq - meaning_jax)
        
        # Add diffusion term
        meaning_laplacian = self._compute_laplacian(meaning_jax)
        diffusion = 0.01 * meaning_laplacian
        
        return evolution + diffusion
    
    def _compute_collective_response_evolution(self) -> jnp.ndarray:
        """
        Compute evolution of collective response field.
        
        Collective response evolves with damping and coupling to main field.
        """
        response_jax = jnp.array(self.collective_response)
        field_jax = jnp.array(self.construct_field)
        
        # Damped response with coupling to main field
        damping = -self.collective_response_damping * response_jax
        coupling = 0.05 * field_jax
        
        # Add noise for stochastic collective behavior
        noise_strength = 0.01
        field_shape = self.collective_response.shape
        noise = noise_strength * (np.random.randn(*field_shape) + 
                                1j * np.random.randn(*field_shape))
        
        return damping + coupling + noise
    
    def _update_topological_defects(self):
        """
        Identify and track topological defects in the social construct field.
        
        Topological defects represent stable configurations in conceptual space
        (e.g., persistent social beliefs, cultural patterns).
        """
        field_jax = jnp.array(self.construct_field)
        
        # Compute topological charge density
        if self.dim >= 2:
            # For 2D, compute winding number density
            phase = jnp.angle(field_jax)
            
            # Compute phase gradients
            phase_x = jnp.gradient(phase, axis=0)
            phase_y = jnp.gradient(phase, axis=1)
            
            # Topological charge density (winding number)
            charge_density = (phase_x * jnp.gradient(phase_y, axis=1) - 
                            phase_y * jnp.gradient(phase_x, axis=0)) / (2 * jnp.pi)
            
            self.topological_charge_density = np.array(charge_density)
    
    def compute_conceptual_charge(self, 
                                concept_location: jnp.ndarray,
                                observational_state: float) -> complex:
        """
        Compute conceptual charge at specific location in conceptual space.
        
        This is the field theory analog of conceptual charge - represents
        the field distortion created by a concept at a given location.
        
        Args:
            concept_location: Position in conceptual space
            observational_state: Current observational state
            
        Returns:
            Complex conceptual charge value
        """
        # Interpolate field value at concept location
        field_value = self._interpolate_field_at_location(concept_location)
        
        # Compute local field gradient
        field_gradient = self._compute_local_field_gradient(concept_location)
        
        # Conceptual charge includes field value and its gradient
        charge_magnitude = jnp.abs(field_value) * (1 + observational_state)
        charge_phase = jnp.angle(field_value) + 0.1 * jnp.linalg.norm(field_gradient)
        
        return charge_magnitude * jnp.exp(1j * charge_phase)
    
    def _interpolate_field_at_location(self, location: jnp.ndarray) -> complex:
        """Interpolate field value at arbitrary location in conceptual space."""
        # For simplicity, use nearest neighbor interpolation
        # In practice, would use proper interpolation schemes
        
        if self.dim == 1:
            idx = int(jnp.clip((location[0] - self.xmin) / self.dx, 0, self.xRes-1))
            return self.construct_field[idx]
        elif self.dim == 2:
            idx_x = int(jnp.clip((location[0] - self.xmin) / self.dx, 0, self.xRes-1))
            idx_y = int(jnp.clip((location[1] - self.ymin) / self.dy, 0, self.yRes-1))
            return self.construct_field[idx_x, idx_y]
        else:
            # 3D case
            idx_x = int(jnp.clip((location[0] - self.xmin) / self.dx, 0, self.xRes-1))
            idx_y = int(jnp.clip((location[1] - self.ymin) / self.dy, 0, self.yRes-1))
            idx_z = int(jnp.clip((location[2] - self.zmin) / self.dz, 0, self.zRes-1))
            return self.construct_field[idx_x, idx_y, idx_z]
    
    def _compute_local_field_gradient(self, location: jnp.ndarray) -> jnp.ndarray:
        """Compute field gradient at specific location."""
        # Use finite differences around the location
        delta = 0.1
        
        if self.dim == 1:
            loc_plus = location + jnp.array([delta])
            loc_minus = location - jnp.array([delta])
            field_plus = self._interpolate_field_at_location(loc_plus)
            field_minus = self._interpolate_field_at_location(loc_minus)
            return (field_plus - field_minus) / (2 * delta)
        else:
            # Multi-dimensional gradient
            gradient = []
            for i in range(self.dim):
                loc_plus = location.at[i].add(delta)
                loc_minus = location.at[i].add(-delta)
                field_plus = self._interpolate_field_at_location(loc_plus)
                field_minus = self._interpolate_field_at_location(loc_minus)
                gradient.append((field_plus - field_minus) / (2 * delta))
            return jnp.array(gradient)
    
    def analyze_field_properties(self) -> Dict[str, Any]:
        """
        Analyze current properties of the social construct field.
        
        Returns:
            Dictionary with field analysis including energy, topological charges, etc.
        """
        field_jax = jnp.array(self.construct_field)
        meaning_jax = jnp.array(self.meaning_density)
        
        # Total field energy
        kinetic_energy = jnp.sum(jnp.abs(field_jax)**2)
        potential_energy = jnp.sum(-0.5 * jnp.abs(field_jax)**2 + 0.25 * jnp.abs(field_jax)**4)
        gradient_energy = jnp.sum(jnp.abs(self._compute_laplacian(field_jax))**2)
        total_energy = kinetic_energy + potential_energy + gradient_energy
        
        # Topological properties
        total_topological_charge = jnp.sum(self.topological_charge_density) if self.dim >= 2 else 0
        
        # Field correlations
        field_correlation_length = self._compute_correlation_length()
        
        # Collective response strength
        response_strength = jnp.mean(jnp.abs(jnp.array(self.collective_response)))
        
        return {
            'total_energy': float(total_energy),
            'kinetic_energy': float(kinetic_energy),
            'potential_energy': float(potential_energy),
            'gradient_energy': float(gradient_energy),
            'total_topological_charge': float(total_topological_charge),
            'field_correlation_length': float(field_correlation_length),
            'collective_response_strength': float(response_strength),
            'mean_field_magnitude': float(jnp.mean(jnp.abs(field_jax))),
            'mean_meaning_density': float(jnp.mean(meaning_jax))
        }
    
    def _compute_correlation_length(self) -> float:
        """
        Compute field correlation length.
        
        This measures the typical scale over which field fluctuations are correlated.
        """
        field_jax = jnp.array(self.construct_field)
        
        # Compute power spectrum
        field_fft = jnp.fft.fftn(field_jax)
        power_spectrum = jnp.abs(field_fft)**2
        
        # Compute correlation function
        correlation_function = jnp.fft.ifftn(power_spectrum).real
        
        # Extract correlation length (simplified)
        if self.dim == 1:
            correlation_length = 1.0 / jnp.argmax(correlation_function[1:]) if jnp.max(correlation_function[1:]) > 0 else 1.0
        else:
            # For multi-dimensional case, use characteristic scale
            correlation_length = jnp.sqrt(jnp.sum(jnp.abs(field_jax)**2) / jnp.sum(jnp.abs(self._compute_laplacian(field_jax))**2))
        
        return correlation_length


class SocialConstructFieldFactory:
    """
    Factory for creating and managing social construct field systems.
    
    This replaces the old transformer-based approach with pure field theory.
    """
    
    def __init__(self,
                 default_conceptual_dimensions: int = 3,
                 default_domain_size: float = 10.0):
        """
        Initialize field factory.
        
        Args:
            default_conceptual_dimensions: Default dimensionality of conceptual space
            default_domain_size: Default size of observational domain
        """
        self.default_conceptual_dimensions = default_conceptual_dimensions
        self.default_domain_size = default_domain_size
        self.active_fields = {}
        
        logger.info("Initialized SocialConstructFieldFactory for pure field theory")
    
    def create_field_system(self,
                          concept_context: str,
                          field_parameters: Optional[Dict[str, Any]] = None) -> SocialConstructField:
        """
        Create new social construct field system for given context.
        
        Args:
            concept_context: Context identifier for the field system
            field_parameters: Optional parameters for field configuration
            
        Returns:
            Configured SocialConstructField system
        """
        # Use default parameters if not provided
        params = field_parameters or {}
        
        conceptual_dims = params.get('conceptual_dimensions', self.default_conceptual_dimensions)
        domain_size = params.get('domain_size', self.default_domain_size)
        coupling_strength = params.get('coupling_strength', 0.1)
        damping = params.get('damping', 0.1)
        
        # Create field system
        field_system = SocialConstructField(
            conceptual_space_dimensions=conceptual_dims,
            observational_domain_size=domain_size,
            field_coupling_strength=coupling_strength,
            collective_response_damping=damping
        )
        
        # Store for management
        self.active_fields[concept_context] = field_system
        
        return field_system
    
    def evolve_all_fields(self, dt: float, num_steps: int = 1):
        """Evolve all active field systems."""
        for context, field_system in self.active_fields.items():
            field_system.evolve_fields(dt, num_steps)
    
    def get_field_system(self, concept_context: str) -> Optional[SocialConstructField]:
        """Get existing field system for context."""
        return self.active_fields.get(concept_context)
    
    def analyze_all_fields(self) -> Dict[str, Dict[str, Any]]:
        """Analyze all active field systems."""
        analysis = {}
        for context, field_system in self.active_fields.items():
            analysis[context] = field_system.analyze_field_properties()
        return analysis