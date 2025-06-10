import numpy as np
from typing import Dict, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from scipy.integrate import quad
from scipy.special import expit

class ConceptualCharge:
    """
    Implementation of complete conceptual charge as defined in the paper:
    
    Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)
    
    This represents the complete field-theoretic formulation where:
    - γ: Global field calibration factor
    - T(τ, C, s): Transformative potential tensor (trajectory-dependent)
    - E^trajectory(τ, s): Emotional trajectory integration
    - Φ^semantic(τ, s): Semantic field generation (dynamic, not static)
    - e^(iθ_total(τ,C,s)): Complete phase integration
    - Ψ_persistence(s-s₀): Observational persistence function
    """
    
    def __init__(self,
                 token: str,
                 semantic_vector: np.ndarray,
                 context: Optional[Dict] = None,
                 observational_state: float = 0.0,
                 gamma: float = 1.0):
        """
        Initialize a conceptual charge with complete field-theoretic formulation.
        
        Args:
            token: The token τ this charge represents
            semantic_vector: Base semantic embedding (1024-dimensional from BGE)
            context: Contextual environment C
            observational_state: Current observational state s
            gamma: Global field calibration factor γ
        """
        self.token = token
        self.semantic_vector = semantic_vector
        self.context = context or {}
        self.observational_state = observational_state
        self.gamma = gamma
        self.s_0 = 0.0  # Initial observational state
        
        # Initialize trajectory parameters
        self.trajectory_history = []
        self.frequency_evolution = {}
        self.phase_accumulation = {}
        
        # Initialize field parameters
        self._initialize_field_parameters()
    
    def _initialize_field_parameters(self):
        """Initialize field parameters for trajectory and emotional dynamics."""
        d = len(self.semantic_vector)
        
        # Trajectory parameters for T_i(τ,s)
        self.omega_base = np.random.uniform(0.1, 1.0, d)  # Base frequency evolution
        self.phi_base = np.random.uniform(0, 2*np.pi, d)  # Base phase relationships
        
        # Emotional trajectory parameters for E^trajectory(τ,s)
        self.alpha_emotional = np.random.uniform(0.5, 2.0, d)  # Emotional amplification
        self.sigma_emotional_sq = np.random.uniform(0.1, 1.0, d)  # Emotional selectivity
        self.v_emotional = np.random.randn(d)  # Emotional alignment vector
        
        # Semantic breathing parameters for Φ^semantic(τ,s)
        self.beta_breathing = np.random.uniform(0.1, 0.5, d)  # Breathing modulation depth
        self.w_weights = np.ones(d)  # Semantic weighting
        
        # Persistence parameters for Ψ_persistence(s-s₀)
        self.sigma_persistence_sq = 1.0  # Gaussian decay width
        self.alpha_persistence = 0.5  # Persistent component amplitude
        self.lambda_persistence = 0.1  # Exponential decay rate
        self.beta_persistence = 0.3  # Oscillatory frequency
    
    def trajectory_operator(self, s: float, dimension: int) -> complex:
        """
        Calculate trajectory operator T_i(τ,s) for dimension i.
        
        T_i(τ,s) = ∫₀ˢ ω_i(τ,s')·e^(iφ_i(τ,s')) ds'
        
        Args:
            s: Current observational state
            dimension: Dimension index i
            
        Returns:
            Complex trajectory operator value
        """
        def integrand(s_prime):
            omega = self.omega_base[dimension] * (1 + 0.1 * np.sin(s_prime))
            phi = self.phi_base[dimension] + 0.2 * s_prime
            return omega * np.exp(1j * phi)
        
        try:
            real_part, _ = quad(lambda x: integrand(x).real, 0, s)
            imag_part, _ = quad(lambda x: integrand(x).imag, 0, s)
            return complex(real_part, imag_part)
        except:
            # Fallback for numerical issues
            return complex(s * self.omega_base[dimension], s * 0.1)
    
    def emotional_trajectory_integration(self, s: float) -> np.ndarray:
        """
        Calculate emotional trajectory integration E^trajectory(τ,s).
        
        Implements trajectory-aware emotional resonance with accumulation.
        
        Args:
            s: Current observational state
            
        Returns:
            Emotional trajectory modulation array
        """
        d = len(self.semantic_vector)
        E_trajectory = np.zeros(d)
        
        for i in range(d):
            # Gaussian alignment component
            alignment = np.exp(-((self.semantic_vector[i] - self.v_emotional[i])**2) / 
                             (2 * self.sigma_emotional_sq[i]))
            
            # Trajectory resonance accumulation (simplified)
            trajectory_accumulation = 1.0 + 0.1 * s * np.exp(-0.1 * s)
            
            E_trajectory[i] = self.alpha_emotional[i] * alignment * trajectory_accumulation
            
        return E_trajectory
    
    def semantic_field_generation(self, s: float, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate semantic field function Φ^semantic(τ,s).
        
        Implements breathing constellation patterns across the narrative sky.
        
        Args:
            s: Current observational state
            x: Position in manifold (optional, defaults to semantic vector)
            
        Returns:
            Dynamic semantic field values
        """
        if x is None:
            x = self.semantic_vector
            
        d = len(self.semantic_vector)
        phi_semantic = np.zeros(d, dtype=complex)
        
        for i in range(d):
            # Trajectory-dependent breathing modulation (use modulo for index)
            traj_idx = i % len(self.omega_base)
            trajectory_integral = s * self.omega_base[traj_idx]  # Simplified integration
            breathing_modulation = 1 + self.beta_breathing[i] * np.cos(trajectory_integral + self.phi_base[traj_idx])
            
            # Field generation with trajectory coupling
            T_i = self.trajectory_operator(s, traj_idx)
            phi_semantic[i] = (self.w_weights[i] * T_i * x[i] * breathing_modulation * 
                             np.exp(1j * (self.phi_base[traj_idx] + 0.1 * s)))
            
        return phi_semantic
    
    def total_phase_integration(self, s: float) -> float:
        """
        Calculate complete phase integration θ_total(τ,C,s).
        
        θ_total = θ_semantic + θ_emotional + ∫ω_temporal ds' + θ_interaction + θ_field
        
        Args:
            s: Current observational state
            
        Returns:
            Total phase value
        """
        # Semantic phase contribution
        theta_semantic = np.angle(np.sum(self.semantic_vector[:3]))
        
        # Emotional phase contribution
        theta_emotional = np.angle(np.sum(self.v_emotional[:3]))
        
        # Temporal trajectory integration
        theta_temporal = s * np.mean(self.omega_base)
        
        # Interaction phase (context-dependent)
        theta_interaction = 0.1 * len(str(self.context))  # Simplified
        
        # Field phase
        theta_field = 0.05 * s
        
        return (theta_semantic + theta_emotional + theta_temporal + 
                theta_interaction + theta_field) % (2 * np.pi)
    
    def observational_persistence(self, s: float) -> float:
        """
        Calculate observational persistence Ψ_persistence(s-s₀).
        
        Dual-decay structure with Gaussian and exponential-cosine components.
        
        Args:
            s: Current observational state
            
        Returns:
            Persistence value
        """
        delta_s = s - self.s_0
        
        # Gaussian decay component (vivid recent chapters)
        gaussian_component = np.exp(-(delta_s**2) / (2 * self.sigma_persistence_sq))
        
        # Exponential-cosine component (persistent character traits)
        exp_cos_component = (self.alpha_persistence * 
                           np.exp(-self.lambda_persistence * delta_s) * 
                           np.cos(self.beta_persistence * delta_s))
        
        return gaussian_component + exp_cos_component
    
    def compute_complete_charge(self, s: Optional[float] = None) -> complex:
        """
        Compute the complete conceptual charge Q(τ, C, s).
        
        Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)
        
        Args:
            s: Observational state (defaults to current state)
            
        Returns:
            Complex-valued complete conceptual charge
        """
        if s is None:
            s = self.observational_state
            
        # Calculate each component
        T_magnitude = np.mean([abs(self.trajectory_operator(s, i)) for i in range(3)])  # Simplified tensor
        E_trajectory = np.mean(self.emotional_trajectory_integration(s))
        Phi_semantic = np.mean(np.abs(self.semantic_field_generation(s)))
        theta_total = self.total_phase_integration(s)
        psi_persistence = self.observational_persistence(s)
        
        # Complete charge calculation
        Q = (self.gamma * T_magnitude * E_trajectory * Phi_semantic * 
             np.exp(1j * theta_total) * psi_persistence)
        
        return Q
    
    def get_charge_magnitude(self) -> float:
        """Calculate the magnitude of the complete conceptual charge."""
        return abs(self.compute_complete_charge())
    
    def get_phase_factor(self) -> float:
        """Calculate the phase factor of the complete conceptual charge."""
        return np.angle(self.compute_complete_charge())
    
    def update_observational_state(self, new_s: float):
        """Update the observational state and record trajectory."""
        self.trajectory_history.append(self.observational_state)
        self.observational_state = new_s