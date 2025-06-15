"""
DTF Mathematical Core - Pure Field Theory Implementation

Implements the fundamental Dynamic Field Theory equations and mathematical
operations without any embedding model dependencies.

MATHEMATICAL FOUNDATION:
Neural Field Equation: τu̇(x,t) = -u(x,t) + h + S(x,t) + ∫w(x-x')f(u(x',t))dx'

Where:
- τ: time constant
- u(x,t): field activation at position x and time t
- h: resting level
- S(x,t): external input
- w(x-x'): lateral interaction kernel
- f(u): activation function

Based on Amari (1977) and modern DTF implementations like Nengo-DTF.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
from scipy import integrate
from scipy.optimize import fsolve
import sys
from pathlib import Path

# Ensure project imports work
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


class DTFMathematicalCore:
    """
    Core DTF mathematical engine implementing fundamental field theory equations.
    
    This class provides pure mathematical DTF operations independent of any
    specific embedding model (BGE, MPNet, etc.). All field computations are
    based on established neural field theory.
    """
    
    def __init__(self, 
                 tau: float = 10.0,
                 resting_level: float = -5.0,
                 activation_threshold: float = 0.0,
                 activation_gain: float = 1.0):
        """
        Initialize DTF mathematical core with field parameters.
        
        Args:
            tau: Time constant for field dynamics
            resting_level: Resting level h in DTF equation
            activation_threshold: Threshold for activation function
            activation_gain: Gain parameter for activation function steepness
        """
        self.tau = tau
        self.resting_level = resting_level
        self.activation_threshold = activation_threshold
        self.activation_gain = activation_gain
        
        logger.debug(f"DTF Core initialized - τ={tau}, h={resting_level}, "
                    f"θ={activation_threshold}, β={activation_gain}")
    
    def activation_function(self, u: np.ndarray) -> np.ndarray:
        """
        DTF sigmoid activation function f(u).
        
        Standard DTF activation: f(u) = 1 / (1 + exp(-β(u - θ)))
        
        Args:
            u: Field activation values
            
        Returns:
            Activated field values
        """
        return 1.0 / (1.0 + np.exp(-self.activation_gain * (u - self.activation_threshold)))
    
    def field_derivative(self, 
                        u: np.ndarray,
                        external_input: np.ndarray,
                        lateral_interaction: np.ndarray) -> np.ndarray:
        """
        Compute field derivative du/dt from DTF equation.
        
        Implements: du/dt = (-u + h + S + ∫w(x-x')f(u(x'))dx') / τ
        
        Args:
            u: Current field state
            external_input: External input S(x,t)
            lateral_interaction: Integrated lateral interaction term
            
        Returns:
            Field derivative du/dt
        """
        return (-u + self.resting_level + external_input + lateral_interaction) / self.tau
    
    def compute_lateral_interaction_term(self,
                                       field_state: np.ndarray,
                                       interaction_kernel: Callable[[np.ndarray], float],
                                       positions: np.ndarray) -> np.ndarray:
        """
        Compute lateral interaction integral ∫w(x-x')f(u(x'))dx'.
        
        This is the core DTF computation that determines how different field
        locations interact with each other.
        
        Args:
            field_state: Current field activation u(x')
            interaction_kernel: Lateral interaction function w(x-x')
            positions: Spatial positions for integration
            
        Returns:
            Lateral interaction contribution for each position
        """
        activated_field = self.activation_function(field_state)
        lateral_terms = np.zeros_like(field_state)
        
        for i, pos_i in enumerate(positions):
            interaction_sum = 0.0
            
            for j, pos_j in enumerate(positions):
                if i != j:  # Don't include self-interaction
                    distance_vector = pos_i - pos_j
                    interaction_weight = interaction_kernel(distance_vector)
                    interaction_sum += interaction_weight * activated_field[j]
            
            lateral_terms[i] = interaction_sum
        
        return lateral_terms
    
    def solve_steady_state(self,
                          external_input: np.ndarray,
                          interaction_kernel: Callable[[np.ndarray], float],
                          positions: np.ndarray,
                          initial_guess: Optional[np.ndarray] = None,
                          max_iterations: int = 100,
                          tolerance: float = 1e-6) -> np.ndarray:
        """
        Solve for DTF steady-state solution.
        
        Finds u* such that: 0 = -u* + h + S + ∫w(x-x')f(u*(x'))dx'
        
        This is essential for computing semantic basis functions from DTF dynamics.
        
        Args:
            external_input: External input pattern S(x)
            interaction_kernel: Lateral interaction function w(x-x')
            positions: Spatial positions
            initial_guess: Initial guess for field state
            max_iterations: Maximum solver iterations
            tolerance: Convergence tolerance
            
        Returns:
            Steady-state field solution u*
        """
        if initial_guess is None:
            initial_guess = np.zeros(len(positions))
        
        def steady_state_equation(u):
            """Equation to solve: F(u) = 0 where F(u) = -u + h + S + lateral"""
            lateral_term = self.compute_lateral_interaction_term(u, interaction_kernel, positions)
            return -u + self.resting_level + external_input + lateral_term
        
        try:
            # Use scipy's nonlinear solver
            solution = fsolve(steady_state_equation, initial_guess, 
                            xtol=tolerance, maxfev=max_iterations)
            
            # Verify convergence
            residual = steady_state_equation(solution)
            max_residual = np.max(np.abs(residual))
            
            if max_residual > tolerance * 10:  # Allow some tolerance relaxation
                logger.warning(f"DTF steady-state solver may not have converged: "
                             f"max_residual={max_residual:.2e}")
            
            return solution
            
        except Exception as e:
            logger.error(f"DTF steady-state solver failed: {e}")
            # Return a reasonable fallback
            return np.full_like(initial_guess, self.resting_level)
    
    def evolve_field(self,
                    initial_state: np.ndarray,
                    external_input: np.ndarray,
                    interaction_kernel: Callable[[np.ndarray], float],
                    positions: np.ndarray,
                    time_steps: np.ndarray,
                    dt: float = 0.01) -> np.ndarray:
        """
        Evolve DTF field through time using numerical integration.
        
        Solves: τdu/dt = -u + h + S + ∫w(x-x')f(u(x'))dx'
        
        Args:
            initial_state: Initial field state u(x,0)
            external_input: External input S(x,t)
            interaction_kernel: Lateral interaction function
            positions: Spatial positions
            time_steps: Time points for evolution
            dt: Integration time step
            
        Returns:
            Field evolution u(x,t) for all time steps
        """
        def field_dynamics(t, u):
            """Field dynamics function for ODE solver."""
            lateral_term = self.compute_lateral_interaction_term(u, interaction_kernel, positions)
            return self.field_derivative(u, external_input, lateral_term)
        
        try:
            # Use scipy's ODE solver
            from scipy.integrate import solve_ivp
            
            solution = solve_ivp(
                field_dynamics,
                [time_steps[0], time_steps[-1]],
                initial_state,
                t_eval=time_steps,
                method='RK45',
                rtol=1e-6,
                atol=1e-8
            )
            
            if solution.success:
                return solution.y.T  # Transpose to get (time, space) shape
            else:
                logger.error(f"DTF field evolution failed: {solution.message}")
                return np.tile(initial_state, (len(time_steps), 1))
                
        except Exception as e:
            logger.error(f"DTF field evolution error: {e}")
            # Return static field as fallback
            return np.tile(initial_state, (len(time_steps), 1))
    
    def compute_field_stability(self,
                               steady_state: np.ndarray,
                               interaction_kernel: Callable[[np.ndarray], float],
                               positions: np.ndarray) -> Dict[str, Any]:
        """
        Analyze stability of DTF steady-state solution.
        
        Computes eigenvalues of linearized dynamics around steady-state to
        determine stability properties.
        
        Args:
            steady_state: Steady-state field solution
            interaction_kernel: Lateral interaction function
            positions: Spatial positions
            
        Returns:
            Dict containing stability analysis results
        """
        n_positions = len(positions)
        
        # Compute Jacobian matrix of linearized dynamics
        jacobian = np.zeros((n_positions, n_positions))
        
        # Activation function derivative at steady state
        activated = self.activation_function(steady_state)
        activation_derivative = self.activation_gain * activated * (1 - activated)
        
        for i in range(n_positions):
            # Diagonal terms: -1/τ (self-decay)
            jacobian[i, i] = -1.0 / self.tau
            
            # Off-diagonal terms: lateral interactions
            for j in range(n_positions):
                if i != j:
                    distance_vector = positions[i] - positions[j]
                    interaction_weight = interaction_kernel(distance_vector)
                    jacobian[i, j] += (interaction_weight * activation_derivative[j]) / self.tau
        
        # Compute eigenvalues for stability analysis
        eigenvalues = np.linalg.eigvals(jacobian)
        real_parts = np.real(eigenvalues)
        
        # Stability: all eigenvalues must have negative real parts
        is_stable = np.all(real_parts < 0)
        max_real_eigenvalue = np.max(real_parts)
        
        return {
            'is_stable': bool(is_stable),
            'eigenvalues': eigenvalues,
            'max_real_eigenvalue': float(max_real_eigenvalue),
            'spectral_radius': float(np.max(np.abs(eigenvalues))),
            'decay_rate': float(-max_real_eigenvalue) if max_real_eigenvalue < 0 else 0.0
        }
    
    def create_mexican_hat_kernel(self, 
                                 excitation_radius: float = 0.5,
                                 inhibition_strength: float = 0.3,
                                 excitation_strength: float = 1.0) -> Callable[[np.ndarray], float]:
        """
        Create a Mexican hat lateral interaction kernel.
        
        Standard DTF kernel: excitation for nearby positions, inhibition for distant ones.
        
        Args:
            excitation_radius: Radius of excitatory region
            inhibition_strength: Strength of inhibitory surround
            excitation_strength: Strength of excitatory center
            
        Returns:
            Lateral interaction function w(x-x')
        """
        def mexican_hat_kernel(distance_vector: np.ndarray) -> float:
            distance = np.linalg.norm(distance_vector)
            
            if distance <= excitation_radius:
                # Excitatory center
                return excitation_strength * np.exp(-distance**2 / (2 * excitation_radius**2))
            else:
                # Inhibitory surround
                surround_width = excitation_radius * 2
                return -inhibition_strength * np.exp(-(distance - excitation_radius)**2 / (2 * surround_width**2))
        
        return mexican_hat_kernel
    
    def create_gaussian_kernel(self, 
                              width: float = 1.0,
                              strength: float = 1.0) -> Callable[[np.ndarray], float]:
        """
        Create a Gaussian lateral interaction kernel.
        
        Pure excitatory kernel for simple DTF dynamics.
        
        Args:
            width: Gaussian width parameter
            strength: Interaction strength
            
        Returns:
            Gaussian interaction function
        """
        def gaussian_kernel(distance_vector: np.ndarray) -> float:
            distance_squared = np.sum(distance_vector**2)
            return strength * np.exp(-distance_squared / (2 * width**2))
        
        return gaussian_kernel
    
    def get_dtf_parameters(self) -> Dict[str, float]:
        """Get current DTF parameters."""
        return {
            'tau': self.tau,
            'resting_level': self.resting_level,
            'activation_threshold': self.activation_threshold,
            'activation_gain': self.activation_gain
        }
    
    def update_parameters(self, **kwargs):
        """Update DTF parameters."""
        if 'tau' in kwargs:
            self.tau = kwargs['tau']
        if 'resting_level' in kwargs:
            self.resting_level = kwargs['resting_level']
        if 'activation_threshold' in kwargs:
            self.activation_threshold = kwargs['activation_threshold']
        if 'activation_gain' in kwargs:
            self.activation_gain = kwargs['activation_gain']
        
        logger.debug(f"DTF parameters updated: {self.get_dtf_parameters()}")


# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing DTF Mathematical Core...")
    
    # Initialize DTF core
    dtf_core = DTFMathematicalCore(tau=10.0, resting_level=-2.0)
    
    # Create test spatial positions (1D for simplicity)
    positions = np.linspace(-5, 5, 21).reshape(-1, 1)
    n_positions = len(positions)
    
    # Create Mexican hat interaction kernel
    interaction_kernel = dtf_core.create_mexican_hat_kernel(
        excitation_radius=1.0,
        inhibition_strength=0.3
    )
    
    # Test activation function
    test_input = np.linspace(-5, 5, 11)
    activated = dtf_core.activation_function(test_input)
    logger.info(f"Activation function test: input range [{test_input[0]:.1f}, {test_input[-1]:.1f}] "
               f"→ output range [{activated.min():.3f}, {activated.max():.3f}]")
    
    # Test steady-state solver with localized input
    external_input = np.exp(-((positions.flatten() - 0)**2) / (2 * 1.0**2))  # Gaussian input at center
    
    steady_state = dtf_core.solve_steady_state(
        external_input=external_input,
        interaction_kernel=interaction_kernel,
        positions=positions
    )
    
    logger.info(f"Steady-state solution: peak={steady_state.max():.3f}, "
               f"center_value={steady_state[n_positions//2]:.3f}")
    
    # Test stability analysis
    stability = dtf_core.compute_field_stability(
        steady_state=steady_state,
        interaction_kernel=interaction_kernel,
        positions=positions
    )
    
    logger.info(f"Stability analysis: stable={stability['is_stable']}, "
               f"max_eigenvalue={stability['max_real_eigenvalue']:.3f}")
    
    # Test field evolution
    time_steps = np.linspace(0, 50, 101)
    initial_state = np.random.randn(n_positions) * 0.1
    
    evolution = dtf_core.evolve_field(
        initial_state=initial_state,
        external_input=external_input,
        interaction_kernel=interaction_kernel,
        positions=positions,
        time_steps=time_steps
    )
    
    logger.info(f"Field evolution: final_peak={evolution[-1].max():.3f}, "
               f"steady_state_reached={np.allclose(evolution[-1], steady_state, atol=0.1)}")
    
    logger.info("DTF Mathematical Core test complete!")