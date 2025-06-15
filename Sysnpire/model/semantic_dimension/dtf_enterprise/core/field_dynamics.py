"""
Field Dynamics Engine - DTF Evolution and Steady-State Solutions

Handles the temporal evolution of DTF fields and computation of steady-state
solutions for semantic basis function generation.

MATHEMATICAL FOUNDATION:
Field Evolution: τdu/dt = -u + h + S + ∫w(x-x')f(u(x'))dx'
Steady-State: 0 = -u* + h + S + ∫w(x-x')f(u*(x'))dx'

This module is essential for:
- Computing semantic basis functions φᵢ(x) from DTF steady-states
- Field evolution for dynamic semantic processing
- Stability analysis and convergence validation
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, root
import warnings
import sys
from pathlib import Path

# Ensure project imports work
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


class FieldDynamicsEngine:
    """
    Engine for DTF field evolution and steady-state computation.
    
    This class handles the computational aspects of DTF field dynamics,
    including numerical integration, steady-state solving, and stability analysis.
    """
    
    def __init__(self,
                 integration_method: str = "RK45",
                 steady_state_method: str = "hybr",
                 convergence_tolerance: float = 1e-6,
                 max_iterations: int = 1000):
        """
        Initialize field dynamics engine.
        
        Args:
            integration_method: ODE integration method ("RK45", "Radau", "BDF")
            steady_state_method: Steady-state solver method ("hybr", "lm", "broyden1")
            convergence_tolerance: Convergence tolerance for solvers
            max_iterations: Maximum iterations for steady-state solver
        """
        self.integration_method = integration_method
        self.steady_state_method = steady_state_method
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        
        # Performance tracking
        self.last_integration_time = None
        self.last_steady_state_time = None
        self.convergence_history = []
        
        logger.debug(f"FieldDynamicsEngine initialized - integration: {integration_method}, "
                    f"steady_state: {steady_state_method}, tolerance: {convergence_tolerance}")
    
    def evolve_field(self,
                    dtf_core,  # DTFMathematicalCore instance
                    initial_state: np.ndarray,
                    external_input: np.ndarray,
                    interaction_kernel: Callable[[np.ndarray], float],
                    positions: np.ndarray,
                    time_span: Tuple[float, float],
                    eval_times: Optional[np.ndarray] = None,
                    **integration_kwargs) -> Dict[str, Any]:
        """
        Evolve DTF field through time using numerical integration.
        
        Solves: τdu/dt = -u + h + S + ∫w(x-x')f(u(x'))dx'
        
        Args:
            dtf_core: DTF mathematical core for field computations
            initial_state: Initial field state u(x,0)
            external_input: External input S(x,t) (assumed constant)
            interaction_kernel: Lateral interaction function w(x-x')
            positions: Spatial positions for field evaluation
            time_span: (start_time, end_time) for integration
            eval_times: Specific times for field evaluation
            **integration_kwargs: Additional arguments for ODE solver
            
        Returns:
            Dict containing field evolution results
        """
        import time
        start_time = time.time()
        
        try:
            def field_dynamics(t, u):
                """Field dynamics function for ODE solver."""
                # Compute lateral interaction term
                lateral_term = dtf_core.compute_lateral_interaction_term(
                    u, interaction_kernel, positions
                )
                
                # Compute field derivative
                return dtf_core.field_derivative(u, external_input, lateral_term)
            
            # Set up integration parameters
            integration_params = {
                'rtol': 1e-8,
                'atol': 1e-10,
                'max_step': np.inf,
                **integration_kwargs
            }
            
            # Perform integration
            solution = solve_ivp(
                field_dynamics,
                time_span,
                initial_state,
                method=self.integration_method,
                t_eval=eval_times,
                **integration_params
            )
            
            self.last_integration_time = time.time() - start_time
            
            if solution.success:
                # Compute additional diagnostics
                final_state = solution.y[:, -1] if solution.y.shape[1] > 0 else initial_state
                
                # Energy/stability metrics
                field_energy = np.sum(final_state**2)
                max_activation = np.max(final_state)
                min_activation = np.min(final_state)
                
                return {
                    'success': True,
                    'times': solution.t,
                    'field_evolution': solution.y.T,  # Shape: (time, space)
                    'final_state': final_state,
                    'integration_time': self.last_integration_time,
                    'n_evaluations': solution.nfev,
                    'diagnostics': {
                        'field_energy': float(field_energy),
                        'max_activation': float(max_activation),
                        'min_activation': float(min_activation),
                        'activation_range': float(max_activation - min_activation)
                    },
                    'solver_message': solution.message
                }
            else:
                logger.error(f"Field evolution failed: {solution.message}")
                return {
                    'success': False,
                    'error': solution.message,
                    'times': getattr(solution, 't', []),
                    'field_evolution': getattr(solution, 'y', np.array([])).T,
                    'final_state': initial_state,
                    'integration_time': self.last_integration_time
                }
                
        except Exception as e:
            logger.error(f"Field evolution error: {e}")
            return {
                'success': False,
                'error': str(e),
                'times': [],
                'field_evolution': np.array([]),
                'final_state': initial_state,
                'integration_time': time.time() - start_time
            }
    
    def compute_steady_state(self,
                            dtf_core,  # DTFMathematicalCore instance
                            external_input: np.ndarray,
                            interaction_kernel: Callable[[np.ndarray], float],
                            positions: np.ndarray,
                            initial_guess: Optional[np.ndarray] = None,
                            **solver_kwargs) -> Dict[str, Any]:
        """
        Compute DTF steady-state solution for basis function generation.
        
        Solves: 0 = -u* + h + S + ∫w(x-x')f(u*(x'))dx'
        
        This is the core method for generating semantic basis functions φᵢ(x).
        
        Args:
            dtf_core: DTF mathematical core
            external_input: External input pattern S(x)
            interaction_kernel: Lateral interaction function w(x-x')
            positions: Spatial positions
            initial_guess: Initial guess for steady-state
            **solver_kwargs: Additional solver parameters
            
        Returns:
            Dict containing steady-state solution and diagnostics
        """
        import time
        start_time = time.time()
        
        try:
            # Set initial guess
            if initial_guess is None:
                initial_guess = np.full(len(positions), dtf_core.resting_level)
            
            def steady_state_equation(u):
                """Steady-state equation: F(u) = 0"""
                lateral_term = dtf_core.compute_lateral_interaction_term(
                    u, interaction_kernel, positions
                )
                return -u + dtf_core.resting_level + external_input + lateral_term
            
            # Solver parameters
            solver_params = {
                'method': self.steady_state_method,
                'options': {
                    'xtol': self.convergence_tolerance,
                    'maxfev': self.max_iterations,
                    'diag': np.ones(len(initial_guess))  # Equal scaling
                },
                **solver_kwargs
            }
            
            # Solve steady-state equation
            solution = root(steady_state_equation, initial_guess, **solver_params)
            
            self.last_steady_state_time = time.time() - start_time
            
            if solution.success:
                steady_state = solution.x
                
                # Verify solution quality
                residual = steady_state_equation(steady_state)
                max_residual = np.max(np.abs(residual))
                rms_residual = np.sqrt(np.mean(residual**2))
                
                # Check convergence quality
                converged = max_residual < self.convergence_tolerance * 10
                
                # Compute diagnostics
                field_magnitude = np.linalg.norm(steady_state)
                peak_activation = np.max(steady_state)
                peak_location = positions[np.argmax(steady_state)]
                
                # Store convergence history
                self.convergence_history.append({
                    'max_residual': max_residual,
                    'rms_residual': rms_residual,
                    'iterations': solution.nfev,
                    'converged': converged
                })
                
                return {
                    'success': True,
                    'steady_state': steady_state,
                    'computation_time': self.last_steady_state_time,
                    'iterations': solution.nfev,
                    'convergence': {
                        'converged': converged,
                        'max_residual': float(max_residual),
                        'rms_residual': float(rms_residual),
                        'tolerance': self.convergence_tolerance
                    },
                    'diagnostics': {
                        'field_magnitude': float(field_magnitude),
                        'peak_activation': float(peak_activation),
                        'peak_location': peak_location.tolist() if hasattr(peak_location, 'tolist') else float(peak_location),
                        'activation_range': float(np.max(steady_state) - np.min(steady_state))
                    },
                    'solver_message': solution.message
                }
            else:
                logger.warning(f"Steady-state solver failed: {solution.message}")
                # Return best attempt
                return {
                    'success': False,
                    'steady_state': solution.x if hasattr(solution, 'x') else initial_guess,
                    'computation_time': self.last_steady_state_time,
                    'iterations': getattr(solution, 'nfev', 0),
                    'error': solution.message,
                    'convergence': {'converged': False}
                }
                
        except Exception as e:
            logger.error(f"Steady-state computation error: {e}")
            return {
                'success': False,
                'steady_state': initial_guess if initial_guess is not None else np.zeros(len(positions)),
                'computation_time': time.time() - start_time,
                'error': str(e),
                'convergence': {'converged': False}
            }
    
    def compute_field_jacobian(self,
                              dtf_core,  # DTFMathematicalCore instance
                              field_state: np.ndarray,
                              interaction_kernel: Callable[[np.ndarray], float],
                              positions: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix of DTF field dynamics for stability analysis.
        
        The Jacobian J[i,j] = ∂(du_i/dt)/∂u_j is essential for:
        - Linear stability analysis around steady-states
        - Eigenvalue computation for field modes
        - Convergence rate estimation
        
        Args:
            dtf_core: DTF mathematical core
            field_state: Current field state for linearization
            interaction_kernel: Lateral interaction function
            positions: Spatial positions
            
        Returns:
            Jacobian matrix [n_positions × n_positions]
        """
        n_positions = len(positions)
        jacobian = np.zeros((n_positions, n_positions))
        
        # Compute activation function derivative at current state
        activated = dtf_core.activation_function(field_state)
        activation_derivative = dtf_core.activation_gain * activated * (1 - activated)
        
        # Fill Jacobian matrix
        for i in range(n_positions):
            # Diagonal terms: self-decay
            jacobian[i, i] = -1.0 / dtf_core.tau
            
            # Off-diagonal terms: lateral interactions
            for j in range(n_positions):
                if i != j:
                    distance_vector = positions[i] - positions[j]
                    interaction_weight = interaction_kernel(distance_vector)
                    jacobian[i, j] += (interaction_weight * activation_derivative[j]) / dtf_core.tau
        
        return jacobian
    
    def analyze_stability(self,
                         dtf_core,  # DTFMathematicalCore instance
                         steady_state: np.ndarray,
                         interaction_kernel: Callable[[np.ndarray], float],
                         positions: np.ndarray) -> Dict[str, Any]:
        """
        Perform linear stability analysis of DTF steady-state.
        
        Computes eigenvalues of linearized dynamics to determine:
        - Stability (all eigenvalues negative real parts)
        - Convergence rate (largest real eigenvalue)
        - Oscillatory modes (complex eigenvalues)
        
        Args:
            dtf_core: DTF mathematical core
            steady_state: Steady-state solution to analyze
            interaction_kernel: Lateral interaction function
            positions: Spatial positions
            
        Returns:
            Dict containing stability analysis results
        """
        try:
            # Compute Jacobian at steady-state
            jacobian = self.compute_field_jacobian(
                dtf_core, steady_state, interaction_kernel, positions
            )
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(jacobian)
            
            # Analyze eigenvalues
            real_parts = np.real(eigenvalues)
            imag_parts = np.imag(eigenvalues)
            
            # Stability criteria
            is_stable = np.all(real_parts < 0)
            max_real_eigenvalue = np.max(real_parts)
            
            # Find modes
            stable_modes = np.sum(real_parts < -1e-10)
            marginal_modes = np.sum(np.abs(real_parts) < 1e-10)
            unstable_modes = np.sum(real_parts > 1e-10)
            
            # Oscillatory behavior
            oscillatory_modes = np.sum(np.abs(imag_parts) > 1e-10)
            dominant_frequency = np.max(np.abs(imag_parts)) if oscillatory_modes > 0 else 0.0
            
            # Convergence rate (from largest real eigenvalue)
            convergence_rate = -max_real_eigenvalue if max_real_eigenvalue < 0 else 0.0
            
            return {
                'is_stable': bool(is_stable),
                'eigenvalues': eigenvalues,
                'max_real_eigenvalue': float(max_real_eigenvalue),
                'convergence_rate': float(convergence_rate),
                'spectral_radius': float(np.max(np.abs(eigenvalues))),
                'mode_analysis': {
                    'stable_modes': int(stable_modes),
                    'marginal_modes': int(marginal_modes),
                    'unstable_modes': int(unstable_modes),
                    'oscillatory_modes': int(oscillatory_modes)
                },
                'oscillation_frequency': float(dominant_frequency),
                'jacobian_condition_number': float(np.linalg.cond(jacobian))
            }
            
        except Exception as e:
            logger.error(f"Stability analysis failed: {e}")
            return {
                'is_stable': False,
                'error': str(e),
                'eigenvalues': np.array([]),
                'max_real_eigenvalue': 0.0,
                'convergence_rate': 0.0
            }
    
    def compute_basis_function_from_input(self,
                                         dtf_core,  # DTFMathematicalCore instance
                                         input_pattern: np.ndarray,
                                         interaction_kernel: Callable[[np.ndarray], float],
                                         positions: np.ndarray) -> Dict[str, Any]:
        """
        Compute DTF-based basis function from input pattern.
        
        This is the key method for generating semantic basis functions φᵢ(x).
        Each input pattern generates a steady-state field that serves as a basis function.
        
        Args:
            dtf_core: DTF mathematical core
            input_pattern: Input pattern S(x) (e.g., from embedding)
            interaction_kernel: Lateral interaction function
            positions: Spatial positions
            
        Returns:
            Dict containing basis function and analysis
        """
        # Compute steady-state for this input pattern
        steady_state_result = self.compute_steady_state(
            dtf_core, input_pattern, interaction_kernel, positions
        )
        
        if not steady_state_result['success']:
            logger.warning("Steady-state computation failed for basis function")
            return {
                'basis_function': np.zeros(len(positions)),
                'success': False,
                'error': steady_state_result.get('error', 'Unknown error')
            }
        
        basis_function = steady_state_result['steady_state']
        
        # Analyze basis function properties
        stability_result = self.analyze_stability(
            dtf_core, basis_function, interaction_kernel, positions
        )
        
        # Compute basis function quality metrics
        peak_value = np.max(basis_function)
        peak_position = positions[np.argmax(basis_function)]
        effective_width = self._compute_effective_width(basis_function, positions)
        
        # Normalize basis function (optional)
        normalized_basis = basis_function / np.linalg.norm(basis_function) if np.linalg.norm(basis_function) > 0 else basis_function
        
        return {
            'basis_function': basis_function,
            'normalized_basis': normalized_basis,
            'success': True,
            'properties': {
                'peak_value': float(peak_value),
                'peak_position': peak_position.tolist() if hasattr(peak_position, 'tolist') else float(peak_position),
                'effective_width': float(effective_width),
                'field_magnitude': float(np.linalg.norm(basis_function)),
                'activation_range': float(np.max(basis_function) - np.min(basis_function))
            },
            'stability': stability_result,
            'computation_info': {
                'convergence': steady_state_result['convergence'],
                'computation_time': steady_state_result['computation_time'],
                'iterations': steady_state_result['iterations']
            }
        }
    
    def _compute_effective_width(self, field: np.ndarray, positions: np.ndarray) -> float:
        """
        Compute effective width of field activation.
        
        Uses second moment to estimate the spatial extent of the field.
        """
        if len(field) == 0 or np.all(field <= 0):
            return 0.0
        
        # Ensure field is positive for width computation
        field_positive = np.maximum(field - np.min(field), 0)
        
        if np.sum(field_positive) == 0:
            return 0.0
        
        # Compute center of mass
        if len(positions.shape) == 1:
            # 1D positions
            center_of_mass = np.sum(positions * field_positive) / np.sum(field_positive)
            # Compute second moment
            second_moment = np.sum(field_positive * (positions - center_of_mass)**2) / np.sum(field_positive)
        else:
            # Multi-dimensional positions
            center_of_mass = np.sum(positions * field_positive.reshape(-1, 1), axis=0) / np.sum(field_positive)
            # Compute second moment (average over dimensions)
            distances_squared = np.sum((positions - center_of_mass)**2, axis=1)
            second_moment = np.sum(field_positive * distances_squared) / np.sum(field_positive)
        
        # Effective width is standard deviation
        return float(np.sqrt(second_moment))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the dynamics engine."""
        recent_convergence = self.convergence_history[-10:] if self.convergence_history else []
        
        avg_iterations = np.mean([c['iterations'] for c in recent_convergence]) if recent_convergence else 0
        avg_residual = np.mean([c['max_residual'] for c in recent_convergence]) if recent_convergence else 0
        convergence_rate = np.mean([c['converged'] for c in recent_convergence]) if recent_convergence else 0
        
        return {
            'last_integration_time': self.last_integration_time,
            'last_steady_state_time': self.last_steady_state_time,
            'recent_performance': {
                'avg_iterations': float(avg_iterations),
                'avg_residual': float(avg_residual),
                'convergence_rate': float(convergence_rate),
                'total_computations': len(self.convergence_history)
            },
            'solver_settings': {
                'integration_method': self.integration_method,
                'steady_state_method': self.steady_state_method,
                'tolerance': self.convergence_tolerance,
                'max_iterations': self.max_iterations
            }
        }


# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing Field Dynamics Engine...")
    
    # Import DTF core for testing
    from .dtf_core import DTFMathematicalCore
    
    # Initialize components
    dtf_core = DTFMathematicalCore(tau=10.0, resting_level=-2.0)
    dynamics_engine = FieldDynamicsEngine()
    
    # Create test setup
    positions = np.linspace(-3, 3, 15).reshape(-1, 1)  # 1D spatial positions
    n_positions = len(positions)
    
    # Create Mexican hat interaction kernel
    interaction_kernel = dtf_core.create_mexican_hat_kernel(
        excitation_radius=0.8, inhibition_strength=0.4
    )
    
    # Test steady-state computation
    external_input = np.exp(-((positions.flatten() - 0.5)**2) / (2 * 0.5**2))  # Gaussian input
    
    logger.info("Testing steady-state computation...")
    steady_result = dynamics_engine.compute_steady_state(
        dtf_core, external_input, interaction_kernel, positions
    )
    
    logger.info(f"Steady-state: success={steady_result['success']}, "
               f"peak={np.max(steady_result['steady_state']):.3f}, "
               f"converged={steady_result['convergence']['converged']}")
    
    # Test stability analysis
    if steady_result['success']:
        logger.info("Testing stability analysis...")
        stability = dynamics_engine.analyze_stability(
            dtf_core, steady_result['steady_state'], interaction_kernel, positions
        )
        
        logger.info(f"Stability: stable={stability['is_stable']}, "
                   f"convergence_rate={stability['convergence_rate']:.3f}")
    
    # Test field evolution
    logger.info("Testing field evolution...")
    initial_state = np.random.randn(n_positions) * 0.1
    time_span = (0, 30)
    eval_times = np.linspace(0, 30, 51)
    
    evolution_result = dynamics_engine.evolve_field(
        dtf_core, initial_state, external_input, interaction_kernel, 
        positions, time_span, eval_times
    )
    
    logger.info(f"Evolution: success={evolution_result['success']}, "
               f"final_peak={np.max(evolution_result['final_state']):.3f}, "
               f"integration_time={evolution_result['integration_time']:.3f}s")
    
    # Test basis function computation
    logger.info("Testing basis function computation...")
    basis_result = dynamics_engine.compute_basis_function_from_input(
        dtf_core, external_input, interaction_kernel, positions
    )
    
    logger.info(f"Basis function: success={basis_result['success']}, "
               f"peak={basis_result['properties']['peak_value']:.3f}, "
               f"width={basis_result['properties']['effective_width']:.3f}")
    
    # Performance metrics
    metrics = dynamics_engine.get_performance_metrics()
    logger.info(f"Performance: steady_state_time={metrics['last_steady_state_time']:.3f}s, "
               f"integration_time={metrics['last_integration_time']:.3f}s")
    
    logger.info("Field Dynamics Engine test complete!")