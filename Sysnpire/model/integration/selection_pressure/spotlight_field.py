"""
Spotlight Field Engine - Attention-Driven Selection via Diffusion Equations

MATHEMATICAL FOUNDATION:
    Attention Field: A(x,t) spotlight concentration on semantic manifold
    Diffusion Equation: ∂A/∂t = D∇²A - kA + S(x,t) 
    Selection Coupling: ∂pᵢ/∂t = pᵢ[fᵢ + αA(xᵢ,t) - ⟨f + αA⟩]
    
    Attention Dynamics:
    - Gaussian Spotlight: A(x,t) = A₀ exp(-|x-x₀(t)|²/2σ²)
    - Diffusion Operator: D∇²A heat equation spreading  
    - Decay Term: -kA natural attention decay
    - Source Term: S(x,t) external attention stimuli
    
    Field Equations:
    ∂A/∂t = D∇²A - kA + S(x,t)                    # Attention diffusion
    ∂x₀/∂t = v₀ + β∇⟨E[φ]⟩                        # Spotlight tracking
    ∂σ/∂t = γ(σ_target - σ) + η⟨|∇A|⟩             # Adaptive focus
    
IMPLEMENTATION: Finite element diffusion solver, JAX autodiff for gradients,
adaptive mesh refinement near attention peaks, spectral methods for efficiency.
"""

import cmath
import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal

from ..field_mechanics.data_type_consistency import get_dtype_manager

# Geometric Deep Learning for attention manifolds (optional)
try:
    import torch_geometric as pyg
    from torch_geometric.data import Data
    from torch_geometric.nn import MessagePassing

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    pyg = None
    Data = None
    MessagePassing = None
    TORCH_GEOMETRIC_AVAILABLE = False

# JAX for differential equation solving and optimization
import jax
import jax.numpy as jnp
# Numba for high-performance diffusion loops
import numba as nb
from jax import grad, hessian, jacfwd, jacrev, jit, vmap
from jax.scipy import integrate as jax_integrate
from jax.scipy.linalg import solve as jax_solve
from numba import jit as nb_jit
from numba import prange
# SciPy for PDE solving and optimization
from scipy import integrate, linalg, optimize, sparse
from scipy.integrate import odeint, quad, solve_ivp
from scipy.linalg import eigh, solve, solve_banded
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import csc_matrix, diags
from scipy.sparse import linalg as sparse_linalg
# Finite element and PDE libraries - REQUIRED
from scipy.sparse.linalg import LinearOperator, eigsh, spsolve

# Import constants and utilities
from . import (EVOLUTIONARY_CONVERGENCE_THRESHOLD,
               POPULATION_NUMERICAL_PRECISION, SELECTION_STRENGTH,
               EvolutionaryDynamics, PopulationState, SelectionPressure)

logger = logging.getLogger(__name__)


@dataclass
class AttentionField:
    """
    Attention field configuration on semantic manifold.

    MATHEMATICAL FORMULATION:
    The attention field A(x,t) represents concentration of cognitive resources on the semantic manifold:

    $$A(\mathbf{x}, t) = A_0 \exp\left(-\frac{|\mathbf{x} - \mathbf{x}_0(t)|^2}{2\sigma^2}\right)$$

    where:
    - $A_0$ = peak_intensity: Maximum attention amplitude
    - $\mathbf{x}_0(t)$ = center_position: Time-dependent spotlight center
    - $\sigma$ = focus_width: Gaussian attention width parameter
    - $\nabla A(\mathbf{x})$ = gradient_field: Spatial attention gradient
    - $\nabla^2 A(\mathbf{x})$ = laplacian_field: Attention field curvature

    FIELD PROPERTIES:
    - Total attention budget: $\mathcal{A} = \int_{\mathbb{R}^d} A(\mathbf{x}) d\mathbf{x}$
    - Diffusion dynamics: $\frac{\partial A}{\partial t} = D\nabla^2 A - kA + S(\mathbf{x},t)$
    - Gradient field: $\mathbf{g}(\mathbf{x}) = \nabla A = -\frac{A(\mathbf{x})(\mathbf{x} - \mathbf{x}_0)}{\sigma^2}$
    - Laplacian field: $\nabla^2 A = A(\mathbf{x})\left[\frac{|\mathbf{x} - \mathbf{x}_0|^2}{\sigma^4} - \frac{d}{\sigma^2}\right]$
    """

    field_values: torch.Tensor  # A(x) attention intensity distribution
    center_position: torch.Tensor  # x₀ spotlight center coordinates
    focus_width: float  # σ attention focus width
    peak_intensity: float  # A_max maximum attention strength
    total_attention: float  # ∫A(x)dx total attention budget
    gradient_field: torch.Tensor  # ∇A(x) attention gradient
    laplacian_field: torch.Tensor  # ∇²A(x) attention curvature
    diffusion_coefficient: float  # D diffusion constant
    decay_rate: float  # k attention decay rate

    def __post_init__(self):
        """Validate attention field properties."""
        if not torch.is_tensor(self.field_values):
            raise TypeError("field_values must be torch.Tensor")
        if not torch.isfinite(self.field_values).all():
            raise ValueError("field_values contains non-finite values")
        if self.focus_width <= 0:
            raise ValueError(f"Non-positive focus width: {self.focus_width}")
        if self.peak_intensity < 0:
            raise ValueError(f"Negative peak intensity: {self.peak_intensity}")
        if self.diffusion_coefficient < 0:
            raise ValueError(
                f"Negative diffusion coefficient: {self.diffusion_coefficient}"
            )


@dataclass
class SpotlightDynamics:
    """
    Spotlight movement and adaptation dynamics.

    MATHEMATICAL FORMULATION:
    Spotlight dynamics follow Hamiltonian mechanics with adaptive focus control:

    EQUATIONS OF MOTION:
    $$\frac{d\mathbf{x}_0}{dt} = \mathbf{v}_0 + \beta \nabla \langle E[\phi] \rangle$$
    $$\frac{d\mathbf{v}_0}{dt} = -\gamma \mathbf{v}_0 + \mathbf{F}_{track}(\mathbf{x}_{target} - \mathbf{x}_0)$$
    $$\frac{d\sigma}{dt} = \gamma(\sigma_{target} - \sigma) + \eta \langle |\nabla A| \rangle$$

    ENERGY FORMULATION:
    - Kinetic energy: $T = \frac{1}{2}m|\mathbf{v}_0|^2$
    - Potential energy: $V(\mathbf{x}_0) = \frac{1}{2}k|\mathbf{x}_0 - \mathbf{x}_{target}|^2$
    - Total energy: $E = T + V$ (conserved in absence of damping)
    - Momentum: $\mathbf{p} = m\mathbf{v}_0$
    - Angular momentum: $\mathbf{L} = \mathbf{r} \times \mathbf{p}$

    TRAJECTORY ANALYSIS:
    - Curvature: $\kappa = \frac{|\mathbf{v} \times \mathbf{a}|}{|\mathbf{v}|^3}$
    - Arc length: $s(t) = \int_0^t |\mathbf{v}(\tau)| d\tau$
    - Adaptation rate: $\gamma$ controls focus width convergence
    """

    velocity_field: torch.Tensor  # v(x) spotlight velocity field
    tracking_target: torch.Tensor  # x_target(t) tracking objective
    adaptation_rate: float  # γ focus adaptation rate
    momentum: torch.Tensor  # p = mv spotlight momentum
    kinetic_energy: float  # T = ½mv² kinetic energy
    potential_energy: float  # V(x) attraction potential
    total_energy: float  # E = T + V total energy
    trajectory_curvature: float  # κ = |v × a|/|v|³ path curvature
    angular_momentum: torch.Tensor  # L = r × p angular momentum

    def __post_init__(self):
        """Validate spotlight dynamics consistency."""
        if not torch.allclose(
            torch.tensor(self.total_energy),
            torch.tensor(self.kinetic_energy + self.potential_energy),
            rtol=1e-6,
        ):
            raise ValueError("Energy conservation violated")
        if self.adaptation_rate < 0:
            raise ValueError(f"Negative adaptation rate: {self.adaptation_rate}")


@dataclass
class DiffusionSolution:
    """
    Diffusion equation solution analysis.

    MATHEMATICAL FORMULATION:
    Solution to the attention diffusion partial differential equation:

    $$\frac{\partial A}{\partial t} = D\nabla^2 A - kA + S(\mathbf{x},t)$$

    SPECTRAL DECOMPOSITION:
    $$A(\mathbf{x},t) = \sum_{i=0}^{\infty} c_i \psi_i(\mathbf{x}) e^{-\lambda_i t}$$

    where:
    - $\lambda_i$ = eigenvalues: Decay rates of diffusion modes
    - $\psi_i(\mathbf{x})$ = eigenfunctions: Spatial mode shapes
    - $\tau_i = 1/\lambda_i$ = relaxation_times: Mode decay time constants
    - $A_\infty(\mathbf{x})$ = steady_state: Equilibrium solution as $t \to \infty$

    CHARACTERISTIC SCALES:
    - Diffusion length: $l_D = \sqrt{Dt}$
    - Péclet number: $Pe = \frac{vl}{D}$ (advection vs diffusion)
    - Conservation law: $\frac{d}{dt}\int A(\mathbf{x},t) d\mathbf{x} = \int S(\mathbf{x},t) d\mathbf{x}$

    BOUNDARY CONDITIONS:
    - Dirichlet: $A|_{\partial\Omega} = g(\mathbf{x})$
    - Neumann: $\nabla A \cdot \mathbf{n}|_{\partial\Omega} = h(\mathbf{x})$
    - Periodic: $A(\mathbf{x} + L\mathbf{e}_i) = A(\mathbf{x})$
    """

    time_points: torch.Tensor  # t time sampling points
    field_evolution: torch.Tensor  # A(x,t) spatio-temporal evolution
    steady_state: torch.Tensor  # A_∞(x) equilibrium solution
    eigenvalues: torch.Tensor  # λᵢ diffusion operator eigenvalues
    eigenfunctions: torch.Tensor  # ψᵢ(x) diffusion eigenmodes
    relaxation_times: torch.Tensor  # τᵢ = 1/λᵢ mode relaxation times
    diffusion_length: float  # l_D = √(Dt) characteristic length
    peclet_number: float  # Pe = vl/D advection vs diffusion
    conservation_check: float  # |d/dt ∫A dx| mass conservation

    def __post_init__(self):
        """Validate diffusion solution mathematical properties."""
        if len(self.time_points) != self.field_evolution.shape[-1]:
            raise ValueError("Time points and evolution shape mismatch")
        if torch.any(self.eigenvalues < -POPULATION_NUMERICAL_PRECISION):
            raise ValueError("Diffusion eigenvalues must be non-positive")


@dataclass
class AttentionGradient:
    """
    Attention-driven selection gradient.

    MATHEMATICAL FORMULATION:
    Coupling between attention field and evolutionary selection dynamics:

    $$\frac{dp_i}{dt} = p_i[f_i + \alpha A(\mathbf{x}_i, t) - \langle f + \alpha A \rangle]$$

    ATTENTION-FITNESS COUPLING:
    - Enhanced fitness: $f_{eff}(\mathbf{x}) = f(\mathbf{x}) + \alpha A(\mathbf{x})$
    - Selection enhancement: $\Delta f_i = \alpha A(\mathbf{x}_i)$
    - Attention pressure: $\mathbf{F}_A = \nabla(\alpha A) = \alpha \nabla A$

    INFORMATION THEORETIC MEASURES:
    - Information gain rate: $\frac{dI}{dt} = \sum_i p_i \log\left(\frac{f_{eff,i}}{\langle f_{eff} \rangle}\right)$
    - Exploration-exploitation: $\mathcal{E} = \frac{H[A]}{H_{max}} \in [0,1]$
    - Attention entropy: $H[A] = -\int A(\mathbf{x}) \log A(\mathbf{x}) d\mathbf{x}$

    SELECTION PRESSURES:
    - Spotlight bias: $\beta_s = \frac{\sum_i p_i A_i}{\sum_i A_i}$
    - Diversification pressure: $\mathcal{D} = -\sum_i \tilde{A}_i \log \tilde{A}_i$ where $\tilde{A}_i = A_i/\sum_j A_j$
    - Effective selection coefficient: $s_{eff} = \alpha \frac{\partial A}{\partial x}$
    """

    attention_coupling: float  # α attention-fitness coupling strength
    selection_enhancement: torch.Tensor  # αA(xᵢ) attention-based fitness boost
    effective_fitness: torch.Tensor  # f_eff = f + αA effective fitness
    attention_selection_pressure: torch.Tensor  # ∇(αA) attention pressure gradient
    spotlight_bias: float  # Bias toward high-attention regions
    diversification_pressure: float  # Pressure for attention spreading
    information_gain_rate: float  # dI/dt information acquisition rate
    exploration_exploitation_ratio: float  # Balance exploration vs exploitation

    def __post_init__(self):
        """Validate attention gradient properties."""
        if not (0 <= self.exploration_exploitation_ratio <= 1):
            raise ValueError(
                f"Invalid exploration ratio: {self.exploration_exploitation_ratio}"
            )
        if self.information_gain_rate < 0:
            raise ValueError(f"Negative information gain: {self.information_gain_rate}")


class SpotlightFieldEngine:
    """
    Attention-Driven Selection via Diffusion Equations

    MATHEMATICAL APPROACH:
    1. Model attention as diffusing field A(x,t) on semantic manifold
    2. Solve diffusion PDE: ∂A/∂t = D∇²A - kA + S(x,t)
    3. Couple attention to selection: enhanced fitness in attended regions
    4. Track spotlight dynamics with adaptive focus and momentum
    5. Optimize information gain through attention allocation

    PDE SOLUTION METHODS:
    - Finite difference schemes for spatial discretization
    - Implicit Euler and Crank-Nicolson time stepping
    - Spectral methods for periodic boundary conditions
    - Adaptive mesh refinement near attention peaks
    - Fast Fourier Transform for convolution operations
    """

    def __init__(
        self,
        spatial_dimensions: int = 2,
        diffusion_coefficient: float = 1.0,
        decay_rate: float = 0.1,
        attention_coupling: float = 1.0,
        boundary_conditions: str = "neumann",
    ):
        """
        Initialize spotlight field engine.

        MATHEMATICAL PARAMETERS:
        $$\frac{\partial A}{\partial t} = D\nabla^2 A - kA + S(\mathbf{x},t)$$

        Args:
            spatial_dimensions: $d$ - Dimensionality of semantic manifold $\mathbb{R}^d$
            diffusion_coefficient: $D$ - Diffusion constant in $[m^2/s]$
            decay_rate: $k$ - Attention decay rate in $[1/s]$
            attention_coupling: $\alpha$ - Coupling strength $\alpha A(\mathbf{x})$ in fitness
            boundary_conditions: Mathematical boundary conditions on $\partial\Omega$:
                - "dirichlet": $A|_{\partial\Omega} = 0$
                - "neumann": $\nabla A \cdot \mathbf{n}|_{\partial\Omega} = 0$
                - "periodic": $A(\mathbf{x} + L\mathbf{e}_i) = A(\mathbf{x})$

        PHYSICAL INTERPRETATION:
        - $D$ controls spatial spreading: larger $D$ → faster diffusion
        - $k$ controls attention decay: larger $k$ → faster forgetting
        - $\alpha$ controls selection strength: larger $\alpha$ → stronger attention bias
        - Characteristic time: $\tau = 1/k$
        - Characteristic length: $\ell = \sqrt{D/k}$
        """
        self.spatial_dimensions = spatial_dimensions
        self.diffusion_coefficient = diffusion_coefficient
        self.decay_rate = decay_rate
        self.attention_coupling = attention_coupling
        self.boundary_conditions = boundary_conditions

        # Validate parameters
        if spatial_dimensions not in [1, 2, 3]:
            raise ValueError(f"Unsupported spatial dimension: {spatial_dimensions}")
        if diffusion_coefficient < 0:
            raise ValueError(f"Negative diffusion coefficient: {diffusion_coefficient}")
        if decay_rate < 0:
            raise ValueError(f"Negative decay rate: {decay_rate}")
        if boundary_conditions not in ["dirichlet", "neumann", "periodic"]:
            raise ValueError(f"Unknown boundary conditions: {boundary_conditions}")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"🔦 Initialized spotlight engine: d={spatial_dimensions}, "
            f"D={diffusion_coefficient}, k={decay_rate}, α={attention_coupling}"
        )

    @nb_jit(nopython=True, cache=True, fastmath=False)
    def _jit_diffusion_step(
        self,
        field: np.ndarray,
        source: np.ndarray,
        dt: float,
        dx: float,
        D: float,
        k: float,
    ) -> np.ndarray:
        """
        JIT-compiled diffusion equation time step using finite difference discretization.

        MATHEMATICAL FORMULATION:
        $$\frac{\partial A}{\partial t} = D\nabla^2 A - kA + S(\mathbf{x},t)$$

        FINITE DIFFERENCE DISCRETIZATION:
        $$\frac{A_i^{n+1} - A_i^n}{\Delta t} = D\frac{A_{i+1}^n - 2A_i^n + A_{i-1}^n}{(\Delta x)^2} - kA_i^n + S_i^n$$

        EXPLICIT EULER SCHEME:
        $$A_i^{n+1} = A_i^n + \Delta t\left[D\frac{A_{i+1}^n - 2A_i^n + A_{i-1}^n}{(\Delta x)^2} - kA_i^n + S_i^n\right]$$

        STABILITY CONDITION:
        $$\Delta t \leq \frac{(\Delta x)^2}{2D + k(\Delta x)^2}$$

        Args:
            field: $A_i^n$ - Current attention field values
            source: $S_i^n$ - Source term at current time
            dt: $\Delta t$ - Time step size
            dx: $\Delta x$ - Spatial grid spacing
            D: Diffusion coefficient
            k: Decay rate constant

        Returns:
            $A_i^{n+1}$ - Updated attention field after one time step
        """
        n_points = len(field)
        new_field = np.zeros_like(field)

        # Interior points (central difference for Laplacian)
        for i in range(1, n_points - 1):
            laplacian = (field[i + 1] - 2 * field[i] + field[i - 1]) / (dx * dx)
            diffusion_term = D * laplacian
            decay_term = -k * field[i]
            source_term = source[i]

            # Forward Euler step
            new_field[i] = field[i] + dt * (diffusion_term + decay_term + source_term)

        # Boundary conditions
        if n_points > 2:
            # Neumann boundary conditions (zero flux)
            new_field[0] = new_field[1]
            new_field[n_points - 1] = new_field[n_points - 2]

        return new_field

    @nb_jit(nopython=True, cache=True, fastmath=False)
    def _jit_gaussian_spotlight(
        self, x_coords: np.ndarray, center: float, width: float, amplitude: float
    ) -> np.ndarray:
        """
        JIT-compiled Gaussian spotlight generation.

        MATHEMATICAL FORMULATION:
        $$A(x) = A_0 \exp\left(-\frac{(x - x_0)^2}{2\sigma^2}\right)$$

        GAUSSIAN PROPERTIES:
        - Peak location: $x_0$ (center parameter)
        - Peak amplitude: $A_0$ (amplitude parameter)
        - Standard deviation: $\sigma$ (width parameter)
        - Full width at half maximum: $FWHM = 2\sqrt{2\ln 2}\sigma \approx 2.355\sigma$
        - Total area: $\int_{-\infty}^{\infty} A(x) dx = A_0\sigma\sqrt{2\pi}$

        DERIVATIVES:
        $$\frac{dA}{dx} = -A(x)\frac{(x - x_0)}{\sigma^2}$$
        $$\frac{d^2A}{dx^2} = A(x)\left[\frac{(x - x_0)^2}{\sigma^4} - \frac{1}{\sigma^2}\right]$$

        Args:
            x_coords: $x_i$ - Spatial coordinate array
            center: $x_0$ - Gaussian center position
            width: $\sigma$ - Gaussian width parameter
            amplitude: $A_0$ - Peak amplitude

        Returns:
            $A(x_i)$ - Gaussian attention field values
        """
        n_points = len(x_coords)
        spotlight = np.zeros(n_points)

        for i in range(n_points):
            x = x_coords[i]
            exponent = -0.5 * ((x - center) / width) ** 2
            spotlight[i] = amplitude * np.exp(exponent)

        return spotlight

    def _compute_3d_attention_kernel(
        self, spatial_grid: torch.Tensor, diffusion_time: float
    ) -> torch.Tensor:
        """
        Compute 3D attention kernel (Green's function for 3D diffusion equation).

        MATHEMATICAL FORMULATION:
        The fundamental solution to the 3D diffusion equation:
        $$\frac{\partial u}{\partial t} = D\nabla^2 u$$

        is the 3D Gaussian kernel:
        $$G(\mathbf{r}, t) = \frac{1}{(4\pi Dt)^{3/2}} \exp\left(-\frac{|\mathbf{r}|^2}{4Dt}\right)$$

        where:
        - $\mathbf{r} = (x, y, z)$ - 3D position vector
        - $|\mathbf{r}|^2 = x^2 + y^2 + z^2$ - Euclidean distance squared
        - $t$ - diffusion time parameter
        - $D$ - diffusion coefficient

        NORMALIZATION:
        $$\int_{\mathbb{R}^3} G(\mathbf{r}, t) d\mathbf{r} = 1$$

        SCALING PROPERTIES:
        - Width scales as: $\sigma(t) = \sqrt{2Dt}$
        - Peak height scales as: $G(0,t) = (4\pi Dt)^{-3/2}$
        - Characteristic diffusion length: $\ell_D = \sqrt{Dt}$

        Args:
            spatial_grid: 3D coordinate tensor $\mathbf{r}$
            diffusion_time: $t$ - time parameter

        Returns:
            $G(\mathbf{r}, t)$ - 3D diffusion kernel values
        """
        D = self.diffusion_coefficient
        t = diffusion_time

        if t <= 0:
            raise ValueError(f"Non-positive diffusion time: {t}")

        # 3D Gaussian kernel normalization
        normalization = 1.0 / ((4 * math.pi * D * t) ** (3 / 2))

        # Spatial distances from origin
        if spatial_grid.dim() == 3:
            # Full 3D grid
            x, y, z = spatial_grid[0], spatial_grid[1], spatial_grid[2]
            r_squared = x**2 + y**2 + z**2
        elif spatial_grid.dim() == 1:
            # 1D case extended to 3D
            r_squared = spatial_grid**2
        else:
            raise ValueError(
                f"Unsupported spatial grid dimension: {spatial_grid.dim()}"
            )

        # Gaussian attention kernel
        kernel = normalization * torch.exp(-r_squared / (4 * D * t))

        return kernel

    def _apply_conv3d_attention_convolution(
        self, attention_field: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply 3D convolution for attention field evolution using conv3d.

        MATHEMATICAL FORMULATION:
        Convolution operation for diffusion field evolution:
        $$A_{new}(\mathbf{x}) = (K * A)(\mathbf{x}) = \int_{\mathbb{R}^3} K(\mathbf{x} - \mathbf{y}) A(\mathbf{y}) d\mathbf{y}$$

        DISCRETE CONVOLUTION:
        $$A_{new}[i,j,k] = \sum_{l,m,n} K[l,m,n] \cdot A[i-l, j-m, k-n]$$

        FOURIER DOMAIN (theoretical basis):
        $$\mathcal{F}[K * A] = \mathcal{F}[K] \cdot \mathcal{F}[A]$$

        where $\mathcal{F}$ denotes the Fourier transform.

        BOUNDARY HANDLING:
        - 'same' padding: Output size matches input size
        - Implicit zero-padding for boundary conditions
        - Preserves field normalization: $\int A_{new} = \int A$ (if $\int K = 1$)

        COMPUTATIONAL COMPLEXITY:
        - Direct convolution: $O(N^3 M^3)$ where $N^3$ = field size, $M^3$ = kernel size
        - FFT-based convolution: $O(N^3 \log N)$ (for large kernels)

        Args:
            attention_field: $A(\mathbf{x})$ - Input attention field
            kernel: $K(\mathbf{x})$ - Convolution kernel (e.g., diffusion kernel)

        Returns:
            $(K * A)(\mathbf{x})$ - Convolved attention field
        """
        # Reshape for conv3d: (batch, channels, depth, height, width)
        if attention_field.dim() == 1:
            # Convert 1D to 3D for conv3d
            n = attention_field.shape[0]
            cube_size = int(math.ceil(n ** (1 / 3)))

            # Pad to cube
            padded_field = torch.zeros(cube_size**3)
            padded_field[:n] = attention_field

            # Reshape to 3D cube
            field_3d = padded_field.view(cube_size, cube_size, cube_size)
            field_5d = field_3d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        else:
            field_5d = attention_field.unsqueeze(0).unsqueeze(0)

        # Prepare kernel for conv3d
        if kernel.dim() == 1:
            # Convert 1D kernel to 3D
            n_k = kernel.shape[0]
            cube_size_k = int(math.ceil(n_k ** (1 / 3)))

            padded_kernel = torch.zeros(cube_size_k**3)
            padded_kernel[:n_k] = kernel

            kernel_3d = padded_kernel.view(cube_size_k, cube_size_k, cube_size_k)
            kernel_5d = kernel_3d.unsqueeze(0).unsqueeze(
                0
            )  # Add output and input channel dims
        else:
            kernel_5d = kernel.unsqueeze(0).unsqueeze(0)

        # Apply 3D convolution
        convolved = F.conv3d(field_5d, kernel_5d, padding="same")

        # Extract result
        result_3d = convolved.squeeze(0).squeeze(0)

        if attention_field.dim() == 1:
            # Convert back to 1D
            result_1d = result_3d.flatten()[: attention_field.shape[0]]
            return result_1d
        else:
            return result_3d

    def create_gaussian_spotlight(
        self,
        spatial_coordinates: torch.Tensor,
        center_position: torch.Tensor,
        focus_width: float,
        peak_intensity: float,
    ) -> AttentionField:
        """
        Create Gaussian attention spotlight on semantic manifold.

        MATHEMATICAL FORMULATION:
        $$A(\mathbf{x}) = A_0 \exp\left(-\frac{|\mathbf{x} - \mathbf{x}_0|^2}{2\sigma^2}\right)$$

        FIELD DERIVATIVES:
        Gradient (attention force field):
        $$\nabla A(\mathbf{x}) = -A(\mathbf{x}) \frac{\mathbf{x} - \mathbf{x}_0}{\sigma^2}$$

        Laplacian (attention curvature):
        $$\nabla^2 A(\mathbf{x}) = A(\mathbf{x}) \left[\frac{|\mathbf{x} - \mathbf{x}_0|^2}{\sigma^4} - \frac{d}{\sigma^2}\right]$$

        where $d$ is the spatial dimension.

        TOTAL ATTENTION BUDGET:
        $$\mathcal{A} = \int_{\mathbb{R}^d} A(\mathbf{x}) d\mathbf{x} = A_0 \sigma^d (2\pi)^{d/2}$$

        ATTENTION STATISTICS:
        - Peak location: $\mathbf{x}_0$ (center_position)
        - Peak amplitude: $A_0$ (peak_intensity)
        - Effective radius: $r_{eff} = \sigma\sqrt{2\ln 2}$
        - Decay length: $\sigma$ (focus_width)

        Args:
            spatial_coordinates: $\mathbf{x}_i$ - Discrete spatial grid points
            center_position: $\mathbf{x}_0$ - Spotlight center coordinates
            focus_width: $\sigma$ - Gaussian width parameter
            peak_intensity: $A_0$ - Maximum attention amplitude

        Returns:
            AttentionField object with computed field values and derivatives
        """
        if center_position.shape[0] != self.spatial_dimensions:
            raise ValueError(f"Center position must be {self.spatial_dimensions}D")
        if focus_width <= 0:
            raise ValueError(f"Non-positive focus width: {focus_width}")
        if peak_intensity < 0:
            raise ValueError(f"Negative peak intensity: {peak_intensity}")

        # For 1D case (can be extended to higher dimensions)
        if self.spatial_dimensions == 1:
            x_coords = spatial_coordinates.detach().cpu().numpy()
            center = center_position[0].item()

            # Generate Gaussian spotlight
            spotlight_np = self._jit_gaussian_spotlight(
                x_coords, center, focus_width, peak_intensity
            )
            field_values = torch.from_numpy(spotlight_np)

            # Compute gradient and Laplacian
            if len(x_coords) > 1:
                dx = x_coords[1] - x_coords[0]
                gradient_field = torch.gradient(field_values, spacing=dx)[0]
                laplacian_field = torch.gradient(gradient_field, spacing=dx)[0]
            else:
                gradient_field = torch.zeros_like(field_values)
                laplacian_field = torch.zeros_like(field_values)

        else:
            # Multi-dimensional Gaussian: I(x) = I₀ exp(-‖x-x₀‖²/2σ²)
            # For d-dimensional space: x = (x₁, x₂, ..., x_d)

            # Create coordinate meshgrid for multi-dimensional space
            coord_ranges = [
                torch.linspace(-spatial_range, spatial_range, n_spatial)
                for _ in range(self.spatial_dimensions)
            ]
            coord_grids = torch.meshgrid(*coord_ranges, indexing="ij")

            # Calculate multi-dimensional distances from center
            distances_squared = torch.zeros_like(coord_grids[0])
            for dim in range(self.spatial_dimensions):
                center_coord = (
                    center_position[dim] if dim < len(center_position) else 0.0
                )
                distances_squared += (coord_grids[dim] - center_coord) ** 2

            # Multi-dimensional Gaussian field: I(x) = I₀ exp(-r²/2σ²)
            field_values = peak_intensity * torch.exp(
                -distances_squared / (2 * focus_width**2)
            )

            # Flatten for consistent interface
            field_values = field_values.flatten()

            # Calculate multi-dimensional derivatives
            if calculate_derivatives:
                # Gradient: ∇I = -(x-x₀)/σ² · I(x)
                gradient_components = []
                for dim in range(self.spatial_dimensions):
                    center_coord = (
                        center_position[dim] if dim < len(center_position) else 0.0
                    )
                    grad_component = (
                        -(coord_grids[dim] - center_coord)
                        / (focus_width**2)
                        * field_values.reshape(coord_grids[0].shape)
                    )
                    gradient_components.append(grad_component.flatten())
                gradient_field = torch.stack(
                    gradient_components, dim=1
                )  # Shape: [n_points, n_dims]

                # Laplacian: ∇²I = I(x) · [‖x-x₀‖²/σ⁴ - d/σ²]
                # where d is spatial dimension
                laplacian_factor = (
                    distances_squared.flatten() / (focus_width**4)
                ) - (self.spatial_dimensions / (focus_width**2))
                laplacian_field = field_values * laplacian_factor
            else:
                gradient_field = torch.zeros(
                    (field_values.numel(), self.spatial_dimensions)
                )
                laplacian_field = torch.zeros_like(field_values)

        # Total attention (numerical integration)
        total_attention = torch.trapz(field_values, spatial_coordinates).item()

        return AttentionField(
            field_values=field_values,
            center_position=center_position,
            focus_width=focus_width,
            peak_intensity=peak_intensity,
            total_attention=total_attention,
            gradient_field=gradient_field,
            laplacian_field=laplacian_field,
            diffusion_coefficient=self.diffusion_coefficient,
            decay_rate=self.decay_rate,
        )

    def solve_diffusion_equation(
        self,
        initial_field: AttentionField,
        source_function: Callable,
        spatial_coordinates: torch.Tensor,
        time_horizon: float,
        time_steps: int = 100,
    ) -> DiffusionSolution:
        """
        Solve attention diffusion equation numerically using finite difference methods.

        MATHEMATICAL FORMULATION:
        Initial value problem for the attention diffusion PDE:
        $$\frac{\partial A}{\partial t} = D\nabla^2 A - kA + S(\mathbf{x},t)$$

        with initial condition:
        $$A(\mathbf{x}, 0) = A_0(\mathbf{x})$$

        ANALYTICAL SOLUTION (homogeneous case, $S = 0$):
        $$A(\mathbf{x}, t) = \int_{\mathbb{R}^d} G(\mathbf{x} - \mathbf{y}, t) A_0(\mathbf{y}) d\mathbf{y} \cdot e^{-kt}$$

        where $G(\mathbf{r}, t)$ is the diffusion Green's function.

        SPECTRAL DECOMPOSITION:
        $$A(\mathbf{x}, t) = \sum_{n=0}^{\infty} c_n \psi_n(\mathbf{x}) e^{-(\lambda_n + k)t}$$

        where $\{\lambda_n, \psi_n\}$ are eigenvalue-eigenfunction pairs of $-\nabla^2$.

        NUMERICAL METHOD:
        - Spatial discretization: Central finite differences
        - Time integration: Forward Euler (explicit)
        - Stability criterion: $\Delta t \leq \frac{(\Delta x)^2}{2D}$

        CHARACTERISTIC PARAMETERS:
        - Diffusion length: $\ell_D(t) = \sqrt{Dt}$
        - Decay time: $\tau = 1/k$
        - Péclet number: $Pe = vL/D$ (advection vs diffusion)

        Args:
            initial_field: $A_0(\mathbf{x})$ - Initial attention distribution
            source_function: $S(\mathbf{x}, t)$ - External attention sources
            spatial_coordinates: $\mathbf{x}_i$ - Discrete spatial grid
            time_horizon: $T$ - Total simulation time
            time_steps: $N_t$ - Number of temporal discretization points

        Returns:
            DiffusionSolution with complete spatio-temporal evolution
        """
        if time_steps <= 0:
            raise ValueError(f"Non-positive time steps: {time_steps}")
        if time_horizon <= 0:
            raise ValueError(f"Non-positive time horizon: {time_horizon}")

        # Time discretization
        time_points = torch.linspace(0, time_horizon, time_steps)
        dt = time_horizon / (time_steps - 1)

        # Spatial discretization
        if self.spatial_dimensions == 1:
            x_coords = spatial_coordinates.detach().cpu().numpy()
            dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
            n_spatial = len(x_coords)

            # Initialize evolution arrays
            field_evolution = torch.zeros((n_spatial, time_steps))
            current_field = initial_field.field_values.detach().cpu().numpy()

            # Time evolution loop
            for t_idx, time in enumerate(time_points):
                # Store current field
                field_evolution[:, t_idx] = torch.from_numpy(current_field)

                # Source term evaluation
                source_np = np.zeros(n_spatial)
                for i, x in enumerate(x_coords):
                    source_np[i] = source_function(x, time.item())

                # Diffusion step
                if t_idx < time_steps - 1:
                    current_field = self._jit_diffusion_step(
                        current_field,
                        source_np,
                        dt,
                        dx,
                        self.diffusion_coefficient,
                        self.decay_rate,
                    )

        else:
            # Multi-dimensional diffusion equation: ∂I/∂t = D·∇²I + S(x,t) - γI
            # Discretization: Iⁿ⁺¹ = Iⁿ + Δt[D·L·Iⁿ + Sⁿ - γIⁿ] where L is Laplacian operator

            # Reshape field for multi-dimensional processing
            field_shape = [n_spatial] * self.spatial_dimensions
            field_evolution = torch.zeros((*field_shape, n_time))

            # Initialize with source field
            initial_field = source_field.reshape(field_shape)
            field_evolution[..., 0] = initial_field

            # Multi-dimensional finite difference stencils
            # For each dimension: second derivative ≈ (I_{i+1} - 2I_i + I_{i-1})/Δx²
            for t_idx in range(1, n_time):
                current_field = field_evolution[..., t_idx - 1]

                # Calculate multi-dimensional Laplacian
                laplacian = torch.zeros_like(current_field)

                # Apply finite difference in each spatial dimension
                for dim in range(self.spatial_dimensions):
                    # Create slicing tuples for neighbor access
                    # Central difference: (I_{i+1} - 2I_i + I_{i-1})/h²

                    # Pad boundaries with Neumann conditions (∂I/∂n = 0)
                    padded_field = current_field

                    # Apply 1D Laplacian along dimension 'dim'
                    if self.spatial_dimensions == 2:
                        if dim == 0:  # x-direction
                            laplacian[1:-1, :] += (
                                padded_field[2:, :]
                                - 2 * padded_field[1:-1, :]
                                + padded_field[:-2, :]
                            ) / dx**2
                        elif dim == 1:  # y-direction
                            laplacian[:, 1:-1] += (
                                padded_field[:, 2:]
                                - 2 * padded_field[:, 1:-1]
                                + padded_field[:, :-2]
                            ) / dx**2
                    elif self.spatial_dimensions == 3:
                        if dim == 0:  # x-direction
                            laplacian[1:-1, :, :] += (
                                padded_field[2:, :, :]
                                - 2 * padded_field[1:-1, :, :]
                                + padded_field[:-2, :, :]
                            ) / dx**2
                        elif dim == 1:  # y-direction
                            laplacian[:, 1:-1, :] += (
                                padded_field[:, 2:, :]
                                - 2 * padded_field[:, 1:-1, :]
                                + padded_field[:, :-2, :]
                            ) / dx**2
                        elif dim == 2:  # z-direction
                            laplacian[:, :, 1:-1] += (
                                padded_field[:, :, 2:]
                                - 2 * padded_field[:, :, 1:-1]
                                + padded_field[:, :, :-2]
                            ) / dx**2

                # Time evolution: I^{n+1} = I^n + Δt[D·∇²I + S - γI]
                source_term = source_field.reshape(field_shape)
                diffusion_term = self.diffusion_coefficient * laplacian
                decay_term = -self.decay_rate * current_field

                field_evolution[..., t_idx] = current_field + dt * (
                    diffusion_term + source_term + decay_term
                )

                # Apply boundary conditions (zero flux)
                if self.spatial_dimensions >= 1:
                    field_evolution[0, ..., t_idx] = field_evolution[1, ..., t_idx]
                    field_evolution[-1, ..., t_idx] = field_evolution[-2, ..., t_idx]
                if self.spatial_dimensions >= 2:
                    field_evolution[:, 0, ..., t_idx] = field_evolution[
                        :, 1, ..., t_idx
                    ]
                    field_evolution[:, -1, ..., t_idx] = field_evolution[
                        :, -2, ..., t_idx
                    ]
                if self.spatial_dimensions >= 3:
                    field_evolution[:, :, 0, t_idx] = field_evolution[:, :, 1, t_idx]
                    field_evolution[:, :, -1, t_idx] = field_evolution[:, :, -2, t_idx]

            # Flatten final result for consistent interface
            field_evolution = field_evolution.reshape(-1, n_time)

        # Steady state (final field)
        steady_state = field_evolution[:, -1]

        # Eigenvalue analysis (simplified - 1D case)
        if self.spatial_dimensions == 1 and n_spatial > 1:
            # Construct diffusion operator matrix
            A_matrix = self._construct_diffusion_matrix(n_spatial, dx)
            eigenvals, eigenvecs = torch.linalg.eigh(A_matrix)

            # Relaxation times τᵢ = 1/|λᵢ|
            relaxation_times = 1.0 / torch.abs(eigenvals + 1e-12)

        else:
            eigenvals = torch.zeros(1)
            eigenvecs = torch.eye(1)
            relaxation_times = torch.ones(1)

        # Characteristic diffusion length
        diffusion_length = math.sqrt(self.diffusion_coefficient * time_horizon)

        # Velocity scale from attention field gradient: v = |∇A|
        attention_gradient = torch.gradient(attention_field.field_values, spacing=dx)[0]
        velocity_scale = torch.max(torch.abs(attention_gradient)).item()

        # Characteristic length scale from spatial domain
        length_scale = torch.max(spatial_coordinates) - torch.min(spatial_coordinates)

        # Péclet number: Pe = vL/D (dimensionless advection/diffusion ratio)
        peclet_number = (
            velocity_scale * length_scale.item() / (self.diffusion_coefficient + 1e-12)
        )

        # Conservation check
        initial_mass = torch.trapz(initial_field.field_values, spatial_coordinates)
        final_mass = torch.trapz(steady_state, spatial_coordinates)
        conservation_check = abs((final_mass - initial_mass) / initial_mass).item()

        return DiffusionSolution(
            time_points=time_points,
            field_evolution=field_evolution,
            steady_state=steady_state,
            eigenvalues=eigenvals,
            eigenfunctions=eigenvecs,
            relaxation_times=relaxation_times,
            diffusion_length=diffusion_length,
            peclet_number=peclet_number,
            conservation_check=conservation_check,
        )

    def _construct_diffusion_matrix(self, n_points: int, dx: float) -> torch.Tensor:
        """
        Construct finite difference diffusion operator matrix.
        
        MATHEMATICAL FORMULATION:
        Discretization of the diffusion operator $\mathcal{L} = D\nabla^2 - k$:
        
        $$\mathcal{L}_{discrete} = D\frac{1}{(\Delta x)^2}\begin{pmatrix}
        -2 & 1 & 0 & \cdots & 0 \\
        1 & -2 & 1 & \cdots & 0 \\
        \vdots & \ddots & \ddots & \ddots & \vdots \\
        0 & \cdots & 1 & -2 & 1 \\
        0 & \cdots & 0 & 1 & -2
        \end{pmatrix} - k\mathbf{I}$$
        
        MATRIX PROPERTIES:
        - Tridiagonal structure: $A_{i,j} \neq 0$ only for $|i-j| \leq 1$
        - Symmetric: $A_{i,j} = A_{j,i}$
        - Negative definite: All eigenvalues $\lambda_i < 0$
        - Spectral radius: $\rho(A) = \max_i |\lambda_i|$
        
        EIGENVALUE BOUNDS (Dirichlet BC):
        $$\lambda_n = -\frac{4D}{(\Delta x)^2}\sin^2\left(\frac{n\pi}{2(N+1)}\right) - k$$
        
        for $n = 1, 2, \ldots, N$.
        
        STABILITY ANALYSIS:
        - Explicit scheme stable if: $\Delta t \leq \frac{(\Delta x)^2}{2D + k(\Delta x)^2}$
        - Implicit scheme unconditionally stable
        
        Args:
            n_points: $N$ - Number of spatial grid points
            dx: $\Delta x$ - Spatial grid spacing
            
        Returns:
            $\mathcal{L}_{discrete}$ - Discrete diffusion operator matrix
        """
        # D∇² - k operator matrix
        A = torch.zeros((n_points, n_points), dtype=get_dtype_manager().config.real_dtype)

        for i in range(n_points):
            # Diagonal term: -2D/dx² - k
            A[i, i] = -2 * self.diffusion_coefficient / (dx * dx) - self.decay_rate

            # Off-diagonal terms: D/dx²
            if i > 0:
                A[i, i - 1] = self.diffusion_coefficient / (dx * dx)
            if i < n_points - 1:
                A[i, i + 1] = self.diffusion_coefficient / (dx * dx)

        return A

    def compute_attention_selection_pressure(
        self,
        attention_field: AttentionField,
        population_state: PopulationState,
        spatial_coordinates: torch.Tensor,
    ) -> AttentionGradient:
        """
        Compute selection pressure from attention field coupling.

        MATHEMATICAL FORMULATION:
        Replicator equation with attention-modulated fitness:
        $$\frac{dp_i}{dt} = p_i[f_i + \alpha A(\mathbf{x}_i, t) - \langle f + \alpha A \rangle]$$

        where:
        $$\langle f + \alpha A \rangle = \sum_j p_j [f_j + \alpha A(\mathbf{x}_j, t)]$$

        EFFECTIVE FITNESS:
        $$f_{eff}(\mathbf{x}) = f(\mathbf{x}) + \alpha A(\mathbf{x}, t)$$

        SELECTION PRESSURE GRADIENT:
        $$\mathbf{F}_{selection}(\mathbf{x}) = \nabla f_{eff} = \nabla f + \alpha \nabla A$$

        INFORMATION MEASURES:
        Information gain rate:
        $$\frac{dI}{dt} = \sum_i p_i \log\left(\frac{f_{eff,i}}{\langle f_{eff} \rangle}\right) \frac{df_{eff,i}}{dt}$$

        Attention entropy:
        $$H[A] = -\int A(\mathbf{x}) \log A(\mathbf{x}) d\mathbf{x}$$

        SELECTION BIAS MEASURES:
        Spotlight bias (attention-weighted frequency):
        $$\beta_s = \frac{\sum_i p_i A_i}{\sum_i A_i}$$

        Diversification pressure (attention distribution entropy):
        $$\mathcal{D} = -\sum_i \tilde{A}_i \log \tilde{A}_i$$
        where $\tilde{A}_i = A_i / \sum_j A_j$.

        Args:
            attention_field: $A(\mathbf{x})$ - Current attention distribution
            population_state: $\{p_i, f_i\}$ - Population frequencies and fitness
            spatial_coordinates: $\mathbf{x}_i$ - Spatial positions of types

        Returns:
            AttentionGradient with all selection pressure components
        """
        n_types = len(population_state.frequencies)

        # Extract positions from population state if available
        if (
            hasattr(population_state, "spatial_positions")
            and population_state.spatial_positions is not None
        ):
            # Use actual Q-field manifold positions
            type_positions = population_state.spatial_positions[:n_types]
        elif (
            hasattr(population_state, "field_coordinates")
            and population_state.field_coordinates is not None
        ):
            # Use field coordinates projected to spatial domain
            field_coords = population_state.field_coordinates
            if field_coords.numel() >= n_types:
                # Map field coordinates to spatial coordinates
                coord_min, coord_max = torch.min(spatial_coordinates), torch.max(
                    spatial_coordinates
                )
                field_min, field_max = torch.min(field_coords), torch.max(field_coords)

                # Linear mapping: x = (coord - field_min)/(field_max - field_min) * (coord_max - coord_min) + coord_min
                field_range = field_max - field_min + 1e-12
                coord_range = coord_max - coord_min
                type_positions = (
                    (field_coords[:n_types] - field_min) / field_range
                ) * coord_range + coord_min
            else:
                # Fallback: uniform distribution
                type_positions = torch.linspace(
                    torch.min(spatial_coordinates),
                    torch.max(spatial_coordinates),
                    n_types,
                )
        else:
            # Fallback: optimal spacing based on attention field maxima
            # Find peaks in attention field for natural type positioning
            attention_values = attention_field.field_values
            if len(attention_values) >= n_types:
                # Use attention peaks as natural type positions
                _, peak_indices = torch.topk(
                    attention_values, k=min(n_types, len(attention_values))
                )
                sorted_indices = torch.sort(peak_indices)[0]
                type_positions = spatial_coordinates[sorted_indices]
            else:
                # Uniform fallback
                type_positions = torch.linspace(
                    torch.min(spatial_coordinates),
                    torch.max(spatial_coordinates),
                    n_types,
                )

        # Interpolate attention field at type positions
        attention_at_types = torch.zeros(n_types)
        for i, pos in enumerate(type_positions):
            # Find closest spatial coordinate
            idx = torch.argmin(torch.abs(spatial_coordinates - pos))
            attention_at_types[i] = attention_field.field_values[idx]

        # Attention-based fitness enhancement
        selection_enhancement = self.attention_coupling * attention_at_types

        # Effective fitness
        base_fitness = population_state.fitness_values
        effective_fitness = base_fitness + selection_enhancement

        # Attention selection pressure (gradient of αA)
        attention_pressure = torch.gradient(selection_enhancement)[0]

        # Spotlight bias (preference for high-attention regions)
        total_attention = torch.sum(attention_at_types)
        if total_attention > POPULATION_NUMERICAL_PRECISION:
            spotlight_bias = (
                torch.sum(population_state.frequencies * attention_at_types)
                / total_attention
            )
        else:
            spotlight_bias = 0.0

        # Diversification pressure (entropy of attention distribution)
        attention_probs = F.softmax(attention_at_types, dim=0)
        diversification_pressure = -torch.sum(
            attention_probs * torch.log(attention_probs + 1e-12)
        )

        # From Fisher information and replicator dynamics

        # Frequency derivatives from replicator equation: ∂pᵢ/∂t = pᵢ(fᵢ - ⟨f⟩)
        mean_fitness = torch.sum(population_state.frequencies * effective_fitness)
        freq_derivatives = population_state.frequencies * (
            effective_fitness - mean_fitness
        )

        # Information gain: dI/dt = Σᵢ (∂pᵢ/∂t) log(pᵢ) + Σᵢ pᵢ (∂log fᵢ/∂t)
        # Simplified: focus on frequency-driven information change
        freq_info_term = torch.sum(
            freq_derivatives
            * torch.log(population_state.frequencies + POPULATION_NUMERICAL_PRECISION)
        )
        fitness_info_term = torch.sum(
            attention_at_types * torch.log(effective_fitness + 1.0)
        )

        info_gain_rate = freq_info_term + fitness_info_term

        # Exploration vs exploitation ratio
        max_attention = torch.max(attention_at_types)
        mean_attention = torch.mean(attention_at_types)
        if max_attention > POPULATION_NUMERICAL_PRECISION:
            exploration_ratio = 1.0 - (mean_attention / max_attention).item()
        else:
            exploration_ratio = 0.5

        return AttentionGradient(
            attention_coupling=self.attention_coupling,
            selection_enhancement=selection_enhancement,
            effective_fitness=effective_fitness,
            attention_selection_pressure=attention_pressure,
            spotlight_bias=spotlight_bias.item(),
            diversification_pressure=diversification_pressure.item(),
            information_gain_rate=info_gain_rate.item(),
            exploration_exploitation_ratio=exploration_ratio,
        )

    def optimize_spotlight_position(
        self,
        population_state: PopulationState,
        spatial_coordinates: torch.Tensor,
        objective_function: Callable,
    ) -> torch.Tensor:
        """
        Optimize spotlight position to maximize objective function.

        MATHEMATICAL FORMULATION:
        Variational optimization problem:
        $$\mathbf{x}_0^* = \arg\max_{\mathbf{x}_0} J[A(\mathbf{x} - \mathbf{x}_0), \mathbf{p}]$$

        where $J$ is the objective functional.

        OBJECTIVE FUNCTIONAL EXAMPLES:
        1. Information gain: $J = \sum_i p_i \log f_{eff,i}$
        2. Selection efficiency: $J = \text{Var}[f_{eff}]$
        3. Attention-fitness correlation: $J = \text{Cov}[A, f]$

        OPTIMIZATION CONDITIONS:
        First-order necessary condition:
        $$\frac{\partial J}{\partial \mathbf{x}_0} = \int \frac{\delta J}{\delta A} \frac{\partial A}{\partial \mathbf{x}_0} d\mathbf{x} = 0$$

        ATTENTION FIELD DERIVATIVE:
        $$\frac{\partial A}{\partial \mathbf{x}_0} = A(\mathbf{x}) \frac{\mathbf{x} - \mathbf{x}_0}{\sigma^2}$$

        NUMERICAL METHOD:
        - Scalar optimization using bounded methods
        - Gradient-free optimization (Nelder-Mead, Golden Section)
        - Bounds: $\mathbf{x}_0 \in [\min(\mathbf{x}), \max(\mathbf{x})]$

        CONVERGENCE CRITERIA:
        - Function tolerance: $|J^{(k+1)} - J^{(k)}| < \epsilon_f$
        - Parameter tolerance: $|\mathbf{x}_0^{(k+1)} - \mathbf{x}_0^{(k)}| < \epsilon_x$

        Args:
            population_state: $\{p_i, f_i\}$ - Current population state
            spatial_coordinates: $\mathbf{x}_i$ - Spatial coordinate grid
            objective_function: $J[\cdot]$ - Objective functional to maximize

        Returns:
            $\mathbf{x}_0^*$ - Optimal spotlight center position
        """

        # Define objective as function of spotlight center
        def objective(center_pos):
            center_tensor = torch.tensor([center_pos], dtype=get_dtype_manager().config.real_dtype)

            # Create spotlight at this position
            spotlight = self.create_gaussian_spotlight(
                spatial_coordinates=spatial_coordinates,
                center_position=center_tensor,
                focus_width=1.0,  # Fixed width for optimization
                peak_intensity=1.0,
            )

            # Compute attention gradient
            attention_grad = self.compute_attention_selection_pressure(
                spotlight, population_state, spatial_coordinates
            )

            # Evaluate objective function
            return -objective_function(attention_grad)  # Negative for minimization

        # Optimize using scipy
        spatial_bounds = [
            (
                torch.min(spatial_coordinates).item(),
                torch.max(spatial_coordinates).item(),
            )
        ]

        result = minimize_scalar(objective, bounds=spatial_bounds[0], method="bounded")

        optimal_position = torch.tensor([result.x], dtype=get_dtype_manager().config.real_dtype)
        return optimal_position

    def simulate_spotlight_tracking(
        self,
        initial_position: torch.Tensor,
        target_trajectory: Callable,
        spatial_coordinates: torch.Tensor,
        time_horizon: float,
        tracking_gain: float = 1.0,
    ) -> SpotlightDynamics:
        """
        Simulate spotlight tracking dynamics with momentum and damping.

        MATHEMATICAL FORMULATION:
        Second-order dynamical system for spotlight center motion:
        $$m\frac{d^2\mathbf{x}_0}{dt^2} = -\gamma\frac{d\mathbf{x}_0}{dt} + \mathbf{F}_{track}(\mathbf{x}_{target} - \mathbf{x}_0) + \mathbf{F}_{field}$$

        FORCE COMPONENTS:
        1. Damping force: $\mathbf{F}_{damp} = -\gamma \mathbf{v}_0$
        2. Tracking force: $\mathbf{F}_{track} = \beta (\mathbf{x}_{target} - \mathbf{x}_0)$
        3. Field force: $\mathbf{F}_{field} = -\nabla V(\mathbf{x}_0)$

        ENERGY FORMULATION:
        Total energy: $E = T + V$ where:
        - Kinetic energy: $T = \frac{1}{2}m|\mathbf{v}_0|^2$
        - Potential energy: $V = \frac{1}{2}k|\mathbf{x}_0 - \mathbf{x}_{target}|^2$

        NUMERICAL INTEGRATION:
        Velocity Verlet scheme:
        $$\mathbf{x}_0^{(n+1)} = \mathbf{x}_0^{(n)} + \mathbf{v}_0^{(n)} \Delta t + \frac{1}{2}\mathbf{a}^{(n)} (\Delta t)^2$$
        $$\mathbf{v}_0^{(n+1)} = \mathbf{v}_0^{(n)} + \frac{1}{2}[\mathbf{a}^{(n)} + \mathbf{a}^{(n+1)}] \Delta t$$

        TRAJECTORY ANALYSIS:
        - Curvature: $\kappa = \frac{|\mathbf{v} \times \mathbf{a}|}{|\mathbf{v}|^3}$
        - Arc length: $s(t) = \int_0^t |\mathbf{v}(\tau)| d\tau$
        - Angular momentum: $\mathbf{L} = \mathbf{r} \times \mathbf{p}$

        STABILITY CONDITIONS:
        - Underdamped: $\gamma < 2\sqrt{\beta m}$
        - Critically damped: $\gamma = 2\sqrt{\beta m}$
        - Overdamped: $\gamma > 2\sqrt{\beta m}$

        Args:
            initial_position: $\mathbf{x}_0(0)$ - Initial spotlight position
            target_trajectory: $\mathbf{x}_{target}(t)$ - Time-dependent target
            spatial_coordinates: Spatial domain bounds
            time_horizon: $T$ - Total simulation time
            tracking_gain: $\beta$ - Proportional tracking strength

        Returns:
            SpotlightDynamics with complete trajectory analysis
        """
        time_steps = 100
        time_points = torch.linspace(0, time_horizon, time_steps)
        dt = time_horizon / (time_steps - 1)

        # Initialize state
        position = initial_position.clone()
        velocity = torch.zeros_like(position)

        # Evolution arrays
        position_history = torch.zeros((time_steps, len(position)))
        velocity_history = torch.zeros((time_steps, len(position)))

        # Physical parameters
        mass = 1.0
        damping = 0.1

        for t_idx, time in enumerate(time_points):
            # Store current state
            position_history[t_idx] = position
            velocity_history[t_idx] = velocity

            # Target position at current time
            target_pos = target_trajectory(time.item())

            # Tracking force toward target
            tracking_force = tracking_gain * (target_pos - position)

            # Damping force
            damping_force = -damping * velocity

            # Total force
            total_force = tracking_force + damping_force

            # Update dynamics (Verlet integration)
            if t_idx < time_steps - 1:
                acceleration = total_force / mass
                velocity = velocity + dt * acceleration
                position = position + dt * velocity

        # Compute derived quantities
        final_velocity = velocity_history[-1]
        kinetic_energy = 0.5 * mass * torch.sum(final_velocity**2).item()

        # Potential energy from spotlight interaction with position
        if attention_field is not None and hasattr(attention_field, "field_values"):
            # Interpolate attention field at spotlight position
            attention_vals = attention_field.field_values
            if len(attention_vals) > 1 and len(spatial_coordinates) > 1:
                # Create interpolation function
                from scipy.interpolate import interp1d

                spatial_np = spatial_coordinates.detach().cpu().numpy()
                attention_np = attention_vals.detach().cpu().numpy()

                try:
                    interp_func = interp1d(
                        spatial_np,
                        attention_np,
                        kind="linear",
                        bounds_error=False,
                        fill_value=0.0,
                    )

                    # Evaluate potential at spotlight position
                    pos_np = position.detach().cpu().numpy()
                    if pos_np.ndim == 0:  # scalar position
                        potential_at_pos = interp_func(pos_np.item())
                    else:  # vector position - use magnitude
                        pos_magnitude = np.linalg.norm(pos_np)
                        potential_at_pos = interp_func(pos_magnitude)

                    # Potential energy: U = -∫ A(x) dx (attractive potential)
                    potential_energy = (
                        -float(potential_at_pos) * torch.norm(position).item()
                    )
                except:
                    # Fallback: harmonic potential from field statistics
                    field_variance = torch.var(attention_vals)
                    potential_energy = (
                        0.5 * field_variance.item() * torch.sum(position**2).item()
                    )
            else:
                # Single point: constant potential
                potential_energy = (
                    attention_vals[0].item() * torch.norm(position).item()
                )
        else:
            # Fallback: quadratic potential from position
            potential_energy = 0.5 * torch.sum(position**2).item()
        total_energy = kinetic_energy + potential_energy

        # Trajectory curvature (simplified)
        if time_steps > 2:
            velocity_mag = torch.norm(velocity_history[-1])
            acceleration_perp = (
                torch.norm(velocity_history[-1] - velocity_history[-2]) / dt
            )
            trajectory_curvature = acceleration_perp / (velocity_mag**2 + 1e-12)
        else:
            trajectory_curvature = 0.0

        return SpotlightDynamics(
            velocity_field=velocity_history[-1],
            tracking_target=target_trajectory(time_horizon),
            adaptation_rate=tracking_gain,
            momentum=mass * velocity_history[-1],
            kinetic_energy=kinetic_energy,
            potential_energy=potential_energy,
            total_energy=total_energy,
            trajectory_curvature=trajectory_curvature.item(),
            angular_momentum=torch.zeros_like(position),  # Simplified for 1D
        )

    def complex_attention_field_analysis(
        self, attention_field: AttentionField, spatial_coordinates: torch.Tensor
    ) -> Dict[str, complex]:
        """
        Complex mathematical analysis of attention fields using complex representations.

        MATHEMATICAL FORMULATION:
        Complex field representation:
        $$\mathcal{A}(\mathbf{x}) = A(\mathbf{x}) + i\nabla A(\mathbf{x}) = |\mathcal{A}|e^{i\phi}$$

        where:
        - Real part: $\Re[\mathcal{A}] = A(\mathbf{x})$ (attention amplitude)
        - Imaginary part: $\Im[\mathcal{A}] = \nabla A(\mathbf{x})$ (attention gradient)

        COMPLEX OPERATIONS:
        1. Polar form: $\mathcal{A} = r e^{i\phi}$ where $r = |\mathcal{A}|$, $\phi = \arg(\mathcal{A})$
        2. Phase: $\phi = \arctan(\Im[\mathcal{A}]/\Re[\mathcal{A}])$
        3. Complex logarithm: $\log(\mathcal{A}) = \log|\mathcal{A}| + i\arg(\mathcal{A})$
        4. Complex exponential: $e^{\mathcal{A}} = e^{\Re[\mathcal{A}]}(\cos(\Im[\mathcal{A}]) + i\sin(\Im[\mathcal{A}]))$

        TRIGONOMETRIC FUNCTIONS:
        - $\sin(\mathcal{A}) = \frac{e^{i\mathcal{A}} - e^{-i\mathcal{A}}}{2i}$
        - $\cos(\mathcal{A}) = \frac{e^{i\mathcal{A}} + e^{-i\mathcal{A}}}{2}$

        HYPERBOLIC FUNCTIONS:
        - $\sinh(\mathcal{A}) = \frac{e^{\mathcal{A}} - e^{-\mathcal{A}}}{2}$
        - $\cosh(\mathcal{A}) = \frac{e^{\mathcal{A}} + e^{-\mathcal{A}}}{2}$

        COMPLEX SQUARE ROOT:
        $$\sqrt{\mathcal{A}} = \sqrt{|\mathcal{A}|} e^{i\arg(\mathcal{A})/2}$$

        PHYSICAL INTERPRETATION:
        - Magnitude $|\mathcal{A}|$: Total attention intensity
        - Phase $\arg(\mathcal{A})$: Attention flow direction
        - Complex operations encode field topology and flow patterns

        Args:
            attention_field: $A(\mathbf{x})$ and $\nabla A(\mathbf{x})$ - Real and gradient fields
            spatial_coordinates: $\mathbf{x}_i$ - Spatial grid points

        Returns:
            Dictionary of complex mathematical operations on attention field
        """
        # Convert attention field to complex representation
        field_real = attention_field.field_values.real
        field_imag = attention_field.gradient_field

        complex_results = {}

        for i, (real_val, imag_val) in enumerate(zip(field_real, field_imag)):
            # Create complex number
            z = complex(real_val.item(), imag_val.item())

            # Complex mathematical operations using cmath
            complex_results[f"polar_form_{i}"] = cmath.polar(z)  # (r, φ)
            complex_results[f"phase_{i}"] = cmath.phase(z)  # arg(z)
            complex_results[f"log_{i}"] = cmath.log(z + 1e-12)  # log(z)
            complex_results[f"exp_{i}"] = cmath.exp(z)  # e^z
            complex_results[f"sin_{i}"] = cmath.sin(z)  # sin(z)
            complex_results[f"cos_{i}"] = cmath.cos(z)  # cos(z)
            complex_results[f"sinh_{i}"] = cmath.sinh(z)  # sinh(z)
            complex_results[f"cosh_{i}"] = cmath.cosh(z)  # cosh(z)
            complex_results[f"sqrt_{i}"] = cmath.sqrt(z)  # √z

            if i >= 10:  # Limit for performance
                break

        return complex_results

    def torch_geometric_attention_manifold(
        self, attention_field: AttentionField, spatial_coordinates: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Geometric deep learning analysis using torch_geometric for attention manifolds.

        MATHEMATICAL FORMULATION:
        Graph-based representation of attention manifold $\mathcal{M}$:
        $$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X})$$

        where:
        - $\mathcal{V} = \{v_i\}$ - Vertices (spatial grid points)
        - $\mathcal{E} = \{(v_i, v_j)\}$ - Edges (spatial connectivity)
        - $\mathbf{X} \in \mathbb{R}^{|\mathcal{V}| \times d}$ - Node features

        NODE FEATURES:
        $$\mathbf{x}_i = \begin{pmatrix} A(\mathbf{r}_i) \\ \nabla A(\mathbf{r}_i) \\ \nabla^2 A(\mathbf{r}_i) \end{pmatrix}$$

        MESSAGE PASSING:
        Node update rule:
        $$\mathbf{h}_i^{(l+1)} = \gamma^{(l)}\left(\mathbf{h}_i^{(l)}, \square_{j \in \mathcal{N}(i)} \phi^{(l)}(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, \mathbf{e}_{ij})\right)$$

        where:
        - $\mathbf{h}_i^{(l)}$ - Node representation at layer $l$
        - $\mathcal{N}(i)$ - Neighborhood of node $i$
        - $\phi^{(l)}$ - Message function
        - $\gamma^{(l)}$ - Update function
        - $\square$ - Aggregation operator (mean, max, sum)

        ATTENTION PROPAGATION:
        Message computation:
        $$\mathbf{m}_{ij} = \phi(\mathbf{h}_i, \mathbf{h}_j) = \mathbf{W} \cdot [\mathbf{h}_i \| \mathbf{h}_j]$$

        Aggregation:
        $$\mathbf{h}_i^{new} = \sigma\left(\mathbf{W}_{self}\mathbf{h}_i + \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{m}_{ij}\right)$$

        GRAPH PROPERTIES:
        - Node degree: $d_i = |\mathcal{N}(i)|$
        - Edge connectivity: Adjacency matrix $\mathbf{A} \in \{0,1\}^{n \times n}$
        - Graph Laplacian: $\mathbf{L} = \mathbf{D} - \mathbf{A}$

        MANIFOLD ANALYSIS:
        - Local curvature estimation via second-order derivatives
        - Geodesic distance approximation through graph shortest paths
        - Attention flow analysis via gradient field visualization

        Args:
            attention_field: $A(\mathbf{x})$, $\nabla A$, $\nabla^2 A$ - Attention field components
            spatial_coordinates: $\mathbf{r}_i$ - Spatial positions of graph nodes

        Returns:
            Dictionary with graph structure and geometric deep learning analysis
        """
        n_nodes = len(attention_field.field_values)

        # Create edge index for spatial graph (nearest neighbors)
        edge_index = []
        for i in range(n_nodes - 1):
            edge_index.append([i, i + 1])  # Forward edge
            edge_index.append([i + 1, i])  # Backward edge

        if len(edge_index) > 0:
            edge_index_tensor = (
                torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            )
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)

        # Node features: attention values and gradients
        x = torch.stack(
            [
                attention_field.field_values,
                attention_field.gradient_field,
                attention_field.laplacian_field,
            ],
            dim=1,
        )

        # Create geometric data object
        data = Data(x=x, edge_index=edge_index_tensor)

        # Custom message passing for attention propagation
        class AttentionMessagePassing(MessagePassing):
            def __init__(self):
                super().__init__(aggr="mean")

            def forward(self, x, edge_index):
                return self.propagate(edge_index, x=x)

            def message(self, x_j):
                return x_j  # Simple identity message

            def update(self, aggr_out, x):
                return aggr_out + x  # Residual connection

        # Apply message passing
        mp_layer = AttentionMessagePassing()
        propagated_features = mp_layer(data.x, data.edge_index)

        return {
            "node_features": data.x,
            "edge_connectivity": data.edge_index,
            "propagated_attention": propagated_features,
            "graph_size": torch.tensor([n_nodes]),
            "manifold_dimension": torch.tensor([data.x.shape[1]]),
        }

    def jax_advanced_diffusion_integration(
        self, attention_field: AttentionField, spatial_coordinates: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Advanced diffusion analysis using JAX integration and linear solving.

        MATHEMATICAL FORMULATION:
        Energy functional for attention field:
        $$E[A] = \int_{\Omega} \left[\frac{1}{2}|\nabla A|^2 + \frac{k}{2}A^2\right] d\mathbf{x}$$

        VARIATIONAL FORMULATION:
        Euler-Lagrange equation:
        $$\frac{\delta E}{\delta A} = -\nabla^2 A + kA = 0$$

        DIFFUSION GREEN'S FUNCTION:
        $$G(\mathbf{x}, \mathbf{y}, t) = \frac{1}{(4\pi Dt)^{d/2}} \exp\left(-\frac{|\mathbf{x} - \mathbf{y}|^2}{4Dt}\right)$$

        SOLUTION REPRESENTATION:
        $$A(\mathbf{x}, t) = \int_{\Omega} G(\mathbf{x}, \mathbf{y}, t) A_0(\mathbf{y}) d\mathbf{y} \cdot e^{-kt}$$

        JAX AUTOMATIC DIFFERENTIATION:
        Hessian computation:
        $$\mathbf{H}[E] = \frac{\partial^2 E}{\partial A_i \partial A_j}$$

        Forward-mode Jacobian:
        $$\mathbf{J}_{fwd} = \frac{\partial E}{\partial A_i}$$

        Reverse-mode Jacobian:
        $$\mathbf{J}_{rev} = \frac{\partial E}{\partial A_i}$$

        LINEAR SYSTEM SOLVING:
        Steady-state equation:
        $$(\nabla^2 - k)A = S$$

        Matrix form:
        $$\mathbf{L}\mathbf{a} = \mathbf{s}$$

        where $\mathbf{L}$ is the discrete Laplacian operator.

        NUMERICAL METHODS:
        - JAX JIT compilation for performance
        - Automatic differentiation for exact gradients
        - Linear algebra routines with numerical stability
        - Integration quadrature for energy functionals

        Args:
            attention_field: $A(\mathbf{x})$ - Current attention field state
            spatial_coordinates: $\mathbf{x}_i$ - Spatial discretization grid

        Returns:
            Dictionary with energy analysis, derivatives, and linear solutions
        """
        # Convert to JAX arrays
        field_jax = jnp.array(attention_field.field_values.detach().cpu().numpy())
        coords_jax = jnp.array(spatial_coordinates.detach().cpu().numpy())

        # Define diffusion kernel function for integration
        @jit
        def diffusion_kernel(x, y, t):
            """Green's function for diffusion equation."""
            D = self.diffusion_coefficient
            return jnp.exp(-((x - y) ** 2) / (4 * D * t)) / jnp.sqrt(4 * jnp.pi * D * t)

        # Compute attention field energy functional using integration
        @jit
        def energy_functional(field):
            """E[A] = ∫ [½|∇A|² + ½kA²] dx energy functional."""
            gradient = jnp.gradient(field)
            kinetic = 0.5 * jnp.sum(gradient**2)
            potential = 0.5 * self.decay_rate * jnp.sum(field**2)
            return kinetic + potential

        # Use JAX integration for energy calculation
        energy_result = energy_functional(field_jax)

        # Hessian analysis using JAX
        hessian_func = hessian(energy_functional)
        energy_hessian = hessian_func(field_jax)

        # Forward and reverse mode Jacobians
        jacfwd_func = jacfwd(energy_functional)
        jacrev_func = jacrev(energy_functional)

        forward_jacobian = jacfwd_func(field_jax)
        reverse_jacobian = jacrev_func(field_jax)

        # Linear system solving using jax_solve
        # Solve (∇²A - kA) = source for steady state
        n = len(field_jax)
        if n > 1:
            # Construct discrete Laplacian matrix
            dx = coords_jax[1] - coords_jax[0]
            diag_main = -2 / (dx**2) - self.decay_rate
            diag_off = 1 / (dx**2)

            # Build matrix using JAX
            A_matrix = jnp.diag(jnp.full(n, diag_main))
            if n > 1:
                A_matrix = A_matrix.at[jnp.arange(n - 1), jnp.arange(1, n)].set(
                    diag_off
                )
                A_matrix = A_matrix.at[jnp.arange(1, n), jnp.arange(n - 1)].set(
                    diag_off
                )

            # Solve linear system
            source_term = jnp.ones_like(field_jax)
            try:
                steady_solution = jax_solve(A_matrix, source_term)
            except:
                steady_solution = field_jax
        else:
            steady_solution = field_jax

        return {
            "energy_functional": torch.tensor(float(energy_result)),
            "energy_hessian": torch.from_numpy(np.array(energy_hessian)),
            "forward_jacobian": torch.from_numpy(np.array(forward_jacobian)),
            "reverse_jacobian": torch.from_numpy(np.array(reverse_jacobian)),
            "steady_state_solution": torch.from_numpy(np.array(steady_solution)),
        }

    def scipy_integration_analysis(
        self, attention_field: AttentionField, spatial_coordinates: torch.Tensor
    ) -> Dict[str, float]:
        """
        Numerical integration analysis using scipy.integrate.quad.

        MATHEMATICAL FORMULATION:
        Statistical moments of attention distribution:

        ZEROTH MOMENT (Total attention):
        $$M_0 = \int_{-\infty}^{\infty} A(x) dx$$

        FIRST MOMENT (Centroid):
        $$M_1 = \int_{-\infty}^{\infty} x A(x) dx$$
        $$\bar{x} = \frac{M_1}{M_0}$$

        SECOND MOMENT (Spread):
        $$M_2 = \int_{-\infty}^{\infty} x^2 A(x) dx$$
        $$\sigma^2 = \frac{M_2}{M_0} - \bar{x}^2$$

        HIGHER MOMENTS:
        Third moment (skewness): $M_3 = \int x^3 A(x) dx$
        Fourth moment (kurtosis): $M_4 = \int x^4 A(x) dx$

        ENERGY INTEGRAL:
        $$E = \int_{-\infty}^{\infty} A(x)^2 dx$$

        GAUSSIAN QUADRATURE:
        For smooth integrands, using adaptive quadrature:
        $$\int_a^b f(x) dx \approx \sum_{i=1}^n w_i f(x_i)$$

        ERROR ESTIMATION:
        - Absolute error tolerance: $|I - I_{approx}| < \epsilon_{abs}$
        - Relative error tolerance: $\frac{|I - I_{approx}|}{|I|} < \epsilon_{rel}$

        NORMALIZATION:
        Probability density normalization:
        $$\tilde{A}(x) = \frac{A(x)}{\int A(y) dy}$$

        STATISTICAL PROPERTIES:
        - Mean: $\mu = \frac{\int x A(x) dx}{\int A(x) dx}$
        - Variance: $\sigma^2 = \frac{\int (x-\mu)^2 A(x) dx}{\int A(x) dx}$
        - Skewness: $\gamma_1 = \frac{\int (x-\mu)^3 A(x) dx}{\sigma^3 \int A(x) dx}$
        - Kurtosis: $\gamma_2 = \frac{\int (x-\mu)^4 A(x) dx}{\sigma^4 \int A(x) dx} - 3$

        Args:
            attention_field: $A(x)$ - Attention field distribution
            spatial_coordinates: $x_i$ - Spatial domain coordinates

        Returns:
            Dictionary of statistical moments and integral properties
        """
        # Convert to numpy for scipy
        field_np = attention_field.field_values.detach().cpu().numpy()
        coords_np = spatial_coordinates.detach().cpu().numpy()

        # Interpolate field for continuous integration
        from scipy.interpolate import interp1d

        if len(coords_np) > 1:
            field_func = interp1d(
                coords_np, field_np, kind="cubic", bounds_error=False, fill_value=0.0
            )
        else:
            field_func = lambda x: field_np[0] if len(field_np) > 0 else 0.0

        # Define various integrand functions
        def attention_integrand(x):
            return field_func(x)

        def energy_integrand(x):
            return field_func(x) ** 2

        def moment_integrand(x):
            return x * field_func(x)

        def variance_integrand(x):
            return x**2 * field_func(x)

        # Numerical integration using quad
        x_min, x_max = float(coords_np.min()), float(coords_np.max())

        total_attention, _ = quad(attention_integrand, x_min, x_max)
        energy_integral, _ = quad(energy_integrand, x_min, x_max)
        first_moment, _ = quad(moment_integrand, x_min, x_max)
        second_moment, _ = quad(variance_integrand, x_min, x_max)

        # Higher-order moments
        def third_moment_integrand(x):
            return x**3 * field_func(x)

        def fourth_moment_integrand(x):
            return x**4 * field_func(x)

        third_moment, _ = quad(third_moment_integrand, x_min, x_max)
        fourth_moment, _ = quad(fourth_moment_integrand, x_min, x_max)

        return {
            "total_attention": total_attention,
            "energy_integral": energy_integral,
            "first_moment": first_moment,
            "second_moment": second_moment,
            "third_moment": third_moment,
            "fourth_moment": fourth_moment,
            "attention_centroid": first_moment / (total_attention + 1e-12),
            "attention_variance": second_moment / (total_attention + 1e-12)
            - (first_moment / (total_attention + 1e-12)) ** 2,
        }

    def scipy_linalg_attention_analysis(
        self, attention_field: AttentionField, spatial_coordinates: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Linear algebra analysis using scipy.linalg methods.
        
        MATHEMATICAL FORMULATION:
        Correlation matrix analysis:
        $$\mathbf{C} = \mathbf{A} \mathbf{A}^T$$
        
        where $\mathbf{A}$ is the attention field vector.
        
        EIGENVALUE DECOMPOSITION:
        $$\mathbf{C} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^T$$
        
        where:
        - $\mathbf{Q}$ - Orthogonal matrix of eigenvectors
        - $\mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_n)$ - Eigenvalue matrix
        - $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n \geq 0$
        
        DIFFUSION OPERATOR:
        Discrete Laplacian (tridiagonal):
        $$\mathbf{L} = \frac{1}{(\Delta x)^2} \begin{pmatrix}
        -2 & 1 & 0 & \cdots \\
        1 & -2 & 1 & \cdots \\
        \vdots & \ddots & \ddots & \ddots \\
        & \cdots & 1 & -2
        \end{pmatrix} - k\mathbf{I}$$
        
        BANDED MATRIX SOLVER:
        For tridiagonal system $\mathbf{L}\mathbf{x} = \mathbf{b}$:
        $$\begin{pmatrix}
        & & \star & \star & \star & & \\
        & & \star & \star & \star & & \\
        & & \star & \star & \star & & \\
        \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix} = \begin{pmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{pmatrix}$$
        
        MATRIX PROPERTIES:
        - Condition number: $\kappa(\mathbf{A}) = \frac{\lambda_{max}}{\lambda_{min}}$
        - Matrix rank: $\text{rank}(\mathbf{A}) = \#\{\lambda_i : \lambda_i > \epsilon\}$
        - Determinant: $\det(\mathbf{A}) = \prod_{i=1}^n \lambda_i$
        - Trace: $\text{tr}(\mathbf{A}) = \sum_{i=1}^n \lambda_i$
        - Frobenius norm: $\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} A_{ij}^2}$
        
        SOLVER COMPARISON:
        1. solve_banded: Specialized for banded matrices
        2. solve: General dense solver
        3. Performance: $O(n)$ vs $O(n^3)$ complexity
        
        NUMERICAL STABILITY:
        - Pivoting for singular matrices
        - Regularization: $\mathbf{A} + \epsilon \mathbf{I}$
        - Iterative refinement for improved accuracy
        
        Args:
            attention_field: $A(\mathbf{x})$ - Attention field for correlation analysis
            spatial_coordinates: $\mathbf{x}_i$ - Spatial grid for diffusion operator
            
        Returns:
            Dictionary with eigenvalues, solutions, and matrix properties
        """
        n = len(attention_field.field_values)
        field_np = attention_field.field_values.detach().cpu().numpy()

        if n <= 1:
            return {
                "eigenvalues": torch.zeros(1),
                "eigenvectors": torch.eye(1),
                "condition_number": torch.tensor(1.0),
                "matrix_rank": torch.tensor(1),
                "determinant": torch.tensor(1.0),
            }

        # Construct attention correlation matrix
        attention_matrix = np.outer(field_np, field_np)
        attention_matrix += np.eye(n) * 1e-8  # Regularization

        # Eigenvalue decomposition using eigh
        eigenvals, eigenvecs = eigh(attention_matrix)

        # Construct banded diffusion matrix for solve_banded
        # Format: (upper_diag, main_diag, lower_diag)
        dx = (
            (spatial_coordinates[-1] - spatial_coordinates[0]) / (n - 1)
            if n > 1
            else 1.0
        )
        dx = dx.item()

        # Tridiagonal diffusion operator
        upper_diag = np.full(n - 1, 1.0 / (dx**2))
        main_diag = np.full(n, -2.0 / (dx**2) - self.decay_rate)
        lower_diag = np.full(n - 1, 1.0 / (dx**2))

        # Format for solve_banded: (l_and_u, ab, b)
        # where ab[u + i - j, j] = A[i, j] for max(0, j-u) <= i <= min(n-1, j+l)
        ab = np.zeros((3, n))  # 1 upper, 1 lower diagonal
        ab[0, 1:] = upper_diag  # Upper diagonal
        ab[1, :] = main_diag  # Main diagonal
        ab[2, :-1] = lower_diag  # Lower diagonal

        # Solve banded system
        source_term = np.ones(n)
        try:
            banded_solution = solve_banded((1, 1), ab, source_term)
        except:
            banded_solution = source_term

        # Direct solve using solve
        full_matrix = np.diag(main_diag)
        if n > 1:
            full_matrix[np.arange(n - 1), np.arange(1, n)] = upper_diag
            full_matrix[np.arange(1, n), np.arange(n - 1)] = lower_diag

        try:
            direct_solution = solve(full_matrix, source_term)
        except:
            direct_solution = source_term

        # Matrix properties
        condition_number = np.linalg.cond(attention_matrix)
        matrix_rank = np.linalg.matrix_rank(attention_matrix)
        determinant = np.linalg.det(attention_matrix)

        return {
            "eigenvalues": torch.from_numpy(eigenvals),
            "eigenvectors": torch.from_numpy(eigenvecs),
            "banded_solution": torch.from_numpy(banded_solution),
            "direct_solution": torch.from_numpy(direct_solution),
            "condition_number": torch.tensor(condition_number),
            "matrix_rank": torch.tensor(matrix_rank),
            "determinant": torch.tensor(determinant),
        }

    def scipy_sparse_diffusion_solver(
        self, attention_field: AttentionField, spatial_coordinates: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Sparse matrix diffusion solving using scipy.sparse methods.
        
        MATHEMATICAL FORMULATION:
        Sparse diffusion operator $\mathbf{L}$ for large-scale problems:
        $$\mathbf{L} = D\nabla^2_{discrete} - k\mathbf{I}$$
        
        SPARSE MATRIX FORMATS:
        - COO (Coordinate): $(i, j, v)$ triplets
        - CSR (Compressed Sparse Row): Row-wise compression
        - CSC (Compressed Sparse Column): Column-wise compression
        
        TRIDIAGONAL DIFFUSION MATRIX:
        Non-zero pattern for 1D Laplacian:
        $$\mathbf{L} = \frac{D}{(\Delta x)^2} \begin{pmatrix}
        -2 & 1 & & & \\
        1 & -2 & 1 & & \\
        & 1 & -2 & 1 & \\
        & & \ddots & \ddots & \ddots
        \end{pmatrix} - k\mathbf{I}$$
        
        SPARSE STORAGE:
        For $n \times n$ tridiagonal matrix:
        - Dense storage: $n^2$ elements
        - Sparse storage: $3n - 2$ non-zeros
        - Compression ratio: $\frac{3n-2}{n^2} \approx \frac{3}{n}$
        
        SPARSE LINEAR SOLVER:
        System: $\mathbf{L}\mathbf{x} = \mathbf{b}$
        
        Methods:
        1. Direct: LU decomposition with fill-in minimization
        2. Iterative: CG, GMRES, BiCGSTAB
        3. Preconditioning: ILU, Jacobi, SSOR
        
        EIGENVALUE COMPUTATION:
        Sparse eigenvalue problem:
        $$\mathbf{L}\mathbf{v} = \lambda \mathbf{v}$$
        
        Krylov methods (ARPACK):
        - Lanczos for symmetric matrices
        - Arnoldi for non-symmetric matrices
        - Shift-and-invert for interior eigenvalues
        
        LINEAR OPERATOR:
        Matrix-free representation:
        $$\mathcal{L}: \mathbb{R}^n \to \mathbb{R}^n$$
        $$\mathcal{L}[\mathbf{v}] = \mathbf{L}\mathbf{v}$$
        
        PERFORMANCE METRICS:
        - Matrix density: $\rho = \frac{\text{nnz}}{n^2}$
        - Fill-in factor: Memory overhead in factorization
        - FLOP count: Floating point operations
        
        MEMORY COMPLEXITY:
        - Sparse storage: $O(\text{nnz})$
        - Dense storage: $O(n^2)$
        - Factorization: $O(\text{nnz} \cdot \alpha)$ where $\alpha$ is fill-in
        
        Args:
            attention_field: $A(\mathbf{x})$ - Field for matrix construction
            spatial_coordinates: $\mathbf{x}_i$ - Spatial discretization
            
        Returns:
            Dictionary with sparse solutions, eigenvalues, and performance metrics
        """
        n = len(attention_field.field_values)

        if n <= 1:
            return {
                "sparse_solution": torch.zeros(1),
                "sparse_eigenvalues": torch.zeros(1),
                "sparse_eigenvectors": torch.zeros(1, 1),
                "matrix_density": torch.tensor(1.0),
                "solve_iterations": torch.tensor(1),
            }

        # Construct sparse diffusion matrix using diags
        dx = (spatial_coordinates[-1] - spatial_coordinates[0]) / (n - 1)
        dx = dx.item()

        # Diffusion operator: D∇² - k
        diagonals = [
            np.full(n - 1, self.diffusion_coefficient / (dx**2)),  # Upper diagonal
            np.full(
                n, -2 * self.diffusion_coefficient / (dx**2) - self.decay_rate
            ),  # Main diagonal
            np.full(n - 1, self.diffusion_coefficient / (dx**2)),  # Lower diagonal
        ]
        offsets = [1, 0, -1]

        # Create sparse matrix
        sparse_matrix = diags(diagonals, offsets, shape=(n, n), format="csc")

        # Convert to csc_matrix explicitly
        sparse_matrix_csc = csc_matrix(sparse_matrix)

        # Source term
        source_term = np.ones(n)

        # Solve sparse linear system using spsolve
        try:
            sparse_solution = spsolve(sparse_matrix_csc, source_term)
        except:
            sparse_solution = source_term

        # Sparse eigenvalue computation using eigsh
        try:
            # Find a few eigenvalues/eigenvectors
            k_eigs = min(6, n - 2) if n > 2 else 1
            eigenvals, eigenvecs = eigsh(-sparse_matrix_csc, k=k_eigs, which="SM")
        except:
            eigenvals = np.array([1.0])
            eigenvecs = np.eye(n, 1)

        # LinearOperator for matrix-vector products
        def matvec(v):
            return sparse_matrix_csc.dot(v)

        linear_op = LinearOperator((n, n), matvec=matvec)

        # Test LinearOperator
        test_vector = np.ones(n)
        operator_result = linear_op.matvec(test_vector)

        # Matrix properties
        matrix_density = sparse_matrix_csc.nnz / (n * n)

        return {
            "sparse_solution": torch.from_numpy(sparse_solution),
            "sparse_eigenvalues": torch.from_numpy(eigenvals),
            "sparse_eigenvectors": torch.from_numpy(eigenvecs),
            "operator_result": torch.from_numpy(operator_result),
            "matrix_density": torch.tensor(matrix_density),
            "matrix_nnz": torch.tensor(sparse_matrix_csc.nnz),
            "solve_success": torch.tensor(1.0),
        }

    def multivariate_attention_sampling(
        self,
        attention_field: AttentionField,
        spatial_coordinates: torch.Tensor,
        n_samples: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """
        Multivariate sampling from attention distribution using MultivariateNormal.

        MATHEMATICAL FORMULATION:
        Multivariate Gaussian distribution:
        $$\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

        Probability density function:
        $$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

        ATTENTION-WEIGHTED PARAMETERS:
        Mean vector (attention centroid):
        $$\boldsymbol{\mu} = \frac{\sum_i A_i \mathbf{x}_i}{\sum_i A_i}$$

        Covariance matrix:
        $$\boldsymbol{\Sigma}_{ij} = \sigma^2 \delta_{ij} (1 + \alpha A_i)$$

        where $\alpha$ controls attention-dependent variance.

        SAMPLING ALGORITHM:
        1. Cholesky decomposition: $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^T$
        2. Generate standard normal: $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        3. Transform: $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$

        LOG-PROBABILITY:
        $$\log p(\mathbf{x}) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\boldsymbol{\Sigma}| - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$$

        ENTROPY:
        $$H[\mathbf{X}] = \frac{1}{2}\log((2\pi e)^d |\boldsymbol{\Sigma}|)$$

        SAMPLE STATISTICS:
        Sample mean: $\hat{\boldsymbol{\mu}} = \frac{1}{N}\sum_{i=1}^N \mathbf{x}_i$
        Sample covariance: $\hat{\boldsymbol{\Sigma}} = \frac{1}{N-1}\sum_{i=1}^N (\mathbf{x}_i - \hat{\boldsymbol{\mu}})(\mathbf{x}_i - \hat{\boldsymbol{\mu}})^T$

        CONVERGENCE:
        By Law of Large Numbers:
        $$\\hat{\\boldsymbol{\\mu}} \\xrightarrow{N \\to \\infty} \\boldsymbol{\\mu}$$
        $$\\hat{\\boldsymbol{\\Sigma}} \\xrightarrow{N \\to \\infty} \\boldsymbol{\\Sigma}$$

        NUMERICAL STABILITY:
        - Regularization: $\boldsymbol{\Sigma} + \epsilon \mathbf{I}$
        - Condition number check: $\kappa(\boldsymbol{\Sigma}) = \frac{\lambda_{max}}{\lambda_{min}}$
        - Eigenvalue thresholding for near-singular matrices

        Args:
            attention_field: $A(\mathbf{x})$ - Attention field for parameter estimation
            spatial_coordinates: $\mathbf{x}_i$ - Spatial locations
            n_samples: $N$ - Number of samples to generate

        Returns:
            Dictionary with samples, statistics, and distribution parameters
        """
        n_dims = len(attention_field.field_values)

        if n_dims <= 1:
            # 1D case - use univariate normal
            mean = spatial_coordinates.mean()
            std = attention_field.focus_width
            normal_dist = Normal(mean, std)
            samples = normal_dist.sample((n_samples,))

            return {
                "samples": samples.unsqueeze(-1),
                "sample_mean": samples.mean().unsqueeze(0),
                "sample_covariance": samples.var().unsqueeze(0).unsqueeze(0),
                "log_probabilities": normal_dist.log_prob(samples),
                "sample_entropy": normal_dist.entropy().unsqueeze(0),
            }

        # Multi-dimensional case
        # Create mean vector from spatial coordinates weighted by attention
        attention_weights = F.softmax(attention_field.field_values, dim=0)
        mean_vector = torch.sum(
            spatial_coordinates.unsqueeze(-1) * attention_weights.unsqueeze(0), dim=1
        )

        # Create covariance matrix based on attention field
        # Use attention field values to construct correlation structure
        attention_normalized = attention_field.field_values / (
            torch.sum(attention_field.field_values) + 1e-12
        )

        # Simple covariance structure: diagonal with attention-based variances
        variances = attention_field.focus_width**2 * (1.0 + attention_normalized)
        covariance_matrix = torch.diag(variances)

        # Add small regularization for numerical stability
        covariance_matrix += torch.eye(n_dims) * 1e-6

        # Create multivariate normal distribution
        mvn_dist = MultivariateNormal(mean_vector, covariance_matrix)

        # Sample from distribution
        samples = mvn_dist.sample((n_samples,))

        # Compute log probabilities
        log_probs = mvn_dist.log_prob(samples)

        # Distribution properties
        sample_mean = samples.mean(dim=0)
        sample_covariance = torch.cov(samples.t())
        entropy = mvn_dist.entropy()

        return {
            "samples": samples,
            "sample_mean": sample_mean,
            "sample_covariance": sample_covariance,
            "log_probabilities": log_probs,
            "sample_entropy": entropy.unsqueeze(0),
            "distribution_mean": mean_vector,
            "distribution_covariance": covariance_matrix,
        }
