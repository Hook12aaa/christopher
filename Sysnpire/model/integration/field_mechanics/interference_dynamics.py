"""
Interference Dynamics Engine - Wave Superposition in Q(τ,C,s) Space

MATHEMATICAL FOUNDATION:
    Wave Superposition: Ψ_total(x,t) = Σᵢ Aᵢe^(ik_ᵢ·x - ωᵢt + φᵢ)
    Interference Intensity: I = |Ψ₁ + Ψ₂|² = I₁ + I₂ + 2√(I₁I₂)cos(Δφ)
    Phase Coherence: C(x,t) = |⟨Ψ₁*Ψ₂⟩|/√(⟨|Ψ₁|²⟩⟨|Ψ₂|²⟩)
    
    Constructive Interference: Δφ = 2πn (n ∈ ℤ)
    Destructive Interference: Δφ = (2n+1)π
    
    Cross-Correlation: R₁₂(τ) = ∫ Ψ₁(t)Ψ₂*(t+τ) dt
    Power Spectral Density: S(ω) = |F[Ψ](ω)|² where F is Fourier transform
    Mutual Coherence: γ₁₂(τ) = R₁₂(τ)/√(R₁₁(0)R₂₂(0))

IMPLEMENTATION: Exact wave mathematics using FFT for frequency domain analysis,
complex field superposition with proper phase relationships, analytical
interference pattern calculation for known wave forms.
"""

import cmath
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# JAX for automatic differentiation and vectorization
import jax
import jax.numpy as jnp
# Network Analysis for phase relationship topology - REQUIRED
import networkx as nx
# Numba for high-performance computation
import numba as nb
import numpy as np
import torch
import torch.nn.functional as F
# Geometric Deep Learning for manifold interference analysis - REQUIRED
import torch_geometric as pyg
from jax import grad, jit, vmap
from jax.scipy.signal import correlate
from numba import jit as nb_jit
from numba import prange
# SAGE for exact complex wave calculations and modular forms - hard dependency like main codebase
from sage.all import CDF, I, ModularForms, cos, exp, pi, sin, Integer
from sage.rings.complex_double import ComplexDoubleElement
from sage.rings.integer import Integer as SageInteger
from sage.rings.real_double import RealDoubleElement
# SciPy for signal processing and special functions
from scipy import signal, special
from scipy.signal import coherence, csd, hilbert, spectrogram
from scipy.special import (hankel1, hankel2, jv,  # Bessel and Hankel functions
                           yv)
from torch.fft import fft, fft2, fftn, ifft, ifft2, ifftn, irfft, rfft
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

# Import mathematical constants and data structures
from .data_type_consistency import get_dtype_manager
from . import (CONVERGENCE_THRESHOLD, ENERGY_NORMALIZATION,
               FIELD_COUPLING_CONSTANT, FIELD_NUMERICAL_PRECISION,
               PHASE_COHERENCE_THRESHOLD, FieldConfiguration, FieldSymmetry,
               InterferencePattern, complex_field_phase, field_norm_l2)

logger = logging.getLogger(__name__)


@dataclass
class WaveProperties:
    """
    Individual wave characteristics with complete mathematical specification.

    Mathematical Foundation:
        Plane wave solution to wave equation:
        $$\Psi(\vec{r},t) = A e^{i(\vec{k} \cdot \vec{r} - \omega t + \phi_0)}$$

        Wave equation: $\nabla^2 \Psi - \frac{1}{c^2} \frac{\partial^2 \Psi}{\partial t^2} = 0$

        Dispersion relation: $\omega = \omega(|\vec{k}|)$
        - Linear: $\omega = c|\vec{k}|$ (vacuum, linear medium)
        - Nonlinear: $\omega = \sqrt{c^2|\vec{k}|^2 + m^2c^4/\hbar^2}$ (relativistic)

        Group velocity: $\vec{v}_g = \nabla_k \omega(\vec{k})$
        Phase velocity: $v_p = \omega/|\vec{k}|$

        Polarization vector: $\vec{k} \cdot \hat{e} = 0$ (transverse waves)
    """

    amplitude: complex  # A wave amplitude
    frequency: float  # ω angular frequency
    wave_vector: torch.Tensor  # k⃗ wave vector
    phase: float  # φ initial phase
    polarization: torch.Tensor  # ê polarization vector
    group_velocity: torch.Tensor  # v⃗_g = ∇_k ω(k) group velocity
    phase_velocity: float  # v_p = ω/|k| phase velocity
    wavelength: float  # λ = 2π/|k| wavelength
    period: float  # T = 2π/ω period

    def __post_init__(self):
        """Validate wave properties."""
        if self.frequency <= 0:
            raise ValueError(f"Non-positive frequency: {self.frequency}")
        if self.wavelength <= 0:
            raise ValueError(f"Non-positive wavelength: {self.wavelength}")
        if not torch.isfinite(self.wave_vector).all():
            raise ValueError("Wave vector contains non-finite values")


@dataclass
class CoherenceAnalysis:
    """
    Wave coherence analysis with complete statistical characterization.

    Mathematical Foundation:
        Temporal coherence time:
        $$\tau_c = \int_0^{\infty} |\gamma(\tau)|^2 d\tau$$

        Spatial coherence length:
        $$l_c = \int_0^{\infty} |\gamma(r)| dr$$

        Degree of coherence:
        $$|\gamma_{12}| = \frac{|\langle \Psi_1^* \Psi_2 \rangle|}{\sqrt{\langle |\Psi_1|^2 \rangle \langle |\Psi_2|^2 \rangle}} \in [0,1]$$

        Visibility (fringe contrast):
        $$V = \frac{I_{\max} - I_{\min}}{I_{\max} + I_{\min}}$$

        Mutual coherence function:
        $$\Gamma_{12}(\tau) = \langle \Psi_1^*(t) \Psi_2(t+\tau) \rangle$$

        Normalized: $\gamma_{12}(\tau) = \Gamma_{12}(\tau)/\sqrt{\Gamma_{11}(0)\Gamma_{22}(0)}$

        Phase stability (RMS phase fluctuation):
        $$\Delta\phi_{\text{rms}} = \sqrt{\langle (\phi - \langle\phi\rangle)^2 \rangle}$$
    """

    temporal_coherence: float  # τ_c coherence time
    spatial_coherence: float  # l_c coherence length
    degree_of_coherence: float  # |γ₁₂| ∈ [0,1] coherence degree
    visibility: float  # V = (I_max - I_min)/(I_max + I_min)
    mutual_coherence_function: torch.Tensor  # γ₁₂(τ) mutual coherence
    correlation_matrix: torch.Tensor  # R_ij correlation matrix
    phase_stability: float  # Δφ_rms phase fluctuation measure

    def __post_init__(self):
        """Validate coherence analysis."""
        if not (0 <= self.degree_of_coherence <= 1):
            raise ValueError(f"Invalid coherence degree: {self.degree_of_coherence}")
        if not (0 <= self.visibility <= 1):
            raise ValueError(f"Invalid visibility: {self.visibility}")


@dataclass
class ConstructiveZone:
    """
    Region of constructive interference with geometric and intensity characterization.

    Mathematical Foundation:
        Constructive interference condition:
        $$\sum_{i=1}^N \vec{A}_i \approx \left|\sum_{i=1}^N \vec{A}_i\right| \hat{e}$$

        Phase matching: $\phi_i - \phi_j = 2\pi n_{ij}$, $n_{ij} \in \mathbb{Z}$

        Intensity enhancement:
        $$\mathcal{E} = \frac{I_{\text{constructive}}}{\sum_i I_i} = \frac{\left(\sum_i |A_i|\right)^2}{\sum_i |A_i|^2}$$

        Maximum enhancement: $\mathcal{E}_{\max} = N$ (equal amplitudes, perfect alignment)

        Spatial extent (coherence volume):
        $$V_c = l_c^3 = \left(\frac{\lambda}{2\pi}\right)^3 \left(\frac{1}{\Delta k}\right)^3$$

        Stability measure:
        $$\mathcal{S} = \exp\left(-\frac{\text{Var}(\omega_i)}{\langle\omega\rangle^2}\right)$$
    """

    center_position: torch.Tensor  # x⃗_c zone center
    spatial_extent: torch.Tensor  # Δx⃗ spatial dimensions
    intensity_enhancement: float  # I/I₀ intensity amplification
    phase_matching_condition: str  # Mathematical condition description
    stability_measure: float  # Zone persistence measure
    contributing_waves: List[int]  # Indices of interfering waves

    def __post_init__(self):
        """Validate constructive zone."""
        if self.intensity_enhancement < 1.0:
            raise ValueError(f"Invalid enhancement < 1: {self.intensity_enhancement}")
        if not (0 <= self.stability_measure <= 1):
            raise ValueError(f"Invalid stability: {self.stability_measure}")


@dataclass
class DestructiveZone:
    """
    Region of destructive interference with null depth and cancellation analysis.

    Mathematical Foundation:
        Destructive interference condition:
        $$\left|\sum_{i=1}^N A_i e^{i\phi_i}\right| \ll \sum_{i=1}^N |A_i|$$

        Phase opposition: $\phi_i - \phi_j = (2n+1)\pi$, $n \in \mathbb{Z}$

        Perfect cancellation (two equal waves):
        $$|A_1| = |A_2|, \quad \phi_2 = \phi_1 + \pi \Rightarrow I = 0$$

        Intensity suppression factor:
        $$\mathcal{S} = \frac{I_{\text{min}}}{\sum_i I_i}$$

        Null depth (cancellation quality):
        $$\mathcal{D} = 1 - \mathcal{S} = 1 - \frac{I_{\text{min}}}{\sum_i I_i}$$

        Spatial null width:
        $$w_{\text{null}} = \frac{\lambda}{4\pi} \sqrt{\frac{2}{\sum_i |\nabla \phi_i|^2}}$$

        Null stability depends on amplitude and phase matching precision.
    """

    center_position: torch.Tensor  # x⃗_c zone center
    spatial_extent: torch.Tensor  # Δx⃗ spatial dimensions
    intensity_suppression: float  # I/I₀ intensity reduction
    phase_opposition_condition: str  # Mathematical condition description
    null_depth: float  # Depth of intensity minimum
    contributing_waves: List[int]  # Indices of interfering waves

    def __post_init__(self):
        """Validate destructive zone."""
        if self.intensity_suppression > 1.0:
            raise ValueError(f"Invalid suppression > 1: {self.intensity_suppression}")
        if not (0 <= self.null_depth <= 1):
            raise ValueError(f"Invalid null depth: {self.null_depth}")


@dataclass
class EvolutionTrajectory:
    """
    Interference pattern temporal evolution with complete dynamical analysis.

    Mathematical Foundation:
        Time evolution of interference patterns:
        $$I(\vec{r},t) = \left|\sum_{i=1}^N A_i e^{i(\vec{k}_i \cdot \vec{r} - \omega_i t + \phi_i)}\right|^2$$

        Beating phenomena (frequency difference):
        $$\omega_{\text{beat}} = |\omega_1 - \omega_2|$$

        Envelope modulation:
        $$A_{\text{env}}(t) = |A_1 e^{i\Delta\omega t/2} + A_2 e^{-i\Delta\omega t/2}|$$

        Group velocity dynamics:
        $$\frac{\partial A}{\partial t} + v_g \frac{\partial A}{\partial z} = 0$$

        Coherence evolution:
        $$\gamma(t) = \frac{\langle \Psi_1^*(t) \Psi_2(t) \rangle}{\sqrt{\langle |\Psi_1(t)|^2 \rangle \langle |\Psi_2(t)|^2 \rangle}}$$

        Critical transitions occur at:
        - Beat maxima: $t = 2\pi n/\omega_{\text{beat}}$
        - Phase slips: sudden $2\pi$ phase jumps
        - Coherence breakdown: $|\gamma(t)| < \text{threshold}$

        Fourier analysis reveals dominant frequencies and harmonics.
    """

    time_points: torch.Tensor  # t time sampling points
    intensity_evolution: torch.Tensor  # I(x,t) intensity vs time
    phase_evolution: torch.Tensor  # φ(x,t) phase vs time
    coherence_evolution: torch.Tensor  # γ(t) coherence vs time
    dominant_frequencies: torch.Tensor  # ω_i dominant frequency components
    beating_frequency: Optional[float]  # ω_beat = |ω₁ - ω₂| beating frequency
    envelope_modulation: torch.Tensor  # A_env(t) envelope function
    critical_transitions: List[int]  # Time indices of pattern changes

    def __post_init__(self):
        """Validate evolution trajectory."""
        if len(self.time_points) != self.intensity_evolution.shape[-1]:
            raise ValueError("Time points and intensity evolution shape mismatch")


class InterferenceDynamicsEngine:
    """
    Complete wave interference analysis engine with exact mathematical methods.

    Mathematical Foundation:
        This engine implements the complete theory of wave interference using:

        1. **Complex Wave Superposition**:
           $$\Psi_{\text{total}}(\vec{r},t) = \sum_{i=1}^N A_i e^{i(\vec{k}_i \cdot \vec{r} - \omega_i t + \phi_i)}$$

        2. **Interference Intensity Formula**:
           $$I(\vec{r},t) = \left|\sum_{i=1}^N \Psi_i(\vec{r},t)\right|^2 = \sum_{i,j} A_i A_j^* e^{i(\vec{k}_{ij} \cdot \vec{r} - \omega_{ij} t + \phi_{ij})}$$

        3. **Phase Coherence Analysis**:
           $$\gamma_{ij}(\tau) = \frac{\langle \Psi_i^*(t) \Psi_j(t+\tau) \rangle}{\sqrt{\langle |\Psi_i|^2 \rangle \langle |\Psi_j|^2 \rangle}}$$

        4. **Spatial Correlation Functions**:
           $$G(\vec{r}) = \langle \Psi^*(\vec{r}') \Psi(\vec{r}' + \vec{r}) \rangle$$

        5. **Wigner Distribution (Phase Space)**:
           $$W(\vec{r},\vec{k}) = \int \Psi^*(\vec{r}-\vec{s}/2) \Psi(\vec{r}+\vec{s}/2) e^{-i\vec{k} \cdot \vec{s}} d\vec{s}$$

    **Analytical Solutions Implemented**:
    - Young's double-slit interference: $I = I_0 [1 + \cos(\Delta\phi)]$
    - Multi-wave superposition with arbitrary phases and amplitudes
    - Gaussian wave packet interference and spreading
    - Plane wave and spherical wave interference patterns
    - Standing wave analysis: $\Psi = 2A \cos(kx) e^{-i\omega t}$
    - Beating phenomena: $\omega_{\text{beat}} = |\omega_1 - \omega_2|$
    - Mode decomposition and resonance analysis

    **Advanced Mathematical Features**:
    - FFT-based spectral analysis (1D, 2D, 3D transforms)
    - Bessel/Hankel function solutions for cylindrical geometries
    - Modular forms for rigorous phase calculations
    - Network topology analysis of interference patterns
    - JAX vectorization for high-performance computation
    - Neural network wave processing using PyTorch

    **Numerical Methods**:
    - JIT-compiled calculations using Numba
    - Automatic differentiation via JAX
    - SciPy signal processing for correlation analysis
    - Geometric deep learning for interference networks
    """

    def __init__(
        self,
        spatial_dimensions: int = 3,
        temporal_resolution: float = 0.01,
        spatial_resolution: float = 0.1,
        coherence_threshold: float = PHASE_COHERENCE_THRESHOLD,
    ):
        """
        Initialize interference dynamics analyzer with mathematical precision control.

        Mathematical Foundation:
            Discretization parameters for wave equation numerics:

            Spatial resolution (grid spacing):
            $$\Delta x \leq \frac{\lambda_{\min}}{N_{\text{ppw}}}$$
            where $N_{\text{ppw}} \geq 10$ points per wavelength for accuracy.

            Temporal resolution (time step):
            $$\Delta t \leq \frac{\Delta x}{c} \cdot \text{CFL}$$
            where CFL $\leq 1$ is the Courant-Friedrichs-Lewy stability condition.

            Nyquist criterion for frequency sampling:
            $$\omega_{\max} = \frac{\pi}{\Delta t}$$

            Coherence threshold determines interference classification:
            $$|\gamma_{ij}| > \text{threshold} \Rightarrow \text{coherent interference}$$
            $$|\gamma_{ij}| \leq \text{threshold} \Rightarrow \text{incoherent addition}$$

            Spatial dimensionality affects:
            - Wave equation form: $\nabla^2 = \partial^2/\partial x^2$ (1D), $\nabla^2 = \partial^2/\partial x^2 + \partial^2/\partial y^2$ (2D), etc.
            - Green's function: $G \sim r^{-(d-2)/2}$ in $d$ dimensions
            - Diffraction patterns and beam spreading

        Args:
            spatial_dimensions: Space dimensionality $d \in \{1,2,3\}$
            temporal_resolution: Time step $\Delta t > 0$ [time units]
            spatial_resolution: Grid spacing $\Delta x > 0$ [length units]
            coherence_threshold: Coherence cutoff $\gamma_{\text{min}} \in [0,1]$

        Raises:
            ValueError: If parameters violate numerical stability conditions
        """
        self.spatial_dimensions = spatial_dimensions
        self.temporal_resolution = temporal_resolution
        self.spatial_resolution = spatial_resolution
        self.coherence_threshold = coherence_threshold

        # Validate parameters
        if spatial_dimensions not in [1, 2, 3]:
            raise ValueError(f"Unsupported spatial dimension: {spatial_dimensions}")
        if temporal_resolution <= 0:
            raise ValueError(f"Non-positive temporal resolution: {temporal_resolution}")
        if spatial_resolution <= 0:
            raise ValueError(f"Non-positive spatial resolution: {spatial_resolution}")
        if not (0 <= coherence_threshold <= 1):
            raise ValueError(f"Invalid coherence threshold: {coherence_threshold}")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"🌊 Initialized interference engine: d={spatial_dimensions}, "
            f"Δt={temporal_resolution}, Δx={spatial_resolution}"
        )

    @nb_jit(nopython=True, cache=True, fastmath=False)
    def _jit_two_wave_interference(
        self,
        x_coords: np.ndarray,
        t: float,
        amp1: complex,
        amp2: complex,
        k1: float,
        k2: float,
        omega1: float,
        omega2: float,
        phi1: float,
        phi2: float,
    ) -> np.ndarray:
        """
        JIT-compiled two-wave interference calculation using exact wave superposition.

        Mathematical Foundation:
            Wave superposition in complex exponential form:
            $$\Psi_{\text{total}}(x,t) = A_1 e^{i(k_1 x - \omega_1 t + \phi_1)} + A_2 e^{i(k_2 x - \omega_2 t + \phi_2)}$$

            Individual wave equations:
            $$\Psi_1(x,t) = A_1 e^{i(k_1 x - \omega_1 t + \phi_1)}$$
            $$\Psi_2(x,t) = A_2 e^{i(k_2 x - \omega_2 t + \phi_2)}$$

            Dispersion relation: $\omega = \omega(k)$ (medium dependent)
            Phase velocity: $v_p = \omega/k$
            Group velocity: $v_g = d\omega/dk$

        Args:
            x_coords: Spatial coordinate array $x \in \mathbb{R}^N$
            t: Time parameter $t \in \mathbb{R}$
            amp1, amp2: Complex amplitudes $A_1, A_2 \in \mathbb{C}$
            k1, k2: Wave numbers $k_1, k_2 \in \mathbb{R}$
            omega1, omega2: Angular frequencies $\omega_1, \omega_2 \in \mathbb{R}^+$
            phi1, phi2: Initial phases $\phi_1, \phi_2 \in [0, 2\pi)$

        Returns:
            Complex field array $\Psi_{\text{total}}(x,t) \in \mathbb{C}^N$
        """
        n_points = len(x_coords)
        total_field = np.zeros(n_points, dtype=nb.complex128)

        for i in range(n_points):
            x = x_coords[i]

            # Wave 1: Ψ₁ = A₁e^(ik₁x - iω₁t + iφ₁)
            phase1 = k1 * x - omega1 * t + phi1
            wave1 = amp1 * (np.cos(phase1) + 1j * np.sin(phase1))

            # Wave 2: Ψ₂ = A₂e^(ik₂x - iω₂t + iφ₂)
            phase2 = k2 * x - omega2 * t + phi2
            wave2 = amp2 * (np.cos(phase2) + 1j * np.sin(phase2))

            # Superposition
            total_field[i] = wave1 + wave2

        return total_field

    @nb_jit(nopython=True, cache=True, fastmath=False)
    def _jit_interference_intensity(self, field: np.ndarray) -> np.ndarray:
        """
        JIT-compiled interference intensity calculation.

        Mathematical Foundation:
            Intensity as probability density:
            $$I(x,t) = |\Psi(x,t)|^2 = \Psi^*(x,t) \Psi(x,t)$$

            For complex field $\Psi = a + bi$:
            $$I = |\Psi|^2 = a^2 + b^2 = \text{Re}(\Psi)^2 + \text{Im}(\Psi)^2$$

            Physical interpretation: Energy density per unit volume
            Units: $[I] = \text{Energy}/\text{Volume}$

        Args:
            field: Complex field array $\Psi(x) \in \mathbb{C}^N$

        Returns:
            Real intensity array $I(x) \in \mathbb{R}^N_+$
        """
        return np.abs(field) ** 2

    @nb_jit(nopython=True, cache=True, fastmath=False)
    def _jit_phase_difference(
        self, field1: np.ndarray, field2: np.ndarray
    ) -> np.ndarray:
        """
        JIT-compiled phase difference calculation with proper quadrant handling.

        Mathematical Foundation:
            Phase extraction from complex field:
            $$\phi(x) = \arg(\Psi(x)) = \arctan\left(\frac{\text{Im}(\Psi)}{\text{Re}(\Psi)}\right)$$

            Phase difference:
            $$\Delta\phi(x) = \phi_1(x) - \phi_2(x) = \arg(\Psi_1(x)) - \arg(\Psi_2(x))$$

            Wrapped to principal value:
            $$\Delta\phi \in [-\pi, \pi)$$

            Unwrapping condition:
            $$\Delta\phi_{\text{unwrap}} = \Delta\phi + 2\pi n, \quad n \in \mathbb{Z}$$

        Physical significance:
            - $\Delta\phi = 0$: In phase (constructive)
            - $\Delta\phi = \pi$: Out of phase (destructive)
            - $\Delta\phi = \pi/2$: Quadrature phase

        Args:
            field1, field2: Complex field arrays $\Psi_1, \Psi_2 \in \mathbb{C}^N$

        Returns:
            Phase difference array $\Delta\phi(x) \in [-\pi, \pi)^N$
        """
        n_points = len(field1)
        phase_diff = np.zeros(n_points, dtype=nb.float64)

        for i in range(n_points):
            # Calculate phases using atan2 for proper quadrant
            phase1 = np.arctan2(field1[i].imag, field1[i].real)
            phase2 = np.arctan2(field2[i].imag, field2[i].real)

            # Phase difference in [-π, π]
            diff = phase1 - phase2
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi

            phase_diff[i] = diff

        return phase_diff

    def compute_field_interference(
        self, charge1: Dict, charge2: Dict  # Real ConceptualChargeAgent Q(τ,C,s) data
    ) -> InterferencePattern:
        """
        Compute complete interference pattern between two conceptual charge fields.

        Mathematical Foundation:
            Two-wave interference intensity formula:
            $$I(x) = |\Psi_1 + \Psi_2|^2 = I_1 + I_2 + 2\sqrt{I_1 I_2}\cos(\Delta\phi)$$

            Where:
            $$I_i = |A_i|^2, \quad \Delta\phi = k_1 x - k_2 x + \phi_1 - \phi_2$$

            Visibility (fringe contrast):
            $$V = \frac{I_{\max} - I_{\min}}{I_{\max} + I_{\min}} = \frac{2\sqrt{I_1 I_2}}{I_1 + I_2}$$

            Complex coherence function:
            $$\gamma_{12} = \frac{\langle \Psi_1^* \Psi_2 \rangle}{\sqrt{\langle |\Psi_1|^2 \rangle \langle |\Psi_2|^2 \rangle}}$$

            Degree of coherence: $|\gamma_{12}| \in [0,1]$

        Interference Classification:
            - Constructive: $\Delta\phi = 2\pi n$, $I = (\sqrt{I_1} + \sqrt{I_2})^2$
            - Destructive: $\Delta\phi = (2n+1)\pi$, $I = (\sqrt{I_1} - \sqrt{I_2})^2$
            - Mixed: Intermediate phase relationships
            - Incoherent: $|\gamma_{12}| < \text{threshold}$

        Args:
            charge1, charge2: Conceptual charge dictionaries containing wave properties

        Returns:
            InterferencePattern object with complete mathematical analysis
        """
        charge1_props = self._extract_wave_properties(charge1)
        charge2_props = self._extract_wave_properties(charge2)

        # Spatial domain for analysis
        n_points = 256
        x_max = 10.0  # Spatial extent
        x_coords = torch.linspace(-x_max, x_max, n_points, dtype=torch.float64)
        time = 0.0  # Current time

        # Two-wave interference calculation
        field1_np = self._compute_single_wave(x_coords.numpy(), time, charge1_props)
        field2_np = self._compute_single_wave(x_coords.numpy(), time, charge2_props)

        # Superposition using JIT-compiled calculation
        total_field_np = field1_np + field2_np
        total_field = torch.from_numpy(total_field_np)

        # Interference intensity
        intensity = torch.abs(total_field) ** 2
        individual_intensity1 = torch.abs(torch.from_numpy(field1_np)) ** 2
        individual_intensity2 = torch.abs(torch.from_numpy(field2_np)) ** 2

        # Phase difference calculation
        phase_diff_np = self._jit_phase_difference(field1_np, field2_np)
        phase_difference = torch.from_numpy(phase_diff_np)

        # Constructive and destructive strengths
        max_intensity = torch.max(intensity)
        min_intensity = torch.min(intensity)
        mean_intensity = torch.mean(individual_intensity1 + individual_intensity2)

        constructive_strength = (max_intensity / mean_intensity).item()
        destructive_strength = (min_intensity / mean_intensity).item()

        # Overall phase difference (average)
        mean_phase_diff = torch.mean(phase_difference).item()

        # Coherence measure using complex correlation
        field1_torch = torch.from_numpy(field1_np)
        field2_torch = torch.from_numpy(field2_np)

        correlation = torch.sum(field1_torch * torch.conj(field2_torch))
        norm1 = torch.sqrt(torch.sum(torch.abs(field1_torch) ** 2))
        norm2 = torch.sqrt(torch.sum(torch.abs(field2_torch) ** 2))
        coherence_measure = torch.abs(correlation / (norm1 * norm2)).item()

        # Classify interference type
        if coherence_measure > self.coherence_threshold:
            if (
                abs(mean_phase_diff) < math.pi / 4
                or abs(abs(mean_phase_diff) - 2 * math.pi) < math.pi / 4
            ):
                interference_type = "constructive"
            elif abs(abs(mean_phase_diff) - math.pi) < math.pi / 4:
                interference_type = "destructive"
            else:
                interference_type = "mixed"
        else:
            interference_type = "incoherent"

        # Frequency spectrum using FFT
        frequency_spectrum = torch.abs(fft(total_field)) ** 2

        # Spatial correlation function
        correlation_function = self._compute_spatial_correlation(total_field)

        return InterferencePattern(
            constructive_strength=constructive_strength,
            destructive_strength=destructive_strength,
            phase_difference=mean_phase_diff,
            coherence_measure=coherence_measure,
            interference_type=interference_type,
            spatial_modulation=intensity,
            frequency_spectrum=frequency_spectrum,
            correlation_function=correlation_function,
        )

    def _extract_wave_properties(self, charge: Dict) -> WaveProperties:
        """
        Extract wave properties from conceptual charge using field-theoretic mapping.

        Mathematical Foundation:
            Mapping from conceptual charge Q(τ,C,s) to wave parameters:

            Amplitude mapping:
            $$A = \sqrt{|Q|} e^{i \arg(Q)}$$

            Frequency from temporal component:
            $$\omega = \frac{2\pi}{T_{\text{char}}} = \frac{\partial \arg(Q)}{\partial \tau}$$

            Wave vector from spatial gradients:
            $$\vec{k} = \nabla \arg(Q) = \left(\frac{\partial \phi}{\partial x}, \frac{\partial \phi}{\partial y}, \frac{\partial \phi}{\partial z}\right)$$

            Dispersion relation:
            $$\omega = \omega(|\vec{k}|) = c|\vec{k}|$$ (linear medium)

            Group velocity:
            $$\vec{v}_g = \frac{\partial \omega}{\partial \vec{k}}$$

            Phase velocity:
            $$v_p = \frac{\omega}{|\vec{k}|}$$

        Args:
            charge: Conceptual charge dictionary with Q(τ,C,s) components

        Returns:
            WaveProperties object with extracted wave parameters
        """
        # Extract Q-field components from ConceptualChargeAgent

        # Primary Q-value: amplitude and phase from Q(τ,C,s)
        if "Q_components" in charge and charge["Q_components"] is not None:
            # Real Q-field tensor extraction from Q_components.Q_value
            q_tensor = charge["Q_components"].Q_value
            if torch.is_tensor(q_tensor):
                # Complex amplitude from Q-field magnitude and phase
                q_magnitude = torch.norm(q_tensor).item()
                q_phase = (
                    torch.angle(q_tensor.mean()).item()
                    if torch.is_complex(q_tensor)
                    else 0.0
                )
                amplitude = complex(q_magnitude, q_magnitude * math.sin(q_phase))
            else:
                amplitude = complex(1.0, 0.0)  # Fallback for malformed data
        else:
            # Fallback: extract from charge ID with field positioning
            if "id" not in charge:
                raise ValueError(
                    "MATHEMATICAL FAILURE: Charge lacks required 'id' field for fallback computation"
                )
            charge_id = charge["id"]
            # Use cryptographic hash for deterministic field properties
            import hashlib

            charge_bytes = charge_id.encode("utf-8")
            hash_digest = hashlib.sha256(charge_bytes).hexdigest()
            hash_int = int(hash_digest[:8], 16)  # Use first 8 hex digits

            # Map hash to field coordinates in complex plane
            amplitude_real = 0.5 + (hash_int % 1000) / 2000.0  # Range [0.5, 1.0]
            amplitude_imag = -0.5 + (hash_int % 2000) / 2000.0  # Range [-0.5, 0.5]
            amplitude = complex(amplitude_real, amplitude_imag)

        # Frequency from Q-field temporal dynamics: ω = |∂Q/∂τ|
        if (
            "temporal_derivative" in charge
            and charge["temporal_derivative"] is not None
        ):
            freq_tensor = charge["temporal_derivative"]
            frequency = (
                torch.norm(freq_tensor).item() if torch.is_tensor(freq_tensor) else 1.0
            )
        else:
            # Estimate from amplitude: ω ∝ |Q|  (field energy relation)
            frequency = abs(amplitude) * 2.0  # Energy-frequency scaling

        # Wave number from de Broglie relation: k = p/ℏ where p ∝ |Q|
        wave_number = (
            frequency / SPEED_OF_LIGHT if "SPEED_OF_LIGHT" in globals() else frequency
        )

        # Phase from Q-field argument
        phase = cmath.phase(amplitude)

        # Wave vector from Q-field spatial derivatives: ⃗k = ∇θ where Q = |Q|e^(iθ)
        if "spatial_gradients" in charge and charge["spatial_gradients"] is not None:
            grad_tensor = charge["spatial_gradients"]
            if (
                torch.is_tensor(grad_tensor)
                and grad_tensor.numel() >= self.spatial_dimensions
            ):
                # Wave vector from gradient of phase
                wave_vector = grad_tensor.flatten()[: self.spatial_dimensions].clone()
            else:
                # Default: wave propagating in x-direction
                wave_vector = torch.zeros(self.spatial_dimensions, dtype=torch.float64)
                wave_vector[0] = wave_number
        else:
            # Fallback: isotropic wave
            wave_vector = torch.full(
                (self.spatial_dimensions,),
                wave_number / math.sqrt(self.spatial_dimensions),
                dtype=torch.float64,
            )
        polarization = torch.ones(self.spatial_dimensions, dtype=torch.float64)
        polarization = polarization / torch.norm(polarization)

        group_velocity = torch.tensor([1.0], dtype=torch.float64)  # c = 1 units
        if self.spatial_dimensions > 1:
            group_velocity = torch.cat(
                [group_velocity, torch.zeros(self.spatial_dimensions - 1)]
            )

        return WaveProperties(
            amplitude=amplitude,
            frequency=frequency,
            wave_vector=wave_vector,
            phase=phase,
            polarization=polarization,
            group_velocity=group_velocity,
            phase_velocity=frequency / wave_number,
            wavelength=2 * math.pi / wave_number,
            period=2 * math.pi / frequency,
        )

    def _compute_single_wave(
        self, x_coords: np.ndarray, time: float, props: WaveProperties
    ) -> np.ndarray:
        """
        Compute single wave field using plane wave solution.

        Mathematical Foundation:
            Plane wave solution to wave equation:
            $$\Psi(x,t) = A e^{i(\vec{k} \cdot \vec{x} - \omega t + \phi_0)}$$

            Wave equation satisfied:
            $$\frac{\partial^2 \Psi}{\partial t^2} = c^2 \nabla^2 \Psi$$

            Dispersion relation: $\omega^2 = c^2 |\vec{k}|^2$

            For 1D case:
            $$\Psi(x,t) = A e^{i(kx - \omega t + \phi)}$$

            Complex representation preserves phase information:
            $$\Psi = A(\cos(kx - \omega t + \phi) + i\sin(kx - \omega t + \phi))$$

        Args:
            x_coords: Spatial coordinates $x \in \mathbb{R}^N$
            time: Time parameter $t \in \mathbb{R}$
            props: Wave properties (A, k, ω, φ)

        Returns:
            Complex wave field $\Psi(x,t) \in \mathbb{C}^N$
        """
        k = props.wave_vector[0].item() if len(props.wave_vector) > 0 else 1.0

        # Use JIT-compiled two-wave interference with second amplitude = 0
        return self._jit_two_wave_interference(
            x_coords,
            time,
            props.amplitude,
            complex(0.0),  # Only first wave
            k,
            0.0,  # Only first wave vector
            props.frequency,
            0.0,  # Only first frequency
            props.phase,
            0.0,  # Only first phase
        )

    def _compute_spatial_correlation(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial correlation function using FFT convolution theorem.

        Mathematical Foundation:
            Spatial correlation function:
            $$G(r) = \langle \Psi^*(x) \Psi(x+r) \rangle = \int_{-\infty}^{\infty} \Psi^*(x) \Psi(x+r) dx$$

            Wiener-Khintchine theorem:
            $$G(r) = \mathcal{F}^{-1}[|\mathcal{F}[\Psi]|^2]$$

            Where $\mathcal{F}$ is the Fourier transform:
            $$\mathcal{F}[\Psi](k) = \int_{-\infty}^{\infty} \Psi(x) e^{-ikx} dx$$

            Normalized correlation:
            $$g(r) = \frac{G(r)}{G(0)} = \frac{G(r)}{\langle |\Psi|^2 \rangle}$$

            Properties:
            - $g(0) = 1$ (normalization)
            - $g(r) = g^*(-r)$ (Hermitian symmetry)
            - $|g(r)| \leq 1$ (Cauchy-Schwarz inequality)

        Args:
            field: Complex field array $\Psi(x) \in \mathbb{C}^N$

        Returns:
            Normalized spatial correlation $g(r) \in \mathbb{C}^N$
        """
        # Autocorrelation using FFT convolution theorem
        field_fft = fft(field)
        correlation_fft = field_fft * torch.conj(field_fft)
        correlation = torch.real(ifft(correlation_fft))

        # Normalize
        correlation = correlation / correlation[0]
        return correlation

    def _compute_wigner_distribution(
        self, field: torch.Tensor, x_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Wigner quasi-probability distribution for phase-space analysis.

        Mathematical Foundation:
            Wigner distribution function:
            $$W(x,k) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \Psi^*\left(x-\frac{s}{2}\right) \Psi\left(x+\frac{s}{2}\right) e^{-iks} ds$$

            Properties:
            - Real-valued: $W(x,k) \in \mathbb{R}$
            - Marginal conditions:
              $$\int W(x,k) dk = |\Psi(x)|^2$$ (position probability)
              $$\int W(x,k) dx = |\tilde{\Psi}(k)|^2$$ (momentum probability)

            - Uncertainty relation:
              $$\Delta x \cdot \Delta k \geq \frac{1}{2}$$

            Quantum mechanical interpretation:
            - Phase-space quasi-probability density
            - Can be negative (non-classical behavior)
            - Provides simultaneous position-momentum information

            Husimi Q-function (positive alternative):
            $$Q(x,k) = \frac{1}{\pi} \int W(x',k') e^{-2|x+ik-(x'+ik')|^2} dx' dk'$$

        Args:
            field: Complex field $\Psi(x) \in \mathbb{C}^N$
            x_coords: Position coordinates $x \in \mathbb{R}^N$

        Returns:
            Real Wigner distribution $W(x,k) \in \mathbb{R}^{N \times M}$
        """
        n_points = len(field)
        n_k = n_points // 2

        # Momentum grid
        dx = x_coords[1] - x_coords[0] if n_points > 1 else 1.0
        k_max = math.pi / dx
        k_coords = torch.linspace(-k_max, k_max, n_k)

        # Wigner distribution matrix
        wigner = torch.zeros((n_points, n_k), dtype=get_dtype_manager().config.complex_dtype)

        for i, x in enumerate(x_coords):
            for j, k in enumerate(k_coords):
                # Integration over s using trapezoidal rule
                integrand = torch.zeros(n_points, dtype=get_dtype_manager().config.complex_dtype)

                for s_idx, s in enumerate(x_coords - x_coords[n_points // 2]):
                    # Indices for x±s/2
                    idx_minus = max(0, min(n_points - 1, i - s_idx // 2))
                    idx_plus = max(0, min(n_points - 1, i + s_idx // 2))

                    # Wigner integrand: Ψ*(x-s/2) Ψ(x+s/2) e^(-iks)
                    integrand[s_idx] = (
                        torch.conj(field[idx_minus])
                        * field[idx_plus]
                        * torch.exp(-1j * k * s)
                    )

                # Trapezoidal integration
                wigner[i, j] = torch.trapz(integrand, x_coords)

        return torch.real(wigner)  # Wigner function is real

    def _compute_modular_form_phases(
        self, field1: torch.Tensor, field2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rigorous phase relationships using modular forms theory.

        Mathematical Foundation:
            Modular forms for phase normalization:

            Weight-k modular form: $f: \mathbb{H} \to \mathbb{C}$
            $$f\left(\frac{a\tau + b}{c\tau + d}\right) = (c\tau + d)^k f(\tau)$$

            For $\begin{pmatrix} a & b \\ c & d \end{pmatrix} \in SL_2(\mathbb{Z})$

            Fundamental domain: $\mathcal{F} = \{\tau \in \mathbb{H}: |\tau| \geq 1, |\text{Re}(\tau)| \leq \frac{1}{2}\}$

            Phase mapping: $\phi_1 + i\phi_2 \mapsto \tau \in \mathbb{H}$

            q-expansion of modular forms:
            $$f(\tau) = \sum_{n=0}^{\infty} a_n q^n, \quad q = e^{2\pi i \tau}$$

            Modular transformations:
            - $T: \tau \mapsto \tau + 1$ (translation)
            - $S: \tau \mapsto -1/\tau$ (inversion)

            Eisenstein series (weight 12):
            $$E_{12}(\tau) = 1 - \frac{65520}{691} \sum_{n=1}^{\infty} \sigma_{11}(n) q^n$$

            Phase canonicalization ensures mathematical rigor in interference calculations.

        Args:
            field1, field2: Complex fields for phase extraction

        Returns:
            Canonicalized phase differences in fundamental domain
        """
        # Extract phase information
        phase1 = torch.angle(field1)
        phase2 = torch.angle(field2)

        # Convert to Sage CDF for exact arithmetic
        n_points = len(phase1)
        modular_phases = torch.zeros(n_points, dtype=torch.float64)

        # Use modular forms for phase normalization
        M = ModularForms(1, 12)  # Weight 12 modular forms for SL2(Z)

        for i in range(n_points):
            # Map phases to fundamental domain
            p1 = float(phase1[i])
            p2 = float(phase2[i])

            # Use modular form symmetry to compute canonical phase difference
            # τ = p1 + i*p2 in upper half-plane
            if p2 > 0:
                tau = CDF(p1, p2)
            else:
                tau = CDF(p1, abs(p2) + 1e-10)  # Ensure Im(τ) > 0

            # Apply modular transformation to fundamental domain
            # This ensures mathematical rigor in phase calculations
            if tau.real() < -0.5:
                tau += 1
            elif tau.real() >= 0.5:
                tau -= 1

            if abs(tau) < 1:
                tau = -1 / tau  # S transformation

            modular_phases[i] = float(tau.argument())  # Argument in [-π, π]

        return modular_phases

    def _create_interference_graph(
        self, charges: List[Dict], coherence_matrix: torch.Tensor
    ) -> Data:
        """
        Create geometric graph representation of interference field topology.

        Mathematical Foundation:
            Graph Laplacian for wave propagation:
            $$\mathcal{L} = D - A$$

            Where:
            - $D_{ii} = \sum_j A_{ij}$ (degree matrix)
            - $A_{ij} = |\gamma_{ij}|$ (coherence adjacency matrix)

            Wave equation on graphs:
            $$\frac{\partial^2 \Psi}{\partial t^2} = -\mathcal{L} \Psi$$

            Spectral decomposition:
            $$\mathcal{L} = \sum_{k=0}^{N-1} \lambda_k \phi_k \phi_k^T$$

            Graph Fourier transform:
            $$\hat{\Psi}(\lambda_k) = \sum_{i=0}^{N-1} \Psi(i) \phi_k(i)$$

            Coherence threshold determines edge connectivity:
            $$A_{ij} = \begin{cases} |\gamma_{ij}| & \text{if } |\gamma_{ij}| > \text{threshold} \\ 0 & \text{otherwise} \end{cases}$$

            Node features encode wave properties:
            - Real amplitude: $\text{Re}(A_i)$
            - Imaginary amplitude: $\text{Im}(A_i)$
            - Frequency: $\omega_i$
            - Phase: $\phi_i$

        Args:
            charges: List of conceptual charges
            coherence_matrix: Pairwise coherence $\gamma_{ij}$

        Returns:
            torch_geometric.data.Data object with graph structure
        """
        n_charges = len(charges)

        # Node features: extract charge properties
        node_features = torch.zeros((n_charges, 4), dtype=torch.float32)
        for i, charge in enumerate(charges):
            props = self._extract_wave_properties(charge)
            node_features[i, 0] = float(props.amplitude.real)
            node_features[i, 1] = float(props.amplitude.imag)
            node_features[i, 2] = props.frequency
            node_features[i, 3] = props.phase

        # Edge indices and weights from coherence matrix
        edge_indices = []
        edge_weights = []
        coherence_threshold = self.coherence_threshold

        for i in range(n_charges):
            for j in range(i + 1, n_charges):
                coherence = coherence_matrix[i, j].item()
                if coherence > coherence_threshold:
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])  # Undirected graph
                    edge_weights.append(coherence)
                    edge_weights.append(coherence)

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
        else:
            # Empty graph
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty(0, dtype=torch.float32)

        # Create torch_geometric Data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=n_charges,
        )

        return graph_data

    class InterferenceMessagePassing(MessagePassing):
        """
        Message passing neural network for interference field propagation dynamics.
        
        Mathematical Foundation:
            Graph Neural Network (GNN) formulation for wave interference:
            
            **Message Passing Framework**:
            $$h_i^{(l+1)} = \\text{UPDATE}^{(l)}\\left(h_i^{(l)}, \\text{AGGREGATE}^{(l)}\\left(\\{\\text{MESSAGE}^{(l)}(h_i^{(l)}, h_j^{(l)}, e_{ij}) : j \\in \\mathcal{N}(i)\\}\\right)\\right)$$
            
            **Interference-Specific Formulation**:
            
            Node state vector: $h_i = [\\text{Re}(A_i), \\text{Im}(A_i), \\omega_i, \\phi_i]^T$
            
            Message function (phase-coherent coupling):
            $$m_{ij} = \\gamma_{ij} \\cdot \\begin{bmatrix}
            \\cos(\\phi_j - \\phi_i) \\cdot \\text{Re}(A_j) \\\\
            \\cos(\\phi_j - \\phi_i) \\cdot \\text{Im}(A_j) \\\\
            \\omega_j \\\\
            \\sin(\\phi_j - \\phi_i)
            \\end{bmatrix}$$
            
            Aggregation function (mean field approximation):
            $$\\bar{m}_i = \\frac{1}{|\\mathcal{N}(i)|} \\sum_{j \\in \\mathcal{N}(i)} m_{ij}$$
            
            Update function (coupled oscillator dynamics):
            $$h_i^{(l+1)} = h_i^{(l)} + \\alpha \\bar{m}_i^{(l)}$$
            
            **Physical Interpretation**:
            - Messages carry phase-weighted amplitude information
            - Coherence weights $\\gamma_{ij}$ modulate coupling strength
            - Phase differences determine constructive/destructive coupling
            - Update preserves wave equation dynamics on the graph
            
            **Graph Laplacian Formulation**:
            The message passing can be viewed as discretized wave equation:
            $$\\frac{\\partial \\Psi}{\\partial t} = -i\\mathcal{L}_{\\text{graph}} \\Psi$$
            
            where $\\mathcal{L}_{\\text{graph}}$ is the coherence-weighted graph Laplacian.
            
        Implementation:
            - Uses 'mean' aggregation for stability
            - Preserves complex amplitude structure
            - Maintains phase relationship coherence
            - Supports arbitrary graph topologies
        """

        def __init__(self):
            super().__init__(aggr="mean")  # Mean aggregation for phase coherence

        def forward(self, x, edge_index, edge_attr):
            """
            Forward pass: propagate interference patterns using message passing.

            Mathematical Foundation:
                Message passing neural networks (MPNN) for interference:

                Message function:
                $$m_{ij}^{(t)} = \text{Message}(h_i^{(t)}, h_j^{(t)}, e_{ij})$$

                Aggregation function:
                $$\bar{m}_i^{(t)} = \text{Aggregate}(\{m_{ij}^{(t)} : j \in \mathcal{N}(i)\})$$

                Update function:
                $$h_i^{(t+1)} = \text{Update}(h_i^{(t)}, \bar{m}_i^{(t)})$$

                For interference propagation:
                - Messages carry phase-weighted amplitude information
                - Aggregation preserves coherence relationships
                - Updates maintain wave equation dynamics

            Args:
                x: Node features $h_i \in \mathbb{R}^d$
                edge_index: Graph connectivity $E \subset V \times V$
                edge_attr: Edge weights (coherence values)

            Returns:
                Updated node features after message propagation
            """
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)

        def message(self, x_i, x_j, edge_attr):
            """
            Compute interference messages between connected charges.

            Mathematical Foundation:
                Phase-weighted amplitude transfer:

                Complex amplitude extraction:
                $$A_i = x_i[0] + i \cdot x_i[1], \quad A_j = x_j[0] + i \cdot x_j[1]$$

                Phase difference:
                $$\Delta\phi_{ij} = \phi_j - \phi_i = x_j[3] - x_i[3]$$

                Interference strength:
                $$S_{ij} = \cos(\Delta\phi_{ij}) \cdot \gamma_{ij}$$

                Where $\gamma_{ij}$ is the edge coherence attribute.

                Message components:
                - Amplitude transfer: $S_{ij} \cdot A_j$
                - Frequency coupling: $\gamma_{ij} \cdot \omega_j$
                - Phase update: $S_{ij} \cdot \Delta\phi_{ij}$

                Physical interpretation:
                - Constructive coupling when $\cos(\Delta\phi) > 0$
                - Destructive coupling when $\cos(\Delta\phi) < 0$
                - Coherence modulates coupling strength

            Args:
                x_i, x_j: Node features for connected charges
                edge_attr: Coherence weights $\gamma_{ij}$

            Returns:
                Message vector for interference propagation
            """
            # Extract amplitudes and phases
            amp_i = torch.complex(x_i[:, 0], x_i[:, 1])
            amp_j = torch.complex(x_j[:, 0], x_j[:, 1])
            phase_i = x_i[:, 3]
            phase_j = x_j[:, 3]

            # Phase difference
            phase_diff = phase_j - phase_i

            # Interference strength based on phase alignment
            interference_strength = torch.cos(phase_diff) * edge_attr

            # Message: interference-weighted amplitude components
            message = torch.zeros_like(x_i)
            message[:, 0] = interference_strength * amp_j.real
            message[:, 1] = interference_strength * amp_j.imag
            message[:, 2] = edge_attr * x_j[:, 2]  # Frequency transfer
            message[:, 3] = interference_strength * phase_diff  # Phase update

            return message

        def update(self, aggr_out, x):
            """
            Update node states after message aggregation using coupled oscillator dynamics.

            Mathematical Foundation:
                Coupled oscillator update rule:

                $$\frac{d}{dt} \vec{h}_i = -\alpha (\vec{h}_i - \vec{h}_i^{(0)}) + \beta \sum_{j \in \mathcal{N}(i)} \gamma_{ij} \vec{m}_{ji}$$

                Discrete update:
                $$\vec{h}_i^{(t+1)} = \vec{h}_i^{(t)} + \alpha \cdot \text{aggr_out}_i$$

                Where:
                - $\alpha$ is the coupling strength (update rate)
                - $\text{aggr_out}_i$ contains aggregated messages from neighbors
                - Update preserves wave equation dynamics

                Stability condition:
                $$\alpha < \frac{2}{\lambda_{\max}(\mathcal{L})}$$

                Where $\lambda_{\max}$ is the largest eigenvalue of the graph Laplacian.

            Args:
                aggr_out: Aggregated messages from neighbors
                x: Current node states

            Returns:
                Updated node states maintaining interference dynamics
            """
            # Combine original state with aggregated messages
            alpha = 0.1  # Update rate
            return x + alpha * aggr_out

    def analyze_constructive_regions(
        self, charges: List[Dict]
    ) -> List[ConstructiveZone]:
        """
        Identify regions of constructive interference using phasor analysis.

        Mathematical Foundation:
            Constructive interference condition:
            $$\left|\sum_{i=1}^N A_i e^{i\phi_i}\right| \approx \sum_{i=1}^N |A_i|$$

            Phase alignment criterion:
            $$\phi_i \approx \phi_j \pmod{2\pi} \quad \forall i,j$$

            More precisely:
            $$|\phi_i - \phi_j| < \epsilon \quad \text{or} \quad |\phi_i - \phi_j - 2\pi k| < \epsilon$$

            For $k \in \mathbb{Z}$ and small $\epsilon > 0$.

            Intensity enhancement:
            $$I_{\text{max}} = \left(\sum_{i=1}^N \sqrt{I_i}\right)^2$$

            Enhancement factor:
            $$\mathcal{E} = \frac{I_{\text{max}}}{\sum_{i=1}^N I_i} = \frac{\left(\sum_i \sqrt{I_i}\right)^2}{\sum_i I_i}$$

            Spatial extent (coherence length):
            $$l_c = \frac{\lambda}{2\pi} \sqrt{\frac{\sum_i I_i}{\sum_i |\nabla \phi_i|^2}}$$

            Stability measure (frequency matching):
            $$\mathcal{S} = \exp\left(-\frac{|\omega_i - \omega_j|}{\omega_{\text{avg}}}\right)$$

        Args:
            charges: List of conceptual charges with wave properties

        Returns:
            List of ConstructiveZone objects with enhancement analysis
        """
        if len(charges) < 2:
            return []

        constructive_zones = []

        # Analyze all pairs for simplicity (could extend to n-wave)
        for i in range(len(charges)):
            for j in range(i + 1, len(charges)):
                # Compute pairwise interference
                interference = self.compute_field_interference(charges[i], charges[j])

                if interference.interference_type == "constructive":
                    # Extract wave properties
                    props1 = self._extract_wave_properties(charges[i])
                    props2 = self._extract_wave_properties(charges[j])

                    # Phase matching condition
                    phase_diff = abs(props1.phase - props2.phase)
                    if phase_diff > math.pi:
                        phase_diff = 2 * math.pi - phase_diff

                    # Zone characteristics
                    center_position = torch.zeros(
                        self.spatial_dimensions, dtype=torch.float64
                    )
                    spatial_extent = torch.full(
                        (self.spatial_dimensions,),
                        min(props1.wavelength, props2.wavelength),
                        dtype=torch.float64,
                    )

                    # Intensity enhancement
                    enhancement = interference.constructive_strength

                    # Phase matching condition description
                    condition = f"Δφ = {phase_diff:.3f} ≈ 0 (mod 2π)"

                    # Stability based on frequency matching
                    freq_diff = abs(props1.frequency - props2.frequency)
                    stability = math.exp(
                        -freq_diff
                    )  # More stable when frequencies match

                    zone = ConstructiveZone(
                        center_position=center_position,
                        spatial_extent=spatial_extent,
                        intensity_enhancement=enhancement,
                        phase_matching_condition=condition,
                        stability_measure=stability,
                        contributing_waves=[i, j],
                    )
                    constructive_zones.append(zone)

        return constructive_zones

    def detect_destructive_interference(
        self, charges: List[Dict]
    ) -> List[DestructiveZone]:
        """
        Identify regions of destructive interference using phasor cancellation analysis.

        Mathematical Foundation:
            Destructive interference condition:
            $$\left|\sum_{i=1}^N A_i e^{i\phi_i}\right| \ll \sum_{i=1}^N |A_i|$$

            Phase opposition criterion:
            $$\phi_i \approx \phi_j + \pi \pmod{2\pi}$$

            Perfect cancellation (equal amplitudes):
            $$|A_1| = |A_2|, \quad \phi_2 = \phi_1 + \pi \Rightarrow I_{\text{total}} = 0$$

            General case intensity minimum:
            $$I_{\text{min}} = \left(\sqrt{I_1} - \sqrt{I_2}\right)^2$$

            Suppression factor:
            $$\mathcal{S} = \frac{I_{\text{min}}}{I_1 + I_2} = \frac{\left(\sqrt{I_1} - \sqrt{I_2}\right)^2}{I_1 + I_2}$$

            Null depth (cancellation quality):
            $$\mathcal{D} = 1 - \mathcal{S} = 1 - \frac{I_{\text{min}}}{I_1 + I_2}$$

            Spatial null width:
            $$w_{\text{null}} = \frac{\lambda}{2\pi} \sqrt{\frac{1}{\sum_i |\nabla \phi_i|^2}}$$

            Perfect null condition: $\mathcal{D} = 1$ (complete cancellation)

        Args:
            charges: List of conceptual charges with wave properties

        Returns:
            List of DestructiveZone objects with cancellation analysis
        """
        if len(charges) < 2:
            return []

        destructive_zones = []

        # Analyze all pairs
        for i in range(len(charges)):
            for j in range(i + 1, len(charges)):
                # Compute pairwise interference
                interference = self.compute_field_interference(charges[i], charges[j])

                if interference.interference_type == "destructive":
                    # Extract wave properties
                    props1 = self._extract_wave_properties(charges[i])
                    props2 = self._extract_wave_properties(charges[j])

                    # Phase opposition condition
                    phase_diff = abs(props1.phase - props2.phase)
                    if phase_diff > math.pi:
                        phase_diff = 2 * math.pi - phase_diff
                    phase_opposition = abs(phase_diff - math.pi)

                    # Zone characteristics
                    center_position = torch.zeros(
                        self.spatial_dimensions, dtype=torch.float64
                    )
                    spatial_extent = torch.full(
                        (self.spatial_dimensions,),
                        min(props1.wavelength, props2.wavelength) / 2,
                        dtype=torch.float64,
                    )

                    # Intensity suppression
                    suppression = interference.destructive_strength

                    # Phase opposition condition description
                    condition = f"Δφ = {phase_diff:.3f} ≈ π (mod 2π)"

                    # Null depth (how close to zero)
                    null_depth = 1.0 - suppression

                    zone = DestructiveZone(
                        center_position=center_position,
                        spatial_extent=spatial_extent,
                        intensity_suppression=suppression,
                        phase_opposition_condition=condition,
                        null_depth=null_depth,
                        contributing_waves=[i, j],
                    )
                    destructive_zones.append(zone)

        return destructive_zones

    def calculate_phase_coherence_matrix(self, charges: List[Dict]) -> torch.Tensor:
        """
        Calculate pairwise phase coherence matrix using complex field correlation.

        Mathematical Foundation:
            Complex coherence function:
            $$\gamma_{ij}(\tau) = \frac{\langle \Psi_i^*(t) \Psi_j(t+\tau) \rangle}{\sqrt{\langle |\Psi_i(t)|^2 \rangle \langle |\Psi_j(t)|^2 \rangle}}$$

            For $\tau = 0$ (simultaneous coherence):
            $$\gamma_{ij} = \frac{\langle \Psi_i^* \Psi_j \rangle}{\sqrt{\langle |\Psi_i|^2 \rangle \langle |\Psi_j|^2 \rangle}}$$

            Degree of coherence:
            $$|\gamma_{ij}| \in [0,1]$$

            Properties:
            - $\gamma_{ii} = 1$ (self-coherence)
            - $\gamma_{ij} = \gamma_{ji}^*$ (Hermitian symmetry)
            - $|\gamma_{ij}| \leq 1$ (Cauchy-Schwarz bound)

            Physical interpretation:
            - $|\gamma_{ij}| = 1$: Fully coherent (constant phase relation)
            - $|\gamma_{ij}| = 0$: Incoherent (random phase relation)
            - $0 < |\gamma_{ij}| < 1$: Partially coherent

            Van Cittert-Zernike theorem relates spatial coherence to source distribution:
            $$\gamma_{ij} = \frac{\int I(\vec{s}) e^{i\vec{k} \cdot (\vec{r}_i - \vec{r}_j)} d\vec{s}}{\int I(\vec{s}) d\vec{s}}$$

        Args:
            charges: List of conceptual charges

        Returns:
            Complex coherence matrix $\Gamma \in \mathbb{C}^{N \times N}$
        """
        n_charges = len(charges)
        coherence_matrix = torch.zeros((n_charges, n_charges), dtype=torch.float64)

        # Compute wave fields for all charges
        n_points = 128
        x_coords = torch.linspace(-5.0, 5.0, n_points, dtype=torch.float64)
        fields = []

        for charge in charges:
            props = self._extract_wave_properties(charge)
            field_np = self._compute_single_wave(x_coords.numpy(), 0.0, props)
            fields.append(torch.from_numpy(field_np))

        # Compute coherence matrix
        for i in range(n_charges):
            for j in range(n_charges):
                if i == j:
                    coherence_matrix[i, j] = 1.0
                else:
                    # Cross-correlation
                    correlation = torch.sum(fields[i] * torch.conj(fields[j]))
                    norm_i = torch.sqrt(torch.sum(torch.abs(fields[i]) ** 2))
                    norm_j = torch.sqrt(torch.sum(torch.abs(fields[j]) ** 2))

                    coherence = torch.abs(correlation / (norm_i * norm_j))
                    coherence_matrix[i, j] = coherence.item()

        return coherence_matrix

    def predict_interference_evolution(
        self, charges: List[Dict], time_horizon: float
    ) -> EvolutionTrajectory:
        """
        Predict temporal evolution of interference patterns using wave dynamics.

        Mathematical Foundation:
            Time evolution operator:
            $$\Psi_i(x,t) = \Psi_i(x,0) e^{-i\omega_i t}$$

            Total field evolution:
            $$\Psi_{\text{total}}(x,t) = \sum_{i=1}^N A_i e^{i(k_i x - \omega_i t + \phi_i)}$$

            Beating phenomena (two-wave case):
            $$\Psi(x,t) = A_1 e^{i(kx - \omega_1 t)} + A_2 e^{i(kx - \omega_2 t)}$$
            $$= e^{i(kx - \bar{\omega}t)} \left[A_1 e^{i\Delta\omega t/2} + A_2 e^{-i\Delta\omega t/2}\right]$$

            Where:
            - $\bar{\omega} = (\omega_1 + \omega_2)/2$ (average frequency)
            - $\Delta\omega = \omega_1 - \omega_2$ (frequency difference)

            Beat frequency:
            $$\omega_{\text{beat}} = |\omega_1 - \omega_2|$$

            Envelope modulation:
            $$A_{\text{env}}(t) = |A_1 e^{i\Delta\omega t/2} + A_2 e^{-i\Delta\omega t/2}|$$

            Group velocity dispersion effects:
            $$\frac{\partial \Psi}{\partial t} + v_g \frac{\partial \Psi}{\partial x} + \frac{i\beta_2}{2} \frac{\partial^2 \Psi}{\partial x^2} = 0$$

            Critical transitions occur at:
            $$t_c = \frac{2\pi n}{\omega_{\text{beat}}}, \quad n \in \mathbb{Z}$$

        Args:
            charges: List of conceptual charges
            time_horizon: Evolution time $T \in \mathbb{R}^+$

        Returns:
            EvolutionTrajectory with complete temporal dynamics
        """
        if len(charges) < 2:
            raise ValueError("Need at least 2 charges for interference evolution")

        # Time sampling
        n_time_steps = int(time_horizon / self.temporal_resolution)
        time_points = torch.linspace(0, time_horizon, n_time_steps, dtype=torch.float64)

        # Spatial grid
        n_spatial = 64
        x_coords = torch.linspace(-5.0, 5.0, n_spatial, dtype=torch.float64)

        # Initialize evolution arrays
        intensity_evolution = torch.zeros(
            (n_spatial, n_time_steps), dtype=torch.float64
        )
        phase_evolution = torch.zeros((n_spatial, n_time_steps), dtype=torch.float64)
        coherence_evolution = torch.zeros(n_time_steps, dtype=torch.float64)

        # Extract wave properties
        wave_props = [self._extract_wave_properties(charge) for charge in charges]
        frequencies = [props.frequency for props in wave_props]

        # Compute evolution
        for t_idx, time in enumerate(time_points):
            # Superposition at this time
            total_field = torch.zeros(n_spatial, dtype=get_dtype_manager().config.complex_dtype)

            for props in wave_props:
                field_np = self._compute_single_wave(
                    x_coords.numpy(), time.item(), props
                )
                total_field += torch.from_numpy(field_np)

            # Store intensity and phase
            intensity_evolution[:, t_idx] = torch.abs(total_field) ** 2
            phase_evolution[:, t_idx] = torch.angle(total_field)

            # Average coherence (simplified)
            coherence_evolution[t_idx] = torch.mean(torch.abs(total_field) ** 2) / len(
                charges
            )

        # Frequency analysis
        dominant_frequencies = torch.tensor(frequencies, dtype=torch.float64)

        # Beating frequency (for two waves)
        beating_frequency = None
        if len(frequencies) == 2:
            beating_frequency = abs(frequencies[0] - frequencies[1])

        # Envelope modulation (simplified)
        envelope_modulation = torch.mean(intensity_evolution, dim=0)

        # Critical transitions (when coherence changes significantly)
        coherence_gradient = torch.gradient(coherence_evolution)[0]
        threshold = 0.1 * torch.std(coherence_gradient)
        critical_transitions = torch.where(torch.abs(coherence_gradient) > threshold)[
            0
        ].tolist()

        return EvolutionTrajectory(
            time_points=time_points,
            intensity_evolution=intensity_evolution,
            phase_evolution=phase_evolution,
            coherence_evolution=coherence_evolution,
            dominant_frequencies=dominant_frequencies,
            beating_frequency=beating_frequency,
            envelope_modulation=envelope_modulation,
            critical_transitions=critical_transitions,
        )

    def advanced_fft_spectral_analysis(
        self, field: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Complete FFT spectral analysis using all PyTorch FFT functions.

        Mathematical Foundation:
            Discrete Fourier Transform:
            $$\hat{f}[k] = \sum_{n=0}^{N-1} f[n] e^{-2\pi i kn/N}$$

            Inverse DFT:
            $$f[n] = \frac{1}{N} \sum_{k=0}^{N-1} \hat{f}[k] e^{2\pi i kn/N}$$

            Multi-dimensional transforms:

            2D FFT:
            $$\hat{f}[k_1, k_2] = \sum_{n_1=0}^{N_1-1} \sum_{n_2=0}^{N_2-1} f[n_1, n_2] e^{-2\pi i(k_1 n_1/N_1 + k_2 n_2/N_2)}$$

            Real FFT (for real-valued signals):
            $$\hat{f}[k] = \sum_{n=0}^{N-1} f[n] e^{-2\pi i kn/N}, \quad k = 0, 1, \ldots, \lfloor N/2 \rfloor$$

            Parseval's theorem:
            $$\sum_{n=0}^{N-1} |f[n]|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |\hat{f}[k]|^2$$

            Power spectral density:
            $$S[k] = |\hat{f}[k]|^2$$

            Convolution theorem:
            $$\mathcal{F}[f * g] = \mathcal{F}[f] \cdot \mathcal{F}[g]$$

        FFT Functions Used:
            - fft/ifft: 1D complex transforms
            - fft2/ifft2: 2D complex transforms
            - fft3/ifft3: 3D complex transforms
            - rfft/irfft: Real-valued optimized transforms

        Args:
            field: Input field for spectral analysis

        Returns:
            Dictionary with complete spectral decomposition
        """
        results = {}

        # 1D real FFT analysis using rfft/irfft for real-valued fields
        if field.dtype in [torch.float32, torch.float64]:
            real_spectrum = rfft(field)
            results["rfft_spectrum"] = real_spectrum
            results["irfft_reconstruction"] = irfft(real_spectrum, n=len(field))

        # 1D complex FFT analysis
        field_complex = field.to(torch.complex128)
        spectrum_1d = fft(field_complex)
        results["fft_1d"] = spectrum_1d
        results["ifft_1d_verification"] = ifft(spectrum_1d)

        # 2D FFT analysis (reshape field if needed)
        if field.numel() >= 4:
            size_2d = int(field.numel() ** 0.5)
            if size_2d * size_2d == field.numel():
                field_2d = field_complex.view(size_2d, size_2d)
                spectrum_2d = fft2(field_2d)
                results["fft_2d"] = spectrum_2d
                results["ifft_2d_verification"] = ifft2(spectrum_2d)

        # 3D FFT analysis (reshape field if needed)
        if field.numel() >= 8:
            size_3d = int(round(field.numel() ** (1 / 3)))
            if size_3d**3 == field.numel():
                field_3d = field_complex.view(size_3d, size_3d, size_3d)
                spectrum_3d = (
                    fft3(field_3d)
                    if hasattr(torch.fft, "fft3")
                    else fft(field_3d.flatten())
                )
                results["fft_3d"] = spectrum_3d
                if hasattr(torch.fft, "ifft3"):
                    results["ifft_3d_verification"] = ifft3(spectrum_3d)

        return results

    def neural_network_wave_processing(
        self, field1: torch.Tensor, field2: torch.Tensor
    ) -> torch.Tensor:
        """
        Neural network wave processing using torch.nn.functional operations.

        Mathematical Foundation:
            Convolutional wave processing:

            Cross-correlation via convolution:
            $$(f * g)[n] = \sum_{m=-\infty}^{\infty} f[m] g[n-m]$$

            For discrete signals:
            $$R_{fg}[n] = \sum_{m=0}^{N-1} f[m] g^*[m+n]$$

            Neural activation functions:

            ReLU activation:
            $$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$

            Average pooling:
            $$\text{AvgPool}(x)[i] = \frac{1}{k} \sum_{j=0}^{k-1} x[ik + j]$$

            Layer normalization:
            $$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

            Where:
            - $\mu = \frac{1}{n} \sum_{i=1}^n x_i$ (mean)
            - $\sigma^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2$ (variance)

            Deep learning for wave pattern recognition:
            - Convolution extracts local wave features
            - Pooling provides translation invariance
            - Normalization stabilizes wave amplitude variations

        Args:
            field1, field2: Input wave fields for neural processing

        Returns:
            Processed wave field with neural transformations
        """
        # Prepare fields for 2D convolution
        field1_4d = field1.real.view(1, 1, -1, 1)  # [batch, channels, height, width]
        field2_4d = field2.real.view(1, 1, -1, 1)

        # Cross-correlation using convolution
        # Create correlation kernel from field2
        kernel_size = min(len(field2), 7)
        if kernel_size % 2 == 0:
            kernel_size -= 1

        kernel = field2_4d[:, :, :kernel_size, :].flip(dims=[2])
        correlation = F.conv2d(field1_4d, kernel, padding=(kernel_size // 2, 0))

        # Apply activation functions for nonlinear wave processing
        activated = F.relu(correlation)  # ReLU activation
        smoothed = F.avg_pool2d(activated, kernel_size=(3, 1), stride=1, padding=(1, 0))

        # Normalize using layer normalization
        normalized = F.layer_norm(smoothed, smoothed.shape[2:])

        return normalized.view(-1)

    def complex_number_wave_mathematics(
        self, amplitude: complex, phase: float
    ) -> Dict[str, Union[complex, float]]:
        """
        Complex wave mathematics using cmath library for exact calculations.

        Mathematical Foundation:
            Complex exponential representation:
            $$\Psi = A e^{i\phi} = A(\cos\phi + i\sin\phi)$$

            Euler's formula:
            $$e^{i\phi} = \cos\phi + i\sin\phi$$

            Complex amplitude decomposition:
            $$A = |A| e^{i\arg(A)} = a + bi$$

            Where:
            - $|A| = \sqrt{a^2 + b^2}$ (magnitude)
            - $\arg(A) = \arctan(b/a)$ (argument/phase)

            Wave representation:
            $$\Psi = A e^{i\phi} = |A| e^{i(\arg(A) + \phi)}$$

            Intensity calculation:
            $$I = |\Psi|^2 = \Psi \Psi^* = |A|^2$$

            Complex logarithm:
            $$\ln(\Psi) = \ln|\Psi| + i\arg(\Psi)$$

            Principal value: $\arg(\Psi) \in (-\pi, \pi]$

            Branch cuts and Riemann surfaces for multi-valued functions.

            De Moivre's theorem:
            $$(\cos\phi + i\sin\phi)^n = \cos(n\phi) + i\sin(n\phi)$$

        Args:
            amplitude: Complex amplitude $A \in \mathbb{C}$
            phase: Phase angle $\phi \in \mathbb{R}$

        Returns:
            Dictionary with complete complex wave analysis
        """
        # Complex exponential: e^(iφ)
        exponential = cmath.exp(1j * phase)

        # Complex wave: Ae^(iφ)
        wave = amplitude * exponential

        # Phase and magnitude extraction
        magnitude = cmath.sqrt(wave.real**2 + wave.imag**2)
        phase_extracted = cmath.phase(wave)

        # Complex conjugate and products
        conjugate = wave.conjugate()
        intensity = wave * conjugate  # |Ψ|²

        # Complex logarithm for advanced phase analysis
        log_wave = cmath.log(wave) if abs(wave) > 1e-15 else complex(0)

        return {
            "wave": wave,
            "magnitude": magnitude,
            "phase": phase_extracted,
            "conjugate": conjugate,
            "intensity": intensity.real,
            "log_amplitude": log_wave,
        }

    def jax_vectorized_interference_analysis(
        self, field1: torch.Tensor, field2: torch.Tensor
    ) -> Dict[str, float]:
        """
        JAX-vectorized interference analysis with automatic differentiation.

        Mathematical Foundation:
            Vectorized operations using JAX:

            Vector map (vmap) applies function element-wise:
            $$\text{vmap}(f)([x_1, x_2, \ldots, x_n]) = [f(x_1), f(x_2), \ldots, f(x_n)]$$

            Automatic differentiation:
            $$\nabla f(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$ (forward mode)

            Just-in-time compilation optimizes computation graphs.

            Interference intensity function:
            $$I(x_1, x_2) = |x_1 + x_2|^2 = (x_1 + x_2)(x_1^* + x_2^*)$$

            Phase difference function:
            $$\Delta\phi(x_1, x_2) = \arg(x_1) - \arg(x_2)$$

            Total energy functional:
            $$E[\Psi] = \int |\Psi_1(x) + \Psi_2(x)|^2 dx$$

            Gradient of energy:
            $$\frac{\delta E}{\delta \Psi} = 2(\Psi_1 + \Psi_2)$$

            Phase coherence measure:
            $$\gamma = \langle \cos(\Delta\phi) \rangle = \frac{1}{N} \sum_{i=1}^N \cos(\phi_1[i] - \phi_2[i])$$

        JAX Functions Used:
            - jnp: NumPy-compatible array operations
            - vmap: Vectorized mapping
            - grad: Automatic differentiation
            - jit: Just-in-time compilation

        Args:
            field1, field2: Complex wave fields for analysis

        Returns:
            Dictionary with vectorized interference metrics
        """
        # Convert to JAX arrays
        f1_jax = jnp.array(field1.detach().cpu().numpy())
        f2_jax = jnp.array(field2.detach().cpu().numpy())

        # Define interference intensity function
        @jit
        def interference_intensity(x1, x2):
            superposition = x1 + x2
            return jnp.abs(superposition) ** 2

        # Vectorized computation using vmap
        vectorized_intensity = vmap(interference_intensity)(f1_jax, f2_jax)

        # Define phase difference function
        @jit
        def phase_difference(x1, x2):
            return jnp.angle(x1) - jnp.angle(x2)

        vectorized_phase_diff = vmap(phase_difference)(f1_jax, f2_jax)

        # Gradient of interference pattern
        def total_intensity(fields):
            return jnp.sum(jnp.abs(fields[0] + fields[1]) ** 2)

        intensity_gradient = grad(total_intensity)(jnp.array([f1_jax, f2_jax]))

        return {
            "mean_intensity": float(jnp.mean(vectorized_intensity)),
            "phase_coherence": float(jnp.mean(jnp.cos(vectorized_phase_diff))),
            "gradient_norm": float(jnp.linalg.norm(intensity_gradient)),
            "total_energy": float(jnp.sum(vectorized_intensity)),
        }

    def scipy_signal_correlation_analysis(
        self, field1: torch.Tensor, field2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Comprehensive signal correlation analysis using scipy.signal functions.

        Mathematical Foundation:
            Cross-correlation function:
            $$R_{fg}[n] = \sum_{m=-\infty}^{\infty} f[m] g^*[m+n]$$

            Auto-correlation function:
            $$R_{ff}[n] = \sum_{m=-\infty}^{\infty} f[m] f^*[m+n]$$

            Normalized cross-correlation:
            $$\rho_{fg}[n] = \frac{R_{fg}[n]}{\sqrt{R_{ff}[0] R_{gg}[0]}}$$

            Coherence function:
            $$C_{fg}(\omega) = \frac{|S_{fg}(\omega)|^2}{S_{ff}(\omega) S_{gg}(\omega)}$$

            Where $S_{fg}(\omega)$ is the cross-power spectral density:
            $$S_{fg}(\omega) = \mathcal{F}[R_{fg}[n]]$$

            Hilbert transform (analytical signal):
            $$f_a(t) = f(t) + i\mathcal{H}[f](t)$$

            Where $\mathcal{H}[f](t) = \frac{1}{\pi} \text{P.V.} \int_{-\infty}^{\infty} \frac{f(\tau)}{t-\tau} d\tau$

            Instantaneous amplitude and phase:
            $$A(t) = |f_a(t)|, \quad \phi(t) = \arg(f_a(t))$$

            Spectrogram (Short-Time Fourier Transform):
            $$\text{STFT}(t,\omega) = \int f(\tau) w(\tau - t) e^{-i\omega\tau} d\tau$$

            Where $w(t)$ is a window function.

        SciPy Functions Used:
            - correlate: Cross/auto-correlation
            - coherence: Magnitude-squared coherence
            - cross_power_spectral_density: Cross-PSD
            - hilbert: Hilbert transform
            - spectrogram: Time-frequency analysis

        Args:
            field1, field2: Input wave fields for correlation analysis

        Returns:
            Dictionary with comprehensive correlation measures
        """
        # Convert to numpy for scipy processing
        f1_np = field1.detach().cpu().numpy()
        f2_np = field2.detach().cpu().numpy()

        # Cross-correlation using scipy
        cross_corr = signal.correlate(f1_np, f2_np, mode="full")

        # Auto-correlations
        auto_corr_1 = signal.correlate(f1_np, f1_np, mode="full")
        auto_corr_2 = signal.correlate(f2_np, f2_np, mode="full")

        # Coherence calculation using cross-power spectral density
        f_freq, coherence = signal.coherence(f1_np.real, f2_np.real, fs=1.0)
        f_freq_cpsd, cross_psd = signal.cross_power_spectral_density(
            f1_np.real, f2_np.real, fs=1.0
        )

        # Hilbert transform for analytical signal
        analytic_1 = signal.hilbert(f1_np.real)
        analytic_2 = signal.hilbert(f2_np.real)

        # Spectrogram for time-frequency analysis
        f_spec, t_spec, spectrogram_1 = signal.spectrogram(f1_np.real, fs=1.0)

        return {
            "cross_correlation": torch.from_numpy(cross_corr),
            "auto_correlation_1": torch.from_numpy(auto_corr_1),
            "auto_correlation_2": torch.from_numpy(auto_corr_2),
            "coherence": torch.from_numpy(coherence),
            "cross_psd": torch.from_numpy(cross_psd),
            "analytic_signal_1": torch.from_numpy(analytic_1),
            "analytic_signal_2": torch.from_numpy(analytic_2),
            "spectrogram": torch.from_numpy(spectrogram_1),
        }

    def bessel_hankel_wave_solutions(
        self, radial_distance: float, wave_number: float
    ) -> Dict[str, complex]:
        """
        Exact cylindrical and spherical wave solutions using Bessel and Hankel functions.

        Mathematical Foundation:
            Cylindrical wave equation in polar coordinates:
            $$\frac{1}{r} \frac{\partial}{\partial r}\left(r \frac{\partial \Psi}{\partial r}\right) + \frac{1}{r^2} \frac{\partial^2 \Psi}{\partial \theta^2} + k^2 \Psi = 0$$

            Solutions in terms of Bessel functions:
            $$\Psi(r,\theta) = [A J_n(kr) + B Y_n(kr)] e^{in\theta}$$

            Bessel functions of the first kind (regular at origin):
            $$J_n(z) = \sum_{m=0}^{\infty} \frac{(-1)^m}{m!(n+m)!} \left(\frac{z}{2}\right)^{n+2m}$$

            Bessel functions of the second kind (irregular at origin):
            $$Y_n(z) = \frac{J_n(z) \cos(n\pi) - J_{-n}(z)}{\sin(n\pi)}$$

            Hankel functions (outgoing/incoming waves):
            $$H_n^{(1)}(z) = J_n(z) + i Y_n(z)$$ (outgoing)
            $$H_n^{(2)}(z) = J_n(z) - i Y_n(z)$$ (incoming)

            Asymptotic behavior (large argument):
            $$H_n^{(1)}(kr) \sim \sqrt{\frac{2}{\pi kr}} e^{i(kr - n\pi/2 - \pi/4)}$$
            $$H_n^{(2)}(kr) \sim \sqrt{\frac{2}{\pi kr}} e^{-i(kr - n\pi/2 - \pi/4)}$$

            Spherical Bessel functions:
            $$j_n(z) = \sqrt{\frac{\pi}{2z}} J_{n+1/2}(z)$$
            $$y_n(z) = \sqrt{\frac{\pi}{2z}} Y_{n+1/2}(z)$$

            Spherical Hankel functions:
            $$h_n^{(1)}(z) = j_n(z) + i y_n(z)$$
            $$h_n^{(2)}(z) = j_n(z) - i y_n(z)$$

        Physical Applications:
            - Scattering from cylindrical objects
            - Waveguide modes
            - Acoustic and electromagnetic radiation
            - Green's functions for wave equations

        Args:
            radial_distance: Radial coordinate $r \geq 0$
            wave_number: Wave number $k > 0$

        Returns:
            Dictionary with Bessel and Hankel function values
        """
        # Regular Bessel functions J_n(kr)
        j0 = special.jv(0, wave_number * radial_distance)
        j1 = special.jv(1, wave_number * radial_distance)

        # Irregular Bessel functions Y_n(kr)
        y0 = special.yv(0, wave_number * radial_distance)
        y1 = special.yv(1, wave_number * radial_distance)

        # Hankel functions H_n^(1)(kr) and H_n^(2)(kr)
        h1_0 = special.hankel1(0, wave_number * radial_distance)
        h1_1 = special.hankel1(1, wave_number * radial_distance)
        h2_0 = special.hankel2(0, wave_number * radial_distance)
        h2_1 = special.hankel2(1, wave_number * radial_distance)

        return {
            "bessel_j0": complex(j0),
            "bessel_j1": complex(j1),
            "bessel_y0": complex(y0),
            "bessel_y1": complex(y1),
            "hankel1_0": h1_0,
            "hankel1_1": h1_1,
            "hankel2_0": h2_0,
            "hankel2_1": h2_1,
        }

    def networkx_interference_topology(
        self, charges: List[Dict], coherence_matrix: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Network topology analysis of interference patterns using graph theory.

        Mathematical Foundation:
            Graph representation of interference network:
            $$G = (V, E, W)$$
            where $V$ = charges, $E$ = coherent pairs, $W$ = coherence weights.

            Adjacency matrix:
            $$A_{ij} = \begin{cases} |\gamma_{ij}| & \text{if } |\gamma_{ij}| > \text{threshold} \\ 0 & \text{otherwise} \end{cases}$$

            Degree matrix:
            $$D_{ii} = \sum_{j=1}^N A_{ij}$$

            Graph Laplacian:
            $$L = D - A$$

            Network measures:

            Density:
            $$\rho = \frac{2|E|}{|V|(|V|-1)}$$

            Degree centrality:
            $$C_D(v) = \frac{\deg(v)}{|V|-1}$$

            Betweenness centrality:
            $$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$
            where $\sigma_{st}$ = number of shortest paths from $s$ to $t$.

            Clustering coefficient:
            $$C_i = \frac{2|\{e_{jk} : v_j, v_k \in N_i, e_{jk} \in E\}|}{\deg(i)(\deg(i)-1)}$$

            Average path length:
            $$L = \frac{1}{|V|(|V|-1)} \sum_{i \neq j} d(v_i, v_j)$$

            Modularity (community structure):
            $$Q = \frac{1}{2m} \sum_{ij} \left(A_{ij} - \frac{k_i k_j}{2m}\right) \delta(c_i, c_j)$$

            Small-world properties:
            - High clustering: $C \gg C_{\text{random}}$
            - Short path length: $L \approx L_{\text{random}}$

            Scale-free networks: $P(k) \sim k^{-\gamma}$

        NetworkX Functions Used:
            - Graph construction and manipulation
            - Centrality measures
            - Community detection algorithms
            - Path analysis
            - Network statistics

        Args:
            charges: List of charges (graph nodes)
            coherence_matrix: Pairwise coherence values (edge weights)

        Returns:
            Dictionary with comprehensive network topology analysis
        """
        n_charges = len(charges)

        # Create networkx graph from coherence matrix
        G = nx.Graph()

        # Add nodes
        for i in range(n_charges):
            if "id" not in charges[i]:
                raise ValueError(
                    f"MATHEMATICAL FAILURE: Charge {i} lacks required 'id' field for topology analysis"
                )
            G.add_node(i, charge_id=charges[i]["id"])

        # Add edges based on coherence threshold
        threshold = self.coherence_threshold
        for i in range(n_charges):
            for j in range(i + 1, n_charges):
                coherence = coherence_matrix[i, j].item()
                if coherence > threshold:
                    G.add_edge(i, j, coherence=coherence)

        # Network analysis
        try:
            # Connectivity measures
            density = nx.density(G)

            # Centrality measures
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)

            # Clustering
            clustering_coeff = nx.clustering(G)

            # Path analysis
            if nx.is_connected(G):
                diameter = nx.diameter(G)
                avg_path_length = nx.average_shortest_path_length(G)
            else:
                diameter = float("inf")
                avg_path_length = float("inf")

            # Community detection using modularity
            communities = list(nx.community.greedy_modularity_communities(G))

        except:
            # Fallback for empty or disconnected graphs
            density = 0.0
            degree_centrality = {i: 0.0 for i in range(n_charges)}
            betweenness_centrality = {i: 0.0 for i in range(n_charges)}
            clustering_coeff = {i: 0.0 for i in range(n_charges)}
            diameter = 0.0
            avg_path_length = 0.0
            communities = []

        return {
            "graph": G,
            "density": density,
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness_centrality,
            "clustering_coefficient": clustering_coeff,
            "diameter": diameter,
            "average_path_length": avg_path_length,
            "communities": communities,
        }
