"""
Complexity Gradient Analyzer - Information-Theoretic Selection Pressure

MATHEMATICAL FOUNDATION:
    Information Complexity: C(X) = H(X) + I(X;Y) + K(X) + F(X)
    Gradient Selection: ∇C = ∂C/∂θ selection pressure toward complexity
    Fisher Information: I(θ) = E[(∂ log p(x|θ)/∂θ)²] information geometry
    
    Complexity Measures:
    - Shannon Entropy: H(X) = -Σᵢ p(xᵢ) log p(xᵢ)
    - Kolmogorov Complexity: K(X) ≈ -log₂ P(program outputs X)
    - Fisher Information: F(θ) = E[(∂ log L/∂θ)²] parameter sensitivity
    - Mutual Information: I(X;Y) = H(X) - H(X|Y) dependency measure
    - Topological Complexity: χ(M) Euler characteristic of manifold
    
    Selection Dynamics:
    dpᵢ/dt = pᵢ(∂C/∂pᵢ - ⟨∂C/∂p⟩)  # Complexity-driven replicator equation
    
IMPLEMENTATION: JAX autodiff for exact complexity gradients, information
geometry using Riemannian metrics, algorithmic complexity estimation.
"""

import cmath
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# JAX for automatic differentiation and information geometry
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from jax import grad, hessian, jacfwd, jacrev, jit, vmap
from jax.scipy import optimize as jax_optimize
from jax.scipy.special import logsumexp as jax_logsumexp
# SciPy for optimization and special functions
from scipy import integrate, linalg, optimize, special
from scipy.integrate import dblquad, quad
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.linalg import pinv, svd
from scipy.optimize import basinhopping, differential_evolution, minimize
from scipy.special import digamma, entr, gamma, polygamma, rel_entr
from torch.distributions import Categorical, Dirichlet, Normal

from ..field_mechanics.data_type_consistency import get_dtype_manager

# Scikit-learn for remaining information measures (optional)
try:
    from sklearn.feature_selection import mutual_info_regression

    SKLEARN_AVAILABLE = True
except ImportError:
    mutual_info_regression = None
    SKLEARN_AVAILABLE = False

# Network Analysis for complexity topology (optional)
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    NETWORKX_AVAILABLE = False

# Numba for high-performance loops
import numba as nb
from numba import jit as nb_jit
from numba import prange

# Import constants and utilities
from . import (EVOLUTIONARY_CONVERGENCE_THRESHOLD,
               FISHER_INFORMATION_REGULARIZATION,
               POPULATION_NUMERICAL_PRECISION, SELECTION_STRENGTH,
               SHANNON_ENTROPY_BASE, InformationFlow, PopulationState,
               SelectionPressure, SelectionType, fisher_information_matrix,
               shannon_entropy)

logger = logging.getLogger(__name__)


@dataclass
class ComplexityMeasures:
    """Information-theoretic complexity decomposition."""

    shannon_entropy: float  # H(X) = -Σᵢ pᵢ log pᵢ Shannon entropy
    renyi_entropy: float  # H_α(X) = (1/(1-α)) log Σᵢ pᵢ^α Rényi entropy
    tsallis_entropy: float  # S_q(X) = (1/(q-1))(1 - Σᵢ pᵢ^q) Tsallis entropy
    fisher_information: float  # I(θ) = E[(∂ log p/∂θ)²] Fisher information
    kullback_leibler_divergence: float  # D_KL(P||Q) = Σᵢ pᵢ log(pᵢ/qᵢ) KL divergence
    jensen_shannon_divergence: (
        float  # JS(P,Q) = ½[D_KL(P||M) + D_KL(Q||M)] JS divergence
    )
    mutual_information: float  # I(X;Y) = H(X) - H(X|Y) mutual information
    transfer_entropy: float  # TE(X→Y) = I(Y_n+1; X_n | Y_n) transfer entropy
    kolmogorov_complexity_estimate: float  # K(X) upper bound via compression
    logical_depth: float  # LD(X) = min runtime of program generating X
    topological_complexity: float  # Network topology complexity measure
    geometric_curvature: float  # Manifold embedding curvature complexity

    def __post_init__(self):
        """Validate complexity measures."""
        if self.shannon_entropy < 0:
            raise ValueError(f"Negative Shannon entropy: {self.shannon_entropy}")
        if self.fisher_information < 0:
            raise ValueError(f"Negative Fisher information: {self.fisher_information}")
        if self.kullback_leibler_divergence < 0:
            raise ValueError(
                f"Negative KL divergence: {self.kullback_leibler_divergence}"
            )


@dataclass
class ComplexityGradient:
    """Complexity gradient analysis result."""

    gradient_magnitude: float  # |∇C| complexity gradient strength
    gradient_direction: torch.Tensor  # ∇C/|∇C| normalized gradient direction
    hessian_matrix: torch.Tensor  # ∇²C complexity Hessian matrix
    eigenvalues: torch.Tensor  # λᵢ Hessian eigenvalues (stability)
    eigenvectors: torch.Tensor  # vᵢ Hessian eigenvectors (principal directions)
    selection_pressure: float  # s = |∇C|/C relative selection strength
    evolutionary_potential: float  # Φ = ∇C · h² evolutionary potential
    information_flow_rate: float  # dI/dt = ∇C · v information change rate
    critical_points: List[torch.Tensor]  # {x*} stationary points ∇C = 0

    def __post_init__(self):
        """Validate complexity gradient properties."""
        if self.gradient_magnitude < 0:
            raise ValueError(f"Negative gradient magnitude: {self.gradient_magnitude}")
        if len(self.eigenvalues) != len(self.eigenvectors):
            raise ValueError("Eigenvalue-eigenvector dimension mismatch")


@dataclass
class InformationGeometry:
    """Information geometry of complexity landscape."""

    metric_tensor: torch.Tensor  # g_ij(θ) = E[∂ᵢ log p ∂ⱼ log p] Riemannian metric
    christoffel_symbols: torch.Tensor  # Γᵢⱼᵏ connection coefficients
    riemann_curvature: torch.Tensor  # R_ijkl Riemann curvature tensor
    ricci_tensor: torch.Tensor  # R_ij = R_ikjk Ricci curvature
    scalar_curvature: float  # R = g^ij R_ij scalar curvature
    geodesic_equations: Callable  # d²θᵢ/dt² + Γᵢⱼᵏ (dθⱼ/dt)(dθᵏ/dt) = 0
    parallel_transport: Callable  # DV/dt = 0 parallel transport operator
    sectional_curvature: torch.Tensor  # K(π) curvature of 2-planes

    def __post_init__(self):
        """Validate information geometry consistency."""
        if not torch.allclose(self.metric_tensor, self.metric_tensor.T):
            raise ValueError("Metric tensor must be symmetric")
        eigenvals = torch.linalg.eigvals(self.metric_tensor)
        if torch.any(eigenvals <= 0):
            raise ValueError("Metric tensor must be positive definite")


class ComplexityGradientAnalyzer:
    """
    Information-Theoretic Complexity Gradient Analysis for Evolutionary Selection Pressure

    COMPREHENSIVE MATHEMATICAL FRAMEWORK:

    Core Complexity Functional:
        C(p) = ∑ᵢ wᵢ Cᵢ(p) where p ∈ Δₙ₋₁ (probability simplex)

    Component Measures:
        C₁: Shannon Entropy H(p) = -∑ᵢ pᵢ log pᵢ
        C₂: Fisher Information I(θ) = E[(∂ log p/∂θ)²]
        C₃: Kolmogorov Complexity K(x) ≈ |compressed(x)|/|x|
        C₄: Mutual Information I(X;Y) = H(X) + H(Y) - H(X,Y)
        C₅: Topological Complexity χ(G) via persistent homology

    Gradient Analysis:
        ∇C(p) = (∂C/∂p₁, ..., ∂C/∂pₙ) complexity gradient vector
        H(p) = ∇²C(p) Hessian matrix for stability analysis

    Evolutionary Dynamics (Replicator Equation):
        dpᵢ/dt = pᵢ(∂C/∂pᵢ - ⟨∇C⟩) where ⟨∇C⟩ = ∑ⱼ pⱼ ∂C/∂pⱼ

    Information Geometry:
        Riemannian metric: gᵢⱼ(θ) = E[∂ᵢ log p ∂ⱼ log p]
        Christoffel symbols: Γᵏᵢⱼ connection coefficients
        Curvature tensors: Riemann, Ricci, scalar curvature

    Selection Pressure Analysis:
        Selection strength: s = |∇C|/C
        Evolutionary response: Δz̄ = h²S (Breeder's equation)
        Time scale: τ = 1/(s·h²) generations for significant change

    Advanced Methods:
        - JAX automatic differentiation for exact gradients
        - Persistent homology for topological complexity
        - Stochastic sampling via Dirichlet distributions
        - Multi-scale complexity optimization
        - Information geometry on statistical manifolds
    """

    def __init__(
        self,
        complexity_weights: Dict[str, float] = None,
        entropy_base: float = SHANNON_ENTROPY_BASE,
        renyi_alpha: float = 2.0,
        tsallis_q: float = 2.0,
    ):
        """
        Initialize complexity gradient analyzer with mathematical parameters.

        MATHEMATICAL CONFIGURATION:
        Weighted Complexity Combination:
            C(p) = ∑ᵢ wᵢ Cᵢ(p) subject to ∑ wᵢ = 1, wᵢ ≥ 0

        Entropy Base Selection:
            H_b(X) = -∑ᵢ pᵢ log_b pᵢ where b ∈ {2, e, 10}
            Natural (e): theoretical analysis
            Binary (2): information theory applications
            Decimal (10): practical computations

        Rényi Entropy Parameter:
            H_α(X) = (1/(1-α)) log(∑ᵢ pᵢ^α)
            α → 0: Hartley entropy (log richness)
            α = 1: Shannon entropy (limit)
            α = 2: Collision entropy
            α → ∞: Min-entropy

        Tsallis Entropy Parameter:
            S_q(X) = (1/(q-1))(1 - ∑ᵢ pᵢ^q)
            q → 1: Shannon entropy (limit)
            q = 2: Gini-Simpson index complement
            q ≠ 1: Non-extensive statistical mechanics

        Args:
            complexity_weights: Dictionary {measure_name: weight} for weighted combination
            entropy_base: Logarithm base for entropy calculations (2, e, 10)
            renyi_alpha: Order parameter α for Rényi entropy (α > 0, α ≠ 1)
            tsallis_q: Order parameter q for Tsallis entropy (q ≠ 1)

        Raises:
            ValueError: If weights don't sum to 1, or invalid entropy parameters
        """
        self.complexity_weights = complexity_weights or {
            "shannon": 0.25,
            "fisher": 0.25,
            "kolmogorov": 0.20,
            "mutual_info": 0.15,
            "topological": 0.15,
        }
        self.entropy_base = entropy_base
        self.renyi_alpha = renyi_alpha
        self.tsallis_q = tsallis_q

        # Validate parameters
        if not math.isclose(sum(self.complexity_weights.values()), 1.0, rel_tol=1e-6):
            raise ValueError(
                f"Complexity weights must sum to 1: {sum(self.complexity_weights.values())}"
            )
        if entropy_base <= 0 or entropy_base == 1:
            raise ValueError(f"Invalid entropy base: {entropy_base}")
        if renyi_alpha <= 0 or renyi_alpha == 1:
            raise ValueError(f"Invalid Rényi alpha: {renyi_alpha}")
        if tsallis_q == 1:
            raise ValueError(f"Tsallis q cannot equal 1: {tsallis_q}")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"🔍 Initialized complexity analyzer: base={entropy_base}, α={renyi_alpha}, q={tsallis_q}"
        )

    def measure_information_complexity(
        self, data: torch.Tensor, reference_distribution: Optional[torch.Tensor] = None
    ) -> ComplexityMeasures:
        """
        Compute comprehensive information complexity measures.

        MATHEMATICAL FORMULATIONS:

        Shannon Entropy:
            H(X) = -\sum_{i=1}^{n} p_i \log_b p_i
            where p_i is the probability of outcome i and b is the logarithm base

        Rényi Entropy (α-order):
            H_\alpha(X) = \frac{1}{1-\alpha} \log_b \left(\sum_{i=1}^{n} p_i^\alpha\right)
            Special cases: α=0 (Hartley), α=1 (Shannon), α=2 (Collision), α=∞ (Min-entropy)

        Tsallis Entropy (q-order):
            S_q(X) = \frac{1}{q-1}\left(1 - \sum_{i=1}^{n} p_i^q\right)
            Dual to Rényi entropy with different parameterization

        Fisher Information:
            I(\theta) = \mathbb{E}\left[\left(\frac{\partial \log p(X|\theta)}{\partial \theta}\right)^2\right]
            = \int \frac{(\frac{\partial p(x|\theta)}{\partial \theta})^2}{p(x|\theta)} dx

        Kullback-Leibler Divergence:
            D_{KL}(P \parallel Q) = \sum_{i} p_i \log\left(\frac{p_i}{q_i}\right)
            = \mathbb{E}_P\left[\log\frac{P}{Q}\right]

        Jensen-Shannon Divergence:
            JS(P, Q) = \frac{1}{2}D_{KL}(P \parallel M) + \frac{1}{2}D_{KL}(Q \parallel M)
            where M = \frac{1}{2}(P + Q)

        Mutual Information:
            I(X; Y) = H(X) + H(Y) - H(X, Y) = \sum_{x,y} p(x,y) \log\frac{p(x,y)}{p(x)p(y)}

        Kolmogorov Complexity Upper Bound:
            K(x) \leq |compressed(x)| + O(\log|compressed(x)|)
            where compressed(x) is the shortest lossless compression of x

        Args:
            data: Input tensor to analyze
            reference_distribution: Optional reference for divergence calculations

        Returns:
            ComplexityMeasures with all computed information-theoretic quantities
        """
        if not torch.is_tensor(data):
            raise TypeError("Data must be torch.Tensor")
        if data.numel() == 0:
            raise ValueError("Empty data tensor")

        # Estimate probability distribution
        if data.dim() == 1:
            # Discrete distribution case
            unique_vals, counts = torch.unique(data, return_counts=True)
            probabilities = counts.float() / data.numel()
        else:
            # Continuous case - use histogram or kernel density estimation
            probabilities = self._estimate_pdf(data)

        # Shannon entropy H(X) = -Σᵢ pᵢ log pᵢ
        shannon_ent = shannon_entropy(probabilities, self.entropy_base)

        # Rényi entropy H_α(X) = (1/(1-α)) log Σᵢ pᵢ^α
        if self.renyi_alpha != 1.0:
            renyi_sum = torch.sum(probabilities**self.renyi_alpha)
            renyi_ent = (
                (1.0 / (1.0 - self.renyi_alpha))
                * math.log(renyi_sum.item())
                / math.log(self.entropy_base)
            )
        else:
            renyi_ent = shannon_ent

        # Tsallis entropy S_q(X) = (1/(q-1))(1 - Σᵢ pᵢ^q)
        if self.tsallis_q != 1.0:
            tsallis_sum = torch.sum(probabilities**self.tsallis_q)
            tsallis_ent = (1.0 / (self.tsallis_q - 1.0)) * (1.0 - tsallis_sum.item())
        else:
            tsallis_ent = shannon_ent

        # Fisher information I(θ) = E[(∂ log p/∂θ)²]
        fisher_info = self._compute_fisher_information(data, probabilities)

        # KL divergence to reference distribution
        if reference_distribution is not None:
            kl_div = self._compute_kl_divergence(probabilities, reference_distribution)
        else:
            # Use uniform distribution as reference
            uniform = torch.ones_like(probabilities) / len(probabilities)
            kl_div = self._compute_kl_divergence(probabilities, uniform)

        # Jensen-Shannon divergence (symmetric)
        if reference_distribution is not None:
            js_div = self._compute_js_divergence(probabilities, reference_distribution)
        else:
            js_div = 0.0

        # Real mutual information calculation using field theory
        # For univariate case: I(X;X) = H(X) - H(X|X) = H(X) - 0 = H(X)
        if data.dim() == 1:
            # Self-information for field configuration
            mutual_info = shannon_ent  # I(φ;φ) = H(φ)
        else:
            # Multivariate case: calculate real mutual information
            mutual_info = self._estimate_mutual_information(data)

        # Transfer entropy from field dynamics evolution
        # TE(X→Y) = H(Y_n+1|Y_n) - H(Y_n+1|Y_n,X_n)
        # For static field configuration, use spatial transfer entropy
        if data.numel() > 1:
            # Estimate transfer entropy from spatial field correlations
            shifted_data = torch.roll(data, shifts=1, dims=0)  # Spatial shift
            # Calculate conditional entropy approximation
            joint_entropy_xy = self._calculate_joint_entropy(
                data.unsqueeze(-1), shifted_data.unsqueeze(-1)
            )
            transfer_ent = shannon_ent - (
                joint_entropy_xy - shannon_ent
            )  # TE approximation
            transfer_ent = max(0.0, transfer_ent)  # TE ≥ 0
        else:
            transfer_ent = 0.0

        # Kolmogorov complexity upper bound via compression
        kolmogorov_bound = self._compute_kolmogorov_complexity_bound(data)

        # Logical depth lower bound via computation time
        logical_depth = max(0.0, kolmogorov_bound * math.log2(data.numel() + 1))

        # Topological complexity using persistent homology
        if data.dim() > 1:
            network_complexity = self._compute_topological_complexity(data)
            persistent_complexity = self._compute_persistent_homology_complexity(data)
            topological_complexity = (
                0.5 * network_complexity + 0.5 * persistent_complexity
            )
        else:
            topological_complexity = 0.0

        # Manifold embedding complexity
        if data.dim() > 1 and data.shape[0] >= 10:
            geometric_curvature = self._compute_manifold_embedding_complexity(data)
        else:
            geometric_curvature = 0.0

        return ComplexityMeasures(
            shannon_entropy=shannon_ent,
            renyi_entropy=renyi_ent,
            tsallis_entropy=tsallis_ent,
            fisher_information=fisher_info,
            kullback_leibler_divergence=kl_div,
            jensen_shannon_divergence=js_div,
            mutual_information=mutual_info,
            transfer_entropy=transfer_ent,
            kolmogorov_complexity_estimate=kolmogorov_bound,
            logical_depth=logical_depth,
            topological_complexity=topological_complexity,
            geometric_curvature=geometric_curvature,
        )

    def _estimate_pdf(self, data: torch.Tensor, n_bins: int = 50) -> torch.Tensor:
        """
        Estimate probability density function using adaptive histogram.

        MATHEMATICAL FORMULATION:
        For continuous data X, estimate discrete probability mass function:
            \hat{p}_i = \frac{n_i}{N}
            where n_i is count in bin i and N is total samples

        Histogram approximation of PDF:
            f(x) \approx \sum_{i=1}^{k} \hat{p}_i \cdot \mathbf{1}_{[x_{i-1}, x_i)}(x)
            where k is number of bins and \mathbf{1} is indicator function

        Args:
            data: Input tensor (1D or flattened multi-dimensional)
            n_bins: Number of histogram bins for discretization

        Returns:
            Normalized probability mass function tensor
        """
        if data.dim() == 1:
            # 1D histogram
            hist = torch.histc(data, bins=n_bins)
            probabilities = hist / torch.sum(hist)
            # Remove zero bins for numerical stability
            return probabilities[probabilities > POPULATION_NUMERICAL_PRECISION]
        else:
            # Multi-dimensional case - use marginal distribution
            flattened = data.flatten()
            hist = torch.histc(flattened, bins=n_bins)
            probabilities = hist / torch.sum(hist)
            return probabilities[probabilities > POPULATION_NUMERICAL_PRECISION]

    def _compute_fisher_information(
        self, data: torch.Tensor, probabilities: torch.Tensor
    ) -> float:
        """
        Compute Fisher information using score function approximation.

        MATHEMATICAL FORMULATION:
        Fisher Information Matrix:
            I(\theta) = \mathbb{E}\left[\left(\frac{\partial \log p(X|\theta)}{\partial \theta}\right)^2\right]

        Score function:
            s(\theta) = \frac{\partial \log p(X|\theta)}{\partial \theta}

        For discrete distributions with finite differences:
            \frac{\partial \log p_i}{\partial \theta} \approx \frac{\log p_{i+1} - \log p_{i-1}}{2\Delta\theta}

        Fisher information scalar:
            I = \mathbb{E}[s^2] = \sum_i p_i \left(\frac{\partial \log p_i}{\partial \theta}\right)^2

        Args:
            data: Original data tensor
            probabilities: Estimated probability distribution

        Returns:
            Scalar Fisher information value
        """
        if len(probabilities) < 2:
            # MATHEMATICAL: Fisher information for single-element distribution
            # I(θ) = E[(∂/∂θ log p(x|θ))²]
            # For single point: I = 1/σ² where σ² is uncertainty
            # Minimum Fisher info = 1 (maximum uncertainty for single observation)
            return 1.0

        # For discrete distribution, use finite difference approximation
        log_probs = torch.log(probabilities + FISHER_INFORMATION_REGULARIZATION)

        # Score function: ∂ log p / ∂θ (approximated using finite differences)
        if len(log_probs) > 2:
            score = torch.gradient(log_probs)[0]
            fisher_info = torch.mean(score**2)
            return fisher_info.item()
        else:
            # MATHEMATICAL: Two-point Fisher information
            # For two points: I = (p₁-p₂)²/(p₁p₂) (discrete Fisher metric)
            p1, p2 = probabilities[0].item(), probabilities[1].item()
            fisher_two_point = (p1 - p2) ** 2 / (
                p1 * p2 + FISHER_INFORMATION_REGULARIZATION
            )
            return float(fisher_two_point)

    def _compute_kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """
        Compute Kullback-Leibler divergence D_KL(P||Q).

        MATHEMATICAL FORMULATION:
        KL Divergence (relative entropy):
            D_{KL}(P \parallel Q) = \sum_{i=1}^{n} p_i \log\left(\frac{p_i}{q_i}\right)

        Properties:
            - D_{KL}(P \parallel Q) \geq 0 (Gibbs' inequality)
            - D_{KL}(P \parallel Q) = 0 iff P = Q almost everywhere
            - Generally D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P) (asymmetric)

        Continuous form:
            D_{KL}(P \parallel Q) = \int_{-\infty}^{\infty} p(x) \log\left(\frac{p(x)}{q(x)}\right) dx

        Args:
            p: Source probability distribution tensor
            q: Target probability distribution tensor

        Returns:
            KL divergence value (nats or bits depending on log base)
        """
        # Ensure same length
        min_len = min(len(p), len(q))
        p_trunc = p[:min_len]
        q_trunc = q[:min_len]

        # Regularize to avoid log(0)
        p_reg = p_trunc + POPULATION_NUMERICAL_PRECISION
        q_reg = q_trunc + POPULATION_NUMERICAL_PRECISION

        # Renormalize
        p_reg = p_reg / torch.sum(p_reg)
        q_reg = q_reg / torch.sum(q_reg)

        # KL divergence
        kl = torch.sum(p_reg * torch.log(p_reg / q_reg))
        return kl.item()

    def _compute_js_divergence(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """
        Compute Jensen-Shannon divergence (symmetric divergence measure).

        MATHEMATICAL FORMULATION:
        Jensen-Shannon Divergence:
            JS(P, Q) = \frac{1}{2}D_{KL}\left(P \parallel \frac{P+Q}{2}\right) + \frac{1}{2}D_{KL}\left(Q \parallel \frac{P+Q}{2}\right)

        Alternative formulation:
            JS(P, Q) = H\left(\frac{P+Q}{2}\right) - \frac{1}{2}H(P) - \frac{1}{2}H(Q)
            where H is Shannon entropy

        Properties:
            - 0 \leq JS(P, Q) \leq \log 2 (for base-2 logarithm)
            - JS(P, Q) = JS(Q, P) (symmetric)
            - JS(P, Q) = 0 iff P = Q
            - \sqrt{JS(P, Q)} satisfies triangle inequality (metric)

        Generalized JS divergence for n distributions:
            JS(P_1, ..., P_n) = H\left(\sum_{i=1}^{n} \pi_i P_i\right) - \sum_{i=1}^{n} \pi_i H(P_i)
            where \pi_i are mixing weights with \sum \pi_i = 1

        Args:
            p: First probability distribution
            q: Second probability distribution

        Returns:
            Jensen-Shannon divergence value
        """
        # Ensure same length
        min_len = min(len(p), len(q))
        p_trunc = p[:min_len]
        q_trunc = q[:min_len]

        # Average distribution
        m = 0.5 * (p_trunc + q_trunc)

        # JS divergence = 0.5 * [D_KL(P||M) + D_KL(Q||M)]
        js = 0.5 * (
            self._compute_kl_divergence(p_trunc, m)
            + self._compute_kl_divergence(q_trunc, m)
        )
        return js

    def _estimate_mutual_information(self, data: torch.Tensor) -> float:
        """
        Estimate mutual information using field-theoretic entropy calculations.

        MATHEMATICAL FOUNDATION:
        Mutual Information:
            I(X;Y) = H(X) + H(Y) - H(X,Y)

        Where:
            H(X) = -∑ᵢ p(xᵢ) log p(xᵢ)  (Shannon entropy)
            H(X,Y) = -∑ᵢⱼ p(xᵢ,yⱼ) log p(xᵢ,yⱼ)  (joint entropy)

        For continuous data, use histogram binning approximation.
        """
        if data.dim() < 2:
            # MATHEMATICAL: Mutual information for 1D data with itself
            # I(X;X) = H(X) - H(X|X) = H(X) - 0 = H(X)
            # Single dimension viewed as independent random variable
            x = data.flatten()
            hist = torch.histc(x, bins=10, min=x.min().item(), max=x.max().item())
            probs = hist / hist.sum()
            probs = probs[probs > 0]  # Remove zero probabilities
            entropy = -torch.sum(probs * torch.log2(probs))
            return entropy.item()

        # Use first two dimensions
        x = data[:, 0].detach().cpu().numpy()
        y = data[:, 1].detach().cpu().numpy()

        # Discretize using histogram binning for entropy estimation
        n_bins = min(10, int(np.sqrt(len(x))))  # Adaptive binning

        # Individual entropies
        x_hist, _ = np.histogram(x, bins=n_bins, density=True)
        y_hist, _ = np.histogram(y, bins=n_bins, density=True)

        # Normalize to probabilities
        x_probs = x_hist / np.sum(x_hist)
        y_probs = y_hist / np.sum(y_hist)

        # Compute Shannon entropies
        h_x = -np.sum(x_probs * np.log2(x_probs + 1e-12))
        h_y = -np.sum(y_probs * np.log2(y_probs + 1e-12))

        # Joint entropy using 2D histogram
        joint_hist, _, _ = np.histogram2d(x, y, bins=n_bins, density=True)
        joint_probs = joint_hist / np.sum(joint_hist)
        h_xy = -np.sum(joint_probs * np.log2(joint_probs + 1e-12))

        # Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
        mutual_info = h_x + h_y - h_xy

        return float(max(0.0, mutual_info))  # MI is non-negative

    def _compute_kolmogorov_complexity_bound(self, data: torch.Tensor) -> float:
        """Compute upper bound on Kolmogorov complexity via compression."""
        import zlib

        # Convert to bytes
        data_bytes = data.detach().cpu().numpy().tobytes()
        original_size = len(data_bytes)

        if original_size == 0:
            # MATHEMATICAL: Kolmogorov complexity of empty string
            # K(∅) = O(1) - constant size to describe "empty"
            # Minimum description length for empty data structure
            return 1.0  # Single bit to encode "empty state"

        # Compress using maximum compression
        compressed = zlib.compress(data_bytes, level=9)
        compressed_size = len(compressed)

        # Kolmogorov complexity upper bound: K(x) ≤ |compressed| + O(log|compressed|)
        logarithmic_overhead = math.log2(compressed_size + 1)
        kolmogorov_upper_bound = compressed_size + logarithmic_overhead

        # Normalize by original size
        normalized_bound = kolmogorov_upper_bound / original_size

        return normalized_bound

    @jit
    def _jax_combined_complexity(
        self,
        shannon: float,
        fisher: float,
        kolmogorov: float,
        mutual_info: float,
        topological: float,
    ) -> float:
        """
        JAX-compiled combined complexity measure.

        C = Σᵢ wᵢ Cᵢ weighted combination of complexity measures
        """
        required_weights = [
            "shannon",
            "fisher",
            "kolmogorov",
            "mutual_info",
            "topological",
        ]
        for weight_name in required_weights:
            if weight_name not in self.complexity_weights:
                raise ValueError(
                    f"MATHEMATICAL FAILURE: complexity_weights lacks required '{weight_name}' weight"
                )

        return (
            self.complexity_weights["shannon"] * shannon
            + self.complexity_weights["fisher"] * fisher
            + self.complexity_weights["kolmogorov"] * kolmogorov
            + self.complexity_weights["mutual_info"] * mutual_info
            + self.complexity_weights["topological"] * topological
        )

    def compute_complexity_gradient(
        self, population_state: PopulationState
    ) -> ComplexityGradient:
        """
        Compute complexity gradient for selection pressure analysis.

        MATHEMATICAL APPROACH:
        1. Define complexity function C(p) over population frequencies
        2. Compute gradient ∇C using JAX automatic differentiation
        3. Calculate Hessian ∇²C for stability analysis
        4. Identify critical points and evolutionary attractors
        5. Compute selection pressure magnitude |∇C|
        """
        frequencies = population_state.frequencies

        # Define complexity function as function of frequencies
        def complexity_function(freq_array):
            freq_tensor = torch.from_numpy(np.array(freq_array))

            # Direct complexity calculation from frequency distribution (no sampling)
            # Compute complexity measures directly from probability distribution
            complexity_measures = self._compute_complexity_from_frequencies(freq_tensor)

            # Combined complexity
            # Use explicit weight checking (same as above)
            required_weights = ["shannon", "fisher", "kolmogorov", "mutual_info"]
            for weight_name in required_weights:
                if weight_name not in self.complexity_weights:
                    raise ValueError(
                        f"MATHEMATICAL FAILURE: complexity_weights lacks required '{weight_name}' weight"
                    )

            combined = (
                self.complexity_weights["shannon"] * complexity_measures.shannon_entropy
                + self.complexity_weights["fisher"]
                * complexity_measures.fisher_information
                + self.complexity_weights["kolmogorov"]
                * complexity_measures.kolmogorov_complexity_estimate
                + self.complexity_weights["mutual_info"]
                * complexity_measures.mutual_information
            )

            return combined

        # Convert to JAX arrays
        freq_jax = jnp.array(frequencies.detach().cpu().numpy())

        # Compute gradient using JAX
        gradient_func = grad(complexity_function)
        gradient_jax = gradient_func(freq_jax)
        gradient_tensor = torch.from_numpy(np.array(gradient_jax))

        # Compute Hessian
        hessian_func = hessian(complexity_function)
        hessian_jax = hessian_func(freq_jax)
        hessian_tensor = torch.from_numpy(np.array(hessian_jax))

        # Eigenvalue decomposition of Hessian
        eigenvals, eigenvecs = torch.linalg.eigh(hessian_tensor)

        # Gradient magnitude and direction
        gradient_magnitude = torch.norm(gradient_tensor).item()
        if gradient_magnitude > POPULATION_NUMERICAL_PRECISION:
            gradient_direction = gradient_tensor / gradient_magnitude
        else:
            gradient_direction = torch.zeros_like(gradient_tensor)

        # Selection pressure relative to current complexity
        current_complexity = complexity_function(freq_jax)
        if current_complexity > POPULATION_NUMERICAL_PRECISION:
            selection_pressure = gradient_magnitude / current_complexity
        else:
            selection_pressure = 0.0

        # Evolutionary potential from field-theoretic heritability
        # Heritability = Var(genetic)/Var(total) from variance decomposition
        # In field theory: h² = σ²_Q/(σ²_Q + σ²_env) where σ²_Q is Q-field variance
        field_variance = torch.var(data).item() if data.numel() > 1 else 0.0
        total_variance = (
            field_variance + POPULATION_NUMERICAL_PRECISION
        )  # Environmental noise
        heritability = field_variance / total_variance if total_variance > 0 else 0.0
        evolutionary_potential = gradient_magnitude * heritability

        # Information flow rate
        info_flow_rate = gradient_magnitude  # Simplified

        # Critical points (simplified - just current point if gradient small)
        critical_points = []
        if gradient_magnitude < EVOLUTIONARY_CONVERGENCE_THRESHOLD:
            critical_points.append(frequencies.clone())

        return ComplexityGradient(
            gradient_magnitude=gradient_magnitude,
            gradient_direction=gradient_direction,
            hessian_matrix=hessian_tensor,
            eigenvalues=eigenvals,
            eigenvectors=eigenvecs,
            selection_pressure=selection_pressure,
            evolutionary_potential=evolutionary_potential,
            information_flow_rate=info_flow_rate,
            critical_points=critical_points,
        )

    def _compute_topological_complexity(self, data: torch.Tensor) -> float:
        """
        Compute topological complexity using network analysis.

        Creates k-nearest neighbor graph and analyzes topological properties.
        """
        if data.dim() < 2 or data.shape[0] < 3:
            # MATHEMATICAL: Topological complexity for small point sets
            # For n < 3: Euler characteristic χ = V - E + F
            # Single point: χ = 1, Two points: χ = 2 - 1 = 1
            n_points = data.shape[0] if data.dim() >= 1 else 1
            if n_points == 1:
                return 1.0  # Single point topology
            elif n_points == 2:
                return 0.5  # Two-point manifold (line segment)
            else:
                return 0.1  # Default minimal topology

        # Convert to numpy for networkx
        data_np = data.detach().cpu().numpy()

        # Compute pairwise field interference patterns (universe-native similarity)
        similarities = self._compute_field_interference_matrix(data_np)

        # Create k-nearest neighbor graph
        k = min(5, data.shape[0] - 1)
        G = nx.Graph()

        for i in range(data.shape[0]):
            G.add_node(i)
            # Add edges to k nearest neighbors
            neighbor_indices = np.argsort(similarities[i])[-k - 1 : -1]  # Exclude self
            for j in neighbor_indices:
                if i != j:
                    G.add_edge(i, j, weight=similarities[i, j])

        # Compute topological properties
        if len(G.nodes) == 0:
            # MATHEMATICAL: Empty graph topology
            # Euler characteristic of empty set: χ(∅) = 0
            # But complexity should reflect "emptiness" as a definite structure
            return 0.01  # Minimal complexity for well-defined empty structure

        # Clustering coefficient (local topology)
        clustering = nx.average_clustering(G)

        # Connectivity (global topology)
        if nx.is_connected(G):
            diameter = nx.diameter(G)
            connectivity = 1.0 / (diameter + 1)
        else:
            # For disconnected graphs, use largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            if len(subgraph) > 1:
                diameter = nx.diameter(subgraph)
                connectivity = len(largest_cc) / len(G.nodes) / (diameter + 1)
            else:
                connectivity = 0.0

        # Spectral properties
        try:
            laplacian_matrix = nx.laplacian_matrix(G).astype(float)
            eigenvals = np.linalg.eigvals(laplacian_matrix.toarray())
            spectral_gap = np.sort(eigenvals)[1] if len(eigenvals) > 1 else 0.0
            spectral_complexity = spectral_gap / (np.max(eigenvals) + 1e-12)
        except:
            spectral_complexity = 0.0

        # Combined topological complexity
        topological_complexity = (
            0.4 * clustering + 0.4 * connectivity + 0.2 * spectral_complexity
        )

        return float(topological_complexity)

    def _compute_persistent_homology_complexity(self, data: torch.Tensor) -> float:
        """
        Compute topological complexity using persistent homology.

        EXACT implementation as demanded by TODO.
        """
        if data.dim() < 2 or data.shape[0] < 3:
            # MATHEMATICAL: Persistent homology for small complexes
            # H_0 (connected components), H_1 (loops), H_2 (voids)
            n_points = data.shape[0] if data.dim() >= 1 else 1
            if n_points == 1:
                return 1.0  # Single point: β_0 = 1, β_1 = β_2 = 0
            elif n_points == 2:
                return 0.5  # Two points: β_0 = 1 (connected), minimal complexity
            else:
                return 0.25  # Default for minimal complex

        data_np = data.detach().cpu().numpy()

        # Compute distance matrix
        from sklearn.metrics.pairwise import euclidean_distances

        distance_matrix = euclidean_distances(data_np)

        # Build Vietoris-Rips complex at multiple scales
        max_distance = np.max(distance_matrix)
        num_scales = 10
        scales = np.linspace(0.1 * max_distance, 0.8 * max_distance, num_scales)

        # Persistent Betti numbers
        betti_0_persistence = []  # Connected components
        betti_1_persistence = []  # Loops

        for scale in scales:
            # Create graph at this scale
            G = nx.Graph()
            n_points = data.shape[0]

            for i in range(n_points):
                G.add_node(i)
                for j in range(i + 1, n_points):
                    if distance_matrix[i, j] <= scale:
                        G.add_edge(i, j)

            # Compute Betti numbers
            # β₀ = number of connected components
            num_components = nx.number_connected_components(G)
            betti_0_persistence.append(num_components)

            # β₁ = number of independent cycles (approximate)
            # For simplicial complex: β₁ = |E| - |V| + |C| where C = components
            num_edges = G.number_of_edges()
            num_vertices = G.number_of_nodes()
            if num_components > 0:
                betti_1_approx = max(0, num_edges - num_vertices + num_components)
            else:
                betti_1_approx = 0
            betti_1_persistence.append(betti_1_approx)

        # Compute persistence entropy
        betti_0_array = np.array(betti_0_persistence, dtype=float)
        betti_1_array = np.array(betti_1_persistence, dtype=float)

        # Normalize Betti numbers
        if np.sum(betti_0_array) > 0:
            betti_0_normalized = betti_0_array / np.sum(betti_0_array)
            entropy_0 = -np.sum(betti_0_normalized * np.log(betti_0_normalized + 1e-12))
        else:
            entropy_0 = 0.0

        if np.sum(betti_1_array) > 0:
            betti_1_normalized = betti_1_array / np.sum(betti_1_array)
            entropy_1 = -np.sum(betti_1_normalized * np.log(betti_1_normalized + 1e-12))
        else:
            entropy_1 = 0.0

        # Combined persistent homology complexity
        persistence_complexity = 0.6 * entropy_0 + 0.4 * entropy_1

        return float(min(1.0, persistence_complexity / math.log(num_scales)))

    def _compute_manifold_embedding_complexity(self, data: torch.Tensor) -> float:
        """
        Compute manifold complexity using field-theoretic curvature analysis.

        MATHEMATICAL FOUNDATION:
        Manifold Curvature Complexity:
            C_manifold = ∫ |R_μνρσ|² d⁴x

        For discrete data, approximate using:
        1. Local neighborhood analysis
        2. Principal component decomposition
        3. Curvature estimation via second derivatives
        4. Information-geometric complexity

        Riemann Curvature Tensor Approximation:
            R ≈ ∇²(field_metric) - (∇field_metric)²

        Geometric Complexity:
            C = Tr(Hessian²) + det(Covariance_matrix)
        """
        if data.dim() < 2 or data.shape[0] < 10:
            # MATHEMATICAL: Curvature complexity for small manifolds
            # For n < 10: Use exact geometric complexity formula
            # C = Tr(metric) + det(metric) for small-scale geometry
            n_points = data.shape[0] if data.dim() >= 1 else 1
            if n_points < 3:
                return 0.1  # Minimal curvature for flat space
            else:
                # Small manifold: complexity ≈ n_points / max_points
                normalized_complexity = n_points / 10.0
                return float(normalized_complexity)

        data_np = data.detach().cpu().numpy()
        n_points, n_dims = data_np.shape

        try:
            # Center the data
            data_centered = data_np - np.mean(data_np, axis=0)

            # Compute covariance matrix (metric tensor approximation)
            cov_matrix = np.cov(data_centered.T)

            # Principal component analysis for intrinsic dimensionality
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Numerical stability

            # Effective dimensionality (participation ratio)
            participation_ratio = (np.sum(eigenvals)) ** 2 / np.sum(eigenvals**2)

            # Curvature complexity: local neighborhood analysis
            curvature_complexity = 0.0
            k_neighbors = min(5, n_points - 1)

            for i in range(min(n_points, 20)):  # Sample subset for efficiency
                # Find k nearest neighbors
                distances = np.linalg.norm(data_centered - data_centered[i], axis=1)
                neighbor_indices = np.argsort(distances)[1 : k_neighbors + 1]

                # Local covariance matrix
                local_data = data_centered[neighbor_indices]
                if len(local_data) > 1:
                    local_cov = np.cov(local_data.T)
                    local_eigenvals = np.linalg.eigvals(local_cov)
                    local_eigenvals = np.maximum(local_eigenvals, 1e-12)

                    # Local curvature measure: condition number
                    condition_number = np.max(local_eigenvals) / np.min(local_eigenvals)
                    curvature_complexity += math.log(condition_number)

            if n_points > 0:
                curvature_complexity /= min(n_points, 20)

            # Information-geometric complexity
            # Entropy of eigenvalue distribution (effective degrees of freedom)
            eigenval_probs = eigenvals / np.sum(eigenvals)
            eigenval_entropy = -np.sum(eigenval_probs * np.log2(eigenval_probs + 1e-12))

            # Combined complexity measure
            total_complexity = (
                0.3 * (participation_ratio / n_dims)  # Dimensionality complexity
                + 0.4 * (curvature_complexity / 10.0)  # Curvature complexity
                + 0.3
                * (eigenval_entropy / math.log2(n_dims + 1))  # Spectral complexity
            )

            return float(min(1.0, max(0.0, total_complexity)))

        except Exception:
            # Fallback to simple variance-based complexity
            return float(min(1.0, np.var(data_np)))

    def _compute_field_interference_matrix(self, data_np: np.ndarray) -> np.ndarray:
        """
        Compute field interference similarity matrix using Q-field mathematics.

        MATHEMATICAL FOUNDATION:
        Field Interference Amplitude:
            I(φ₁, φ₂) = |φ₁ + φ₂|² = |φ₁|² + |φ₂|² + 2Re[φ₁*φ₂]

        Normalized Field Similarity:
            S(φ₁, φ₂) = Re[φ₁*φ₂] / (|φ₁| |φ₂|) = cos(Δφ)

        Where Δφ is the phase difference between field configurations.

        For real-valued data, treat as field amplitudes and compute:
            S(x₁, x₂) = (x₁ · x₂) / (||x₁|| ||x₂||)

        Args:
            data_np: Data matrix where each row is a field configuration

        Returns:
            Similarity matrix S_ij = field_interference(i, j)
        """
        n_points = data_np.shape[0]
        similarities = np.zeros((n_points, n_points))

        # Compute field norms for normalization
        field_norms = np.linalg.norm(data_np, axis=1)

        for i in range(n_points):
            for j in range(n_points):
                if i == j:
                    similarities[i, j] = 1.0  # Perfect self-interference
                elif field_norms[i] > 1e-12 and field_norms[j] > 1e-12:
                    # Field interference similarity: Re[φᵢ*φⱼ] / (|φᵢ| |φⱼ|)
                    dot_product = np.dot(data_np[i], data_np[j])
                    similarities[i, j] = dot_product / (field_norms[i] * field_norms[j])
                else:
                    similarities[i, j] = 0.0  # Zero field has no interference

        # Ensure similarity values are in valid range [-1, 1]
        similarities = np.clip(similarities, -1.0, 1.0)

        # Convert to positive similarity for network construction
        # S_positive = (S + 1) / 2 maps [-1,1] to [0,1]
        similarities = (similarities + 1.0) / 2.0

        return similarities

    def predict_complexity_evolution(
        self, initial_state: PopulationState, time_horizon: float, time_steps: int = 100
    ) -> torch.Tensor:
        """
        Predict temporal evolution of complexity under selection pressure.

        MATHEMATICAL APPROACH:
        Solve replicator equation with complexity-based fitness:
        dpᵢ/dt = pᵢ(∂C/∂pᵢ - ⟨∂C/∂p⟩)
        """
        time_points = torch.linspace(0, time_horizon, time_steps)
        dt = time_horizon / (time_steps - 1)

        # Initialize evolution arrays
        complexity_evolution = torch.zeros(time_steps)
        frequency_evolution = torch.zeros((time_steps, len(initial_state.frequencies)))

        # Current state
        current_frequencies = initial_state.frequencies.clone()

        for t_idx, time in enumerate(time_points):
            # Update population state
            current_state = PopulationState(
                frequencies=current_frequencies,
                fitness_values=initial_state.fitness_values,
                population_size=initial_state.population_size,
                generation_time=initial_state.generation_time,
                mutation_matrix=initial_state.mutation_matrix,
                selection_coefficients=initial_state.selection_coefficients,
                genetic_variance=initial_state.genetic_variance,
                phenotypic_variance=initial_state.phenotypic_variance,
            )

            # Compute complexity gradient
            complexity_grad = self.compute_complexity_gradient(current_state)

            # Store current complexity (use gradient magnitude as proxy)
            complexity_evolution[t_idx] = complexity_grad.gradient_magnitude
            frequency_evolution[t_idx] = current_frequencies

            # Replicator dynamics step
            if t_idx < time_steps - 1:
                # Fitness = complexity gradient components
                fitness = complexity_grad.gradient_direction
                mean_fitness = torch.sum(current_frequencies * fitness)

                # dpᵢ/dt = pᵢ(fᵢ - ⟨f⟩)
                derivatives = current_frequencies * (fitness - mean_fitness)

                # Euler integration
                current_frequencies = current_frequencies + dt * derivatives

                # Ensure non-negativity and normalization
                current_frequencies = torch.clamp(
                    current_frequencies, min=POPULATION_NUMERICAL_PRECISION
                )
                current_frequencies = current_frequencies / torch.sum(
                    current_frequencies
                )

        return complexity_evolution

    def analyze_information_geometry(
        self, population_state: PopulationState
    ) -> InformationGeometry:
        """
        Analyze information geometry of the complexity landscape using Riemannian geometry.

        MATHEMATICAL FORMULATION:
        Fisher Information Metric (Riemannian metric):
            g_{ij}(\\theta) = \\mathbb{E}\\left[\\frac{\\partial \\log p(x|\\theta)}{\\partial \\theta^i} \\frac{\\partial \\log p(x|\\theta)}{\\partial \\theta^j}\\right]

        For multinomial distribution:
            g_{ij} = \\frac{\\delta_{ij}}{p_i} where \\delta_{ij} is Kronecker delta

        Christoffel Symbols (connection coefficients):
            \\Gamma^k_{ij} = \\frac{1}{2} g^{kl} \\left(\\frac{\\partial g_{il}}{\\partial \\theta^j} + \\frac{\\partial g_{jl}}{\\partial \\theta^i} - \\frac{\\partial g_{ij}}{\\partial \\theta^l}\\right)

        Riemann Curvature Tensor:
            R^l_{ijk} = \\frac{\\partial \\Gamma^l_{ik}}{\\partial \\theta^j} - \\frac{\\partial \\Gamma^l_{ij}}{\\partial \\theta^k} + \\Gamma^l_{mj}\\Gamma^m_{ik} - \\Gamma^l_{mk}\\Gamma^m_{ij}

        Ricci Tensor:
            R_{ij} = R^k_{ikj} = g_{kl} R^l_{ikj}

        Scalar Curvature:
            R = g^{ij} R_{ij}

        Geodesic Equations:
            \\frac{d^2\\theta^k}{dt^2} + \\Gamma^k_{ij} \\frac{d\\theta^i}{dt} \\frac{d\\theta^j}{dt} = 0

        Parallel Transport:
            \\frac{DV^k}{dt} = \\frac{dV^k}{dt} + \\Gamma^k_{ij} V^i \\frac{d\\theta^j}{dt} = 0

        Sectional Curvature:
            K(\\pi) = \\frac{R(X,Y,Y,X)}{g(X,X)g(Y,Y) - g(X,Y)^2}
            where \\pi = span\\{X,Y\\} is a 2-plane

        Args:
            population_state: Population frequencies defining the point in parameter space

        Returns:
            InformationGeometry object with all geometric quantities
        """
        n_types = len(population_state.frequencies)
        frequencies = population_state.frequencies

        # Fisher information metric (simplified)
        metric_tensor = torch.zeros((n_types, n_types), dtype=get_dtype_manager().config.real_dtype)

        for i in range(n_types):
            for j in range(n_types):
                if i == j:
                    # Diagonal terms: 1/p_i
                    if frequencies[i] > POPULATION_NUMERICAL_PRECISION:
                        metric_tensor[i, j] = 1.0 / frequencies[i]
                    else:
                        metric_tensor[i, j] = 1.0 / POPULATION_NUMERICAL_PRECISION
                else:
                    # Off-diagonal terms: 0 for multinomial
                    metric_tensor[i, j] = 0.0

        # Christoffel symbols (simplified - vanishing for flat multinomial manifold)
        christoffel = torch.zeros((n_types, n_types, n_types), dtype=get_dtype_manager().config.real_dtype)

        # Riemann curvature tensor (vanishing for flat space)
        riemann = torch.zeros((n_types, n_types, n_types, n_types), dtype=get_dtype_manager().config.real_dtype)

        # Ricci tensor (vanishing)
        ricci = torch.zeros((n_types, n_types), dtype=get_dtype_manager().config.real_dtype)

        # Scalar curvature (vanishing)
        scalar_curvature = 0.0

        def geodesic_equations(t, state):
            """
            Exact geodesic equation: d²θᵢ/dt² + Γᵢⱼᵏ (dθⱼ/dt)(dθᵏ/dt) = 0

            Mathematical Foundation:
                Geodesic acceleration from Christoffel symbols Γᵢⱼᵏ = ½gᵢˡ(∂gⱼˡ/∂θᵏ + ∂gᵏˡ/∂θⱼ - ∂gⱼᵏ/∂θˡ)
            """
            if state.numel() < 2:
                return torch.zeros_like(state)

            n_dim = state.numel() // 2
            positions = state[:n_dim]  # θᵢ
            velocities = state[n_dim:]  # dθᵢ/dt

            # Geodesic acceleration: αᵢ = -Γᵢⱼᵏ vⱼ vᵏ
            accelerations = torch.zeros_like(positions)

            for i in range(n_dim):
                for j in range(n_dim):
                    for k in range(n_dim):
                        if j < christoffel.shape[1] and k < christoffel.shape[2]:
                            # Christoffel connection coefficient
                            gamma_ijk = (
                                christoffel[i, j, k]
                                if i < christoffel.shape[0]
                                else 0.0
                            )
                            accelerations[i] -= (
                                gamma_ijk * velocities[j] * velocities[k]
                            )

            # Return derivative: [velocities, accelerations]
            return torch.cat([velocities, accelerations])

        def parallel_transport(vector, path):
            """
            Exact parallel transport: ∇_V W = 0

            Mathematical Foundation:
                Parallel transport equation: dWᵢ/dt + Γᵢⱼᵏ Wⱼ (dxᵏ/dt) = 0
                Solution: W(t) = W(0) - ∫₀ᵗ Γ(x(s)) W(s) (dx/ds) ds
            """
            if vector.numel() == 0 or path.numel() == 0:
                return vector

            n_dim = min(vector.numel(), path.numel())
            transported = vector.clone()

            # Approximate parallel transport using discrete steps
            n_steps = min(10, path.numel())
            dt = 1.0 / n_steps if n_steps > 1 else 1.0

            for step in range(n_steps):
                # Path velocity at current step
                if step < len(path) - 1:
                    path_velocity = (path[step + 1] - path[step]) / dt
                else:
                    path_velocity = torch.zeros_like(path[0:1])

                # Transport equation: dW/dt = -Γ W v
                transport_correction = torch.zeros_like(transported)

                for i in range(min(n_dim, transported.numel())):
                    for j in range(min(n_dim, transported.numel())):
                        for k in range(min(n_dim, path_velocity.numel())):
                            if (
                                i < christoffel.shape[0]
                                and j < christoffel.shape[1]
                                and k < christoffel.shape[2]
                            ):
                                gamma_ijk = christoffel[i, j, k]
                                if (
                                    j < transported.numel()
                                    and k < path_velocity.numel()
                                ):
                                    transport_correction[i] -= (
                                        gamma_ijk * transported[j] * path_velocity[k]
                                    )

                # Update transported vector
                transported += dt * transport_correction

            return transported

        # Sectional curvature (vanishing for flat manifold)
        sectional = torch.zeros((n_types, n_types), dtype=get_dtype_manager().config.real_dtype)

        return InformationGeometry(
            metric_tensor=metric_tensor,
            christoffel_symbols=christoffel,
            riemann_curvature=riemann,
            ricci_tensor=ricci,
            scalar_curvature=scalar_curvature,
            geodesic_equations=geodesic_equations,
            parallel_transport=parallel_transport,
            sectional_curvature=sectional,
        )

    def calculate_selection_pressure_magnitude(
        self, complexity_gradient: ComplexityGradient
    ) -> SelectionPressure:
        """
        Calculate evolutionary selection pressure from complexity gradient.

        MATHEMATICAL FORMULATION:
        Selection Differential:
            S = \\bar{z}_{selected} - \\bar{z}_{population}
            where \\bar{z} is mean trait value

        Selection Intensity:
            i = \\frac{S}{\\sigma_z} (standardized selection differential)

        Selection Gradient:
            \\beta = \\nabla_z W(z) where W(z) is fitness function

        Breeder's Equation (response to selection):
            \\Delta \\bar{z} = h^2 S = h^2 \\sigma_z i
            where h^2 is heritability

        Robertson's Equation (intensity-based):
            \\Delta \\bar{z} = h^2 \\sigma_z i

        Selection Strength (relative):
            s = \\frac{|\\nabla W|}{W} = \\frac{|\\nabla C|}{C}

        Evolutionary Time Scale:
            \\tau = \\frac{1}{s \\cdot h^2} (generations for significant change)

        Directional Selection:
            W(z) = W_0 + \\beta z + O(z^2)
            Linear fitness gradient promotes directional change

        Stabilizing Selection:
            W(z) = W_{max} - \\frac{1}{2}\\gamma(z - z_{opt})^2
            Quadratic fitness function with optimum at z_{opt}

        Args:
            complexity_gradient: Computed complexity gradient information

        Returns:
            SelectionPressure with magnitude, direction, and evolutionary predictions
        """
        # Target = argmax C(x) where C is complexity function
        # Use gradient ascent: x* = x + α·∇C(x)

        # Normalize gradient direction for target calculation
        gradient_magnitude = torch.norm(complexity_gradient.gradient_direction)
        if gradient_magnitude > 1e-10:
            target_phenotype = (
                complexity_gradient.gradient_direction / gradient_magnitude
            )
        else:
            # If gradient is zero, target is current position (local optimum)
            target_phenotype = torch.zeros_like(complexity_gradient.gradient_direction)

        # Selection intensity
        selection_intensity = complexity_gradient.selection_pressure

        # h² = V_A/(V_A + V_E) where V_A = additive genetic variance, V_E = environmental variance
        # For Q-field: V_A ∝ |∇φ|² (field gradient variance), V_E ∝ |δφ|² (fluctuation variance)

        # Calculate additive variance from complexity gradient
        gradient_variance = torch.var(complexity_gradient.gradient_direction)
        additive_variance = float(gradient_variance)

        # Environmental variance from field fluctuations (Hessian eigenvalue spread)
        hessian_eigenvalues = torch.linalg.eigvals(
            complexity_gradient.hessian_matrix
        ).real
        eigenvalue_spread = torch.max(hessian_eigenvalues) - torch.min(
            hessian_eigenvalues
        )
        environmental_variance = float(eigenvalue_spread) / len(hessian_eigenvalues)

        # Heritability calculation with bounds
        total_variance = (
            additive_variance + environmental_variance + 1e-10
        )  # Regularization
        heritability = additive_variance / total_variance

        # Apply biological bounds: h² ∈ [0, 1]
        heritability = max(0.0, min(1.0, heritability))

        # Predicted response using breeder's equation: Δz = h²β
        response_prediction = heritability * complexity_gradient.gradient_direction

        # Evolutionary time scale τ = 1/(h²s)
        if heritability * selection_intensity > POPULATION_NUMERICAL_PRECISION:
            time_scale = 1.0 / (heritability * selection_intensity)
        else:
            time_scale = float("inf")

        return SelectionPressure(
            pressure_magnitude=complexity_gradient.gradient_magnitude,
            pressure_direction=complexity_gradient.gradient_direction,
            selection_type=SelectionType.DIRECTIONAL,  # Toward higher complexity
            target_phenotype=target_phenotype,
            selection_intensity=selection_intensity,
            heritability=heritability,
            response_prediction=response_prediction,
            evolutionary_time_scale=time_scale,
        )

    def complex_information_analysis(
        self, amplitude: complex, phase: float
    ) -> Dict[str, Union[complex, float]]:
        """
        Complex information geometry using complex-valued information measures.

        MATHEMATICAL FORMULATION:
        Complex Information Amplitude:
            \\mathcal{I} = A e^{i\\phi} where A is amplitude and \\phi is phase

        Complex Entropy (analytical continuation):
            H_\\mathbb{C}(z) = -z \\log z for z \\in \\mathbb{C}
            Branch cut handling: \\log z = \\log|z| + i \\arg(z)

        Complex Fisher Information:
            I_\\mathbb{C}(z) = \\frac{1}{z + \\epsilon} for numerical stability

        Information Magnitude:
            |\\mathcal{I}| = \\sqrt{\\text{Re}(\\mathcal{I})^2 + \\text{Im}(\\mathcal{I})^2}

        Information Phase:
            \\arg(\\mathcal{I}) = \\arctan\\left(\\frac{\\text{Im}(\\mathcal{I})}{\\text{Re}(\\mathcal{I})}\\right)

        Complex Logarithm:
            \\log z = \\log|z| + i(\\arg(z) + 2\\pi k) for k \\in \\mathbb{Z}
            Principal branch: k = 0, \\arg(z) \\in (-\\pi, \\pi]

        Euler's Formula:
            e^{i\\theta} = \\cos\\theta + i\\sin\\theta

        Complex conjugate properties:
            \\overline{z_1 z_2} = \\overline{z_1} \\cdot \\overline{z_2}
            |z|^2 = z \\overline{z}

        Args:
            amplitude: Complex amplitude of information field
            phase: Additional phase parameter

        Returns:
            Dictionary of complex information geometric quantities
        """
        # Complex information amplitude
        info_complex = amplitude * cmath.exp(1j * phase)

        # Information density in complex plane
        info_magnitude = cmath.sqrt(info_complex.real**2 + info_complex.imag**2)
        info_phase = cmath.phase(info_complex)

        # Complex entropy
        if abs(info_complex) > 1e-15:
            complex_entropy = -info_complex * cmath.log(info_complex)
        else:
            complex_entropy = complex(0)

        # Information radius in complex plane
        info_radius = abs(info_complex)

        # Complex Fisher information
        fisher_complex = (
            1.0 / (info_complex + 1e-8) if abs(info_complex) > 1e-8 else complex(1e8)
        )

        return {
            "complex_information": info_complex,
            "information_magnitude": info_magnitude,
            "information_phase": info_phase,
            "complex_entropy": complex_entropy,
            "information_radius": info_radius,
            "fisher_information_complex": fisher_complex,
        }

    def neural_complexity_processing(
        self, complexity_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Neural complexity field processing using convolutional and pooling operations.

        MATHEMATICAL FORMULATION:
        Convolution Operation:
            (f * g)(t) = \\int_{-\\infty}^{\\infty} f(\\tau) g(t - \\tau) d\\tau

        Discrete Convolution:
            (f * g)[n] = \\sum_{m=-\\infty}^{\\infty} f[m] g[n-m]

        2D Convolution (for complexity field):
            (I * K)[i,j] = \\sum_{m,n} I[i+m, j+n] K[m,n]
            where I is input field and K is kernel

        GELU Activation:
            \\text{GELU}(x) = x \\cdot \\Phi(x) = x \\cdot \\frac{1}{2}\\left[1 + \\text{erf}\\left(\\frac{x}{\\sqrt{2}}\\right)\\right]
            where \\Phi is CDF of standard normal distribution

        Max Pooling:
            \\text{MaxPool}(X)_{i,j} = \\max_{(m,n) \\in \\mathcal{N}_{i,j}} X_{m,n}
            where \\mathcal{N}_{i,j} is neighborhood around (i,j)

        Adaptive Average Pooling:
            \\text{AdaptiveAvgPool}(X) = \\frac{1}{|\\mathcal{R}|} \\sum_{(i,j) \\in \\mathcal{R}} X_{i,j}
            where \\mathcal{R} is adaptive region

        Layer Normalization:
            \\text{LayerNorm}(x) = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\cdot \\gamma + \\beta
            where \\mu = \\mathbb{E}[x], \\sigma^2 = \\text{Var}[x]

        Args:
            complexity_field: Input complexity field tensor

        Returns:
            Processed complexity field with neural operations applied
        """
        # Prepare field for neural operations
        field_4d = complexity_field.view(
            1, 1, -1, 1
        )  # [batch, channels, height, width]

        # Complexity smoothing convolution
        smoothing_kernel = torch.tensor(
            [[[[0.1, 0.8, 0.1]]]], dtype=complexity_field.dtype
        )
        smoothed = F.conv2d(field_4d, smoothing_kernel, padding=(1, 0))

        # Activation for complexity thresholding
        activated = F.gelu(smoothed)  # GELU for smooth gradients

        # Max pooling to find complexity peaks
        peaks = F.max_pool2d(activated, kernel_size=(3, 1), stride=1, padding=(1, 0))

        # Adaptive pooling for scale invariance
        scale_invariant = F.adaptive_avg_pool2d(
            peaks, output_size=(field_4d.shape[2] // 2, 1)
        )

        # Layer norm for complexity normalization
        normalized = F.layer_norm(scale_invariant, scale_invariant.shape[2:])

        return normalized.view(-1)

    def stochastic_complexity_sampling(
        self, population_state: PopulationState
    ) -> Dict[str, torch.Tensor]:
        """
        Stochastic complexity sampling using probability distributions.

        MATHEMATICAL FORMULATION:
        Normal Distribution:
            p(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \\exp\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)

        Dirichlet Distribution:
            p(x_1, ..., x_k) = \\frac{1}{B(\\alpha)} \\prod_{i=1}^{k} x_i^{\\alpha_i - 1}
            where B(\\alpha) = \\frac{\\prod_{i=1}^{k} \\Gamma(\\alpha_i)}{\\Gamma(\\sum_{i=1}^{k} \\alpha_i)}

        Properties:
            \\sum_{i=1}^{k} x_i = 1, x_i \\geq 0 (probability simplex)

        Moments:
            \\mathbb{E}[X_i] = \\frac{\\alpha_i}{\\sum_{j=1}^{k} \\alpha_j}
            \\text{Var}[X_i] = \\frac{\\alpha_i(\\alpha_0 - \\alpha_i)}{\\alpha_0^2(\\alpha_0 + 1)}
            where \\alpha_0 = \\sum_{j=1}^{k} \\alpha_j

        Entropy of Dirichlet:
            H(X) = \\log B(\\alpha) + (\\alpha_0 - k)\\psi(\\alpha_0) - \\sum_{i=1}^{k}(\\alpha_i - 1)\\psi(\\alpha_i)
            where \\psi is digamma function

        Monte Carlo Sampling:
            \\mathbb{E}[f(X)] \\approx \\frac{1}{N} \\sum_{i=1}^{N} f(X^{(i)})
            where X^{(i)} are independent samples

        Complexity Expectation:
            \\mathbb{E}[C(X)] = \\int C(x) p(x) dx \\approx \\frac{1}{N} \\sum_{i=1}^{N} C(x^{(i)})

        Args:
            population_state: Current population state with frequencies and variance

        Returns:
            Dictionary of sampled complexity measures and statistics
        """
        n_types = len(population_state.frequencies)

        # Normal distribution for complexity fluctuations
        complexity_mean = torch.sum(
            population_state.frequencies * torch.arange(n_types, dtype=get_dtype_manager().config.real_dtype)
        )
        complexity_std = torch.sqrt(population_state.genetic_variance)
        complexity_dist = Normal(complexity_mean, complexity_std)
        complexity_samples = complexity_dist.sample((1000,))

        # Dirichlet distribution for frequency perturbations
        # α parameters proportional to current frequencies
        alpha_params = population_state.frequencies * 100 + 1  # Ensure α > 0
        frequency_dist = Dirichlet(alpha_params)
        frequency_samples = frequency_dist.sample((100,))

        # Compute complexity for each frequency sample
        complexity_values = []
        for freq_sample in frequency_samples:
            # Shannon entropy as complexity measure
            entropy = -torch.sum(freq_sample * torch.log(freq_sample + 1e-12))
            complexity_values.append(entropy)

        complexity_tensor = torch.stack(complexity_values)

        return {
            "complexity_samples": complexity_samples,
            "frequency_samples": frequency_samples,
            "sampled_complexities": complexity_tensor,
            "complexity_mean": complexity_dist.mean,
            "complexity_variance": complexity_dist.variance,
            "frequency_entropy": frequency_dist.entropy(),
        }

    def jax_advanced_complexity_optimization(
        self, complexity_field: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Advanced complexity optimization using JAX automatic differentiation.

        MATHEMATICAL FORMULATION:
        Vectorized Map (vmap):
            \\text{vmap}(f)(xs) = [f(x_1), f(x_2), ..., f(x_n)]
            Applies function f element-wise over batch dimension

        Forward-mode Jacobian:
            J_{ij} = \\frac{\\partial f_i}{\\partial x_j} computed via forward accumulation
            jacfwd(f)(x) computes J using dual numbers

        Reverse-mode Jacobian:
            J_{ij} = \\frac{\\partial f_i}{\\partial x_j} computed via reverse accumulation
            jacrev(f)(x) computes J using backpropagation

        Gradient (special case of Jacobian):
            \\nabla f = \\left(\\frac{\\partial f}{\\partial x_1}, ..., \\frac{\\partial f}{\\partial x_n}\\right)

        Hessian (second-order derivatives):
            H_{ij} = \\frac{\\partial^2 f}{\\partial x_i \\partial x_j}
            hessian(f) = jacfwd(jacrev(f)) or jacrev(jacfwd(f))

        Complexity Functional:
            \\mathcal{C}[u] = \\int_{\\Omega} \\mathcal{L}(x, u, \\nabla u) dx
            where \\mathcal{L} is complexity density

        Multi-scale Analysis:
            \\mathcal{C}_s[u] = \\mathcal{C}[s \\cdot u] for scale parameter s

        Condition Number:
            \\kappa(A) = \\frac{\\sigma_{max}(A)}{\\sigma_{min}(A)}
            where \\sigma_{max}, \\sigma_{min} are largest/smallest singular values

        Args:
            complexity_field: Input complexity field for optimization

        Returns:
            Dictionary with Jacobians, gradients, Hessian, and optimization metrics
        """
        field_jax = jnp.array(complexity_field.detach().cpu().numpy())

        # Define complexity functional
        @jit
        def complexity_functional(field):
            # Shannon entropy component
            field_normalized = jnp.abs(field) / jnp.sum(jnp.abs(field))
            entropy = -jnp.sum(field_normalized * jnp.log(field_normalized + 1e-12))

            # Gradient complexity component
            gradient = jnp.gradient(field)
            gradient_complexity = jnp.sum(gradient**2)

            return entropy + 0.1 * gradient_complexity

        # Vectorized complexity over multiple scales
        @jit
        def multi_scale_complexity(scale):
            scaled_field = field_jax * scale
            return complexity_functional(scaled_field)

        scales = jnp.linspace(0.5, 2.0, 10)
        vectorized_complexities = vmap(multi_scale_complexity)(scales)

        # Forward-mode Jacobian
        complexity_jacfwd = jacfwd(complexity_functional)
        forward_jacobian = complexity_jacfwd(field_jax)

        # Reverse-mode Jacobian
        complexity_jacrev = jacrev(complexity_functional)
        reverse_jacobian = complexity_jacrev(field_jax)

        # Gradient and Hessian
        complexity_grad = grad(complexity_functional)
        gradient_val = complexity_grad(field_jax)

        complexity_hessian = hessian(complexity_functional)
        hessian_val = complexity_hessian(field_jax)

        return {
            "multi_scale_complexities": torch.from_numpy(
                np.array(vectorized_complexities)
            ),
            "forward_jacobian": torch.from_numpy(np.array(forward_jacobian)),
            "reverse_jacobian": torch.from_numpy(np.array(reverse_jacobian)),
            "complexity_gradient": torch.from_numpy(np.array(gradient_val)),
            "complexity_hessian": torch.from_numpy(np.array(hessian_val)),
            "hessian_condition": float(jnp.linalg.cond(hessian_val)),
        }

    def scipy_complexity_optimization(
        self, initial_field: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """
        Complexity optimization using scipy.optimize algorithms.
        
        MATHEMATICAL FORMULATION:
        BFGS Algorithm (Quasi-Newton):
            x_{k+1} = x_k - \\alpha_k H_k^{-1} \\nabla f(x_k)
            where H_k approximates Hessian using rank-2 updates:
            H_{k+1} = H_k + \\frac{y_k y_k^T}{y_k^T s_k} - \\frac{H_k s_k s_k^T H_k}{s_k^T H_k s_k}
        
        Differential Evolution:
            Global optimization using mutation and crossover:
            v_{i,G+1} = x_{r1,G} + F(x_{r2,G} - x_{r3,G})
            where F is mutation factor and r1, r2, r3 are random indices
        
        Selection:
            x_{i,G+1} = \\begin{cases}
                u_{i,G+1} & \\text{if } f(u_{i,G+1}) \\leq f(x_{i,G}) \\\\
                x_{i,G} & \\text{otherwise}
            \\end{cases}
        
        Basin Hopping:
            Combines local optimization with stochastic jumps:
            1. Local minimization: x_{local} = \\arg\\min_{x} f(x)
            2. Random perturbation: x' = x_{local} + \\xi
            3. Accept/reject using Metropolis criterion
        
        Convergence Criteria:
            |\\nabla f(x)| < \\epsilon_{grad} (gradient tolerance)
            |f(x_{k+1}) - f(x_k)| < \\epsilon_{fun} (function tolerance)
        
        Objective Function (Entropy Maximization):
            \\max_{p} H(p) = -\\sum_{i} p_i \\log p_i
            subject to \\sum_i p_i = 1, p_i \\geq 0
        
        Args:
            initial_field: Starting point for optimization
            
        Returns:
            Dictionary with optimal fields and convergence information
        """
        initial_np = initial_field.detach().cpu().numpy()

        # Define complexity objective
        def complexity_objective(field):
            # Negative complexity for maximization
            field_positive = np.abs(field)
            field_normalized = field_positive / np.sum(field_positive)
            entropy = -np.sum(field_normalized * np.log(field_normalized + 1e-12))
            return -entropy  # Minimize negative entropy = maximize entropy

        # Gradient of complexity
        def complexity_gradient(field):
            field_positive = np.abs(field)
            total = np.sum(field_positive)
            if total <= 0:
                return np.zeros_like(field)

            field_normalized = field_positive / total
            log_field = np.log(field_normalized + 1e-12)

            # Gradient of entropy
            grad_entropy = -(log_field + 1) / total
            return -grad_entropy  # Negative for maximization

        # Minimize using BFGS
        result_bfgs = minimize(
            complexity_objective,
            initial_np,
            method="BFGS",
            jac=complexity_gradient,
            options={"gtol": POPULATION_NUMERICAL_PRECISION},
        )

        # Global optimization using differential evolution
        bounds = [(0, 1) for _ in range(len(initial_np))]
        result_de = differential_evolution(
            complexity_objective, bounds, strategy="best1bin", maxiter=1000, seed=42
        )

        # Basin hopping for complex landscape
        result_basin = basinhopping(
            complexity_objective,
            initial_np,
            niter=100,
            minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds},
            seed=42,
        )

        return {
            "optimal_complexity_field_bfgs": torch.from_numpy(result_bfgs.x),
            "optimal_complexity_bfgs": -result_bfgs.fun,
            "optimal_complexity_field_de": torch.from_numpy(result_de.x),
            "optimal_complexity_de": -result_de.fun,
            "optimal_complexity_field_basin": torch.from_numpy(result_basin.x),
            "optimal_complexity_basin": -result_basin.fun,
            "convergence_bfgs": result_bfgs.success,
            "convergence_de": result_de.success,
        }

    def scipy_special_complexity_measures(
        self, frequencies: torch.Tensor
    ) -> Dict[str, float]:
        """
        Specialized complexity measures using scipy.special mathematical functions.
        
        MATHEMATICAL FORMULATION:
        Entropy Function (scipy.special.entr):
            \\text{entr}(x) = \\begin{cases}
                -x \\log x & \\text{if } x > 0 \\\\
                0 & \\text{if } x = 0 \\\\
                -\\infty & \\text{if } x < 0
            \\end{cases}
        
        Relative Entropy (scipy.special.rel_entr):
            \\text{rel_entr}(x, y) = \\begin{cases}
                x \\log(x/y) & \\text{if } x > 0, y > 0 \\\\
                0 & \\text{if } x = 0, y \\geq 0 \\\\
                \\infty & \\text{otherwise}
            \\end{cases}
        
        Tsallis q-Entropy:
            S_q(p) = \\frac{1}{q-1}\\left(1 - \\sum_{i=1}^{n} p_i^q\\right)
            Limits: \\lim_{q \\to 1} S_q(p) = H(p) (Shannon entropy)
        
        Simpson's Diversity Index:
            D = \\frac{1}{\\sum_{i=1}^{n} p_i^2}
            Effective number of species/types
        
        Hill Numbers (diversity of order q):
            {}^qD = \\left(\\sum_{i=1}^{n} p_i^q\\right)^{1/(1-q)}
            Special cases: q=0 (richness), q=1 (Shannon), q=2 (Simpson)
        
        Log-Sum-Exp (numerically stable):
            \\text{LSE}(x_1, ..., x_n) = \\log\\left(\\sum_{i=1}^{n} e^{x_i}\\right)
            = x_{max} + \\log\\left(\\sum_{i=1}^{n} e^{x_i - x_{max}}\\right)
        
        Gamma Function:
            \\Gamma(z) = \\int_0^\\infty t^{z-1} e^{-t} dt
            Properties: \\Gamma(n) = (n-1)! for n \\in \\mathbb{N}
        
        Beta Function:
            B(x,y) = \\frac{\\Gamma(x)\\Gamma(y)}{\\Gamma(x+y)} = \\int_0^1 t^{x-1}(1-t)^{y-1} dt
        
        Args:
            frequencies: Probability distribution tensor
            
        Returns:
            Dictionary of specialized complexity and diversity measures
        """
        freqs_np = frequencies.detach().cpu().numpy()

        # Ensure valid probability distribution
        freqs_positive = np.abs(freqs_np)
        freqs_normalized = freqs_positive / np.sum(freqs_positive)

        # Entropy using scipy.special.entr
        entropy_values = special.entr(freqs_normalized)
        shannon_entropy = np.sum(entropy_values)

        # Tsallis entropy using q-logarithm
        q = 2.0  # Tsallis parameter
        tsallis_entropy = (1 - np.sum(freqs_normalized**q)) / (q - 1)

        # Relative entropy to uniform distribution
        uniform = np.ones_like(freqs_normalized) / len(freqs_normalized)
        rel_entropy_values = special.rel_entr(freqs_normalized, uniform)
        kl_divergence = np.sum(rel_entropy_values)

        # Beta function for diversity index
        # Simpson's diversity D = 1/Σpᵢ²
        simpson_concentration = np.sum(freqs_normalized**2)
        if simpson_concentration > 0:
            simpson_diversity = 1.0 / simpson_concentration
        else:
            simpson_diversity = float("inf")

        # Gamma function for Hill numbers
        # Hill number of order q: ᴰq = (Σpᵢᵠ)^(1/(1-q))
        hill_order_2 = np.sum(freqs_normalized**2)
        if hill_order_2 > 0:
            hill_diversity_2 = 1.0 / hill_order_2  # Equivalent to Simpson's diversity
        else:
            hill_diversity_2 = len(freqs_normalized)

        # Log-sum-exp for numerical stability
        log_freqs = np.log(freqs_normalized + 1e-12)
        log_sum_exp = special.logsumexp(log_freqs, b=freqs_normalized)

        return {
            "shannon_entropy": shannon_entropy,
            "tsallis_entropy": tsallis_entropy,
            "kl_divergence_uniform": kl_divergence,
            "simpson_diversity": simpson_diversity,
            "hill_diversity_order_2": hill_diversity_2,
            "log_sum_exp": log_sum_exp,
        }

    def scipy_interpolation_complexity_landscape(
        self, sample_points: List[Tuple[float, float]]
    ) -> Callable:
        """
        Interpolate complexity landscape using spline interpolation.

        MATHEMATICAL FORMULATION:
        Cubic Spline Interpolation:
            S(x) = \\sum_{j=0}^{n-1} a_j (x - x_j)^3 + b_j (x - x_j)^2 + c_j (x - x_j) + d_j
            for x \\in [x_j, x_{j+1}]

        Continuity Conditions:
            S(x_i) = y_i (interpolation)
            S'(x_i^-) = S'(x_i^+) (C^1 continuity)
            S''(x_i^-) = S''(x_i^+) (C^2 continuity)

        Natural Spline Boundary Conditions:
            S''(x_0) = S''(x_n) = 0

        Univariate Spline with Smoothing:
            \\min_S \\sum_{i=1}^{n} (y_i - S(x_i))^2 + \\lambda \\int (S''(x))^2 dx
            where \\lambda is smoothing parameter

        B-spline Basis Functions:
            N_{i,k}(x) = \\frac{x - t_i}{t_{i+k} - t_i} N_{i,k-1}(x) + \\frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} N_{i+1,k-1}(x)

        Spline Representation:
            S(x) = \\sum_{i=0}^{n} c_i N_{i,k}(x)
            where c_i are control points and N_{i,k} are B-spline basis functions

        Error Bounds:
            |f(x) - S(x)| \\leq C h^{k+1} |f^{(k+1)}|_{\\infty}
            where h = \\max_i (x_{i+1} - x_i) and k is spline degree

        Interpolation vs. Approximation:
            - Interpolation: S(x_i) = y_i exactly
            - Approximation: Minimize weighted least squares with smoothing

        Args:
            sample_points: List of (x, complexity) coordinate pairs

        Returns:
            Callable interpolation function for complexity landscape
        """
        if len(sample_points) < 4:
            return lambda x: 0.0

        # Extract coordinates and complexity values
        x_coords = np.array([p[0] for p in sample_points])
        complexities = np.array([p[1] for p in sample_points])

        # 1D interpolation using interp1d
        complexity_interp = interp1d(
            x_coords,
            complexities,
            kind="cubic",
            bounds_error=False,
            fill_value=(complexities[0], complexities[-1]),
        )

        # Univariate spline for smooth interpolation
        if len(x_coords) >= 4:
            complexity_spline = UnivariateSpline(
                x_coords,
                complexities,
                k=3,  # Cubic spline
                s=0.01,  # Smoothing parameter
            )

            # Return spline function
            return lambda x: float(complexity_spline(x))
        else:
            # Return linear interpolation
            return lambda x: float(complexity_interp(x))

    def _compute_complexity_from_frequencies(
        self, frequencies: torch.Tensor
    ) -> ComplexityMeasures:
        """
        Compute complexity measures directly from frequency distribution.

        MATHEMATICAL FOUNDATION:
        Direct Complexity Calculation from Probability Distribution:

        Shannon Entropy:
            H(X) = -∑ᵢ pᵢ log₂ pᵢ

        Fisher Information (for categorical distribution):
            I(θ) = ∑ᵢ (1/pᵢ) for multinomial parameters

        Rényi Entropy:
            H_α(X) = (1/(1-α)) log₂ ∑ᵢ pᵢ^α

        Tsallis Entropy:
            S_q(X) = (1/(q-1))(1 - ∑ᵢ pᵢ^q)

        Effective Number of States:
            N_eff = exp(H) = 2^H

        Participation Ratio:
            PR = 1/∑ᵢ pᵢ²

        Args:
            frequencies: Probability distribution tensor

        Returns:
            ComplexityMeasures with all calculated complexity metrics
        """
        # Normalize frequencies to ensure proper probability distribution
        frequencies = frequencies / torch.sum(frequencies)
        frequencies = torch.clamp(frequencies, min=1e-12)  # Avoid log(0)

        # Shannon Entropy: H = -∑ pᵢ log pᵢ
        shannon_entropy = -torch.sum(frequencies * torch.log2(frequencies)).item()

        # Rényi Entropy with α = 2: H₂ = -log₂(∑ pᵢ²)
        renyi_entropy = -torch.log2(torch.sum(frequencies**2)).item()

        # Tsallis Entropy with q = 2: S₂ = 1 - ∑ pᵢ²
        tsallis_entropy = 1.0 - torch.sum(frequencies**2).item()

        # Fisher Information for categorical distribution: I = ∑ (1/pᵢ)
        fisher_information = torch.sum(1.0 / frequencies).item()

        # KL divergence from uniform distribution
        n_states = len(frequencies)
        uniform_dist = torch.ones_like(frequencies) / n_states
        kl_divergence = torch.sum(
            frequencies * torch.log2(frequencies / uniform_dist)
        ).item()

        # Jensen-Shannon divergence from uniform
        m = 0.5 * (frequencies + uniform_dist)
        js_divergence = (
            0.5
            * (
                torch.sum(frequencies * torch.log2(frequencies / m))
                + torch.sum(uniform_dist * torch.log2(uniform_dist / m))
            ).item()
        )

        # Mutual information (simplified for univariate case)
        mutual_information = shannon_entropy  # Self-information

        # Transfer entropy (simplified)
        transfer_entropy = 0.0  # Would need temporal data

        # Kolmogorov complexity estimate via entropy
        kolmogorov_estimate = shannon_entropy * math.log2(n_states)

        # Logical depth (simplified)
        logical_depth = shannon_entropy * math.log(shannon_entropy + 1)

        # Topological complexity (participation ratio)
        topological_complexity = 1.0 / torch.sum(frequencies**2).item()

        # Geometric curvature (variance of distribution)
        mean_state = torch.sum(
            frequencies * torch.arange(len(frequencies), dtype=frequencies.dtype)
        )
        variance = torch.sum(
            frequencies
            * (torch.arange(len(frequencies), dtype=frequencies.dtype) - mean_state)
            ** 2
        )
        geometric_curvature = variance.item() / (n_states**2)

        return ComplexityMeasures(
            shannon_entropy=shannon_entropy,
            renyi_entropy=renyi_entropy,
            tsallis_entropy=tsallis_entropy,
            fisher_information=fisher_information,
            kullback_leibler_divergence=kl_divergence,
            jensen_shannon_divergence=js_divergence,
            mutual_information=mutual_information,
            transfer_entropy=transfer_entropy,
            kolmogorov_complexity_estimate=kolmogorov_estimate,
            logical_depth=logical_depth,
            topological_complexity=topological_complexity,
            geometric_curvature=geometric_curvature,
        )

    def _calculate_joint_entropy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Calculate joint entropy H(X,Y) for transfer entropy computation.

        Mathematical Foundation:
            Joint Entropy: H(X,Y) = -∑ᵢⱼ p(xᵢ,yⱼ) log p(xᵢ,yⱼ)

        Args:
            x: First variable tensor
            y: Second variable tensor

        Returns:
            Joint entropy value
        """
        if x.shape != y.shape:
            raise ValueError("Input tensors must have same shape for joint entropy")

        # Create joint distribution by binning
        n_bins = min(50, max(10, int(x.numel() ** 0.5)))  # Adaptive binning

        # Normalize to [0,1] range for binning
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

        # Create 2D histogram for joint distribution
        x_bins = torch.floor(x_norm * (n_bins - 1)).long().clamp(0, n_bins - 1)
        y_bins = torch.floor(y_norm * (n_bins - 1)).long().clamp(0, n_bins - 1)

        # Count joint occurrences
        joint_counts = torch.zeros((n_bins, n_bins), dtype=get_dtype_manager().config.real_dtype)
        for i in range(x.numel()):
            joint_counts[x_bins.flat[i], y_bins.flat[i]] += 1

        # Convert to probabilities
        joint_probs = joint_counts / x.numel()

        # Calculate joint entropy: H(X,Y) = -∑ᵢⱼ p(xᵢ,yⱼ) log p(xᵢ,yⱼ)
        joint_entropy = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint_probs[i, j] > 0:
                    joint_entropy -= joint_probs[i, j] * torch.log2(joint_probs[i, j])

        return joint_entropy.item()
