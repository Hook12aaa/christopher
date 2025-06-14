"""
Spectral Analysis Utilities

Advanced spectral analysis functions for frequency domain operations
in field theory applications. Supports phase integration and heat kernel
evolution calculations essential for Q(τ, C, s) conceptual charges.
"""

from .frequency_analysis import FrequencyAnalyzer
from .heat_kernel_processor import HeatKernelEvolutionEngine

__all__ = ['FrequencyAnalyzer', 'HeatKernelEvolutionEngine']