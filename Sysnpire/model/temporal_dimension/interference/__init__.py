"""
Temporal Field Interference Computation

Modular system for computing field interference between temporal biographies
using data-driven optimization strategies.
"""

from .sparsity_analyzer import SparsityAnalyzer, SparsityAnalysis
from .chunked_computer import ChunkedComputer

__all__ = ['SparsityAnalyzer', 'SparsityAnalysis', 'ChunkedComputer']