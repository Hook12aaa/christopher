"""
Field Theory Optimization Utilities

High-performance numba-optimized functions for field theory calculations.
Shared across BGE and MPNet ingestion systems for mathematical consistency.
"""

from .similarity_calculations import SimilarityCalculator

__all__ = [
    'SimilarityCalculator'
]