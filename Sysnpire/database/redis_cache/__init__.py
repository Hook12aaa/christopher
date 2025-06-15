"""
Redis Cache - Needle Speed

Fast access caching for <100ms query responses.
Stores hot regions and frequently accessed field patterns.

Components:
- hot_regions: Cache frequently accessed field regions
- field_patterns: Cache computed field patterns
- spectral_cache: Pre-computed spectral bases for O(n) evolution
"""

from .hot_regions import HotRegionsCache
from .field_patterns import FieldPatternsCache
from .spectral_cache import SpectralCache

__all__ = [
    'HotRegionsCache',
    'FieldPatternsCache',
    'SpectralCache'
]