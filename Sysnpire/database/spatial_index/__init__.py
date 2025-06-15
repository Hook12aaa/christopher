"""
Spatial Index - Track Positioning

Spatial organization using Hilbert curves for O(log n) spatial locality.
Maps field positions to efficient storage locations.

Components:
- hilbert_encoder: Encode 3D positions using Hilbert curves
- region_mapper: Map charges to field regions  
- duckdb_queries: SQL queries on Lance datasets with spatial support
"""

from .hilbert_encoder import HilbertEncoder
from .region_mapper import RegionMapper
from .duckdb_queries import DuckDBQueries

__all__ = [
    'HilbertEncoder',
    'RegionMapper',
    'DuckDBQueries'
]