"""
Query Interface - Needle Reading

Query the manifold to find meaning patterns.
Provides interfaces for navigating and sampling the field.

Components:
- field_navigator: Navigate field patterns to find meaning
- pattern_search: Search for specific field patterns
- duckdb_bridge: SQL interface to Lance tensor data
"""

from .field_navigator import FieldNavigator
from .pattern_search import PatternSearch
from .duckdb_bridge import DuckDBBridge

__all__ = [
    'FieldNavigator',
    'PatternSearch',
    'DuckDBBridge'
]