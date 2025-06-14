"""
Semantic Dimension Processing - Field Pool and Pipeline Components

This module contains the processing pipeline components for semantic dimension
operations, including field pools, batch processing, and pipeline management.

Components:
- field_pool.py: Main field pool for embedding → field transformation pipeline
- batch_processor.py: Batch processing utilities (future)
- pipeline_manager.py: Pipeline orchestration (future)
"""

from .field_pool import SemanticFieldPool, FieldPoolEntry, create_field_pool_from_manifold

__all__ = ['SemanticFieldPool', 'FieldPoolEntry', 'create_field_pool_from_manifold']