"""
Abstraction Layer - Vinyl Pressing & Needle Operations

This layer handles the transformation of ConceptualChargeObjects into optimized
formats for tensor storage. Think of it as the "vinyl pressing" process that
prepares charges for efficient storage and the "needle operations" that read them.

Components:
- charge_transformer: Process charges for optimal tensor representation
- field_calculator: Compute field interactions and dynamics
- manifold_processor: Handle manifold geometry operations
- collective_analyzer: Analyze collective response patterns
- intake_processor: Process incoming charges for storage pipeline
"""

# Import available components
from .intake_processor import IntakeProcessor
from .charge_transformer import ChargeTransformer, TensorTransformationConfig

# TODO: Import when implemented
# from .field_calculator import FieldCalculator
# from .manifold_processor import ManifoldProcessor
# from .collective_analyzer import CollectiveAnalyzer

__all__ = [
    'IntakeProcessor',
    'ChargeTransformer',
    'TensorTransformationConfig', 
    # 'FieldCalculator',
    # 'ManifoldProcessor',
    # 'CollectiveAnalyzer'
]