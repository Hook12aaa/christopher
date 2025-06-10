"""
Semantic Dimension - Field Generation (Section 3.1.2)

Transforms static embeddings into dynamic field-generating functions that actively
shape the semantic landscape through field effects.

Components:
- field_generators.py: S_τ(x) field function implementations
- embedding_reconstruction.py: Transform embeddings to field representations
- manifold_coupling.py: Semantic-manifold interaction mechanisms
"""

from .field_generators import SemanticFieldGenerator
from .embedding_reconstruction import EmbeddingReconstructor  
from .manifold_coupling import SemanticManifoldCoupler

__all__ = ['SemanticFieldGenerator', 'EmbeddingReconstructor', 'SemanticManifoldCoupler']