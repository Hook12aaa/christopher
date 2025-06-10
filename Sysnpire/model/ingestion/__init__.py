"""
Ingestion Module - Text to Embedding Pipeline with Semantic Space Analysis

This module handles the conversion of raw text into semantic embeddings
that serve as the foundation for conceptual charge generation. Includes
comprehensive semantic space analysis for universe conversion.

Components:
- BGEIngestion: BGE-Large-v1.5 simulation for text processing
- MPNetIngestion: all-mpnet-base-v2 simulation for social universe construction
- SemanticEmbedding: Simplified embedding representation with utility methods
- SemanticSpaceAnalyzer: Comprehensive embedding space analysis for field theory
"""

from .bge_ingestion import BGEIngestion
from .mpnet_ingestion import MPNetIngestion
from .semantic_embedding import SemanticEmbedding
from .semantic_space_analyzer import SemanticSpaceAnalyzer

__all__ = ['BGEIngestion', 'MPNetIngestion', 'SemanticEmbedding', 'SemanticSpaceAnalyzer']