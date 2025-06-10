"""
Ingestion Module - Text to Embedding Pipeline

This module handles the conversion of raw text into semantic embeddings
that serve as the foundation for conceptual charge generation.

Components:
- BGEIngestion: Main ingestion class for text processing
- SemanticEmbedding: Simplified embedding representation
"""

from .bge_ingestion import BGEIngestion
from .semantic_embedding import SemanticEmbedding

__all__ = ['BGEIngestion', 'SemanticEmbedding']