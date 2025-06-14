"""
Ingestion Module - Text to Embedding Pipeline with Semantic Space Analysis

This module focuses on the initial part of creating the universe (product manifold \mathcal{M})
by allowing illustration over each embedding inside the model 



"""

from .bge_ingestion import BGEIngestion
from .mpnet_ingestion import MPNetIngestion


__all__ = ['BGEIngestion', 'MPNetIngestion']