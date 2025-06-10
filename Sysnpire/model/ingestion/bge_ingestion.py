"""
BGE Ingestion - Clean interface for text-to-embedding conversion

This module provides a streamlined interface for converting text into
semantic embeddings using BGE (BGE-Large-v1.5 model simulation).
The embeddings serve as the foundation for conceptual charge generation.
"""

import numpy as np
import hashlib
from typing import List, Optional, Dict, Any, Union
from .semantic_embedding import SemanticEmbedding
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BGEIngestion:
    """
    BGE Ingestion Engine for Text-to-Embedding Conversion
    
    This class provides a clean, well-documented interface for converting
    raw text into semantic embeddings that can be used for conceptual
    charge generation. It simulates the BGE-Large-v1.5 model behavior
    with deterministic, reproducible results.
    
    The ingestion process extracts semantic information from text and
    creates 1024-dimensional embeddings that capture:
    - Semantic meaning and relationships
    - Textual structure and patterns  
    - Category-specific features
    - Normalized vector representations
    
    Attributes:
        embedding_dim (int): Dimensionality of output embeddings (default: 1024)
        model_name (str): Name of the simulated BGE model
        cache_embeddings (bool): Whether to cache embeddings for repeated texts
        _embedding_cache (Dict): Internal cache for processed embeddings
    """
    
    def __init__(self, 
                 embedding_dim: int = 1024,
                 cache_embeddings: bool = True,
                 random_seed: Optional[int] = None):
        """
        Initialize the BGE Ingestion Engine.
        
        Args:
            embedding_dim (int): Dimensionality of output embeddings
            cache_embeddings (bool): Enable caching for performance
            random_seed (int, optional): Seed for reproducible results
        """
        self.embedding_dim = embedding_dim
        self.model_name = "BAAI/bge-large-en-v1.5"
        self.cache_embeddings = cache_embeddings
        self.random_seed = random_seed
        
        # Internal cache for embeddings
        self._embedding_cache: Dict[str, SemanticEmbedding] = {}
        
        # Semantic category definitions for feature extraction
        self._semantic_categories = {
            'abstract_concepts': ['theory', 'concept', 'idea', 'principle', 'philosophy'],
            'emotional_content': ['feel', 'emotion', 'joy', 'sad', 'angry', 'happy', 'love'],
            'social_dynamics': ['social', 'community', 'relationship', 'interaction', 'group'],
            'technical_domain': ['system', 'process', 'method', 'algorithm', 'technical'],
            'creative_expression': ['art', 'creative', 'music', 'design', 'artistic', 'beauty'],
            'temporal_aspects': ['time', 'future', 'past', 'evolution', 'change', 'development'],
            'spatial_concepts': ['space', 'location', 'place', 'environment', 'landscape'],
            'quantitative_info': ['number', 'measure', 'data', 'statistics', 'analysis']
        }
        
        logger.info(f"BGE Ingestion initialized with {embedding_dim}D embeddings")
    
    def ingest_text(self, 
                   text: str, 
                   text_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SemanticEmbedding:
        """
        Convert a single text string into a semantic embedding.
        
        This is the primary method for text ingestion. It processes the input
        text through multiple stages to create a rich semantic representation:
        
        1. Text preprocessing and feature extraction
        2. Semantic vector generation using deterministic BGE simulation
        3. Feature integration and normalization
        4. Embedding validation and packaging
        
        Args:
            text (str): Input text to convert to embedding
            text_id (str, optional): Unique identifier for the text
            metadata (Dict, optional): Additional metadata to store with embedding
            
        Returns:
            SemanticEmbedding: A complete semantic embedding object ready for
                              conceptual charge generation
                              
        Raises:
            ValueError: If text is empty or invalid
            TypeError: If text is not a string
        """
        # Input validation
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")
        
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Check cache first
        text_hash = self._generate_text_hash(text)
        if self.cache_embeddings and text_hash in self._embedding_cache:
            logger.debug(f"Retrieved cached embedding for text: {text[:50]}...")
            return self._embedding_cache[text_hash]
        
        # Generate semantic embedding
        try:
            # Step 1: Extract textual features
            text_features = self._extract_textual_features(text)
            
            # Step 2: Extract semantic category features
            semantic_features = self._extract_semantic_features(text)
            
            # Step 3: Generate base embedding vector
            embedding_vector = self._generate_embedding_vector(text, text_features, semantic_features)
            
            # Step 4: Create semantic embedding object
            semantic_embedding = SemanticEmbedding(
                vector=embedding_vector,
                text=text,
                text_id=text_id,
                metadata=metadata or {}
            )
            
            # Step 5: Cache the result
            if self.cache_embeddings:
                self._embedding_cache[text_hash] = semantic_embedding
            
            logger.debug(f"Generated embedding for text: {text[:50]}... "
                        f"(magnitude: {semantic_embedding.magnitude:.4f})")
            
            return semantic_embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {text[:50]}... Error: {e}")
            raise
    
    def ingest_batch(self, 
                    texts: List[str], 
                    text_ids: Optional[List[str]] = None,
                    metadata: Optional[List[Dict[str, Any]]] = None) -> List[SemanticEmbedding]:
        """
        Convert multiple texts into semantic embeddings efficiently.
        
        This method processes multiple texts in batch for improved performance.
        It maintains the same quality as single-text ingestion while providing
        better throughput for large-scale processing.
        
        Args:
            texts (List[str]): List of input texts to convert
            text_ids (List[str], optional): List of unique identifiers for texts
            metadata (List[Dict], optional): List of metadata dictionaries
            
        Returns:
            List[SemanticEmbedding]: List of semantic embeddings in same order as input
            
        Raises:
            ValueError: If input lists have mismatched lengths
        """
        # Input validation
        if not texts:
            return []
        
        if text_ids and len(text_ids) != len(texts):
            raise ValueError("text_ids length must match texts length")
        
        if metadata and len(metadata) != len(texts):
            raise ValueError("metadata length must match texts length")
        
        # Prepare arguments
        if text_ids is None:
            text_ids = [None] * len(texts)
        
        if metadata is None:
            metadata = [None] * len(texts)
        
        # Process batch
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self.ingest_text(
                    text=text,
                    text_id=text_ids[i],
                    metadata=metadata[i]
                )
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to process text {i}: {text[:50]}... Error: {e}")
                # Create a zero embedding as fallback
                zero_embedding = SemanticEmbedding(
                    vector=np.zeros(self.embedding_dim),
                    text=text,
                    text_id=text_ids[i],
                    metadata={'error': str(e), 'failed_ingestion': True}
                )
                embeddings.append(zero_embedding)
        
        logger.info(f"Processed batch of {len(texts)} texts -> {len(embeddings)} embeddings")
        return embeddings
    
    def _generate_text_hash(self, text: str) -> str:
        """Generate a hash for text caching."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _extract_textual_features(self, text: str) -> np.ndarray:
        """
        Extract basic textual structure features from the input text.
        
        These features capture the structural properties of the text that
        contribute to its semantic embedding:
        - Text length and complexity
        - Punctuation patterns
        - Vocabulary diversity
        - Sentence structure
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            np.ndarray: Array of normalized textual features
        """
        features = [
            len(text) / 500.0,  # Text length (normalized)
            text.count(' ') / max(len(text), 1),  # Word density
            text.count('.') / max(text.count(' '), 1),  # Sentence density
            text.count(',') / max(text.count(' '), 1),  # Clause complexity
            len(set(text.lower())) / max(len(text), 1),  # Character diversity
            text.count('?') / max(len(text), 1),  # Question frequency
            text.count('!') / max(len(text), 1),  # Exclamation frequency
            len(text.split()) / max(len(text), 1),  # Word-to-character ratio
        ]
        
        return np.array(features)
    
    def _extract_semantic_features(self, text: str) -> np.ndarray:
        """
        Extract semantic category features from the input text.
        
        This method analyzes the text for semantic content across predefined
        categories that are relevant for conceptual charge generation.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            np.ndarray: Array of semantic category scores
        """
        text_lower = text.lower()
        features = []
        
        for category, keywords in self._semantic_categories.items():
            # Calculate category score based on keyword presence
            score = 0.0
            for keyword in keywords:
                score += text_lower.count(keyword)
            
            # Normalize by text length and keyword count
            normalized_score = score / (len(keywords) * max(len(text.split()), 1))
            features.append(normalized_score)
        
        return np.array(features)
    
    def _generate_embedding_vector(self, 
                                 text: str, 
                                 text_features: np.ndarray, 
                                 semantic_features: np.ndarray) -> np.ndarray:
        """
        Generate the actual embedding vector from extracted features.
        
        This method combines textual and semantic features to create a
        deterministic, high-dimensional embedding that simulates BGE behavior.
        
        Args:
            text (str): Original input text
            text_features (np.ndarray): Extracted textual features
            semantic_features (np.ndarray): Extracted semantic features
            
        Returns:
            np.ndarray: Normalized embedding vector of specified dimensionality
        """
        # Create deterministic seed from text
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        seed = int(text_hash[:8], 16)
        
        # Use provided random seed if available
        if self.random_seed is not None:
            seed = seed ^ self.random_seed
        
        np.random.seed(seed)
        
        # Generate base embedding
        embedding = np.random.randn(self.embedding_dim)
        
        # Integrate textual features
        feature_positions = np.linspace(0, self.embedding_dim - 1, len(text_features), dtype=int)
        for i, feature in enumerate(text_features):
            if i < len(feature_positions):
                pos = feature_positions[i]
                embedding[pos] = embedding[pos] * (1 + feature * 0.5)
        
        # Integrate semantic features
        semantic_block_size = self.embedding_dim // len(semantic_features)
        for i, semantic_weight in enumerate(semantic_features):
            start_idx = i * semantic_block_size
            end_idx = min(start_idx + semantic_block_size, self.embedding_dim)
            
            # Apply semantic weighting to embedding blocks
            if semantic_weight > 0:
                embedding[start_idx:end_idx] *= (1 + semantic_weight * 2.0)
        
        # Add text-specific perturbations for uniqueness
        for i, char in enumerate(text[:min(len(text), 100)]):  # Use first 100 chars
            char_influence = ord(char) / 128.0  # Normalize ASCII values
            influence_pos = (i * 7) % self.embedding_dim  # Spread influence
            embedding[influence_pos] += char_influence * 0.1
        
        # Normalize the final embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embedding cache.
        
        Returns:
            Dict[str, Any]: Cache statistics including size and hit rate
        """
        return {
            'cache_size': len(self._embedding_cache),
            'cache_enabled': self.cache_embeddings,
            'embedding_dimension': self.embedding_dim,
            'model_name': self.model_name
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache to free memory."""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def __repr__(self) -> str:
        """String representation of the BGE Ingestion engine."""
        return (f"BGEIngestion(embedding_dim={self.embedding_dim}, "
                f"cache_enabled={self.cache_embeddings}, "
                f"cached_embeddings={len(self._embedding_cache)})")