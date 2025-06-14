"""
MPNet Ingestion - all-mpnet-base-v2 embedding generation for social universe construction

This module provides a specialized interface for converting text into
semantic embeddings using the all-mpnet-base-v2 model architecture.
Optimized for social construct analysis and conceptual charge generation
in multi-dimensional social universes.
"""

import numpy as np
import hashlib
from typing import List, Optional, Dict, Any, Union
from .semantic_embedding import SemanticEmbedding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MPNetIngestion:
    """
    MPNet Ingestion Engine for Social Universe Construction
    
    This class provides a specialized interface for converting raw text into
    semantic embeddings using the all-mpnet-base-v2 model simulation.
    Specifically designed for social universe construction with enhanced
    social construct detection and multi-dimensional semantic analysis.
    
    The ingestion process creates 768-dimensional embeddings optimized for:
    - Social construct identification and analysis
    - Power dynamics and relationship modeling
    - Collective behavior pattern recognition
    - Identity formation and cultural context extraction
    - Enhanced semantic space characteristics for field theory
    
    Attributes:
        embedding_dim (int): Fixed at 768 dimensions for MPNet compatibility
        model_name (str): Full name of the all-mpnet-base-v2 model
        cache_embeddings (bool): Whether to cache embeddings for repeated texts
        social_categories (Dict): Extended social construct categories
        _embedding_cache (Dict): Internal cache for processed embeddings
    """
    
    def __init__(self, 
                 cache_embeddings: bool = True,
                 random_seed: Optional[int] = None,
                 social_focus: bool = True):
        """
        Initialize the MPNet Ingestion Engine for social universe construction.
        
        Args:
            cache_embeddings (bool): Enable caching for performance optimization
            random_seed (int, optional): Seed for reproducible deterministic results
            social_focus (bool): Enable enhanced social construct analysis features
        """
        self.embedding_dim = 768
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.cache_embeddings = cache_embeddings
        self.random_seed = random_seed
        self.social_focus = social_focus
        
        self._embedding_cache: Dict[str, SemanticEmbedding] = {}
        
        self.social_categories = self._initialize_social_categories()
        
        logger.info(f"MPNet Ingestion initialized with {self.embedding_dim}D embeddings "
                   f"(social_focus={'enabled' if social_focus else 'disabled'})")
    
    def _initialize_social_categories(self) -> Dict[str, List[str]]:
        """
        Initialize comprehensive social construct categories for MPNet analysis.
        
        These categories are specifically designed for social universe construction
        and capture the multi-dimensional aspects of social constructs that are
        essential for field theory applications.
        
        Returns:
            Dict[str, List[str]]: Comprehensive social semantic categories
        """
        categories = {
            'social_constructs': [
                'culture', 'society', 'norm', 'value', 'belief', 'tradition',
                'custom', 'ritual', 'practice', 'institution', 'convention'
            ],
            'power_dynamics': [
                'authority', 'control', 'influence', 'hierarchy', 'leadership',
                'dominance', 'submission', 'governance', 'command', 'rule'
            ],
            'collective_behavior': [
                'movement', 'trend', 'consensus', 'collective', 'mass',
                'crowd', 'mob', 'solidarity', 'unity', 'cooperation'
            ],
            'identity_formation': [
                'identity', 'self', 'belonging', 'membership', 'role',
                'persona', 'character', 'reputation', 'status', 'position'
            ],
            'social_relationships': [
                'relationship', 'connection', 'bond', 'tie', 'link',
                'association', 'partnership', 'alliance', 'network', 'community'
            ],
            'emotional_resonance': [
                'empathy', 'compassion', 'sympathy', 'understanding', 'care',
                'concern', 'support', 'comfort', 'encouragement', 'validation'
            ],
            'conflict_resolution': [
                'conflict', 'resolution', 'negotiation', 'compromise', 'mediation',
                'arbitration', 'peace', 'harmony', 'reconciliation', 'agreement'
            ],
            'cultural_transmission': [
                'education', 'learning', 'teaching', 'knowledge', 'wisdom',
                'tradition', 'heritage', 'legacy', 'inheritance', 'passing'
            ],
            'social_change': [
                'change', 'transformation', 'evolution', 'revolution', 'reform',
                'progress', 'innovation', 'adaptation', 'development', 'growth'
            ],
            'communication_patterns': [
                'communication', 'language', 'speech', 'dialogue', 'conversation',
                'discourse', 'narrative', 'story', 'message', 'expression'
            ],
            'structural_elements': [
                'structure', 'system', 'organization', 'framework', 'pattern',
                'order', 'arrangement', 'design', 'architecture', 'foundation'
            ],
            'temporal_dynamics': [
                'time', 'duration', 'sequence', 'rhythm', 'cycle',
                'phase', 'stage', 'period', 'era', 'moment'
            ]
        }
        
        if not self.social_focus:
            return {k: v for k, v in list(categories.items())[:6]}
        
        return categories
    
    def ingest_text(self, 
                   text: str, 
                   text_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SemanticEmbedding:
        """
        Convert text into MPNet semantic embedding optimized for social analysis.
        
        This method processes input text through multiple stages specifically
        designed for social universe construction:
        
        1. Social construct feature extraction and analysis
        2. Multi-dimensional semantic vector generation using MPNet simulation
        3. Social relationship pattern identification
        4. Cultural and temporal context integration
        5. Field-theoretic embedding optimization
        
        Args:
            text (str): Input text to convert to social semantic embedding
            text_id (str, optional): Unique identifier for the text
            metadata (Dict, optional): Additional metadata for social context
            
        Returns:
            SemanticEmbedding: Social-optimized semantic embedding ready for
                              conceptual charge generation and universe placement
                              
        Raises:
            ValueError: If text is empty or invalid for social analysis
            TypeError: If text is not a string
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")
        
        if not text.strip():
            raise ValueError("Input text cannot be empty for social analysis")
        
        text_hash = self._generate_text_hash(text)
        if self.cache_embeddings and text_hash in self._embedding_cache:
            logger.debug(f"Retrieved cached MPNet embedding for: {text[:50]}...")
            return self._embedding_cache[text_hash]
        
        try:
            social_features = self._extract_social_features(text)
            linguistic_features = self._extract_linguistic_features(text)
            contextual_features = self._extract_contextual_features(text)
            
            embedding_vector = self._generate_mpnet_embedding(
                text, social_features, linguistic_features, contextual_features
            )
            
            enhanced_metadata = metadata or {}
            enhanced_metadata.update({
                'model_type': 'mpnet',
                'social_score': float(np.mean(social_features)),
                'linguistic_complexity': float(np.std(linguistic_features)),
                'contextual_richness': float(np.sum(contextual_features))
            })
            
            semantic_embedding = SemanticEmbedding(
                vector=embedding_vector,
                text=text,
                text_id=text_id,
                metadata=enhanced_metadata
            )
            
            if self.cache_embeddings:
                self._embedding_cache[text_hash] = semantic_embedding
            
            logger.debug(f"Generated MPNet embedding: magnitude={semantic_embedding.magnitude:.4f}, "
                        f"social_score={enhanced_metadata['social_score']:.4f}")
            
            return semantic_embedding
            
        except Exception as e:
            logger.error(f"MPNet embedding generation failed for: {text[:50]}... Error: {e}")
            raise
    
    def ingest_batch(self, 
                    texts: List[str], 
                    text_ids: Optional[List[str]] = None,
                    metadata: Optional[List[Dict[str, Any]]] = None) -> List[SemanticEmbedding]:
        """
        Convert multiple texts into MPNet embeddings with social optimization.
        
        Processes multiple texts efficiently while maintaining the quality of
        social construct analysis and semantic space characteristics needed
        for social universe construction.
        
        Args:
            texts (List[str]): List of input texts for social analysis
            text_ids (List[str], optional): List of unique identifiers
            metadata (List[Dict], optional): List of metadata dictionaries
            
        Returns:
            List[SemanticEmbedding]: List of social-optimized semantic embeddings
            
        Raises:
            ValueError: If input lists have mismatched lengths
        """
        if not texts:
            return []
        
        if text_ids and len(text_ids) != len(texts):
            raise ValueError("text_ids length must match texts length")
        
        if metadata and len(metadata) != len(texts):
            raise ValueError("metadata length must match texts length")
        
        if text_ids is None:
            text_ids = [None] * len(texts)
        
        if metadata is None:
            metadata = [None] * len(texts)
        
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
                logger.warning(f"Failed to process text {i} in batch: {text[:50]}... Error: {e}")
                fallback_embedding = SemanticEmbedding(
                    vector=np.zeros(self.embedding_dim),
                    text=text,
                    text_id=text_ids[i],
                    metadata={'error': str(e), 'failed_mpnet_ingestion': True}
                )
                embeddings.append(fallback_embedding)
        
        logger.info(f"MPNet batch processing: {len(texts)} texts -> {len(embeddings)} embeddings")
        return embeddings
    
    def _generate_text_hash(self, text: str) -> str:
        """Generate hash for MPNet-specific text caching."""
        mpnet_prefix = "mpnet_"
        return mpnet_prefix + hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _extract_social_features(self, text: str) -> np.ndarray:
        """
        Extract social construct features optimized for MPNet analysis.
        
        Analyzes text for social constructs, power dynamics, collective behavior,
        and other social universe characteristics that are essential for
        field-theoretic social modeling.
        
        Args:
            text (str): Input text for social feature extraction
            
        Returns:
            np.ndarray: Array of social construct feature scores
        """
        text_lower = text.lower()
        social_features = []
        
        for category, keywords in self.social_categories.items():
            category_score = 0.0
            total_keywords = len(keywords)
            
            for keyword in keywords:
                keyword_count = text_lower.count(keyword)
                category_score += keyword_count
            
            normalized_score = category_score / (total_keywords * max(len(text.split()), 1))
            social_features.append(normalized_score)
        
        return np.array(social_features)
    
    def _extract_linguistic_features(self, text: str) -> np.ndarray:
        """
        Extract linguistic complexity features for MPNet semantic analysis.
        
        Args:
            text (str): Input text for linguistic analysis
            
        Returns:
            np.ndarray: Array of linguistic complexity features
        """
        features = [
            len(text) / 1000.0,
            text.count(' ') / max(len(text), 1),
            text.count('.') / max(text.count(' '), 1),
            text.count(',') / max(text.count(' '), 1),
            text.count(';') / max(text.count(' '), 1),
            text.count(':') / max(text.count(' '), 1),
            len(set(text.lower())) / max(len(text), 1),
            text.count('?') / max(len(text), 1),
            text.count('!') / max(len(text), 1),
            len([w for w in text.split() if len(w) > 6]) / max(len(text.split()), 1)
        ]
        
        return np.array(features)
    
    def _extract_contextual_features(self, text: str) -> np.ndarray:
        """
        Extract contextual features for social universe placement.
        
        Args:
            text (str): Input text for contextual analysis
            
        Returns:
            np.ndarray: Array of contextual features
        """
        text_lower = text.lower()
        
        contextual_markers = {
            'temporal_markers': ['when', 'time', 'during', 'before', 'after', 'now', 'then'],
            'spatial_markers': ['where', 'here', 'there', 'place', 'location', 'space'],
            'causal_markers': ['because', 'cause', 'reason', 'why', 'therefore', 'thus'],
            'modal_markers': ['might', 'could', 'should', 'would', 'may', 'can', 'must'],
            'emotional_markers': ['feel', 'emotion', 'mood', 'sentiment', 'feeling'],
            'social_markers': ['we', 'us', 'they', 'group', 'together', 'community']
        }
        
        contextual_features = []
        for category, markers in contextual_markers.items():
            score = sum(text_lower.count(marker) for marker in markers)
            normalized_score = score / max(len(text.split()), 1)
            contextual_features.append(normalized_score)
        
        return np.array(contextual_features)
    
    def _generate_mpnet_embedding(self, 
                                 text: str,
                                 social_features: np.ndarray,
                                 linguistic_features: np.ndarray,
                                 contextual_features: np.ndarray) -> np.ndarray:
        """
        Generate MPNet-style embedding vector with social optimization.
        
        Creates a 768-dimensional embedding that simulates all-mpnet-base-v2
        behavior while incorporating social construct analysis for enhanced
        semantic space characteristics.
        
        Args:
            text (str): Original input text
            social_features (np.ndarray): Extracted social construct features
            linguistic_features (np.ndarray): Extracted linguistic features
            contextual_features (np.ndarray): Extracted contextual features
            
        Returns:
            np.ndarray: Normalized 768-dimensional MPNet embedding vector
        """
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        seed = int(text_hash[:8], 16)
        
        if self.random_seed is not None:
            seed = seed ^ self.random_seed
        
        np.random.seed(seed)
        
        base_embedding = np.random.randn(self.embedding_dim)
        
        social_block_size = self.embedding_dim // len(social_features)
        for i, social_weight in enumerate(social_features):
            start_idx = i * social_block_size
            end_idx = min(start_idx + social_block_size, self.embedding_dim)
            
            if social_weight > 0:
                enhancement_factor = 1 + (social_weight * 3.0)
                base_embedding[start_idx:end_idx] *= enhancement_factor
        
        linguistic_positions = np.linspace(0, self.embedding_dim - 1, 
                                         len(linguistic_features), dtype=int)
        for i, feature in enumerate(linguistic_features):
            if i < len(linguistic_positions):
                pos = linguistic_positions[i]
                base_embedding[pos] += feature * 0.8
        
        contextual_stride = self.embedding_dim // (len(contextual_features) * 4)
        for i, context_score in enumerate(contextual_features):
            for j in range(4):
                pos = (i * 4 + j) * contextual_stride
                if pos < self.embedding_dim:
                    base_embedding[pos] *= (1 + context_score * 0.6)
        
        char_influences = np.array([ord(c) for c in text[:min(len(text), 200)]])
        char_positions = np.linspace(0, self.embedding_dim - 1, 
                                   len(char_influences), dtype=int)
        for i, char_val in enumerate(char_influences):
            if i < len(char_positions):
                pos = char_positions[i]
                base_embedding[pos] += (char_val / 128.0) * 0.2
        
        final_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        return final_embedding
    
    def get_social_analysis_stats(self) -> Dict[str, Any]:
        """
        Get statistics about social construct analysis capabilities.
        
        Returns:
            Dict[str, Any]: Statistics about social analysis features
        """
        return {
            'social_categories': len(self.social_categories),
            'total_social_keywords': sum(len(keywords) for keywords in self.social_categories.values()),
            'social_focus_enabled': self.social_focus,
            'embedding_dimension': self.embedding_dim,
            'cache_size': len(self._embedding_cache),
            'model_name': self.model_name
        }
    
    def compare_social_content(self, 
                              embedding1: SemanticEmbedding, 
                              embedding2: SemanticEmbedding) -> Dict[str, float]:
        """
        Compare social content between two MPNet embeddings.
        
        Args:
            embedding1 (SemanticEmbedding): First embedding for comparison
            embedding2 (SemanticEmbedding): Second embedding for comparison
            
        Returns:
            Dict[str, float]: Social comparison metrics
        """
        social_similarity = embedding1.similarity(embedding2)
        
        meta1 = embedding1.metadata or {}
        meta2 = embedding2.metadata or {}
        
        social_score_diff = abs(meta1.get('social_score', 0) - meta2.get('social_score', 0))
        
        return {
            'semantic_similarity': social_similarity,
            'social_score_difference': social_score_diff,
            'combined_social_metric': social_similarity - (social_score_diff * 0.5)
        }
    
    def clear_cache(self) -> None:
        """Clear the MPNet embedding cache to free memory."""
        self._embedding_cache.clear()
        logger.info("MPNet embedding cache cleared")
    
    def __repr__(self) -> str:
        """String representation of the MPNet Ingestion engine."""
        return (f"MPNetIngestion(embedding_dim={self.embedding_dim}, "
                f"social_focus={self.social_focus}, "
                f"cached_embeddings={len(self._embedding_cache)})")