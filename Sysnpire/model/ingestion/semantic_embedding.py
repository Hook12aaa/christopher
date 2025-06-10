"""
Semantic Embedding - Simplified representation of text embeddings

This class provides a clean interface for working with BGE embeddings
before they are transformed into conceptual charges.
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import hashlib


@dataclass
class SemanticEmbedding:
    """
    A simplified representation of a semantic embedding vector.
    
    This class wraps raw BGE embeddings with metadata and utility methods
    needed for conceptual charge generation.
    
    Attributes:
        vector (np.ndarray): The 1024-dimensional BGE embedding vector
        text (str): Original text that generated this embedding
        text_id (str): Unique identifier for the text
        metadata (Dict[str, Any]): Additional metadata about the embedding
        dimension (int): Dimensionality of the embedding (should be 1024)
        magnitude (float): L2 norm of the embedding vector
    """
    
    vector: np.ndarray
    text: str
    text_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize computed properties after dataclass creation."""
        if self.text_id is None:
            self.text_id = self._generate_text_id()
        
        if self.metadata is None:
            self.metadata = {}
        
        # Validate vector properties
        if not isinstance(self.vector, np.ndarray):
            raise ValueError("Vector must be a numpy ndarray")
        
        if len(self.vector.shape) != 1:
            raise ValueError("Vector must be 1-dimensional")
    
    @property
    def dimension(self) -> int:
        """Get the dimensionality of the embedding vector."""
        return self.vector.shape[0]
    
    @property
    def magnitude(self) -> float:
        """Get the L2 norm (magnitude) of the embedding vector."""
        return float(np.linalg.norm(self.vector))
    
    @property
    def normalized_vector(self) -> np.ndarray:
        """Get the L2-normalized version of the embedding vector."""
        if self.magnitude == 0:
            return self.vector.copy()
        return self.vector / self.magnitude
    
    def _generate_text_id(self) -> str:
        """Generate a unique ID for the text based on its content."""
        text_hash = hashlib.md5(self.text.encode('utf-8')).hexdigest()
        return f"emb_{text_hash[:12]}"
    
    def similarity(self, other: 'SemanticEmbedding') -> float:
        """
        Compute cosine similarity with another semantic embedding.
        
        Args:
            other (SemanticEmbedding): Another semantic embedding to compare with
            
        Returns:
            float: Cosine similarity score between -1 and 1
        """
        if self.dimension != other.dimension:
            raise ValueError("Embeddings must have the same dimensionality")
        
        # Compute cosine similarity
        dot_product = np.dot(self.normalized_vector, other.normalized_vector)
        return float(dot_product)
    
    def distance(self, other: 'SemanticEmbedding') -> float:
        """
        Compute Euclidean distance to another semantic embedding.
        
        Args:
            other (SemanticEmbedding): Another semantic embedding to compare with
            
        Returns:
            float: Euclidean distance between the embeddings
        """
        if self.dimension != other.dimension:
            raise ValueError("Embeddings must have the same dimensionality")
        
        return float(np.linalg.norm(self.vector - other.vector))
    
    def extract_semantic_components(self) -> Dict[str, np.ndarray]:
        """
        Extract semantic components from the embedding for charge generation.
        
        This method prepares the embedding data in the format needed for
        conceptual charge mathematics.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary containing semantic components:
                - 'tau': The core semantic vector (τ in the charge formula)
                - 'semantic_clusters': Grouped semantic features
                - 'magnitude_profile': Magnitude distribution across dimensions
        """
        components = {
            'tau': self.vector.copy(),  # Core semantic vector for charge formula
            'semantic_clusters': self._extract_semantic_clusters(),
            'magnitude_profile': self._extract_magnitude_profile()
        }
        
        return components
    
    def _extract_semantic_clusters(self) -> np.ndarray:
        """Extract clustered semantic features from the embedding."""
        # Group embedding dimensions into semantic clusters
        cluster_size = 128  # 1024 / 8 = 128 dimensions per cluster
        num_clusters = self.dimension // cluster_size
        
        clusters = []
        for i in range(num_clusters):
            start_idx = i * cluster_size
            end_idx = start_idx + cluster_size
            cluster_vector = self.vector[start_idx:end_idx]
            cluster_magnitude = np.linalg.norm(cluster_vector)
            clusters.append(cluster_magnitude)
        
        return np.array(clusters)
    
    def _extract_magnitude_profile(self) -> np.ndarray:
        """Extract magnitude distribution profile across embedding dimensions."""
        # Create a profile of how the embedding magnitude is distributed
        window_size = 64  # Sliding window for magnitude analysis
        profile = []
        
        for i in range(0, self.dimension - window_size + 1, window_size):
            window = self.vector[i:i + window_size]
            window_magnitude = np.linalg.norm(window)
            profile.append(window_magnitude)
        
        return np.array(profile)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the embedding to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the embedding
        """
        return {
            'text_id': self.text_id,
            'text': self.text,
            'vector': self.vector.tolist(),
            'dimension': self.dimension,
            'magnitude': self.magnitude,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticEmbedding':
        """
        Create a SemanticEmbedding from a dictionary representation.
        
        Args:
            data (Dict[str, Any]): Dictionary containing embedding data
            
        Returns:
            SemanticEmbedding: Reconstructed embedding object
        """
        return cls(
            vector=np.array(data['vector']),
            text=data['text'],
            text_id=data.get('text_id'),
            metadata=data.get('metadata', {})
        )
    
    def __repr__(self) -> str:
        """String representation of the semantic embedding."""
        return (f"SemanticEmbedding(text_id='{self.text_id}', "
                f"dimension={self.dimension}, magnitude={self.magnitude:.4f}, "
                f"text='{self.text[:50]}{'...' if len(self.text) > 50 else ''}')")