
"""
    BGEIngestion class for loading and managing BGE embeddings.

    For my Conceptual Charge I need to create an initial universe of charges from embeddings.
    This class handles loading the BGE model, managing device compatibility, and an access point for our embeddings.

    Unlike other ingestion classes, we aren't focused on converting text to embeddings here.
    Instead, we need to disect the BGE model and provide a way to access the embeddings
    directly for further processing.

    It provides the initial rendering for our Product Manifold, as we need to convert these embeddings into conceptual charges.
    
    It supports automatic detection of available on-device
    hardware (CPU, CUDA GPU, or MPS for Apple Silicon) and handles model loading accordingly.

    
"""



import sys
from pathlib import Path


# Ensure the project root is in the path for imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import necessary modules from the project
import numpy as np
import numba as nb
import hashlib
from typing import List, Optional, Dict, Any, Union
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import scipy.fft

from Sysnpire.utils.logger import get_logger
logger = get_logger(__name__)
HAS_RICH_LOGGER = True



class BGEIngestion():
    def  __init__(self,model_name: str = "BAAI/bge-large-en-v1.5", random_seed: Optional[int] = None) -> None:
        """
        Initialize the BGEIngestion class with a specific model.

        Will automatically detect available hardware and load the BGE model
        on the most appropriate device (CUDA GPU, MPS for Apple Silicon, or CPU).
    

        Args:
            model_name (str): Name of the BGE model to use.
            random_seed (Optional[int]): Seed for reproducibility.
        """
        self.model_name = model_name
        self.random_seed = random_seed
        self.model = self._load_model()
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        self.cache = {}

    def _load_model(self) -> None:
        """
        Load the BGE model with automatic CPU/GPU detection.
        
        This method intelligently detects available hardware and loads the
        BGE-Large-v1.5 model on the most appropriate device:
        - CUDA GPU if available and working
        - MPS (Apple Silicon) if available 
        - CPU as fallback
        
        The model is cached after first load for efficiency.
        """
                
        try:
            # Detect best available device
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Using CUDA device: {device_name}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                logger.info(f"Using MPS device: Apple Silicon")
            else:
                self.device = torch.device('cpu')
                logger.info("Using CPU for BGE model")
            
            # Load the model
            logger.info(f"Loading BGE model '{self.model_name}' on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)

            
            # Verify model loaded correctly
            if self.model is None:
                raise RuntimeError("Failed to load BGE model")
            
            logger.info(f"BGE model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load BGE model: {e}")
            # Fallback to CPU if GPU loading fails
            if self.device != torch.device('cpu'):
                logger.warning("Attempting fallback to CPU...")
                try:
                    self.device = torch.device('cpu')
                    self.model = SentenceTransformer(self.model_name, device=self.device)
                    logger.info("Successfully loaded model on CPU fallback")
                except Exception as cpu_error:
                    logger.error(f"CPU fallback also failed: {cpu_error}")
                    raise RuntimeError(f"Unable to load BGE model on any device: {e}")
            else:
                raise RuntimeError(f"Unable to load BGE model: {e}")
        return self.model
    
    
    def load_total_embeddings(self) -> Dict[str, Any]:
        """
        Extract all token embeddings and metadata from the BGE model.
        
        Returns:
            Dict containing:
            - 'embeddings': Token embedding matrix [vocab_size, embedding_dim]
            - 'vocab_size': Size of vocabulary
            - 'embedding_dim': Dimension of embeddings
            - 'tokenizer': Tokenizer for token-to-id mapping
            - 'token_to_id': Dictionary mapping tokens to IDs
            - 'id_to_token': Dictionary mapping IDs to tokens
        """
        if self.model is None:
            raise RuntimeError("BGE model is not loaded. Call _load_model() first.")
        
        try:
            # Access the underlying transformer model
            transformer_model = self.model[0].auto_model
            tokenizer = self.model[0].tokenizer
            
            # Extract token embeddings from the embedding layer
            embedding_layer = transformer_model.embeddings.word_embeddings
            token_embeddings = embedding_layer.weight.detach().cpu().numpy()
            
            vocab_size, embedding_dim = token_embeddings.shape
            
            # Get vocabulary mappings
            vocab = tokenizer.get_vocab()
            token_to_id = dict(vocab)
            id_to_token = {v: k for k, v in vocab.items()}
            
            logger.info(f"Extracted {vocab_size} token embeddings of dimension {embedding_dim}")
            logger.info(f"Vocabulary size: {len(vocab)}")
            
            return {
                'embeddings': token_embeddings,
                'vocab_size': vocab_size,
                'embedding_dim': embedding_dim,
                'tokenizer': tokenizer,
                'token_to_id': token_to_id,
                'id_to_token': id_to_token,
                'device': str(self.device)
            }
            
        except AttributeError as e:
            logger.error(f"Failed to access BGE model internals: {e}")
            logger.info("Attempting alternative extraction method...")
            
            # Alternative approach: access through model components
            try:
                # Get the first module (usually the transformer)
                first_module = self.model._modules['0']
                transformer = first_module.auto_model
                tokenizer = first_module.tokenizer
                
                # Extract embeddings
                embeddings = transformer.get_input_embeddings().weight.detach().cpu().numpy()
                vocab = tokenizer.get_vocab()
                
                vocab_size, embedding_dim = embeddings.shape
                token_to_id = dict(vocab)
                id_to_token = {v: k for k, v in vocab.items()}
                
                logger.info(f"Successfully extracted {vocab_size} embeddings via alternative method")
                
                return {
                    'embeddings': embeddings,
                    'vocab_size': vocab_size,
                    'embedding_dim': embedding_dim,
                    'tokenizer': tokenizer,
                    'token_to_id': token_to_id,
                    'id_to_token': id_to_token,
                    'device': str(self.device)
                }
                
            except Exception as alt_error:
                logger.error(f"Alternative extraction also failed: {alt_error}")
                raise RuntimeError(f"Unable to extract token embeddings from BGE model: {e}")
        
        except Exception as e:
            logger.error(f"Unexpected error during embedding extraction: {e}")
            raise RuntimeError(f"Failed to extract embeddings: {e}")
        

    def search_embeddings(self, query: str, top_k: int = 100) -> Dict[str, Any]:
        """
        Search embeddings and extract manifold properties for conceptual charge calculation.
        
        For each embedding, extracts mathematical context including geometric, field-theoretic,
        and topological properties needed for Q(τ, C, s) computation.

        Args:
            query (str): The query string to search for in the embeddings.
            top_k (int): Number of top similar embeddings to analyze.

        Returns:
            Dict containing search results with manifold properties for each embedding.
        """
        if self.model is None:
            raise RuntimeError("BGE model is not loaded. Call _load_model() first.")
        
        # Get embedding data if not cached
        if not hasattr(self, '_embedding_data'):
            self._embedding_data = self.load_total_embeddings()
        
        # Encode the query
        query_embedding = self.model.encode(query, convert_to_tensor=True).cpu().numpy()
        all_embeddings = self._embedding_data['embeddings']
        id_to_token = self._embedding_data['id_to_token']
        
        # Calculate similarities
        similarities = np.dot(all_embeddings, query_embedding) / (
            np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k most similar embeddings
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Prepare manifold analysis tools
        pca = PCA(n_components=min(50, all_embeddings.shape[1]))
        pca.fit(all_embeddings[top_indices])
        
        knn_model = NearestNeighbors(n_neighbors=min(20, len(top_indices)), metric='cosine')
        knn_model.fit(all_embeddings[top_indices])
        
        # Extract manifold properties for each top embedding
        results = {
            'query': query,
            'query_embedding': query_embedding,
            'top_k': top_k,
            'embeddings': []
        }
        
        for i, idx in enumerate(top_indices):
            embedding = all_embeddings[idx]
            token = id_to_token.get(idx, f"<UNK_{idx}>")
            similarity = similarities[idx]
            
            # Extract manifold properties
            manifold_props = self.extract_manifold_properties(
                embedding, i, all_embeddings[top_indices], pca, knn_model
            )
            
            embedding_result = {
                'index': int(idx),
                'token': token,
                'similarity': float(similarity),
                'manifold_properties': manifold_props
            }
            
            results['embeddings'].append(embedding_result)
        
        logger.info(f"Extracted manifold properties for {len(top_indices)} embeddings")
        return results
    
    def extract_manifold_properties(self, embedding: np.ndarray, index: int, 
                                   all_embeddings: np.ndarray, pca: PCA, 
                                   knn_model: NearestNeighbors) -> Dict[str, Any]:
        """
        Extract comprehensive manifold properties from an embedding for field theory calculations.
        
        Args:
            embedding: The embedding vector to analyze
            index: Index in the top-k set  
            all_embeddings: All embeddings in the analysis set
            pca: Fitted PCA model for dimensionality analysis
            knn_model: Fitted KNN model for local neighborhood analysis
            
        Returns:
            Dict containing all manifold properties needed for conceptual charge calculation
        """
        # Basic properties
        magnitude = np.linalg.norm(embedding)
        
        # Find k-nearest neighbors for local analysis
        distances, neighbor_indices = knn_model.kneighbors([embedding])
        neighbors = all_embeddings[neighbor_indices[0]]
        
        # Geometric properties
        local_density = 1.0 / (np.mean(distances[0]) + 1e-8)
        
        # Local curvature estimation via neighbor variance
        neighbor_center = np.mean(neighbors, axis=0)
        neighbor_deviations = neighbors - neighbor_center
        local_curvature = np.trace(np.cov(neighbor_deviations.T))
        
        # Metric tensor eigenvalues (local metric properties)
        if len(neighbors) > 1:
            cov_matrix = np.cov(neighbor_deviations.T)
            metric_eigenvalues = np.real(eigh(cov_matrix)[0])
        else:
            metric_eigenvalues = np.ones(embedding.shape[0])
        
        # Principal component projection
        e_i_projected = pca.transform([embedding])[0]
        
        # Phase angles in complex representation
        complex_embedding = embedding[:len(embedding)//2] + 1j * embedding[len(embedding)//2:]
        phase_angles = np.angle(complex_embedding)
        
        # Field properties - gradient estimation
        if len(neighbors) > 2:
            # Approximate gradient using finite differences with neighbors
            gradient = np.mean(neighbors - embedding, axis=0)
            gradient_magnitude = np.linalg.norm(gradient)
            
            # Hessian eigenvalues (second-order field properties)
            try:
                hessian_approx = np.outer(gradient, gradient) / (gradient_magnitude + 1e-8)
                eigenvalues = np.real(eigh(hessian_approx)[0])
            except:
                eigenvalues = np.zeros(20)
        else:
            gradient = np.zeros_like(embedding)
            gradient_magnitude = 0.0
            eigenvalues = np.zeros(20)
        
        # Persistence properties
        persistence_radius = np.max(distances[0])
        persistence_score = local_density * persistence_radius
        
        # Coupling properties (correlation with neighbors)
        if len(neighbors) > 1:
            correlations = [np.corrcoef(embedding, neighbor)[0,1] for neighbor in neighbors]
            correlations = [c for c in correlations if not np.isnan(c)]
            coupling_mean = np.mean(correlations) if correlations else 0.0
            coupling_variance = np.var(correlations) if correlations else 0.0
        else:
            coupling_mean = 0.0
            coupling_variance = 0.0
        
        # Spectral properties via FFT
        fft_result = scipy.fft.fft(embedding)
        power_spectrum = np.abs(fft_result)**2
        dominant_freq_indices = np.argsort(power_spectrum)[-10:]  # Top 10 frequencies
        dominant_frequencies = dominant_freq_indices.astype(float) / len(embedding)
        frequency_magnitudes = power_spectrum[dominant_freq_indices]
        
        # Topological properties
        # Boundary score: how much the point differs from local average
        local_mean = np.mean(neighbors, axis=0)
        boundary_score = np.linalg.norm(embedding - local_mean)
        
        # Loop detection via homology approximation
        if len(neighbors) >= 3:
            # Simple approximation: check if neighbors form loops in projection
            neighbor_distances = pdist(neighbors[:,:3])  # Use first 3 dims for speed
            distance_matrix = squareform(neighbor_distances)
            # Check triangular inequality violations as loop indicator
            violations = 0
            n_neighbors = len(neighbors)
            for i in range(min(n_neighbors, 5)):
                for j in range(i+1, min(n_neighbors, 5)):
                    for k in range(j+1, min(n_neighbors, 5)):
                        if i < len(distance_matrix) and j < len(distance_matrix) and k < len(distance_matrix):
                            d_ij, d_jk, d_ik = distance_matrix[i,j], distance_matrix[j,k], distance_matrix[i,k]
                            if d_ij + d_jk < d_ik * 0.9:  # Significant violation
                                violations += 1
            local_loops = violations > 0
        else:
            local_loops = False
        
        features = {
            # Basic properties
            'magnitude': float(magnitude),
            'vector': embedding.tolist(),
            
            # Geometric
            'local_density': float(local_density),
            'local_curvature': float(local_curvature),
            'metric_eigenvalues': metric_eigenvalues[:20].tolist(),  # Top 20
            
            # Directional
            'principal_components': e_i_projected[:50].tolist(),  # Top 50 PCs
            'phase_angles': phase_angles.tolist(),
            
            # Field properties
            'gradient': gradient.tolist(),
            'gradient_magnitude': float(gradient_magnitude),
            'hessian_eigenvalues': eigenvalues[:20].tolist(),  # Top 20
            
            # Persistence
            'persistence_radius': float(persistence_radius),
            'persistence_score': float(persistence_score),
            
            # Coupling
            'coupling_mean': float(coupling_mean),
            'coupling_variance': float(coupling_variance),
            
            # Spectral
            'dominant_frequencies': dominant_frequencies.tolist(),
            'frequency_magnitudes': frequency_magnitudes.tolist(),
            
            # Topological
            'boundary_score': float(boundary_score),
            'has_loops': bool(local_loops)
        }
        
        return features





if __name__ == "__main__":
    bge_ingestion = BGEIngestion(model_name="BAAI/bge-large-en-v1.5", random_seed=42)
    embedding_data = bge_ingestion.load_total_embeddings()
    
    logger.info(f"Vocabulary size: {embedding_data['vocab_size']}")
    logger.info(f"Embedding dimension: {embedding_data['embedding_dim']}")
    logger.info(f"Device used: {embedding_data['device']}")
    
    # Show sample tokens and their embeddings
    embeddings = embedding_data['embeddings']
    id_to_token = embedding_data['id_to_token']
    
    logger.info("Sample tokens and embedding info:")
    for i in range(min(10, len(embeddings))):
        token = id_to_token.get(i, f"<UNK_{i}>")
        embedding_norm = np.linalg.norm(embeddings[i])
        logger.info(f"Token {i}: '{token}' | Embedding norm: {embedding_norm:.4f}")
    
    logger.info(f"Total token embeddings extracted: {len(embeddings)}")

    # Example search for embeddings similar to  from our embeddings list
    query = embeddings[40]
    search_results = bge_ingestion.search_embeddings(query=query, top_k=5)
    logger.info(f"Search results for query embedding: {query[:10]}... (truncated)")
    for result in search_results['embeddings']:
        logger.info(f"Token: {result['token']} | Similarity: {result['similarity']:.4f} | "
                    f"Manifold properties: {result['manifold_properties']}")
    logger.info("BGE ingestion and search completed successfully.")