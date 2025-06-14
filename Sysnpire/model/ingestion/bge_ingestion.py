
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
