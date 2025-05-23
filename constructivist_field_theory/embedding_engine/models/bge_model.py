"""
BGE model integration for accessing rich semantic embeddings.

This module provides functionality to interact with the BGE-Large-v1.5 model,
which serves as our foundational semantic space before projection into
hyperbolic space for constructivist mathematics implementation.
"""

import os
import sys
import numpy as np
import mlx.core as mx
import logging
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USE_SENTENCE_TRANSFORMERS = True
try:
    from sentence_transformers import SentenceTransformer
except (ImportError, RuntimeError) as e:
    logger.warning(f"Failed to import SentenceTransformer: {str(e)}")
    logger.warning("Falling back to alternative embedding method")
    USE_SENTENCE_TRANSFORMERS = False
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to import torch/transformers: {str(e)}")
        logger.warning("Using simple word embedding fallback")


class BGEModel:
    """
    Interface for the BGE-Large-v1.5 embedding model.
    
    BGE (BAAI General Embedding) is a state-of-the-art embedding model that
    creates rich semantic representations of text for our constructivist framework.
    """
    
    def __init__(self, model_name="BAAI/bge-large-v1.5", device=None, cache_dir=None):
        """Initialize the BGE model."""
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.embedding_dim = 1024
        self.model_type = "unknown"
        
        if device is None:
            try:
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
        
        self.device = device
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Model loading timed out")
        
        try:
            if sys.platform != 'win32':
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                
            if USE_SENTENCE_TRANSFORMERS:
                try:
                    if "bge" in model_name.lower() and "/" in model_name:
                        logger.info(f"Using BGE model: {model_name}")
                    
                    self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
                    self.embedding_dim = self.model.get_sentence_embedding_dimension()
                    self.model_type = "sentence_transformers"
                    logger.info(f"Using SentenceTransformer with model: {model_name}")
                    logger.info(f"Model successfully loaded with dimension: {self.embedding_dim}")
                except Exception as e:
                    logger.error(f"Error loading SentenceTransformer: {str(e)}")
                    self._setup_fallback_model()
            else:
                self._setup_fallback_model()
                
            if sys.platform != 'win32':
                signal.alarm(0)
                
        except TimeoutError:
            logger.error("Model loading timed out. Using fallback model.")
            self._setup_fallback_model()
        except Exception as e:
            logger.error(f"Unexpected error during model loading: {str(e)}")
            self._setup_fallback_model()
            
    def _setup_fallback_model(self):
        """Set up a fallback model when SentenceTransformer is not available."""
        try:
            fallback_model_name = "bert-base-uncased"
            logger.info(f"Attempting to load fallback model: {fallback_model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model_name, cache_dir=self.cache_dir)
            self.model = AutoModel.from_pretrained(fallback_model_name, cache_dir=self.cache_dir)
            
            if self.device != "cpu":
                self.model = self.model.to(self.device)
            
            self.model_type = "huggingface"
            logger.info(f"Using HuggingFace transformers with model: {fallback_model_name}")
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {str(e)}")
            self.model_type = "simple"
            logger.info("Using simple word embedding fallback")
            
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for the HuggingFace transformers model."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
    def _simple_encode(self, texts):
        """Simple fallback encoding method using basic word hashing."""
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        for text in texts:
            words = text.lower().split()
            embedding = np.zeros(self.embedding_dim)
            
            for i, word in enumerate(words):
                hash_val = hash(word) % 10000
                np.random.seed(hash_val)
                word_vec = np.random.randn(self.embedding_dim)
                embedding += word_vec
                
            if words:
                embedding = embedding / np.sqrt((embedding**2).sum())
                
            embeddings.append(embedding)
            
        return np.array(embeddings)
        
    def encode(self, texts, batch_size=32, normalize=True, convert_to_mlx=True):
        """Encode text into embedding vectors."""
        if isinstance(texts, str):
            texts = [texts]
            
        if self.model_type == "sentence_transformers":
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size, 
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 100
            )
        elif self.model_type == "huggingface":
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                encoded_input = self.tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                )
                
                if self.device != "cpu":
                    encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                    
                    if normalize:
                        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                        
                    embeddings.append(batch_embeddings.cpu().numpy())
                    
            embeddings = np.vstack(embeddings)
        else:
            embeddings = self._simple_encode(texts)
        
        if convert_to_mlx:
            return mx.array(embeddings)
        
        return embeddings
    
    def semantic_search(self, query_embeddings, corpus_embeddings, top_k=5):
        """Perform semantic search using cosine similarity."""
        if hasattr(query_embeddings, 'numpy'):
            query_embeddings = query_embeddings.numpy()
            
        if hasattr(corpus_embeddings, 'numpy'):
            corpus_embeddings = corpus_embeddings.numpy()
            
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
            
        similarities = np.matmul(query_embeddings, corpus_embeddings.T)
        
        results = []
        for i in range(len(query_embeddings)):
            scores = []
            for j in range(len(corpus_embeddings)):
                scores.append({
                    'corpus_id': j,
                    'score': float(similarities[i, j])
                })
            scores = sorted(scores, key=lambda x: x['score'], reverse=True)
            results.append(scores[:top_k])
            
        return results
    
    def get_embedding_dim(self):
        """Get the dimensionality of the embeddings."""
        return self.embedding_dim
    
    def get_model_info(self):
        """Get information about the model."""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'model_type': self.model_type
        }
    
if __name__ == "__main__":
    bge_model = BGEModel()
    texts = ["Hello, world!", "This is a test sentence."]
    
    try:
        embeddings = bge_model.encode(texts)
        
        print("Model type:", bge_model.model_type)
        print("Embeddings shape:", embeddings.shape)
        
        if hasattr(embeddings[0], 'numpy'):
            first_embedding = embeddings[0].numpy()
        else:
            first_embedding = embeddings[0][:10]
            
        print(f"First embedding (first 10 values): {first_embedding[:10]}")
        
        search_results = bge_model.semantic_search(embeddings[0], embeddings)
        print("Semantic search results:", search_results)
    except Exception as e:
        logger.error(f"Error during embedding or search: {str(e)}")
        import traceback
        traceback.print_exc()