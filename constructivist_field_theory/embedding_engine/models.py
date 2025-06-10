from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Union, Optional
from ..core_mathematics.conceptual_charge import ConceptualCharge

class ConceptualChargeGenerator:
    def __init__(self, model_name: str = "BAAI/bge-large-v1.5"):
        """
        Initialize the conceptual charge generator with BGE model.
        
        Args:
            model_name: Name of the BGE model to use for semantic embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Convert raw text into semantic embeddings.
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            numpy array of embeddings (1024 dimensions for BGE-large)
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_numpy=True)
    
    def create_conceptual_charge(self,
                               text: str,
                               context: Optional[Dict] = None,
                               observational_state: float = 0.0,
                               gamma: float = 1.0) -> ConceptualCharge:
        """
        Create a complete conceptual charge Q(τ, C, s) from text input.
        
        Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)
        
        Args:
            text: Input text (token τ) to create charge from
            context: Contextual environment C (can include semantic, emotional, social context)
            observational_state: Current observational state s
            gamma: Global field calibration factor γ
            
        Returns:
            ConceptualCharge object implementing complete field-theoretic formulation
        """
        # Get semantic embedding vector from BGE
        semantic_vector = self.encode_text(text)[0]  # Get first embedding if batch
        
        # Create and return complete conceptual charge
        return ConceptualCharge(
            token=text,
            semantic_vector=semantic_vector,
            context=context,
            observational_state=observational_state,
            gamma=gamma
        )
    
    def create_batch_charges(self,
                           texts: List[str],
                           contexts: Optional[List[Dict]] = None,
                           observational_states: Optional[List[float]] = None,
                           gamma: float = 1.0) -> List[ConceptualCharge]:
        """
        Create multiple conceptual charges in batch using field-theoretic formulation.
        
        Args:
            texts: List of input texts (tokens τ)
            contexts: Optional list of contextual environments C
            observational_states: Optional list of observational states s
            gamma: Global field calibration factor γ
            
        Returns:
            List of ConceptualCharge objects implementing complete formulation
        """
        # Get all semantic embeddings at once for efficiency
        semantic_vectors = self.encode_text(texts)
        
        charges = []
        for i, (text, semantic_vector) in enumerate(zip(texts, semantic_vectors)):
            context = contexts[i] if contexts and i < len(contexts) else {}
            observational_state = observational_states[i] if observational_states and i < len(observational_states) else 0.0
            
            charge = ConceptualCharge(
                token=text,
                semantic_vector=semantic_vector,
                context=context,
                observational_state=observational_state,
                gamma=gamma
            )
            charges.append(charge)
            
        return charges 