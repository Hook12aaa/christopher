"""
MAIN SEMANTIC PROCESSING HELPER CLASS - Entry point for semantic dimension.


This module provides a helper class for adding and managing semantic dimensions
to data structures and models. It handles the creation, validation, and 
integration of semantic dimensions into existing systems.



This is the primary function called by ChargeFactory for any part of our  semantic dimension work .
Implements the Φ^semantic(τ, s) component of Q(τ, C, s).

MATHEMATICAL FOUNDATION:
Φ^semantic(τ, s) = w_i * T_i * x[i] * breathing_modulation * e^(iθ)

"""


import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from Sysnpire.model.semantic_dimension.vector_transformation import VectorTransformation
from Sysnpire.utils.logger import get_logger


logger = get_logger(__name__)



class SemanticDimensionHelper():
    """
    Helper class for managing semantic dimensions in data structures and models.
    
    This class provides methods to add, validate, and integrate semantic dimensions
    into existing systems, ensuring coherence and consistency across the model.
    """

    def __init__(self, from_base:bool, model_info=None, helper = None) -> None:
        """
        Initialize the SemanticDimensionHelper with a specific dimension name.

        """

        self.helper = helper
        self.from_base = from_base 


    def key_component(self, key: str) -> str:
        # Get embedding dimension from model info or helper
        if self.from_base and hasattr(self.helper, 'info'):
            model_info = self.helper.info()
            embedding_dim = model_info.get('dimension', 1024)  # BGE default
        else:
            embedding_dim = 1024  # Default for BGE
            
        self.vector = VectorTransformation(
            from_base=self.from_base, 
            embedding_dimension=embedding_dim, 
            helper=self.helper, 
            phase_computation_method="component_based"
        )
        
    

    def convert_vector_to_field_respentation(self, total_embeddings) -> dict:
        """
        Convert embedding vectors to semantic field representations.

        Implements the S_τ(x) = Σᵢ e_τ,ᵢ · φᵢ(x) · e^(iθ_τ,ᵢ) transformation
        for each embedding in the input list.

        Args:
            total_embeddings (list): List of BGE search result dictionaries or embedding dictionaries to convert

        Returns:
            dict: Dictionary containing field representations for all embeddings
        """
        # Extract individual embeddings from BGE search results
        individual_embeddings = []
        for item in total_embeddings:
            if 'embeddings' in item and isinstance(item['embeddings'], list):
                # This is a BGE search result with embedded list
                individual_embeddings.extend(item['embeddings'])
            else:
                # This is already an individual embedding dictionary
                individual_embeddings.append(item)
        
        logger.info(f"Converting {len(individual_embeddings)} embeddings to semantic fields")
        
        # Initialize vector transformation if not already done
        if not hasattr(self, 'vector'):
            self.key_component("default")  # Initialize transformation
        
        field_representations = []
        
        # Transform each embedding to semantic field
        for i, embedding_dict in enumerate(individual_embeddings):
            try:
                logger.debug(f"Transforming embedding {i+1}/{len(individual_embeddings)}")
                field_result = self.vector.model_transform_to_field(embedding_dict)
                field_representations.append(field_result)
                
            except Exception as e:
                logger.error(f"Failed to transform embedding {i}: {e}")
                raise RuntimeError(f"Semantic field transformation failed for embedding {i}: {e}")
        
        logger.info(f"Successfully converted {len(field_representations)} embeddings to semantic fields")
        
        return {
            'field_representations': field_representations,
            'total_converted': len(field_representations),
            'transformation_method': 'S_τ(x) = Σᵢ e_τ,ᵢ · φᵢ(x) · e^(iθ_τ,ᵢ)',
            'conversion_complete': True
        }