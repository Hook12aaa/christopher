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
        self.model_info = model_info 


    def key_component(self, key: str) -> str:
        self.vector = VectorTransformation(from_base=self.from_base, embedding_dimension= self.model_info['dimension'] if self.from_base == True else 1024, helper=self.helper, phase_computation_method="component_based")
        
    

    def convert_vector_to_field_respentation(self, total_embeddings) -> dict:
        """
        Convert a vector representation into a semantic field representation.

        Args:
            total_embeddings (list): List of embedding dictionaries to be converted.

        Returns:
            dict: A dictionary representing the semantic field results.
        """
        logger.info(f"Converting {len(total_embeddings)} embeddings to field representation")
        
        # Initialize the vector transformation tool
        self.vector_transformer = VectorTransformation(
            from_base=self.from_base, 
            embedding_dimension=self.model_info['dimension'] if self.from_base else 1024, 
            helper=self.helper, 
            phase_computation_method="component_based"
        )
        
        field_results = []
        
        # We need to create a field representation of our vector, this is the core transformation
        for i, embedding_data in enumerate(total_embeddings):
            logger.info(f"Processing embedding {i+1}/{len(total_embeddings)}")
            
            # Transform each embedding to field representation
            field_result = self.vector_transformer.model_transform_to_field(embedding_data)
            field_results.append(field_result)
        
        logger.info(f"Completed field transformation for {len(field_results)} embeddings")
        
        return {
            'field_results': field_results,
            'total_processed': len(field_results),
            'transformation_method': 'vector_to_field',
            'success': True
        }