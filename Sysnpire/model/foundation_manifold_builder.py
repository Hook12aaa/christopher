"""
This script is deadicated with transforming a model into our inital universal manifold.
We are taking preivous emeddings from models such as mpnet, bge and others others. Then converting their geospatial embeddings into a universal manifold representation.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


# Reduce Jax spam
import os
os.environ["JAX_LOG_COMPILE"] = "0"

# Model that are supported, these are our helper classes that will extract our key information from the model.
from Sysnpire.model.intial.bge_ingestion import BGEIngestion
from Sysnpire.model.intial.mpnet_ingestion import MPNetIngestion
from Sysnpire.model.charge_factory import ChargeFactory

from Sysnpire.utils.logger import get_logger
logger = get_logger(__name__)



class FoundationManifoldBuilder():
    """
    Class for building a universal manifold from foundation models.
    
    This class provides methods to transform existing embeddings into a universal manifold representation,
    integrating various semantic dimensions and ensuring coherence across the model.
    """

    def __init__(self, model:str = 'bge'):
        """
        Initialize the FoundationManifoldBuilder with an optional model.
        
        Args:
            model (optional): The model to be transformed into a universal manifold.
        """


        self.model =self.select_model(model)
        self.model_info = self.model.info()
        self.charge_factory = ChargeFactory(from_base=True, model_info= self.model_info,model=self.model)
        self.__readout_info()

    def __readout_info(self):
        """
        Reads out information from the model and logs it.
        This method retrieves and logs the model's information, such as its name and other relevant details.

        Returns:
            None
        """
        # Log the model information
        logger.info(f"Reading out information from model: {self.model_info['model_name']}")
        for key, value in (self.model.info()).items():
            logger.info(f"{key}: {value}")


    def __load_model_and_check(self)->dict:
        loaded_model = self.model.load_total_embeddings()

        if loaded_model is None:
            raise ValueError("Model embeddings could not be loaded. Please check the model and try again.")
        
        if int(loaded_model['embedding_dim']) != int(self.model_info['dimension']):
            raise ValueError(f"Model embedding dimension {loaded_model['embedding_dim']} does not match expected dimension {self.model_info['dimension']}. Please check the model configuration.")
        
        if loaded_model['vocab_size'] != self.model_info['vocab_size']:
            raise ValueError(f"Model vocabulary size {loaded_model['vocab_size']} does not match expected size {self.model_info['vocab_size']}. Please check the model configuration.")
        
        return loaded_model


    def select_model(self, model_name)-> "BGEIngestion | MPNetIngestion":
        """
        Selects the appropriate model based on the provided model name.
        Args:
            model_name (str): The name of the model to select.
        Returns:
            An instance of the selected model ingestion class.
        Raises:
            ValueError: If the model name is not recognized.
        """
        logger.info(f"Selecting model: {model_name}")
        if model_name == "bge":
            return BGEIngestion(model_name="BAAI/bge-large-en-v1.5", random_seed=42)
        if model_name == "mpnet":
            return  MPNetIngestion(model_name="sentence-transformers/all-mpnet-base-v2", random_seed=42)
        else:
            raise ValueError(f"Model {model_name} is not supported. Please choose 'bge' or 'mpnet'.")
        


    def build_manifold(self):

        logger.info("Building the universal manifold from the selected model.")

        model_loaded = self.__load_model_and_check()
        
        # FOR TESTING: Use only embeddings 550-560 (10 embeddings)
        test_embeddings = model_loaded['embeddings'][550:560]
        logger.info(f"🧪 Testing with {len(test_embeddings)} embeddings (indices 550-560)")
        
        enriched_e = [self.model.search_embeddings(e, top_k=1) for e in test_embeddings]
        self.charge_factory.build(enriched_e, model_loaded)
        



        


        




if __name__ == "__main__":

    builder = FoundationManifoldBuilder(model = "bge")
    builder.build_manifold()
