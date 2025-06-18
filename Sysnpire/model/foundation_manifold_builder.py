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
        test_embeddings = model_loaded['embeddings'][580:590]  # Adjusted to use 580-590 for 10 embeddings
        logger.info(f"🧪 Testing with {len(test_embeddings)} embeddings (indices 550-560)")
        
        enriched_e = [self.model.search_embeddings(e, top_k=1) for e in test_embeddings]
        
        # 🚀 BUILD THE COMPLETE PIPELINE WITH LIQUID RESULTS!
        complete_results = self.charge_factory.build(enriched_e, model_loaded)
        
        # 🎉 WE NOW HAVE BOTH combined_results AND liquid_results!
        combined_results = {
            'semantic_results': complete_results['semantic_results'],
            'temporal_results': complete_results['temporal_results'], 
            'emotional_results': complete_results['emotional_results'],
            'field_components_ready': complete_results['field_components_ready']
        }
        
        liquid_results = complete_results['liquid_results']
        
        # 📁 DUMP BOTH RESULTS TO FILES FOR EXPLORATION
        import os
        project_root = Path(__file__).resolve().parent.parent.parent
        
        # Create dump function
        def dump_object(obj, indent=0):
            """Recursively dump object structure to readable text."""
            spaces = "  " * indent
            if isinstance(obj, dict):
                lines = [f"{spaces}{k}: {dump_object(v, indent+1)}" for k, v in obj.items()]
                return "{\n" + "\n".join(lines) + f"\n{spaces}" + "}"
            elif isinstance(obj, list):
                if len(obj) > 3:  # Truncate long lists
                    items = [dump_object(item, indent+1) for item in obj[:3]]
                    return f"[{len(obj)} items: " + ", ".join(items) + ", ...]"
                else:
                    items = [dump_object(item, indent+1) for item in obj]
                    return "[" + ", ".join(items) + "]"
            elif hasattr(obj, '__dict__'):  # Custom objects
                attrs = {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
                obj_info = f"{obj.__class__.__name__}("
                attr_strs = []
                for k, v in attrs.items():
                    if hasattr(v, '__len__') and len(str(v)) > 100:  # Long values
                        attr_strs.append(f"{k}=<{type(v).__name__} len={len(v)}>")
                    else:
                        attr_strs.append(f"{k}={dump_object(v, indent+1)}")
                return obj_info + ", ".join(attr_strs) + ")"
            elif hasattr(obj, '__len__') and len(str(obj)) > 200:  # Long strings/arrays
                return f"<{type(obj).__name__} len={len(obj)}>"
            else:
                return str(obj)

        # 📄 DUMP combined_results
        combined_path = os.path.join(project_root, "Sysnpire", "combined_results_example.txt")
        with open(combined_path, 'w') as f:
            f.write("=== combined_results RESULTS DUMP ===\n\n")
            f.write(dump_object(combined_results))
        
        logger.info(f"✅ Combined results dumped to {combined_path}")
        
        # 🌊 DUMP liquid_results  
        liquid_path = os.path.join(project_root, "Sysnpire", "liquid_results_example.txt")
        with open(liquid_path, 'w') as f:
            f.write("=== LIQUID RESULTS DUMP ===\n\n")
            f.write(f"🚀 REVOLUTIONARY O(log N) LIQUID UNIVERSE CREATED!\n\n")
            f.write(dump_object(liquid_results))
        
        logger.info(f"🌊 Liquid results dumped to {liquid_path}")
        
        # 📊 LOG SUCCESS METRICS
        num_agents = liquid_results.get('num_agents', 0)
        optimization_stats = liquid_results.get('optimization_stats', {})
        
        logger.info(f"🎉 LIQUID UNIVERSE PIPELINE SUCCESS!")
        logger.info(f"   🚀 Created {num_agents} living Q(τ,C,s) agents")
        logger.info(f"   ⚡ Performance mode: {optimization_stats.get('performance_mode', 'Unknown')}")
        logger.info(f"   📈 Optimization factor: {optimization_stats.get('complexity_reduction', 'Unknown')}")
        
        return complete_results



        


        




if __name__ == "__main__":

    builder = FoundationManifoldBuilder(model = "bge")
    builder.build_manifold()
