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


def _search_embedding_worker(args):
    """Module-level worker function for multiprocessing BGE search (must be picklable)."""
    embedding_idx, embedding_data = args
    try:
        # Import here to avoid circular imports
        from Sysnpire.model.intial.bge_ingestion import BGEIngestion
        
        # Create a fresh BGE instance for this worker process
        bge_worker = BGEIngestion(model_name="BAAI/bge-large-en-v1.5", random_seed=42)
        result = bge_worker.search_embeddings(embedding_data, top_k=1)
        return (embedding_idx, result, None)
        
    except Exception as exc:
        return (embedding_idx, None, str(exc))



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
        
        # FOR TESTING: Use embeddings 5000-5099 (100 meaningful vocabulary words)
        test_embeddings = model_loaded['embeddings'][5000:5100]  # Use 5000-5099 for meaningful words  
        logger.info(f"🧪 Testing with {len(test_embeddings)} embeddings (indices 5000-5099)")
        
        # 🚀 OPTIMIZED SEQUENTIAL: Fast sequential processing with progress updates
        logger.info(f"🚀 Starting optimized BGE search for {len(test_embeddings)} embeddings...")
        
        enriched_e = []
        for i, embedding in enumerate(test_embeddings):
            try:
                result = self.model.search_embeddings(embedding, top_k=1)
                enriched_e.append(result)
                
                # Progress updates every 10 completions
                if (i + 1) % 10 == 0:
                    progress_pct = ((i + 1) / len(test_embeddings)) * 100
                    logger.info(f"   ✅ Completed {i + 1}/{len(test_embeddings)} searches ({progress_pct:.1f}%)")
                    
            except Exception as exc:
                logger.error(f"   ❌ BGE search {i} failed: {exc}")
                # Provide fallback result
                enriched_e.append([{'embedding': embedding, 'token': f'<ERROR_{i}>'}])
        
        logger.info(f"🎉 Sequential BGE search completed: {len(enriched_e)} results processed")
        
        # 🧬 EXTRACT VOCAB MAPPINGS: Get actual vocabulary words for our embeddings
        id_to_token = model_loaded.get('id_to_token')
        token_to_id = model_loaded.get('token_to_id')
        embedding_indices = list(range(5000, 5100))  # Track which embeddings we're using
        
        # 🔍 Extract actual vocabulary words for our embedding indices (optimized)
        vocab_words = [id_to_token.get(idx, f"<UNK_{idx}>") for idx in embedding_indices]
        
        logger.info(f"📚 BGE Vocabulary tokens for indices 5000-5099:")
        # Log first 10 and last 10 to avoid spam
        sample_tokens = [(idx, token) for idx, token in zip(embedding_indices[:10], vocab_words[:10])]
        sample_tokens += [(idx, token) for idx, token in zip(embedding_indices[-10:], vocab_words[-10:])]
        
        for idx, token in sample_tokens[:10]:
            logger.info(f"   Index {idx}: '{token}'")
        logger.info(f"   ... [skipped {len(vocab_words)-20} tokens] ...")
        for idx, token in sample_tokens[10:]:
            logger.info(f"   Index {idx}: '{token}'")
        
        vocab_mappings = {
            'id_to_token': id_to_token,
            'token_to_id': token_to_id,
            'embedding_indices': embedding_indices,
            'vocab_words': vocab_words  # Actual tokens for our embeddings
        }
        logger.info(f"📚 Extracted vocab mappings: {len(vocab_mappings['id_to_token'])} tokens available")
        
        # 🚀 BUILD THE COMPLETE PIPELINE WITH LIQUID RESULTS AND VOCAB!
        complete_results = self.charge_factory.build(enriched_e, model_loaded, vocab_mappings)
        
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
