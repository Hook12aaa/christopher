"""
This script is deadicated with transforming a model into our inital universal manifold.
We are taking preivous emeddings from models such as mpnet, bge and others others. Then converting their geospatial embeddings into a universal manifold representation.
"""

import sys
from pathlib import Path
import argparse

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


# Reduce Jax spam and optimize MPS memory usage
import os

os.environ["JAX_LOG_COMPILE"] = "0"
# Prevent multiprocessing resource leaks
os.environ["PYTORCH_MULTIPROCESSING_SHARING_STRATEGY"] = "file_descriptor"

# BLAS OPTIMIZATION: Fix OpenBLAS threading and memory issues
os.environ["OPENBLAS_NUM_THREADS"] = "8"  # Conservative thread count for stability
os.environ["MKL_NUM_THREADS"] = "8"       # Intel MKL threading control
os.environ["NUMEXPR_NUM_THREADS"] = "8"   # NumExpr threading control
os.environ["OMP_NUM_THREADS"] = "8"       # OpenMP threading control
# Prevent OpenBLAS memory corruption on Apple Silicon
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
os.environ["BLAS_NUM_THREADS"] = "8"
# Additional numerical stability for field theory computations
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"  # Apple's Accelerate framework
os.environ["PYTORCH_MPS_ALLOCATOR_POLICY"] = "garbage_collection"  # Aggressive cleanup

# MPS MEMORY OPTIMIZATION: Set both watermark ratios for proper memory management
# This allows more aggressive memory allocation for large field theory computations
if not os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"):
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"  # Conservative limit to prevent memory corruption
if not os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"):
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.5"   # Must be lower than high watermark

# Model that are supported, these are our helper classes that will extract our key information from the model.
from Sysnpire.model.intial.bge_ingestion import BGEIngestion
from Sysnpire.model.intial.mpnet_ingestion import MPNetIngestion
from Sysnpire.model.charge_factory import ChargeFactory

# Database integration for persistent storage
from Sysnpire.database import FieldUniverse, FieldUniverseConfig
import time

from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


def parse_arguments():
    """
    Parse command-line arguments for the foundation manifold builder.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Build universal manifold from foundation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python foundation_manifold_builder.py --total 1000
  python foundation_manifold_builder.py --range 550 580
  python foundation_manifold_builder.py --range 550 580 --model mpnet
  python foundation_manifold_builder.py --total 500 --storage-path ./my_universe
        """,
    )

    parser.add_argument("--model", default="bge", choices=["bge", "mpnet"], help="Model to use (default: bge)")

    parser.add_argument("--total", type=int, help="Total number of embeddings to process from the beginning")

    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Range of embeddings to process (start end). Example: --range 550 580",
    )

    parser.add_argument("--storage-path", type=str, help="Path for database storage (default: timestamped directory)")

    return parser.parse_args()


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


class FoundationManifoldBuilder:
    """
    Class for building a universal manifold from foundation models.

    This class provides methods to transform existing embeddings into a universal manifold representation,
    integrating various semantic dimensions and ensuring coherence across the model.
    """

    def __init__(self, model: str = "bge", storage_path: str = None):
        """
        Initialize the FoundationManifoldBuilder with an optional model and storage path.

        Args:
            model (optional): The model to be transformed into a universal manifold.
            storage_path (optional): Path for persistent database storage. If None, uses timestamped default.
        """

        # Always enable database, just configurable path
        if storage_path is None:
            self.storage_path = f"./liquid_universes/{int(time.time())}"
        else:
            self.storage_path = storage_path

        self.model = self.select_model(model)
        self.model_info = self.model.info()
        self.charge_factory = ChargeFactory(from_base=True, model_info=self.model_info, model=self.model)
        self.__readout_info()

        logger.info(f"💾 Database storage configured: {self.storage_path}")

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

    def __load_model_and_check(self) -> dict:
        loaded_model = self.model.load_total_embeddings()

        if loaded_model is None:
            raise ValueError("Model embeddings could not be loaded. Please check the model and try again.")

        if int(loaded_model["embedding_dim"]) != int(self.model_info["dimension"]):
            raise ValueError(
                f"Model embedding dimension {loaded_model['embedding_dim']} does not match expected dimension {self.model_info['dimension']}. Please check the model configuration."
            )

        if loaded_model["vocab_size"] != self.model_info["vocab_size"]:
            raise ValueError(
                f"Model vocabulary size {loaded_model['vocab_size']} does not match expected size {self.model_info['vocab_size']}. Please check the model configuration."
            )

        return loaded_model

    def select_model(self, model_name) -> "BGEIngestion | MPNetIngestion":
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
            return MPNetIngestion(model_name="sentence-transformers/all-mpnet-base-v2", random_seed=42)
        else:
            raise ValueError(f"Model {model_name} is not supported. Please choose 'bge' or 'mpnet'.")

    def build_manifold(self, start_idx=None, end_idx=None, total=None):
        """
        Build the universal manifold from the selected model.

        Args:
            start_idx (int, optional): Starting index for embedding range
            end_idx (int, optional): Ending index for embedding range
            total (int, optional): Total number of embeddings to process from beginning
        """
        logger.info("Building the universal manifold from the selected model.")

        model_loaded = self.__load_model_and_check()

        # Get all available embeddings
        all_embeddings = model_loaded["embeddings"]
        total_available = len(all_embeddings)

        # STEP 1: DETERMINE EMBEDDING SLICE BASED ON PARAMETERS
        if start_idx is not None and end_idx is not None:
            # Validate range parameters
            if start_idx < 0:
                raise ValueError(f"Start index must be >= 0, got {start_idx}")
            if end_idx <= start_idx:
                raise ValueError(f"End index ({end_idx}) must be greater than start index ({start_idx})")
            if start_idx >= total_available:
                raise ValueError(f"Start index ({start_idx}) exceeds available embeddings ({total_available})")
            if end_idx > total_available:
                logger.warning(
                    f"End index ({end_idx}) exceeds available embeddings ({total_available}), using {total_available}"
                )
                end_idx = total_available

            test_embeddings = all_embeddings[start_idx:end_idx]
            embedding_indices = list(range(start_idx, end_idx))
            logger.info(f"🎯 Processing embedding range {start_idx}-{end_idx-1} ({len(test_embeddings)} embeddings)")

        elif total is not None:
            # Validate total parameter
            if total <= 0:
                raise ValueError(f"Total must be > 0, got {total}")
            
            # Start from real vocabulary (index 999) instead of unused tokens
            vocab_start = 999
            vocab_end = min(vocab_start + total, total_available)
            
            if vocab_end > total_available:
                logger.warning(
                    f"Requested total ({total}) from vocab start ({vocab_start}) exceeds available embeddings, using {total_available - vocab_start}"
                )
                vocab_end = total_available
                total = vocab_end - vocab_start

            test_embeddings = all_embeddings[vocab_start:vocab_end]
            embedding_indices = list(range(vocab_start, vocab_end))
            logger.info(f"🔢 Processing {total} real vocabulary embeddings starting from index {vocab_start}")

        else:
            # Default behavior - use real vocabulary range starting at 999
            default_start = 999  # First real vocabulary index
            default_count = 10   # Process 10 real vocabulary tokens
            default_end = min(default_start + default_count, total_available)
            test_embeddings = all_embeddings[default_start:default_end]
            embedding_indices = list(range(default_start, default_end))
            logger.info(
                f"🧪 Default mode: Processing real vocabulary indices {default_start}-{default_end-1} (use --total or --range for more)"
            )

        logger.info(f"📊 Selected {len(test_embeddings)} embeddings from {total_available} available")

        # 🚀 OPTIMIZED SEQUENTIAL: Fast sequential processing with progress updates
        logger.info(f"🚀 Starting optimized BGE search for {len(test_embeddings)} embeddings...")

        # ⚡ PERFORMANCE FIX: Pre-load model data once instead of reloading for each search
        logger.info("⚡ Pre-loading model data to avoid repeated loading...")
        if not hasattr(self.model, "_embedding_data"):
            self.model._embedding_data = self.model.load_total_embeddings()
        logger.info("✅ Model data cached for batch processing")

        enriched_e = []
        for i, embedding in enumerate(test_embeddings):
            try:
                # ⚡ DIRECT LOOKUP: Skip expensive search since we know the index
                actual_index = embedding_indices[i]  # Use the correct index from our range/selection
                token = model_loaded["id_to_token"].get(actual_index)

                # Create result structure directly without search
                result = {
                    "query": embedding,
                    "query_embedding": embedding,
                    "top_k": 1,
                    "embeddings": [
                        {
                            "index": actual_index,
                            "token": token,
                            "vector": embedding.tolist(),
                            "similarity": 1.0,  # Perfect match since it's the same embedding
                            "manifold_properties": {},  # Will be computed by ChargeFactory if needed
                        }
                    ],
                }
                enriched_e.append(result)

                # Progress updates every 10 completions
                if (i + 1) % 10 == 0:
                    progress_pct = ((i + 1) / len(test_embeddings)) * 100
                    logger.info(f"   ✅ Completed {i + 1}/{len(test_embeddings)} searches ({progress_pct:.1f}%)")

            except Exception as exc:
                logger.error(f"   ❌ BGE search {i} failed: {exc}")
                # Provide fallback result
                enriched_e.append(
                    [{"embedding": embedding, "token": f"<ERROR_{actual_index if 'actual_index' in locals() else i}>"}]
                )

        logger.info(f"🎉 Sequential BGE search completed: {len(enriched_e)} results processed")

        # 🧬 EXTRACT VOCAB MAPPINGS: Get actual vocabulary words for our embeddings
        id_to_token = model_loaded["id_to_token"]
        token_to_id = model_loaded["token_to_id"]
        # embedding_indices already defined above based on range/total parameters
        
        # Extract actual vocabulary words for our embedding indices
        vocab_words = []
        for idx in embedding_indices:
            token = id_to_token.get(idx)
            vocab_words.append(token)

        # Display range information
        start_idx_display = embedding_indices[0] if embedding_indices else 0
        end_idx_display = embedding_indices[-1] if embedding_indices else 0
        logger.info(f"📚 BGE Vocabulary tokens for indices {start_idx_display}-{end_idx_display}:")

        # Log first 10 and last 10 to avoid spam
        sample_tokens = [(idx, token) for idx, token in zip(embedding_indices[:10], vocab_words[:10])]
        if len(embedding_indices) > 20:
            sample_tokens += [(idx, token) for idx, token in zip(embedding_indices[-10:], vocab_words[-10:])]

        for idx, token in sample_tokens[:10]:
            logger.info(f"   Index {idx}: '{token}'")
        if len(sample_tokens) > 10:
            logger.info(f"   ... [skipped {len(vocab_words)-20} tokens] ...")
            for idx, token in sample_tokens[10:]:
                logger.info(f"   Index {idx}: '{token}'")

        vocab_mappings = {
            "id_to_token": id_to_token,
            "token_to_id": token_to_id,
            "embedding_indices": embedding_indices,
            "vocab_words": vocab_words,  # Actual tokens for our embeddings
        }
        logger.info(f"📚 Extracted vocab mappings: {len(vocab_mappings['id_to_token'])} tokens available")

        # 🚀 BUILD THE COMPLETE PIPELINE WITH LIQUID RESULTS AND VOCAB!
        complete_results = self.charge_factory.build(enriched_e, model_loaded, vocab_mappings)

        # 🎉 WE NOW HAVE BOTH combined_results AND liquid_results!
        combined_results = {
            "semantic_results": complete_results["semantic_results"],
            "temporal_results": complete_results["temporal_results"],
            "emotional_results": complete_results["emotional_results"],
            "field_components_ready": complete_results["field_components_ready"],
        }

        liquid_results = complete_results["liquid_results"]

        # 📁 DUMP BOTH RESULTS TO FILES FOR EXPLORATION
        import os

        project_root = Path(__file__).resolve().parent.parent.parent

        # Create dump function
        def dump_object(obj, indent=0, max_depth=5):
            """Recursively dump object structure to readable text with depth limit."""
            if indent > max_depth:
                return f"<max_depth_reached: {type(obj).__name__}>"
                
            spaces = "  " * indent
            if isinstance(obj, dict):
                lines = []
                for k, v in obj.items():
                    # Skip heavy objects that cause hangs
                    if k == "orchestrator":
                        lines.append(f"{spaces}{k}: <LiquidOrchestrator instance skipped>")
                    elif k == "agent_pool" and isinstance(v, dict) and len(v) > 10:
                        lines.append(f"{spaces}{k}: <{len(v)} agents skipped>")
                    else:
                        lines.append(f"{spaces}{k}: {dump_object(v, indent+1, max_depth)}")
                return "{\n" + "\n".join(lines) + f"\n{spaces}" + "}"
            elif isinstance(obj, list):
                if len(obj) > 3:  # Truncate long lists
                    items = [dump_object(item, indent + 1, max_depth) for item in obj[:3]]
                    return f"[{len(obj)} items: " + ", ".join(items) + ", ...]"
                else:
                    items = [dump_object(item, indent + 1, max_depth) for item in obj]
                    return "[" + ", ".join(items) + "]"
            elif hasattr(obj, "__dict__"):  # Custom objects
                # Skip complex objects that can cause infinite recursion
                if "torch" in str(type(obj)) or "sage" in str(type(obj)).lower():
                    return f"<{type(obj).__name__} instance skipped>"
                    
                attrs = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
                obj_info = f"{obj.__class__.__name__}("
                attr_strs = []
                for k, v in attrs.items():
                    if hasattr(v, "__len__") and len(str(v)) > 100:  # Long values
                        try:
                            attr_strs.append(f"{k}=<{type(v).__name__} len={len(v)}>")
                        except TypeError:
                            if hasattr(v, 'shape'):
                                attr_strs.append(f"{k}=<{type(v).__name__} shape={v.shape}>")
                            else:
                                attr_strs.append(f"{k}=<{type(v).__name__}>")
                    else:
                        attr_strs.append(f"{k}={dump_object(v, indent+1, max_depth)}")
                return obj_info + ", ".join(attr_strs) + ")"
            elif hasattr(obj, "__len__") and len(str(obj)) > 200:  # Long strings/arrays
                try:
                    return f"<{type(obj).__name__} len={len(obj)}>"
                except TypeError:
                    if hasattr(obj, 'shape'):
                        return f"<{type(obj).__name__} shape={obj.shape}>"
                    else:
                        return f"<{type(obj).__name__}>"
            else:
                return str(obj)

        # 📄 DUMP combined_results
        combined_path = os.path.join(project_root, "Sysnpire", "combined_results_example_all.txt")
        with open(combined_path, "w") as f:
            f.write("=== combined_results RESULTS DUMP ===\n\n")
            f.write(dump_object(combined_results))

        logger.info(f"✅ Combined results dumped to {combined_path}")

        # 🌊 DUMP liquid_results
        liquid_path = os.path.join(project_root, "Sysnpire", "liquid_results_example_all.txt")
        with open(liquid_path, "w") as f:
            f.write("=== LIQUID RESULTS DUMP ===\n\n")
            f.write(f"🚀 REVOLUTIONARY O(log N) LIQUID UNIVERSE CREATED!\n\n")
            f.write(dump_object(liquid_results))

        logger.info(f"🌊 Liquid results dumped to {liquid_path}")

        # 📊 LOG SUCCESS METRICS
        num_agents = liquid_results.get("num_agents")
        optimization_stats = liquid_results.get("optimization_stats")

        logger.info(f"🎉 LIQUID UNIVERSE PIPELINE SUCCESS!")
        logger.info(f"   🚀 Created {num_agents} living Q(τ,C,s) agents")
        logger.info(f"   ⚡ Performance mode: {optimization_stats.get('performance_mode')}")
        logger.info(f"   📈 Optimization factor: {optimization_stats.get('complexity_reduction')}")

        # 💾 STEP 5: BURN TO PERSISTENT DATABASE (MANDATORY)
        logger.info("💾 STEP 5: Burning liquid universe to persistent database...")
        try:
            config = FieldUniverseConfig(storage_path=Path(self.storage_path))
            field_universe = FieldUniverse(config)
            burning_results = field_universe.burn_liquid_universe(complete_results)

            logger.info(f"🔥 Universe burned successfully!")
            logger.info(f"   📁 Storage: {burning_results['storage_path']}")
            logger.info(f"   🎯 Agents: {burning_results['agents_burned']}")
            logger.info(f"   📚 Vocab tokens: {burning_results['vocab_context']['tokens_count']}")
            logger.info(f"   ⏱️ Burn time: {burning_results['burn_time_seconds']:.2f}s")

            # Include database results in return
            complete_results["database_results"] = burning_results

        except Exception as e:
            logger.error(f"❌ Database burning failed: {e}")
            logger.error(f"   📁 Attempted storage path: {self.storage_path}")
            logger.error("   💡 Check storage path permissions and disk space")
            raise RuntimeError(f"Mandatory database burning failed: {e}")

        # Memory cleanup to prevent leaks
        try:
            if hasattr(self.model, '_embedding_data'):
                del self.model._embedding_data
        except:
            pass
        
        return complete_results


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Validate arguments
    if args.range and args.total:
        logger.error("❌ Cannot specify both --range and --total. Choose one.")
        sys.exit(1)

    # Additional validation for range
    if args.range:
        start, end = args.range
        if start >= end:
            logger.error(f"❌ Range start ({start}) must be less than end ({end})")
            sys.exit(1)
        if start < 0:
            logger.error(f"❌ Range start must be >= 0, got {start}")
            sys.exit(1)

    # Additional validation for total
    if args.total and args.total <= 0:
        logger.error(f"❌ Total must be > 0, got {args.total}")
        sys.exit(1)

    # Log the configuration
    logger.info("🚀 Foundation Manifold Builder Starting")
    logger.info(f"   Model: {args.model}")
    if args.range:
        logger.info(f"   Range: {args.range[0]}-{args.range[1]-1} ({args.range[1] - args.range[0]} embeddings)")
    elif args.total:
        logger.info(f"   Total: {args.total} embeddings")
    else:
        logger.info("   Mode: Default (100 embeddings)")
    if args.storage_path:
        logger.info(f"   Storage: {args.storage_path}")

    # Create builder with specified model and storage path
    builder = FoundationManifoldBuilder(model=args.model, storage_path=args.storage_path)

    # Execute build with appropriate parameters
    try:
        if args.range:
            start, end = args.range
            builder.build_manifold(start_idx=start, end_idx=end)
        elif args.total:
            builder.build_manifold(total=args.total)
        else:
            builder.build_manifold()  # Default behavior

        logger.info("🎉 Foundation manifold build completed successfully!")

    except Exception as e:
        # Handle potential memory corruption issues
        import gc
        gc.collect()
        
        # Check if it's a memory corruption issue
        if "malloc" in str(e) or "checksum" in str(e) or "freed object" in str(e):
            logger.error("❌ Memory corruption detected - try reducing batch size or using CPU-only mode")
            logger.error("💡 Consider setting PYTORCH_MPS_HIGH_WATERMARK_RATIO to a lower value")
        
        raise ValueError(f"Foundation manifold build failed: {e}") from e
