"""
Charge Factory - Focused Q(τ, C, s) Transformation Engine

FOCUSED RESPONSIBILITY: This factory has ONE job - take embedding vectors with their
mathematical properties and transform them into dynamic conceptual charges using the
complete Q(τ, C, s) field theory formula. It does NOT handle data sourcing.

MATHEMATICAL TRANSFORMATION:
Input: Static embedding + model_geometric + field 
Process: Apply complete Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)
Output: Uploads our dynamic conceptual charge to a live universe, ready for interaction.

DESIGN PRINCIPLE: This factory is model-agnostic and source-agnostic. It can process
embeddings from BGE models, MPNet models, scraped data, user inputs, or any other
source that provides embedding vectors + mathematical properties.

If it is starting from a base model, from base must be set to True. This is usually the BGE or MPNet model.

USAGE CONTEXTS:
- Initial "Big Bang" from model vocabularies (separate script)
- For building the universe from base model use .build()
- For Rejection/Accepting a new data scrapt content, use .integrate()



"""

import sys
from pathlib import Path
from typing import Dict, List, Any

# Add memory tracking:
import psutil
import os
import gc
import time


project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.model.semantic_dimension.SemanticDimensionHelper import (
    SemanticDimensionHelper,
)
from Sysnpire.model.temporal_dimension.TemporalDimensionHelper import (
    TemporalDimensionHelper,
)
from Sysnpire.model.emotional_dimension.EmotionalDimensionHelper import (
    EmotionalDimensionHelper,
)
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


# Liqued Universe this is universe stage one
from Sysnpire.model.liquid.liquid_orchestrator import LiquidOrchestrator


class ChargeFactory:
    """
    Focused Field Theory Charge Generator

    SINGLE RESPONSIBILITY: Transforms embeddings + mathematical properties
    into dynamic conceptual charges using Q(τ, C, s) field theory mathematics.

    AGNOSTIC DESIGN: Works with embeddings from any source - BGE models, MPNet models,
    scraped data, user text, or external systems. Only requires embedding vector
    and mathematical properties of either living universe or the base model.

    FROM BASE MODEL: If starting from a base model, set from_base to True.
    This is usually the BGE or MPNet model. This allows us to create the universe

    """

    def __init__(self, from_base: bool = True, model_info: dict = None, model=None):
        """

        Args:
        - from_base (bool): Whether to initialize with base model support from BGE or MPNet. This is usually at ground zero of creating the universe
        - model (Optional[Any]): If we are starting from base model, we need to understand what model we are using. This is usually the BGE or MPNet model.


        TODO: Initialize charge factory for Q(τ, C, s) transformations

        Implementation tasks:
        - Set up minimal initialization (no model loading, no data dependencies)
        - Initialize charge counter for tracking
        - Set up trajectory operator engine for T(τ, C, s) component
        - Initialize temporal orchestrator for cross-dimensional integration
        - Integrate temporal orchestrator with charge factory
        """
        self.from_base = from_base
        self.model_info = model_info

        # Set helper first before initializing factory helpers
        if self.from_base:
            # if we are from base, we need to pass the model as that will be our helper, (they share methods)
            self.helper = model
            logger.info("ChargeFactory initialized with base model support.")
        else:
            # if we are not from base, we need to pass the universe as that will be our helper,
            self.helper = None
            logger.info("ChargeFactory initialized without base model support.")

        # initialize  our factory helpers (after self.helper is set)
        self.__init_factory_helpers()

        # Initialize memory tracking
        self._memory_baseline = None
        self._memory_history = []

    def __init_factory_helpers(self):
        self.semantic_helper = SemanticDimensionHelper(
            self.from_base, model_info=self.model_info, helper=self.helper
        )  # This is our handler for semantic field generation (3.1.2)
        self.temporal_helper = TemporalDimensionHelper(
            self.from_base, model_info=self.model_info, helper=self.helper
        )  # This is our handler for temporal breathing patterns (3.1.4)
        self.emotional_helper = EmotionalDimensionHelper(
            self.from_base, model_info=self.model_info, helper=self.helper
        )  # This is our conductor for emotional field modulation (3.1.3)

    def _track_and_report_memory(self, step_name: str) -> Dict[str, float]:
        """
        Track memory usage and report changes.

        Args:
            step_name: Name of the processing step for logging

        Returns:
            Dict with memory statistics
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        current_memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB

        # Set baseline on first call
        if self._memory_baseline is None:
            self._memory_baseline = current_memory_mb
            memory_delta = 0.0
            logger.info(f"📊 MEMORY BASELINE: {current_memory_mb:.1f} MB")
        else:
            memory_delta = current_memory_mb - self._memory_baseline

        # Calculate delta from previous step
        step_delta = 0.0
        if self._memory_history:
            step_delta = current_memory_mb - self._memory_history[-1]["memory_mb"]

        memory_stats = {
            "step_name": step_name,
            "memory_mb": current_memory_mb,
            "delta_from_baseline_mb": memory_delta,
            "step_delta_mb": step_delta,
            "timestamp": time.time(),
        }

        self._memory_history.append(memory_stats)

        # Log memory usage with appropriate emoji
        if step_delta > 50:
            logger.warning(
                f"📈 MEMORY: {step_name} | {current_memory_mb:.1f} MB (+{step_delta:.1f} MB) | Total: +{memory_delta:.1f} MB"
            )
        elif step_delta < -50:
            logger.info(
                f"📉 MEMORY: {step_name} | {current_memory_mb:.1f} MB ({step_delta:.1f} MB) | Total: +{memory_delta:.1f} MB"
            )
        else:
            logger.info(f"📊 MEMORY: {step_name} | {current_memory_mb:.1f} MB | Total: +{memory_delta:.1f} MB")

        return memory_stats

    def _aggressive_cache_cleanup(self):
        """
        Aggressively purge all intermediate caches to free memory.
        Preserves only essential results.
        """
        cleanup_count = 0

        # Clear basis function caches from semantic dimension
        if hasattr(self, "semantic_helper") and hasattr(self.semantic_helper, "vector"):
            if hasattr(self.semantic_helper.vector, "model") and hasattr(
                self.semantic_helper.vector.model, "_spatial_analysis_cache"
            ):
                cache_size = len(self.semantic_helper.vector.model._spatial_analysis_cache)
                self.semantic_helper.vector.model._spatial_analysis_cache.clear()
                cleanup_count += cache_size
                logger.debug(f"🗑️  Cleared spatial analysis cache ({cache_size} entries)")

        # Clear basis caches if they exist
        if hasattr(self, "semantic_fields"):
            for field in self.semantic_fields:
                if hasattr(field, "basis_functions") and hasattr(field.basis_functions, "basis_cache"):
                    cache_size = len(field.basis_functions.basis_cache)
                    field.basis_functions.basis_cache.clear()
                    cleanup_count += cache_size

        # Clear helper internal caches
        for helper_name in ["semantic_helper", "temporal_helper", "emotional_helper"]:
            if hasattr(self, helper_name):
                helper = getattr(self, helper_name)
                # Clear any cached data in helpers
                if hasattr(helper, "_cache"):
                    helper._cache.clear()
                if hasattr(helper, "vector") and hasattr(helper.vector, "model"):
                    model = helper.vector.model
                    # Clear BGE model caches
                    for cache_attr in ["_spatial_analysis_cache", "_laplacian_cache", "_embedding_data"]:
                        if hasattr(model, cache_attr):
                            if isinstance(getattr(model, cache_attr), dict):
                                getattr(model, cache_attr).clear()
                            else:
                                delattr(model, cache_attr)

        # Force garbage collection
        gc.collect()

        if cleanup_count > 0:
            logger.info(f"🗑️  CACHE CLEANUP: Cleared {cleanup_count} cache entries + forced garbage collection")
        else:
            logger.debug("🗑️  CACHE CLEANUP: No caches found to clear, forced garbage collection")

    def _clear_dimension_caches(self, dimension_name: str):
        """
        Clear caches specific to a dimension helper.

        Args:
            dimension_name: Name of the dimension ('semantic', 'temporal', 'emotional')
        """
        helper_attr = f"{dimension_name}_helper"
        if hasattr(self, helper_attr):
            helper = getattr(self, helper_attr)
            if hasattr(helper, "vector") and hasattr(helper.vector, "model"):
                model = helper.vector.model
                if hasattr(model, "_spatial_analysis_cache"):
                    cache_size = len(model._spatial_analysis_cache)
                    model._spatial_analysis_cache.clear()
                    logger.debug(f"🗑️  Cleared {dimension_name} spatial analysis cache ({cache_size} entries)")

            # Clear basis function caches if available
            if dimension_name == "semantic" and hasattr(self, "semantic_fields"):
                for field in self.semantic_fields:
                    if hasattr(field, "basis_functions") and hasattr(field.basis_functions, "basis_cache"):
                        field.basis_functions.basis_cache.clear()

        gc.collect()

    def __build_safety_checks(self, all: List[Dict]) -> None:
        """
        Perform safety checks on the input list of embedding vectors.
        This method ensures that the input list is not empty, is a list of dictionaries,
        and that each dictionary contains the required 'embedding_vector' key.
        Args:
            all (List[Dict]): List of embedding vectors to be checked.
        Raises:
            ValueError: If the input list is empty or does not contain the required keys.
            TypeError: If the input is not a list of dictionaries.
        """

        if not all:
            raise ValueError("The input list 'all' cannot be empty. Please provide a list of embedding vectors.")
        if not isinstance(all, list):
            raise TypeError("The input 'all' must be a list of embedding vectors. Please provide a valid list.")
        if not self.from_base:
            raise ValueError(
                "ChargeFactory must be initialized with from_base=True to build the universe. Please check your initialization parameters."
            )

    def build(
        self,
        all: List[Dict],
        total_info: Dict[str, Any],
        vocab_mappings: Dict[str, Any] = None,
    ) -> Any:
        """
        Build the initial universe from a list of embedding vectors with vocabulary context.
        This method will take a list of embedding vectors and transform them into
        dynamic conceptual charges using the Q(τ, C, s) field theory mathematics.

        And then upload them into our Universe as found in /Sysnpire/database/field_universe.py

        Args:
            all (List[Dict]): List of embedding vectors to be transformed.
            total_info (Dict[str, Any]): Complete model information
            vocab_mappings (Dict[str, Any]): Vocabulary mappings (id_to_token, token_to_id)

        Returns:
            Any: The transformed dynamic conceptual charges with vocab context

        """
        # Perform safety checks on the input list
        self.__build_safety_checks(all)

        # 📊 MEMORY TRACKING: Establish baseline
        self._track_and_report_memory("Build Started")

        # 📚 VOCAB CONTEXT: Prepare vocabulary mappings for dimensional helpers
        if vocab_mappings is None:
            vocab_mappings = {
                "id_to_token": {},
                "token_to_id": {},
                "embedding_indices": [],
            }
        logger.info(f"📚 Threading vocab context: {len(vocab_mappings.get('id_to_token', {}))} tokens available")

        # STEP 1: Convert embeddings to semantic fields with vocab context
        semantic_results = self.semantic_helper.convert_vector_to_field_respentation(all, vocab_mappings)
        self.semantic_fields = semantic_results["field_representations"]

        logger.info(f"✅ Generated {len(self.semantic_fields)} semantic fields with vocab context")

        # 📊 MEMORY TRACKING: After semantic processing + cleanup
        self._track_and_report_memory("Semantic Processing Complete")
        self._clear_dimension_caches("semantic")
        self._track_and_report_memory("After Semantic Cache Cleanup")

        # STEP 2: Convert embeddings to temporal breathing patterns with vocab context
        temporal_results = self.temporal_helper.convert_embedding_to_temporal_field(all, vocab_mappings)
        self.temporal_biographies = temporal_results["temporal_biographies"]

        logger.info(f"🌊 Generated {len(self.temporal_biographies)} temporal breathing patterns with vocab context")

        # 📊 MEMORY TRACKING: After temporal processing + cleanup
        self._track_and_report_memory("Temporal Processing Complete")
        self._clear_dimension_caches("temporal")
        self._track_and_report_memory("After Temporal Cache Cleanup")

        # STEP 3: Emotional conductor with vocab context - coordinate field modulation parameters
        emotional_results = self.emotional_helper.convert_embeddings_to_emotional_modulation(all, vocab_mappings)
        self.emotional_modulations = emotional_results["emotional_modulations"]

        logger.info(f"🎭 Generated emotional field conductor with {len(self.emotional_modulations)} modulations")

        # Log field strength from signature
        field_signature = emotional_results["field_signature"]
        logger.info(f"   Field strength: {field_signature.field_modulation_strength:.3f}")
        logger.info(f"   Pattern confidence: {field_signature.pattern_confidence:.3f}")

        # 📊 MEMORY TRACKING: After emotional processing + cleanup
        self._track_and_report_memory("Emotional Processing Complete")
        self._clear_dimension_caches("emotional")
        self._track_and_report_memory("After Emotional Cache Cleanup")

        # Combine results with emotional coordination ready AND vocab context
        combined_results = {
            "semantic_results": semantic_results,
            "temporal_results": temporal_results,
            "emotional_results": emotional_results,
            "field_components_ready": {
                "semantic_fields": len(self.semantic_fields),
                "temporal_biographies": len(self.temporal_biographies),
                "emotional_modulations": len(self.emotional_modulations),
                "emotional_conductor_active": True,
                "ready_for_unified_assembly": True,
            },
            # 📚 VOCAB THREADING: Include vocab mappings for LiquidOrchestrator
            "vocab_mappings": vocab_mappings,
        }

        # STEP 4: Hand off to LiquidOrchestrator for liquid universe creation
        # This is where all dimensions CLASH together in cascading feedback loops:
        # - Emotional conductor modulates semantic fields AND temporal patterns
        # - Semantic fields breathe with temporal rhythms
        # - Temporal trajectories reshape semantic landscapes
        # - Everything flows together like liquid metal forming Q(τ,C,s)
        logger.info("🎭 STEP 4: All dimensions ready to clash in liquid stage...")

        # Create LiquidOrchestrator and pass combined results for agent creation
        liquid_orchestrator = LiquidOrchestrator(device="mps")  # Use MPS for Apple Silicon
        liquid_results = liquid_orchestrator.create_liquid_universe(combined_results)

        logger.info(f"🌊 Liquid universe created with {liquid_results['num_agents']} living Q(τ,C,s) entities")

        # 📊 MEMORY TRACKING: After liquid processing + aggressive cleanup
        self._track_and_report_memory("Liquid Processing Complete")
        self._aggressive_cache_cleanup()
        self._track_and_report_memory("After Aggressive Cache Cleanup")

        # 📊 MEMORY TRACKING: Final memory report
        final_stats = self._track_and_report_memory("Build Complete - Ready to Return")

        # Return complete results with memory summary
        result = {
            **combined_results,  # Includes semantic, temporal, emotional results
            "liquid_results": liquid_results,  # The liquid universe with agent pool
            "memory_summary": {
                "total_memory_used_mb": final_stats["delta_from_baseline_mb"],
                "processing_steps": len(self._memory_history),
                "memory_history": (
                    self._memory_history[-3:] if len(self._memory_history) > 3 else self._memory_history
                ),  # Last 3 steps
            },
        }

        logger.info(f"📊 MEMORY SUMMARY: Total processing used {final_stats['delta_from_baseline_mb']:.1f} MB")
        return result

    def integrate():
        """
        Integrate new data into the charge factory.

        This method will handle the integration of new data, transforming it into
        dynamic conceptual charges using the Q(τ, C, s) field theory mathematics.
        """
        # TODO: This is a later stage, we are focusing on the initial charge generation.
        pass


# TODO: Add example usage section showing source-agnostic design
if __name__ == "__main__":
    """
    TODO: Implement example usage demonstrating:
    1. Factory initialization (no dependencies)
    2. Example parameter setup
    3. Logging of factory capabilities
    4. Source-agnostic design examples (BGE, MPNet, user text, scraped data, etc.)
    5. Factory statistics demonstration
    """
    # TODO: Implement example usage code
    pass
