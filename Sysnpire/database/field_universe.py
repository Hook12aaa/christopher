"""
Field Universe - Main Orchestrator for Hybrid Storage Architecture

Orchestrates the complete pipeline for burning liquid universe results into persistent storage
and reconstructing them back into living liquid universes. Uses hybrid HDF5 + Arrow/Parquet
storage for optimal mathematical precision and query performance.

Mathematical Foundation:
Preserves the complete field equation: ∂M/∂t = ∇²M + F[M] + Σᵢ T[Qᵢ]

Architecture:
- Liquid Burning: ConceptualChargeAgent → Persistent storage (HDF5 + Arrow)
- Universe Reconstruction: Persistent storage → Living liquid universe
- Dynamic Dimensionality: Model-agnostic field handling
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Core imports
from .conceptual_charge_object import ConceptualChargeObject
from .hybrid_storage.storage_coordinator import StorageCoordinator

# Hybrid storage components
from .liquid_burning.burning_orchestrator import BurningOrchestrator
from .universe_reconstruction.reconstructor import UniverseReconstructor

logger = logging.getLogger(__name__)


@dataclass
class FieldUniverseConfig:
    """Configuration for field universe operations."""

    storage_path: Path
    hdf5_compression: str = "lzf"
    arrow_batch_size: int = 1000
    enable_validation: bool = True
    dimension_agnostic: bool = True
    preserve_precision: bool = True


class FieldUniverse:
    """
    Main orchestrator for hybrid storage architecture.

    Coordinates the burning of liquid universe results into persistent storage
    and reconstruction back into living universes while preserving complete
    mathematical accuracy of Q(τ,C,s) field theory.

    Key Capabilities:
    - Burn liquid universe → persistent hybrid storage
    - Reconstruct persistent storage → living liquid universe
    - Dynamic dimensionality handling (BGE 1024, MPNet 768, etc.)
    - Mathematical precision preservation
    - Efficient query and retrieval operations
    """

    def __init__(self, config: FieldUniverseConfig):
        """
        Initialize field universe orchestrator.

        Args:
            config: Configuration for storage operations
        """
        self.config = config
        self.storage_path = Path(config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize orchestrator components
        self._init_orchestrators()

        logger.info(f"FieldUniverse initialized with storage: {self.storage_path}")
        logger.info(f"Dimension-agnostic mode: {config.dimension_agnostic}")

    def _init_orchestrators(self):
        """Initialize the orchestrator components."""
        # Storage coordinator for hybrid HDF5 + Arrow management
        self.storage_coordinator = StorageCoordinator(
            storage_path=self.storage_path,
            hdf5_compression=self.config.hdf5_compression,
            arrow_batch_size=self.config.arrow_batch_size,
        )

        # Burning orchestrator for liquid → persistent conversion
        self.burning_orchestrator = BurningOrchestrator(
            storage_coordinator=self.storage_coordinator,
            enable_validation=self.config.enable_validation,
            preserve_precision=self.config.preserve_precision,
        )

        # Universe reconstructor for persistent → liquid conversion
        self.universe_reconstructor = UniverseReconstructor(
            storage_coordinator=self.storage_coordinator,
            dimension_agnostic=self.config.dimension_agnostic,
        )

    def burn_liquid_universe(self, liquid_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Burn a liquid universe into persistent hybrid storage.

        This is the main entry point for converting liquid universe results
        from ChargeFactory.build() into persistent storage while preserving
        all mathematical components of the field theory.

        Args:
            liquid_results: Complete results from ChargeFactory.build() containing:
                          - semantic_results, temporal_results, emotional_results, vocab_mappings
                          - liquid_results (nested): Contains agent_pool, active_charges, field_statistics

        Returns:
            Dict containing burning results and storage metadata
        """
        logger.info("🔥 Starting liquid universe burning process...")
        start_time = time.time()

        # Extract the nested liquid_results containing actual universe data
        nested_liquid_results = liquid_results.get("liquid_results", {})

        # Extract key metrics for logging from nested structure
        num_agents = nested_liquid_results.get("num_agents", 0)
        field_stats = nested_liquid_results.get("field_statistics", {})

        logger.info(f"   🌊 Burning {num_agents} liquid agents")
        logger.info(f"   ⚡ Field energy: {field_stats.get('field_energy', 0):.6f}")

        # Include vocab context in burning process
        vocab_mappings = liquid_results.get("vocab_mappings", {})
        logger.info(f"   📚 Vocab context: {len(vocab_mappings.get('id_to_token', {}))} tokens")

        # Delegate to burning orchestrator (pass full structure for vocab access)
        burning_results = self.burning_orchestrator.burn_universe(liquid_results)

        # Calculate timing
        burn_time = time.time() - start_time

        # Compile results
        results = {
            "status": "burned",
            "burn_time_seconds": burn_time,
            "agents_burned": num_agents,
            "storage_path": str(self.storage_path),
            "burning_results": burning_results,
            "field_statistics": field_stats,
            "vocab_context": {
                "tokens_count": len(vocab_mappings.get("id_to_token", {})),
                "has_model_info": "model_info" in vocab_mappings,
            },
            "timestamp": time.time(),
        }

        logger.info(f"🔥 Liquid universe burned successfully in {burn_time:.2f}s")
        logger.info(f"   📁 Storage: {self.storage_path}")
        logger.info(f"   🎯 Agents burned: {num_agents}")
        logger.info(f"   📚 Vocab tokens: {len(vocab_mappings.get('id_to_token', {}))}")

        return results

    def reconstruct_liquid_universe(self, universe_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Reconstruct a liquid universe from persistent storage.

        Takes burned storage data and reconstructs it back into a complete
        living liquid universe with all mathematical components restored.

        Args:
            universe_id: Optional specific universe to reconstruct (latest if None)

        Returns:
            Dict containing reconstructed liquid universe results
        """
        logger.info("🔄 Starting liquid universe reconstruction...")
        start_time = time.time()

        # Delegate to universe reconstructor
        reconstruction_results = self.universe_reconstructor.reconstruct_universe(universe_id)

        # Calculate timing
        reconstruction_time = time.time() - start_time

        logger.info(f"🔄 Liquid universe reconstructed in {reconstruction_time:.2f}s")

        return {
            "status": "reconstructed",
            "reconstruction_time_seconds": reconstruction_time,
            "reconstruction_results": reconstruction_results,
            "timestamp": time.time(),
        }

    def list_burned_universes(self) -> List[Dict[str, Any]]:
        """
        List all burned universes available for reconstruction.

        Returns:
            List of universe metadata dictionaries
        """
        return self.storage_coordinator.list_universes()

    def get_universe_metadata(self, universe_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific burned universe.

        Args:
            universe_id: Universe identifier

        Returns:
            Universe metadata dictionary
        """
        return self.storage_coordinator.get_universe_metadata(universe_id)

    def query_charges(
        self,
        text_filter: Optional[str] = None,
        magnitude_range: Optional[tuple] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query charges from stored universes using fast Arrow metadata.

        Args:
            text_filter: Filter by text source content
            magnitude_range: (min_magnitude, max_magnitude) filter
            limit: Maximum number of results

        Returns:
            List of charge metadata dictionaries
        """
        return self.storage_coordinator.query_charges(
            text_filter=text_filter, magnitude_range=magnitude_range, limit=limit
        )

    def get_charge_mathematical_components(self, charge_id: str) -> Dict[str, Any]:
        """
        Retrieve complete mathematical components for a specific charge.

        Uses HDF5 storage to get full precision Q(τ,C,s) components,
        field arrays, and all mathematical data.

        Args:
            charge_id: Charge identifier

        Returns:
            Complete mathematical components dictionary
        """
        return self.storage_coordinator.get_charge_components(charge_id)

    def validate_storage_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of stored universe data.

        Returns:
            Validation results dictionary
        """
        logger.info("🔍 Validating storage integrity...")

        validation_results = self.storage_coordinator.validate_integrity()

        logger.info(f"✅ Storage validation complete")

        return validation_results

    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get storage usage and performance statistics.

        Returns:
            Storage statistics dictionary
        """
        return self.storage_coordinator.get_statistics()
    
    def reconstruct_liquid_universe(self, universe_id: Optional[str] = None, 
                                  device: str = "mps") -> Dict[str, Any]:
        """
        Reconstruct a complete liquid universe from storage.
        
        This is the main high-level interface for universe reconstruction.
        Creates a living LiquidOrchestrator with all agents and field dynamics restored.
        
        Args:
            universe_id: Universe to reconstruct (latest if None)
            device: PyTorch device for reconstruction
            
        Returns:
            Complete reconstruction results with living LiquidOrchestrator
        """
        logger.info("🌊 FIELD UNIVERSE RECONSTRUCTION")
        logger.info(f"   Storage path: {self.config.storage_path}")
        logger.info(f"   Universe ID: {universe_id or 'latest'}")
        logger.info(f"   Target device: {device}")
        
        start_time = time.time()
        
        try:
            # Get universe ID if not specified
            if universe_id is None:
                universes = self.storage_coordinator.list_universes()
                if not universes:
                    raise ValueError("No universes available for reconstruction")
                
                # Get the most recent universe
                universe_id = universes[0]["universe_id"]
                logger.info(f"   Selected latest universe: {universe_id}")
            
            # Create LiquidOrchestrator for reconstruction
            from Sysnpire.model.liquid.liquid_orchestrator import LiquidOrchestrator
            
            logger.info("🏗️  Creating LiquidOrchestrator for reconstruction...")
            orchestrator = LiquidOrchestrator(device=device, field_resolution=256)
            
            # Load universe into orchestrator
            logger.info("🔄 Loading universe into orchestrator...")
            reconstruction_result = orchestrator.load_universe_from_storage(
                storage_coordinator=self.storage_coordinator,
                universe_id=universe_id
            )
            
            if reconstruction_result["status"] != "success":
                raise ValueError(f"Orchestrator reconstruction failed: {reconstruction_result.get('error')}")
            
            total_time = time.time() - start_time
            
            # Complete reconstruction results
            final_result = {
                "status": "success",
                "universe_id": universe_id,
                "reconstruction_time": total_time,
                "orchestrator": orchestrator,  # Living LiquidOrchestrator
                "agents_count": reconstruction_result["agents_reconstructed"],
                "field_energy": reconstruction_result["field_energy"],
                "validation_passed": reconstruction_result["validation_passed"],
                "ready_for_simulation": reconstruction_result["ready_for_simulation"],
                "reconstruction_details": reconstruction_result
            }
            
            logger.info("🎉 FIELD UNIVERSE RECONSTRUCTION COMPLETE")
            logger.info(f"   Total time: {total_time:.2f}s")
            logger.info(f"   Agents: {final_result['agents_count']}")
            logger.info(f"   Field energy: {final_result['field_energy']:.6f}")
            logger.info(f"   Ready for simulation: {final_result['ready_for_simulation']}")
            
            return final_result
            
        except Exception as e:
            error_msg = f"Field universe reconstruction failed: {e}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "failed",
                "error": error_msg,
                "universe_id": universe_id,
                "reconstruction_time": time.time() - start_time
            }


# Convenience functions for direct usage
def burn_liquid_universe(
    liquid_results: Dict[str, Any], storage_path: Union[str, Path], **config_kwargs
) -> Dict[str, Any]:
    """
    Convenience function to burn a liquid universe directly.

    Args:
        liquid_results: Liquid universe results from LiquidOrchestrator
        storage_path: Path for persistent storage
        **config_kwargs: Additional configuration options

    Returns:
        Burning results dictionary
    """
    config = FieldUniverseConfig(storage_path=Path(storage_path), **config_kwargs)

    universe = FieldUniverse(config)
    return universe.burn_liquid_universe(liquid_results)


def reconstruct_liquid_universe(
    storage_path: Union[str, Path], universe_id: Optional[str] = None, **config_kwargs
) -> Dict[str, Any]:
    """
    Convenience function to reconstruct a liquid universe directly.

    Args:
        storage_path: Path to persistent storage
        universe_id: Optional specific universe to reconstruct
        **config_kwargs: Additional configuration options

    Returns:
        Reconstruction results dictionary
    """
    config = FieldUniverseConfig(storage_path=Path(storage_path), **config_kwargs)

    universe = FieldUniverse(config)
    return universe.reconstruct_liquid_universe(universe_id)


if __name__ == "__main__":
    # Example usage
    print("Field Universe - Hybrid Storage Orchestrator")
    print("Ready for liquid universe burning and reconstruction")
