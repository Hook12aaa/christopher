"""
Universe Reconstructor - Main Reconstruction Engine

Reconstructs living liquid universes from persistent storage, restoring
complete mathematical components and field dynamics with dimension-agnostic
processing.

Key Features:
- Complete liquid universe reconstruction
- Mathematical precision restoration
- Dynamic dimensionality handling
- Field dynamics restoration
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class UniverseReconstructor:
    """
    Main orchestrator for liquid universe reconstruction.

    Converts persistent storage back into living liquid universe
    with all mathematical components and field dynamics restored.
    """

    def __init__(self, storage_coordinator: "StorageCoordinator", dimension_agnostic: bool = True):
        """
        Initialize universe reconstructor.

        Args:
            storage_coordinator: Hybrid storage coordinator
            dimension_agnostic: Support dynamic dimensionality
        """
        self.storage_coordinator = storage_coordinator
        self.dimension_agnostic = dimension_agnostic

        logger.info("UniverseReconstructor initialized")
        logger.info(f"  Dimension-agnostic: {dimension_agnostic}")

    def reconstruct_universe(self, universe_id: Optional[str] = None, 
                           device: str = "mps") -> Dict[str, Any]:
        """
        Reconstruct a liquid universe from storage with complete living agents.

        Args:
            universe_id: Universe to reconstruct (latest if None)
            device: PyTorch device for reconstruction

        Returns:
            Complete reconstructed liquid universe with living LiquidOrchestrator
        """
        if universe_id is None:
            # Get latest universe
            universes = self.storage_coordinator.list_universes()
            if not universes:
                raise ValueError("No universes available for reconstruction")
            universe_id = universes[0]["universe_id"]

        logger.info(f"🔄 Reconstructing universe: {universe_id}")
        logger.info(f"   Target device: {device}")
        start_time = time.time()

        try:
            # Create LiquidOrchestrator for reconstruction
            from Sysnpire.model.liquid.liquid_orchestrator import LiquidOrchestrator
            
            orchestrator = LiquidOrchestrator(device=device, field_resolution=256)
            
            # Use orchestrator's reconstruction method
            reconstruction_result = orchestrator.load_universe_from_storage(
                storage_coordinator=self.storage_coordinator,
                universe_id=universe_id
            )
            
            if reconstruction_result["status"] != "success":
                raise ValueError(f"Reconstruction failed: {reconstruction_result.get('error')}")
            
            reconstruction_time = time.time() - start_time
            
            logger.info(f"🔄 Universe reconstruction complete in {reconstruction_time:.2f}s")
            logger.info(f"   Agents reconstructed: {reconstruction_result['agents_reconstructed']}")
            logger.info(f"   Field energy: {reconstruction_result['field_energy']:.6f}")

            return {
                "status": "success",
                "universe_id": universe_id,
                "reconstruction_time": reconstruction_time,
                "orchestrator": orchestrator,  # Living LiquidOrchestrator
                "agents_count": reconstruction_result["agents_reconstructed"],
                "field_energy": reconstruction_result["field_energy"],
                "validation_passed": reconstruction_result["validation_passed"],
                "ready_for_simulation": reconstruction_result["ready_for_simulation"],
                "reconstruction_details": reconstruction_result
            }
            
        except Exception as e:
            error_msg = f"Universe reconstruction failed: {e}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "failed",
                "error": error_msg,
                "universe_id": universe_id,
                "reconstruction_time": time.time() - start_time
            }


if __name__ == "__main__":
    print("UniverseReconstructor ready for liquid universe restoration")
