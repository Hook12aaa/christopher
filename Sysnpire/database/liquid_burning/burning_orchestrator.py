"""
Burning Orchestrator - Coordinate Full Burning Process

Orchestrates the complete liquid universe burning process, coordinating
liquid processing, agent serialization, and hybrid storage to convert
living liquid universes into persistent mathematical storage.

Key Features:
- Complete pipeline orchestration from liquid results to storage
- Mathematical validation and precision preservation
- Performance monitoring and optimization
- Error handling and recovery
- Dimension-agnostic processing
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .agent_serializer import AgentSerializer

# Import burning components
from .liquid_processor import ExtractedLiquidData, LiquidProcessor
from .mathematical_validator import MathematicalValidator

logger = logging.getLogger(__name__)


@dataclass
class BurningMetrics:
    """Complete metrics for the burning process."""

    total_burning_time: float = 0.0
    liquid_processing_time: float = 0.0
    storage_coordination_time: float = 0.0
    validation_time: float = 0.0
    agents_burned: int = 0
    mathematical_validation_passed: bool = False
    storage_size_mb: float = 0.0
    compression_achieved: float = 0.0


class BurningOrchestrator:
    """
    Main orchestrator for liquid universe burning process.

    Coordinates the complete pipeline from liquid universe results
    to persistent hybrid storage while preserving mathematical
    accuracy and optimizing performance.

    Process Flow:
    1. Liquid Processing: Extract mathematical components
    2. Mathematical Validation: Verify component integrity
    3. Storage Coordination: Store in hybrid HDF5 + Arrow system
    4. Final Validation: Verify storage consistency
    """

    def __init__(
        self,
        storage_coordinator: "StorageCoordinator",
        enable_validation: bool = True,
        preserve_precision: bool = True,
        optimize_performance: bool = True,
    ):
        """
        Initialize burning orchestrator.

        Args:
            storage_coordinator: Hybrid storage coordinator
            enable_validation: Enable mathematical validation
            preserve_precision: Ensure exact mathematical precision
            optimize_performance: Apply performance optimizations
        """
        self.storage_coordinator = storage_coordinator
        self.enable_validation = enable_validation
        self.preserve_precision = preserve_precision
        self.optimize_performance = optimize_performance

        # Initialize processing components
        self.liquid_processor = LiquidProcessor(
            preserve_precision=preserve_precision, validate_mathematics=enable_validation
        )

        self.agent_serializer = AgentSerializer(
            preserve_precision=preserve_precision, optimize_storage=optimize_performance
        )

        if enable_validation:
            self.mathematical_validator = MathematicalValidator(
                strict_validation=preserve_precision
            )

        logger.info("BurningOrchestrator initialized")
        logger.info(f"  Validation enabled: {enable_validation}")
        logger.info(f"  Precision preservation: {preserve_precision}")
        logger.info(f"  Performance optimization: {optimize_performance}")

    def burn_universe(self, liquid_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Burn complete liquid universe into persistent storage.

        Args:
            liquid_results: Complete liquid universe results from LiquidOrchestrator

        Returns:
            Complete burning results with metrics and validation
        """
        logger.info("🔥 Starting liquid universe burning orchestration...")
        start_time = time.time()

        metrics = BurningMetrics()

        try:
            # STEP 1: Process liquid universe results
            logger.info("🔄 Step 1: Processing liquid universe components...")
            processing_start = time.time()

            extracted_data = self.liquid_processor.process_liquid_universe(liquid_results)

            metrics.liquid_processing_time = time.time() - processing_start
            metrics.agents_burned = extracted_data.extraction_metrics.agents_processed

            logger.info(f"✅ Liquid processing complete in {metrics.liquid_processing_time:.2f}s")
            logger.info(f"   Agents processed: {metrics.agents_burned}")
            logger.info(
                f"   Field dimensions: {extracted_data.extraction_metrics.field_dimensions_detected}"
            )

            # STEP 2: Mathematical validation (if enabled)
            if self.enable_validation:
                logger.info("🔍 Step 2: Validating mathematical components...")
                validation_start = time.time()

                validation_results = self.mathematical_validator.validate_extracted_data(
                    extracted_data
                )

                metrics.validation_time = time.time() - validation_start
                metrics.mathematical_validation_passed = validation_results["validation_passed"]

                logger.info(f"✅ Validation complete in {metrics.validation_time:.2f}s")
                logger.info(f"   Validation passed: {metrics.mathematical_validation_passed}")

                if not metrics.mathematical_validation_passed:
                    logger.warning("⚠️ Mathematical validation failed - proceeding with warnings")
                    logger.warning(f"   Validation errors: {validation_results.get('errors')}")

            # STEP 3: Storage coordination
            logger.info("💾 Step 3: Coordinating hybrid storage...")
            storage_start = time.time()

            storage_results = self.storage_coordinator.store_universe(extracted_data)

            metrics.storage_coordination_time = time.time() - storage_start

            if storage_results["status"] == "success":
                logger.info(
                    f"✅ Storage coordination complete in {metrics.storage_coordination_time:.2f}s"
                )
                logger.info(f"   Universe ID: {storage_results['universe_id']}")

                # Extract storage metrics
                coord_metrics = storage_results.get("coordination_metrics")
                if coord_metrics:
                    metrics.storage_size_mb = (
                        coord_metrics.hdf5_storage_size_mb + coord_metrics.arrow_index_size_mb
                    )
            else:
                logger.error(f"❌ Storage coordination failed: {storage_results.get('error')}")
                return self._create_failure_result(metrics, storage_results.get("error"))

            # STEP 4: Final consistency validation
            logger.info("🔍 Step 4: Final consistency validation...")
            consistency_check = storage_results.get("consistency_check")

            if consistency_check.get("consistent"):
                logger.info("✅ Storage consistency verified")
            else:
                logger.warning("⚠️ Storage consistency issues detected")
                logger.warning(f"   Consistency details: {consistency_check}")

            # Calculate final metrics
            metrics.total_burning_time = time.time() - start_time

            # Calculate compression if available
            if metrics.storage_size_mb > 0 and metrics.agents_burned > 0:
                # Estimate original size (rough approximation)
                estimated_original_mb = metrics.agents_burned * 0.5  # Rough estimate
                metrics.compression_achieved = (
                    estimated_original_mb / metrics.storage_size_mb
                    if metrics.storage_size_mb > 0
                    else 1.0
                )

            logger.info(f"🔥 Liquid universe burning orchestration complete!")
            logger.info(f"   Total time: {metrics.total_burning_time:.2f}s")
            logger.info(f"   Storage size: {metrics.storage_size_mb:.2f}MB")
            logger.info(f"   Agents burned: {metrics.agents_burned}")

            return {
                "status": "success",
                "universe_id": storage_results["universe_id"],
                "burning_metrics": metrics,
                "extraction_metrics": extracted_data.extraction_metrics,
                "storage_results": storage_results,
                "validation_results": validation_results if self.enable_validation else None,
                "consistency_check": consistency_check,
                "performance_summary": {
                    "total_time_seconds": metrics.total_burning_time,
                    "processing_time_seconds": metrics.liquid_processing_time,
                    "storage_time_seconds": metrics.storage_coordination_time,
                    "validation_time_seconds": metrics.validation_time,
                    "agents_per_second": (
                        metrics.agents_burned / metrics.total_burning_time
                        if metrics.total_burning_time > 0
                        else 0
                    ),
                    "storage_efficiency_mb_per_agent": (
                        metrics.storage_size_mb / metrics.agents_burned
                        if metrics.agents_burned > 0
                        else 0
                    ),
                },
            }

        except Exception as e:
            logger.error(f"🔥 Burning orchestration failed: {e}")
            metrics.total_burning_time = time.time() - start_time
            return self._create_failure_result(metrics, str(e))

    def _create_failure_result(self, metrics: BurningMetrics, error_message: str) -> Dict[str, Any]:
        """Create failure result with partial metrics."""
        return {
            "status": "failed",
            "error": error_message,
            "partial_metrics": metrics,
            "failure_time_seconds": metrics.total_burning_time,
            "agents_processed_before_failure": metrics.agents_burned,
        }

    def estimate_burning_requirements(self, liquid_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate storage requirements and processing time for burning.

        Args:
            liquid_results: Liquid universe results to analyze

        Returns:
            Estimation results
        """
        num_agents = liquid_results.get("num_agents")
        field_stats = liquid_results.get("field_statistics")

        # Rough estimation based on agent count and field complexity
        estimated_processing_time = num_agents * 0.01  # ~10ms per agent
        estimated_storage_mb = num_agents * 0.5  # ~500KB per agent

        # Adjust for field complexity
        field_energy = field_stats.get("field_energy")
        if field_energy > 0.01:  # High field energy
            estimated_processing_time *= 1.5
            estimated_storage_mb *= 1.2

        return {
            "estimated_processing_time_seconds": estimated_processing_time,
            "estimated_storage_size_mb": estimated_storage_mb,
            "estimated_agents_per_second": (
                num_agents / estimated_processing_time if estimated_processing_time > 0 else 0
            ),
            "recommendations": {
                "enable_compression": estimated_storage_mb > 100,
                "use_batch_processing": num_agents > 1000,
                "parallel_processing": num_agents > 5000,
            },
        }

    def get_burning_statistics(self) -> Dict[str, Any]:
        """Get statistics for burning operations."""
        storage_stats = self.storage_coordinator.get_statistics()

        return {
            "burning_orchestrator": {
                "validation_enabled": self.enable_validation,
                "precision_preservation": self.preserve_precision,
                "performance_optimization": self.optimize_performance,
            },
            "storage_statistics": storage_stats,
            "component_status": {
                "liquid_processor": "active",
                "agent_serializer": "active",
                "mathematical_validator": "active" if self.enable_validation else "disabled",
                "storage_coordinator": "active",
            },
        }


if __name__ == "__main__":
    # Example usage
    print("BurningOrchestrator ready for liquid universe burning")
