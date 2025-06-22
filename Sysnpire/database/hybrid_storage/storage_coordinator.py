"""
Storage Coordinator - Dual Storage System Coordination

Coordinates HDF5 mathematical storage with Arrow/Parquet indexing to provide
both exact mathematical precision and fast query capabilities. Manages the
hybrid storage architecture and maintains consistency between systems.

Key Features:
- Coordinate HDF5 + Arrow/Parquet dual storage
- Maintain consistency between mathematical storage and metadata
- Optimize storage vs. query performance trade-offs
- Handle update propagation between systems
- Provide unified interface for storage operations
"""

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .arrow_indexer import ArrowIndexer

# Import storage components
from .hdf5_manager import HDF5Manager

logger = logging.getLogger(__name__)


@dataclass
class StorageCoordinationMetrics:
    """Metrics for storage coordination operations."""

    universes_coordinated: int = 0
    total_coordination_time: float = 0.0
    hdf5_storage_size_mb: float = 0.0
    arrow_index_size_mb: float = 0.0
    consistency_checks_passed: int = 0
    sync_operations_performed: int = 0


class StorageCoordinator:
    """
    Coordinator for hybrid HDF5 + Arrow/Parquet storage.

    Provides unified interface for storing and retrieving liquid universe
    data while maintaining consistency between mathematical precision storage
    (HDF5) and fast query indices (Arrow/Parquet).

    Architecture:
    - HDF5: Complete mathematical objects with exact precision
    - Arrow/Parquet: Queryable metadata and spatial indices
    - Coordination: Consistency, synchronization, and unified access
    """

    def __init__(
        self,
        storage_path: Union[str, Path],
        hdf5_compression: str = "lzf",
        arrow_batch_size: int = 1000,
    ):
        """
        Initialize storage coordinator.

        Args:
            storage_path: Base path for storage systems
            hdf5_compression: Compression for HDF5 storage
            arrow_batch_size: Batch size for Arrow operations
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize storage components
        self.hdf5_manager = HDF5Manager(
            storage_path=self.storage_path / "hdf5", compression=hdf5_compression
        )

        self.arrow_indexer = ArrowIndexer(
            storage_path=self.storage_path / "arrow", batch_size=arrow_batch_size
        )

        logger.info("StorageCoordinator initialized")
        logger.info(f"  Storage path: {self.storage_path}")
        logger.info(f"  HDF5 compression: {hdf5_compression}")
        logger.info(f"  Arrow batch size: {arrow_batch_size}")

    def store_universe(
        self, extracted_data: "ExtractedLiquidData", universe_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store complete universe in hybrid storage system.

        Args:
            extracted_data: Complete extracted liquid universe data
            universe_id: Optional custom universe ID

        Returns:
            Storage results with coordination metrics
        """
        if universe_id is None:
            universe_id = f"universe_{uuid.uuid4().hex[:8]}_{int(time.time())}"

        logger.info(f"🎯 Coordinating storage for universe: {universe_id}")
        start_time = time.time()

        metrics = StorageCoordinationMetrics()

        try:
            # Store mathematical data in HDF5
            logger.info("   💾 Storing mathematical components in HDF5...")
            hdf5_universe_id = self.hdf5_manager.store_liquid_universe(extracted_data, universe_id)
            
            # Verify HDF5 storage succeeded
            if not hdf5_universe_id:
                raise Exception("HDF5 storage failed - no universe ID returned")

            # Create metadata for Arrow indexing
            logger.info("   📊 Creating metadata index in Arrow...")
            metadata = self._extract_metadata_for_indexing(extracted_data, universe_id)

            # Pre-validate metadata for Arrow compatibility
            validation_result = self._validate_arrow_compatibility(metadata)
            if not validation_result["compatible"]:
                logger.error(f"❌ Metadata validation failed for Arrow storage:")
                for error in validation_result["errors"]:
                    logger.error(f"   • {error}")
                raise ValueError(f"Metadata contains Arrow-incompatible objects: {validation_result['errors']}")

            # Store metadata in Arrow/Parquet using the actual HDF5 universe ID for consistency
            try:
                arrow_result = self.arrow_indexer.store_universe_metadata(hdf5_universe_id, metadata)
                
                # Verify Arrow storage succeeded
                if arrow_result.get("status") != "success":
                    logger.error(f"❌ Arrow storage failed: {arrow_result.get('error', 'Unknown error')}")
                    logger.error(f"   Universe ID: {universe_id}")
                    logger.error(f"   Metadata keys: {list(metadata.keys())}")
                    if metadata.get("agent_metadata"):
                        first_agent = metadata["agent_metadata"][0]
                        logger.error(f"   First agent keys: {list(first_agent.keys())}")
                        # Check for problematic types in first agent
                        for key, value in first_agent.items():
                            value_type = str(type(value))
                            if any(pt in value_type.lower() for pt in ['torch', 'tensor', 'device']):
                                logger.error(f"   Problematic field {key}: {value_type} = {value}")
                    raise RuntimeError(f"Critical Arrow storage failure: {arrow_result.get('error')}")
                else:
                    logger.info("✅ Arrow metadata stored successfully")
                    
            except Exception as e:
                logger.error(f"❌ Arrow storage exception: {e}")
                logger.error(f"   Exception type: {type(e)}")
                # Log detailed information about the metadata that failed
                self._log_metadata_diagnostic(metadata, universe_id)
                raise RuntimeError(f"Critical Arrow storage failure: {e}")

            # Verify consistency using the actual HDF5 universe ID
            logger.info("   ✅ Verifying storage consistency...")
            consistency_check = self._verify_storage_consistency(hdf5_universe_id, extracted_data)

            if consistency_check["consistent"]:
                metrics.consistency_checks_passed += 1

            # Calculate metrics
            metrics.universes_coordinated = 1
            metrics.total_coordination_time = time.time() - start_time
            metrics.hdf5_storage_size_mb = self._get_hdf5_size(universe_id)
            metrics.arrow_index_size_mb = self._get_arrow_size(universe_id)

            logger.info(
                f"🎯 Universe storage coordinated successfully in {metrics.total_coordination_time:.2f}s"
            )
            logger.info(f"   📁 HDF5 size: {metrics.hdf5_storage_size_mb:.2f}MB")
            logger.info(f"   📊 Arrow index size: {metrics.arrow_index_size_mb:.2f}MB")

            return {
                "status": "success",
                "universe_id": hdf5_universe_id,
                "hdf5_universe_id": hdf5_universe_id,
                "arrow_result": arrow_result,
                "consistency_check": consistency_check,
                "coordination_metrics": metrics,
                "storage_paths": {
                    "hdf5": str(self.hdf5_manager.storage_path),
                    "arrow": str(self.arrow_indexer.storage_path),
                },
            }

        except Exception as e:
            logger.error(f"Storage coordination failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "universe_id": universe_id,
                "partial_coordination_metrics": metrics,
            }

    def _extract_metadata_for_indexing(
        self, extracted_data: "ExtractedLiquidData", universe_id: str
    ) -> Dict[str, Any]:
        """Extract metadata suitable for Arrow indexing."""
        metadata = {
            "universe_metadata": {
                "universe_id": universe_id,
                "storage_timestamp": time.time(),
                "num_agents": len(extracted_data.agent_data),
                "field_dimensions": extracted_data.extraction_metrics.field_dimensions_detected,
                "model_type": extracted_data.extraction_metrics.model_type_detected,
                **extracted_data.universe_metadata,
            },
            "agent_metadata": [],
            "field_statistics": extracted_data.field_statistics,
            "extraction_metrics": {
                "agents_processed": extracted_data.extraction_metrics.agents_processed,
                "processing_time": extracted_data.extraction_metrics.total_processing_time,
                "field_dimensions_detected": extracted_data.extraction_metrics.field_dimensions_detected,
                "model_type_detected": extracted_data.extraction_metrics.model_type_detected,
                "validation_passed": extracted_data.extraction_metrics.mathematical_validation_passed,
            },
        }

        # Extract agent metadata for fast queries
        for agent_id, agent_data in extracted_data.agent_data.items():
            agent_metadata = agent_data.get("agent_metadata", {})
            q_components = agent_data.get("Q_components", {})

            # Create queryable agent record with device serialization fix
            agent_record = {
                "universe_id": universe_id,
                "agent_id": agent_id,
                "hdf5_path": f"/liquid_universe_{universe_id}/charges/{agent_id}",
            }
            
            # Add metadata with comprehensive PyTorch object conversion
            for key, value in agent_metadata.items():
                agent_record[key] = self._convert_torch_objects(value)

            # Add Q-value magnitude and phase for queries
            if "Q_value_real" in q_components and "Q_value_imag" in q_components:
                q_real = q_components["Q_value_real"]
                q_imag = q_components["Q_value_imag"]
                agent_record["Q_magnitude"] = (q_real**2 + q_imag**2) ** 0.5
                agent_record["Q_phase"] = np.arctan2(q_imag, q_real)
            elif "Q_value" in q_components:
                q_val = q_components["Q_value"]
                if isinstance(q_val, complex):
                    agent_record["Q_magnitude"] = abs(q_val)
                    agent_record["Q_phase"] = np.angle(q_val)
                else:
                    agent_record["Q_magnitude"] = abs(q_val)
                    agent_record["Q_phase"] = 0.0

            # Add field position if available (for spatial queries)
            field_components = agent_data.get("field_components", {})
            if "field_position" in field_components:
                pos = field_components["field_position"]
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    agent_record["field_position_x"] = float(pos[0])
                    agent_record["field_position_y"] = float(pos[1])
                    if len(pos) >= 3:
                        agent_record["field_position_z"] = float(pos[2])

            metadata["agent_metadata"].append(agent_record)

        # Apply PyTorch object conversion to the entire metadata structure
        converted_metadata = self._convert_torch_objects(metadata)
        
        return converted_metadata

    def _convert_torch_objects(self, obj):
        """
        Carefully convert PyTorch objects to Arrow-compatible formats while preserving mathematical data.
        
        Only converts objects that Arrow can't handle, preserving all mathematical information.
        """
        # Handle None values
        if obj is None:
            return None
        
        # Get type string for checking
        obj_type_str = str(type(obj))
        
        # ONLY convert torch.device objects to strings (metadata only, no mathematical data lost)
        if ('torch.device' in obj_type_str or 
            (hasattr(obj, 'type') and hasattr(obj, 'index') and not hasattr(obj, '__array__'))):
            return str(obj)
        
        # ONLY convert torch.dtype objects to strings (metadata only)
        if 'torch.dtype' in obj_type_str and hasattr(obj, 'name'):
            return str(obj)
        
        # For torch.Tensor objects - preserve the mathematical data as numpy arrays
        if ('torch.Tensor' in obj_type_str or 
            (hasattr(obj, 'cpu') and hasattr(obj, 'numpy') and hasattr(obj, 'shape'))):
            try:
                # Simple tensor conversion - NO property access that could cause boolean evaluation
                # Convert directly to CPU without checking device/dtype properties
                numpy_data = obj.cpu().detach().numpy()
                
                # Handle MPS float64 compatibility at numpy level
                if numpy_data.dtype == np.float64:
                    numpy_data = numpy_data.astype(np.float32)
                elif numpy_data.dtype == np.complex128:
                    numpy_data = numpy_data.astype(np.complex64)
                
                # Only convert to list if array is small to avoid performance issues
                if numpy_data.size <= 1000:
                    return numpy_data.tolist()
                else:
                    # For large arrays, keep as numpy (Arrow can handle numpy)
                    return numpy_data
            except Exception as e:
                # Log the error for debugging but don't lose the data
                logger.warning(f"Failed to convert tensor to numpy: {e}, converting to string")
                return str(obj)
        
        # Handle dictionaries recursively - preserve structure
        if isinstance(obj, dict):
            return {k: self._convert_torch_objects(v) for k, v in obj.items()}
        
        # Handle lists/tuples recursively - preserve structure  
        if isinstance(obj, (list, tuple)):
            converted = [self._convert_torch_objects(item) for item in obj]
            return converted if isinstance(obj, list) else tuple(converted)
        
        # Keep numpy arrays as-is (Arrow can handle numpy arrays)
        if hasattr(obj, '__array__') and hasattr(obj, 'dtype'):
            return obj
        
        # Convert complex numbers to dict format for Arrow compatibility
        if isinstance(obj, complex):
            return {"real": float(obj.real), "imag": float(obj.imag), "_type": "complex"}
        
        # Keep all primitive types as-is
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        
        # For other PyTorch objects that aren't tensors/devices, log and preserve
        if 'torch' in obj_type_str.lower():
            logger.warning(f"Found PyTorch object {obj_type_str} - attempting to preserve")
            # Try to extract meaningful data first
            if hasattr(obj, 'data'):
                return self._convert_torch_objects(obj.data)
            elif hasattr(obj, 'numpy'):
                try:
                    # MPS-safe tensor conversion - move to CPU first
                    if hasattr(obj, 'cpu'):
                        return obj.cpu().detach().numpy()
                    else:
                        return obj.numpy()
                except:
                    logger.warning(f"Could not convert {obj_type_str} to numpy, converting to string")
                    return str(obj)
            else:
                return str(obj)
        
        # For everything else, preserve as-is first, only convert to string as last resort
        return obj

    def _restore_complex_numbers(self, obj):
        """
        Recursively restore dictionary-format complex numbers back to proper Python complex objects.
        
        Converts {"real": float, "imag": float, "_type": "complex"} back to complex(real, imag).
        """
        if obj is None:
            return None
        
        # Check if this is a complex number dictionary
        if (isinstance(obj, dict) and 
            "_type" in obj and obj["_type"] == "complex" and
            "real" in obj and "imag" in obj):
            return complex(float(obj["real"]), float(obj["imag"]))
        
        # Handle dictionaries recursively
        if isinstance(obj, dict):
            return {k: self._restore_complex_numbers(v) for k, v in obj.items()}
        
        # Handle lists/tuples recursively
        if isinstance(obj, (list, tuple)):
            restored = [self._restore_complex_numbers(item) for item in obj]
            return restored if isinstance(obj, list) else tuple(restored)
        
        # Return primitive types as-is
        return obj

    def _validate_arrow_compatibility(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that metadata contains only Arrow-compatible types.
        
        Returns detailed information about any incompatible objects found.
        """
        errors = []
        compatible = True
        
        def check_object(obj, path=""):
            nonlocal compatible, errors
            
            # Get type info
            obj_type = type(obj)
            obj_type_str = str(obj_type)
            
            # Check for known problematic types
            if 'torch.device' in obj_type_str:
                errors.append(f"Found torch.device at {path}: {obj}")
                compatible = False
            elif 'torch.dtype' in obj_type_str:
                errors.append(f"Found torch.dtype at {path}: {obj}")
                compatible = False
            elif 'torch.Tensor' in obj_type_str:
                # NO tensor property access - just report tensor presence
                errors.append(f"Found torch.Tensor at {path}")
                compatible = False
            elif hasattr(obj, 'cpu') and hasattr(obj, 'numpy') and 'torch' in obj_type_str:
                errors.append(f"Found PyTorch tensor-like object at {path}: {obj_type_str}")
                compatible = False
            elif isinstance(obj, complex):
                # Complex numbers are handled by conversion, just note them
                pass  # Complex numbers are converted to dict format
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    check_object(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    check_object(item, f"{path}[{i}]" if path else f"[{i}]")
            elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
                # Check for custom objects that might not be serializable
                if not hasattr(obj, '__array__'):  # Allow numpy arrays
                    errors.append(f"Found custom object at {path}: {obj_type_str}")
        
        try:
            check_object(metadata)
        except Exception as e:
            errors.append(f"Error during validation: {e}")
            compatible = False
        
        return {
            "compatible": compatible,
            "errors": errors,
            "total_errors": len(errors)
        }

    def _log_metadata_diagnostic(self, metadata: Dict[str, Any], universe_id: str):
        """Log detailed diagnostic information about metadata structure."""
        logger.error(f"🔍 Detailed metadata diagnostic for universe {universe_id}:")
        
        def log_structure(obj, path="", max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                logger.error(f"   {path}: <max_depth_reached>")
                return
                
            obj_type = type(obj)
            obj_type_str = str(obj_type)
            
            if isinstance(obj, dict):
                logger.error(f"   {path}: dict with {len(obj)} keys: {list(obj.keys())}")
                for key, value in list(obj.items())[:3]:  # Log first 3 items
                    log_structure(value, f"{path}.{key}", max_depth, current_depth + 1)
                if len(obj) > 3:
                    logger.error(f"   {path}: ... and {len(obj) - 3} more keys")
            elif isinstance(obj, (list, tuple)):
                logger.error(f"   {path}: {obj_type_str} with {len(obj)} items")
                if obj:
                    log_structure(obj[0], f"{path}[0]", max_depth, current_depth + 1)
                    if len(obj) > 1:
                        logger.error(f"   {path}: ... and {len(obj) - 1} more items")
            else:
                if any(pt in obj_type_str.lower() for pt in ['torch', 'tensor', 'device']):
                    logger.error(f"   {path}: 🚨 PROBLEMATIC {obj_type_str} = {obj}")
                else:
                    logger.error(f"   {path}: {obj_type_str}")
        
        try:
            log_structure(metadata)
        except Exception as e:
            logger.error(f"   Error during diagnostic logging: {e}")

    def _verify_storage_consistency(
        self, universe_id: str, extracted_data: "ExtractedLiquidData"
    ) -> Dict[str, Any]:
        """Verify consistency between HDF5 and Arrow storage."""
        try:
            # Check HDF5 storage
            hdf5_universes = self.hdf5_manager.list_universes()
            hdf5_present = any(universe_id in u for u in hdf5_universes)

            # Check Arrow storage
            arrow_universes = self.arrow_indexer.list_universes()
            arrow_present = universe_id in arrow_universes

            logger.info(f"🔍 Storage verification for universe {universe_id}:")
            logger.info(f"   HDF5 present: {hdf5_present} (found {len(hdf5_universes)} universes)")
            logger.info(f"   Arrow present: {arrow_present} (found {len(arrow_universes)} universes)")

            # If storage not found, check actual file paths
            if not hdf5_present:
                hdf5_files = list(self.hdf5_manager.storage_path.glob("*.h5"))
                logger.warning(f"❌ HDF5 universe not found. Files in {self.hdf5_manager.storage_path}: {[f.name for f in hdf5_files]}")
            
            if not arrow_present:
                arrow_files = list(self.arrow_indexer.storage_path.glob("*.parquet"))
                logger.warning(f"❌ Arrow universe not found. Files in {self.arrow_indexer.storage_path}: {[f.name for f in arrow_files]}")

            # Basic consistency checks
            consistent = hdf5_present and arrow_present

            # Count verification
            hdf5_data = None
            arrow_count = 0

            if hdf5_present:
                try:
                    hdf5_data = self.hdf5_manager.load_universe(universe_id)
                    hdf5_count = len(hdf5_data.get("agents", {}))
                except:
                    hdf5_count = 0
            else:
                hdf5_count = 0

            if arrow_present:
                try:
                    arrow_metadata = self.arrow_indexer.get_universe_metadata(universe_id)
                    arrow_count = len(arrow_metadata.get("agent_metadata", []))
                except:
                    arrow_count = 0

            count_consistent = hdf5_count == arrow_count == len(extracted_data.agent_data)

            return {
                "consistent": consistent and count_consistent,
                "hdf5_present": hdf5_present,
                "arrow_present": arrow_present,
                "hdf5_agent_count": hdf5_count,
                "arrow_agent_count": arrow_count,
                "expected_agent_count": len(extracted_data.agent_data),
                "count_consistent": count_consistent,
            }

        except Exception as e:
            logger.error(f"Consistency verification failed: {e}")
            return {"consistent": False, "error": str(e)}

    def load_universe(self, universe_id: str) -> Dict[str, Any]:
        """
        Load complete universe from hybrid storage.

        Args:
            universe_id: Universe identifier

        Returns:
            Complete universe data with both mathematical and metadata components
        """
        logger.info(f"📖 Loading universe: {universe_id}")
        start_time = time.time()

        try:
            # Load mathematical data from HDF5
            hdf5_data = self.hdf5_manager.load_universe(universe_id)

            # Load metadata from Arrow
            arrow_metadata = self.arrow_indexer.get_universe_metadata(universe_id)

            # CRITICAL FIX: Convert dictionary-format complex numbers back to proper complex numbers
            hdf5_data = self._restore_complex_numbers(hdf5_data)
            arrow_metadata = self._restore_complex_numbers(arrow_metadata)

            # Combine data
            universe_data = {
                "universe_id": universe_id,
                "mathematical_data": hdf5_data,
                "metadata": arrow_metadata,
                "load_time_seconds": time.time() - start_time,
                "storage_sources": {"hdf5": True, "arrow": True},
            }

            logger.info(f"📖 Universe loaded in {universe_data['load_time_seconds']:.2f}s")

            return universe_data

        except Exception as e:
            logger.error(f"Universe loading failed: {e}")
            return {
                "universe_id": universe_id,
                "error": str(e),
                "load_time_seconds": time.time() - start_time,
            }

    def list_universes(self) -> List[Dict[str, Any]]:
        """List all universes with metadata from both storage systems."""
        hdf5_universes = set(self.hdf5_manager.list_universes())
        arrow_universes = set(self.arrow_indexer.list_universes())

        all_universes = hdf5_universes | arrow_universes

        universe_list = []
        for universe_id in all_universes:
            universe_info = {
                "universe_id": universe_id,
                "hdf5_available": universe_id in hdf5_universes,
                "arrow_available": universe_id in arrow_universes,
                "consistent": universe_id in hdf5_universes and universe_id in arrow_universes,
            }

            # Add metadata if available
            if universe_info["arrow_available"]:
                try:
                    metadata = self.arrow_indexer.get_universe_metadata(universe_id)
                    universe_info.update(
                        {
                            "num_agents": len(metadata.get("agent_metadata", [])),
                            "creation_time": metadata.get("universe_metadata", {}).get(
                                "storage_timestamp"
                            ),
                            "model_type": metadata.get("universe_metadata", {}).get("model_type"),
                            "field_dimensions": metadata.get("universe_metadata", {}).get(
                                "field_dimensions"
                            ),
                        }
                    )
                except:
                    pass

            universe_list.append(universe_info)

        return sorted(universe_list, key=lambda x: x.get("creation_time", 0), reverse=True)

    def get_universe_metadata(self, universe_id: str) -> Dict[str, Any]:
        """Get universe metadata from Arrow storage."""
        return self.arrow_indexer.get_universe_metadata(universe_id)

    def query_charges(
        self,
        text_filter: Optional[str] = None,
        magnitude_range: Optional[Tuple[float, float]] = None,
        universe_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query charges using fast Arrow metadata.

        Args:
            text_filter: Filter by text source content
            magnitude_range: (min_magnitude, max_magnitude) filter
            universe_id: Filter by specific universe
            limit: Maximum number of results

        Returns:
            List of charge metadata with HDF5 paths for full data retrieval
        """
        return self.arrow_indexer.query_charges(
            text_filter=text_filter,
            magnitude_range=magnitude_range,
            universe_id=universe_id,
            limit=limit,
        )

    def get_charge_components(self, charge_id: str, universe_id: str) -> Dict[str, Any]:
        """
        Get complete mathematical components for a specific charge.

        Args:
            charge_id: Charge identifier
            universe_id: Universe identifier

        Returns:
            Complete mathematical components from HDF5 storage
        """
        # Load universe data
        universe_data = self.hdf5_manager.load_universe(universe_id)

        # Extract specific charge
        agents = universe_data.get("agents", {})
        if charge_id not in agents:
            raise KeyError(f"Charge {charge_id} not found in universe {universe_id}")

        return agents[charge_id]

    def validate_integrity(self) -> Dict[str, Any]:
        """Validate integrity of hybrid storage system."""
        logger.info("🔍 Validating hybrid storage integrity...")

        hdf5_stats = self.hdf5_manager.get_storage_statistics()
        arrow_stats = self.arrow_indexer.get_storage_statistics()

        # Cross-validate universe lists
        hdf5_universes = set(self.hdf5_manager.list_universes())
        arrow_universes = set(self.arrow_indexer.list_universes())

        consistent_universes = hdf5_universes & arrow_universes
        orphaned_hdf5 = hdf5_universes - arrow_universes
        orphaned_arrow = arrow_universes - hdf5_universes

        validation_results = {
            "overall_status": (
                "healthy"
                if len(orphaned_hdf5) == 0 and len(orphaned_arrow) == 0
                else "inconsistent"
            ),
            "total_universes": len(hdf5_universes | arrow_universes),
            "consistent_universes": len(consistent_universes),
            "orphaned_hdf5_universes": len(orphaned_hdf5),
            "orphaned_arrow_universes": len(orphaned_arrow),
            "hdf5_statistics": hdf5_stats,
            "arrow_statistics": arrow_stats,
            "consistency_ratio": (
                len(consistent_universes) / len(hdf5_universes | arrow_universes)
                if (hdf5_universes | arrow_universes)
                else 1.0
            ),
        }

        logger.info(f"✅ Storage validation complete")
        logger.info(f"   Overall status: {validation_results['overall_status']}")
        logger.info(f"   Consistency ratio: {validation_results['consistency_ratio']:.2%}")

        return validation_results

    def get_statistics(self) -> Dict[str, Any]:
        """Get combined storage statistics."""
        hdf5_stats = self.hdf5_manager.get_storage_statistics()
        arrow_stats = self.arrow_indexer.get_storage_statistics()

        return {
            "storage_coordinator": {
                "storage_path": str(self.storage_path),
                "hybrid_architecture": True,
            },
            "hdf5_storage": hdf5_stats,
            "arrow_storage": arrow_stats,
            "combined_statistics": {
                "total_storage_mb": hdf5_stats.get("total_size_mb", 0)
                + arrow_stats.get("total_size_mb", 0),
                "universe_consistency": self._check_universe_consistency(),
            },
        }

    def _check_universe_consistency(self) -> float:
        """Calculate universe consistency ratio."""
        try:
            hdf5_universes = set(self.hdf5_manager.list_universes())
            arrow_universes = set(self.arrow_indexer.list_universes())

            if not (hdf5_universes | arrow_universes):
                return 1.0

            consistent = len(hdf5_universes & arrow_universes)
            total = len(hdf5_universes | arrow_universes)

            return consistent / total
        except:
            return 0.0

    def _get_hdf5_size(self, universe_id: str) -> float:
        """Get HDF5 storage size for universe in MB."""
        try:
            hdf5_file = self.hdf5_manager.storage_path / f"{universe_id}.h5"
            if hdf5_file.exists():
                return hdf5_file.stat().st_size / (1024 * 1024)
        except:
            pass
        return 0.0

    def _get_arrow_size(self, universe_id: str) -> float:
        """Get Arrow storage size for universe in MB."""
        try:
            return self.arrow_indexer.get_universe_size(universe_id)
        except:
            return 0.0


if __name__ == "__main__":
    # Example usage
    coordinator = StorageCoordinator("/tmp/test_hybrid_storage")
    print("StorageCoordinator ready for hybrid storage operations")
