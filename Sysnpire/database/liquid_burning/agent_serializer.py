"""
Agent Serializer - ConceptualChargeAgent → HDF5 Serialization

Serializes ConceptualChargeAgent objects into HDF5 hierarchical structure
with flexible dimensions. Preserves complete mathematical components
without any hardcoded dimensional assumptions.

Key Features:
- Dynamic dimension detection and handling
- Complex number precision preservation
- Hierarchical structure matching agent organization
- Efficient tensor compression for large field arrays
- Model-agnostic serialization (BGE, MPNet, future models)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SerializationMetrics:
    """Metrics for agent serialization performance."""

    agents_serialized: int = 0
    total_serialization_time: float = 0.0
    field_components_serialized: int = 0
    complex_numbers_preserved: int = 0
    tensor_arrays_compressed: int = 0
    average_compression_ratio: float = 0.0


class AgentSerializer:
    """
    Serializer for ConceptualChargeAgent objects to HDF5 format.

    Handles the complete serialization of liquid universe agents into
    HDF5 hierarchical structure while preserving all mathematical
    precision and supporting dynamic field dimensionalities.

    Serialization Structure:
    agent_id/
    ├── metadata/                  # Agent identification and timestamps
    ├── Q_components/             # Complete Q(τ,C,s) mathematics
    │   ├── Q_value              # Complex Q value
    │   ├── gamma                # Field coupling strength
    │   ├── T_tensor             # Trajectory tensor (complex)
    │   ├── E_trajectory         # Emotional trajectory (complex)
    │   ├── phi_semantic         # Semantic phase
    │   ├── theta_components/    # Phase components
    │   ├── phase_factor         # Overall phase factor
    │   └── psi_persistence      # Persistence factor
    ├── field_components/        # Dynamic-dimensional field arrays
    │   ├── semantic_embedding   # Main semantic field (N-dimensional)
    │   ├── semantic_phase_factors
    │   ├── emotional_trajectory
    │   └── trajectory_operators
    ├── temporal_biography/      # Time evolution data
    │   ├── trajectory_operators
    │   ├── vivid_layer
    │   ├── character_layer
    │   ├── frequency_evolution
    │   ├── phase_coordination
    │   ├── temporal_momentum
    │   └── breathing_coherence
    ├── emotional_modulation/    # Emotional field data
    │   ├── semantic_modulation_tensor
    │   ├── unified_phase_shift
    │   ├── trajectory_attractors
    │   ├── resonance_frequencies
    │   └── field_modulation_strength
    └── agent_state/            # Complete agent state
        ├── evolution_parameters
        ├── breathing_coefficients
        └── interaction_memory
    """

    def __init__(
        self,
        compression: str = "lzf",
        preserve_precision: bool = True,
        optimize_storage: bool = True,
    ):
        """
        Initialize agent serializer.

        Args:
            compression: HDF5 compression algorithm
            preserve_precision: Ensure exact mathematical precision
            optimize_storage: Apply storage optimizations
        """
        self.compression = compression
        self.preserve_precision = preserve_precision
        self.optimize_storage = optimize_storage

        logger.info("AgentSerializer initialized")
        logger.info(f"  Compression: {compression}")
        logger.info(f"  Precision preservation: {preserve_precision}")
        logger.info(f"  Storage optimization: {optimize_storage}")

    def serialize_agents(
        self, agents_data: Dict[str, Dict[str, Any]], hdf5_group: h5py.Group
    ) -> SerializationMetrics:
        """
        Serialize multiple agents to HDF5 group.

        Args:
            agents_data: Dictionary mapping agent IDs to extracted agent data
            hdf5_group: HDF5 group to store agents in

        Returns:
            SerializationMetrics with performance data
        """
        logger.info(f"🔗 Serializing {len(agents_data)} agents to HDF5...")
        start_time = time.time()

        metrics = SerializationMetrics()

        for agent_id, agent_data in agents_data.items():
            try:
                self._serialize_single_agent(agent_id, agent_data, hdf5_group, metrics)
                metrics.agents_serialized += 1

            except Exception as e:
                logger.error(f"Failed to serialize agent {agent_id}: {e}")
                continue

        metrics.total_serialization_time = time.time() - start_time

        if metrics.tensor_arrays_compressed > 0:
            metrics.average_compression_ratio = (
                metrics.average_compression_ratio / metrics.tensor_arrays_compressed
            )

        logger.info(f"🔗 Agent serialization complete in {metrics.total_serialization_time:.2f}s")
        logger.info(f"   ✅ Agents serialized: {metrics.agents_serialized}")
        logger.info(f"   🧮 Complex numbers preserved: {metrics.complex_numbers_preserved}")
        logger.info(f"   📊 Tensor arrays compressed: {metrics.tensor_arrays_compressed}")

        return metrics

    def _serialize_single_agent(
        self,
        agent_id: str,
        agent_data: Dict[str, Any],
        hdf5_group: h5py.Group,
        metrics: SerializationMetrics,
    ):
        """Serialize a single agent to HDF5."""
        # Create agent group
        agent_group = hdf5_group.create_group(agent_id)

        # Serialize each component
        self._serialize_agent_metadata(agent_group, agent_data.get("agent_metadata"), metrics)
        self._serialize_q_components(agent_group, agent_data.get("Q_components"), metrics)
        self._serialize_field_components(
            agent_group, agent_data.get("field_components"), metrics
        )
        self._serialize_temporal_components(
            agent_group, agent_data.get("temporal_components"), metrics
        )
        self._serialize_emotional_components(
            agent_group, agent_data.get("emotional_components"), metrics
        )
        self._serialize_agent_state(agent_group, agent_data.get("agent_state"), metrics)

    def _serialize_agent_metadata(
        self, agent_group: h5py.Group, metadata: Dict[str, Any], metrics: SerializationMetrics
    ):
        """Serialize agent metadata."""
        metadata_group = agent_group.create_group("metadata")

        for key, value in metadata.items():
            try:
                if isinstance(value, (str, int, float, bool)):
                    metadata_group.attrs[key] = value
                elif value is not None:
                    metadata_group.attrs[key] = str(value)
            except Exception as e:
                logger.warning(f"Could not serialize metadata {key}: {e}")

    def _serialize_q_components(
        self, agent_group: h5py.Group, q_components: Dict[str, Any], metrics: SerializationMetrics
    ):
        """Serialize Q(τ,C,s) mathematical components."""
        if not q_components:
            return

        q_group = agent_group.create_group("Q_components")

        for key, value in q_components.items():
            try:
                if isinstance(value, complex) or (
                    isinstance(value, (int, float)) and "complex" in key.lower()
                ):
                    # Handle complex numbers with exact precision
                    self._store_complex_number(q_group, key, value, metrics)

                elif key.endswith("_real") or key.endswith("_imag"):
                    # These are components of complex numbers, store as float
                    q_group.create_dataset(key, data=float(value), compression=self.compression)

                elif isinstance(value, np.ndarray):
                    # Store numpy arrays with optimal chunking
                    self._store_array_data(q_group, key, value, metrics)

                elif isinstance(value, (list, tuple)):
                    # Convert to numpy and store
                    array_data = np.array(value)
                    self._store_array_data(q_group, key, array_data, metrics)

                elif isinstance(value, (int, float)):
                    q_group.create_dataset(key, data=value, compression=self.compression)

                else:
                    # Store as attribute for simple reconstruction
                    q_group.attrs[key] = str(value)

            except Exception as e:
                logger.warning(f"Could not serialize Q component {key}: {e}")

    def _serialize_field_components(
        self,
        agent_group: h5py.Group,
        field_components: Dict[str, Any],
        metrics: SerializationMetrics,
    ):
        """Serialize field components with dynamic dimensionality."""
        if not field_components:
            return

        field_group = agent_group.create_group("field_components")

        for key, value in field_components.items():
            try:
                if isinstance(value, np.ndarray):
                    # Dynamic field arrays - preserve exact dimensions
                    self._store_field_array(field_group, key, value, metrics)

                elif isinstance(value, (list, tuple)):
                    # Convert to numpy array and store
                    array_data = np.array(value)
                    self._store_field_array(field_group, key, array_data, metrics)

                elif isinstance(value, (int, float, complex)):
                    if isinstance(value, complex):
                        self._store_complex_number(field_group, key, value, metrics)
                    else:
                        field_group.create_dataset(key, data=value, compression=self.compression)

                else:
                    field_group.attrs[key] = str(value)

            except Exception as e:
                logger.warning(f"Could not serialize field component {key}: {e}")

        metrics.field_components_serialized += len(field_components)

    def _serialize_temporal_components(
        self,
        agent_group: h5py.Group,
        temporal_components: Dict[str, Any],
        metrics: SerializationMetrics,
    ):
        """Serialize temporal biography and evolution data."""
        if not temporal_components:
            return

        temporal_group = agent_group.create_group("temporal_biography")

        for key, value in temporal_components.items():
            try:
                if isinstance(value, np.ndarray):
                    self._store_array_data(temporal_group, key, value, metrics)

                elif isinstance(value, complex) or key.endswith("_real") or key.endswith("_imag"):
                    if isinstance(value, complex):
                        self._store_complex_number(temporal_group, key, value, metrics)
                    else:
                        temporal_group.create_dataset(
                            key, data=float(value), compression=self.compression
                        )

                elif isinstance(value, (list, tuple)):
                    array_data = np.array(value)
                    self._store_array_data(temporal_group, key, array_data, metrics)

                elif isinstance(value, (int, float)):
                    temporal_group.create_dataset(key, data=value, compression=self.compression)

                else:
                    temporal_group.attrs[key] = str(value)

            except Exception as e:
                logger.warning(f"Could not serialize temporal component {key}: {e}")

    def _serialize_emotional_components(
        self,
        agent_group: h5py.Group,
        emotional_components: Dict[str, Any],
        metrics: SerializationMetrics,
    ):
        """Serialize emotional modulation and field signature data."""
        if not emotional_components:
            return

        emotional_group = agent_group.create_group("emotional_modulation")

        for key, value in emotional_components.items():
            try:
                if isinstance(value, np.ndarray):
                    self._store_array_data(emotional_group, key, value, metrics)

                elif isinstance(value, complex) or key.endswith("_real") or key.endswith("_imag"):
                    if isinstance(value, complex):
                        self._store_complex_number(emotional_group, key, value, metrics)
                    else:
                        emotional_group.create_dataset(
                            key, data=float(value), compression=self.compression
                        )

                elif isinstance(value, (list, tuple)):
                    array_data = np.array(value)
                    self._store_array_data(emotional_group, key, array_data, metrics)

                elif isinstance(value, (int, float)):
                    emotional_group.create_dataset(key, data=value, compression=self.compression)

                else:
                    emotional_group.attrs[key] = str(value)

            except Exception as e:
                logger.warning(f"Could not serialize emotional component {key}: {e}")

    def _serialize_agent_state(
        self, agent_group: h5py.Group, agent_state: Dict[str, Any], metrics: SerializationMetrics
    ):
        """Serialize complete agent state for reconstruction."""
        if not agent_state:
            return

        state_group = agent_group.create_group("agent_state")

        for key, value in agent_state.items():
            try:
                if isinstance(value, dict):
                    # Handle nested dictionaries (evolution_rates, etc.)
                    self._serialize_nested_dict(state_group, key, value, metrics)

                elif isinstance(value, np.ndarray):
                    self._store_array_data(state_group, key, value, metrics)

                elif isinstance(value, complex) or key.endswith("_real") or key.endswith("_imag"):
                    if isinstance(value, complex):
                        self._store_complex_number(state_group, key, value, metrics)
                    else:
                        state_group.create_dataset(
                            key, data=float(value), compression=self.compression
                        )

                elif isinstance(value, (list, tuple)):
                    array_data = np.array(value)
                    self._store_array_data(state_group, key, array_data, metrics)

                elif isinstance(value, (int, float)):
                    state_group.create_dataset(key, data=value, compression=self.compression)

                else:
                    state_group.attrs[key] = str(value)

            except Exception as e:
                logger.warning(f"Could not serialize agent state component {key}: {e}")

    def _serialize_nested_dict(
        self,
        parent_group: h5py.Group,
        dict_name: str,
        dict_data: Dict[str, Any],
        metrics: SerializationMetrics,
    ):
        """Serialize nested dictionary structures."""
        dict_group = parent_group.create_group(dict_name)

        for key, value in dict_data.items():
            try:
                if isinstance(value, (int, float, complex, str, bool)):
                    if isinstance(value, complex):
                        self._store_complex_number(dict_group, key, value, metrics)
                    else:
                        dict_group.attrs[key] = value

                elif isinstance(value, np.ndarray):
                    self._store_array_data(dict_group, key, value, metrics)

                elif isinstance(value, (list, tuple)):
                    array_data = np.array(value)
                    self._store_array_data(dict_group, key, array_data, metrics)

                else:
                    dict_group.attrs[key] = str(value)

            except Exception as e:
                logger.warning(f"Could not serialize nested dict item {dict_name}/{key}: {e}")

    def _store_complex_number(
        self, group: h5py.Group, name: str, value: complex, metrics: SerializationMetrics
    ):
        """Store complex number with exact precision."""
        if isinstance(value, complex):
            # Create compound dataset for complex numbers
            complex_dtype = np.dtype([("real", "f8"), ("imag", "f8")])
            complex_data = np.array([(value.real, value.imag)], dtype=complex_dtype)
            group.create_dataset(name, data=complex_data, compression=self.compression)
            metrics.complex_numbers_preserved += 1
        else:
            # Fallback for non-complex values
            group.create_dataset(name, data=float(value), compression=self.compression)

    def _store_array_data(
        self, group: h5py.Group, name: str, array: np.ndarray, metrics: SerializationMetrics
    ):
        """Store numpy array with optimal compression and chunking."""
        if array.size == 0:
            return

        # Calculate optimal chunks
        chunks = self._calculate_optimal_chunks(array.shape)

        # Determine if array contains complex data
        if np.iscomplexobj(array):
            # Store complex arrays as compound datasets
            complex_dtype = np.dtype([("real", array.real.dtype), ("imag", array.imag.dtype)])
            complex_data = np.empty(array.shape, dtype=complex_dtype)
            complex_data["real"] = array.real
            complex_data["imag"] = array.imag

            group.create_dataset(
                name,
                data=complex_data,
                compression=self.compression,
                chunks=chunks,
                shuffle=True,
                fletcher32=self.preserve_precision,
            )
            metrics.complex_numbers_preserved += 1
        else:
            # Store real arrays
            group.create_dataset(
                name,
                data=array,
                compression=self.compression,
                chunks=chunks,
                shuffle=True,
                fletcher32=self.preserve_precision,
            )

        # Store metadata for verification
        group[name].attrs["original_shape"] = array.shape
        group[name].attrs["original_dtype"] = str(array.dtype)

        metrics.tensor_arrays_compressed += 1

        # Calculate compression ratio (approximate)
        uncompressed_size = array.nbytes
        compressed_size = group[name].id.get_storage_size()
        if compressed_size > 0:
            compression_ratio = uncompressed_size / compressed_size
            metrics.average_compression_ratio += compression_ratio

    def _store_field_array(
        self, group: h5py.Group, name: str, array: np.ndarray, metrics: SerializationMetrics
    ):
        """Store field array with optimizations for semantic/embedding data."""
        if array.size == 0:
            return

        # Field arrays often benefit from specific optimizations
        chunks = self._calculate_field_chunks(array.shape)

        if np.iscomplexobj(array):
            # Complex field arrays
            complex_dtype = np.dtype([("real", array.real.dtype), ("imag", array.imag.dtype)])
            complex_data = np.empty(array.shape, dtype=complex_dtype)
            complex_data["real"] = array.real
            complex_data["imag"] = array.imag

            group.create_dataset(
                name,
                data=complex_data,
                compression=self.compression,
                chunks=chunks,
                shuffle=True,
                fletcher32=self.preserve_precision,
            )
        else:
            # Real field arrays
            group.create_dataset(
                name,
                data=array,
                compression=self.compression,
                chunks=chunks,
                shuffle=True,
                fletcher32=self.preserve_precision,
            )

        # Store field-specific metadata
        group[name].attrs["field_dimension"] = array.shape[0] if len(array.shape) > 0 else 0
        group[name].attrs["field_type"] = "semantic" if "semantic" in name else "general"
        group[name].attrs["original_dtype"] = str(array.dtype)

        metrics.tensor_arrays_compressed += 1

    def _calculate_optimal_chunks(self, shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
        """Calculate optimal chunk size for general arrays."""
        if not self.optimize_storage or len(shape) == 0:
            return None

        if len(shape) == 1:
            size = shape[0]
            if size < 1000:
                return None
            # Chunk to roughly 64KB for 1D arrays
            chunk_size = min(size, 8192)
            return (chunk_size,)

        # For higher dimensions, use auto-chunking
        return True

    def _calculate_field_chunks(self, shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
        """Calculate optimal chunk size specifically for field arrays."""
        if not self.optimize_storage or len(shape) == 0:
            return None

        if len(shape) == 1:
            # Field vectors (semantic embeddings, etc.)
            size = shape[0]
            if size < 512:
                return None

            # Optimize for common field dimensions
            if size >= 1024:  # BGE-style
                return (1024,)
            elif size >= 768:  # MPNet-style
                return (768,)
            else:
                return (min(size, 512),)

        return True


if __name__ == "__main__":
    # Example usage
    serializer = AgentSerializer()
    print("AgentSerializer ready for flexible dimension serialization")
