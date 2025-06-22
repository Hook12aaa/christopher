"""
HDF5 Manager - Mathematical Object Storage with Dynamic Schemas

Manages HDF5 storage for complete mathematical liquid objects with dynamic
dimensionality support. Preserves exact complex number precision and handles
any field dimensionality without hardcoded assumptions.

Key Features:
- Dynamic schema creation based on actual field dimensions
- Complex number preservation with exact precision
- Hierarchical organization following liquid agent structure
- Efficient compression for large tensor arrays
- Model-agnostic storage (BGE 1024, MPNet 768, future models)
"""

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HDF5StorageMetrics:
    """Metrics for HDF5 storage operations."""

    universes_stored: int = 0
    agents_stored: int = 0
    total_storage_size_mb: float = 0.0
    compression_ratio: float = 0.0
    storage_time_seconds: float = 0.0
    field_dimensions: Optional[int] = None


class HDF5Manager:
    """
    Manager for mathematical object storage using HDF5.

    Provides dynamic schema creation and exact mathematical precision
    preservation for liquid universe components. Adapts automatically
    to any field dimensionality without hardcoded constraints.

    Storage Structure:
    /liquid_universe_{uuid}/
    ├── metadata/
    │   ├── creation_timestamp
    │   ├── field_statistics
    │   ├── model_info/
    │   │   ├── embedding_dimension    # Dynamic: 768, 1024, 1536, etc.
    │   │   ├── model_type            # BGE, MPNet, etc.
    │   │   └── field_architecture    # Model-specific properties
    │   └── optimization_params
    ├── charges/
    │   ├── charge_0/
    │   │   ├── Q_components/         # Complete Q(τ,C,s) mathematics
    │   │   ├── field_components/     # Dynamic-dimensional arrays
    │   │   ├── temporal_biography/   # Time evolution data
    │   │   ├── emotional_modulation/ # Field modulation data
    │   │   └── agent_state/         # Complete agent state
    │   └── charge_N/...
    └── collective_properties/
        ├── field_statistics/
        ├── interaction_matrices/
        └── optimization_stats/
    """

    def __init__(
        self,
        storage_path: Union[str, Path],
        compression: str = "lzf",
        chunk_cache_size: int = 1024 * 1024 * 16,
    ):  # 16MB cache
        """
        Initialize HDF5 manager.

        Args:
            storage_path: Base path for HDF5 storage
            compression: Compression algorithm ('lzf', 'gzip', 'szip')
            chunk_cache_size: HDF5 chunk cache size in bytes
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.chunk_cache_size = chunk_cache_size

        # Configure HDF5 library settings (version-safe)
        try:
            h5py.get_config().cache_size = chunk_cache_size
        except AttributeError:
            # h5py version doesn't support cache_size configuration
            logger.debug("h5py cache_size configuration not available in this version")

        logger.info(f"HDF5Manager initialized")
        logger.info(f"  Storage path: {self.storage_path}")
        logger.info(f"  Compression: {compression}")
        logger.info(f"  Cache size: {chunk_cache_size/1024/1024:.1f}MB")

    def store_liquid_universe(
        self, extracted_data: "ExtractedLiquidData", universe_id: Optional[str] = None
    ) -> str:
        """
        Store complete liquid universe data in HDF5.

        Args:
            extracted_data: Complete extracted liquid universe data
            universe_id: Optional custom universe ID (auto-generated if None)

        Returns:
            Universe ID for the stored data
        """
        if universe_id is None:
            universe_id = f"universe_{uuid.uuid4().hex[:8]}_{int(time.time())}"

        hdf5_file = self.storage_path / f"{universe_id}.h5"

        logger.info(f"💾 Storing liquid universe: {universe_id}")
        logger.info(f"   📁 File: {hdf5_file}")

        start_time = time.time()

        with h5py.File(hdf5_file, "w") as f:
            # Create universe root group
            universe_group = f.create_group(f"liquid_universe_{universe_id}")

            # Store metadata
            self._store_universe_metadata(universe_group, extracted_data)

            # Store agent data with dynamic schemas
            self._store_agent_data(universe_group, extracted_data)

            # Store collective properties
            self._store_collective_properties(universe_group, extracted_data)

            # Store extraction metrics
            self._store_extraction_metrics(universe_group, extracted_data)

        storage_time = time.time() - start_time
        file_size_mb = hdf5_file.stat().st_size / (1024 * 1024)

        logger.info(f"💾 Universe stored successfully in {storage_time:.2f}s")
        logger.info(f"   📊 File size: {file_size_mb:.2f}MB")
        logger.info(f"   🎯 Agents: {len(extracted_data.agent_data)}")

        return universe_id

    def _store_universe_metadata(
        self, universe_group: h5py.Group, extracted_data: "ExtractedLiquidData"
    ):
        """Store universe-level metadata."""
        metadata_group = universe_group.create_group("metadata")

        # Store basic metadata
        for key, value in extracted_data.universe_metadata.items():
            if value is not None:
                if isinstance(value, (str, int, float, bool)):
                    metadata_group.attrs[key] = value
                else:
                    # Handle complex types
                    try:
                        metadata_group.attrs[key] = str(value)
                    except:
                        logger.warning(f"Could not store metadata key: {key}")

        # Store field statistics
        if extracted_data.field_statistics:
            field_stats_group = metadata_group.create_group("field_statistics")
            for key, value in extracted_data.field_statistics.items():
                if isinstance(value, (int, float)):
                    field_stats_group.attrs[key] = value
                elif isinstance(value, str):
                    field_stats_group.attrs[key] = value

        # Store model information (critical for reconstruction)
        model_info_group = metadata_group.create_group("model_info")

        # Store detected field dimensions
        if extracted_data.extraction_metrics.field_dimensions_detected:
            model_info_group.attrs["embedding_dimension"] = (
                extracted_data.extraction_metrics.field_dimensions_detected
            )

        if extracted_data.extraction_metrics.model_type_detected:
            model_info_group.attrs["model_type"] = (
                extracted_data.extraction_metrics.model_type_detected
            )

        # Store timestamp for reconstruction ordering
        model_info_group.attrs["storage_timestamp"] = time.time()

        # Store vocabulary context (critical for reconstruction)
        self._store_vocabulary_context(metadata_group, extracted_data)

    def _store_vocabulary_context(
        self, metadata_group: h5py.Group, extracted_data: "ExtractedLiquidData"
    ):
        """
        Store vocabulary context for complete universe reconstruction.

        Vocabulary context is critical for reconstructing the exact semantic
        relationships and token mappings from the original foundation model.
        """
        # Check if we have vocabulary mappings in the extracted data
        # This should come from the top-level liquid_results structure
        vocab_mappings = getattr(extracted_data, "vocab_mappings", {})

        if not vocab_mappings:
            logger.warning("No vocabulary mappings found in extracted data")
            return

        vocab_group = metadata_group.create_group("vocabulary_context")

        # Store vocabulary mappings with efficient encoding
        id_to_token = vocab_mappings.get("id_to_token")
        token_to_id = vocab_mappings.get("token_to_id")
        embedding_indices = vocab_mappings.get("embedding_indices")

        if id_to_token:
            logger.info(f"   📚 Storing {len(id_to_token)} id_to_token mappings")

            # Convert to parallel arrays for efficient HDF5 storage
            token_ids = list(id_to_token.keys())
            token_strings = list(id_to_token.values())

            # Store as parallel datasets for efficient lookup
            vocab_group.create_dataset(
                "token_ids",
                data=np.array(token_ids, dtype=np.int32),
                compression=self.compression,
            )

            # Handle string encoding for HDF5
            string_dtype = h5py.special_dtype(vlen=str)
            vocab_group.create_dataset(
                "token_strings",
                data=token_strings,
                dtype=string_dtype,
                compression=self.compression,
            )

            # Store count for validation
            vocab_group.attrs["vocab_size"] = len(id_to_token)

        if token_to_id:
            logger.info(
                f"   🔢 Storing reverse token mapping for {len(token_to_id)} tokens"
            )
            # Store validation data for reverse mapping
            vocab_group.attrs["reverse_mapping_size"] = len(token_to_id)

        if embedding_indices:
            logger.info(f"   📊 Storing {len(embedding_indices)} embedding indices")
            vocab_group.create_dataset(
                "embedding_indices",
                data=np.array(embedding_indices, dtype=np.int32),
                compression=self.compression,
            )

        # Store additional model-specific vocabulary context
        if "model_info" in vocab_mappings:
            model_vocab_info = vocab_mappings["model_info"]
            model_vocab_group = vocab_group.create_group("model_vocab_info")

            for key, value in model_vocab_info.items():
                if isinstance(value, (str, int, float, bool)):
                    model_vocab_group.attrs[key] = value
                else:
                    model_vocab_group.attrs[key] = str(value)

        # Store creation timestamp for vocabulary versioning
        vocab_group.attrs["vocab_storage_timestamp"] = time.time()
        vocab_group.attrs["vocab_source"] = "ChargeFactory.build()"

        logger.info(f"   ✅ Vocabulary context stored successfully")

    def _store_agent_data(
        self, universe_group: h5py.Group, extracted_data: "ExtractedLiquidData"
    ):
        """Store all agent data with dynamic schemas."""
        charges_group = universe_group.create_group("charges")

        for agent_id, agent_data in extracted_data.agent_data.items():
            agent_group = charges_group.create_group(agent_id)

            # Store each component type
            self._store_q_components(agent_group, agent_data.get("Q_components"))
            self._store_field_components(
                agent_group, agent_data.get("field_components")
            )
            self._store_temporal_components(
                agent_group, agent_data.get("temporal_components")
            )
            self._store_emotional_components(
                agent_group, agent_data.get("emotional_components")
            )
            self._store_agent_state(agent_group, agent_data.get("agent_state"))
            self._store_agent_metadata(
                agent_group, agent_data.get("agent_metadata")
            )

            # Store enhanced evolution data with temporal tracking
            self._store_evolution_data(agent_group, agent_data)

    def _store_q_components(
        self, agent_group: h5py.Group, q_components: Dict[str, Any]
    ):
        """Store Q(τ,C,s) mathematical components."""
        if not q_components:
            return

        q_group = agent_group.create_group("Q_components")

        for key, value in q_components.items():
            try:
                if isinstance(value, (int, float, complex)):
                    if isinstance(value, complex):
                        # Store complex numbers as compound dataset
                        complex_dtype = np.dtype([("real", "f8"), ("imag", "f8")])
                        complex_data = np.array(
                            [(value.real, value.imag)], dtype=complex_dtype
                        )
                        q_group.create_dataset(
                            key, data=complex_data, compression=self.compression
                        )
                    else:
                        # Scalar datasets don't support compression
                        q_group.create_dataset(key, data=value)

                elif isinstance(value, np.ndarray):
                    # Handle numpy arrays with appropriate chunking
                    chunks = self._calculate_optimal_chunks(value.shape)
                    q_group.create_dataset(
                        key, data=value, compression=self.compression, chunks=chunks
                    )

                elif isinstance(value, (list, tuple)):
                    # Convert to numpy array for storage
                    array_data = np.array(value)
                    chunks = self._calculate_optimal_chunks(array_data.shape)
                    q_group.create_dataset(
                        key,
                        data=array_data,
                        compression=self.compression,
                        chunks=chunks,
                    )

                else:
                    # Store as string for reconstruction
                    q_group.attrs[key] = str(value)

            except Exception as e:
                logger.warning(f"Could not store Q component {key}: {e}")

    def _store_field_components(
        self, agent_group: h5py.Group, field_components: Dict[str, Any]
    ):
        """Store field components with dynamic dimensionality."""
        if not field_components:
            return

        field_group = agent_group.create_group("field_components")

        for key, value in field_components.items():
            try:
                if isinstance(value, np.ndarray):
                    # Dynamic chunking based on actual array dimensions
                    chunks = self._calculate_optimal_chunks(value.shape)

                    # Store with compression optimized for field data
                    field_group.create_dataset(
                        key,
                        data=value,
                        compression=self.compression,
                        chunks=chunks,
                        shuffle=True,  # Improve compression for field data
                        fletcher32=True,  # Error detection
                    )

                    # Store shape metadata for verification
                    field_group[key].attrs["shape"] = value.shape
                    field_group[key].attrs["dtype"] = str(value.dtype)

                elif isinstance(value, (list, tuple)):
                    # Convert and store with metadata
                    array_data = np.array(value)
                    chunks = self._calculate_optimal_chunks(array_data.shape)
                    field_group.create_dataset(
                        key,
                        data=array_data,
                        compression=self.compression,
                        chunks=chunks,
                    )

                elif isinstance(value, (int, float)):
                    # Scalar datasets don't support compression
                    field_group.create_dataset(key, data=value)
                elif isinstance(value, complex):
                    complex_dtype = np.dtype([("real", "f8"), ("imag", "f8")])
                    complex_data = np.array([(value.real, value.imag)], dtype=complex_dtype)
                    field_group.create_dataset(
                        key, data=complex_data, compression=self.compression
                    )

                else:
                    field_group.attrs[key] = str(value)

            except Exception as e:
                logger.warning(f"Could not store field component {key}: {e}")

    def _store_temporal_components(
        self, agent_group: h5py.Group, temporal_components: Dict[str, Any]
    ):
        """Store temporal biography and evolution data."""
        if not temporal_components:
            return

        temporal_group = agent_group.create_group("temporal_biography")

        for key, value in temporal_components.items():
            try:
                if isinstance(value, np.ndarray):
                    chunks = self._calculate_optimal_chunks(value.shape)
                    temporal_group.create_dataset(
                        key, data=value, compression=self.compression, chunks=chunks
                    )

                elif isinstance(value, complex):
                    # Store complex temporal momentum, etc.
                    complex_dtype = np.dtype([("real", "f8"), ("imag", "f8")])
                    complex_data = np.array(
                        [(value.real, value.imag)], dtype=complex_dtype
                    )
                    temporal_group.create_dataset(
                        key, data=complex_data, compression=self.compression
                    )

                elif isinstance(value, (int, float)):
                    # Scalar datasets don't support compression
                    temporal_group.create_dataset(key, data=value)

                elif isinstance(value, (list, tuple)):
                    array_data = np.array(value)
                    chunks = self._calculate_optimal_chunks(array_data.shape)
                    temporal_group.create_dataset(
                        key,
                        data=array_data,
                        compression=self.compression,
                        chunks=chunks,
                    )

                else:
                    temporal_group.attrs[key] = str(value)

            except Exception as e:
                logger.warning(f"Could not store temporal component {key}: {e}")

    def _store_emotional_components(
        self, agent_group: h5py.Group, emotional_components: Dict[str, Any]
    ):
        """Store emotional modulation and field signature data."""
        if not emotional_components:
            return

        emotional_group = agent_group.create_group("emotional_modulation")

        for key, value in emotional_components.items():
            try:
                if isinstance(value, np.ndarray):
                    chunks = self._calculate_optimal_chunks(value.shape)
                    emotional_group.create_dataset(
                        key, data=value, compression=self.compression, chunks=chunks
                    )

                elif isinstance(value, complex):
                    complex_dtype = np.dtype([("real", "f8"), ("imag", "f8")])
                    complex_data = np.array(
                        [(value.real, value.imag)], dtype=complex_dtype
                    )
                    emotional_group.create_dataset(
                        key, data=complex_data, compression=self.compression
                    )

                elif isinstance(value, (int, float)):
                    # Scalar datasets don't support compression
                    emotional_group.create_dataset(key, data=value)

                elif isinstance(value, (list, tuple)):
                    array_data = np.array(value)
                    chunks = self._calculate_optimal_chunks(array_data.shape)
                    emotional_group.create_dataset(
                        key,
                        data=array_data,
                        compression=self.compression,
                        chunks=chunks,
                    )

                else:
                    emotional_group.attrs[key] = str(value)

            except Exception as e:
                logger.warning(f"Could not store emotional component {key}: {e}")

    def _store_agent_state(self, agent_group: h5py.Group, agent_state: Dict[str, Any]):
        """Store complete agent state for reconstruction."""
        if not agent_state:
            return

        state_group = agent_group.create_group("agent_state")

        for key, value in agent_state.items():
            try:
                if isinstance(value, dict):
                    # Handle nested dictionaries (evolution_rates, etc.)
                    dict_group = state_group.create_group(key)
                    for subkey, subvalue in value.items():
                        # Convert integer keys to strings for HDF5 compatibility
                        str_subkey = str(subkey)
                        if isinstance(subvalue, (int, float, complex)):
                            dict_group.attrs[str_subkey] = subvalue
                        else:
                            dict_group.attrs[str_subkey] = str(subvalue)

                elif isinstance(value, complex):
                    complex_dtype = np.dtype([("real", "f8"), ("imag", "f8")])
                    complex_data = np.array(
                        [(value.real, value.imag)], dtype=complex_dtype
                    )
                    state_group.create_dataset(
                        key, data=complex_data, compression=self.compression
                    )

                elif isinstance(value, (int, float)):
                    # Scalar datasets don't support compression
                    state_group.create_dataset(key, data=value)

                elif isinstance(value, (list, tuple, np.ndarray)):
                    array_data = np.array(value)
                    chunks = self._calculate_optimal_chunks(array_data.shape)
                    state_group.create_dataset(
                        key,
                        data=array_data,
                        compression=self.compression,
                        chunks=chunks,
                    )

                else:
                    state_group.attrs[key] = str(value)

            except Exception as e:
                logger.warning(f"Could not store agent state component {key}: {e}")

    def _store_agent_metadata(
        self, agent_group: h5py.Group, agent_metadata: Dict[str, Any]
    ):
        """Store agent metadata."""
        for key, value in agent_metadata.items():
            try:
                if isinstance(value, (str, int, float, bool)):
                    agent_group.attrs[key] = value
                else:
                    agent_group.attrs[key] = str(value)
            except Exception as e:
                logger.warning(f"Could not store agent metadata {key}: {e}")

    def _store_evolution_data(
        self, agent_group: h5py.Group, agent_data: Dict[str, Any]
    ):
        """
        Store enhanced evolution data with temporal tracking and mathematical precision.

        Organizes evolution data into dedicated structure for agent development tracking:
        - Evolution parameters (breathing, modulation, position)
        - Mathematical evolution (hecke eigenvalues, L-function coefficients)
        - Temporal progression tracking
        - Cross-dimensional coupling evolution
        """
        evolution_group = agent_group.create_group("evolution_data")

        # Extract evolution-related data from different components
        agent_state = agent_data.get("agent_state")
        temporal_components = agent_data.get("temporal_components")
        emotional_components = agent_data.get("emotional_components")

        # Store mathematical evolution parameters
        math_evolution_group = evolution_group.create_group("mathematical_evolution")

        # Hecke eigenvalues and modular form data
        mathematical_attrs = [
            "hecke_eigenvalues",
            "l_function_coefficients",
            "breathing_q_coefficients",
        ]
        for attr in mathematical_attrs:
            if attr in agent_state:
                value = agent_state[attr]
                if isinstance(value, dict):
                    # Create subgroup for complex mathematical structures
                    attr_group = math_evolution_group.create_group(attr)
                    for subkey, subvalue in value.items():
                        try:
                            # Convert integer keys to strings for HDF5 compatibility
                            str_subkey = str(subkey)
                            
                            if isinstance(subvalue, complex):
                                complex_dtype = np.dtype(
                                    [("real", "f8"), ("imag", "f8")]
                                )
                                complex_data = np.array(
                                    [(subvalue.real, subvalue.imag)],
                                    dtype=complex_dtype,
                                )
                                attr_group.create_dataset(
                                    str_subkey,
                                    data=complex_data,
                                    compression=self.compression,
                                )
                            elif isinstance(subvalue, (int, float)):
                                # Scalar datasets don't support compression
                                attr_group.create_dataset(str_subkey, data=subvalue)
                            elif isinstance(subvalue, (list, tuple, np.ndarray)):
                                array_data = np.array(subvalue)
                                chunks = self._calculate_optimal_chunks(
                                    array_data.shape
                                )
                                attr_group.create_dataset(
                                    str_subkey,
                                    data=array_data,
                                    compression=self.compression,
                                    chunks=chunks,
                                )
                            else:
                                attr_group.attrs[str_subkey] = str(subvalue)
                        except Exception as e:
                            logger.warning(
                                f"Could not store mathematical evolution {attr}/{subkey}: {e}"
                            )

        # Store dynamical evolution parameters
        dynamics_group = evolution_group.create_group("dynamical_evolution")

        # Breathing and temporal dynamics
        breathing_attrs = [
            "breath_frequency",
            "breath_phase",
            "breath_amplitude",
            "breathing_coherence",
        ]
        for attr in breathing_attrs:
            if attr in agent_state:
                value = agent_state[attr]
                self._store_evolution_value(dynamics_group, attr, value)
            elif attr in temporal_components:
                value = temporal_components[attr]
                self._store_evolution_value(dynamics_group, attr, value)

        # Field evolution parameters
        field_evolution_group = evolution_group.create_group("field_evolution")

        # Conductivity and coupling evolution
        field_attrs = [
            "emotional_conductivity",
            "coupling_strength",
            "gradient_magnitude",
            "field_modulation_strength",
            "pattern_confidence",
        ]
        for attr in field_attrs:
            if attr in agent_state:
                value = agent_state[attr]
                self._store_evolution_value(field_evolution_group, attr, value)
            elif attr in emotional_components:
                value = emotional_components[attr]
                self._store_evolution_value(field_evolution_group, attr, value)

        # Store collective evolution data
        collective_evolution_group = evolution_group.create_group(
            "collective_evolution"
        )

        # Evolution rates and cascade momentum
        collective_attrs = ["evolution_rates", "cascade_momentum"]
        for attr in collective_attrs:
            if attr in agent_state:
                value = agent_state[attr]
                if isinstance(value, dict):
                    attr_group = collective_evolution_group.create_group(attr)
                    for subkey, subvalue in value.items():
                        self._store_evolution_value(attr_group, subkey, subvalue)

        # Store positional evolution
        position_group = evolution_group.create_group("positional_evolution")

        # Tau position and spatial coordinates
        position_attrs = [
            "tau_position",
            "modular_weight",
            "sigma_i",
            "alpha_i",
            "lambda_i",
            "beta_i",
        ]
        for attr in position_attrs:
            if attr in agent_state:
                value = agent_state[attr]
                self._store_evolution_value(position_group, attr, value)

        # Store evolution metadata
        evolution_group.attrs["evolution_storage_timestamp"] = time.time()
        evolution_group.attrs["evolution_components_stored"] = len(
            list(evolution_group.keys())
        )

        logger.debug(
            f"Stored evolution data with {len(list(evolution_group.keys()))} component groups"
        )

    def _store_evolution_value(self, group: h5py.Group, key: str, value: Any):
        """Helper method to store evolution values with appropriate type handling."""
        try:
            if isinstance(value, complex):
                complex_dtype = np.dtype([("real", "f8"), ("imag", "f8")])
                complex_data = np.array([(value.real, value.imag)], dtype=complex_dtype)
                group.create_dataset(
                    key, data=complex_data, compression=self.compression
                )

            elif isinstance(value, (int, float)):
                # Scalar datasets don't support compression/chunks
                group.create_dataset(key, data=value)

            elif isinstance(value, (list, tuple, np.ndarray)):
                array_data = np.array(value)
                chunks = self._calculate_optimal_chunks(array_data.shape)
                group.create_dataset(
                    key, data=array_data, compression=self.compression, chunks=chunks
                )

            else:
                group.attrs[key] = str(value)

        except Exception as e:
            logger.warning(f"Could not store evolution value {key}: {e}")

    def _store_collective_properties(
        self, universe_group: h5py.Group, extracted_data: "ExtractedLiquidData"
    ):
        """Store collective properties and optimization statistics."""
        if not extracted_data.collective_properties:
            return

        collective_group = universe_group.create_group("collective_properties")

        for key, value in extracted_data.collective_properties.items():
            try:
                if isinstance(value, dict):
                    # Create subgroup for nested dictionaries
                    subgroup = collective_group.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float, str, bool)):
                            subgroup.attrs[subkey] = subvalue
                        elif isinstance(subvalue, (list, tuple, np.ndarray)):
                            array_data = np.array(subvalue)
                            chunks = self._calculate_optimal_chunks(array_data.shape)
                            subgroup.create_dataset(
                                subkey,
                                data=array_data,
                                compression=self.compression,
                                chunks=chunks,
                            )
                        else:
                            subgroup.attrs[subkey] = str(subvalue)

                elif isinstance(value, (list, tuple, np.ndarray)):
                    array_data = np.array(value)
                    chunks = self._calculate_optimal_chunks(array_data.shape)
                    collective_group.create_dataset(
                        key,
                        data=array_data,
                        compression=self.compression,
                        chunks=chunks,
                    )

                else:
                    collective_group.attrs[key] = str(value)

            except Exception as e:
                logger.warning(f"Could not store collective property {key}: {e}")

    def _store_extraction_metrics(
        self, universe_group: h5py.Group, extracted_data: "ExtractedLiquidData"
    ):
        """Store extraction metrics for validation."""
        metrics_group = universe_group.create_group("extraction_metrics")

        metrics = extracted_data.extraction_metrics

        metrics_group.attrs["agents_processed"] = metrics.agents_processed
        metrics_group.attrs["total_processing_time"] = metrics.total_processing_time
        metrics_group.attrs["q_components_extracted"] = metrics.q_components_extracted
        metrics_group.attrs["field_arrays_extracted"] = metrics.field_arrays_extracted
        metrics_group.attrs["mathematical_validation_passed"] = (
            metrics.mathematical_validation_passed
        )

        if metrics.field_dimensions_detected:
            metrics_group.attrs["field_dimensions_detected"] = (
                metrics.field_dimensions_detected
            )

        if metrics.model_type_detected:
            metrics_group.attrs["model_type_detected"] = metrics.model_type_detected

    def _calculate_optimal_chunks(
        self, shape: Tuple[int, ...]
    ) -> Optional[Tuple[int, ...]]:
        """
        Calculate optimal chunk size for HDF5 storage.

        Args:
            shape: Array shape

        Returns:
            Optimal chunk shape or None for auto-chunking
        """
        if len(shape) == 0:
            return None

        # For 1D arrays (most field components)
        if len(shape) == 1:
            size = shape[0]
            if size < 1000:
                return None  # No chunking for small arrays
            else:
                # Chunk to roughly 64KB
                chunk_size = min(size, 8192)
                return (chunk_size,)

        # For 2D arrays and higher, use auto-chunking
        return True

    def load_universe(self, universe_id: str) -> Dict[str, Any]:
        """
        Load complete universe data from HDF5.

        Args:
            universe_id: Universe identifier

        Returns:
            Complete universe data dictionary
        """
        hdf5_file = self.storage_path / f"{universe_id}.h5"

        if not hdf5_file.exists():
            raise FileNotFoundError(f"Universe file not found: {hdf5_file}")

        logger.info(f"📖 Loading universe: {universe_id}")

        with h5py.File(hdf5_file, "r") as f:
            universe_key = f"liquid_universe_{universe_id}"
            if universe_key not in f:
                raise KeyError(f"Universe group not found: {universe_key}")

            universe_group = f[universe_key]

            # Load all components
            universe_data = {
                "universe_id": universe_id,
                "metadata": self._load_metadata(universe_group),
                "agents": self._load_agent_data(universe_group),
                "collective_properties": self._load_collective_properties(
                    universe_group
                ),
                "extraction_metrics": self._load_extraction_metrics(universe_group),
            }

        logger.info(f"📖 Universe loaded: {len(universe_data['agents'])} agents")

        return universe_data

    def _load_metadata(self, universe_group: h5py.Group) -> Dict[str, Any]:
        """Load universe metadata."""
        if "metadata" not in universe_group:
            return {}

        metadata_group = universe_group["metadata"]
        metadata = dict(metadata_group.attrs)

        # Load field statistics if present
        if "field_statistics" in metadata_group:
            metadata["field_statistics"] = dict(
                metadata_group["field_statistics"].attrs
            )

        # Load model info if present
        if "model_info" in metadata_group:
            metadata["model_info"] = dict(metadata_group["model_info"].attrs)

        # Load vocabulary context if present
        if "vocabulary_context" in metadata_group:
            metadata["vocabulary_context"] = self._load_vocabulary_context(
                metadata_group["vocabulary_context"]
            )

        return metadata

    def _load_agent_data(self, universe_group: h5py.Group) -> Dict[str, Any]:
        """Load all agent data."""
        if "charges" not in universe_group:
            return {}

        charges_group = universe_group["charges"]
        agents = {}

        for agent_id in charges_group.keys():
            agent_group = charges_group[agent_id]
            agents[agent_id] = {
                "Q_components": self._load_group_data(agent_group, "Q_components"),
                "field_components": self._load_group_data(
                    agent_group, "field_components"
                ),
                "temporal_biography": self._load_group_data(
                    agent_group, "temporal_biography"
                ),
                "emotional_modulation": self._load_group_data(
                    agent_group, "emotional_modulation"
                ),
                "agent_state": self._load_group_data(agent_group, "agent_state"),
                "evolution_data": self._load_group_data(agent_group, "evolution_data"),
                "metadata": dict(agent_group.attrs),
            }

        return agents

    def _load_collective_properties(self, universe_group: h5py.Group) -> Dict[str, Any]:
        """Load collective properties."""
        if "collective_properties" not in universe_group:
            return {}

        return self._load_group_data(universe_group, "collective_properties")

    def _load_extraction_metrics(self, universe_group: h5py.Group) -> Dict[str, Any]:
        """Load extraction metrics."""
        if "extraction_metrics" not in universe_group:
            return {}

        return dict(universe_group["extraction_metrics"].attrs)

    def _load_group_data(
        self, parent_group: h5py.Group, group_name: str
    ) -> Dict[str, Any]:
        """Load data from a specific group."""
        if group_name not in parent_group:
            return {}

        group = parent_group[group_name]
        data = {}

        # Load attributes
        data.update(dict(group.attrs))

        # Load datasets
        for key in group.keys():
            try:
                if isinstance(group[key], h5py.Dataset):
                    dataset_data = group[key][...]

                    # Handle complex numbers
                    if (
                        dataset_data.dtype.names
                        and "real" in dataset_data.dtype.names
                        and "imag" in dataset_data.dtype.names
                    ):
                        # Reconstruct complex number
                        if dataset_data.shape == (1,):
                            data[key] = complex(
                                dataset_data["real"][0], dataset_data["imag"][0]
                            )
                        else:
                            data[key] = dataset_data["real"] + 1j * dataset_data["imag"]
                    else:
                        data[key] = dataset_data

                elif isinstance(group[key], h5py.Group):
                    # Recursively load subgroups
                    data[key] = self._load_group_data(group, key)

            except Exception as e:
                logger.warning(f"Could not load {group_name}/{key}: {e}")

        return data

    def _load_vocabulary_context(self, vocab_group: h5py.Group) -> Dict[str, Any]:
        """
        Load vocabulary context for universe reconstruction.

        Reconstructs the complete vocabulary mappings needed to restore
        exact semantic relationships from the original foundation model.
        """
        vocab_context = {}

        # Load basic vocabulary attributes
        vocab_context.update(dict(vocab_group.attrs))

        # Reconstruct id_to_token mapping
        if "token_ids" in vocab_group and "token_strings" in vocab_group:
            token_ids = vocab_group["token_ids"][...]
            token_strings = vocab_group["token_strings"][...]

            # Reconstruct dictionary
            id_to_token = {}
            for token_id, token_string in zip(token_ids, token_strings):
                # Handle different string encoding
                if isinstance(token_string, bytes):
                    token_string = token_string.decode("utf-8")
                id_to_token[int(token_id)] = token_string

            vocab_context["id_to_token"] = id_to_token

            # Reconstruct reverse mapping
            token_to_id = {token: token_id for token_id, token in id_to_token.items()}
            vocab_context["token_to_id"] = token_to_id

            logger.info(f"   📚 Loaded {len(id_to_token)} vocabulary mappings")

        # Load embedding indices if present
        if "embedding_indices" in vocab_group:
            embedding_indices = vocab_group["embedding_indices"][...]
            vocab_context["embedding_indices"] = embedding_indices.tolist()
            logger.info(f"   📊 Loaded {len(embedding_indices)} embedding indices")

        # Load model-specific vocabulary info if present
        if "model_vocab_info" in vocab_group:
            model_vocab_group = vocab_group["model_vocab_info"]
            vocab_context["model_info"] = dict(model_vocab_group.attrs)
            logger.info(f"   🤖 Loaded model vocabulary info")

        logger.info(f"   ✅ Vocabulary context loaded successfully")

        return vocab_context

    def _load_arrow_metadata(self, universe_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Load Arrow metadata for all agents to merge with HDF5 data.
        
        Args:
            universe_id: Universe identifier to find corresponding Arrow file
            
        Returns:
            Dictionary mapping agent_id -> metadata from Arrow parquet
        """
        # Find corresponding Arrow file in the liquid_universes directory structure
        arrow_metadata = {}
        
        try:
            # Look for Arrow files in parent directories
            base_path = self.storage_path.parent  # Go up from hdf5/ to universe dir
            if base_path.name == "hdf5":
                base_path = base_path.parent  # Go up to liquid_universes/timestamp/
            
            # Look for arrow directory
            arrow_path = base_path / "arrow"
            if not arrow_path.exists():
                logger.warning(f"Arrow metadata directory not found: {arrow_path}")
                return arrow_metadata
            
            # Find parquet file matching universe_id
            parquet_files = list(arrow_path.glob(f"{universe_id}_agents.parquet"))
            if not parquet_files:
                # Try pattern match for similar universe IDs
                parquet_files = list(arrow_path.glob("*_agents.parquet"))
                if parquet_files:
                    logger.info(f"Using first available Arrow file: {parquet_files[0]}")
                else:
                    logger.warning(f"No Arrow metadata files found in {arrow_path}")
                    return arrow_metadata
            
            parquet_file = parquet_files[0]
            logger.info(f"📋 Loading Arrow metadata from: {parquet_file}")
            
            # Load parquet data
            df = pd.read_parquet(parquet_file)
            
            # Convert to dictionary mapping agent_id -> metadata
            for _, row in df.iterrows():
                agent_id = row.get('agent_id', row.get('charge_id', 'unknown'))
                arrow_metadata[agent_id] = {
                    'text_source': row.get('text_source', f'agent_{agent_id}'),
                    'charge_id': row.get('charge_id', agent_id),
                    'vocab_token_string': row.get('vocab_token_string', ''),
                    'vocab_token_id': row.get('vocab_token_id', ''),
                    'Q_magnitude': row.get('Q_magnitude', 0.0),
                    'Q_phase': row.get('Q_phase', 0.0),
                    'creation_timestamp': row.get('creation_timestamp', 0.0),
                    'last_updated': row.get('last_updated', 0.0)
                }
            
            logger.info(f"✅ Loaded Arrow metadata for {len(arrow_metadata)} agents")
            
        except Exception as e:
            logger.warning(f"Failed to load Arrow metadata: {e}")
            logger.info("Continuing without Arrow metadata - will use fallbacks")
        
        return arrow_metadata

    def load_universe_agents(self, universe_id: str) -> Dict[str, Any]:
        """
        Load all agent data from a stored universe with hybrid Arrow+HDF5 integration.
        
        This method retrieves stored agent data in the format that AgentFactory
        expects, including proper complex number reconstruction and metadata
        merging from both Arrow (text_source, etc.) and HDF5 (mathematical components).
        
        Args:
            universe_id: Universe identifier to load agents from
            
        Returns:
            Dictionary containing agent data and metadata
            
        Raises:
            FileNotFoundError: If universe file doesn't exist
            ValueError: If universe data is corrupted or invalid
        """
        logger.info(f"📦 Loading agents from universe: {universe_id}")
        start_time = time.time()
        
        # Handle both formats: just the ID or full filename
        if not universe_id.endswith('.h5'):
            universe_file = self.storage_path / f"{universe_id}.h5"
        else:
            universe_file = self.storage_path / universe_id
            universe_id = universe_id.replace('.h5', '')
        
        if not universe_file.exists():
            raise FileNotFoundError(f"Universe file not found: {universe_file}")
        
        # Load Arrow metadata first (if available)
        arrow_metadata = self._load_arrow_metadata(universe_id)
        
        try:
            with h5py.File(universe_file, "r") as f:
                # Find the universe group (handle different naming patterns)
                universe_group_name = None
                for key in f.keys():
                    if key.startswith("liquid_universe_"):
                        universe_group_name = key
                        break
                
                if not universe_group_name:
                    raise ValueError(f"No universe group found in {universe_file}")
                
                universe_group = f[universe_group_name]
                
                # Load universe metadata
                universe_metadata = {}
                if "metadata" in universe_group:
                    metadata_group = universe_group["metadata"]
                    
                    # Load basic metadata attributes
                    universe_metadata.update(dict(metadata_group.attrs))
                    
                    # Load field statistics if present
                    if "field_statistics" in metadata_group:
                        field_stats = dict(metadata_group["field_statistics"].attrs)
                        universe_metadata["field_statistics"] = field_stats
                    
                    # Load model info if present
                    if "model_info" in metadata_group:
                        model_info = dict(metadata_group["model_info"].attrs)
                        universe_metadata["model_info"] = model_info
                    
                    # Load vocabulary context if present  
                    if "vocabulary_context" in metadata_group:
                        vocab_context = self._load_vocabulary_context(metadata_group["vocabulary_context"])
                        universe_metadata["vocabulary_context"] = vocab_context
                
                # Load agent data
                agents_data = {}
                if "charges" in universe_group:
                    charges_group = universe_group["charges"]
                    
                    for agent_id in charges_group.keys():
                        agent_group = charges_group[agent_id]
                        agent_data = {}
                        
                        # Load each component group
                        component_groups = [
                            "Q_components", 
                            "field_components", 
                            "temporal_biography", 
                            "emotional_modulation", 
                            "agent_state"
                        ]
                        
                        for component_name in component_groups:
                            if component_name in agent_group:
                                component_data = self._load_group_data(agent_group, component_name)
                                
                                # Map temporal_biography back to temporal_components for compatibility
                                if component_name == "temporal_biography":
                                    agent_data["temporal_components"] = component_data
                                # Map emotional_modulation back to emotional_components
                                elif component_name == "emotional_modulation":
                                    agent_data["emotional_components"] = component_data
                                else:
                                    agent_data[component_name] = component_data
                        
                        # Load agent metadata (stored as attributes or in a separate group)
                        agent_metadata = {}
                        if "agent_metadata" in agent_group:
                            if isinstance(agent_group["agent_metadata"], h5py.Group):
                                agent_metadata = self._load_group_data(agent_group, "agent_metadata")
                            else:
                                # Handle case where metadata is stored as dataset
                                agent_metadata = dict(agent_group["agent_metadata"].attrs)
                        else:
                            # Load any attributes directly on the agent group
                            agent_metadata.update(dict(agent_group.attrs))
                        
                        # Ensure charge_id is set
                        if "charge_id" not in agent_metadata:
                            agent_metadata["charge_id"] = agent_id
                        
                        # HYBRID INTEGRATION: Merge Arrow metadata with HDF5 metadata
                        if agent_id in arrow_metadata:
                            arrow_agent_metadata = arrow_metadata[agent_id]
                            # Add text_source from Arrow (key missing field)
                            agent_metadata["text_source"] = arrow_agent_metadata["text_source"]
                            # Add other Arrow metadata
                            agent_metadata.update({
                                "vocab_token_string": arrow_agent_metadata["vocab_token_string"],
                                "vocab_token_id": arrow_agent_metadata["vocab_token_id"],
                                "arrow_Q_magnitude": arrow_agent_metadata["Q_magnitude"],
                                "arrow_Q_phase": arrow_agent_metadata["Q_phase"]
                            })
                            logger.debug(f"✅ Merged Arrow metadata for {agent_id}: text_source='{arrow_agent_metadata['text_source']}'")
                        else:
                            # Fallback text_source if Arrow metadata not available
                            agent_metadata["text_source"] = f"agent_{agent_id}"
                            logger.debug(f"⚠️ No Arrow metadata for {agent_id}, using fallback text_source")
                        
                        # TEMPORAL INTEGRATION: Map observational_state from temporal_biography.breathing_coherence
                        if "agent_state" in agent_data and "temporal_components" in agent_data:
                            temporal_components = agent_data["temporal_components"]
                            agent_state = agent_data["agent_state"]
                            
                            # Map breathing_coherence -> observational_state (the 's' parameter in Q(τ,C,s))
                            if "breathing_coherence" in temporal_components:
                                observational_state = temporal_components["breathing_coherence"]
                                agent_state["observational_state"] = float(observational_state)
                                logger.debug(f"✅ Mapped observational_state for {agent_id}: {observational_state}")
                            else:
                                # Fallback to default observational state
                                agent_state["observational_state"] = 1.0
                                logger.debug(f"⚠️ No breathing_coherence for {agent_id}, using default observational_state=1.0")
                        
                        agent_data["agent_metadata"] = agent_metadata
                        agents_data[agent_id] = agent_data
                
                loading_time = time.time() - start_time
                
                logger.info(f"📦 Loaded {len(agents_data)} agents in {loading_time:.2f}s")
                logger.info(f"   Universe metadata keys: {list(universe_metadata.keys())}")
                
                if agents_data:
                    first_agent = next(iter(agents_data.values()))
                    logger.info(f"   Sample agent components: {list(first_agent.keys())}")
                    
                    # Log Q_components structure for debugging
                    if "Q_components" in first_agent:
                        q_components = first_agent["Q_components"]
                        logger.info(f"   Sample Q_components keys: {list(q_components.keys())}")
                        
                        # Check for complex number reconstruction
                        for key, value in q_components.items():
                            if isinstance(value, complex):
                                logger.info(f"   ✅ Complex number reconstructed: {key} = {value} (magnitude: {abs(value):.2e})")
                            elif key.endswith(('_real', '_imag')):
                                logger.info(f"   📝 Real/imag component: {key} = {value}")
                
                return {
                    "status": "success",
                    "universe_id": universe_id,
                    "agents": agents_data,
                    "metadata": universe_metadata,
                    "loading_time": loading_time,
                    "agents_count": len(agents_data)
                }
                
        except Exception as e:
            error_msg = f"Failed to load universe agents: {e}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg) from e

    def list_universes(self) -> List[str]:
        """List all stored universe IDs."""
        universe_files = list(self.storage_path.glob("universe_*.h5"))
        universe_ids = []

        for file_path in universe_files:
            # Extract universe ID from filename - keep the full ID including "universe_" prefix
            filename = file_path.stem
            if filename.startswith("universe_"):
                universe_ids.append(filename)  # Keep full universe ID

        return sorted(universe_ids)

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        universe_files = list(self.storage_path.glob("universe_*.h5"))

        total_size = sum(f.stat().st_size for f in universe_files)

        return {
            "total_universes": len(universe_files),
            "total_size_mb": total_size / (1024 * 1024),
            "storage_path": str(self.storage_path),
            "compression": self.compression,
        }


if __name__ == "__main__":
    # Example usage
    manager = HDF5Manager("/tmp/test_storage")
    print("HDF5Manager ready for dynamic mathematical storage")
