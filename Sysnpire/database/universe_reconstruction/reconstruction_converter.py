"""
Reconstruction Converter - Pure Data Type Conversion

Converts validated raw data (numpy arrays, lists, primitives) to PyTorch tensors
for mathematical computation. This converter assumes data is ALREADY VALIDATED
and focuses purely on type conversion without any boolean evaluation of tensors.

Key Features:
- Real/imaginary pairs → complex numbers
- NumPy arrays → PyTorch tensors with device placement
- Lists/tuples → PyTorch tensors
- NO validation - assumes pre-validated data
- NO tensor boolean evaluation
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from Sysnpire.utils.log_polar_cdf import LogPolarCDF

logger = logging.getLogger(__name__)


class ReconstructionConversionError(Exception):
    """Raised when data type conversion fails during reconstruction."""
    pass


class ReconstructionConverter:
    """
    Pure data type converter for ALREADY VALIDATED raw data.
    
    Converts numpy arrays, lists, and primitives to PyTorch tensors.
    ASSUMES data is already validated by RawDataValidator.
    NO tensor validation or boolean evaluation performed here.
    """
    
    def __init__(self, device: str = "mps"):
        """
        Initialize reconstruction converter.
        
        Args:
            device: Target device for PyTorch tensors
        """
        # Device handling with fallback
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("⚠️ MPS not available, falling back to CPU")
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available, falling back to CPU") 
            device = "cpu"
            
        self.device = torch.device(device)
        
        # CRITICAL FIX: Set default dtype to float32 for MPS compatibility
        if self.device.type == "mps":
            torch.set_default_dtype(torch.float32)
            logger.info("🔧 MPS detected: Using float32 precision for compatibility")
            logger.info(f"   PyTorch default dtype set to: {torch.get_default_dtype()}")
        
        # No validation in converter - data is pre-validated
        
        logger.info("ReconstructionConverter initialized")
        logger.info(f"  Target device: {self.device}")
        logger.info("  Pure conversion mode - no validation performed")
    
    def convert_agent_batch(self, stored_agents: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Convert batch of agents from storage format to runtime format.
        
        Args:
            stored_agents: Dictionary mapping agent_id -> stored_agent_data
            
        Returns:
            Dictionary mapping agent_id -> converted_agent_data
        """
        logger.info(f"🔄 Converting batch of {len(stored_agents)} agents to tensors")
        start_time = time.time()
        
        converted_batch = {}
        conversion_errors = []
        
        try:
            for agent_id, stored_agent_data in stored_agents.items():
                try:
                    converted_batch[agent_id] = self.convert_agent_data(stored_agent_data)
                except Exception as e:
                    conversion_errors.append(f"Agent {agent_id}: {e}")
                    continue
            
            conversion_time = time.time() - start_time
            success_count = len(converted_batch)
            logger.info(f"✅ Batch conversion complete: {success_count}/{len(stored_agents)} agents in {conversion_time:.3f}s")
            
            if conversion_errors:
                logger.warning(f"⚠️  {len(conversion_errors)} conversion errors occurred")
                for error in conversion_errors[:3]:  # Log first 3 errors
                    logger.warning(f"   - {error}")
            
            return converted_batch
            
        except Exception as e:
            error_msg = f"Batch agent conversion failed: {e}"
            logger.error(f"❌ {error_msg}")
            raise ReconstructionConversionError(error_msg) from e

    def convert_agent_data(self, stored_agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert complete agent data from storage format to runtime format.
        
        Args:
            stored_agent_data: VALIDATED agent data from storage
            
        Returns:
            Converted agent data with tensors ready for computation
            
        Raises:
            ReconstructionConversionError: If conversion fails (not validation failures)
        """
        logger.debug("🔄 Converting validated data to tensors")
        start_time = time.time()
        
        try:
            converted_data = {}
            
            # Convert each component section
            if "agent_metadata" in stored_agent_data:
                converted_data["agent_metadata"] = self._convert_metadata(
                    stored_agent_data["agent_metadata"]
                )
            
            if "Q_components" in stored_agent_data:
                converted_data["Q_components"] = self._convert_q_components(
                    stored_agent_data["Q_components"]
                )
            
            if "field_components" in stored_agent_data:
                converted_data["field_components"] = self._convert_field_components(
                    stored_agent_data["field_components"]
                )
            
            if "temporal_components" in stored_agent_data:
                converted_data["temporal_components"] = self._convert_temporal_components(
                    stored_agent_data["temporal_components"]
                )
            
            if "emotional_components" in stored_agent_data:
                converted_data["emotional_components"] = self._convert_emotional_components(
                    stored_agent_data["emotional_components"]
                )
            
            if "agent_state" in stored_agent_data:
                converted_data["agent_state"] = self._convert_agent_state(
                    stored_agent_data["agent_state"]
                )
            
            conversion_time = time.time() - start_time
            # Agent data conversion complete (timing moved to batch summaries)
            
            return converted_data
            
        except Exception as e:
            error_msg = f"Agent data conversion failed: {e}"
            logger.error(f"❌ {error_msg}")
            raise ReconstructionConversionError(error_msg) from e
    
    def _convert_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metadata (usually no conversion needed)."""
        return dict(metadata)  # Simple copy, metadata is typically primitive types
    
    def _convert_q_components(self, q_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Q(τ,C,s) components from storage to runtime format.
        
        Key conversions:
        - Real/imag pairs → complex numbers
        - Separate mathematical components → integrated objects
        """
        converted = {}
        
        # Convert Q_value from real/imag pairs to complex number
        if "Q_value_real" in q_components and "Q_value_imag" in q_components:
            try:
                real_part = float(q_components["Q_value_real"])
                imag_part = float(q_components["Q_value_imag"])
                
                # Data is pre-validated, no need to check finite values
                
                converted["Q_value"] = complex(real_part, imag_part)
                # Q_value reconstructed successfully
                
            except (ValueError, TypeError) as e:
                # Data should be pre-validated, so this indicates a real error
                raise ReconstructionConversionError(f"Q_value conversion failed on validated data: {e}")
        
        elif "Q_value" in q_components:
            # Q_value already in complex format or needs conversion
            q_val = q_components["Q_value"]
            if isinstance(q_val, dict) and "_type" in q_val:
                # Handle special storage format
                if q_val["_type"] == "complex":
                    converted["Q_value"] = complex(q_val["real"], q_val["imag"])
                else:
                    converted["Q_value"] = q_val
            else:
                converted["Q_value"] = q_val
        
        # CRITICAL FIX: Reconstruct complex fields from real/imag pairs first
        # Apply same pattern as temporal_momentum reconstruction above
        complex_mathematical_components = ["T_tensor", "E_trajectory", "phi_semantic", "phase_factor"]
        
        for component in complex_mathematical_components:
            real_key = f"{component}_real"
            imag_key = f"{component}_imag"
            
            # Check if we have both real and imag components (same pattern as temporal_momentum)
            if real_key in q_components and imag_key in q_components:
                real_part = float(q_components[real_key])
                imag_part = float(q_components[imag_key])
                
                # NO DEFAULTS - Either the values exist or it fails (same as temporal_momentum)
                converted[component] = complex(real_part, imag_part)
                # Complex component converted from real/imag pairs
        
        # Convert mathematical component arrays to tensors (handle scalars vs arrays)
        mathematical_components = ["gamma", "T_tensor", "E_trajectory", "phi_semantic", "theta_components", "phase_factor", "psi_persistence", "psi_gaussian", "psi_exponential_cosine"]
        
        for component in mathematical_components:
            for key in q_components:
                if key.startswith(component):
                    # Skip if we already processed this as a complex reconstruction
                    if key.endswith("_real") or key.endswith("_imag"):
                        # Check if the base component was already reconstructed
                        base_component = key.replace("_real", "").replace("_imag", "")
                        if base_component in converted:
                            continue  # Skip individual real/imag components
                    
                    value = q_components[key]
                    if isinstance(value, (int, float)):
                        # Keep scalars as scalars, don't convert to tensors
                        converted[key] = float(value)
                    elif isinstance(value, (np.number, np.floating, np.integer)):
                        # CRITICAL FIX: Convert numpy scalars to Python primitives
                        # This prevents numpy.float64 values from becoming tensors
                        if np.issubdtype(type(value), np.floating):
                            converted[key] = float(value)
                        elif np.issubdtype(type(value), np.integer):
                            converted[key] = int(value)
                        else:
                            converted[key] = value.item()
                        logger.debug(f"🔧 Converted numpy scalar Q.{key}: {type(value)} -> {type(converted[key])}")
                    elif isinstance(value, np.ndarray):
                        converted[key] = self._convert_array_to_tensor(value, f"Q.{key}")
                    elif isinstance(value, (list, tuple)):
                        array_val = np.array(value)
                        converted[key] = self._convert_array_to_tensor(array_val, f"Q.{key}")
                    else:
                        # Keep other values as-is
                        converted[key] = value
        
        return converted
    
    def _convert_field_components(self, field_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert field components from numpy arrays to PyTorch tensors.
        
        Handles dynamic dimensionality while preserving mathematical precision.
        """
        converted = {}
        
        for key, value in field_components.items():
            if isinstance(value, np.ndarray):
                converted[key] = self._convert_array_to_tensor(value, f"field.{key}")
            elif isinstance(value, (list, tuple)):
                try:
                    array_val = np.array(value)
                    converted[key] = self._convert_array_to_tensor(array_val, f"field.{key}")
                except Exception as e:
                    # Data should be pre-validated, so this indicates a real error
                    raise ReconstructionConversionError(f"Cannot convert validated field component {key} to tensor: {e}")
            else:
                converted[key] = value
        
        return converted
    
    def _convert_temporal_components(self, temporal_components: Dict[str, Any]) -> Dict[str, Any]:
        """Convert temporal biography components to tensors."""
        converted = {}
        
        # MATHEMATICAL INTEGRITY: Convert temporal_momentum to LogPolarCDF representation
        
        # Check for new log-polar storage format first
        if "temporal_momentum_log_mag" in temporal_components and "temporal_momentum_phase" in temporal_components:
            logger.info("🔍 DEBUG: Found log-polar format temporal_momentum")
            log_mag = float(temporal_components["temporal_momentum_log_mag"])
            phase = float(temporal_components["temporal_momentum_phase"])
            
            # NO DEFAULTS - validate or fail hard
            if not math.isfinite(log_mag):
                raise ReconstructionConversionError(f"temporal_momentum_log_mag is {log_mag} - storage corrupted!")
            if not math.isfinite(phase):
                raise ReconstructionConversionError(f"temporal_momentum_phase is {phase} - storage corrupted!")
            
            converted["temporal_momentum"] = LogPolarCDF(log_mag, phase)
            logger.info(f"✅ temporal_momentum loaded from log-polar: exp({log_mag:.3f}) * exp(i*{phase:.3f})")
        
        # Legacy format: convert (real, imag) to log-polar
        elif "temporal_momentum_real" in temporal_components and "temporal_momentum_imag" in temporal_components:
            real_part = float(temporal_components["temporal_momentum_real"])
            imag_part = float(temporal_components["temporal_momentum_imag"])
            
            # NO GRACEFUL HANDLING - let LogPolarComplex.from_real_imag validate and fail if needed
            try:
                converted["temporal_momentum"] = LogPolarCDF.from_real_imag(real_part, imag_part)
            except ValueError as e:
                # NO DEFAULTS - if conversion fails, storage is corrupted
                raise ReconstructionConversionError(f"Legacy temporal_momentum conversion failed: {e}. Real={real_part}, Imag={imag_part}")
        
        # Already in some other format
        elif "temporal_momentum" in temporal_components:
            logger.error("❌ UNSUPPORTED: temporal_momentum in unknown format")
            raise ReconstructionConversionError("temporal_momentum found but not in log-polar or real/imag format - unsupported storage version")
        
        # Missing entirely
        else:
            logger.error(f"❌ CRITICAL: No temporal_momentum found in temporal_components!")
            logger.error(f"   Available keys: {list(temporal_components.keys())}")
            raise ReconstructionConversionError("Missing temporal_momentum in temporal_components - storage incomplete!")
        
        temporal_arrays = ["trajectory_operators", "vivid_layer", "character_layer", 
                          "frequency_evolution", "phase_coordination", "breathing_coherence"]
        
        for key, value in temporal_components.items():
            if key in temporal_arrays and isinstance(value, np.ndarray):
                converted[key] = self._convert_array_to_tensor(value, f"temporal.{key}")
            elif key not in ["temporal_momentum_real", "temporal_momentum_imag", "temporal_momentum"]:
                # Skip the real/imag components we already processed, and the combined one
                converted[key] = value
                
                # 🔍 TRACE: Debug breathing_coherence value as it flows through converter
                if key == "breathing_coherence":
                    logger.info(f"🔍 RECONSTRUCTION_CONVERTER: breathing_coherence = {value} (type: {type(value)}, finite: {np.isfinite(value) if hasattr(value, '__array__') or isinstance(value, (int, float)) else 'unknown'})")
        
        return converted
    
    def _convert_emotional_components(self, emotional_components: Dict[str, Any]) -> Dict[str, Any]:
        """Convert emotional modulation components."""
        converted = {}
        
        for key, value in emotional_components.items():
            if isinstance(value, np.ndarray):
                converted[key] = self._convert_array_to_tensor(value, f"emotional.{key}")
            elif isinstance(value, (int, float)):
                # Data is pre-validated, direct conversion
                converted[key] = float(value)
            elif isinstance(value, (np.number, np.floating, np.integer)):
                # CRITICAL FIX: Convert numpy scalars to Python primitives
                if np.issubdtype(type(value), np.floating):
                    converted[key] = float(value)
                elif np.issubdtype(type(value), np.integer):
                    converted[key] = int(value)
                else:
                    converted[key] = value.item()
                logger.debug(f"🔧 Converted numpy scalar emotional.{key}: {type(value)} -> {type(converted[key])}")
            else:
                converted[key] = value
        
        return converted
    
    def _convert_agent_state(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert agent state components."""
        converted = {}
        
        for key, value in agent_state.items():
            if isinstance(value, np.ndarray):
                converted[key] = self._convert_array_to_tensor(value, f"state.{key}")
            elif isinstance(value, (np.number, np.floating, np.integer)):
                # CRITICAL FIX: Convert numpy scalars to Python primitives
                # This prevents numpy.float64 values from becoming tensors during evolution
                original_type = type(value)
                if np.issubdtype(type(value), np.floating):
                    converted[key] = float(value)
                elif np.issubdtype(type(value), np.integer):
                    converted[key] = int(value)
                else:
                    converted[key] = value.item()  # Generic numpy scalar conversion
                
                # Log evolution parameter conversions specifically
                if key in ["sigma_i", "alpha_i", "lambda_i", "beta_i"]:
                    logger.info(f"🔧 EVOLUTION PARAM CONVERTED: {key}: {original_type} -> {type(converted[key])} (value: {converted[key]})")
                else:
                    logger.debug(f"🔧 Converted numpy scalar {key}: {original_type} -> {type(converted[key])}")
            else:
                converted[key] = value
        
        return converted
    
    def _safe_tensor_operation(self, tensor_or_array, operation_name: str = "") -> torch.Tensor:
        """
        Simple tensor conversion for validated data.
        
        Converts numpy arrays/lists to PyTorch tensors with MPS compatibility.
        ASSUMES data is already validated - no validation performed here.
        
        Args:
            tensor_or_array: PyTorch tensor or numpy array (already validated)
            operation_name: Name of operation for error reporting
            
        Returns:
            PyTorch tensor on target device
        """
        try:
            # STEP 1: Convert to numpy 
            if hasattr(tensor_or_array, 'cpu'):
                # It's a PyTorch tensor - move to CPU first
                numpy_array = tensor_or_array.cpu().detach().numpy()
            elif isinstance(tensor_or_array, np.ndarray):
                # It's already numpy
                numpy_array = tensor_or_array.copy()
            else:
                # Convert from list/tuple to numpy
                numpy_array = np.array(tensor_or_array)
            
            # STEP 2: Handle MPS compatibility (dtype conversion only)
            if self.device.type == "mps":
                # For MPS: ensure numpy array is float32 before tensor creation
                if numpy_array.dtype == np.float64:
                    numpy_array = numpy_array.astype(np.float32)
                elif numpy_array.dtype == np.complex128:
                    numpy_array = numpy_array.astype(np.complex64)
                # Create tensor with explicit float32 dtype for floating point arrays
                if numpy_array.dtype in [np.float32, np.float64]:
                    tensor = torch.from_numpy(numpy_array).to(dtype=torch.float32)
                else:
                    tensor = torch.from_numpy(numpy_array)
            else:
                # For CPU/CUDA: use default behavior
                tensor = torch.from_numpy(numpy_array)
            
            # STEP 3: Ensure compatible dtype for non-float arrays
            if tensor.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
                if self.device.type == "mps":
                    tensor = tensor.float()  # float32 for MPS
                else:
                    tensor = tensor.float()  # Default float type
            
            # STEP 4: Move to target device
            tensor = tensor.to(self.device)
            
            # Tensor conversion completed (shape/device info moved to batch summaries)
            return tensor
            
        except Exception as e:
            error_msg = f"Tensor conversion failed in {operation_name}: {e}"
            logger.error(f"❌ {error_msg}")
            raise ReconstructionConversionError(error_msg) from e

    def _convert_array_to_tensor(self, array: np.ndarray, context: str = "") -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor using safe operation pattern.
        
        Args:
            array: NumPy array to convert
            context: Context string for error reporting
            
        Returns:
            PyTorch tensor on target device
        """
        return self._safe_tensor_operation(array, f"array_to_tensor_{context}")
    
    def validate_q_magnitude(self, q_value: complex, agent_id: str = "unknown") -> bool:
        """
        Validate Q magnitude using AgentFactory patterns.
        
        Args:
            q_value: Complex Q value to validate
            agent_id: Agent identifier for logging
            
        Returns:
            True if magnitude is within valid range
        """
        try:
            q_magnitude = abs(q_value)
            
            # Validate using AgentFactory ranges
            if q_magnitude > 1e10:
                logger.error(f"💥 Agent {agent_id} - Q_magnitude ASTRONOMICAL: {q_magnitude:.2e}")
                return False
            elif q_magnitude < 1e-15:
                logger.error(f"💥 Agent {agent_id} - Q_magnitude TOO SMALL: {q_magnitude:.2e}")
                return False
            else:
                # Q_magnitude validation passed
                return True
                
        except Exception as e:
            logger.error(f"❌ Agent {agent_id} - Q_magnitude validation failed: {e}")
            return False


if __name__ == "__main__":
    converter = ReconstructionConverter()
    print("ReconstructionConverter ready for data type alignment")