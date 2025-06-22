"""
Raw Data Validator - Validate Data BEFORE Tensor Conversion

Validates numpy arrays, lists, and primitive types BEFORE converting to tensors.
This prevents the need for tensor boolean evaluation by ensuring all validation
happens on raw data types that support safe boolean operations.

CRITICAL PRINCIPLE: NO TENSORS - Only numpy arrays, lists, and primitives
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RawValidationResult:
    """Results from raw data validation."""
    
    validation_passed: bool = False
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class RawDataValidationError(Exception):
    """Raised when raw data validation fails."""
    pass


class RawDataValidator:
    """
    Validates raw data (numpy arrays, lists, primitives) BEFORE tensor conversion.
    
    This validator works exclusively with data types that support safe boolean
    evaluation (numpy arrays, lists, primitives) and never touches tensors.
    
    Key Features:
    - Validates numpy arrays using safe numpy operations
    - Validates lists and tuples using len() and standard Python operations
    - Validates complex numbers and primitive types
    - NO tensor validation - only raw data
    """
    
    def __init__(self, strict_validation: bool = True):
        """
        Initialize raw data validator.
        
        Args:
            strict_validation: Enable strict validation with detailed checks
        """
        self.strict_validation = strict_validation
        logger.info("RawDataValidator initialized")
        logger.info(f"  Strict validation: {strict_validation}")
    
    def validate_raw_agent_data(self, stored_agent_data: Dict[str, Any], agent_id: str = "unknown") -> RawValidationResult:
        """
        Validate complete raw agent data before tensor conversion.
        
        Args:
            stored_agent_data: Raw agent data from storage (numpy/lists/primitives)
            agent_id: Agent identifier for error reporting
            
        Returns:
            RawValidationResult with validation status and detailed results
        """
        logger.debug(f"🔍 Validating raw data for agent {agent_id}")
        
        result = RawValidationResult()
        
        try:
            # Validate metadata (primitives)
            if "agent_metadata" in stored_agent_data:
                self._validate_raw_metadata(stored_agent_data["agent_metadata"], result, agent_id)
            else:
                result.failed_checks += 1
                result.errors.append(f"Agent {agent_id}: Missing agent_metadata")
            
            # Validate Q components (complex numbers and primitives)
            if "Q_components" in stored_agent_data:
                self._validate_raw_q_components(stored_agent_data["Q_components"], result, agent_id)
            else:
                result.warnings.append(f"Agent {agent_id}: Missing Q_components")
            
            # Validate field components (numpy arrays and lists)
            if "field_components" in stored_agent_data:
                self._validate_raw_field_components(stored_agent_data["field_components"], result, agent_id)
            else:
                result.warnings.append(f"Agent {agent_id}: Missing field_components")
            
            # Validate temporal components (numpy arrays)
            if "temporal_components" in stored_agent_data:
                self._validate_raw_temporal_components(stored_agent_data["temporal_components"], result, agent_id)
            else:
                result.warnings.append(f"Agent {agent_id}: Missing temporal_components")
            
            # Validate emotional components (numpy arrays and primitives)
            if "emotional_components" in stored_agent_data:
                self._validate_raw_emotional_components(stored_agent_data["emotional_components"], result, agent_id)
            else:
                result.warnings.append(f"Agent {agent_id}: Missing emotional_components")
            
            # Validate agent state (primitives and dicts)
            if "agent_state" in stored_agent_data:
                self._validate_raw_agent_state(stored_agent_data["agent_state"], result, agent_id)
            else:
                result.warnings.append(f"Agent {agent_id}: Missing agent_state")
            
            # Final validation result
            result.validation_passed = result.failed_checks == 0
            
            logger.debug(f"✅ Raw data validation complete for agent {agent_id}")
            logger.debug(f"   Passed: {result.validation_passed}, Checks: {result.passed_checks}/{result.total_checks}")
            
            return result
            
        except Exception as e:
            result.validation_passed = False
            result.failed_checks += 1
            result.errors.append(f"Agent {agent_id}: Raw data validation exception: {e}")
            logger.error(f"❌ Raw data validation failed for agent {agent_id}: {e}")
            return result
    
    def _validate_raw_metadata(self, metadata: Dict[str, Any], result: RawValidationResult, agent_id: str):
        """Validate metadata (primitives only)."""
        result.total_checks += 1
        
        # Check for required primitive fields
        required_fields = ["charge_id", "text_source"]
        missing_fields = []
        
        for field in required_fields:
            if field not in metadata or metadata[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            result.failed_checks += 1
            result.errors.append(f"Agent {agent_id}: Missing required metadata fields: {missing_fields}")
        else:
            result.passed_checks += 1
            logger.debug(f"✅ Agent {agent_id}: Metadata validation passed")
    
    def _validate_raw_q_components(self, q_components: Dict[str, Any], result: RawValidationResult, agent_id: str):
        """Validate Q components (complex numbers and primitives)."""
        result.total_checks += 1
        
        # Check for Q value (real/imag pairs or complex)
        has_q_value = False
        
        if "Q_value_real" in q_components and "Q_value_imag" in q_components:
            # Validate real/imag pairs (primitives)
            q_real = q_components["Q_value_real"]
            q_imag = q_components["Q_value_imag"]
            
            if isinstance(q_real, (int, float)) and isinstance(q_imag, (int, float)):
                if np.isfinite(q_real) and np.isfinite(q_imag):
                    has_q_value = True
                    logger.debug(f"✅ Agent {agent_id}: Q_value real/imag validation passed")
                else:
                    result.errors.append(f"Agent {agent_id}: Non-finite Q_value components")
            else:
                result.errors.append(f"Agent {agent_id}: Invalid Q_value component types")
        
        elif "Q_value" in q_components:
            # Validate complex Q value
            q_value = q_components["Q_value"]
            if isinstance(q_value, complex):
                if np.isfinite(q_value.real) and np.isfinite(q_value.imag):
                    has_q_value = True
                    logger.debug(f"✅ Agent {agent_id}: Q_value complex validation passed")
                else:
                    result.errors.append(f"Agent {agent_id}: Non-finite complex Q_value")
            else:
                result.errors.append(f"Agent {agent_id}: Q_value not a complex number")
        
        if has_q_value:
            result.passed_checks += 1
        else:
            result.failed_checks += 1
            result.errors.append(f"Agent {agent_id}: No valid Q_value found")
    
    def _validate_raw_field_components(self, field_components: Dict[str, Any], result: RawValidationResult, agent_id: str):
        """Validate field components (numpy arrays and lists)."""
        result.total_checks += 1
        
        # Check for semantic field presence
        has_semantic = any("semantic" in key.lower() for key in field_components.keys())
        if not has_semantic:
            result.failed_checks += 1
            result.errors.append(f"Agent {agent_id}: Missing semantic field components")
            return
        
        # Validate each field component
        valid_components = 0
        total_components = 0
        
        for key, value in field_components.items():
            total_components += 1
            
            if isinstance(value, np.ndarray):
                # ✅ SAFE: numpy array validation
                if value.size > 0 and np.isfinite(value).all():
                    valid_components += 1
                    logger.debug(f"✅ Agent {agent_id}: Field component {key} (numpy) validation passed")
                else:
                    if value.size == 0:
                        result.errors.append(f"Agent {agent_id}: Empty field component {key}")
                    else:
                        result.errors.append(f"Agent {agent_id}: Non-finite values in field component {key}")
            
            elif isinstance(value, (list, tuple)):
                # ✅ SAFE: list/tuple validation  
                if len(value) > 0:
                    try:
                        # Convert to numpy for finite check
                        array_val = np.array(value)
                        if np.isfinite(array_val).all():
                            valid_components += 1
                            logger.debug(f"✅ Agent {agent_id}: Field component {key} (list) validation passed")
                        else:
                            result.errors.append(f"Agent {agent_id}: Non-finite values in field component {key}")
                    except Exception as e:
                        result.errors.append(f"Agent {agent_id}: Cannot validate field component {key}: {e}")
                else:
                    result.errors.append(f"Agent {agent_id}: Empty field component {key}")
            
            elif isinstance(value, (int, float)):
                # ✅ SAFE: primitive validation
                if np.isfinite(value):
                    valid_components += 1
                    logger.debug(f"✅ Agent {agent_id}: Field component {key} (primitive) validation passed")
                else:
                    result.errors.append(f"Agent {agent_id}: Non-finite field component {key}")
            
            else:
                result.warnings.append(f"Agent {agent_id}: Unknown field component type {key}: {type(value)}")
        
        if valid_components == total_components and total_components > 0:
            result.passed_checks += 1
        else:
            result.failed_checks += 1
    
    def _validate_raw_temporal_components(self, temporal_components: Dict[str, Any], result: RawValidationResult, agent_id: str):
        """Validate temporal components (numpy arrays)."""
        result.total_checks += 1
        
        if len(temporal_components) == 0:
            result.warnings.append(f"Agent {agent_id}: No temporal components to validate")
            result.passed_checks += 1
            return
        
        valid_components = 0
        total_components = len(temporal_components)
        
        for key, value in temporal_components.items():
            if isinstance(value, np.ndarray):
                # ✅ SAFE: numpy array validation
                if value.size > 0 and np.isfinite(value).all():
                    valid_components += 1
                    logger.debug(f"✅ Agent {agent_id}: Temporal component {key} validation passed")
                else:
                    result.errors.append(f"Agent {agent_id}: Invalid temporal component {key}")
            
            elif isinstance(value, (int, float, complex)):
                # ✅ SAFE: primitive validation
                if isinstance(value, complex):
                    if np.isfinite(value.real) and np.isfinite(value.imag):
                        valid_components += 1
                    else:
                        result.errors.append(f"Agent {agent_id}: Non-finite temporal component {key}")
                else:
                    if np.isfinite(value):
                        valid_components += 1
                    else:
                        result.errors.append(f"Agent {agent_id}: Non-finite temporal component {key}")
            
            else:
                result.warnings.append(f"Agent {agent_id}: Unknown temporal component type {key}: {type(value)}")
        
        if valid_components == total_components:
            result.passed_checks += 1
        else:
            result.failed_checks += 1
    
    def _validate_raw_emotional_components(self, emotional_components: Dict[str, Any], result: RawValidationResult, agent_id: str):
        """Validate emotional components (numpy arrays and primitives)."""
        result.total_checks += 1
        
        if len(emotional_components) == 0:
            result.warnings.append(f"Agent {agent_id}: No emotional components to validate")
            result.passed_checks += 1
            return
        
        valid_components = 0
        total_components = len(emotional_components)
        
        for key, value in emotional_components.items():
            if isinstance(value, np.ndarray):
                # ✅ SAFE: numpy array validation
                if value.size > 0 and np.isfinite(value).all():
                    valid_components += 1
                    logger.debug(f"✅ Agent {agent_id}: Emotional component {key} validation passed")
                else:
                    result.errors.append(f"Agent {agent_id}: Invalid emotional component {key}")
            
            elif isinstance(value, (int, float)):
                # ✅ SAFE: primitive validation
                if np.isfinite(value):
                    valid_components += 1
                else:
                    result.errors.append(f"Agent {agent_id}: Non-finite emotional component {key}")
            
            else:
                result.warnings.append(f"Agent {agent_id}: Unknown emotional component type {key}: {type(value)}")
        
        if valid_components == total_components:
            result.passed_checks += 1
        else:
            result.failed_checks += 1
    
    def _validate_raw_agent_state(self, agent_state: Dict[str, Any], result: RawValidationResult, agent_id: str):
        """Validate agent state (primitives and dicts)."""
        result.total_checks += 1
        
        if len(agent_state) == 0:
            result.warnings.append(f"Agent {agent_id}: Empty agent state")
            result.passed_checks += 1
            return
        
        # Validate state contains reasonable data
        valid_state_fields = 0
        total_state_fields = len(agent_state)
        
        for key, value in agent_state.items():
            if isinstance(value, (int, float)):
                # ✅ SAFE: primitive validation
                if np.isfinite(value):
                    valid_state_fields += 1
                else:
                    result.errors.append(f"Agent {agent_id}: Non-finite agent state {key}")
            
            elif isinstance(value, dict):
                # ✅ SAFE: dict validation
                if len(value) > 0:
                    valid_state_fields += 1
                else:
                    result.warnings.append(f"Agent {agent_id}: Empty agent state dict {key}")
                    valid_state_fields += 1  # Empty dict is not an error
            
            elif value is None:
                result.warnings.append(f"Agent {agent_id}: None agent state {key}")
                valid_state_fields += 1  # None is acceptable for some fields
            
            else:
                result.warnings.append(f"Agent {agent_id}: Unknown agent state type {key}: {type(value)}")
                valid_state_fields += 1  # Don't fail on unknown types
        
        if valid_state_fields == total_state_fields:
            result.passed_checks += 1
        else:
            result.failed_checks += 1


if __name__ == "__main__":
    # Test raw data validator
    print("Testing Raw Data Validator...")
    
    # Create test data with numpy arrays and lists (NO TENSORS)
    test_data = {
        "agent_metadata": {
            "charge_id": "test_001",
            "text_source": "test data"
        },
        "Q_components": {
            "Q_value_real": 1.5,
            "Q_value_imag": 0.8,
            "gamma": 1.0
        },
        "field_components": {
            "semantic_embedding": np.array([0.1, 0.2, 0.3, 0.4]),  # ✅ numpy array
            "phase_total": 1.57
        },
        "temporal_components": {
            "trajectory_operators": np.array([[1.0, 2.0], [3.0, 4.0]])  # ✅ numpy array
        },
        "emotional_components": {
            "some_emotional_array": np.array([0.5, 0.6, 0.7])  # ✅ numpy array
        },
        "agent_state": {
            "sigma_i": 0.1,
            "evolution_rates": {"cascading": 0.05}
        }
    }
    
    validator = RawDataValidator()
    result = validator.validate_raw_agent_data(test_data, "test_001")
    
    print(f"✅ Validation passed: {result.validation_passed}")
    print(f"   Total checks: {result.total_checks}")
    print(f"   Passed: {result.passed_checks}")
    print(f"   Failed: {result.failed_checks}")
    if result.errors:
        print(f"   Errors: {result.errors}")
    if result.warnings:
        print(f"   Warnings: {result.warnings}")
    
    print("\n✅ Raw data validation test completed!")