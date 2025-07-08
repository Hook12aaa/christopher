"""
Attribute Validator - Runtime Validation for Integration Layer

MATHEMATICAL FOUNDATION: Validates all attribute access patterns to prevent
AttributeError and ensure perfect integration with main codebase.

VALIDATION SCOPE:
1. ConceptualChargeAgent attribute structure
2. FieldConfiguration object attributes  
3. Mathematical engine interface compliance
4. Device and type consistency validation
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class AttributeValidator:
    """Runtime attribute validation for integration layer components."""

    @staticmethod
    def validate_conceptual_charge_agent(agent: Any) -> Dict[str, bool]:
        """
        Validate ConceptualChargeAgent attribute structure.

        Args:
            agent: ConceptualChargeAgent instance to validate

        Returns:
            Dict mapping attribute names to validation results
        """
        required_attributes = {
            "living_Q_value": "complex",
            "breathing_q_coefficients": "dict",
            "field_state": "object",
            "charge_id": "str",
            "device": "object",
            "temporal_biography": "object",
            "emotional_modulation": "object",
            "semantic_field": "object",
        }

        validation_results = {}

        for attr_name, expected_type in required_attributes.items():
            has_attr = hasattr(agent, attr_name)
            validation_results[f"has_{attr_name}"] = has_attr

            if has_attr:
                attr_value = getattr(agent, attr_name)
                if expected_type == "complex":
                    validation_results[f"{attr_name}_type_correct"] = isinstance(
                        attr_value, complex
                    )
                elif expected_type == "dict":
                    validation_results[f"{attr_name}_type_correct"] = isinstance(
                        attr_value, dict
                    )
                elif expected_type == "str":
                    validation_results[f"{attr_name}_type_correct"] = isinstance(
                        attr_value, str
                    )
                else:
                    validation_results[f"{attr_name}_type_correct"] = (
                        attr_value is not None
                    )
            else:
                validation_results[f"{attr_name}_type_correct"] = False

        # Special validation for nested attributes
        if hasattr(agent, "field_state"):
            field_state = agent.field_state
            validation_results["field_state_has_field_position"] = hasattr(
                field_state, "field_position"
            )

        if hasattr(agent, "breathing_q_coefficients"):
            breathing_coeffs = agent.breathing_q_coefficients
            if isinstance(breathing_coeffs, dict):
                validation_results["breathing_coefficients_are_complex"] = all(
                    isinstance(v, complex) for v in breathing_coeffs.values()
                )
            else:
                validation_results["breathing_coefficients_are_complex"] = False

        return validation_results

    @staticmethod
    def validate_field_configuration(field_config: Any) -> Dict[str, bool]:
        """
        Validate FieldConfiguration object structure.

        Args:
            field_config: FieldConfiguration instance to validate

        Returns:
            Dict mapping attribute names to validation results
        """
        required_attributes = [
            "field_values",
            "spatial_gradients",
            "temporal_derivatives",
            "energy_density",
            "total_energy",
            "correlation_length",
        ]

        validation_results = {}

        for attr_name in required_attributes:
            has_attr = hasattr(field_config, attr_name)
            validation_results[f"has_{attr_name}"] = has_attr

            if has_attr:
                attr_value = getattr(field_config, attr_name)
                validation_results[f"{attr_name}_not_none"] = attr_value is not None

        return validation_results

    @staticmethod
    def validate_mathematical_engine(
        engine: Any, expected_methods: List[str]
    ) -> Dict[str, bool]:
        """
        Validate mathematical engine interface compliance.

        Args:
            engine: Mathematical engine instance
            expected_methods: List of required method names

        Returns:
            Dict mapping method names to validation results
        """
        validation_results = {}

        for method_name in expected_methods:
            has_method = hasattr(engine, method_name)
            validation_results[f"has_{method_name}"] = has_method

            if has_method:
                method_obj = getattr(engine, method_name)
                validation_results[f"{method_name}_callable"] = callable(method_obj)
            else:
                validation_results[f"{method_name}_callable"] = False

        return validation_results

    @staticmethod
    def validate_device_consistency(obj: Any) -> Dict[str, bool]:
        """
        Validate device placement consistency.

        Args:
            obj: Object with potential device attributes

        Returns:
            Dict with device validation results
        """
        validation_results = {}

        # Check for device attribute
        has_device = hasattr(obj, "device")
        validation_results["has_device"] = has_device

        if has_device:
            device = obj.device
            validation_results["device_not_none"] = device is not None

            # Check if it's a torch.device
            try:
                import torch

                validation_results["is_torch_device"] = isinstance(device, torch.device)

                if isinstance(device, torch.device):
                    device_type = device.type
                    validation_results["device_type_valid"] = device_type in [
                        "cpu",
                        "cuda",
                        "mps",
                    ]
                else:
                    validation_results["device_type_valid"] = False

            except ImportError:
                validation_results["is_torch_device"] = False
                validation_results["device_type_valid"] = False
        else:
            validation_results["device_not_none"] = False
            validation_results["is_torch_device"] = False
            validation_results["device_type_valid"] = False

        return validation_results


def safe_getattr(
    obj: Any, attr_name: str, default: Any = None, validate_type: Optional[type] = None
) -> Any:
    """
    Safe attribute access with type validation.

    Args:
        obj: Object to get attribute from
        attr_name: Name of attribute
        default: Default value if attribute missing
        validate_type: Expected type for validation

    Returns:
        Attribute value or default

    Raises:
        AttributeError: If attribute missing and no default
        TypeError: If attribute exists but wrong type
    """
    if not hasattr(obj, attr_name):
        if default is not None:
            return default
        else:
            raise AttributeError(
                f"Object {type(obj).__name__} lacks required attribute '{attr_name}'"
            )

    value = getattr(obj, attr_name)

    if validate_type is not None and not isinstance(value, validate_type):
        raise TypeError(
            f"Attribute '{attr_name}' expected {validate_type.__name__}, got {type(value).__name__}"
        )

    return value


def safe_method_call(obj: Any, method_name: str, *args, **kwargs) -> Any:
    """
    Safe method call with existence validation.

    Args:
        obj: Object to call method on
        method_name: Name of method
        *args: Method arguments
        **kwargs: Method keyword arguments

    Returns:
        Method result

    Raises:
        AttributeError: If method doesn't exist
        TypeError: If attribute exists but not callable
    """
    if not hasattr(obj, method_name):
        raise AttributeError(
            f"Object {type(obj).__name__} lacks method '{method_name}'"
        )

    method = getattr(obj, method_name)

    if not callable(method):
        raise TypeError(f"Attribute '{method_name}' is not callable")

    return method(*args, **kwargs)


def validate_integration_layer_connections() -> Dict[str, Dict[str, bool]]:
    """
    Validate all connections in integration layer.

    Returns:
        Comprehensive validation report
    """
    validation_report = {}

    # Validate imports (use absolute imports when run directly)
    try:
        import field_mechanics

        validation_report["field_mechanics_import"] = {"success": True}
    except ImportError as e:
        validation_report["field_mechanics_import"] = {
            "success": False,
            "error": str(e),
        }

    try:
        import selection_pressure

        validation_report["selection_pressure_import"] = {"success": True}
    except ImportError as e:
        validation_report["selection_pressure_import"] = {
            "success": False,
            "error": str(e),
        }

    try:
        from field_integrator import FieldIntegrator

        validation_report["field_integrator_import"] = {"success": True}

        # Validate FieldIntegrator interface
        integrator = FieldIntegrator()
        expected_methods = [
            "get_universe_field_state",
            "evaluate_mathematical_weight",
            "compute_field_compatibility",
            "text_to_field_signature",
        ]

        validation_report["field_integrator_interface"] = (
            AttributeValidator.validate_mathematical_engine(
                integrator, expected_methods
            )
        )

    except Exception as e:
        validation_report["field_integrator_import"] = {
            "success": False,
            "error": str(e),
        }

    return validation_report


if __name__ == "__main__":
    # Run validation when executed directly
    logger.info("🔍 ATTRIBUTE VALIDATION: Starting comprehensive validation...")

    report = validate_integration_layer_connections()

    all_passed = all(
        result.get("success", True) if isinstance(result, dict) else True
        for result in report.values()
    )

    if all_passed:
        logger.info("✅ ATTRIBUTE VALIDATION: All connections validated successfully")
    else:
        logger.error("❌ ATTRIBUTE VALIDATION: Issues detected")
        for component, result in report.items():
            if isinstance(result, dict) and not result.get("success", True):
                logger.error(f"   {component}: {result.get('error', 'Unknown issue')}")
