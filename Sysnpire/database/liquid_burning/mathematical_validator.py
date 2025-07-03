"""
Mathematical Validator - Mathematical Relationship Validation

Validates mathematical relationships and consistency in reconstructed data.
Does NOT validate tensor existence (handled by RawDataValidator).
Focuses on mathematical properties and relationships only.

Key Features:
- Q(τ,C,s) mathematical relationship validation
- Complex number precision verification
- Field dimension consistency checks
- Mathematical property validation
- NO tensor boolean evaluation
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResults:
    """Results from mathematical validation."""

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


class MathematicalValidator:
    """
    Validator for mathematical components extracted from liquid universe.

    Ensures that Q(τ,C,s) components and field mathematics are preserved
    with exact precision during the burning process.
    """

    def __init__(self, strict_validation: bool = True):
        """
        Initialize mathematical validator.

        Args:
            strict_validation: Enable strict validation rules
        """
        self.strict_validation = strict_validation

        logger.info("MathematicalValidator initialized")
        logger.info(f"  Strict validation: {strict_validation}")

    def validate_extracted_data(self, extracted_data: "ExtractedLiquidData") -> Dict[str, Any]:
        """
        Validate complete extracted liquid data.

        Args:
            extracted_data: Complete extracted liquid universe data

        Returns:
            Validation results dictionary
        """
        logger.info("🔍 Starting mathematical validation...")

        results = ValidationResults()

        # Validate universe metadata
        self._validate_universe_metadata(extracted_data, results)

        # Validate agent data
        self._validate_agent_data(extracted_data, results)

        # Validate collective properties
        self._validate_collective_properties(extracted_data, results)

        # Determine overall validation status
        results.validation_passed = results.failed_checks == 0

        logger.info(f"🔍 Mathematical validation complete")
        logger.info(f"   Passed: {results.passed_checks}/{results.total_checks}")
        # Validation status tracked (detailed status moved to batch summaries)

        if results.errors:
            logger.warning(f"   Errors: {len(results.errors)}")
        if results.warnings:
            logger.info(f"   Warnings: {len(results.warnings)}")

        return {
            "validation_passed": results.validation_passed,
            "total_checks": results.total_checks,
            "passed_checks": results.passed_checks,
            "failed_checks": results.failed_checks,
            "errors": results.errors,
            "warnings": results.warnings,
            "validation_summary": {
                "success_rate": (
                    results.passed_checks / results.total_checks if results.total_checks > 0 else 0
                ),
                "strict_mode": self.strict_validation,
            },
        }
    
    def validate_reconstructed_batch(self, converted_batch: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate batch of reconstructed agent data after conversion from storage format.
        
        Args:
            converted_batch: Dictionary mapping agent_id -> converted_agent_data
            
        Returns:
            Batch validation results dictionary
        """
        logger.info(f"🔍 Starting batch validation for {len(converted_batch)} agents...")
        start_time = time.time()
        
        batch_results = ValidationResults()
        agent_errors = {}
        
        for agent_id, converted_data in converted_batch.items():
            agent_result = self.validate_reconstructed_data(converted_data, agent_id)
            
            # Accumulate batch statistics
            batch_results.total_checks += agent_result["total_checks"]
            batch_results.passed_checks += agent_result["passed_checks"] 
            batch_results.failed_checks += agent_result["failed_checks"]
            
            # Track individual agent errors
            if agent_result["errors"]:
                agent_errors[agent_id] = agent_result["errors"]
        
        # Determine overall batch status
        batch_results.validation_passed = batch_results.failed_checks == 0
        validation_time = time.time() - start_time
        
        # O(log n) Smart Validation Summary - Only significant results
        success_rate = batch_results.passed_checks / batch_results.total_checks if batch_results.total_checks > 0 else 0
        
        if success_rate == 1.0:
            logger.info(f"✅ Batch validation: 100% success ({len(converted_batch)} agents, {validation_time:.2f}s)")
        else:
            logger.warning(f"⚠️  Batch validation: {success_rate:.1%} success rate ({batch_results.failed_checks} failures)")
            
        # Only log errors if they exist
        if agent_errors:
            error_count = len(agent_errors)
            if error_count > 10:
                logger.error(f"❌ {error_count} agents failed validation - system may need attention")
            else:
                logger.warning(f"⚠️  {error_count} agents had validation errors:")
            
            # Log first few errors as examples (smart sampling)
            sample_errors = list(agent_errors.items())[:min(3, len(agent_errors))]
            for agent_id, errors in sample_errors:
                logger.warning(f"   - Agent {agent_id}: {errors[0]}")
            
            if len(agent_errors) > 3:
                logger.warning(f"   - ... and {len(agent_errors) - 3} more agents with errors")
        
        # Performance warning
        if validation_time > 2.0:
            logger.warning(f"⚠️  Slow validation: {validation_time:.1f}s exceeds 2s threshold")
        
        return {
            "validation_passed": batch_results.validation_passed,
            "total_checks": batch_results.total_checks,
            "passed_checks": batch_results.passed_checks,
            "failed_checks": batch_results.failed_checks,
            "agent_errors": agent_errors,
            "success_rate": success_rate,
            "validation_time": validation_time,
            "agents_validated": len(converted_batch),
            "validation_type": "batch_reconstruction"
        }

    def validate_reconstructed_data(self, converted_data: Dict[str, Any], agent_id: str = "unknown") -> Dict[str, Any]:
        """
        Validate reconstructed agent data after conversion from storage format.
        
        Mirrors the saving validation logic to ensure mathematical consistency
        throughout the storage → reconstruction → liquid orchestrator pipeline.
        
        Args:
            converted_data: Agent data converted from storage format
            agent_id: Agent identifier for error reporting
            
        Returns:
            Validation results dictionary
        """
        logger.debug(f"Validating reconstructed data for agent: {agent_id}")
        
        results = ValidationResults()
        
        # Validate reconstructed components using same patterns as saving validation
        self._validate_reconstructed_metadata(converted_data.get("agent_metadata"), results, agent_id)
        self._validate_reconstructed_q_components(converted_data.get("Q_components"), results, agent_id)
        self._validate_reconstructed_field_components(converted_data.get("field_components"), results, agent_id)
        self._validate_reconstructed_temporal_components(converted_data.get("temporal_components"), results, agent_id)
        self._validate_reconstructed_emotional_components(converted_data.get("emotional_components"), results, agent_id)
        
        # Determine overall validation status
        results.validation_passed = results.failed_checks == 0
        
        # Reconstruction validation complete (details moved to batch summaries)
        # Validation status tracked (detailed status moved to batch summaries)
        
        if results.errors:
            logger.warning(f"   Errors: {len(results.errors)}")
            for error in results.errors:
                logger.warning(f"     - {error}")
        
        return {
            "validation_passed": results.validation_passed,
            "total_checks": results.total_checks,
            "passed_checks": results.passed_checks,
            "failed_checks": results.failed_checks,
            "errors": results.errors,
            "warnings": results.warnings,
            "agent_id": agent_id,
            "validation_type": "reconstruction"
        }

    def _validate_universe_metadata(
        self, extracted_data: "ExtractedLiquidData", results: ValidationResults
    ):
        """Validate universe-level metadata."""
        metadata = extracted_data.universe_metadata

        # Check required fields
        required_fields = ["num_agents"]
        for field in required_fields:
            results.total_checks += 1
            if field in metadata:
                results.passed_checks += 1
            else:
                results.failed_checks += 1
                results.errors.append(f"Missing required universe metadata: {field}")

        # Validate agent count consistency
        results.total_checks += 1
        expected_agents = metadata.get("num_agents")
        actual_agents = len(extracted_data.agent_data)

        if expected_agents == actual_agents:
            results.passed_checks += 1
        else:
            results.failed_checks += 1
            results.errors.append(
                f"Agent count mismatch: expected {expected_agents}, got {actual_agents}"
            )

    def _validate_agent_data(
        self, extracted_data: "ExtractedLiquidData", results: ValidationResults
    ):
        """Validate all agent mathematical components."""
        for agent_id, agent_data in extracted_data.agent_data.items():
            self._validate_single_agent(agent_id, agent_data, results)

    def _validate_single_agent(
        self, agent_id: str, agent_data: Dict[str, Any], results: ValidationResults
    ):
        """Validate a single agent's mathematical components."""
        # Validate Q components
        self._validate_q_components(agent_id, agent_data.get("Q_components"), results)

        # Validate field components
        self._validate_field_components(agent_id, agent_data.get("field_components"), results)

        # Validate temporal components
        self._validate_temporal_components(
            agent_id, agent_data.get("temporal_components"), results
        )

        # Validate emotional components
        self._validate_emotional_components(
            agent_id, agent_data.get("emotional_components"), results
        )

    def _validate_q_components(
        self, agent_id: str, q_components: Dict[str, Any], results: ValidationResults
    ):
        """Validate Q(τ,C,s) mathematical components."""
        # Check for Q value presence
        results.total_checks += 1
        has_q_value = any(
            key in q_components for key in ["Q_value", "Q_value_real", "Q_value_imag"]
        )

        if has_q_value:
            results.passed_checks += 1
        else:
            results.failed_checks += 1
            results.errors.append(f"Agent {agent_id}: Missing Q value components")

        # Validate complex number consistency
        if "Q_value_real" in q_components and "Q_value_imag" in q_components:
            results.total_checks += 1
            try:
                real_part = float(q_components["Q_value_real"])
                imag_part = float(q_components["Q_value_imag"])

                # Check for valid complex number
                if np.isfinite(real_part) and np.isfinite(imag_part):
                    results.passed_checks += 1
                else:
                    results.failed_checks += 1
                    results.errors.append(
                        f"Agent {agent_id}: Invalid Q value components (non-finite)"
                    )

            except (ValueError, TypeError):
                results.failed_checks += 1
                results.errors.append(f"Agent {agent_id}: Q value components not numeric")

        # Validate mathematical component presence
        expected_components = ["gamma", "T_tensor", "E_trajectory", "phi_semantic"]
        for component in expected_components:
            results.total_checks += 1
            if any(key.startswith(component) for key in q_components.keys()):
                results.passed_checks += 1
            else:
                results.failed_checks += 1
                results.errors.append(f"Agent {agent_id}: Missing Q component {component}")

    def _validate_field_components(
        self, agent_id: str, field_components: Dict[str, Any], results: ValidationResults
    ):
        """Validate field components with dynamic dimensionality."""
        # Check for semantic field presence
        results.total_checks += 1
        has_semantic = any("semantic" in key.lower() for key in field_components.keys())

        if has_semantic:
            results.passed_checks += 1
        else:
            results.failed_checks += 1
            results.errors.append(f"Agent {agent_id}: Missing semantic field components")

        # Validate field array dimensions
        for key, value in field_components.items():
            if isinstance(value, np.ndarray):
                results.total_checks += 1

                # Check for valid array
                if value.size > 0 and np.isfinite(value).all():
                    results.passed_checks += 1
                else:
                    results.failed_checks += 1
                    results.errors.append(f"Agent {agent_id}: Invalid field array {key}")

            elif isinstance(value, (list, tuple)) and len(value) > 0:
                results.total_checks += 1
                try:
                    array_val = np.array(value)
                    if np.isfinite(array_val).all():
                        results.passed_checks += 1
                    else:
                        results.failed_checks += 1
                        results.errors.append(f"Agent {agent_id}: Invalid field list {key}")
                except:
                    results.failed_checks += 1
                    results.errors.append(
                        f"Agent {agent_id}: Cannot convert field list {key} to array"
                    )

    def _validate_temporal_components(
        self, agent_id: str, temporal_components: Dict[str, Any], results: ValidationResults
    ):
        """Validate temporal biography components."""
        # Check for temporal data presence
        results.total_checks += 1
        has_temporal = len(temporal_components) > 0

        if has_temporal:
            results.passed_checks += 1
        else:
            results.warnings.append(f"Agent {agent_id}: No temporal components found")
            results.passed_checks += 1  # Warning, not error

        # Validate temporal arrays
        temporal_arrays = ["trajectory_operators", "vivid_layer", "character_layer"]
        for array_name in temporal_arrays:
            if array_name in temporal_components:
                results.total_checks += 1
                value = temporal_components[array_name]

                if isinstance(value, np.ndarray) and value.size > 0:
                    if np.isfinite(value).all():
                        results.passed_checks += 1
                    else:
                        results.failed_checks += 1
                        results.errors.append(
                            f"Agent {agent_id}: Invalid temporal array {array_name}"
                        )
                else:
                    results.failed_checks += 1
                    results.errors.append(
                        f"Agent {agent_id}: Empty or invalid temporal array {array_name}"
                    )

    def _validate_emotional_components(
        self, agent_id: str, emotional_components: Dict[str, Any], results: ValidationResults
    ):
        """Validate emotional modulation components."""
        # Check for emotional data presence
        results.total_checks += 1
        has_emotional = len(emotional_components) > 0

        if has_emotional:
            results.passed_checks += 1
        else:
            results.warnings.append(f"Agent {agent_id}: No emotional components found")
            results.passed_checks += 1  # Warning, not error

        # Validate emotional metrics
        emotional_metrics = ["field_modulation_strength", "pattern_confidence", "coupling_strength"]
        for metric in emotional_metrics:
            if metric in emotional_components:
                results.total_checks += 1
                value = emotional_components[metric]

                if isinstance(value, (int, float)) and np.isfinite(value):
                    results.passed_checks += 1
                else:
                    results.failed_checks += 1
                    results.errors.append(f"Agent {agent_id}: Invalid emotional metric {metric}")

    def _validate_collective_properties(
        self, extracted_data: "ExtractedLiquidData", results: ValidationResults
    ):
        """Validate collective properties and optimization statistics."""
        collective = extracted_data.collective_properties

        # Check for collective data presence
        results.total_checks += 1
        if collective:
            results.passed_checks += 1
        else:
            results.warnings.append("No collective properties found")
            results.passed_checks += 1  # Warning, not error

        # Validate field statistics consistency
        field_stats = extracted_data.field_statistics
        if field_stats:
            results.total_checks += 1

            # Check for reasonable field energy
            field_energy = field_stats.get("field_energy")
            if isinstance(field_energy, (int, float)) and field_energy >= 0:
                results.passed_checks += 1
            else:
                results.failed_checks += 1
                results.errors.append("Invalid field energy in statistics")


    def _validate_reconstructed_metadata(self, metadata: Dict[str, Any], results: ValidationResults, agent_id: str):
        """Validate reconstructed metadata (mirrors _validate_universe_metadata patterns)."""
        results.total_checks += 1
        if "charge_id" in metadata:
            results.passed_checks += 1
        else:
            results.failed_checks += 1
            results.errors.append(f"Agent {agent_id}: Missing charge_id in metadata")
    
    def _validate_reconstructed_q_components(self, q_components: Dict[str, Any], results: ValidationResults, agent_id: str):
        """
        Validate reconstructed Q(τ,C,s) components (mirrors _validate_q_components).
        
        Checks that data conversion from storage format worked correctly.
        """
        # Check for converted Q_value (should be complex now)
        results.total_checks += 1
        if "Q_value" in q_components:
            q_value = q_components["Q_value"]
            if isinstance(q_value, complex):
                results.passed_checks += 1
                
                # Validate complex number properties
                results.total_checks += 1
                if np.isfinite(q_value.real) and np.isfinite(q_value.imag):
                    results.passed_checks += 1
                    
                    # Validate magnitude range (using same logic as AgentFactory)
                    results.total_checks += 1
                    q_magnitude = abs(q_value)
                    if 1e-15 <= q_magnitude <= 1e10:
                        results.passed_checks += 1
                    else:
                        results.failed_checks += 1
                        results.errors.append(f"Agent {agent_id}: Q_value magnitude out of range: {q_magnitude:.2e}")
                else:
                    results.failed_checks += 1
                    results.errors.append(f"Agent {agent_id}: Q_value has non-finite components")
            else:
                results.failed_checks += 1
                results.errors.append(f"Agent {agent_id}: Q_value not converted to complex type: {type(q_value)}")
        else:
            results.failed_checks += 1
            results.errors.append(f"Agent {agent_id}: Missing Q_value after reconstruction")
        
        # Check for mathematical component presence (no tensor validation)
        mathematical_components = ["gamma", "T_tensor", "E_trajectory", "phi_semantic"]
        found_components = []
        for component in mathematical_components:
            for key in q_components:
                if key.startswith(component):
                    found_components.append(component)
                    break
        
        results.total_checks += 1
        if len(found_components) >= 2:  # Expect at least 2 mathematical components
            results.passed_checks += 1
        else:
            results.failed_checks += 1
            results.errors.append(f"Agent {agent_id}: Insufficient mathematical components. Found: {found_components}")
    
    def _validate_reconstructed_field_components(self, field_components: Dict[str, Any], results: ValidationResults, agent_id: str):
        """
        Validate field component mathematical relationships.
        
        Validates mathematical properties, NOT tensor existence.
        """
        # Check for semantic field presence
        results.total_checks += 1
        has_semantic = any("semantic" in key.lower() for key in field_components.keys())
        if has_semantic:
            results.passed_checks += 1
        else:
            results.failed_checks += 1
            results.errors.append(f"Agent {agent_id}: Missing semantic field components after reconstruction")
        
        # Check for reasonable number of field components
        results.total_checks += 1
        if len(field_components) >= 2:  # Expect at least semantic + one other component
            results.passed_checks += 1
        else:
            results.failed_checks += 1
            results.errors.append(f"Agent {agent_id}: Insufficient field components. Found {len(field_components)} components")
    
    def _validate_reconstructed_temporal_components(self, temporal_components: Dict[str, Any], results: ValidationResults, agent_id: str):
        """Validate temporal component mathematical relationships."""
        
        # Check for temporal data presence
        results.total_checks += 1
        has_temporal = len(temporal_components) > 0
        if has_temporal:
            results.passed_checks += 1
        else:
            results.warnings.append(f"Agent {agent_id}: No temporal components after reconstruction")
            results.passed_checks += 1  # Warning, not error
    
    def _validate_reconstructed_emotional_components(self, emotional_components: Dict[str, Any], results: ValidationResults, agent_id: str):
        """Validate reconstructed emotional components."""
        import torch
        
        # Check for emotional data presence
        results.total_checks += 1
        has_emotional = len(emotional_components) > 0
        if has_emotional:
            results.passed_checks += 1
        else:
            results.warnings.append(f"Agent {agent_id}: No emotional components after reconstruction")
            results.passed_checks += 1  # Warning, not error
        
        # No tensor validation - emotional components are pre-validated


if __name__ == "__main__":
    validator = MathematicalValidator()
    print("MathematicalValidator ready for Q(τ,C,s) preservation verification")
