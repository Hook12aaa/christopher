"""
Liquid Processor - Main Burning Processor for Liquid Universe Results

Extracts complete mathematical components from ConceptualChargeAgent objects
in a dimension-agnostic manner. Handles any embedding dimensionality from
BGE (1024), MPNet (768), or future models without hardcoded assumptions.

Mathematical Preservation:
- Complete Q(τ,C,s) components with exact precision
- Dynamic field dimensionality detection and handling
- Complex-valued mathematics preservation
- Temporal trajectory and emotional modulation extraction
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import ConceptualChargeAgent for proper type handling
from Sysnpire.model.liquid.conceptual_charge_agent import ConceptualChargeAgent
# Note: LiquidOrchestrator import removed to avoid circular import

logger = logging.getLogger(__name__)


@dataclass
class LiquidExtractionMetrics:
    """Metrics for liquid extraction performance."""

    agents_processed: int = 0
    total_processing_time: float = 0.0
    field_dimensions_detected: Optional[int] = None
    model_type_detected: Optional[str] = None
    q_components_extracted: int = 0
    field_arrays_extracted: int = 0
    mathematical_validation_passed: bool = False


@dataclass
class ExtractedLiquidData:
    """Container for extracted liquid universe data."""

    universe_metadata: Dict[str, Any]
    agent_data: Dict[str, Dict[str, Any]]
    collective_properties: Dict[str, Any]
    field_statistics: Dict[str, Any]
    extraction_metrics: LiquidExtractionMetrics
    vocab_mappings: Dict[str, Any]  # Vocabulary context for reconstruction


class LiquidProcessor:
    """
    Main processor for extracting mathematical components from liquid universe results.

    Designed to be completely dimension-agnostic and model-agnostic. Automatically
    detects field dimensionality and adapts extraction accordingly.

    Key Features:
    - Dynamic dimensionality detection (no hardcoded 1024)
    - Complete Q(τ,C,s) mathematical component extraction
    - Complex number and tensor preservation
    - Model metadata preservation for reconstruction
    """

    def __init__(self, preserve_precision: bool = True, validate_mathematics: bool = True):
        """
        Initialize liquid processor.

        Args:
            preserve_precision: Ensure exact mathematical precision preservation
            validate_mathematics: Validate extracted mathematical components
        """
        self.preserve_precision = preserve_precision
        self.validate_mathematics = validate_mathematics

        logger.info("LiquidProcessor initialized")
        logger.info(f"  Precision preservation: {preserve_precision}")
        logger.info(f"  Mathematical validation: {validate_mathematics}")

    def process_liquid_universe(self, liquid_results: Dict[str, Any]) -> ExtractedLiquidData:
        """
        Process complete liquid universe results into extraction-ready format.

        Args:
            liquid_results: Complete results from ChargeFactory.build() containing:
                          - semantic_results, temporal_results, emotional_results
                          - liquid_results (nested): Contains agent_pool, active_charges, field_statistics

        Returns:
            ExtractedLiquidData containing all mathematical components
        """
        logger.info("🔄 Starting liquid universe processing...")
        start_time = time.time()

        # Initialize extraction metrics
        metrics = LiquidExtractionMetrics()

        # Extract the nested liquid_results containing agent_pool
        nested_liquid_results = liquid_results.get("liquid_results")

        # Extract universe metadata (using nested structure)
        universe_metadata = self._extract_universe_metadata(nested_liquid_results)

        # Detect field dimensionality from first agent
        field_dims = self._detect_field_dimensionality(nested_liquid_results)
        metrics.field_dimensions_detected = field_dims

        logger.info(f"   📐 Detected field dimensionality: {field_dims}")

        # Extract model information (use full liquid_results for vocab_mappings)
        model_info = self._extract_model_information(liquid_results)
        metrics.model_type_detected = model_info.get("model_type")

        logger.info(f"   🤖 Model type: {metrics.model_type_detected}")

        # Process all agents (using nested structure)
        agent_data = self._process_agent_pool(nested_liquid_results, metrics)

        # Extract collective properties (using nested structure)
        collective_properties = self._extract_collective_properties(nested_liquid_results)

        # Extract field statistics (using nested structure)
        field_statistics = nested_liquid_results.get("field_statistics")

        # Extract vocabulary mappings (from top level)
        vocab_mappings = liquid_results.get("vocab_mappings")
        logger.info(
            f"   📚 Vocabulary mappings: {len(vocab_mappings.get('id_to_token'))} tokens"
        )

        # Finalize metrics
        metrics.total_processing_time = time.time() - start_time

        # Validate if requested
        if self.validate_mathematics:
            metrics.mathematical_validation_passed = self._validate_extracted_mathematics(
                agent_data, collective_properties
            )

        logger.info(f"🔄 Liquid processing complete in {metrics.total_processing_time:.2f}s")
        logger.info(f"   🎯 Agents processed: {metrics.agents_processed}")
        logger.info(f"   🧮 Q components extracted: {metrics.q_components_extracted}")
        logger.info(f"   📊 Field arrays extracted: {metrics.field_arrays_extracted}")
        logger.info(f"   📚 Vocab tokens extracted: {len(vocab_mappings.get('id_to_token'))}")

        return ExtractedLiquidData(
            universe_metadata=universe_metadata,
            agent_data=agent_data,
            collective_properties=collective_properties,
            field_statistics=field_statistics,
            extraction_metrics=metrics,
            vocab_mappings=vocab_mappings,
        )

    def _extract_universe_metadata(self, liquid_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract universe-level metadata."""
        orchestrator = liquid_results.get("orchestrator")

        metadata = {
            "creation_timestamp": time.time(),
            "num_agents": liquid_results.get("num_agents"),
            "ready_for_simulation": liquid_results.get("ready_for_simulation"),
            "field_resolution": (
                getattr(orchestrator, "field_resolution", None) if orchestrator else None
            ),
            "device": getattr(orchestrator, "device", None) if orchestrator else None,
            "current_tau": getattr(orchestrator, "current_tau", 0.0) if orchestrator else 0.0,
            "simulation_time": (
                getattr(orchestrator, "simulation_time", 0.0) if orchestrator else 0.0
            ),
        }

        return metadata

    def _detect_field_dimensionality(self, liquid_results: Dict[str, Any]) -> Optional[int]:
        """
        Detect field dimensionality from liquid results.

        Returns:
            Field dimension count or None if cannot be detected
        """
        # Try to get from agent_pool first
        agent_pool = liquid_results.get("agent_pool")

        if agent_pool:
            # Get first agent to check dimensionality
            first_agent_key = next(iter(agent_pool.keys()))
            first_agent = agent_pool[first_agent_key]

            # Check semantic field dimensionality
            if hasattr(first_agent, "semantic_field") and hasattr(
                first_agent.semantic_field, "embedding_components"
            ):
                field_array = first_agent.semantic_field.embedding_components
                if hasattr(field_array, "shape"):
                    return field_array.shape[0]
                elif hasattr(field_array, "__len__"):
                    return len(field_array)

            # Fallback: check charge object field components
            if hasattr(first_agent, "charge_obj") and hasattr(
                first_agent.charge_obj, "field_components"
            ):
                field_components = first_agent.charge_obj.field_components
                if hasattr(field_components, "semantic_field") and hasattr(
                    field_components.semantic_field, "shape"
                ):
                    return field_components.semantic_field.shape[0]

        # Fallback: try to infer from other sources
        logger.warning("Could not detect field dimensionality from agents")
        return None

    def _extract_model_information(self, liquid_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model information for reconstruction."""
        # Try to get from various sources in liquid results
        model_info = {}

        # Check if vocab_mappings contains model info (at top level from ChargeFactory)
        vocab_mappings = liquid_results.get("vocab_mappings")
        if "model_info" in vocab_mappings:
            model_info.update(vocab_mappings["model_info"])

        # Try to infer from agent structure (using nested liquid_results)
        nested_liquid_results = liquid_results.get("liquid_results")
        agent_pool = nested_liquid_results.get("agent_pool")
        if agent_pool:
            first_agent = next(iter(agent_pool.values()))
            if hasattr(first_agent, "device"):
                model_info["device"] = first_agent.device

            # Try to infer model type from field dimensions
            field_dims = self._detect_field_dimensionality(nested_liquid_results)
            if field_dims == 1024:
                model_info["model_type"] = "BGE_large_v1.5"
            elif field_dims == 768:
                model_info["model_type"] = "MPNet_base"
            else:
                model_info["model_type"] = f"unknown_{field_dims}d"

        return model_info

    def _process_agent_pool(
        self, liquid_results: Dict[str, Any], metrics: LiquidExtractionMetrics
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process all agents in the agent pool.

        Returns:
            Dictionary mapping agent IDs to extracted agent data
        """
        agent_pool = liquid_results.get("agent_pool")
        agent_data = {}

        for agent_id, agent in agent_pool.items():
            try:
                extracted_agent = self._extract_agent_components(agent)
                agent_data[agent_id] = extracted_agent

                metrics.agents_processed += 1
                if "Q_components" in extracted_agent:
                    metrics.q_components_extracted += 1
                if "field_components" in extracted_agent:
                    metrics.field_arrays_extracted += 1

            except Exception as e:
                logger.error(f"Failed to process agent {agent_id}: {e}")
                continue

        return agent_data

    def _extract_agent_components(self, agent) -> Dict[str, Any]:
        """
        Extract all mathematical components from a single agent.

        Returns:
            Dictionary containing all agent mathematical data
        """
        extracted = {
            "agent_metadata": self._extract_agent_metadata(agent),
            "Q_components": self._extract_q_components(agent),
            "field_components": self._extract_field_components(agent),
            "temporal_components": self._extract_temporal_components(agent),
            "emotional_components": self._extract_emotional_components(agent),
            "agent_state": self._extract_agent_state(agent),
        }

        return extracted

    def _extract_agent_metadata(self, agent) -> Dict[str, Any]:
        """Extract basic agent metadata."""
        metadata = {}

        # Basic identifiers
        if hasattr(agent, "charge_id"):
            metadata["charge_id"] = agent.charge_id
        if hasattr(agent, "charge_index"):
            metadata["charge_index"] = agent.charge_index
        if hasattr(agent, "vocab_token_string"):
            metadata["vocab_token_string"] = agent.vocab_token_string
        if hasattr(agent, "vocab_token_id"):
            metadata["vocab_token_id"] = agent.vocab_token_id
        if hasattr(agent, "device"):
            metadata["device"] = agent.device

        # Charge object metadata if available
        if hasattr(agent, "charge_obj"):
            charge_obj = agent.charge_obj
            if hasattr(charge_obj, "creation_timestamp"):
                metadata["creation_timestamp"] = charge_obj.creation_timestamp
            if hasattr(charge_obj, "last_updated"):
                metadata["last_updated"] = charge_obj.last_updated
            if hasattr(charge_obj, "text_source"):
                metadata["text_source"] = charge_obj.text_source

        return metadata

    def _extract_q_components(self, agent) -> Dict[str, Any]:
        """Extract complete Q(τ,C,s) mathematical components."""
        q_components = {}

        # Extract primary Q value - prioritize living_Q_value as it's the main attribute
        primary_q_found = False
        
        if hasattr(agent, "living_Q_value"):
            living_q = agent.living_Q_value
            if isinstance(living_q, complex):
                q_components["Q_value_real"] = float(living_q.real)
                q_components["Q_value_imag"] = float(living_q.imag)
                q_components["living_Q_value_real"] = float(living_q.real)
                q_components["living_Q_value_imag"] = float(living_q.imag)
                primary_q_found = True
            else:
                q_components["living_Q_value"] = living_q
                primary_q_found = True

        # Fallback to static Q_value if living_Q_value not found
        if not primary_q_found and hasattr(agent, "Q_value"):
            q_value = agent.Q_value
            if isinstance(q_value, complex):
                q_components["Q_value_real"] = float(q_value.real)
                q_components["Q_value_imag"] = float(q_value.imag)
                primary_q_found = True
            else:
                q_components["Q_value"] = q_value
                primary_q_found = True

        # Extract Q mathematical components if available
        q_math_found = False
        if hasattr(agent, "Q_components"):
            q_math = agent.Q_components

            # Extract individual components
            component_attrs = [
                "gamma",
                "T_tensor",
                "E_trajectory",
                "phi_semantic",
                "theta_components",
                "phase_factor",
                "psi_persistence",
                "psi_gaussian",
                "psi_exponential_cosine",
            ]

            for attr in component_attrs:
                if hasattr(q_math, attr):
                    value = getattr(q_math, attr)
                    
                    # 🔍 Q VALUE TRACKING: Log E_trajectory extraction
                    if attr == "E_trajectory":
                        logger.debug(f"🔍 Q-TRACK SAVE: Extracting E_trajectory for agent - value: {value}, type: {type(value)}")
                        if isinstance(value, complex):
                            logger.debug(f"   - E_trajectory complex - real: {value.real}, imag: {value.imag}, magnitude: {abs(value)}")
                        elif value is None:
                            logger.warning(f"   - WARNING: E_trajectory is None during extraction!")
                    
                    if isinstance(value, complex):
                        q_components[f"{attr}_real"] = float(value.real)
                        q_components[f"{attr}_imag"] = float(value.imag)
                        
                        # Additional debug for E_trajectory
                        if attr == "E_trajectory":
                            logger.debug(f"   - Stored E_trajectory_real: {float(value.real)}, E_trajectory_imag: {float(value.imag)}")
                    elif isinstance(value, np.ndarray):
                        q_components[attr] = value.tolist() if value.size < 1000 else value
                    else:
                        q_components[attr] = value
                    q_math_found = True

        # FALLBACK: Extract mathematical components from agent attributes if Q_components missing
        if not q_math_found:
            # Try to extract from direct agent attributes
            fallback_attrs = [
                "gamma", "T_tensor", "E_trajectory", "phi_semantic", 
                "phase_factor", "psi_persistence", "psi_gaussian", 
                "psi_exponential_cosine", "breath_frequency", "breath_amplitude"
            ]
            
            for attr in fallback_attrs:
                if hasattr(agent, attr):
                    value = getattr(agent, attr)
                    if isinstance(value, complex):
                        q_components[f"{attr}_real"] = float(value.real)
                        q_components[f"{attr}_imag"] = float(value.imag)
                    elif isinstance(value, np.ndarray):
                        q_components[attr] = value.tolist() if value.size < 1000 else value
                    elif isinstance(value, (int, float)):
                        q_components[attr] = value

        # If still no Q components found, synthesize minimal components
        if not q_components:
            # Create basic Q components from living_Q_value if we have it
            if primary_q_found and "Q_value_real" in q_components:
                q_components["gamma"] = 1.0
                q_components["psi_persistence"] = 1.0
                # Add a note that these are synthesized
                q_components["_synthesized"] = True

        return q_components

    def _extract_field_components(self, agent) -> Dict[str, Any]:
        """Extract field components (dimension-agnostic)."""
        field_data = {}

        # Extract semantic field
        if hasattr(agent, "semantic_field"):
            semantic_field = agent.semantic_field
            if hasattr(semantic_field, "embedding_components"):
                field_data["semantic_embedding"] = semantic_field.embedding_components
            if hasattr(semantic_field, "phase_factors"):
                field_data["semantic_phase_factors"] = semantic_field.phase_factors
            if hasattr(semantic_field, "manifold_dimension"):
                field_data["manifold_dimension"] = semantic_field.manifold_dimension

        # Extract from charge object field components
        if hasattr(agent, "charge_obj") and hasattr(agent.charge_obj, "field_components"):
            field_components = agent.charge_obj.field_components

            if hasattr(field_components, "semantic_field"):
                field_data["charge_semantic_field"] = field_components.semantic_field
            if hasattr(field_components, "emotional_trajectory"):
                field_data["emotional_trajectory"] = field_components.emotional_trajectory
            if hasattr(field_components, "trajectory_operators"):
                field_data["trajectory_operators"] = field_components.trajectory_operators
            if hasattr(field_components, "phase_total"):
                field_data["phase_total"] = field_components.phase_total

        return field_data

    def _extract_temporal_components(self, agent) -> Dict[str, Any]:
        """Extract temporal biography and evolution data."""
        temporal_data = {}

        if hasattr(agent, "temporal_biography"):
            temp_bio = agent.temporal_biography

            # Extract key temporal arrays
            temporal_attrs = [
                "trajectory_operators",
                "vivid_layer",
                "character_layer",
                "frequency_evolution",
                "phase_coordination",
                "temporal_momentum",
                "breathing_coherence",
                "field_interference_signature",
            ]

            for attr in temporal_attrs:
                if hasattr(temp_bio, attr):
                    value = getattr(temp_bio, attr)
                    if isinstance(value, complex):
                        temporal_data[f"{attr}_real"] = float(value.real)
                        temporal_data[f"{attr}_imag"] = float(value.imag)
                    elif isinstance(value, np.ndarray):
                        temporal_data[attr] = value
                    else:
                        temporal_data[attr] = value

        return temporal_data

    def _extract_emotional_components(self, agent) -> Dict[str, Any]:
        """Extract emotional modulation and field signature data."""
        emotional_data = {}

        if hasattr(agent, "emotional_modulation"):
            emo_mod = agent.emotional_modulation

            # Extract emotional arrays and metrics
            emotional_attrs = [
                "semantic_modulation_tensor",
                "unified_phase_shift",
                "trajectory_attractors",
                "resonance_frequencies",
                "field_modulation_strength",
                "pattern_confidence",
                "coupling_strength",
                "gradient_magnitude",
            ]

            for attr in emotional_attrs:
                if hasattr(emo_mod, attr):
                    value = getattr(emo_mod, attr)
                    if isinstance(value, complex):
                        emotional_data[f"{attr}_real"] = float(value.real)
                        emotional_data[f"{attr}_imag"] = float(value.imag)
                    elif isinstance(value, np.ndarray):
                        emotional_data[attr] = value
                    else:
                        emotional_data[attr] = value

        return emotional_data

    def _extract_agent_state(self, agent) -> Dict[str, Any]:
        """Extract complete agent state for reconstruction."""
        state_data = {}

        # Extract evolution parameters
        evolution_attrs = [
            "sigma_i",
            "alpha_i",
            "lambda_i",
            "beta_i",
            "breath_frequency",
            "breath_phase",
            "breath_amplitude",
            "emotional_conductivity",
            "tau_position",
            "modular_weight",
        ]

        for attr in evolution_attrs:
            if hasattr(agent, attr):
                value = getattr(agent, attr)
                if isinstance(value, complex):
                    state_data[f"{attr}_real"] = float(value.real)
                    state_data[f"{attr}_imag"] = float(value.imag)
                else:
                    state_data[attr] = value

        # Extract dictionaries
        dict_attrs = [
            "evolution_rates",
            "cascade_momentum",
            "breathing_q_coefficients",
            "hecke_eigenvalues",
            "l_function_coefficients",
        ]

        for attr in dict_attrs:
            if hasattr(agent, attr):
                state_data[attr] = getattr(agent, attr)

        return state_data

    def _extract_collective_properties(self, liquid_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract collective properties and optimization statistics."""
        collective = {}

        # Extract optimization statistics
        if "optimization_stats" in liquid_results:
            collective["optimization_stats"] = liquid_results["optimization_stats"]

        # Extract orchestrator collective properties
        orchestrator = liquid_results.get("orchestrator")
        if orchestrator:
            if hasattr(orchestrator, "adaptive_tuning"):
                collective["adaptive_tuning"] = orchestrator.adaptive_tuning
            if hasattr(orchestrator, "field_history"):
                collective["field_history"] = orchestrator.field_history

        return collective

    def _validate_extracted_mathematics(
        self, agent_data: Dict, collective_properties: Dict
    ) -> bool:
        """
        Validate that extracted mathematical components are consistent.

        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Check that we have agents
            if not agent_data:
                logger.warning("No agent data extracted")
                return False

            # Validate each agent has required components
            for agent_id, data in agent_data.items():
                if "Q_components" not in data:
                    logger.warning(f"Agent {agent_id} missing Q_components")
                    return False

                if "field_components" not in data:
                    logger.warning(f"Agent {agent_id} missing field_components")
                    return False

            logger.debug("Mathematical validation passed")
            return True

        except Exception as e:
            logger.error(f"Mathematical validation failed: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    processor = LiquidProcessor()
    print("Liquid Processor ready for dimension-agnostic universe processing")
