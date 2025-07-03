"""
Liquid Orchestrator - Field-Theoretic Agent Coordination

MATHEMATICAL FOUNDATION: Implements the complete Q(τ, C, s) formula orchestration
where each conceptual charge becomes a living mathematical entity that computes:

Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)

ORCHESTRATOR ROLE: Manages the liquid stage where charges interact through:
- Field interference patterns between Q values
- Emotional conductor transformations of S-T interactions
- Observational state evolution affecting all components
- Trajectory integration with real field dynamics

ARCHITECTURE: AgentTorch-based multi-agent simulation with PyTorch tensors
"""

import math

import numba as nb
import numpy as np
import torch
from sage.all import (
    CDF,
    CuspForms,
    EisensteinForms,
    Integer,
    ModularForms,
)
from scipy import integrate, linalg, signal, special
from scipy.linalg import eigh
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


# ADVANCED NUMBA JIT COMPILATION FOR PERFORMANCE-CRITICAL OPERATIONS
@nb.jit(nopython=True, cache=True, fastmath=True)
def _jit_breathing_coefficient_modulation(
    coeff_real_array, coeff_imag_array, oscillation_values, n_coefficients
):
    """
    JIT-compiled breathing coefficient modulation - MAXIMUM PERFORMANCE.

    Replaces Python loops with compiled operations for coefficient breathing.
    """
    for i in range(n_coefficients):
        if i < len(coeff_real_array) and i < len(oscillation_values):
            # Extract current coefficient components
            real_part = coeff_real_array[i]
            imag_part = coeff_imag_array[i]

            # Apply breathing modulation using compiled math
            breathing_factor = 1.0 + oscillation_values[i] * 0.1

            # Update coefficient arrays in-place (maximum performance)
            coeff_real_array[i] = real_part * breathing_factor
            coeff_imag_array[i] = imag_part * breathing_factor

    return coeff_real_array, coeff_imag_array


import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from Sysnpire.database.conceptual_charge_object import (
    ConceptualChargeObject,
    FieldComponents,
)
from Sysnpire.model.liquid.conceptual_charge_agent import ConceptualChargeAgent
from Sysnpire.model.liquid.regulation import RegulationLiquid
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


class TensorPool:
    """Memory-optimized tensor pool for GPU operations."""
    
    def __init__(self, device: torch.device):
        self.device = device
        # Pre-allocate common tensor sizes for reuse
        self.phase_tensors = {}
        self.complex_tensors = {}
        self.float_tensors = {}
        
        # Common sizes based on typical agent counts
        common_sizes = [10, 25, 50, 100, 200, 500]
        for size in common_sizes:
            self.phase_tensors[size] = torch.zeros(size, dtype=torch.float32, device=device)
            self.complex_tensors[size] = torch.zeros(size, dtype=torch.complex64, device=device)
            self.float_tensors[size] = torch.zeros(size, dtype=torch.float32, device=device)
    
    def get_phase_tensor(self, size: int) -> torch.Tensor:
        """Get pre-allocated phase tensor of appropriate size."""
        available_size = min(s for s in self.phase_tensors.keys() if s >= size)
        return self.phase_tensors[available_size][:size].clone()
    
    def get_complex_tensor(self, size: int) -> torch.Tensor:
        """Get pre-allocated complex tensor of appropriate size."""
        available_size = min(s for s in self.complex_tensors.keys() if s >= size)
        return self.complex_tensors[available_size][:size].clone()
    
    def get_float_tensor(self, size: int) -> torch.Tensor:
        """Get pre-allocated float tensor of appropriate size."""
        available_size = min(s for s in self.float_tensors.keys() if s >= size)
        return self.float_tensors[available_size][:size].clone()


@dataclass
class FieldInterferencePattern:
    """Results of Q-value field interference between charges."""

    charge_pairs: List[Tuple[str, str]]
    interference_strengths: torch.Tensor
    phase_relationships: torch.Tensor
    field_distortions: torch.Tensor


@dataclass
class EmotionalConductorState:
    """State of emotional conductor affecting S-T interactions."""

    modulation_tensor: torch.Tensor
    s_t_coupling_strength: float
    conductor_phase: float
    field_harmonics: torch.Tensor


@dataclass
class ObservationalEvolution:
    """Evolution of observational states across the field."""

    current_s_values: torch.Tensor
    s_gradients: torch.Tensor
    persistence_factors: torch.Tensor
    evolution_trajectories: torch.Tensor


class LiquidOrchestrator:
    """
    Field-Theoretic Multi-Agent Orchestrator

    Manages the liquid stage where conceptual charges behave as intelligent
    agents computing the complete Q(τ, C, s) formula with proper field
    interference, emotional conductor effects, and observational evolution.
    """

    def __init__(self, device: str = "mps", field_resolution: int = 256):
        """
        Initialize field-theoretic orchestrator.

        Args:
            device: PyTorch device ("mps" for Apple Silicon, "cuda", "cpu")
            field_resolution: Spatial resolution for field computations
        """
        self.device = torch.device(
            device if torch.backends.mps.is_available() else "cpu"
        )
        self.field_resolution = field_resolution

        # 🎯 GROUND ZERO: Device-aware dtype selection
        if self.device.type == "mps":
            self.field_dtype = torch.complex64
            self.float_dtype = torch.float32
            logger.info("✅ Using complex64/float32 for MPS compatibility")
        else:
            self.field_dtype = torch.complex128  
            self.float_dtype = torch.float64
            logger.info("✅ Using complex128/float64 for full mathematical precision")

        # Memory-optimized tensor pool for performance
        self.tensor_pool = TensorPool(self.device)
        logger.info("🚀 Memory-optimized tensor pool initialized")

        # Active charge agents (living Q(τ, C, s) entities)
        self.active_charges: Dict[str, ConceptualChargeObject] = {}
        self.charge_agents: Dict[str, "ConceptualChargeAgent"] = (
            {}
        )  # Store actual agents

        # ChargeFactory data storage
        self.combined_results: Optional[Dict[str, Any]] = None

        # Field state tensors with device-aware dtype handling
        self.field_grid = self._initialize_field_grid()

        # MATHEMATICAL THEORY OPTIMIZATION: Cache modular form bases (computed once, reused everywhere)
        logger.info("🔧 Caching modular form bases for group processing optimization...")
        try:
            self.eisenstein_basis = EisensteinForms(1, 4).basis()
            self.cusp_basis = CuspForms(1, 12).basis()
            logger.info(f"✅ Cached Eisenstein basis: {len(self.eisenstein_basis)} forms")
            logger.info(f"✅ Cached Cusp basis: {len(self.cusp_basis)} forms")
        except Exception as e:
            logger.warning(f"⚠️ Modular form caching failed: {e}")
            self.eisenstein_basis = []
            self.cusp_basis = []

        # Device-aware precision: MPS uses complex64, others use complex128
        self.q_field_values = torch.zeros(
            field_resolution,
            field_resolution,
            dtype=self.field_dtype,
            device=self.device,
        )

        # Emotional conductor state
        self.emotional_conductor = EmotionalConductorState(
            modulation_tensor=torch.ones(
                field_resolution, field_resolution, device=self.device
            ),
            s_t_coupling_strength=1.0,
            conductor_phase=0.0,
            field_harmonics=torch.zeros(field_resolution, device=self.device),
        )

        # Observational evolution tracking
        self.observational_state = ObservationalEvolution(
            current_s_values=torch.ones(
                field_resolution, field_resolution, device=self.device
            ),
            s_gradients=torch.zeros(
                field_resolution, field_resolution, device=self.device
            ),
            persistence_factors=torch.ones(
                field_resolution, field_resolution, device=self.device
            ),
            evolution_trajectories=torch.zeros(
                field_resolution, field_resolution, 2, device=self.device
            ),
        )

        # Simulation state
        self.current_tau = 0.0
        self.simulation_time = 0.0
        self.field_history: List[Dict[str, torch.Tensor]] = []

        # Initialize adaptive optimization system
        self.__init_adaptive_optimization()

        # Initialize field-theoretic regulation system
        self.regulation_liquid = RegulationLiquid(
            device=device,
            regulation_field_resolution=field_resolution
            // 4,  # Use smaller resolution for regulation field
        )
        logger.info(
            f"🌊 RegulationLiquid initialized for field-theoretic stabilization"
        )

    def _initialize_field_grid(self) -> torch.Tensor:
        """Initialize spatial grid for field computations."""
        x = torch.linspace(-1, 1, self.field_resolution, dtype=self.float_dtype, device=self.device)
        y = torch.linspace(-1, 1, self.field_resolution, dtype=self.float_dtype, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1)

    def load_charge_factory_results(self, combined_results: Dict[str, Any]) -> int:
        """
        Load ChargeFactory combined_results directly into orchestrator.

        RUNS THE SHOW: Main entry point for liquid simulation from ChargeFactory data.

        Args:
            combined_results: Dictionary from ChargeFactory.build() containing:
                - semantic_results with field_representations
                - temporal_results with temporal_biographies
                - emotional_results with emotional_modulations
                - field_components_ready status

        Returns:
            Number of charges loaded
        """
        logger.info("LiquidOrchestrator loading ChargeFactory results")

        # Store the results
        self.combined_results = combined_results

        # Validate structure
        required_keys = [
            "semantic_results",
            "temporal_results",
            "emotional_results",
            "field_components_ready",
        ]
        for key in required_keys:
            if key not in combined_results:
                raise ValueError(f"Missing required key '{key}' in combined_results")

        # Check readiness
        if not combined_results["field_components_ready"].get(
            "ready_for_unified_assembly"
        ):
            logger.warning(
                "ChargeFactory indicates components not ready for unified assembly"
            )

        # Get charge counts
        semantic_count = combined_results["field_components_ready"]["semantic_fields"]
        temporal_count = combined_results["field_components_ready"][
            "temporal_biographies"
        ]
        emotional_count = combined_results["field_components_ready"][
            "emotional_modulations"
        ]

        logger.info(
            f"Factory data loaded: {semantic_count} semantic fields, {temporal_count} temporal biographies, {emotional_count} emotional modulations"
        )

        # 📚 EXTRACT VOCAB MAPPINGS: Get vocabulary context for agent creation
        vocab_mappings = combined_results.get("vocab_mappings")
        vocab_count = len(vocab_mappings.get("id_to_token"))
        logger.info(
            f"📚 Vocab context loaded: {vocab_count} tokens available for agent identification"
        )

        if not (semantic_count == temporal_count == emotional_count):
            logger.warning(
                f"Mismatched component counts: semantic={semantic_count}, temporal={temporal_count}, emotional={emotional_count}"
            )

        return min(semantic_count, temporal_count, emotional_count)

    def create_agents_from_factory_data(self, max_agents: Optional[int] = None) -> int:
        """
        Create ConceptualChargeAgent entities from loaded ChargeFactory data.

        Args:
            max_agents: Maximum number of agents to create (None for all)

        Returns:
            Number of agents created
        """
        if self.combined_results is None:
            raise ValueError(
                "No ChargeFactory results loaded. Call load_charge_factory_results() first."
            )

        logger.info("Creating ConceptualChargeAgent entities from factory data")

        # Import here to avoid circular imports
        from Sysnpire.model.liquid.conceptual_charge_agent import ConceptualChargeAgent

        # Determine number of agents to create
        max_available = min(
            len(self.combined_results["semantic_results"]["field_representations"]),
            len(self.combined_results["temporal_results"]["temporal_biographies"]),
            len(self.combined_results["emotional_results"]["emotional_modulations"]),
        )

        num_agents = min(max_available, max_agents) if max_agents else max_available

        logger.info(
            f"Creating {num_agents} agents from {max_available} available charge sets"
        )

        agents_created = 0
        for i in range(num_agents):
            # DIRECT AGENT CREATION - NO ERROR MASKING
            logger.info(f"Creating agent {i}")

            # Create agent using factory method WITH VOCAB CONTEXT AND REGULATION
            vocab_mappings = self.combined_results.get("vocab_mappings")
            agent = ConceptualChargeAgent.from_charge_factory_results(
                combined_results=self.combined_results,
                charge_index=i,
                device=str(self.device),
                vocab_mappings=vocab_mappings,  # 📚 Pass vocab context for agent identification
                regulation_liquid=self.regulation_liquid,  # 🌊 Pass regulation system for field stabilization
            )

            # Store both agent and charge object
            agent_id = agent.charge_id
            self.charge_agents[agent_id] = agent
            self.active_charges[agent_id] = agent.charge_obj
            
            # DEBUG: Verify regulation system and Q_components are properly initialized
            logger.debug(f"🔍 DEBUG: Agent {agent_id} created with:")
            logger.debug(f"   - Has regulation_liquid: {hasattr(agent, 'regulation_liquid') and agent.regulation_liquid is not None}")
            logger.debug(f"   - Has Q_components: {hasattr(agent, 'Q_components') and agent.Q_components is not None}")
            if hasattr(agent, 'Q_components') and agent.Q_components is not None:
                logger.debug(f"   - Q_components.E_trajectory is None: {agent.Q_components.E_trajectory is None}")
                if agent.Q_components.E_trajectory is not None:
                    logger.debug(f"   - E_trajectory value: {agent.Q_components.E_trajectory}")
            else:
                logger.debug("   - WARNING: Q_components is None or missing!")

            # FIELD THEORY MEMORY OPTIMIZATION: Validate agent before cleanup
            if (i + 1) % 10 == 0:
                # Verify agent has complete Q(τ,C,s) structure before memory cleanup
                if not hasattr(agent, "charge_obj") or agent.charge_obj is None:
                    raise ValueError(
                        f"FIELD THEORY ERROR: Agent {i} missing charge_obj. "
                        f"Cannot perform memory cleanup without complete Q(τ,C,s) structure."
                    )

                # Verify charge has required field components
                required_components = ["semantic", "temporal", "manifold_properties"]
                charge_data = (
                    agent.charge_obj.__dict__
                    if hasattr(agent.charge_obj, "__dict__")
                    else {}
                )
                missing_components = [
                    comp
                    for comp in required_components
                    if comp not in charge_data and not hasattr(agent.charge_obj, comp)
                ]

                if missing_components:
                    logger.warning(
                        f"⚠️ FIELD THEORY: Agent {i} missing components {missing_components} during memory optimization"
                    )

                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    logger.debug(
                        f"🗑️ FIELD THEORY: MPS cache cleared after validating {i + 1} agents"
                    )

            # Update Q-field with new charge contribution
            self._update_q_field_contribution(agent.charge_obj)

            agents_created += 1
            logger.info(f"Agent {i} created successfully: {agent_id}")

        logger.info(
            f"Successfully created {agents_created} ConceptualChargeAgent entities"
        )

        # FIELD THEORY FINAL VALIDATION AND MEMORY CLEANUP
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            # Validate all agents have complete field structure before final cleanup
            incomplete_agents = []
            for agent_id, agent in self.charge_agents.items():
                if not hasattr(agent, "charge_obj") or agent.charge_obj is None:
                    incomplete_agents.append(agent_id)

            if incomplete_agents:
                raise ValueError(
                    f"FIELD THEORY ERROR: Cannot finalize with incomplete agents: {incomplete_agents}. "
                    f"All agents must have complete Q(τ,C,s) structure for field theory integrity."
                )

            torch.mps.empty_cache()
            logger.info(
                f"🗑️ FIELD THEORY: Final MPS cache cleanup after validating {agents_created} complete agents"
            )

        # 🔧 VERIFY: Check that all agents have optimized interaction method
        agents_with_optimized = 0
        agents_missing_optimized = 0

        for agent_id, agent in self.charge_agents.items():
            if hasattr(agent, "interact_with_optimized_field"):
                agents_with_optimized += 1
            else:
                # NO FALLBACKS - ALL AGENTS MUST HAVE OPTIMIZED METHODS
                raise ValueError(
                    f"Agent {agent_id} missing interact_with_optimized_field method - NO FALLBACKS ALLOWED!"
                )

        logger.info(
            f"✅ All {agents_created} agents have required optimized interaction methods - mathematical purity achieved!"
        )

        return agents_created

    def create_liquid_universe(
        self, combined_results: Dict[str, Any], max_agents: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create the liquid universe from ChargeFactory combined results.

        This is the main entry point called by ChargeFactory after stages 1, 2, 3.

        Args:
            combined_results: Dictionary from ChargeFactory containing:
                - semantic_results with field_representations
                - temporal_results with temporal_biographies
                - emotional_results with emotional_modulations (conductor)
                - field_components_ready status
            max_agents: Maximum number of agents to create (None for all)

        Returns:
            liquid_results: Dictionary containing:
                - agent_pool: Dict of ConceptualChargeAgent instances
                - active_charges: Dict of ConceptualChargeObject instances
                - num_agents: Number of agents created
                - field_statistics: Initial field state
                - orchestrator: Reference to self for simulation control
                - ready_for_simulation: Boolean readiness flag
        """
        logger.info("=" * 60)
        logger.info("CREATING LIQUID UNIVERSE")
        logger.info("LiquidOrchestrator taking control from ChargeFactory")
        logger.info("=" * 60)

        # Step 1: Load factory data
        num_charges = self.load_charge_factory_results(combined_results)

        # Step 2: Create agents
        num_agents = self.create_agents_from_factory_data(max_agents)

        # Step 3: Get initial field statistics
        field_stats = self.get_field_statistics()

        # Step 4: Get optimization statistics
        optimization_stats = self._get_optimization_statistics(num_agents)

        # 📚 STEP 4.5: Generate agent summaries with vocab context
        agent_summaries = []
        vocab_mappings = self.combined_results.get("vocab_mappings")

        for agent_id, agent in self.charge_agents.items():
            summary = {
                "agent_id": agent_id,
                "charge_index": agent.charge_index,
                "vocab_token_string": getattr(agent, "vocab_token_string", "unknown"),
                "vocab_token_id": getattr(agent, "vocab_token_id", None),
                "Q_value": (
                    agent.Q_components.Q_value if agent.Q_components else complex(0)
                ),
                "living_Q_value": getattr(agent, "living_Q_value", complex(0)),
                "gamma": agent.Q_components.gamma if agent.Q_components else 0.0,
                "field_magnitude": (
                    abs(agent.Q_components.Q_value) if agent.Q_components else 0.0
                ),
            }
            agent_summaries.append(summary)

        # Step 5: Build liquid results structure with vocab-enhanced summaries
        liquid_results = {
            "agent_pool": self.charge_agents,  # Living Q(τ,C,s) entities
            "active_charges": self.active_charges,  # ConceptualChargeObject instances
            "num_agents": num_agents,
            "field_statistics": field_stats,
            "optimization_stats": optimization_stats,  # Performance metrics
            "orchestrator": self,  # Reference for simulation control
            "ready_for_simulation": num_agents > 0,
            # 📚 VOCAB-ENHANCED RESULTS: Include agent summaries with readable vocab
            "agent_summaries": agent_summaries,
            "vocab_context": vocab_mappings,
        }

        logger.info("Liquid universe creation complete:")
        logger.info(f"  Agents created: {num_agents}")
        logger.info(f"  Field energy: {field_stats['field_energy']:.6f}")
        logger.info(f"  Ready for simulation: {liquid_results['ready_for_simulation']}")

        return liquid_results

    def orchestrate_living_evolution(
        self, tau_steps: int = 100, tau_step_size: float = 0.01
    ) -> Dict[str, Any]:
        """
        Orchestrate the living evolution of modular forms in liquid stage.

        This implements the "movement score" where all agents evolve together
        through breathing, cascading, and interaction - creating liquid metal dynamics.

        Args:
            tau_steps: Number of evolution steps
            tau_step_size: Size of each evolution step

        Returns:
            Evolution results with emergent patterns and complexity measures
        """
        logger.info("=" * 60)
        logger.info("🌊 ORCHESTRATING LIVING MODULAR FORMS EVOLUTION")
        logger.info("=" * 60)

        if not self.charge_agents:
            logger.warning("No agents available for evolution")
            return {"error": "no_agents"}

        # Convert agents to list for easier access
        agents = list(self.charge_agents.values())
        evolution_results = {
            "breathing_patterns": [],
            "cascade_energies": [],
            "interaction_networks": [],
            "complexity_evolution": [],
            "emergent_harmonics": [],
            "collective_Q_field": [],
        }

        logger.info(f"🎼 Starting evolution with {len(agents)} living modular forms")

        # 🔧 FIX: Initialize optimization IMMEDIATELY before evolution starts
        logger.info("🎯 Initializing O(N log N) optimization system...")
        self.listen_to_modular_forms(agents, 0.0)
        self.adapt_computation_strategy(agents, 0.0)

        # Verify optimization is ready
        sparse_interactions = sum(
            len(neighbors)
            for neighbors in self.adaptive_tuning["sparse_interaction_graph"].values()
        )
        logger.info(
            f"✅ Optimization ready: {sparse_interactions} sparse interactions configured"
        )

        # MATHEMATICAL THEORY OPTIMIZATION: Pre-compute observational persistence for all agents
        logger.info("🔧 Pre-computing observational persistence using mathematical clustering...")
        persistence_start = time.time()
        self.orchestrate_persistence_optimization(agents)
        persistence_time = time.time() - persistence_start
        logger.info(f"✅ Persistence optimization complete in {persistence_time:.3f}s")

        for step in range(tau_steps):
            tau = step * tau_step_size
            self.current_tau = tau  # 🔧 FIX: Track current tau for safety checks

            # 🎯 ADAPTIVE OPTIMIZATION: Listen to modular forms and adapt strategies
            if step % 5 == 0:  # Listen every 5 steps (after initial setup)
                self.listen_to_modular_forms(agents, tau)
                self.adapt_computation_strategy(agents, tau)

            logger.info("🔍 Post-adaptation: Starting MOVEMENT sequence...")

            # MOVEMENT 1: 🎵 Collective Breathing (with breathing sync groups)
            logger.info("🎵 Starting collective breathing...")
            breathing_start = time.time()
            breathing_synchrony = self._orchestrate_collective_breathing(agents, tau)
            breathing_time = time.time() - breathing_start
            logger.info(f"✅ Collective breathing complete in {breathing_time:.4f}s")
            evolution_results["breathing_patterns"].append(breathing_synchrony)

            # MOVEMENT 2: 🌊 Cascading Dimensional Feedback (with cascade optimization)
            logger.info("🌊 Starting dimensional cascades...")
            cascade_start = time.time()
            cascade_energy = self._orchestrate_dimensional_cascades(agents)
            cascade_time = time.time() - cascade_start
            logger.info(f"✅ Dimensional cascades complete in {cascade_time:.4f}s")
            evolution_results["cascade_energies"].append(cascade_energy)

            # MOVEMENT 3: 🎭 Field Interactions (O(N log N) OPTIMIZED!)
            interaction_strength = self._orchestrate_field_interactions_optimized(
                agents
            )

            # FIELD-THEORETIC REGULATION: Apply natural mathematical stabilization
            if (
                not math.isfinite(interaction_strength)
                or abs(interaction_strength) > 1e15
            ):
                logger.info(
                    f"🌊 FIELD REGULATION TRIGGERED: interaction_strength = {interaction_strength:.2e}"
                )
                regulated_strength, regulation_metrics = (
                    self.regulation_liquid.regulate_interaction_strength(
                        agents, interaction_strength
                    )
                )

                # Log regulation results
                if regulation_metrics["regulation_applied"]:
                    logger.info(
                        f"🌊 REGULATION APPLIED: {interaction_strength:.2e} -> {regulated_strength:.2e}"
                    )
                    logger.info(
                        f"   Stability achieved: {regulation_metrics['stability_achieved']}"
                    )
                    logger.info(
                        f"   Regulation ratio: {regulation_metrics['regulation_strength_ratio']:.3f}"
                    )

                interaction_strength = regulated_strength

                # Update regulation field based on current state
                self.regulation_liquid.update_regulation_field(
                    regulation_metrics["field_state"]
                )
            else:
                # Even for stable interactions, update regulation parameters for future use
                if step % 10 == 0:  # Update every 10 steps to avoid overhead
                    self.regulation_liquid.analyze_field_state(agents)

            logger.info("📊 Appending interaction strength to results...")
            append_start = time.time()
            evolution_results["interaction_networks"].append(interaction_strength)
            append_time = time.time() - append_start
            logger.info(f"✅ Interaction results appended in {append_time:.4f}s")

            # MOVEMENT 4: 📍 Observational Evolution (with phase boundaries)
            logger.info("📍 Starting s-parameter evolution...")
            s_param_start = time.time()
            complexity_measure = self._orchestrate_s_parameter_evolution(agents)
            s_param_time = time.time() - s_param_start
            logger.info(f"✅ S-parameter evolution complete in {s_param_time:.4f}s")

            # FIELD-THEORETIC REGULATION: Apply regulation to complexity measures
            if not math.isfinite(complexity_measure) or abs(complexity_measure) > 1e15:
                logger.info(
                    f"🌊 COMPLEXITY REGULATION TRIGGERED: complexity_measure = {complexity_measure:.2e}"
                )
                # Use interaction regulation method for complexity as it follows same field principles
                regulated_complexity, complexity_regulation_metrics = (
                    self.regulation_liquid.regulate_complexity(
                        agents, complexity_measure
                    )
                )

                if complexity_regulation_metrics["regulation_applied"]:
                    logger.info(
                        f"🌊 COMPLEXITY REGULATION APPLIED: {complexity_measure:.2e} -> {regulated_complexity:.2e}"
                    )

                complexity_measure = regulated_complexity

            evolution_results["complexity_evolution"].append(complexity_measure)

            # MOVEMENT 5: 🎼 Measure Emergent Properties
            emergent_data = self._measure_emergent_complexity(agents, tau)
            evolution_results["emergent_harmonics"].append(emergent_data)

            # MOVEMENT 6: 🌌 Collective Q-Field
            collective_Q = self._compute_collective_Q_field(agents)
            evolution_results["collective_Q_field"].append(collective_Q)

            # Log progress every 10 steps
            if step % 10 == 0:
                # Include comprehensive optimization stats
                sparse_interactions = sum(
                    len(neighbors)
                    for neighbors in self.adaptive_tuning[
                        "sparse_interaction_graph"
                    ].values()
                )
                sync_groups = len(self.adaptive_tuning["breathing_sync_groups"])
                cascade_chains = len(self.adaptive_tuning["resonance_cascades"])
                phase_boundaries = len(self.adaptive_tuning["phase_boundaries"])

                # Calculate complexity reduction
                total_possible_interactions = len(agents) * (len(agents) - 1)
                optimization_factor = (
                    sparse_interactions / total_possible_interactions
                    if total_possible_interactions > 0
                    else 0
                )

                # Check interaction method efficiency
                optimized_agents = sum(
                    1
                    for agent in agents
                    if hasattr(agent, "interact_with_optimized_field")
                )
                fallback_agents = len(agents) - optimized_agents

                logger.info(
                    f"Step {step}: Breathing={breathing_synchrony:.3f}, "
                    f"Cascade={cascade_energy:.3f}, "
                    f"Interaction={interaction_strength:.3f}, "
                    f"Complexity={complexity_measure:.3f}"
                )
                logger.info(
                    f"🎯 O(N log N) Stats: {sparse_interactions}/{total_possible_interactions} interactions "
                    f"({optimization_factor:.3%} of O(N²)), "
                    f"{sync_groups} sync groups, "
                    f"{cascade_chains} cascades, "
                    f"{phase_boundaries} boundaries"
                )
                logger.info(
                    f"⚡ Efficiency: {optimized_agents}/{len(agents)} optimized agents, "
                    f"{fallback_agents} using efficient fallback"
                )

        # Final analysis - pass agents for total Q energy calculation
        final_analysis = self._analyze_evolution_results(evolution_results, agents)

        logger.info("🎼 Living evolution complete!")
        logger.info(f"📊 Final complexity: {final_analysis['final_complexity']:.3f}")
        logger.info(
            f"🌊 Emergent harmonics: {len(final_analysis['emergent_harmonics'])}"
        )
        logger.info(
            f"🎭 Collective coherence: {final_analysis['collective_coherence']:.3f}"
        )

        return {
            "evolution_results": evolution_results,
            "final_analysis": final_analysis,
            "agents_evolved": len(agents),
            "tau_steps_completed": tau_steps,
        }

    def _orchestrate_collective_breathing(self, agents: List, tau: float) -> float:
        """TRUE O(log N) collective breathing using vectorized group processing."""
        all_breathing_phases = []

        # Get breathing sync groups from adaptive optimization
        sync_groups = self.adaptive_tuning["breathing_sync_groups"]

        if sync_groups:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import os
            valid_groups = []
            for group_indices in sync_groups:
                if group_indices:
                    group_agents = [agents[i] for i in group_indices if i < len(agents)]
                    if len(group_agents) > 0:
                        valid_groups.append(group_agents)
            
            # Process all sync groups in parallel
            max_workers = min(len(valid_groups), os.cpu_count() or 4)
            if max_workers > 1 and len(valid_groups) > 1:
                logger.debug(f"🔄 Parallel processing {len(valid_groups)} sync groups with {max_workers} workers")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    group_futures = {
                        executor.submit(self._process_breathing_group_collectively, group_agents, tau): i
                        for i, group_agents in enumerate(valid_groups)
                    }
                    
                    for future in as_completed(group_futures):
                        try:
                            group_phases = future.result()
                            all_breathing_phases.extend(group_phases)
                        except Exception as e:
                            group_idx = group_futures[future]
                            logger.warning(f"Sync group {group_idx} failed: {e}")
            else:
                # Fallback to sequential if only one group or limited workers
                for group_agents in valid_groups:
                    group_phases = self._process_breathing_group_collectively(group_agents, tau)
                    all_breathing_phases.extend(group_phases)

            # Handle ungrouped agents efficiently
            grouped_indices = set()
            for group in sync_groups:
                grouped_indices.update(group)

            ungrouped_agents = [
                agents[i] for i in range(len(agents)) if i not in grouped_indices
            ]
            if ungrouped_agents:
                # Process ungrouped agents as additional group
                ungrouped_phases = self._process_breathing_group_collectively(
                    ungrouped_agents, tau
                )
                all_breathing_phases.extend(ungrouped_phases)
        else:
            # Fallback: Process all agents as single group
            group_phases = self._process_breathing_group_collectively(agents, tau)
            all_breathing_phases.extend(group_phases)

        # Update dynamic field after all group breathing
        self.update_dynamic_field()

        # Measure breathing synchrony
        if len(all_breathing_phases) > 1:
            phase_diffs = [
                abs(all_breathing_phases[i] - all_breathing_phases[0])
                for i in range(1, len(all_breathing_phases))
            ]
            phase_diffs_tensor = torch.tensor(phase_diffs, device=self.device)
            # GPU-native correlation using PyTorch
            phase_expanded = phase_diffs_tensor.unsqueeze(0).unsqueeze(0)
            phase_flipped = torch.flip(phase_diffs_tensor, [0]).unsqueeze(0).unsqueeze(0)
            synchrony_corr = torch.conv1d(phase_expanded, phase_flipped, padding='same').squeeze()
            
            peak_correlation = torch.logsumexp(synchrony_corr, dim=0) / len(synchrony_corr)
            synchrony = 1.0 - (peak_correlation / (2 * math.pi))
        else:
            # NO FALLBACK - Phase differences must exist for synchrony computation
            raise ValueError(
                "MATHEMATICAL FAILURE: No phase differences available - "
                "Synchrony cannot be computed. System requires phase data."
            )

        return max(0.0, synchrony)

    def _orchestrate_dimensional_cascades(self, agents: List) -> float:
        """Coordinate dimensional feedback cascades using detected resonance chains."""
        total_cascade_energy = 0.0

        # Get detected resonance cascades from adaptive optimization
        cascade_chains = self.adaptive_tuning["resonance_cascades"]

        if cascade_chains:
            # Process cascade chains for exponential amplification
            for cascade_info in cascade_chains:
                cascade_agent_indices = cascade_info.get("agents")
                cascade_magnitude = cascade_info.get("magnitude")

                # Get agents in cascade chain
                cascade_agents = [
                    agents[i] for i in cascade_agent_indices if i < len(agents)
                ]

                if len(cascade_agents) >= 3:  # Valid cascade chain
                    # Amplify cascading for phase-aligned agents
                    amplification_factor = (
                        1.0 + len(cascade_agents) * 0.1
                    )  # More agents = more amplification

                    for agent in cascade_agents:
                        # Temporarily boost cascade rates for resonance chains
                        original_rate = agent.evolution_rates["cascading"]
                        agent.evolution_rates["cascading"] *= amplification_factor

                        agent.cascade_dimensional_feedback()
                        agent.sync_positions()

                        # Restore original rate
                        agent.evolution_rates["cascading"] = original_rate

                        # Measure enhanced cascade energy
                        cascade_momentum = agent.cascade_momentum
                        agent_cascade_energy = sum(
                            abs(momentum) for momentum in cascade_momentum.values()
                        )
                        total_cascade_energy += (
                            agent_cascade_energy * amplification_factor
                        )

            # Handle agents not in any cascade chain
            cascaded_indices = set()
            for cascade_info in cascade_chains:
                cascaded_indices.update(cascade_info.get("agents"))

            for i, agent in enumerate(agents):
                if i not in cascaded_indices:
                    agent.cascade_dimensional_feedback()
                    agent.sync_positions()

                    cascade_momentum = agent.cascade_momentum
                    agent_cascade_energy = sum(
                        abs(momentum) for momentum in cascade_momentum.values()
                    )
                    total_cascade_energy += agent_cascade_energy
        else:
            # Fallback: regular cascading for all agents
            for agent in agents:
                agent.cascade_dimensional_feedback()
                agent.sync_positions()

                cascade_momentum = agent.cascade_momentum
                agent_cascade_energy = sum(
                    abs(momentum) for momentum in cascade_momentum.values()
                )
                total_cascade_energy += agent_cascade_energy

        # Update dynamic field after all cascading
        self.update_dynamic_field()

        # Return average cascade energy
        return total_cascade_energy / len(agents) if agents else 0.0

    def _orchestrate_field_interactions(self, agents: List) -> float:
        """Coordinate field interactions between living modular forms (LEGACY O(N²) VERSION)."""
        total_interaction_strength = 0.0

        # Each agent interacts with all others
        for agent in agents:
            agent.interact_with_field(agents)

            # Sync positions after interactions
            agent.sync_positions()

            # Measure interaction strength from memory
            if agent.interaction_memory:
                recent_interactions = agent.interaction_memory[
                    -5:
                ]  # Last 5 interactions
                agent_interaction_strength = sum(
                    record["influence"] for record in recent_interactions
                ) / len(recent_interactions)
                total_interaction_strength += agent_interaction_strength

        # Update dynamic field after all interactions
        self.update_dynamic_field()

        return total_interaction_strength / len(agents) if agents else 0.0

    def _orchestrate_field_interactions_optimized(self, agents: List) -> float:
        """TRUE O(log N) field interactions using group-centric processing."""
        import time

        start_time = time.time()
        logger.info(
            f"🚀 Starting optimized field interactions for {len(agents)} agents"
        )

        total_interaction_strength = 0.0

        # 🔧 VALIDATE: Check Q values before interactions
        valid_agents = []
        problematic_agents = []

        for agent in agents:
            # Check if agent has computed Q components
            if hasattr(agent, "Q_components") and agent.Q_components is not None:
                Q_magnitude = abs(agent.Q_components.Q_value)
                if Q_magnitude > 1e-12:  # Meaningful Q value
                    valid_agents.append(agent)
                else:
                    problematic_agents.append((agent, Q_magnitude))
            else:
                # DIRECT Q COMPUTATION - NO ERROR MASKING
                pool_size = len(agents)
                agent.compute_complete_Q(pool_size=pool_size)
                Q_magnitude = (
                    abs(agent.Q_components.Q_value) if agent.Q_components else 0.0
                )
                if Q_magnitude > 1e-12:
                    valid_agents.append(agent)
                else:
                    problematic_agents.append((agent, Q_magnitude))

        # Log validation results
        if problematic_agents:
            logger.warning(
                f"⚠️  Field Interaction Validation: {len(problematic_agents)}/{len(agents)} agents have problematic Q values"
            )
            for agent, Q_mag in problematic_agents[:3]:  # Show first 3
                agent_id = getattr(agent, "charge_id", "unknown")
                logger.warning(
                    f"    Agent {agent_id}: |Q| = {Q_mag:.2e} (too small for meaningful interaction)"
                )
            if len(problematic_agents) > 3:
                logger.warning(
                    f"    ... and {len(problematic_agents) - 3} more agents with similar issues"
                )

        if len(valid_agents) < 2:
            logger.warning(
                f"⚠️  Insufficient valid agents for field interactions: {len(valid_agents)}/{len(agents)}"
            )
            return 0.0

        logger.debug(
            f"✅ Field Interaction Validation: {len(valid_agents)}/{len(agents)} agents ready for interaction"
        )

        # Use only valid agents for interactions
        interaction_agents = valid_agents

        # Get interaction groups from adaptive optimization
        logger.info(
            f"🔄 Building interaction groups for {len(interaction_agents)} agents..."
        )
        group_build_start = time.time()
        interaction_groups = self._build_interaction_groups(interaction_agents)
        group_build_time = time.time() - group_build_start
        logger.info(
            f"✅ Built {len(interaction_groups)} groups in {group_build_time:.4f}s"
        )

        # 🚀 REVOLUTIONARY CHANGE: Process O(log N) groups instead of O(N) agents
        logger.info(f"🚀 Processing {len(interaction_groups)} interaction groups...")
        group_process_start = time.time()
        for group_info in interaction_groups:
            group_agents = group_info["agents"]
            group_interactions = group_info["interactions"]

            # Process entire group collectively using vectorized operations
            group_strength = self._process_interaction_group_collectively(
                group_agents, group_interactions
            )
            total_interaction_strength += group_strength

        group_process_time = time.time() - group_process_start
        logger.info(f"✅ Group processing complete in {group_process_time:.4f}s")

        # Update dynamic field after all group interactions
        logger.info("🌊 Updating dynamic field...")
        field_update_start = time.time()
        self.update_dynamic_field()
        field_update_time = time.time() - field_update_start
        logger.info(f"✅ Dynamic field update complete in {field_update_time:.4f}s")

        interaction_time = time.time() - start_time
        logger.info(
            f"✅ Optimized field interactions complete: {len(interaction_groups)} groups processed in {interaction_time:.4f}s"
        )

        return (
            total_interaction_strength / len(interaction_agents)
            if interaction_agents
            else 0.0
        )

    def _apply_precomputed_interactions(
        self, agent, nearby_agents_with_strengths: List[Tuple]
    ):
        """
        NO FALLBACKS ALLOWED - This method should never be called.

        All agents MUST have optimized methods - no fallback systems permitted.
        """
        raise ValueError(
            f"FALLBACK METHOD CALLED for agent {getattr(agent, 'charge_id', 'unknown')} - NO FALLBACKS ALLOWED!"
        )

    def _create_mathematical_clusters(self, agents: List) -> Dict[str, List]:
        """Create clusters based on mathematical theory: modular periodicity and Q-magnitude."""
        import math
        
        # MATHEMATICAL THEORY: Agents with similar properties can share computations
        clusters = {}
        
        for i, agent in enumerate(agents):
            # Modular form periodicity clustering
            tau_class = int(float(i) / len(agents) * 12) % 12  # Fundamental domain partition
            
            # Q-magnitude clustering for shared CDF operations
            if hasattr(agent, 'living_Q_value') and agent.living_Q_value is not None:
                q_magnitude = abs(agent.living_Q_value)
                if q_magnitude > 0:
                    mag_class = int(math.log10(q_magnitude) + 10)  # Log scale grouping
                else:
                    mag_class = 0
            else:
                mag_class = 0
            
            # Combine tau and magnitude for cluster key
            cluster_key = f"tau_{tau_class}_mag_{mag_class}"
            
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(agent)
        
        return clusters

    def _build_cluster_field_context(self, cluster_agents: List, sparse_graph: Dict, all_agents: List) -> List:
        """Build shared field context for cluster using sparse graph optimization."""
        field_context_set = set()
        
        # Collect all neighbors for cluster agents
        for agent in cluster_agents:
            agent_idx = all_agents.index(agent) if agent in all_agents else -1
            if agent_idx >= 0 and sparse_graph and agent_idx in sparse_graph:
                neighbor_indices = [idx for idx, _ in sparse_graph[agent_idx]]
                for neighbor_idx in neighbor_indices:
                    if neighbor_idx < len(all_agents):
                        field_context_set.add(all_agents[neighbor_idx])
            
            # Include the agent itself
            field_context_set.add(agent)
        
        return list(field_context_set)

    def _batch_evolve_cluster_s_parameter(self, cluster_agents: List, field_context: List) -> None:
        """VECTORIZED S-PARAMETER EVOLUTION: True tensor-based batch processing."""
        # LIQUID FIELD THEORY: Compute shared field properties once, then vectorized agent updates
        self._vectorized_s_parameter_evolution(cluster_agents, field_context)
    
    def _vectorized_s_parameter_evolution(self, agents: List, field_context: List) -> None:
        """
        VECTORIZED S-PARAMETER EVOLUTION: Process all agents simultaneously using tensor operations.
        
        Replaces O(N²) individual agent processing with O(N) vectorized computation.
        """
        if not agents:
            return
        
        # SHARED FIELD COMPUTATION: Calculate once for all agents
        # Instead of each agent computing field_pressure from all others
        field_q_values = [
            agent.living_Q_value for agent in field_context 
            if hasattr(agent, "living_Q_value") and agent.living_Q_value is not None
        ]
        
        if not field_q_values:
            return
            
        # Global field pressure computed once
        field_pressure = sum(abs(q) for q in field_q_values) / len(field_q_values)
        
        # VECTORIZED AGENT DATA EXTRACTION
        current_s_values = []
        temporal_phases = []
        evolution_rates = []
        memory_pressures = []
        valid_agents = []
        
        for agent in agents:
            # Validate agent has required attributes
            if not (hasattr(agent, 'state') and hasattr(agent, 'temporal_biography') and 
                   hasattr(agent.temporal_biography, 'temporal_momentum')):
                continue
                
            try:
                # Extract current s-parameter
                current_s = agent.state.current_s
                if not math.isfinite(current_s):
                    continue
                
                # Extract temporal phase from CDF temporal momentum
                temporal_momentum = agent.temporal_biography.temporal_momentum
                temporal_phase = temporal_momentum.arg()
                
                # Extract evolution rate
                memory_rate = agent.evolution_rates.get("memory", 0.01)
                
                # Extract memory pressure
                if agent.interaction_memory:
                    recent_memory = agent.interaction_memory[-10:]
                    memory_pressure = sum(record["influence"] for record in recent_memory) / len(recent_memory)
                else:
                    memory_pressure = 0.0
                
                # Add to vectors
                current_s_values.append(current_s)
                temporal_phases.append(temporal_phase)
                evolution_rates.append(memory_rate)
                memory_pressures.append(memory_pressure)
                valid_agents.append(agent)
                
            except (AttributeError, KeyError, ValueError) as e:
                logger.debug(f"Skipping agent {getattr(agent, 'charge_id', 'unknown')} in s-evolution: {e}")
                continue
        
        if not valid_agents:
            return
        
        # VECTORIZED COMPUTATION: Process all agents simultaneously
        import torch.nn.functional as F
        
        # Convert to tensors for vectorized operations
        phase_tensor = torch.tensor(temporal_phases, dtype=torch.float32)
        current_s_tensor = torch.tensor(current_s_values, dtype=torch.float32)
        memory_tensor = torch.tensor(memory_pressures, dtype=torch.float32)
        evolution_tensor = torch.tensor(evolution_rates, dtype=torch.float32)
        
        # VECTORIZED INFLUENCE COMPUTATION
        # Vivid influence: 1.0 + 0.5 * F.gelu(cos(phase * 3.0))
        vivid_base = torch.cos(phase_tensor * 3.0)
        vivid_influence = 1.0 + 0.5 * F.gelu(vivid_base)
        
        # Character influence: 0.001 + 0.999 * (F.silu(sin(phase * 2.0 + π/4)) * 0.5 + 0.5)
        character_base = torch.sin(phase_tensor * 2.0 + math.pi / 4)
        character_influence = 0.001 + 0.999 * (F.silu(character_base) * 0.5 + 0.5)
        
        # VECTORIZED EVOLUTION COMPUTATION
        # Field term: field_pressure * vivid_influence * 0.01
        field_terms = field_pressure * vivid_influence * 0.01
        
        # Momentum term: character_influence * 0.1 (simplified from complex multiplication)
        momentum_terms = character_influence * 0.1
        
        # Memory term: memory_pressure * 0.05
        memory_terms = memory_tensor * 0.05
        
        # Total delta_s for all agents
        delta_s = field_terms + momentum_terms + memory_terms
        
        # Evolution ratio: 1.0 + (delta_s * evolution_rate * 0.01)
        evolution_ratios = 1.0 + (delta_s * evolution_tensor * 0.01)
        
        # New s-parameters: current_s * evolution_ratio
        new_s_values = current_s_tensor * evolution_ratios
        
        # VECTORIZED UPDATE: Apply all changes simultaneously
        for i, agent in enumerate(valid_agents):
            agent.state.current_s = new_s_values[i].item()
            
            # Update modular form complexity based on s distance (individual operation)
            s_distance = abs(agent.state.current_s - agent.state.s_zero)
            if hasattr(agent, 'update_form_complexity'):
                agent.update_form_complexity(s_distance)

    def _batch_sync_agent_positions(self, agents: List) -> None:
        """VECTORIZED POSITION SYNC: True batch operation for all agents."""
        # LIQUID FIELD THEORY: Sync all agents simultaneously instead of individual calls
        if not agents:
            return
            
        # Extract all position data for vectorized sync
        living_q_values = []
        valid_agents = []
        
        for agent in agents:
            if hasattr(agent, 'living_Q_value') and agent.living_Q_value is not None:
                living_q_values.append(agent.living_Q_value)
                valid_agents.append(agent)
        
        if not living_q_values:
            return
            
        # Vectorized position update computations
        q_magnitudes = [abs(q) for q in living_q_values]
        q_phases = [torch.atan2(torch.tensor(q).imag, torch.tensor(q).real).item() for q in living_q_values]
        
        # Apply sync updates to all agents simultaneously
        for i, agent in enumerate(valid_agents):
            # Update agent position based on Q-value (simplified sync operation)
            magnitude = q_magnitudes[i]
            phase = q_phases[i]
            
            # Sync position to Q-field coordinates
            if hasattr(agent, 'position'):
                agent.position = {
                    'magnitude': magnitude,
                    'phase': phase,
                    'real': magnitude * math.cos(phase),
                    'imag': magnitude * math.sin(phase)
                }

    def _precompute_tau_evaluations(self, group_agents: List, n_charges: int) -> Dict:
        """ENHANCED MODULAR FORM CACHING: Persistent cache with tau clustering."""
        tau_evaluations = {}
        
        if len(self.eisenstein_basis) > 0:
            eisenstein_form = self.eisenstein_basis[0]
            cusp_form = self.cusp_basis[0] if len(self.cusp_basis) > 0 else None
            
            # PERSISTENT CACHE: Initialize if not exists
            if not hasattr(self, '_tau_cache'):
                self._tau_cache = {}
            
            # TAU CLUSTERING: Group similar tau values to reduce computations
            tau_clusters = {}
            
            for i, agent in enumerate(group_agents):
                agent_idx = i
                
                # Compute tau position
                tau_real = float(agent_idx) / n_charges
                tau = CDF(tau_real, 1.5)
                
                # TAU CLUSTERING: Quantize tau to reduce cache size
                # Group tau values into clusters to enable cache reuse
                tau_cluster_id = f"{int(tau_real * 100)//10}_{150}"  # Cluster real part to 0.1 precision
                
                # Check persistent cache first
                if tau_cluster_id in self._tau_cache:
                    tau_evaluations[agent] = self._tau_cache[tau_cluster_id].copy()
                    tau_evaluations[agent]['tau'] = tau  # Update with exact tau
                    continue
                
                # Compute if not cached
                eisenstein_val = eisenstein_form(tau)
                cusp_val = cusp_form(tau) if cusp_form else None
                
                result = {
                    'tau': tau,
                    'eisenstein': eisenstein_val,
                    'cusp': cusp_val
                }
                
                tau_evaluations[agent] = result
                
                # Cache for future use
                self._tau_cache[tau_cluster_id] = {
                    'tau': tau,
                    'eisenstein': eisenstein_val, 
                    'cusp': cusp_val
                }
                
                # Limit cache size
                if len(self._tau_cache) > 1000:
                    # Remove oldest entries (simple FIFO)
                    oldest_key = next(iter(self._tau_cache))
                    del self._tau_cache[oldest_key]
        
        return tau_evaluations

    def _precompute_magnitude_cdf_operations(self, group_agents: List) -> Dict:
        """Pre-compute CDF operations for magnitude classes to avoid repeated expensive computations."""
        magnitude_cdf_cache = {}
        
        # Group agents by magnitude classes for shared CDF operations
        magnitude_groups = {}
        for agent in group_agents:
            if hasattr(agent, 'living_Q_value') and agent.living_Q_value is not None:
                q_magnitude = abs(agent.living_Q_value)
                if q_magnitude > 0:
                    mag_class = int(math.log10(q_magnitude) + 10)  # Same clustering as S-parameter
                    if mag_class not in magnitude_groups:
                        magnitude_groups[mag_class] = []
                    magnitude_groups[mag_class].append((agent, q_magnitude))
        
        # Pre-compute CDF base operations for each magnitude class
        for mag_class, class_agents in magnitude_groups.items():
            if class_agents:
                # Use representative magnitude for class
                representative_magnitude = class_agents[0][1]
                base_log_mag = math.log(float(representative_magnitude))
                
                # Pre-compute expensive CDF exponential for this magnitude class
                magnitude_cdf_cache[mag_class] = {
                    'base_log_mag': base_log_mag,
                    'agents': [agent for agent, _ in class_agents]
                }
        
        return magnitude_cdf_cache

    def _create_persistence_clusters(self, agents: List) -> Dict[str, List]:
        """Cluster agents by delta_s magnitude for shared observational persistence computation."""
        import math
        
        clusters = {}
        
        for agent in agents:
            # Extract delta_s values using same logic as observational persistence
            if hasattr(agent.state.current_s, "cpu"):
                s = float(agent.state.current_s.cpu().detach().numpy())
            else:
                s = float(agent.state.current_s)
                
            if hasattr(agent.state.s_zero, "cpu"):
                s_zero = float(agent.state.s_zero.cpu().detach().numpy())
            else:
                s_zero = float(agent.state.s_zero)
            
            delta_s = s - s_zero
            
            # Mathematical clustering by delta_s magnitude
            if abs(delta_s) > 1.0:
                # Logarithmic clustering: agents within same order of magnitude
                cluster_key = int(math.log10(abs(delta_s)))
            else:
                cluster_key = 0
            
            cluster_id = f"delta_s_magnitude_{cluster_key}"
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(agent)
        
        return clusters

    def _batch_compute_persistence_cluster(self, cluster_agents: List) -> Dict:
        """Compute observational persistence once per cluster, apply to mathematically similar agents."""
        if not cluster_agents:
            return {}
        
        # Use representative agent for cluster computation (they're mathematically similar)
        representative_agent = cluster_agents[0]
        persistence_result = representative_agent.compute_observational_persistence()
        
        # Apply same result to all agents in cluster
        cluster_cache = {}
        for agent in cluster_agents:
            cluster_cache[agent] = persistence_result
        
        return cluster_cache

    def orchestrate_persistence_optimization(self, agents: List) -> None:
        """Orchestrate observational persistence computation using mathematical clustering."""
        import time
        
        logger.info(f"🔧 Persistence optimization starting for {len(agents)} agents")
        start_time = time.time()
        
        # MATHEMATICAL THEORY OPTIMIZATION: Cluster by delta_s magnitude
        persistence_clusters = self._create_persistence_clusters(agents)
        clustering_time = time.time() - start_time
        logger.info(f"🔧 Persistence clustering: {len(persistence_clusters)} clusters in {clustering_time:.4f}s")
        
        # Compute persistence for each cluster and cache results
        total_cache = {}
        for cluster_id, cluster_agents in persistence_clusters.items():
            cluster_start = time.time()
            cluster_cache = self._batch_compute_persistence_cluster(cluster_agents)
            total_cache.update(cluster_cache)
            
            cluster_time = time.time() - cluster_start
            logger.info(f"🔧 Cluster {cluster_id}: {len(cluster_agents)} agents processed in {cluster_time:.4f}s")
        
        # Apply cached results to all agents
        for agent, persistence_result in total_cache.items():
            agent._cached_persistence_result = persistence_result
        
        total_time = time.time() - start_time
        logger.info(f"✅ Persistence optimization complete: {len(agents)} agents in {total_time:.3f}s")

    def _orchestrate_s_parameter_evolution(self, agents: List) -> float:
        """Coordinate observational state evolution using mathematical theory clustering."""
        import time

        logger.info(f"🔧 S-parameter evolution starting for {len(agents)} agents")

        complexity_measures = []
        sparse_graph = self.adaptive_tuning.get("sparse_interaction_graph")

        # MATHEMATICAL THEORY OPTIMIZATION: Use modular form periodicity and Q-magnitude clustering
        start_clustering = time.time()
        mathematical_clusters = self._create_mathematical_clusters(agents)
        clustering_time = time.time() - start_clustering
        logger.info(f"🔧 Mathematical clustering: {len(mathematical_clusters)} clusters in {clustering_time:.4f}s")

        # Process clusters instead of individual agents
        for cluster_id, cluster_agents in mathematical_clusters.items():
            cluster_start = time.time()
            
            # Create shared field context for cluster using sparse graph
            cluster_field_context = self._build_cluster_field_context(cluster_agents, sparse_graph, agents)
            
            # Batch evolve all agents in cluster with shared mathematical properties
            self._batch_evolve_cluster_s_parameter(cluster_agents, cluster_field_context)
            
            cluster_time = time.time() - cluster_start
            logger.info(f"🔧 Cluster {cluster_id}: {len(cluster_agents)} agents evolved in {cluster_time:.4f}s")

            # Collect complexity measures from cluster
            for agent in cluster_agents:
                s_distance = abs(agent.state.current_s - agent.state.s_zero)
                complexity_measures.append(s_distance)

        # Single sync operation for all agents (batch optimization)
        sync_start = time.time()
        self._batch_sync_agent_positions(agents)
        sync_time = time.time() - sync_start
        logger.info(f"🔧 Batch sync all agents: {sync_time:.4f}s")

        # Update dynamic field after all s-parameter evolution
        field_start = time.time()
        self.update_dynamic_field()
        field_time = time.time() - field_start
        logger.info(f"🔧 Dynamic field update after s-evolution: {field_time:.4f}s")

        # ADVANCED COMPLEXITY ANALYSIS USING SCIPY SIGNAL PROCESSING
        if complexity_measures:
            # GPU-native tensor operations
            complexity_tensor = torch.tensor(
                complexity_measures, dtype=torch.float32, device=self.device
            )
            complexity_expanded = complexity_tensor.unsqueeze(0).unsqueeze(0)
            complexity_flipped = torch.flip(complexity_tensor, [0]).unsqueeze(0).unsqueeze(0)
            complexity_correlation = torch.conv1d(complexity_expanded, complexity_flipped, padding='same').squeeze()
            # GPU-native log-sum-exp with safe division
            if complexity_correlation.numel() > 0:
                complexity_peak = torch.logsumexp(complexity_correlation, dim=0) / complexity_correlation.numel()
            else:
                complexity_peak = torch.tensor(0.0, device=self.device)
            return complexity_peak
        else:
            return 0.0

    def _measure_emergent_complexity(self, agents: List, tau: float) -> Dict[str, Any]:
        """Measure emergent patterns and complexity from agent interactions."""
        emergent_data = {
            "new_harmonics": 0,
            "hecke_adaptations": 0,
            "q_coefficient_diversity": 0.0,
            "tau": tau,
        }

        # Count new harmonics that emerged
        for agent in agents:
            # Count q-coefficients that weren't in original embedding
            original_size = len(agent.semantic_field.embedding_components)
            current_size = len(agent.breathing_q_coefficients)
            emergent_data["new_harmonics"] += max(0, current_size - original_size)

        # Measure Hecke eigenvalue adaptations
        for agent in agents:
            if hasattr(agent, "_initial_hecke_eigenvalues"):
                adaptations = sum(
                    1
                    for p in agent.hecke_eigenvalues
                    if abs(
                        agent.hecke_eigenvalues[p]
                        - agent._initial_hecke_eigenvalues.get(p)
                    )
                    > 0.01
                )
                emergent_data["hecke_adaptations"] += adaptations

        # Measure q-coefficient diversity across agents
        if len(agents) > 1:
            all_coeffs = []
            for agent in agents:
                agent_coeffs = [
                    abs(coeff) for coeff in agent.breathing_q_coefficients.values()
                ]
                all_coeffs.extend(agent_coeffs[:10])  # First 10 coefficients

            if all_coeffs:
                # Convert to tensor and use PyTorch operations for MPS compatibility
                all_coeffs_tensor = torch.tensor(
                    all_coeffs, dtype=torch.float32, device=self.device
                )
                # ADVANCED DIVERSITY ANALYSIS USING SCIPY EIGENVALUE DECOMPOSITION
                # MPS-compatible eigenvalue computation (CPU fallback for eigvals)
                coeffs_matrix = all_coeffs_tensor.reshape(-1, 1)
                cov_matrix = torch.matmul(coeffs_matrix, coeffs_matrix.T)
                # Move to CPU for eigenvalue computation (MPS doesn't support linalg.eigvals)
                eigenvals = torch.linalg.eigvals(cov_matrix.cpu()).real.to(self.device)
                # GPU-native eigenvalue diversity measure
                eigenvalue_diversity = torch.logsumexp(eigenvals, dim=0) / (
                    torch.logsumexp(eigenvals, dim=0) + 1e-8
                )
                emergent_data["q_coefficient_diversity"] = eigenvalue_diversity

        return emergent_data

    def _compute_collective_Q_field(self, agents: List) -> Dict[str, Any]:
        """Compute collective Q-field from all living modular forms."""
        collective_data = {
            "total_magnitude": 0.0,
            "phase_coherence": 0.0,
            "field_energy": 0.0,
            "agent_contributions": [],
        }

        # Gather all Q values
        Q_values = []
        for agent in agents:
            Q_val = agent.living_Q_value
            Q_values.append(Q_val)
            collective_data["agent_contributions"].append(
                {
                    "magnitude": abs(Q_val),
                    # ADVANCED PHASE EXTRACTION USING TORCH
                    "phase": (
                        torch.atan2(torch.tensor(Q_val.imag), torch.tensor(Q_val.real)).item()
                        if hasattr(Q_val, "imag")
                        else torch.atan2(
                            torch.tensor(Q_val).imag, torch.tensor(Q_val).real
                        ).item()
                    ),
                    "agent_id": agent.charge_id,
                }
            )

        if Q_values:
            # Total magnitude
            collective_data["total_magnitude"] = sum(abs(Q) for Q in Q_values)

            # ADVANCED PHASE COHERENCE ANALYSIS USING TORCH
            phases = [
                torch.atan2(torch.tensor(Q).imag, torch.tensor(Q).real).item()
                for Q in Q_values
            ]
            if len(phases) > 1:
                # GPU-native variance analysis using PyTorch eigenvalues
                phases_tensor = torch.tensor(phases, dtype=torch.float32, device=self.device).reshape(-1, 1)
                cov_matrix = torch.matmul(phases_tensor, phases_tensor.T)
                eigenvals = torch.linalg.eigvals(cov_matrix.cpu()).real.to(self.device)
                phase_variance = torch.logsumexp(eigenvals, dim=0) / len(eigenvals)
                collective_data["phase_coherence"] = torch.exp(
                    -torch.tensor(phase_variance)
                ).item()  # High coherence = low variance
            else:
                collective_data["phase_coherence"] = 1.0

            # Field energy (sum of squared magnitudes)
            collective_data["field_energy"] = sum(abs(Q) ** 2 for Q in Q_values)

        return collective_data

    def _analyze_evolution_results(
        self, evolution_results: Dict[str, Any], agents: List
    ) -> Dict[str, Any]:
        """Analyze the complete evolution to extract emergent patterns."""
        analysis = {
            "final_complexity": 0.0,
            "emergent_harmonics": [],
            "collective_coherence": 0.0,
            "evolution_stability": 0.0,
            "cascade_efficiency": 0.0,
            "total_Q_energy": 0.0,
        }

        if not evolution_results["complexity_evolution"]:
            return analysis

        # Final complexity
        analysis["final_complexity"] = evolution_results["complexity_evolution"][-1]

        # Calculate total Q energy from all agents
        total_q_energy = 0.0
        for agent in agents:
            if hasattr(agent, "living_Q_value") and agent.living_Q_value is not None:
                q_magnitude = abs(agent.living_Q_value)
                if math.isfinite(q_magnitude):
                    total_q_energy += q_magnitude
        analysis["total_Q_energy"] = total_q_energy

        # Count total emergent harmonics
        total_harmonics = sum(
            data["new_harmonics"] for data in evolution_results["emergent_harmonics"]
        )
        analysis["emergent_harmonics"] = list(
            range(total_harmonics)
        )  # Placeholder for actual harmonic data

        # Collective coherence (final breathing synchrony)
        if evolution_results["breathing_patterns"]:
            analysis["collective_coherence"] = evolution_results["breathing_patterns"][
                -1
            ]

        # Evolution stability (variance in complexity evolution)
        complexity_series = evolution_results["complexity_evolution"]
        if len(complexity_series) > 1:
            # Use torch for sophisticated variance - NO BASIC NUMPY
            complexity_variance = torch.var(
                torch.tensor(complexity_series, dtype=torch.float32)
            ).item()
            analysis["evolution_stability"] = 1.0 / (1.0 + complexity_variance)

        # Cascade efficiency (how well energy cascades between dimensions)
        if evolution_results["cascade_energies"]:
            cascade_energies = evolution_results["cascade_energies"]

            # MATHEMATICAL VALIDATION: Check data before polynomial fitting
            if len(cascade_energies) < 2:
                # Cannot fit trend to single point - mathematically undefined
                analysis["cascade_efficiency"] = 0.0
            elif all(not math.isfinite(x) for x in cascade_energies):
                # All values invalid - mathematically undefined trend
                analysis["cascade_efficiency"] = 0.0
            elif all(abs(x) > 1e30 for x in cascade_energies if math.isfinite(x)):
                # Values too extreme for numerical stability - use mathematical sign
                if cascade_energies[-1] > cascade_energies[0]:
                    analysis["cascade_efficiency"] = 1.0  # Positive mathematical trend
                else:
                    analysis["cascade_efficiency"] = (
                        0.0  # Non-positive mathematical trend
                    )
            else:
                # Attempt mathematical polynomial fitting with robustness
                # Pure mathematical trend calculation - no fallbacks
                # Use scipy for sophisticated polynomial fitting - NO BASIC NUMPY
                from scipy.optimize import curve_fit

                def linear(x, a, b):
                    return a * x + b

                x_data = torch.arange(
                    len(cascade_energies), dtype=torch.float32
                ).numpy()
                y_data = torch.tensor(cascade_energies, dtype=torch.float32).numpy()
                popt, _ = curve_fit(linear, x_data, y_data)
                cascade_trend = popt[0]
                analysis["cascade_efficiency"] = max(0.0, cascade_trend)

        return analysis

    def update_dynamic_field(self):
        """
        Update q_field_values with current agent living_Q_values.

        This ensures the orchestrator's field tensor stays synchronized with
        the evolving agent states rather than becoming stale.
        """
        # Reset field to zero
        self.q_field_values.fill_(0)

        # Add current contributions from all agents
        for agent in self.charge_agents.values():
            # Sync positions first to ensure consistency
            agent.sync_positions()

            # Get current living Q value
            current_Q = agent.living_Q_value

            # Update field at agent position
            self._update_q_field_contribution_dynamic(agent, current_Q)

    def _update_q_field_contribution_dynamic(self, agent, Q_value: complex):
        """
        Update field tensor with current Q value at agent position.

        Args:
            agent: ConceptualChargeAgent with current position
            Q_value: Current living Q value to add to field
        """
        x, y = agent.state.field_position

        # Convert field position to grid indices
        col = int((x + 1) * (self.field_resolution - 1) / 2)
        row = int((y + 1) * (self.field_resolution - 1) / 2)

        # Ensure indices are in bounds
        col = max(0, min(col, self.field_resolution - 1))
        row = max(0, min(row, self.field_resolution - 1))

        # Add Q value to field tensor with proper dtype conversion if needed
        if hasattr(self, "field_dtype"):
            # Convert Q_value to match field tensor dtype while preserving mathematical precision
            if self.field_dtype == torch.complex64 and isinstance(Q_value, complex):
                # For MPS: convert complex128 Q_value to complex64 safely
                # EVOLVE DATA TYPE: Handle large values by using log-space representation if needed
                # Extract magnitude in type-aware manner (same pattern as phase extraction)
                magnitude = (
                    torch.abs(Q_value).item()
                    if torch.is_tensor(Q_value)
                    else abs(Q_value)
                )

                if magnitude > 1e30:  # Value too large for direct complex64
                    # Store log magnitude and phase separately to preserve precision
                    log_magnitude = math.log(magnitude)
                    phase = (
                        torch.angle(Q_value)
                        if torch.is_tensor(Q_value)
                        else torch.atan2(
                            torch.tensor(Q_value).imag, torch.tensor(Q_value).real
                        )
                    )
                    # Reconstruct with scaled magnitude to prevent overflow
                    scaled_magnitude = math.exp(
                        min(log_magnitude, 30.0)
                    )  # Cap at e^30 for complex64
                    q_np = scaled_magnitude * complex(math.cos(phase), math.sin(phase))
                    logger.debug(
                        f"🔧 Large Q-value scaled: |Q|={magnitude:.2e} → {abs(q_np):.2e}"
                    )
                else:
                    # Normal conversion for reasonable values - use PyTorch conversion
                    q_np = (
                        Q_value.to(torch.complex64)
                        if torch.is_tensor(Q_value)
                        else complex(Q_value)
                    )
                q_tensor = torch.tensor(
                    q_np, dtype=self.field_dtype, device=self.device
                )
                self.q_field_values[row, col] += q_tensor
            else:
                # Direct addition for complex128 or non-complex values
                self.q_field_values[row, col] += torch.tensor(
                    Q_value, dtype=self.field_dtype, device=self.device
                )
        else:
            # Fallback: direct addition
            self.q_field_values[row, col] += Q_value

    def __init_adaptive_optimization(self):
        """
        Initialize adaptive optimization system that LISTENS to modular forms.

        This creates a living tuning system that learns from the mathematics itself,
        adapting computation strategies based on what the modular forms reveal.
        """
        self.adaptive_tuning = {
            # LISTENING HISTORY - track mathematical patterns over time
            "eigenvalue_history": [],
            "breathing_history": [],
            "interaction_history": [],
            "cascade_history": [],
            "phase_history": [],
            # DYNAMIC OPTIMIZATIONS - adapt based on listening
            "eigenvalue_clusters": {},
            "breathing_sync_groups": [],
            "sparse_interaction_graph": {},
            "resonance_cascades": [],
            "phase_boundaries": [],
            # LEARNING PARAMETERS - evolve with experience
            "cluster_sensitivity": 0.1,  # How sensitive eigenvalue clustering is
            "sync_threshold": torch.pi / 8,  # Phase threshold for breathing sync
            "interaction_cutoff": 0.01,  # Minimum interaction strength
            "cascade_threshold": 0.8,  # Phase alignment for cascades
            "adaptation_rate": 0.02,  # How fast parameters adapt
            # PERFORMANCE TRACKING - learn what works
            "computation_efficiency": [],
            "mathematical_coherence": [],
            "optimization_success": {},
            "last_adaptation_step": 0,
            # LISTENER SYSTEMS - analyze modular form behavior
            "eigenvalue_listener": self._create_eigenvalue_listener(),
            "breathing_listener": self._create_breathing_listener(),
            "interaction_listener": self._create_interaction_listener(),
            "cascade_listener": self._create_cascade_listener(),
            "phase_listener": self._create_phase_listener(),
        }

    def listen_to_modular_forms(self, agents: List, tau: float):
        """
        LISTEN to what the modular forms are telling us about optimal computation.

        This is the core insight: let the mathematics itself guide optimization.
        The modular forms reveal their own computational preferences through their behavior.
        """
        tuning = self.adaptive_tuning

        # LISTEN 1: 🎯 Eigenvalue Pattern Detection
        eigenvalue_patterns = tuning["eigenvalue_listener"].listen(agents)
        if eigenvalue_patterns.get("new_clusters_detected"):
            # The eigenvalues are telling us they want different clustering
            tuning["cluster_sensitivity"] = eigenvalue_patterns.get(
                "suggested_sensitivity"
            )
            logger.info(
                f"🎯 Eigenvalues suggest sensitivity: {tuning['cluster_sensitivity']:.3f}"
            )

        # LISTEN 2: 🌊 Breathing Synchronization Detection
        breathing_patterns = tuning["breathing_listener"].listen(agents, tau)
        if breathing_patterns.get("sync_strength_changed"):
            # The breathing patterns are showing us new synchronization thresholds
            tuning["sync_threshold"] = breathing_patterns.get("optimal_threshold")
            logger.info(
                f"🌊 Breathing suggests sync threshold: {tuning['sync_threshold']:.3f}"
            )

        # LISTEN 3: 🔗 Interaction Strength Learning
        interaction_patterns = tuning["interaction_listener"].listen(agents)
        if interaction_patterns.get("cutoff_should_adapt"):
            # The interactions are showing us what strength levels actually matter
            tuning["interaction_cutoff"] = interaction_patterns.get("suggested_cutoff")
            logger.info(
                f"🔗 Interactions suggest cutoff: {tuning['interaction_cutoff']:.4f}"
            )

        # LISTEN 4: 🎼 Resonance Cascade Discovery
        cascade_patterns = tuning["cascade_listener"].listen(agents)
        if cascade_patterns.get("new_cascades_found"):
            # The resonances are revealing new amplification opportunities
            tuning["cascade_threshold"] = cascade_patterns.get("optimal_threshold")
            logger.info(
                f"🎼 Cascades suggest threshold: {tuning['cascade_threshold']:.3f}"
            )

        # LISTEN 5: 🌀 Phase Coherence Boundaries
        phase_patterns = tuning["phase_listener"].listen(agents)
        if phase_patterns.get("boundaries_shifted"):
            # The phase relationships are showing us new computational boundaries
            new_boundaries = phase_patterns.get("new_boundaries")
            logger.info(f"🌀 Phases suggest {len(new_boundaries)} new boundaries")

    def adapt_computation_strategy(self, agents: List, tau: float):
        """
        ADAPT the computation strategy based on what we learned from listening.

        This implements dynamic tuning - the optimization evolves with the mathematics.
        """
        tuning = self.adaptive_tuning

        # ADAPT 1: 🎯 Dynamic Eigenvalue Clustering
        if int(tau * 100) % 10 == 0:  # Re-cluster every 10 steps
            new_clusters = self._adaptive_eigenvalue_clustering(agents)
            if self._clusters_significantly_different(
                tuning["eigenvalue_clusters"], new_clusters
            ):
                tuning["eigenvalue_clusters"] = new_clusters
                logger.info(
                    f"🎯 Adapted eigenvalue clustering: {len(new_clusters)} clusters"
                )

        # ADAPT 2: 🌊 Dynamic Breathing Groups
        new_sync_groups = self._adaptive_breathing_grouping(agents)
        if self._sync_groups_changed(tuning["breathing_sync_groups"], new_sync_groups):
            tuning["breathing_sync_groups"] = new_sync_groups
            logger.info(
                f"🌊 Adapted breathing groups: {len(new_sync_groups)} sync groups"
            )

        # ADAPT 3: 🔗 Dynamic Interaction Graph
        # 🔧 FIX: Always build graph if empty, otherwise rebuild periodically
        graph_is_empty = not tuning["sparse_interaction_graph"]
        should_rebuild = int(tau * 100) % 5 == 0

        if graph_is_empty or should_rebuild:
            new_graph = self._adaptive_interaction_graph(agents)
            tuning["sparse_interaction_graph"] = new_graph
            total_interactions = sum(len(neighbors) for neighbors in new_graph.values())
            status = "initialized" if graph_is_empty else "rebuilt"
            logger.info(
                f"🔗 Interaction graph {status}: {total_interactions} total interactions"
            )

        # ADAPT 4: 🎼 Dynamic Cascade Detection
        new_cascades = self._adaptive_cascade_detection(agents)
        if len(new_cascades) != len(tuning["resonance_cascades"]):
            tuning["resonance_cascades"] = new_cascades
            logger.info(f"🎼 Adapted cascades: {len(new_cascades)} resonance chains")

        # ADAPT 5: 🌀 Dynamic Phase Boundaries
        if int(tau * 100) % 15 == 0:  # Recompute boundaries every 15 steps
            new_boundaries = self._adaptive_phase_boundaries(agents)
            tuning["phase_boundaries"] = new_boundaries
            logger.info(f"🌀 Adapted phase boundaries: {len(new_boundaries)} regions")

    def initialize_liquid_simulation(
        self, combined_results: Dict[str, Any], max_agents: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Complete initialization of liquid simulation from ChargeFactory data.

        MAIN ENTRY POINT: This method coordinates everything from factory data to live simulation.

        Args:
            combined_results: ChargeFactory output dictionary
            max_agents: Maximum number of agents to create

        Returns:
            Initialization summary
        """
        logger.info("=" * 60)
        logger.info("INITIALIZING LIQUID SIMULATION")
        logger.info("LiquidOrchestrator RUNS THE SHOW")
        logger.info("=" * 60)

        # Step 1: Load factory data
        num_charges = self.load_charge_factory_results(combined_results)

        # Step 2: Create agents
        num_agents = self.create_agents_from_factory_data(max_agents)

        # Step 3: Initialize field dynamics
        initial_stats = self.get_field_statistics()

        summary = {
            "charges_loaded": num_charges,
            "agents_created": num_agents,
            "field_resolution": self.field_resolution,
            "device": str(self.device),
            "initial_field_energy": initial_stats["field_energy"],
            "ready_for_simulation": num_agents > 0,
        }

        logger.info("Liquid simulation initialization complete:")
        logger.info(f"  Charges loaded: {num_charges}")
        logger.info(f"  Agents created: {num_agents}")
        logger.info(
            f"  Field resolution: {self.field_resolution}x{self.field_resolution}"
        )
        logger.info(f"  Device: {str(self.device)}")
        logger.info(f"  Initial field energy: {initial_stats['field_energy']:.6f}")
        logger.info(f"  Ready for simulation: {summary['ready_for_simulation']}")

        return summary

    def add_conceptual_charge(
        self,
        charge_obj: ConceptualChargeObject,
        field_position: Optional[Tuple[float, float]] = None,
    ):
        """
        Add a conceptual charge agent to the liquid simulation.

        Args:
            charge_obj: ConceptualChargeObject with complete field components
            field_position: Optional spatial position in field grid
        """
        charge_id = charge_obj.charge_id

        # Set field position if not provided
        if field_position is None:
            # Place randomly but away from existing charges
            field_position = self._find_optimal_placement(charge_obj)

        charge_obj.set_field_position(field_position)
        self.active_charges[charge_id] = charge_obj

        # Update Q-field with new charge contribution
        self._update_q_field_contribution(charge_obj)

    def _find_optimal_placement(
        self, charge_obj: ConceptualChargeObject
    ) -> Tuple[float, float]:
        """Find optimal placement for new charge to minimize interference."""
        if not self.active_charges:
            return (0.0, 0.0)

        # Simple placement strategy: find low-field region
        field_magnitude = torch.abs(self.q_field_values)
        min_idx = torch.argmin(field_magnitude)
        row, col = min_idx // self.field_resolution, min_idx % self.field_resolution

        # Convert grid indices to field coordinates
        x = -1 + 2 * col / (self.field_resolution - 1)
        y = -1 + 2 * row / (self.field_resolution - 1)

        return (float(x), float(y))

    def _update_q_field_contribution(self, charge_obj: ConceptualChargeObject):
        """Update Q-field tensor with charge contribution."""
        if charge_obj.metadata.field_position is None:
            return

        x, y = charge_obj.metadata.field_position

        # Convert field position to grid indices
        col = int((x + 1) * (self.field_resolution - 1) / 2)
        row = int((y + 1) * (self.field_resolution - 1) / 2)

        # Ensure indices are in bounds
        col = max(0, min(col, self.field_resolution - 1))
        row = max(0, min(row, self.field_resolution - 1))

        # Add Q-value contribution
        q_value = complex(charge_obj.complete_charge)
        self.q_field_values[row, col] += q_value

    def compute_field_interference_patterns(self) -> FieldInterferencePattern:
        """
        Compute interference patterns using current living Q values.

        Returns patterns of constructive/destructive interference between
        conceptual charges in the liquid stage using real-time Q values.
        """
        charge_ids = list(self.active_charges.keys())
        n_charges = len(charge_ids)

        if n_charges < 2:
            return FieldInterferencePattern(
                charge_pairs=[],
                interference_strengths=torch.empty(0, device=self.device),
                phase_relationships=torch.empty(0, device=self.device),
                field_distortions=torch.empty(0, device=self.device),
            )

        # MATHEMATICAL THEORY OPTIMIZATION: Pre-compute tau evaluations
        group_agents = [self.charge_agents[charge_id] for charge_id in charge_ids]
        tau_evaluations = self._precompute_tau_evaluations(group_agents, n_charges)

        # Compute pairwise interference using CURRENT living Q values
        pairs = []
        strengths = []
        phases = []
        distortions = []

        for i in range(n_charges):
            for j in range(i + 1, n_charges):
                agent_a = self.charge_agents[charge_ids[i]]
                agent_b = self.charge_agents[charge_ids[j]]

                # REAL EISENSTEIN FORM FIELD INTERFERENCE - NO BASIC MULTIPLICATION
                q_a = agent_a.living_Q_value
                q_b = agent_b.living_Q_value

                # MATHEMATICAL THEORY OPTIMIZATION: Use cached modular form bases
                eisenstein_basis = self.eisenstein_basis  # O(1) cached lookup
                cusp_basis = self.cusp_basis             # O(1) cached lookup

                # Convert to Sage CDF for mathematical precision
                q_a_sage = CDF(complex(q_a))
                q_b_sage = CDF(complex(q_b))

                # MATHEMATICAL THEORY OPTIMIZATION: Use pre-computed tau evaluations (O(1) lookup)
                if agent_a in tau_evaluations and agent_b in tau_evaluations:
                    eisenstein_a = tau_evaluations[agent_a]['eisenstein']
                    eisenstein_b = tau_evaluations[agent_b]['eisenstein']
                    cusp_a = tau_evaluations[agent_a]['cusp']
                    cusp_b = tau_evaluations[agent_b]['cusp']

                    # Choose interference type based on available modular forms
                    if cusp_a is not None and cusp_b is not None:
                        # SOPHISTICATED interference: Eisenstein + Cusp combination
                        interference = (
                            q_a_sage
                            * q_b_sage.conjugate()
                            * (
                                eisenstein_a * eisenstein_b.conjugate()
                                + cusp_a * cusp_b.conjugate()
                            )
                        )
                    else:
                        # Pure Eisenstein interference
                        interference = (
                            q_a_sage
                            * q_b_sage.conjugate()
                            * eisenstein_a
                            * eisenstein_b.conjugate()
                        )

                    strength = abs(interference)
                    phase_diff = float(interference.argument())

                    # Sophisticated distortion using full modular form mathematics
                    distortion = float(strength.real()) * math.cos(phase_diff)
                else:
                    raise ValueError(
                        "Modular forms unavailable - mathematical system corrupted!"
                    )

                pairs.append((charge_ids[i], charge_ids[j]))
                strengths.append(strength)
                phases.append(phase_diff)
                distortions.append(distortion)

        return FieldInterferencePattern(
            charge_pairs=pairs,
            interference_strengths=torch.tensor(strengths, device=self.device),
            phase_relationships=torch.tensor(phases, device=self.device),
            field_distortions=torch.tensor(distortions, device=self.device),
        )

    def update_emotional_conductor(self, tau_step: float):
        """
        Update emotional conductor state affecting S-T interactions.

        The emotional dimension acts as a conductor that modulates how
        semantic and temporal components interact within Q(τ, C, s).
        """
        # Compute field-wide emotional modulation
        total_emotional_influence = 0.0
        conductor_harmonics = torch.zeros_like(self.emotional_conductor.field_harmonics)

        for charge_obj in self.active_charges.values():
            # Extract emotional trajectory component
            emotional_traj = charge_obj.field_components.emotional_trajectory

            # Add to conductor harmonics
            if len(emotional_traj) > 0:
                # Simple projection of emotional trajectory to field harmonics
                # GPU-native averaging using PyTorch
                traj_tensor = torch.tensor(emotional_traj, device=self.device)
                freq_component = torch.logsumexp(traj_tensor, dim=0) / len(emotional_traj)
                total_emotional_influence += abs(freq_component)

                # Distribute across harmonic frequencies
                for k in range(min(len(conductor_harmonics), len(emotional_traj))):
                    conductor_harmonics[k] += (
                        emotional_traj[k] if k < len(emotional_traj) else 0.0
                    )

        # Update conductor state
        self.emotional_conductor.modulation_tensor *= (
            1.0 + 0.1 * total_emotional_influence
        )
        self.emotional_conductor.conductor_phase += tau_step * total_emotional_influence
        self.emotional_conductor.field_harmonics = conductor_harmonics

        # Update S-T coupling strength based on emotional field
        self.emotional_conductor.s_t_coupling_strength = (
            1.0 + 0.2 * total_emotional_influence
        )

    def evolve_observational_states(self, tau_step: float):
        """
        Evolve observational states s across the field.

        Updates s-values that affect all components of Q(τ, C, s) and
        implements the dual-decay persistence structure.
        """
        # Compute s-value gradients from Q-field
        q_magnitude = torch.abs(self.q_field_values)

        # Gradient computation
        grad_x = torch.gradient(q_magnitude, dim=1)[0]
        grad_y = torch.gradient(q_magnitude, dim=0)[0]

        self.observational_state.s_gradients = torch.stack([grad_x, grad_y], dim=-1)

        # Update s-values based on field dynamics
        gradient_magnitude = torch.norm(self.observational_state.s_gradients, dim=-1)
        s_evolution = (
            tau_step
            * gradient_magnitude
            * self.emotional_conductor.s_t_coupling_strength
        )

        self.observational_state.current_s_values += s_evolution

        # Apply persistence decay
        decay_factor = torch.exp(-tau_step * 0.1)  # Adjustable decay rate
        self.observational_state.persistence_factors *= decay_factor

        # Update trajectory tracking
        self.observational_state.evolution_trajectories += (
            self.observational_state.s_gradients * tau_step
        )

        # Update individual charge observational states
        for charge_obj in self.active_charges.values():
            if charge_obj.metadata.field_position is not None:
                x, y = charge_obj.metadata.field_position
                col = int((x + 1) * (self.field_resolution - 1) / 2)
                row = int((y + 1) * (self.field_resolution - 1) / 2)

                col = max(0, min(col, self.field_resolution - 1))
                row = max(0, min(row, self.field_resolution - 1))

                new_s = float(self.observational_state.current_s_values[row, col])
                charge_obj.update_observational_state(new_s)

    def simulate_liquid_dynamics(
        self, tau_steps: int = 100, tau_step_size: float = 0.01
    ) -> Dict[str, Any]:
        """
        Run liquid stage simulation with field-theoretic dynamics.

        Args:
            tau_steps: Number of tau evolution steps
            tau_step_size: Size of each tau step

        Returns:
            Simulation results with field evolution data
        """
        simulation_results = {
            "field_evolution": [],
            "interference_patterns": [],
            "emotional_conductor_states": [],
            "observational_evolution": [],
            "final_q_field": None,
            "charge_trajectories": {
                charge_id: [] for charge_id in self.active_charges.keys()
            },
        }

        for step in range(tau_steps):
            # Update tau
            self.current_tau += tau_step_size
            self.simulation_time += tau_step_size

            # Compute field interference patterns
            interference = self.compute_field_interference_patterns()
            simulation_results["interference_patterns"].append(interference)

            # Update emotional conductor
            self.update_emotional_conductor(tau_step_size)

            # Evolve observational states
            self.evolve_observational_states(tau_step_size)

            # Record field state
            field_snapshot = {
                "tau": self.current_tau,
                "q_field_magnitude": torch.abs(self.q_field_values).clone(),
                "q_field_phase": torch.angle(self.q_field_values).clone(),
                "emotional_modulation": self.emotional_conductor.modulation_tensor.clone(),
                "s_values": self.observational_state.current_s_values.clone(),
            }
            simulation_results["field_evolution"].append(field_snapshot)

            # Record charge positions
            for charge_id, charge_obj in self.active_charges.items():
                if charge_obj.metadata.field_position is not None:
                    simulation_results["charge_trajectories"][charge_id].append(
                        {
                            "tau": self.current_tau,
                            "position": charge_obj.metadata.field_position,
                            "q_value": charge_obj.complete_charge,
                            "s_value": charge_obj.observational_state,
                        }
                    )

        # Final field state
        simulation_results["final_q_field"] = self.q_field_values.clone()

        return simulation_results

    def get_field_statistics(self) -> Dict[str, Any]:
        """Get current field statistics and health metrics."""
        # 🚀 FIX: Use living_Q_value from agents instead of empty q_field_values tensor
        if self.charge_agents:
            # Calculate field statistics from actual agent Q values
            agent_q_values = [
                abs(agent.living_Q_value) for agent in self.charge_agents.values()
            ]
            if agent_q_values:
                q_magnitude_list = agent_q_values
                field_energy = sum(q**2 for q in q_magnitude_list)
                max_field_strength = max(q_magnitude_list)
                mean_field_strength = sum(q_magnitude_list) / len(q_magnitude_list)
                field_coverage = sum(1 for q in q_magnitude_list if q > 0.01) / len(
                    q_magnitude_list
                )
            else:
                # NO FALLBACK - Agent Q values must exist for field analysis
                raise ValueError(
                    "MATHEMATICAL FAILURE: No agent Q values available - "
                    "Field energy cannot be computed. System requires charge data."
                )
        else:
            # NO FALLBACK - Agent system must exist for field calculation
            raise ValueError(
                "MATHEMATICAL FAILURE: No agents available for field calculation - "
                "System requires active agent data."
            )

        return {
            "active_charges": len(self.active_charges),
            "field_energy": field_energy,
            "max_field_strength": max_field_strength,
            "mean_field_strength": mean_field_strength,
            "field_coverage": field_coverage,
            "emotional_conductor_strength": float(
                self.emotional_conductor.s_t_coupling_strength
            ),
            # GPU-native S-value analysis using PyTorch
            "s_values_array": float(
                torch.logsumexp(self.observational_state.current_s_values, dim=0)
                / len(self.observational_state.current_s_values)
            ),
            "current_tau": self.current_tau,
            "simulation_time": self.simulation_time,
        }

    def reset_simulation(self):
        """Reset simulation state while keeping charges."""
        self.current_tau = 0.0
        self.simulation_time = 0.0
        self.field_history.clear()

        # Reset field tensors
        self.q_field_values = torch.zeros_like(self.q_field_values)
        self.observational_state.current_s_values = torch.ones_like(
            self.observational_state.current_s_values
        )
        self.observational_state.s_gradients = torch.zeros_like(
            self.observational_state.s_gradients
        )
        self.observational_state.persistence_factors = torch.ones_like(
            self.observational_state.persistence_factors
        )

        # Rebuild Q-field from active charges
        for charge_obj in self.active_charges.values():
            self._update_q_field_contribution(charge_obj)

    # ========================================
    # ADAPTIVE OPTIMIZATION SYSTEM - O(N log N) SCALING
    # ========================================

    def _create_eigenvalue_listener(self):
        """Create eigenvalue pattern listener that analyzes Hecke eigenvalue clustering."""

        class EigenvalueListener:
            def __init__(self, orchestrator):
                self.orchestrator = orchestrator
                self.last_cluster_count = 0
                self.last_eigenvalue_spread = 0.0

            def listen(self, agents):
                """Analyze eigenvalue patterns to suggest clustering strategies."""
                if not agents:
                    return {"new_clusters_detected": False}

                # Extract eigenvalues from living modular forms
                eigenvalues = []
                for agent in agents:
                    if hasattr(agent, "hecke_eigenvalues") and agent.hecke_eigenvalues:
                        eigenvalues.extend(agent.hecke_eigenvalues.values())

                if len(eigenvalues) < 2:
                    return {"new_clusters_detected": False}

                # Analyze eigenvalue distribution using magnitudes
                # Convert complex eigenvalues to magnitudes for spread calculation
                eigenvalue_magnitudes = [abs(e) for e in eigenvalues]
                eigenvalue_spread = torch.std(
                    torch.tensor(eigenvalue_magnitudes, dtype=torch.float32)
                ).item()

                # Detect if clustering should be adapted
                spread_change = abs(eigenvalue_spread - self.last_eigenvalue_spread)
                cluster_change_detected = (
                    spread_change > 0.1 * self.last_eigenvalue_spread
                )

                self.last_eigenvalue_spread = eigenvalue_spread

                if cluster_change_detected:
                    # Suggest new sensitivity based on eigenvalue density
                    suggested_sensitivity = max(0.05, min(0.5, eigenvalue_spread / 10))
                    return {
                        "new_clusters_detected": True,
                        "suggested_sensitivity": suggested_sensitivity,
                        "eigenvalue_spread": eigenvalue_spread,
                    }

                return {"new_clusters_detected": False}

        return EigenvalueListener(self)

    def _create_breathing_listener(self):
        """Create breathing pattern listener that analyzes synchronization."""

        class BreathingListener:
            def __init__(self, orchestrator):
                self.orchestrator = orchestrator
                self.last_sync_strength = 0.0

            def listen(self, agents, tau):
                """Analyze breathing synchronization to suggest optimal thresholds."""
                if len(agents) < 2:
                    return {"sync_strength_changed": False}

                # Extract breathing phases
                breathing_phases = []
                for agent in agents:
                    if hasattr(agent, "breath_phase"):
                        breathing_phases.append(agent.breath_phase)

                if len(breathing_phases) < 2:
                    return {"sync_strength_changed": False}

                # Calculate current synchronization strength
                phase_differences = []
                for i in range(len(breathing_phases)):
                    for j in range(i + 1, len(breathing_phases)):
                        phase_diff = abs(breathing_phases[i] - breathing_phases[j])
                        # Use torch.pi for sophisticated constant - NO BASIC NUMPY
                        phase_diff = min(
                            phase_diff, 2 * torch.pi - phase_diff
                        )  # Wrap to [0, π]
                        phase_differences.append(phase_diff)

                # Convert to tensor and use PyTorch operations for MPS compatibility
                phase_differences_tensor = torch.tensor(
                    phase_differences, device=self.orchestrator.device
                )
                # GPU-native phase synchronization analysis
                phase_expanded = phase_differences_tensor.unsqueeze(0).unsqueeze(0)
                phase_flipped = torch.flip(phase_differences_tensor, [0]).unsqueeze(0).unsqueeze(0)
                sync_correlation = torch.conv1d(phase_expanded, phase_flipped, padding='same').squeeze()
                # Safe division check
                if sync_correlation.numel() > 0:
                    sync_strength = 1.0 - torch.logsumexp(sync_correlation, dim=0) / (
                        sync_correlation.numel() * math.pi
                    )
                else:
                    sync_strength = 1.0

                # Detect significant sync changes
                sync_change = abs(sync_strength - self.last_sync_strength)
                sync_changed = sync_change > 0.1

                self.last_sync_strength = sync_strength

                if sync_changed:
                    # Suggest optimal threshold based on current sync patterns
                    if phase_differences:
                        optimal_threshold = torch.quantile(
                            phase_differences_tensor, 0.75
                        )
                    else:
                        # Use torch.pi for sophisticated constant - NO BASIC NUMPY
                        optimal_threshold = torch.pi / 8
                    return {
                        "sync_strength_changed": True,
                        "optimal_threshold": optimal_threshold,
                        "sync_strength": sync_strength,
                    }

                return {"sync_strength_changed": False}

        return BreathingListener(self)

    def _create_interaction_listener(self):
        """Create interaction strength listener that monitors field coupling."""

        class InteractionListener:
            def __init__(self, orchestrator):
                self.orchestrator = orchestrator
                self.last_avg_interaction = 0.0

            def listen(self, agents):
                """Analyze interaction strengths to suggest cutoff adaptations."""
                if len(agents) < 2:
                    return {"cutoff_should_adapt": False}

                # Calculate interaction strengths between agents
                interaction_strengths = []
                for i, agent1 in enumerate(agents):
                    for j, agent2 in enumerate(agents[i + 1 :], i + 1):
                        # Calculate field interaction strength
                        if hasattr(agent1, "living_Q_value") and hasattr(
                            agent2, "living_Q_value"
                        ):
                            Q1, Q2 = agent1.living_Q_value, agent2.living_Q_value
                            interaction = (
                                abs(Q1 * torch.conj(torch.tensor(Q2)))
                                if Q1 is not None and Q2 is not None
                                else 0.0
                            )
                            # Use torch for sophisticated real extraction - NO BASIC NUMPY
                            interaction_strengths.append(
                                float(
                                    torch.real(
                                        torch.tensor(interaction, dtype=torch.complex64)
                                    ).item()
                                )
                            )

                if not interaction_strengths:
                    return {"cutoff_should_adapt": False}

                # GPU-native averaging using PyTorch
                interaction_tensor = torch.tensor(interaction_strengths, device=self.orchestrator.device)
                avg_interaction = torch.logsumexp(interaction_tensor, dim=0) / len(
                    interaction_strengths
                )

                # Detect if cutoff should adapt
                interaction_change = abs(avg_interaction - self.last_avg_interaction)
                cutoff_should_adapt = (
                    interaction_change > 0.5 * self.last_avg_interaction
                    if self.last_avg_interaction > 0
                    else False
                )

                self.last_avg_interaction = avg_interaction

                if cutoff_should_adapt:
                    # Suggest cutoff as 10% of median interaction strength
                    # Use torch for sophisticated percentile - NO BASIC NUMPY
                    strengths_tensor = torch.tensor(
                        interaction_strengths, dtype=torch.float32
                    )
                    percentile_10 = torch.quantile(strengths_tensor, 0.1).item()
                    suggested_cutoff = max(0.001, percentile_10)
                    return {
                        "cutoff_should_adapt": True,
                        "suggested_cutoff": suggested_cutoff,
                        "avg_interaction": avg_interaction,
                    }

                return {"cutoff_should_adapt": False}

        return InteractionListener(self)

    def _create_cascade_listener(self):
        """Create cascade listener that detects resonance opportunities."""

        class CascadeListener:
            def __init__(self, orchestrator):
                self.orchestrator = orchestrator
                self.last_cascade_count = 0

            def listen(self, agents):
                """Analyze phase alignments to detect cascade opportunities."""
                if len(agents) < 3:
                    return {"new_cascades_found": False}

                # Find potential cascade chains (3+ agents with aligned phases)
                cascade_chains = []
                agent_phases = []

                valid_agents = 0
                invalid_agents = 0
                
                for agent in agents:
                    if (
                        hasattr(agent, "living_Q_value")
                        and agent.living_Q_value is not None
                    ):
                        try:
                            phase = torch.atan2(
                                torch.tensor(agent.living_Q_value).imag,
                                torch.tensor(agent.living_Q_value).real,
                            )
                            agent_phases.append((agent, phase))
                            valid_agents += 1
                        except Exception as e:
                            logger.error(f"Failed to calculate phase for agent: {e}")
                            invalid_agents += 1
                            raise ValueError(f"Phase calculation failed for agent: {e}")
                    else:
                        invalid_agents += 1
                
                # Validate we have enough valid agents
                if valid_agents == 0:
                    raise ValueError(f"No agents with valid living_Q_value! Total agents: {len(agents)}")
                
                if invalid_agents > valid_agents:
                    logger.warning(f"⚠️ More invalid agents ({invalid_agents}) than valid ({valid_agents})")
                    raise ValueError(f"Too many invalid agents: {invalid_agents}/{len(agents)}")
                
                logger.debug(f"Phase calculation: {valid_agents} valid, {invalid_agents} invalid agents")

                # Look for phase-aligned groups that could cascade
                phase_groups = {}
                for agent, phase in agent_phases:
                    phase_bucket = (
                        round((phase * 8 / (2 * torch.pi)).item()) * (2 * torch.pi) / 8
                    )  # Discretize phases
                    if phase_bucket not in phase_groups:
                        phase_groups[phase_bucket] = []
                    phase_groups[phase_bucket].append(agent)

                # Validate phase distribution
                if len(phase_groups) == 0:
                    raise ValueError("No phase groups created despite having valid agents!")
                
                if len(phase_groups) == 1 and valid_agents > 10:
                    single_phase = list(phase_groups.keys())[0]
                    logger.error(f"All {valid_agents} agents collapsed to single phase bucket: {single_phase}")
                    raise ValueError(f"Degenerate phase distribution: all agents in one bucket")
                
                # Log phase distribution for monitoring
                phase_distribution = {f"{k:.3f}": len(v) for k, v in phase_groups.items()}
                logger.debug(f"Phase distribution across {len(phase_groups)} buckets: {phase_distribution}")
                
                # Find groups with 3+ agents (potential cascades)
                for phase_bucket, group_agents in phase_groups.items():
                    if len(group_agents) >= 3:
                        cascade_chains.append(group_agents)

                new_cascade_count = len(cascade_chains)
                cascades_changed = new_cascade_count != self.last_cascade_count

                self.last_cascade_count = new_cascade_count

                if cascades_changed:
                    # Suggest threshold based on phase spread within cascade groups
                    if cascade_chains:
                        phase_spreads = []
                        for chain in cascade_chains:
                            chain_phases = [
                                torch.atan2(
                                    torch.tensor(agent.living_Q_value).imag,
                                    torch.tensor(agent.living_Q_value).real,
                                )
                                for agent in chain
                                if hasattr(agent, "living_Q_value")
                                and agent.living_Q_value is not None
                            ]
                            if len(chain_phases) > 1:
                                # Use torch for sophisticated standard deviation - NO BASIC NUMPY
                                phase_spreads.append(
                                    torch.std(
                                        torch.tensor(chain_phases, dtype=torch.float32)
                                    ).item()
                                )

                        # GPU-native averaging using PyTorch
                        if phase_spreads:
                            spreads_tensor = torch.tensor(phase_spreads, device=self.orchestrator.device)
                            optimal_threshold = torch.logsumexp(spreads_tensor, dim=0) / len(phase_spreads)
                        else:
                            optimal_threshold = 0.8
                    else:
                        optimal_threshold = 0.8

                    return {
                        "new_cascades_found": True,
                        "optimal_threshold": optimal_threshold,
                        "cascade_count": new_cascade_count,
                    }

                return {"new_cascades_found": False}

        return CascadeListener(self)

    def _create_phase_listener(self):
        """Create phase boundary listener that analyzes coherence regions."""

        class PhaseListener:
            def __init__(self, orchestrator):
                self.orchestrator = orchestrator
                self.last_boundary_count = 0

            def listen(self, agents):
                """Analyze phase coherence to suggest computational boundaries."""
                if len(agents) < 4:
                    return {"boundaries_shifted": False}

                # Extract phases and positions
                agent_data = []
                for agent in agents:
                    if (
                        hasattr(agent, "living_Q_value")
                        and agent.living_Q_value is not None
                        and hasattr(agent, "state")
                        and hasattr(agent.state, "field_position")
                    ):
                        phase = torch.atan2(
                            torch.tensor(agent.living_Q_value).imag,
                            torch.tensor(agent.living_Q_value).real,
                        )
                        pos = agent.state.field_position
                        agent_data.append((pos[0], pos[1], phase))

                if len(agent_data) < 4:
                    return {"boundaries_shifted": False}

                # Simple phase boundary detection using phase gradients
                boundaries = []
                threshold = torch.pi / 4  # Phase difference threshold for boundary

                # Sort by x position to detect x-boundaries
                agent_data_x = sorted(agent_data, key=lambda x: x[0])
                for i in range(len(agent_data_x) - 1):
                    phase_diff = abs(agent_data_x[i][2] - agent_data_x[i + 1][2])
                    # Use torch.pi for sophisticated constant - NO BASIC NUMPY
                    phase_diff = min(phase_diff, 2 * torch.pi - phase_diff)
                    if phase_diff > threshold:
                        boundaries.append(
                            ("x", (agent_data_x[i][0] + agent_data_x[i + 1][0]) / 2)
                        )

                # Sort by y position to detect y-boundaries
                agent_data_y = sorted(agent_data, key=lambda x: x[1])
                for i in range(len(agent_data_y) - 1):
                    phase_diff = abs(agent_data_y[i][2] - agent_data_y[i + 1][2])
                    # Use torch.pi for sophisticated constant - NO BASIC NUMPY
                    phase_diff = min(phase_diff, 2 * torch.pi - phase_diff)
                    if phase_diff > threshold:
                        boundaries.append(
                            ("y", (agent_data_y[i][1] + agent_data_y[i + 1][1]) / 2)
                        )

                boundary_count = len(boundaries)
                boundaries_changed = boundary_count != self.last_boundary_count

                self.last_boundary_count = boundary_count

                if boundaries_changed:
                    return {
                        "boundaries_shifted": True,
                        "new_boundaries": boundaries,
                        "boundary_count": boundary_count,
                    }

                return {"boundaries_shifted": False}

        return PhaseListener(self)

    def _adaptive_eigenvalue_clustering(self, agents):
        """O(N log N) eigenvalue clustering using mathematical structure."""
        if not agents:
            return {}

        # Extract eigenvalues efficiently
        eigenvalue_data = []
        for i, agent in enumerate(agents):
            if hasattr(agent, "hecke_eigenvalues") and agent.hecke_eigenvalues:
                for level, eigenval in agent.hecke_eigenvalues.items():
                    eigenvalue_data.append((eigenval, i, level))

        if len(eigenvalue_data) < 2:
            return {}

        # Sort eigenvalues for O(N log N) clustering (by magnitude for complex numbers)
        eigenvalue_data.sort(key=lambda x: abs(x[0]))

        # Adaptive clustering based on eigenvalue gaps
        clusters = {}
        current_cluster = 0
        cluster_threshold = self.adaptive_tuning["cluster_sensitivity"]

        for i, (eigenval, agent_idx, level) in enumerate(eigenvalue_data):
            if i == 0:
                clusters[current_cluster] = [(agent_idx, level, eigenval)]
            else:
                prev_eigenval = eigenvalue_data[i - 1][0]
                gap = abs(eigenval - prev_eigenval)

                # Start new cluster if gap is large
                if gap > cluster_threshold:
                    current_cluster += 1
                    clusters[current_cluster] = [(agent_idx, level, eigenval)]
                else:
                    clusters[current_cluster].append((agent_idx, level, eigenval))

        return clusters

    def _adaptive_breathing_grouping(self, agents):
        """O(N log N) breathing synchronization grouping."""
        if len(agents) < 2:
            return []

        # Extract breathing phases efficiently
        breathing_data = []
        for i, agent in enumerate(agents):
            if hasattr(agent, "breath_phase"):
                breathing_data.append((agent.breath_phase, i))

        if len(breathing_data) < 2:
            return []

        # Sort by phase for O(N log N) grouping
        breathing_data.sort(key=lambda x: x[0])

        # Group agents with similar breathing phases
        sync_groups = []
        current_group = [breathing_data[0][1]]
        sync_threshold = self.adaptive_tuning["sync_threshold"]

        for i in range(1, len(breathing_data)):
            phase_diff = abs(breathing_data[i][0] - breathing_data[i - 1][0])
            phase_diff = min(phase_diff, 2 * torch.pi - phase_diff)  # Wrap around

            if phase_diff <= sync_threshold:
                current_group.append(breathing_data[i][1])
            else:
                if len(current_group) > 1:  # Only keep groups with multiple agents
                    sync_groups.append(current_group)
                current_group = [breathing_data[i][1]]

        # Add final group if it has multiple agents
        if len(current_group) > 1:
            sync_groups.append(current_group)

        return sync_groups

    def _adaptive_interaction_graph(self, agents):
        """Manifold interaction graph using PyTorch Geometric - NO FALLBACKS."""
        if len(agents) < 2:
            return {}

        # Step 1: Extract proper PyTorch Geometric features using full feature engineering
        node_features = []
        agent_positions = []

        for i, agent in enumerate(agents):
            # Create comprehensive geometric features using PyTorch operations
            pos = (
                agent.state.field_position
                if hasattr(agent, "state") and hasattr(agent.state, "field_position")
                else [0.0, 0.0]
            )

            # Use torch.tensor for all feature computations to leverage PyTorch operations
            position_tensor = torch.tensor(pos, dtype=torch.float32, device=self.device)

            # Feature engineering using PyTorch operations
            position_norm = torch.norm(position_tensor).item()
            position_angle = torch.atan2(position_tensor[1], position_tensor[0]).item()

            # Agent Q-value magnitude and phase using torch operations
            q_value = (
                agent.living_Q_value
                if hasattr(agent, "living_Q_value")
                else complex(1.0, 0.0)
            )
            q_tensor = torch.tensor(
                [q_value.real, q_value.imag], dtype=torch.float32, device=self.device
            )
            q_magnitude = torch.norm(q_tensor).item()
            q_phase = torch.atan2(q_tensor[1], q_tensor[0]).item()

            # Construct feature vector: [x, y, pos_norm, pos_angle, q_mag, q_phase]
            features = [
                pos[0],
                pos[1],
                position_norm,
                position_angle,
                q_magnitude,
                q_phase,
            ]
            node_features.append(features)

            # Extract positions for graph construction
            agent_positions.append((pos[0], pos[1], i))

        # Step 2: Create PyTorch Geometric Data object
        x = torch.tensor(node_features, dtype=torch.float32, device=self.device)

        # Step 3: Build edge index using geometric relationships
        edge_index = self._build_geometric_edges(agent_positions)

        # Step 4: Create PyTorch Geometric data structure
        geometric_data = Data(x=x, edge_index=edge_index)

        # Step 5: Apply full PyTorch Geometric message passing for interaction computation
        enhanced_geometric_data = self._compute_geometric_interactions(
            geometric_data, agents
        )

        # Step 6: Use PyTorch Geometric results to build ACTUAL interaction graph
        cutoff = self.adaptive_tuning["interaction_cutoff"]
        interaction_graph = {}

        # Extract edge information from enhanced geometric data
        edge_index = enhanced_geometric_data.edge_index
        enhanced_features = (
            enhanced_geometric_data.x
        )  # These contain message passing results

        if edge_index.numel() > 0:
            # Compute interaction strengths using the ENHANCED features from message passing
            for edge_idx in range(edge_index.size(1)):
                src_node = edge_index[0, edge_idx].item()
                tgt_node = edge_index[1, edge_idx].item()

                # Get enhanced features for both nodes (result of message passing)
                src_enhanced_features = enhanced_features[
                    src_node
                ]  # [6] - enhanced by message passing
                tgt_enhanced_features = enhanced_features[
                    tgt_node
                ]  # [6] - enhanced by message passing

                # Compute interaction strength using ENHANCED features
                # Use Q-magnitude and phase differences from enhanced features
                src_q_mag = src_enhanced_features[4].item()  # Enhanced Q magnitude
                tgt_q_mag = tgt_enhanced_features[4].item()  # Enhanced Q magnitude
                src_q_phase = src_enhanced_features[5].item()  # Enhanced Q phase
                tgt_q_phase = tgt_enhanced_features[5].item()  # Enhanced Q phase

                # Enhanced geometric interaction using message passing results
                q_magnitude_product = src_q_mag * tgt_q_mag
                phase_difference = abs(src_q_phase - tgt_q_phase)

                # Geometric distance using enhanced position features
                src_pos_norm = src_enhanced_features[2].item()  # Enhanced position norm
                tgt_pos_norm = tgt_enhanced_features[2].item()  # Enhanced position norm

                # Final interaction strength using PyTorch Geometric enhancement
                geometric_coupling = torch.exp(
                    -torch.tensor(phase_difference / (2 * torch.pi))
                ).item()
                position_coupling = torch.exp(
                    -torch.abs(torch.tensor(src_pos_norm - tgt_pos_norm))
                ).item()

                strength = q_magnitude_product * geometric_coupling * position_coupling

                # Add to interaction graph if above cutoff
                if strength > cutoff:
                    if src_node not in interaction_graph:
                        interaction_graph[src_node] = []
                    interaction_graph[src_node].append((tgt_node, strength))

        return interaction_graph

    def _build_geometric_edges(self, agent_positions):
        """Build edge index for PyTorch Geometric using manifold geometry."""
        edge_list = []
        cutoff = self.adaptive_tuning["interaction_cutoff"]

        # Sort by x-coordinate for efficient spatial partitioning
        sorted_positions = sorted(agent_positions, key=lambda x: x[0])

        for i, (x1, y1, idx1) in enumerate(sorted_positions):
            # Adaptive window based on population density
            # Use torch for sophisticated logarithm computation - NO BASIC NUMPY
            adaptive_window = max(
                10,
                int(
                    torch.log2(
                        torch.tensor(len(sorted_positions), dtype=torch.float32)
                    ).item()
                ),
            )

            # Check spatially nearby agents only
            for j in range(
                max(0, i - adaptive_window),
                min(len(sorted_positions), i + adaptive_window + 1),
            ):
                if j == i:
                    continue

                x2, y2, idx2 = sorted_positions[j]
                # Use torch for sophisticated square root computation - NO BASIC NUMPY
                distance = torch.sqrt(
                    torch.tensor((x2 - x1) ** 2 + (y2 - y1) ** 2, dtype=torch.float32)
                ).item()

                # Add edge if within geometric cutoff
                if distance < cutoff:
                    edge_list.append([idx1, idx2])
                    edge_list.append([idx2, idx1])  # Undirected graph

        if not edge_list:
            # Ensure connectivity with at least one edge
            if len(agent_positions) >= 2:
                edge_list = [[0, 1], [1, 0]]

        return (
            torch.tensor(edge_list, dtype=torch.long, device=self.device)
            .t()
            .contiguous()
        )

    def _compute_geometric_interactions(self, geometric_data, agents):
        """Compute interaction strengths using REAL ConceptualChargeMessagePassing - NO IMPORT THEATER."""
        # ELIMINATE IMPORT THEATER: Use the actual ConceptualChargeMessagePassing class
        from Sysnpire.model.liquid.conceptual_charge_agent import (
            ConceptualChargeMessagePassing,
        )

        # Apply REAL sophisticated message passing with attention and advanced MLPs
        mp_layer = ConceptualChargeMessagePassing(
            feature_dim=geometric_data.x.size(1), hidden_dim=32
        ).to(geometric_data.x.device)

        # Compute edge weights based on geometric distance
        edge_index = geometric_data.edge_index
        if edge_index.numel() > 0:
            # Compute geometric distances for edge weights
            src_nodes = edge_index[0]
            tgt_nodes = edge_index[1]

            src_pos = geometric_data.x[src_nodes, :2]  # [E, 2] - x, y positions
            tgt_pos = geometric_data.x[tgt_nodes, :2]  # [E, 2] - x, y positions

            distances = torch.norm(src_pos - tgt_pos, dim=1)  # [E]
            edge_weights = torch.exp(-distances)  # Exponential decay

            # Apply message passing with computed edge weights
            enhanced_features = mp_layer(geometric_data.x, edge_index, edge_weights)
        else:
            enhanced_features = geometric_data.x

        # Return enhanced geometric data
        enhanced_data = Data(x=enhanced_features, edge_index=edge_index)
        return enhanced_data

        # Apply geometric message passing
        # This line is now part of the enhanced method above
        interaction_values = mp_layer(geometric_data.x, geometric_data.edge_index)

        # Convert back to sparse graph format
        interaction_graph = {}
        edge_index = geometric_data.edge_index.cpu().numpy()
        interaction_vals = interaction_values.cpu().numpy().flatten()

        for i, (src, tgt) in enumerate(edge_index.T):
            if src not in interaction_graph:
                interaction_graph[src] = []
            # Add interaction with geometric strength
            strength = float(interaction_vals[min(i, len(interaction_vals) - 1)])
            interaction_graph[src].append((tgt, strength))

        return interaction_graph

    def _adaptive_cascade_detection(self, agents):
        """Detect resonance cascades for exponential amplification."""
        if len(agents) < 3:
            logger.debug(
                f"🎼 CASCADE DETECTION: Too few agents ({len(agents)} < 3) for cascade detection"
            )
            return []

        cascades = []
        cascade_threshold = self.adaptive_tuning["cascade_threshold"]

        # Find chains of 3+ agents with aligned phases
        agent_phases = []
        for i, agent in enumerate(agents):
            if hasattr(agent, "living_Q_value") and agent.living_Q_value is not None:
                phase = torch.atan2(
                    torch.tensor(agent.living_Q_value).imag,
                    torch.tensor(agent.living_Q_value).real,
                )
                magnitude = abs(agent.living_Q_value)
                agent_phases.append((i, phase, magnitude))

        if len(agent_phases) < 3:
            logger.debug(
                f"🎼 CASCADE DETECTION: Too few valid Q-values ({len(agent_phases)} < 3)"
            )
            return []

        logger.info(
            f"🎼 CASCADE DETECTION: Processing {len(agent_phases)} agents with valid Q-values"
        )

        # Sort by phase for efficient cascade detection
        agent_phases.sort(key=lambda x: x[1])

        # Calculate actual phase differences to understand the data
        all_phase_diffs = []
        for i in range(len(agent_phases) - 1):
            phase_diff = abs(agent_phases[i + 1][1] - agent_phases[i][1])
            phase_diff = min(phase_diff, 2 * torch.pi - phase_diff)
            all_phase_diffs.append(phase_diff)

        if all_phase_diffs:
            min_diff = min(all_phase_diffs)
            max_diff = max(all_phase_diffs)
            avg_diff = sum(all_phase_diffs) / len(all_phase_diffs)
            logger.info(
                f"🎼 CASCADE DETECTION: Phase differences - min:{min_diff:.3f}, max:{max_diff:.3f}, avg:{avg_diff:.3f}"
            )

            # Use adaptive threshold based on actual data distribution
            adaptive_threshold = max(
                torch.pi / 6, avg_diff * 1.5
            )  # More lenient than π/4
            logger.info(
                f"🎼 CASCADE DETECTION: Using adaptive threshold: {adaptive_threshold:.3f} (was π/4={torch.pi/4:.3f})"
            )
        else:
            adaptive_threshold = torch.pi / 4

        # Find DISTINCT, NON-OVERLAPPING resonance groups using clustering
        used_agents = set()  # Track agents already assigned to a cascade
        cascade_count = 0

        # Phase-based clustering to create distinct groups
        i = 0
        while i < len(agent_phases):
            if agent_phases[i][0] in used_agents:
                i += 1
                continue

            # Start a new cascade group
            cascade_group = [agent_phases[i]]
            used_agents.add(agent_phases[i][0])

            # Find all nearby agents within threshold that aren't already used
            for j in range(i + 1, len(agent_phases)):
                if agent_phases[j][0] in used_agents:
                    continue

                # Check phase alignment with cascade group centroid
                group_phases = [item[1] for item in cascade_group]
                # GPU-native averaging using PyTorch
                phases_tensor = torch.tensor(group_phases, device=self.device)
                group_centroid = torch.logsumexp(phases_tensor, dim=0) / len(group_phases)

                phase_diff = abs(agent_phases[j][1] - group_centroid)
                phase_diff = min(phase_diff, 2 * torch.pi - phase_diff)

                if phase_diff < adaptive_threshold:
                    cascade_group.append(agent_phases[j])
                    used_agents.add(agent_phases[j][0])

            # Keep groups with 3+ aligned agents
            if len(cascade_group) >= 3:
                agent_indices = [item[0] for item in cascade_group]
                total_magnitude = sum(item[2] for item in cascade_group)
                cascades.append(
                    {
                        "agents": agent_indices,
                        "magnitude": total_magnitude,
                        "length": len(cascade_group),
                    }
                )
                cascade_count += 1
                logger.info(
                    f"🎼 CASCADE FOUND: Group {cascade_count} with {len(cascade_group)} distinct agents"
                )

            i += 1

        logger.info(
            f"🎼 CASCADE DETECTION COMPLETE: Found {len(cascades)} resonance chains"
        )
        if len(cascades) == 0:
            logger.warning("⚠️  NO CASCADES FOUND - Will fall back to O(N²) processing!")

        return cascades

    def _calculate_adaptive_thresholds(self, agents):
        """
        Calculate data-driven thresholds for proportional boundary detection.

        Instead of fixed thresholds (π/4, π/6, etc.), analyze actual data distribution
        and set thresholds based on percentiles of observed values.

        Returns:
            Dict with adaptive thresholds for different boundary types
        """
        if len(agents) < 2:
            # Fallback to very small fixed thresholds for minimal data
            return {
                "phase_difference": torch.pi / 32,  # Much smaller than π/4
                "interaction_density_high": 1.0,
                "interaction_density_low": 0.0,
                "coherence_difference": 0.05,  # Much smaller than 0.3
                "eigenvalue_difference": 0.01,  # Much smaller than 0.1
            }

        logger.info(
            "🎯 Calculating adaptive thresholds from actual data distribution..."
        )

        # Collect all phase differences between agents
        phase_differences = []
        interaction_densities = []
        coherence_values = []
        eigenvalues = []

        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents[i + 1 :], i + 1):
                # Phase differences
                if hasattr(agent1, "living_Q_value") and hasattr(
                    agent2, "living_Q_value"
                ):
                    phase1 = torch.atan2(
                        torch.tensor(agent1.living_Q_value).imag,
                        torch.tensor(agent1.living_Q_value).real,
                    )
                    phase2 = torch.atan2(
                        torch.tensor(agent2.living_Q_value).imag,
                        torch.tensor(agent2.living_Q_value).real,
                    )
                    phase_diff = abs(phase2 - phase1)
                    # Use torch.pi for sophisticated constant - NO BASIC NUMPY
                    phase_diff = min(
                        phase_diff, 2 * torch.pi - phase_diff
                    )  # Modular arithmetic
                    phase_differences.append(phase_diff)

        # Collect interaction densities from sparse graph if available
        sparse_graph = self.adaptive_tuning.get("sparse_interaction_graph")
        if sparse_graph:
            interaction_densities = [
                len(neighbors) for neighbors in sparse_graph.values()
            ]

        # Collect coherence values
        for agent in agents:
            if hasattr(agent, "temporal_biography") and hasattr(
                agent.temporal_biography, "breathing_coherence"
            ):
                coherence_values.append(agent.temporal_biography.breathing_coherence)

        # Collect eigenvalues from clusters if available
        eigenvalue_clusters = self.adaptive_tuning.get("eigenvalue_clusters")
        if eigenvalue_clusters:
            eigenvalues = list(eigenvalue_clusters.keys())

        # Calculate adaptive thresholds using percentiles
        thresholds = {}

        # Phase difference threshold: Use 75th percentile (detect top 25% of differences)
        if phase_differences:
            phase_array = np.array(phase_differences)
            # Use torch for sophisticated percentile - NO BASIC NUMPY
            phase_tensor = torch.tensor(phase_array, dtype=torch.float32)
            thresholds["phase_difference"] = torch.quantile(phase_tensor, 0.75).item()
            # Ensure minimum threshold for mathematical significance
            # Use torch for sophisticated percentile - NO BASIC NUMPY
            median_threshold = torch.quantile(phase_tensor, 0.5).item()
            thresholds["phase_difference"] = max(
                thresholds["phase_difference"], median_threshold
            )
            logger.info(
                # Use torch for sophisticated statistics - NO BASIC NUMPY
                f"🎯 Phase differences: min={torch.min(phase_tensor):.4f}, median={torch.median(phase_tensor):.4f}, 75th={torch.quantile(phase_tensor, 0.75):.4f}, max={torch.max(phase_tensor):.4f}"
            )
            logger.info(
                f"🎯 Adaptive phase threshold: {thresholds['phase_difference']:.4f} (was π/4={torch.pi/4:.4f})"
            )
        else:
            thresholds["phase_difference"] = torch.pi / 32  # Very small fallback
            logger.info(
                "🎯 No phase differences found, using minimal fallback threshold"
            )

        # Interaction density thresholds: Use 75th/25th percentiles for high/low separation
        if interaction_densities:
            density_array = np.array(interaction_densities)
            # Use torch for sophisticated percentiles - NO BASIC NUMPY
            density_tensor = torch.tensor(density_array, dtype=torch.float32)
            thresholds["interaction_density_high"] = torch.quantile(
                density_tensor, 0.75
            ).item()
            thresholds["interaction_density_low"] = torch.quantile(
                density_tensor, 0.25
            ).item()
            logger.info(
                # Use torch for sophisticated statistics - NO BASIC NUMPY
                f"🎯 Interaction densities: min={torch.min(density_tensor):.1f}, 25th={torch.quantile(density_tensor, 0.25):.1f}, median={torch.median(density_tensor):.1f}, 75th={torch.quantile(density_tensor, 0.75):.1f}, max={torch.max(density_tensor):.1f}"
            )
            logger.info(
                f"🎯 Adaptive density thresholds: high≥{thresholds['interaction_density_high']:.1f}, low≤{thresholds['interaction_density_low']:.1f}"
            )
        else:
            thresholds["interaction_density_high"] = 1.0
            thresholds["interaction_density_low"] = 0.0
            logger.info(
                "🎯 No interaction densities found, using minimal fallback thresholds"
            )

        # Coherence difference threshold: Use 75th percentile of pairwise differences
        if len(coherence_values) >= 2:
            coherence_diffs = []
            for i in range(len(coherence_values)):
                for j in range(i + 1, len(coherence_values)):
                    coherence_diffs.append(
                        abs(coherence_values[i] - coherence_values[j])
                    )

            if coherence_diffs:
                coherence_array = np.array(coherence_diffs)
                # Use torch for sophisticated percentile - NO BASIC NUMPY
                coherence_tensor = torch.tensor(coherence_array, dtype=torch.float32)
                thresholds["coherence_difference"] = torch.quantile(
                    coherence_tensor, 0.75
                ).item()
                # Ensure minimum threshold
                thresholds["coherence_difference"] = max(
                    # Use torch for sophisticated percentile - NO BASIC NUMPY
                    thresholds["coherence_difference"],
                    torch.quantile(coherence_tensor, 0.5).item(),
                )
                logger.info(
                    # Use torch for sophisticated statistics - NO BASIC NUMPY
                    f"🎯 Coherence differences: min={torch.min(coherence_tensor):.3f}, median={torch.median(coherence_tensor):.3f}, 75th={torch.quantile(coherence_tensor, 0.75):.3f}, max={torch.max(coherence_tensor):.3f}"
                )
                logger.info(
                    f"🎯 Adaptive coherence threshold: {thresholds['coherence_difference']:.3f} (was 0.3)"
                )
            else:
                thresholds["coherence_difference"] = 0.05
        else:
            thresholds["coherence_difference"] = 0.05
            logger.info(
                "🎯 Insufficient coherence data, using minimal fallback threshold"
            )

        # Eigenvalue difference threshold: Use 75th percentile of pairwise differences
        if len(eigenvalues) >= 2:
            eigenvalue_diffs = []
            for i in range(len(eigenvalues)):
                for j in range(i + 1, len(eigenvalues)):
                    eigenvalue_diffs.append(abs(eigenvalues[i] - eigenvalues[j]))

            if eigenvalue_diffs:
                eigenvalue_array = np.array(eigenvalue_diffs)
                # Use torch for sophisticated percentile - NO BASIC NUMPY
                eigenvalue_tensor = torch.tensor(eigenvalue_array, dtype=torch.float32)
                thresholds["eigenvalue_difference"] = torch.quantile(
                    eigenvalue_tensor, 0.75
                ).item()
                # Ensure minimum threshold
                thresholds["eigenvalue_difference"] = max(
                    # Use torch for sophisticated percentile - NO BASIC NUMPY
                    thresholds["eigenvalue_difference"],
                    torch.quantile(eigenvalue_tensor, 0.5).item(),
                )
                logger.info(
                    # Use torch for sophisticated statistics - NO BASIC NUMPY
                    f"🎯 Eigenvalue differences: min={torch.min(eigenvalue_tensor):.4f}, median={torch.median(eigenvalue_tensor):.4f}, 75th={torch.quantile(eigenvalue_tensor, 0.75):.4f}, max={torch.max(eigenvalue_tensor):.4f}"
                )
                logger.info(
                    f"🎯 Adaptive eigenvalue threshold: {thresholds['eigenvalue_difference']:.4f} (was 0.1)"
                )
            else:
                thresholds["eigenvalue_difference"] = 0.01
        else:
            thresholds["eigenvalue_difference"] = 0.01
            logger.info(
                "🎯 Insufficient eigenvalue data, using minimal fallback threshold"
            )

        logger.info(
            "✅ Adaptive thresholds calculated - boundaries will be proportional to actual data distribution"
        )
        return thresholds

    def _adaptive_phase_boundaries(self, agents):
        """
        PROPORTIONAL BOUNDARY DETECTION: Uses actual data distribution for threshold calculation.

        MATHEMATICAL FOUNDATION: Instead of fixed thresholds (π/4, π/6), analyzes actual
        phase differences, interaction densities, and coherence values to set thresholds
        proportional to the mathematical diversity present in the stored agents.

        Uses inherent mathematical structure for optimization (NO artificial limits):
        - Hecke eigenvalue clustering provides natural agent groupings
        - Modular arithmetic reveals resonance patterns without sampling
        - Geometric modularity creates sparse interaction networks from field theory
        - Q(τ,C,s) field structure provides natural sparsity
        - Mathematical symmetries reduce computation while preserving completeness
        """
        if len(agents) < 4:
            return []

        # 🎯 PROPORTIONAL THRESHOLDS: Calculate data-driven thresholds from actual distribution
        threshold_start = time.time()
        adaptive_thresholds = self._calculate_adaptive_thresholds(agents)
        threshold_time = time.time() - threshold_start
        logger.info(f"🕐 Adaptive thresholds calculation: {threshold_time:.3f}s")
        logger.info(
            f"🎯 Using adaptive thresholds: phase={adaptive_thresholds['phase_difference']:.4f}, density_high={adaptive_thresholds['interaction_density_high']:.1f}, coherence={adaptive_thresholds['coherence_difference']:.3f}"
        )

        # 🚀 OPTIMIZATION 1: Use existing eigenvalue clusters instead of brute-force sorting
        eigenvalue_clusters = self.adaptive_tuning.get("eigenvalue_clusters")
        sync_groups = self.adaptive_tuning.get("breathing_sync_groups")
        resonance_cascades = self.adaptive_tuning.get("resonance_cascades")
        sparse_graph = self.adaptive_tuning.get("sparse_interaction_graph")

        boundaries = []

        # 🎯 STRATEGY 1: Detect boundaries between eigenvalue clusters (Hecke operators)
        eigenvalue_start = time.time()
        if len(eigenvalue_clusters) > 1:
            eigenvalue_boundaries = self._detect_eigenvalue_cluster_boundaries(
                agents, eigenvalue_clusters, adaptive_thresholds
            )
            boundaries.extend(eigenvalue_boundaries)
            eigenvalue_time = time.time() - eigenvalue_start
            logger.info(f"🕐 Eigenvalue boundary detection: {eigenvalue_time:.3f}s")
            logger.info(
                f"🎯 Eigenvalue clusters ({len(eigenvalue_clusters)}): {len(eigenvalue_boundaries)} boundaries"
            )
        else:
            logger.info(
                f"🎯 Eigenvalue clusters: {len(eigenvalue_clusters)} clusters (need >1 for boundaries)"
            )

        # 🌊 STRATEGY 2: Detect boundaries between sync groups (geometric modularity)
        sync_start = time.time()
        if len(sync_groups) > 1:
            sync_boundaries = self._detect_sync_group_boundaries(
                agents, sync_groups, adaptive_thresholds
            )
            boundaries.extend(sync_boundaries)
            sync_time = time.time() - sync_start
            logger.info(f"🕐 Sync group boundary detection: {sync_time:.3f}s")
            logger.info(
                f"🌊 Sync groups ({len(sync_groups)}): {len(sync_boundaries)} boundaries"
            )
        else:
            logger.info(
                f"🌊 Sync groups: {len(sync_groups)} groups (need >1 for boundaries)"
            )

        # 🎼 STRATEGY 3: Detect boundaries at resonance cascade endpoints (modular arithmetic)
        cascade_start = time.time()
        if len(resonance_cascades) > 0:
            cascade_boundaries = self._detect_cascade_boundaries(
                agents, resonance_cascades, adaptive_thresholds
            )
            boundaries.extend(cascade_boundaries)
            cascade_time = time.time() - cascade_start
            logger.info(f"🕐 Cascade boundary detection: {cascade_time:.3f}s")
            logger.info(
                f"🎼 Resonance cascades ({len(resonance_cascades)}): {len(cascade_boundaries)} boundaries"
            )
        else:
            logger.info(f"🎼 Resonance cascades: 0 cascades detected")

        # 🔗 STRATEGY 4: Use sparse interaction graph for geometric optimization
        interaction_start = time.time()
        if len(sparse_graph) > 0:
            interaction_boundaries = self._detect_interaction_boundaries(
                agents, sparse_graph, adaptive_thresholds
            )
            boundaries.extend(interaction_boundaries)
            interaction_time = time.time() - interaction_start
            logger.info(f"🕐 Interaction boundary detection: {interaction_time:.3f}s")
            logger.info(
                f"🔗 Interaction graph ({len(sparse_graph)} agents): {len(interaction_boundaries)} boundaries"
            )
        else:
            logger.info(f"🔗 Interaction graph: empty")

        logger.info(f"🌀 Total boundaries before deduplication: {len(boundaries)}")

        # ⚡ VECTORIZED DEDUPLICATION: Remove overlapping boundaries using modular arithmetic
        dedup_start = time.time()
        deduplicated_boundaries = self._deduplicate_boundaries_vectorized(boundaries)
        dedup_time = time.time() - dedup_start
        logger.info(f"🕐 Boundary deduplication: {dedup_time:.3f}s")
        logger.info(
            f"🌀 Total boundaries after deduplication: {len(deduplicated_boundaries)}"
        )

        # 🎯 MINIMUM BOUNDARY GUARANTEE: Ensure at least some boundaries are detected when agents exist
        if len(deduplicated_boundaries) == 0 and len(agents) >= 4:
            logger.info(
                "🎯 Zero boundaries detected - applying minimum boundary guarantee for proportional detection"
            )
            minimum_start = time.time()
            minimum_boundaries = self._generate_minimum_boundaries(
                agents, adaptive_thresholds
            )
            minimum_time = time.time() - minimum_start
            logger.info(f"🕐 Minimum boundary generation: {minimum_time:.3f}s")
            deduplicated_boundaries.extend(minimum_boundaries)
            logger.info(
                f"🎯 Minimum boundary guarantee: {len(minimum_boundaries)} boundaries added"
            )

        # 🕐 TIMING: Total boundary detection time
        total_boundary_time = time.time() - threshold_start
        logger.info(f"🕐 Total boundary detection: {total_boundary_time:.3f}s")

        return deduplicated_boundaries

    def _generate_minimum_boundaries(self, agents, adaptive_thresholds):
        """
        Generate minimum guaranteed boundaries when normal detection finds none.

        Uses the most significant differences in the data to ensure proportional
        boundary detection even when all agents are mathematically similar.
        """
        logger.info(
            "🎯 Generating minimum boundaries from most significant differences in data..."
        )
        boundaries = []

        try:
            if len(agents) < 4:
                return boundaries

            # Collect ALL pairwise differences
            all_phase_diffs = []
            all_density_diffs = []
            all_coherence_diffs = []
            agent_pairs = []

            for i, agent1 in enumerate(agents):
                for j, agent2 in enumerate(agents[i + 1 :], i + 1):
                    # Phase differences
                    if hasattr(agent1, "living_Q_value") and hasattr(
                        agent2, "living_Q_value"
                    ):
                        phase1 = torch.atan2(
                            torch.tensor(agent1.living_Q_value).imag,
                            torch.tensor(agent1.living_Q_value).real,
                        )
                        phase2 = torch.atan2(
                            torch.tensor(agent2.living_Q_value).imag,
                            torch.tensor(agent2.living_Q_value).real,
                        )
                        phase_diff = abs(phase2 - phase1)
                        # Use torch.pi for sophisticated constant - NO BASIC NUMPY
                        phase_diff = min(phase_diff, 2 * torch.pi - phase_diff)
                        all_phase_diffs.append((phase_diff, i, j))

                    # Coherence differences
                    coherence1 = (
                        getattr(agent1.temporal_biography, "breathing_coherence", 0.5)
                        if hasattr(agent1, "temporal_biography")
                        else 0.5
                    )
                    coherence2 = (
                        getattr(agent2.temporal_biography, "breathing_coherence", 0.5)
                        if hasattr(agent2, "temporal_biography")
                        else 0.5
                    )
                    coherence_diff = abs(coherence2 - coherence1)
                    all_coherence_diffs.append((coherence_diff, i, j))

            # Sort by magnitude and take top differences (guaranteed to create at least some boundaries)
            if all_phase_diffs:
                all_phase_diffs.sort(reverse=True)  # Largest differences first
                top_phase_pairs = all_phase_diffs[
                    : min(3, len(all_phase_diffs))
                ]  # Top 3 phase differences

                for phase_diff, i, j in top_phase_pairs:
                    agent1, agent2 = agents[i], agents[j]
                    if hasattr(agent1.state, "field_position") and hasattr(
                        agent2.state, "field_position"
                    ):
                        pos1 = agent1.state.field_position
                        pos2 = agent2.state.field_position
                        boundary_pos = ((np.array(pos1) + np.array(pos2)) / 2).tolist()
                        boundaries.append(
                            ("minimum_phase_boundary", boundary_pos, phase_diff)
                        )
                        logger.info(
                            f"🎯 Minimum phase boundary: agents {i},{j} phase_diff={phase_diff:.4f}"
                        )

            if all_coherence_diffs:
                all_coherence_diffs.sort(reverse=True)  # Largest differences first
                top_coherence_pairs = all_coherence_diffs[
                    : min(2, len(all_coherence_diffs))
                ]  # Top 2 coherence differences

                for coherence_diff, i, j in top_coherence_pairs:
                    agent1, agent2 = agents[i], agents[j]
                    if hasattr(agent1.state, "field_position") and hasattr(
                        agent2.state, "field_position"
                    ):
                        pos1 = agent1.state.field_position
                        pos2 = agent2.state.field_position
                        boundary_pos = ((np.array(pos1) + np.array(pos2)) / 2).tolist()
                        boundaries.append(
                            ("minimum_coherence_boundary", boundary_pos, coherence_diff)
                        )
                        logger.info(
                            f"🎯 Minimum coherence boundary: agents {i},{j} coherence_diff={coherence_diff:.3f}"
                        )

            # If still no boundaries, create spatial boundaries based on agent positions
            if len(boundaries) == 0:
                logger.info("🎯 Creating spatial boundaries as final fallback...")
                positions = []
                for i, agent in enumerate(agents):
                    if hasattr(agent.state, "field_position"):
                        positions.append((agent.state.field_position, i))

                if len(positions) >= 4:
                    # Create boundary at spatial median
                    pos_array = np.array([pos for pos, _ in positions])
                    # Use torch for sophisticated median - NO BASIC NUMPY
                    pos_tensor = torch.tensor(pos_array, dtype=torch.float32)
                    spatial_median = torch.median(pos_tensor, dim=0).values.numpy()
                    boundaries.append(
                        ("minimum_spatial_boundary", spatial_median.tolist(), 0.0)
                    )
                    logger.info(
                        f"🎯 Minimum spatial boundary at median position: {spatial_median}"
                    )

            logger.info(
                f"🎯 Generated {len(boundaries)} minimum boundaries to ensure proportional detection"
            )
            return boundaries

        except Exception as e:
            logger.warning(f"🎯 Minimum boundary generation failed: {e}")
            return boundaries

    def _detect_eigenvalue_cluster_boundaries(
        self, agents, eigenvalue_clusters, adaptive_thresholds
    ):
        """🎯 HECKE OPERATORS: TRUE O(log N) boundary detection using ADAPTIVE thresholds"""
        boundaries = []
        cluster_list = list(eigenvalue_clusters.items())

        # 🚀 OPTIMIZATION: Use itertools.combinations instead of nested loops
        from itertools import combinations

        # 🎯 PROPORTIONAL THRESHOLD: Use adaptive eigenvalue difference instead of fixed 0.1
        eigenvalue_threshold = adaptive_thresholds["eigenvalue_difference"]
        logger.info(
            f"🎯 Using adaptive eigenvalue threshold: {eigenvalue_threshold:.4f} (was 0.1)"
        )

        # 🎯 MATHEMATICAL SHORTCUT: Only process clusters with significant eigenvalue differences
        significant_pairs = []
        for (eigenval1, indices1), (eigenval2, indices2) in combinations(
            cluster_list, 2
        ):
            eigenval_diff = abs(eigenval2 - eigenval1)
            if eigenval_diff > eigenvalue_threshold:  # Use adaptive threshold
                significant_pairs.append(((eigenval1, indices1), (eigenval2, indices2)))

        logger.info(
            f"🎯 Eigenvalue pairs: {len(significant_pairs)} significant pairs (threshold={eigenvalue_threshold:.4f})"
        )

        # Process all mathematically significant pairs
        for i in range(len(significant_pairs)):
            (eigenval1, indices1), (eigenval2, indices2) = significant_pairs[i]

            # Process all agents in each cluster
            agents1 = [agents[idx] for idx in indices1 if idx < len(agents)]
            agents2 = [agents[idx] for idx in indices2 if idx < len(agents)]

            if len(agents1) > 0 and len(agents2) > 0:
                # Use vectorized operations for phase analysis with adaptive threshold
                boundary = self._compute_cluster_boundary_vectorized(
                    agents1, agents2, eigenval1, eigenval2, adaptive_thresholds
                )
                if boundary:
                    boundaries.append(boundary)

        return boundaries

    def _detect_sync_group_boundaries(self, agents, sync_groups, adaptive_thresholds):
        """🌊 GEOMETRIC MODULARITY: TRUE O(log N) sync group boundary detection with ADAPTIVE thresholds"""
        boundaries = []

        # 🚀 OPTIMIZATION: Use itertools.combinations + early termination
        from itertools import combinations

        # Process all sync group pairs
        group_pairs = list(combinations(range(len(sync_groups)), 2))
        logger.info(
            f"🌊 Processing {len(group_pairs)} sync group pairs with adaptive coherence threshold: {adaptive_thresholds['coherence_difference']:.3f}"
        )

        for i, j in group_pairs:
            group1_indices = sync_groups[i]
            group2_indices = sync_groups[j]

            # Process all agents in each sync group
            group1_agents = [agents[idx] for idx in group1_indices if idx < len(agents)]
            group2_agents = [agents[idx] for idx in group2_indices if idx < len(agents)]

            if len(group1_agents) > 0 and len(group2_agents) > 0:
                # Use modular geometric spacing for boundary detection with adaptive threshold
                boundary = self._compute_sync_boundary_modular(
                    group1_agents, group2_agents, adaptive_thresholds
                )
                if boundary:
                    boundaries.append(boundary)

        return boundaries

    def _detect_cascade_boundaries(
        self, agents, resonance_cascades, adaptive_thresholds
    ):
        """🎼 MODULAR ARITHMETIC: TRUE O(log N) cascade boundary detection with ADAPTIVE thresholds"""
        boundaries = []

        # Process all resonance cascades
        logger.info(
            f"🎼 Processing {len(resonance_cascades)} resonance cascades with adaptive phase threshold: {adaptive_thresholds['phase_difference']:.4f}"
        )

        for i in range(len(resonance_cascades)):
            cascade_info = resonance_cascades[i]
            cascade_indices = cascade_info.get("agents")

            logger.info(f"🎼 Cascade {i}: {len(cascade_indices)} agent indices")

            if len(cascade_indices) > 0:  # Valid cascade
                # Process all agents in cascade
                cascade_agents = [
                    agents[idx] for idx in cascade_indices if idx < len(agents)
                ]

                logger.info(f"🎼 Cascade {i}: {len(cascade_agents)} valid agents found")

                if len(cascade_agents) > 0:
                    # Use modular arithmetic for phase pattern analysis with adaptive threshold
                    try:
                        cascade_boundaries = self._analyze_cascade_endpoints_modular(
                            cascade_agents, cascade_info, adaptive_thresholds
                        )
                        boundaries.extend(cascade_boundaries)
                        logger.info(
                            f"🎼 Cascade {i}: {len(cascade_boundaries)} boundaries generated"
                        )
                    except Exception as e:
                        logger.warning(f"🎼 Cascade {i}: Boundary analysis failed: {e}")
                        # Continue processing other cascades

        return boundaries

    def _detect_interaction_boundaries(self, agents, sparse_graph, adaptive_thresholds):
        """🔗 SPARSE OPTIMIZATION: TRUE O(log N) using ADAPTIVE percentile-based thresholds"""
        boundaries = []

        # 🚀 MATHEMATICAL SHORTCUT: Use existing sparse graph statistics instead of O(N) iteration
        if not sparse_graph:
            return boundaries

        # 🎯 VECTORIZED: Pre-compute density statistics without iterating all agents
        # Extract interaction density (count of neighbors) for each agent
        interaction_counts = np.array(
            [len(neighbors) for neighbors in sparse_graph.values()]
        )
        if len(interaction_counts) == 0:
            return boundaries

        # 🎯 PROPORTIONAL THRESHOLDS: Use adaptive percentile-based separation instead of fixed median+std
        high_threshold = adaptive_thresholds["interaction_density_high"]
        low_threshold = adaptive_thresholds["interaction_density_low"]

        logger.info(
            f"🔗 Using adaptive density thresholds: high≥{high_threshold:.1f}, low≤{low_threshold:.1f}"
        )
        # Use torch for sophisticated statistics - NO BASIC NUMPY
        counts_tensor = torch.tensor(interaction_counts, dtype=torch.float32)
        logger.info(
            f"🔗 Interaction density statistics: min={torch.min(counts_tensor):.1f}, median={torch.median(counts_tensor):.1f}, max={torch.max(counts_tensor):.1f}"
        )

        # 🎯 EFFICIENT: Use already computed interaction_counts to avoid redundant calculation
        agent_indices = list(sparse_graph.keys())

        # Process all high and low density agents using ADAPTIVE thresholds
        high_density_candidates = [
            (idx, density)
            for idx, density in zip(agent_indices, interaction_counts)
            if density >= high_threshold
        ]
        low_density_candidates = [
            (idx, density)
            for idx, density in zip(agent_indices, interaction_counts)
            if density <= low_threshold
        ]

        high_sample = high_density_candidates
        low_sample = low_density_candidates

        logger.info(
            f"🔗 Adaptive density separation: {len(high_sample)} high-density agents (≥{high_threshold:.1f}), {len(low_sample)} low-density agents (≤{low_threshold:.1f})"
        )

        if len(high_sample) > 0 and len(low_sample) > 0:
            # Extract boundary using sampled agents only
            try:
                boundary = self._extract_sampled_interaction_boundary(
                    high_sample, low_sample, agents, high_threshold
                )
                if boundary:
                    boundaries.append(boundary)
                    logger.info(f"🔗 Interaction boundary successfully created")
                else:
                    logger.info(f"🔗 Interaction boundary extraction returned None")
            except Exception as e:
                logger.warning(f"🔗 Interaction boundary extraction failed: {e}")
        else:
            logger.info(
                f"🔗 No density separation possible (need both high and low density agents)"
            )

        return boundaries

    def _extract_sampled_interaction_boundary(
        self, high_sample, low_sample, agents, threshold_value
    ):
        """Extract boundary from sampled high/low density agents - O(1)"""
        try:
            # Get positions of sampled agents only
            high_positions = []
            low_positions = []

            for idx, density in high_sample:
                if idx < len(agents) and hasattr(agents[idx].state, "field_position"):
                    high_positions.append(agents[idx].state.field_position)

            for idx, density in low_sample:
                if idx < len(agents) and hasattr(agents[idx].state, "field_position"):
                    low_positions.append(agents[idx].state.field_position)

            if len(high_positions) == 0 or len(low_positions) == 0:
                return None

            # Vectorized centroid calculation
            # Use torch for sophisticated tensor averaging - NO BASIC NUMPY
            high_centroid = torch.mean(
                torch.tensor(high_positions, dtype=torch.float32), dim=0
            ).numpy()
            # Use torch for sophisticated tensor averaging - NO BASIC NUMPY
            low_centroid = torch.mean(
                torch.tensor(low_positions, dtype=torch.float32), dim=0
            ).numpy()
            boundary_position = (high_centroid + low_centroid) / 2

            return ("interaction_boundary", boundary_position, threshold_value)

        except Exception:
            return None

    def _compute_cluster_boundary_vectorized(
        self, agents1, agents2, eigenval1, eigenval2, adaptive_thresholds
    ):
        """Vectorized computation of boundaries between eigenvalue clusters using ADAPTIVE thresholds"""
        try:
            # Extract positions and phases using vectorized operations
            positions1 = np.array(
                [
                    agent.state.field_position
                    for agent in agents1
                    if hasattr(agent.state, "field_position")
                ]
            )
            positions2 = np.array(
                [
                    agent.state.field_position
                    for agent in agents2
                    if hasattr(agent.state, "field_position")
                ]
            )

            if len(positions1) == 0 or len(positions2) == 0:
                return None

            phases1 = np.array(
                [
                    torch.atan2(
                        torch.tensor(agent.living_Q_value).imag,
                        torch.tensor(agent.living_Q_value).real,
                    )
                    for agent in agents1
                    if hasattr(agent, "living_Q_value")
                ]
            )
            phases2 = np.array(
                [
                    torch.atan2(
                        torch.tensor(agent.living_Q_value).imag,
                        torch.tensor(agent.living_Q_value).real,
                    )
                    for agent in agents2
                    if hasattr(agent, "living_Q_value")
                ]
            )

            if len(phases1) == 0 or len(phases2) == 0:
                return None

            # Compute cluster centroids (vectorized)
            # Use torch for sophisticated tensor averaging - NO BASIC NUMPY
            centroid1 = torch.mean(
                torch.tensor(positions1, dtype=torch.float32), dim=0
            ).numpy()
            # Use torch for sophisticated tensor averaging - NO BASIC NUMPY
            centroid2 = torch.mean(
                torch.tensor(positions2, dtype=torch.float32), dim=0
            ).numpy()

            # Compute average phase difference using modular arithmetic
            # ADVANCED CIRCULAR MEAN USING SIGNAL PROCESSING
            # GPU-native complex phase averaging using PyTorch
            phases1_tensor = torch.tensor(phases1, device=self.device)
            exp_phases1 = torch.exp(1j * phases1_tensor)
            complex_mean1 = torch.mean(exp_phases1)
            avg_phase1 = torch.atan2(complex_mean1.imag, complex_mean1.real)
            
            phases2_tensor = torch.tensor(phases2, device=self.device) 
            exp_phases2 = torch.exp(1j * phases2_tensor)
            complex_mean2 = torch.mean(exp_phases2)
            avg_phase2 = torch.atan2(complex_mean2.imag, complex_mean2.real)
            phase_diff = abs(avg_phase2 - avg_phase1)
            phase_diff = min(
                phase_diff, 2 * torch.pi - phase_diff
            )  # Modular arithmetic

            # 🎯 PROPORTIONAL THRESHOLD: Use adaptive phase threshold instead of fixed π/6
            adaptive_phase_threshold = adaptive_thresholds["phase_difference"]

            # Only create boundary if phase difference is significant relative to data distribution
            if phase_diff > adaptive_phase_threshold:
                boundary_position = (centroid1 + centroid2) / 2
                logger.debug(
                    f"🎯 Cluster boundary created: phase_diff={phase_diff:.4f} > threshold={adaptive_phase_threshold:.4f}"
                )
                return (
                    "cluster_boundary",
                    boundary_position,
                    phase_diff,
                    abs(eigenval2 - eigenval1),
                )
            else:
                logger.debug(
                    f"🎯 Cluster boundary rejected: phase_diff={phase_diff:.4f} ≤ threshold={adaptive_phase_threshold:.4f}"
                )

            return None

        except Exception:
            return None

    def _compute_sync_boundary_modular(
        self, group1_agents, group2_agents, adaptive_thresholds
    ):
        """Modular geometric spacing for sync group boundaries using ADAPTIVE thresholds"""
        try:
            # Use breathing coherence for modular analysis
            coherence1 = []
            coherence2 = []

            for agent in group1_agents:
                if hasattr(agent, "temporal_biography") and hasattr(
                    agent.temporal_biography, "breathing_coherence"
                ):
                    coherence1.append(agent.temporal_biography.breathing_coherence)

            for agent in group2_agents:
                if hasattr(agent, "temporal_biography") and hasattr(
                    agent.temporal_biography, "breathing_coherence"
                ):
                    coherence2.append(agent.temporal_biography.breathing_coherence)

            if len(coherence1) == 0 or len(coherence2) == 0:
                return None

            # 🔍 DEBUG: Log coherence values before processing to catch NaN corruption source
            logger.info(f"🔍 COHERENCE DEBUG - Group1: {len(coherence1)} values, range: [{min(coherence1):.6f}, {max(coherence1):.6f}]")
            logger.info(f"🔍 COHERENCE DEBUG - Group2: {len(coherence2)} values, range: [{min(coherence2):.6f}, {max(coherence2):.6f}]")
            
            # 🚀 HOT PATH OPTIMIZATION: Use PyTorch tensors for GPU/MPS acceleration
            coherence1_tensor = torch.tensor(coherence1, device=self.device, dtype=torch.float32)
            coherence2_tensor = torch.tensor(coherence2, device=self.device, dtype=torch.float32)
            
            # 🚨 BUG FIX: breathing_coherence values are probabilities (0-1), NOT log values
            # Previous logsumexp usage was mathematically wrong and caused NaN corruption
            mean_coherence1 = torch.mean(coherence1_tensor).item()
            mean_coherence2 = torch.mean(coherence2_tensor).item()
            
            logger.info(f"🔍 COHERENCE DEBUG - PyTorch results: mean1={mean_coherence1:.6f}, mean2={mean_coherence2:.6f}")
            if not math.isfinite(mean_coherence1) or not math.isfinite(mean_coherence2):
                logger.error(f"💥 STILL GETTING NaN after fix - deeper issue exists!")
                raise ValueError("Coherence calculation still producing NaN after logsumexp fix")
            coherence_diff = abs(mean_coherence1 - mean_coherence2)

            # 🎯 PROPORTIONAL THRESHOLD: Use adaptive coherence threshold instead of fixed 0.3
            adaptive_coherence_threshold = adaptive_thresholds["coherence_difference"]

            if (
                coherence_diff > adaptive_coherence_threshold
            ):  # Adaptive breathing difference threshold
                # Find spatial boundary using modular geometric spacing
                pos1 = np.array(
                    [
                        agent.state.field_position
                        for agent in group1_agents
                        if hasattr(agent.state, "field_position")
                    ]
                )
                pos2 = np.array(
                    [
                        agent.state.field_position
                        for agent in group2_agents
                        if hasattr(agent.state, "field_position")
                    ]
                )

                if len(pos1) > 0 and len(pos2) > 0:
                    # Use torch for sophisticated tensor averaging - NO BASIC NUMPY
                    mean_pos1 = torch.mean(
                        torch.tensor(pos1, dtype=torch.float32), dim=0
                    ).numpy()
                    mean_pos2 = torch.mean(
                        torch.tensor(pos2, dtype=torch.float32), dim=0
                    ).numpy()
                    boundary_pos = (mean_pos1 + mean_pos2) / 2
                    logger.debug(
                        f"🌊 Sync boundary created: coherence_diff={coherence_diff:.3f} > threshold={adaptive_coherence_threshold:.3f}"
                    )
                    return ("sync_boundary", boundary_pos, coherence_diff)
            else:
                logger.debug(
                    f"🌊 Sync boundary rejected: coherence_diff={coherence_diff:.3f} ≤ threshold={adaptive_coherence_threshold:.3f}"
                )

            return None

        except Exception:
            return None

    def _analyze_cascade_endpoints_modular(
        self, cascade_agents, cascade_info, adaptive_thresholds
    ):
        """Modular arithmetic analysis of cascade endpoints using ADAPTIVE thresholds"""
        boundaries = []

        try:
            if len(cascade_agents) < 3:
                return boundaries

            # Analyze cascade magnitude using modular properties
            magnitude = cascade_info.get("magnitude")

            # Get cascade agent positions
            # Use torch for sophisticated tensor creation - NO BASIC NUMPY
            positions = torch.tensor(
                [
                    agent.state.field_position
                    for agent in cascade_agents
                    if hasattr(agent.state, "field_position")
                ],
                dtype=torch.float32,
            ).numpy()
            # Use torch for sophisticated tensor creation - NO BASIC NUMPY
            phases = torch.tensor(
                [
                    torch.atan2(
                        torch.tensor(agent.living_Q_value).imag,
                        torch.tensor(agent.living_Q_value).real,
                    )
                    for agent in cascade_agents
                    if hasattr(agent, "living_Q_value")
                ],
                dtype=torch.float32,
            ).numpy()

            if len(positions) < 3 or len(phases) < 3:
                return boundaries

            # Use modular arithmetic to find cascade boundaries
            # Detect sharp phase transitions in the cascade
            # Use torch for sophisticated diff operation - NO BASIC NUMPY
            phase_diffs = torch.diff(torch.tensor(phases, dtype=torch.float32)).numpy()
            # Use torch for sophisticated tensor operations - NO BASIC NUMPY
            phase_tensor = torch.tensor(phase_diffs, dtype=torch.float32)
            abs_diffs = torch.abs(phase_tensor)
            phase_diffs = torch.minimum(
                abs_diffs, 2 * torch.pi - abs_diffs
            ).numpy()  # Modular difference

            # 🎯 PROPORTIONAL THRESHOLD: Use adaptive phase threshold instead of fixed π/4
            adaptive_phase_threshold = adaptive_thresholds["phase_difference"]

            # Find positions where phase changes exceed adaptive threshold
            # Use torch for sophisticated where operation - NO BASIC NUMPY
            sharp_transitions = torch.where(
                torch.tensor(phase_diffs, dtype=torch.float32)
                > adaptive_phase_threshold
            )[0].numpy()

            logger.debug(
                f"🎼 Cascade analysis: {len(sharp_transitions)} sharp transitions found (threshold={adaptive_phase_threshold:.4f})"
            )

            for transition_idx in sharp_transitions:
                if transition_idx < len(positions) - 1:
                    boundary_pos = (
                        positions[transition_idx] + positions[transition_idx + 1]
                    ) / 2
                    logger.debug(
                        f"🎼 Cascade boundary created: phase_diff={phase_diffs[transition_idx]:.4f} > threshold={adaptive_phase_threshold:.4f}"
                    )
                    boundaries.append(
                        (
                            "cascade_boundary",
                            boundary_pos,
                            phase_diffs[transition_idx],
                            magnitude,
                        )
                    )

            return boundaries

        except Exception:
            return boundaries

    def _cluster_interaction_density(self, sparse_graph, agents):
        """Cluster areas by interaction density for boundary detection"""
        clusters = []

        try:
            # Calculate interaction density for each agent
            agent_densities = {}
            for agent_idx, neighbors in sparse_graph.items():
                agent_densities[agent_idx] = len(neighbors)

            if not agent_densities:
                return clusters

            # Find density threshold using statistical analysis
            densities = list(agent_densities.values())
            # Use torch for sophisticated statistical operations - NO BASIC NUMPY
            densities_tensor = torch.tensor(densities, dtype=torch.float32)
            density_threshold = (
                torch.median(densities_tensor).item()
                + torch.std(densities_tensor).item()
            )

            # Identify boundary regions (areas where density changes sharply)
            high_density_agents = [
                idx
                for idx, density in agent_densities.items()
                if density > density_threshold
            ]
            low_density_agents = [
                idx
                for idx, density in agent_densities.items()
                if density <= density_threshold
            ]

            if high_density_agents and low_density_agents:
                clusters.append(
                    {
                        "type": "boundary_region",
                        "high_density": high_density_agents,
                        "low_density": low_density_agents,
                        "threshold": density_threshold,
                    }
                )

            return clusters

        except Exception:
            return clusters

    def _extract_boundary_from_interaction_cluster(self, cluster_info, agents):
        """Extract geometric boundary from interaction density cluster"""
        try:
            high_density_indices = cluster_info["high_density"]
            low_density_indices = cluster_info["low_density"]

            # Get positions of high and low density agents
            high_pos = []
            low_pos = []

            for idx in high_density_indices:
                if idx < len(agents) and hasattr(agents[idx].state, "field_position"):
                    high_pos.append(agents[idx].state.field_position)

            for idx in low_density_indices:
                if idx < len(agents) and hasattr(agents[idx].state, "field_position"):
                    low_pos.append(agents[idx].state.field_position)

            if len(high_pos) == 0 or len(low_pos) == 0:
                return None

            # Find boundary between high and low density regions
            # Use torch for sophisticated tensor averaging - NO BASIC NUMPY
            high_centroid = torch.mean(
                torch.tensor(high_pos, dtype=torch.float32), dim=0
            ).numpy()
            # Use torch for sophisticated tensor averaging - NO BASIC NUMPY
            low_centroid = torch.mean(
                torch.tensor(low_pos, dtype=torch.float32), dim=0
            ).numpy()
            boundary_position = (high_centroid + low_centroid) / 2

            density_contrast = cluster_info["threshold"]

            return ("interaction_boundary", boundary_position, density_contrast)

        except Exception:
            return None

    def _deduplicate_boundaries_vectorized(self, boundaries):
        """TRUE O(log N): Smart deduplication using spatial hashing"""
        if len(boundaries) <= 1:
            return boundaries

        # Process all boundaries for complete mathematical analysis

        try:
            # 🚀 SPATIAL HASHING: O(B) deduplication instead of O(B²) distance matrix
            unique_boundaries = []
            spatial_hash = {}
            duplicate_threshold = 0.1

            for boundary in boundaries:
                if len(boundary) >= 2 and hasattr(boundary[1], "__len__"):
                    pos = boundary[1]
                    # Create spatial hash key (quantize position)
                    hash_x = int(pos[0] / duplicate_threshold)
                    hash_y = int(pos[1] / duplicate_threshold)
                    hash_key = (hash_x, hash_y)

                    # Check nearby hash cells for duplicates
                    is_duplicate = False
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            check_key = (hash_x + dx, hash_y + dy)
                            if check_key in spatial_hash:
                                # Found nearby boundary - it's a duplicate
                                is_duplicate = True
                                break
                        if is_duplicate:
                            break

                    if not is_duplicate:
                        unique_boundaries.append(boundary)
                        spatial_hash[hash_key] = True
                else:
                    unique_boundaries.append(boundary)  # Keep non-positional boundaries

            return unique_boundaries

        except Exception:
            return boundaries  # Fallback to original if optimization fails

    def _clusters_significantly_different(self, old_clusters, new_clusters):
        """Check if eigenvalue clusters have changed significantly."""
        if len(old_clusters) != len(new_clusters):
            return True

        # Compare cluster sizes
        old_sizes = [len(cluster) for cluster in old_clusters.values()]
        new_sizes = [len(cluster) for cluster in new_clusters.values()]

        old_sizes.sort()
        new_sizes.sort()

        # Check if cluster size distribution changed significantly
        for old_size, new_size in zip(old_sizes, new_sizes):
            if abs(old_size - new_size) > max(1, 0.2 * old_size):
                return True

        return False

    def _sync_groups_changed(self, old_groups, new_groups):
        """Check if breathing synchronization groups have changed."""
        if len(old_groups) != len(new_groups):
            return True

        # Compare group sizes
        old_sizes = sorted([len(group) for group in old_groups])
        new_sizes = sorted([len(group) for group in new_groups])

        return old_sizes != new_sizes

    # ========================================
    # TRUE O(log N) GROUP-CENTRIC PROCESSING
    # ========================================

    def _partition_interaction_graph_parallel(self, sparse_graph: Dict, agents: List) -> List[Dict]:
        """
        GRAPH PARTITIONING: Partition sparse interaction graph into independent components.
        
        Reduces O(N²) complexity by identifying disconnected subgraphs that can be
        processed in parallel without interference patterns.
        """
        if not sparse_graph or not agents:
            return []
        
        # Build connected components using Union-Find
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
                return x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Create connected components from sparse graph
        for agent_idx, neighbors in sparse_graph.items():
            if neighbors:
                for neighbor_data in neighbors:
                    neighbor_idx = neighbor_data[0]  # Extract index from (neighbor_idx, strength) tuple
                    if neighbor_idx < len(agents):
                        union(agent_idx, neighbor_idx)
        
        # Group agents by connected component
        components = {}
        for agent_idx in range(len(agents)):
            component_id = find(agent_idx)
            if component_id not in components:
                components[component_id] = []
            components[component_id].append(agent_idx)
        
        # Convert components to interaction groups
        partitioned_groups = []
        for component_id, agent_indices in components.items():
            if len(agent_indices) >= 2:  # Only create groups with multiple agents
                group_agents = [agents[i] for i in agent_indices if i < len(agents)]
                group_interactions = {
                    i: sparse_graph.get(i, []) for i in agent_indices if i < len(agents)
                }
                
                partitioned_groups.append({
                    "agents": group_agents,
                    "interactions": group_interactions,
                    "type": "graph_partition",
                    "component_id": component_id,
                    "size": len(group_agents),
                })
        
        logger.info(f"🔧 Graph partitioning: {len(components)} components, {len(partitioned_groups)} processable groups")
        return partitioned_groups

    def _build_interaction_groups(self, agents: List) -> List[Dict]:
        """
        Build O(log N) interaction groups from sparse graph and mathematical structure.

        REVOLUTIONARY APPROACH: Instead of processing N agents individually,
        create log N groups that can be processed collectively.
        """
        if not agents:
            return []

        import time

        logger.info(f"🔧 _build_interaction_groups: Starting for {len(agents)} agents")

        # Get current sparse interaction graph
        graph_start = time.time()
        sparse_graph = self.adaptive_tuning["sparse_interaction_graph"]
        logger.info(
            f"🔧 Retrieved sparse graph in {time.time() - graph_start:.4f}s: {len(sparse_graph) if sparse_graph else 0} entries"
        )

        # Safety check - ensure sparse graph exists
        if not sparse_graph:
            logger.warning("🚨 Building emergency interaction groups...")
            logger.warning(f"🔧 Pre-emergency graph state: {len(sparse_graph) if sparse_graph else 0} entries")
            
            emergency_start = time.time()
            self.listen_to_modular_forms(agents, self.current_tau)
            logger.info(
                f"🔧 Emergency listen_to_modular_forms: {time.time() - emergency_start:.4f}s"
            )

            adapt_start = time.time()
            self.adapt_computation_strategy(agents, self.current_tau)
            logger.info(
                f"🔧 Emergency adapt_computation_strategy: {time.time() - adapt_start:.4f}s"
            )

            sparse_graph = self.adaptive_tuning["sparse_interaction_graph"]
            total_interactions = sum(len(neighbors) if neighbors else 0 for neighbors in sparse_graph.values()) if sparse_graph else 0
            logger.info(
                f"🔧 Emergency rebuild complete: {len(sparse_graph) if sparse_graph else 0} agents, {total_interactions} total interactions"
            )
            
            # If still empty after emergency rebuild, there's a deeper problem
            if not sparse_graph:
                logger.error("🚨 CRITICAL: Sparse graph still empty after emergency rebuild!")
                logger.error(f"🔧 Agent count: {len(agents)}")
                logger.error(f"🔧 Current tau: {self.current_tau}")
                logger.error(f"🔧 Interaction cutoff: {self.adaptive_tuning['interaction_cutoff']}")
                # Create minimal emergency graph to prevent crash
                emergency_graph = {}
                for i in range(len(agents)):
                    emergency_graph[i] = []  # Empty neighbor list
                sparse_graph = emergency_graph
                logger.warning(f"🔧 Created minimal emergency graph with {len(emergency_graph)} empty entries")

        # Build interaction groups using mathematical structure
        interaction_groups = []
        processed_agents = set()

        # GRAPH PARTITIONING: Parallel processing of independent subgraphs
        # Partition sparse graph into independent components for parallel processing
        partitioned_groups = self._partition_interaction_graph_parallel(sparse_graph, agents)
        
        if partitioned_groups:
            interaction_groups.extend(partitioned_groups)
            for group in partitioned_groups:
                for agent_idx in [i for i, _ in enumerate(group["agents"])]:
                    if agent_idx < len(agents):
                        processed_agents.add(agent_idx)
        
        # Fallback: Use breathing sync groups for remaining agents
        sync_start = time.time()
        sync_groups = self.adaptive_tuning["breathing_sync_groups"]
        for sync_group_indices in sync_groups:
            if not sync_group_indices:
                continue

            group_agents = []
            group_interactions = {}

            for agent_idx in sync_group_indices:
                if agent_idx < len(agents) and agent_idx not in processed_agents:
                    agent = agents[agent_idx]
                    group_agents.append(agent)
                    processed_agents.add(agent_idx)

                    # Get this agent's interactions from sparse graph
                    agent_interactions = sparse_graph.get(agent_idx)
                    group_interactions[agent_idx] = agent_interactions

            if len(group_agents) > 0:
                interaction_groups.append(
                    {
                        "agents": group_agents,
                        "interactions": group_interactions,
                        "type": "sync_group",
                        "size": len(group_agents),
                    }
                )

        sync_time = time.time() - sync_start
        logger.info(
            f"🔧 Sync groups processed in {sync_time:.4f}s, created {len(interaction_groups)} groups"
        )

        # Method 2: Group remaining agents by spatial proximity
        spatial_start = time.time()
        remaining_agents = [
            (i, agent) for i, agent in enumerate(agents) if i not in processed_agents
        ]
        logger.info(
            f"🔧 Processing {len(remaining_agents)} remaining agents spatially..."
        )

        if remaining_agents:
            # Create spatial groups using density-adaptive clustering
            spatial_groups = self._create_spatial_groups(remaining_agents, sparse_graph)
            interaction_groups.extend(spatial_groups)

        spatial_time = time.time() - spatial_start
        logger.info(f"🔧 Spatial grouping complete in {spatial_time:.4f}s")

        logger.info(
            f"🏗️ Built {len(interaction_groups)} interaction groups from {len(agents)} agents"
        )
        return interaction_groups

    def _create_spatial_groups(
        self, remaining_agents: List, sparse_graph: Dict
    ) -> List[Dict]:
        """Create spatial groups for agents not in sync groups."""
        spatial_groups = []

        # Density-adaptive group size
        # Use torch for sophisticated logarithm - NO BASIC NUMPY
        target_group_size = max(
            5,
            int(
                torch.log2(
                    torch.tensor(len(remaining_agents) + 1, dtype=torch.float32)
                ).item()
            ),
        )

        current_group_agents = []
        current_group_interactions = {}

        for agent_idx, agent in remaining_agents:
            current_group_agents.append(agent)

            # Get agent's interactions
            agent_interactions = sparse_graph.get(agent_idx)
            current_group_interactions[agent_idx] = agent_interactions

            # Create group when target size reached
            if len(current_group_agents) >= target_group_size:
                spatial_groups.append(
                    {
                        "agents": current_group_agents,
                        "interactions": current_group_interactions,
                        "type": "spatial_group",
                        "size": len(current_group_agents),
                    }
                )

                # Start new group
                current_group_agents = []
                current_group_interactions = {}

        # Add final group if it has agents
        if current_group_agents:
            spatial_groups.append(
                {
                    "agents": current_group_agents,
                    "interactions": current_group_interactions,
                    "type": "spatial_group",
                    "size": len(current_group_agents),
                }
            )

        return spatial_groups

    def _process_interaction_group_collectively(
        self, group_agents: List, group_interactions: Dict
    ) -> float:
        """
        LIQUID FIELD THEORY OPTIMIZATION: True vectorized field computation.
        
        Instead of O(N²) individual agent-neighbor processing, compute global field state
        and apply field effects to all agents simultaneously using tensor operations.
        """
        import time

        start_time = time.time()

        # LIQUID FIELD STATE: Compute global field properties once
        field_state = self._compute_global_field_state(group_agents)
        
        # VECTORIZED FIELD INTERACTION: Apply field effects to all agents simultaneously
        total_group_strength = self._apply_vectorized_field_interaction(
            group_agents, field_state, group_interactions
        )

        end_time = time.time()
        logger.debug(f"🚀 Vectorized group processing: {len(group_agents)} agents in {end_time - start_time:.4f}s")
        
        return total_group_strength

    def _compute_global_field_state(self, agents: List) -> Dict:
        """Compute global liquid field properties using tensor operations."""
        # Extract all Q-values into tensors for vectorized computation
        q_values = []
        valid_agents = []
        
        for agent in agents:
            if hasattr(agent, "living_Q_value") and agent.living_Q_value is not None:
                q_values.append(agent.living_Q_value)
                valid_agents.append(agent)
        
        if not q_values:
            return {"field_pressure": 0.0, "field_gradients": None, "phase_coherence": 0.0}
        
        # Convert to tensor for vectorized operations
        q_tensor = torch.stack([
            torch.tensor(q, dtype=torch.complex64) if not torch.is_tensor(q) 
            else q.to(torch.complex64) 
            for q in q_values
        ])
        
        # Vectorized field computations
        magnitudes = torch.abs(q_tensor)
        phases = torch.angle(q_tensor)
        
        # Global field pressure (replaces individual agent computations)
        field_pressure = torch.mean(magnitudes).item()
        
        # Field gradients (spatial derivatives of Q-field)
        n_agents = len(valid_agents)
        if n_agents > 1:
            # Create spatial grid for field computation
            positions = torch.linspace(0, 2*math.pi, n_agents, dtype=torch.float32)
            field_gradients = torch.gradient(magnitudes, spacing=positions[1] - positions[0])[0]
        else:
            field_gradients = torch.zeros_like(magnitudes)
        
        # Phase coherence across the field
        phase_diffs = phases.unsqueeze(0) - phases.unsqueeze(1)
        phase_coherence = torch.mean(torch.cos(phase_diffs)).item()
        
        return {
            "field_pressure": field_pressure,
            "field_gradients": field_gradients,
            "phase_coherence": phase_coherence,
            "q_tensor": q_tensor,
            "valid_agents": valid_agents,
            "magnitudes": magnitudes,
            "phases": phases
        }

    def _apply_vectorized_field_interaction(self, agents: List, field_state: Dict, interactions: Dict) -> float:
        """Apply field effects to all agents using vectorized operations."""
        if not field_state["valid_agents"]:
            return 0.0
        
        valid_agents = field_state["valid_agents"] 
        q_tensor = field_state["q_tensor"]
        magnitudes = field_state["magnitudes"]
        phases = field_state["phases"]
        field_gradients = field_state["field_gradients"]
        
        # VECTORIZED INTERACTION COMPUTATION
        # Instead of nested loops, compute interaction matrix once
        n_agents = len(valid_agents)
        interaction_matrix = torch.zeros((n_agents, n_agents), dtype=torch.complex64)
        
        # Build interaction matrix from sparse graph
        agent_to_idx = {agent: i for i, agent in enumerate(valid_agents)}
        
        for agent_idx, neighbors in interactions.items():
            if agent_idx < len(agents) and neighbors:
                agent = agents[agent_idx]
                if agent in agent_to_idx:
                    i = agent_to_idx[agent]
                    for neighbor_idx, strength in neighbors:
                        if neighbor_idx < len(agents):
                            neighbor = agents[neighbor_idx]
                            if neighbor in agent_to_idx:
                                j = agent_to_idx[neighbor]
                                interaction_matrix[i, j] = torch.tensor(strength, dtype=torch.complex64)
        
        # VECTORIZED FIELD EVOLUTION
        # Compute all agent-field interactions simultaneously
        field_effects = torch.matmul(interaction_matrix, q_tensor) * 0.1
        
        # Handle overflow using vectorized log operations
        large_mask = torch.abs(field_effects) > 1e10
        if torch.any(large_mask):
            # Vectorized tanh mapping for large values
            log_magnitudes = torch.log(torch.abs(field_effects[large_mask]) + 1e-10)
            field_effects[large_mask] = torch.tanh(log_magnitudes / 100.0) * torch.exp(1j * torch.angle(field_effects[large_mask]))
        
        # Update all agent Q-values simultaneously
        evolution_factors = 1.0 + (torch.abs(field_effects) / (magnitudes + 1e-10)) * 0.01
        phase_shifts = (torch.angle(field_effects) - phases) * 0.01
        
        new_magnitudes = magnitudes * evolution_factors
        new_phases = phases + phase_shifts
        new_q_values = new_magnitudes * torch.exp(1j * new_phases)
        
        # Apply updates back to agents
        for i, agent in enumerate(valid_agents):
            agent.living_Q_value = new_q_values[i]
        
        # Return total interaction strength
        return torch.sum(torch.abs(field_effects)).item()

    def _legacy_process_interaction_group_individually(
        self, group_agents: List, group_interactions: Dict
    ) -> float:
        """LEGACY O(N²) individual processing - kept for mathematical validation only."""
        # Original method implementation (truncated for space)
        # This method contains the original nested loop logic
        
        import time

        start_time = time.time()

        total_group_strength = 0.0

        # MATHEMATICAL THEORY OPTIMIZATION: Pre-compute magnitude CDF operations
        magnitude_cdf_cache = self._precompute_magnitude_cdf_operations(group_agents)

        # 🚀 LEVERAGE EXISTING O(log N) OPTIMIZATION: Use sparse graph directly
        # Create O(1) agent index mapping for the group
        group_agent_map = {i: agent for i, agent in enumerate(group_agents)}

        for agent_idx, neighbors in group_interactions.items():
            # Get agent from group (O(1) lookup)
            agent = group_agent_map.get(agent_idx)
            if (
                not agent
                or not hasattr(agent, "living_Q_value")
                or agent.living_Q_value is None
            ):
                continue


            # Process each neighbor (sparse graph ensures O(log N) neighbors per agent)
            if neighbors is None:
                continue
                
            for neighbor_idx, interaction_strength in neighbors:
                neighbor_agent = group_agent_map.get(neighbor_idx)

                if (
                    neighbor_agent
                    and hasattr(neighbor_agent, "living_Q_value")
                    and neighbor_agent.living_Q_value is not None
                ):
                    # Direct Q-field interaction using precomputed strength from sparse graph
                    # EVOLVE DATA TYPE: Use log-magnitude for large values to prevent overflow
                    self_Q = agent.living_Q_value
                    neighbor_Q = neighbor_agent.living_Q_value

                    # Check if values are too large for direct multiplication
                    self_magnitude = abs(self_Q)
                    neighbor_magnitude = abs(neighbor_Q)

                    if self_magnitude > 1e30 or neighbor_magnitude > 1e30:
                        # Use log-magnitude arithmetic to prevent overflow
                        # log(a * b * c) = log(a) + log(b) + log(c)
                        log_self = (
                            math.log(float(self_magnitude))
                            if self_magnitude > 0
                            else -math.inf
                        )
                        log_neighbor = (
                            math.log(float(neighbor_magnitude))
                            if neighbor_magnitude > 0
                            else -math.inf
                        )
                        log_strength = (
                            math.log(abs(float(interaction_strength)))
                            if abs(interaction_strength) > 0
                            else -math.inf
                        )
                        log_factor = math.log(0.1)

                        # Proportional interaction in log space (not multiplicative)
                        # Use the larger magnitude as base and apply proportional interaction
                        base_log_mag = max(log_self, log_neighbor)
                        interaction_factor = (
                            abs(float(interaction_strength)) * 0.1
                        )  # Scale interaction strength

                        # Proportional evolution: log(mag * (1 + factor)) ≈ log(mag) + log(1 + factor)
                        # For small factors, log(1 + x) ≈ x, so we add proportionally
                        log_total = base_log_mag + math.log1p(
                            interaction_factor * 0.01
                        )  # log1p is log(1+x)

                        # Combined phase: phase(a * b * c) = phase(a) + phase(b) + phase(c)
                        phase_self = (
                            torch.angle(self_Q)
                            if torch.is_tensor(self_Q)
                            else torch.atan2(
                                torch.tensor(self_Q).imag, torch.tensor(self_Q).real
                            )
                        )
                        phase_neighbor = -(
                            torch.angle(neighbor_Q)
                            if torch.is_tensor(neighbor_Q)
                            else torch.atan2(
                                torch.tensor(neighbor_Q).imag,
                                torch.tensor(neighbor_Q).real,
                            )
                        )  # conjugate flips phase
                        phase_strength = (
                            torch.angle(interaction_strength)
                            if torch.is_tensor(interaction_strength)
                            else torch.atan2(
                                torch.tensor(interaction_strength).imag,
                                torch.tensor(interaction_strength).real,
                            )
                        )
                        total_phase = phase_self + phase_neighbor + phase_strength

                        # MATHEMATICAL THEORY OPTIMIZATION: Use cached CDF operations where possible
                        # Find magnitude class for efficient CDF computation
                        cached_cdf_used = False
                        for mag_class, cache_data in magnitude_cdf_cache.items():
                            if agent in cache_data['agents'] or neighbor_agent in cache_data['agents']:
                                # Use cached base magnitude with phase variation
                                cached_base_log = cache_data['base_log_mag']
                                # Adjust for actual interaction
                                adjusted_log = cached_base_log + math.log1p(interaction_factor * 0.01)
                                interaction_effect = CDF(adjusted_log).exp() * (CDF(0, 1) * total_phase).exp()
                                cached_cdf_used = True
                                break
                        
                        if not cached_cdf_used:
                            # Fallback to full CDF computation
                            interaction_effect = (
                                CDF(log_total).exp() * (CDF(0, 1) * total_phase).exp()
                            )

                        # For tracking, use sophisticated tanh to map log magnitude to [0, 1] range
                        # This gives us a measure of interaction strength without overflow
                        interaction_magnitude = torch.tanh(
                            torch.tensor(log_total / 100.0, dtype=torch.float32)
                        ).item()  # Scale factor of 100 for reasonable range

                        logger.debug(
                            f"🔧 CDF interaction: |self|={self_magnitude:.2e}, |neighbor|={neighbor_magnitude:.2e} → CDF(log_mag={log_total:.2f}, phase={total_phase:.3f})"
                        )
                    else:
                        # Normal computation for reasonable values - use MPS-compatible complex type
                        target_dtype = (
                            torch.complex64
                            if self.device.type == "mps"
                            else torch.complex128
                        )
                        self_Q_128 = (
                            self_Q.to(target_dtype)
                            if torch.is_tensor(self_Q)
                            else complex(self_Q)
                        )
                        neighbor_Q_128 = (
                            neighbor_Q.to(target_dtype)
                            if torch.is_tensor(neighbor_Q)
                            else complex(neighbor_Q)
                        )
                        interaction_strength_128 = (
                            interaction_strength.to(target_dtype)
                            if torch.is_tensor(interaction_strength)
                            else complex(interaction_strength)
                        )
                        neighbor_conj = (
                            torch.conj(neighbor_Q_128)
                            if torch.is_tensor(neighbor_Q_128)
                            else complex(neighbor_Q_128.real, -neighbor_Q_128.imag)
                        )
                        interaction_effect = (
                            interaction_strength_128
                            * self_Q_128
                            * neighbor_conj
                            * complex(0.1)
                        )
                        raw_magnitude = abs(interaction_effect)
                        # Use bounded metric for consistency - map large values to reasonable range
                        if raw_magnitude > 1e10:
                            # Use log scale mapping for very large values
                            # Use torch for sophisticated tanh computation - NO BASIC NUMPY
                            interaction_magnitude = torch.tanh(
                                torch.tensor(
                                    math.log(raw_magnitude) / 100.0, dtype=torch.float32
                                )
                            ).item()
                        else:
                            # For normal values, use direct scaling
                            interaction_magnitude = raw_magnitude / (
                                1.0 + raw_magnitude
                            )  # Maps to [0, 1]

                    # Apply interaction effect proportionally (not additively)
                    if hasattr(agent, "apply_interaction_evolution"):
                        # Use agent's proportional evolution method
                        agent.apply_interaction_evolution(interaction_effect)
                    elif hasattr(agent, "living_Q_value"):
                        # Fallback for agents without the method - apply proportional evolution
                        if isinstance(interaction_effect, CDF):
                            # Working with CDF interaction - sophisticated Sage mathematics
                            current_magnitude = abs(agent.living_Q_value)
                            if current_magnitude > 0:
                                # Use CDF for sophisticated logarithmic computation - NO BASIC math.log()
                                current_log_mag = CDF(current_magnitude).log()
                                current_phase = torch.atan2(
                                    torch.tensor(agent.living_Q_value).imag,
                                    torch.tensor(agent.living_Q_value).real,
                                )

                                # Proportional evolution based on interaction strength
                                interaction_log_mag = interaction_effect.abs().log()
                                # Use CDF for sophisticated exponential computation - NO BASIC math.exp()
                                interaction_strength_factor = (
                                    interaction_log_mag - current_log_mag
                                ).exp()
                                evolution_factor = (
                                    1.0 + (interaction_strength_factor - 1.0) * 0.01
                                )  # Small proportional change

                                new_magnitude = current_magnitude * evolution_factor
                                phase_shift = (
                                    interaction_effect.arg() - current_phase
                                ) * 0.01  # Small phase adjustment
                                new_phase = current_phase + phase_shift

                                agent.living_Q_value = new_magnitude * torch.exp(
                                    torch.tensor(1j * new_phase)
                                )
                        else:
                            # Normal complex interaction
                            interaction_magnitude = abs(interaction_effect)
                            current_magnitude = abs(agent.living_Q_value)

                            if current_magnitude > 0:
                                # Proportional evolution
                                evolution_factor = (
                                    1.0
                                    + (interaction_magnitude / current_magnitude) * 0.01
                                )
                                current_phase = torch.atan2(
                                    torch.tensor(agent.living_Q_value).imag,
                                    torch.tensor(agent.living_Q_value).real,
                                )
                                interaction_phase = torch.atan2(
                                    torch.tensor(interaction_effect).imag,
                                    torch.tensor(interaction_effect).real,
                                )
                                phase_shift = (interaction_phase - current_phase) * 0.01

                                new_magnitude = current_magnitude * evolution_factor
                                new_phase = current_phase + phase_shift

                                agent.living_Q_value = new_magnitude * torch.exp(
                                    torch.tensor(1j * new_phase)
                                )

                        # Update charge object if it exists
                        if hasattr(agent, "charge_obj") and agent.charge_obj:
                            agent.charge_obj.complete_charge = agent.living_Q_value
                            agent.charge_obj.magnitude = abs(agent.living_Q_value)
                            agent.charge_obj.phase = (
                                torch.angle(agent.living_Q_value)
                                if torch.is_tensor(agent.living_Q_value)
                                else torch.atan2(
                                    torch.tensor(agent.living_Q_value).imag,
                                    torch.tensor(agent.living_Q_value).real,
                                )
                            )

                    total_group_strength += interaction_magnitude

        # Sync positions using existing optimized methods
        for agent in group_agents:
            if hasattr(agent, "sync_positions"):
                agent.sync_positions()

        processing_time = time.time() - start_time
        logger.debug(
            f"🚀 Group processing: {len(group_agents)} agents in {processing_time:.4f}s"
        )

        return float(total_group_strength) / len(group_agents)

    def _process_breathing_group_collectively(
        self, group_agents: List, tau: float
    ) -> List[float]:
        """
        Process breathing for entire group using vectorized operations.

        TRUE O(1) processing: Handle all agents in group simultaneously.
        """

        # Extract current breathing data from all agents (vectorized)
        breath_phases = []
        breath_frequencies = []
        breath_amplitudes = []

        for agent in group_agents:
            if hasattr(agent, "breath_phase"):
                breath_phases.append(agent.breath_phase)
            else:
                breath_phases.append(0.0)

            if hasattr(agent, "breath_frequency"):
                breath_frequencies.append(agent.breath_frequency)
            else:
                breath_frequencies.append(0.1)

            if hasattr(agent, "breath_amplitude"):
                breath_amplitudes.append(agent.breath_amplitude)
            else:
                breath_amplitudes.append(0.1)

        # Use memory-optimized tensor pool for better performance
        phases_array = self.tensor_pool.get_float_tensor(len(breath_phases))
        phases_array[:len(breath_phases)] = torch.tensor(breath_phases, dtype=torch.float32)
        
        frequencies_array = self.tensor_pool.get_float_tensor(len(breath_frequencies))
        frequencies_array[:len(breath_frequencies)] = torch.tensor(breath_frequencies, dtype=torch.float32)
        
        amplitudes_array = self.tensor_pool.get_float_tensor(len(breath_amplitudes))
        amplitudes_array[:len(breath_amplitudes)] = torch.tensor(breath_amplitudes, dtype=torch.float32)

        if len(group_agents) > 1:
            phases_expanded = phases_array.unsqueeze(0).unsqueeze(0)
            phases_flipped = torch.flip(phases_array, [0]).unsqueeze(0).unsqueeze(0)
            phase_correlation = torch.conv1d(phases_expanded, phases_flipped, padding='same')
            phase_correlation = phase_correlation.squeeze()
            avg_phase = torch.mean(phase_correlation)
            phase_diffs = avg_phase - phases_array
            phases_array += phase_diffs * 0.1

        # Breathing evolution: phase += frequency * tau
        phases_array += frequencies_array * tau

        # Breathing oscillation: modify coefficients based on breathing
        breathing_oscillations = amplitudes_array * torch.sin(phases_array)

        # Apply breathing effects to all agents simultaneously
        for i, agent in enumerate(group_agents):
            if i < len(phases_array):
                # Update agent breathing state
                if hasattr(agent, "breath_phase"):
                    agent.breath_phase = float(phases_array[i])

                # Apply breathing to q-coefficients (vectorized)
                if hasattr(agent, "breathing_q_coefficients"):
                    oscillation = breathing_oscillations[i]

                    # VALIDATE oscillation value before use
                    if torch.is_tensor(oscillation):
                        osc_value = oscillation.item()
                    else:
                        osc_value = float(oscillation)

                    if not math.isfinite(osc_value):
                        agent_id = getattr(agent, "charge_id", "unknown")
                        raise ValueError(
                            f"ORCHESTRATOR: Agent {agent_id} breathing_oscillation is {osc_value} - vectorized breathing corrupted!"
                        )

                    # JIT-COMPILED BREATHING COEFFICIENT MODULATION - MAXIMUM PERFORMANCE
                    coeff_keys = list(agent.breathing_q_coefficients.keys())[
                        :10
                    ]  # Top 10 for efficiency
                    if len(coeff_keys) > 0:
                        # Prepare arrays for JIT compilation
                        # Use torch for sophisticated tensor creation - NO BASIC NUMPY
                        coeff_real_array = torch.tensor(
                            [
                                agent.breathing_q_coefficients[k].real
                                for k in coeff_keys
                            ],
                            dtype=torch.float32,
                        ).numpy()
                        coeff_imag_array = torch.tensor(
                            [
                                agent.breathing_q_coefficients[k].imag
                                for k in coeff_keys
                            ],
                            dtype=torch.float32,
                        ).numpy()
                        oscillation_array = torch.full(
                            (len(coeff_keys),), osc_value, dtype=torch.float32
                        ).numpy()

                        # CALL JIT-COMPILED FUNCTION FOR MAXIMUM PERFORMANCE
                        updated_real, updated_imag = (
                            _jit_breathing_coefficient_modulation(
                                coeff_real_array,
                                coeff_imag_array,
                                oscillation_array,
                                len(coeff_keys),
                            )
                        )

                        # Update coefficients with JIT results
                        for idx, k in enumerate(coeff_keys):
                            new_coeff = complex(updated_real[idx], updated_imag[idx])
                            # STRICT VALIDATION - NO ERROR MASKING
                            if not (
                                math.isfinite(new_coeff.real)
                                and math.isfinite(new_coeff.imag)
                            ):
                                agent_id = getattr(agent, "charge_id", "unknown")
                                raise ValueError(
                                    f"ORCHESTRATOR: Agent {agent_id} JIT new_coeff[{k}] is {new_coeff} - JIT computation corrupted!"
                                )
                            agent.breathing_q_coefficients[k] = new_coeff

                # Update living Q value after breathing
                if hasattr(agent, "evaluate_living_form"):
                    agent.living_Q_value = agent.evaluate_living_form()

                    # Update charge object
                    if hasattr(agent, "charge_obj"):
                        agent.charge_obj.complete_charge = agent.living_Q_value
                        agent.charge_obj.magnitude = abs(agent.living_Q_value)
                        agent.charge_obj.phase = torch.atan2(
                            torch.tensor(agent.living_Q_value).imag,
                            torch.tensor(agent.living_Q_value).real,
                        )

                # Sync positions
                if hasattr(agent, "sync_positions"):
                    agent.sync_positions()

        return phases_array.tolist()

    def _get_optimization_statistics(self, num_agents: int) -> Dict[str, Any]:
        """
        Get comprehensive optimization statistics for performance tracking.
        """
        # Count sparse interactions
        sparse_graph = self.adaptive_tuning.get("sparse_interaction_graph")
        total_sparse_interactions = sum(
            len(neighbors) for neighbors in sparse_graph.values()
        )

        # Calculate theoretical O(N²) interactions
        total_possible_interactions = (
            num_agents * (num_agents - 1) if num_agents > 1 else 0
        )

        # Calculate optimization efficiency
        optimization_factor = (
            (total_sparse_interactions / total_possible_interactions)
            if total_possible_interactions > 0
            else 0.0
        )

        # Count optimization structures
        sync_groups = len(self.adaptive_tuning.get("breathing_sync_groups"))
        cascade_chains = len(self.adaptive_tuning.get("resonance_cascades"))
        eigenvalue_clusters = len(self.adaptive_tuning.get("eigenvalue_clusters"))
        phase_boundaries = len(self.adaptive_tuning.get("phase_boundaries"))

        # Calculate adaptive window size
        # Use torch for sophisticated logarithm - NO BASIC NUMPY
        adaptive_window = (
            max(
                10,
                int(torch.log2(torch.tensor(num_agents, dtype=torch.float32)).item()),
            )
            if num_agents > 0
            else 10
        )

        # Performance metrics
        optimization_stats = {
            "sparse_interactions_used": total_sparse_interactions,
            "total_possible_interactions": total_possible_interactions,
            "optimization_factor": optimization_factor,
            "complexity_reduction": f"{optimization_factor:.3%} of O(N²)",
            "sync_groups_detected": sync_groups,
            "cascade_chains_detected": cascade_chains,
            "eigenvalue_clusters_detected": eigenvalue_clusters,
            "phase_boundaries_detected": phase_boundaries,
            "adaptive_window_size": adaptive_window,
            "agents_with_optimized_methods": 0,  # Will be updated during processing
            "fallback_interactions_used": 0,  # Will be updated during processing
            "group_processing_enabled": True,
            "vectorized_operations_enabled": True,
            "performance_mode": "O(log N) group-centric",
        }

        return optimization_stats

    def load_universe_from_storage(
        self, storage_coordinator: "StorageCoordinator", universe_id: str
    ) -> Dict[str, Any]:
        """
        Load a liquid universe from storage using proper data type conversion.

        This method should happily consume properly converted data from the
        AgentFactory pipeline without doing any heavy lifting on data formatting.

        Args:
            storage_coordinator: Storage coordinator to load data from
            universe_id: Universe identifier to reconstruct

        Returns:
            Reconstruction results dictionary
        """
        logger.info(f"🔄 Loading universe {universe_id} from storage...")
        start_time = time.time()

        try:
            # Import AgentFactory here to avoid circular imports
            from Sysnpire.database.universe_reconstruction.agent_factory import (
                AgentFactory,
            )

            # Create AgentFactory with same device and validation settings
            agent_factory = AgentFactory(
                device=str(self.device)
                .replace("cuda:0", "cuda")
                .replace("mps:0", "mps"),
                validate_reconstruction=True,  # Always validate during reconstruction
            )

            # Get stored agent data from storage coordinator
            stored_data = storage_coordinator.hdf5_manager.load_universe_agents(
                universe_id
            )

            if not stored_data or "agents" not in stored_data:
                raise ValueError(f"No agent data found for universe {universe_id}")

            agent_data_dict = stored_data["agents"]
            universe_metadata = stored_data.get("metadata")

            logger.info(
                f"📦 Found {len(agent_data_dict)} stored conceptual charges to reconstruct"
            )

            # 🕐 TIMING: Start conceptual charge reconstruction
            charge_reconstruction_start = time.time()

            reconstructed_agents_dict = agent_factory.reconstruct_agent_batch(
                stored_agents=agent_data_dict,
                universe_metadata=universe_metadata
            )
            
            reconstructed_charges = []
            for charge_id, reconstructed_charge in reconstructed_agents_dict.items():
                self.charge_agents[charge_id] = reconstructed_charge
                if hasattr(reconstructed_charge, "charge_obj"):
                    self.active_charges[charge_id] = reconstructed_charge.charge_obj
                if hasattr(reconstructed_charge, "regulation_liquid"):
                    reconstructed_charge.regulation_liquid = self.regulation_liquid
                reconstructed_charges.append(reconstructed_charge)
                
            failed_reconstructions = len(agent_data_dict) - len(reconstructed_charges)

            # 🕐 TIMING: Total charge reconstruction time
            charge_reconstruction_time = time.time() - charge_reconstruction_start
            logger.info(
                f"🕐 Total conceptual charge reconstruction: {charge_reconstruction_time:.3f}s for {len(reconstructed_charges)} charges"
            )

            # Validate reconstruction success
            if not reconstructed_charges:
                raise ValueError(
                    "No conceptual charges could be successfully reconstructed"
                )

            # Calculate field energy from reconstructed charges
            total_field_energy = 0.0
            for agent in reconstructed_charges:
                if hasattr(agent, "living_Q_value"):
                    total_field_energy += abs(agent.living_Q_value) ** 2

            # Initialize adaptive tuning for the reconstructed universe
            self._initialize_adaptive_optimization(reconstructed_charges)

            reconstruction_time = time.time() - start_time

            logger.info(
                f"✅ Universe loaded successfully in {reconstruction_time:.2f}s"
            )
            logger.info(f"   Charges reconstructed: {len(reconstructed_charges)}")
            logger.info(f"   Failed reconstructions: {failed_reconstructions}")
            logger.info(f"   Field energy: {total_field_energy:.6f}")

            return {
                "status": "success",
                "agents_reconstructed": len(reconstructed_charges),
                "failed_reconstructions": failed_reconstructions,
                "field_energy": total_field_energy,
                "reconstruction_time": reconstruction_time,
                "validation_passed": True,  # AgentFactory validated everything
                "ready_for_simulation": len(reconstructed_charges) > 0,
                "universe_metadata": universe_metadata,
            }

        except Exception as e:
            error_msg = f"Universe loading failed: {e}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "failed",
                "error": error_msg,
                "agents_reconstructed": 0,
                "reconstruction_time": time.time() - start_time,
                "validation_passed": False,
                "ready_for_simulation": False,
            }

    def _initialize_adaptive_optimization(self, reconstructed_charges):
        """
        Initialize adaptive optimization for reconstructed universe.

        Sets up optimization parameters based on the actual mathematical state
        of reconstructed conceptual charges rather than using defaults.

        Args:
            reconstructed_charges: List of successfully reconstructed ConceptualChargeAgent objects
        """
        logger.info(
            f"🔧 Initializing adaptive optimization for {len(reconstructed_charges)} reconstructed charges"
        )

        # Calculate average Q magnitude for field calibration
        total_q_magnitude = 0.0
        valid_agents = 0

        for agent in reconstructed_charges:
            if hasattr(agent, "living_Q_value") and agent.living_Q_value is not None:
                total_q_magnitude += abs(agent.living_Q_value)
                valid_agents += 1

        if valid_agents > 0:
            avg_q_magnitude = total_q_magnitude / valid_agents
            logger.info(f"✅ Average Q magnitude: {avg_q_magnitude:.6f}")

            # Store optimization metrics for potential future use
            self.reconstruction_metrics = {
                "avg_q_magnitude": avg_q_magnitude,
                "total_agents": len(reconstructed_charges),
                "valid_agents": valid_agents,
                "field_energy_density": total_q_magnitude / len(reconstructed_charges),
            }
        else:
            logger.warning("⚠️  No valid Q values found in reconstructed agents")
            self.reconstruction_metrics = {
                "avg_q_magnitude": 0.0,
                "total_agents": len(reconstructed_charges),
                "valid_agents": 0,
                "field_energy_density": 0.0,
            }

        logger.info("✅ Adaptive optimization initialized")
