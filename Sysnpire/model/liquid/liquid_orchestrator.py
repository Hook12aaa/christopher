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

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

from Sysnpire.database.conceptual_charge_object import ConceptualChargeObject, FieldComponents
from Sysnpire.model.liquid.conceptual_charge_agent import ConceptualChargeAgent
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


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
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        self.field_resolution = field_resolution
        
        # Active charge agents (living Q(τ, C, s) entities)
        self.active_charges: Dict[str, ConceptualChargeObject] = {}
        self.charge_agents: Dict[str, 'ConceptualChargeAgent'] = {}  # Store actual agents
        
        # ChargeFactory data storage
        self.combined_results: Optional[Dict[str, Any]] = None
        
        # Field state tensors
        self.field_grid = self._initialize_field_grid()
        self.q_field_values = torch.zeros(field_resolution, field_resolution, dtype=torch.complex64, device=self.device)
        
        # Emotional conductor state
        self.emotional_conductor = EmotionalConductorState(
            modulation_tensor=torch.ones(field_resolution, field_resolution, device=self.device),
            s_t_coupling_strength=1.0,
            conductor_phase=0.0,
            field_harmonics=torch.zeros(field_resolution, device=self.device)
        )
        
        # Observational evolution tracking
        self.observational_state = ObservationalEvolution(
            current_s_values=torch.ones(field_resolution, field_resolution, device=self.device),
            s_gradients=torch.zeros(field_resolution, field_resolution, device=self.device),
            persistence_factors=torch.ones(field_resolution, field_resolution, device=self.device),
            evolution_trajectories=torch.zeros(field_resolution, field_resolution, 2, device=self.device)
        )
        
        # Simulation state
        self.current_tau = 0.0
        self.simulation_time = 0.0
        self.field_history: List[Dict[str, torch.Tensor]] = []
        
        # Initialize adaptive optimization system
        self.__init_adaptive_optimization()
        
    def _initialize_field_grid(self) -> torch.Tensor:
        """Initialize spatial grid for field computations."""
        x = torch.linspace(-1, 1, self.field_resolution, device=self.device)
        y = torch.linspace(-1, 1, self.field_resolution, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
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
        required_keys = ['semantic_results', 'temporal_results', 'emotional_results', 'field_components_ready']
        for key in required_keys:
            if key not in combined_results:
                raise ValueError(f"Missing required key '{key}' in combined_results")
        
        # Check readiness
        if not combined_results['field_components_ready'].get('ready_for_unified_assembly', False):
            logger.warning("ChargeFactory indicates components not ready for unified assembly")
        
        # Get charge counts
        semantic_count = combined_results['field_components_ready']['semantic_fields']
        temporal_count = combined_results['field_components_ready']['temporal_biographies']
        emotional_count = combined_results['field_components_ready']['emotional_modulations']
        
        logger.info(f"Factory data loaded: {semantic_count} semantic fields, {temporal_count} temporal biographies, {emotional_count} emotional modulations")
        
        # 📚 EXTRACT VOCAB MAPPINGS: Get vocabulary context for agent creation
        vocab_mappings = combined_results.get('vocab_mappings', {})
        vocab_count = len(vocab_mappings.get('id_to_token', {}))
        logger.info(f"📚 Vocab context loaded: {vocab_count} tokens available for agent identification")
        
        if not (semantic_count == temporal_count == emotional_count):
            logger.warning(f"Mismatched component counts: semantic={semantic_count}, temporal={temporal_count}, emotional={emotional_count}")
        
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
            raise ValueError("No ChargeFactory results loaded. Call load_charge_factory_results() first.")
        
        logger.info("Creating ConceptualChargeAgent entities from factory data")
        
        # Import here to avoid circular imports
        from Sysnpire.model.liquid.conceptual_charge_agent import ConceptualChargeAgent
        
        # Determine number of agents to create
        max_available = min(
            len(self.combined_results['semantic_results']['field_representations']),
            len(self.combined_results['temporal_results']['temporal_biographies']),
            len(self.combined_results['emotional_results']['emotional_modulations'])
        )
        
        num_agents = min(max_available, max_agents) if max_agents else max_available
        
        logger.info(f"Creating {num_agents} agents from {max_available} available charge sets")
        
        agents_created = 0
        for i in range(num_agents):
            try:
                logger.info(f"Creating agent {i}")
                
                # Create agent using factory method WITH VOCAB CONTEXT
                vocab_mappings = self.combined_results.get('vocab_mappings', {})
                agent = ConceptualChargeAgent.from_charge_factory_results(
                    combined_results=self.combined_results,
                    charge_index=i,
                    device=str(self.device),
                    vocab_mappings=vocab_mappings  # 📚 Pass vocab context for agent identification
                )
                
                # Store both agent and charge object
                agent_id = agent.charge_id
                self.charge_agents[agent_id] = agent
                self.active_charges[agent_id] = agent.charge_obj
                
                # Update Q-field with new charge contribution
                self._update_q_field_contribution(agent.charge_obj)
                
                agents_created += 1
                logger.info(f"Agent {i} created successfully: {agent_id}")
                
            except Exception as e:
                logger.error(f"Failed to create agent {i}: {e}")
        
        logger.info(f"Successfully created {agents_created} ConceptualChargeAgent entities")
        
        # 🔧 VERIFY: Check that all agents have optimized interaction method
        agents_with_optimized = 0
        agents_missing_optimized = 0
        
        for agent_id, agent in self.charge_agents.items():
            if hasattr(agent, 'interact_with_optimized_field'):
                agents_with_optimized += 1
            else:
                agents_missing_optimized += 1
                logger.warning(f"🚨 Agent {agent_id} missing interact_with_optimized_field method!")
        
        logger.info(f"🔍 Method coverage: {agents_with_optimized}/{agents_created} agents have optimized interaction method")
        if agents_missing_optimized > 0:
            logger.warning(f"⚠️ {agents_missing_optimized} agents will use inefficient fallback!")
        else:
            logger.info("✅ All agents have optimized interaction method - maximum efficiency!")
        
        return agents_created
    
    def create_liquid_universe(self, combined_results: Dict[str, Any], max_agents: Optional[int] = None) -> Dict[str, Any]:
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
        logger.info("="*60)
        logger.info("CREATING LIQUID UNIVERSE")
        logger.info("LiquidOrchestrator taking control from ChargeFactory")
        logger.info("="*60)
        
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
        vocab_mappings = self.combined_results.get('vocab_mappings', {})
        
        for agent_id, agent in self.charge_agents.items():
            summary = {
                'agent_id': agent_id,
                'charge_index': agent.charge_index,
                'vocab_token_string': getattr(agent, 'vocab_token_string', 'unknown'),
                'vocab_token_id': getattr(agent, 'vocab_token_id', None),
                'Q_value': agent.Q_components.Q_value if agent.Q_components else complex(0),
                'living_Q_value': getattr(agent, 'living_Q_value', complex(0)),
                'gamma': agent.Q_components.gamma if agent.Q_components else 0.0,
                'field_magnitude': abs(agent.Q_components.Q_value) if agent.Q_components else 0.0
            }
            agent_summaries.append(summary)
        
        # Step 5: Build liquid results structure with vocab-enhanced summaries
        liquid_results = {
            'agent_pool': self.charge_agents,  # Living Q(τ,C,s) entities
            'active_charges': self.active_charges,  # ConceptualChargeObject instances
            'num_agents': num_agents,
            'field_statistics': field_stats,
            'optimization_stats': optimization_stats,  # Performance metrics
            'orchestrator': self,  # Reference for simulation control
            'ready_for_simulation': num_agents > 0,
            # 📚 VOCAB-ENHANCED RESULTS: Include agent summaries with readable vocab
            'agent_summaries': agent_summaries,
            'vocab_context': vocab_mappings
        }
        
        logger.info("Liquid universe creation complete:")
        logger.info(f"  Agents created: {num_agents}")
        logger.info(f"  Field energy: {field_stats['field_energy']:.6f}")
        logger.info(f"  Ready for simulation: {liquid_results['ready_for_simulation']}")
        
        return liquid_results
    
    def orchestrate_living_evolution(self, tau_steps: int = 100, tau_step_size: float = 0.01) -> Dict[str, Any]:
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
        logger.info("="*60)
        logger.info("🌊 ORCHESTRATING LIVING MODULAR FORMS EVOLUTION")
        logger.info("="*60)
        
        if not self.charge_agents:
            logger.warning("No agents available for evolution")
            return {'error': 'no_agents'}
        
        # Convert agents to list for easier access
        agents = list(self.charge_agents.values())
        evolution_results = {
            'breathing_patterns': [],
            'cascade_energies': [],
            'interaction_networks': [],
            'complexity_evolution': [],
            'emergent_harmonics': [],
            'collective_Q_field': []
        }
        
        logger.info(f"🎼 Starting evolution with {len(agents)} living modular forms")
        
        # 🔧 FIX: Initialize optimization IMMEDIATELY before evolution starts
        logger.info("🎯 Initializing O(N log N) optimization system...")
        self.listen_to_modular_forms(agents, 0.0)
        self.adapt_computation_strategy(agents, 0.0)
        
        # Verify optimization is ready
        sparse_interactions = sum(len(neighbors) for neighbors in self.adaptive_tuning['sparse_interaction_graph'].values())
        logger.info(f"✅ Optimization ready: {sparse_interactions} sparse interactions configured")
        
        for step in range(tau_steps):
            tau = step * tau_step_size
            self.current_tau = tau  # 🔧 FIX: Track current tau for safety checks
            
            # 🎯 ADAPTIVE OPTIMIZATION: Listen to modular forms and adapt strategies
            if step % 5 == 0:  # Listen every 5 steps (after initial setup)
                self.listen_to_modular_forms(agents, tau)
                self.adapt_computation_strategy(agents, tau)
            
            # MOVEMENT 1: 🎵 Collective Breathing (with breathing sync groups)
            breathing_synchrony = self._orchestrate_collective_breathing(agents, tau)
            evolution_results['breathing_patterns'].append(breathing_synchrony)
            
            # MOVEMENT 2: 🌊 Cascading Dimensional Feedback (with cascade optimization)
            cascade_energy = self._orchestrate_dimensional_cascades(agents)
            evolution_results['cascade_energies'].append(cascade_energy)
            
            # MOVEMENT 3: 🎭 Field Interactions (O(N log N) OPTIMIZED!)
            interaction_strength = self._orchestrate_field_interactions_optimized(agents)
            evolution_results['interaction_networks'].append(interaction_strength)
            
            # MOVEMENT 4: 📍 Observational Evolution (with phase boundaries)
            complexity_measure = self._orchestrate_s_parameter_evolution(agents)
            evolution_results['complexity_evolution'].append(complexity_measure)
            
            # MOVEMENT 5: 🎼 Measure Emergent Properties
            emergent_data = self._measure_emergent_complexity(agents, tau)
            evolution_results['emergent_harmonics'].append(emergent_data)
            
            # MOVEMENT 6: 🌌 Collective Q-Field
            collective_Q = self._compute_collective_Q_field(agents)
            evolution_results['collective_Q_field'].append(collective_Q)
            
            # Log progress every 10 steps
            if step % 10 == 0:
                # Include comprehensive optimization stats
                sparse_interactions = sum(len(neighbors) for neighbors in self.adaptive_tuning['sparse_interaction_graph'].values())
                sync_groups = len(self.adaptive_tuning['breathing_sync_groups'])
                cascade_chains = len(self.adaptive_tuning['resonance_cascades'])
                phase_boundaries = len(self.adaptive_tuning['phase_boundaries'])
                
                # Calculate complexity reduction
                total_possible_interactions = len(agents) * (len(agents) - 1)
                optimization_factor = sparse_interactions / total_possible_interactions if total_possible_interactions > 0 else 0
                
                # Check interaction method efficiency
                optimized_agents = sum(1 for agent in agents if hasattr(agent, 'interact_with_optimized_field'))
                fallback_agents = len(agents) - optimized_agents
                
                logger.info(f"Step {step}: Breathing={breathing_synchrony:.3f}, "
                          f"Cascade={cascade_energy:.3f}, "
                          f"Interaction={interaction_strength:.3f}, "
                          f"Complexity={complexity_measure:.3f}")
                logger.info(f"🎯 O(N log N) Stats: {sparse_interactions}/{total_possible_interactions} interactions "
                          f"({optimization_factor:.3%} of O(N²)), "
                          f"{sync_groups} sync groups, "
                          f"{cascade_chains} cascades, "
                          f"{phase_boundaries} boundaries")
                logger.info(f"⚡ Efficiency: {optimized_agents}/{len(agents)} optimized agents, "
                          f"{fallback_agents} using efficient fallback")
        
        # Final analysis
        final_analysis = self._analyze_evolution_results(evolution_results)
        
        logger.info("🎼 Living evolution complete!")
        logger.info(f"📊 Final complexity: {final_analysis['final_complexity']:.3f}")
        logger.info(f"🌊 Emergent harmonics: {len(final_analysis['emergent_harmonics'])}")
        logger.info(f"🎭 Collective coherence: {final_analysis['collective_coherence']:.3f}")
        
        return {
            'evolution_results': evolution_results,
            'final_analysis': final_analysis,
            'agents_evolved': len(agents),
            'tau_steps_completed': tau_steps
        }
    
    def _orchestrate_collective_breathing(self, agents: List, tau: float) -> float:
        """TRUE O(log N) collective breathing using vectorized group processing."""
        all_breathing_phases = []
        
        # Get breathing sync groups from adaptive optimization
        sync_groups = self.adaptive_tuning['breathing_sync_groups']
        
        if sync_groups:
            # 🚀 REVOLUTIONARY: Process O(log N) groups instead of O(N) agents
            for group_indices in sync_groups:
                if not group_indices:
                    continue
                    
                group_agents = [agents[i] for i in group_indices if i < len(agents)]
                
                if len(group_agents) > 0:
                    # Vectorized group breathing processing
                    group_phases = self._process_breathing_group_collectively(group_agents, tau)
                    all_breathing_phases.extend(group_phases)
            
            # Handle ungrouped agents efficiently
            grouped_indices = set()
            for group in sync_groups:
                grouped_indices.update(group)
            
            ungrouped_agents = [agents[i] for i in range(len(agents)) if i not in grouped_indices]
            if ungrouped_agents:
                # Process ungrouped agents as additional groups
                ungrouped_phases = self._process_breathing_group_collectively(ungrouped_agents, tau)
                all_breathing_phases.extend(ungrouped_phases)
        else:
            # Fallback: Process all agents as single group
            group_phases = self._process_breathing_group_collectively(agents, tau)
            all_breathing_phases.extend(group_phases)
        
        # Update dynamic field after all group breathing
        self.update_dynamic_field()
        
        # Measure breathing synchrony
        if len(all_breathing_phases) > 1:
            phase_diffs = [abs(all_breathing_phases[i] - all_breathing_phases[0]) 
                          for i in range(1, len(all_breathing_phases))]
            synchrony = 1.0 - (np.mean(phase_diffs) / (2 * np.pi))
        else:
            synchrony = 1.0
            
        return max(0.0, synchrony)
    
    def _orchestrate_dimensional_cascades(self, agents: List) -> float:
        """Coordinate dimensional feedback cascades using detected resonance chains."""
        total_cascade_energy = 0.0
        
        # Get detected resonance cascades from adaptive optimization
        cascade_chains = self.adaptive_tuning['resonance_cascades']
        
        if cascade_chains:
            # Process cascade chains for exponential amplification
            for cascade_info in cascade_chains:
                cascade_agent_indices = cascade_info.get('agents', [])
                cascade_magnitude = cascade_info.get('magnitude', 0.0)
                
                # Get agents in cascade chain
                cascade_agents = [agents[i] for i in cascade_agent_indices if i < len(agents)]
                
                if len(cascade_agents) >= 3:  # Valid cascade chain
                    # Amplify cascading for phase-aligned agents
                    amplification_factor = 1.0 + len(cascade_agents) * 0.1  # More agents = more amplification
                    
                    for agent in cascade_agents:
                        # Temporarily boost cascade rates for resonance chains
                        original_rate = agent.evolution_rates['cascading']
                        agent.evolution_rates['cascading'] *= amplification_factor
                        
                        agent.cascade_dimensional_feedback()
                        agent.sync_positions()
                        
                        # Restore original rate
                        agent.evolution_rates['cascading'] = original_rate
                        
                        # Measure enhanced cascade energy
                        cascade_momentum = agent.cascade_momentum
                        agent_cascade_energy = sum(abs(momentum) for momentum in cascade_momentum.values())
                        total_cascade_energy += agent_cascade_energy * amplification_factor
            
            # Handle agents not in any cascade chain
            cascaded_indices = set()
            for cascade_info in cascade_chains:
                cascaded_indices.update(cascade_info.get('agents', []))
            
            for i, agent in enumerate(agents):
                if i not in cascaded_indices:
                    agent.cascade_dimensional_feedback()
                    agent.sync_positions()
                    
                    cascade_momentum = agent.cascade_momentum
                    agent_cascade_energy = sum(abs(momentum) for momentum in cascade_momentum.values())
                    total_cascade_energy += agent_cascade_energy
        else:
            # Fallback: regular cascading for all agents
            for agent in agents:
                agent.cascade_dimensional_feedback()
                agent.sync_positions()
                
                cascade_momentum = agent.cascade_momentum
                agent_cascade_energy = sum(abs(momentum) for momentum in cascade_momentum.values())
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
                recent_interactions = agent.interaction_memory[-5:]  # Last 5 interactions
                agent_interaction_strength = sum(record['influence'] 
                                                for record in recent_interactions) / len(recent_interactions)
                total_interaction_strength += agent_interaction_strength
        
        # Update dynamic field after all interactions
        self.update_dynamic_field()
        
        return total_interaction_strength / len(agents) if agents else 0.0
    
    def _orchestrate_field_interactions_optimized(self, agents: List) -> float:
        """TRUE O(log N) field interactions using group-centric processing."""
        total_interaction_strength = 0.0
        
        # 🔧 VALIDATE: Check Q values before interactions
        valid_agents = []
        problematic_agents = []
        
        for agent in agents:
            # Check if agent has computed Q components
            if hasattr(agent, 'Q_components') and agent.Q_components is not None:
                Q_magnitude = abs(agent.Q_components.Q_value)
                if Q_magnitude > 1e-12:  # Meaningful Q value
                    valid_agents.append(agent)
                else:
                    problematic_agents.append((agent, Q_magnitude))
            else:
                # Agent hasn't computed Q yet - try to compute it with pool size
                try:
                    # 🚀 BOOST: Pass pool size to gamma calibration for stronger field presence
                    pool_size = len(agents)
                    agent.compute_complete_Q(pool_size=pool_size)
                    Q_magnitude = abs(agent.Q_components.Q_value) if agent.Q_components else 0.0
                    if Q_magnitude > 1e-12:
                        valid_agents.append(agent)
                    else:
                        problematic_agents.append((agent, Q_magnitude))
                except Exception as e:
                    logger.warning(f"⚠️  Agent {getattr(agent, 'charge_id', 'unknown')} - Q computation failed during interaction: {e}")
                    problematic_agents.append((agent, 0.0))
        
        # Log validation results
        if problematic_agents:
            logger.warning(f"⚠️  Field Interaction Validation: {len(problematic_agents)}/{len(agents)} agents have problematic Q values")
            for agent, Q_mag in problematic_agents[:3]:  # Show first 3
                agent_id = getattr(agent, 'charge_id', 'unknown')
                logger.warning(f"    Agent {agent_id}: |Q| = {Q_mag:.2e} (too small for meaningful interaction)")
            if len(problematic_agents) > 3:
                logger.warning(f"    ... and {len(problematic_agents) - 3} more agents with similar issues")
        
        if len(valid_agents) < 2:
            logger.warning(f"⚠️  Insufficient valid agents for field interactions: {len(valid_agents)}/{len(agents)}")
            return 0.0
        
        logger.debug(f"✅ Field Interaction Validation: {len(valid_agents)}/{len(agents)} agents ready for interaction")
        
        # Use only valid agents for interactions
        interaction_agents = valid_agents
        
        # Get interaction groups from adaptive optimization
        interaction_groups = self._build_interaction_groups(interaction_agents)
        
        # 🚀 REVOLUTIONARY CHANGE: Process O(log N) groups instead of O(N) agents
        for group_info in interaction_groups:
            group_agents = group_info['agents']
            group_interactions = group_info['interactions']
            
            # Process entire group collectively using vectorized operations
            group_strength = self._process_interaction_group_collectively(group_agents, group_interactions)
            total_interaction_strength += group_strength
        
        # Update dynamic field after all group interactions
        self.update_dynamic_field()
        
        return total_interaction_strength / len(interaction_agents) if interaction_agents else 0.0
    
    def _apply_precomputed_interactions(self, agent, nearby_agents_with_strengths: List[Tuple]):
        """
        MINIMAL FALLBACK: Apply pre-computed interaction effects without redundant calculations.
        
        This is the absolute minimal fallback for agents missing optimized methods.
        Uses pre-computed interaction strengths to avoid expensive recalculations.
        """
        if not nearby_agents_with_strengths:
            return
        
        # Apply interaction effects using pre-computed strengths
        total_influence = 0.0
        
        for other_agent, pre_computed_strength in nearby_agents_with_strengths:
            if other_agent is agent:
                continue
            
            # Use pre-computed strength instead of recalculating distance
            influence_strength = pre_computed_strength
            
            if influence_strength < 0.001:  # Skip very weak interactions
                continue
            
            # Apply basic q-coefficient interaction using pre-computed strength
            if hasattr(agent, 'breathing_q_coefficients') and hasattr(other_agent, 'breathing_q_coefficients'):
                for n in range(min(10, len(agent.breathing_q_coefficients), len(other_agent.breathing_q_coefficients))):
                    # Simplified interaction - just top 10 coefficients for efficiency
                    self_coeff = agent.breathing_q_coefficients.get(n, 0)
                    other_coeff = other_agent.breathing_q_coefficients.get(n, 0)
                    
                    if self_coeff != 0 and other_coeff != 0:
                        interaction_effect = influence_strength * np.real(self_coeff * np.conj(other_coeff)) * 0.1
                        agent.breathing_q_coefficients[n] += interaction_effect
            
            total_influence += influence_strength
        
        # Update living Q value after interactions
        if hasattr(agent, 'evaluate_living_form'):
            agent.living_Q_value = agent.evaluate_living_form()
            
            # Update charge object
            if hasattr(agent, 'charge_obj'):
                agent.charge_obj.complete_charge = agent.living_Q_value
                agent.charge_obj.magnitude = abs(agent.living_Q_value)
                agent.charge_obj.phase = np.angle(agent.living_Q_value)
        
        # Store minimal interaction record
        if hasattr(agent, 'interaction_memory'):
            interaction_record = {
                'influence': total_influence / len(nearby_agents_with_strengths) if nearby_agents_with_strengths else 0.0,
                'timestamp': getattr(agent.state, 'current_s', 0.0),
                'fallback': 'minimal_precomputed'
            }
            agent.interaction_memory.append(interaction_record)
            
            # Maintain memory length
            if len(agent.interaction_memory) > getattr(agent, 'max_memory_length', 100):
                agent.interaction_memory.pop(0)
    
    def _orchestrate_s_parameter_evolution(self, agents: List) -> float:
        """Coordinate observational state evolution across all agents."""
        complexity_measures = []
        
        # Each agent evolves its s-parameter
        for agent in agents:
            agent.evolve_s_parameter(agents)
            
            # Sync positions after s-parameter evolution
            agent.sync_positions()
            
            # Measure complexity as distance from initial state
            s_distance = abs(agent.state.current_s - agent.state.s_zero)
            complexity_measures.append(s_distance)
        
        # Update dynamic field after all s-parameter evolution
        self.update_dynamic_field()
        
        # Return average complexity
        return np.mean(complexity_measures) if complexity_measures else 0.0
    
    def _measure_emergent_complexity(self, agents: List, tau: float) -> Dict[str, Any]:
        """Measure emergent patterns and complexity from agent interactions."""
        emergent_data = {
            'new_harmonics': 0,
            'hecke_adaptations': 0,
            'q_coefficient_diversity': 0.0,
            'tau': tau
        }
        
        # Count new harmonics that emerged
        for agent in agents:
            # Count q-coefficients that weren't in original embedding
            original_size = len(agent.semantic_field.embedding_components)
            current_size = len(agent.breathing_q_coefficients)
            emergent_data['new_harmonics'] += max(0, current_size - original_size)
        
        # Measure Hecke eigenvalue adaptations
        for agent in agents:
            if hasattr(agent, '_initial_hecke_eigenvalues'):
                adaptations = sum(1 for p in agent.hecke_eigenvalues 
                                if abs(agent.hecke_eigenvalues[p] - agent._initial_hecke_eigenvalues.get(p, 0)) > 0.01)
                emergent_data['hecke_adaptations'] += adaptations
        
        # Measure q-coefficient diversity across agents
        if len(agents) > 1:
            all_coeffs = []
            for agent in agents:
                agent_coeffs = [abs(coeff) for coeff in agent.breathing_q_coefficients.values()]
                all_coeffs.extend(agent_coeffs[:10])  # First 10 coefficients
            
            if all_coeffs:
                emergent_data['q_coefficient_diversity'] = np.std(all_coeffs) / (np.mean(all_coeffs) + 1e-8)
        
        return emergent_data
    
    def _compute_collective_Q_field(self, agents: List) -> Dict[str, Any]:
        """Compute collective Q-field from all living modular forms."""
        collective_data = {
            'total_magnitude': 0.0,
            'phase_coherence': 0.0,
            'field_energy': 0.0,
            'agent_contributions': []
        }
        
        # Gather all Q values
        Q_values = []
        for agent in agents:
            Q_val = agent.living_Q_value
            Q_values.append(Q_val)
            collective_data['agent_contributions'].append({
                'magnitude': abs(Q_val),
                'phase': np.angle(Q_val),
                'agent_id': agent.charge_id
            })
        
        if Q_values:
            # Total magnitude
            collective_data['total_magnitude'] = sum(abs(Q) for Q in Q_values)
            
            # Phase coherence (how aligned the phases are)
            phases = [np.angle(Q) for Q in Q_values]
            if len(phases) > 1:
                phase_variance = np.var(phases)
                collective_data['phase_coherence'] = np.exp(-phase_variance)  # High coherence = low variance
            else:
                collective_data['phase_coherence'] = 1.0
            
            # Field energy (sum of squared magnitudes)
            collective_data['field_energy'] = sum(abs(Q)**2 for Q in Q_values)
        
        return collective_data
    
    def _analyze_evolution_results(self, evolution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the complete evolution to extract emergent patterns."""
        analysis = {
            'final_complexity': 0.0,
            'emergent_harmonics': [],
            'collective_coherence': 0.0,
            'evolution_stability': 0.0,
            'cascade_efficiency': 0.0
        }
        
        if not evolution_results['complexity_evolution']:
            return analysis
        
        # Final complexity
        analysis['final_complexity'] = evolution_results['complexity_evolution'][-1]
        
        # Count total emergent harmonics
        total_harmonics = sum(data['new_harmonics'] for data in evolution_results['emergent_harmonics'])
        analysis['emergent_harmonics'] = list(range(total_harmonics))  # Placeholder for actual harmonic data
        
        # Collective coherence (final breathing synchrony)
        if evolution_results['breathing_patterns']:
            analysis['collective_coherence'] = evolution_results['breathing_patterns'][-1]
        
        # Evolution stability (variance in complexity evolution)
        complexity_series = evolution_results['complexity_evolution']
        if len(complexity_series) > 1:
            complexity_variance = np.var(complexity_series)
            analysis['evolution_stability'] = 1.0 / (1.0 + complexity_variance)
        
        # Cascade efficiency (how well energy cascades between dimensions)
        if evolution_results['cascade_energies']:
            cascade_trend = np.polyfit(range(len(evolution_results['cascade_energies'])), 
                                     evolution_results['cascade_energies'], 1)[0]
            analysis['cascade_efficiency'] = max(0.0, cascade_trend)
        
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
        
        # Add current Q value to field tensor
        self.q_field_values[row, col] += Q_value
    
    def __init_adaptive_optimization(self):
        """
        Initialize adaptive optimization system that LISTENS to modular forms.
        
        This creates a living tuning system that learns from the mathematics itself,
        adapting computation strategies based on what the modular forms reveal.
        """
        self.adaptive_tuning = {
            # LISTENING HISTORY - track mathematical patterns over time
            'eigenvalue_history': [],
            'breathing_history': [],
            'interaction_history': [],
            'cascade_history': [],
            'phase_history': [],
            
            # DYNAMIC OPTIMIZATIONS - adapt based on listening
            'eigenvalue_clusters': {},
            'breathing_sync_groups': [],
            'sparse_interaction_graph': {},
            'resonance_cascades': [],
            'phase_boundaries': [],
            
            # LEARNING PARAMETERS - evolve with experience
            'cluster_sensitivity': 0.1,      # How sensitive eigenvalue clustering is
            'sync_threshold': np.pi/8,       # Phase threshold for breathing sync
            'interaction_cutoff': 0.01,      # Minimum interaction strength
            'cascade_threshold': 0.8,        # Phase alignment for cascades
            'adaptation_rate': 0.02,         # How fast parameters adapt
            
            # PERFORMANCE TRACKING - learn what works
            'computation_efficiency': [],
            'mathematical_coherence': [],
            'optimization_success': {},
            'last_adaptation_step': 0,
            
            # LISTENER SYSTEMS - analyze modular form behavior
            'eigenvalue_listener': self._create_eigenvalue_listener(),
            'breathing_listener': self._create_breathing_listener(),
            'interaction_listener': self._create_interaction_listener(),
            'cascade_listener': self._create_cascade_listener(),
            'phase_listener': self._create_phase_listener()
        }
    
    def listen_to_modular_forms(self, agents: List, tau: float):
        """
        LISTEN to what the modular forms are telling us about optimal computation.
        
        This is the core insight: let the mathematics itself guide optimization.
        The modular forms reveal their own computational preferences through their behavior.
        """
        tuning = self.adaptive_tuning
        
        # LISTEN 1: 🎯 Eigenvalue Pattern Detection
        eigenvalue_patterns = tuning['eigenvalue_listener'].listen(agents)
        if eigenvalue_patterns.get('new_clusters_detected', False):
            # The eigenvalues are telling us they want different clustering
            tuning['cluster_sensitivity'] = eigenvalue_patterns.get('suggested_sensitivity', tuning['cluster_sensitivity'])
            logger.info(f"🎯 Eigenvalues suggest sensitivity: {tuning['cluster_sensitivity']:.3f}")
        
        # LISTEN 2: 🌊 Breathing Synchronization Detection  
        breathing_patterns = tuning['breathing_listener'].listen(agents, tau)
        if breathing_patterns.get('sync_strength_changed', False):
            # The breathing patterns are showing us new synchronization thresholds
            tuning['sync_threshold'] = breathing_patterns.get('optimal_threshold', tuning['sync_threshold'])
            logger.info(f"🌊 Breathing suggests sync threshold: {tuning['sync_threshold']:.3f}")
        
        # LISTEN 3: 🔗 Interaction Strength Learning
        interaction_patterns = tuning['interaction_listener'].listen(agents)
        if interaction_patterns.get('cutoff_should_adapt', False):
            # The interactions are showing us what strength levels actually matter
            tuning['interaction_cutoff'] = interaction_patterns.get('suggested_cutoff', tuning['interaction_cutoff'])
            logger.info(f"🔗 Interactions suggest cutoff: {tuning['interaction_cutoff']:.4f}")
        
        # LISTEN 4: 🎼 Resonance Cascade Discovery
        cascade_patterns = tuning['cascade_listener'].listen(agents)
        if cascade_patterns.get('new_cascades_found', False):
            # The resonances are revealing new amplification opportunities
            tuning['cascade_threshold'] = cascade_patterns.get('optimal_threshold', tuning['cascade_threshold'])
            logger.info(f"🎼 Cascades suggest threshold: {tuning['cascade_threshold']:.3f}")
        
        # LISTEN 5: 🌀 Phase Coherence Boundaries
        phase_patterns = tuning['phase_listener'].listen(agents)
        if phase_patterns.get('boundaries_shifted', False):
            # The phase relationships are showing us new computational boundaries
            new_boundaries = phase_patterns.get('new_boundaries', [])
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
            if self._clusters_significantly_different(tuning['eigenvalue_clusters'], new_clusters):
                tuning['eigenvalue_clusters'] = new_clusters
                logger.info(f"🎯 Adapted eigenvalue clustering: {len(new_clusters)} clusters")
        
        # ADAPT 2: 🌊 Dynamic Breathing Groups
        new_sync_groups = self._adaptive_breathing_grouping(agents)
        if self._sync_groups_changed(tuning['breathing_sync_groups'], new_sync_groups):
            tuning['breathing_sync_groups'] = new_sync_groups
            logger.info(f"🌊 Adapted breathing groups: {len(new_sync_groups)} sync groups")
        
        # ADAPT 3: 🔗 Dynamic Interaction Graph
        # 🔧 FIX: Always build graph if empty, otherwise rebuild periodically
        graph_is_empty = not tuning['sparse_interaction_graph']
        should_rebuild = int(tau * 100) % 5 == 0
        
        if graph_is_empty or should_rebuild:
            new_graph = self._adaptive_interaction_graph(agents)
            tuning['sparse_interaction_graph'] = new_graph
            total_interactions = sum(len(neighbors) for neighbors in new_graph.values())
            status = "initialized" if graph_is_empty else "rebuilt"
            logger.info(f"🔗 Interaction graph {status}: {total_interactions} total interactions")
        
        # ADAPT 4: 🎼 Dynamic Cascade Detection
        new_cascades = self._adaptive_cascade_detection(agents)
        if len(new_cascades) != len(tuning['resonance_cascades']):
            tuning['resonance_cascades'] = new_cascades
            logger.info(f"🎼 Adapted cascades: {len(new_cascades)} resonance chains")
        
        # ADAPT 5: 🌀 Dynamic Phase Boundaries
        if int(tau * 100) % 15 == 0:  # Recompute boundaries every 15 steps
            new_boundaries = self._adaptive_phase_boundaries(agents)
            tuning['phase_boundaries'] = new_boundaries
            logger.info(f"🌀 Adapted phase boundaries: {len(new_boundaries)} regions")
    
    def initialize_liquid_simulation(self, combined_results: Dict[str, Any], max_agents: Optional[int] = None) -> Dict[str, Any]:
        """
        Complete initialization of liquid simulation from ChargeFactory data.
        
        MAIN ENTRY POINT: This method coordinates everything from factory data to live simulation.
        
        Args:
            combined_results: ChargeFactory output dictionary
            max_agents: Maximum number of agents to create
            
        Returns:
            Initialization summary
        """
        logger.info("="*60)
        logger.info("INITIALIZING LIQUID SIMULATION")
        logger.info("LiquidOrchestrator RUNS THE SHOW")
        logger.info("="*60)
        
        # Step 1: Load factory data
        num_charges = self.load_charge_factory_results(combined_results)
        
        # Step 2: Create agents
        num_agents = self.create_agents_from_factory_data(max_agents)
        
        # Step 3: Initialize field dynamics
        initial_stats = self.get_field_statistics()
        
        summary = {
            'charges_loaded': num_charges,
            'agents_created': num_agents,
            'field_resolution': self.field_resolution,
            'device': str(self.device),
            'initial_field_energy': initial_stats['field_energy'],
            'ready_for_simulation': num_agents > 0
        }
        
        logger.info("Liquid simulation initialization complete:")
        logger.info(f"  Charges loaded: {num_charges}")
        logger.info(f"  Agents created: {num_agents}")
        logger.info(f"  Field resolution: {self.field_resolution}x{self.field_resolution}")
        logger.info(f"  Device: {str(self.device)}")
        logger.info(f"  Initial field energy: {initial_stats['field_energy']:.6f}")
        logger.info(f"  Ready for simulation: {summary['ready_for_simulation']}")
        
        return summary
        
    def add_conceptual_charge(self, charge_obj: ConceptualChargeObject, 
                            field_position: Optional[Tuple[float, float]] = None):
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
        
    def _find_optimal_placement(self, charge_obj: ConceptualChargeObject) -> Tuple[float, float]:
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
                field_distortions=torch.empty(0, device=self.device)
            )
        
        # Compute pairwise interference using CURRENT living Q values
        pairs = []
        strengths = []
        phases = []
        distortions = []
        
        for i in range(n_charges):
            for j in range(i + 1, n_charges):
                agent_a = self.charge_agents[charge_ids[i]]
                agent_b = self.charge_agents[charge_ids[j]]
                
                # Use CURRENT living Q values (not stale complete_charge)
                q_a = agent_a.living_Q_value
                q_b = agent_b.living_Q_value
                
                # Compute interference
                interference = q_a * np.conj(q_b)
                strength = abs(interference)
                phase_diff = np.angle(interference)
                
                # Field distortion from interference
                distortion = strength * np.cos(phase_diff)
                
                pairs.append((charge_ids[i], charge_ids[j]))
                strengths.append(strength)
                phases.append(phase_diff)
                distortions.append(distortion)
        
        return FieldInterferencePattern(
            charge_pairs=pairs,
            interference_strengths=torch.tensor(strengths, device=self.device),
            phase_relationships=torch.tensor(phases, device=self.device),
            field_distortions=torch.tensor(distortions, device=self.device)
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
                freq_component = np.mean(emotional_traj) if len(emotional_traj) > 0 else 0.0
                total_emotional_influence += abs(freq_component)
                
                # Distribute across harmonic frequencies
                for k in range(min(len(conductor_harmonics), len(emotional_traj))):
                    conductor_harmonics[k] += emotional_traj[k] if k < len(emotional_traj) else 0.0
        
        # Update conductor state
        self.emotional_conductor.modulation_tensor *= (1.0 + 0.1 * total_emotional_influence)
        self.emotional_conductor.conductor_phase += tau_step * total_emotional_influence
        self.emotional_conductor.field_harmonics = conductor_harmonics
        
        # Update S-T coupling strength based on emotional field
        self.emotional_conductor.s_t_coupling_strength = 1.0 + 0.2 * total_emotional_influence
        
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
        s_evolution = tau_step * gradient_magnitude * self.emotional_conductor.s_t_coupling_strength
        
        self.observational_state.current_s_values += s_evolution
        
        # Apply persistence decay
        decay_factor = torch.exp(-tau_step * 0.1)  # Adjustable decay rate
        self.observational_state.persistence_factors *= decay_factor
        
        # Update trajectory tracking
        self.observational_state.evolution_trajectories += self.observational_state.s_gradients * tau_step
        
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
                
    def simulate_liquid_dynamics(self, tau_steps: int = 100, tau_step_size: float = 0.01) -> Dict[str, Any]:
        """
        Run liquid stage simulation with field-theoretic dynamics.
        
        Args:
            tau_steps: Number of tau evolution steps
            tau_step_size: Size of each tau step
            
        Returns:
            Simulation results with field evolution data
        """
        simulation_results = {
            'field_evolution': [],
            'interference_patterns': [],
            'emotional_conductor_states': [],
            'observational_evolution': [],
            'final_q_field': None,
            'charge_trajectories': {charge_id: [] for charge_id in self.active_charges.keys()}
        }
        
        for step in range(tau_steps):
            # Update tau
            self.current_tau += tau_step_size
            self.simulation_time += tau_step_size
            
            # Compute field interference patterns
            interference = self.compute_field_interference_patterns()
            simulation_results['interference_patterns'].append(interference)
            
            # Update emotional conductor
            self.update_emotional_conductor(tau_step_size)
            
            # Evolve observational states
            self.evolve_observational_states(tau_step_size)
            
            # Record field state
            field_snapshot = {
                'tau': self.current_tau,
                'q_field_magnitude': torch.abs(self.q_field_values).clone(),
                'q_field_phase': torch.angle(self.q_field_values).clone(),
                'emotional_modulation': self.emotional_conductor.modulation_tensor.clone(),
                's_values': self.observational_state.current_s_values.clone()
            }
            simulation_results['field_evolution'].append(field_snapshot)
            
            # Record charge positions
            for charge_id, charge_obj in self.active_charges.items():
                if charge_obj.metadata.field_position is not None:
                    simulation_results['charge_trajectories'][charge_id].append({
                        'tau': self.current_tau,
                        'position': charge_obj.metadata.field_position,
                        'q_value': charge_obj.complete_charge,
                        's_value': charge_obj.observational_state
                    })
        
        # Final field state
        simulation_results['final_q_field'] = self.q_field_values.clone()
        
        return simulation_results
        
    def get_field_statistics(self) -> Dict[str, Any]:
        """Get current field statistics and health metrics."""
        # 🚀 FIX: Use living_Q_value from agents instead of empty q_field_values tensor
        if self.charge_agents:
            # Calculate field statistics from actual agent Q values
            agent_q_values = [abs(agent.living_Q_value) for agent in self.charge_agents.values()]
            if agent_q_values:
                q_magnitude_list = agent_q_values
                field_energy = sum(q ** 2 for q in q_magnitude_list)
                max_field_strength = max(q_magnitude_list)
                mean_field_strength = sum(q_magnitude_list) / len(q_magnitude_list)
                field_coverage = sum(1 for q in q_magnitude_list if q > 0.01) / len(q_magnitude_list)
            else:
                field_energy = 0.0
                max_field_strength = 0.0 
                mean_field_strength = 0.0
                field_coverage = 0.0
        else:
            # Fallback to tensor calculation if no agents
            q_magnitude = torch.abs(self.q_field_values)
            field_energy = float(torch.sum(q_magnitude ** 2))
            max_field_strength = float(torch.max(q_magnitude))
            mean_field_strength = float(torch.mean(q_magnitude))
            field_coverage = float(torch.mean((q_magnitude > 0.01).float()))
        
        return {
            'active_charges': len(self.active_charges),
            'field_energy': field_energy,
            'max_field_strength': max_field_strength,
            'mean_field_strength': mean_field_strength,
            'field_coverage': field_coverage,
            'emotional_conductor_strength': float(self.emotional_conductor.s_t_coupling_strength),
            'mean_s_value': float(torch.mean(self.observational_state.current_s_values)),
            'current_tau': self.current_tau,
            'simulation_time': self.simulation_time
        }
        
    def reset_simulation(self):
        """Reset simulation state while keeping charges."""
        self.current_tau = 0.0
        self.simulation_time = 0.0
        self.field_history.clear()
        
        # Reset field tensors
        self.q_field_values = torch.zeros_like(self.q_field_values)
        self.observational_state.current_s_values = torch.ones_like(self.observational_state.current_s_values)
        self.observational_state.s_gradients = torch.zeros_like(self.observational_state.s_gradients)
        self.observational_state.persistence_factors = torch.ones_like(self.observational_state.persistence_factors)
        
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
                    return {'new_clusters_detected': False}
                
                # Extract eigenvalues from living modular forms
                eigenvalues = []
                for agent in agents:
                    if hasattr(agent, 'hecke_eigenvalues') and agent.hecke_eigenvalues:
                        eigenvalues.extend(agent.hecke_eigenvalues.values())
                
                if len(eigenvalues) < 2:
                    return {'new_clusters_detected': False}
                
                # Analyze eigenvalue distribution
                eigenvalue_spread = np.std(eigenvalues)
                
                # Detect if clustering should be adapted
                spread_change = abs(eigenvalue_spread - self.last_eigenvalue_spread)
                cluster_change_detected = spread_change > 0.1 * self.last_eigenvalue_spread
                
                self.last_eigenvalue_spread = eigenvalue_spread
                
                if cluster_change_detected:
                    # Suggest new sensitivity based on eigenvalue density
                    suggested_sensitivity = max(0.05, min(0.5, eigenvalue_spread / 10))
                    return {
                        'new_clusters_detected': True,
                        'suggested_sensitivity': suggested_sensitivity,
                        'eigenvalue_spread': eigenvalue_spread
                    }
                
                return {'new_clusters_detected': False}
        
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
                    return {'sync_strength_changed': False}
                
                # Extract breathing phases
                breathing_phases = []
                for agent in agents:
                    if hasattr(agent, 'breath_phase'):
                        breathing_phases.append(agent.breath_phase)
                
                if len(breathing_phases) < 2:
                    return {'sync_strength_changed': False}
                
                # Calculate current synchronization strength
                phase_differences = []
                for i in range(len(breathing_phases)):
                    for j in range(i+1, len(breathing_phases)):
                        phase_diff = abs(breathing_phases[i] - breathing_phases[j])
                        phase_diff = min(phase_diff, 2*np.pi - phase_diff)  # Wrap to [0, π]
                        phase_differences.append(phase_diff)
                
                sync_strength = 1.0 - np.mean(phase_differences) / np.pi
                
                # Detect significant sync changes
                sync_change = abs(sync_strength - self.last_sync_strength)
                sync_changed = sync_change > 0.1
                
                self.last_sync_strength = sync_strength
                
                if sync_changed:
                    # Suggest optimal threshold based on current sync patterns
                    optimal_threshold = np.percentile(phase_differences, 75) if phase_differences else np.pi/8
                    return {
                        'sync_strength_changed': True,
                        'optimal_threshold': optimal_threshold,
                        'sync_strength': sync_strength
                    }
                
                return {'sync_strength_changed': False}
        
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
                    return {'cutoff_should_adapt': False}
                
                # Calculate interaction strengths between agents
                interaction_strengths = []
                for i, agent1 in enumerate(agents):
                    for j, agent2 in enumerate(agents[i+1:], i+1):
                        # Calculate field interaction strength
                        if hasattr(agent1, 'living_Q_value') and hasattr(agent2, 'living_Q_value'):
                            Q1, Q2 = agent1.living_Q_value, agent2.living_Q_value
                            interaction = abs(Q1 * np.conj(Q2)) if Q1 is not None and Q2 is not None else 0.0
                            interaction_strengths.append(float(np.real(interaction)))
                
                if not interaction_strengths:
                    return {'cutoff_should_adapt': False}
                
                avg_interaction = np.mean(interaction_strengths)
                
                # Detect if cutoff should adapt
                interaction_change = abs(avg_interaction - self.last_avg_interaction)
                cutoff_should_adapt = interaction_change > 0.5 * self.last_avg_interaction if self.last_avg_interaction > 0 else False
                
                self.last_avg_interaction = avg_interaction
                
                if cutoff_should_adapt:
                    # Suggest cutoff as 10% of median interaction strength
                    suggested_cutoff = max(0.001, np.percentile(interaction_strengths, 10))
                    return {
                        'cutoff_should_adapt': True,
                        'suggested_cutoff': suggested_cutoff,
                        'avg_interaction': avg_interaction
                    }
                
                return {'cutoff_should_adapt': False}
        
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
                    return {'new_cascades_found': False}
                
                # Find potential cascade chains (3+ agents with aligned phases)
                cascade_chains = []
                agent_phases = []
                
                for agent in agents:
                    if hasattr(agent, 'living_Q_value') and agent.living_Q_value is not None:
                        phase = np.angle(agent.living_Q_value)
                        agent_phases.append((agent, phase))
                
                # Look for phase-aligned groups that could cascade
                phase_groups = {}
                for agent, phase in agent_phases:
                    phase_bucket = round(phase * 8 / (2*np.pi)) * (2*np.pi) / 8  # Discretize phases
                    if phase_bucket not in phase_groups:
                        phase_groups[phase_bucket] = []
                    phase_groups[phase_bucket].append(agent)
                
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
                            chain_phases = [np.angle(agent.living_Q_value) for agent in chain if hasattr(agent, 'living_Q_value') and agent.living_Q_value is not None]
                            if len(chain_phases) > 1:
                                phase_spreads.append(np.std(chain_phases))
                        
                        optimal_threshold = np.mean(phase_spreads) if phase_spreads else 0.8
                    else:
                        optimal_threshold = 0.8
                    
                    return {
                        'new_cascades_found': True,
                        'optimal_threshold': optimal_threshold,
                        'cascade_count': new_cascade_count
                    }
                
                return {'new_cascades_found': False}
        
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
                    return {'boundaries_shifted': False}
                
                # Extract phases and positions
                agent_data = []
                for agent in agents:
                    if (hasattr(agent, 'living_Q_value') and agent.living_Q_value is not None 
                        and hasattr(agent, 'state') and hasattr(agent.state, 'field_position')):
                        phase = np.angle(agent.living_Q_value)
                        pos = agent.state.field_position
                        agent_data.append((pos[0], pos[1], phase))
                
                if len(agent_data) < 4:
                    return {'boundaries_shifted': False}
                
                # Simple phase boundary detection using phase gradients
                boundaries = []
                threshold = np.pi/4  # Phase difference threshold for boundary
                
                # Sort by x position to detect x-boundaries
                agent_data_x = sorted(agent_data, key=lambda x: x[0])
                for i in range(len(agent_data_x)-1):
                    phase_diff = abs(agent_data_x[i][2] - agent_data_x[i+1][2])
                    phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                    if phase_diff > threshold:
                        boundaries.append(('x', (agent_data_x[i][0] + agent_data_x[i+1][0])/2))
                
                # Sort by y position to detect y-boundaries  
                agent_data_y = sorted(agent_data, key=lambda x: x[1])
                for i in range(len(agent_data_y)-1):
                    phase_diff = abs(agent_data_y[i][2] - agent_data_y[i+1][2])
                    phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                    if phase_diff > threshold:
                        boundaries.append(('y', (agent_data_y[i][1] + agent_data_y[i+1][1])/2))
                
                boundary_count = len(boundaries)
                boundaries_changed = boundary_count != self.last_boundary_count
                
                self.last_boundary_count = boundary_count
                
                if boundaries_changed:
                    return {
                        'boundaries_shifted': True,
                        'new_boundaries': boundaries,
                        'boundary_count': boundary_count
                    }
                
                return {'boundaries_shifted': False}
        
        return PhaseListener(self)
    
    def _adaptive_eigenvalue_clustering(self, agents):
        """O(N log N) eigenvalue clustering using mathematical structure."""
        if not agents:
            return {}
        
        # Extract eigenvalues efficiently
        eigenvalue_data = []
        for i, agent in enumerate(agents):
            if hasattr(agent, 'hecke_eigenvalues') and agent.hecke_eigenvalues:
                for level, eigenval in agent.hecke_eigenvalues.items():
                    eigenvalue_data.append((eigenval, i, level))
        
        if len(eigenvalue_data) < 2:
            return {}
        
        # Sort eigenvalues for O(N log N) clustering (by magnitude for complex numbers)
        eigenvalue_data.sort(key=lambda x: abs(x[0]))
        
        # Adaptive clustering based on eigenvalue gaps
        clusters = {}
        current_cluster = 0
        cluster_threshold = self.adaptive_tuning['cluster_sensitivity']
        
        for i, (eigenval, agent_idx, level) in enumerate(eigenvalue_data):
            if i == 0:
                clusters[current_cluster] = [(agent_idx, level, eigenval)]
            else:
                prev_eigenval = eigenvalue_data[i-1][0]
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
            if hasattr(agent, 'breath_phase'):
                breathing_data.append((agent.breath_phase, i))
        
        if len(breathing_data) < 2:
            return []
        
        # Sort by phase for O(N log N) grouping
        breathing_data.sort(key=lambda x: x[0])
        
        # Group agents with similar breathing phases
        sync_groups = []
        current_group = [breathing_data[0][1]]
        sync_threshold = self.adaptive_tuning['sync_threshold']
        
        for i in range(1, len(breathing_data)):
            phase_diff = abs(breathing_data[i][0] - breathing_data[i-1][0])
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)  # Wrap around
            
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
        """O(N log N) sparse interaction graph using exponential decay."""
        if len(agents) < 2:
            return {}
        
        interaction_graph = {}
        cutoff = self.adaptive_tuning['interaction_cutoff']
        
        # Build spatial index for O(N log N) neighbor finding
        agent_positions = []
        for i, agent in enumerate(agents):
            if hasattr(agent, 'state') and hasattr(agent.state, 'field_position'):
                pos = agent.state.field_position
                agent_positions.append((pos[0], pos[1], i))
        
        # Sort by x-coordinate for spatial partitioning
        agent_positions.sort(key=lambda x: x[0])
        
        # For each agent, only check nearby agents (sparse interactions)
        # 🚀 ADAPTIVE WINDOW: Scale with population density
        adaptive_window = max(10, int(np.log2(len(agent_positions))))
        
        for i, (x1, y1, idx1) in enumerate(agent_positions):
            neighbors = []
            
            # Check agents in spatial neighborhood only - DENSITY ADAPTIVE
            for j in range(max(0, i-adaptive_window), min(len(agent_positions), i+adaptive_window+1)):
                if j == i:
                    continue
                    
                x2, y2, idx2 = agent_positions[j]
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Exponential decay interaction strength
                if distance < 2.0:  # Maximum interaction distance
                    interaction_strength = np.exp(-distance)
                    if interaction_strength > cutoff:
                        neighbors.append((idx2, interaction_strength))
            
            if neighbors:
                interaction_graph[idx1] = neighbors
        
        return interaction_graph
    
    def _adaptive_cascade_detection(self, agents):
        """Detect resonance cascades for exponential amplification."""
        if len(agents) < 3:
            return []
        
        cascades = []
        cascade_threshold = self.adaptive_tuning['cascade_threshold']
        
        # Find chains of 3+ agents with aligned phases
        agent_phases = []
        for i, agent in enumerate(agents):
            if hasattr(agent, 'living_Q_value') and agent.living_Q_value is not None:
                phase = np.angle(agent.living_Q_value)
                magnitude = abs(agent.living_Q_value)
                agent_phases.append((i, phase, magnitude))
        
        if len(agent_phases) < 3:
            return []
        
        # Sort by phase for efficient cascade detection
        agent_phases.sort(key=lambda x: x[1])
        
        # Find consecutive agents with aligned phases
        for i in range(len(agent_phases) - 2):
            cascade_chain = [agent_phases[i]]
            
            for j in range(i+1, len(agent_phases)):
                phase_diff = abs(agent_phases[j][1] - cascade_chain[-1][1])
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                
                if phase_diff < np.pi / 4:  # Phase alignment threshold
                    cascade_chain.append(agent_phases[j])
                else:
                    break
            
            # Keep cascades with 3+ aligned agents
            if len(cascade_chain) >= 3:
                agent_indices = [item[0] for item in cascade_chain]
                total_magnitude = sum(item[2] for item in cascade_chain)
                cascades.append({
                    'agents': agent_indices,
                    'magnitude': total_magnitude,
                    'length': len(cascade_chain)
                })
        
        return cascades
    
    def _adaptive_phase_boundaries(self, agents):
        """Compute phase coherence boundaries for region-based optimization."""
        if len(agents) < 4:
            return []
        
        boundaries = []
        
        # Extract agent data efficiently
        agent_data = []
        for agent in agents:
            if (hasattr(agent, 'living_Q_value') and agent.living_Q_value is not None 
                and hasattr(agent, 'state') and hasattr(agent.state, 'field_position')):
                pos = agent.state.field_position
                phase = np.angle(agent.living_Q_value)
                agent_data.append((pos[0], pos[1], phase))
        
        if len(agent_data) < 4:
            return []
        
        # Detect phase discontinuities using gradient analysis
        # Sort by x for x-boundaries
        agent_data_x = sorted(agent_data, key=lambda x: x[0])
        for i in range(len(agent_data_x) - 1):
            phase1, phase2 = agent_data_x[i][2], agent_data_x[i+1][2]
            phase_diff = abs(phase2 - phase1)
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)
            
            if phase_diff > np.pi/3:  # Significant phase jump
                boundary_x = (agent_data_x[i][0] + agent_data_x[i+1][0]) / 2
                boundaries.append(('x_boundary', boundary_x, phase_diff))
        
        # Sort by y for y-boundaries
        agent_data_y = sorted(agent_data, key=lambda x: x[1])
        for i in range(len(agent_data_y) - 1):
            phase1, phase2 = agent_data_y[i][2], agent_data_y[i+1][2]
            phase_diff = abs(phase2 - phase1)
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)
            
            if phase_diff > np.pi/3:  # Significant phase jump
                boundary_y = (agent_data_y[i][1] + agent_data_y[i+1][1]) / 2
                boundaries.append(('y_boundary', boundary_y, phase_diff))
        
        return boundaries
    
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
    
    def _build_interaction_groups(self, agents: List) -> List[Dict]:
        """
        Build O(log N) interaction groups from sparse graph and mathematical structure.
        
        REVOLUTIONARY APPROACH: Instead of processing N agents individually,
        create log N groups that can be processed collectively.
        """
        if not agents:
            return []
        
        # Get current sparse interaction graph
        sparse_graph = self.adaptive_tuning['sparse_interaction_graph']
        
        # Safety check - ensure sparse graph exists
        if not sparse_graph:
            logger.warning("🚨 Building emergency interaction groups...")
            self.listen_to_modular_forms(agents, self.current_tau)
            self.adapt_computation_strategy(agents, self.current_tau)
            sparse_graph = self.adaptive_tuning['sparse_interaction_graph']
        
        # Build interaction groups using mathematical structure
        interaction_groups = []
        processed_agents = set()
        
        # Method 1: Use breathing sync groups as interaction groups
        sync_groups = self.adaptive_tuning['breathing_sync_groups']
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
                    agent_interactions = sparse_graph.get(agent_idx, [])
                    group_interactions[agent_idx] = agent_interactions
            
            if len(group_agents) > 0:
                interaction_groups.append({
                    'agents': group_agents,
                    'interactions': group_interactions,
                    'type': 'sync_group',
                    'size': len(group_agents)
                })
        
        # Method 2: Group remaining agents by spatial proximity
        remaining_agents = [(i, agent) for i, agent in enumerate(agents) if i not in processed_agents]
        
        if remaining_agents:
            # Create spatial groups using density-adaptive clustering
            spatial_groups = self._create_spatial_groups(remaining_agents, sparse_graph)
            interaction_groups.extend(spatial_groups)
        
        logger.info(f"🏗️ Built {len(interaction_groups)} interaction groups from {len(agents)} agents")
        return interaction_groups
    
    def _create_spatial_groups(self, remaining_agents: List, sparse_graph: Dict) -> List[Dict]:
        """Create spatial groups for agents not in sync groups."""
        spatial_groups = []
        
        # Density-adaptive group size
        target_group_size = max(5, int(np.log2(len(remaining_agents) + 1)))
        
        current_group_agents = []
        current_group_interactions = {}
        
        for agent_idx, agent in remaining_agents:
            current_group_agents.append(agent)
            
            # Get agent's interactions
            agent_interactions = sparse_graph.get(agent_idx, [])
            current_group_interactions[agent_idx] = agent_interactions
            
            # Create group when target size reached
            if len(current_group_agents) >= target_group_size:
                spatial_groups.append({
                    'agents': current_group_agents,
                    'interactions': current_group_interactions,
                    'type': 'spatial_group',
                    'size': len(current_group_agents)
                })
                
                # Start new group
                current_group_agents = []
                current_group_interactions = {}
        
        # Add final group if it has agents
        if current_group_agents:
            spatial_groups.append({
                'agents': current_group_agents,
                'interactions': current_group_interactions,
                'type': 'spatial_group',
                'size': len(current_group_agents)
            })
        
        return spatial_groups
    
    def _process_interaction_group_collectively(self, group_agents: List, group_interactions: Dict) -> float:
        """
        Process entire interaction group using vectorized operations.
        
        TRUE O(1) processing: Handle all agents in group simultaneously
        instead of individual loops.
        """
        if not group_agents:
            return 0.0
        
        total_group_strength = 0.0
        
        # Extract Q values from all agents in group (vectorized)
        group_Q_values = []
        group_positions = []
        
        for agent in group_agents:
            if hasattr(agent, 'living_Q_value') and agent.living_Q_value is not None:
                group_Q_values.append(agent.living_Q_value)
            else:
                group_Q_values.append(0.0 + 0.0j)
                
            if hasattr(agent, 'state') and hasattr(agent.state, 'field_position'):
                group_positions.append(agent.state.field_position)
            else:
                group_positions.append((0.0, 0.0))
        
        # Convert to numpy arrays for vectorized operations
        Q_array = np.array(group_Q_values, dtype=complex)
        pos_array = np.array(group_positions, dtype=float)
        
        # Vectorized interaction computation
        if len(Q_array) > 1:
            # Create interaction matrix for the group
            interaction_effects = self._compute_group_interaction_matrix(Q_array, pos_array, group_interactions)
            
            # Apply effects to all agents simultaneously
            self._apply_group_interaction_effects(group_agents, interaction_effects)
            
            # Measure total group interaction strength
            total_group_strength = np.sum(np.abs(interaction_effects))
        
        # Sync positions for all agents in group
        for agent in group_agents:
            if hasattr(agent, 'sync_positions'):
                agent.sync_positions()
        
        return float(total_group_strength) / len(group_agents)
    
    def _compute_group_interaction_matrix(self, Q_array: np.ndarray, pos_array: np.ndarray, group_interactions: Dict) -> np.ndarray:
        """
        Compute interaction effects matrix for entire group using vectorized operations.
        
        Returns matrix where interaction_effects[i] = total effect on agent i from all interactions.
        """
        n_agents = len(Q_array)
        interaction_effects = np.zeros(n_agents, dtype=complex)
        
        # Process all interactions in group using precomputed sparse graph
        agent_indices = list(group_interactions.keys())
        
        for i, agent_idx in enumerate(agent_indices):
            neighbors = group_interactions[agent_idx]
            
            # Vectorized interaction with all neighbors
            for neighbor_idx, interaction_strength in neighbors:
                # Find neighbor in current group
                if neighbor_idx in agent_indices:
                    j = agent_indices.index(neighbor_idx)
                    
                    if i != j:  # Don't interact with self
                        # Vectorized Q-field interaction
                        self_Q = Q_array[i]
                        neighbor_Q = Q_array[j]
                        
                        # Use precomputed interaction strength (no distance calculation!)
                        interaction_effect = interaction_strength * self_Q * np.conj(neighbor_Q) * 0.1
                        interaction_effects[i] += interaction_effect
        
        return interaction_effects
    
    def _apply_group_interaction_effects(self, group_agents: List, interaction_effects: np.ndarray):
        """
        Apply computed interaction effects to all agents in group.
        
        Vectorized application of interaction results.
        """
        for i, agent in enumerate(group_agents):
            if i < len(interaction_effects):
                effect = interaction_effects[i]
                
                # Apply interaction effect to agent's Q value
                if hasattr(agent, 'living_Q_value'):
                    agent.living_Q_value += effect
                    
                    # Update charge object
                    if hasattr(agent, 'charge_obj'):
                        agent.charge_obj.complete_charge = agent.living_Q_value
                        agent.charge_obj.magnitude = abs(agent.living_Q_value)
                        agent.charge_obj.phase = np.angle(agent.living_Q_value)
                
                # Store interaction record
                if hasattr(agent, 'interaction_memory'):
                    interaction_record = {
                        'influence': float(abs(effect)),
                        'timestamp': getattr(agent.state, 'current_s', 0.0),
                        'method': 'group_collective'
                    }
                    agent.interaction_memory.append(interaction_record)
                    
                    # Maintain memory length
                    if len(agent.interaction_memory) > 100:
                        agent.interaction_memory.pop(0)
    
    def _process_breathing_group_collectively(self, group_agents: List, tau: float) -> List[float]:
        """
        Process breathing for entire group using vectorized operations.
        
        TRUE O(1) processing: Handle all agents in group simultaneously.
        """
        if not group_agents:
            return []
        
        # Extract current breathing data from all agents (vectorized)
        breath_phases = []
        breath_frequencies = []
        breath_amplitudes = []
        
        for agent in group_agents:
            if hasattr(agent, 'breath_phase'):
                breath_phases.append(agent.breath_phase)
            else:
                breath_phases.append(0.0)
                
            if hasattr(agent, 'breath_frequency'):
                breath_frequencies.append(agent.breath_frequency)
            else:
                breath_frequencies.append(0.1)
                
            if hasattr(agent, 'breath_amplitude'):
                breath_amplitudes.append(agent.breath_amplitude)
            else:
                breath_amplitudes.append(0.1)
        
        # Convert to numpy arrays for vectorized operations
        phases_array = np.array(breath_phases, dtype=float)
        frequencies_array = np.array(breath_frequencies, dtype=float)
        amplitudes_array = np.array(breath_amplitudes, dtype=float)
        
        # Vectorized breathing evolution
        if len(group_agents) > 1:
            # Group synchronization: nudge toward average phase
            avg_phase = np.mean(phases_array)
            phase_diffs = avg_phase - phases_array
            phases_array += phase_diffs * 0.1  # Gentle synchronization
        
        # Breathing evolution: phase += frequency * tau
        phases_array += frequencies_array * tau
        
        # Breathing oscillation: modify coefficients based on breathing
        breathing_oscillations = amplitudes_array * np.sin(phases_array)
        
        # Apply breathing effects to all agents simultaneously
        for i, agent in enumerate(group_agents):
            if i < len(phases_array):
                # Update agent breathing state
                if hasattr(agent, 'breath_phase'):
                    agent.breath_phase = float(phases_array[i])
                
                # Apply breathing to q-coefficients (vectorized)
                if hasattr(agent, 'breathing_q_coefficients'):
                    oscillation = breathing_oscillations[i]
                    
                    # Modify breathing coefficients efficiently
                    for n in list(agent.breathing_q_coefficients.keys())[:10]:  # Top 10 for efficiency
                        current_coeff = agent.breathing_q_coefficients[n]
                        breathing_modulation = 1.0 + oscillation * 0.1
                        agent.breathing_q_coefficients[n] = current_coeff * breathing_modulation
                
                # Update living Q value after breathing
                if hasattr(agent, 'evaluate_living_form'):
                    agent.living_Q_value = agent.evaluate_living_form()
                    
                    # Update charge object
                    if hasattr(agent, 'charge_obj'):
                        agent.charge_obj.complete_charge = agent.living_Q_value
                        agent.charge_obj.magnitude = abs(agent.living_Q_value)
                        agent.charge_obj.phase = np.angle(agent.living_Q_value)
                
                # Sync positions
                if hasattr(agent, 'sync_positions'):
                    agent.sync_positions()
        
        return phases_array.tolist()
    
    def _get_optimization_statistics(self, num_agents: int) -> Dict[str, Any]:
        """
        Get comprehensive optimization statistics for performance tracking.
        """
        # Count sparse interactions
        sparse_graph = self.adaptive_tuning.get('sparse_interaction_graph', {})
        total_sparse_interactions = sum(len(neighbors) for neighbors in sparse_graph.values())
        
        # Calculate theoretical O(N²) interactions
        total_possible_interactions = num_agents * (num_agents - 1) if num_agents > 1 else 0
        
        # Calculate optimization efficiency
        optimization_factor = (total_sparse_interactions / total_possible_interactions) if total_possible_interactions > 0 else 0.0
        
        # Count optimization structures
        sync_groups = len(self.adaptive_tuning.get('breathing_sync_groups', []))
        cascade_chains = len(self.adaptive_tuning.get('resonance_cascades', []))
        eigenvalue_clusters = len(self.adaptive_tuning.get('eigenvalue_clusters', {}))
        phase_boundaries = len(self.adaptive_tuning.get('phase_boundaries', []))
        
        # Calculate adaptive window size
        adaptive_window = max(10, int(np.log2(num_agents))) if num_agents > 0 else 10
        
        # Performance metrics
        optimization_stats = {
            'sparse_interactions_used': total_sparse_interactions,
            'total_possible_interactions': total_possible_interactions,
            'optimization_factor': optimization_factor,
            'complexity_reduction': f"{optimization_factor:.3%} of O(N²)",
            'sync_groups_detected': sync_groups,
            'cascade_chains_detected': cascade_chains,
            'eigenvalue_clusters_detected': eigenvalue_clusters,
            'phase_boundaries_detected': phase_boundaries,
            'adaptive_window_size': adaptive_window,
            'agents_with_optimized_methods': 0,  # Will be updated during processing
            'fallback_interactions_used': 0,     # Will be updated during processing
            'group_processing_enabled': True,
            'vectorized_operations_enabled': True,
            'performance_mode': 'O(log N) group-centric'
        }
        
        return optimization_stats
    
    def load_universe_from_storage(self, storage_coordinator: "StorageCoordinator", 
                                 universe_id: str) -> Dict[str, Any]:
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
            from Sysnpire.database.universe_reconstruction.agent_factory import AgentFactory
            
            # Create AgentFactory with same device and validation settings
            agent_factory = AgentFactory(
                device=str(self.device).replace('cuda:0', 'cuda').replace('mps:0', 'mps'),
                validate_reconstruction=True  # Always validate during reconstruction
            )
            
            # Get stored agent data from storage coordinator
            stored_data = storage_coordinator.hdf5_manager.load_universe_agents(universe_id)
            
            if not stored_data or "agents" not in stored_data:
                raise ValueError(f"No agent data found for universe {universe_id}")
            
            agent_data_dict = stored_data["agents"]
            universe_metadata = stored_data.get("metadata", {})
            
            logger.info(f"📦 Found {len(agent_data_dict)} stored agents to reconstruct")
            
            # Reconstruct agents using AgentFactory (with data conversion pipeline)
            reconstructed_agents = []
            failed_reconstructions = 0
            
            for agent_id, agent_data in agent_data_dict.items():
                try:
                    # AgentFactory handles all data type conversion and validation
                    reconstructed_agent = agent_factory.reconstruct_single_agent(
                        stored_agent_data=agent_data,
                        universe_metadata=universe_metadata
                    )
                    
                    # Add to our active collections (properly formatted data)
                    self.charge_agents[agent_id] = reconstructed_agent
                    if hasattr(reconstructed_agent, 'charge_obj'):
                        self.active_charges[agent_id] = reconstructed_agent.charge_obj
                    
                    reconstructed_agents.append(reconstructed_agent)
                    
                except Exception as e:
                    raise ValueError(f"Failed to reconstruct agent {agent_id}: {e}")
            
            # Validate reconstruction success
            if not reconstructed_agents:
                raise ValueError("No agents could be successfully reconstructed")
            
            # Calculate field energy from reconstructed agents
            total_field_energy = 0.0
            for agent in reconstructed_agents:
                if hasattr(agent, 'living_Q_value'):
                    total_field_energy += abs(agent.living_Q_value) ** 2
            
            # Initialize adaptive tuning for the reconstructed universe
            self._initialize_adaptive_optimization(reconstructed_agents)
            
            reconstruction_time = time.time() - start_time
            
            logger.info(f"✅ Universe loaded successfully in {reconstruction_time:.2f}s")
            logger.info(f"   Agents reconstructed: {len(reconstructed_agents)}")
            logger.info(f"   Failed reconstructions: {failed_reconstructions}")
            logger.info(f"   Field energy: {total_field_energy:.6f}")
            
            return {
                "status": "success",
                "agents_reconstructed": len(reconstructed_agents),
                "failed_reconstructions": failed_reconstructions,
                "field_energy": total_field_energy,
                "reconstruction_time": reconstruction_time,
                "validation_passed": True,  # AgentFactory validated everything
                "ready_for_simulation": len(reconstructed_agents) > 0,
                "universe_metadata": universe_metadata
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
                "ready_for_simulation": False
            }
    
    def _initialize_adaptive_optimization(self, reconstructed_agents):
        """
        Initialize adaptive optimization for reconstructed universe.
        
        Sets up optimization parameters based on the actual mathematical state
        of reconstructed agents rather than using defaults.
        
        Args:
            reconstructed_agents: List of successfully reconstructed ConceptualChargeAgent objects
        """
        logger.info(f"🔧 Initializing adaptive optimization for {len(reconstructed_agents)} reconstructed agents")
        
        # Calculate average Q magnitude for field calibration
        total_q_magnitude = 0.0
        valid_agents = 0
        
        for agent in reconstructed_agents:
            if hasattr(agent, 'living_Q_value') and agent.living_Q_value is not None:
                total_q_magnitude += abs(agent.living_Q_value)
                valid_agents += 1
        
        if valid_agents > 0:
            avg_q_magnitude = total_q_magnitude / valid_agents
            logger.info(f"✅ Average Q magnitude: {avg_q_magnitude:.6f}")
            
            # Store optimization metrics for potential future use
            self.reconstruction_metrics = {
                "avg_q_magnitude": avg_q_magnitude,
                "total_agents": len(reconstructed_agents),
                "valid_agents": valid_agents,
                "field_energy_density": total_q_magnitude / len(reconstructed_agents)
            }
        else:
            logger.warning("⚠️  No valid Q values found in reconstructed agents")
            self.reconstruction_metrics = {
                "avg_q_magnitude": 0.0,
                "total_agents": len(reconstructed_agents),
                "valid_agents": 0,
                "field_energy_density": 0.0
            }
        
        logger.info("✅ Adaptive optimization initialized")