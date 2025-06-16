"""
Field Orchestrator - Pure Field Theory Coordination

This orchestrates multiple social construct field systems without any transformer concepts.
Manages field evolution, inter-field coupling, and collective dynamics across 
different conceptual domains.

Mathematical Foundation: Multi-field system with cross-field interactions
∂φᵢ/∂t = -δF/δφᵢ + Σⱼ Γᵢⱼ(φⱼ) + ξᵢ(r,t)

Where φᵢ are different social construct fields, Γᵢⱼ represents inter-field coupling,
and F is the total free energy functional.
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

from .social_construct_field import SocialConstructField, SocialConstructFieldFactory

logger = logging.getLogger(__name__)


class FieldOrchestrator:
    """
    Orchestrates multiple social construct field systems.
    
    This is pure field theory - manages field evolution across different
    conceptual domains with inter-field coupling and collective dynamics.
    NO transformer concepts, NO embedding dimensions, NO attention.
    """
    
    def __init__(self,
                 conceptual_space_dimensions: int,
                 observational_domain_size: float,
                 inter_field_coupling_strength: float):
        """
        Initialize field orchestrator.
        
        Args:
            conceptual_space_dimensions: Dimensionality of conceptual space
            observational_domain_size: Size of observational domain
            inter_field_coupling_strength: Strength of coupling between different fields
        """
        self.conceptual_space_dims = conceptual_space_dimensions
        self.domain_size = observational_domain_size
        self.inter_field_coupling = inter_field_coupling_strength
        
        # Field factory for creating field systems
        self.field_factory = SocialConstructFieldFactory(
            default_conceptual_dimensions=conceptual_space_dimensions,
            default_domain_size=observational_domain_size
        )
        
        # Active field systems
        self.field_systems: Dict[str, SocialConstructField] = {}
        
        # Inter-field coupling matrix
        self.coupling_matrix = {}
        
        # Field evolution history
        self.evolution_history = []
        
        # Global field properties
        self.global_field_energy = 0.0
        self.total_topological_charge = 0.0
        
        logger.info(f"Initialized FieldOrchestrator for {conceptual_space_dimensions}D conceptual space")
    
    def add_field_system(self,
                        field_context: str,
                        field_parameters: Optional[Dict[str, Any]] = None) -> SocialConstructField:
        """
        Add new social construct field system to orchestration.
        
        Args:
            field_context: Context identifier for the field
            field_parameters: Optional field configuration parameters
            
        Returns:
            Created field system
        """
        # Create field system
        field_system = self.field_factory.create_field_system(
            concept_context=field_context,
            field_parameters=field_parameters
        )
        
        # Add to orchestration
        self.field_systems[field_context] = field_system
        
        # Initialize coupling with existing fields
        self._initialize_inter_field_coupling(field_context)
        
        logger.info(f"Added field system: {field_context}")
        return field_system
    
    def _initialize_inter_field_coupling(self, new_field_context: str):
        """
        Initialize coupling between new field and existing fields.
        
        This sets up the interaction terms Γᵢⱼ(φⱼ) in the field equations.
        """
        # Initialize coupling with all existing fields
        for existing_context in self.field_systems.keys():
            if existing_context != new_field_context:
                # Coupling strength based on conceptual similarity
                coupling_strength = self._compute_conceptual_coupling_strength(
                    new_field_context, existing_context
                )
                
                # Bidirectional coupling
                self.coupling_matrix[(new_field_context, existing_context)] = coupling_strength
                self.coupling_matrix[(existing_context, new_field_context)] = coupling_strength
    
    def _compute_conceptual_coupling_strength(self, context1: str, context2: str) -> float:
        """
        Compute coupling strength between two conceptual contexts.
        
        This is based on semantic/conceptual proximity, not transformer similarity.
        Uses field-theoretic principles to determine interaction strength.
        """
        # Simple hash-based coupling (in practice, would use conceptual distance metrics)
        hash1 = hash(context1) % 1000
        hash2 = hash(context2) % 1000
        
        # Coupling strength based on conceptual proximity
        conceptual_distance = abs(hash1 - hash2) / 1000.0
        coupling_strength = self.inter_field_coupling * jnp.exp(-5.0 * conceptual_distance)
        
        return float(coupling_strength)
    
    def evolve_coupled_fields(self, dt: float, num_steps: int = 1):
        """
        Evolve all field systems with inter-field coupling.
        
        Implements coupled field evolution:
        ∂φᵢ/∂t = -δF/δφᵢ + Σⱼ Γᵢⱼ(φⱼ) + ξᵢ(r,t)
        
        Args:
            dt: Time step
            num_steps: Number of evolution steps
        """
        for step in range(num_steps):
            # Compute inter-field coupling terms for all fields
            coupling_terms = self._compute_inter_field_coupling_terms()
            
            # Evolve each field with coupling
            for field_context, field_system in self.field_systems.items():
                # Get coupling term for this field
                coupling_term = coupling_terms.get(field_context, 0.0)
                
                # Evolve field with coupling
                self._evolve_single_field_with_coupling(
                    field_system, coupling_term, dt
                )
            
            # Update global properties
            self._update_global_field_properties()
            
            # Store evolution history
            self._record_evolution_state(step)
    
    def _compute_inter_field_coupling_terms(self) -> Dict[str, jnp.ndarray]:
        """
        Compute inter-field coupling terms Γᵢⱼ(φⱼ) for all fields.
        
        Returns:
            Dictionary mapping field contexts to their coupling terms
        """
        coupling_terms = {}
        
        for field_context, field_system in self.field_systems.items():
            total_coupling = 0.0
            
            # Sum coupling from all other fields
            for other_context, other_field_system in self.field_systems.items():
                if other_context != field_context:
                    # Get coupling strength
                    coupling_key = (field_context, other_context)
                    coupling_strength = self.coupling_matrix.get(coupling_key, 0.0)
                    
                    if coupling_strength > 0:
                        # Compute field coupling term
                        field_coupling = self._compute_field_coupling_term(
                            field_system, other_field_system, coupling_strength
                        )
                        total_coupling += field_coupling
            
            coupling_terms[field_context] = total_coupling
        
        return coupling_terms
    
    def _compute_field_coupling_term(self,
                                   target_field: SocialConstructField,
                                   source_field: SocialConstructField,
                                   coupling_strength: float) -> jnp.ndarray:
        """
        Compute coupling term between two fields.
        
        This implements the interaction term Γᵢⱼ(φⱼ) representing how
        field φⱼ influences field φᵢ.
        """
        # Get field configurations
        target_field_config = jnp.array(target_field.construct_field)
        source_field_config = jnp.array(source_field.construct_field)
        
        # Ensure compatible shapes (might need interpolation in practice)
        if target_field_config.shape != source_field_config.shape:
            # For now, just pad/truncate to match
            min_shape = tuple(min(d1, d2) for d1, d2 in zip(target_field_config.shape, source_field_config.shape))
            target_slice = tuple(slice(0, d) for d in min_shape)
            source_slice = tuple(slice(0, d) for d in min_shape)
            
            target_field_config = target_field_config[target_slice]
            source_field_config = source_field_config[source_slice]
        
        # Compute coupling term - non-linear interaction
        coupling_term = coupling_strength * (
            source_field_config * jnp.conj(target_field_config) +
            jnp.conj(source_field_config) * target_field_config
        )
        
        return coupling_term
    
    def _evolve_single_field_with_coupling(self,
                                         field_system: SocialConstructField,
                                         coupling_term: Union[float, jnp.ndarray],
                                         dt: float):
        """
        Evolve single field system with inter-field coupling.
        
        This modifies the field evolution to include coupling terms.
        """
        # Store original field for coupling calculation
        original_field = field_system.construct_field.copy()
        
        # Evolve field normally
        field_system.evolve_fields(dt, num_steps=1)
        
        # Add coupling term
        if isinstance(coupling_term, (int, float)) and coupling_term == 0:
            # No coupling
            pass
        else:
            # Apply coupling term
            field_system.construct_field += dt * np.array(coupling_term)
    
    def _update_global_field_properties(self):
        """
        Update global properties across all field systems.
        
        Computes total energy, topological charge, etc. across all fields.
        """
        total_energy = 0.0
        total_topo_charge = 0.0
        
        for field_system in self.field_systems.values():
            field_props = field_system.analyze_field_properties()
            total_energy += field_props['total_energy']
            total_topo_charge += field_props['total_topological_charge']
        
        self.global_field_energy = total_energy
        self.total_topological_charge = total_topo_charge
    
    def _record_evolution_state(self, step: int):
        """Record current state in evolution history."""
        state = {
            'step': step,
            'global_energy': self.global_field_energy,
            'total_topological_charge': self.total_topological_charge,
            'field_properties': {}
        }
        
        # Record properties of each field
        for context, field_system in self.field_systems.items():
            state['field_properties'][context] = field_system.analyze_field_properties()
        
        self.evolution_history.append(state)
        
        # Maintain history size
        if len(self.evolution_history) > 1000:
            self.evolution_history = self.evolution_history[-1000:]
    
    def compute_conceptual_charge_field_theory(self,
                                             concept_location: jnp.ndarray,
                                             field_context: str,
                                             observational_state: float) -> complex:
        """
        Compute conceptual charge using pure field theory.
        
        This is the field theory version of conceptual charge computation.
        NO transformer concepts - pure field distortion measurement.
        
        Args:
            concept_location: Position in conceptual space
            field_context: Which field system to use
            observational_state: Current observational state
            
        Returns:
            Complex conceptual charge from field theory
        """
        if field_context not in self.field_systems:
            raise ValueError(f"Field context '{field_context}' not found")
        
        field_system = self.field_systems[field_context]
        
        # Compute charge using field theory
        charge = field_system.compute_conceptual_charge(
            concept_location, observational_state
        )
        
        # Add contributions from inter-field coupling
        coupling_contribution = self._compute_charge_coupling_contribution(
            concept_location, field_context, observational_state
        )
        
        return charge + coupling_contribution
    
    def _compute_charge_coupling_contribution(self,
                                            concept_location: jnp.ndarray,
                                            target_field_context: str,
                                            observational_state: float) -> complex:
        """
        Compute contribution to conceptual charge from inter-field coupling.
        
        This represents how other fields influence the charge at a location.
        """
        total_contribution = 0.0 + 0.0j
        
        target_field = self.field_systems[target_field_context]
        
        for other_context, other_field in self.field_systems.items():
            if other_context != target_field_context:
                # Get coupling strength
                coupling_key = (target_field_context, other_context)
                coupling_strength = self.coupling_matrix.get(coupling_key, 0.0)
                
                if coupling_strength > 0:
                    # Compute other field's contribution
                    other_charge = other_field.compute_conceptual_charge(
                        concept_location, observational_state
                    )
                    
                    # Weight by coupling strength
                    contribution = coupling_strength * other_charge
                    total_contribution += contribution
        
        return total_contribution
    
    def analyze_field_orchestration(self) -> Dict[str, Any]:
        """
        Analyze current state of field orchestration.
        
        Returns:
            Comprehensive analysis of all field systems and their interactions
        """
        analysis = {
            'global_properties': {
                'total_energy': self.global_field_energy,
                'total_topological_charge': self.total_topological_charge,
                'num_field_systems': len(self.field_systems),
                'num_couplings': len(self.coupling_matrix)
            },
            'field_systems': {},
            'coupling_analysis': {},
            'evolution_statistics': {}
        }
        
        # Analyze each field system
        for context, field_system in self.field_systems.items():
            analysis['field_systems'][context] = field_system.analyze_field_properties()
        
        # Analyze coupling strengths
        for (field1, field2), strength in self.coupling_matrix.items():
            coupling_key = f"{field1}-{field2}"
            analysis['coupling_analysis'][coupling_key] = {
                'strength': float(strength),
                'field1': field1,
                'field2': field2
            }
        
        # Evolution statistics
        if self.evolution_history:
            energies = [state['global_energy'] for state in self.evolution_history]
            topo_charges = [state['total_topological_charge'] for state in self.evolution_history]
            
            analysis['evolution_statistics'] = {
                'energy_mean': float(jnp.mean(jnp.array(energies))),
                'energy_std': float(jnp.std(jnp.array(energies))),
                'energy_trend': float(jnp.polyfit(jnp.arange(len(energies), dtype=float), jnp.array(energies), 1)[0]),
                'topological_charge_mean': float(jnp.mean(jnp.array(topo_charges))),
                'evolution_steps': len(self.evolution_history)
            }
        
        return analysis
    
    def get_field_system(self, field_context: str) -> Optional[SocialConstructField]:
        """Get field system for given context."""
        return self.field_systems.get(field_context)
    
    def remove_field_system(self, field_context: str):
        """Remove field system and its couplings."""
        if field_context in self.field_systems:
            # Remove field system
            del self.field_systems[field_context]
            
            # Remove couplings
            coupling_keys_to_remove = [
                key for key in self.coupling_matrix.keys()
                if field_context in key
            ]
            for key in coupling_keys_to_remove:
                del self.coupling_matrix[key]
            
            logger.info(f"Removed field system: {field_context}")


class CollectiveResponseOrchestrator:
    """
    Orchestrates collective response dynamics across multiple field systems.
    
    This manages the collective behavior emerging from field interactions,
    representing social consensus formation, cultural evolution, etc.
    """
    
    def __init__(self, field_orchestrator: FieldOrchestrator):
        """
        Initialize collective response orchestrator.
        
        Args:
            field_orchestrator: Main field orchestrator to work with
        """
        self.field_orchestrator = field_orchestrator
        self.collective_patterns = {}
        self.consensus_states = {}
        self.cultural_evolution_history = []
        
        logger.info("Initialized CollectiveResponseOrchestrator")
    
    def detect_collective_patterns(self) -> Dict[str, Any]:
        """
        Detect collective patterns across all field systems.
        
        This identifies emergent collective behaviors like consensus formation,
        cultural patterns, social movements, etc.
        """
        patterns = {}
        
        # Analyze each field system for collective behavior
        for context, field_system in self.field_orchestrator.field_systems.items():
            field_props = field_system.analyze_field_properties()
            
            # Detect consensus patterns
            consensus_strength = self._compute_consensus_strength(field_system)
            
            # Detect cultural patterns
            cultural_stability = self._compute_cultural_stability(field_system)
            
            # Detect social movement patterns
            movement_activity = self._compute_movement_activity(field_system)
            
            patterns[context] = {
                'consensus_strength': consensus_strength,
                'cultural_stability': cultural_stability,
                'movement_activity': movement_activity,
                'field_properties': field_props
            }
        
        self.collective_patterns = patterns
        return patterns
    
    def _compute_consensus_strength(self, field_system: SocialConstructField) -> float:
        """
        Compute consensus strength from field configuration.
        
        High consensus = low field variation, high collective response.
        """
        field_array = jnp.array(field_system.construct_field)
        response_array = jnp.array(field_system.collective_response)
        
        # Consensus indicated by low field variance, high response magnitude
        field_variance = jnp.var(jnp.abs(field_array))
        response_magnitude = jnp.mean(jnp.abs(response_array))
        
        # Consensus strength (inverse of variance, scaled by response)
        consensus = response_magnitude / (1.0 + field_variance)
        
        return float(consensus)
    
    def _compute_cultural_stability(self, field_system: SocialConstructField) -> float:
        """
        Compute cultural stability from topological charge and field energy.
        
        Stable culture = persistent topological structures, low energy fluctuations.
        """
        # Topological charge indicates stable cultural patterns
        topo_charge = jnp.sum(jnp.abs(field_system.topological_charge_density))
        
        # Field energy stability
        field_props = field_system.analyze_field_properties()
        energy_stability = 1.0 / (1.0 + field_props['total_energy'])
        
        cultural_stability = topo_charge * energy_stability
        
        return float(cultural_stability)
    
    def _compute_movement_activity(self, field_system: SocialConstructField) -> float:
        """
        Compute social movement activity from field dynamics.
        
        Active movements = high field gradients, rapid changes.
        """
        field_array = jnp.array(field_system.construct_field)
        
        # Compute field gradients
        field_gradient = field_system._compute_laplacian(field_array)
        gradient_magnitude = jnp.mean(jnp.abs(field_gradient))
        
        # Movement activity proportional to field dynamics
        movement_activity = gradient_magnitude
        
        return float(movement_activity)
    
    def evolve_collective_dynamics(self, dt: float, num_steps: int = 1):
        """
        Evolve collective dynamics across all field systems.
        
        This manages the collective behavior evolution while fields evolve.
        """
        for step in range(num_steps):
            # Evolve coupled fields
            self.field_orchestrator.evolve_coupled_fields(dt, 1)
            
            # Detect new collective patterns
            current_patterns = self.detect_collective_patterns()
            
            # Update consensus states
            self._update_consensus_states(current_patterns)
            
            # Record cultural evolution
            self._record_cultural_evolution(step, current_patterns)
    
    def _update_consensus_states(self, patterns: Dict[str, Any]):
        """Update consensus states based on current patterns."""
        for context, pattern_data in patterns.items():
            consensus_strength = pattern_data['consensus_strength']
            
            # Update consensus state
            if context not in self.consensus_states:
                self.consensus_states[context] = {
                    'strength': consensus_strength,
                    'stability': 0.0,
                    'formation_time': 0
                }
            else:
                # Update with exponential moving average
                old_strength = self.consensus_states[context]['strength']
                new_strength = 0.9 * old_strength + 0.1 * consensus_strength
                self.consensus_states[context]['strength'] = new_strength
                
                # Update stability (how consistent the consensus is)
                strength_change = abs(new_strength - old_strength)
                stability = self.consensus_states[context]['stability']
                new_stability = 0.95 * stability + 0.05 * (1.0 - strength_change)
                self.consensus_states[context]['stability'] = new_stability
                
                self.consensus_states[context]['formation_time'] += 1
    
    def _record_cultural_evolution(self, step: int, patterns: Dict[str, Any]):
        """Record cultural evolution history."""
        evolution_state = {
            'step': step,
            'patterns': patterns,
            'consensus_states': self.consensus_states.copy(),
            'global_consensus': jnp.mean(jnp.array([
                state['strength'] for state in self.consensus_states.values()
            ])) if self.consensus_states else 0.0
        }
        
        self.cultural_evolution_history.append(evolution_state)
        
        # Maintain history size
        if len(self.cultural_evolution_history) > 500:
            self.cultural_evolution_history = self.cultural_evolution_history[-500:]