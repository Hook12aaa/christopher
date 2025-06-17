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
        
    def _initialize_field_grid(self) -> torch.Tensor:
        """Initialize spatial grid for field computations."""
        x = torch.linspace(-1, 1, self.field_resolution, device=self.device)
        y = torch.linspace(-1, 1, self.field_resolution, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=-1)
        
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
        Compute interference patterns between Q-field values.
        
        Returns patterns of constructive/destructive interference between
        conceptual charges in the liquid stage.
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
        
        # Compute pairwise interference
        pairs = []
        strengths = []
        phases = []
        distortions = []
        
        for i in range(n_charges):
            for j in range(i + 1, n_charges):
                charge_a = self.active_charges[charge_ids[i]]
                charge_b = self.active_charges[charge_ids[j]]
                
                # Get Q-values
                q_a = charge_a.complete_charge
                q_b = charge_b.complete_charge
                
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
        q_magnitude = torch.abs(self.q_field_values)
        
        return {
            'active_charges': len(self.active_charges),
            'field_energy': float(torch.sum(q_magnitude ** 2)),
            'max_field_strength': float(torch.max(q_magnitude)),
            'mean_field_strength': float(torch.mean(q_magnitude)),
            'field_coverage': float(torch.mean((q_magnitude > 0.01).float())),
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