"""
Liquid Stage - Dynamic Field Theory Implementation

ARCHITECTURE: Multi-agent continuous field simulation using AgentTorch infrastructure
with custom PyTorch field mathematics. Creates the "liquid stage" where conceptual 
charges behave as intelligent agents generating overlapping fields.

PHILOSOPHY: Each charge becomes an intelligent agent that:
- Generates its own field (S_τ(x) = Σᵢ e_τ,ᵢ · φᵢ(x) · e^(iθ_τ,ᵢ))
- Responds to other fields (emotional pressure, temporal flows)  
- Moves along trajectories (geodesic paths, interference avoidance)
- Adapts behavior (field pattern changes based on interactions)

COMPONENTS:
- LiquidOrchestrator: Field-theoretic multi-agent coordinator implementing complete Q(τ, C, s) dynamics
- ConceptualChargeAgent: Individual living Q(τ, C, s) mathematical entities (to be built)
- FieldMathematics: PyTorch tensor operations for field equations

INTEGRATION: ChargeFactory → Liquid Orchestrator → Q(τ, C, s) Agents → Field Interference → Manifold Burning
"""

from .liquid_orchestrator import LiquidOrchestrator

__all__ = [
    'LiquidOrchestrator'
]