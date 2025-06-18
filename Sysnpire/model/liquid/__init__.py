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
- LiquidOrchestrator: Field-theoretic multi-agent coordinator implementing complete Q(τ, C, s) dynamics (RUNS THE SHOW)
- ConceptualChargeAgent: Individual living Q(τ, C, s) mathematical entities
- FieldMathematics: PyTorch tensor operations for field equations

INTEGRATION FLOW - CASCADING DIMENSIONAL FEEDBACK:
1. ChargeFactory.build() creates combined_results from stages 1, 2, 3
   - Stage 1: Semantic field generation (S_τ(x))
   - Stage 2: Temporal breathing patterns (T(τ,C,s))
   - Stage 3: Emotional conductor (E^trajectory) - Coordinates S+T interactions
   
2. ChargeFactory hands off to LiquidOrchestrator.create_liquid_universe()
   - This is where all dimensions CLASH together
   - Emotional conductor modulates both semantic fields AND temporal patterns
   
3. 🔄 Stage 4: LIQUID DIMENSIONAL INTEGRATION (Feedback Loop)
   - Semantic fields breathe with temporal patterns
   - Emotional conductor modulates field strengths  
   - Temporal trajectories reshape semantic landscapes
   - All dimensions flow together like liquid metal
   
4. LiquidOrchestrator creates living Q(τ,C,s) agents
   - Each agent IS the complete formula in action
   - Agents influence each other through field interference
   - Creates cascading conceptual charge dynamics
   
5. ChargeFactory returns complete results with liquid universe
   - Dimensions continue to feed back into each other
   - Q(τ, C, s) → Field Interference → Manifold Burning

USAGE:
From ChargeFactory: liquid_results = charge_factory.build(embeddings, model_info)
Direct access: liquid_results['agent_pool'] contains living Q(τ,C,s) entities
"""

from .liquid_orchestrator import LiquidOrchestrator
from .conceptual_charge_agent import ConceptualChargeAgent

__all__ = [
    'LiquidOrchestrator',
    'ConceptualChargeAgent'
]