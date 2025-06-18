"""
Hybrid Storage Database - Field-Theoretic Liquid Universe Persistence

Complete hybrid storage architecture for burning liquid universe results
into persistent storage and reconstructing them back to living universes.

Architecture:
- Liquid Burning: ConceptualChargeAgent → Persistent storage
- Hybrid Storage: HDF5 (mathematical) + Arrow (metadata) 
- Universe Reconstruction: Persistent → Living liquid universe
- Dynamic Dimensionality: Model-agnostic field handling

Mathematical Foundation:
Preserves complete field equation: ∂M/∂t = ∇²M + F[M] + Σᵢ T[Qᵢ]
"""

# Core data structures
from .conceptual_charge_object import (
    ConceptualChargeObject,
    FieldComponents,
    FieldMetadata,
)

# Main orchestrator
from .field_universe import (
    FieldUniverse,
    FieldUniverseConfig,
    burn_liquid_universe,
    reconstruct_liquid_universe,
)

# Hybrid storage system
from .hybrid_storage import ArrowIndexer, HDF5Manager, StorageCoordinator

# Liquid burning pipeline
from .liquid_burning import (
    AgentSerializer,
    BurningOrchestrator,
    ExtractedLiquidData,
    FieldCompressor,
    LiquidProcessor,
    MathematicalValidator,
)

# Universe reconstruction
from .universe_reconstruction import UniverseReconstructor

__all__ = [
    # Main interface
    "FieldUniverse",
    "FieldUniverseConfig",
    "burn_liquid_universe",
    "reconstruct_liquid_universe",
    # Core data structures
    "ConceptualChargeObject",
    "FieldComponents",
    "FieldMetadata",
    # Liquid burning
    "LiquidProcessor",
    "ExtractedLiquidData",
    "AgentSerializer",
    "FieldCompressor",
    "MathematicalValidator",
    "BurningOrchestrator",
    # Hybrid storage
    "HDF5Manager",
    "ArrowIndexer",
    "StorageCoordinator",
    # Reconstruction
    "UniverseReconstructor",
]
