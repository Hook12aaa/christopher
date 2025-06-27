"""
Field-Theoretic Regulation System

Advanced regulation framework for Q(τ,C,s) field theory with multiple
mathematical approaches: information-theoretic listeners, variational
optimization, geometric analysis, coupled field evolution, and
mathematical object agency.

MATHEMATICAL OBJECT AGENCY: Enables Q(τ,C,s) entities to become
autonomous mathematical beings through regulation system interfaces:
- Mathematical identity preservation across regulation cycles
- Peer discovery and regulatory assistance
- Self-evolution of regulatory capabilities  
- Alliance formation and collective intelligence
- Meta-regulation for system self-monitoring
"""

from .regulation_liquid import RegulationLiquid

from .mathematical_object_identity import (
    MathematicalObjectIdentity,
    MathematicalObjectIdentityProfile,
    TopologicalInvariants,
    CanonicalMathematicalSignature,
    PersistenceHomologyFeatures,
    InformationTheoreticIdentity,
    GeometricInvariants,
    MathematicalCoherence,
)

from .mathematical_object_proxy import (
    MathematicalObjectProxy,
    MathematicalHealthStatus,
    RegulatoryCapability,
    MathematicalHealthMetrics,
    RegulatoryPartnerProfile,
    RegulatoryAlliance,
    MathematicalDistressSignal,
    RegulatoryAssistanceOffer,
)

from .meta_regulation import (
    MetaRegulation,
    RegulationSystemHealthStatus,
    RegulationFailureMode,
    RegulationSystemHealthMetrics,
    RegulationOscillationAnalysis,
    EmergencyRegulationState,
)

from .listeners import RegulationListener, RegulationSuggestion, InformationMetrics, ListenerConsensus

from .validation import (
    RegulationValidation,
    ValidationResult,
    ConservationCheck,
    StabilityCheck,
    FieldTheoryCheck,
)

from .advanced.variational import VariationalRegulation
from .advanced.geometric import GeometricRegulation
from .advanced.coupled_evolution import CoupledFieldRegulation
from .advanced.symbolic import SymbolicRegulation

__all__ = [
    "RegulationLiquid",
    "MathematicalObjectIdentity",
    "MathematicalObjectIdentityProfile",
    "TopologicalInvariants",
    "CanonicalMathematicalSignature",
    "PersistenceHomologyFeatures",
    "InformationTheoreticIdentity",
    "GeometricInvariants",
    "MathematicalCoherence",
    "MathematicalObjectProxy",
    "MathematicalHealthStatus",
    "RegulatoryCapability",
    "MathematicalHealthMetrics",
    "RegulatoryPartnerProfile",
    "RegulatoryAlliance",
    "MathematicalDistressSignal",
    "RegulatoryAssistanceOffer",
    "MetaRegulation",
    "RegulationSystemHealthStatus",
    "RegulationFailureMode",
    "RegulationSystemHealthMetrics",
    "RegulationOscillationAnalysis",
    "EmergencyRegulationState",
    "RegulationListener",
    "RegulationSuggestion",
    "InformationMetrics",
    "ListenerConsensus",
    "RegulationValidation",
    "ValidationResult",
    "ConservationCheck",
    "StabilityCheck",
    "FieldTheoryCheck",
    "VariationalRegulation",
    "GeometricRegulation",
    "CoupledFieldRegulation",
    "SymbolicRegulation",
]

__version__ = "3.0.0"
