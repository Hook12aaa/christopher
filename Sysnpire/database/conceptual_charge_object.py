"""
Conceptual Charge Object - Rich field-theoretic entity

Enterprise-grade conceptual charge with complete field properties,
historical states, and relational information.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FieldComponents:
    """Complete field components for a conceptual charge."""

    trajectory_operators: List[complex]
    emotional_trajectory: np.ndarray
    semantic_field: np.ndarray
    phase_total: float
    observational_persistence: float


@dataclass
class FieldMetadata:
    """Metadata for field placement and relationships."""

    field_position: Optional[Tuple[float, ...]] = None
    nearby_charges: List[str] = field(default_factory=list)
    collective_influences: Dict[str, float] = field(default_factory=dict)
    field_region: Optional[str] = None


class ConceptualChargeObject:
    """
    Rich conceptual charge object for universe storage.

    Represents the complete field-theoretic entity with all mathematical
    properties, temporal evolution, and relational context.
    """

    def __init__(
        self,
        charge_id: str,
        text_source: str,
        complete_charge: complex,
        field_components: FieldComponents,
        observational_state: float = 1.0,
        gamma: float = 1.0,
    ):
        """
        Initialize rich conceptual charge object.

        Args:
            charge_id: Unique identifier
            text_source: Original text input
            complete_charge: Q(τ,C,s) - main field value
            field_components: All field computation results
            observational_state: Current s value
            gamma: Field calibration factor
        """
        # Core identity
        self.charge_id = charge_id
        self.text_source = text_source
        self.creation_timestamp = time.time()
        self.last_updated = time.time()

        # Field properties
        self.complete_charge = complete_charge
        self.magnitude = abs(complete_charge)
        self.phase = np.angle(complete_charge)
        self.observational_state = observational_state
        self.gamma = gamma

        # Field components
        self.field_components = field_components

        # Universe placement
        self.metadata = FieldMetadata()

        # Historical evolution
        self.historical_states: List[Dict[str, Any]] = []
        self._record_current_state()

    def _record_current_state(self):
        """Record current state for historical tracking."""
        state = {
            "timestamp": time.time(),
            "observational_state": self.observational_state,
            "complete_charge": {
                "real": self.complete_charge.real,
                "imag": self.complete_charge.imag,
                "magnitude": self.magnitude,
                "phase": self.phase,
            },
            "field_position": self.metadata.field_position,
            "nearby_count": len(self.metadata.nearby_charges),
        }
        self.historical_states.append(state)

    def update_observational_state(self, new_s: float):
        """Update observational state and recompute field values."""
        self.observational_state = new_s
        self.last_updated = time.time()

        # Would trigger recomputation of field values here
        # For now, just record the change
        self._record_current_state()

    def set_field_position(self, position: Tuple[float, ...]):
        """Set position in field-theoretic universe."""
        self.metadata.field_position = position
        self.last_updated = time.time()
        self._record_current_state()

    def add_nearby_charge(self, charge_id: str, influence_strength: float = 1.0):
        """Add a nearby charge with influence strength."""
        if charge_id not in self.metadata.nearby_charges:
            self.metadata.nearby_charges.append(charge_id)
            self.metadata.collective_influences[charge_id] = influence_strength
            self.last_updated = time.time()

    def remove_nearby_charge(self, charge_id: str):
        """Remove a nearby charge."""
        if charge_id in self.metadata.nearby_charges:
            self.metadata.nearby_charges.remove(charge_id)
            self.metadata.collective_influences.pop(charge_id, None)
            self.last_updated = time.time()

    def compute_collective_response(self) -> complex:
        """
        Compute collective response from nearby charges.

        This would implement the collective response mathematics
        from the product manifold theory.
        """
        if not self.metadata.nearby_charges:
            return self.complete_charge

        # Simplified collective response
        total_influence = sum(self.metadata.collective_influences.values())
        if total_influence == 0:
            return self.complete_charge

        # Weight the charge by collective influences
        collective_factor = 1.0 + 0.1 * (total_influence - 1.0)
        return self.complete_charge * collective_factor

    def get_field_evolution(self) -> List[Tuple[float, complex]]:
        """Get temporal evolution of field values."""
        evolution = []
        for state in self.historical_states:
            timestamp = state["timestamp"]
            charge = complex(state["complete_charge"]["real"], state["complete_charge"]["imag"])
            evolution.append((timestamp, charge))
        return evolution

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "charge_id": self.charge_id,
            "text_source": self.text_source,
            "creation_timestamp": self.creation_timestamp,
            "last_updated": self.last_updated,
            "complete_charge": {
                "real": self.complete_charge.real,
                "imag": self.complete_charge.imag,
            },
            "magnitude": self.magnitude,
            "phase": self.phase,
            "observational_state": self.observational_state,
            "gamma": self.gamma,
            "field_components": {
                "trajectory_operators": [
                    {"real": t.real, "imag": t.imag}
                    for t in self.field_components.trajectory_operators
                ],
                "emotional_trajectory": self.field_components.emotional_trajectory.tolist(),
                "semantic_field": [
                    {"real": s.real, "imag": s.imag} for s in self.field_components.semantic_field
                ],
                "phase_total": self.field_components.phase_total,
                "observational_persistence": self.field_components.observational_persistence,
            },
            "metadata": {
                "field_position": self.metadata.field_position,
                "nearby_charges": self.metadata.nearby_charges,
                "collective_influences": self.metadata.collective_influences,
                "field_region": self.metadata.field_region,
            },
            "historical_states": self.historical_states,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptualChargeObject":
        """Create from dictionary (for loading from storage)."""
        # Reconstruct complex values
        complete_charge = complex(data["complete_charge"]["real"], data["complete_charge"]["imag"])

        trajectory_ops = [
            complex(t["real"], t["imag"]) for t in data["field_components"]["trajectory_operators"]
        ]

        semantic_field = np.array(
            [complex(s["real"], s["imag"]) for s in data["field_components"]["semantic_field"]]
        )

        field_components = FieldComponents(
            trajectory_operators=trajectory_ops,
            emotional_trajectory=np.array(data["field_components"]["emotional_trajectory"]),
            semantic_field=semantic_field,
            phase_total=data["field_components"]["phase_total"],
            observational_persistence=data["field_components"]["observational_persistence"],
        )

        # Create object
        obj = cls(
            charge_id=data["charge_id"],
            text_source=data["text_source"],
            complete_charge=complete_charge,
            field_components=field_components,
            observational_state=data["observational_state"],
            gamma=data["gamma"],
        )

        # Restore metadata
        obj.creation_timestamp = data["creation_timestamp"]
        obj.last_updated = data["last_updated"]
        obj.magnitude = data["magnitude"]
        obj.phase = data["phase"]

        obj.metadata.field_position = data["metadata"]["field_position"]
        obj.metadata.nearby_charges = data["metadata"]["nearby_charges"]
        obj.metadata.collective_influences = data["metadata"]["collective_influences"]
        obj.metadata.field_region = data["metadata"]["field_region"]

        obj.historical_states = data["historical_states"]

        return obj

    def __str__(self) -> str:
        """String representation."""
        return (
            f"ConceptualCharge({self.charge_id}, "
            f"magnitude={self.magnitude:.6f}, "
            f"phase={self.phase:.3f}, "
            f"nearby_charges={len(self.metadata.nearby_charges)})"
        )

    def __repr__(self) -> str:
        return self.__str__()
