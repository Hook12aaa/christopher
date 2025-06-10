"""
Conceptual Charge - Core Implementation (Section 3.1.0)

The foundation of our theoretical framework: the fundamental reconceptualization 
of how meaning exists and propagates within social systems.

Components:
- base_charge.py: Core Q(τ,C,s) implementation
- charge_factory.py: Factory pattern for charge creation
"""

from .base_charge import ConceptualCharge
from .charge_factory import ChargeFactory

__all__ = ['ConceptualCharge', 'ChargeFactory']