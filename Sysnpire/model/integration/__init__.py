"""
Integration Layer - Complete Charge Assembly (Section 3.1.5)

Synthesizes all dimensions into the complete conceptual charge formula:
Q(τ,C,s) = γ · T(τ,C,s) · E^trajectory(τ,s) · Φ^semantic(τ,s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)

Components:
- charge_assembler.py: Complete charge integration and assembly
- validation_framework.py: Scientific validation and threshold testing
"""

from .charge_assembler import ConceptualChargeAssembler
from .validation_framework import ValidationFramework

__all__ = ['ConceptualChargeAssembler', 'ValidationFramework']