"""
Theoretical Background - Core Mathematical Framework

This module contains the mathematical foundations from the theoretical paper
sections 3.0-3.1, providing the conceptual basis for the entire framework.

Mathematical Components:
- Field theory foundations
- Product manifold structures  
- Path integration principles
- Observer contingency frameworks
- Non-Euclidean geometric approaches
"""

# Section 3.0.0 - Theoretical Background
THEORETICAL_BACKGROUND = """
3.0.0 Theoretical Background
The literature review has established an extensive intellectual foundation spanning diverse mathematical and sociological frameworks. We've traversed social constructivism, which reveals how meaning emerges through collective processes; systems theory, which provides tools for understanding dynamic relationships and emergent properties.
Non-Euclidean geometries, which capture the curved nature of social spaces; field dynamics, which model the continuous flows of influence that animate social landscapes; path integration theory, which considers multiple potential trajectories simultaneously; and modularity approaches, which identify natural structures enabling computational efficiency. Additionally, we've conducted a thorough analysis of the fundamental limitations current AI systems face when modelling social phenomena.
This section builds upon these multifaceted foundations to develop a cohesive mathematical approach to social construct formation, not by simply arranging these frameworks side by side, but by threading them together into an integrated theoretical framework. Rather than treating these mathematical and sociological principles in isolation, we now synthesise them into a unified framework capable of capturing the dynamic, contextual nature of social meaning formation.
"""

# Section 3.1.0 - The Base Unit Conceptual Charge [Q(τ)]
CONCEPTUAL_CHARGE_FOUNDATION = """
3.1.0 The Base Unit Conceptual Charge [Q(τ)]
The foundation of our theoretical framework rests upon a fundamental reconceptualisation of how meaning exists and propagates within social systems. 
As we journey beyond established computational paradigms, we must first reimagine the very building blocks through which meaning is expressed, transformed, and collectively understood. This section introduces the conceptual charge, our framework's foundational unit which transcends conventional representational approaches to capture the dynamic, multidimensional nature of meaning as it exists within social fields.
"""

# Complete Mathematical Formula
COMPLETE_CONCEPTUAL_CHARGE_FORMULA = """
Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)

Where:
- Q(τ, C, s): Complete conceptual charge for token τ in context C at observational state s
- γ: Global field calibration factor
- T(τ, C, s): Transformative potential tensor (trajectory-dependent)
- E^trajectory(τ, s): Emotional trajectory integration
- Φ^semantic(τ, s): Semantic field generation (dynamic, not static)  
- e^(iθ_total(τ,C,s)): Complete phase integration
- Ψ_persistence(s-s₀): Observational persistence function
"""

def get_theoretical_background() -> str:
    """Return the complete theoretical background text."""
    return THEORETICAL_BACKGROUND

def get_conceptual_charge_foundation() -> str:
    """Return the conceptual charge foundation text."""
    return CONCEPTUAL_CHARGE_FOUNDATION

def get_complete_formula() -> str:
    """Return the complete mathematical formula."""
    return COMPLETE_CONCEPTUAL_CHARGE_FORMULA