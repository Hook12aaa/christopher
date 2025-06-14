# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mathematical Foundation

This codebase implements the complete Field Theory of Social Constructs as defined in the research paper. **Critical**: All development must respect the mathematical formulations detailed below.

### The Complete Conceptual Charge Formula

The core implementation centers on this complete mathematical formulation:

```
Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)
```

**Component Definitions:**
- **γ**: Global field calibration factor
- **T(τ, C, s)**: Transformative potential tensor with trajectory integration: `T_i(τ,s) = ∫₀ˢ ω_i(τ,s')·e^(iφ_i(τ,s')) ds'`
- **E^trajectory(τ, s)**: Emotional trajectory integration (NOT static valence/arousal/dominance)
- **Φ^semantic(τ, s)**: Dynamic semantic field generation with breathing patterns
- **e^(iθ_total(τ,C,s))**: Complete phase integration from all sources
- **Ψ_persistence(s-s₀)**: Observational persistence with dual-decay structure

### Key Theoretical Principles

1. **Field-Theoretic Approach**: Conceptual charges are dynamic field generators, not static vector coordinates
2. **Trajectory Dependence**: All components evolve based on accumulated observational experience
3. **Complex-Valued Mathematics**: Uses complex numbers for phase relationships and interference patterns
4. **Observer Contingency**: Meaning emerges through contextual observation, not predetermined categories
5. **Non-Euclidean Geometry**: Field effects create curved semantic landscapes

## Commands

### Installation
```bash
cd constructivist_field_theory
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
pytest

# Run complete field theory tests
pytest tests/test_complete_field_theory.py

# Run with coverage
pytest --cov
```

### Linting and Formatting
```bash
flake8
black .
isort .
```

### Running the API Server
```bash
# Development mode
python api/main.py

# Production mode
hypercorn api.main:app --bind 0.0.0.0:8000
```

## Architecture Overview

### Core Mathematical Components

1. **Core Mathematics** (`/core_mathematics/`)
   - `ConceptualCharge`: Complete Q(τ, C, s) implementation
   - Methods: `trajectory_operator()`, `emotional_trajectory_integration()`, `semantic_field_generation()`, `total_phase_integration()`, `observational_persistence()`, `compute_complete_charge()`

2. **Embedding Engine** (`/embedding_engine/`)
   - `ConceptualChargeGenerator`: Creates charges using BGE-Large-v1.5
   - Supports context, observational_state, and gamma parameters
   - Batch processing with field-theoretic formulation

3. **API Layer** (`/api/`)
   - Field-theoretic endpoints accepting complex mathematical parameters
   - `/charges/generate`: Full Q(τ, C, s) creation
   - `/charges/batch`: Multiple charge generation
   - `/charges/analyze`: Trajectory and field analysis

### Mathematical Implementation Details

#### Trajectory Operators
```python
# Complex integration over observational states
T_i = ∫₀ˢ ω_i(τ,s')·e^(iφ_i(τ,s')) ds'
```

#### Emotional Trajectory Integration
```python
# Gaussian alignment with trajectory accumulation
E_trajectory[i] = α_i * exp(-|v_i - v_E|²/2σ²) * trajectory_accumulation
```

#### Semantic Field Generation
```python
# Breathing constellation patterns
φ_semantic[i] = w_i * T_i * x[i] * breathing_modulation * e^(iθ)
```

#### Phase Integration
```python
# Complete phase synthesis
θ_total = θ_semantic + θ_emotional + θ_temporal + θ_interaction + θ_field
```

#### Observational Persistence
```python
# Dual-decay structure
Ψ = exp(-(s-s₀)²/2σ²) + α*exp(-λ(s-s₀))*cos(β(s-s₀))
```

### Key Data Flow

1. **Text Input** → BGE-Large-v1.5 → **Semantic Vector** (1024d)
2. **Context + Observational State** → **Field Parameters**
3. **Trajectory Integration** → **Dynamic Components**
4. **Phase Synthesis** → **Complete Charge Q(τ, C, s)**
5. **Field Effects** → **Complex-valued Results**

### Development Guidelines

#### Mathematical Accuracy
- All implementations must match the paper's mathematical formulations exactly
- Use complex numbers for field calculations, not just real values
- Implement trajectory integration using scipy.integrate.quad when possible
- Preserve phase relationships in all calculations

#### Field Theory Principles
- Never treat charges as static - all components must evolve with observational state
- Implement breathing patterns for semantic fields
- Use trajectory-aware emotional resonance, not categorical emotions
- Maintain complex-valued field effects throughout

#### API Design
- Accept `context`, `observational_state`, and `gamma` parameters
- Return both magnitude and phase information
- Provide field effect breakdowns for analysis
- Support trajectory evolution through observational state updates

#### Testing Requirements
- Test mathematical components against known theoretical properties
- Verify trajectory dependence (different observational states → different results)
- Confirm complex-valued outputs where expected
- Test field effect relationships and phase coherence

### Important Implementation Notes

- **NO static emotional categories**: Use trajectory-dependent emotional field modulation
- **NO simple vector concatenation**: Implement proper field-theoretic integration
- **NO Euclidean assumptions**: Account for field-induced metric warping
- **Complex mathematics required**: Use `complex` type for field calculations
- **Trajectory integration essential**: All components depend on accumulated observational experience
- **NO simulated or default values**: Never use placeholder, mock, or default values for mathematical parameters. All field theory calculations must use actual computed values from the mathematical formulations. When parameters are needed, they must be explicitly provided or calculated from the theoretical framework.
- **NO simulated model outputs**: Never use np.random, fake embeddings, or simulated model responses. Always use actual model inference from loaded models (BGE, transformers, etc.). If a model is not available, request it be provided rather than creating fake outputs.
- **NO half-fixes or workarounds**: Never create simplified test scripts or bypass actual model loading. All functionality must work with the complete technology stack (TensorFlow, BGE models, etc.). If there are environment issues, they must be properly resolved, not worked around.

### Module Structure and Context

Each module contains mathematical context files (`THEORY.md`) that detail the specific paper sections implemented. Refer to these for component-specific mathematical requirements and theoretical foundations.

### Performance Considerations

- Trajectory integration can be computationally intensive - use caching for repeated calculations
- Complex field calculations may require numerical precision considerations
- Batch processing should maintain mathematical accuracy while optimizing performance
- Consider approximations for real-time applications while preserving core field-theoretic properties