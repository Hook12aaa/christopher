# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mathematical Foundation

This codebase implements the complete Field Theory of Social Constructs as defined in the research paper. **Critical**: All development must respect the mathematical formulations.

### The Complete Conceptual Charge Formula

The core implementation centers on this complete mathematical formulation:

```LaTexcl

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

1. **Model Layer** (`/Sysnpire/model/`)
   - `ConceptualCharge`: Complete Q(τ, C, s) implementation with field-theoretic mathematics
   - `ChargeFactory`: Transforms embeddings into dynamic conceptual charges using complete field theory
   - Dimensions: `/semantic_dimension/`, `/temporal_dimension/`, `/emotional_dimension/`, `/shared_components/`
   - `FoundationManifoldBuilder`: Orchestrates transformation from foundation models (BGE, MPNet) to universal manifold
   - Initial ingestion: `/initial/bge_ingestion.py`, `/initial/mpnet_ingestion.py`

2. **Database Layer** (`/Sysnpire/database/`)
   - `FieldUniverse`: Main orchestrator for tensor-native field-theoretic manifold storage
   - Lance+Arrow tensor storage with spatial indexing (primary storage)
   - Redis hot cache for frequently accessed field regions
   - `ConceptualChargeObject`: Storage representation with field components
   - Abstraction layer: intake processor and charge transformer
   - DuckDB SQL interface to Lance datasets
   - Spatial indexing via Hilbert encoding for field proximity queries

3. **Processing Pipeline** (`/Sysnpire/`)
   - **Intake**: Foundation model embeddings → `ChargeFactory` 
   - **Transformation**: Static embeddings → Dynamic field charges via Q(τ, C, s)
   - **Storage**: Tensor-native storage in Lance with field positioning
   - **Query**: Field-theoretic proximity and trajectory analysis

### Key Architecture Differences

**ChargeFactory Design:**

- Model-agnostic: Works with BGE, MPNet, or any embedding source
- Two modes: `from_base=True` (foundation model) vs integration mode
- Focused responsibility: Pure mathematical transformation to Q(τ, C, s)
- No data sourcing - receives preprocessed embeddings with mathematical properties

**FoundationManifoldBuilder:**

- Orchestrates complete pipeline from foundation model to field universe
- Handles model selection, loading, validation, and enrichment
- Integrates with `ChargeFactory` for mathematical transformation
- Supports BGE Large v1.5 and MPNet base models

**FieldUniverse Storage:**

- Tensor-native architecture (no legacy SQL)
- Lance+Arrow for primary tensor storage with spatial indexing
- Redis cache for hot field regions and frequently accessed charges
- Abstraction layer for validation and optimization
- Batch processing for efficient field computation

### What TO Do

- Use actual model inference (BGE embeddings, etc.)
- Implement complete mathematical formulations
- Test field-theoretic properties
- Maintain complex-valued calculations where specified
- Reuse existing mathematical components

### What NOT To Do

- Never use `np.random` or simulated values for mathematical parameters
- Don't create simplified test versions that bypass the actual system
- Avoid duplicating mathematical implementations across modules
- Don't use static emotional categories or fixed semantic labels
- Never approximate field calculations unless explicitly required for performance

### Module Structure and Context

Each module contains mathematical context files (`readme.md`) that detail the specific paper sections implemented. Refer to these for component-specific mathematical requirements and theoretical foundations.

### Performance Considerations

- Trajectory integration can be computationally intensive - use caching for repeated calculations
- Complex field calculations may require numerical precision considerations