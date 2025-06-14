# Sysnpire - Field Theory of Social Constructs

Transform text into mathematical fields that capture meaning, emotion, and semantic relationships. Built on cutting-edge research in conceptual charge theory.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sysnpire.git
cd sysnpire

# Install dependencies
pip install -r requirements.txt
```

### Basic Example

```python
from model.charge_factory import ChargeFactory
from database.field_universe import FieldUniverse

# Initialize the system
factory = ChargeFactory()
universe = FieldUniverse()

# Create a conceptual charge from text
charge = factory.create_charge(
    text="The beauty of mathematics lies in its simplicity",
    observational_state=1.0,  # Current observational state
    gamma=1.2                 # Field calibration factor
)

# Access charge properties
print(f"Magnitude: {charge.magnitude:.4f}")
print(f"Field Position: {charge.field_position}")
print(f"Semantic Field: {charge.semantic_field[:5]}...")  # First 5 components

# Store in universe for persistence
universe.store_charge(charge)
```

### Run the API Server

```bash
# Start the server
python api/main.py

# The API will be available at http://localhost:8080
# Docs at http://localhost:8080/docs
```

### API Usage Examples

```bash
# Create a single charge
curl -X POST http://localhost:8080/charge/create \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Understanding emerges from observation",
    "observational_state": 1.0,
    "gamma": 1.2
  }'

# Batch process multiple texts
curl -X POST http://localhost:8080/charge/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["First concept", "Second concept", "Third concept"],
    "observational_state": 1.0,
    "gamma": 1.2
  }'

# Analyze charge properties
curl -X POST http://localhost:8080/charge/analyze \
  -H "Content-Type: application/json" \
  -d '{"charge_id": "your_charge_id"}'
```

## 🏗️ Project Structure

```
Sysnpire/
├── model/                    # Core mathematical models
│   ├── mathematics/         # Field theory implementations
│   │   ├── conceptual_charge.py    # Q(τ, C, s) formula
│   │   └── theoretical_background.py # Mathematical foundations
│   ├── bge_encoder.py      # BGE-Large embeddings (1024d)
│   ├── charge_factory.py   # High-level charge creation
│   └── field_enhancer.py   # Field parameter extraction
│
├── database/               # Data persistence layer
│   ├── field_universe.py   # SQLite storage with field indexing
│   ├── conceptual_charge_object.py  # Rich charge objects
│   └── manifold_manager.py # Collective charge operations
│
├── api/                    # REST API layer
│   ├── main.py            # FastAPI application
│   ├── routers/           # API endpoints
│   └── api_endpoints/     # Business logic handlers
│
├── dashboard/             # Visualization tools
│   ├── field_visualizer.py # 3D field visualization
│   └── universe_monitor.py # Real-time monitoring
│
└── docs/                  # Documentation
    └── ARCHITECTURE.md    # Detailed architecture guide
```

## 💡 Key Concepts

### What is a Conceptual Charge?

A conceptual charge is a mathematical object that represents text as a dynamic field in high-dimensional space. Unlike simple embeddings, charges capture:

- **Semantic meaning** through 1024-dimensional BGE embeddings
- **Emotional trajectory** based on observational context
- **Phase relationships** between different semantic components
- **Field effects** that influence nearby charges

### Why Use Sysnpire?

- **Rich Representations**: Go beyond flat embeddings to capture nuanced meaning
- **Context-Aware**: Charges evolve based on observational state
- **Field Interactions**: Discover relationships through field resonance
- **Mathematical Rigor**: Based on field theory mathematics from physics

## 🛠️ Development Guide

### Setting Up Your Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt  # If you have dev dependencies
```

### Running Tests

```bash
# Run the test suite
python test_enterprise_system.py

# Expected output:
# ✓ Creating 1000 charges...
# ✓ Throughput: 500+ charges/second
# ✓ Database persistence verified
# ✓ Field calculations accurate
```

### Working with the Codebase

#### Creating Custom Charges

```python
from model.mathematics.conceptual_charge import ConceptualCharge
import numpy as np

# Create a custom charge with specific parameters
charge = ConceptualCharge(
    tau=np.random.randn(1024),  # Semantic vector
    context="scientific_discovery",
    observational_state=2.5,
    gamma=1.0
)

# Compute the complete charge
Q = charge.compute_complete_charge()
print(f"Charge magnitude: {np.abs(Q)}")
print(f"Charge phase: {np.angle(Q)}")
```

#### Extending the API

```python
# Add a new endpoint in api/routers/charges.py
@router.post("/charge/similarity")
async def compute_similarity(charge_ids: List[str]):
    """Compute field resonance between charges"""
    charges = [universe.get_charge(id) for id in charge_ids]
    resonance = compute_field_resonance(charges)
    return {"resonance": resonance}
```

#### Database Operations

```python
from database.field_universe import FieldUniverse
from database.manifold_manager import ManifoldManager

# Advanced queries
universe = FieldUniverse()
manifold = ManifoldManager()

# Find charges in a specific field region
charges = universe.query_by_field_region(x_range=(0, 5), y_range=(-2, 2))

# Compute collective field effects
collective = manifold.compute_collective_field(charge_ids)
print(f"Field strength: {collective['field_strength']}")
print(f"Coherence: {collective['coherence']}")
```

### Visualization Dashboard

```bash
# Run the visualization dashboard
python dashboard/field_visualizer.py

# Monitor system performance
python dashboard/universe_monitor.py
```

## 📊 Core Mathematics

The system implements the complete conceptual charge formula:

```
Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)
```

Each component contributes to the final charge:
- **γ**: Calibrates field strength
- **T**: Captures transformative potential
- **E**: Tracks emotional evolution
- **Φ**: Generates semantic fields
- **e^(iθ)**: Maintains phase coherence
- **Ψ**: Models observational decay

## 🚀 Advanced Features

### Batch Processing
```python
# Process multiple texts efficiently
texts = ["concept one", "concept two", "concept three"]
charges = factory.create_charges_batch(
    texts=texts,
    observational_state=1.0,
    gamma=1.2
)

# Analyze batch statistics
stats = factory.get_batch_statistics(charges)
print(f"Average magnitude: {stats['avg_magnitude']}")
print(f"Field variance: {stats['field_variance']}")
```

### Real-time Streaming
```python
# Stream charges as they're created
async for charge in factory.stream_charges(text_source):
    universe.store_charge(charge)
    if charge.magnitude > threshold:
        await notify_high_energy_charge(charge)
```

### Field Analysis
```python
# Analyze field topology
topology = universe.analyze_field_topology()
print(f"Field curvature: {topology['curvature']}")
print(f"Energy wells: {topology['energy_wells']}")
print(f"Resonance peaks: {topology['resonance_peaks']}")
```

## 📈 Performance Optimization

- **Caching**: BGE embeddings are cached for repeated texts
- **Batch Operations**: Process multiple charges in parallel
- **Async Support**: API endpoints support async operations
- **Field Indexing**: Spatial indexing for fast field queries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8
- Add type hints for new functions
- Document mathematical formulas in docstrings
- Include unit tests for new features

## 📚 Learn More

- **Architecture Guide**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Mathematical Theory**: [database/MANIFOLD_THEORY.md](database/MANIFOLD_THEORY.md)
- **API Documentation**: Run server and visit `/docs`
- **Research Paper**: [Field Theory of Social Constructs](#)

## 🎯 Use Cases

- **Semantic Search**: Find conceptually similar content
- **Emotion Analysis**: Track emotional trajectories in text
- **Content Understanding**: Deep semantic analysis
- **Recommendation Systems**: Field-based similarity matching
- **Research Applications**: Analyze conceptual evolution

---

**Built with ❤️ using Field Theory and Modern Mathematics**