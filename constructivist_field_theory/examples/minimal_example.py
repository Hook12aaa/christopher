#!/usr/bin/env python3
"""
Minimal Working Example: Field Theory of Social Constructs

This demonstrates the core concept with simple, working implementations.
Goal: Get something running to validate the approach before building complexity.

Run this first to verify your setup:
    python examples/minimal_example.py
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test that we can import basic scientific computing packages"""
    print("🔍 Testing basic imports...")
    
    try:
        import numpy as np
        import scipy
        print("  ✅ NumPy and SciPy available")
        
        # Test complex math operations
        result = np.exp(1j * np.pi) + 1
        assert np.abs(result) < 1e-10, "Complex exponential test failed"
        print("  ✅ Complex mathematics working")
        
        return True
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_embedding_model():
    """Test BGE model availability"""
    print("\n🔍 Testing embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Try to load BGE model (this might download on first run)
        print("  📥 Loading BGE-Large-v1.5 model (may download first time)...")
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # Test encoding
        test_text = "Field theory of social constructs"
        embedding = model.encode(test_text)
        
        print(f"  ✅ Model loaded, embedding dimension: {len(embedding)}")
        print(f"  ✅ Test embedding magnitude: {np.linalg.norm(embedding):.3f}")
        
        return model, embedding
    except Exception as e:
        print(f"  ❌ Model error: {e}")
        return None, None

class SimpleConceptualCharge:
    """
    Simplified conceptual charge for demonstration.
    
    This implements the basic mathematical structure without full complexity:
    Q = magnitude * e^(i*phase) 
    
    Later we'll expand to: Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)
    """
    
    def __init__(self, text: str, embedding: np.ndarray, observational_state: float = 1.0):
        """
        Initialize simple conceptual charge.
        
        Args:
            text: Source text token
            embedding: Semantic embedding vector
            observational_state: Simple trajectory parameter
        """
        self.text = text
        self.embedding = embedding
        self.observational_state = observational_state
        
        # Simple field parameters (placeholders for full implementation)
        self.gamma = 1.0  # Field calibration
        
    def compute_charge(self) -> complex:
        """
        Compute simplified conceptual charge.
        
        This is a placeholder for the full Q(τ, C, s) computation.
        Currently implements: Q = ||embedding|| * e^(i*phase)
        """
        # Simple magnitude from embedding norm
        magnitude = np.linalg.norm(self.embedding) * self.gamma
        
        # Simple phase from embedding characteristics and observational state
        embedding_mean = np.mean(self.embedding)
        phase = embedding_mean * self.observational_state * np.pi
        
        # Complex charge
        Q = magnitude * np.exp(1j * phase)
        
        return Q
    
    def evolve_observational_state(self, new_state: float):
        """Update observational state (trajectory evolution)"""
        self.observational_state = new_state

def demonstrate_simple_charges():
    """Demonstrate basic conceptual charge creation and computation"""
    print("\n🧪 Testing Simple Conceptual Charges...")
    
    # Get model and test embedding
    model, test_embedding = test_embedding_model()
    if model is None:
        print("  ❌ Cannot demo charges without embedding model")
        return False
    
    try:
        # Create sample texts for testing
        texts = [
            "Jazz improvisation in intimate venues",
            "Corporate culture and creative expression",
            "Community arts festivals and social bonding"
        ]
        
        charges = []
        
        # Create charges for each text
        print("\n  📝 Creating conceptual charges:")
        for text in texts:
            embedding = model.encode(text)
            charge = SimpleConceptualCharge(text, embedding)
            charges.append(charge)
            
            Q = charge.compute_charge()
            print(f"    '{text[:30]}...'")
            print(f"      Charge: {abs(Q):.3f} ∠ {np.angle(Q):.3f}")
        
        # Demonstrate trajectory evolution
        print("\n  🔄 Demonstrating trajectory evolution:")
        charge = charges[0]
        initial_Q = charge.compute_charge()
        
        charge.evolve_observational_state(2.0)
        evolved_Q = charge.compute_charge()
        
        print(f"    Initial charge:  {abs(initial_Q):.3f} ∠ {np.angle(initial_Q):.3f}")
        print(f"    Evolved charge:  {abs(evolved_Q):.3f} ∠ {np.angle(evolved_Q):.3f}")
        print(f"    Change in phase: {np.angle(evolved_Q) - np.angle(initial_Q):.3f}")
        
        # Demonstrate charge interactions (simple interference)
        print("\n  🌊 Demonstrating simple interference:")
        Q1, Q2 = charges[0].compute_charge(), charges[1].compute_charge()
        Q_combined = Q1 + Q2
        Q_incoherent = abs(Q1) + abs(Q2)
        
        print(f"    Charge 1: {abs(Q1):.3f}")
        print(f"    Charge 2: {abs(Q2):.3f}")
        print(f"    Coherent sum: {abs(Q_combined):.3f}")
        print(f"    Incoherent sum: {Q_incoherent:.3f}")
        print(f"    Interference factor: {abs(Q_combined)/Q_incoherent:.3f}")
        
        if abs(Q_combined)/Q_incoherent > 1.1:
            print("    → Constructive interference detected!")
        elif abs(Q_combined)/Q_incoherent < 0.9:
            print("    → Destructive interference detected!")
        else:
            print("    → Neutral interference")
        
        print("  ✅ Simple charge system working!")
        return True
        
    except Exception as e:
        print(f"  ❌ Charge demo error: {e}")
        return False

def demonstrate_simple_manifold():
    """Demonstrate basic manifold concept with simple spatial representation"""
    print("\n🌍 Testing Simple Manifold Concept...")
    
    try:
        # Create a simple 2D spatial grid
        grid_size = 32
        x = np.linspace(-5, 5, grid_size)
        y = np.linspace(-5, 5, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Initialize empty manifold field
        manifold_field = np.zeros((grid_size, grid_size), dtype=complex)
        
        print(f"  📐 Created {grid_size}x{grid_size} spatial grid")
        
        # Get model for testing
        model, _ = test_embedding_model()
        if model is None:
            print("  ❌ Cannot demo manifold without embedding model")
            return False
        
        # Create a few charges and place them on the manifold
        texts = [
            "Artistic collaboration",
            "Community gathering", 
            "Cultural expression"
        ]
        
        positions = [(0, 0), (-2, 2), (2, -1)]  # Simple positions
        
        print("\n  🎯 Placing charges on manifold:")
        for text, (px, py) in zip(texts, positions):
            # Create charge
            embedding = model.encode(text)
            charge = SimpleConceptualCharge(text, embedding)
            Q = charge.compute_charge()
            
            # Simple transformation: charge → spatial imprint
            # This is a placeholder for the full T[Q] transformation operator
            distance = np.sqrt((X - px)**2 + (Y - py)**2)
            spatial_profile = np.exp(-distance**2 / 2) * abs(Q)  # Gaussian profile
            phase_pattern = np.angle(Q) * np.ones_like(X)  # Uniform phase
            
            charge_imprint = spatial_profile * np.exp(1j * phase_pattern)
            manifold_field += charge_imprint
            
            print(f"    '{text}' at ({px}, {py}) with magnitude {abs(Q):.3f}")
        
        # Simple manifold analysis
        total_energy = np.sum(np.abs(manifold_field)**2)
        peak_magnitude = np.max(np.abs(manifold_field))
        
        print(f"\n  📊 Manifold properties:")
        print(f"    Total energy: {total_energy:.3f}")
        print(f"    Peak magnitude: {peak_magnitude:.3f}")
        print(f"    Number of significant points: {np.sum(np.abs(manifold_field) > 0.1 * peak_magnitude)}")
        
        # Find center of mass
        magnitude = np.abs(manifold_field)
        total_magnitude = np.sum(magnitude)
        if total_magnitude > 0:
            center_x = np.sum(magnitude * X) / total_magnitude
            center_y = np.sum(magnitude * Y) / total_magnitude
            print(f"    Center of mass: ({center_x:.2f}, {center_y:.2f})")
        
        print("  ✅ Simple manifold concept working!")
        
        # Return manifold data for potential visualization
        return {
            'field': manifold_field,
            'coordinates': (X, Y),
            'charges': list(zip(texts, positions)),
            'total_energy': total_energy
        }
        
    except Exception as e:
        print(f"  ❌ Manifold demo error: {e}")
        return None

def main():
    """Run complete minimal example"""
    print("🚀 Field Theory of Social Constructs - Minimal Example")
    print("=" * 60)
    
    # Test basic functionality
    success = True
    success &= test_basic_imports()
    success &= demonstrate_simple_charges()
    
    manifold_data = demonstrate_simple_manifold()
    success &= manifold_data is not None
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS: Minimal example completed successfully!")
        print("\nNext steps:")
        print("1. Run full development tests: python tests/test_development.py")
        print("2. Begin implementing full mathematical components")
        print("3. Follow DEVELOPMENT_GUIDE.md for systematic building")
        
        if manifold_data:
            print(f"\n📊 Manifold Summary:")
            print(f"   - {len(manifold_data['charges'])} charges placed")
            print(f"   - Total field energy: {manifold_data['total_energy']:.3f}")
            print(f"   - Grid size: {manifold_data['field'].shape}")
    else:
        print("❌ FAILURE: Some components not working")
        print("\nTroubleshooting:")
        print("1. Check requirements.txt installation")
        print("2. Verify internet connection for model download")
        print("3. Check Python version (requires 3.8+)")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)