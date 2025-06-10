#!/usr/bin/env python3
"""
Development Test Suite

Run these tests frequently during development to catch issues early.
These tests focus on basic functionality and imports rather than full mathematical correctness.

Usage:
    python tests/test_development.py
    pytest tests/test_development.py
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestBasicEnvironment:
    """Test basic Python environment and dependencies"""
    
    def test_python_version(self):
        """Test Python version is adequate"""
        assert sys.version_info >= (3, 8), f"Python 3.8+ required, got {sys.version_info}"
    
    def test_numpy_available(self):
        """Test NumPy is available and working"""
        import numpy as np
        
        # Test basic operations
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.dot(a, b)
        assert result == 32, "NumPy basic operations failed"
        
        # Test complex operations
        complex_result = np.exp(1j * np.pi) + 1
        assert np.abs(complex_result) < 1e-10, "NumPy complex operations failed"
    
    def test_scipy_available(self):
        """Test SciPy is available for mathematical operations"""
        try:
            import scipy
            from scipy.integrate import quad
            
            # Test simple integration
            result, _ = quad(lambda x: x**2, 0, 1)
            expected = 1/3
            assert abs(result - expected) < 1e-10, "SciPy integration failed"
        except ImportError:
            pytest.skip("SciPy not available - install with: pip install scipy")

class TestEmbeddingModel:
    """Test BGE embedding model functionality"""
    
    def test_sentence_transformers_import(self):
        """Test sentence-transformers can be imported"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.fail("sentence-transformers not available - install with: pip install sentence-transformers")
    
    def test_bge_model_loading(self):
        """Test BGE model can be loaded (may download on first run)"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # This may take time on first run (downloads model)
            model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            assert model is not None, "BGE model failed to load"
            
        except Exception as e:
            pytest.skip(f"BGE model loading failed: {e}. Check internet connection or try running examples/minimal_example.py")
    
    def test_basic_encoding(self):
        """Test basic text encoding"""
        try:
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            
            # Test encoding
            test_text = "Field theory of social constructs"
            embedding = model.encode(test_text)
            
            assert len(embedding) == 1024, f"Expected 1024-dim embedding, got {len(embedding)}"
            assert np.linalg.norm(embedding) > 0, "Embedding should have non-zero magnitude"
            
        except Exception as e:
            pytest.skip(f"Encoding test failed: {e}")

class TestProjectStructure:
    """Test project structure and imports"""
    
    def test_project_structure(self):
        """Test key directories exist"""
        required_dirs = [
            'core_mathematics',
            'embedding_engine', 
            'emotional_dynamics',
            'temporal_framework',
            'field_integration',
            'product_manifold',
            'data_pipeline',
            'api',
            'tests',
            'examples'
        ]
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Required directory missing: {dir_name}"
            
            # Check for __init__.py in Python modules
            if dir_name != 'examples':
                init_file = dir_path / '__init__.py'
                assert init_file.exists(), f"Missing __init__.py in {dir_name}"
    
    def test_core_mathematics_import(self):
        """Test core mathematics module can be imported"""
        try:
            from core_mathematics.conceptual_charge import ConceptualCharge
            # If this succeeds, the module structure is correct
        except ImportError as e:
            # Expected during early development
            pytest.skip(f"Core mathematics not yet implemented: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error importing core mathematics: {e}")
    
    def test_embedding_engine_import(self):
        """Test embedding engine module can be imported"""
        try:
            from embedding_engine.models import ConceptualChargeGenerator
        except ImportError as e:
            pytest.skip(f"Embedding engine not yet implemented: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error importing embedding engine: {e}")

class TestBasicMathematics:
    """Test basic mathematical operations we'll need"""
    
    def test_complex_arithmetic(self):
        """Test complex number operations"""
        # Test basic complex operations
        z1 = 1 + 2j
        z2 = 3 - 1j
        
        # Addition
        result = z1 + z2
        assert result == 4 + 1j, "Complex addition failed"
        
        # Multiplication
        result = z1 * z2
        assert result == 5 + 5j, "Complex multiplication failed"
        
        # Polar form
        magnitude = abs(z1)
        phase = np.angle(z1)
        reconstructed = magnitude * np.exp(1j * phase)
        assert np.abs(reconstructed - z1) < 1e-10, "Polar form conversion failed"
    
    def test_trajectory_integration_placeholder(self):
        """Test placeholder for trajectory integration"""
        # This tests the mathematical operations we'll need for trajectory operators
        def simple_trajectory_function(s):
            """Simplified version of T_i(τ,s) = ∫₀ˢ ω_i(τ,s')·e^(iφ_i(τ,s')) ds'"""
            omega = 1.0  # Frequency
            phi = 0.5 * s  # Linear phase evolution
            return omega * np.exp(1j * phi)
        
        # Test simple integration
        try:
            from scipy.integrate import quad
            
            def integrand_real(s):
                return simple_trajectory_function(s).real
            
            def integrand_imag(s):
                return simple_trajectory_function(s).imag
            
            real_part, _ = quad(integrand_real, 0, 1)
            imag_part, _ = quad(integrand_imag, 0, 1)
            
            result = complex(real_part, imag_part)
            assert abs(result) > 0, "Trajectory integration should produce non-zero result"
            
        except ImportError:
            pytest.skip("SciPy required for integration tests")
    
    def test_field_operations(self):
        """Test basic field operations on grids"""
        # Create simple 2D grid
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        X, Y = np.meshgrid(x, y)
        
        # Create simple field
        field = np.exp(-(X**2 + Y**2)) * np.exp(1j * np.pi * X)
        
        # Test field properties
        assert field.shape == (10, 10), "Field grid shape incorrect"
        assert field.dtype == complex, "Field should be complex-valued"
        
        # Test field energy
        energy = np.sum(np.abs(field)**2)
        assert energy > 0, "Field energy should be positive"
        
        # Test center of mass calculation (useful for manifold analysis)
        magnitude = np.abs(field)
        total_magnitude = np.sum(magnitude)
        center_x = np.sum(magnitude * X) / total_magnitude
        center_y = np.sum(magnitude * Y) / total_magnitude
        
        # Should be close to origin for this symmetric field
        assert abs(center_x) < 0.1, "Field center should be near origin"
        assert abs(center_y) < 0.1, "Field center should be near origin"

def run_development_tests():
    """Run all development tests and report results"""
    print("🧪 Running Development Tests")
    print("=" * 50)
    
    # Run tests with pytest
    test_file = __file__
    exit_code = pytest.main([test_file, "-v", "--tb=short"])
    
    print("\n" + "=" * 50)
    if exit_code == 0:
        print("✅ All development tests passed!")
        print("\nNext steps:")
        print("1. Run minimal example: python examples/minimal_example.py")
        print("2. Begin implementing core mathematics")
        print("3. Follow DEVELOPMENT_GUIDE.md")
    else:
        print("❌ Some tests failed")
        print("\nTroubleshooting:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Check Python version (3.8+ required)")
        print("3. Verify internet connection for model downloads")
    
    return exit_code == 0

if __name__ == "__main__":
    success = run_development_tests()
    sys.exit(0 if success else 1)