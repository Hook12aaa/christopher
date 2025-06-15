"""
Simple test script to validate optimization decorators work correctly.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Ensure project imports work
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.utils.performance_optimizers import (
    cupy_optimize, jax_optimize, numba_jit, auto_optimize, profile_decorator,
    get_optimization_status, get_performance_summary, log_optimization_status
)

def test_basic_decorators():
    """Test basic decorator functionality."""
    print("🧪 Testing Optimization Decorators")
    print("=" * 50)
    
    # Log optimization library status
    log_optimization_status()
    
    # Test data
    size = 1000
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    
    print(f"\n📊 Testing with {size}x{size} matrices")
    
    # Test profile decorator
    @profile_decorator(name="matrix_mult_baseline")
    def matrix_multiply_baseline(x, y):
        return np.dot(x, y)
    
    # Test numba decorator
    @numba_jit(profile=True)
    def matrix_multiply_numba(x, y):
        return np.dot(x, y)
    
    # Test jax decorator  
    @jax_optimize(profile=True)
    def matrix_multiply_jax(x, y):
        return np.dot(x, y)
    
    # Test cupy decorator
    @cupy_optimize(profile=True)
    def matrix_multiply_cupy(x, y):
        return np.dot(x, y)
    
    # Test auto optimization
    @auto_optimize(profile=True)
    def matrix_multiply_auto(x, y):
        return np.dot(x, y)
    
    print("\n🔄 Running tests...")
    
    try:
        # Run baseline
        print("Testing baseline...")
        result1 = matrix_multiply_baseline(a, b)
        print(f"✅ Baseline completed: {result1.shape}")
        
        # Run numba
        print("Testing Numba...")
        result2 = matrix_multiply_numba(a, b)
        print(f"✅ Numba completed: {result2.shape}")
        
        # Run JAX
        print("Testing JAX...")
        result3 = matrix_multiply_jax(a, b)
        print(f"✅ JAX completed: {result3.shape}")
        
        # Run CuPy
        print("Testing CuPy...")
        result4 = matrix_multiply_cupy(a, b)
        print(f"✅ CuPy completed: {result4.shape}")
        
        # Run auto
        print("Testing Auto-optimize...")
        result5 = matrix_multiply_auto(a, b)
        print(f"✅ Auto-optimize completed: {result5.shape}")
        
        # Check results are similar
        print("\n🔍 Validating results...")
        tolerance = 1e-4
        if np.allclose(result1, result2, atol=tolerance):
            print("✅ Numba result matches baseline")
        else:
            print("❌ Numba result differs from baseline")
            
        if np.allclose(result1, result3, atol=tolerance):
            print("✅ JAX result matches baseline")
        else:
            print("❌ JAX result differs from baseline")
            
        if np.allclose(result1, result4, atol=tolerance):
            print("✅ CuPy result matches baseline")
        else:
            print("❌ CuPy result differs from baseline")
            
        if np.allclose(result1, result5, atol=tolerance):
            print("✅ Auto-optimize result matches baseline")
        else:
            print("❌ Auto-optimize result differs from baseline")
        
        # Show performance summary
        print("\n📈 Performance Summary:")
        summary = get_performance_summary()
        if summary:
            for func_name, stats in summary.items():
                print(f"  • {func_name}: {stats['avg_speedup']:.2f}x speedup")
        else:
            print("  No performance data available")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎉 Test completed!")


def test_simple_operations():
    """Test with simpler operations to ensure decorators work."""
    print("\n🔬 Testing Simple Operations")
    print("-" * 30)
    
    # Simple vector operations
    @profile_decorator(name="vector_ops")
    def vector_operations(x):
        return np.sum(x**2) + np.mean(x) * np.std(x)
    
    @numba_jit(profile=True) 
    def vector_operations_numba(x):
        return np.sum(x**2) + np.mean(x) * np.std(x)
    
    # Test data
    x = np.random.randn(10000).astype(np.float32)
    
    print("Testing vector operations...")
    result1 = vector_operations(x)
    result2 = vector_operations_numba(x)
    
    print(f"Baseline result: {result1:.6f}")
    print(f"Numba result: {result2:.6f}")
    print(f"Results match: {np.isclose(result1, result2)}")


if __name__ == "__main__":
    test_basic_decorators()
    test_simple_operations()