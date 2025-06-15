"""
Performance Optimization Decorators for Sysnpire Math Operations

This module provides decorators to optimize math-heavy operations using:
- CuPy for GPU acceleration (with Mac compatibility checks)
- JAX for JIT compilation and automatic differentiation 
- Numba for CPU JIT compilation
- Performance profiling and benchmarking

Usage:
    @cupy_optimize
    def matrix_operation(a, b):
        return np.dot(a, b)
    
    @jax_optimize
    def field_calculation(x):
        return np.exp(-x**2) * np.cos(x)
    
    @numba_jit
    def loop_heavy_computation(arr):
        return np.sum(arr**2)
"""

import functools
import time
import warnings
from typing import Any, Callable, Dict, Optional, Union
import numpy as np
from pathlib import Path
import sys

# Ensure project imports work
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.utils.logger import get_logger, SysnpireLogger

base_logger = get_logger(__name__)
logger = SysnpireLogger()

# Optional dependency checks
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    # Check if Mac M1/M2 - CuPy doesn't work well on Mac
    import platform
    if platform.system() == "Darwin":
        logger.log_warning("CuPy detected on macOS - GPU acceleration may not work properly")
        CUPY_AVAILABLE = False
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

try:
    import numba
    from numba import jit as numba_jit_func, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    numba = None


class PerformanceProfiler:
    """Performance profiling utility for tracking optimization effectiveness."""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_function(self, func_name: str, original_time: float, optimized_time: float, 
                        input_shape: Optional[tuple] = None, optimizer: str = "unknown"):
        """Record performance comparison with detailed logging."""
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        
        profile_data = {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'speedup': speedup,
            'input_shape': input_shape,
            'optimizer': optimizer,
            'timestamp': time.time()
        }
        
        if func_name not in self.profiles:
            self.profiles[func_name] = []
        self.profiles[func_name].append(profile_data)
        
        # Enhanced logging with performance metrics
        logger.log_success(f"🚀 {optimizer} optimization for {func_name}: {speedup:.2f}x speedup")
        logger.log_performance(base_logger, f"{optimizer} {func_name}", optimized_time)
        
        if input_shape:
            logger.log_debug(f"📐 Input shape: {input_shape}, Data size: {np.prod(input_shape) if hasattr(input_shape, '__iter__') else input_shape} elements")
        
        # Log memory efficiency if significant improvement
        if speedup > 2.0:
            logger.log_success(f"🎯 Significant optimization achieved: {original_time:.4f}s → {optimized_time:.4f}s")
        elif speedup < 0.9:
            logger.log_warning(f"⚠️  Optimization overhead detected: {original_time:.4f}s → {optimized_time:.4f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary across all profiled functions."""
        summary = {}
        for func_name, profiles in self.profiles.items():
            avg_speedup = np.mean([p['speedup'] for p in profiles])
            best_speedup = max([p['speedup'] for p in profiles])
            summary[func_name] = {
                'avg_speedup': avg_speedup,
                'best_speedup': best_speedup,
                'num_profiles': len(profiles)
            }
        return summary


# Global profiler instance
profiler = PerformanceProfiler()


def benchmark_function(func: Callable, *args, iterations: int = 10, **kwargs) -> float:
    """Benchmark a function by running it multiple times and taking the mean."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times)


def cupy_optimize(fallback_to_numpy: bool = True, profile: bool = True):
    """
    Decorator to optimize numpy operations using CuPy GPU acceleration.
    
    Args:
        fallback_to_numpy: If True, falls back to numpy if CuPy fails
        profile: If True, profiles performance improvement
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not CUPY_AVAILABLE:
                if profile:
                    logger.log_warning(f"CuPy not available for {func.__name__}, using numpy")
                return func(*args, **kwargs)
            
            try:
                # Convert numpy arrays to cupy
                cupy_args = []
                for arg in args:
                    if isinstance(arg, np.ndarray):
                        cupy_args.append(cp.asarray(arg))
                    else:
                        cupy_args.append(arg)
                
                cupy_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, np.ndarray):
                        cupy_kwargs[k] = cp.asarray(v)
                    else:
                        cupy_kwargs[k] = v
                
                # Replace numpy references in the function's global namespace
                func_globals = func.__globals__.copy()
                original_np = func_globals.get('np', np)
                func_globals['np'] = cp
                
                # Create new function with cupy globals
                import types
                cupy_func = types.FunctionType(
                    func.__code__, func_globals, func.__name__, 
                    func.__defaults__, func.__closure__
                )
                
                if profile:
                    # Benchmark original
                    original_time = benchmark_function(func, *args, **kwargs)
                    
                    # Run optimized version
                    start = time.perf_counter()
                    result = cupy_func(*cupy_args, **cupy_kwargs)
                    end = time.perf_counter()
                    optimized_time = end - start
                    
                    # Convert result back to numpy
                    if hasattr(result, 'get'):
                        result = result.get()
                    
                    # Profile performance
                    input_shape = args[0].shape if args and hasattr(args[0], 'shape') else None
                    profiler.profile_function(func.__name__, original_time, optimized_time, 
                                            input_shape, "CuPy")
                else:
                    result = cupy_func(*cupy_args, **cupy_kwargs)
                    if hasattr(result, 'get'):
                        result = result.get()
                
                return result
                
            except Exception as e:
                if fallback_to_numpy:
                    logger.log_warning(f"CuPy optimization failed for {func.__name__}, falling back to numpy: {e}")
                    return func(*args, **kwargs)
                else:
                    raise e
        
        return wrapper
    return decorator


def jax_optimize(static_argnums: Optional[tuple] = None, profile: bool = True):
    """
    Decorator to optimize numpy operations using JAX JIT compilation.
    
    Args:
        static_argnums: Argument indices to treat as static for JIT
        profile: If True, profiles performance improvement
    """
    def decorator(func: Callable) -> Callable:
        if not JAX_AVAILABLE:
            logger.log_warning(f"JAX not available for {func.__name__}, using original function")
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Convert numpy arrays to jax
                jax_args = []
                for arg in args:
                    if isinstance(arg, np.ndarray):
                        jax_args.append(jnp.array(arg))
                    else:
                        jax_args.append(arg)
                
                jax_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, np.ndarray):
                        jax_kwargs[k] = jnp.array(v)
                    else:
                        jax_kwargs[k] = v
                
                # Replace numpy with jax.numpy in function globals
                func_globals = func.__globals__.copy()
                func_globals['np'] = jnp
                
                # Create JAX version of function
                import types
                jax_func = types.FunctionType(
                    func.__code__, func_globals, func.__name__,
                    func.__defaults__, func.__closure__
                )
                
                # JIT compile
                if static_argnums:
                    jit_func = jax.jit(jax_func, static_argnums=static_argnums)
                else:
                    jit_func = jax.jit(jax_func)
                
                if profile:
                    # Benchmark original
                    original_time = benchmark_function(func, *args, **kwargs)
                    
                    # First call to trigger compilation
                    _ = jit_func(*jax_args, **jax_kwargs)
                    
                    # Benchmark optimized (post-compilation)
                    start = time.perf_counter()
                    result = jit_func(*jax_args, **jax_kwargs)
                    end = time.perf_counter()
                    optimized_time = end - start
                    
                    # Convert back to numpy
                    result = np.array(result)
                    
                    # Profile performance
                    input_shape = args[0].shape if args and hasattr(args[0], 'shape') else None
                    profiler.profile_function(func.__name__, original_time, optimized_time,
                                            input_shape, "JAX")
                else:
                    result = jit_func(*jax_args, **jax_kwargs)
                    result = np.array(result)
                
                return result
                
            except Exception as e:
                logger.log_warning(f"JAX optimization failed for {func.__name__}: {e}")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def numba_jit(nopython: bool = True, parallel: bool = False, cache: bool = True, profile: bool = True):
    """
    Decorator to optimize functions using Numba JIT compilation.
    
    Args:
        nopython: Use nopython mode for better performance
        parallel: Enable automatic parallelization
        cache: Cache compiled functions
        profile: If True, profiles performance improvement
    """
    def decorator(func: Callable) -> Callable:
        if not NUMBA_AVAILABLE:
            logger.log_warning(f"Numba not available for {func.__name__}, using original function")
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Create numba-compiled version
                numba_func = numba_jit_func(nopython=nopython, parallel=parallel, cache=cache)(func)
                
                if profile:
                    # Benchmark original
                    original_time = benchmark_function(func, *args, **kwargs)
                    
                    # First call to trigger compilation
                    _ = numba_func(*args, **kwargs)
                    
                    # Benchmark optimized (post-compilation)
                    start = time.perf_counter()
                    result = numba_func(*args, **kwargs)
                    end = time.perf_counter()
                    optimized_time = end - start
                    
                    # Profile performance
                    input_shape = args[0].shape if args and hasattr(args[0], 'shape') else None
                    profiler.profile_function(func.__name__, original_time, optimized_time,
                                            input_shape, "Numba")
                else:
                    result = numba_func(*args, **kwargs)
                
                return result
                
            except Exception as e:
                logger.log_warning(f"Numba optimization failed for {func.__name__}: {e}")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def auto_optimize(prefer_gpu: bool = True, profile: bool = True):
    """
    Automatically choose the best optimization strategy based on available libraries.
    
    Args:
        prefer_gpu: Prefer GPU-based optimizations when available
        profile: If True, profiles performance improvement
    """
    def decorator(func: Callable) -> Callable:
        # Choose optimization strategy
        if prefer_gpu and CUPY_AVAILABLE:
            return cupy_optimize(profile=profile)(func)
        elif JAX_AVAILABLE:
            return jax_optimize(profile=profile)(func)
        elif NUMBA_AVAILABLE:
            return numba_jit(profile=profile)(func)
        else:
            logger.log_warning(f"No optimization libraries available for {func.__name__}")
            return func
    
    return decorator


def profile_decorator(name: Optional[str] = None, log_memory: bool = False):
    """Enhanced performance profiling decorator with detailed metrics."""
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Memory tracking if requested
            if log_memory:
                try:
                    import psutil
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                except ImportError:
                    memory_before = None
            
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            
            duration = end - start
            input_shape = args[0].shape if args and hasattr(args[0], 'shape') else None
            
            # Enhanced performance logging
            logger.log_performance(base_logger, f"{func_name}", duration)
            
            if input_shape:
                data_size = np.prod(input_shape) if hasattr(input_shape, '__iter__') else input_shape
                throughput = data_size / duration if duration > 0 else 0
                logger.log_debug(f"📊 {func_name} - Shape: {input_shape}, Throughput: {throughput:.0f} items/s")
            
            # Memory usage logging
            if log_memory and memory_before is not None:
                try:
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    memory_delta = memory_after - memory_before
                    if abs(memory_delta) > 10:  # Only log significant changes
                        logger.log_debug(f"💾 {func_name} memory: {memory_delta:+.1f} MB")
                except:
                    pass
            
            # Performance warnings for slow operations
            if duration > 5.0:
                logger.log_warning(f"⏰ {func_name} took {duration:.2f}s - consider optimization")
            elif duration > 1.0:
                logger.log_debug(f"⏱️  {func_name} took {duration:.2f}s")
            
            return result
        return wrapper
    return decorator


# Convenience functions for getting optimization info
def get_optimization_status() -> Dict[str, bool]:
    """Get status of available optimization libraries."""
    return {
        'cupy': CUPY_AVAILABLE,
        'jax': JAX_AVAILABLE,
        'numba': NUMBA_AVAILABLE
    }


def get_performance_summary() -> Dict[str, Any]:
    """Get performance profiling summary."""
    return profiler.get_summary()


def log_optimization_status():
    """Log the status of optimization libraries."""
    status = get_optimization_status()
    logger.log_info("🔧 Optimization Libraries Status:")
    for lib, available in status.items():
        status_icon = "✅" if available else "❌"
        logger.log_info(f"  {status_icon} {lib.upper()}: {'Available' if available else 'Not Available'}")