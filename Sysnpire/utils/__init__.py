"""
Utilities package for Sysnpire

This package contains utility modules for the Sysnpire project including
logging, performance monitoring, optimization decorators, and other helper functions.
"""

from .logger import (
    get_logger,
    log_device_info,
    log_model_info, 
    log_embedding_stats,
    log_performance,
    log_error_context,
    SysnpireLogger
)

from .performance_optimizers import (
    cupy_optimize,
    jax_optimize,
    numba_jit,
    auto_optimize,
    profile_decorator,
    get_optimization_status,
    get_performance_summary,
    log_optimization_status,
    PerformanceProfiler
)

__all__ = [
    # Logging utilities
    'get_logger',
    'log_device_info', 
    'log_model_info',
    'log_embedding_stats',
    'log_performance',
    'log_error_context',
    'SysnpireLogger',
    
    # Performance optimization decorators
    'cupy_optimize',
    'jax_optimize', 
    'numba_jit',
    'auto_optimize',
    'profile_decorator',
    'get_optimization_status',
    'get_performance_summary',
    'log_optimization_status',
    'PerformanceProfiler'
]