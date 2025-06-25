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
    jax_optimize,
    numba_jit,
    auto_optimize,
    profile_decorator,
    get_optimization_status,
    get_performance_summary,
    log_optimization_status,
    PerformanceProfiler
)

from .field_theory_optimizers import (
    field_theory_jax_optimize,
    field_theory_numba_optimize,
    field_theory_trajectory_optimize,
    field_theory_auto_optimize,
    get_field_theory_optimization_status,
    get_field_theory_performance_summary,
    log_field_theory_optimization_status,
    FieldTheoryPerformanceProfiler
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
    'jax_optimize', 
    'numba_jit',
    'auto_optimize',
    'profile_decorator',
    'get_optimization_status',
    'get_performance_summary',
    'log_optimization_status',
    'PerformanceProfiler',
    
    # Field theory optimization decorators (CLAUDE.md compliant)
    'field_theory_jax_optimize',
    'field_theory_numba_optimize', 
    'field_theory_trajectory_optimize',
    'field_theory_auto_optimize',
    'get_field_theory_optimization_status',
    'get_field_theory_performance_summary',
    'log_field_theory_optimization_status',
    'FieldTheoryPerformanceProfiler'
]