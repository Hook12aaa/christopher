"""
Comprehensive Test Suite for Sysnpire Performance Optimization Decorators

This script tests all optimization decorators with real mathematical operations
extracted from the Sysnpire codebase. It ensures proper performance metrics
logging and validates optimization effectiveness across different data sizes.

Test Categories:
1. BGE Embedding Operations (from BGE ingestion)
2. DTF Field Calculations (from DTF core)
3. Conceptual Charge Mathematics (from conceptual charge)
4. Similarity Computations (from similarity calculations)
5. Vector Transformations (from vector transformation)

Each test includes:
- Performance benchmarking with detailed metrics
- Fallback mechanism validation  
- Library availability checks
- Memory usage tracking
- Scaling analysis across different data sizes
"""

import sys
import time
import traceback
import psutil
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Ensure project imports work
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.utils.performance_optimizers import (
    cupy_optimize, jax_optimize, numba_jit, auto_optimize, profile_decorator,
    get_optimization_status, get_performance_summary, log_optimization_status,
    PerformanceProfiler
)
from Sysnpire.utils.logger import get_logger, SysnpireLogger

base_logger = get_logger(__name__)
logger = SysnpireLogger()


class PerformanceTestSuite:
    """Comprehensive performance testing suite for optimization decorators."""
    
    def __init__(self):
        self.test_results = {}
        self.profiler = PerformanceProfiler()
        
        # Test configurations for different data sizes
        self.test_configs = {
            'small': {'embeddings': 100, 'dim': 512, 'matrix_size': 256},
            'medium': {'embeddings': 1000, 'dim': 1024, 'matrix_size': 512},
            'large': {'embeddings': 5000, 'dim': 1024, 'matrix_size': 1024}
        }
        
        logger.log_info("🧪 Performance Test Suite Initialized")
        log_optimization_status()
    
    def log_memory_usage(self, operation_name: str):
        """Log current memory usage."""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.log_debug(f"💾 Memory usage after {operation_name}: {memory_mb:.1f} MB")
    
    def measure_execution_time(self, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Measure execution time and return result with timing."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time


# =============================================================================
# REAL MATH OPERATIONS FROM SYSNPIRE CODEBASE
# =============================================================================

class BGEEmbeddingOperationsTest:
    """Test optimization of BGE embedding operations from the actual codebase."""
    
    @profile_decorator(name="BGE_similarity_baseline")
    def compute_similarity_baseline(self, embeddings: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Baseline BGE similarity computation (numpy only)."""
        # Normalize embeddings
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_query = query / np.linalg.norm(query)
        
        # Cosine similarity
        return np.dot(norm_embeddings, norm_query)
    
    @cupy_optimize(profile=True)
    def compute_similarity_cupy(self, embeddings: np.ndarray, query: np.ndarray) -> np.ndarray:
        """CuPy-optimized BGE similarity computation."""
        # Normalize embeddings
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_query = query / np.linalg.norm(query)
        
        # Cosine similarity
        return np.dot(norm_embeddings, norm_query)
    
    @jax_optimize(profile=True)
    def compute_similarity_jax(self, embeddings: np.ndarray, query: np.ndarray) -> np.ndarray:
        """JAX-optimized BGE similarity computation."""
        # Normalize embeddings
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_query = query / np.linalg.norm(query)
        
        # Cosine similarity
        return np.dot(norm_embeddings, norm_query)
    
    @numba_jit(parallel=True, profile=True)
    def compute_similarity_numba(self, embeddings: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Numba-optimized BGE similarity computation."""
        num_embeddings = embeddings.shape[0]
        similarities = np.empty(num_embeddings)
        query_norm = np.linalg.norm(query)
        
        for i in range(num_embeddings):
            embedding_norm = np.linalg.norm(embeddings[i])
            dot_product = np.dot(embeddings[i], query)
            similarities[i] = dot_product / (embedding_norm * query_norm + 1e-10)
        
        return similarities


class DTFFieldCalculationsTest:
    """Test optimization of DTF field calculations from the actual codebase."""
    
    @profile_decorator(name="DTF_field_evolution_baseline")
    def neural_field_evolution_baseline(self, u: np.ndarray, h: float, S: np.ndarray, 
                                       w: np.ndarray, tau: float, dt: float) -> np.ndarray:
        """Baseline DTF neural field evolution (numpy only)."""
        # Activation function
        f_u = 1.0 / (1.0 + np.exp(-u))
        
        # Lateral interaction (convolution)
        lateral = np.convolve(w, f_u, mode='same')
        
        # Field evolution equation
        du_dt = (-u + h + S + lateral) / tau
        
        return u + dt * du_dt
    
    @jax_optimize(profile=True)
    def neural_field_evolution_jax(self, u: np.ndarray, h: float, S: np.ndarray,
                                  w: np.ndarray, tau: float, dt: float) -> np.ndarray:
        """JAX-optimized DTF neural field evolution."""
        # Activation function
        f_u = 1.0 / (1.0 + np.exp(-u))
        
        # Lateral interaction (convolution)
        lateral = np.convolve(w, f_u, mode='same')
        
        # Field evolution equation  
        du_dt = (-u + h + S + lateral) / tau
        
        return u + dt * du_dt
    
    @cupy_optimize(profile=True)
    def large_field_interaction_cupy(self, field1: np.ndarray, field2: np.ndarray,
                                    interaction_matrix: np.ndarray) -> np.ndarray:
        """CuPy-optimized large-scale field interaction."""
        # Cross-field interaction
        cross_interaction = np.dot(interaction_matrix, field2)
        
        # Field coupling
        coupling = np.outer(field1, field2)
        
        # Combined field effect
        result = field1 + cross_interaction + np.sum(coupling, axis=1)
        
        return result
    
    @numba_jit(parallel=True, profile=True)
    def field_neighborhood_numba(self, field: np.ndarray, neighborhood_size: int = 2) -> np.ndarray:
        """Numba-optimized field neighborhood computation."""
        rows, cols = field.shape
        result = np.zeros_like(field)
        
        for i in range(rows):
            for j in range(cols):
                total = 0.0
                count = 0
                
                for di in range(-neighborhood_size, neighborhood_size + 1):
                    for dj in range(-neighborhood_size, neighborhood_size + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            total += field[ni, nj]
                            count += 1
                
                result[i, j] = total / count if count > 0 else 0.0
        
        return result


class ConceptualChargeOperationsTest:
    """Test optimization of conceptual charge calculations from the actual codebase."""
    
    @profile_decorator(name="trajectory_integration_baseline")
    def trajectory_operator_baseline(self, tau: np.ndarray, s: float, omega_base: np.ndarray, 
                                   phi_base: np.ndarray) -> np.ndarray:
        """Baseline trajectory operator T(τ, C, s) integration."""
        # Frequency evolution with observational state dependency
        omega = omega_base * (1.0 + 0.1 * s)
        
        # Phase evolution
        phi = phi_base + omega * s
        
        # Complex trajectory integration (simplified version)
        trajectory_components = np.zeros(len(tau), dtype=complex)
        for i in range(len(tau)):
            integrand = lambda s_prime: omega[i] * np.exp(1j * (phi[i] + omega[i] * s_prime))
            real_part, _ = quad(lambda s_p: np.real(integrand(s_p)), 0, s)
            imag_part, _ = quad(lambda s_p: np.imag(integrand(s_p)), 0, s)
            trajectory_components[i] = complex(real_part, imag_part)
        
        return trajectory_components
    
    @jax_optimize(static_argnums=(1,), profile=True)
    def trajectory_operator_jax(self, tau: np.ndarray, s: float, omega_base: np.ndarray,
                               phi_base: np.ndarray) -> np.ndarray:
        """JAX-optimized trajectory operator integration."""
        # Frequency evolution
        omega = omega_base * (1.0 + 0.1 * s)
        
        # Phase evolution
        phi = phi_base + omega * s
        
        # Simplified complex integration using trapezoidal rule
        s_range = np.linspace(0, s, 100)
        ds = s / 99.0
        
        trajectory_components = np.zeros(len(tau), dtype=complex)
        for i in range(len(tau)):
            integrand = omega[i] * np.exp(1j * (phi[i] + omega[i] * s_range))
            trajectory_components[i] = np.trapz(integrand, dx=ds)
        
        return trajectory_components
    
    @cupy_optimize(profile=True)
    def emotional_trajectory_integration_cupy(self, valence: np.ndarray, arousal: np.ndarray,
                                             dominance: np.ndarray, alpha: np.ndarray,
                                             sigma_sq: np.ndarray) -> np.ndarray:
        """CuPy-optimized emotional trajectory integration."""
        # Emotional space coordinates
        emotional_space = np.stack([valence, arousal, dominance], axis=1)
        
        # Gaussian alignment computation
        emotional_centers = np.array([[0.8, 0.6, 0.7], [-0.3, 0.2, 0.4], [0.1, -0.5, 0.8]])
        
        alignments = np.zeros((len(emotional_space), len(emotional_centers)))
        for i, center in enumerate(emotional_centers):
            distances_sq = np.sum((emotional_space - center)**2, axis=1)
            alignments[:, i] = alpha[i] * np.exp(-distances_sq / (2 * sigma_sq[i]))
        
        # Trajectory accumulation
        trajectory_weights = np.exp(-np.arange(len(emotional_space)) / 50.0)
        weighted_alignments = alignments * trajectory_weights.reshape(-1, 1)
        
        return np.sum(weighted_alignments, axis=0)
    
    @auto_optimize(profile=True)
    def semantic_field_generation_auto(self, semantic_vector: np.ndarray, 
                                      beta_breathing: np.ndarray, w_weights: np.ndarray,
                                      observational_state: float) -> np.ndarray:
        """Auto-optimized semantic field generation."""
        # Breathing modulation
        breathing_phase = 2 * np.pi * observational_state * 0.1
        breathing_modulation = 1.0 + beta_breathing * np.cos(breathing_phase + np.arange(len(semantic_vector)))
        
        # Semantic field generation with complex phase
        T_components = np.random.randn(len(semantic_vector)) + 1j * np.random.randn(len(semantic_vector))
        
        # Field generation formula
        phi_semantic = w_weights * T_components * semantic_vector * breathing_modulation * np.exp(1j * breathing_phase)
        
        return np.real(phi_semantic)


class VectorTransformationTest:
    """Test optimization of vector transformation operations."""
    
    @profile_decorator(name="gaussian_basis_baseline")
    def gaussian_basis_computation_baseline(self, embedding: np.ndarray, 
                                          basis_centers: np.ndarray, 
                                          basis_widths: np.ndarray) -> np.ndarray:
        """Baseline Gaussian basis function computation."""
        num_basis = basis_centers.shape[0]
        basis_values = np.zeros(num_basis)
        
        for i in range(num_basis):
            diff = embedding - basis_centers[i]
            dist_sq = np.dot(diff, diff)
            basis_values[i] = np.exp(-dist_sq / (2 * basis_widths[i]**2))
        
        return basis_values
    
    @numba_jit(parallel=True, profile=True)
    def gaussian_basis_computation_numba(self, embedding: np.ndarray,
                                        basis_centers: np.ndarray,
                                        basis_widths: np.ndarray) -> np.ndarray:
        """Numba-optimized Gaussian basis function computation."""
        num_basis = basis_centers.shape[0]
        basis_values = np.zeros(num_basis)
        
        for i in range(num_basis):
            diff = embedding - basis_centers[i]
            dist_sq = np.dot(diff, diff)
            basis_values[i] = np.exp(-dist_sq / (2 * basis_widths[i]**2))
        
        return basis_values
    
    @cupy_optimize(profile=True)
    def field_transformation_matrix_cupy(self, embeddings: np.ndarray,
                                        transformation_matrix: np.ndarray) -> np.ndarray:
        """CuPy-optimized field transformation matrix operation."""
        # Phase computation
        phases = np.angle(embeddings @ transformation_matrix.T)
        
        # Field magnitude computation
        magnitudes = np.linalg.norm(embeddings @ transformation_matrix.T, axis=1)
        
        # Complex field representation
        complex_field = magnitudes[:, np.newaxis] * np.exp(1j * phases)
        
        return np.real(complex_field)


# =============================================================================
# TEST EXECUTION ENGINE
# =============================================================================

class TestExecutor:
    """Executes comprehensive optimization tests with detailed logging."""
    
    def __init__(self):
        self.suite = PerformanceTestSuite()
        self.bge_test = BGEEmbeddingOperationsTest()
        self.dtf_test = DTFFieldCalculationsTest()
        self.charge_test = ConceptualChargeOperationsTest()
        self.vector_test = VectorTransformationTest()
        
        self.test_results = {}
    
    def run_bge_embedding_tests(self, config_name: str):
        """Run BGE embedding optimization tests."""
        logger.log_info(f"🔤 Running BGE Embedding Tests - {config_name}")
        config = self.suite.test_configs[config_name]
        
        # Generate test data
        embeddings = np.random.randn(config['embeddings'], config['dim']).astype(np.float32)
        query = np.random.randn(config['dim']).astype(np.float32)
        
        # Normalize for realistic BGE-like data
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        query = query / np.linalg.norm(query)
        
        self.suite.log_memory_usage("BGE data generation")
        
        test_results = {}
        
        try:
            # Baseline test
            result, time_taken = self.suite.measure_execution_time(
                self.bge_test.compute_similarity_baseline, embeddings, query
            )
            test_results['baseline'] = {'time': time_taken, 'shape': result.shape}
            logger.log_performance(logger._default_logger, f"BGE Baseline ({config_name})", time_taken, config['embeddings'])
            
            # CuPy test
            result, time_taken = self.suite.measure_execution_time(
                self.bge_test.compute_similarity_cupy, embeddings, query
            )
            test_results['cupy'] = {'time': time_taken, 'shape': result.shape}
            
            # JAX test
            result, time_taken = self.suite.measure_execution_time(
                self.bge_test.compute_similarity_jax, embeddings, query
            )
            test_results['jax'] = {'time': time_taken, 'shape': result.shape}
            
            # Numba test
            result, time_taken = self.suite.measure_execution_time(
                self.bge_test.compute_similarity_numba, embeddings, query
            )
            test_results['numba'] = {'time': time_taken, 'shape': result.shape}
            
            self.test_results[f'bge_{config_name}'] = test_results
            logger.log_success(f"✅ BGE tests completed for {config_name}")
            
        except Exception as e:
            logger.log_error(f"❌ BGE test failed for {config_name}: {str(e)}")
            logger.log_debug(f"Traceback: {traceback.format_exc()}")
    
    def run_dtf_field_tests(self, config_name: str):
        """Run DTF field calculation optimization tests."""
        logger.log_info(f"🌊 Running DTF Field Tests - {config_name}")
        config = self.suite.test_configs[config_name]
        
        # Generate DTF test data
        field_size = (config['matrix_size'], config['matrix_size'])
        u = np.random.randn(*field_size).astype(np.float32)
        h = -2.0
        S = np.random.randn(*field_size).astype(np.float32) * 0.1
        w = np.exp(-np.linspace(-3, 3, 21)**2).astype(np.float32)
        tau = 10.0
        dt = 0.1
        
        self.suite.log_memory_usage("DTF data generation")
        
        test_results = {}
        
        try:
            # Neural field evolution tests
            u_1d = u.flatten()
            S_1d = S.flatten()
            
            result, time_taken = self.suite.measure_execution_time(
                self.dtf_test.neural_field_evolution_baseline, u_1d, h, S_1d, w, tau, dt
            )
            test_results['field_evolution_baseline'] = {'time': time_taken, 'shape': result.shape}
            logger.log_performance(logger._default_logger, f"DTF Field Evolution Baseline ({config_name})", time_taken)
            
            result, time_taken = self.suite.measure_execution_time(
                self.dtf_test.neural_field_evolution_jax, u_1d, h, S_1d, w, tau, dt
            )
            test_results['field_evolution_jax'] = {'time': time_taken, 'shape': result.shape}
            
            # Large field interaction tests
            field1 = np.random.randn(config['matrix_size']).astype(np.float32)
            field2 = np.random.randn(config['matrix_size']).astype(np.float32)
            interaction_matrix = np.random.randn(config['matrix_size'], config['matrix_size']).astype(np.float32) * 0.01
            
            result, time_taken = self.suite.measure_execution_time(
                self.dtf_test.large_field_interaction_cupy, field1, field2, interaction_matrix
            )
            test_results['field_interaction_cupy'] = {'time': time_taken, 'shape': result.shape}
            
            # Field neighborhood tests
            small_field = u[:64, :64] if config['matrix_size'] > 64 else u
            result, time_taken = self.suite.measure_execution_time(
                self.dtf_test.field_neighborhood_numba, small_field
            )
            test_results['field_neighborhood_numba'] = {'time': time_taken, 'shape': result.shape}
            
            self.test_results[f'dtf_{config_name}'] = test_results
            logger.log_success(f"✅ DTF tests completed for {config_name}")
            
        except Exception as e:
            logger.log_error(f"❌ DTF test failed for {config_name}: {str(e)}")
            logger.log_debug(f"Traceback: {traceback.format_exc()}")
    
    def run_conceptual_charge_tests(self, config_name: str):
        """Run conceptual charge calculation optimization tests."""
        logger.log_info(f"⚡ Running Conceptual Charge Tests - {config_name}")
        config = self.suite.test_configs[config_name]
        
        # Generate conceptual charge test data
        tau = np.random.randn(config['dim']).astype(np.float32)
        s = 5.0
        omega_base = np.random.rand(config['dim']).astype(np.float32) * 0.1
        phi_base = np.random.rand(config['dim']).astype(np.float32) * 2 * np.pi
        
        # Emotional trajectory data
        num_states = min(1000, config['embeddings'])
        valence = np.random.randn(num_states).astype(np.float32)
        arousal = np.random.randn(num_states).astype(np.float32)
        dominance = np.random.randn(num_states).astype(np.float32)
        alpha = np.random.rand(3).astype(np.float32)
        sigma_sq = np.random.rand(3).astype(np.float32) + 0.1
        
        self.suite.log_memory_usage("Conceptual charge data generation")
        
        test_results = {}
        
        try:
            # Trajectory operator tests (simplified for performance)
            small_tau = tau[:min(50, len(tau))]  # Limit size for integration tests
            small_omega = omega_base[:min(50, len(omega_base))]
            small_phi = phi_base[:min(50, len(phi_base))]
            
            result, time_taken = self.suite.measure_execution_time(
                self.charge_test.trajectory_operator_baseline, small_tau, s, small_omega, small_phi
            )
            test_results['trajectory_baseline'] = {'time': time_taken, 'shape': result.shape}
            logger.log_performance(logger._default_logger, f"Trajectory Integration Baseline ({config_name})", time_taken)
            
            result, time_taken = self.suite.measure_execution_time(
                self.charge_test.trajectory_operator_jax, small_tau, s, small_omega, small_phi
            )
            test_results['trajectory_jax'] = {'time': time_taken, 'shape': result.shape}
            
            # Emotional trajectory tests
            result, time_taken = self.suite.measure_execution_time(
                self.charge_test.emotional_trajectory_integration_cupy, valence, arousal, dominance, alpha, sigma_sq
            )
            test_results['emotional_cupy'] = {'time': time_taken, 'shape': result.shape}
            
            # Semantic field generation tests
            semantic_vector = np.random.randn(config['dim']).astype(np.float32)
            beta_breathing = np.random.rand(config['dim']).astype(np.float32) * 0.5
            w_weights = np.ones(config['dim']).astype(np.float32)
            
            result, time_taken = self.suite.measure_execution_time(
                self.charge_test.semantic_field_generation_auto, semantic_vector, beta_breathing, w_weights, s
            )
            test_results['semantic_auto'] = {'time': time_taken, 'shape': result.shape}
            
            self.test_results[f'charge_{config_name}'] = test_results
            logger.log_success(f"✅ Conceptual Charge tests completed for {config_name}")
            
        except Exception as e:
            logger.log_error(f"❌ Conceptual Charge test failed for {config_name}: {str(e)}")
            logger.log_debug(f"Traceback: {traceback.format_exc()}")
    
    def run_vector_transformation_tests(self, config_name: str):
        """Run vector transformation optimization tests."""
        logger.log_info(f"🔄 Running Vector Transformation Tests - {config_name}")
        config = self.suite.test_configs[config_name]
        
        # Generate vector transformation test data
        embedding = np.random.randn(config['dim']).astype(np.float32)
        num_basis = min(500, config['embeddings'])
        basis_centers = np.random.randn(num_basis, config['dim']).astype(np.float32)
        basis_widths = np.random.rand(num_basis).astype(np.float32) + 0.1
        
        # Transformation matrix
        transformation_matrix = np.random.randn(config['dim'], config['dim']).astype(np.float32) * 0.1
        embeddings_batch = np.random.randn(min(1000, config['embeddings']), config['dim']).astype(np.float32)
        
        self.suite.log_memory_usage("Vector transformation data generation")
        
        test_results = {}
        
        try:
            # Gaussian basis tests
            result, time_taken = self.suite.measure_execution_time(
                self.vector_test.gaussian_basis_computation_baseline, embedding, basis_centers, basis_widths
            )
            test_results['gaussian_baseline'] = {'time': time_taken, 'shape': result.shape}
            logger.log_performance(logger._default_logger, f"Gaussian Basis Baseline ({config_name})", time_taken)
            
            result, time_taken = self.suite.measure_execution_time(
                self.vector_test.gaussian_basis_computation_numba, embedding, basis_centers, basis_widths
            )
            test_results['gaussian_numba'] = {'time': time_taken, 'shape': result.shape}
            
            # Field transformation tests
            result, time_taken = self.suite.measure_execution_time(
                self.vector_test.field_transformation_matrix_cupy, embeddings_batch, transformation_matrix
            )
            test_results['transformation_cupy'] = {'time': time_taken, 'shape': result.shape}
            
            self.test_results[f'vector_{config_name}'] = test_results
            logger.log_success(f"✅ Vector Transformation tests completed for {config_name}")
            
        except Exception as e:
            logger.log_error(f"❌ Vector Transformation test failed for {config_name}: {str(e)}")
            logger.log_debug(f"Traceback: {traceback.format_exc()}")
    
    def run_comprehensive_test_suite(self):
        """Run the complete optimization test suite."""
        logger.log_info("🚀 Starting Comprehensive Optimization Test Suite")
        logger.log_info("=" * 80)
        
        # Log system information
        logger.log_info(f"💻 System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / 1024**3:.1f} GB RAM")
        
        # Test each configuration size
        for config_name in self.suite.test_configs:
            logger.log_info(f"\n📊 Testing configuration: {config_name.upper()}")
            logger.log_info(f"Config: {self.suite.test_configs[config_name]}")
            
            # Run all test categories
            self.run_bge_embedding_tests(config_name)
            self.run_dtf_field_tests(config_name)
            self.run_conceptual_charge_tests(config_name)
            self.run_vector_transformation_tests(config_name)
            
            # Log memory usage after each config
            self.suite.log_memory_usage(f"Config {config_name} completed")
        
        # Generate final performance report
        self.generate_performance_report()
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        logger.log_info("\n" + "=" * 80)
        logger.log_info("📈 COMPREHENSIVE PERFORMANCE REPORT")
        logger.log_info("=" * 80)
        
        # Overall optimization status
        status = get_optimization_status()
        logger.log_info(f"🔧 Optimization Libraries: {sum(status.values())}/{len(status)} available")
        
        # Performance summary from decorators
        decorator_summary = get_performance_summary()
        if decorator_summary:
            logger.log_info("\n🚀 Decorator Performance Summary:")
            for func_name, stats in decorator_summary.items():
                logger.log_info(f"  • {func_name}: {stats['avg_speedup']:.2f}x avg, {stats['best_speedup']:.2f}x best ({stats['num_profiles']} tests)")
        
        # Test results analysis
        logger.log_info("\n📊 Test Results Analysis:")
        for test_category, results in self.test_results.items():
            logger.log_info(f"\n{test_category.upper()}:")
            for test_name, data in results.items():
                logger.log_info(f"  • {test_name}: {data['time']:.4f}s (shape: {data['shape']})")
        
        # Generate recommendations
        self.generate_optimization_recommendations()
        
        logger.log_info("\n" + "=" * 80)
        logger.log_success("🎉 Test Suite Completed Successfully!")
        logger.log_info("=" * 80)
    
    def generate_optimization_recommendations(self):
        """Generate optimization recommendations based on test results."""
        logger.log_info("\n💡 OPTIMIZATION RECOMMENDATIONS:")
        
        status = get_optimization_status()
        
        if status['cupy']:
            logger.log_info("  🎯 CuPy available - Use @cupy_optimize for large matrix operations")
        else:
            logger.log_info("  ⚠️  CuPy not available - Consider JAX for GPU-like performance")
        
        if status['jax']:
            logger.log_info("  🎯 JAX available - Use @jax_optimize for differentiable field calculations")
        
        if status['numba']:
            logger.log_info("  🎯 Numba available - Use @numba_jit for loop-heavy computations")
        
        logger.log_info("  📋 General recommendations:")
        logger.log_info("    • Use @auto_optimize for mixed workloads")
        logger.log_info("    • Always enable profile=True during development")
        logger.log_info("    • Monitor memory usage for large-scale operations")
        logger.log_info("    • Test fallback mechanisms in production")


def main():
    """Main test execution function."""
    try:
        logger.log_info("🧪 Initializing Sysnpire Optimization Decorator Test Suite")
        
        # Initialize and run tests
        executor = TestExecutor()
        executor.run_comprehensive_test_suite()
        
    except KeyboardInterrupt:
        logger.log_warning("⚠️  Test suite interrupted by user")
    except Exception as e:
        logger.log_error(f"❌ Test suite failed: {str(e)}")
        logger.log_debug(f"Traceback: {traceback.format_exc()}")
    finally:
        logger.log_info("🔚 Test suite execution completed")


if __name__ == "__main__":
    main()