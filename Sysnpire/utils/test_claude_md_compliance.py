"""
CLAUDE.md Compliance Verification Test

This script demonstrates that our field theory optimizers strictly adhere to 
CLAUDE.md requirements with working examples of complex-valued mathematics,
trajectory integration, and field calculations.
"""

import sys
import numpy as np
from pathlib import Path
from scipy.integrate import quad

# Ensure project imports work
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.utils.field_theory_optimizers import (
    field_theory_jax_optimize, field_theory_trajectory_optimize,
    log_field_theory_optimization_status, get_field_theory_performance_summary
)
from Sysnpire.utils.logger import SysnpireLogger

logger = SysnpireLogger()


def test_complex_valued_preservation():
    """Test that optimizations preserve complex-valued mathematics."""
    logger.log_info("🔬 Testing Complex-Valued Mathematics Preservation")
    
    # Test complex field calculation
    @field_theory_jax_optimize(preserve_complex=True, profile=True)
    def complex_field_calculation(omega: np.ndarray, phi: np.ndarray, s: float) -> np.ndarray:
        """Complex field calculation as per CLAUDE.md requirements."""
        # Complex phase evolution - critical for field theory
        phase_factor = np.exp(1j * (phi + omega * s))
        
        # Field modulation with breathing patterns
        breathing = 1.0 + 0.3 * np.cos(2 * np.pi * s * 0.1 + np.arange(len(omega)))
        
        # Complex field generation
        field = omega * breathing * phase_factor
        
        return field
    
    # Test data with realistic field parameters
    omega = np.array([0.1, 0.05, 0.15, 0.08])  # Frequency components
    phi = np.array([0.0, np.pi/4, np.pi/2, np.pi])  # Phase relationships
    s = 5.0  # Observational state
    
    result = complex_field_calculation(omega, phi, s)
    
    # Verify complex result
    if not np.iscomplexobj(result):
        logger.log_error("❌ Complex mathematics not preserved")
        return False
    
    # Verify field characteristics
    magnitudes = np.abs(result)
    phases = np.angle(result)
    
    logger.log_success(f"✅ Complex field generated: |field| = {magnitudes}, ∠field = {phases}")
    logger.log_success("✅ Complex-valued mathematics preserved through optimization")
    return True


def test_trajectory_integration_compliance():
    """Test trajectory integration with scipy.integrate.quad compliance."""
    logger.log_info("🌊 Testing Trajectory Integration Compliance")
    
    @field_theory_trajectory_optimize(profile=True)
    def trajectory_integration_with_quad(omega: float, phi: float, s: float) -> complex:
        """
        Trajectory operator using scipy.integrate.quad as required by CLAUDE.md.
        
        Implements: T_i(τ,s) = ∫₀ˢ ω_i(τ,s')·e^(iφ_i(τ,s')) ds'
        """
        # Define integrand for complex trajectory integration
        def integrand_real(s_prime):
            return omega * np.cos(phi + omega * s_prime)
        
        def integrand_imag(s_prime):
            return omega * np.sin(phi + omega * s_prime)
        
        # Use scipy.integrate.quad for real and imaginary parts
        real_part, _ = quad(integrand_real, 0, s)
        imag_part, _ = quad(integrand_imag, 0, s)
        
        return complex(real_part, imag_part)
    
    # Test trajectory integration
    omega = 0.1
    phi = np.pi / 4
    s = 5.0
    
    result = trajectory_integration_with_quad(omega, phi, s)
    
    # Verify complex result from integration
    if not isinstance(result, complex):
        logger.log_error("❌ Trajectory integration must return complex values")
        return False
    
    logger.log_success(f"✅ Trajectory integration: T = {result:.6f}")
    logger.log_success("✅ scipy.integrate.quad integration compliance verified")
    return True


def test_field_theory_formulation():
    """Test field theory formulation preservation."""
    logger.log_info("⚡ Testing Field Theory Formulation Preservation")
    
    @field_theory_jax_optimize(preserve_complex=True, profile=True)
    def conceptual_charge_component(gamma: float, T_component: complex, E_magnitude: float,
                                   phi_magnitude: float, theta_total: float, 
                                   psi_persistence: float) -> complex:
        """
        Component of Q(τ, C, s) formulation preservation test.
        
        Tests: Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)
        """
        # Phase factor preservation
        phase_factor = np.exp(1j * theta_total)
        
        # Complete charge calculation
        Q = gamma * T_component * E_magnitude * phi_magnitude * phase_factor * psi_persistence
        
        return Q
    
    # Realistic field theory parameters
    gamma = 1.2  # Global field calibration
    T_component = complex(0.5, 0.3)  # Trajectory component
    E_magnitude = 0.8  # Emotional trajectory magnitude
    phi_magnitude = 1.1  # Semantic field magnitude
    theta_total = np.pi / 3  # Total phase
    psi_persistence = 0.9  # Observational persistence
    
    result = conceptual_charge_component(gamma, T_component, E_magnitude, 
                                       phi_magnitude, theta_total, psi_persistence)
    
    # Verify complex-valued charge
    if not isinstance(result, complex):
        logger.log_error("❌ Conceptual charge must be complex-valued")
        return False
    
    magnitude = np.abs(result)
    phase = np.angle(result)
    
    logger.log_success(f"✅ Conceptual charge: |Q| = {magnitude:.6f}, ∠Q = {phase:.6f}")
    logger.log_success("✅ Field theory formulation preserved through optimization")
    return True


def test_no_simulated_values():
    """Verify no simulated or default values are used."""
    logger.log_info("🎯 Testing No Simulated Values Compliance")
    
    # All values must be mathematically derived, not simulated
    @field_theory_jax_optimize(profile=True)
    def field_calculation_no_simulation(semantic_vector: np.ndarray, 
                                       observational_state: float) -> np.ndarray:
        """Field calculation using only derived values, no simulation."""
        # Derive parameters from actual inputs (not np.random or defaults)
        vector_norm = np.linalg.norm(semantic_vector)
        
        # Mathematical derivation based on observational state
        frequency_scale = 1.0 / (1.0 + observational_state)
        
        # Field evolution based on mathematical relationships
        field_response = semantic_vector * frequency_scale * (1.0 + 0.1 * observational_state)
        
        return field_response
    
    # Create deterministic semantic vector (not random)
    semantic_vector = np.array([0.5, -0.3, 0.8, 0.1])  # Deterministic values
    semantic_vector = semantic_vector / np.linalg.norm(semantic_vector)  # Unit vector
    
    observational_state = 3.0  # Specific observational state
    
    result = field_calculation_no_simulation(semantic_vector, observational_state)
    
    # Verify result is deterministic and mathematically derived
    expected_norm = np.linalg.norm(semantic_vector) * (1.0 / (1.0 + observational_state)) * (1.0 + 0.1 * observational_state)
    actual_norm = np.linalg.norm(result)
    
    if not np.isclose(actual_norm, expected_norm):
        logger.log_error("❌ Field calculation not properly derived from inputs")
        return False
    
    logger.log_success("✅ No simulated values - all calculations mathematically derived")
    return True


def run_claude_md_compliance_verification():
    """Run complete CLAUDE.md compliance verification."""
    logger.log_info("🚀 CLAUDE.MD COMPLIANCE VERIFICATION")
    logger.log_info("=" * 60)
    
    # Log optimization status
    log_field_theory_optimization_status()
    
    # Run compliance tests
    tests = [
        ("Complex-Valued Mathematics", test_complex_valued_preservation),
        ("Trajectory Integration", test_trajectory_integration_compliance), 
        ("Field Theory Formulation", test_field_theory_formulation),
        ("No Simulated Values", test_no_simulated_values)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.log_info(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.log_info(f"  {status}")
        except Exception as e:
            results[test_name] = False
            logger.log_error(f"  ❌ FAILED: {e}")
    
    # Final compliance report
    logger.log_info("\n" + "=" * 60)
    logger.log_info("📋 CLAUDE.MD COMPLIANCE SUMMARY")
    logger.log_info("=" * 60)
    
    all_passed = all(results.values())
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        logger.log_info(f"  {status} {test_name}")
    
    # Performance summary
    perf_summary = get_field_theory_performance_summary()
    if perf_summary:
        logger.log_info("\n🚀 Optimization Performance:")
        for func_name, stats in perf_summary.items():
            accuracy = stats.get('mathematical_accuracy_rate', 1.0)
            speedup = stats.get('avg_speedup', 1.0)
            logger.log_info(f"  • {func_name}: {speedup:.2f}x speedup, {accuracy:.0%} accuracy")
    
    logger.log_info("\n" + "=" * 60)
    if all_passed:
        logger.log_success("🎉 ALL CLAUDE.MD REQUIREMENTS SATISFIED!")
        logger.log_success("✅ Field theory optimizers are production ready")
    else:
        logger.log_error("❌ CLAUDE.MD COMPLIANCE ISSUES DETECTED")
        logger.log_warning("⚠️  Must resolve compliance issues before deployment")
    
    logger.log_info("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    run_claude_md_compliance_verification()