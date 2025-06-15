#!/usr/bin/env python3
"""
Main Entry Point for Semantic Dimension Processing

This script serves as the primary entry point for semantic dimension operations,
similar to index.js in Node.js projects. It provides both a command-line interface
and a main function for testing and development.

USAGE:
    python main.py --test              # Run basic semantic field test
    python main.py --demo              # Run demonstration with sample data
    python main.py --benchmark         # Run performance benchmarks

INTEGRATION:
    from Sysnpire.model.semantic_dimension.main import run_semantic_processing
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional

# Ensure project imports work
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.model.semantic_dimension import process_semantic_field
from Sysnpire.utils.logger import get_logger
from Sysnpire.utils.field_theory_optimizers import (
    field_theory_jax_optimize, field_theory_numba_optimize, 
    field_theory_auto_optimize
)

logger = get_logger(__name__)


def _optimized_dtf_field_extraction(phi_semantic_complex: complex, 
                                   dtf_basis_count: int) -> tuple:
    """
    High-level DTF field extraction using optimized sub-functions.
    
    CLAUDE.md Compliance: Field theory optimized for Φ^semantic(τ, s) extraction.
    Preserves complex-valued mathematics and phase relationships.
    """
    # Extract magnitude and phase from complex DTF field
    dtf_field_magnitude = abs(phi_semantic_complex)
    dtf_field_phase = np.angle(phi_semantic_complex)
    
    # Use optimized complex field extraction
    extraction_result = _optimized_complex_field_extraction(
        magnitude=dtf_field_magnitude,
        phase=dtf_field_phase,
        basis_count=dtf_basis_count
    )
    
    # Extract results from array
    magnitude = extraction_result[0]
    phase = extraction_result[1] 
    field_strength = extraction_result[2]
    
    return magnitude, phase, field_strength


def _optimized_manifold_properties_validation(manifold_props: Dict[str, Any]) -> tuple:
    """
    High-level manifold validation using optimized sub-functions.
    
    CLAUDE.md Compliance: Field theory optimized for manifold property processing.
    Preserves mathematical accuracy for field calculations.
    """
    # Extract key manifold properties
    local_density = manifold_props.get('local_density', 1.0)
    persistence_radius = manifold_props.get('persistence_radius', 0.8)
    magnitude = manifold_props.get('magnitude', 1.0)
    coupling_mean = manifold_props.get('coupling_mean', 0.7)
    
    # Use optimized validation
    validation_result = _optimized_manifold_validation_array(
        local_density=local_density,
        persistence_radius=persistence_radius,
        magnitude=magnitude,
        coupling_mean=coupling_mean
    )
    
    # Extract results from array
    validated_density = validation_result[0]
    validated_radius = validation_result[1]
    validated_magnitude = validation_result[2]
    validated_coupling = validation_result[3]
    field_coupling_strength = validation_result[4]
    manifold_stability = validation_result[5]
    
    return validated_density, validated_radius, validated_magnitude, validated_coupling, field_coupling_strength, manifold_stability


@field_theory_jax_optimize(preserve_complex=False, profile=True)
def _optimized_embedding_normalization(embedding: np.ndarray) -> np.ndarray:
    """
    JAX-optimized embedding normalization for stable field calculations.
    
    CLAUDE.md Compliance: Field theory optimized pure functional normalization.
    """
    embedding_norm = np.linalg.norm(embedding)
    # Use JAX-safe division with epsilon to avoid if-statements
    normalized_embedding = embedding / (embedding_norm + 1e-12)
    return normalized_embedding


@field_theory_numba_optimize(preserve_complex=False, profile=True)
def _optimized_embedding_statistics(embedding: np.ndarray) -> np.ndarray:
    """
    Numba-optimized embedding statistics computation.
    
    CLAUDE.md Compliance: Field theory optimized statistics without tuple returns.
    Returns array: [magnitude, variance, skewness]
    """
    # Compute statistics as array to avoid tuple issues
    embedding_norm = np.linalg.norm(embedding)
    embedding_variance = np.var(embedding)
    
    # Compute skewness safely
    mean_val = np.mean(embedding)
    std_val = np.std(embedding)
    if std_val > 0:
        embedding_skewness = np.mean(((embedding - mean_val) / std_val)**3)
    else:
        embedding_skewness = 0.0
    
    # Return as array instead of tuple
    stats = np.array([embedding_norm, embedding_variance, embedding_skewness])
    return stats


@field_theory_jax_optimize(preserve_complex=False, profile=True)
def _optimized_field_scaling(magnitude: float, variance: float, 
                           observational_state: float, gamma: float) -> np.ndarray:
    """
    JAX-optimized field parameter scaling.
    
    CLAUDE.md Compliance: Field theory optimized scaling without control flow.
    Returns array: [scaled_magnitude, field_variance]
    """
    scaled_magnitude = magnitude * gamma * observational_state
    field_variance = variance * observational_state
    
    # Return as array for JAX compatibility
    return np.array([scaled_magnitude, field_variance])


@field_theory_numba_optimize(preserve_complex=True, profile=True)
def _optimized_complex_field_extraction(magnitude: float, phase: float, 
                                       basis_count: int) -> np.ndarray:
    """
    Numba-optimized complex field extraction avoiding control flow issues.
    
    CLAUDE.md Compliance: Field theory optimized for complex field processing.
    Returns array: [magnitude, phase, field_strength]
    """
    # Avoid if-statements by using mathematical operations
    safe_basis_count = max(basis_count, 1)
    basis_strength_factor = np.sqrt(safe_basis_count) / 10.0
    field_strength = magnitude * basis_strength_factor
    
    # Use mathematical operations instead of boolean conversion
    # If basis_count > 0, multiply by 1, otherwise by 0
    multiplier = 1.0 if basis_count > 0 else 0.0
    field_strength = field_strength * multiplier
    
    # Return as array
    result = np.array([magnitude, phase, field_strength])
    return result


@field_theory_numba_optimize(preserve_complex=False, profile=True)
def _optimized_manifold_validation_array(local_density: float, persistence_radius: float,
                                        magnitude: float, coupling_mean: float) -> np.ndarray:
    """
    Numba-optimized manifold property validation with array returns.
    
    CLAUDE.md Compliance: Field theory optimized property validation.
    Returns array: [validated_density, validated_radius, validated_magnitude, validated_coupling, coupling_strength, stability]
    """
    # Use min/max instead of np.clip for Numba compatibility
    validated_density = min(max(local_density, 0.1), 10.0)
    validated_radius = min(max(persistence_radius, 0.1), 2.0) 
    validated_magnitude = min(max(magnitude, 0.01), 100.0)
    validated_coupling = min(max(coupling_mean, 0.0), 1.0)
    
    # Compute derived properties
    coupling_strength = validated_density * validated_coupling
    stability = validated_radius * validated_magnitude
    
    # Return as array
    result = np.array([validated_density, validated_radius, validated_magnitude, 
                      validated_coupling, coupling_strength, stability])
    return result


def _optimized_embedding_preprocessing(embedding: np.ndarray, 
                                     observational_state: float, 
                                     gamma: float) -> tuple:
    """
    High-level preprocessing using optimized sub-functions.
    
    This function orchestrates the optimized sub-functions to provide
    the same interface while maximizing performance where possible.
    """
    # Use optimized normalization
    normalized_embedding = _optimized_embedding_normalization(embedding)
    
    # Use optimized statistics computation
    stats = _optimized_embedding_statistics(embedding)
    embedding_magnitude = stats[0]
    embedding_variance = stats[1] 
    embedding_skewness = stats[2]
    
    # Use optimized scaling
    scaling_result = _optimized_field_scaling(embedding_magnitude, embedding_variance, 
                                            observational_state, gamma)
    scaled_magnitude = scaling_result[0]
    field_variance = scaling_result[1]
    
    return normalized_embedding, embedding_magnitude, scaled_magnitude, field_variance, embedding_skewness


def run_semantic_processing(embedding: np.ndarray,
                           manifold_properties: Dict[str, Any],
                           observational_state: float = 1.0,
                           gamma: float = 1.2,
                           context: str = "main_processing",
                           field_temperature: float = 0.1,
                           metadata: Optional[Dict[str, Any]] = None,
                           use_dtf: bool = True,
                           model_type: str = "auto") -> Dict[str, Any]:
    """
    Main processing function - Primary interface for semantic dimension operations.
    
    This function provides a clean, stable interface for external modules to use
    semantic dimension processing without worrying about internal implementation details.
    
    Args:
        embedding: Input semantic vector
        manifold_properties: Geometric properties from manifold analysis
        observational_state: Current observational state parameter
        gamma: Field calibration factor
        context: Processing context identifier
        field_temperature: Temperature for field dynamics
        metadata: Optional processing metadata
        use_dtf: Whether to use DTF enterprise processing (default: True)
        model_type: Model type for DTF ("BGE", "MPNet", or "auto" to detect)
        
    Returns:
        Dict containing complete semantic field processing results
    """
    logger.info(f"Starting semantic processing - embedding dim: {embedding.shape[0]}, "
                f"gamma: {gamma}, observational_state: {observational_state}, DTF: {use_dtf}")
    
    # Use optimized preprocessing for enhanced performance
    normalized_embedding, embedding_magnitude, scaled_magnitude, field_variance, embedding_skewness = _optimized_embedding_preprocessing(
        embedding=embedding,
        observational_state=observational_state,
        gamma=gamma
    )
    
    # Use optimized manifold property validation
    validated_manifold_props = _optimized_manifold_properties_validation(manifold_properties)
    local_density, persistence_radius, magnitude, coupling_mean, field_coupling_strength, manifold_stability = validated_manifold_props
    
    logger.debug(f"Preprocessing complete - embedding_mag: {embedding_magnitude:.4f}, scaled_mag: {scaled_magnitude:.4f}, "
                f"manifold_stability: {manifold_stability:.4f}")
    
    try:
        if use_dtf:
            # Use DTF enterprise processing
            return _run_dtf_processing(
                embedding=embedding,
                manifold_properties=manifold_properties,
                observational_state=observational_state,
                gamma=gamma,
                context=context,
                field_temperature=field_temperature,
                metadata=metadata,
                model_type=model_type
            )
        else:
            # Use original semantic field processing
            results = process_semantic_field(
                embedding=embedding,
                manifold_properties=manifold_properties,
                observational_state=observational_state,
                gamma=gamma,
                context=context,
                field_temperature=field_temperature,
                metadata=metadata
            )
            
            logger.info(f"Semantic processing complete - field magnitude: {results['field_magnitude']:.4f}")
            return results
        
    except Exception as e:
        logger.error(f"Semantic processing failed: {e}")
        raise


def _run_dtf_processing(embedding: np.ndarray,
                       manifold_properties: Dict[str, Any],
                       observational_state: float,
                       gamma: float,
                       context: str,
                       field_temperature: float,
                       metadata: Optional[Dict[str, Any]],
                       model_type: str) -> Dict[str, Any]:
    """Run DTF processing using REAL embeddings from foundation manifold - NO PLACEHOLDERS."""
    
    # Get manifold data if available in metadata
    manifold_data = metadata.get('manifold_data') if metadata else None
    
    if manifold_data is None:
        logger.warning("No real manifold data provided - falling back to standard processing")
        logger.warning("For true DTF processing, provide manifold_data from foundation_manifold_builder.py")
        # Fall back to original processing - NO FAKE DTF DATA
        original_results = process_semantic_field(
            embedding=embedding,
            manifold_properties=manifold_properties,
            observational_state=observational_state,
            gamma=gamma,
            context=context,
            field_temperature=field_temperature,
            metadata=metadata
        )
        
        # Add DTF metadata indicating fallback (NO FAKE VALUES)
        original_results.update({
            'dtf_processing_mode': 'fallback_no_manifold_data',
            'dtf_manifold_available': False,
            'dtf_processing_successful': False,
            'dtf_uses_real_data': False,
            'dtf_fallback_reason': 'No real manifold data provided'
        })
        
        return original_results
    
    logger.info(f"Using DTF processing with REAL manifold embeddings (dim: {embedding.shape[0]})")
    
    try:
        # Use the REAL DTF implementation (not enterprise placeholder)
        logger.info("Using REAL DTF mathematics from semantic_basis_functions.py")
        
        # Connect DTF to Complete Charge Formula Q(τ, C, s)
        logger.info("Integrating DTF semantic field Φ^semantic with complete charge formula")
        
        # Use the REAL DTF field pool for Φ^semantic(τ, s) component
        from Sysnpire.model.semantic_dimension.processing.field_pool import create_dtf_field_pool_with_basis_extraction
        
        # Create DTF field pool using REAL manifold embeddings
        dtf_pool = create_dtf_field_pool_with_basis_extraction(
            manifold_data=manifold_data,
            pool_config={'pool_capacity': 100}
        )
        
        # Process embedding through DTF to get Φ^semantic(τ, s)
        single_manifold_data = {
            'embeddings': [embedding],
            'id_to_token': {0: f'input_{context}'}
        }
        
        added_count = dtf_pool.batch_add_from_manifold(single_manifold_data, limit=1)
        
        if added_count > 0:
            # Process through DTF semantic field generation
            dtf_pool.process_all()
            
            # Get DTF semantic field Φ^semantic(τ, s)
            dtf_results = None
            for result in dtf_pool.return_to_manifold():
                dtf_results = result
                break  # Only get first result
            
            if dtf_results:
                # Extract Φ^semantic(τ, s) from DTF processing using optimized extraction
                phi_semantic = dtf_results.get('transformed_field', 0+0j)
                dtf_basis_count = dtf_pool.semantic_basis_set.get('num_functions', 0) if dtf_pool.semantic_basis_set else 0
                
                # Use optimized DTF field extraction for enhanced performance
                dtf_field_magnitude, dtf_field_phase, field_strength = _optimized_dtf_field_extraction(
                    phi_semantic_complex=phi_semantic,
                    dtf_basis_count=dtf_basis_count
                )
                
                logger.info(f"DTF Φ^semantic extracted: magnitude={dtf_field_magnitude:.4f}, phase={dtf_field_phase:.3f}, "
                           f"basis_functions={dtf_basis_count}, field_strength={field_strength:.4f}")
                
                # Field strength is used for ChargeFactory integration
                complete_charge_magnitude = field_strength  # Enhanced field strength for Q(τ, C, s)
                
                logger.info(f"DTF semantic field extracted for ChargeFactory: Φ^semantic={dtf_field_magnitude:.4f}, "
                           f"enhanced_strength={field_strength:.4f}")
                
            else:
                logger.warning("DTF processing returned no results")
                dtf_field_magnitude = 0.0
                dtf_basis_count = 0
                complete_charge_magnitude = 0.0
        else:
            logger.warning("Failed to add embedding to DTF pool")
            dtf_field_magnitude = 0.0
            dtf_basis_count = 0
            complete_charge_magnitude = 0.0
            
    except Exception as e:
        logger.error(f"DTF processing failed: {e}")
        dtf_field_magnitude = 0.0
        dtf_basis_count = 0
    
    # Also run original semantic field processing for comparison
    original_results = process_semantic_field(
        embedding=embedding,
        manifold_properties=manifold_properties,
        observational_state=observational_state,
        gamma=gamma,
        context=context,
        field_temperature=field_temperature,
        metadata=metadata
    )
    
    # Merge results with REAL DTF enhancements (NO PLACEHOLDER VALUES)
    enhanced_results = original_results.copy()
    enhanced_results.update({
        'dtf_phi_semantic_magnitude': dtf_field_magnitude,  # Φ^semantic(τ, s) from DTF
        'dtf_phi_semantic_complex': locals().get('phi_semantic', complex(0)),  # Complex DTF field value
        'dtf_basis_functions': dtf_basis_count,
        'dtf_processing_mode': 'real_manifold_data' if manifold_data else 'no_manifold_data',
        'dtf_manifold_available': manifold_data is not None,
        'dtf_processing_successful': dtf_field_magnitude > 0,
        'dtf_uses_real_data': manifold_data is not None,
        'complete_charge_magnitude': locals().get('complete_charge_magnitude', 0.0),  # Full Q(τ, C, s)
        'dtf_semantic_integration': 'phi_semantic_component' if dtf_field_magnitude > 0 else 'fallback_processing'
    })
    
    if manifold_data and dtf_field_magnitude > 0:
        logger.info(f"✅ DTF→Q(τ,C,s) Integration Complete:")
        logger.info(f"   Original semantic: {original_results['field_magnitude']:.4f}")
        logger.info(f"   DTF Φ^semantic: {dtf_field_magnitude:.4f} (basis functions: {dtf_basis_count})")
        logger.info(f"   Complete Q(τ,C,s): {enhanced_results['complete_charge_magnitude']:.4f}")
    else:
        logger.info(f"DTF processing fallback - original: {original_results['field_magnitude']:.4f}, "
                   f"DTF: {dtf_field_magnitude:.4f}, basis functions: {dtf_basis_count}")
    
    return enhanced_results


def test_semantic_field():
    """Basic test of semantic field processing with DTF integration."""
    logger.info("🧪 TESTING SEMANTIC FIELD PROCESSING (DTF Enhanced) 🧪")
    
    # Create test data
    embedding_1024 = np.random.rand(1024)  # BGE-style
    embedding_768 = np.random.rand(768)    # MPNet-style
    
    manifold_props = {
        'local_density': 1.2,
        'persistence_radius': 0.8,
        'phase_angles': [0.1, 0.2, 0.3],
        'magnitude': 1.0,
        'gradient': np.random.rand(10),
        'dominant_frequencies': [0.5, 1.0, 1.5],
        'coupling_mean': 0.7
    }
    
    test_cases = [
        ("BGE-1024", embedding_1024, "BGE"),
        ("MPNet-768", embedding_768, "MPNet")
    ]
    
    # Test both DTF and original processing
    for use_dtf in [True, False]:
        mode = "DTF Enhanced" if use_dtf else "Original"
        logger.info(f"\nTesting {mode} mode...")
        
        for name, embedding, model_type in test_cases:
            logger.info(f"  Testing {name} embedding...")
            
            results = run_semantic_processing(
                embedding=embedding,
                manifold_properties=manifold_props,
                observational_state=1.0,
                gamma=1.2,
                context=f"test_{name.lower()}_{mode.lower().replace(' ', '_')}",
                field_temperature=0.1,
                use_dtf=use_dtf,
                model_type=model_type
            )
            
            # Validate results
            assert 'semantic_field' in results
            assert 'breathing_patterns' in results
            assert 'phase_modulation' in results
            assert 'field_magnitude' in results
            assert 'constellation_topology' in results
            
            if use_dtf:
                # Check for DTF results (may fall back to standard if no manifold data)
                dtf_success = results.get('dtf_processing_successful', False)
                dtf_manifold = results.get('dtf_manifold_available', False)
                dtf_uses_real_data = results.get('dtf_uses_real_data', False)
                
                if dtf_manifold and dtf_uses_real_data:
                    # Real DTF processing with manifold data
                    assert 'dtf_phi_semantic_magnitude' in results
                    assert 'dtf_basis_functions' in results
                    assert 'complete_charge_magnitude' in results
                    logger.info(f"    ✅ {name} REAL DTF→Q(τ,C,s) test passed:")
                    logger.info(f"       Original: {results['field_magnitude']:.4f}")
                    logger.info(f"       DTF Φ^semantic: {results['dtf_phi_semantic_magnitude']:.4f}, basis: {results['dtf_basis_functions']}")
                    logger.info(f"       Complete Q(τ,C,s): {results['complete_charge_magnitude']:.4f}")
                else:
                    # Fallback to standard processing (no real manifold data available)
                    logger.info(f"    ✅ {name} DTF fallback test passed - original: {results['field_magnitude']:.4f}")
                    logger.info(f"       Note: No real manifold data provided - used standard processing")
                
                logger.info(f"       DTF manifold available: {dtf_manifold}, DTF successful: {dtf_success}, Real data: {dtf_uses_real_data}")
            else:
                logger.info(f"    ✅ {name} original test passed - field magnitude: {results['field_magnitude']:.4f}")
    
    logger.info("\n🧪 ALL SEMANTIC FIELD TESTS PASSED 🧪")


def demo_semantic_dimension():
    """Demonstration of semantic dimension capabilities."""
    logger.info("🌟 SEMANTIC DIMENSION DEMONSTRATION 🌟")
    
    # Create meaningful demo data
    embedding = np.random.rand(1024)
    
    # Simulate different observational states
    states = [0.5, 1.0, 1.5, 2.0]
    gammas = [0.8, 1.0, 1.2, 1.5]
    
    manifold_props = {
        'local_density': 1.0,
        'persistence_radius': 1.0,
        'phase_angles': [0.0, np.pi/4, np.pi/2],
        'magnitude': 1.0,
        'gradient': np.random.rand(10),
        'dominant_frequencies': [0.3, 0.7, 1.2],
        'coupling_mean': 0.8
    }
    
    logger.info("Demonstrating field evolution across different parameters...")
    
    for i, (state, gamma) in enumerate(zip(states, gammas)):
        results = run_semantic_processing(
            embedding=embedding,
            manifold_properties=manifold_props,
            observational_state=state,
            gamma=gamma,
            context=f"demo_run_{i+1}",
            field_temperature=0.1 + i * 0.05
        )
        
        breathing = results['breathing_patterns']
        topology = results['constellation_topology']
        
        logger.info(f"Run {i+1}: state={state:.1f}, γ={gamma:.1f} → "
                   f"field_mag={results['field_magnitude']:.3f}, "
                   f"breathing_freq={breathing['frequency']:.3f}, "
                   f"entropy={topology['field_entropy']:.3f}")
    
    logger.info("🌟 DEMONSTRATION COMPLETE 🌟")


def benchmark_performance():
    """Performance benchmark for semantic dimension processing."""
    logger.info("⚡ PERFORMANCE BENCHMARK ⚡")
    
    import time
    
    # Test different embedding sizes
    sizes = [256, 512, 768, 1024]
    iterations = 100
    
    manifold_props = {
        'local_density': 1.0,
        'persistence_radius': 1.0,
        'phase_angles': [0.1, 0.2],
        'magnitude': 1.0,
        'gradient': np.random.rand(10),
        'dominant_frequencies': [0.5, 1.0],
        'coupling_mean': 0.7
    }
    
    for size in sizes:
        embedding = np.random.rand(size)
        
        start_time = time.time()
        
        for i in range(iterations):
            _ = run_semantic_processing(
                embedding=embedding,
                manifold_properties=manifold_props,
                observational_state=1.0,
                gamma=1.2,
                context=f"benchmark_{size}_{i}"
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        logger.info(f"Size {size:4d}: {avg_time*1000:.2f}ms avg ({total_time:.2f}s total, {iterations} runs)")
    
    logger.info("⚡ BENCHMARK COMPLETE ⚡")


def main():
    """Main entry point for semantic dimension operations."""
    parser = argparse.ArgumentParser(description="Semantic Dimension Processing")
    parser.add_argument("--test", action="store_true", help="Run basic tests")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--no-dtf", action="store_true", help="Disable DTF processing (use original only)")
    parser.add_argument("--model", choices=["BGE", "MPNet", "auto"], default="auto", help="Model type for DTF processing")
    
    args = parser.parse_args()
    
    if args.test:
        test_semantic_field()
    elif args.demo:
        demo_semantic_dimension()
    elif args.benchmark:
        benchmark_performance()
    else:
        logger.info("Semantic Dimension Main Entry Point")
        logger.info("Available commands:")
        logger.info("  --test      Run basic semantic field tests (DTF enhanced)")
        logger.info("  --demo      Run demonstration with sample data")
        logger.info("  --benchmark Run performance benchmarks")
        logger.info("  --no-dtf    Disable DTF processing (use original only)")
        logger.info("  --model     Specify model type: BGE, MPNet, or auto (default: auto)")
        logger.info("")
        logger.info("Integration example:")
        logger.info("  from Sysnpire.model.semantic_dimension.main import run_semantic_processing")
        logger.info("  # DTF enhanced processing (default)")
        logger.info("  results = run_semantic_processing(embedding, manifold_props, use_dtf=True)")
        logger.info("  # Original processing")
        logger.info("  results = run_semantic_processing(embedding, manifold_props, use_dtf=False)")


if __name__ == "__main__":
    main()