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

logger = get_logger(__name__)


def run_semantic_processing(embedding: np.ndarray,
                           manifold_properties: Dict[str, Any],
                           observational_state: float = 1.0,
                           gamma: float = 1.2,
                           context: str = "main_processing",
                           field_temperature: float = 0.1,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        
    Returns:
        Dict containing complete semantic field processing results
    """
    logger.info(f"Starting semantic processing - embedding dim: {embedding.shape[0]}, "
                f"gamma: {gamma}, observational_state: {observational_state}")
    
    try:
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


def test_semantic_field():
    """Basic test of semantic field processing."""
    logger.info("🧪 TESTING SEMANTIC FIELD PROCESSING 🧪")
    
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
        ("BGE-1024", embedding_1024),
        ("MPNet-768", embedding_768)
    ]
    
    for name, embedding in test_cases:
        logger.info(f"Testing {name} embedding...")
        
        results = run_semantic_processing(
            embedding=embedding,
            manifold_properties=manifold_props,
            observational_state=1.0,
            gamma=1.2,
            context=f"test_{name.lower()}",
            field_temperature=0.1
        )
        
        # Validate results
        assert 'semantic_field' in results
        assert 'breathing_patterns' in results
        assert 'phase_modulation' in results
        assert 'field_magnitude' in results
        assert 'constellation_topology' in results
        
        logger.info(f"✅ {name} test passed - field magnitude: {results['field_magnitude']:.4f}")
    
    logger.info("🧪 ALL SEMANTIC FIELD TESTS PASSED 🧪")


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
        logger.info("  --test      Run basic semantic field tests")
        logger.info("  --demo      Run demonstration with sample data")
        logger.info("  --benchmark Run performance benchmarks")
        logger.info("")
        logger.info("Integration example:")
        logger.info("  from Sysnpire.model.semantic_dimension.main import run_semantic_processing")


if __name__ == "__main__":
    main()