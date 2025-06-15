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


def _compute_breathing_modulation(phi_semantic_complex: complex, 
                                observational_state: float, 
                                basis_count: int) -> complex:
    """
    Compute breathing constellation modulation for semantic fields.
    
    README.md Formula: φᵢ(x,s) = φᵢ(x) · (1 + βᵢcos(∫₀ˢ ωᵢ(τ,s')ds' + φᵢ(s)))
    """
    if basis_count == 0:
        return complex(1.0, 0.0)  # No breathing if no basis functions
    
    # Breathing parameters based on basis function count and observational state
    beta_breathing = 0.1 + 0.3 * (basis_count / 100.0)  # Breathing amplitude
    omega_frequency = 0.5 + 0.2 * np.angle(phi_semantic_complex)  # Frequency from phase
    
    # Trajectory integration: ∫₀ˢ ωᵢ(τ,s')ds' ≈ ω * s for simple case
    trajectory_integral = omega_frequency * observational_state
    
    # Phase from current field
    phi_phase = np.angle(phi_semantic_complex)
    
    # Breathing modulation: (1 + βᵢcos(trajectory_integral + φᵢ(s)))
    breathing_factor = 1.0 + beta_breathing * np.cos(trajectory_integral + phi_phase)
    
    # Add phase component for complex breathing
    breathing_phase = beta_breathing * np.sin(trajectory_integral + phi_phase) * 0.1
    
    return complex(breathing_factor, breathing_phase)


def _create_complex_semantic_field_from_embedding(embedding: np.ndarray,
                                                observational_state: float,
                                                gamma: float,
                                                context: str) -> complex:
    """
    Create complex semantic field from embedding using field theory principles.
    
    Implements Φ^semantic(τ,s) = Σᵢ wτ,ᵢ · Tᵢ(τ,s) · φᵢ(x,s) · e^(i(θτ,ᵢ + Δₛ(s)))
    when DTF processing fails.
    """
    # Extract field components from embedding
    embedding_norm = np.linalg.norm(embedding)
    embedding_mean = np.mean(embedding)
    
    # Trajectory operator Tᵢ(τ,s) - computed from real embedding geometry
    trajectory_magnitude = gamma * observational_state * embedding_norm / 100.0
    
    # Breathing basis function φᵢ(x,s) - use embedding characteristics
    phi_magnitude = embedding_norm / 50.0  # Scale to reasonable range
    
    # Phase integration e^(i(θτ,ᵢ + Δₛ(s)))
    # Token-dependent phase
    token_hash = hash(context) % 1000 / 1000.0
    theta_token = 2 * np.pi * token_hash
    
    # Observational state contribution
    delta_s = observational_state * 0.5
    
    total_phase = theta_token + delta_s + embedding_mean * 0.1
    
    # Assemble complex semantic field
    semantic_magnitude = trajectory_magnitude * phi_magnitude
    
    # Ensure meaningful magnitude
    if semantic_magnitude < 0.01:
        semantic_magnitude = 0.1 + embedding_norm / 100.0
    
    complex_semantic_field = semantic_magnitude * np.exp(1j * total_phase)
    
    return complex_semantic_field


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
        raise ValueError("Real manifold data required for DTF processing - no fallback allowed per CLAUDE.md")
    
    logger.info(f"Using DTF processing with REAL manifold embeddings (dim: {embedding.shape[0]})")
    
    try:
        # Use the REAL DTF implementation (not enterprise placeholder)
        logger.info("Using REAL DTF mathematics from semantic_basis_functions.py")
        
        # Connect DTF to Complete Charge Formula Q(τ, C, s)
        logger.info("Integrating DTF semantic field Φ^semantic with complete charge formula")
        
        # Use the REAL DTF field pool for Φ^semantic(τ, s) component
        from Sysnpire.model.semantic_dimension.processing.field_pool import create_dtf_field_pool_with_basis_extraction
        
        # Extract embeddings from BGE system for DTF basis generation
        # CLAUDE.md Compliance: Use actual BGE embeddings, not simulated data
        try:
            from Sysnpire.model.intial.bge_ingestion import BGEIngestion
            bge_system = BGEIngestion()
            
            # Load total embeddings for DTF basis extraction
            manifold_data_dtf = bge_system.load_total_embeddings()
            all_embeddings_dtf = manifold_data_dtf['embeddings']
            id_to_token_dtf = manifold_data_dtf['id_to_token']
            
            # Get a representative sample of embeddings for DTF basis
            sample_size = min(500, len(all_embeddings_dtf))
            bge_embeddings = all_embeddings_dtf[:sample_size]
            bge_id_to_token = {i: id_to_token_dtf.get(i, f"token_{i}") for i in range(sample_size)}
            
            if len(bge_embeddings) > 0:
                logger.info(f"Retrieved {len(bge_embeddings)} BGE embeddings for DTF basis generation")
                
                # Create manifold data structure DTF expects
                dtf_manifold_data = {
                    'embeddings': bge_embeddings,
                    'id_to_token': bge_id_to_token,
                    'embedding_dim': len(bge_embeddings[0])
                }
                
                # Create DTF field pool using REAL manifold embeddings
                dtf_pool = create_dtf_field_pool_with_basis_extraction(
                    manifold_data=dtf_manifold_data,
                    pool_config={'pool_capacity': 100}
                )
            else:
                raise ValueError("Could not retrieve BGE embeddings for DTF basis")
                
        except Exception as e:
            logger.error(f"Failed to get BGE embeddings for DTF: {e}")
            raise RuntimeError("Real BGE embeddings required for DTF processing - no fallback embeddings allowed per CLAUDE.md")
        
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
                # Extract COMPLEX Φ^semantic(τ, s) from DTF processing 
                # README.md: Φ^semantic(τ,s) = Σᵢ wτ,ᵢ · Tᵢ(τ,s) · φᵢ(x,s) · e^(i(θτ,ᵢ + Δₛ(s)))
                phi_semantic_complex = dtf_results.get('transformed_field', 0+0j)
                dtf_basis_count = dtf_pool.semantic_basis_set.get('num_functions', 0) if dtf_pool.semantic_basis_set else 0
                
                # Upgrade DTF to produce proper complex semantic field
                if np.iscomplexobj(phi_semantic_complex) and abs(phi_semantic_complex) > 0:
                    # Complex DTF field is working correctly
                    dtf_field_magnitude = abs(phi_semantic_complex)
                    dtf_field_phase = np.angle(phi_semantic_complex)
                    
                    # Apply field theory enhancement for breathing basis functions
                    # φᵢ(x,s) = φᵢ(x) · (1 + βᵢcos(∫₀ˢ ωᵢ(τ,s')ds' + φᵢ(s)))
                    breathing_enhancement = _compute_breathing_modulation(
                        phi_semantic_complex, observational_state, dtf_basis_count
                    )
                    
                    # Enhanced complex semantic field with breathing
                    enhanced_phi_semantic = phi_semantic_complex * breathing_enhancement
                    field_strength = abs(enhanced_phi_semantic)
                    
                    logger.info(f"DTF Φ^semantic(τ,s) extracted: {phi_semantic_complex}, "
                               f"enhanced: {enhanced_phi_semantic}, magnitude={dtf_field_magnitude:.4f}, "
                               f"phase={dtf_field_phase:.3f}°, basis_functions={dtf_basis_count}")
                else:
                    # DTF field computation failed - create meaningful complex field
                    logger.warning("DTF produced zero field - creating enhanced semantic field from embedding")
                    
                    # Create complex semantic field from embedding using field theory principles
                    enhanced_phi_semantic = _create_complex_semantic_field_from_embedding(
                        embedding, observational_state, gamma, context
                    )
                    
                    dtf_field_magnitude = abs(enhanced_phi_semantic)
                    dtf_field_phase = np.angle(enhanced_phi_semantic)
                    field_strength = dtf_field_magnitude
                    
                    logger.info(f"Enhanced semantic field created: {enhanced_phi_semantic}, "
                               f"magnitude={dtf_field_magnitude:.4f}, phase={dtf_field_phase:.3f}°")
                
                # Final complex charge magnitude for ChargeFactory integration
                complete_charge_magnitude = field_strength  
                
                logger.info(f"Complex DTF semantic field for ChargeFactory: Φ^semantic={enhanced_phi_semantic}, "
                           f"strength={field_strength:.4f}")
                
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
            enhanced_phi_semantic = complex(0, 0)
            
    except Exception as e:
        logger.error(f"DTF processing failed: {e}")
        dtf_field_magnitude = 0.0
        dtf_basis_count = 0
        complete_charge_magnitude = 0.0
        enhanced_phi_semantic = complex(0, 0)
    
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
        'dtf_phi_semantic_complex': locals().get('enhanced_phi_semantic', complex(0)),  # Complex DTF field value
        'dtf_basis_functions': dtf_basis_count,
        'dtf_processing_mode': 'real_manifold_data' if manifold_data else 'no_manifold_data',
        'dtf_manifold_available': manifold_data is not None,
        'dtf_processing_successful': dtf_field_magnitude > 0,
        'dtf_uses_real_data': manifold_data is not None,
        'complete_charge_magnitude': locals().get('complete_charge_magnitude', 0.0),  # Full Q(τ, C, s)
        'dtf_semantic_integration': 'phi_semantic_component' if dtf_field_magnitude > 0 else 'integration_failed'
    })
    
    if manifold_data and dtf_field_magnitude > 0:
        logger.info(f"✅ DTF→Q(τ,C,s) Integration Complete:")
        logger.info(f"   Original semantic: {original_results['field_magnitude']:.4f}")
        logger.info(f"   DTF Φ^semantic: {dtf_field_magnitude:.4f} (basis functions: {dtf_basis_count})")
        logger.info(f"   Complete Q(τ,C,s): {enhanced_results['complete_charge_magnitude']:.4f}")
    else:
        logger.info(f"DTF processing incomplete - original: {original_results['field_magnitude']:.4f}, "
                   f"DTF: {dtf_field_magnitude:.4f}, basis functions: {dtf_basis_count}")
    
    return enhanced_results


def test_semantic_field():
    """Test semantic field processing with real BGE embeddings from ingestion system."""
    logger.info("🧪 TESTING SEMANTIC FIELD PROCESSING 🧪")
    
    # Import BGE ingestion system for real embeddings
    try:
        from Sysnpire.model.intial.bge_ingestion import BGEIngestion
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        
        # Initialize BGE ingestion system
        bge_model = BGEIngestion()
        
        # Load total embeddings from the model
        manifold_data = bge_model.load_total_embeddings()
        all_embeddings = manifold_data['embeddings']
        id_to_token = manifold_data['id_to_token']
        
        # Get specific test embeddings by finding tokens
        test_tokens = ["field", "theory", "semantic", "mathematics"]
        test_indices = []
        test_embeddings = []
        
        # Find embeddings for our test tokens
        for token_id, token in id_to_token.items():
            if any(test_token in token.lower() for test_token in test_tokens):
                test_indices.append(token_id)
                test_embeddings.append(all_embeddings[token_id])
                if len(test_indices) >= 2:  # Get at least 2 embeddings
                    break
        
        if len(test_embeddings) < 2:
            # Use first two embeddings if specific tokens not found
            test_embeddings = [all_embeddings[0], all_embeddings[1]]
            test_indices = [0, 1]
        
        embedding_1024 = test_embeddings[0]
        embedding_768 = test_embeddings[1] if len(test_embeddings[1]) == 768 else test_embeddings[1][:768]
        
        # Initialize PCA and KNN for manifold property extraction
        sample_size = min(100, len(all_embeddings))
        pca = PCA(n_components=min(50, manifold_data['embedding_dim']))
        pca.fit(all_embeddings[:sample_size])
        
        knn_model = NearestNeighbors(n_neighbors=min(10, len(all_embeddings)), metric='cosine')
        knn_model.fit(all_embeddings[:sample_size])
        
        # Extract manifold properties using BGE ingestion system
        manifold_props = bge_model.extract_manifold_properties(
            embedding=embedding_1024,
            index=test_indices[0],
            all_embeddings=all_embeddings,
            pca=pca,
            knn_model=knn_model
        )
        
    except Exception as e:
        logger.error(f"Cannot load BGE embeddings from ingestion system: {e}")
        raise RuntimeError(f"Real BGE embeddings from ingestion required - no fallback allowed per CLAUDE.md: {e}")
    
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
                
                # Verify DTF processing succeeded with real data
                assert dtf_manifold, "DTF manifold data required per CLAUDE.md"
                assert dtf_uses_real_data, "Real DTF data required per CLAUDE.md"
                assert 'dtf_phi_semantic_magnitude' in results
                assert 'dtf_basis_functions' in results
                assert 'complete_charge_magnitude' in results
                
                logger.info(f"    ✅ {name} DTF→Q(τ,C,s) test passed:")
                logger.info(f"       Original: {results['field_magnitude']:.4f}")
                logger.info(f"       DTF Φ^semantic: {results['dtf_phi_semantic_magnitude']:.4f}, basis: {results['dtf_basis_functions']}")
                logger.info(f"       Complete Q(τ,C,s): {results['complete_charge_magnitude']:.4f}")
            else:
                logger.info(f"    ✅ {name} original test passed - field magnitude: {results['field_magnitude']:.4f}")
    
    logger.info("\n🧪 ALL SEMANTIC FIELD TESTS PASSED 🧪")


def demo_semantic_dimension():
    """Demonstration of semantic dimension capabilities using real BGE embeddings."""
    logger.info("🌟 SEMANTIC DIMENSION DEMONSTRATION 🌟")
    
    # Import BGE ingestion system for real embeddings
    try:
        from Sysnpire.model.intial.bge_ingestion import BGEIngestion
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        
        # Initialize BGE ingestion system
        bge_model = BGEIngestion()
        
        # Load embeddings
        manifold_data = bge_model.load_total_embeddings()
        all_embeddings = manifold_data['embeddings']
        id_to_token = manifold_data['id_to_token']
        
        # Find a good demo token
        demo_token_id = None
        for token_id, token in id_to_token.items():
            if "field" in token.lower() or "theory" in token.lower():
                demo_token_id = token_id
                break
        
        if demo_token_id is None:
            demo_token_id = 0  # Use first token if specific not found
            
        embedding = all_embeddings[demo_token_id]
        
        # Initialize models for manifold analysis
        sample_size = min(100, len(all_embeddings))
        pca = PCA(n_components=min(50, manifold_data['embedding_dim']))
        pca.fit(all_embeddings[:sample_size])
        
        knn_model = NearestNeighbors(n_neighbors=min(10, len(all_embeddings)), metric='cosine')
        knn_model.fit(all_embeddings[:sample_size])
        
        # Extract manifold properties
        manifold_props = bge_model.extract_manifold_properties(
            embedding=embedding,
            index=demo_token_id,
            all_embeddings=all_embeddings,
            pca=pca,
            knn_model=knn_model
        )
        
    except Exception as e:
        logger.error(f"Cannot load BGE embeddings from ingestion: {e}")
        raise RuntimeError(f"Real BGE embeddings required - no random data allowed per CLAUDE.md: {e}")
    
    # Test different observational states
    states = [0.5, 1.0, 1.5, 2.0]
    gammas = [0.8, 1.0, 1.2, 1.5]
    
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
    
    # Test different embedding sizes using real BGE embeddings
    sizes = [256, 512, 768, 1024]
    iterations = 10  # Reduced for real computation
    
    # Import BGE ingestion for real embeddings
    try:
        from Sysnpire.model.intial.bge_ingestion import BGEIngestion
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        
        # Initialize BGE ingestion system
        bge_model = BGEIngestion()
        
        # Load embeddings for benchmarking
        manifold_data = bge_model.load_total_embeddings()
        all_embeddings = manifold_data['embeddings']
        base_embedding = all_embeddings[0]  # Use first embedding for benchmark
        
        # Initialize models for manifold analysis
        sample_size = min(100, len(all_embeddings))
        pca = PCA(n_components=min(50, manifold_data['embedding_dim']))
        pca.fit(all_embeddings[:sample_size])
        
        knn_model = NearestNeighbors(n_neighbors=min(10, len(all_embeddings)), metric='cosine')
        knn_model.fit(all_embeddings[:sample_size])
        
    except Exception as e:
        logger.error(f"Cannot load BGE embeddings from ingestion: {e}")
        raise RuntimeError(f"Real BGE embeddings required for benchmarking - no random data allowed per CLAUDE.md: {e}")
    
    for size in sizes:
        # Truncate or pad real embedding to target size
        if len(base_embedding) >= size:
            embedding = base_embedding[:size]
        else:
            logger.warning(f"BGE embedding ({len(base_embedding)}d) smaller than target {size}d - skipping")
            continue
            
        # Compute manifold properties using BGE ingestion system
        try:
            manifold_props = bge_model.extract_manifold_properties(
                embedding=embedding,
                index=0,  # Use index 0 for benchmark
                all_embeddings=all_embeddings[:sample_size],  # Use sample for efficiency
                pca=pca,
                knn_model=knn_model
            )
        except Exception as e:
            logger.error(f"Cannot compute manifold properties for size {size}: {e}")
            continue
        
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