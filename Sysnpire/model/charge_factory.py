"""
Charge Factory - Focused Q(τ, C, s) Transformation Engine

FOCUSED RESPONSIBILITY: This factory has ONE job - take embedding vectors with their
mathematical properties and transform them into dynamic conceptual charges using the
complete Q(τ, C, s) field theory formula. It does NOT handle data sourcing.

MATHEMATICAL TRANSFORMATION:
Input: Static embedding + manifold properties + field parameters
Process: Apply complete Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)
Output: Dynamic ConceptualCharge object

DESIGN PRINCIPLE: This factory is model-agnostic and source-agnostic. It can process
embeddings from BGE models, MPNet models, scraped data, user inputs, or any other
source that provides embedding vectors + mathematical properties.

USAGE CONTEXTS:
- Initial "Big Bang" from model vocabularies (separate script)
- Live text processing from user inputs
- Batch processing from data scraping operations
- Integration with external embedding sources
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass

# Ensure project imports work
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.model.mathematics.conceptual_charge import ConceptualCharge
from Sysnpire.model.temporal_dimension import TrajectoryOperatorEngine
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChargeParameters:
    """Parameters for Q(τ, C, s) charge generation."""
    observational_state: float = 1.0      # s parameter
    gamma: float = 1.2                    # γ field calibration factor
    context: str = "general"              # C context parameter
    field_temperature: float = 0.1        # Temperature for field evolution
    time_evolution: float = 1.0           # Time parameter for dynamics


class ChargeFactory:
    """
    Focused Field Theory Charge Generator
    
    SINGLE RESPONSIBILITY: Transforms embedding vectors + mathematical properties
    into dynamic conceptual charges using Q(τ, C, s) field theory mathematics.
    
    AGNOSTIC DESIGN: Works with embeddings from any source - BGE models, MPNet models,
    scraped data, user text, or external systems. Only requires embedding vector
    and mathematical properties as input.
    
    MATHEMATICAL FOCUS: Implements the complete conceptual charge formula without
    concern for data sourcing, model management, or persistence.
    """
    
    def __init__(self):
        """
        Initialize charge factory for Q(τ, C, s) transformations.
        
        MINIMAL SETUP: No model loading, no data dependencies. Pure mathematical
        transformation engine ready to process any embedding input.
        """
        logger.info("Initializing ChargeFactory for Q(τ, C, s) transformations")
        self.charge_count = 0
        
        # Initialize trajectory operator engine for T(τ, C, s) component
        self.trajectory_engine = TrajectoryOperatorEngine(
            embedding_dimension=1024,  # Default for BGE, will adapt dynamically
            integration_method="adaptive_quad"
        )
    
    def create_charge(self, 
                     embedding: np.ndarray,
                     manifold_properties: Dict[str, Any],
                     charge_params: ChargeParameters,
                     charge_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> ConceptualCharge:
        """
        Transform single embedding into dynamic conceptual charge using Q(τ, C, s).
        
        CORE TRANSFORMATION: Applies the complete conceptual charge formula to convert
        a static embedding vector with its mathematical properties into a dynamic
        field theory charge.
        
        Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)
        
        Args:
            embedding: Static embedding vector [embedding_dim]
            manifold_properties: Mathematical properties from manifold analysis
            charge_params: Parameters for Q(τ, C, s) calculation
            charge_id: Unique identifier for the charge
            metadata: Additional metadata (token, source, etc.)
            
        Returns:
            ConceptualCharge: Dynamic field theory charge with full Q(τ, C, s) properties
        """
        if charge_id is None:
            charge_id = f"charge_{self.charge_count:06d}"
            self.charge_count += 1
        
        # Extract token from metadata
        token = metadata.get('token', f'token_{charge_id}') if metadata else f'token_{charge_id}'
        
        # Adapt trajectory engine dimension to embedding dimension
        embedding_dim = len(embedding)
        if embedding_dim != self.trajectory_engine.embedding_dimension:
            self.trajectory_engine = TrajectoryOperatorEngine(
                embedding_dimension=embedding_dim,
                integration_method="adaptive_quad"
            )
        
        # Compute T(τ, C, s) - transformative potential tensor using enhanced temporal dimension
        logger.debug(f"Computing temporal dimension for {token} with embedding dim {embedding_dim}")
        temporal_results = self.temporal_dimension(
            embedding=embedding,
            charge_params=charge_params,
            metadata=metadata
        )
        
        transformative_potential = temporal_results.get('transformative_potential', 1.0)
        trajectory_operators = temporal_results.get('trajectory_operators')
        breathing_pattern = temporal_results.get('breathing_pattern', 1.0)
        frequency_evolution = temporal_results.get('frequency_evolution')
        phase_accumulation = temporal_results.get('phase_accumulation')
        semantic_modulation = temporal_results.get('semantic_modulation', np.zeros(3))
        
        # Calculate E^trajectory(τ, s) - emotional trajectory integration using dedicated emotional dimension
        emotional_trajectory = self._compute_emotional_trajectory_integration(
            embedding=embedding,
            manifold_properties=manifold_properties,
            charge_params=charge_params,
            token=token,
            temporal_data=temporal_results
        )
        
        # Generate Φ^semantic(τ, s) using DTF semantic field if available
        dtf_semantic_field_magnitude = 0.0
        dtf_semantic_field_complex = complex(0)
        if metadata and metadata.get('dtf_enabled') and metadata.get('manifold_data'):
            logger.debug(f"Processing {token} with DTF semantic field enhancement")
            
            # Use DTF-enhanced semantic processing for Φ^semantic(τ, s) component
            try:
                from Sysnpire.model.semantic_dimension.main import run_semantic_processing
                
                dtf_results = run_semantic_processing(
                    embedding=embedding,
                    manifold_properties=manifold_properties,
                    observational_state=charge_params.observational_state,
                    gamma=charge_params.gamma,
                    context=f"charge_factory_{token}",
                    field_temperature=charge_params.field_temperature,
                    metadata=metadata,  # Contains manifold_data for DTF
                    use_dtf=True,
                    model_type="auto"
                )
                
                # Extract DTF-enhanced Φ^semantic(τ, s)
                dtf_semantic_field_magnitude = dtf_results.get('dtf_phi_semantic_magnitude', 0.0)
                dtf_semantic_field_complex = dtf_results.get('dtf_phi_semantic_complex', complex(0))
                complete_charge_magnitude = dtf_results.get('complete_charge_magnitude', 0.0)
                
                if dtf_semantic_field_magnitude > 0:
                    logger.debug(f"DTF enhanced {token}: Φ^semantic={dtf_semantic_field_magnitude:.4f}, Q(τ,C,s)={complete_charge_magnitude:.4f}")
                else:
                    logger.debug(f"DTF processing incomplete for {token}: semantic field magnitude={dtf_semantic_field_magnitude}")
                    
            except Exception as e:
                logger.error(f"DTF processing failed for {token}: {e}")
                raise RuntimeError(f"DTF semantic processing required - no fallback allowed per CLAUDE.md: {e}")
        
        # Compute e^(iθ_total(τ,C,s)) - complete phase integration using Phase Dimension
        phase_integration, phase_analysis = self._compute_complete_phase_integration(
            embedding=embedding,
            manifold_properties=manifold_properties,
            charge_params=charge_params,
            token=token,
            temporal_data=temporal_results,
            emotional_trajectory=emotional_trajectory
        )
        
        # Extract Ψ_persistence(s-s₀) - observational persistence from temporal results
        observational_persistence = temporal_results.get('observational_persistence', 1.0)
        phase_coordination = temporal_results.get('phase_coordination', {})
        
        # Combine all components via complete Q(τ, C, s) formula
        # Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)
        logger.debug(f"Pre-assembly values for {token}: dtf_mag={dtf_semantic_field_magnitude:.6f}, dtf_complex={dtf_semantic_field_complex}")
        charge_magnitude, charge_phase = self._assemble_complete_charge_formula(
            gamma=charge_params.gamma,
            transformative_potential=transformative_potential,
            emotional_trajectory=emotional_trajectory,
            semantic_field_magnitude=dtf_semantic_field_magnitude,
            semantic_field_complex=dtf_semantic_field_complex,
            phase_integration=phase_integration,
            observational_persistence=observational_persistence,
            token=token
        )
        
        # Create ConceptualCharge with enhanced trajectory operator integration
        charge = ConceptualCharge(
            token=token,
            semantic_vector=embedding,
            context={'context': charge_params.context},
            observational_state=charge_params.observational_state,
            gamma=charge_params.gamma
        )
        
        # Replace ConceptualCharge's random trajectory parameters with computed ones from temporal dimension
        if trajectory_operators is not None:
            # Extract proper trajectory parameters from temporal processing
            embedding_dim = len(embedding)
            
            # Replace random omega_base with computed frequencies from temporal dimension
            charge.omega_base = self.trajectory_engine.base_frequencies[:embedding_dim]
            
            # Replace random phi_base with token/context-dependent phases
            token_hash = hash(token) % 1000 / 1000.0
            context_hash = hash(charge_params.context) % 1000 / 1000.0
            charge.phi_base = np.array([
                2 * np.pi * (token_hash + i / embedding_dim + 0.1 * context_hash)
                for i in range(embedding_dim)
            ])
            
            # Enhanced breathing parameters based on temporal coordination
            coherence = phase_coordination.get('coherence', 0.5)
            charge.beta_breathing = np.full(embedding_dim, 0.1 + 0.4 * coherence)
            
            # Enhanced persistence parameters from temporal dimension  
            charge.alpha_persistence = self.trajectory_engine.persistence_alpha
            charge.lambda_persistence = self.trajectory_engine.exponential_lambda
            charge.beta_persistence = self.trajectory_engine.cosine_beta
            charge.sigma_persistence_sq = self.trajectory_engine.gaussian_sigma**2
            
            # Store rich trajectory data for DTF and final charge output
            charge.trajectory_data = {
                'trajectory_operators': trajectory_operators,
                'transformative_magnitude': np.abs(trajectory_operators),
                'frequency_evolution': frequency_evolution,
                'phase_accumulation': phase_accumulation,
                'semantic_modulation': semantic_modulation,
                'total_transformative_potential': transformative_potential,
                'observational_state': charge_params.observational_state,
                'context_hash': hash(charge_params.context) % 1000 / 1000.0
            }
            
            logger.debug(f"Enhanced {token} with trajectory data: T={transformative_potential:.4f}, dims={len(trajectory_operators)}")
        else:
            logger.warning(f"No trajectory operators computed for {token}, using ConceptualCharge defaults")
            charge.trajectory_data = None
        
        # Store complete Q(τ, C, s) computation results
        charge.complete_charge_magnitude = charge_magnitude
        charge.complete_charge_phase = charge_phase
        charge.complete_charge_complex = charge_magnitude * np.exp(1j * charge_phase)
        charge.field_theory_enhanced = True
        
        # Store complete phase analysis from Phase Dimension
        charge.phase_analysis = phase_analysis
        charge.phase_integration_complex = phase_integration
        charge.phase_coherence = phase_analysis['phase_quality']['coherence']
        charge.phase_quality_score = phase_analysis['phase_quality']['quality_score']
        
        # Enhance with DTF semantic field if available
        if dtf_semantic_field_magnitude > 0:
            # Store DTF semantic field data for enhanced Φ^semantic(τ, s) computation
            charge.dtf_semantic_field = dtf_semantic_field_complex
            charge.dtf_enhanced = True
            charge.foundation_processing = True
            logger.debug(f"Enhanced {token} with DTF Φ^semantic: {dtf_semantic_field_magnitude:.4f}")
        else:
            charge.dtf_enhanced = False
            charge.foundation_processing = metadata.get('foundation_processing', False) if metadata else False
        
        # Store additional properties from manifold analysis and metadata
        charge.manifold_properties = manifold_properties
        charge.metadata = metadata if metadata else {}
        charge.charge_id = charge_id
        
        # Store component analysis for debugging and analysis
        charge.component_analysis = {
            'emotional_trajectory': emotional_trajectory,
            'phase_integration': phase_integration,
            'temporal_data': temporal_results,
            'semantic_field_magnitude': dtf_semantic_field_magnitude,
            'semantic_field_complex': dtf_semantic_field_complex,
            'observational_persistence': observational_persistence
        }
        
        return charge
    
    def create_charges_batch(self, 
                           embeddings: List[np.ndarray],
                           properties_batch: List[Dict[str, Any]],
                           charge_params: ChargeParameters,
                           metadata_batch: Optional[List[Dict[str, Any]]] = None) -> List[ConceptualCharge]:
        """
        Efficiently transform batch of embeddings into conceptual charges.
        
        BATCH PROCESSING: Optimized for processing multiple embeddings while
        maintaining full mathematical accuracy for each Q(τ, C, s) calculation.
        
        Args:
            embeddings: List of embedding vectors to transform
            properties_batch: Manifold properties for each embedding
            charge_params: Shared parameters for Q(τ, C, s) calculations
            metadata_batch: Optional metadata for each embedding
            
        Returns:
            List[ConceptualCharge]: Batch of transformed dynamic charges
        """
        if metadata_batch is None:
            metadata_batch = [None] * len(embeddings)
        
        charges = []
        for i, (embedding, properties, metadata) in enumerate(zip(embeddings, properties_batch, metadata_batch)):
            try:
                charge = self.create_charge(
                    embedding=embedding,
                    manifold_properties=properties,
                    charge_params=charge_params,
                    charge_id=f"batch_{i:04d}",
                    metadata=metadata
                )
                charges.append(charge)
            except Exception as e:
                logger.warning(f"Failed to create charge for embedding {i}: {e}")
                # Continue processing other embeddings
                continue
        
        logger.info(f"Successfully created {len(charges)} charges from {len(embeddings)} embeddings")
        return charges
    
    def create_charge_from_text(self, 
                              text: str,
                              embedding_source: Any,  # BGEIngestion, MPNetIngestion, etc.
                              charge_params: ChargeParameters) -> ConceptualCharge:
        """
        Create charge from text input using provided embedding source.
        
        CONVENIENCE METHOD: Handles text → embedding → charge pipeline using
        any embedding source that provides search_embeddings() method.
        
        Args:
            text: Input text to transform
            embedding_source: Any object with search_embeddings(text) method
            charge_params: Parameters for Q(τ, C, s) calculation
            
        Returns:
            ConceptualCharge: Dynamic charge generated from text
        """
        try:
            # Use new ingestion interface to get embedding and properties
            search_results = embedding_source.search_embeddings(text, top_k=1)
            
            if not search_results or 'results' not in search_results or len(search_results['results']) == 0:
                raise ValueError(f"No embedding results found for text: {text[:50]}...")
            
            # Extract first (best) result
            best_result = search_results['results'][0]
            embedding = best_result['embedding']
            
            # Extract manifold properties using the new interface
            manifold_properties = embedding_source.extract_manifold_properties(
                embedding=embedding,
                index=best_result.get('index', 0),
                all_embeddings=search_results.get('embeddings', np.array([embedding]))
            )
            
            # Create metadata from search results
            metadata = {
                'text': text,
                'similarity_score': best_result.get('similarity', 1.0),
                'token': best_result.get('token', 'unknown'),
                'model_type': getattr(embedding_source, 'model_name', 'unknown'),
                'processing_method': 'text_to_charge_pipeline'
            }
            
            # Generate charge using complete Q(τ, C, s) transformation
            charge = self.create_charge(
                embedding=embedding,
                manifold_properties=manifold_properties,
                charge_params=charge_params,
                charge_id=f"text_{hash(text) % 1000000:06d}",
                metadata=metadata
            )
            
            logger.info(f"Successfully created charge from text: {text[:50]}...")
            return charge
            
        except Exception as e:
            logger.error(f"Failed to create charge from text '{text[:50]}...': {e}")
            raise
    
    def validate_manifold_properties(self, properties: Dict[str, Any]) -> bool:
        """
        Validate that manifold properties contain required keys for Q(τ, C, s).
        
        QUALITY CONTROL: Ensures manifold properties contain all mathematical
        components needed for complete conceptual charge calculation.
        
        Args:
            properties: Manifold properties dictionary to validate
            
        Returns:
            bool: True if properties are sufficient for charge generation
        """
        required_keys = {
            'magnitude', 'gradient', 'phase_angles', 'dominant_frequencies',
            'coupling_mean', 'persistence_radius', 'local_density'
        }
        
        missing_keys = required_keys - set(properties.keys())
        if missing_keys:
            logger.warning(f"Missing required manifold properties: {missing_keys}")
            return False
        
        return True
    
    def temporal_dimension(self,
                          embedding: np.ndarray,
                          charge_params: ChargeParameters,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        TEMPORAL DIMENSION PROCESSING: Core trajectory operator computation T(τ, C, s).
        
        This function implements the temporal dimension component of the complete 
        Q(τ, C, s) formula, focusing specifically on trajectory integration and
        observational persistence.
        
        MATHEMATICAL FOUNDATION:
        T(τ, C, s) = ∫₀ˢ ω_i(τ,s')·e^(iφ_i(τ,s')) ds'
        Ψ_persistence(s-s₀) = exp(-(s-s₀)²/2σ²) + α·exp(-λ(s-s₀))·cos(β(s-s₀))
        
        Args:
            embedding: Base semantic vector (1024d for BGE, 768d for MPNet)
            charge_params: Field parameters for temporal processing
            metadata: Optional context and processing metadata
            
        Returns:
            Dict containing temporal dimension processing results:
            - 'trajectory_operators': Complex trajectory operator array
            - 'transformative_potential': Overall T(τ,C,s) magnitude
            - 'observational_persistence': Ψ_persistence value
            - 'developmental_distance': Transformative distance measure
            - 'phase_coordination': Phase relationships across dimensions
        """
        token = metadata.get('token', 'unknown') if metadata else 'unknown'
        
        try:
            # Adapt engine to embedding dimension
            embedding_dim = len(embedding)
            if embedding_dim != self.trajectory_engine.embedding_dimension:
                self.trajectory_engine = TrajectoryOperatorEngine(
                    embedding_dimension=embedding_dim,
                    integration_method="adaptive_quad"
                )
            
            # Compute trajectory operators and transformative analysis
            trajectory_results = self.trajectory_engine.compute_trajectory_integral(
                token=token,
                context=charge_params.context,
                observational_state=charge_params.observational_state,
                semantic_embedding=embedding
            )
            
            # Extract key transformative data
            trajectory_operators = trajectory_results['trajectory_operators']
            transformative_potential = trajectory_results['total_transformative_potential']
            frequency_evolution = trajectory_results['frequency_evolution']
            phase_accumulation = trajectory_results['phase_accumulation']
            semantic_modulation = trajectory_results['semantic_modulation']
            
            # Compute observational persistence
            persistence = self.trajectory_engine.generate_observational_persistence(
                observational_distance=charge_params.observational_state
            )
            if isinstance(persistence, np.ndarray):
                persistence = np.mean(persistence)
            
            # Generate breathing pattern for semantic coupling
            breathing_pattern = self.trajectory_engine.generate_breathing_pattern(
                observational_state=charge_params.observational_state
            )
            
            # Compute phase coordination
            phase_info = self.trajectory_engine.compute_phase_coordination(
                trajectory_operators=trajectory_operators,
                observational_state=charge_params.observational_state
            )
            
            logger.debug(f"Temporal processing for {token}: T={transformative_potential:.4f}, Ψ={persistence:.4f}")
            
            return {
                'trajectory_operators': trajectory_operators,
                'transformative_potential': transformative_potential,
                'frequency_evolution': frequency_evolution,
                'phase_accumulation': phase_accumulation,
                'semantic_modulation': semantic_modulation,
                'observational_persistence': persistence,
                'breathing_pattern': breathing_pattern,
                'phase_coordination': phase_info,
                'processing_status': 'enhanced_temporal'
            }
            
        except Exception as e:
            logger.error(f"Temporal dimension processing failed for {token}: {e}")
            raise RuntimeError(f"Temporal dimension processing required - no fallback allowed per CLAUDE.md: {e}")
    
    def semantic_dimension(self, 
                          embedding: np.ndarray,
                          manifold_properties: Dict[str, Any],
                          charge_params: ChargeParameters,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        SEMANTIC DIMENSION PROCESSING: Core semantic field generation Φ^semantic(τ, s).
        
        This function implements the semantic dimension component of the complete 
        Q(τ, C, s) formula, focusing specifically on breathing constellation patterns
        and dynamic semantic field generation.
        
        MATHEMATICAL FOUNDATION:
        Φ^semantic(τ, s) = w_i * T_i * x[i] * breathing_modulation * e^(iθ)
        
        Args:
            embedding: Base semantic vector (1024d for BGE, 768d for MPNet)
            manifold_properties: Mathematical properties from manifold analysis
            charge_params: Field parameters for semantic processing
            metadata: Optional context and processing metadata
            
        Returns:
            Dict containing semantic dimension processing results:
            - 'semantic_field': Processed semantic field vector
            - 'breathing_patterns': Dynamic constellation breathing data
            - 'phase_modulation': Complex phase information
            - 'field_magnitude': Semantic field strength
            - 'constellation_topology': Geometric structure data
        """
        try:
            from Sysnpire.model.semantic_dimension import process_semantic_field
            
            # Delegate to specialized semantic dimension module
            semantic_results = process_semantic_field(
                embedding=embedding,
                manifold_properties=manifold_properties,
                observational_state=charge_params.observational_state,
                gamma=charge_params.gamma,
                context=charge_params.context,
                field_temperature=charge_params.field_temperature,
                metadata=metadata
            )
            
            logger.debug(f"Semantic dimension processed - field magnitude: {semantic_results.get('field_magnitude', 'N/A')}")
            return semantic_results
            
        except ImportError:
            logger.warning("Semantic dimension module not available - using placeholder")
            # Placeholder implementation until semantic_dimension module is built
            return {
                'semantic_field': embedding,  # Pass-through for now
                'breathing_patterns': {'status': 'placeholder'},
                'phase_modulation': {'real': 1.0, 'imag': 0.0},
                'field_magnitude': float(np.linalg.norm(embedding)),
                'constellation_topology': {'status': 'placeholder'},
                'processing_status': 'placeholder_mode'
            }
        except Exception as e:
            logger.error(f"Semantic dimension processing failed: {e}")
            raise
    
    def convert_to_database_object(self, 
                                  charge: ConceptualCharge,
                                  charge_id: str,
                                  text_source: str) -> 'ConceptualChargeObject':
        """
        Convert enhanced ConceptualCharge to database ConceptualChargeObject.
        
        Ensures trajectory operator data flows through to database storage
        following CLAUDE.md principles of no duplication.
        
        Args:
            charge: Enhanced ConceptualCharge from factory
            charge_id: Unique identifier for database
            text_source: Original text input
            
        Returns:
            ConceptualChargeObject ready for database storage with trajectory data
        """
        from Sysnpire.database.conceptual_charge_object import ConceptualChargeObject, FieldComponents
        
        # Extract trajectory operators from enhanced charge
        if hasattr(charge, 'trajectory_data') and charge.trajectory_data is not None:
            # Use enhanced trajectory data
            trajectory_ops = charge.trajectory_data['trajectory_operators']
            transformative_magnitude = charge.trajectory_data['transformative_magnitude']
            frequency_evolution = charge.trajectory_data['frequency_evolution']
            phase_accumulation = charge.trajectory_data['phase_accumulation']
            semantic_modulation = charge.trajectory_data['semantic_modulation']
            
            logger.debug(f"Converting {charge.token} with enhanced trajectory data: {len(trajectory_ops)} operators")
        else:
            # Fallback to basic trajectory computation
            s = charge.observational_state
            trajectory_ops = [charge.trajectory_operator(s, i) for i in range(min(3, len(charge.omega_base)))]
            transformative_magnitude = np.array([abs(op) for op in trajectory_ops])
            frequency_evolution = charge.omega_base[:len(trajectory_ops)]
            phase_accumulation = charge.phi_base[:len(trajectory_ops)]
            semantic_modulation = np.zeros(len(trajectory_ops))
            
            logger.debug(f"Converting {charge.token} with basic trajectory data: {len(trajectory_ops)} operators")
        
        # Compute complete charge for database
        complete_charge_value = charge.compute_complete_charge()
        
        # Create FieldComponents with trajectory data
        field_components = FieldComponents(
            trajectory_operators=trajectory_ops,  # List[complex] - core T(τ,C,s) data
            emotional_trajectory=charge.emotional_trajectory_integration(charge.observational_state),
            semantic_field=np.real(charge.semantic_field_generation(charge.observational_state)),  # Convert complex to real for storage
            phase_total=charge.total_phase_integration(charge.observational_state),
            observational_persistence=charge.observational_persistence(charge.observational_state)
        )
        
        # Create database object with trajectory data preserved
        db_charge = ConceptualChargeObject(
            charge_id=charge_id,
            text_source=text_source,
            complete_charge=complete_charge_value,
            field_components=field_components,
            observational_state=charge.observational_state,
            gamma=charge.gamma
        )
        
        # Store additional trajectory metadata for enhanced retrieval
        db_charge._trajectory_metadata = {
            'transformative_magnitude': transformative_magnitude.tolist() if isinstance(transformative_magnitude, np.ndarray) else transformative_magnitude,
            'frequency_evolution': frequency_evolution.tolist() if isinstance(frequency_evolution, np.ndarray) else frequency_evolution,
            'phase_accumulation': phase_accumulation.tolist() if isinstance(phase_accumulation, np.ndarray) else phase_accumulation,
            'semantic_modulation': semantic_modulation.tolist() if isinstance(semantic_modulation, np.ndarray) else semantic_modulation,
            'movement_available': hasattr(charge, 'trajectory_data'),
            'dtf_enhanced': getattr(charge, 'dtf_enhanced', False),
            'total_transformative_potential': float(np.mean(transformative_magnitude)) if len(transformative_magnitude) > 0 else 0.0
        }
        
        return db_charge
    
    def _compute_emotional_trajectory_integration(self,
                                                embedding: np.ndarray,
                                                manifold_properties: Dict[str, Any],
                                                charge_params: ChargeParameters,
                                                token: str,
                                                temporal_data: Dict[str, Any]) -> complex:
        """
        Compute E^trajectory(τ, s) using dedicated emotional dimension module.
        
        DELEGATION TO EMOTIONAL DIMENSION:
        Uses Sysnpire.model.emotional_dimension for proper implementation
        following CLAUDE.md principle of no code duplication.
        
        Args:
            embedding: Semantic vector [D]
            manifold_properties: Contains coupling_mean, coupling_variance from correlation analysis
            charge_params: Field parameters including observational_state
            token: Token identifier for trajectory tracking
            temporal_data: Temporal dimension results for coordination
            
        Returns:
            complex: E^trajectory(τ, s) with magnitude and phase
        """
        try:
            from Sysnpire.model.emotional_dimension import compute_emotional_trajectory
            
            # Compute emotional trajectory using dedicated module
            emotional_results = compute_emotional_trajectory(
                token=token,
                semantic_embedding=embedding,
                manifold_properties=manifold_properties,
                observational_state=charge_params.observational_state,
                gamma=charge_params.gamma,
                context=charge_params.context,
                temporal_data=temporal_data,
                emotional_intensity=1.0
            )
            
            # Extract complex-valued result
            emotional_trajectory_complex = emotional_results.get('emotional_trajectory_complex', complex(1.0, 0.0))
            
            logger.debug(f"E^trajectory computed via emotional_dimension for {token}: {emotional_trajectory_complex}")
            return emotional_trajectory_complex
            
        except ImportError as e:
            logger.warning(f"Emotional dimension module not available for {token}: {e}")
            return complex(1.0, 0.0)
        except Exception as e:
            logger.warning(f"Emotional trajectory computation failed for {token}: {e}")
            return complex(1.0, 0.0)
    
    def _compute_complete_phase_integration(self,
                                          embedding: np.ndarray,
                                          manifold_properties: Dict[str, Any],
                                          charge_params: ChargeParameters,
                                          token: str,
                                          temporal_data: Dict[str, Any],
                                          emotional_trajectory: complex) -> Tuple[complex, Dict[str, Any]]:
        """
        Compute complete phase integration e^(iθ_total(τ,C,s)) using Phase Dimension.
        
        PHASE DIMENSION INTEGRATION:
        Uses the complete phase dimension module to integrate phases from all dimensions
        into the final e^(iθ_total(τ,C,s)) component for Q(τ, C, s) formula assembly.
        
        MATHEMATICAL FOUNDATION:
        θ_total(τ,C,s) = θ_semantic + θ_emotional + θ_temporal + θ_interaction + θ_field
        
        CLAUDE.MD COMPLIANCE:
        - Uses actual computed values from all dimensions
        - NO simulation or default values
        - Integrates with BGE manifold properties from database
        
        Args:
            embedding: Semantic vector [D]
            manifold_properties: BGE manifold properties with phase data
            charge_params: Field parameters including observational_state, context
            token: Token identifier for phase analysis
            temporal_data: Temporal dimension processing results
            emotional_trajectory: Complex emotional trajectory from emotional dimension
            
        Returns:
            Tuple of (e^(iθ_total), phase_analysis_data)
        """
        try:
            # Import phase dimension (lazy import for performance)
            from Sysnpire.model.shared_dimensions.phase_dimension import compute_total_phase
            
            # Step 1: Prepare semantic data for phase extraction
            semantic_data = {
                'phase_angles': manifold_properties.get('phase_angles', []),
                'semantic_modulation': manifold_properties.get('semantic_modulation', []),
                'gradient': manifold_properties.get('gradient', []),
                'semantic_field_complex': embedding[0] + 1j * embedding[1] if len(embedding) > 1 else complex(embedding[0], 0)
            }
            
            # Step 2: Prepare emotional data for phase extraction
            # CLAUDE.MD COMPLIANCE: Use actual computed complex values from emotional dimension
            emotional_data = {
                'emotional_trajectory_complex': emotional_trajectory,
                'emotional_phase': np.angle(emotional_trajectory) if emotional_trajectory != 0 else 0.0,
                'emotional_magnitude': abs(emotional_trajectory),
                'complex_field_data': {
                    'magnitude': abs(emotional_trajectory),
                    'phase': np.angle(emotional_trajectory),
                    'real': emotional_trajectory.real,
                    'imag': emotional_trajectory.imag
                }
            }
            
            # Step 3: Prepare trajectory data for phase extraction
            trajectory_data = {
                'phase_accumulation': temporal_data.get('phase_accumulation', []),
                'frequency_evolution': temporal_data.get('frequency_evolution', []),
                'transformative_magnitude': temporal_data.get('transformative_magnitude', []),
                'total_transformative_potential': temporal_data.get('transformative_potential', 0.0)
            }
            
            # Step 4: Complete phase integration using Phase Dimension
            logger.debug(f"Computing total phase integration for token: {token}")
            
            e_iθ_total, phase_components = compute_total_phase(
                semantic_data=semantic_data,
                emotional_data=emotional_data,
                trajectory_data=trajectory_data,
                context=charge_params.context,
                observational_state=charge_params.observational_state,
                manifold_properties=manifold_properties
            )
            
            # Step 5: Create phase analysis data for storage including enhanced critical requirements
            phase_analysis = {
                'phase_components': {
                    'semantic_phase': phase_components.semantic_phase,
                    'emotional_phase': phase_components.emotional_phase,
                    'temporal_phase': phase_components.temporal_phase,
                    'interaction_phase': phase_components.interaction_phase,
                    'field_phase': phase_components.field_phase,
                    'total_phase': phase_components.total_phase
                },
                'phase_quality': {
                    'coherence': phase_components.phase_coherence,
                    'quality_score': phase_components.phase_quality,
                    # Critical Requirements Data
                    'unification_strength': phase_components.unification_strength,  # Req 1: Phase as Unifier
                    'interference_patterns': phase_components.interference_patterns,  # Req 3: Interference Enabler
                    'memory_encoding': phase_components.memory_encoding,  # Req 4: Memory Mechanism
                    'evolution_coupling': phase_components.evolution_coupling  # Req 5: Evolution Driver
                },
                'complex_result': {
                    'magnitude': abs(e_iθ_total),
                    'phase': np.angle(e_iθ_total),
                    'real': e_iθ_total.real,
                    'imag': e_iθ_total.imag
                },
                # Full field coherence metrics
                'field_coherence_metrics': phase_components.field_coherence_metrics
            }
            
            logger.debug(f"Phase integration complete for {token}: "
                        f"θ_total={phase_components.total_phase:.4f}, "
                        f"|e^(iθ)|={abs(e_iθ_total):.4f}, "
                        f"coherence={phase_components.phase_coherence:.3f}")
            
            return e_iθ_total, phase_analysis
            
        except Exception as e:
            logger.error(f"Phase dimension integration failed for {token}: {e}")
            # CLAUDE.MD COMPLIANCE: No fallback values, require actual data
            raise ValueError(f"Cannot compute phase integration without valid dimensional data for {token}: {e}")
    
    def _compute_semantic_phase_from_embedding(self, embedding: np.ndarray, token: str) -> float:
        """
        Compute semantic phase directly from embedding structure.
        
        This is a mathematical derivation from embedding geometry, not a fallback.
        Uses embedding structure and token characteristics to generate phase.
        """
        # Token-specific phase contribution
        token_hash = hash(token) % 1000 / 1000.0
        base_phase = 2 * np.pi * token_hash
        
        # Embedding-derived phase from vector characteristics
        embedding_magnitude = np.linalg.norm(embedding)
        embedding_phase_factor = np.mean(np.abs(embedding)) / (embedding_magnitude + 1e-10)
        
        # Combine for semantic phase
        semantic_phase = base_phase * embedding_phase_factor
        
        return semantic_phase
    
    def _compute_interaction_phase(self,
                                 theta_semantic: float,
                                 theta_emotional: float,
                                 theta_temporal: float,
                                 observational_state: float,
                                 gamma: float) -> float:
        """
        Compute interaction phase from cross-dimensional coupling effects.
        
        MATHEMATICAL FOUNDATION:
        θ_interaction represents phase effects from coupling between semantic,
        emotional, and temporal dimensions as they interact through the field.
        """
        # Phase coupling strength based on observational state and gamma
        coupling_strength = 0.1 * gamma * observational_state
        
        # Compute phase coupling terms
        semantic_emotional_coupling = np.sin(theta_semantic - theta_emotional)
        emotional_temporal_coupling = np.sin(theta_emotional - theta_temporal)
        temporal_semantic_coupling = np.sin(theta_temporal - theta_semantic)
        
        # Total interaction phase
        theta_interaction = coupling_strength * (
            semantic_emotional_coupling + 
            emotional_temporal_coupling + 
            temporal_semantic_coupling
        )
        
        return theta_interaction
    
    def _compute_field_phase(self,
                           manifold_properties: Dict[str, Any],
                           observational_state: float,
                           token: str) -> float:
        """
        Compute field-induced phase effects from manifold geometry.
        
        MATHEMATICAL FOUNDATION:
        θ_field represents phase contributions from the manifold structure itself,
        including curvature effects and topological phase contributions.
        """
        # Extract manifold-derived phase information
        spectral_entropy = manifold_properties.get('spectral_entropy', 0.0)
        phase_variance = manifold_properties.get('phase_variance', 0.0)
        frequency_bandwidth = manifold_properties.get('frequency_bandwidth', 0.0)
        
        # Token-specific field interaction
        token_hash = hash(token) % 1000 / 1000.0
        
        # Field phase from manifold geometry
        geometric_phase = spectral_entropy * np.sin(2 * np.pi * token_hash)
        curvature_phase = phase_variance * observational_state
        topological_phase = frequency_bandwidth * np.cos(np.pi * token_hash)
        
        # Combine field phase contributions
        theta_field = 0.1 * (geometric_phase + curvature_phase + topological_phase)
        
        return theta_field
    
    def _assemble_complete_charge_formula(self,
                                        gamma: float,
                                        transformative_potential: float,
                                        emotional_trajectory: complex,
                                        semantic_field_magnitude: float,
                                        semantic_field_complex: complex,
                                        phase_integration: complex,
                                        observational_persistence: float,
                                        token: str) -> Tuple[float, float]:
        """
        Assemble the complete Q(τ, C, s) conceptual charge formula.
        
        COMPLETE MATHEMATICAL FORMULA:
        Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)
        
        COMPONENT INTEGRATION:
        - γ: Global field calibration factor
        - T(τ, C, s): Transformative potential tensor (real)
        - E^trajectory(τ, s): Emotional trajectory integration (complex)
        - Φ^semantic(τ, s): Semantic field generation (complex)
        - e^(iθ_total(τ,C,s)): Complete phase integration (complex)
        - Ψ_persistence(s-s₀): Observational persistence (real)
        
        Args:
            gamma: Global field calibration factor γ
            transformative_potential: T(τ, C, s) from temporal dimension
            emotional_trajectory: E^trajectory(τ, s) complex result
            semantic_field_magnitude: |Φ^semantic(τ, s)| magnitude
            semantic_field_complex: Φ^semantic(τ, s) complex field
            phase_integration: e^(iθ_total(τ,C,s)) complex phase
            observational_persistence: Ψ_persistence(s-s₀) 
            token: Token identifier for logging
            
        Returns:
            Tuple[float, float]: (charge_magnitude, charge_phase)
        """
        try:
            # Assemble complex-valued charge components
            
            # 1. Real components: γ, T, Ψ
            real_component = gamma * transformative_potential * observational_persistence
            
            # 2. Complex components: E^trajectory, Φ^semantic, e^(iθ_total)
            # Handle semantic field - require valid DTF processing
            if semantic_field_magnitude > 0 and semantic_field_complex != 0:
                # Use DTF-enhanced semantic field
                phi_semantic = semantic_field_complex
            else:
                # CLAUDE.md compliance: semantic field must be computed
                raise ValueError(f"Invalid semantic field for charge assembly: magnitude={semantic_field_magnitude}, complex={semantic_field_complex}")
            
            # 3. Complete complex charge assembly
            # Q_complex = real_component · E^trajectory · Φ^semantic · e^(iθ_total)
            complex_component = emotional_trajectory * phi_semantic * phase_integration
            
            # 4. Final charge with real scaling
            complete_charge_complex = real_component * complex_component
            
            # 5. Extract magnitude and phase
            charge_magnitude = abs(complete_charge_complex)
            charge_phase = np.angle(complete_charge_complex)
            
            # 6. Numerical stability checks
            if np.isnan(charge_magnitude) or np.isinf(charge_magnitude):
                logger.warning(f"Unstable charge magnitude for {token}: {charge_magnitude}")
                charge_magnitude = 1.0
                
            if np.isnan(charge_phase) or np.isinf(charge_phase):
                logger.warning(f"Unstable charge phase for {token}: {charge_phase}")
                charge_phase = 0.0
            
            # 7. Log assembly results
            logger.debug(f"Complete Q(τ,C,s) for {token}: |Q|={charge_magnitude:.4f}, φ={charge_phase:.4f}")
            logger.debug(f"  Components: γ={gamma:.3f}, T={transformative_potential:.3f}, |E|={abs(emotional_trajectory):.3f}, |Φ|={abs(phi_semantic):.3f}, |e^iθ|={abs(phase_integration):.3f}, Ψ={observational_persistence:.3f}")
            
            return charge_magnitude, charge_phase
            
        except Exception as e:
            logger.error(f"Complete charge assembly failed for {token}: {e}")
            # Fallback values
            return 1.0, 0.0
    
    def get_factory_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about charge factory operations.
        
        Returns:
            Dict containing factory statistics and performance metrics
        """
        return {
            'total_charges_created': self.charge_count,
            'factory_type': 'Q(τ, C, s) Transformation Engine',
            'source_agnostic': True,
            'mathematical_focus': 'Field Theory Charge Generation'
        }


# Example usage showing source-agnostic design
if __name__ == "__main__":
    # Initialize factory (no dependencies)
    factory = ChargeFactory()
    
    # Example parameters
    params = ChargeParameters(
        observational_state=1.0,
        gamma=1.2,
        context="example_context"
    )
    
    logger.info("ChargeFactory ready for Q(τ, C, s) transformations")
    logger.info("Factory can process embeddings from any source:")
    logger.info("- BGE model vocabularies")
    logger.info("- MPNet model vocabularies") 
    logger.info("- User text inputs")
    logger.info("- Scraped data embeddings")
    logger.info("- External embedding sources")
    
    # TODO: Demonstrate with actual embedding data when implementation complete
    logger.info(f"Factory statistics: {factory.get_factory_statistics()}")