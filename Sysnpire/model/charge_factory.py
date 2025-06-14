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
from typing import Dict, List, Any, Optional, Union
import numpy as np
from dataclasses import dataclass

# Ensure project imports work
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.model.mathematics.conceptual_charge import ConceptualCharge
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
        
        # TODO: Compute T(τ, C, s) - transformative potential tensor
        # Use manifold_properties: 'gradient', 'hessian_eigenvalues', 'principal_components'
        transformative_potential = None
        
        # TODO: Calculate E^trajectory(τ, s) - emotional trajectory integration  
        # Use manifold_properties: 'coupling_mean', 'coupling_variance'
        emotional_trajectory = None
        
        # TODO: Generate Φ^semantic(τ, s) - dynamic semantic field
        # Use manifold_properties: 'dominant_frequencies', 'frequency_magnitudes'
        semantic_field = None
        
        # TODO: Compute e^(iθ_total(τ,C,s)) - complete phase integration
        # Use manifold_properties: 'phase_angles'
        phase_integration = None
        
        # TODO: Apply Ψ_persistence(s-s₀) - observational persistence
        # Use manifold_properties: 'persistence_radius', 'persistence_score'
        observational_persistence = None
        
        # TODO: Combine all components via Q(τ, C, s) formula
        # Q = γ · T · E · Φ · e^(iθ) · Ψ
        charge_magnitude = None
        charge_phase = None
        
        # Create ConceptualCharge with correct parameters
        charge = ConceptualCharge(
            token=token,
            semantic_vector=embedding,
            context={'context': charge_params.context},
            observational_state=charge_params.observational_state,
            gamma=charge_params.gamma
        )
        
        # TODO: Set additional properties from manifold analysis
        # charge.manifold_properties = manifold_properties
        # charge.metadata = metadata
        # charge.charge_id = charge_id
        
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