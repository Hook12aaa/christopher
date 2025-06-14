"""
Foundation Manifold Builder - Platform Seeding and Core Universe Provisioning

PLATFORM FOUNDATION CREATION: This service implements the systematic foundation
building process that transforms pre-trained model vocabularies into the core
product manifold using field theory mathematics.

FOUNDATION BUILDING PROCESS:
1. Extract complete token embedding manifolds from BGE/MPNet models (30K+ tokens)
2. Compute comprehensive manifold properties for every embedding vector
3. Apply Q(τ, C, s) transformations to create dynamic conceptual charges
4. Provision baseline social universe ready for field theory operations

SCOPE: This is the PLATFORM FOUNDATION service. Once the core manifold is established,
additional charges can be integrated via the ChargeFactory from user inputs, scraped data,
or other sources without re-running this foundation building process.

FOUNDATION SCALE:
- BGE: ~30,522 tokens × 1024 dimensions → ~30K conceptual charges
- MPNet: ~30,522 tokens × 768 dimensions → ~30K conceptual charges
- Total foundation manifold: ~60K+ dynamic field theory charges

This provisions the foundational product manifold for all subsequent
social construct analysis and field theory platform operations.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
import numpy as np
from dataclasses import dataclass
import time
import json

# Ensure project imports work
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.model.intial.bge_ingestion import BGEIngestion
from Sysnpire.model.intial.mpnet_ingestion import MPNetIngestion
from Sysnpire.model.charge_factory import ChargeFactory, ChargeParameters
from Sysnpire.database.field_universe import FieldUniverse
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FoundationConfig:
    """Configuration for platform foundation and manifold building operations."""
    models_to_process: List[str] = None  # ["BGE", "MPNet"] or subset
    charge_params: ChargeParameters = None
    batch_size: int = 100
    save_checkpoints: bool = True
    checkpoint_interval: int = 1000
    output_directory: Path = None
    enable_statistics: bool = True
    random_seed: int = 42
    
    def __post_init__(self):
        if self.models_to_process is None:
            self.models_to_process = ["BGE", "MPNet"]
        if self.charge_params is None:
            self.charge_params = ChargeParameters(
                observational_state=1.0,
                gamma=1.2,
                context="foundation_manifold_building"
            )
        if self.output_directory is None:
            self.output_directory = Path("./foundation_manifold")


class FoundationManifoldBuilder:
    """
    Platform Foundation and Core Universe Provisioner
    
    FOUNDATION BUILDING: Transforms complete pre-trained model vocabularies into
    the foundational product manifold using field theory mathematics.
    
    FOUNDATION PROCESS:
    1. Load complete embedding manifolds from selected models
    2. Extract field-theoretic properties for every token embedding
    3. Transform static embeddings into dynamic conceptual charges
    4. Provision core manifold for subsequent platform operations
    
    SCALE: Processes tens of thousands of embeddings to create a comprehensive
    foundation manifold representing the baseline semantic knowledge for the platform.
    """
    
    def __init__(self, config: FoundationConfig):
        """
        Initialize platform foundation and manifold building system.
        
        Args:
            config: Configuration for foundation manifold building process
        """
        self.config = config
        self.charge_factory = ChargeFactory()
        self.universe = FieldUniverse()
        self.models = {}
        self.statistics = {
            'total_embeddings_processed': 0,
            'total_charges_created': 0,
            'processing_time': 0,
            'models_processed': [],
            'errors_encountered': []
        }
        
        logger.info("Initializing platform foundation manifold building system")
        self._setup_output_directory()
    
    def _setup_output_directory(self) -> None:
        """Setup output directory for foundation manifold building data."""
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Foundation manifold directory: {self.config.output_directory}")
    
    def initialize_models(self) -> None:
        """
        Initialize and load all selected embedding models for manifold extraction.
        
        FOUNDATION PREPARATION: Loads the complete embedding models that will provide
        the foundational semantic field data for platform manifold building.
        """
        logger.info(f"Initializing models: {self.config.models_to_process}")
        
        for model_type in self.config.models_to_process:
            try:
                if model_type == "BGE":
                    logger.info("Loading BGE-Large-v1.5 model...")
                    self.models["BGE"] = BGEIngestion(
                        model_name="BAAI/bge-large-en-v1.5",
                        random_seed=self.config.random_seed
                    )
                    logger.info("BGE model loaded successfully")
                    
                elif model_type == "MPNet":
                    logger.info("Loading MPNet-base-v2 model...")
                    self.models["MPNet"] = MPNetIngestion(
                        model_name="sentence-transformers/all-mpnet-base-v2", 
                        random_seed=self.config.random_seed
                    )
                    logger.info("MPNet model loaded successfully")
                    
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    
            except Exception as e:
                logger.error(f"Failed to load {model_type} model: {e}")
                self.statistics['errors_encountered'].append(f"Model loading error: {model_type} - {e}")
    
    def extract_complete_manifold(self, model_type: str) -> Dict[str, Any]:
        """
        Extract complete token embedding manifold from specified model.
        
        MANIFOLD HARVESTING: Gets the entire vocabulary embedding matrix that
        will serve as discrete samples of the continuous semantic field.
        
        Args:
            model_type: Type of model to extract manifold from ("BGE" or "MPNet")
            
        Returns:
            Dict containing complete manifold data and properties
        """
        logger.info(f"Extracting complete manifold from {model_type} model...")
        
        model = self.models[model_type]
        
        # TODO: Extract complete embedding manifold
        manifold_data = model.load_total_embeddings()
        
        # TODO: Compute manifold properties for all embeddings
        # This may take significant time for 30K+ embeddings
        # Consider batch processing and progress tracking
        
        logger.info(f"Manifold extraction complete for {model_type}")
        logger.info(f"Embeddings: {manifold_data['vocab_size']} x {manifold_data['embedding_dim']}")
        
        return manifold_data
    
    def create_universe_from_manifold(self, 
                                    manifold_data: Dict[str, Any],
                                    model_type: str) -> Iterator[Any]:
        """
        Transform complete embedding manifold into universe of conceptual charges.
        
        MANIFOLD TRANSFORMATION: Applies Q(τ, C, s) field theory mathematics to
        every embedding in the manifold to create dynamic conceptual charges.
        
        Args:
            manifold_data: Complete embedding manifold with properties
            model_type: Source model type for metadata
            
        Yields:
            ConceptualCharge: Dynamic charges created from embedding manifold
        """
        embeddings = manifold_data['embeddings']
        id_to_token = manifold_data['id_to_token']
        
        logger.info(f"Beginning manifold transformation of {len(embeddings)} embeddings...")
        
        # Process embeddings in batches for efficiency
        batch_size = self.config.batch_size
        total_batches = (len(embeddings) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(embeddings))
            batch_embeddings = embeddings[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} "
                       f"(embeddings {start_idx}-{end_idx})")
            
            # Compute manifold properties for batch using new ingestion interface
            model = self.models[model_type]
            properties_batch = []
            metadata_batch = []
            
            for i, embedding in enumerate(batch_embeddings):
                global_idx = start_idx + i
                token = id_to_token.get(global_idx, f"<UNK_{global_idx}>")
                
                # Extract complete manifold properties using new ingestion interface
                manifold_properties = model.extract_manifold_properties(
                    embedding=embedding,
                    index=global_idx,
                    all_embeddings=embeddings  # Pass full embedding matrix for context
                )
                
                metadata = {
                    'token': token,
                    'source_model': model_type,
                    'batch_id': batch_idx,
                    'global_index': global_idx,
                    'foundation_building': True
                }
                
                properties_batch.append(manifold_properties)
                metadata_batch.append(metadata)
            
            # Transform batch into conceptual charges
            try:
                charges = self.charge_factory.create_charges_batch(
                    embeddings=batch_embeddings,
                    properties_batch=properties_batch,
                    charge_params=self.config.charge_params,
                    metadata_batch=metadata_batch
                )
                
                # Yield charges for processing/storage
                for charge in charges:
                    yield charge
                    
                self.statistics['total_embeddings_processed'] += len(batch_embeddings)
                self.statistics['total_charges_created'] += len(charges)
                
                # Save checkpoint if configured
                if (self.config.save_checkpoints and 
                    (batch_idx + 1) % (self.config.checkpoint_interval // batch_size) == 0):
                    self._save_checkpoint(batch_idx, model_type)
                    
            except Exception as e:
                logger.error(f"Failed to process batch {batch_idx}: {e}")
                self.statistics['errors_encountered'].append(f"Batch processing error: {batch_idx} - {e}")
                continue
    
    def execute_foundation_building(self) -> None:
        """
        Execute the complete platform foundation manifold building process.
        
        FOUNDATION CREATION: The main orchestration method that coordinates the entire
        foundation building process from model loading to manifold provisioning.
        """
        start_time = time.time()
        
        logger.info("🏗️ INITIATING PLATFORM FOUNDATION MANIFOLD BUILDING 🏗️")
        logger.info(f"Models to process: {self.config.models_to_process}")
        logger.info(f"Expected scale: ~30K+ charges per model")
        
        try:
            # Step 1: Initialize models
            self.initialize_models()
            
            # Step 2: Process each model
            for model_type in self.config.models_to_process:
                if model_type not in self.models:
                    logger.warning(f"Skipping {model_type} - model not loaded")
                    continue
                
                logger.info(f"🏗️ Processing {model_type} model for foundation manifold building...")
                
                # Extract manifold
                manifold_data = self.extract_complete_manifold(model_type)
                
                # Transform to charges and store
                charge_count = 0
                for charge in self.create_universe_from_manifold(manifold_data, model_type):
                    # TODO: Store charge in universe database
                    # self.universe.store_charge(charge)
                    charge_count += 1
                    
                    if charge_count % 1000 == 0:
                        logger.info(f"Provisioned {charge_count} charges from {model_type}")
                
                logger.info(f"✅ Completed {model_type}: {charge_count} charges provisioned")
                self.statistics['models_processed'].append(model_type)
            
            # Step 3: Finalize foundation manifold
            end_time = time.time()
            self.statistics['processing_time'] = end_time - start_time
            
            self._save_final_statistics()
            logger.info("🏗️ PLATFORM FOUNDATION MANIFOLD BUILDING COMPLETE 🏗️")
            self._print_foundation_summary()
            
        except Exception as e:
            logger.error(f"Foundation manifold building failed: {e}")
            raise
    
    def _save_checkpoint(self, batch_idx: int, model_type: str) -> None:
        """Save processing checkpoint for resumable foundation building."""
        checkpoint_data = {
            'batch_idx': batch_idx,
            'model_type': model_type,
            'statistics': self.statistics,
            'timestamp': time.time()
        }
        
        checkpoint_path = self.config.output_directory / f"checkpoint_{model_type}_{batch_idx}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_statistics(self) -> None:
        """Save final foundation manifold building statistics."""
        stats_path = self.config.output_directory / "foundation_building_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        
        logger.info(f"Final statistics saved: {stats_path}")
    
    def _print_foundation_summary(self) -> None:
        """Print comprehensive foundation manifold building summary."""
        logger.info("=" * 60)
        logger.info("FOUNDATION MANIFOLD BUILDING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total embeddings processed: {self.statistics['total_embeddings_processed']:,}")
        logger.info(f"Total charges provisioned: {self.statistics['total_charges_created']:,}")
        logger.info(f"Models processed: {', '.join(self.statistics['models_processed'])}")
        logger.info(f"Processing time: {self.statistics['processing_time']:.2f} seconds")
        logger.info(f"Average rate: {self.statistics['total_charges_created'] / self.statistics['processing_time']:.1f} charges/second")
        
        if self.statistics['errors_encountered']:
            logger.warning(f"Errors encountered: {len(self.statistics['errors_encountered'])}")
        
        logger.info("🏗️ Platform foundation manifold ready for social construct field theory operations 🏗️")


# Main execution for foundation manifold building
if __name__ == "__main__":
    # Configure foundation manifold building
    config = FoundationConfig(
        models_to_process=["BGE"],  # Start with BGE only for testing
        batch_size=50,
        save_checkpoints=True,
        checkpoint_interval=1000,
        output_directory=Path("./foundation_manifold"),
        random_seed=42
    )
    
    # Initialize and execute foundation building
    builder = FoundationManifoldBuilder(config)
    
    logger.info("Foundation manifold building configured and ready")
    logger.info("Uncomment the following line to execute foundation building:")
    logger.info("# builder.execute_foundation_building()")
    
    # TODO: Uncomment when ready for full foundation manifold building
    # builder.execute_foundation_building()