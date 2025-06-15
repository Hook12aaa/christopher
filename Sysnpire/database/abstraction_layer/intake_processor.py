"""
Intake Processor - First Stage of Abstraction Layer

Processes incoming ConceptualChargeObjects for optimal storage pipeline.
Validates, normalizes, and prepares charges for tensor storage.

This is the entry point where charges are "pressed" for storage efficiency.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from ..conceptual_charge_object import ConceptualChargeObject

logger = logging.getLogger(__name__)


@dataclass
class IntakeMetrics:
    """Metrics for intake processing performance"""
    charges_processed: int = 0
    total_processing_time: float = 0.0
    validation_failures: int = 0
    normalization_applied: int = 0
    field_extractions: int = 0
    average_processing_time: float = 0.0


class IntakeProcessor:
    """
    Process incoming ConceptualChargeObjects for optimal storage.
    
    The "vinyl pressing" stage that prepares charges for tensor storage.
    Validates data integrity, normalizes field components, and extracts
    tensor representations for Lance storage.
    """
    
    def __init__(self, 
                 validation_strict: bool = True,
                 normalize_field_components: bool = True,
                 extract_tensor_representations: bool = True):
        """
        Initialize intake processor.
        
        Args:
            validation_strict: Whether to apply strict validation
            normalize_field_components: Whether to normalize field components
            extract_tensor_representations: Whether to extract tensor data
        """
        self.validation_strict = validation_strict
        self.normalize_field_components = normalize_field_components
        self.extract_tensor_representations = extract_tensor_representations
        
        # Processing metrics
        self.metrics = IntakeMetrics()
        
        # Validation thresholds
        self.magnitude_range = (1e-10, 1e10)  # Valid magnitude range
        self.phase_range = (-2*np.pi, 2*np.pi)  # Valid phase range
        self.field_component_threshold = 1e-12  # Minimum field component value
        
        logger.info(f"IntakeProcessor initialized - strict validation: {validation_strict}")
    
    def extract_trajectory_features(self, charge: ConceptualChargeObject) -> Dict[str, Any]:
        """
        Extract trajectory operator features for database indexing and querying.
        
        Following CLAUDE.md principles: reuse existing trajectory data rather than recomputing.
        
        Args:
            charge: ConceptualChargeObject with trajectory data
            
        Returns:
            Dictionary with trajectory features for database storage/indexing
        """
        trajectory_features = {}
        
        try:
            # Extract core trajectory operators (T_i(τ,s))
            if hasattr(charge.field_components, 'trajectory_operators'):
                trajectory_ops = charge.field_components.trajectory_operators
                
                # Compute transformative characteristics
                magnitudes = [abs(op) for op in trajectory_ops]
                phases = [np.angle(op) for op in trajectory_ops]
                
                trajectory_features.update({
                    'trajectory_operator_count': len(trajectory_ops),
                    'total_transformative_potential': np.mean(magnitudes) if magnitudes else 0.0,
                    'max_transformative_magnitude': max(magnitudes) if magnitudes else 0.0,
                    'min_transformative_magnitude': min(magnitudes) if magnitudes else 0.0,
                    'transformative_coherence': 1.0 / (1.0 + np.std(magnitudes)) if len(magnitudes) > 1 else 1.0,
                    'phase_distribution_entropy': -np.sum([p * np.log(abs(p) + 1e-10) for p in phases]) if phases else 0.0,
                    'trajectory_complexity': np.std(magnitudes) + np.std(phases) if len(magnitudes) > 1 else 0.0
                })
                
                logger.debug(f"Extracted trajectory features for {charge.charge_id}: T_potential={trajectory_features['total_transformative_potential']:.4f}")
            
            # Extract trajectory metadata if available (from enhanced charges)
            if hasattr(charge, '_trajectory_metadata'):
                metadata = charge._trajectory_metadata
                trajectory_features.update({
                    'movement_available': metadata.get('movement_available', False),
                    'dtf_enhanced': metadata.get('dtf_enhanced', False),
                    'frequency_evolution_available': len(metadata.get('frequency_evolution', [])) > 0,
                    'semantic_modulation_strength': np.mean(metadata.get('semantic_modulation', [0.0])),
                    'enhanced_transformative_potential': metadata.get('total_transformative_potential', 0.0)
                })
                
                logger.debug(f"Enhanced trajectory metadata for {charge.charge_id}: movement={metadata.get('movement_available', False)}")
            
        except Exception as e:
            logger.warning(f"Failed to extract trajectory features for {charge.charge_id}: {e}")
            # Provide minimal fallback features
            trajectory_features = {
                'trajectory_operator_count': 0,
                'total_transformative_potential': 0.0,
                'movement_available': False,
                'trajectory_complexity': 0.0
            }
        
        return trajectory_features
    
    def process_charge(self, charge: ConceptualChargeObject) -> Optional[Dict[str, Any]]:
        """
        Process a single ConceptualChargeObject for storage.
        
        Args:
            charge: ConceptualChargeObject to process
            
        Returns:
            Processed charge data ready for tensor storage, or None if invalid
        """
        start_time = time.time()
        
        try:
            # Step 1: Validate charge
            if not self._validate_charge(charge):
                self.metrics.validation_failures += 1
                logger.warning(f"Charge validation failed: {charge.charge_id}")
                return None
            
            # Step 2: Extract core data
            processed_data = self._extract_core_data(charge)
            
            # Step 3: Normalize field components if enabled
            if self.normalize_field_components:
                processed_data = self._normalize_field_components(processed_data)
                self.metrics.normalization_applied += 1
            
            # Step 4: Extract trajectory features for movement analysis
            trajectory_features = self.extract_trajectory_features(charge)
            processed_data['trajectory_features'] = trajectory_features
            
            # Step 5: Extract tensor representations
            if self.extract_tensor_representations:
                processed_data.update(self._extract_tensor_data(charge))
                self.metrics.field_extractions += 1
            
            # Step 6: Add processing metadata
            processed_data['intake_metadata'] = {
                'processed_timestamp': time.time(),
                'validation_passed': True,
                'normalization_applied': self.normalize_field_components,
                'tensor_extracted': self.extract_tensor_representations,
                'processing_time': time.time() - start_time
            }
            
            # Update metrics
            self.metrics.charges_processed += 1
            self.metrics.total_processing_time += time.time() - start_time
            self.metrics.average_processing_time = (
                self.metrics.total_processing_time / self.metrics.charges_processed
            )
            
            logger.debug(f"Successfully processed charge: {charge.charge_id}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to process charge {charge.charge_id}: {e}")
            return None
    
    def process_batch(self, charges: List[ConceptualChargeObject]) -> List[Dict[str, Any]]:
        """
        Process a batch of charges efficiently.
        
        Args:
            charges: List of ConceptualChargeObjects to process
            
        Returns:
            List of processed charge data ready for tensor storage
        """
        logger.info(f"Processing batch of {len(charges)} charges...")
        
        processed_batch = []
        for charge in charges:
            processed_charge = self.process_charge(charge)
            if processed_charge is not None:
                processed_batch.append(processed_charge)
        
        success_rate = len(processed_batch) / len(charges) if charges else 0
        logger.info(f"Batch processing complete: {len(processed_batch)}/{len(charges)} "
                   f"charges processed ({success_rate:.1%} success rate)")
        
        return processed_batch
    
    def _validate_charge(self, charge: ConceptualChargeObject) -> bool:
        """
        Validate charge data integrity.
        
        Args:
            charge: Charge to validate
            
        Returns:
            True if charge is valid for processing
        """
        try:
            # Basic existence checks
            if not hasattr(charge, 'charge_id') or not charge.charge_id:
                logger.warning("Charge missing charge_id")
                return False
            
            if not hasattr(charge, 'magnitude') or charge.magnitude is None:
                logger.warning(f"Charge {charge.charge_id} missing magnitude")
                return False
            
            if not hasattr(charge, 'phase') or charge.phase is None:
                logger.warning(f"Charge {charge.charge_id} missing phase")
                return False
            
            # Range validation
            if not (self.magnitude_range[0] <= charge.magnitude <= self.magnitude_range[1]):
                if self.validation_strict:
                    logger.warning(f"Charge {charge.charge_id} magnitude out of range: {charge.magnitude}")
                    return False
            
            if not (self.phase_range[0] <= charge.phase <= self.phase_range[1]):
                if self.validation_strict:
                    logger.warning(f"Charge {charge.charge_id} phase out of range: {charge.phase}")
                    return False
            
            # Field components validation
            if hasattr(charge, 'field_components') and charge.field_components:
                if not self._validate_field_components(charge.field_components):
                    if self.validation_strict:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error for charge {getattr(charge, 'charge_id', 'unknown')}: {e}")
            return False
    
    def _validate_field_components(self, field_components) -> bool:
        """Validate field components data structure"""
        try:
            # Check for required field component attributes
            required_attrs = ['trajectory_operators', 'emotional_trajectory', 'semantic_field']
            
            for attr in required_attrs:
                if not hasattr(field_components, attr):
                    logger.warning(f"Field components missing required attribute: {attr}")
                    return not self.validation_strict
            
            # Validate trajectory operators
            if hasattr(field_components, 'trajectory_operators'):
                trajectory_ops = field_components.trajectory_operators
                if isinstance(trajectory_ops, (list, np.ndarray)):
                    if len(trajectory_ops) == 0:
                        logger.warning("Empty trajectory operators")
                        return not self.validation_strict
            
            return True
            
        except Exception as e:
            logger.warning(f"Field components validation error: {e}")
            return not self.validation_strict
    
    def _extract_core_data(self, charge: ConceptualChargeObject) -> Dict[str, Any]:
        """
        Extract core charge data for tensor storage.
        
        Args:
            charge: Charge to extract data from
            
        Returns:
            Dictionary with core charge data
        """
        core_data = {
            'charge_id': charge.charge_id,
            'magnitude': float(charge.magnitude),
            'phase': float(charge.phase),
            'observational_state': getattr(charge, 'observational_state', 1.0),
            'gamma': getattr(charge, 'gamma', 1.0),
            'text_source': getattr(charge, 'text_source', ''),
            'creation_timestamp': getattr(charge, 'creation_timestamp', time.time()),
            'last_updated': getattr(charge, 'last_updated', time.time())
        }
        
        # Extract complete charge if available
        if hasattr(charge, 'complete_charge'):
            if isinstance(charge.complete_charge, complex):
                core_data['complete_charge_real'] = float(charge.complete_charge.real)
                core_data['complete_charge_imag'] = float(charge.complete_charge.imag)
            else:
                core_data['complete_charge_real'] = float(charge.complete_charge)
                core_data['complete_charge_imag'] = 0.0
        
        # Extract field position if available
        if hasattr(charge, 'metadata') and charge.metadata:
            if hasattr(charge.metadata, 'field_position') and charge.metadata.field_position:
                position = charge.metadata.field_position
                if isinstance(position, (list, tuple, np.ndarray)) and len(position) >= 3:
                    core_data['field_position'] = [float(x) for x in position[:3]]
                else:
                    core_data['field_position'] = [0.0, 0.0, 0.0]
            else:
                core_data['field_position'] = [0.0, 0.0, 0.0]
        else:
            core_data['field_position'] = [0.0, 0.0, 0.0]
        
        return core_data
    
    def _normalize_field_components(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize field components for consistent tensor operations.
        
        Args:
            data: Charge data to normalize
            
        Returns:
            Data with normalized field components
        """
        # Normalize magnitude to reasonable range
        if 'magnitude' in data:
            magnitude = data['magnitude']
            if magnitude > 1e6:
                data['magnitude'] = np.log10(magnitude) + 6  # Log compression for large values
            elif magnitude < 1e-6:
                data['magnitude'] = max(magnitude, 1e-10)  # Floor for tiny values
        
        # Normalize phase to [-π, π]
        if 'phase' in data:
            phase = data['phase']
            data['phase'] = np.arctan2(np.sin(phase), np.cos(phase))
        
        # Normalize field position to unit scale
        if 'field_position' in data:
            position = np.array(data['field_position'])
            position_norm = np.linalg.norm(position)
            if position_norm > 1e-10:
                # Scale to reasonable range while preserving direction
                scale_factor = min(1.0, 100.0 / position_norm)
                data['field_position'] = (position * scale_factor).tolist()
        
        return data
    
    def _extract_tensor_data(self, charge: ConceptualChargeObject) -> Dict[str, Any]:
        """
        Extract tensor representations for Lance storage.
        
        Args:
            charge: Charge to extract tensors from
            
        Returns:
            Dictionary with tensor data ready for Arrow/Lance
        """
        tensor_data = {}
        
        # Extract field components as tensors
        if hasattr(charge, 'field_components') and charge.field_components:
            fc = charge.field_components
            
            # Trajectory operators tensor
            if hasattr(fc, 'trajectory_operators') and fc.trajectory_operators is not None:
                traj_ops = fc.trajectory_operators
                if isinstance(traj_ops, (list, np.ndarray)):
                    # Convert complex numbers to real tensor
                    if len(traj_ops) > 0 and isinstance(traj_ops[0], complex):
                        tensor_real = [float(op.real) for op in traj_ops]
                        tensor_imag = [float(op.imag) for op in traj_ops]
                        tensor_data['trajectory_operators_real'] = tensor_real
                        tensor_data['trajectory_operators_imag'] = tensor_imag
                    else:
                        tensor_data['trajectory_operators_real'] = [float(op) for op in traj_ops]
                        tensor_data['trajectory_operators_imag'] = [0.0] * len(traj_ops)
            
            # Emotional trajectory tensor
            if hasattr(fc, 'emotional_trajectory') and fc.emotional_trajectory is not None:
                emotional_traj = fc.emotional_trajectory
                if isinstance(emotional_traj, (list, np.ndarray)):
                    tensor_data['emotional_trajectory'] = [float(x) for x in emotional_traj]
            
            # Semantic field tensor
            if hasattr(fc, 'semantic_field') and fc.semantic_field is not None:
                semantic_field = fc.semantic_field
                if isinstance(semantic_field, (list, np.ndarray)):
                    # Handle complex semantic field
                    if len(semantic_field) > 0 and isinstance(semantic_field[0], complex):
                        tensor_real = [float(x.real) for x in semantic_field]
                        tensor_imag = [float(x.imag) for x in semantic_field]
                        tensor_data['semantic_field_real'] = tensor_real
                        tensor_data['semantic_field_imag'] = tensor_imag
                    else:
                        tensor_data['semantic_field_real'] = [float(x) for x in semantic_field]
                        tensor_data['semantic_field_imag'] = [0.0] * len(semantic_field)
        
        return tensor_data
    
    def get_metrics(self) -> IntakeMetrics:
        """Get current processing metrics"""
        return self.metrics
    
    def reset_metrics(self):
        """Reset processing metrics"""
        self.metrics = IntakeMetrics()
        logger.info("Intake processor metrics reset")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create test intake processor
    processor = IntakeProcessor(validation_strict=True)
    
    print("✅ IntakeProcessor created successfully")
    print(f"   Validation strict: {processor.validation_strict}")
    print(f"   Normalize components: {processor.normalize_field_components}")
    print(f"   Extract tensors: {processor.extract_tensor_representations}")