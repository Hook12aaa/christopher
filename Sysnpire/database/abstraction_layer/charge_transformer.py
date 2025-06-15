"""
Charge Transformer - Enhanced T[Q] Operations

Evolved version of transformation_operator.py optimized for tensor storage.
Transforms conceptual charges into geometric imprints ready for Lance storage.

This is the "vinyl pressing" process that converts charges into efficient
tensor representations for the manifold storage system.
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import logging

try:
    from ..conceptual_charge_object import ConceptualChargeObject
    from ..transformation_operator import TransformationOperator, TransformationParameters
except ImportError:
    # Development fallback
    ConceptualChargeObject = Any
    TransformationOperator = Any
    TransformationParameters = Any

logger = logging.getLogger(__name__)


@dataclass
class TensorTransformationConfig:
    """Configuration for tensor-optimized charge transformation"""
    tensor_dimensions: Tuple[int, ...] = (64, 64, 64)  # 3D tensor grid
    spatial_extent: float = 10.0  # Spatial extent in field units
    preserve_phase_information: bool = True  # Keep complex phase data
    normalize_for_storage: bool = True  # Normalize tensors for Lance
    batch_optimization: bool = True  # Enable batch processing optimizations
    compression_threshold: float = 1e-8  # Sparse compression threshold


class ChargeTransformer:
    """
    Transform ConceptualChargeObjects into tensor representations for Lance storage.
    
    Enhanced version of TransformationOperator optimized for:
    - Native tensor operations
    - Lance+Arrow storage efficiency  
    - Batch processing
    - Zero-copy data sharing
    """
    
    def __init__(self, 
                 config: Optional[TensorTransformationConfig] = None,
                 legacy_operator: Optional[TransformationOperator] = None):
        """
        Initialize charge transformer.
        
        Args:
            config: Tensor transformation configuration
            legacy_operator: Optional legacy TransformationOperator for compatibility
        """
        self.config = config or TensorTransformationConfig()
        self.legacy_operator = legacy_operator
        
        # Tensor grid setup
        self.grid_shape = self.config.tensor_dimensions
        self.spatial_extent = self.config.spatial_extent
        
        # Create spatial coordinate grids
        self.coordinate_grids = self._create_coordinate_grids()
        
        # Processing statistics
        self.stats = {
            'transforms_completed': 0,
            'batch_transforms': 0,
            'compression_applied': 0,
            'average_sparsity': 0.0,
            'total_processing_time': 0.0
        }
        
        logger.info(f"ChargeTransformer initialized - grid shape: {self.grid_shape}")
    
    def transform_charge(self, 
                        charge: ConceptualChargeObject,
                        target_position: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Transform a single charge into tensor representation.
        
        Args:
            charge: ConceptualChargeObject to transform
            target_position: Optional target position in tensor grid
            
        Returns:
            Dictionary with tensor data ready for Lance storage
        """
        start_time = time.time()
        
        try:
            # Extract charge properties
            magnitude = charge.magnitude
            phase = charge.phase if hasattr(charge, 'phase') else 0.0
            
            # Determine spatial positioning
            if target_position is not None:
                center_position = target_position
            elif hasattr(charge, 'metadata') and charge.metadata:
                if hasattr(charge.metadata, 'field_position') and charge.metadata.field_position:
                    center_position = np.array(charge.metadata.field_position[:3])
                else:
                    center_position = self._compute_natural_position(charge)
            else:
                center_position = self._compute_natural_position(charge)
            
            # Generate tensor imprint
            tensor_imprint = self._generate_tensor_imprint(
                magnitude=magnitude,
                phase=phase,
                center_position=center_position,
                charge=charge
            )
            
            # Apply tensor optimizations
            tensor_data = self._optimize_for_storage(tensor_imprint, charge)
            
            # Update statistics
            self.stats['transforms_completed'] += 1
            self.stats['total_processing_time'] += time.time() - start_time
            
            logger.debug(f"Transformed charge {charge.charge_id} to tensor shape {tensor_imprint.shape}")
            
            return tensor_data
            
        except Exception as e:
            logger.error(f"Failed to transform charge {getattr(charge, 'charge_id', 'unknown')}: {e}")
            raise
    
    def transform_batch(self, 
                       charges: List[ConceptualChargeObject],
                       positions: Optional[List[np.ndarray]] = None) -> List[Dict[str, Any]]:
        """
        Transform multiple charges efficiently in batch.
        
        Args:
            charges: List of charges to transform
            positions: Optional list of target positions
            
        Returns:
            List of tensor representations
        """
        logger.info(f"Batch transforming {len(charges)} charges...")
        start_time = time.time()
        
        if positions is None:
            positions = [None] * len(charges)
        
        batch_results = []
        
        if self.config.batch_optimization and len(charges) > 1:
            # Optimized batch processing
            batch_results = self._batch_transform_optimized(charges, positions)
        else:
            # Individual processing
            for charge, position in zip(charges, positions):
                tensor_data = self.transform_charge(charge, position)
                batch_results.append(tensor_data)
        
        # Update batch statistics
        self.stats['batch_transforms'] += 1
        batch_time = time.time() - start_time
        
        logger.info(f"Batch transformation complete: {len(batch_results)} tensors "
                   f"generated in {batch_time:.3f}s")
        
        return batch_results
    
    def _create_coordinate_grids(self) -> Dict[str, np.ndarray]:
        """Create coordinate grids for tensor space"""
        # Create 3D coordinate grids
        if len(self.grid_shape) >= 3:
            x_coords = np.linspace(-self.spatial_extent/2, self.spatial_extent/2, self.grid_shape[0])
            y_coords = np.linspace(-self.spatial_extent/2, self.spatial_extent/2, self.grid_shape[1])
            z_coords = np.linspace(-self.spatial_extent/2, self.spatial_extent/2, self.grid_shape[2])
            
            X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
            
            return {
                'x': X,
                'y': Y, 
                'z': Z,
                'r': np.sqrt(X**2 + Y**2 + Z**2)  # Radial coordinate
            }
        else:
            raise ValueError(f"Grid shape must be 3D, got {self.grid_shape}")
    
    def _compute_natural_position(self, charge: ConceptualChargeObject) -> np.ndarray:
        """
        Compute natural position for charge in tensor space.
        
        Uses field-theoretic principles to find where charge "wants to sit"
        """
        # Default to origin if no other information
        default_position = np.array([0.0, 0.0, 0.0])
        
        try:
            # Try to use field components to determine position
            if hasattr(charge, 'field_components') and charge.field_components:
                fc = charge.field_components
                
                # Use trajectory operators for x-position
                if hasattr(fc, 'trajectory_operators') and fc.trajectory_operators:
                    traj_ops = fc.trajectory_operators
                    if isinstance(traj_ops, (list, np.ndarray)) and len(traj_ops) > 0:
                        x_pos = np.real(np.mean(traj_ops)) * self.spatial_extent * 0.1
                        default_position[0] = x_pos
                
                # Use emotional trajectory for y-position
                if hasattr(fc, 'emotional_trajectory') and fc.emotional_trajectory:
                    emotional_traj = fc.emotional_trajectory
                    if isinstance(emotional_traj, (list, np.ndarray)) and len(emotional_traj) > 0:
                        y_pos = np.mean(emotional_traj) * self.spatial_extent * 0.1
                        default_position[1] = y_pos
                
                # Use semantic field for z-position
                if hasattr(fc, 'semantic_field') and fc.semantic_field:
                    semantic_field = fc.semantic_field
                    if isinstance(semantic_field, (list, np.ndarray)) and len(semantic_field) > 0:
                        z_pos = np.real(np.mean(semantic_field)) * self.spatial_extent * 0.1
                        default_position[2] = z_pos
            
            # Clamp to grid bounds
            max_extent = self.spatial_extent * 0.4  # Stay within bounds
            default_position = np.clip(default_position, -max_extent, max_extent)
            
            return default_position
            
        except Exception as e:
            logger.warning(f"Failed to compute natural position for charge: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def _generate_tensor_imprint(self, 
                                magnitude: float,
                                phase: float,
                                center_position: np.ndarray,
                                charge: ConceptualChargeObject) -> np.ndarray:
        """
        Generate 3D tensor imprint for the charge.
        
        Creates a spatial field pattern centered at the given position.
        """
        # Compute distances from center
        X, Y, Z = self.coordinate_grids['x'], self.coordinate_grids['y'], self.coordinate_grids['z']
        
        dx = X - center_position[0]
        dy = Y - center_position[1] 
        dz = Z - center_position[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Characteristic length scale based on charge magnitude
        sigma = max(0.5, min(2.0, magnitude)) * self.spatial_extent * 0.1
        
        # Generate spatial profile (Gaussian with exponential decay)
        spatial_profile = magnitude * np.exp(-r**2 / (2 * sigma**2))
        
        # Apply phase modulation if preserving phase information
        if self.config.preserve_phase_information:
            # Complex tensor with phase information
            phase_modulation = np.exp(1j * phase)
            tensor_imprint = spatial_profile * phase_modulation
        else:
            # Real tensor 
            tensor_imprint = spatial_profile * np.cos(phase)
        
        return tensor_imprint
    
    def _optimize_for_storage(self, 
                             tensor_imprint: np.ndarray,
                             charge: ConceptualChargeObject) -> Dict[str, Any]:
        """
        Optimize tensor for Lance storage.
        
        Applies compression, normalization, and formatting for Arrow schema.
        """
        tensor_data = {
            'charge_id': charge.charge_id,
            'tensor_shape': list(tensor_imprint.shape),
            'spatial_extent': self.spatial_extent,
            'center_position': getattr(charge.metadata, 'field_position', [0.0, 0.0, 0.0]) if hasattr(charge, 'metadata') else [0.0, 0.0, 0.0]
        }
        
        # Handle complex vs real tensors
        if np.iscomplexobj(tensor_imprint):
            # Split complex tensor for storage
            tensor_data['tensor_real'] = tensor_imprint.real.astype(np.float32)
            tensor_data['tensor_imag'] = tensor_imprint.imag.astype(np.float32)
            tensor_data['is_complex'] = True
        else:
            # Real tensor
            tensor_data['tensor_real'] = tensor_imprint.astype(np.float32)
            tensor_data['tensor_imag'] = None
            tensor_data['is_complex'] = False
        
        # Apply sparse compression if enabled
        if self.config.compression_threshold > 0:
            tensor_data = self._apply_sparse_compression(tensor_data)
        
        # Normalization for storage consistency
        if self.config.normalize_for_storage:
            tensor_data = self._normalize_tensor_data(tensor_data)
        
        return tensor_data
    
    def _apply_sparse_compression(self, tensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply sparse compression to reduce storage size"""
        threshold = self.config.compression_threshold
        
        # Compress real part
        if 'tensor_real' in tensor_data and tensor_data['tensor_real'] is not None:
            tensor_real = tensor_data['tensor_real']
            mask = np.abs(tensor_real) > threshold
            
            if np.sum(mask) < tensor_real.size * 0.5:  # If more than 50% sparse
                # Store as sparse format
                indices = np.where(mask)
                values = tensor_real[mask]
                
                tensor_data['tensor_real_sparse'] = {
                    'indices': indices,
                    'values': values,
                    'shape': tensor_real.shape,
                    'is_sparse': True
                }
                tensor_data['tensor_real'] = None  # Remove dense version
                self.stats['compression_applied'] += 1
                
                # Update sparsity statistics
                sparsity = 1.0 - (len(values) / tensor_real.size)
                self.stats['average_sparsity'] = (
                    (self.stats['average_sparsity'] * (self.stats['compression_applied'] - 1) + sparsity)
                    / self.stats['compression_applied']
                )
        
        # Compress imaginary part if present
        if 'tensor_imag' in tensor_data and tensor_data['tensor_imag'] is not None:
            tensor_imag = tensor_data['tensor_imag']
            mask = np.abs(tensor_imag) > threshold
            
            if np.sum(mask) < tensor_imag.size * 0.5:
                indices = np.where(mask)
                values = tensor_imag[mask]
                
                tensor_data['tensor_imag_sparse'] = {
                    'indices': indices,
                    'values': values,
                    'shape': tensor_imag.shape,
                    'is_sparse': True
                }
                tensor_data['tensor_imag'] = None
        
        return tensor_data
    
    def _normalize_tensor_data(self, tensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize tensor data for consistent storage"""
        # Normalize real part
        if 'tensor_real' in tensor_data and tensor_data['tensor_real'] is not None:
            tensor_real = tensor_data['tensor_real']
            max_val = np.max(np.abs(tensor_real))
            if max_val > 0:
                tensor_data['tensor_real'] = tensor_real / max_val
                tensor_data['normalization_factor_real'] = float(max_val)
            else:
                tensor_data['normalization_factor_real'] = 1.0
        
        # Normalize imaginary part  
        if 'tensor_imag' in tensor_data and tensor_data['tensor_imag'] is not None:
            tensor_imag = tensor_data['tensor_imag']
            max_val = np.max(np.abs(tensor_imag))
            if max_val > 0:
                tensor_data['tensor_imag'] = tensor_imag / max_val
                tensor_data['normalization_factor_imag'] = float(max_val)
            else:
                tensor_data['normalization_factor_imag'] = 1.0
        
        return tensor_data
    
    def _batch_transform_optimized(self,
                                  charges: List[ConceptualChargeObject],
                                  positions: List[Optional[np.ndarray]]) -> List[Dict[str, Any]]:
        """Optimized batch transformation using vectorized operations"""
        # For now, fall back to individual processing
        # TODO: Implement true vectorized batch processing
        batch_results = []
        for charge, position in zip(charges, positions):
            tensor_data = self.transform_charge(charge, position)
            batch_results.append(tensor_data)
        
        return batch_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get transformation statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset transformation statistics"""
        self.stats = {
            'transforms_completed': 0,
            'batch_transforms': 0,
            'compression_applied': 0,
            'average_sparsity': 0.0,
            'total_processing_time': 0.0
        }


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create test transformer
    config = TensorTransformationConfig(
        tensor_dimensions=(32, 32, 32),  # Smaller for testing
        spatial_extent=5.0
    )
    transformer = ChargeTransformer(config=config)
    
    print("✅ ChargeTransformer created successfully")
    print(f"   Grid shape: {transformer.grid_shape}")
    print(f"   Spatial extent: {transformer.spatial_extent}")
    print(f"   Statistics: {transformer.get_statistics()}")