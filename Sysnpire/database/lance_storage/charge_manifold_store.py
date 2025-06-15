"""
Charge Manifold Store - Lance Tensor Storage for ConceptualCharge Objects

Connects the ChargeFactory output (ConceptualCharge objects) directly to Lance tensor storage.
Extracts computed field theory results and stores them as tensors without duplicating math.

Data Flow:
ChargeFactory → ConceptualCharge objects → ChargeManifo1dStore → Lance tensors
"""

import numpy as np
import lance
import pyarrow as pa
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time
import logging

# Import actual ConceptualCharge from model layer
try:
    from ...model.mathematics.conceptual_charge import ConceptualCharge
    CONCEPTUAL_CHARGE_AVAILABLE = True
except ImportError:
    # Development fallback
    ConceptualCharge = Any
    CONCEPTUAL_CHARGE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChargeManifoldStore:
    """
    Tensor storage connector for ConceptualCharge objects.
    
    Extracts computed field theory results from ConceptualCharge objects
    and stores them efficiently in Lance tensor format. Does NOT duplicate
    the mathematical calculations - uses the actual results from ChargeFactory.
    """
    
    def __init__(self, storage_path: Union[str, Path]):
        """
        Initialize charge manifold storage.
        
        Args:
            storage_path: Path to Lance dataset storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Lance dataset paths
        self.charges_dataset_path = self.storage_path / "charges"
        self.field_components_dataset_path = self.storage_path / "field_components"
        self.manifold_state_dataset_path = self.storage_path / "manifold_state"
        
        # Initialize Arrow schemas
        self._init_schemas()
        
        # Statistics
        self.stats = {
            'charges_stored': 0,
            'total_storage_time': 0.0,
            'tensor_compression_ratio': 0.0,
            'field_extractions': 0
        }
        
        logger.info(f"ChargeManifoldStore initialized at {self.storage_path}")
    
    def _init_schemas(self):
        """Initialize Arrow schemas for different data types"""
        
        # Schema for core charge data
        self.charge_schema = pa.schema([
            ('charge_id', pa.string()),
            ('token', pa.string()),
            ('timestamp', pa.timestamp('us')),
            ('observational_state', pa.float64()),
            ('gamma', pa.float64()),
            
            # Complete charge results (from compute_complete_charge())
            ('complete_charge_real', pa.float64()),
            ('complete_charge_imag', pa.float64()),
            ('charge_magnitude', pa.float64()),
            ('charge_phase', pa.float64()),
            
            # Context and metadata
            ('context_json', pa.string()),
            ('semantic_vector', pa.list_(pa.float32())),  # BGE 1024-dim vector
            
            # DTF enhancement data
            ('dtf_enhanced', pa.bool_()),
            ('dtf_semantic_field_real', pa.float64()),
            ('dtf_semantic_field_imag', pa.float64())
        ])
        
        # Schema for field components (trajectory operators, emotional field, etc.)
        self.field_components_schema = pa.schema([
            ('charge_id', pa.string()),
            ('observational_state', pa.float64()),
            ('timestamp', pa.timestamp('us')),
            
            # Trajectory operators T(τ, C, s) - complex values
            ('trajectory_operators_real', pa.list_(pa.float64())),
            ('trajectory_operators_imag', pa.list_(pa.float64())),
            
            # Emotional trajectory integration E^trajectory(τ, s)
            ('emotional_trajectory', pa.list_(pa.float64())),
            
            # Semantic field generation Φ^semantic(τ, s) - complex values
            ('semantic_field_real', pa.list_(pa.float64())),
            ('semantic_field_imag', pa.list_(pa.float64())),
            
            # Phase integration θ_total(τ,C,s)
            ('total_phase', pa.float64()),
            
            # Observational persistence Ψ_persistence(s-s₀)
            ('persistence_value', pa.float64()),
        ])
        
        # Schema for manifold state evolution
        self.manifold_state_schema = pa.schema([
            ('charge_id', pa.string()),
            ('observational_state', pa.float64()),
            ('timestamp', pa.timestamp('us')),
            ('trajectory_history', pa.list_(pa.float64())),
            ('frequency_evolution_json', pa.string()),
            ('phase_accumulation_json', pa.string())
        ])
    
    def store_charge(self, charge: ConceptualCharge) -> bool:
        """
        Store a single ConceptualCharge object in Lance tensor format.
        
        Extracts all computed field theory results from the charge object
        and stores them as tensors without duplicating calculations.
        
        Args:
            charge: ConceptualCharge object from ChargeFactory
            
        Returns:
            True if successfully stored
        """
        if not CONCEPTUAL_CHARGE_AVAILABLE:
            logger.error("ConceptualCharge not available - check imports")
            return False
        
        start_time = time.time()
        
        try:
            # Extract computed charge results (uses actual Q(τ, C, s) math)
            complete_charge = charge.compute_complete_charge()
            charge_magnitude = charge.get_charge_magnitude()
            charge_phase = charge.get_phase_factor()
            
            # Extract field components using actual methods
            field_components = self._extract_field_components(charge)
            manifold_state = self._extract_manifold_state(charge)
            
            # Extract DTF information if available
            dtf_enhanced = getattr(charge, 'dtf_enhanced', False)
            dtf_semantic_field = getattr(charge, 'dtf_semantic_field', complex(0, 0))
            
            # Create charge record
            charge_record = {
                'charge_id': [getattr(charge, 'charge_id', f'charge_{int(time.time() * 1000000)}')],
                'token': [charge.token],
                'timestamp': [pa.scalar(int(time.time() * 1000000), type=pa.timestamp('us'))],
                'observational_state': [charge.observational_state],
                'gamma': [charge.gamma],
                'complete_charge_real': [float(complete_charge.real)],
                'complete_charge_imag': [float(complete_charge.imag)],
                'charge_magnitude': [float(charge_magnitude)],
                'charge_phase': [float(charge_phase)],
                'context_json': [str(charge.context)],
                'semantic_vector': [charge.semantic_vector.tolist()],
                'dtf_enhanced': [dtf_enhanced],
                'dtf_semantic_field_real': [float(dtf_semantic_field.real)],
                'dtf_semantic_field_imag': [float(dtf_semantic_field.imag)]
            }
            
            # Store to Lance datasets
            self._append_to_dataset(charge_record, self.charges_dataset_path, self.charge_schema)
            self._append_to_dataset(field_components, self.field_components_dataset_path, self.field_components_schema)
            self._append_to_dataset(manifold_state, self.manifold_state_dataset_path, self.manifold_state_schema)
            
            # Update statistics
            self.stats['charges_stored'] += 1
            self.stats['total_storage_time'] += time.time() - start_time
            self.stats['field_extractions'] += 1
            
            logger.debug(f"Stored charge {charge.token} with magnitude {charge_magnitude:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store charge {getattr(charge, 'token', 'unknown')}: {e}")
            return False
    
    def _extract_field_components(self, charge: ConceptualCharge) -> Dict[str, List]:
        """
        Extract field components using actual ConceptualCharge methods.
        
        Uses the real field theory calculations from the charge object
        rather than duplicating the mathematics.
        """
        s = charge.observational_state
        charge_id = getattr(charge, 'charge_id', f'charge_{int(time.time() * 1000000)}')
        
        # Extract trajectory operators (complex values)
        trajectory_ops = []
        for dim in range(3):  # 3D trajectory operators
            t_op = charge.trajectory_operator(s, dim)
            trajectory_ops.append(t_op)
        
        trajectory_real = [float(t.real) for t in trajectory_ops]
        trajectory_imag = [float(t.imag) for t in trajectory_ops]
        
        # Extract emotional trajectory (real values)
        emotional_traj = charge.emotional_trajectory_integration(s)
        emotional_trajectory = [float(x) for x in emotional_traj]
        
        # Extract semantic field (complex values)
        semantic_field = charge.semantic_field_generation(s)
        if len(semantic_field) > 0 and hasattr(semantic_field[0], 'real'):
            # Complex semantic field
            semantic_real = [float(x.real) for x in semantic_field]
            semantic_imag = [float(x.imag) for x in semantic_field]
        else:
            # Real semantic field
            semantic_real = [float(x) for x in semantic_field]
            semantic_imag = [0.0] * len(semantic_field)
        
        # Extract phase and persistence
        total_phase = charge.total_phase_integration(s)
        persistence_value = charge.observational_persistence(s)
        
        return {
            'charge_id': [charge_id],
            'observational_state': [s],
            'timestamp': [pa.scalar(int(time.time() * 1000000), type=pa.timestamp('us'))],
            'trajectory_operators_real': [trajectory_real],
            'trajectory_operators_imag': [trajectory_imag],
            'emotional_trajectory': [emotional_trajectory],
            'semantic_field_real': [semantic_real],
            'semantic_field_imag': [semantic_imag],
            'total_phase': [float(total_phase)],
            'persistence_value': [float(persistence_value)]
        }
    
    def _extract_manifold_state(self, charge: ConceptualCharge) -> Dict[str, List]:
        """Extract manifold evolution state from charge object"""
        charge_id = getattr(charge, 'charge_id', f'charge_{int(time.time() * 1000000)}')
        
        return {
            'charge_id': [charge_id],
            'observational_state': [charge.observational_state],
            'timestamp': [pa.scalar(int(time.time() * 1000000), type=pa.timestamp('us'))],
            'trajectory_history': [charge.trajectory_history],
            'frequency_evolution_json': [str(charge.frequency_evolution)],
            'phase_accumulation_json': [str(charge.phase_accumulation)]
        }
    
    def _append_to_dataset(self, data: Dict[str, List], dataset_path: Path, schema: pa.Schema):
        """Append data to Lance dataset"""
        # Convert to Arrow table
        table = pa.table(data, schema=schema)
        
        # Append to existing dataset or create new one
        if dataset_path.exists():
            # Append to existing dataset
            dataset = lance.dataset(str(dataset_path))
            lance.write_dataset(table, str(dataset_path), mode='append')
        else:
            # Create new dataset
            lance.write_dataset(table, str(dataset_path))
    
    def store_charges_batch(self, charges: List[ConceptualCharge]) -> int:
        """
        Store multiple charges efficiently in batch.
        
        Args:
            charges: List of ConceptualCharge objects from ChargeFactory
            
        Returns:
            Number of successfully stored charges
        """
        logger.info(f"Batch storing {len(charges)} charges to Lance tensors...")
        
        success_count = 0
        batch_start = time.time()
        
        # Collect all data for batch processing
        charge_records = []
        field_components_batch = []
        manifold_states_batch = []
        
        for charge in charges:
            try:
                # Extract all data using actual ConceptualCharge methods
                complete_charge = charge.compute_complete_charge()
                charge_magnitude = charge.get_charge_magnitude()
                charge_phase = charge.get_phase_factor()
                
                charge_id = getattr(charge, 'charge_id', f'charge_{int(time.time() * 1000000)}_{success_count}')
                
                # Extract DTF information if available
                dtf_enhanced = getattr(charge, 'dtf_enhanced', False)
                dtf_semantic_field = getattr(charge, 'dtf_semantic_field', complex(0, 0))
                
                # Charge record
                charge_records.append({
                    'charge_id': charge_id,
                    'token': charge.token,
                    'timestamp': pa.scalar(int(time.time() * 1000000), type=pa.timestamp('us')),
                    'observational_state': charge.observational_state,
                    'gamma': charge.gamma,
                    'complete_charge_real': float(complete_charge.real),
                    'complete_charge_imag': float(complete_charge.imag),
                    'charge_magnitude': float(charge_magnitude),
                    'charge_phase': float(charge_phase),
                    'context_json': str(charge.context),
                    'semantic_vector': charge.semantic_vector.tolist(),
                    'dtf_enhanced': dtf_enhanced,
                    'dtf_semantic_field_real': float(dtf_semantic_field.real),
                    'dtf_semantic_field_imag': float(dtf_semantic_field.imag)
                })
                
                # Field components
                field_data = self._extract_field_components(charge)
                field_components_batch.append({k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in field_data.items()})
                
                # Manifold state
                state_data = self._extract_manifold_state(charge)
                manifold_states_batch.append({k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in state_data.items()})
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process charge {getattr(charge, 'token', 'unknown')} in batch: {e}")
                continue
        
        # Write batches to Lance
        if charge_records:
            self._write_batch_to_dataset(charge_records, self.charges_dataset_path, self.charge_schema)
            self._write_batch_to_dataset(field_components_batch, self.field_components_dataset_path, self.field_components_schema)
            self._write_batch_to_dataset(manifold_states_batch, self.manifold_state_dataset_path, self.manifold_state_schema)
        
        batch_time = time.time() - batch_start
        logger.info(f"Batch storage complete: {success_count}/{len(charges)} charges stored in {batch_time:.3f}s")
        
        return success_count
    
    def _write_batch_to_dataset(self, records: List[Dict], dataset_path: Path, schema: pa.Schema):
        """Write batch of records to Lance dataset"""
        if not records:
            return
        
        # Convert to proper format for Arrow table
        table_data = {}
        for key in schema.names:
            table_data[key] = [record[key] for record in records]
        
        table = pa.table(table_data, schema=schema)
        
        # Write to Lance
        if dataset_path.exists():
            lance.write_dataset(table, str(dataset_path), mode='append')
        else:
            lance.write_dataset(table, str(dataset_path))
    
    def load_charge(self, charge_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a charge and its field components from Lance storage.
        
        Args:
            charge_id: ID of charge to load
            
        Returns:
            Dictionary with charge data and field components
        """
        try:
            # Load from charges dataset
            if not self.charges_dataset_path.exists():
                return None
            
            charges_dataset = lance.dataset(str(self.charges_dataset_path))
            charge_table = charges_dataset.to_table(filter=f"charge_id = '{charge_id}'")
            
            if len(charge_table) == 0:
                return None
            
            # Load field components
            components_data = {}
            if self.field_components_dataset_path.exists():
                components_dataset = lance.dataset(str(self.field_components_dataset_path))
                components_table = components_dataset.to_table(filter=f"charge_id = '{charge_id}'")
                if len(components_table) > 0:
                    components_data = components_table.to_pandas().iloc[-1].to_dict()
            
            # Combine charge and field data
            charge_data = charge_table.to_pandas().iloc[-1].to_dict()
            charge_data['field_components'] = components_data
            
            return charge_data
            
        except Exception as e:
            logger.error(f"Failed to load charge {charge_id}: {e}")
            return None
    
    def query_charges_by_magnitude(self, min_magnitude: float, max_magnitude: float) -> List[Dict[str, Any]]:
        """Query charges by magnitude range using Lance filtering"""
        try:
            if not self.charges_dataset_path.exists():
                return []
            
            dataset = lance.dataset(str(self.charges_dataset_path))
            table = dataset.to_table(filter=f"charge_magnitude >= {min_magnitude} AND charge_magnitude <= {max_magnitude}")
            
            return table.to_pandas().to_dict('records')
            
        except Exception as e:
            logger.error(f"Failed to query charges by magnitude: {e}")
            return []
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = self.stats.copy()
        
        # Add dataset sizes
        for name, path in [
            ('charges_dataset_size', self.charges_dataset_path),
            ('field_components_dataset_size', self.field_components_dataset_path),
            ('manifold_state_dataset_size', self.manifold_state_dataset_path)
        ]:
            if path.exists():
                try:
                    dataset = lance.dataset(str(path))
                    stats[name] = len(dataset.to_table())
                except:
                    stats[name] = 0
            else:
                stats[name] = 0
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test storage initialization
    store = ChargeManifoldStore("test_charge_manifold")
    
    print("✅ ChargeManifoldStore created successfully")
    print(f"   Storage path: {store.storage_path}")
    print(f"   Schemas initialized: {len([store.charge_schema, store.field_components_schema, store.manifold_state_schema])}")
    print(f"   Statistics: {store.get_storage_stats()}")