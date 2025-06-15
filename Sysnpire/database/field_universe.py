"""
Field Universe - Main Orchestrator for Field-Theoretic Manifold Storage

Evolved enterprise-grade storage system using Lance+Arrow with abstraction layer.
Orchestrates the complete pipeline: intake → abstraction → storage → handler

Data Flow:
1. Intake: Receives ConceptualChargeObjects
2. Abstraction: Validates, transforms, and optimizes for tensor storage  
3. Storage: Stores in Lance+Arrow with spatial indexing
4. Handler: Provides query interface and field computations
"""

import numpy as np
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass

# Core imports
from .conceptual_charge_object import ConceptualChargeObject, FieldComponents

# Import actual ConceptualCharge from model layer (for ChargeFactory output)
try:
    from ..model.mathematics.conceptual_charge import ConceptualCharge
    CONCEPTUAL_CHARGE_AVAILABLE = True
except ImportError:
    # Fallback during development
    ConceptualCharge = ConceptualChargeObject
    CONCEPTUAL_CHARGE_AVAILABLE = False

# New abstraction layer imports
from .abstraction_layer.intake_processor import IntakeProcessor
from .abstraction_layer.charge_transformer import ChargeTransformer, TensorTransformationConfig

# Storage layer imports
try:
    from .lance_storage.charge_manifold_store import ChargeManifoldStore
    CHARGE_MANIFOLD_STORE_AVAILABLE = True
except ImportError as e:
    ChargeManifoldStore = None
    CHARGE_MANIFOLD_STORE_AVAILABLE = False

try:
    from .spatial_index.hilbert_encoder import HilbertEncoder
    SPATIAL_INDEX_AVAILABLE = True
except ImportError:
    HilbertEncoder = None
    SPATIAL_INDEX_AVAILABLE = False

try:
    from .redis_cache.hot_regions import HotRegionsCache
    REDIS_CACHE_AVAILABLE = True
except ImportError:
    HotRegionsCache = None
    REDIS_CACHE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class UniverseMetrics:
    """Universe-wide metrics and statistics."""
    total_charges: int
    total_energy: float
    average_magnitude: float
    field_density: float
    last_updated: float
    # New metrics for abstraction layer
    abstraction_efficiency: float = 0.0
    tensor_storage_size: int = 0
    cache_hit_rate: float = 0.0


@dataclass
class FieldUniverseConfig:
    """Configuration for FieldUniverse orchestrator"""
    # Storage paths
    lance_storage_path: str = "manifold_tensors"
    redis_cache_url: str = "redis://localhost:6379"
    
    # Processing configuration
    enable_abstraction_layer: bool = True
    enable_redis_cache: bool = True
    strict_validation: bool = True
    
    # Performance settings
    batch_size: int = 100
    cache_ttl: int = 3600  # 1 hour
    
    # Query interface
    enable_sql_interface: bool = True  # DuckDB SQL interface to Lance data


class FieldUniverse:
    """
    Main Orchestrator for Tensor-Native Field-Theoretic Manifold Storage
    
    Pure tensor storage architecture:
    - Intake: Receives ConceptualChargeObjects
    - Abstraction: Validates and transforms charges to tensors
    - Storage: Lance tensor storage with spatial indexing (NO SQL)
    - Query: DuckDB SQL interface TO Lance data + Redis cache
    - Handler: Field computations on tensor data
    """
    
    def __init__(self, config: Optional[FieldUniverseConfig] = None):
        """
        Initialize the tensor-native field universe.
        
        Args:
            config: Configuration for tensor storage and processing
        """
        self.config = config or FieldUniverseConfig()
        
        # Tensor storage only - no legacy SQL storage
        self.charges: Dict[str, ConceptualChargeObject] = {}  # Hot memory cache only
        self.field_regions: Dict[str, List[str]] = {}  # Spatial organization cache
        
        # Initialize abstraction layer
        self._init_abstraction_layer()
        
        # Initialize tensor storage backends  
        self._init_tensor_storage()
        
        # Load existing tensor data
        self._load_existing_tensors()
        
        logger.info(f"🌌 Tensor-Native Field Universe initialized")
        logger.info(f"   Abstraction layer: {'✅' if self.abstraction_enabled else '❌'}")
        logger.info(f"   Lance tensor storage: {'✅' if self.lance_storage_enabled else '❌'}")
        logger.info(f"   Redis cache: {'✅' if self.cache_enabled else '❌'}")
        logger.info(f"   SQL interface: {'✅' if self.sql_interface_enabled else '❌'}")
        logger.info(f"   Cached charges: {len(self.charges)}")
    
    def _init_abstraction_layer(self):
        """Initialize the abstraction layer components"""
        try:
            if self.config.enable_abstraction_layer:
                # Initialize intake processor
                self.intake_processor = IntakeProcessor(
                    validation_strict=self.config.strict_validation,
                    normalize_field_components=True,
                    extract_tensor_representations=True
                )
                
                # Initialize charge transformer
                tensor_config = TensorTransformationConfig(
                    tensor_dimensions=(64, 64, 64),
                    spatial_extent=10.0,
                    preserve_phase_information=True,
                    normalize_for_storage=True,
                    batch_optimization=True
                )
                self.charge_transformer = ChargeTransformer(config=tensor_config)
                
                self.abstraction_enabled = True
                logger.info("✅ Abstraction layer initialized")
            else:
                self.intake_processor = None
                self.charge_transformer = None
                self.abstraction_enabled = False
                logger.info("❌ Abstraction layer disabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize abstraction layer: {e}")
            self.abstraction_enabled = False
            self.intake_processor = None
            self.charge_transformer = None
    
    def _init_tensor_storage(self):
        """Initialize tensor-native storage backends"""
        # Lance tensor storage (PRIMARY storage)
        if CHARGE_MANIFOLD_STORE_AVAILABLE:
            try:
                self.manifold_store = ChargeManifoldStore(self.config.lance_storage_path)
                self.spatial_indexer = HilbertEncoder() if HilbertEncoder else None
                self.lance_storage_enabled = True
                logger.info("✅ Lance tensor storage initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Lance storage: {e}")
                self.lance_storage_enabled = False
                self.manifold_store = None
                self.spatial_indexer = None
                raise RuntimeError("Lance tensor storage is required - no fallback available")
        else:
            self.lance_storage_enabled = False
            self.manifold_store = None
            self.spatial_indexer = None
            logger.warning("❌ Lance storage not available - install pylance")
            logger.info("⚠️  Continuing without Lance storage (fallback mode)")
        
        # Redis hot cache
        if REDIS_CACHE_AVAILABLE and self.config.enable_redis_cache:
            try:
                self.hot_cache = HotRegionsCache(
                    redis_url=self.config.redis_cache_url,
                    ttl=self.config.cache_ttl
                )
                self.cache_enabled = True
                logger.info("✅ Redis cache initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Redis cache: {e}")
                self.cache_enabled = False
                self.hot_cache = None
        else:
            self.cache_enabled = False
            self.hot_cache = None
            logger.info("❌ Redis cache not available")
        
        # DuckDB SQL interface to Lance data
        if self.config.enable_sql_interface:
            try:
                # TODO: Initialize DuckDB connection to Lance datasets
                # self.sql_interface = DuckDBBridge(self.config.lance_storage_path)
                self.sql_interface_enabled = True
                logger.info("✅ SQL interface ready (TODO: implement)")
            except Exception as e:
                logger.error(f"Failed to initialize SQL interface: {e}")
                self.sql_interface_enabled = False
        else:
            self.sql_interface_enabled = False
    
    def _load_existing_tensors(self):
        """Load existing tensor data from Lance storage"""
        if self.lance_storage_enabled:
            try:
                # TODO: Load charges from Lance datasets
                # cached_charges = self.field_store.load_all_charges()
                # self.charges.update(cached_charges)
                logger.info("Tensor data loading ready (TODO: implement)")
            except Exception as e:
                logger.error(f"Failed to load tensor data: {e}")
        
        # Load hot regions from Redis cache
        if self.cache_enabled:
            try:
                # TODO: Load hot regions from Redis
                # hot_regions = self.hot_cache.load_hot_regions()
                logger.info("Cache data loading ready (TODO: implement)")
            except Exception as e:
                logger.error(f"Failed to load cache data: {e}")
    
    def add_charge(self, charge_obj: Union[ConceptualCharge, ConceptualChargeObject]) -> bool:
        """
        Add a ConceptualCharge object directly to tensor storage.
        
        Args:
            charge_obj: ConceptualCharge object from ChargeFactory (with computed Q(τ, C, s))
            
        Returns:
            True if successfully stored
        """
        try:
            charge_id = getattr(charge_obj, 'charge_id', getattr(charge_obj, 'token', 'unknown'))
            logger.debug(f"Storing ConceptualCharge {charge_id} directly to tensor storage...")
            
            # Store ConceptualCharge directly (uses actual Q(τ, C, s) math)
            storage_success = self._store_charge_tensor_native(charge_obj)
            
            if storage_success:
                # Update in-memory cache for fast access
                self.charges[charge_id] = charge_obj
                
                # Calculate magnitude for logging (uses actual field theory)
                try:
                    magnitude = charge_obj.get_charge_magnitude()
                    phase = charge_obj.get_phase_factor()
                    logger.info(f"✅ ConceptualCharge {charge_id} stored - magnitude: {magnitude:.4f}, phase: {phase:.4f}")
                except Exception as e:
                    logger.info(f"✅ ConceptualCharge {charge_id} stored successfully")
                
                return True
            else:
                logger.error(f"Failed to store ConceptualCharge {charge_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add ConceptualCharge: {e}")
            return False
    
    def _store_charge_tensor_native(self, 
                                   charge_obj: ConceptualChargeObject) -> bool:
        """
        Store ConceptualCharge directly in tensor-native backends.
        
        Args:
            charge_obj: ConceptualCharge object from ChargeFactory (with computed Q(τ, C, s))
            
        Returns:
            True if stored successfully in Lance tensor storage
        """
        # Primary storage: Lance tensor backend
        if self.lance_storage_enabled and self.manifold_store:
            try:
                # Store actual ConceptualCharge object (with real Q(τ, C, s) math)
                lance_success = self.manifold_store.store_charge(charge_obj)
                if lance_success:
                    logger.debug(f"Lance tensor storage for {getattr(charge_obj, 'token', 'unknown')}: ✅")
                else:
                    logger.error(f"Lance storage failed for {getattr(charge_obj, 'token', 'unknown')}")
            except Exception as e:
                logger.error(f"Lance storage failed for {getattr(charge_obj, 'token', 'unknown')}: {e}")
                lance_success = False
        else:
            logger.error("Lance storage not available - cannot store charge")
            return False
        
        # Secondary: Redis hot cache
        if self.cache_enabled:
            try:
                # TODO: Implement when HotRegionsCache is available
                # self.hot_cache.cache_charge_data(charge_obj.charge_id, charge_obj)
                logger.debug(f"Redis cache for {getattr(charge_obj, 'token', 'unknown')}: Ready (TODO: implement)")
            except Exception as e:
                logger.error(f"Redis cache failed for {getattr(charge_obj, 'token', 'unknown')}: {e}")
        
        return lance_success
    
    def _legacy_process_charge(self, charge_obj: ConceptualChargeObject) -> Dict[str, Any]:
        """
        Legacy charge processing without abstraction layer.
        
        Args:
            charge_obj: Charge to process
            
        Returns:
            Basic processed data dictionary
        """
        # Calculate field position using legacy method
        field_position = self._calculate_field_position(charge_obj)
        if hasattr(charge_obj, 'metadata') and charge_obj.metadata:
            if hasattr(charge_obj.metadata, 'set_field_position'):
                charge_obj.metadata.set_field_position(field_position)
        
        return {
            'charge_id': charge_obj.charge_id,
            'magnitude': charge_obj.magnitude,
            'phase': getattr(charge_obj, 'phase', 0.0),
            'field_position': field_position,
            'processing_method': 'legacy'
        }
    
    def batch_add_charges(self, charges: List[Union[ConceptualCharge, ConceptualChargeObject]]) -> int:
        """
        Add multiple ConceptualCharge objects efficiently using Lance batch processing.
        
        Args:
            charges: List of ConceptualCharge objects from ChargeFactory
            
        Returns:
            Number of successfully added charges
        """
        logger.info(f"Batch storing {len(charges)} ConceptualCharge objects...")
        
        if self.lance_storage_enabled and self.manifold_store and len(charges) > 1:
            # Use Lance batch processing
            try:
                success_count = self.manifold_store.store_charges_batch(charges)
                
                # Update in-memory cache
                for charge in charges[:success_count]:
                    charge_id = getattr(charge, 'charge_id', getattr(charge, 'token', 'unknown'))
                    self.charges[charge_id] = charge
                
                logger.info(f"Lance batch storage complete: {success_count}/{len(charges)} charges stored")
                return success_count
                
            except Exception as e:
                logger.error(f"Lance batch storage failed: {e}")
                # Fallback to individual processing
        
        # Individual processing fallback
        success_count = 0
        for charge in charges:
            if self.add_charge(charge):
                success_count += 1
        
        logger.info(f"Individual storage complete: {success_count}/{len(charges)} charges stored")
        return success_count
    
    def _calculate_field_position(self, charge_obj: ConceptualChargeObject) -> Tuple[float, float, float]:
        """
        Calculate field-theoretic position based on charge properties.
        
        Uses the mathematical formulation to place charges in field-adjacent positions.
        """
        # Use field components to determine position
        fc = charge_obj.field_components
        
        # X position based on trajectory operators
        x_pos = np.mean([t.real for t in fc.trajectory_operators])
        
        # Y position based on emotional trajectory
        y_pos = np.mean(fc.emotional_trajectory) if len(fc.emotional_trajectory) > 0 else 0.0
        
        # Z position based on semantic field
        z_pos = np.mean([s.real for s in fc.semantic_field]) if len(fc.semantic_field) > 0 else 0.0
        
        # Scale to reasonable field coordinates
        x_pos = x_pos * 100.0  # Scale trajectory influence
        y_pos = y_pos * 50.0   # Scale emotional influence  
        z_pos = z_pos * 75.0   # Scale semantic influence
        
        return (float(x_pos), float(y_pos), float(z_pos))
    
    def _determine_field_region(self, position: Tuple[float, float, float]) -> str:
        """Determine field region based on position."""
        x, y, z = position
        
        # Simple grid-based regions
        region_x = int(x // 25)  # 25-unit grid
        region_y = int(y // 25)
        region_z = int(z // 25)
        
        return f"region_{region_x}_{region_y}_{region_z}"
    
    def _establish_field_relationships(self, charge_obj: ConceptualChargeObject):
        """Find nearby charges and establish field relationships."""
        if not charge_obj.metadata.field_position:
            return
        
        x, y, z = charge_obj.metadata.field_position
        proximity_threshold = 50.0  # Field units
        
        for other_id, other_charge in self.charges.items():
            if other_id == charge_obj.charge_id:
                continue
                
            if not other_charge.metadata.field_position:
                continue
            
            # Calculate field distance
            ox, oy, oz = other_charge.metadata.field_position
            distance = np.sqrt((x - ox)**2 + (y - oy)**2 + (z - oz)**2)
            
            if distance < proximity_threshold:
                # Calculate influence strength based on field theory
                influence = self._calculate_field_influence(charge_obj, other_charge, distance)
                
                # Establish bidirectional relationship
                charge_obj.add_nearby_charge(other_id, influence)
                other_charge.add_nearby_charge(charge_obj.charge_id, influence)
    
    def _calculate_field_influence(self, charge1: ConceptualChargeObject, 
                                 charge2: ConceptualChargeObject, distance: float) -> float:
        """Calculate field influence between two charges."""
        # Field influence based on magnitude and proximity
        magnitude_factor = min(charge1.magnitude, charge2.magnitude)
        distance_factor = 1.0 / (1.0 + distance / 10.0)  # Decay with distance
        
        # Phase alignment factor
        phase_diff = abs(charge1.phase - charge2.phase)
        phase_factor = np.cos(phase_diff)  # Stronger influence for aligned phases
        
        return magnitude_factor * distance_factor * phase_factor
    
    def _save_charge_to_db(self, charge_obj: ConceptualChargeObject):
        """Save charge to SQLite database."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            
            # Prepare position data
            pos = charge_obj.metadata.field_position
            pos_x, pos_y, pos_z = (pos[0], pos[1], pos[2]) if pos else (None, None, None)
            
            # Save charge
            cursor.execute("""
                INSERT OR REPLACE INTO charges (
                    charge_id, text_source, creation_timestamp, last_updated,
                    complete_charge_real, complete_charge_imag, magnitude, phase,
                    observational_state, gamma, field_position_x, field_position_y, field_position_z,
                    field_region, data_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                charge_obj.charge_id,
                charge_obj.text_source,
                charge_obj.creation_timestamp,
                charge_obj.last_updated,
                charge_obj.complete_charge.real,
                charge_obj.complete_charge.imag,
                charge_obj.magnitude,
                charge_obj.phase,
                charge_obj.observational_state,
                charge_obj.gamma,
                pos_x, pos_y, pos_z,
                charge_obj.metadata.field_region,
                json.dumps(charge_obj.to_dict())
            ))
            
            # Save relationships
            cursor.execute("DELETE FROM charge_relationships WHERE charge_id = ?", (charge_obj.charge_id,))
            for nearby_id, influence in charge_obj.metadata.collective_influences.items():
                cursor.execute("""
                    INSERT INTO charge_relationships (charge_id, nearby_charge_id, influence_strength)
                    VALUES (?, ?, ?)
                """, (charge_obj.charge_id, nearby_id, influence))
            
            conn.commit()
    
    def _load_charges(self):
        """Load existing charges from database."""
        if not self.storage_path.exists():
            return
            
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data_json FROM charges")
            
            for (data_json,) in cursor.fetchall():
                try:
                    data = json.loads(data_json)
                    charge_obj = ConceptualChargeObject.from_dict(data)
                    self.charges[charge_obj.charge_id] = charge_obj
                    
                    # Add to field regions
                    region = charge_obj.metadata.field_region
                    if region:
                        if region not in self.field_regions:
                            self.field_regions[region] = []
                        self.field_regions[region].append(charge_obj.charge_id)
                        
                except Exception as e:
                    print(f"   ⚠️ Failed to load charge: {e}")
    
    def get_charge(self, charge_id: str) -> Optional[ConceptualChargeObject]:
        """Get charge by ID."""
        return self.charges.get(charge_id)
    
    def query_charges_by_region(self, field_region: str) -> List[ConceptualChargeObject]:
        """Get all charges in a field region."""
        charge_ids = self.field_regions.get(field_region, [])
        return [self.charges[cid] for cid in charge_ids if cid in self.charges]
    
    def query_charges_by_magnitude(self, min_magnitude: float, max_magnitude: float) -> List[ConceptualChargeObject]:
        """Query charges by magnitude range."""
        return [
            charge for charge in self.charges.values()
            if min_magnitude <= charge.magnitude <= max_magnitude
        ]
    
    def query_nearby_charges(self, charge_id: str, max_distance: float = 50.0) -> List[ConceptualChargeObject]:
        """Get charges near a specific charge."""
        base_charge = self.charges.get(charge_id)
        if not base_charge or not base_charge.metadata.field_position:
            return []
        
        x, y, z = base_charge.metadata.field_position
        nearby = []
        
        for other_charge in self.charges.values():
            if other_charge.charge_id == charge_id:
                continue
                
            if not other_charge.metadata.field_position:
                continue
                
            ox, oy, oz = other_charge.metadata.field_position
            distance = np.sqrt((x - ox)**2 + (y - oy)**2 + (z - oz)**2)
            
            if distance <= max_distance:
                nearby.append(other_charge)
        
        return nearby
    
    def get_universe_metrics(self) -> UniverseMetrics:
        """Get universe-wide metrics."""
        if not self.charges:
            return UniverseMetrics(0, 0.0, 0.0, 0.0, time.time())
        
        magnitudes = [charge.magnitude for charge in self.charges.values()]
        total_energy = sum(magnitudes)
        average_magnitude = total_energy / len(magnitudes)
        
        # Field density (charges per region)
        field_density = len(self.charges) / max(1, len(self.field_regions))
        
        return UniverseMetrics(
            total_charges=len(self.charges),
            total_energy=total_energy,
            average_magnitude=average_magnitude,
            field_density=field_density,
            last_updated=time.time()
        )
    
    def compute_collective_response(self, charge_ids: List[str]) -> complex:
        """
        Compute collective response for a group of charges.
        
        Implements the collective response mathematics from product manifold theory.
        """
        if not charge_ids:
            return complex(0, 0)
        
        charges = [self.charges[cid] for cid in charge_ids if cid in self.charges]
        if not charges:
            return complex(0, 0)
        
        # Simple collective response - sum with interference
        total_response = complex(0, 0)
        for charge in charges:
            individual_response = charge.compute_collective_response()
            total_response += individual_response
        
        # Normalize by number of charges
        return total_response / len(charges)
    
    def get_field_evolution(self, hours_back: float = 24.0) -> List[Tuple[float, int, float]]:
        """Get universe evolution over time."""
        cutoff_time = time.time() - (hours_back * 3600)
        evolution = []
        
        # Collect all timestamps
        timestamps = set()
        for charge in self.charges.values():
            for state in charge.historical_states:
                if state['timestamp'] >= cutoff_time:
                    timestamps.add(state['timestamp'])
        
        # Calculate metrics at each timestamp
        for timestamp in sorted(timestamps):
            active_charges = []
            total_energy = 0.0
            
            for charge in self.charges.values():
                if charge.creation_timestamp <= timestamp:
                    active_charges.append(charge)
                    # Find closest historical state
                    closest_state = min(
                        charge.historical_states,
                        key=lambda s: abs(s['timestamp'] - timestamp)
                    )
                    total_energy += closest_state['complete_charge']['magnitude']
            
            evolution.append((timestamp, len(active_charges), total_energy))
        
        return evolution
    
    def __len__(self) -> int:
        """Number of charges in universe."""
        return len(self.charges)
    
    def __str__(self) -> str:
        """String representation."""
        metrics = self.get_universe_metrics()
        return (f"FieldUniverse(charges={metrics.total_charges}, "
                f"energy={metrics.total_energy:.3f}, "
                f"regions={len(self.field_regions)})")
    
    def __repr__(self) -> str:
        return self.__str__()