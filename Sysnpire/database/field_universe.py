"""
Field Universe - Database layer for field-theoretic storage

Enterprise-grade storage system for conceptual charges with field-adjacent
placement using mathematical formulation from the paper.
"""

import numpy as np
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sqlite3
from dataclasses import dataclass

from .conceptual_charge_object import ConceptualChargeObject, FieldComponents

@dataclass
class UniverseMetrics:
    """Universe-wide metrics and statistics."""
    total_charges: int
    total_energy: float
    average_magnitude: float
    field_density: float
    last_updated: float

class FieldUniverse:
    """
    Field-theoretic universe for storing and managing conceptual charges.
    
    Implements field-adjacent placement using the mathematical formulation
    and provides enterprise-grade storage with queries and analytics.
    """
    
    def __init__(self, storage_path: str = "universe.db"):
        """
        Initialize the field universe.
        
        Args:
            storage_path: Path to SQLite database file
        """
        self.storage_path = Path(storage_path)
        self.charges: Dict[str, ConceptualChargeObject] = {}
        self.field_regions: Dict[str, List[str]] = {}
        
        # Initialize database
        self._init_database()
        
        # Load existing charges
        self._load_charges()
        
        print(f"🌌 Field Universe initialized")
        print(f"   Storage: {self.storage_path}")
        print(f"   Loaded charges: {len(self.charges)}")
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            
            # Charges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS charges (
                    charge_id TEXT PRIMARY KEY,
                    text_source TEXT NOT NULL,
                    creation_timestamp REAL NOT NULL,
                    last_updated REAL NOT NULL,
                    complete_charge_real REAL NOT NULL,
                    complete_charge_imag REAL NOT NULL,
                    magnitude REAL NOT NULL,
                    phase REAL NOT NULL,
                    observational_state REAL NOT NULL,
                    gamma REAL NOT NULL,
                    field_position_x REAL,
                    field_position_y REAL,
                    field_position_z REAL,
                    field_region TEXT,
                    data_json TEXT NOT NULL
                )
            """)
            
            # Field regions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS field_regions (
                    region_id TEXT PRIMARY KEY,
                    center_x REAL NOT NULL,
                    center_y REAL NOT NULL,
                    center_z REAL NOT NULL,
                    radius REAL NOT NULL,
                    charge_count INTEGER DEFAULT 0,
                    total_energy REAL DEFAULT 0.0,
                    created_timestamp REAL NOT NULL
                )
            """)
            
            # Charge relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS charge_relationships (
                    charge_id TEXT NOT NULL,
                    nearby_charge_id TEXT NOT NULL,
                    influence_strength REAL NOT NULL,
                    distance REAL,
                    PRIMARY KEY (charge_id, nearby_charge_id),
                    FOREIGN KEY (charge_id) REFERENCES charges(charge_id),
                    FOREIGN KEY (nearby_charge_id) REFERENCES charges(charge_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_charges_magnitude ON charges(magnitude)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_charges_field_region ON charges(field_region)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_charges_timestamp ON charges(creation_timestamp)")
            
            conn.commit()
    
    def add_charge(self, charge_obj: ConceptualChargeObject) -> bool:
        """
        Add a conceptual charge to the universe with field placement.
        
        Args:
            charge_obj: Rich conceptual charge object
            
        Returns:
            True if successfully added
        """
        try:
            # Calculate field position based on field theory
            field_position = self._calculate_field_position(charge_obj)
            charge_obj.set_field_position(field_position)
            
            # Determine field region
            field_region = self._determine_field_region(field_position)
            charge_obj.metadata.field_region = field_region
            
            # Find nearby charges and establish relationships
            self._establish_field_relationships(charge_obj)
            
            # Store in memory
            self.charges[charge_obj.charge_id] = charge_obj
            
            # Update field region
            if field_region not in self.field_regions:
                self.field_regions[field_region] = []
            self.field_regions[field_region].append(charge_obj.charge_id)
            
            # Persist to database
            self._save_charge_to_db(charge_obj)
            
            print(f"   ✅ Added charge {charge_obj.charge_id} to region {field_region}")
            print(f"      Position: {field_position}")
            print(f"      Nearby charges: {len(charge_obj.metadata.nearby_charges)}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to add charge {charge_obj.charge_id}: {e}")
            return False
    
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