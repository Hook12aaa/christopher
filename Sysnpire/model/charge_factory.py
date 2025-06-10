"""
Charge Factory - Main production interface for conceptual charges

Commercial-grade factory for creating conceptual charges from text inputs.
"""

import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from .bge_encoder import BGEEncoder
from .field_enhancer import FieldEnhancer

# Import universe components
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.conceptual_charge_object import ConceptualChargeObject, FieldComponents
from database.field_universe import FieldUniverse

class ChargeFactory:
    """
    Enterprise charge factory for producing rich conceptual charges.
    
    Integrates with field universe for persistent storage and field placement.
    """
    
    def __init__(self, universe: Optional[FieldUniverse] = None):
        """
        Initialize the enterprise charge production pipeline.
        
        Args:
            universe: Field universe for storage (creates new if None)
        """
        self.encoder = BGEEncoder()
        self.enhancer = FieldEnhancer()
        self.universe = universe or FieldUniverse("enterprise_universe.db")
        
        print("🏭 Enterprise Charge Factory initialized")
        print("   BGE Encoder: Ready")
        print("   Field Enhancer: Ready")
        print(f"   Universe: {self.universe}")
        print(f"   Storage: Connected to field universe database")
    
    def create_charge(self, text: str, 
                     observational_state: float = 1.0,
                     gamma: float = 1.0,
                     store_in_universe: bool = True) -> ConceptualChargeObject:
        """
        Create a rich conceptual charge from text with universe storage.
        
        Args:
            text: Input text
            observational_state: Trajectory position s
            gamma: Field calibration factor
            store_in_universe: Whether to store in field universe
            
        Returns:
            Rich conceptual charge object
        """
        start_time = time.time()
        
        # Step 1: Text to BGE embedding
        embedding = self.encoder.encode(text)
        
        # Step 2: BGE to conceptual charge (legacy)
        legacy_charge = self.enhancer.enhance_embedding(
            embedding, text, observational_state, gamma
        )
        
        # Step 3: Compute field values
        field_values = self.enhancer.compute_charge_values(legacy_charge)
        
        # Step 4: Create rich conceptual charge object
        charge_id = f"charge_{uuid.uuid4().hex[:8]}"
        
        field_components = FieldComponents(
            trajectory_operators=field_values['trajectory_operators'],
            emotional_trajectory=field_values['emotional_trajectory'],
            semantic_field=field_values['semantic_field'],
            phase_total=field_values['phase_total'],
            observational_persistence=field_values['persistence']
        )
        
        rich_charge = ConceptualChargeObject(
            charge_id=charge_id,
            text_source=text,
            complete_charge=field_values['complete_charge'],
            field_components=field_components,
            observational_state=observational_state,
            gamma=gamma
        )
        
        # Step 5: Store in field universe
        if store_in_universe:
            success = self.universe.add_charge(rich_charge)
            if not success:
                print(f"   ⚠️ Failed to store charge in universe")
        
        processing_time = time.time() - start_time
        print(f"   ⚡ Created charge {charge_id} in {processing_time:.3f}s")
        print(f"      Magnitude: {rich_charge.magnitude:.6f}")
        print(f"      Field position: {rich_charge.metadata.field_position}")
        print(f"      Nearby charges: {len(rich_charge.metadata.nearby_charges)}")
        
        return rich_charge
    
    def create_charges_batch(self, texts: List[str],
                           observational_states: Optional[List[float]] = None,
                           gamma_values: Optional[List[float]] = None,
                           store_in_universe: bool = True) -> List[ConceptualChargeObject]:
        """
        Create multiple rich conceptual charges from text list.
        
        Enterprise batch processing with universe storage.
        """
        if observational_states is None:
            observational_states = [1.0 + i * 0.2 for i in range(len(texts))]
        
        if gamma_values is None:
            gamma_values = [1.0] * len(texts)
        
        results = []
        print(f"🏭 Enterprise batch processing: {len(texts)} texts...")
        
        for i, (text, obs_state, gamma) in enumerate(zip(texts, observational_states, gamma_values)):
            print(f"\n  [{i+1}/{len(texts)}] Processing: '{text[:50]}...'")
            
            charge_obj = self.create_charge(text, obs_state, gamma, store_in_universe)
            results.append(charge_obj)
        
        print(f"\n✅ Enterprise batch processing complete: {len(results)} charges created")
        print(f"   Universe now contains: {len(self.universe)} total charges")
        print(f"   Field regions: {len(self.universe.field_regions)}")
        
        return results
    
    def get_universe_metrics(self) -> Dict[str, Any]:
        """Get comprehensive universe metrics."""
        metrics = self.universe.get_universe_metrics()
        
        analysis = {
            'universe_metrics': {
                'total_charges': metrics.total_charges,
                'total_energy': metrics.total_energy,
                'average_magnitude': metrics.average_magnitude,
                'field_density': metrics.field_density,
                'last_updated': metrics.last_updated
            },
            'field_regions': len(self.universe.field_regions),
            'storage_path': str(self.universe.storage_path),
            'timestamp': time.time()
        }
        
        print(f"📊 Enterprise Universe Analysis:")
        print(f"   Total charges: {metrics.total_charges}")
        print(f"   Total energy: {metrics.total_energy:.3f}")
        print(f"   Average magnitude: {metrics.average_magnitude:.6f}")
        print(f"   Field density: {metrics.field_density:.2f} charges/region")
        print(f"   Field regions: {len(self.universe.field_regions)}")
        print(f"   Database: {self.universe.storage_path}")
        
        return analysis
    
    def query_charges(self, 
                     magnitude_range: Optional[Tuple[float, float]] = None,
                     field_region: Optional[str] = None,
                     nearby_charge_id: Optional[str] = None,
                     max_distance: float = 50.0) -> List[ConceptualChargeObject]:
        """
        Query charges from universe with various filters.
        
        Args:
            magnitude_range: (min, max) magnitude filter
            field_region: Specific field region
            nearby_charge_id: Find charges near this one
            max_distance: Maximum distance for nearby search
            
        Returns:
            List of matching conceptual charges
        """
        if magnitude_range:
            return self.universe.query_charges_by_magnitude(magnitude_range[0], magnitude_range[1])
        elif field_region:
            return self.universe.query_charges_by_region(field_region)
        elif nearby_charge_id:
            return self.universe.query_nearby_charges(nearby_charge_id, max_distance)
        else:
            return list(self.universe.charges.values())
    
    def compute_collective_response(self, charge_ids: List[str]) -> complex:
        """Compute collective response for specific charges."""
        return self.universe.compute_collective_response(charge_ids)
    
    def get_field_evolution(self, hours_back: float = 24.0) -> List[Tuple[float, int, float]]:
        """Get universe evolution over time."""
        return self.universe.get_field_evolution(hours_back)