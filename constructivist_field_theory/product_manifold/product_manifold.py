"""
Product Manifold: Complete Integration

Main class orchestrating the complete product manifold framework.

Integrates:
- Conceptual charges → geometric imprints (TransformationOperator)  
- Manifold field evolution (ManifoldFieldEquation)
- Collective response computation (CollectiveResponseFunction)
- Dynamic visualization and analysis

Creates the "living map of sociology" from conceptual charge interactions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import time
import warnings

# Import core components
try:
    from .transformation_operator import TransformationOperator, TransformationParameters
    from .manifold_field_equation import ManifoldFieldEquation, ManifoldParameters, EvolutionState
    from .collective_response import CollectiveResponseFunction, CollectiveResponseParameters
    from ..core_mathematics.conceptual_charge import ConceptualCharge
except ImportError:
    # Fallback for development
    warnings.warn("Some imports failed - using fallback types")
    TransformationOperator = Any
    ManifoldFieldEquation = Any
    CollectiveResponseFunction = Any
    ConceptualCharge = Any


@dataclass
class ManifoldConfiguration:
    """Complete configuration for product manifold"""
    # Spatial configuration
    spatial_dimensions: Tuple[int, int] = (64, 64)
    spatial_extent: Tuple[float, float] = (10.0, 10.0)
    
    # Component parameters
    transformation_params: TransformationParameters = field(default_factory=TransformationParameters)
    manifold_params: ManifoldParameters = field(default_factory=ManifoldParameters)  
    response_params: CollectiveResponseParameters = field(default_factory=CollectiveResponseParameters)
    
    # Evolution parameters
    default_dt: float = 0.01
    default_evolution_method: str = 'RK45'
    max_evolution_time: float = 10.0
    
    # Analysis parameters
    auto_analysis: bool = True
    history_length: int = 1000


@dataclass
class ManifoldState:
    """Complete state of product manifold at a given time"""
    time: float
    field: np.ndarray
    charges: List[ConceptualCharge]
    charge_positions: List[Tuple[float, float]]
    evolution_state: EvolutionState
    collective_response: Dict[str, Any]
    field_properties: Dict[str, Any]


class ProductManifold:
    """
    Complete product manifold implementation.
    
    Creates the geometric space where conceptual charges collectively operate,
    forming a "living map of sociology" through field-theoretic dynamics.
    
    Key capabilities:
    1. Assemble conceptual charges into geometric manifold
    2. Evolve manifold through field equation dynamics
    3. Compute observable collective response patterns
    4. Analyze emergent sociological structures
    5. Provide interactive visualization and exploration
    """
    
    def __init__(self, config: Optional[ManifoldConfiguration] = None):
        """
        Initialize complete product manifold system.
        
        Args:
            config: Manifold configuration parameters
        """
        self.config = config or ManifoldConfiguration()
        
        # Initialize core components
        self._initialize_components()
        
        # Current state
        self.current_charges: List[ConceptualCharge] = []
        self.current_positions: List[Tuple[float, float]] = []
        
        # Evolution and analysis history
        self.manifold_history: List[ManifoldState] = []
        self.analysis_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {
            'evolution_times': [],
            'response_computation_times': [],
            'total_computation_times': []
        }
    
    def _initialize_components(self):
        """Initialize all mathematical components"""
        # Transformation operator: Q → geometric imprints
        self.transformation_operator = TransformationOperator(
            spatial_dimensions=self.config.spatial_dimensions,
            spatial_extent=self.config.spatial_extent,
            params=self.config.transformation_params
        )
        
        # Manifold field equation: ∂M/∂t = ∇²M + F[M] + Σᵢ T[Qᵢ]
        self.manifold_equation = ManifoldFieldEquation(
            spatial_dimensions=self.config.spatial_dimensions,
            spatial_extent=self.config.spatial_extent,
            transformation_operator=self.transformation_operator,
            params=self.config.manifold_params
        )
        
        # Collective response: R_collective = M · G[M] · M†
        self.collective_response_function = CollectiveResponseFunction(
            manifold_equation=self.manifold_equation,
            params=self.config.response_params
        )
    
    def add_conceptual_charges(self, 
                             charges: List[ConceptualCharge],
                             positions: Optional[List[Tuple[float, float]]] = None,
                             replace_existing: bool = False) -> Dict[str, Any]:
        """
        Add conceptual charges to the manifold.
        
        Args:
            charges: List of conceptual charges to add
            positions: Positions for charge placement (default: auto-distribute)
            replace_existing: Whether to replace existing charges
            
        Returns:
            Addition operation summary
        """
        if replace_existing:
            self.current_charges = []
            self.current_positions = []
        
        # Auto-generate positions if not provided
        if positions is None:
            positions = self._generate_charge_positions(len(charges))
        elif len(positions) != len(charges):
            raise ValueError(f"Number of positions ({len(positions)}) must match number of charges ({len(charges)})")
        
        # Add charges and positions
        self.current_charges.extend(charges)
        self.current_positions.extend(positions)
        
        # Update manifold with new charge configuration
        self._update_manifold_from_charges()
        
        return {
            'num_charges_added': len(charges),
            'total_charges': len(self.current_charges),
            'charge_positions': positions,
            'manifold_updated': True
        }
    
    def _generate_charge_positions(self, num_charges: int) -> List[Tuple[float, float]]:
        """Generate well-distributed positions for charges"""
        Lx, Ly = self.config.spatial_extent
        
        if num_charges == 1:
            # Single charge at center
            return [(0.0, 0.0)]
        
        elif num_charges <= 4:
            # Small number: arrange in symmetric pattern
            positions = []
            for i in range(num_charges):
                angle = 2 * np.pi * i / num_charges
                radius = min(Lx, Ly) / 4
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions.append((x, y))
            return positions
        
        else:
            # Larger number: quasi-random distribution avoiding boundaries
            margin = 0.2
            positions = []
            for _ in range(num_charges):
                x = np.random.uniform(-Lx/2 + margin*Lx, Lx/2 - margin*Lx)
                y = np.random.uniform(-Ly/2 + margin*Ly, Ly/2 - margin*Ly)
                positions.append((x, y))
            return positions
    
    def _update_manifold_from_charges(self):
        """Update manifold field based on current charges"""
        if not self.current_charges:
            # No charges: initialize with small random field
            self.manifold_equation.initialize_field('random', amplitude=0.1, noise_level=0.05)
        else:
            # Compute initial imprint from charges
            initial_imprint = self.transformation_operator.batch_transform(
                self.current_charges, self.current_positions
            )
            
            # Set as initial field (add small diffusion to avoid singularities)
            self.manifold_equation.field = initial_imprint.astype(np.complex128)
            self.manifold_equation.time = 0.0
    
    def evolve_manifold(self, 
                       evolution_time: float,
                       dt: Optional[float] = None,
                       method: Optional[str] = None,
                       record_history: bool = True) -> Dict[str, Any]:
        """
        Evolve the manifold through field equation dynamics.
        
        Args:
            evolution_time: Total time to evolve
            dt: Time step size (default: config.default_dt)
            method: Integration method (default: config.default_evolution_method)
            record_history: Whether to record evolution history
            
        Returns:
            Evolution summary and final state
        """
        dt = dt or self.config.default_dt
        method = method or self.config.default_evolution_method
        
        start_time = time.time()
        
        # Evolution parameters
        num_steps = int(evolution_time / dt)
        evolution_summary = {
            'evolution_time': evolution_time,
            'num_steps': num_steps,
            'dt': dt,
            'method': method
        }
        
        # Record initial state if requested
        if record_history:
            initial_state = self._capture_current_state()
            self.manifold_history.append(initial_state)
        
        # Evolution loop
        evolution_states = []
        
        for step in range(num_steps):
            # Evolve one time step
            evolution_state = self.manifold_equation.evolve_field(
                dt, self.current_charges, self.current_positions, method=method
            )
            
            evolution_states.append(evolution_state)
            
            # Record history at regular intervals
            if record_history and (step % max(1, num_steps // 50) == 0):
                current_state = self._capture_current_state()
                self.manifold_history.append(current_state)
        
        # Final state analysis
        final_field_properties = self.manifold_equation.analyze_field_properties()
        final_collective_response = self.collective_response_function.compute_collective_response()
        
        # Performance tracking
        computation_time = time.time() - start_time
        self.performance_metrics['evolution_times'].append(computation_time)
        
        evolution_summary.update({
            'computation_time': computation_time,
            'final_energy': evolution_states[-1].energy if evolution_states else 0.0,
            'energy_change': (evolution_states[-1].energy - evolution_states[0].energy) if len(evolution_states) > 1 else 0.0,
            'final_field_properties': final_field_properties,
            'final_collective_response': final_collective_response,
            'evolution_states': evolution_states
        })
        
        return evolution_summary
    
    def _capture_current_state(self) -> ManifoldState:
        """Capture complete current state of manifold"""
        # Compute current analysis
        evolution_state = self.manifold_equation._compute_evolution_state()
        collective_response = self.collective_response_function.compute_collective_response()
        field_properties = self.manifold_equation.analyze_field_properties()
        
        return ManifoldState(
            time=self.manifold_equation.time,
            field=self.manifold_equation.field.copy(),
            charges=self.current_charges.copy(),
            charge_positions=self.current_positions.copy(),
            evolution_state=evolution_state,
            collective_response=collective_response,
            field_properties=field_properties
        )
    
    def compute_collective_phenomena(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of collective phenomena in the manifold.
        
        Returns:
            Complete collective phenomena analysis
        """
        start_time = time.time()
        
        # Core collective response
        collective_response = self.collective_response_function.compute_collective_response()
        
        # Multi-scale analysis
        multi_scale_response = self.collective_response_function.multi_scale_response_analysis()
        
        # Field structure analysis
        field_properties = self.manifold_equation.analyze_field_properties()
        
        # Charge interaction analysis
        charge_interaction_analysis = self._analyze_charge_interactions()
        
        # Emergent structure detection
        emergent_structures = self._detect_emergent_structures(field_properties, collective_response)
        
        # Performance tracking
        computation_time = time.time() - start_time
        self.performance_metrics['response_computation_times'].append(computation_time)
        
        return {
            'collective_response': collective_response,
            'multi_scale_response': multi_scale_response,
            'field_properties': field_properties,
            'charge_interactions': charge_interaction_analysis,
            'emergent_structures': emergent_structures,
            'computation_time': computation_time,
            'analysis_timestamp': time.time()
        }
    
    def _analyze_charge_interactions(self) -> Dict[str, Any]:
        """Analyze interactions between conceptual charges"""
        if len(self.current_charges) < 2:
            return {'message': 'Need at least 2 charges for interaction analysis'}
        
        # Compute pairwise interactions
        charge_similarities = []
        spatial_distances = []
        
        for i in range(len(self.current_charges)):
            for j in range(i + 1, len(self.current_charges)):
                # Semantic similarity (dot product of semantic vectors)
                sim = np.dot(
                    self.current_charges[i].semantic_vector,
                    self.current_charges[j].semantic_vector
                ) / (np.linalg.norm(self.current_charges[i].semantic_vector) * 
                     np.linalg.norm(self.current_charges[j].semantic_vector))
                charge_similarities.append(sim)
                
                # Spatial distance
                x1, y1 = self.current_positions[i]
                x2, y2 = self.current_positions[j]
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                spatial_distances.append(dist)
        
        # Interference pattern analysis
        interference_analysis = self.transformation_operator.compute_interference_pattern(
            self.current_charges, self.current_positions
        )
        
        return {
            'charge_similarities': charge_similarities,
            'spatial_distances': spatial_distances,
            'mean_similarity': np.mean(charge_similarities),
            'similarity_std': np.std(charge_similarities),
            'mean_spatial_distance': np.mean(spatial_distances),
            'interference_analysis': interference_analysis
        }
    
    def _detect_emergent_structures(self, 
                                   field_properties: Dict[str, Any],
                                   collective_response: Dict[str, Any]) -> Dict[str, Any]:
        """Detect emergent collective structures in the manifold"""
        field_magnitude = field_properties['field_magnitude']
        response_field = collective_response['collective_response_field']
        
        # Stable structure detection (local maxima)
        from scipy.ndimage import maximum_filter
        
        # Field-based structures
        field_local_maxima = (field_magnitude == maximum_filter(field_magnitude, size=3))
        field_threshold = 0.2 * field_properties['peak_magnitude']
        field_structures = field_local_maxima & (field_magnitude > field_threshold)
        
        # Response-based structures
        response_local_maxima = (response_field == maximum_filter(response_field, size=3))
        response_threshold = 0.2 * collective_response['peak_response']
        response_structures = response_local_maxima & (response_field > response_threshold)
        
        # Coherent structures (overlap between field and response)
        coherent_structures = field_structures & response_structures
        
        # Structure statistics
        num_field_structures = np.sum(field_structures)
        num_response_structures = np.sum(response_structures)
        num_coherent_structures = np.sum(coherent_structures)
        
        # Structure persistence (if history available)
        structure_persistence = self._analyze_structure_persistence(coherent_structures)
        
        return {
            'field_structures': field_structures,
            'response_structures': response_structures,
            'coherent_structures': coherent_structures,
            'num_field_structures': num_field_structures,
            'num_response_structures': num_response_structures,
            'num_coherent_structures': num_coherent_structures,
            'structure_coherence_ratio': num_coherent_structures / max(1, num_field_structures),
            'structure_persistence': structure_persistence
        }
    
    def _analyze_structure_persistence(self, current_structures: np.ndarray) -> Dict[str, Any]:
        """Analyze how stable structures persist over time"""
        if len(self.manifold_history) < 2:
            return {'message': 'Insufficient history for persistence analysis'}
        
        # Compare with previous states
        recent_states = self.manifold_history[-5:]  # Last 5 states
        structure_overlaps = []
        
        for state in recent_states:
            # Detect structures in historical state
            field_mag = np.abs(state.field)
            local_maxima = (field_mag == maximum_filter(field_mag, size=3))
            threshold = 0.2 * np.max(field_mag)
            historical_structures = local_maxima & (field_mag > threshold)
            
            # Compute overlap with current structures
            overlap = np.sum(current_structures & historical_structures) / max(1, np.sum(current_structures | historical_structures))
            structure_overlaps.append(overlap)
        
        return {
            'structure_overlaps': structure_overlaps,
            'mean_persistence': np.mean(structure_overlaps),
            'persistence_trend': np.polyfit(range(len(structure_overlaps)), structure_overlaps, 1)[0] if len(structure_overlaps) > 1 else 0.0
        }
    
    def create_sociology_map(self) -> Dict[str, Any]:
        """
        Generate the complete "living map of sociology" visualization data.
        
        Returns:
            Comprehensive data for visualizing sociological field dynamics
        """
        # Ensure we have current analysis
        collective_phenomena = self.compute_collective_phenomena()
        
        # Spatial grids for visualization
        field_magnitude = np.abs(self.manifold_equation.field)
        field_phase = np.angle(self.manifold_equation.field)
        response_field = collective_phenomena['collective_response']['collective_response_field']
        
        # Create layered visualization data
        sociology_map = {
            # Core field visualization
            'field_layers': {
                'magnitude': field_magnitude,
                'phase': field_phase,
                'collective_response': response_field,
                'constructive_regions': collective_phenomena['collective_response']['constructive_regions'],
                'destructive_regions': collective_phenomena['collective_response']['destructive_regions']
            },
            
            # Charge visualization
            'charge_data': {
                'positions': self.current_positions,
                'magnitudes': [abs(charge.compute_complete_charge()) for charge in self.current_charges],
                'phases': [np.angle(charge.compute_complete_charge()) for charge in self.current_charges],
                'tokens': [getattr(charge, 'token', f'charge_{i}') for i, charge in enumerate(self.current_charges)]
            },
            
            # Emergent structures
            'emergent_structures': collective_phenomena['emergent_structures'],
            
            # Spatial coordinates
            'coordinates': {
                'X': self.manifold_equation.X,
                'Y': self.manifold_equation.Y,
                'spatial_extent': self.config.spatial_extent
            },
            
            # Analysis metadata
            'analysis_metadata': {
                'time': self.manifold_equation.time,
                'total_energy': collective_phenomena['field_properties']['total_energy'],
                'num_charges': len(self.current_charges),
                'phase_coherence': collective_phenomena['collective_response']['phase_coherence'],
                'total_response': collective_phenomena['collective_response']['total_response']
            }
        }
        
        return sociology_map
    
    def get_manifold_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of current manifold state"""
        return {
            'configuration': {
                'spatial_dimensions': self.config.spatial_dimensions,
                'spatial_extent': self.config.spatial_extent,
                'num_charges': len(self.current_charges)
            },
            'current_state': {
                'time': self.manifold_equation.time,
                'field_energy': np.sum(np.abs(self.manifold_equation.field)**2),
                'field_peak': np.max(np.abs(self.manifold_equation.field)),
                'field_mean': np.mean(np.abs(self.manifold_equation.field))
            },
            'evolution_history': {
                'num_recorded_states': len(self.manifold_history),
                'evolution_duration': (self.manifold_history[-1].time - self.manifold_history[0].time) if len(self.manifold_history) > 1 else 0.0
            },
            'performance_metrics': {
                'total_evolution_time': sum(self.performance_metrics['evolution_times']),
                'total_response_time': sum(self.performance_metrics['response_computation_times']),
                'average_evolution_time': np.mean(self.performance_metrics['evolution_times']) if self.performance_metrics['evolution_times'] else 0.0
            }
        }


def create_sociology_manifold(charges: List[ConceptualCharge],
                            grid_size: int = 64,
                            spatial_size: float = 10.0,
                            evolution_time: float = 1.0) -> ProductManifold:
    """
    Create and initialize a complete sociology manifold from conceptual charges.
    
    Convenience function for quick manifold creation and basic evolution.
    
    Args:
        charges: List of conceptual charges
        grid_size: Spatial resolution
        spatial_size: Physical domain size
        evolution_time: Initial evolution time
        
    Returns:
        Initialized and evolved ProductManifold
    """
    # Configuration
    config = ManifoldConfiguration(
        spatial_dimensions=(grid_size, grid_size),
        spatial_extent=(spatial_size, spatial_size)
    )
    
    # Create manifold
    manifold = ProductManifold(config)
    
    # Add charges
    manifold.add_conceptual_charges(charges)
    
    # Initial evolution
    if evolution_time > 0:
        manifold.evolve_manifold(evolution_time)
    
    return manifold