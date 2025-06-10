"""
Manifold Assembly Pipeline

Complete pipeline from conceptual charges to living sociology manifold.

Pipeline stages:
1. Charge Collection & Preprocessing
2. Manifold Initialization & Assembly  
3. Dynamic Evolution Processing
4. Collective Phenomena Analysis
5. Visualization & Export

Creates the "living map of sociology" from charge interactions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import time
import warnings
from pathlib import Path
import json

# Import core components
try:
    from ..product_manifold.product_manifold import ProductManifold, ManifoldConfiguration, create_sociology_manifold
    from ..core_mathematics.conceptual_charge import ConceptualCharge
    from ..embedding_engine.models import ConceptualChargeGenerator
except ImportError:
    warnings.warn("Some imports failed - using fallback types")
    ProductManifold = Any
    ConceptualCharge = Any
    ConceptualChargeGenerator = Any


@dataclass
class PipelineConfiguration:
    """Configuration for complete manifold assembly pipeline"""
    # Input processing
    batch_size: int = 50
    max_charges_per_manifold: int = 100
    charge_preprocessing: bool = True
    
    # Manifold assembly
    manifold_config: ManifoldConfiguration = field(default_factory=ManifoldConfiguration)
    auto_position_charges: bool = True
    position_strategy: str = 'semantic_clustering'  # 'random', 'grid', 'semantic_clustering'
    
    # Evolution processing
    auto_evolve: bool = True
    evolution_time: float = 2.0
    evolution_steps: int = 100
    adaptive_time_stepping: bool = True
    
    # Analysis configuration
    auto_analysis: bool = True
    multi_scale_analysis: bool = True
    temporal_analysis: bool = True
    export_intermediate_results: bool = False
    
    # Performance optimization
    parallel_processing: bool = True
    memory_optimization: bool = True
    cache_results: bool = True
    max_memory_gb: float = 4.0
    
    # Export configuration
    export_formats: List[str] = field(default_factory=lambda: ['numpy', 'json'])
    visualization_export: bool = True
    high_resolution_export: bool = False


@dataclass
class PipelineResult:
    """Complete result from manifold assembly pipeline"""
    # Core manifold
    manifold: ProductManifold
    
    # Processing metadata
    num_input_charges: int
    processing_time: float
    pipeline_config: PipelineConfiguration
    
    # Analysis results
    collective_phenomena: Dict[str, Any]
    sociology_map: Dict[str, Any]
    emergent_structures: Dict[str, Any]
    
    # Evolution data
    evolution_summary: Dict[str, Any]
    temporal_analysis: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class ManifoldAssemblyPipeline:
    """
    Complete pipeline for assembling conceptual charges into sociology manifolds.
    
    Orchestrates the full process:
    1. Charge preprocessing and optimization
    2. Intelligent charge positioning  
    3. Manifold assembly and initialization
    4. Dynamic evolution processing
    5. Comprehensive analysis and export
    
    Designed for both interactive exploration and batch processing.
    """
    
    def __init__(self, 
                 config: Optional[PipelineConfiguration] = None,
                 charge_generator: Optional[ConceptualChargeGenerator] = None):
        """
        Initialize manifold assembly pipeline.
        
        Args:
            config: Pipeline configuration
            charge_generator: Conceptual charge generator for text processing
        """
        self.config = config or PipelineConfiguration()
        self.charge_generator = charge_generator
        
        # Pipeline state
        self.current_manifold: Optional[ProductManifold] = None
        self.processing_history: List[PipelineResult] = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_charges_processed': 0,
            'total_manifolds_created': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        
        # Result caching
        self.result_cache: Dict[str, Any] = {}
    
    def process_text_to_manifold(self, 
                                texts: List[str],
                                contexts: Optional[List[Dict[str, Any]]] = None,
                                observational_states: Optional[List[float]] = None,
                                gamma_values: Optional[List[float]] = None) -> PipelineResult:
        """
        Complete pipeline: texts → conceptual charges → sociology manifold.
        
        Args:
            texts: Input texts to convert to conceptual charges
            contexts: Context dictionaries for each text
            observational_states: Observational states for charges
            gamma_values: Field calibration values
            
        Returns:
            Complete pipeline result with manifold and analysis
        """
        start_time = time.time()
        
        # Stage 1: Generate conceptual charges from texts
        if self.charge_generator is None:
            # Initialize default charge generator
            from ..embedding_engine.models import ConceptualChargeGenerator
            self.charge_generator = ConceptualChargeGenerator()
        
        charges = self._generate_charges_from_texts(
            texts, contexts, observational_states, gamma_values
        )
        
        # Stage 2: Process charges through manifold assembly
        result = self.process_charges_to_manifold(charges)
        
        # Update processing metadata
        result.processing_time = time.time() - start_time
        result.num_input_charges = len(texts)
        
        return result
    
    def process_charges_to_manifold(self, charges: List[ConceptualCharge]) -> PipelineResult:
        """
        Core pipeline: conceptual charges → sociology manifold.
        
        Args:
            charges: List of conceptual charges
            
        Returns:
            Complete pipeline result
        """
        start_time = time.time()
        
        # Stage 1: Charge preprocessing and optimization
        processed_charges = self._preprocess_charges(charges)
        
        # Stage 2: Intelligent charge positioning
        charge_positions = self._compute_charge_positions(processed_charges)
        
        # Stage 3: Manifold assembly and initialization
        manifold = self._assemble_manifold(processed_charges, charge_positions)
        
        # Stage 4: Dynamic evolution processing
        evolution_summary = self._process_manifold_evolution(manifold)
        
        # Stage 5: Comprehensive analysis
        analysis_results = self._perform_comprehensive_analysis(manifold)
        
        # Stage 6: Create sociology map
        sociology_map = manifold.create_sociology_map()
        
        # Compile complete result
        result = PipelineResult(
            manifold=manifold,
            num_input_charges=len(charges),
            processing_time=time.time() - start_time,
            pipeline_config=self.config,
            collective_phenomena=analysis_results['collective_phenomena'],
            sociology_map=sociology_map,
            emergent_structures=analysis_results['emergent_structures'],
            evolution_summary=evolution_summary,
            temporal_analysis=analysis_results.get('temporal_analysis'),
            performance_metrics=analysis_results['performance_metrics']
        )
        
        # Update pipeline state
        self.current_manifold = manifold
        self.processing_history.append(result)
        self._update_performance_metrics(result)
        
        return result
    
    def _generate_charges_from_texts(self, 
                                   texts: List[str],
                                   contexts: Optional[List[Dict[str, Any]]],
                                   observational_states: Optional[List[float]],
                                   gamma_values: Optional[List[float]]) -> List[ConceptualCharge]:
        """Generate conceptual charges from input texts"""
        charges = []
        
        # Default values
        if contexts is None:
            contexts = [{}] * len(texts)
        if observational_states is None:
            observational_states = [1.0] * len(texts)
        if gamma_values is None:
            gamma_values = [1.0] * len(texts)
        
        # Process in batches for memory efficiency
        batch_size = self.config.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_contexts = contexts[i:i+batch_size]
            batch_obs_states = observational_states[i:i+batch_size]
            batch_gammas = gamma_values[i:i+batch_size]
            
            # Generate charges for batch
            for text, context, obs_state, gamma in zip(
                batch_texts, batch_contexts, batch_obs_states, batch_gammas
            ):
                charge = self.charge_generator.create_conceptual_charge(
                    text=text,
                    context=context,
                    observational_state=obs_state,
                    gamma=gamma
                )
                charges.append(charge)
        
        return charges
    
    def _preprocess_charges(self, charges: List[ConceptualCharge]) -> List[ConceptualCharge]:
        """Preprocess and optimize charges for manifold assembly"""
        if not self.config.charge_preprocessing:
            return charges
        
        processed_charges = []
        
        for charge in charges:
            # Normalize charge parameters if needed
            if hasattr(charge, 'normalize_field_parameters'):
                charge.normalize_field_parameters()
            
            # Filter charges with too low magnitude
            charge_magnitude = abs(charge.compute_complete_charge())
            if charge_magnitude > 1e-6:  # Threshold for numerical stability
                processed_charges.append(charge)
        
        # Limit number of charges for performance
        if len(processed_charges) > self.config.max_charges_per_manifold:
            # Keep charges with highest magnitudes
            charge_magnitudes = [abs(c.compute_complete_charge()) for c in processed_charges]
            sorted_indices = np.argsort(charge_magnitudes)[::-1]  # Descending order
            selected_indices = sorted_indices[:self.config.max_charges_per_manifold]
            processed_charges = [processed_charges[i] for i in selected_indices]
        
        return processed_charges
    
    def _compute_charge_positions(self, charges: List[ConceptualCharge]) -> List[Tuple[float, float]]:
        """Compute intelligent positions for charges based on semantic relationships"""
        if not self.config.auto_position_charges:
            # Use default positioning
            return self._generate_default_positions(len(charges))
        
        if self.config.position_strategy == 'semantic_clustering':
            return self._compute_semantic_clustering_positions(charges)
        elif self.config.position_strategy == 'grid':
            return self._compute_grid_positions(len(charges))
        else:  # 'random'
            return self._generate_default_positions(len(charges))
    
    def _compute_semantic_clustering_positions(self, charges: List[ConceptualCharge]) -> List[Tuple[float, float]]:
        """Position charges based on semantic similarity clustering"""
        if len(charges) <= 1:
            return [(0.0, 0.0)] * len(charges)
        
        # Extract semantic vectors
        semantic_vectors = np.array([charge.semantic_vector for charge in charges])
        
        # Compute similarity matrix
        similarities = np.dot(semantic_vectors, semantic_vectors.T)
        similarities /= (np.linalg.norm(semantic_vectors, axis=1)[:, None] * 
                        np.linalg.norm(semantic_vectors, axis=1)[None, :])
        
        # Convert to distance matrix
        distances = 1.0 - similarities
        np.fill_diagonal(distances, 0.0)
        
        # Multidimensional scaling to 2D positions
        try:
            from sklearn.manifold import MDS
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            positions_2d = mds.fit_transform(distances)
            
            # Scale to fit manifold spatial extent
            Lx, Ly = self.config.manifold_config.spatial_extent
            scale_factor = min(Lx, Ly) * 0.4  # Use 40% of domain
            
            # Center and scale positions
            positions_2d -= np.mean(positions_2d, axis=0)
            positions_2d *= scale_factor / np.max(np.abs(positions_2d))
            
            return [(float(x), float(y)) for x, y in positions_2d]
            
        except ImportError:
            warnings.warn("sklearn not available, falling back to default positioning")
            return self._generate_default_positions(len(charges))
    
    def _compute_grid_positions(self, num_charges: int) -> List[Tuple[float, float]]:
        """Position charges in regular grid pattern"""
        if num_charges <= 1:
            return [(0.0, 0.0)] * num_charges
        
        # Determine grid dimensions
        grid_side = int(np.ceil(np.sqrt(num_charges)))
        
        Lx, Ly = self.config.manifold_config.spatial_extent
        spacing_x = Lx * 0.8 / grid_side
        spacing_y = Ly * 0.8 / grid_side
        
        positions = []
        for i in range(num_charges):
            grid_x = i % grid_side
            grid_y = i // grid_side
            
            x = (grid_x - grid_side/2 + 0.5) * spacing_x
            y = (grid_y - grid_side/2 + 0.5) * spacing_y
            
            positions.append((x, y))
        
        return positions
    
    def _generate_default_positions(self, num_charges: int) -> List[Tuple[float, float]]:
        """Generate default charge positions"""
        Lx, Ly = self.config.manifold_config.spatial_extent
        
        if num_charges == 1:
            return [(0.0, 0.0)]
        elif num_charges <= 6:
            # Symmetric arrangement
            positions = []
            for i in range(num_charges):
                angle = 2 * np.pi * i / num_charges
                radius = min(Lx, Ly) / 4
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions.append((x, y))
            return positions
        else:
            # Random distribution
            margin = 0.2
            positions = []
            for _ in range(num_charges):
                x = np.random.uniform(-Lx/2 + margin*Lx, Lx/2 - margin*Lx)
                y = np.random.uniform(-Ly/2 + margin*Ly, Ly/2 - margin*Ly)
                positions.append((x, y))
            return positions
    
    def _assemble_manifold(self, 
                          charges: List[ConceptualCharge], 
                          positions: List[Tuple[float, float]]) -> ProductManifold:
        """Assemble product manifold from charges and positions"""
        # Create manifold with configuration
        manifold = ProductManifold(self.config.manifold_config)
        
        # Add charges to manifold
        manifold.add_conceptual_charges(charges, positions, replace_existing=True)
        
        return manifold
    
    def _process_manifold_evolution(self, manifold: ProductManifold) -> Dict[str, Any]:
        """Process manifold evolution according to configuration"""
        if not self.config.auto_evolve:
            return {'message': 'Auto evolution disabled'}
        
        # Determine evolution parameters
        evolution_time = self.config.evolution_time
        dt = evolution_time / self.config.evolution_steps
        
        # Adaptive time stepping if enabled
        if self.config.adaptive_time_stepping:
            method = 'RK45'
        else:
            method = 'euler'
        
        # Evolve manifold
        evolution_summary = manifold.evolve_manifold(
            evolution_time=evolution_time,
            dt=dt,
            method=method,
            record_history=True
        )
        
        return evolution_summary
    
    def _perform_comprehensive_analysis(self, manifold: ProductManifold) -> Dict[str, Any]:
        """Perform comprehensive analysis of manifold"""
        analysis_start = time.time()
        
        # Core collective phenomena analysis
        collective_phenomena = manifold.compute_collective_phenomena()
        
        # Temporal analysis if requested and history available
        temporal_analysis = None
        if self.config.temporal_analysis and len(manifold.manifold_history) > 1:
            temporal_analysis = self._perform_temporal_analysis(manifold)
        
        # Emergent structure analysis
        emergent_structures = collective_phenomena['emergent_structures']
        
        # Performance metrics
        analysis_time = time.time() - analysis_start
        performance_metrics = {
            'analysis_computation_time': analysis_time,
            'manifold_summary': manifold.get_manifold_summary()
        }
        
        return {
            'collective_phenomena': collective_phenomena,
            'temporal_analysis': temporal_analysis,
            'emergent_structures': emergent_structures,
            'performance_metrics': performance_metrics
        }
    
    def _perform_temporal_analysis(self, manifold: ProductManifold) -> Dict[str, Any]:
        """Analyze temporal evolution patterns"""
        history = manifold.manifold_history
        
        if len(history) < 2:
            return {'message': 'Insufficient history for temporal analysis'}
        
        # Extract time series data
        times = [state.time for state in history]
        energies = [state.evolution_state.energy for state in history]
        peak_magnitudes = [state.evolution_state.peak_magnitude for state in history]
        
        # Analyze trends
        energy_trend = np.polyfit(times, energies, 1)[0] if len(times) > 1 else 0.0
        magnitude_trend = np.polyfit(times, peak_magnitudes, 1)[0] if len(times) > 1 else 0.0
        
        # Stability analysis
        recent_energies = energies[-10:] if len(energies) >= 10 else energies
        energy_stability = np.std(recent_energies) / np.mean(recent_energies) if np.mean(recent_energies) > 0 else 0.0
        
        return {
            'evolution_duration': times[-1] - times[0],
            'energy_trend': energy_trend,
            'magnitude_trend': magnitude_trend,
            'energy_stability': energy_stability,
            'num_evolution_states': len(history),
            'times': times,
            'energies': energies,
            'peak_magnitudes': peak_magnitudes
        }
    
    def _update_performance_metrics(self, result: PipelineResult):
        """Update pipeline performance metrics"""
        self.performance_metrics['total_charges_processed'] += result.num_input_charges
        self.performance_metrics['total_manifolds_created'] += 1
        self.performance_metrics['total_processing_time'] += result.processing_time
        
        # Compute average
        if self.performance_metrics['total_manifolds_created'] > 0:
            self.performance_metrics['average_processing_time'] = (
                self.performance_metrics['total_processing_time'] / 
                self.performance_metrics['total_manifolds_created']
            )
    
    def export_results(self, 
                      result: PipelineResult,
                      output_directory: Union[str, Path],
                      filename_prefix: str = 'sociology_manifold') -> Dict[str, str]:
        """
        Export pipeline results in multiple formats.
        
        Args:
            result: Pipeline result to export
            output_directory: Directory for output files
            filename_prefix: Prefix for output filenames
            
        Returns:
            Dictionary mapping format names to output file paths
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export manifold field data
        if 'numpy' in self.config.export_formats:
            field_file = output_dir / f"{filename_prefix}_field.npz"
            np.savez_compressed(
                field_file,
                field=result.manifold.manifold_equation.field,
                coordinates_X=result.manifold.manifold_equation.X,
                coordinates_Y=result.manifold.manifold_equation.Y,
                time=result.manifold.manifold_equation.time
            )
            exported_files['numpy_field'] = str(field_file)
        
        # Export analysis results as JSON
        if 'json' in self.config.export_formats:
            # Prepare JSON-serializable analysis data
            analysis_data = {
                'processing_metadata': {
                    'num_input_charges': result.num_input_charges,
                    'processing_time': result.processing_time,
                    'pipeline_config': {
                        'batch_size': self.config.batch_size,
                        'evolution_time': self.config.evolution_time,
                        'position_strategy': self.config.position_strategy
                    }
                },
                'collective_phenomena': self._make_json_serializable(result.collective_phenomena),
                'emergent_structures': self._make_json_serializable(result.emergent_structures),
                'evolution_summary': self._make_json_serializable(result.evolution_summary)
            }
            
            json_file = output_dir / f"{filename_prefix}_analysis.json"
            with open(json_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            exported_files['json_analysis'] = str(json_file)
        
        # Export sociology map for visualization
        if self.config.visualization_export:
            sociology_map = result.sociology_map
            
            # Save visualization data
            viz_file = output_dir / f"{filename_prefix}_sociology_map.npz"
            np.savez_compressed(
                viz_file,
                field_magnitude=sociology_map['field_layers']['magnitude'],
                field_phase=sociology_map['field_layers']['phase'],
                collective_response=sociology_map['field_layers']['collective_response'],
                coordinates_X=sociology_map['coordinates']['X'],
                coordinates_Y=sociology_map['coordinates']['Y'],
                charge_positions=np.array(sociology_map['charge_data']['positions']),
                charge_magnitudes=np.array(sociology_map['charge_data']['magnitudes'])
            )
            exported_files['visualization'] = str(viz_file)
        
        return exported_files
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and complex numbers to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.complex128) or isinstance(obj, complex):
            return {'real': float(obj.real), 'imag': float(obj.imag)}
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline performance and history"""
        return {
            'performance_metrics': self.performance_metrics,
            'num_processed_manifolds': len(self.processing_history),
            'current_manifold_active': self.current_manifold is not None,
            'configuration': {
                'batch_size': self.config.batch_size,
                'max_charges_per_manifold': self.config.max_charges_per_manifold,
                'auto_evolve': self.config.auto_evolve,
                'position_strategy': self.config.position_strategy
            }
        }


# Convenience functions for quick pipeline usage

def create_manifold_from_texts(texts: List[str],
                             grid_size: int = 64,
                             evolution_time: float = 1.0,
                             position_strategy: str = 'semantic_clustering') -> PipelineResult:
    """
    Convenience function: texts → sociology manifold (complete pipeline).
    
    Args:
        texts: Input texts to process
        grid_size: Spatial resolution
        evolution_time: Evolution time
        position_strategy: Charge positioning strategy
        
    Returns:
        Complete pipeline result
    """
    # Configure pipeline
    config = PipelineConfiguration(
        manifold_config=ManifoldConfiguration(
            spatial_dimensions=(grid_size, grid_size)
        ),
        evolution_time=evolution_time,
        position_strategy=position_strategy
    )
    
    # Create and run pipeline
    pipeline = ManifoldAssemblyPipeline(config)
    return pipeline.process_text_to_manifold(texts)


def create_manifold_from_charges(charges: List[ConceptualCharge],
                                grid_size: int = 64,
                                evolution_time: float = 1.0) -> PipelineResult:
    """
    Convenience function: charges → sociology manifold.
    
    Args:
        charges: Conceptual charges
        grid_size: Spatial resolution  
        evolution_time: Evolution time
        
    Returns:
        Complete pipeline result
    """
    config = PipelineConfiguration(
        manifold_config=ManifoldConfiguration(
            spatial_dimensions=(grid_size, grid_size)
        ),
        evolution_time=evolution_time
    )
    
    pipeline = ManifoldAssemblyPipeline(config)
    return pipeline.process_charges_to_manifold(charges)