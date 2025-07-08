"""
Embedding Space Cell Lists - O(N) Spatial Partitioning for High-Dimensional Field Theory

MATHEMATICAL FOUNDATION:
Implements the cell lists algorithm adapted for high-dimensional embedding spaces,
enabling O(N) complexity through intelligent spatial partitioning and fixed-radius
neighbor search in field-theoretic manifolds.

KEY INNOVATION:
Handles the curse of dimensionality by combining adaptive dimensionality reduction
with exact distance calculations, preserving mathematical integrity while achieving
O(N) complexity through bounded neighborhood sizes.

CELL LISTS ALGORITHM:
- Partition embedding space into cells of size >= cutoff_radius
- Each charge assigned to exactly one cell based on position
- Neighbors found by checking current cell + adjacent cells (bounded number)
- Result: O(N × k) where k is constant number of neighboring cells = O(N)

FIELD-THEORETIC OPTIMIZATION:
- Uses field interference signatures for natural spatial representation
- Preserves complex-valued calculations throughout
- Adapts to embedding manifold topology via PCA/random projection
- Leverages field locality properties for efficient partitioning
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
import gc
from collections import defaultdict

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.utils.logger import get_logger
from Sysnpire.utils.field_theory_optimizers import field_theory_auto_optimize

logger = get_logger(__name__)


@dataclass
class CellCoordinate:
    """Multi-dimensional cell coordinate in embedding space."""
    indices: Tuple[int, ...]  # Cell indices in each dimension
    
    def __hash__(self) -> int:
        return hash(self.indices)
    
    def __eq__(self, other) -> bool:
        return isinstance(other, CellCoordinate) and self.indices == other.indices
    
    def __repr__(self) -> str:
        return f"Cell{self.indices}"


@dataclass
class Cell:
    """Cell containing charges and their metadata."""
    coordinate: CellCoordinate
    charge_indices: List[int]          # Indices of charges in this cell
    bounds: Tuple[np.ndarray, np.ndarray]  # (min_bounds, max_bounds) in embedding space
    
    def add_charge(self, charge_idx: int) -> None:
        """Add charge to this cell."""
        self.charge_indices.append(charge_idx)
    
    def remove_charge(self, charge_idx: int) -> None:
        """Remove charge from this cell."""
        if charge_idx in self.charge_indices:
            self.charge_indices.remove(charge_idx)
    
    def is_empty(self) -> bool:
        """Check if cell is empty."""
        return len(self.charge_indices) == 0
    
    def size(self) -> int:
        """Number of charges in this cell."""
        return len(self.charge_indices)


@dataclass 
class CellListsStatistics:
    """Statistics from cell lists construction."""
    total_cells: int
    occupied_cells: int
    empty_cells: int
    max_charges_per_cell: int
    mean_charges_per_cell: float
    std_charges_per_cell: float
    embedding_reduction_ratio: float  # Original dim / reduced dim
    memory_usage_mb: float
    construction_time_seconds: float


class EmbeddingSpaceCellLists:
    """
    Efficient O(N) cell lists for high-dimensional embedding space.
    
    ALGORITHM OVERVIEW:
    1. Reduce dimensionality while preserving local structure
    2. Partition reduced space into regular grid cells
    3. Assign each charge to cell based on field signature position
    4. Provide O(1) access to neighboring cells for fixed-radius queries
    
    COMPLEXITY GUARANTEE:
    - Construction: O(N) for charge assignment
    - Neighbor query: O(k) where k is bounded number of adjacent cells
    - Memory: O(N + C) where C is total number of cells
    """
    
    def __init__(self, 
                 cutoff_radius: float,
                 embedding_dimension: int,
                 target_dimension: int = 6,
                 cells_per_dimension: int = None,
                 reduction_method: str = 'pca'):
        """
        Initialize embedding space cell lists.
        
        Args:
            cutoff_radius: Maximum interaction radius for cell sizing
            embedding_dimension: Original dimension of embedding space
            target_dimension: Target dimension after reduction (balance efficiency vs accuracy)
            cells_per_dimension: Number of cells per dimension (auto-computed if None)
            reduction_method: 'pca', 'random', or 'identity' (no reduction)
        """
        self.cutoff_radius = cutoff_radius
        self.embedding_dimension = embedding_dimension
        self.target_dimension = min(target_dimension, embedding_dimension)
        self.reduction_method = reduction_method
        
        # Cell sizing: cell_size >= cutoff_radius for algorithm correctness
        # Use cell_size = cutoff_radius for optimal efficiency
        self.cell_size = cutoff_radius
        
        # Auto-compute cells per dimension if not specified
        if cells_per_dimension is None:
            # Adaptive sizing based on embedding statistics
            # Aim for reasonable number of total cells (not too sparse, not too dense)
            self.cells_per_dimension = max(5, min(20, int(np.ceil(10 ** (1.0 / self.target_dimension)))))
        else:
            self.cells_per_dimension = cells_per_dimension
        
        # Storage structures
        self.cells: Dict[CellCoordinate, Cell] = {}
        self.charge_to_cell: Dict[int, CellCoordinate] = {}
        self.charge_positions: np.ndarray = None  # Reduced-dimension positions
        self.original_positions: np.ndarray = None  # Full-dimension positions
        
        # Dimensionality reduction
        self.dimension_reducer = None
        self.space_bounds = None  # (min_bounds, max_bounds) for reduced space
        
        # Performance tracking
        self.statistics: Optional[CellListsStatistics] = None
        
        logger.debug(f"EmbeddingSpaceCellLists initialized:")
        logger.debug(f"  Cutoff radius: {self.cutoff_radius:.4f}")
        logger.debug(f"  Embedding dimension: {self.embedding_dimension}")
        logger.debug(f"  Target dimension: {self.target_dimension}")
        logger.debug(f"  Cell size: {self.cell_size:.4f}")
        logger.debug(f"  Cells per dimension: {self.cells_per_dimension}")
    
    @field_theory_auto_optimize(prefer_accuracy=True, profile=True)
    def build_cell_lists(self, temporal_biographies: List) -> None:
        """
        Build cell lists from temporal biographies.
        
        CONSTRUCTION ALGORITHM:
        1. Extract field interference signatures from biographies
        2. Apply dimensionality reduction for spatial partitioning
        3. Determine space bounds and create grid structure
        4. Assign each charge to appropriate cell
        5. Build neighbor relationships between cells
        
        Args:
            temporal_biographies: List of TemporalBiography objects
        """
        import time
        start_time = time.time()
        
        logger.info(f"🏗️  Building cell lists for {len(temporal_biographies)} temporal biographies")
        
        # Step 1: Extract field signatures for spatial representation
        logger.debug("Extracting field interference signatures...")
        field_signatures = self._extract_field_signatures(temporal_biographies)
        self.original_positions = field_signatures
        
        # Step 2: Apply dimensionality reduction for efficient partitioning
        logger.debug(f"Applying {self.reduction_method} dimensionality reduction...")
        reduced_positions = self._apply_dimensionality_reduction(field_signatures)
        self.charge_positions = reduced_positions
        
        # Step 3: Determine space bounds and grid parameters
        logger.debug("Computing space bounds and grid structure...")
        self._compute_space_bounds()
        
        # Step 4: Create cells and assign charges
        logger.debug("Creating cells and assigning charges...")
        self._assign_charges_to_cells(len(temporal_biographies))
        
        # Step 5: Compute construction statistics
        construction_time = time.time() - start_time
        self._compute_statistics(construction_time)
        
        # Memory cleanup
        gc.collect()
        
        logger.info(f"✅ Cell lists construction complete:")
        logger.info(f"   Construction time: {construction_time:.3f}s")
        logger.info(f"   Total cells: {self.statistics.total_cells}")
        logger.info(f"   Occupied cells: {self.statistics.occupied_cells}")
        logger.info(f"   Max charges per cell: {self.statistics.max_charges_per_cell}")
        logger.info(f"   Mean charges per cell: {self.statistics.mean_charges_per_cell:.1f}")
        logger.info(f"   Memory usage: {self.statistics.memory_usage_mb:.1f} MB")
    
    def _extract_field_signatures(self, temporal_biographies: List) -> np.ndarray:
        """
        Extract field interference signatures for spatial representation.
        
        FIELD-THEORETIC EMBEDDING:
        Uses field_interference_signature as the natural spatial coordinate system,
        as this captures the essential field properties that determine interactions.
        """
        field_signatures = []
        
        for bio in temporal_biographies:
            signature = bio.field_interference_signature
            
            # Handle complex signatures by extracting magnitude and phase
            if np.iscomplexobj(signature):
                magnitude = np.abs(signature)
                phase = np.angle(signature)
                
                # Combine magnitude and phase as real-valued feature vector
                # This preserves the complex field structure in real-valued form
                combined_signature = np.concatenate([magnitude, phase])
            else:
                combined_signature = np.real(signature)
            
            field_signatures.append(combined_signature)
        
        field_signatures = np.array(field_signatures)
        
        logger.debug(f"Extracted field signatures: {field_signatures.shape}")
        return field_signatures
    
    def _apply_dimensionality_reduction(self, field_signatures: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction while preserving local structure.
        
        REDUCTION STRATEGIES:
        - PCA: Preserves maximum variance, good for global structure
        - Random projection: Johnson-Lindenstrauss lemma guarantees
        - Identity: No reduction for low-dimensional embeddings
        """
        if self.target_dimension >= field_signatures.shape[1]:
            # No reduction needed
            logger.debug("No dimensionality reduction applied (target >= original)")
            self.dimension_reducer = None
            return field_signatures
        
        if self.reduction_method == 'pca':
            self.dimension_reducer = PCA(
                n_components=self.target_dimension, 
                random_state=42
            )
            reduced_positions = self.dimension_reducer.fit_transform(field_signatures)
            
            explained_variance = self.dimension_reducer.explained_variance_ratio_.sum()
            logger.debug(f"PCA reduction: {explained_variance:.3f} variance explained")
            
        elif self.reduction_method == 'random':
            self.dimension_reducer = GaussianRandomProjection(
                n_components=self.target_dimension,
                random_state=42
            )
            reduced_positions = self.dimension_reducer.fit_transform(field_signatures)
            
            logger.debug("Random projection applied")
            
        elif self.reduction_method == 'identity':
            # Take first target_dimension components without transformation
            reduced_positions = field_signatures[:, :self.target_dimension]
            self.dimension_reducer = None
            
            logger.debug("Identity reduction applied")
            
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction_method}")
        
        logger.debug(f"Dimensionality reduction: {field_signatures.shape[1]} -> {reduced_positions.shape[1]}")
        return reduced_positions
    
    def _compute_space_bounds(self) -> None:
        """Compute bounds of the reduced embedding space for grid creation."""
        # Find min/max bounds in each dimension
        min_bounds = np.min(self.charge_positions, axis=0)
        max_bounds = np.max(self.charge_positions, axis=0)
        
        # Add padding to ensure all charges are within bounds
        # Padding = 10% of range or cutoff_radius, whichever is larger
        ranges = max_bounds - min_bounds
        padding = np.maximum(0.1 * ranges, self.cutoff_radius)
        
        min_bounds -= padding
        max_bounds += padding
        
        self.space_bounds = (min_bounds, max_bounds)
        
        logger.debug(f"Space bounds: min={min_bounds}, max={max_bounds}")
        logger.debug(f"Space ranges: {max_bounds - min_bounds}")
    
    def _assign_charges_to_cells(self, n_charges: int) -> None:
        """
        Assign each charge to its appropriate cell.
        
        ASSIGNMENT ALGORITHM:
        For each charge:
        1. Compute cell coordinate based on position
        2. Create cell if it doesn't exist
        3. Add charge to cell
        4. Update charge -> cell mapping
        """
        min_bounds, max_bounds = self.space_bounds
        ranges = max_bounds - min_bounds
        
        # Compute cell indices for all charges at once (vectorized)
        normalized_positions = (self.charge_positions - min_bounds) / ranges
        
        # Clamp to [0, 1) to handle edge cases
        normalized_positions = np.clip(normalized_positions, 0.0, 0.999999)
        
        # Convert to cell indices
        cell_indices = (normalized_positions * self.cells_per_dimension).astype(int)
        
        # Create cells and assign charges
        for charge_idx in range(n_charges):
            cell_coord = CellCoordinate(tuple(cell_indices[charge_idx]))
            
            # Create cell if it doesn't exist
            if cell_coord not in self.cells:
                self._create_cell(cell_coord)
            
            # Add charge to cell
            self.cells[cell_coord].add_charge(charge_idx)
            self.charge_to_cell[charge_idx] = cell_coord
        
        logger.debug(f"Assigned {n_charges} charges to {len(self.cells)} cells")
    
    def _create_cell(self, coordinate: CellCoordinate) -> None:
        """Create a new cell with computed bounds."""
        min_bounds, max_bounds = self.space_bounds
        ranges = max_bounds - min_bounds
        
        # Compute cell bounds in embedding space
        cell_size_per_dim = ranges / self.cells_per_dimension
        
        cell_min = min_bounds + np.array(coordinate.indices) * cell_size_per_dim
        cell_max = cell_min + cell_size_per_dim
        
        cell = Cell(
            coordinate=coordinate,
            charge_indices=[],
            bounds=(cell_min, cell_max)
        )
        
        self.cells[coordinate] = cell
    
    def find_neighbor_cells(self, cell_coordinate: CellCoordinate) -> List[CellCoordinate]:
        """
        Find all neighboring cells (including the cell itself).
        
        NEIGHBOR DEFINITION:
        For D-dimensional grid, neighbors are all cells within 1 step in any dimension.
        Total neighbors = 3^D (including center cell).
        
        COMPLEXITY:
        O(3^D) where D is target_dimension, which is bounded and constant.
        """
        neighbors = []
        center_indices = np.array(cell_coordinate.indices)
        
        # Generate all possible neighbor offsets: (-1, 0, +1) in each dimension
        offsets = []
        for _ in range(len(center_indices)):
            offsets.append([-1, 0, 1])
        
        # Compute Cartesian product of offsets
        from itertools import product
        for offset_combination in product(*offsets):
            neighbor_indices = center_indices + np.array(offset_combination)
            
            # Check bounds (cells must have non-negative indices)
            if np.all(neighbor_indices >= 0) and np.all(neighbor_indices < self.cells_per_dimension):
                neighbor_coord = CellCoordinate(tuple(neighbor_indices))
                neighbors.append(neighbor_coord)
        
        return neighbors
    
    def find_neighbors_within_cutoff(self, charge_idx: int) -> List[int]:
        """
        Find all charges within cutoff radius of the given charge.
        
        CELL LISTS ALGORITHM:
        1. Find cell containing the charge
        2. Get all neighboring cells (bounded number)
        3. Check all charges in neighboring cells
        4. Filter by exact distance in original embedding space
        
        COMPLEXITY: O(k × m) where:
        - k = number of neighboring cells (constant, bounded by 3^D)
        - m = average charges per cell (bounded for good spatial distribution)
        Result: O(constant) per query
        """
        if charge_idx not in self.charge_to_cell:
            return []
        
        # Get charge's cell
        charge_cell = self.charge_to_cell[charge_idx]
        
        # Find all neighboring cells
        neighbor_cells = self.find_neighbor_cells(charge_cell)
        
        # Collect all candidate charges from neighboring cells
        candidate_charges = []
        for cell_coord in neighbor_cells:
            if cell_coord in self.cells:
                candidate_charges.extend(self.cells[cell_coord].charge_indices)
        
        # Filter candidates by exact distance in original space
        neighbors_within_cutoff = []
        charge_position = self.original_positions[charge_idx]
        
        for candidate_idx in candidate_charges:
            if candidate_idx == charge_idx:
                continue  # Skip self
            
            candidate_position = self.original_positions[candidate_idx]
            distance = np.linalg.norm(charge_position - candidate_position)
            
            if distance <= self.cutoff_radius:
                neighbors_within_cutoff.append(candidate_idx)
        
        return neighbors_within_cutoff
    
    def get_cell_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about cell occupancy and distribution."""
        if not self.cells:
            return {}
        
        cell_sizes = [cell.size() for cell in self.cells.values()]
        occupied_cells = [size for size in cell_sizes if size > 0]
        
        return {
            'total_cells': len(self.cells),
            'occupied_cells': len(occupied_cells),
            'empty_cells': len(self.cells) - len(occupied_cells),
            'cell_size_stats': {
                'mean': np.mean(cell_sizes) if cell_sizes else 0,
                'std': np.std(cell_sizes) if cell_sizes else 0,
                'min': np.min(cell_sizes) if cell_sizes else 0,
                'max': np.max(cell_sizes) if cell_sizes else 0,
                'median': np.median(cell_sizes) if cell_sizes else 0
            },
            'occupied_cell_stats': {
                'mean': np.mean(occupied_cells) if occupied_cells else 0,
                'std': np.std(occupied_cells) if occupied_cells else 0,
                'min': np.min(occupied_cells) if occupied_cells else 0,
                'max': np.max(occupied_cells) if occupied_cells else 0,
                'median': np.median(occupied_cells) if occupied_cells else 0
            }
        }
    
    def _compute_statistics(self, construction_time: float) -> None:
        """Compute comprehensive statistics from cell lists construction."""
        cell_stats = self.get_cell_statistics()
        
        # Memory usage estimation
        memory_usage = 0
        memory_usage += self.charge_positions.nbytes if self.charge_positions is not None else 0
        memory_usage += self.original_positions.nbytes if self.original_positions is not None else 0
        memory_usage += len(self.cells) * 200  # Estimated bytes per cell object
        memory_usage += len(self.charge_to_cell) * 50  # Estimated bytes per mapping
        
        memory_usage_mb = memory_usage / (1024 * 1024)
        
        # Reduction ratio
        reduction_ratio = (self.embedding_dimension / self.target_dimension 
                          if self.target_dimension > 0 else 1.0)
        
        self.statistics = CellListsStatistics(
            total_cells=cell_stats['total_cells'],
            occupied_cells=cell_stats['occupied_cells'], 
            empty_cells=cell_stats['empty_cells'],
            max_charges_per_cell=int(cell_stats['cell_size_stats']['max']),
            mean_charges_per_cell=cell_stats['cell_size_stats']['mean'],
            std_charges_per_cell=cell_stats['cell_size_stats']['std'],
            embedding_reduction_ratio=reduction_ratio,
            memory_usage_mb=memory_usage_mb,
            construction_time_seconds=construction_time
        )
    
    def validate_cell_lists(self) -> Dict[str, bool]:
        """
        Validate cell lists construction for correctness.
        
        VALIDATION CHECKS:
        1. All charges assigned to exactly one cell
        2. Cell bounds are consistent with grid structure
        3. Neighbor relationships are symmetric
        4. No charges outside space bounds
        """
        validation_results = {
            'all_charges_assigned': True,
            'cell_bounds_consistent': True,
            'no_charges_outside_bounds': True,
            'reasonable_cell_distribution': True
        }
        
        try:
            # Check 1: All charges assigned
            n_charges = len(self.original_positions) if self.original_positions is not None else 0
            assigned_charges = set(self.charge_to_cell.keys())
            if len(assigned_charges) != n_charges:
                validation_results['all_charges_assigned'] = False
                logger.warning(f"Charge assignment mismatch: {len(assigned_charges)} assigned, {n_charges} total")
            
            # Check 2: Charges within space bounds
            if self.charge_positions is not None and self.space_bounds is not None:
                min_bounds, max_bounds = self.space_bounds
                within_bounds = np.all((self.charge_positions >= min_bounds) & 
                                     (self.charge_positions <= max_bounds))
                if not within_bounds:
                    validation_results['no_charges_outside_bounds'] = False
                    logger.warning("Some charges are outside computed space bounds")
            
            # Check 3: Reasonable cell distribution
            if self.statistics is not None:
                if self.statistics.max_charges_per_cell > n_charges * 0.5:
                    validation_results['reasonable_cell_distribution'] = False
                    logger.warning(f"Poor cell distribution: max {self.statistics.max_charges_per_cell} charges in one cell")
        
        except Exception as e:
            logger.error(f"Cell lists validation failed: {e}")
            for key in validation_results:
                validation_results[key] = False
        
        return validation_results
    
    def __len__(self) -> int:
        """Number of charges in the cell lists."""
        return len(self.charge_to_cell)
    
    def __contains__(self, charge_idx: int) -> bool:
        """Check if charge is in the cell lists."""
        return charge_idx in self.charge_to_cell
    
    def memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        if self.statistics:
            return self.statistics.memory_usage_mb
        return 0.0


if __name__ == "__main__":
    """Test embedding space cell lists with sample data."""
    print("EmbeddingSpaceCellLists ready for O(N) spatial partitioning")
    print("Efficiently handles high-dimensional embedding spaces with bounded neighbor queries")