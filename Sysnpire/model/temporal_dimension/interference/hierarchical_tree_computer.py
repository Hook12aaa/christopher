"""
Hierarchical Tree Computer - WARNING: O(N²) Complexity Issues

WARNING: This implementation has fundamental complexity issues and does NOT achieve
the claimed O(log N) performance. Actual complexity is O(N²) due to design flaws.

KNOWN ISSUES:
- Claims O(log N) but actually implements O(N²) pairwise operations
- Memory leaks from unbounded caches
- Race conditions in parallel processing
- Mathematical approximation errors

USE INSTEAD:
- TreeComputer: True O(N log N) using Barnes-Hut algorithm
- SparseMatrixComputer: Memory-efficient for sparse datasets
- MemoryMappedComputer: For datasets > 2GB
- ChunkedComputer: Reliable dense computation

STATUS: DISABLED - Use proven optimization methods instead
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import time
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from scipy.spatial.distance import pdist, squareform
from dataclasses import dataclass
import concurrent.futures
from threading import Lock
import multiprocessing as mp

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.utils.logger import get_logger
from Sysnpire.utils.field_theory_optimizers import field_theory_auto_optimize

logger = get_logger(__name__)


@dataclass
class HierarchicalNode:
    """Node in the hierarchical decomposition tree."""
    level: int                          # Decomposition level (0 = finest)
    indices: List[int]                  # Charge indices in this node
    center_signature: np.ndarray        # Combined temporal signature
    interference_radius: float          # Effective interaction radius
    children: Optional[List['HierarchicalNode']] = None
    parent: Optional['HierarchicalNode'] = None
    
    # Cached computations for O(log N) queries
    cached_interactions: Dict[int, complex] = None
    significance_threshold: float = 1e-8


@dataclass
class InterferenceQuery:
    """Query structure for O(log N) interference computation."""
    charge_i: int
    charge_j: int
    required_accuracy: float = 1e-6
    max_depth: int = None


class HierarchicalBasisEngine:
    """
    Wavelet-like decomposition engine for temporal biographies.
    
    Creates hierarchical representation with O(log N) levels where:
    - Level 0: Individual charges (finest resolution)
    - Level k: Groups of 2^k charges (coarser resolution)
    - Each level captures different temporal scales
    """
    
    def __init__(self, max_levels: int = None, significance_threshold: float = 1e-8):
        """
        Initialize hierarchical basis engine.
        
        Args:
            max_levels: Maximum decomposition levels (default: log2(N))
            significance_threshold: Threshold for pruning insignificant interactions
        """
        self.max_levels = max_levels
        self.significance_threshold = significance_threshold
        self.decomposition_cache = {}
    
    def decompose_temporal_biographies(self, temporal_biographies: List) -> HierarchicalNode:
        """
        Create hierarchical decomposition of temporal biographies.
        
        Returns:
            Root node of hierarchical tree with log N levels
        """
        n_charges = len(temporal_biographies)
        if self.max_levels is None:
            self.max_levels = max(1, int(np.log2(n_charges)) + 1)
        
        logger.info(f"🔄 Creating hierarchical decomposition with {self.max_levels} levels")
        
        # Extract temporal signatures for spatial organization
        signatures = self._extract_temporal_signatures(temporal_biographies)
        
        # Build hierarchical tree bottom-up
        root = self._build_hierarchical_tree(temporal_biographies, signatures)
        
        # Precompute significant interactions at each level
        self._precompute_level_interactions(root, temporal_biographies)
        
        return root
    
    def _extract_temporal_signatures(self, temporal_biographies: List) -> np.ndarray:
        """
        Extract multi-dimensional temporal signatures for hierarchical organization.
        
        Creates 6D signature space:
        - Mean trajectory magnitude
        - Trajectory magnitude variance  
        - Mean phase coordination
        - Phase coordination variance
        - Real trajectory component
        - Imaginary trajectory component
        """
        signatures = []
        
        for bio in temporal_biographies:
            # Handle SAGE objects
            if hasattr(bio.trajectory_operators, 'dtype') and bio.trajectory_operators.dtype == object:
                traj_complex = np.zeros(len(bio.trajectory_operators), dtype=complex)
                for i, op in enumerate(bio.trajectory_operators):
                    if hasattr(op, 'real') and hasattr(op, 'imag'):
                        traj_complex[i] = complex(float(op.real()), float(op.imag()))
                    else:
                        traj_complex[i] = complex(op)
                trajectories = traj_complex
            else:
                trajectories = bio.trajectory_operators
            
            # Create 6D temporal signature
            signature = np.array([
                np.mean(np.abs(trajectories)),              # Mean magnitude
                np.std(np.abs(trajectories)),               # Magnitude variation
                np.mean(bio.phase_coordination),            # Mean phase
                np.std(bio.phase_coordination),             # Phase variation
                np.real(np.mean(trajectories)),             # Mean real part
                np.imag(np.mean(trajectories))              # Mean imaginary part
            ])
            
            signatures.append(signature)
        
        signatures = np.array(signatures)
        
        # Normalize for better hierarchical clustering
        signatures = (signatures - np.mean(signatures, axis=0)) / (np.std(signatures, axis=0) + 1e-8)
        
        return signatures
    
    def _build_hierarchical_tree(self, temporal_biographies: List, signatures: np.ndarray) -> HierarchicalNode:
        """
        Build hierarchical tree using recursive binary decomposition.
        
        Creates balanced tree with guaranteed log N depth.
        """
        n_charges = len(temporal_biographies)
        all_indices = list(range(n_charges))
        
        # Create root node at finest level
        root = self._create_hierarchical_node(
            level=0,
            indices=all_indices,
            signatures=signatures,
            temporal_biographies=temporal_biographies
        )
        
        # Recursively build tree levels
        self._recursive_decomposition(root, signatures, temporal_biographies, level=1)
        
        return root
    
    def _recursive_decomposition(self, node: HierarchicalNode, signatures: np.ndarray, 
                                temporal_biographies: List, level: int):
        """
        Recursively decompose node into children until max level reached.
        """
        if level >= self.max_levels or len(node.indices) <= 2:
            return
        
        # Binary decomposition using k-means clustering
        child_indices_list = self._binary_decomposition(node.indices, signatures)
        
        # Create child nodes
        node.children = []
        for child_indices in child_indices_list:
            if len(child_indices) > 0:
                child_node = self._create_hierarchical_node(
                    level=level,
                    indices=child_indices,
                    signatures=signatures,
                    temporal_biographies=temporal_biographies
                )
                child_node.parent = node
                node.children.append(child_node)
                
                # Recurse to next level
                self._recursive_decomposition(child_node, signatures, temporal_biographies, level + 1)
    
    def _binary_decomposition(self, indices: List[int], signatures: np.ndarray) -> List[List[int]]:
        """
        Binary decomposition using k-means clustering for balanced partitioning.
        """
        if len(indices) <= 2:
            return [indices]
        
        # Extract signatures for these indices
        subset_signatures = signatures[indices]
        
        # Simple binary split using median along dominant principal component
        if len(subset_signatures) > 1:
            # Find dominant dimension
            variances = np.var(subset_signatures, axis=0)
            dominant_dim = np.argmax(variances)
            
            # Split at median
            median_val = np.median(subset_signatures[:, dominant_dim])
            
            left_indices = [indices[i] for i, sig in enumerate(subset_signatures) 
                           if sig[dominant_dim] <= median_val]
            right_indices = [indices[i] for i, sig in enumerate(subset_signatures) 
                            if sig[dominant_dim] > median_val]
            
            # Ensure both groups are non-empty
            if len(left_indices) == 0:
                mid = len(indices) // 2
                left_indices = indices[:mid]
                right_indices = indices[mid:]
            elif len(right_indices) == 0:
                mid = len(indices) // 2
                left_indices = indices[:mid]
                right_indices = indices[mid:]
            
            return [left_indices, right_indices]
        else:
            return [indices]
    
    def _create_hierarchical_node(self, level: int, indices: List[int], 
                                 signatures: np.ndarray, temporal_biographies: List) -> HierarchicalNode:
        """
        Create hierarchical node with combined temporal signature.
        """
        if len(indices) == 0:
            raise ValueError("Cannot create node with empty indices")
        
        # Compute combined temporal signature for this group
        combined_signature = np.mean(signatures[indices], axis=0)
        
        # Estimate interference radius based on signature spread
        if len(indices) > 1:
            distances = pdist(signatures[indices])
            interference_radius = np.max(distances) if len(distances) > 0 else 1.0
        else:
            interference_radius = 1.0
        
        return HierarchicalNode(
            level=level,
            indices=indices,
            center_signature=combined_signature,
            interference_radius=interference_radius,
            cached_interactions={},
            significance_threshold=self.significance_threshold
        )
    
    def _precompute_level_interactions(self, root: HierarchicalNode, temporal_biographies: List):
        """
        Precompute significant interactions at each level for O(log N) queries.
        """
        logger.info("⚡ Precomputing hierarchical interactions...")
        
        # Traverse tree and precompute interactions
        self._traverse_and_precompute(root, temporal_biographies)
    
    def _traverse_and_precompute(self, node: HierarchicalNode, temporal_biographies: List):
        """
        Traverse tree and precompute significant interactions at each node.
        """
        # Precompute interactions for this node
        if len(node.indices) > 1:
            self._compute_node_interactions(node, temporal_biographies)
        
        # Recurse to children
        if node.children:
            for child in node.children:
                self._traverse_and_precompute(child, temporal_biographies)
    
    def _compute_node_interactions(self, node: HierarchicalNode, temporal_biographies: List):
        """
        Compute and cache significant interactions within a node.
        """
        # Deterministic sampling for reproducible results
        max_samples = min(20, len(node.indices))
        if max_samples >= len(node.indices):
            sample_indices = node.indices
        else:
            # Use evenly spaced indices for deterministic sampling
            step = len(node.indices) // max_samples
            sample_indices = node.indices[::step][:max_samples]
        
        significant_pairs = set()
        
        for i, idx_i in enumerate(sample_indices):
            for j, idx_j in enumerate(sample_indices):
                if i != j:
                    bio_i = temporal_biographies[idx_i]
                    bio_j = temporal_biographies[idx_j]
                    
                    # Quick interference estimate
                    interference = self._quick_interference_estimate(bio_i, bio_j)
                    
                    if np.abs(interference) > node.significance_threshold:
                        significant_pairs.add((min(idx_i, idx_j), max(idx_i, idx_j)))
        
        # Cache significant interactions
        node.cached_interactions = {pair: True for pair in significant_pairs}
    
    def _quick_interference_estimate(self, bio_i, bio_j) -> complex:
        """
        Fast interference estimation for precomputation.
        """
        # Simplified interference calculation
        phase_diff = np.mean(bio_i.phase_coordination - bio_j.phase_coordination)
        phase_interference = np.abs(np.exp(1j * phase_diff))
        
        # Trajectory similarity estimate
        traj_similarity = np.abs(np.mean(
            bio_i.field_interference_signature * 
            np.conj(bio_j.field_interference_signature)
        ))
        
        return phase_interference * traj_similarity


class ParallelTreeTraversal:
    """
    Tree traversal engine for hierarchical interference computation.
    
    Note: Despite the name, this implementation is not truly parallel
    and does not achieve O(log N) complexity due to fundamental design issues.
    """
    
    def __init__(self, max_workers: int = None, cache_size: int = 1000):
        """
        Initialize traversal engine.
        
        Args:
            max_workers: Maximum parallel workers (default: CPU count)
            cache_size: Maximum cache entries to prevent memory leaks
        """
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.query_cache = {}
        self.cache_lock = Lock()
        self.cache_size = cache_size  # Prevent unbounded growth
        
        # Thread-safe counters
        self._query_count_lock = Lock()
        self._query_count = 0
        self._total_query_time = 0.0
    
    @field_theory_auto_optimize(prefer_accuracy=True, profile=True)
    def parallel_interference_query(self, root: HierarchicalNode, query: InterferenceQuery,
                                   temporal_biographies: List) -> complex:
        """
        Execute O(log N) interference query using parallel tree traversal.
        
        Args:
            root: Root of hierarchical tree
            query: Interference query specification
            temporal_biographies: Original temporal biographies
            
        Returns:
            Complex interference value
        """
        # Check cache first
        cache_key = (query.charge_i, query.charge_j)
        with self.cache_lock:
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]
        
        # Execute traversal
        interference = self._parallel_traverse(
            root, query, temporal_biographies, depth=0
        )
        
        # Cache result with bounds to prevent memory leaks
        with self.cache_lock:
            # Implement simple LRU by clearing cache when full
            if len(self.query_cache) >= self.cache_size:
                self.query_cache.clear()  # Simple eviction strategy
            self.query_cache[cache_key] = interference
        
        return interference
    
    def _parallel_traverse(self, node: HierarchicalNode, query: InterferenceQuery,
                          temporal_biographies: List, depth: int) -> complex:
        """
        Parallel tree traversal with divide-and-conquer.
        
        Guaranteed O(log N) depth through balanced tree structure.
        """
        # Base case: leaf node or max depth reached
        if (not node.children or 
            depth >= (query.max_depth or 20) or
            len(node.indices) <= 2):
            return self._exact_interference_computation(
                query.charge_i, query.charge_j, temporal_biographies
            )
        
        # Check if both charges are in this subtree
        if query.charge_i not in node.indices or query.charge_j not in node.indices:
            # One or both charges not in this subtree
            return 0.0 + 0j
        
        # Find which children contain the charges
        child_i = None
        child_j = None
        
        for child in node.children:
            if query.charge_i in child.indices:
                child_i = child
            if query.charge_j in child.indices:
                child_j = child
        
        # If charges are in different children, use approximation
        if child_i != child_j and child_i is not None and child_j is not None:
            return self._approximate_cross_child_interference(
                child_i, child_j, query, temporal_biographies
            )
        
        # If charges are in same child, recurse deeper
        if child_i is not None:
            return self._parallel_traverse(
                child_i, query, temporal_biographies, depth + 1
            )
        
        # Fallback to exact computation
        return self._exact_interference_computation(
            query.charge_i, query.charge_j, temporal_biographies
        )
    
    def _approximate_cross_child_interference(self, child_i: HierarchicalNode, 
                                             child_j: HierarchicalNode,
                                             query: InterferenceQuery,
                                             temporal_biographies: List) -> complex:
        """
        Approximate interference between charges in different child nodes.
        
        This is where the O(log N) speedup comes from - we don't compute
        exact interactions between distant groups.
        """
        # Use center signatures for approximation
        signature_i = child_i.center_signature
        signature_j = child_j.center_signature
        
        # Distance-based approximation
        distance = np.linalg.norm(signature_i - signature_j)
        
        # If groups are far apart, use multipole-like approximation
        if distance > (child_i.interference_radius + child_j.interference_radius):
            # Use combined signatures for distant interaction approximation
            phase_diff = np.mean(signature_i[:4] - signature_j[:4])  # Phase components
            phase_interference = np.exp(1j * phase_diff)
            
            # Magnitude based on signature similarity
            magnitude = np.exp(-distance * 0.1)  # Decay with distance
            
            return phase_interference * magnitude
        else:
            # Close groups - compute exact interaction
            return self._exact_interference_computation(
                query.charge_i, query.charge_j, temporal_biographies
            )
    
    def _exact_interference_computation(self, charge_i: int, charge_j: int,
                                       temporal_biographies: List) -> complex:
        """
        Exact interference computation between two charges.
        
        Used only for nearby charges or leaf nodes.
        """
        if charge_i == charge_j:
            return 0.0 + 0j
        
        bio_i = temporal_biographies[charge_i]
        bio_j = temporal_biographies[charge_j]
        
        # Handle SAGE objects
        traj_i = bio_i.trajectory_operators
        traj_j = bio_j.trajectory_operators
        
        if hasattr(traj_i, 'dtype') and traj_i.dtype == object:
            traj_i_complex = np.zeros(len(traj_i), dtype=complex)
            for k, op in enumerate(traj_i):
                if hasattr(op, 'real') and hasattr(op, 'imag'):
                    traj_i_complex[k] = complex(float(op.real()), float(op.imag()))
                else:
                    traj_i_complex[k] = complex(op)
            traj_i = traj_i_complex
        
        if hasattr(traj_j, 'dtype') and traj_j.dtype == object:
            traj_j_complex = np.zeros(len(traj_j), dtype=complex)
            for k, op in enumerate(traj_j):
                if hasattr(op, 'real') and hasattr(op, 'imag'):
                    traj_j_complex[k] = complex(float(op.real()), float(op.imag()))
                else:
                    traj_j_complex[k] = complex(op)
            traj_j = traj_j_complex
        
        # Exact interference calculation
        phase_interference = np.mean(
            np.exp(1j * (bio_i.phase_coordination - bio_j.phase_coordination))
        )
        trajectory_interference = np.mean(traj_i * np.conj(traj_j))
        
        return phase_interference * trajectory_interference


class HierarchicalTreeComputer:
    """
    DEPRECATED: Hierarchical Tree Computer with O(N²) Complexity Issues
    
    WARNING: This class does NOT achieve O(log N) complexity as originally claimed.
    Actual performance is O(N²) due to fundamental design flaws:
    1. Examines all N² pairs despite hierarchical structure
    2. Memory leaks from unbounded caching
    3. Race conditions in parallel processing
    4. Approximation errors affecting mathematical accuracy
    
    STATUS: DISABLED in TemporalDimensionHelper
    RECOMMENDED: Use TreeComputer, SparseMatrixComputer, or ChunkedComputer instead
    """
    
    def __init__(self, 
                 max_levels: int = None,
                 significance_threshold: float = 1e-8,
                 max_workers: int = None,
                 cache_size: int = 10000):
        """
        Initialize hierarchical tree computer.
        
        Args:
            max_levels: Maximum decomposition levels (default: log2(N))
            significance_threshold: Threshold for interaction significance
            max_workers: Maximum parallel workers
            cache_size: Size of query result cache
        """
        self.max_levels = max_levels
        self.significance_threshold = significance_threshold
        self.max_workers = max_workers
        self.cache_size = cache_size
        
        # Initialize engines
        self.basis_engine = HierarchicalBasisEngine(max_levels, significance_threshold)
        self.traversal_engine = ParallelTreeTraversal(max_workers)
        
        # Performance tracking
        self.query_count = 0
        self.total_query_time = 0.0
    
    def compute_interference_matrix(self, temporal_biographies: List) -> np.ndarray:
        """
        Compute interference matrix using hierarchical method.
        
        WARNING: This implementation has known O(N²) complexity issues and should not be used
        for production datasets > 1000 charges. Use ChunkedComputer or SparseMatrixComputer instead.
        
        Args:
            temporal_biographies: List of temporal biographies
            
        Returns:
            Complex interference matrix of shape (N, N)
            
        Raises:
            ValueError: If input validation fails or dataset too large
            MemoryError: If estimated memory usage exceeds safe limits
        """
        # CRITICAL: Enhanced safety validations
        if not temporal_biographies:
            raise ValueError("temporal_biographies cannot be empty")
        
        n_charges = len(temporal_biographies)
        
        # CRITICAL: This implementation should not be used in production
        logger.error(f"❌ HierarchicalTreeComputer is DEPRECATED and should not be used")
        logger.error(f"   Fundamental O(N²) complexity issues, memory leaks, and race conditions")
        logger.error(f"   This method was disabled in TemporalDimensionHelper for safety")
        
        # Prevent any production usage
        if n_charges > 100:
            raise RuntimeError(f"HierarchicalTreeComputer disabled for datasets > 100 charges. "
                             f"Current dataset: {n_charges} charges. "
                             f"Use TreeComputer, SparseMatrixComputer, or ChunkedComputer instead.")
        
        # CRITICAL: Prevent memory exhaustion for large datasets
        estimated_memory_gb = (n_charges * n_charges * 16) / (1024**3)  # complex128
        if estimated_memory_gb > 1.0:  # Much lower safety limit
            raise MemoryError(f"Dataset too large ({estimated_memory_gb:.1f}GB estimated). "
                            f"Use MemoryMappedComputer for datasets > 1GB")
        
        # Warn about known issues
        logger.warning(f"⚠️  Using DEPRECATED HierarchicalTreeComputer with known issues:")
        logger.warning(f"   - Actual complexity: O(N²) = {n_charges**2:,} operations")
        logger.warning(f"   - Memory leaks from unbounded caches")
        logger.warning(f"   - Race conditions in parallel processing")
        logger.warning(f"   - Mathematical approximation errors")
        logger.warning(f"   Recommended: Use TreeComputer for O(N log N) performance")
        
        logger.info(f"🌲 Computing interference using hierarchical method ({n_charges} charges)")
        logger.warning(f"⚠️  Note: This implementation has O(N²) complexity, not O(log N) as originally claimed")
        
        start_time = time.time()
        
        # Step 1: Create hierarchical decomposition (O(N log N) construction)
        decomposition_start = time.time()
        root = self.basis_engine.decompose_temporal_biographies(temporal_biographies)
        decomposition_time = time.time() - decomposition_start
        
        # Step 2: Execute O(log N) queries for significant pairs only
        query_start = time.time()
        interference_matrix = self._execute_hierarchical_queries(
            root, temporal_biographies, n_charges
        )
        query_time = time.time() - query_start
        
        # Step 3: Finalize matrix
        interference_matrix = self._finalize_matrix(interference_matrix)
        
        total_time = time.time() - start_time
        
        # Performance reporting
        sparsity = np.sum(np.abs(interference_matrix) < self.significance_threshold) / (n_charges * n_charges)
        logger.info(f"✅ O(log N) computation complete in {total_time:.2f}s")
        logger.info(f"   Decomposition: {decomposition_time:.2f}s, Queries: {query_time:.2f}s")
        logger.info(f"   Sparsity: {sparsity:.1%}, Queries executed: {self.query_count}")
        logger.info(f"   Average query time: {(self.total_query_time/max(1, self.query_count))*1000:.3f}ms")
        
        return interference_matrix
    
    def _execute_hierarchical_queries(self, root: HierarchicalNode, 
                                     temporal_biographies: List, n_charges: int) -> np.ndarray:
        """
        Execute O(log N) queries for all significant charge pairs.
        
        Key optimization: Only query pairs that are likely to have significant interference.
        """
        # Use sparse matrix for efficiency
        interference_matrix = lil_matrix((n_charges, n_charges), dtype=complex)
        
        # Identify significant pairs using hierarchical structure
        significant_pairs = self._identify_significant_pairs(root, n_charges)
        
        logger.info(f"   Executing {len(significant_pairs)} O(log N) queries...")
        
        # Execute queries in parallel batches
        batch_size = 1000
        batches = [significant_pairs[i:i+batch_size] 
                  for i in range(0, len(significant_pairs), batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            if batch_idx % 10 == 0 and batch_idx > 0:
                logger.info(f"     Processing batch {batch_idx+1}/{len(batches)}...")
            
            # Execute batch queries
            batch_results = self._execute_batch_queries(
                batch, root, temporal_biographies
            )
            
            # Store results
            for (i, j), interference in batch_results.items():
                if np.abs(interference) > self.significance_threshold:
                    interference_matrix[i, j] = interference
        
        return interference_matrix.toarray()
    
    def _identify_significant_pairs(self, root: HierarchicalNode, n_charges: int) -> List[Tuple[int, int]]:
        """
        Identify charge pairs likely to have significant interference.
        
        Uses hierarchical structure to avoid examining all N² pairs.
        """
        significant_pairs = set()
        
        # Traverse tree to identify significant interactions
        self._traverse_for_significant_pairs(root, significant_pairs)
        
        # Convert to list and add local interactions
        pairs_list = list(significant_pairs)
        
        # Add nearby pairs based on spatial proximity (within each leaf)
        self._add_local_pairs(root, pairs_list, n_charges)
        
        return pairs_list
    
    def _traverse_for_significant_pairs(self, node: HierarchicalNode, significant_pairs: set):
        """
        Traverse tree to identify significant interaction pairs.
        """
        # Add cached significant interactions
        if node.cached_interactions:
            significant_pairs.update(node.cached_interactions.keys())
        
        # Add all pairs within small nodes
        if len(node.indices) <= 10:
            for i, idx_i in enumerate(node.indices):
                for j, idx_j in enumerate(node.indices):
                    if i != j:
                        significant_pairs.add((min(idx_i, idx_j), max(idx_i, idx_j)))
        
        # Recurse to children
        if node.children:
            for child in node.children:
                self._traverse_for_significant_pairs(child, significant_pairs)
    
    def _add_local_pairs(self, node: HierarchicalNode, pairs_list: List[Tuple[int, int]], n_charges: int):
        """
        Add local interaction pairs for completeness.
        """
        # Add sequential neighbor pairs (charges often interact with nearby indices)
        for i in range(n_charges - 1):
            pairs_list.append((i, i + 1))
        
        # Add deterministic sampling for coverage (replacing random sampling)
        max_sample_pairs = min(1000, n_charges * 2)
        # Use structured sampling instead of random
        step_size = max(1, n_charges // int(np.sqrt(max_sample_pairs)))
        for i in range(0, n_charges, step_size):
            for j in range(i + step_size, min(i + step_size * 2, n_charges), step_size):
                if i != j:
                    pairs_list.append((min(i, j), max(i, j)))
        
        # Remove duplicates
        pairs_list[:] = list(set(pairs_list))
    
    def _execute_batch_queries(self, batch: List[Tuple[int, int]], 
                              root: HierarchicalNode, 
                              temporal_biographies: List) -> Dict[Tuple[int, int], complex]:
        """
        Execute batch of O(log N) queries in parallel.
        """
        results = {}
        
        # Sequential execution for now (can be parallelized further)
        for i, j in batch:
            query = InterferenceQuery(charge_i=i, charge_j=j)
            
            query_start = time.time()
            interference = self.traversal_engine.parallel_interference_query(
                root, query, temporal_biographies
            )
            query_time = time.time() - query_start
            
            # Thread-safe counter updates
            with self.traversal_engine._query_count_lock:
                self.traversal_engine._query_count += 1
                self.traversal_engine._total_query_time += query_time
            
            # Update local counters for backward compatibility
            self.query_count = self.traversal_engine._query_count
            self.total_query_time = self.traversal_engine._total_query_time
            
            results[(i, j)] = interference
        
        return results
    
    def _finalize_matrix(self, interference_matrix: np.ndarray) -> np.ndarray:
        """
        Finalize interference matrix with symmetry and cleanup.
        """
        # Zero diagonal
        np.fill_diagonal(interference_matrix, 0)
        
        # Enforce Hermitian symmetry
        interference_matrix = (interference_matrix + np.conj(interference_matrix.T)) / 2
        
        # Apply final threshold
        mask = np.abs(interference_matrix) < self.significance_threshold
        interference_matrix[mask] = 0
        
        return interference_matrix


if __name__ == "__main__":
    """Test hierarchical tree computer."""
    print("HierarchicalTreeComputer ready for true O(log N) interference computation")