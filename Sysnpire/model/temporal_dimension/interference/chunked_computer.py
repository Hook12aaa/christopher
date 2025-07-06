"""
Chunked Vectorized Interference Computer

Memory-efficient O(N²) computation using chunked vectorization to eliminate
Python loop overhead while managing memory usage.
"""

import numpy as np
from typing import List, Optional
import time

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.utils.logger import get_logger
logger = get_logger(__name__)


class ChunkedComputer:
    """Memory-efficient vectorized interference computation using chunks."""
    
    def __init__(self, chunk_size: Optional[int] = None, max_memory_mb: float = 100.0):
        """
        Initialize chunked computer.
        
        Args:
            chunk_size: Fixed chunk size, or None for automatic sizing
            max_memory_mb: Maximum memory per chunk in MB
        """
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
    
    def compute_interference_matrix(self, temporal_biographies: List) -> np.ndarray:
        """
        Compute full interference matrix using chunked vectorization.
        
        Args:
            temporal_biographies: List of temporal biographies
            
        Returns:
            Complex interference matrix of shape (N, N)
        """
        n_charges = len(temporal_biographies)
        logger.info(f"🚀 Computing interference matrix using chunked vectorization ({n_charges} charges)")
        
        start_time = time.time()
        
        # Determine optimal chunk size
        chunk_size = self._determine_chunk_size(temporal_biographies)
        logger.info(f"   Using chunk size: {chunk_size}")
        
        # Pre-extract all data to avoid repeated access
        all_phases, all_trajectories = self._extract_all_data(temporal_biographies)
        
        # Initialize result matrix
        interference_matrix = np.zeros((n_charges, n_charges), dtype=complex)
        
        # Process in chunks
        total_chunks = (n_charges + chunk_size - 1) // chunk_size
        processed_chunks = 0
        
        for i_start in range(0, n_charges, chunk_size):
            i_end = min(i_start + chunk_size, n_charges)
            
            for j_start in range(0, n_charges, chunk_size):
                j_end = min(j_start + chunk_size, n_charges)
                
                # Compute chunk using vectorization
                chunk_result = self._compute_chunk(
                    all_phases[i_start:i_end], all_trajectories[i_start:i_end],
                    all_phases[j_start:j_end], all_trajectories[j_start:j_end]
                )
                
                # Store result
                interference_matrix[i_start:i_end, j_start:j_end] = chunk_result
                
                processed_chunks += 1
                if processed_chunks % 100 == 0:
                    progress = processed_chunks / (total_chunks ** 2) * 100
                    logger.info(f"   Progress: {progress:.1f}% ({processed_chunks}/{total_chunks**2} chunks)")
        
        # Zero out diagonal (no self-interference)
        np.fill_diagonal(interference_matrix, 0)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Chunked computation complete in {elapsed:.2f}s")
        
        return interference_matrix
    
    def _determine_chunk_size(self, temporal_biographies: List) -> int:
        """Determine optimal chunk size based on memory constraints."""
        if self.chunk_size is not None:
            return self.chunk_size
        
        # Estimate memory usage per element
        if len(temporal_biographies) > 0:
            phase_dim = len(temporal_biographies[0].phase_coordination)
            traj_dim = len(temporal_biographies[0].trajectory_operators)
        else:
            phase_dim = traj_dim = 1024  # Default estimate
        
        # Memory per chunk element (bytes):
        # - Phase differences: chunk_size² × phase_dim × 16 (complex128)
        # - Trajectory products: chunk_size² × traj_dim × 16 (complex128)
        bytes_per_element = (phase_dim + traj_dim) * 16
        
        # Solve: chunk_size² × bytes_per_element = max_memory_mb × 1024² 
        max_bytes = self.max_memory_mb * 1024 * 1024
        chunk_size = int(np.sqrt(max_bytes / bytes_per_element))
        
        # Ensure reasonable bounds
        chunk_size = max(10, min(chunk_size, 1000))
        
        return chunk_size
    
    def _extract_all_data(self, temporal_biographies: List) -> tuple:
        """Extract all phase and trajectory data once to avoid repeated access."""
        logger.info("   Extracting temporal biography data...")
        
        all_phases = np.array([bio.phase_coordination for bio in temporal_biographies])
        all_trajectories = np.array([bio.trajectory_operators for bio in temporal_biographies])
        
        logger.info(f"   Data shapes: phases={all_phases.shape}, trajectories={all_trajectories.shape}")
        
        return all_phases, all_trajectories
    
    def _compute_chunk(self, phases_i: np.ndarray, trajectories_i: np.ndarray,
                      phases_j: np.ndarray, trajectories_j: np.ndarray) -> np.ndarray:
        """
        Compute interference for a chunk using vectorized operations.
        
        Args:
            phases_i: Phase data for i-charges, shape (chunk_i, phase_dim)
            trajectories_i: Trajectory data for i-charges, shape (chunk_i, traj_dim)
            phases_j: Phase data for j-charges, shape (chunk_j, phase_dim)  
            trajectories_j: Trajectory data for j-charges, shape (chunk_j, traj_dim)
            
        Returns:
            Interference matrix chunk, shape (chunk_i, chunk_j)
        """
        # Vectorized phase differences using broadcasting
        # phases_i: (chunk_i, phase_dim) -> (chunk_i, 1, phase_dim)
        # phases_j: (chunk_j, phase_dim) -> (1, chunk_j, phase_dim)
        # Result: (chunk_i, chunk_j, phase_dim)
        phase_diffs = phases_i[:, None, :] - phases_j[None, :, :]
        
        # Vectorized phase interference
        phase_interference = np.mean(np.exp(1j * phase_diffs), axis=2)
        
        # Vectorized trajectory products using broadcasting
        # trajectories_i: (chunk_i, traj_dim) -> (chunk_i, 1, traj_dim)
        # trajectories_j: (chunk_j, traj_dim) -> (1, chunk_j, traj_dim)
        # Result: (chunk_i, chunk_j, traj_dim)
        trajectory_products = trajectories_i[:, None, :] * np.conj(trajectories_j[None, :, :])
        
        # Vectorized trajectory interference
        trajectory_interference = np.mean(trajectory_products, axis=2)
        
        # Combined interference
        return phase_interference * trajectory_interference


if __name__ == "__main__":
    """Test chunked computer with sample data.""" 
    print("ChunkedComputer ready for memory-efficient interference computation")