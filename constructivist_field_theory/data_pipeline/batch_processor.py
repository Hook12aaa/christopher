"""
Batch Processor for Large-Scale Manifold Assembly

Handles efficient batch processing of large datasets for sociology manifold creation.

Key capabilities:
- Memory-efficient processing of large text corpora
- Parallel manifold assembly for multiple datasets
- Incremental manifold updates and evolution
- Distributed processing coordination
- Progress tracking and performance optimization

Designed for research applications requiring large-scale sociological analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Iterator, Callable, Union
from dataclasses import dataclass, field
import time
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import gc

# Import core components
try:
    from .manifold_assembly_pipeline import ManifoldAssemblyPipeline, PipelineConfiguration, PipelineResult
    from ..core_mathematics.conceptual_charge import ConceptualCharge
    from ..product_manifold.product_manifold import ProductManifold
except ImportError:
    warnings.warn("Some imports failed - using fallback types")
    ManifoldAssemblyPipeline = Any
    ConceptualCharge = Any
    ProductManifold = Any


@dataclass
class BatchConfiguration:
    """Configuration for batch processing operations"""
    # Batch processing parameters
    batch_size: int = 100
    max_concurrent_batches: int = 4
    memory_limit_gb: float = 8.0
    
    # Processing strategy
    processing_mode: str = 'parallel'  # 'sequential', 'parallel', 'distributed'
    chunk_strategy: str = 'semantic'   # 'sequential', 'random', 'semantic'
    
    # Pipeline configuration
    pipeline_config: PipelineConfiguration = field(default_factory=PipelineConfiguration)
    
    # Output configuration
    save_intermediate_results: bool = True
    output_directory: Optional[str] = None
    compression_level: int = 6
    
    # Performance optimization
    enable_gc_optimization: bool = True
    memory_monitoring: bool = True
    progress_reporting: bool = True
    
    # Error handling
    continue_on_error: bool = True
    max_retries: int = 3
    error_callback: Optional[Callable] = None


@dataclass
class BatchResult:
    """Result from batch processing operation"""
    # Processing metadata
    total_input_items: int
    successful_batches: int
    failed_batches: int
    total_processing_time: float
    
    # Results
    manifold_results: List[PipelineResult]
    batch_summaries: List[Dict[str, Any]]
    
    # Performance metrics
    processing_rate: float  # items per second
    memory_usage_stats: Dict[str, float]
    error_log: List[Dict[str, Any]]
    
    # Configuration used
    batch_config: BatchConfiguration


class BatchProcessor:
    """
    Efficient batch processor for large-scale manifold assembly.
    
    Handles:
    1. Memory-efficient data chunking and streaming
    2. Parallel/distributed processing coordination
    3. Progress monitoring and performance optimization
    4. Error handling and recovery
    5. Result aggregation and export
    
    Designed for processing large text corpora into sociology manifolds.
    """
    
    def __init__(self, config: Optional[BatchConfiguration] = None):
        """
        Initialize batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfiguration()
        
        # Processing state
        self.current_batch_id = 0
        self.processing_history: List[BatchResult] = []
        
        # Performance monitoring
        self.memory_monitor = MemoryMonitor() if self.config.memory_monitoring else None
        self.progress_tracker = ProgressTracker() if self.config.progress_reporting else None
        
        # Error handling
        self.error_log: List[Dict[str, Any]] = []
    
    def process_text_corpus(self, 
                           texts: List[str],
                           contexts: Optional[List[Dict[str, Any]]] = None,
                           metadata: Optional[List[Dict[str, Any]]] = None) -> BatchResult:
        """
        Process large text corpus into multiple sociology manifolds.
        
        Args:
            texts: List of texts to process
            contexts: Context information for each text
            metadata: Additional metadata for each text
            
        Returns:
            Batch processing result
        """
        start_time = time.time()
        
        # Initialize progress tracking
        if self.progress_tracker:
            self.progress_tracker.initialize(len(texts), self.config.batch_size)
        
        # Create data chunks
        data_chunks = self._create_data_chunks(texts, contexts, metadata)
        
        # Process chunks based on configuration
        if self.config.processing_mode == 'sequential':
            manifold_results = self._process_chunks_sequential(data_chunks)
        elif self.config.processing_mode == 'parallel':
            manifold_results = self._process_chunks_parallel(data_chunks)
        else:  # 'distributed'
            manifold_results = self._process_chunks_distributed(data_chunks)
        
        # Compile batch result
        batch_result = self._compile_batch_result(
            texts, manifold_results, start_time
        )
        
        # Update processing history
        self.processing_history.append(batch_result)
        
        return batch_result
    
    def process_charge_batches(self, 
                             charge_batches: List[List[ConceptualCharge]]) -> BatchResult:
        """
        Process pre-computed charge batches into manifolds.
        
        Args:
            charge_batches: List of charge batches
            
        Returns:
            Batch processing result
        """
        start_time = time.time()
        
        if self.progress_tracker:
            total_charges = sum(len(batch) for batch in charge_batches)
            self.progress_tracker.initialize(total_charges, self.config.batch_size)
        
        # Process charge batches
        manifold_results = []
        
        for i, charges in enumerate(charge_batches):
            try:
                # Create pipeline for this batch
                pipeline = ManifoldAssemblyPipeline(self.config.pipeline_config)
                
                # Process charges to manifold
                result = pipeline.process_charges_to_manifold(charges)
                manifold_results.append(result)
                
                # Update progress
                if self.progress_tracker:
                    self.progress_tracker.update_batch_completed(len(charges))
                
                # Memory cleanup
                if self.config.enable_gc_optimization:
                    gc.collect()
                    
            except Exception as e:
                self._handle_processing_error(e, f"charge_batch_{i}")
                if not self.config.continue_on_error:
                    break
        
        # Compile result
        return self._compile_batch_result(
            [f"charge_batch_{i}" for i in range(len(charge_batches))],
            manifold_results, 
            start_time
        )
    
    def _create_data_chunks(self, 
                           texts: List[str],
                           contexts: Optional[List[Dict[str, Any]]],
                           metadata: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Create data chunks for batch processing"""
        # Default values
        if contexts is None:
            contexts = [{}] * len(texts)
        if metadata is None:
            metadata = [{}] * len(texts)
        
        chunks = []
        
        if self.config.chunk_strategy == 'semantic':
            # Group semantically similar texts together
            chunks = self._create_semantic_chunks(texts, contexts, metadata)
        elif self.config.chunk_strategy == 'random':
            # Randomly shuffle and chunk
            indices = np.random.permutation(len(texts))
            chunks = self._create_index_chunks(texts, contexts, metadata, indices)
        else:  # 'sequential'
            # Process in order
            chunks = self._create_sequential_chunks(texts, contexts, metadata)
        
        return chunks
    
    def _create_semantic_chunks(self, 
                               texts: List[str], 
                               contexts: List[Dict[str, Any]], 
                               metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chunks based on semantic similarity"""
        try:
            # Try to use semantic clustering
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            
            # Compute TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Determine number of clusters
            num_clusters = max(1, len(texts) // self.config.batch_size)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Group by clusters
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)
            
            # Create chunks from clusters
            chunks = []
            for cluster_indices in clusters.values():
                chunk = {
                    'texts': [texts[i] for i in cluster_indices],
                    'contexts': [contexts[i] for i in cluster_indices],
                    'metadata': [metadata[i] for i in cluster_indices],
                    'indices': cluster_indices
                }
                chunks.append(chunk)
            
            return chunks
            
        except ImportError:
            warnings.warn("sklearn not available, falling back to sequential chunking")
            return self._create_sequential_chunks(texts, contexts, metadata)
    
    def _create_sequential_chunks(self, 
                                 texts: List[str], 
                                 contexts: List[Dict[str, Any]], 
                                 metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create sequential chunks"""
        chunks = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            chunk = {
                'texts': texts[i:end_idx],
                'contexts': contexts[i:end_idx],
                'metadata': metadata[i:end_idx],
                'indices': list(range(i, end_idx))
            }
            chunks.append(chunk)
        
        return chunks
    
    def _create_index_chunks(self, 
                            texts: List[str], 
                            contexts: List[Dict[str, Any]], 
                            metadata: List[Dict[str, Any]], 
                            indices: np.ndarray) -> List[Dict[str, Any]]:
        """Create chunks based on provided indices"""
        chunks = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(indices), batch_size):
            end_idx = min(i + batch_size, len(indices))
            batch_indices = indices[i:end_idx]
            
            chunk = {
                'texts': [texts[idx] for idx in batch_indices],
                'contexts': [contexts[idx] for idx in batch_indices],
                'metadata': [metadata[idx] for idx in batch_indices],
                'indices': batch_indices.tolist()
            }
            chunks.append(chunk)
        
        return chunks
    
    def _process_chunks_sequential(self, chunks: List[Dict[str, Any]]) -> List[PipelineResult]:
        """Process chunks sequentially"""
        results = []
        
        for i, chunk in enumerate(chunks):
            try:
                result = self._process_single_chunk(chunk, f"seq_batch_{i}")
                results.append(result)
                
                if self.progress_tracker:
                    self.progress_tracker.update_batch_completed(len(chunk['texts']))
                    
            except Exception as e:
                self._handle_processing_error(e, f"seq_batch_{i}")
                if not self.config.continue_on_error:
                    break
        
        return results
    
    def _process_chunks_parallel(self, chunks: List[Dict[str, Any]]) -> List[PipelineResult]:
        """Process chunks in parallel"""
        results = []
        max_workers = min(self.config.max_concurrent_batches, mp.cpu_count())
        
        # Create partial function for worker
        process_func = partial(self._process_single_chunk_worker, 
                              config=self.config.pipeline_config)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(process_func, chunk, f"par_batch_{i}"): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if self.progress_tracker:
                        self.progress_tracker.update_batch_completed(len(chunks[chunk_idx]['texts']))
                        
                except Exception as e:
                    self._handle_processing_error(e, f"par_batch_{chunk_idx}")
                    if not self.config.continue_on_error:
                        # Cancel remaining tasks
                        for f in future_to_chunk:
                            f.cancel()
                        break
        
        return results
    
    def _process_chunks_distributed(self, chunks: List[Dict[str, Any]]) -> List[PipelineResult]:
        """Process chunks using distributed computing (placeholder)"""
        warnings.warn("Distributed processing not yet implemented, falling back to parallel")
        return self._process_chunks_parallel(chunks)
    
    def _process_single_chunk(self, chunk: Dict[str, Any], batch_id: str) -> PipelineResult:
        """Process a single chunk into manifold"""
        # Memory monitoring
        if self.memory_monitor:
            self.memory_monitor.record_pre_processing()
        
        # Create pipeline
        pipeline = ManifoldAssemblyPipeline(self.config.pipeline_config)
        
        # Process chunk
        result = pipeline.process_text_to_manifold(
            texts=chunk['texts'],
            contexts=chunk['contexts']
        )
        
        # Add batch metadata
        result.batch_id = batch_id
        result.chunk_indices = chunk['indices']
        
        # Memory monitoring
        if self.memory_monitor:
            self.memory_monitor.record_post_processing()
        
        # Save intermediate results if configured
        if self.config.save_intermediate_results and self.config.output_directory:
            self._save_intermediate_result(result, batch_id)
        
        return result
    
    def _process_single_chunk_worker(self, chunk: Dict[str, Any], batch_id: str, config: PipelineConfiguration) -> PipelineResult:
        """Worker function for parallel processing"""
        # This runs in a separate process
        pipeline = ManifoldAssemblyPipeline(config)
        
        result = pipeline.process_text_to_manifold(
            texts=chunk['texts'],
            contexts=chunk['contexts']
        )
        
        result.batch_id = batch_id
        result.chunk_indices = chunk['indices']
        
        return result
    
    def _handle_processing_error(self, error: Exception, batch_id: str):
        """Handle processing errors"""
        error_info = {
            'batch_id': batch_id,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': time.time()
        }
        
        self.error_log.append(error_info)
        
        if self.config.error_callback:
            self.config.error_callback(error_info)
        else:
            warnings.warn(f"Error in batch {batch_id}: {error}")
    
    def _save_intermediate_result(self, result: PipelineResult, batch_id: str):
        """Save intermediate result to disk"""
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save manifold field
        field_file = output_dir / f"{batch_id}_manifold.npz"
        np.savez_compressed(
            field_file,
            field=result.manifold.manifold_equation.field,
            time=result.manifold.manifold_equation.time
        )
        
        # Save sociology map
        map_file = output_dir / f"{batch_id}_sociology_map.npz"
        sociology_map = result.sociology_map
        np.savez_compressed(
            map_file,
            field_magnitude=sociology_map['field_layers']['magnitude'],
            collective_response=sociology_map['field_layers']['collective_response'],
            charge_positions=np.array(sociology_map['charge_data']['positions'])
        )
    
    def _compile_batch_result(self, 
                             input_items: List[Any], 
                             manifold_results: List[PipelineResult], 
                             start_time: float) -> BatchResult:
        """Compile complete batch result"""
        processing_time = time.time() - start_time
        
        # Calculate statistics
        successful_batches = len([r for r in manifold_results if r is not None])
        failed_batches = len(input_items) - successful_batches
        processing_rate = len(input_items) / processing_time if processing_time > 0 else 0.0
        
        # Memory usage stats
        memory_stats = {}
        if self.memory_monitor:
            memory_stats = self.memory_monitor.get_summary()
        
        # Create batch summaries
        batch_summaries = []
        for i, result in enumerate(manifold_results):
            if result is not None:
                summary = {
                    'batch_id': getattr(result, 'batch_id', f'batch_{i}'),
                    'num_charges': result.num_input_charges,
                    'processing_time': result.processing_time,
                    'total_response': result.collective_phenomena['collective_response']['total_response'],
                    'num_emergent_structures': result.emergent_structures['num_coherent_structures']
                }
                batch_summaries.append(summary)
        
        return BatchResult(
            total_input_items=len(input_items),
            successful_batches=successful_batches,
            failed_batches=failed_batches,
            total_processing_time=processing_time,
            manifold_results=[r for r in manifold_results if r is not None],
            batch_summaries=batch_summaries,
            processing_rate=processing_rate,
            memory_usage_stats=memory_stats,
            error_log=self.error_log.copy(),
            batch_config=self.config
        )


class MemoryMonitor:
    """Monitor memory usage during batch processing"""
    
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.memory_snapshots = []
    
    def record_pre_processing(self):
        """Record memory before processing"""
        memory_info = self.process.memory_info()
        self.memory_snapshots.append({
            'timestamp': time.time(),
            'phase': 'pre_processing',
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024
        })
    
    def record_post_processing(self):
        """Record memory after processing"""
        memory_info = self.process.memory_info()
        self.memory_snapshots.append({
            'timestamp': time.time(),
            'phase': 'post_processing',
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024
        })
    
    def get_summary(self) -> Dict[str, float]:
        """Get memory usage summary"""
        if not self.memory_snapshots:
            return {}
        
        rss_values = [s['rss_mb'] for s in self.memory_snapshots]
        return {
            'peak_memory_mb': max(rss_values),
            'final_memory_mb': rss_values[-1],
            'memory_growth_mb': rss_values[-1] - rss_values[0] if len(rss_values) > 1 else 0.0
        }


class ProgressTracker:
    """Track progress during batch processing"""
    
    def __init__(self):
        self.total_items = 0
        self.processed_items = 0
        self.batch_size = 0
        self.start_time = 0
        self.last_update = 0
    
    def initialize(self, total_items: int, batch_size: int):
        """Initialize progress tracking"""
        self.total_items = total_items
        self.processed_items = 0
        self.batch_size = batch_size
        self.start_time = time.time()
        self.last_update = self.start_time
        
        print(f"Starting batch processing: {total_items} items in batches of {batch_size}")
    
    def update_batch_completed(self, batch_items: int):
        """Update progress after batch completion"""
        self.processed_items += batch_items
        current_time = time.time()
        
        # Report progress every 10 seconds or after significant progress
        if (current_time - self.last_update > 10.0 or 
            self.processed_items >= self.total_items):
            
            progress_pct = 100.0 * self.processed_items / self.total_items
            elapsed_time = current_time - self.start_time
            rate = self.processed_items / elapsed_time if elapsed_time > 0 else 0.0
            
            print(f"Progress: {self.processed_items}/{self.total_items} "
                  f"({progress_pct:.1f}%) - {rate:.1f} items/sec")
            
            self.last_update = current_time


# Convenience functions

def batch_process_texts(texts: List[str],
                       batch_size: int = 50,
                       max_concurrent: int = 4,
                       output_dir: Optional[str] = None) -> BatchResult:
    """
    Convenience function for batch processing large text corpora.
    
    Args:
        texts: List of texts to process
        batch_size: Size of each processing batch
        max_concurrent: Maximum concurrent processing batches
        output_dir: Directory to save results
        
    Returns:
        Batch processing result
    """
    config = BatchConfiguration(
        batch_size=batch_size,
        max_concurrent_batches=max_concurrent,
        output_directory=output_dir,
        save_intermediate_results=output_dir is not None
    )
    
    processor = BatchProcessor(config)
    return processor.process_text_corpus(texts)