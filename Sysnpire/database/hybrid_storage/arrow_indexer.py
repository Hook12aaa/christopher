"""
Arrow Indexer - Queryable Metadata Storage

Manages Arrow/Parquet storage for fast queryable metadata and spatial indexing.
Provides rapid charge discovery, filtering, and selection capabilities while
the HDF5 system handles complete mathematical storage.

Key Features:
- Fast columnar queries for charge discovery
- Spatial indexing for field proximity searches
- Metadata filtering and selection
- Integration with existing query patterns
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ArrowIndexer:
    """
    Arrow/Parquet indexer for fast metadata queries.

    Provides rapid queryable access to universe and agent metadata
    while the HDF5 system handles complete mathematical storage.
    """

    def __init__(self, storage_path: Union[str, Path], batch_size: int = 1000):
        """
        Initialize Arrow indexer.

        Args:
            storage_path: Path for Arrow/Parquet storage
            batch_size: Batch size for operations
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

        logger.info("ArrowIndexer initialized")
        logger.info(f"  Storage path: {self.storage_path}")
        logger.info(f"  Batch size: {batch_size}")

    def store_universe_metadata(self, universe_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store universe metadata in Arrow/Parquet format."""
        logger.info(f"📊 Storing Arrow metadata for universe: {universe_id}")

        try:
            # Store universe metadata
            universe_file = self.storage_path / f"{universe_id}_universe.parquet"
            universe_df = pd.DataFrame([metadata["universe_metadata"]])
            universe_df.to_parquet(universe_file, index=False)

            # Store agent metadata
            if metadata["agent_metadata"]:
                agent_file = self.storage_path / f"{universe_id}_agents.parquet"
                agent_df = pd.DataFrame(metadata["agent_metadata"])
                agent_df.to_parquet(agent_file, index=False)

            logger.info("✅ Arrow metadata stored successfully")

            return {
                "status": "success",
                "universe_file": str(universe_file),
                "agent_file": str(agent_file) if metadata["agent_metadata"] else None,
                "agent_count": len(metadata["agent_metadata"]),
            }

        except Exception as e:
            logger.error(f"Arrow metadata storage failed: {e}")
            return {"status": "failed", "error": str(e)}

    def get_universe_metadata(self, universe_id: str) -> Dict[str, Any]:
        """Get universe metadata."""
        universe_file = self.storage_path / f"{universe_id}_universe.parquet"
        agent_file = self.storage_path / f"{universe_id}_agents.parquet"

        metadata = {}

        if universe_file.exists():
            universe_df = pd.read_parquet(universe_file)
            metadata["universe_metadata"] = universe_df.iloc[0].to_dict()

        if agent_file.exists():
            agent_df = pd.read_parquet(agent_file)
            metadata["agent_metadata"] = agent_df.to_dict("records")
        else:
            metadata["agent_metadata"] = []

        return metadata

    def list_universes(self) -> List[str]:
        """List all universe IDs."""
        universe_files = list(self.storage_path.glob("*_universe.parquet"))
        return [f.stem.replace("_universe", "") for f in universe_files]

    def query_charges(
        self,
        text_filter: Optional[str] = None,
        magnitude_range: Optional[Tuple[float, float]] = None,
        universe_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query charges using fast metadata."""
        logger.info("🔍 Querying charges via Arrow metadata...")

        # Implement basic query logic
        results = []

        universe_files = (
            [self.storage_path / f"{universe_id}_agents.parquet"]
            if universe_id
            else list(self.storage_path.glob("*_agents.parquet"))
        )

        for agent_file in universe_files:
            if not agent_file.exists():
                continue

            try:
                df = pd.read_parquet(agent_file)

                # Apply filters
                if text_filter:
                    df = df[df["text_source"].str.contains(text_filter, case=False, na=False)]

                if magnitude_range:
                    min_mag, max_mag = magnitude_range
                    if "Q_magnitude" in df.columns:
                        df = df[(df["Q_magnitude"] >= min_mag) & (df["Q_magnitude"] <= max_mag)]

                if limit and len(results) + len(df) > limit:
                    remaining = limit - len(results)
                    df = df.head(remaining)

                results.extend(df.to_dict("records"))

                if limit and len(results) >= limit:
                    break

            except Exception as e:
                logger.warning(f"Could not query {agent_file}: {e}")

        logger.info(f"🔍 Query complete: {len(results)} results")
        return results

    def get_universe_size(self, universe_id: str) -> float:
        """Get storage size for universe in MB."""
        total_size = 0

        universe_file = self.storage_path / f"{universe_id}_universe.parquet"
        agent_file = self.storage_path / f"{universe_id}_agents.parquet"

        for file_path in [universe_file, agent_file]:
            if file_path.exists():
                total_size += file_path.stat().st_size

        return total_size / (1024 * 1024)

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get Arrow storage statistics."""
        files = list(self.storage_path.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files)

        return {
            "total_files": len(files),
            "total_size_mb": total_size / (1024 * 1024),
            "storage_path": str(self.storage_path),
        }


if __name__ == "__main__":
    indexer = ArrowIndexer("/tmp/test_arrow")
    print("ArrowIndexer ready for fast metadata queries")
