"""
Evolution Manager - Internet Data Acceptance/Rejection System

Manages the evolution of the semantic manifold through selective acceptance
of internet data based on field theory compatibility and resonance patterns.

Evolution Mechanism:
1. Seed Universe: BGE embeddings provide foundational semantic field
2. Internet Ingestion: Scrape/receive new content from web sources  
3. Field Analysis: Calculate resonance with existing manifold
4. Acceptance/Rejection: Accept content that enhances field coherence
5. Manifold Evolution: Integrate accepted content as new conceptual charges
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import asyncio
import aiohttp
from datetime import datetime

# Core imports
from .field_universe import FieldUniverse
from .conceptual_charge_object import ConceptualChargeObject
from ..model.charge_factory import ChargeFactory, ChargeParameters

# Import actual ConceptualCharge from model layer for field calculations
try:
    from ..model.mathematics.conceptual_charge import ConceptualCharge
    CONCEPTUAL_CHARGE_AVAILABLE = True
except ImportError:
    # Fallback during development
    ConceptualCharge = ConceptualChargeObject
    CONCEPTUAL_CHARGE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EvolutionMetrics:
    """Metrics for tracking manifold evolution."""
    total_content_evaluated: int = 0
    content_accepted: int = 0
    content_rejected: int = 0
    acceptance_rate: float = 0.0
    field_coherence_score: float = 0.0
    manifold_complexity: float = 0.0
    last_evolution_time: float = 0.0

@dataclass
class ContentSource:
    """Configuration for internet content sources."""
    url: str
    source_type: str  # "rss", "api", "scrape", "stream"
    priority: float = 1.0
    field_affinity: str = "general"  # Target field region
    rate_limit: float = 1.0  # Requests per second
    enabled: bool = True

@dataclass
class AcceptanceCriteria:
    """Criteria for accepting new content into the manifold."""
    # Field resonance thresholds
    min_field_resonance: float = 0.3
    max_field_disruption: float = 0.7
    
    # Semantic compatibility
    min_semantic_coherence: float = 0.4
    max_semantic_redundancy: float = 0.9
    
    # Content quality filters
    min_content_length: int = 50
    max_content_length: int = 5000
    language_filter: List[str] = None
    
    # Manifold health preservation
    max_acceptance_rate: float = 0.1  # Max 10% of evaluated content
    field_balance_threshold: float = 0.8  # Maintain field region balance

    def __post_init__(self):
        if self.language_filter is None:
            self.language_filter = ["en"]

class EvolutionManager:
    """
    Manages the evolution of the semantic manifold through selective internet data ingestion.
    
    Evolution Strategy:
    - Start with BGE seed universe as foundation
    - Continuously evaluate internet content for field compatibility
    - Accept content that enhances manifold coherence
    - Reject content that disrupts field structure
    - Maintain dynamic equilibrium in the semantic field
    """
    
    def __init__(self, 
                 universe: FieldUniverse,
                 charge_factory: ChargeFactory,
                 criteria: Optional[AcceptanceCriteria] = None):
        """
        Initialize evolution manager.
        
        Args:
            universe: Field universe to evolve
            charge_factory: Factory for creating new charges
            criteria: Acceptance criteria for content filtering
        """
        self.universe = universe
        self.charge_factory = charge_factory
        self.criteria = criteria or AcceptanceCriteria()
        
        # Evolution state
        self.metrics = EvolutionMetrics()
        self.content_sources: List[ContentSource] = []
        self.evolution_history: List[Dict[str, Any]] = []
        
        # Processing queues
        self.evaluation_queue: asyncio.Queue = asyncio.Queue()
        self.integration_queue: asyncio.Queue = asyncio.Queue()
        
        # Evolution control
        self.evolution_active = False
        self.evolution_task: Optional[asyncio.Task] = None
        
        logger.info("🧬 Evolution Manager initialized for manifold evolution")
    
    def add_content_source(self, source: ContentSource):
        """Add a new content source for evolution."""
        self.content_sources.append(source)
        logger.info(f"Added content source: {source.url} ({source.source_type})")
    
    async def start_evolution(self):
        """Start the continuous evolution process."""
        if self.evolution_active:
            logger.warning("Evolution already active")
            return
        
        self.evolution_active = True
        logger.info("🌱 Starting manifold evolution process...")
        
        # Start parallel evolution tasks
        tasks = [
            asyncio.create_task(self._content_ingestion_loop()),
            asyncio.create_task(self._evaluation_loop()),
            asyncio.create_task(self._integration_loop()),
            asyncio.create_task(self._health_monitoring_loop())
        ]
        
        self.evolution_task = asyncio.gather(*tasks)
        
        try:
            await self.evolution_task
        except asyncio.CancelledError:
            logger.info("Evolution process cancelled")
        except Exception as e:
            logger.error(f"Evolution process failed: {e}")
        finally:
            self.evolution_active = False
    
    async def stop_evolution(self):
        """Stop the evolution process."""
        logger.info("🛑 Stopping manifold evolution...")
        self.evolution_active = False
        
        if self.evolution_task:
            self.evolution_task.cancel()
            try:
                await self.evolution_task
            except asyncio.CancelledError:
                pass
    
    async def _content_ingestion_loop(self):
        """Continuously ingest content from internet sources."""
        while self.evolution_active:
            try:
                for source in self.content_sources:
                    if not source.enabled:
                        continue
                    
                    # Respect rate limits
                    await asyncio.sleep(1.0 / source.rate_limit)
                    
                    # Fetch content based on source type
                    content = await self._fetch_content(source)
                    
                    if content:
                        # Add to evaluation queue
                        await self.evaluation_queue.put({
                            'content': content,
                            'source': source,
                            'timestamp': time.time()
                        })
                        
                        logger.debug(f"Queued content from {source.url}")
                
                # Brief pause between ingestion cycles
                await asyncio.sleep(30)  # 30 second cycles
                
            except Exception as e:
                logger.error(f"Content ingestion error: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def _fetch_content(self, source: ContentSource) -> Optional[Dict[str, Any]]:
        """Fetch content from a specific source."""
        try:
            if source.source_type == "rss":
                return await self._fetch_rss_content(source)
            elif source.source_type == "api":
                return await self._fetch_api_content(source)
            elif source.source_type == "scrape":
                return await self._fetch_scraped_content(source)
            else:
                logger.warning(f"Unknown source type: {source.source_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch content from {source.url}: {e}")
            return None
    
    async def _fetch_rss_content(self, source: ContentSource) -> Optional[Dict[str, Any]]:
        """Fetch content from RSS feed."""
        # Placeholder - implement RSS parsing
        # Use feedparser or similar library
        return {
            'text': "Sample RSS content",
            'title': "Sample Title",
            'url': source.url,
            'metadata': {'source_type': 'rss'}
        }
    
    async def _fetch_api_content(self, source: ContentSource) -> Optional[Dict[str, Any]]:
        """Fetch content from API endpoint."""
        async with aiohttp.ClientSession() as session:
            async with session.get(source.url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'text': data.get('text', ''),
                        'title': data.get('title', ''),
                        'url': source.url,
                        'metadata': {'source_type': 'api', 'response_data': data}
                    }
        return None
    
    async def _fetch_scraped_content(self, source: ContentSource) -> Optional[Dict[str, Any]]:
        """Fetch content via web scraping."""
        # Placeholder - implement web scraping
        # Use BeautifulSoup or similar library
        return {
            'text': "Sample scraped content",
            'title': "Sample Scraped Title", 
            'url': source.url,
            'metadata': {'source_type': 'scrape'}
        }
    
    async def _evaluation_loop(self):
        """Evaluate queued content for acceptance into the manifold."""
        while self.evolution_active:
            try:
                # Get content from queue with timeout
                try:
                    item = await asyncio.wait_for(
                        self.evaluation_queue.get(), 
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                content = item['content']
                source = item['source']
                
                # Evaluate content for acceptance
                evaluation_result = await self._evaluate_content(content, source)
                self.metrics.total_content_evaluated += 1
                
                if evaluation_result['accept']:
                    # Add to integration queue
                    await self.integration_queue.put({
                        'content': content,
                        'source': source,
                        'evaluation': evaluation_result,
                        'timestamp': time.time()
                    })
                    
                    self.metrics.content_accepted += 1
                    logger.info(f"✅ Accepted content: {content.get('title', 'Unknown')[:50]}")
                else:
                    self.metrics.content_rejected += 1
                    rejection_reason = evaluation_result.get('rejection_reason', 'Unknown')
                    logger.debug(f"❌ Rejected content: {rejection_reason}")
                
                # Update acceptance rate
                if self.metrics.total_content_evaluated > 0:
                    self.metrics.acceptance_rate = (
                        self.metrics.content_accepted / self.metrics.total_content_evaluated
                    )
                
            except Exception as e:
                logger.error(f"Content evaluation error: {e}")
                await asyncio.sleep(1)
    
    async def _evaluate_content(self, content: Dict[str, Any], source: ContentSource) -> Dict[str, Any]:
        """
        Evaluate content for acceptance into the manifold.
        
        Returns:
            Dictionary with 'accept' boolean and evaluation metrics
        """
        try:
            text = content.get('text', '')
            
            # Basic content quality checks
            if len(text) < self.criteria.min_content_length:
                return {'accept': False, 'rejection_reason': 'content_too_short'}
            
            if len(text) > self.criteria.max_content_length:
                return {'accept': False, 'rejection_reason': 'content_too_long'}
            
            # Create temporary charge for field analysis
            temp_charge_params = ChargeParameters(
                observational_state=1.0,
                gamma=1.0,
                context=f"evaluation_{source.source_type}"
            )
            
            # Use charge factory to create temporary charge
            # This will compute BGE embedding and field properties
            embedding_result = self._create_embedding_for_content(text)
            if not embedding_result:
                return {'accept': False, 'rejection_reason': 'embedding_failed'}
            
            embedding, manifold_properties = embedding_result
            
            # Create temporary charge for field analysis
            temp_charge = self.charge_factory.create_charge(
                embedding=embedding,
                manifold_properties=manifold_properties,
                charge_params=temp_charge_params,
                metadata={'temporary_evaluation': True}
            )
            
            # Analyze field resonance with existing universe
            field_resonance = self._calculate_field_resonance(temp_charge)
            field_disruption = self._calculate_field_disruption(temp_charge)
            semantic_coherence = self._calculate_semantic_coherence(temp_charge)
            semantic_redundancy = self._calculate_semantic_redundancy(temp_charge)
            
            # Apply acceptance criteria
            evaluation_metrics = {
                'field_resonance': field_resonance,
                'field_disruption': field_disruption,
                'semantic_coherence': semantic_coherence,
                'semantic_redundancy': semantic_redundancy
            }
            
            # Decision logic
            accept = (
                field_resonance >= self.criteria.min_field_resonance and
                field_disruption <= self.criteria.max_field_disruption and
                semantic_coherence >= self.criteria.min_semantic_coherence and
                semantic_redundancy <= self.criteria.max_semantic_redundancy and
                self.metrics.acceptance_rate < self.criteria.max_acceptance_rate
            )
            
            rejection_reason = None
            if not accept:
                if field_resonance < self.criteria.min_field_resonance:
                    rejection_reason = 'low_field_resonance'
                elif field_disruption > self.criteria.max_field_disruption:
                    rejection_reason = 'high_field_disruption'
                elif semantic_coherence < self.criteria.min_semantic_coherence:
                    rejection_reason = 'low_semantic_coherence'
                elif semantic_redundancy > self.criteria.max_semantic_redundancy:
                    rejection_reason = 'high_semantic_redundancy'
                elif self.metrics.acceptance_rate >= self.criteria.max_acceptance_rate:
                    rejection_reason = 'acceptance_rate_limit'
            
            return {
                'accept': accept,
                'rejection_reason': rejection_reason,
                'metrics': evaluation_metrics,
                'temp_charge': temp_charge if accept else None
            }
            
        except Exception as e:
            logger.error(f"Content evaluation failed: {e}")
            return {'accept': False, 'rejection_reason': 'evaluation_error'}
    
    def _create_embedding_for_content(self, text: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Create embedding and manifold properties for content text."""
        try:
            # CLAUDE.md Compliance: Use actual BGE model, no simulated outputs
            from ..model.intial.bge_ingestion import BGEIngestion
            
            # Initialize BGE ingestion if not already available
            if not hasattr(self, '_bge_ingestion'):
                self._bge_ingestion = BGEIngestion()
            
            # Generate actual BGE embedding
            embedding = self._bge_ingestion.encode_single_text(text)
            
            # Extract actual manifold properties using BGE ingestion methods
            manifold_properties = self._bge_ingestion.extract_manifold_properties(
                embedding=embedding,
                index=0,  # Single text processing
                all_embeddings=embedding.reshape(1, -1),
                pca=None,  # Will be handled internally
                knn_model=None  # Will be handled internally
            )
            
            return embedding, manifold_properties
            
        except Exception as e:
            logger.error(f"BGE embedding creation failed: {e}")
            logger.error("CLAUDE.md Violation: Cannot use fake embeddings - BGE model required")
            return None
    
    def _calculate_field_resonance(self, charge) -> float:
        """Calculate how well the charge resonates with existing field."""
        try:
            # Find nearby charges in the universe
            if len(self.universe.charges) == 0:
                return 1.0  # First charge always resonates
            
            # Calculate average resonance with existing charges
            resonances = []
            charge_magnitude = abs(charge.compute_complete_charge())
            
            for existing_charge in list(self.universe.charges.values())[:10]:  # Sample 10 charges
                try:
                    existing_magnitude = abs(existing_charge.compute_complete_charge())
                    
                    # Simple resonance based on magnitude similarity
                    magnitude_similarity = 1.0 - abs(charge_magnitude - existing_magnitude) / max(charge_magnitude, existing_magnitude, 1e-6)
                    resonances.append(magnitude_similarity)
                    
                except Exception:
                    continue
            
            return np.mean(resonances) if resonances else 0.5
            
        except Exception as e:
            logger.error(f"Field resonance calculation failed: {e}")
            return 0.0
    
    def _calculate_field_disruption(self, charge) -> float:
        """Calculate how much the charge would disrupt existing field patterns."""
        try:
            # Measure potential disruption to field coherence
            if len(self.universe.charges) < 5:
                return 0.1  # Low disruption for small universes
            
            # Sample field coherence before and after (simulation)
            current_coherence = self._measure_field_coherence()
            
            # Estimate disruption (simplified)
            charge_magnitude = abs(charge.compute_complete_charge())
            avg_universe_magnitude = np.mean([
                abs(c.compute_complete_charge()) 
                for c in list(self.universe.charges.values())[:10]
            ])
            
            # High magnitude difference = high disruption potential
            magnitude_disruption = abs(charge_magnitude - avg_universe_magnitude) / max(avg_universe_magnitude, 1e-6)
            
            return min(magnitude_disruption, 1.0)
            
        except Exception as e:
            logger.error(f"Field disruption calculation failed: {e}")
            return 1.0  # Conservative: assume high disruption on error
    
    def _calculate_semantic_coherence(self, charge) -> float:
        """Calculate semantic coherence with existing content."""
        try:
            # Measure how semantically coherent the new charge is
            # This would use embedding similarity with existing charges
            
            if len(self.universe.charges) == 0:
                return 1.0
            
            # Sample existing charges for comparison
            sample_charges = list(self.universe.charges.values())[:20]
            
            # Calculate semantic similarity (placeholder)
            similarities = []
            for existing_charge in sample_charges:
                try:
                    # This would calculate actual embedding similarity
                    # For now, use magnitude similarity as proxy
                    new_mag = abs(charge.compute_complete_charge())
                    existing_mag = abs(existing_charge.compute_complete_charge())
                    
                    similarity = 1.0 - abs(new_mag - existing_mag) / max(new_mag, existing_mag, 1e-6)
                    similarities.append(similarity)
                    
                except Exception:
                    continue
            
            return np.mean(similarities) if similarities else 0.5
            
        except Exception as e:
            logger.error(f"Semantic coherence calculation failed: {e}")
            return 0.0
    
    def _calculate_semantic_redundancy(self, charge) -> float:
        """Calculate how redundant the charge is with existing content."""
        try:
            # Measure redundancy with existing charges
            if len(self.universe.charges) == 0:
                return 0.0  # No redundancy in empty universe
            
            # Find most similar existing charge
            max_similarity = 0.0
            charge_magnitude = abs(charge.compute_complete_charge())
            
            for existing_charge in list(self.universe.charges.values())[:50]:  # Sample 50 charges
                try:
                    existing_magnitude = abs(existing_charge.compute_complete_charge())
                    
                    # Calculate similarity (this would use actual embeddings)
                    similarity = 1.0 - abs(charge_magnitude - existing_magnitude) / max(charge_magnitude, existing_magnitude, 1e-6)
                    max_similarity = max(max_similarity, similarity)
                    
                except Exception:
                    continue
            
            return max_similarity
            
        except Exception as e:
            logger.error(f"Semantic redundancy calculation failed: {e}")
            return 1.0  # Conservative: assume high redundancy on error
    
    def _measure_field_coherence(self) -> float:
        """Measure overall field coherence of the universe."""
        try:
            if len(self.universe.charges) < 2:
                return 1.0
            
            # Calculate variance in charge magnitudes as coherence measure
            magnitudes = []
            for charge in list(self.universe.charges.values())[:100]:  # Sample 100 charges
                try:
                    magnitude = abs(charge.compute_complete_charge())
                    magnitudes.append(magnitude)
                except Exception:
                    continue
            
            if len(magnitudes) < 2:
                return 1.0
            
            # Lower variance = higher coherence
            variance = np.var(magnitudes)
            mean_magnitude = np.mean(magnitudes)
            
            # Normalize coherence (0 = chaotic, 1 = perfectly coherent)
            relative_variance = variance / max(mean_magnitude**2, 1e-12)
            coherence = 1.0 / (1.0 + relative_variance)
            
            return coherence
            
        except Exception as e:
            logger.error(f"Field coherence measurement failed: {e}")
            return 0.5
    
    async def _integration_loop(self):
        """Integrate accepted content into the manifold."""
        while self.evolution_active:
            try:
                # Get accepted content from queue
                try:
                    item = await asyncio.wait_for(
                        self.integration_queue.get(), 
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                content = item['content']
                evaluation = item['evaluation']
                
                # Get the pre-evaluated charge
                charge = evaluation.get('temp_charge')
                if not charge:
                    logger.error("No charge found for integration")
                    continue
                
                # Remove temporary flag
                if hasattr(charge, 'metadata'):
                    charge.metadata.pop('temporary_evaluation', None)
                
                # Add charge to universe
                success = self.universe.add_charge(charge)
                
                if success:
                    # Log evolution event
                    evolution_event = {
                        'timestamp': time.time(),
                        'action': 'charge_integrated',
                        'content_title': content.get('title', 'Unknown'),
                        'content_url': content.get('url', ''),
                        'source_type': item['source'].source_type,
                        'evaluation_metrics': evaluation.get('metrics', {}),
                        'universe_size': len(self.universe.charges)
                    }
                    
                    self.evolution_history.append(evolution_event)
                    self.metrics.last_evolution_time = time.time()
                    
                    logger.info(f"🧬 Integrated charge into manifold: {content.get('title', 'Unknown')[:50]}")
                    logger.info(f"   Universe size: {len(self.universe.charges)} charges")
                else:
                    logger.error(f"Failed to integrate charge: {content.get('title', 'Unknown')}")
                
            except Exception as e:
                logger.error(f"Content integration error: {e}")
                await asyncio.sleep(1)
    
    async def _health_monitoring_loop(self):
        """Monitor manifold health during evolution."""
        while self.evolution_active:
            try:
                # Update field coherence
                self.metrics.field_coherence_score = self._measure_field_coherence()
                
                # Calculate manifold complexity
                self.metrics.manifold_complexity = len(self.universe.charges) / 1000.0  # Normalize to thousands
                
                # Log health metrics periodically
                logger.info(f"📊 Manifold Health - Coherence: {self.metrics.field_coherence_score:.3f}, "
                           f"Size: {len(self.universe.charges)}, "
                           f"Acceptance Rate: {self.metrics.acceptance_rate:.3f}")
                
                # Check for unhealthy conditions
                if self.metrics.field_coherence_score < 0.3:
                    logger.warning("⚠️ Low field coherence detected - consider reducing acceptance rate")
                
                if self.metrics.acceptance_rate > 0.2:
                    logger.warning("⚠️ High acceptance rate - manifold may be growing too rapidly")
                
                # Sleep between health checks
                await asyncio.sleep(300)  # 5 minute intervals
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status and metrics."""
        return {
            'evolution_active': self.evolution_active,
            'metrics': {
                'total_evaluated': self.metrics.total_content_evaluated,
                'accepted': self.metrics.content_accepted,
                'rejected': self.metrics.content_rejected,
                'acceptance_rate': self.metrics.acceptance_rate,
                'field_coherence': self.metrics.field_coherence_score,
                'manifold_complexity': self.metrics.manifold_complexity,
                'last_evolution': self.metrics.last_evolution_time
            },
            'universe_size': len(self.universe.charges),
            'content_sources': len(self.content_sources),
            'recent_evolution': self.evolution_history[-10:] if self.evolution_history else []
        }
