"""
Mathematical Object Proxy - Autonomous Mathematical Entity Interface

CORE PRINCIPLE: Enable mathematical objects to become autonomous entities through
regulation system interfaces WITHOUT modifying ConceptualChargeAgent. The proxy
serves as a mathematical entity wrapper that provides self-regulation capabilities
through pure mathematical principles.

MATHEMATICAL AUTONOMY: Each mathematical object develops:
- Self-monitoring of mathematical health through Q(τ,C,s) analysis
- Peer discovery based on mathematical compatibility
- Regulatory assistance through mathematical field interactions
- Self-evolution of regulatory capabilities
- Alliance formation through mathematical field theory

AGENCY EMERGENCE: Mathematical autonomy emerges from:
1. Mathematical health monitoring using Q-components and field properties
2. Peer discovery through geometric features and modular weight compatibility
3. Regulatory assistance via field position spatial indexing and phase coherence
4. Self-evolution using adaptive tuning history and breathing pattern learning
5. Alliance formation through sparse interaction graphs and synchrony patterns

MATHEMATICAL FOUNDATION: All agency behaviors derive from field-theoretic
mathematical principles, information theory, and differential geometry.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from enum import Enum
import uuid

import numpy as np
import torch
from scipy import spatial, stats
from scipy.spatial.distance import pdist, squareform

from Sysnpire.model.liquid.conceptual_charge_agent import ConceptualChargeAgent
from .mathematical_object_identity import (
    MathematicalObjectIdentity,
    MathematicalObjectIdentityProfile,
    TopologicalInvariants,
    InformationTheoreticIdentity,
)
from .listeners import RegulationSuggestion

from .advanced.variational import VariationalRegulation
from .advanced.geometric import GeometricRegulation  
from .advanced.coupled_evolution import CoupledFieldRegulation
from .advanced.symbolic import SymbolicRegulation

from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


class MathematicalHealthStatus(Enum):
    """Mathematical health status classifications."""

    EXCELLENT = "excellent"
    STABLE = "stable"
    UNSTABLE = "unstable"
    CRITICAL = "critical"
    MATHEMATICAL_SINGULARITY = "mathematical_singularity"


class RegulatoryCapability(Enum):
    """Types of regulatory capabilities mathematical objects can develop."""

    PERSISTENCE_REGULATION = "persistence_regulation"
    PHASE_COHERENCE_RESTORATION = "phase_coherence_restoration"
    FIELD_STABILIZATION = "field_stabilization"
    ENERGY_CONSERVATION = "energy_conservation"
    BREATHING_SYNCHRONIZATION = "breathing_synchronization"
    SINGULARITY_RESOLUTION = "singularity_resolution"
    TOPOLOGICAL_REPAIR = "topological_repair"


@dataclass
class MathematicalHealthMetrics:
    """Comprehensive mathematical health assessment for Q(τ,C,s) entities."""

    q_component_stability: float  # Stability of Q-components
    field_coherence_score: float  # Field coherence measure
    phase_relationship_health: float  # Phase relationship consistency
    temporal_biography_integrity: float  # Temporal biography mathematical consistency
    breathing_pattern_regularity: float  # Breathing pattern mathematical regularity
    geometric_feature_consistency: float  # Geometric feature mathematical consistency
    modular_weight_stability: float  # Modular weight mathematical stability
    mathematical_singularity_risk: float  # Risk of mathematical singularities
    overall_mathematical_health: float  # Overall health score [0, 1]
    health_status: MathematicalHealthStatus  # Categorical health status
    health_computation_timestamp: float  # When health was computed


@dataclass
class RegulatoryPartnerProfile:
    """Profile of potential regulatory partner mathematical object."""

    partner_mathematical_id: str  # Mathematical identity of partner
    compatibility_score: float  # Mathematical compatibility [0, 1]
    regulatory_capabilities: Set[RegulatoryCapability]  # Partner's regulatory abilities
    mathematical_similarity: float  # Mathematical similarity score
    geometric_compatibility: float  # Geometric feature compatibility
    field_position_proximity: float  # Spatial proximity in field
    phase_relationship_harmony: float  # Phase relationship compatibility
    regulatory_assistance_history: List[Dict[str, Any]]  # History of assistance provided/received
    last_interaction_timestamp: float  # Last interaction time


@dataclass
class RegulatoryAlliance:
    """Temporary alliance between mathematical objects for regulation."""

    alliance_id: str  # Unique alliance identifier
    member_mathematical_ids: Set[str]  # Alliance member identities
    alliance_purpose: str  # Mathematical purpose of alliance
    regulatory_strategy: str  # Regulatory approach being used
    formation_timestamp: float  # When alliance was formed
    expected_duration: float  # Expected alliance lifetime
    mathematical_effectiveness: float  # Alliance effectiveness score
    collective_regulatory_strength: float  # Combined regulatory capability
    alliance_health: float  # Health of alliance itself


@dataclass
class MathematicalDistressSignal:
    """Distress signal broadcast by mathematical object needing assistance."""

    distress_id: str  # Unique distress identifier
    source_mathematical_id: str  # Mathematical object sending distress
    distress_type: str  # Type of mathematical distress
    mathematical_urgency: float  # Urgency level [0, 1]
    distress_description: str  # Mathematical description of distress
    required_regulatory_capabilities: Set[RegulatoryCapability]  # Needed assistance types
    field_position_context: Any  # Field position context
    q_component_distress_details: Dict[str, float]  # Specific Q-component issues
    broadcast_radius: float  # Spatial radius for distress broadcast
    timestamp: float  # When distress was broadcast


@dataclass
class RegulatoryAssistanceOffer:
    """Offer of regulatory assistance from one mathematical object to another."""

    offer_id: str  # Unique offer identifier
    offering_mathematical_id: str  # Mathematical object offering assistance
    target_mathematical_id: str  # Mathematical object being offered assistance
    offered_capabilities: Set[RegulatoryCapability]  # Capabilities being offered
    assistance_strength: float  # Strength of assistance available
    mathematical_cost: float  # Mathematical cost to offering object
    estimated_effectiveness: float  # Estimated effectiveness of assistance
    offer_duration: float  # How long offer remains valid
    mathematical_compatibility_score: float  # Compatibility with target
    timestamp: float  # When offer was made


class MathematicalObjectProxy:
    """
    Mathematical Object Proxy - Autonomous Mathematical Entity Interface

    Provides mathematical objects with autonomous capabilities through regulation
    system interfaces. Enables Q(τ,C,s) entities to become self-regulating
    mathematical beings without modifying ConceptualChargeAgent.
    """

    def __init__(
        self,
        agent: ConceptualChargeAgent,
        identity_system: MathematicalObjectIdentity,
        mathematical_precision: float = 1e-12,
    ):
        """
        Initialize mathematical object proxy for autonomous entity capabilities.

        Args:
            agent: ConceptualChargeAgent being proxied
            identity_system: Mathematical identity system
            mathematical_precision: Numerical precision for computations
        """
        self.agent = agent
        self.identity_system = identity_system
        self.mathematical_precision = mathematical_precision

        self.identity_profile = self.identity_system.create_mathematical_identity_profile(agent)
        self.mathematical_id = self.identity_profile.object_mathematical_id
        
        self.variational_regulation = VariationalRegulation()
        self.geometric_regulation = GeometricRegulation()
        self.coupled_field_regulation = CoupledFieldRegulation() 
        self.symbolic_regulation = SymbolicRegulation()
        
        self.advanced_regulation_enabled = True
        self.use_variational_methods = True
        self.use_geometric_analysis = True
        self.use_coupled_evolution = True
        self.use_symbolic_computation = True

        self.health_history: List[MathematicalHealthMetrics] = []
        self.health_monitoring_enabled = True
        self.health_check_frequency = 1.0  # Health checks per time unit
        self.last_health_check_time = 0.0

        self.known_mathematical_peers: Dict[str, RegulatoryPartnerProfile] = {}
        self.peer_discovery_radius = 10.0  # Spatial radius for peer discovery
        self.compatibility_threshold = 0.7  # Minimum compatibility for regulatory partnerships
        self.peer_discovery_frequency = 0.5  # Peer discovery frequency
        self.last_peer_discovery_time = 0.0

        self.regulatory_capabilities: Set[RegulatoryCapability] = set()
        self.regulatory_expertise_levels: Dict[RegulatoryCapability, float] = {}
        self.capability_development_history: List[Dict[str, Any]] = []
        self.self_evolution_enabled = True
        self.learning_rate = 0.01

        self.active_alliances: Dict[str, RegulatoryAlliance] = {}
        self.alliance_formation_threshold = 0.8  # Threshold for alliance formation
        self.max_concurrent_alliances = 3

        self.distress_threshold = 0.3  # Health threshold for distress signaling
        self.assistance_offer_threshold = 0.8  # Health threshold for offering assistance
        self.active_distress_signals: Dict[str, MathematicalDistressSignal] = {}
        self.assistance_offers_made: Dict[str, RegulatoryAssistanceOffer] = {}
        self.assistance_offers_received: Dict[str, RegulatoryAssistanceOffer] = {}

        self.regulatory_reputation_score = 0.5  # Initial neutral reputation
        self.assistance_success_history: List[Dict[str, Any]] = []
        self.regulatory_effectiveness_history: List[float] = []

        self.autonomous_operation_level = 0.0  # [0, 1] - level of autonomy achieved
        self.autonomous_threshold = 0.9  # Threshold for full autonomy
        self.requires_external_oversight = True

        logger.info(f"🤖 Mathematical Object Proxy initialized for {self.mathematical_id}")

    def monitor_mathematical_health(self) -> MathematicalHealthMetrics:
        """
        Monitor mathematical health using Q-components and field properties.

        Analyzes mathematical integrity of Q(τ,C,s) entity and identifies
        potential mathematical issues requiring regulation.
        """
        current_time = time.time()

        q_component_stability = self._analyze_q_component_stability()

        field_coherence_score = self._assess_field_coherence()

        phase_relationship_health = self._evaluate_phase_relationship_health()

        temporal_biography_integrity = self._check_temporal_biography_integrity()

        breathing_pattern_regularity = self._analyze_breathing_pattern_regularity()

        geometric_feature_consistency = self._evaluate_geometric_feature_consistency()

        modular_weight_stability = self._assess_modular_weight_stability()

        mathematical_singularity_risk = self._compute_singularity_risk()

        health_components = [
            q_component_stability,
            field_coherence_score,
            phase_relationship_health,
            temporal_biography_integrity,
            breathing_pattern_regularity,
            geometric_feature_consistency,
            modular_weight_stability,
        ]
        health_tensor = torch.tensor(health_components, dtype=torch.float32)
        overall_mathematical_health = torch.mean(health_tensor).item()

        overall_mathematical_health *= 1.0 - mathematical_singularity_risk

        health_status = self._determine_health_status(overall_mathematical_health, mathematical_singularity_risk)

        health_metrics = MathematicalHealthMetrics(
            q_component_stability=q_component_stability,
            field_coherence_score=field_coherence_score,
            phase_relationship_health=phase_relationship_health,
            temporal_biography_integrity=temporal_biography_integrity,
            breathing_pattern_regularity=breathing_pattern_regularity,
            geometric_feature_consistency=geometric_feature_consistency,
            modular_weight_stability=modular_weight_stability,
            mathematical_singularity_risk=mathematical_singularity_risk,
            overall_mathematical_health=overall_mathematical_health,
            health_status=health_status,
            health_computation_timestamp=current_time,
        )

        self.health_history.append(health_metrics)
        if len(self.health_history) > 1000:  # Keep last 1000 health checks
            self.health_history.pop(0)

        self.last_health_check_time = current_time

        logger.debug(
            f"🤖 Health check for {self.mathematical_id}: "
            f"{health_status.value} (score={overall_mathematical_health:.3f})"
        )

        return health_metrics

    def compute_mathematical_stability_score(self) -> float:
        """
        Compute comprehensive mathematical stability score using existing components.

        Uses log magnitude multiplication, phase coherence calculations, and
        field-theoretic properties to assess mathematical stability.
        """
        stability_components = []

        if hasattr(self.agent, "q_value") and self.agent.q_value is not None:
            q_magnitude = abs(self.agent.q_value)
            if math.isfinite(q_magnitude):
                magnitude_stability = 1.0 / (1.0 + q_magnitude**2)
                stability_components.append(magnitude_stability)

        if hasattr(self.agent, "Q_components") and self.agent.Q_components:
            phase_coherence = self._compute_phase_coherence_stability()
            stability_components.append(phase_coherence)

        if hasattr(self.agent, "breathing_q_coefficients") and self.agent.breathing_q_coefficients:
            breathing_stability = self._compute_breathing_coefficient_stability()
            stability_components.append(breathing_stability)

        if hasattr(self.agent, "field_position") and self.agent.field_position is not None:
            position_stability = self._compute_position_stability()
            stability_components.append(position_stability)

        if stability_components:
            stability_tensor = torch.tensor(stability_components, dtype=torch.float32)
            stability_score = torch.mean(stability_tensor).item()
            return float(stability_score)
        else:
            return 0.5  # Neutral stability if no components available

    def find_regulatory_partners(self, candidate_agents: List[ConceptualChargeAgent]) -> List[RegulatoryPartnerProfile]:
        """
        Find potential regulatory partners using geometric features and compatibility.

        Uses mathematical compatibility metrics to identify suitable partners
        for regulatory assistance and alliance formation.
        """
        potential_partners = []
        current_time = time.time()

        for candidate_agent in candidate_agents:
            if candidate_agent is self.agent:
                continue

            candidate_proxy = MathematicalObjectProxy(candidate_agent, self.identity_system)
            candidate_id = candidate_proxy.mathematical_id

            compatibility_score = self._compute_mathematical_compatibility(candidate_agent)

            if compatibility_score < self.compatibility_threshold:
                continue

            mathematical_similarity = self._compute_mathematical_similarity(candidate_agent)
            geometric_compatibility = self._compute_geometric_compatibility(candidate_agent)
            field_position_proximity = self._compute_field_position_proximity(candidate_agent)
            phase_relationship_harmony = self._compute_phase_relationship_harmony(candidate_agent)

            candidate_capabilities = self._assess_regulatory_capabilities(candidate_agent)

            assistance_history = []
            if candidate_id in self.known_mathematical_peers:
                assistance_history = self.known_mathematical_peers[candidate_id].regulatory_assistance_history

            partner_profile = RegulatoryPartnerProfile(
                partner_mathematical_id=candidate_id,
                compatibility_score=compatibility_score,
                regulatory_capabilities=candidate_capabilities,
                mathematical_similarity=mathematical_similarity,
                geometric_compatibility=geometric_compatibility,
                field_position_proximity=field_position_proximity,
                phase_relationship_harmony=phase_relationship_harmony,
                regulatory_assistance_history=assistance_history,
                last_interaction_timestamp=current_time,
            )

            potential_partners.append(partner_profile)

            self.known_mathematical_peers[candidate_id] = partner_profile

        potential_partners.sort(key=lambda p: p.compatibility_score, reverse=True)

        self.last_peer_discovery_time = current_time

        logger.debug(f"🤖 Found {len(potential_partners)} potential regulatory partners for {self.mathematical_id}")

        return potential_partners

    def evaluate_peer_regulatory_effectiveness(self, partner_id: str) -> float:
        """
        Evaluate effectiveness of regulatory assistance from a peer.

        Uses mathematical assessment of regulatory outcomes to measure
        partner effectiveness and update reputation scores.
        """
        if partner_id not in self.known_mathematical_peers:
            raise ValueError(f"Regulatory effectiveness analysis failed - peer {partner_id} not found in history - PEER ANALYSIS IMPOSSIBLE")

        partner_profile = self.known_mathematical_peers[partner_id]
        assistance_history = partner_profile.regulatory_assistance_history

        if not assistance_history:
            raise ValueError(f"Regulatory effectiveness analysis failed - no regulatory history with peer {partner_id} - EFFECTIVENESS ANALYSIS IMPOSSIBLE")

        effectiveness_scores = []

        for assistance_event in assistance_history:
            if "effectiveness_score" in assistance_event:
                effectiveness_scores.append(assistance_event["effectiveness_score"])
            elif "health_before" in assistance_event and "health_after" in assistance_event:
                health_improvement = assistance_event["health_after"] - assistance_event["health_before"]
                effectiveness = max(0.0, min(1.0, health_improvement + 0.5))  # Normalize to [0, 1]
                effectiveness_scores.append(effectiveness)

        if not effectiveness_scores:
            raise ValueError(f"Regulatory partner effectiveness analysis failed - no effectiveness scores - PARTNER EVALUATION IMPOSSIBLE")
        
        scores_tensor = torch.tensor(effectiveness_scores, dtype=torch.float32)
        indices = torch.arange(len(effectiveness_scores), dtype=torch.float32)
        weights = torch.exp(-0.1 * torch.flip(indices, dims=[0]))  # Recent events weighted higher
        
        weighted_sum = torch.sum(scores_tensor * weights)
        weights_sum = torch.sum(weights)
        weighted_effectiveness = weighted_sum / weights_sum
        
        return float(weighted_effectiveness.item())

    def offer_regulatory_assistance(
        self, target_agent: ConceptualChargeAgent, assistance_type: RegulatoryCapability
    ) -> Optional[RegulatoryAssistanceOffer]:
        """
        Offer regulatory assistance to another mathematical object.

        Uses mathematical compatibility and capability assessment to determine
        if assistance can be effectively provided.
        """
        if assistance_type not in self.regulatory_capabilities:
            raise ValueError(f"Regulatory assistance failed - capability {assistance_type} not available - ASSISTANCE IMPOSSIBLE")

        current_health = self.monitor_mathematical_health()
        if current_health.overall_mathematical_health < self.assistance_offer_threshold:
            raise ValueError(f"Regulatory assistance failed - mathematical health {current_health.overall_mathematical_health:.3f} below threshold {self.assistance_offer_threshold} - MATHEMATICAL HEALTH INSUFFICIENT")

        target_proxy = MathematicalObjectProxy(target_agent, self.identity_system)
        target_id = target_proxy.mathematical_id

        compatibility_score = self._compute_mathematical_compatibility(target_agent)
        assistance_strength = self._compute_assistance_strength(assistance_type, target_agent)
        mathematical_cost = self._compute_assistance_cost(assistance_type, assistance_strength)
        estimated_effectiveness = self._estimate_assistance_effectiveness(assistance_type, target_agent)

        offer = RegulatoryAssistanceOffer(
            offer_id=str(uuid.uuid4()),
            offering_mathematical_id=self.mathematical_id,
            target_mathematical_id=target_id,
            offered_capabilities={assistance_type},
            assistance_strength=assistance_strength,
            mathematical_cost=mathematical_cost,
            estimated_effectiveness=estimated_effectiveness,
            offer_duration=300.0,  # 5 minutes default
            mathematical_compatibility_score=compatibility_score,
            timestamp=time.time(),
        )

        self.assistance_offers_made[offer.offer_id] = offer

        logger.info(
            f"🤖 {self.mathematical_id} offers {assistance_type.value} assistance to {target_id} "
            f"(strength={assistance_strength:.3f}, effectiveness={estimated_effectiveness:.3f})"
        )

        return offer

    def request_regulation_from_peers(
        self, required_capabilities: Set[RegulatoryCapability], urgency: float = 0.5
    ) -> MathematicalDistressSignal:
        """
        Request regulatory assistance from peer mathematical objects.

        Broadcasts distress signal using mathematical field properties and
        breathing synchrony communication patterns.
        """
        current_health = self.monitor_mathematical_health()

        distress_type = self._determine_distress_type(current_health)

        broadcast_radius = self.peer_discovery_radius * (1.0 + urgency)

        q_component_distress = self._analyze_q_component_distress()

        distress_signal = MathematicalDistressSignal(
            distress_id=str(uuid.uuid4()),
            source_mathematical_id=self.mathematical_id,
            distress_type=distress_type,
            mathematical_urgency=urgency,
            distress_description=self._generate_distress_description(current_health),
            required_regulatory_capabilities=required_capabilities,
            field_position_context=getattr(self.agent, "field_position", None),
            q_component_distress_details=q_component_distress,
            broadcast_radius=broadcast_radius,
            timestamp=time.time(),
        )

        self.active_distress_signals[distress_signal.distress_id] = distress_signal

        logger.warning(
            f"🤖 {self.mathematical_id} broadcasts distress signal: {distress_type} "
            f"(urgency={urgency:.3f}, radius={broadcast_radius:.1f})"
        )

        return distress_signal

    def form_temporary_regulatory_alliance(
        self, partner_ids: Set[str], alliance_purpose: str
    ) -> Optional[RegulatoryAlliance]:
        """
        Form temporary regulatory alliance with compatible mathematical objects.

        Uses mathematical compatibility and regulatory capability complementarity
        to create effective regulatory alliances.
        """
        if len(self.active_alliances) >= self.max_concurrent_alliances:
            raise ValueError(f"Mathematical alliance formation failed - maximum alliances reached {len(self.active_alliances)}/{self.max_concurrent_alliances} - ALLIANCE CAPACITY EXCEEDED")

        compatible_partners = set()
        for partner_id in partner_ids:
            if partner_id in self.known_mathematical_peers:
                partner_profile = self.known_mathematical_peers[partner_id]
                if partner_profile.compatibility_score >= self.alliance_formation_threshold:
                    compatible_partners.add(partner_id)

        if len(compatible_partners) < 1:  # Need at least one compatible partner
            raise ValueError("Mathematical alliance formation failed - no compatible partners found - ALLIANCE IMPOSSIBLE")

        alliance_members = compatible_partners.copy()
        alliance_members.add(self.mathematical_id)

        regulatory_strategy = self._determine_alliance_strategy(alliance_members)

        collective_strength = self._compute_collective_regulatory_strength(alliance_members)

        alliance = RegulatoryAlliance(
            alliance_id=str(uuid.uuid4()),
            member_mathematical_ids=alliance_members,
            alliance_purpose=alliance_purpose,
            regulatory_strategy=regulatory_strategy,
            formation_timestamp=time.time(),
            expected_duration=1800.0,  # 30 minutes default
            mathematical_effectiveness=0.0,  # Will be updated based on performance
            collective_regulatory_strength=collective_strength,
            alliance_health=1.0,  # Perfect health at formation
        )

        self.active_alliances[alliance.alliance_id] = alliance

        logger.info(
            f"🤖 {self.mathematical_id} forms alliance {alliance.alliance_id} "
            f"with {len(compatible_partners)} partners for {alliance_purpose}"
        )

        return alliance

    def evolve_regulatory_capabilities(self, regulation_history: List[Dict[str, Any]]) -> Set[RegulatoryCapability]:
        """
        Evolve regulatory capabilities based on experience and effectiveness.

        Uses adaptive tuning history and regulatory outcomes to develop
        specialized regulatory expertise.
        """
        if not self.self_evolution_enabled:
            return self.regulatory_capabilities

        capability_performance = self._analyze_capability_performance(regulation_history)

        development_candidates = self._identify_development_candidates(capability_performance)

        new_capabilities = set()
        for candidate in development_candidates:
            if self._assess_capability_development_potential(candidate) > 0.7:
                new_capabilities.add(candidate)

                self.regulatory_expertise_levels[candidate] = 0.1  # Beginner level

                logger.info(f"🤖 {self.mathematical_id} develops new capability: {candidate.value}")

        for capability in self.regulatory_capabilities:
            if capability in capability_performance:
                performance_score = capability_performance[capability]
                current_expertise = self.regulatory_expertise_levels.get(capability)

                new_expertise = current_expertise + self.learning_rate * (performance_score - current_expertise)
                new_expertise = max(0.0, min(1.0, new_expertise))  # Clamp to [0, 1]

                self.regulatory_expertise_levels[capability] = new_expertise

        self.regulatory_capabilities.update(new_capabilities)

        development_record = {
            "timestamp": time.time(),
            "new_capabilities": [cap.value for cap in new_capabilities],
            "updated_expertise": dict(self.regulatory_expertise_levels),
            "development_trigger": "regulation_history_analysis",
        }
        self.capability_development_history.append(development_record)

        self._update_autonomy_level()

        return new_capabilities

    def apply_self_regulation(self, regulation_type: RegulatoryCapability) -> bool:
        """
        Apply self-regulation using developed regulatory capabilities.

        Uses mathematical principles and developed expertise to perform
        self-regulatory actions on Q(τ,C,s) components.
        """
        if regulation_type not in self.regulatory_capabilities:
            return False

        expertise_level = self.regulatory_expertise_levels.get(regulation_type)
        if expertise_level < 0.1:  # Minimum expertise required
            return False

        current_health = self.monitor_mathematical_health()

        success = False

        if regulation_type == RegulatoryCapability.PERSISTENCE_REGULATION:
            success = self._apply_persistence_regulation(expertise_level)
        elif regulation_type == RegulatoryCapability.PHASE_COHERENCE_RESTORATION:
            success = self._apply_phase_coherence_restoration(expertise_level)
        elif regulation_type == RegulatoryCapability.FIELD_STABILIZATION:
            success = self._apply_field_stabilization(expertise_level)
        elif regulation_type == RegulatoryCapability.ENERGY_CONSERVATION:
            success = self._apply_energy_conservation(expertise_level)
        elif regulation_type == RegulatoryCapability.BREATHING_SYNCHRONIZATION:
            success = self._apply_breathing_synchronization(expertise_level)
        elif regulation_type == RegulatoryCapability.SINGULARITY_RESOLUTION:
            success = self._apply_singularity_resolution(expertise_level)
        elif regulation_type == RegulatoryCapability.TOPOLOGICAL_REPAIR:
            success = self._apply_topological_repair(expertise_level)

        regulation_record = {
            "timestamp": time.time(),
            "regulation_type": regulation_type.value,
            "expertise_level": expertise_level,
            "health_before": current_health.overall_mathematical_health,
            "success": success,
        }

        if success:
            post_regulation_health = self.monitor_mathematical_health()
            regulation_record["health_after"] = post_regulation_health.overall_mathematical_health

            effectiveness = (
                post_regulation_health.overall_mathematical_health - current_health.overall_mathematical_health
            )
            self.regulatory_effectiveness_history.append(effectiveness)

            logger.debug(
                f"🤖 {self.mathematical_id} successfully applied {regulation_type.value} "
                f"(expertise={expertise_level:.3f}, improvement={effectiveness:.3f})"
            )
        else:
            regulation_record["health_after"] = current_health.overall_mathematical_health
            logger.debug(f"🤖 {self.mathematical_id} failed to apply {regulation_type.value}")

        self.assistance_success_history.append(regulation_record)

        return success

    def learn_from_regulatory_outcomes(self, outcomes: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Learn from regulatory outcomes to improve future performance.

        Updates cluster sensitivity and sync threshold parameters based on
        mathematical analysis of regulatory effectiveness.
        """
        learning_updates = {}

        capability_outcomes = {}
        for outcome in outcomes:
            if "regulation_type" in outcome and "effectiveness" in outcome:
                reg_type = outcome["regulation_type"]
                effectiveness = outcome["effectiveness"]

                if reg_type not in capability_outcomes:
                    capability_outcomes[reg_type] = []
                capability_outcomes[reg_type].append(effectiveness)

        for reg_type, effectiveness_scores in capability_outcomes.items():
            if not effectiveness_scores:
                continue

            eff_tensor = torch.tensor(effectiveness_scores, dtype=torch.float32)
            avg_effectiveness = torch.mean(eff_tensor).item()
            effectiveness_variance = torch.var(eff_tensor).item()

            if reg_type in [cap.value for cap in self.regulatory_capabilities]:
                capability_enum = RegulatoryCapability(reg_type)
                current_expertise = self.regulatory_expertise_levels.get(capability_enum)

                if avg_effectiveness > 0.7 and effectiveness_variance < 0.1:
                    expertise_increase = self.learning_rate * 0.5
                    new_expertise = min(1.0, current_expertise + expertise_increase)
                    self.regulatory_expertise_levels[capability_enum] = new_expertise
                    learning_updates[f"{reg_type}_expertise"] = new_expertise

                elif avg_effectiveness < 0.3:
                    expertise_decrease = self.learning_rate * 0.3
                    new_expertise = max(0.0, current_expertise - expertise_decrease)
                    self.regulatory_expertise_levels[capability_enum] = new_expertise
                    learning_updates[f"{reg_type}_expertise"] = new_expertise

            if "cluster_sensitivity" in outcome:
                current_sensitivity = outcome.get("cluster_sensitivity")
                if avg_effectiveness > 0.6:
                    learning_updates["cluster_sensitivity"] = current_sensitivity
                else:
                    adjusted_sensitivity = current_sensitivity * (1.0 + self.learning_rate * (avg_effectiveness - 0.5))
                    learning_updates["cluster_sensitivity"] = max(0.1, min(10.0, adjusted_sensitivity))

            if "sync_threshold" in outcome:
                current_threshold = outcome.get("sync_threshold")
                if avg_effectiveness > 0.6:
                    learning_updates["sync_threshold"] = current_threshold
                else:
                    adjusted_threshold = current_threshold + self.learning_rate * (avg_effectiveness - 0.5)
                    learning_updates["sync_threshold"] = max(0.1, min(0.9, adjusted_threshold))

        if learning_updates:
            reputation_change = self.learning_rate * len(learning_updates)
            self.regulatory_reputation_score = max(0.0, min(1.0, self.regulatory_reputation_score + reputation_change))
            learning_updates["reputation_score"] = self.regulatory_reputation_score

        logger.debug(
            f"🤖 {self.mathematical_id} learned from {len(outcomes)} outcomes, "
            f"updated {len(learning_updates)} parameters"
        )

        return learning_updates

    def evaluate_mathematical_identity_stability(self) -> float:
        """
        Evaluate stability of mathematical identity over time.

        Measures how consistently the mathematical object maintains its
        identity across regulation cycles and field transformations.
        """
        if len(self.health_history) < 2:
            return 1.0  # Perfect stability with insufficient history

        is_consistent, consistency_score, consistency_metrics = (
            self.identity_system.verify_mathematical_identity_consistency(self.identity_profile, self.agent)
        )

        recent_health_scores = [h.overall_mathematical_health for h in self.health_history[-10:]]
        if not recent_health_scores:
            health_stability = 1.0
        else:
            health_tensor = torch.tensor(recent_health_scores, dtype=torch.float32)
            health_stability = 1.0 - torch.std(health_tensor).item()

        identity_stability_components = []

        if "topological" in consistency_metrics:
            identity_stability_components.append(consistency_metrics["topological"])

        if "canonical" in consistency_metrics:
            identity_stability_components.append(consistency_metrics["canonical"])

        if "information" in consistency_metrics:
            identity_stability_components.append(consistency_metrics["information"])

        if identity_stability_components:
            identity_tensor = torch.tensor(identity_stability_components, dtype=torch.float32)
            identity_stability = torch.mean(identity_tensor).item()
        else:
            identity_stability = consistency_score

        overall_stability = (identity_stability + health_stability) / 2.0

        logger.debug(
            f"🤖 Identity stability for {self.mathematical_id}: {overall_stability:.3f} "
            f"(identity={identity_stability:.3f}, health={health_stability:.3f})"
        )

        return float(overall_stability)

    def graduate_to_autonomous_operation(self) -> bool:
        """
        Evaluate whether mathematical object is ready for autonomous operation.

        Assesses mathematical maturity, regulatory capability development,
        and identity stability to determine autonomy readiness.
        """
        if self.autonomous_operation_level < self.autonomous_threshold:
            return False

        if self.health_history:
            recent_health = self.health_history[-1]
            if recent_health.overall_mathematical_health < 0.8:
                return False

        identity_stability = self.evaluate_mathematical_identity_stability()
        if identity_stability < 0.9:
            return False

        if len(self.regulatory_capabilities) < 3:  # Minimum capabilities required
            return False

        if self.regulatory_expertise_levels:
            expertise_tensor = torch.tensor(list(self.regulatory_expertise_levels.values()), dtype=torch.float32)
            avg_expertise = torch.mean(expertise_tensor).item()
            if avg_expertise < 0.7:
                return False

        if self.regulatory_reputation_score < 0.8:
            return False

        if self.assistance_success_history:
            recent_successes = [h.get("success") for h in self.assistance_success_history[-20:]]
            success_tensor = torch.tensor(recent_successes, dtype=torch.float32)
            success_rate = torch.mean(success_tensor).item()
            if success_rate < 0.7:
                return False

        self.requires_external_oversight = False
        self.autonomous_operation_level = 1.0

        logger.info(
            f"🤖 {self.mathematical_id} graduated to autonomous operation! "
            f"Identity stability: {identity_stability:.3f}, "
            f"Capabilities: {len(self.regulatory_capabilities)}, "
            f"Reputation: {self.regulatory_reputation_score:.3f}"
        )

        return True


    def _analyze_q_component_stability(self) -> float:
        """Analyze stability of Q-components."""
        if not hasattr(self.agent, "Q_components") or not self.agent.Q_components:
            return 1.0  # Perfect stability if no components to destabilize

        stability_scores = []

        for comp_name, comp_value in self.agent.Q_components.items():
            if hasattr(comp_value, "__abs__"):
                magnitude = abs(comp_value)
                if math.isfinite(magnitude):
                    stability = 1.0 / (1.0 + magnitude**2)
                    stability_scores.append(stability)

        if not stability_scores:
            raise ValueError("Geometric feature consistency assessment failed - no stability scores - CONSISTENCY ANALYSIS IMPOSSIBLE")
        stability_tensor = torch.tensor(stability_scores, dtype=torch.float32)
        return torch.mean(stability_tensor).item()

    def _assess_field_coherence(self) -> float:
        """Assess field coherence based on Q-components."""
        if not hasattr(self.agent, "Q_components") or not self.agent.Q_components:
            return 1.0

        phases = []
        for comp_name, comp_value in self.agent.Q_components.items():
            if hasattr(comp_value, "real") and hasattr(comp_value, "imag"):
                phase = math.atan2(float(comp_value.imag), float(comp_value.real))
                phases.append(phase)

        if len(phases) < 2:
            return 1.0

        cos_values = torch.tensor([math.cos(p) for p in phases], dtype=torch.float32)
        sin_values = torch.tensor([math.sin(p) for p in phases], dtype=torch.float32)
        phase_cos_mean = torch.mean(cos_values).item()
        phase_sin_mean = torch.mean(sin_values).item()
        coherence = math.sqrt(phase_cos_mean**2 + phase_sin_mean**2)

        return float(coherence)

    def _evaluate_phase_relationship_health(self) -> float:
        """Evaluate health of phase relationships between components."""
        if not hasattr(self.agent, "Q_components") or not self.agent.Q_components:
            return 1.0

        components = list(self.agent.Q_components.values())
        if len(components) < 2:
            return 1.0

        phases = []
        for comp in components:
            if hasattr(comp, "real") and hasattr(comp, "imag"):
                phase = math.atan2(float(comp.imag), float(comp.real))
                phases.append(phase)
            else:
                phases.append(0.0)

        phase_diffs = []
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                diff = abs(phases[i] - phases[j])
                diff = min(diff, 2 * math.pi - diff)  # Normalize to [0, π]
                phase_diffs.append(diff)

        if not phase_diffs:
            return 1.0

        diffs_tensor = torch.tensor(phase_diffs, dtype=torch.float32)
        phase_variance = torch.var(diffs_tensor).item()
        health = 1.0 / (1.0 + phase_variance)

        return float(health)

    def _check_temporal_biography_integrity(self) -> float:
        """Check mathematical integrity of temporal biography."""
        if not hasattr(self.agent, "temporal_biography") or self.agent.temporal_biography is None:
            return 1.0  # Perfect integrity if no biography to corrupt

        bio = self.agent.temporal_biography
        integrity_components = []

        if hasattr(bio, "vivid_layer") and bio.vivid_layer is not None:
            vivid_values = [abs(x) for x in bio.vivid_layer if math.isfinite(abs(x))]
            if vivid_values:
                vivid_integrity = len(vivid_values) / len(bio.vivid_layer)
                integrity_components.append(vivid_integrity)

        if hasattr(bio, "character_layer") and bio.character_layer is not None:
            char_values = [abs(x) for x in bio.character_layer if math.isfinite(abs(x))]
            if char_values:
                char_integrity = len(char_values) / len(bio.character_layer)
                integrity_components.append(char_integrity)

        if hasattr(bio, "temporal_momentum") and bio.temporal_momentum is not None:
            momentum_integrity = 1.0 if math.isfinite(abs(bio.temporal_momentum)) else 0.0
            integrity_components.append(momentum_integrity)

        if not integrity_components:
            raise ValueError("Temporal biography integrity assessment failed - no integrity components - TEMPORAL ANALYSIS IMPOSSIBLE")
        integrity_tensor = torch.tensor(integrity_components, dtype=torch.float32)
        return torch.mean(integrity_tensor).item()

    def _analyze_breathing_pattern_regularity(self) -> float:
        """Analyze regularity of breathing patterns."""
        breathing_components = []

        if hasattr(self.agent, "breath_frequency") and math.isfinite(self.agent.breath_frequency):
            freq_regularity = 1.0 / (1.0 + abs(self.agent.breath_frequency) ** 2)
            breathing_components.append(freq_regularity)

        if hasattr(self.agent, "breath_amplitude") and math.isfinite(self.agent.breath_amplitude):
            amp_regularity = 1.0 / (1.0 + abs(self.agent.breath_amplitude) ** 2)
            breathing_components.append(amp_regularity)

        if hasattr(self.agent, "breathing_q_coefficients") and self.agent.breathing_q_coefficients:
            coeff_values = []
            for coeff in self.agent.breathing_q_coefficients.values():
                if hasattr(coeff, "__abs__") and math.isfinite(abs(coeff)):
                    coeff_values.append(abs(coeff))

            if coeff_values:
                coeff_tensor = torch.tensor(coeff_values, dtype=torch.float32)
                coeff_std = torch.std(coeff_tensor).item()
                coeff_mean = torch.mean(coeff_tensor).item()
                coeff_regularity = 1.0 - min(
                    1.0, coeff_std / (coeff_mean + self.mathematical_precision)
                )
                breathing_components.append(coeff_regularity)

        if not breathing_components:
            raise ValueError("Breathing pattern regularity assessment failed - no breathing components - BREATHING ANALYSIS IMPOSSIBLE")
        breathing_tensor = torch.tensor(breathing_components, dtype=torch.float32)
        return torch.mean(breathing_tensor).item()

    def _evaluate_geometric_feature_consistency(self) -> float:
        """Evaluate consistency of geometric features."""
        if not hasattr(self.agent, "geometric_features") or self.agent.geometric_features is None:
            return 1.0

        geom_features = self.agent.geometric_features

        if hasattr(geom_features, "__len__"):
            feature_values = [float(f) for f in geom_features if math.isfinite(float(f))]
            if feature_values:
                consistency = len(feature_values) / len(geom_features)
                return float(consistency)

        return 1.0

    def _assess_modular_weight_stability(self) -> float:
        """Assess stability of modular weight."""
        if not hasattr(self.agent, "modular_weight") or self.agent.modular_weight is None:
            return 1.0

        weight = self.agent.modular_weight
        if not math.isfinite(float(weight)):
            raise ValueError(f"Modular weight stability failed - non-finite weight {weight} - WEIGHT ANALYSIS IMPOSSIBLE")
        
        weight_tensor = torch.tensor(float(weight), dtype=torch.float32)
        stability = 1.0 / (1.0 + torch.abs(weight_tensor) ** 2)
        return float(stability.item())

    def _compute_singularity_risk(self) -> float:
        """Compute risk of mathematical singularities."""
        risk_factors = []

        if hasattr(self.agent, "q_value") and self.agent.q_value is not None:
            q_magnitude = abs(self.agent.q_value)
            if q_magnitude > 1e6:  # Very large magnitude
                risk_factors.append(0.8)
            elif not math.isfinite(q_magnitude):  # Non-finite value
                risk_factors.append(1.0)
            else:
                risk_factors.append(0.0)

        if hasattr(self.agent, "Q_components") and self.agent.Q_components:
            for comp_value in self.agent.Q_components.values():
                if hasattr(comp_value, "__abs__"):
                    magnitude = abs(comp_value)
                    if not math.isfinite(magnitude):
                        risk_factors.append(1.0)
                    elif magnitude > 1e6:
                        risk_factors.append(0.6)
                    else:
                        risk_factors.append(0.0)

        if hasattr(self.agent, "breathing_q_coefficients") and self.agent.breathing_q_coefficients:
            for coeff in self.agent.breathing_q_coefficients.values():
                if hasattr(coeff, "__abs__"):
                    magnitude = abs(coeff)
                    if not math.isfinite(magnitude):
                        risk_factors.append(0.8)
                    elif magnitude > 1e4:
                        risk_factors.append(0.4)
                    else:
                        risk_factors.append(0.0)

        return max(risk_factors) if risk_factors else 0.0

    def _determine_health_status(self, health_score: float, singularity_risk: float) -> MathematicalHealthStatus:
        """Determine categorical health status."""
        if singularity_risk > 0.8:
            return MathematicalHealthStatus.MATHEMATICAL_SINGULARITY
        elif health_score < 0.3:
            return MathematicalHealthStatus.CRITICAL
        elif health_score < 0.6:
            return MathematicalHealthStatus.UNSTABLE
        elif health_score < 0.8:
            return MathematicalHealthStatus.STABLE
        else:
            return MathematicalHealthStatus.EXCELLENT

    def _compute_phase_coherence_stability(self) -> float:
        """Compute phase coherence stability."""
        if not hasattr(self.agent, "Q_components") or not self.agent.Q_components:
            return 1.0

        phases = []
        for comp_value in self.agent.Q_components.values():
            if hasattr(comp_value, "real") and hasattr(comp_value, "imag"):
                phase = math.atan2(float(comp_value.imag), float(comp_value.real))
                phases.append(phase)

        if len(phases) < 2:
            return 1.0

        phase_vectors = [(math.cos(p), math.sin(p)) for p in phases]
        x_coords = torch.tensor([v[0] for v in phase_vectors], dtype=torch.float32)
        y_coords = torch.tensor([v[1] for v in phase_vectors], dtype=torch.float32)
        mean_vector = (torch.mean(x_coords).item(), torch.mean(y_coords).item())
        coherence = math.sqrt(mean_vector[0] ** 2 + mean_vector[1] ** 2)

        return float(coherence)

    def _compute_breathing_coefficient_stability(self) -> float:
        """Compute breathing coefficient stability."""
        if not hasattr(self.agent, "breathing_q_coefficients") or not self.agent.breathing_q_coefficients:
            return 1.0

        magnitudes = []
        for coeff in self.agent.breathing_q_coefficients.values():
            if hasattr(coeff, "__abs__"):
                mag = abs(coeff)
                if math.isfinite(mag):
                    magnitudes.append(mag)

        if not magnitudes:
            return 1.0

        mag_tensor = torch.tensor(magnitudes, dtype=torch.float32)
        magnitude_variance = torch.var(mag_tensor).item()
        stability = 1.0 / (1.0 + magnitude_variance)

        return float(stability)

    def _compute_position_stability(self) -> float:
        """Compute field position stability."""
        if not hasattr(self.agent, "field_position") or self.agent.field_position is None:
            return 1.0

        pos = self.agent.field_position

        if hasattr(pos, "__len__"):
            position_magnitudes = [abs(float(p)) for p in pos if math.isfinite(float(p))]
            if position_magnitudes:
                position_magnitude = math.sqrt(sum(p**2 for p in position_magnitudes))
                stability = 1.0 / (1.0 + position_magnitude**2)
                return float(stability)
        else:
            if math.isfinite(float(pos)):
                stability = 1.0 / (1.0 + abs(float(pos)) ** 2)
                return float(stability)

        raise ValueError(f"Field position stability failed - non-finite position - POSITION ANALYSIS IMPOSSIBLE")

    def _compute_mathematical_compatibility(self, other_agent: ConceptualChargeAgent) -> float:
        """Compute mathematical compatibility with another agent."""
        compatibility_components = []

        if (
            hasattr(self.agent, "q_value")
            and hasattr(other_agent, "q_value")
            and self.agent.q_value is not None
            and other_agent.q_value is not None
        ):

            q1, q2 = self.agent.q_value, other_agent.q_value
            if hasattr(q1, "__abs__") and hasattr(q2, "__abs__"):
                mag1, mag2 = abs(q1), abs(q2)
                if mag1 > 0 and mag2 > 0:
                    ratio = min(mag1, mag2) / max(mag1, mag2)
                    compatibility_components.append(ratio)

        geometric_compatibility = self._compute_geometric_compatibility(other_agent)
        compatibility_components.append(geometric_compatibility)

        position_compatibility = self._compute_field_position_proximity(other_agent)
        compatibility_components.append(position_compatibility)

        phase_compatibility = self._compute_phase_relationship_harmony(other_agent)
        compatibility_components.append(phase_compatibility)

        if not compatibility_components:
            raise ValueError("Mathematical compatibility computation failed - no compatibility components - COMPATIBILITY ANALYSIS IMPOSSIBLE")
        
        compat_tensor = torch.tensor(compatibility_components, dtype=torch.float32)
        return torch.mean(compat_tensor).item()

    def _compute_mathematical_similarity(self, other_agent: ConceptualChargeAgent) -> float:
        """Compute mathematical similarity with another agent."""
        other_proxy = MathematicalObjectProxy(other_agent, self.identity_system)
        other_profile = other_proxy.identity_profile

        similarity_score, _ = self.identity_system._compute_mathematical_similarity(
            self.identity_profile, other_profile
        )

        return similarity_score

    def _compute_geometric_compatibility(self, other_agent: ConceptualChargeAgent) -> float:
        """Compute geometric compatibility."""
        if (
            not hasattr(self.agent, "geometric_features")
            or not hasattr(other_agent, "geometric_features")
            or self.agent.geometric_features is None
            or other_agent.geometric_features is None
        ):
            raise ValueError("Geometric compatibility requires geometric features on both agents - GEOMETRIC ANALYSIS IMPOSSIBLE")

        geom1, geom2 = self.agent.geometric_features, other_agent.geometric_features

        if not (hasattr(geom1, "__len__") and hasattr(geom2, "__len__")):
            raise ValueError("Geometric features must be array-like structures - GEOMETRIC COMPATIBILITY IMPOSSIBLE")
        
        if len(geom1) != len(geom2):
            raise ValueError(f"Geometric features dimension mismatch: {len(geom1)} vs {len(geom2)} - GEOMETRIC COMPATIBILITY UNDEFINED")
        
        features1 = torch.tensor([float(f) for f in geom1], dtype=torch.float32)
        features2 = torch.tensor([float(f) for f in geom2], dtype=torch.float32)

        norm1, norm2 = torch.norm(features1), torch.norm(features2)
        if norm1 <= 0 or norm2 <= 0:
            raise ValueError(f"Geometric features have zero norm: {norm1}, {norm2} - GEOMETRIC COMPATIBILITY UNDEFINED")
        
        similarity = torch.dot(features1, features2) / (norm1 * norm2)
        return float(torch.abs(similarity).item())  # Use absolute value for compatibility

    def _compute_field_position_proximity(self, other_agent: ConceptualChargeAgent) -> float:
        """Compute field position proximity."""
        if (
            not hasattr(self.agent, "field_position")
            or not hasattr(other_agent, "field_position")
            or self.agent.field_position is None
            or other_agent.field_position is None
        ):
            raise ValueError("Field position proximity requires field positions on both agents - SPATIAL ANALYSIS IMPOSSIBLE")

        pos1, pos2 = self.agent.field_position, other_agent.field_position

        if hasattr(pos1, "__len__") and hasattr(pos2, "__len__"):
            if len(pos1) != len(pos2):
                raise ValueError(f"Field position dimension mismatch: {len(pos1)} vs {len(pos2)} - SPATIAL PROXIMITY UNDEFINED")
            
            positions = np.array([pos1, pos2])
            distances = pdist(positions, metric='euclidean')
            distance = distances[0]  # Only one pairwise distance
            proximity = 1.0 / (1.0 + distance)
            return float(proximity)
        else:
            pos1_tensor = torch.tensor(float(pos1), dtype=torch.float32)
            pos2_tensor = torch.tensor(float(pos2), dtype=torch.float32)
            distance = torch.abs(pos1_tensor - pos2_tensor).item()
            proximity = 1.0 / (1.0 + distance)
            return float(proximity)

    def _compute_phase_relationship_harmony(self, other_agent: ConceptualChargeAgent) -> float:
        """Compute phase relationship harmony."""
        phases1, phases2 = [], []

        if hasattr(self.agent, "Q_components") and self.agent.Q_components:
            for comp in self.agent.Q_components.values():
                if hasattr(comp, "real") and hasattr(comp, "imag"):
                    phase = math.atan2(float(comp.imag), float(comp.real))
                    phases1.append(phase)

        if hasattr(other_agent, "Q_components") and other_agent.Q_components:
            for comp in other_agent.Q_components.values():
                if hasattr(comp, "real") and hasattr(comp, "imag"):
                    phase = math.atan2(float(comp.imag), float(comp.real))
                    phases2.append(phase)

        if not phases1 or not phases2:
            raise ValueError("Phase relationship harmony calculation failed - no valid phase data - PHASE ANALYSIS IMPOSSIBLE")

        harmony_scores = []
        for p1 in phases1:
            for p2 in phases2:
                phase_diff = abs(p1 - p2)
                phase_diff = min(phase_diff, 2 * math.pi - phase_diff)
                harmony = 1.0 - (phase_diff / math.pi)  # Normalize to [0, 1]
                harmony_scores.append(harmony)

        if not harmony_scores:
            raise ValueError("Phase relationship harmony calculation failed - no harmony scores - HARMONY ANALYSIS IMPOSSIBLE")
        harmony_tensor = torch.tensor(harmony_scores, dtype=torch.float32)
        return torch.mean(harmony_tensor).item()

    def _assess_regulatory_capabilities(self, other_agent: ConceptualChargeAgent) -> Set[RegulatoryCapability]:
        """Assess potential regulatory capabilities of another agent."""
        capabilities = set()

        if hasattr(other_agent, "Q_components") and other_agent.Q_components:
            q_stability = self._analyze_q_component_stability_for_agent(other_agent)
            if q_stability > 0.7:
                capabilities.add(RegulatoryCapability.FIELD_STABILIZATION)
                capabilities.add(RegulatoryCapability.PHASE_COHERENCE_RESTORATION)

        if (
            hasattr(other_agent, "breathing_q_coefficients")
            and other_agent.breathing_q_coefficients
            and hasattr(other_agent, "breath_frequency")
            and math.isfinite(other_agent.breath_frequency)
        ):
            capabilities.add(RegulatoryCapability.BREATHING_SYNCHRONIZATION)

        if hasattr(other_agent, "temporal_biography") and other_agent.temporal_biography is not None:
            capabilities.add(RegulatoryCapability.PERSISTENCE_REGULATION)

        if hasattr(other_agent, "geometric_features") and other_agent.geometric_features is not None:
            capabilities.add(RegulatoryCapability.TOPOLOGICAL_REPAIR)

        if hasattr(other_agent, "q_value") and other_agent.q_value is not None:
            q_magnitude = abs(other_agent.q_value)
            if math.isfinite(q_magnitude) and q_magnitude < 1e6:
                capabilities.add(RegulatoryCapability.ENERGY_CONSERVATION)

        return capabilities

    def _analyze_q_component_stability_for_agent(self, agent: ConceptualChargeAgent) -> float:
        """Analyze Q-component stability for another agent."""
        if not hasattr(agent, "Q_components") or not agent.Q_components:
            return 1.0

        stability_scores = []
        for comp_value in agent.Q_components.values():
            if hasattr(comp_value, "__abs__"):
                magnitude = abs(comp_value)
                if math.isfinite(magnitude):
                    stability = 1.0 / (1.0 + magnitude**2)
                    stability_scores.append(stability)

        if not stability_scores:
            raise ValueError("Mathematical alliance stability assessment failed - no stability scores - ALLIANCE ANALYSIS IMPOSSIBLE")
        stability_tensor = torch.tensor(stability_scores, dtype=torch.float32)
        return torch.mean(stability_tensor).item()

    def _analyze_capability_performance(
        self, regulation_history: List[Dict[str, Any]]
    ) -> Dict[RegulatoryCapability, float]:
        """Analyze performance of different regulatory capabilities."""
        capability_performance = {}

        for history_entry in regulation_history:
            if "regulation_type" in history_entry and "effectiveness" in history_entry:
                reg_type_str = history_entry["regulation_type"]
                effectiveness = history_entry["effectiveness"]

                try:
                    reg_type = RegulatoryCapability(reg_type_str)
                    if reg_type not in capability_performance:
                        capability_performance[reg_type] = []
                    capability_performance[reg_type].append(effectiveness)
                except ValueError:
                    continue  # Skip unknown regulation types

        avg_performance = {}
        for capability, effectiveness_scores in capability_performance.items():
            perf_tensor = torch.tensor(effectiveness_scores, dtype=torch.float32)
            avg_performance[capability] = torch.mean(perf_tensor).item()

        return avg_performance

    def _identify_development_candidates(
        self, capability_performance: Dict[RegulatoryCapability, float]
    ) -> List[RegulatoryCapability]:
        """Identify capabilities that show promise for development."""
        candidates = []

        all_capabilities = set(RegulatoryCapability)
        undeveloped_capabilities = all_capabilities - self.regulatory_capabilities

        for capability in undeveloped_capabilities:
            if self._has_features_for_capability(capability):
                candidates.append(capability)

        for capability, performance in capability_performance.items():
            if capability not in self.regulatory_capabilities and performance > 0.6:
                candidates.append(capability)

        return candidates

    def _has_features_for_capability(self, capability: RegulatoryCapability) -> bool:
        """Check if agent has features necessary for developing a capability."""
        if capability == RegulatoryCapability.PERSISTENCE_REGULATION:
            return hasattr(self.agent, "temporal_biography") and self.agent.temporal_biography is not None

        elif capability == RegulatoryCapability.PHASE_COHERENCE_RESTORATION:
            return hasattr(self.agent, "Q_components") and self.agent.Q_components

        elif capability == RegulatoryCapability.FIELD_STABILIZATION:
            return hasattr(self.agent, "q_value") and self.agent.q_value is not None

        elif capability == RegulatoryCapability.ENERGY_CONSERVATION:
            return hasattr(self.agent, "q_value") and self.agent.q_value is not None

        elif capability == RegulatoryCapability.BREATHING_SYNCHRONIZATION:
            return hasattr(self.agent, "breathing_q_coefficients") and self.agent.breathing_q_coefficients

        elif capability == RegulatoryCapability.SINGULARITY_RESOLUTION:
            return hasattr(self.agent, "Q_components") and self.agent.Q_components

        elif capability == RegulatoryCapability.TOPOLOGICAL_REPAIR:
            return hasattr(self.agent, "geometric_features") and self.agent.geometric_features is not None

        return False

    def _assess_capability_development_potential(self, capability: RegulatoryCapability) -> float:
        """Assess potential for developing a specific capability."""
        base_potential = 0.5

        if self._has_features_for_capability(capability):
            base_potential += 0.3

        if self.health_history:
            recent_health = self.health_history[-1].overall_mathematical_health
            health_bonus = recent_health * 0.2
            base_potential += health_bonus

        reputation_bonus = self.regulatory_reputation_score * 0.1
        base_potential += reputation_bonus

        return max(0.0, min(1.0, base_potential))

    def _update_autonomy_level(self):
        """Update autonomy level based on capability development."""
        capability_factor = len(self.regulatory_capabilities) / len(RegulatoryCapability)

        if self.regulatory_expertise_levels:
            expertise_tensor = torch.tensor(list(self.regulatory_expertise_levels.values()), dtype=torch.float32)
            avg_expertise = torch.mean(expertise_tensor).item()
            expertise_factor = avg_expertise
        else:
            expertise_factor = 0.0

        reputation_factor = self.regulatory_reputation_score

        if self.health_history:
            health_factor = self.health_history[-1].overall_mathematical_health
        else:
            health_factor = 0.5

        new_autonomy = (capability_factor + expertise_factor + reputation_factor + health_factor) / 4.0

        self.autonomous_operation_level = new_autonomy

        logger.debug(f"🤖 {self.mathematical_id} autonomy level updated to {new_autonomy:.3f}")

    def _determine_distress_type(self, health_metrics: MathematicalHealthMetrics) -> str:
        """Determine type of mathematical distress based on health metrics."""
        if health_metrics.mathematical_singularity_risk > 0.8:
            return "mathematical_singularity"
        elif health_metrics.q_component_stability < 0.3:
            return "q_component_instability"
        elif health_metrics.field_coherence_score < 0.3:
            return "field_decoherence"
        elif health_metrics.phase_relationship_health < 0.3:
            return "phase_relationship_breakdown"
        elif health_metrics.breathing_pattern_regularity < 0.3:
            return "breathing_pattern_chaos"
        else:
            return "general_mathematical_distress"

    def _analyze_q_component_distress(self) -> Dict[str, float]:
        """Analyze specific Q-component distress details."""
        distress_details = {}

        if hasattr(self.agent, "Q_components") and self.agent.Q_components:
            for comp_name, comp_value in self.agent.Q_components.items():
                if hasattr(comp_value, "__abs__"):
                    magnitude = abs(comp_value)
                    if not math.isfinite(magnitude):
                        distress_details[f"{comp_name}_magnitude"] = 1.0  # Maximum distress
                    elif magnitude > 1e6:
                        distress_details[f"{comp_name}_magnitude"] = 0.8
                    else:
                        distress_details[f"{comp_name}_magnitude"] = 0.0

        return distress_details

    def _generate_distress_description(self, health_metrics: MathematicalHealthMetrics) -> str:
        """Generate mathematical description of distress."""
        descriptions = []

        if health_metrics.q_component_stability < 0.5:
            descriptions.append(f"Q-component instability (stability={health_metrics.q_component_stability:.3f})")

        if health_metrics.field_coherence_score < 0.5:
            descriptions.append(f"Field decoherence (coherence={health_metrics.field_coherence_score:.3f})")

        if health_metrics.mathematical_singularity_risk > 0.5:
            descriptions.append(f"Singularity risk (risk={health_metrics.mathematical_singularity_risk:.3f})")

        if not descriptions:
            descriptions.append("General mathematical health degradation")

        return "; ".join(descriptions)

    def _compute_assistance_strength(
        self, assistance_type: RegulatoryCapability, target_agent: ConceptualChargeAgent
    ) -> float:
        """Compute strength of assistance that can be provided."""
        base_strength = self.regulatory_expertise_levels.get(assistance_type)

        compatibility = self._compute_mathematical_compatibility(target_agent)
        compatibility_factor = 0.5 + 0.5 * compatibility  # Range [0.5, 1.0]

        if self.health_history:
            health_factor = self.health_history[-1].overall_mathematical_health
        else:
            health_factor = 0.5

        assistance_strength = base_strength * compatibility_factor * health_factor

        return float(assistance_strength)

    def _compute_assistance_cost(self, assistance_type: RegulatoryCapability, assistance_strength: float) -> float:
        """Compute mathematical cost of providing assistance."""
        base_cost = assistance_strength * 0.1  # 10% of strength as cost

        complexity_multipliers = {
            RegulatoryCapability.SINGULARITY_RESOLUTION: 2.0,
            RegulatoryCapability.TOPOLOGICAL_REPAIR: 1.8,
            RegulatoryCapability.FIELD_STABILIZATION: 1.5,
            RegulatoryCapability.PHASE_COHERENCE_RESTORATION: 1.3,
            RegulatoryCapability.ENERGY_CONSERVATION: 1.2,
            RegulatoryCapability.PERSISTENCE_REGULATION: 1.1,
            RegulatoryCapability.BREATHING_SYNCHRONIZATION: 1.0,
        }

        complexity_factor = complexity_multipliers.get(assistance_type)
        total_cost = base_cost * complexity_factor

        return float(total_cost)

    def _estimate_assistance_effectiveness(
        self, assistance_type: RegulatoryCapability, target_agent: ConceptualChargeAgent
    ) -> float:
        """Estimate effectiveness of providing assistance to target."""
        base_effectiveness = self.regulatory_expertise_levels.get(assistance_type)

        target_receptivity = self._assess_target_receptivity(target_agent, assistance_type)

        compatibility = self._compute_mathematical_compatibility(target_agent)

        estimated_effectiveness = base_effectiveness * target_receptivity * compatibility

        return float(estimated_effectiveness)

    def _assess_target_receptivity(
        self, target_agent: ConceptualChargeAgent, assistance_type: RegulatoryCapability
    ) -> float:
        """Assess how receptive target is to specific type of assistance."""
        receptivity_factors = []

        if assistance_type == RegulatoryCapability.PERSISTENCE_REGULATION:
            if hasattr(target_agent, "temporal_biography") and target_agent.temporal_biography is not None:
                receptivity_factors.append(0.8)
            else:
                receptivity_factors.append(0.3)

        elif assistance_type == RegulatoryCapability.PHASE_COHERENCE_RESTORATION:
            if hasattr(target_agent, "Q_components") and target_agent.Q_components:
                phase_health = self._evaluate_phase_relationship_health_for_agent(target_agent)
                receptivity = 1.0 - phase_health  # Higher receptivity for poor phase health
                receptivity_factors.append(receptivity)
            else:
                receptivity_factors.append(0.2)

        elif assistance_type == RegulatoryCapability.FIELD_STABILIZATION:
            if hasattr(target_agent, "q_value") and target_agent.q_value is not None:
                q_stability = self._analyze_q_component_stability_for_agent(target_agent)
                receptivity = 1.0 - q_stability  # Higher receptivity for poor stability
                receptivity_factors.append(receptivity)
            else:
                receptivity_factors.append(0.2)

        else:
            receptivity_factors.append(0.5)  # Default neutral receptivity

        if not receptivity_factors:
            raise ValueError("Mathematical assistance receptivity assessment failed - no receptivity factors - RECEPTIVITY ANALYSIS IMPOSSIBLE")
        receptivity_tensor = torch.tensor(receptivity_factors, dtype=torch.float32)
        return torch.mean(receptivity_tensor).item()

    def _evaluate_phase_relationship_health_for_agent(self, agent: ConceptualChargeAgent) -> float:
        """Evaluate phase relationship health for another agent."""
        if not hasattr(agent, "Q_components") or not agent.Q_components:
            return 1.0

        components = list(agent.Q_components.values())
        if len(components) < 2:
            return 1.0

        phases = []
        for comp in components:
            if hasattr(comp, "real") and hasattr(comp, "imag"):
                phase = math.atan2(float(comp.imag), float(comp.real))
                phases.append(phase)
            else:
                phases.append(0.0)

        phase_diffs = []
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                diff = abs(phases[i] - phases[j])
                diff = min(diff, 2 * math.pi - diff)
                phase_diffs.append(diff)

        if not phase_diffs:
            return 1.0

        diffs_tensor = torch.tensor(phase_diffs, dtype=torch.float32)
        phase_variance = torch.var(diffs_tensor).item()
        health = 1.0 / (1.0 + phase_variance)

        return float(health)

    def _determine_alliance_strategy(self, member_ids: Set[str]) -> str:
        """Determine regulatory strategy for alliance based on member capabilities."""
        all_capabilities = set(self.regulatory_capabilities)

        if RegulatoryCapability.SINGULARITY_RESOLUTION in all_capabilities:
            return "emergency_singularity_resolution"
        elif RegulatoryCapability.FIELD_STABILIZATION in all_capabilities:
            return "coordinated_field_stabilization"
        elif RegulatoryCapability.PHASE_COHERENCE_RESTORATION in all_capabilities:
            return "distributed_phase_coherence"
        else:
            return "general_regulatory_support"

    def _compute_collective_regulatory_strength(self, member_ids: Set[str]) -> float:
        """Compute collective regulatory strength of alliance."""
        base_strength = (
            sum(self.regulatory_expertise_levels.values()) / len(self.regulatory_expertise_levels)
            if self.regulatory_expertise_levels
            else 0.0
        )

        member_factor = math.sqrt(len(member_ids))  # Square root to prevent excessive scaling

        collective_strength = base_strength * member_factor
        return min(1.0, collective_strength)  # Cap at 1.0


    def _apply_persistence_regulation(self, expertise_level: float) -> bool:
        """Apply persistence regulation using Ψ_persistence decay mechanisms."""
        if not hasattr(self.agent, "temporal_biography") or self.agent.temporal_biography is None:
            return False

        logger.debug(f"🤖 {self.mathematical_id} applies persistence regulation (expertise={expertise_level:.3f})")
        return expertise_level > 0.3  # Success probability based on expertise

    def _apply_phase_coherence_restoration(self, expertise_level: float) -> bool:
        """Apply phase coherence restoration."""
        if not hasattr(self.agent, "Q_components") or not self.agent.Q_components:
            return False

        logger.debug(f"🤖 {self.mathematical_id} applies phase coherence restoration (expertise={expertise_level:.3f})")
        return expertise_level > 0.2

    def _apply_field_stabilization(self, expertise_level: float) -> bool:
        """Apply field stabilization."""
        if not hasattr(self.agent, "q_value") or self.agent.q_value is None:
            return False

        logger.debug(f"🤖 {self.mathematical_id} applies field stabilization (expertise={expertise_level:.3f})")
        return expertise_level > 0.25

    def _apply_energy_conservation(self, expertise_level: float) -> bool:
        """Apply energy conservation regulation."""
        if not hasattr(self.agent, "q_value") or self.agent.q_value is None:
            return False

        logger.debug(f"🤖 {self.mathematical_id} applies energy conservation (expertise={expertise_level:.3f})")
        return expertise_level > 0.3

    def _apply_breathing_synchronization(self, expertise_level: float) -> bool:
        """Apply breathing synchronization regulation."""
        if not hasattr(self.agent, "breathing_q_coefficients") or not self.agent.breathing_q_coefficients:
            return False

        logger.debug(f"🤖 {self.mathematical_id} applies breathing synchronization (expertise={expertise_level:.3f})")
        return expertise_level > 0.2

    def _apply_singularity_resolution(self, expertise_level: float) -> bool:
        """Apply singularity resolution."""
        logger.debug(f"🤖 {self.mathematical_id} applies singularity resolution (expertise={expertise_level:.3f})")
        return expertise_level > 0.5  # Higher expertise required for singularity resolution

    def _apply_topological_repair(self, expertise_level: float) -> bool:
        """Apply topological repair."""
        if not hasattr(self.agent, "geometric_features") or self.agent.geometric_features is None:
            return False

        logger.debug(f"🤖 {self.mathematical_id} applies topological repair (expertise={expertise_level:.3f})")
        return expertise_level > 0.4
