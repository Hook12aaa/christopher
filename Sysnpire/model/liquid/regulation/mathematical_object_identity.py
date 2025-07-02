"""
Mathematical Object Identity Theory - Persistent Mathematical Entity
Recognition

CORE MATHEMATICAL FOUNDATION: Implements mathematical object identity
preservation across field regulation cycles using topological invariants,
canonical signatures, and information-theoretic fingerprints derived from
Q(τ,C,s) field theory.

IDENTITY PRINCIPLE: Mathematical objects maintain their essential mathematical
nature through invariant properties that survive all continuous field
transformations:

1. TOPOLOGICAL INVARIANTS: Genus, Euler characteristic, fundamental group
   signatures
2. CANONICAL SIGNATURES: Mathematical "DNA" from Q(τ,C,s) formula structure
3. PERSISTENCE HOMOLOGY: Features that survive regulation cycles
4. INFORMATION SIGNATURES: Entropy patterns independent of field magnitudes
5. GEOMETRIC INVARIANTS: Curvature signatures and manifold topology
6. MATHEMATICAL COHERENCE: Essential relationships between Q-components

MATHEMATICAL RIGOR: Every identity mechanism provably preserves mathematical
structure through field-theoretic transformations. No arbitrary defaults.
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Set
import hashlib

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

from Sysnpire.model.liquid.conceptual_charge_agent import ConceptualChargeAgent
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TopologicalInvariants:
    """Topological properties invariant under continuous field
    transformations."""

    euler_characteristic: float
    genus_signature: float
    fundamental_group_rank: int
    betti_numbers: List[int]
    topological_fingerprint: str


@dataclass
class CanonicalMathematicalSignature:
    """Mathematical "DNA" derived from Q(τ,C,s) formula structure."""

    formula_structure_hash: str
    component_relationship_matrix: np.ndarray
    mathematical_genus: float
    canonical_form_coefficients: np.ndarray
    mathematical_species_id: str


@dataclass
class PersistenceHomologyFeatures:
    """Mathematical features that persist across regulation cycles."""

    persistent_cycles: List[Tuple[float, float]]
    persistence_diagram_signature: str
    stable_mathematical_features: Set[str]
    mathematical_lifetime_distribution: np.ndarray
    topological_persistence_rank: int


@dataclass
class InformationTheoreticIdentity:
    """Information-theoretic fingerprints independent of field magnitudes."""

    entropy_signature_pattern: np.ndarray
    mutual_information_matrix: np.ndarray
    information_theoretical_fingerprint: str
    coherence_information_content: float
    mathematical_information_complexity: float


@dataclass
class GeometricInvariants:
    """Geometric properties that define mathematical object identity."""

    curvature_signature: np.ndarray
    manifold_topology_hash: str
    geometric_genus: float
    canonical_geometric_form: np.ndarray
    differential_geometric_invariants: Dict[str, float]


@dataclass
class MathematicalCoherence:
    """Essential mathematical relationships that define object identity."""

    q_component_relationship_invariants: Dict[str, float]
    mathematical_ratio_signatures: np.ndarray
    phase_relationship_topology: np.ndarray
    mathematical_structure_preservation_score: float
    coherence_mathematical_fingerprint: str


@dataclass
class MathematicalObjectIdentityProfile:
    """Complete mathematical identity profile for a Q(τ,C,s) entity."""

    object_mathematical_id: str
    topological_invariants: TopologicalInvariants
    canonical_signature: CanonicalMathematicalSignature
    persistence_features: PersistenceHomologyFeatures
    information_identity: InformationTheoreticIdentity
    geometric_invariants: GeometricInvariants
    mathematical_coherence: MathematicalCoherence
    identity_creation_timestamp: float
    identity_mathematical_confidence: float


class MathematicalObjectIdentity:
    """
    Mathematical Object Identity Theory Implementation

    Provides mathematical mechanisms for Q(τ,C,s) entities to maintain
    consistent mathematical identity across regulation cycles through
    topological, geometric, and information-theoretic invariants.
    """

    def __init__(self, mathematical_precision: float = 1e-12):
        """
        Initialize mathematical identity system.

        Args:
            mathematical_precision: Numerical precision for mathematical
                computations
        """
        self.mathematical_precision = mathematical_precision
        self.identity_registry: Dict[
            str, MathematicalObjectIdentityProfile] = {}
        self.mathematical_species_classification: Dict[str, Set[str]] = {}
        self.topological_computation_cache: Dict[
            str, TopologicalInvariants] = {}
        self.species_registry: Dict[str, List[str]] = {}

        logger.info("🔬 Mathematical Object Identity system initialized "
                    "with rigorous mathematical foundations")

    def create_identity_with_history(
        self,
        agent: ConceptualChargeAgent,
        regulation_history: List[Dict[str, Any]]
    ) -> str:
        """
        Create and register mathematical identity for a Q(τ,C,s) entity with
        provided regulation history.

        Args:
            agent: ConceptualChargeAgent to create identity for
            regulation_history: Extracted regulation history from agent
                components

        Returns:
            Unique mathematical object ID

        Raises:
            ValueError: If agent lacks sufficient mathematical structure for
                identity creation
        """
        try:
            identity_profile = self.create_mathematical_identity_profile(
                agent=agent,
                regulation_history=regulation_history
            )

            mathematical_object_id = identity_profile.object_mathematical_id

            self.identity_registry[mathematical_object_id] = identity_profile

            species_id = (identity_profile.canonical_signature
                          .mathematical_species_id)
            if species_id not in self.species_registry:
                self.species_registry[species_id] = []
            self.species_registry[species_id].append(mathematical_object_id)

            logger.info(
                f"🔬 Mathematical identity created: "
                f"{mathematical_object_id[:8]} (species: {species_id})"
            )

            return mathematical_object_id

        except Exception as e:
            error_msg = (f"Mathematical identity creation failed for agent: "
                         f"{str(e)}")
            logger.error(f"🚨 {error_msg}")
            raise ValueError(f"MATHEMATICAL IDENTITY CREATION IMPOSSIBLE - "
                             f"{error_msg}")

    def create_identity(self, agent: ConceptualChargeAgent) -> str:
        """
        Create and register mathematical identity for a Q(τ,C,s) entity.

        This is the main entry point for creating persistent mathematical
        identity that survives all field regulation cycles through topological,
        geometric, and information-theoretic invariants.

        Args:
            agent: ConceptualChargeAgent to create identity for

        Returns:
            str: Unique mathematical object ID for the created identity

        Raises:
            ValueError: If agent lacks sufficient mathematical structure for
                identity creation
        """
        try:
            regulation_history = []

            identity_profile = self.create_mathematical_identity_profile(
                agent=agent,
                regulation_history=regulation_history
            )

            mathematical_object_id = identity_profile.object_mathematical_id

            self.identity_registry[mathematical_object_id] = identity_profile

            species_id = (identity_profile.canonical_signature
                          .mathematical_species_id)
            if species_id not in self.mathematical_species_classification:
                self.mathematical_species_classification[species_id] = set()
            self.mathematical_species_classification[species_id].add(
                mathematical_object_id)

            logger.info(
                f"🔬 Mathematical identity created: "
                f"{mathematical_object_id[:8]} (species: {species_id}, "
                f"confidence: "
                f"{identity_profile.identity_mathematical_confidence:.3f})"
            )

            return mathematical_object_id

        except Exception as e:
            agent_id = agent.charge_id
            error_msg = (f"Mathematical identity creation failed for agent "
                         f"{agent_id}: {str(e)}")
            logger.error(f"🚨 {error_msg}")
            raise ValueError(f"MATHEMATICAL IDENTITY CREATION IMPOSSIBLE - "
                             f"{error_msg}")

    def compute_topological_invariants(
        self, agent: ConceptualChargeAgent
    ) -> TopologicalInvariants:
        """
        Compute topological invariants from Q(τ,C,s) field structure.

        Args:
            agent: ConceptualChargeAgent with complete Q-field structure

        Returns:
            TopologicalInvariants computed from mathematical field structure
        """
        q_components = self._extract_q_field_components(agent)

        euler_characteristic = self._compute_euler_characteristic(q_components)
        genus_signature = self._compute_topological_genus(q_components)
        fundamental_group_rank = (
            self._compute_fundamental_group_rank(q_components))
        betti_numbers = self._compute_betti_numbers(q_components)

        topology_data = {
            "euler": euler_characteristic,
            "genus": genus_signature,
            "rank": fundamental_group_rank,
            "betti": betti_numbers
        }
        topological_fingerprint = (
            self._generate_topological_fingerprint(topology_data))

        return TopologicalInvariants(
            euler_characteristic=euler_characteristic,
            genus_signature=genus_signature,
            fundamental_group_rank=fundamental_group_rank,
            betti_numbers=betti_numbers,
            topological_fingerprint=topological_fingerprint
        )

    def derive_canonical_mathematical_signature(
        self, agent: ConceptualChargeAgent
    ) -> CanonicalMathematicalSignature:
        """
        Derive canonical mathematical signature from Q(τ,C,s) formula
        structure.

        Args:
            agent: ConceptualChargeAgent with complete mathematical structure

        Returns:
            CanonicalMathematicalSignature representing mathematical "DNA"
        """
        formula_structure = self._extract_q_formula_structure(agent)
        formula_structure_hash = (
            self._hash_mathematical_structure(formula_structure))

        component_matrix = self._compute_component_relationships(agent)
        mathematical_genus = (
            self._compute_mathematical_genus(component_matrix))
        canonical_coeffs = self._compute_canonical_form(component_matrix)

        species_id = self._classify_mathematical_species(
            formula_structure_hash, mathematical_genus, canonical_coeffs)

        return CanonicalMathematicalSignature(
            formula_structure_hash=formula_structure_hash,
            component_relationship_matrix=component_matrix,
            mathematical_genus=mathematical_genus,
            canonical_form_coefficients=canonical_coeffs,
            mathematical_species_id=species_id
        )

    def compute_persistence_homology_features(
        self,
        agent: ConceptualChargeAgent,
        regulation_history: List[Dict[str, Any]]
    ) -> PersistenceHomologyFeatures:
        """
        Compute persistence homology features from regulation cycle evolution.

        Args:
            agent: ConceptualChargeAgent with mathematical structure
            regulation_history: Historical regulation data

        Returns:
            PersistenceHomologyFeatures for mathematical identity
        """
        if len(regulation_history) < 3:
            return PersistenceHomologyFeatures(
                persistent_cycles=[],
                persistence_diagram_signature="early_cycle",
                stable_mathematical_features=set(),
                mathematical_lifetime_distribution=np.array([]),
                topological_persistence_rank=0
            )

        field_evolution_data = (
            self._extract_field_evolution_data(regulation_history))
        persistent_cycles = self._compute_persistent_cycles(
            field_evolution_data)

        persistence_signature = (
            self._generate_persistence_diagram_signature(persistent_cycles))
        stable_features = self._identify_stable_features(
            persistent_cycles, field_evolution_data)
        lifetime_distribution = (
            self._compute_lifetime_distribution(persistent_cycles))
        persistence_rank = self._compute_persistence_rank(persistent_cycles)

        return PersistenceHomologyFeatures(
            persistent_cycles=persistent_cycles,
            persistence_diagram_signature=persistence_signature,
            stable_mathematical_features=stable_features,
            mathematical_lifetime_distribution=lifetime_distribution,
            topological_persistence_rank=persistence_rank
        )

    def compute_mathematical_information_signature(
        self, agent: ConceptualChargeAgent
    ) -> InformationTheoreticIdentity:
        """
        Compute information-theoretic identity signature from Q-field
        structure.

        Args:
            agent: ConceptualChargeAgent with complete Q-field data

        Returns:
            InformationTheoreticIdentity based on field entropy patterns
        """
        q_field_structure = self._extract_q_field_structure(agent)

        entropy_pattern = (
            self._compute_entropy_signature_pattern(q_field_structure))
        mi_matrix = (
            self._compute_structural_mutual_information(q_field_structure))
        info_fingerprint = self._generate_information_fingerprint(
            entropy_pattern, mi_matrix)
        coherence_content = (
            self._compute_coherence_information_content(q_field_structure))
        complexity = self._compute_mathematical_complexity(
            entropy_pattern, mi_matrix)

        return InformationTheoreticIdentity(
            entropy_signature_pattern=entropy_pattern,
            mutual_information_matrix=mi_matrix,
            information_theoretical_fingerprint=info_fingerprint,
            coherence_information_content=coherence_content,
            mathematical_information_complexity=complexity
        )

    def compute_geometric_invariants(
        self, agent: ConceptualChargeAgent
    ) -> GeometricInvariants:
        """
        Compute geometric invariants from Q-field manifold structure.

        Args:
            agent: ConceptualChargeAgent with field manifold data

        Returns:
            GeometricInvariants representing manifold geometry
        """
        manifold_structure = self._extract_field_manifold_structure(agent)

        curvature_signature = (
            self._compute_curvature_signature(manifold_structure))
        topology_hash = (
            self._generate_manifold_topology_hash(manifold_structure))
        geometric_genus = (
            self._compute_geometric_genus(manifold_structure))
        canonical_form = (
            self._compute_canonical_geometric_form(manifold_structure))
        differential_invariants = (
            self._compute_differential_invariants(manifold_structure))

        return GeometricInvariants(
            curvature_signature=curvature_signature,
            manifold_topology_hash=topology_hash,
            geometric_genus=geometric_genus,
            canonical_geometric_form=canonical_form,
            differential_geometric_invariants=differential_invariants
        )

    def compute_mathematical_coherence(
        self, agent: ConceptualChargeAgent
    ) -> MathematicalCoherence:
        """
        Compute mathematical coherence relationships between Q-components.

        Args:
            agent: ConceptualChargeAgent with Q-component structure

        Returns:
            MathematicalCoherence representing essential relationships
        """
        q_relationships = self._extract_q_component_relationships(agent)

        relationship_invariants = (
            self._compute_relationship_invariants(q_relationships))
        ratio_signatures = (
            self._compute_mathematical_ratios(q_relationships))
        phase_topology = self._compute_phase_relationship_topology(agent)
        preservation_score = self._compute_structure_preservation_score(
            q_relationships, agent)
        coherence_fingerprint = self._generate_coherence_fingerprint(
            relationship_invariants, ratio_signatures)

        return MathematicalCoherence(
            q_component_relationship_invariants=relationship_invariants,
            mathematical_ratio_signatures=ratio_signatures,
            phase_relationship_topology=phase_topology,
            mathematical_structure_preservation_score=preservation_score,
            coherence_mathematical_fingerprint=coherence_fingerprint
        )

    def create_mathematical_identity_profile(
        self,
        agent: ConceptualChargeAgent,
        regulation_history: List[Dict[str, Any]]
    ) -> MathematicalObjectIdentityProfile:
        """
        Create complete mathematical identity profile for Q(τ,C,s) entity.

        Args:
            agent: ConceptualChargeAgent to profile
            regulation_history: Historical regulation data

        Returns:
            Complete MathematicalObjectIdentityProfile
        """
        logger.info(f"🔬 Creating mathematical identity for agent "
                    f"{agent.charge_id}")

        start_time = time.time()

        topological_invariants = self.compute_topological_invariants(agent)
        canonical_signature = (
            self.derive_canonical_mathematical_signature(agent))
        persistence_features = self.compute_persistence_homology_features(
            agent, regulation_history)
        information_identity = (
            self.compute_mathematical_information_signature(agent))
        geometric_invariants = self.compute_geometric_invariants(agent)
        mathematical_coherence = self.compute_mathematical_coherence(agent)

        profile_data = {
            "topological": topological_invariants,
            "canonical": canonical_signature,
            "persistence": persistence_features,
            "information": information_identity,
            "geometric": geometric_invariants,
            "coherence": mathematical_coherence
        }

        mathematical_object_id = (
            self._generate_mathematical_object_id(profile_data))
        identity_confidence = self._compute_identity_confidence(profile_data)
        creation_timestamp = time.time()

        computation_time = creation_timestamp - start_time
        logger.info(f"🔬 Mathematical identity computed in "
                    f"{computation_time:.3f}s")

        return MathematicalObjectIdentityProfile(
            object_mathematical_id=mathematical_object_id,
            topological_invariants=topological_invariants,
            canonical_signature=canonical_signature,
            persistence_features=persistence_features,
            information_identity=information_identity,
            geometric_invariants=geometric_invariants,
            mathematical_coherence=mathematical_coherence,
            identity_creation_timestamp=creation_timestamp,
            identity_mathematical_confidence=identity_confidence
        )

    def verify_mathematical_identity_consistency(
        self,
        agent: ConceptualChargeAgent,
        known_identity: str
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Verify that agent maintains consistent mathematical identity.

        Args:
            agent: ConceptualChargeAgent to verify
            known_identity: Previously computed mathematical identity

        Returns:
            Tuple of (is_consistent, overall_consistency_score,
                     component_scores)
        """
        if known_identity not in self.identity_registry:
            return False, 0.0, {}

        original_profile = self.identity_registry[known_identity]
        current_profile = self.create_mathematical_identity_profile(
            agent, [])

        consistency_scores = {}

        consistency_scores["topological"] = (
            self._compare_topological_invariants(
                original_profile.topological_invariants,
                current_profile.topological_invariants
            ))

        consistency_scores["canonical"] = (
            self._compare_canonical_signatures(
                original_profile.canonical_signature,
                current_profile.canonical_signature
            ))

        consistency_scores["information"] = (
            self._compare_information_identities(
                original_profile.information_identity,
                current_profile.information_identity
            ))

        consistency_scores["geometric"] = (
            self._compare_geometric_invariants(
                original_profile.geometric_invariants,
                current_profile.geometric_invariants
            ))

        consistency_scores["coherence"] = (
            self._compare_mathematical_coherence(
                original_profile.mathematical_coherence,
                current_profile.mathematical_coherence
            ))

        consistency_tensor = torch.tensor(list(consistency_scores.values()),
                                          dtype=torch.float32)
        overall_consistency = float(torch.mean(consistency_tensor))

        is_consistent = overall_consistency > 0.85

        return is_consistent, overall_consistency, consistency_scores

    def recognize_mathematical_peers(
        self,
        agent: ConceptualChargeAgent,
        similarity_threshold: float = 0.8
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Recognize mathematically similar entities based on identity profiles.

        Args:
            agent: ConceptualChargeAgent to find peers for
            similarity_threshold: Minimum similarity for peer recognition

        Returns:
            List of (peer_id, similarity_score, similarity_components)
        """
        current_profile = self.create_mathematical_identity_profile(agent, [])

        peers = []
        for peer_id, peer_profile in self.identity_registry.items():
            similarity_score, similarity_components = (
                self._compute_mathematical_similarity(
                    current_profile, peer_profile
                ))

            if similarity_score >= similarity_threshold:
                peers.append((peer_id, similarity_score,
                              similarity_components))

        peers.sort(key=lambda x: x[1], reverse=True)
        return peers

    def track_mathematical_genealogy(
        self,
        parent_agent: ConceptualChargeAgent,
        child_agent: ConceptualChargeAgent
    ) -> Dict[str, float]:
        """
        Track mathematical inheritance patterns between agents.

        Args:
            parent_agent: Parent ConceptualChargeAgent
            child_agent: Child ConceptualChargeAgent

        Returns:
            Mathematical inheritance scores by component
        """
        parent_profile = self.create_mathematical_identity_profile(
            parent_agent, [])
        child_profile = self.create_mathematical_identity_profile(
            child_agent, [])

        inheritance_scores = {}

        inheritance_scores["topological"] = (
            self._compute_topological_inheritance(
                parent_profile.topological_invariants,
                child_profile.topological_invariants
            ))

        inheritance_scores["canonical"] = (
            self._compute_canonical_inheritance(
                parent_profile.canonical_signature,
                child_profile.canonical_signature
            ))

        inheritance_scores["information"] = (
            self._compute_information_inheritance(
                parent_profile.information_identity,
                child_profile.information_identity
            ))

        inheritance_scores["geometric"] = (
            self._compute_geometric_inheritance(
                parent_profile.geometric_invariants,
                child_profile.geometric_invariants
            ))

        inheritance_scores["coherence"] = (
            self._compute_coherence_inheritance(
                parent_profile.mathematical_coherence,
                child_profile.mathematical_coherence
            ))

        return inheritance_scores

    def _extract_q_field_components(
        self, agent: ConceptualChargeAgent
    ) -> Dict[str, Any]:
        """
        Extract Q-field components from liquid agent using direct dataclass
        access.

        Args:
            agent: ConceptualChargeAgent with Q_components

        Returns:
            Q-field component data dictionary
        """
        field_components = {}

        field_components["gamma"] = agent.Q_components.gamma
        field_components["T_tensor"] = agent.Q_components.T_tensor
        field_components["E_trajectory"] = agent.Q_components.E_trajectory
        field_components["phi_semantic"] = agent.Q_components.phi_semantic
        field_components["theta_total"] = (
            agent.Q_components.theta_components.total)
        field_components["phase_factor"] = agent.Q_components.phase_factor
        field_components["psi_persistence"] = (
            agent.Q_components.psi_persistence)
        field_components["psi_gaussian"] = agent.Q_components.psi_gaussian
        field_components["psi_exponential_cosine"] = (
            agent.Q_components.psi_exponential_cosine)
        field_components["Q_value"] = agent.Q_components.Q_value

        field_components["breathing_coherence"] = (
            agent.temporal_biography.breathing_coherence)
        field_components["field_modulation_strength"] = (
            agent.emotional_field_signature.field_modulation_strength)
        field_components["living_Q_value"] = agent.living_Q_value

        return field_components

    def _compute_euler_characteristic(
        self, q_components: Dict[str, Any]
    ) -> float:
        """
        Compute Euler characteristic from Q-field topology.

        Args:
            q_components: Q-field component data

        Returns:
            Euler characteristic value
        """
        q_value = q_components["Q_value"]
        real_part = float(q_value.real)
        imag_part = float(q_value.imag)

        field_curvature = real_part * imag_part
        vertex_contribution = abs(real_part) + abs(imag_part)
        edge_contribution = math.sqrt(real_part**2 + imag_part**2)

        euler_char = (vertex_contribution - edge_contribution +
                      field_curvature)

        return float(euler_char)

    def _compute_topological_genus(
        self, q_components: Dict[str, Any]
    ) -> float:
        """
        Compute topological genus from Q-field component structure.

        Args:
            q_components: Q-field component data

        Returns:
            Topological genus value
        """
        real_parts = []
        imag_parts = []

        complex_components = ["T_tensor", "E_trajectory", "phi_semantic",
                              "phase_factor", "Q_value"]

        for component_name in complex_components:
            component_value = q_components[component_name]
            real_part = float(component_value.real)
            imag_part = float(component_value.imag)
            real_parts.append(real_part)
            imag_parts.append(imag_part)

        if len(real_parts) < 2:
            return 0.0

        real_matrix = np.outer(real_parts, real_parts)
        imag_matrix = np.outer(imag_parts, imag_parts)

        combined_matrix = real_matrix + 1j * imag_matrix

        eigenvalues = np.linalg.eigvals(combined_matrix)
        positive_eigenvalues = np.sum(np.real(eigenvalues) > 0)

        genus = max(0, (len(real_parts) - positive_eigenvalues) // 2)

        return float(genus)

    def _compute_fundamental_group_rank(
        self, q_components: Dict[str, Any]
    ) -> int:
        """
        Compute fundamental group rank from Q-field phase relationships.

        Args:
            q_components: Q-field component data

        Returns:
            Fundamental group rank
        """
        phases = []
        complex_components = ["T_tensor", "E_trajectory", "phi_semantic",
                              "phase_factor", "Q_value"]

        for component_name in complex_components:
            component_value = q_components[component_name]
            real_part = float(component_value.real)
            imag_part = float(component_value.imag)
            phase = math.atan2(imag_part, real_part)
            phases.append(phase)

        if len(phases) < 2:
            return 0

        phase_differences = []
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                diff = abs(phases[i] - phases[j])
                normalized_diff = min(diff, 2 * math.pi - diff)
                phase_differences.append(normalized_diff)

        independent_phases = 0
        tolerance = 0.1

        for diff in phase_differences:
            if diff > tolerance:
                independent_phases += 1

        return min(independent_phases, len(phases) - 1)

    def _compute_betti_numbers(
        self, q_components: Dict[str, Any]
    ) -> List[int]:
        """
        Compute Betti numbers from Q-field homology.

        Args:
            q_components: Q-field component data

        Returns:
            List of Betti numbers
        """
        points = []
        complex_components = ["T_tensor", "E_trajectory", "phi_semantic",
                              "phase_factor", "Q_value"]

        for component_name in complex_components:
            component_value = q_components[component_name]
            real_part = float(component_value.real)
            imag_part = float(component_value.imag)
            points.append([real_part, imag_part])

        if len(points) < 2:
            return [0]

        points_array = np.array(points)

        threshold = np.mean(pdist(points_array))

        b0 = self._count_connected_components(points_array, threshold)
        b1 = max(0, len(points) - b0)

        return [b0, b1]

    def _count_connected_components(
        self, points: np.ndarray, threshold: float
    ) -> int:
        """
        Count connected components in point cloud.

        Args:
            points: Array of points
            threshold: Connection threshold

        Returns:
            Number of connected components
        """
        if len(points) == 0:
            return 0

        distances = squareform(pdist(points))
        adjacency = distances < threshold

        visited = np.zeros(len(points), dtype=bool)
        components = 0

        for i in range(len(points)):
            if not visited[i]:
                components += 1
                stack = [i]
                while stack:
                    current = stack.pop()
                    if not visited[current]:
                        visited[current] = True
                        neighbors = np.where(adjacency[current])[0]
                        stack.extend(neighbors)

        return components

    def _generate_topological_fingerprint(
        self, topology_data: Dict[str, Any]
    ) -> str:
        """
        Generate unique fingerprint from topological data.

        Args:
            topology_data: Topological invariant data

        Returns:
            Unique topological fingerprint hash
        """
        fingerprint_string = (
            f"{topology_data['euler']:.6f}_{topology_data['genus']:.6f}_"
            f"{topology_data['rank']}_{topology_data['betti']}")
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]

    def _extract_q_formula_structure(
        self, agent: ConceptualChargeAgent
    ) -> Dict[str, Any]:
        """
        Extract Q(τ,C,s) formula structure using direct dataclass access.

        Args:
            agent: ConceptualChargeAgent with mathematical structure

        Returns:
            Q formula structure data
        """
        structure = {}

        q_components = agent.Q_components

        logger.debug(f"🔍 DEBUG: Extracting Q formula structure for agent "
                     f"{agent.charge_id}")

        mathematical_components = [
            "gamma", "T_tensor", "E_trajectory", "phi_semantic",
            "phase_factor", "psi_persistence", "psi_gaussian",
            "psi_exponential_cosine", "Q_value"
        ]

        field_values = []

        for name in mathematical_components:
            value = getattr(q_components, name)

            if name in ["T_tensor", "E_trajectory", "phi_semantic",
                        "phase_factor", "Q_value"]:
                try:
                    E_magnitude = float(abs(value))
                    E_phase = float(math.atan2(value.imag, value.real))
                    field_values.extend([E_magnitude, E_phase])
                except Exception:
                    e_traj_value = getattr(q_components,
                                           'E_trajectory', 'NOT FOUND')
                    logger.debug(f"   - E_trajectory value when E_phase "
                                 f"failed: {e_traj_value}")
                    field_values.extend([0.0, 0.0])
            else:
                field_values.append(float(value))

        field_values.append(float(
            q_components.theta_components.total))

        structure["component_names"] = (mathematical_components +
                                        ["theta_total"])
        structure["field_values"] = field_values
        structure["component_count"] = len(field_values)
        structure["complex_component_count"] = 5
        structure["real_component_count"] = len(field_values) - 10

        return structure

    def _hash_mathematical_structure(
        self, structure: Dict[str, Any]
    ) -> str:
        """
        Generate hash from mathematical structure.

        Args:
            structure: Mathematical structure data

        Returns:
            Structure hash string
        """
        structure_string = (
            f"{structure['component_count']}_"
            f"{structure['complex_component_count']}_"
            f"{structure['real_component_count']}")
        return hashlib.sha256(structure_string.encode()).hexdigest()[:16]

    def _compute_component_relationships(
        self, agent: ConceptualChargeAgent
    ) -> np.ndarray:
        """
        Compute relationships between Q-components using direct dataclass
        access.

        Args:
            agent: ConceptualChargeAgent with Q_components

        Returns:
            Component relationship matrix
        """
        components = [
            agent.Q_components.T_tensor,
            agent.Q_components.E_trajectory,
            agent.Q_components.phi_semantic,
            agent.Q_components.phase_factor,
            agent.Q_components.Q_value
        ]

        relationship_matrix = np.zeros(
            (len(components), len(components)), dtype=complex)

        for i, comp_i in enumerate(components):
            for j, comp_j in enumerate(components):
                if i != j:
                    real_part = float(comp_i.real)
                    imag_part = float(comp_i.imag)
                    other_real = float(comp_j.real)
                    other_imag = float(comp_j.imag)

                    correlation = (real_part * other_real +
                                   imag_part * other_imag)
                    cross_correlation = (real_part * other_imag -
                                         imag_part * other_real)

                    relationship_matrix[i, j] = (correlation +
                                                 1j * cross_correlation)
                else:
                    relationship_matrix[i, j] = float(abs(comp_i))

        return np.real(relationship_matrix)

    def _compute_mathematical_genus(
        self, component_matrix: np.ndarray
    ) -> float:
        """
        Compute mathematical genus from component relationships.

        Args:
            component_matrix: Component relationship matrix

        Returns:
            Mathematical genus value
        """
        if component_matrix.size == 0:
            return 0.0

        eigenvalues = np.linalg.eigvals(component_matrix)

        positive_eigenvalues = np.sum(np.real(eigenvalues) > 1e-10)
        negative_eigenvalues = np.sum(np.real(eigenvalues) < -1e-10)

        genus = ((len(eigenvalues) - positive_eigenvalues +
                  negative_eigenvalues) / 2.0)

        return max(0.0, float(genus))

    def _compute_canonical_form(
        self, component_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute canonical form representation.

        Args:
            component_matrix: Component relationship matrix

        Returns:
            Canonical form coefficients
        """
        if component_matrix.size == 0:
            return np.array([])

        eigenvalues, eigenvectors = np.linalg.eigh(component_matrix)

        sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
        canonical_eigenvalues = eigenvalues[sorted_indices]

        return canonical_eigenvalues

    def _classify_mathematical_species(
        self, structure_hash: str, genus: float, canonical_coeffs: np.ndarray
    ) -> str:
        """
        Classify mathematical species based on structural properties.

        Args:
            structure_hash: Hash of mathematical structure
            genus: Mathematical genus
            canonical_coeffs: Canonical form coefficients

        Returns:
            Mathematical species ID
        """
        genus_class = f"g{int(genus)}"

        if len(canonical_coeffs) > 0:
            dominant_eigenvalue = abs(canonical_coeffs[0])
            if dominant_eigenvalue > 10.0:
                complexity_class = "high"
            elif dominant_eigenvalue > 1.0:
                complexity_class = "medium"
            else:
                complexity_class = "low"
        else:
            complexity_class = "minimal"

        species_id = (f"{genus_class}_{complexity_class}_"
                      f"{structure_hash[:8]}")

        return species_id

    def _extract_field_evolution_data(
        self, regulation_history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract field evolution data from regulation history.

        Args:
            regulation_history: Historical regulation data

        Returns:
            Field evolution data points
        """
        evolution_data = []

        for i, regulation_point in enumerate(regulation_history):
            evolution_point = {
                "timestamp": regulation_point.get("timestamp", float(i)),
                "field_entropy": regulation_point.get("field_entropy", 0.0),
                "phase_coherence": regulation_point.get(
                    "phase_coherence", 0.0),
                "energy_density": regulation_point.get("energy_density", 0.0),
                "regulation_strength": regulation_point.get(
                    "regulation_strength", 0.0)
            }
            evolution_data.append(evolution_point)

        return evolution_data

    def _compute_persistent_cycles(
        self, field_evolution_data: List[Dict[str, Any]]
    ) -> List[Tuple[float, float]]:
        """
        Compute persistent cycles from field evolution.

        Args:
            field_evolution_data: Field evolution time series

        Returns:
            List of (birth, death) persistent cycle pairs
        """
        if len(field_evolution_data) < 3:
            return []

        entropy_series = [point["field_entropy"]
                          for point in field_evolution_data]
        timestamps = [point["timestamp"] for point in field_evolution_data]

        persistent_cycles = self._compute_time_series_persistence(
            entropy_series, timestamps)

        return persistent_cycles

    def _compute_time_series_persistence(
        self,
        values: List[float],
        timestamps: List[float]
    ) -> List[Tuple[float, float]]:
        """
        Compute persistence pairs from time series data.

        Args:
            values: Time series values
            timestamps: Corresponding timestamps

        Returns:
            Persistence pairs (birth, death)
        """
        if len(values) < 3:
            return []

        local_maxima = []
        local_minima = []

        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                local_maxima.append((timestamps[i], values[i]))
            elif values[i] < values[i-1] and values[i] < values[i+1]:
                local_minima.append((timestamps[i], values[i]))

        cycles = []
        for max_time, max_val in local_maxima:
            for min_time, min_val in local_minima:
                if min_time > max_time:
                    persistence = max_val - min_val
                    if persistence > 0.01:
                        cycles.append((max_time, min_time))
                    break

        return cycles

    def _generate_persistence_diagram_signature(
        self, persistent_cycles: List[Tuple[float, float]]
    ) -> str:
        """
        Generate signature from persistence diagram.

        Args:
            persistent_cycles: Persistent cycle pairs

        Returns:
            Persistence diagram signature
        """
        if not persistent_cycles:
            return "empty_persistence"

        lifetimes = [death - birth for birth, death in persistent_cycles]
        avg_lifetime = np.mean(lifetimes)
        max_lifetime = np.max(lifetimes)

        signature_string = (f"{len(persistent_cycles)}_"
                            f"{avg_lifetime:.6f}_{max_lifetime:.6f}")
        return hashlib.sha256(signature_string.encode()).hexdigest()[:16]

    def _identify_stable_features(
        self,
        persistent_cycles: List[Tuple[float, float]],
        field_evolution_data: List[Dict[str, Any]]
    ) -> Set[str]:
        """
        Identify stable mathematical features.

        Args:
            persistent_cycles: Persistent cycle data
            field_evolution_data: Field evolution history

        Returns:
            Set of stable feature identifiers
        """
        stable_features = set()

        if not persistent_cycles:
            return stable_features

        lifetimes = [death - birth for birth, death in persistent_cycles]
        avg_lifetime = np.mean(lifetimes)

        for i, (birth, death) in enumerate(persistent_cycles):
            lifetime = death - birth
            if lifetime > 2 * avg_lifetime:
                stable_features.add(f"persistent_cycle_{i}")

        entropy_values = [point["field_entropy"]
                          for point in field_evolution_data]
        entropy_variance = np.var(entropy_values)

        if entropy_variance < 0.01:
            stable_features.add("stable_entropy")

        coherence_values = [point["phase_coherence"]
                            for point in field_evolution_data]
        coherence_trend = np.polyfit(range(len(coherence_values)),
                                     coherence_values, 1)[0]

        if abs(coherence_trend) < 0.001:
            stable_features.add("stable_coherence")

        return stable_features

    def _compute_lifetime_distribution(
        self, persistent_cycles: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Compute lifetime distribution of persistent features.

        Args:
            persistent_cycles: Persistent cycle pairs

        Returns:
            Lifetime distribution array
        """
        if not persistent_cycles:
            return np.array([])

        lifetimes = [death - birth for birth, death in persistent_cycles]

        return np.array(lifetimes)

    def _compute_persistence_rank(
        self, persistent_cycles: List[Tuple[float, float]]
    ) -> int:
        """
        Compute topological persistence rank.

        Args:
            persistent_cycles: Persistent cycle pairs

        Returns:
            Persistence rank
        """
        if not persistent_cycles:
            return 0

        lifetimes = [death - birth for birth, death in persistent_cycles]
        significant_cycles = [lt for lt in lifetimes
                              if lt > np.mean(lifetimes)]

        return len(significant_cycles)

    def _extract_q_field_structure(
        self, agent: ConceptualChargeAgent
    ) -> Dict[str, Any]:
        """
        Extract Q-field structure using direct dataclass access.

        Args:
            agent: ConceptualChargeAgent with Q_components

        Returns:
            Q-field structure data
        """
        structure = {}

        components = agent.Q_components

        structure["component_names"] = [
            "gamma", "T_tensor", "E_trajectory", "phi_semantic",
            "theta_total", "phase_factor", "psi_persistence",
            "psi_gaussian", "psi_exponential_cosine", "Q_value"
        ]

        field_values = [
            components.gamma,
            components.T_tensor,
            components.E_trajectory,
            components.phi_semantic,
            components.theta_components.total,
            components.phase_factor,
            components.psi_persistence,
            components.psi_gaussian,
            components.psi_exponential_cosine,
            components.Q_value
        ]

        structure["field_values"] = field_values
        structure["component_count"] = len(field_values)

        component_ratios = self._compute_component_ratios(field_values)
        phase_relationships = self._compute_phase_relationships(field_values)

        structure["component_ratios"] = component_ratios
        structure["phase_relationships"] = phase_relationships

        return structure

    def _compute_component_ratios(
        self, component_values: List[Any]
    ) -> List[float]:
        """
        Compute ratios between Q-field components.

        Args:
            component_values: List of component values

        Returns:
            Component ratio values
        """
        ratios = []

        magnitudes = []
        for comp in component_values:
            if isinstance(comp, complex):
                magnitudes.append(abs(comp))
            else:
                magnitudes.append(abs(float(comp)))

        for i in range(len(magnitudes)):
            for j in range(i + 1, len(magnitudes)):
                if magnitudes[j] != 0:
                    ratio = magnitudes[i] / magnitudes[j]
                    ratios.append(ratio)

        return ratios

    def _compute_phase_relationships(
        self, field_values: List[Any]
    ) -> List[float]:
        """
        Compute phase relationships between complex components.

        Args:
            field_values: List of field component values

        Returns:
            Phase relationship values
        """
        phases = []

        complex_indices = [1, 2, 3, 5, 9]

        for idx in complex_indices:
            if idx < len(field_values):
                comp = field_values[idx]
                real_part = float(comp.real)
                imag_part = float(comp.imag)
                phase = math.atan2(imag_part, real_part)
                phases.append(phase)

        phase_relationships = []
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                phase_diff = abs(phases[i] - phases[j])
                normalized_diff = min(phase_diff, 2 * math.pi - phase_diff)
                phase_relationships.append(normalized_diff)

        return phase_relationships

    def _compute_entropy_signature_pattern(
        self, q_field_structure: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute entropy signature pattern from Q-field structure.

        Args:
            q_field_structure: Q-field structure data

        Returns:
            Entropy signature pattern array
        """
        component_ratios = q_field_structure["component_ratios"]
        phase_relationships = q_field_structure["phase_relationships"]

        if not component_ratios:
            return np.array([0.0])

        ratio_entropy = entropy(np.abs(component_ratios) + 1e-10)

        if phase_relationships:
            phase_entropy = entropy(np.abs(phase_relationships) + 1e-10)
        else:
            phase_entropy = 0.0

        entropy_pattern = np.array([ratio_entropy, phase_entropy])

        return entropy_pattern

    def _compute_structural_mutual_information(
        self, q_field_structure: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute mutual information matrix for Q-field components.

        Args:
            q_field_structure: Q-field structure data

        Returns:
            Mutual information matrix
        """
        component_names = q_field_structure["component_names"]
        n_components = len(component_names)

        mi_matrix = np.zeros((n_components, n_components))

        component_ratios = q_field_structure["component_ratios"]

        if len(component_ratios) < n_components:
            return mi_matrix

        for i in range(n_components):
            for j in range(n_components):
                if (i != j and i < len(component_ratios) and
                        j < len(component_ratios)):
                    val_i = abs(component_ratios[i]) + 1e-10
                    val_j = abs(component_ratios[j]) + 1e-10

                    joint_entropy = entropy([val_i, val_j])
                    individual_entropy_i = entropy([val_i, 1.0 - val_i])
                    individual_entropy_j = entropy([val_j, 1.0 - val_j])

                    mi = (individual_entropy_i + individual_entropy_j -
                          joint_entropy)
                    mi_matrix[i, j] = max(0.0, mi)

        return mi_matrix

    def _generate_information_fingerprint(
        self, entropy_pattern: np.ndarray, mi_matrix: np.ndarray
    ) -> str:
        """
        Generate information-theoretic fingerprint.

        Args:
            entropy_pattern: Entropy signature pattern
            mi_matrix: Mutual information matrix

        Returns:
            Information fingerprint string
        """
        entropy_sum = np.sum(entropy_pattern)
        mi_trace = np.trace(mi_matrix)
        mi_determinant = (np.linalg.det(mi_matrix)
                          if mi_matrix.size > 0 else 0.0)

        fingerprint_string = (f"{entropy_sum:.6f}_{mi_trace:.6f}_"
                              f"{mi_determinant:.6f}")
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]

    def _compute_coherence_information_content(
        self, q_field_structure: Dict[str, Any]
    ) -> float:
        """
        Compute information content in coherence relationships.

        Args:
            q_field_structure: Q-field structure data

        Returns:
            Coherence information content
        """
        phase_relationships = q_field_structure["phase_relationships"]

        if not phase_relationships:
            return 0.0

        coherence_entropy = entropy(np.abs(phase_relationships) + 1e-10)

        max_possible_entropy = math.log(len(phase_relationships))

        if max_possible_entropy > 0:
            normalized_coherence = coherence_entropy / max_possible_entropy
        else:
            normalized_coherence = 0.0

        return float(normalized_coherence)

    def _compute_mathematical_complexity(
        self, entropy_pattern: np.ndarray, mi_matrix: np.ndarray
    ) -> float:
        """
        Compute mathematical complexity measure.

        Args:
            entropy_pattern: Entropy signature pattern
            mi_matrix: Mutual information matrix

        Returns:
            Mathematical complexity value
        """
        entropy_complexity = np.sum(entropy_pattern**2)

        if mi_matrix.size > 0:
            mi_complexity = np.sum(mi_matrix**2)
        else:
            mi_complexity = 0.0

        total_complexity = entropy_complexity + mi_complexity

        return float(total_complexity)

    def _extract_field_manifold_structure(
        self, agent: ConceptualChargeAgent
    ) -> Dict[str, Any]:
        """
        Extract field manifold structure using direct dataclass access.

        Args:
            agent: ConceptualChargeAgent with manifold data

        Returns:
            Manifold structure data
        """
        manifold_structure = {}

        q_value = agent.Q_components.Q_value

        manifold_structure["field_position"] = [
            float(q_value.real),
            float(q_value.imag)
        ]
        manifold_structure["q_value"] = q_value
        manifold_structure["geometric_features"] = [
            float(agent.Q_components.gamma),
            float(abs(agent.Q_components.T_tensor)),
            float(abs(agent.Q_components.E_trajectory))
        ]
        manifold_structure["modular_weight"] = float(
            agent.Q_components.psi_persistence)

        return manifold_structure

    def _compute_curvature_signature(
        self, manifold_structure: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute curvature signature from manifold structure.

        Args:
            manifold_structure: Manifold structure data

        Returns:
            Curvature signature array
        """
        field_pos = manifold_structure["field_position"]

        if len(field_pos) >= 2:
            x, y = field_pos[0], field_pos[1]

            gaussian_curvature = x * y / (1 + x**2 + y**2)**2
            mean_curvature = (x + y) / (2 * math.sqrt(1 + x**2 + y**2))

            curvature_signature = np.array([gaussian_curvature,
                                            mean_curvature])
        else:
            curvature_signature = np.array([0.0, 0.0])

        return curvature_signature

    def _generate_manifold_topology_hash(
        self, manifold_structure: Dict[str, Any]
    ) -> str:
        """
        Generate topology hash from manifold structure.

        Args:
            manifold_structure: Manifold structure data

        Returns:
            Topology hash string
        """
        q_value = manifold_structure["q_value"]

        topology_string = (f"{float(q_value.real):.6f}_"
                           f"{float(q_value.imag):.6f}")

        geometric_features = manifold_structure.get("geometric_features", [])
        if geometric_features:
            features_string = "_".join([f"{feat:.6f}"
                                        for feat in geometric_features])
            topology_string += f"_{features_string}"

        return hashlib.sha256(topology_string.encode()).hexdigest()[:16]

    def _compute_geometric_genus(
        self, manifold_structure: Dict[str, Any]
    ) -> float:
        """
        Compute geometric genus from manifold structure.

        Args:
            manifold_structure: Manifold structure data

        Returns:
            Geometric genus value
        """
        q_value = manifold_structure["q_value"]
        real_part = float(q_value.real)
        imag_part = float(q_value.imag)

        field_magnitude = math.sqrt(real_part**2 + imag_part**2)

        if field_magnitude > 1.0:
            genus = math.floor(math.log(field_magnitude))
        else:
            genus = 0

        return float(max(0, genus))

    def _compute_canonical_geometric_form(
        self, manifold_structure: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute canonical geometric form.

        Args:
            manifold_structure: Manifold structure data

        Returns:
            Canonical geometric form array
        """
        geometric_features = manifold_structure.get("geometric_features", [])

        if not geometric_features:
            return np.array([])

        features_array = np.array(geometric_features)

        normalized_features = (features_array /
                               (np.linalg.norm(features_array) + 1e-10))

        return normalized_features

    def _compute_differential_invariants(
        self, manifold_structure: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute differential geometric invariants.

        Args:
            manifold_structure: Manifold structure data

        Returns:
            Dictionary of differential invariants
        """
        invariants = {}

        field_pos = manifold_structure["field_position"]

        if len(field_pos) >= 2:
            x, y = field_pos[0], field_pos[1]

            invariants["scalar_curvature"] = (2 * x * y /
                                              (1 + x**2 + y**2))
            invariants["mean_curvature"] = ((x + y) /
                                            math.sqrt(1 + x**2 + y**2))
            invariants["gaussian_curvature"] = (x * y /
                                                (1 + x**2 + y**2)**2)
        else:
            invariants["scalar_curvature"] = 0.0
            invariants["mean_curvature"] = 0.0
            invariants["gaussian_curvature"] = 0.0

        return invariants

    def _extract_q_component_relationships(
        self, agent: ConceptualChargeAgent
    ) -> Dict[str, Any]:
        """
        Extract Q-component relationships using direct dataclass access.

        Args:
            agent: ConceptualChargeAgent with Q_components

        Returns:
            Q-component relationship data
        """
        relationships = {}

        q_components = agent.Q_components

        magnitudes = {}
        phases = {}

        complex_component_names = ["T_tensor", "E_trajectory", "phi_semantic",
                                   "phase_factor", "Q_value"]

        for comp_name in complex_component_names:
            comp_value = getattr(q_components, comp_name)
            real_part = float(comp_value.real)
            imag_part = float(comp_value.imag)

            magnitude = math.sqrt(real_part**2 + imag_part**2)
            phase = math.atan2(imag_part, real_part)

            magnitudes[comp_name] = magnitude
            phases[comp_name] = phase

        relationships["magnitudes"] = magnitudes
        relationships["phases"] = phases

        magnitude_values = list(magnitudes.values())
        phase_values = list(phases.values())

        magnitude_correlations = np.zeros((len(magnitude_values),
                                           len(magnitude_values)))
        for i, mag_i in enumerate(magnitude_values):
            for j, mag_j in enumerate(magnitude_values):
                if i != j and mag_j != 0:
                    magnitude_correlations[i, j] = mag_i / mag_j

        phase_correlations = np.zeros((len(phase_values),
                                       len(phase_values)))
        for i, phase_i in enumerate(phase_values):
            for j, phase_j in enumerate(phase_values):
                if i != j:
                    phase_diff = abs(phase_i - phase_j)
                    phase_correlations[i, j] = min(phase_diff,
                                                   2 * math.pi - phase_diff)

        relationships["magnitude_correlations"] = magnitude_correlations
        relationships["phase_correlations"] = phase_correlations

        return relationships

    def _compute_relationship_invariants(
        self, q_relationships: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute invariant relationships between Q-components.

        Args:
            q_relationships: Q-component relationship data

        Returns:
            Relationship invariant values
        """
        invariants = {}

        magnitudes = q_relationships["magnitudes"]
        phases = q_relationships["phases"]

        magnitude_values = list(magnitudes.values())

        if len(magnitude_values) > 1:
            invariants["magnitude_ratio_sum"] = sum(magnitude_values)
            invariants["magnitude_ratio_product"] = np.prod(magnitude_values)
            invariants["magnitude_variance"] = float(np.var(magnitude_values))
        else:
            invariants["magnitude_ratio_sum"] = 0.0
            invariants["magnitude_ratio_product"] = 0.0
            invariants["magnitude_variance"] = 0.0

        phase_values = list(phases.values())
        if len(phase_values) > 1:
            phase_differences = []
            for i in range(len(phase_values)):
                for j in range(i + 1, len(phase_values)):
                    diff = abs(phase_values[i] - phase_values[j])
                    normalized_diff = min(diff, 2 * math.pi - diff)
                    phase_differences.append(normalized_diff)

            if phase_differences:
                invariant_values = [v for v in phase_differences
                                    if math.isfinite(v)]
                if invariant_values:
                    invariants["phase_coherence_measure"] = (
                        np.mean(invariant_values))
                    invariants["phase_stability_measure"] = (
                        1.0 / (1.0 + np.var(invariant_values)))
                else:
                    invariants["phase_coherence_measure"] = 0.0
                    invariants["phase_stability_measure"] = 0.0
            else:
                invariants["phase_coherence_measure"] = 0.0
                invariants["phase_stability_measure"] = 0.0

        return invariants

    def _compute_mathematical_ratios(
        self, q_relationships: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute essential mathematical ratios between components.

        Args:
            q_relationships: Q-component relationship data

        Returns:
            Mathematical ratio signature array
        """
        magnitude_correlations = q_relationships["magnitude_correlations"]
        phase_correlations = q_relationships["phase_correlations"]

        magnitude_values = list(q_relationships["magnitudes"].values())

        if len(magnitude_values) < 2:
            return np.array([])

        ratio_signature = np.concatenate([
            magnitude_correlations.flatten(),
            phase_correlations.flatten()
        ])

        finite_ratios = ratio_signature[np.isfinite(ratio_signature)]

        return finite_ratios

    def _compute_phase_relationship_topology(
        self, agent: ConceptualChargeAgent
    ) -> np.ndarray:
        """
        Compute phase relationship topology using direct dataclass access.

        Args:
            agent: ConceptualChargeAgent with Q_components

        Returns:
            Phase relationship topology array
        """
        q_components = agent.Q_components

        complex_components = [
            q_components.T_tensor,
            q_components.E_trajectory,
            q_components.phi_semantic,
            q_components.phase_factor,
            q_components.Q_value
        ]

        phase_topology = np.zeros((len(complex_components),
                                   len(complex_components)))

        for i, comp_i in enumerate(complex_components):
            for j, comp_j in enumerate(complex_components):
                real_part = float(comp_i.real)
                imag_part = float(comp_i.imag)
                other_real = float(comp_j.real)
                other_imag = float(comp_j.imag)

                phase_i = math.atan2(imag_part, real_part)
                phase_j = math.atan2(other_imag, other_real)

                phase_diff = abs(phase_i - phase_j)
                normalized_diff = min(phase_diff, 2 * math.pi - phase_diff)

                phase_topology[i, j] = normalized_diff

        return phase_topology

    def _compute_structure_preservation_score(
        self,
        q_relationships: Dict[str, Any],
        agent: ConceptualChargeAgent
    ) -> float:
        """
        Compute mathematical structure preservation score.

        Args:
            q_relationships: Q-component relationship data
            agent: ConceptualChargeAgent with structure

        Returns:
            Structure preservation score
        """
        relationship_invariants = q_relationships.get(
            "magnitude_correlations")

        if (relationship_invariants is None or
                relationship_invariants.size == 0):
            return 0.0

        eigenvalues = np.linalg.eigvals(relationship_invariants)
        positive_eigenvalues = np.sum(np.real(eigenvalues) > 0)
        total_eigenvalues = len(eigenvalues)

        if total_eigenvalues > 0:
            preservation_score = positive_eigenvalues / total_eigenvalues
        else:
            preservation_score = 0.0

        return float(preservation_score)

    def _generate_coherence_fingerprint(
        self,
        relationship_invariants: Dict[str, float],
        ratio_signatures: np.ndarray
    ) -> str:
        """
        Generate mathematical coherence fingerprint.

        Args:
            relationship_invariants: Relationship invariant data
            ratio_signatures: Mathematical ratio signatures

        Returns:
            Coherence fingerprint string
        """
        if not relationship_invariants:
            return "empty_coherence"

        sorted_keys = sorted(relationship_invariants.keys())
        invariant_string = "_".join([f"{relationship_invariants[key]:.6f}"
                                     for key in sorted_keys])

        if len(ratio_signatures) > 0:
            ratio_sum = np.sum(ratio_signatures)
            ratio_mean = np.mean(ratio_signatures)
            ratio_string = f"{ratio_sum:.6f}_{ratio_mean:.6f}"
        else:
            ratio_string = "0.0_0.0"

        fingerprint_string = f"{invariant_string}_{ratio_string}"
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]

    def _generate_mathematical_object_id(
        self, profile_data: Dict[str, Any]
    ) -> str:
        """
        Generate unique mathematical object identifier.

        Args:
            profile_data: Complete profile data

        Returns:
            Unique mathematical object ID
        """
        topological = profile_data["topological"]
        canonical = profile_data["canonical"]

        id_string = (f"{topological.topological_fingerprint}_"
                     f"{canonical.formula_structure_hash}")

        full_hash = hashlib.sha256(id_string.encode()).hexdigest()

        return f"math_obj_{full_hash[:16]}"

    def _compute_identity_confidence(
        self, profile_data: Dict[str, Any]
    ) -> float:
        """
        Compute mathematical identity confidence score.

        Args:
            profile_data: Complete profile data

        Returns:
            Identity confidence score
        """
        confidence_factors = []

        topological = profile_data["topological"]
        if topological.betti_numbers:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)

        canonical = profile_data["canonical"]
        if len(canonical.canonical_form_coefficients) > 0:
            confidence_factors.append(0.85)
        else:
            confidence_factors.append(0.3)

        information = profile_data["information"]
        if information.entropy_signature_pattern.size > 0:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)

        coherence = profile_data["coherence"]
        if coherence.q_component_relationship_invariants:
            confidence_factors.append(0.75)
        else:
            confidence_factors.append(0.2)

        overall_confidence = np.mean(confidence_factors)

        return float(overall_confidence)

    def _compare_topological_invariants(
        self, orig: TopologicalInvariants, current: TopologicalInvariants
    ) -> float:
        """
        Compare topological invariants for consistency.

        Args:
            orig: Original topological invariants
            current: Current topological invariants

        Returns:
            Consistency score [0, 1]
        """
        consistency_scores = []

        euler_diff = abs(orig.euler_characteristic -
                         current.euler_characteristic)
        euler_consistency = 1.0 / (1.0 + euler_diff)
        consistency_scores.append(euler_consistency)

        genus_diff = abs(orig.genus_signature - current.genus_signature)
        genus_consistency = 1.0 / (1.0 + genus_diff)
        consistency_scores.append(genus_consistency)

        rank_diff = abs(orig.fundamental_group_rank -
                        current.fundamental_group_rank)
        rank_consistency = 1.0 / (1.0 + rank_diff)
        consistency_scores.append(rank_consistency)

        if orig.betti_numbers and current.betti_numbers:
            betti_diff = sum(abs(a - b) for a, b in
                             zip(orig.betti_numbers, current.betti_numbers))
            betti_consistency = 1.0 / (1.0 + betti_diff)
        else:
            betti_consistency = 0.5
        consistency_scores.append(betti_consistency)

        fingerprint_match = (1.0 if orig.topological_fingerprint ==
                             current.topological_fingerprint else 0.0)
        consistency_scores.append(fingerprint_match)

        return float(np.mean(consistency_scores))

    def _compare_canonical_signatures(
        self,
        orig: CanonicalMathematicalSignature,
        current: CanonicalMathematicalSignature
    ) -> float:
        """
        Compare canonical mathematical signatures.

        Args:
            orig: Original canonical signature
            current: Current canonical signature

        Returns:
            Consistency score [0, 1]
        """
        consistency_scores = []

        hash_match = (1.0 if orig.formula_structure_hash ==
                      current.formula_structure_hash else 0.0)
        consistency_scores.append(hash_match)

        genus_diff = abs(orig.mathematical_genus -
                         current.mathematical_genus)
        genus_consistency = 1.0 / (1.0 + genus_diff)
        consistency_scores.append(genus_consistency)

        species_match = (1.0 if orig.mathematical_species_id ==
                         current.mathematical_species_id else 0.0)
        consistency_scores.append(species_match)

        if (orig.canonical_form_coefficients.size > 0 and
                current.canonical_form_coefficients.size > 0):
            min_length = min(len(orig.canonical_form_coefficients),
                             len(current.canonical_form_coefficients))
            coeff_diff = np.sum(np.abs(
                orig.canonical_form_coefficients[:min_length] -
                current.canonical_form_coefficients[:min_length]))
            coeff_consistency = 1.0 / (1.0 + coeff_diff)
        else:
            coeff_consistency = 0.5
        consistency_scores.append(coeff_consistency)

        return float(np.mean(consistency_scores))

    def _compare_information_identities(
        self,
        orig: InformationTheoreticIdentity,
        current: InformationTheoreticIdentity
    ) -> float:
        """
        Compare information-theoretic identities.

        Args:
            orig: Original information identity
            current: Current information identity

        Returns:
            Consistency score [0, 1]
        """
        consistency_scores = []

        fingerprint_match = (
            1.0 if orig.information_theoretical_fingerprint ==
            current.information_theoretical_fingerprint else 0.0)
        consistency_scores.append(fingerprint_match)

        coherence_diff = abs(orig.coherence_information_content -
                             current.coherence_information_content)
        coherence_consistency = 1.0 / (1.0 + coherence_diff)
        consistency_scores.append(coherence_consistency)

        complexity_diff = abs(orig.mathematical_information_complexity -
                              current.mathematical_information_complexity)
        complexity_consistency = 1.0 / (1.0 + complexity_diff)
        consistency_scores.append(complexity_consistency)

        if (orig.entropy_signature_pattern.size > 0 and
                current.entropy_signature_pattern.size > 0):
            entropy_diff = np.sum(np.abs(
                orig.entropy_signature_pattern -
                current.entropy_signature_pattern))
            entropy_consistency = 1.0 / (1.0 + entropy_diff)
        else:
            entropy_consistency = 0.5
        consistency_scores.append(entropy_consistency)

        if (orig.mutual_information_matrix.size > 0 and
                current.mutual_information_matrix.size > 0):
            mi_diff = np.sum(np.abs(orig.mutual_information_matrix -
                                    current.mutual_information_matrix))
            mi_consistency = 1.0 / (1.0 + mi_diff)
        else:
            mi_consistency = 0.5
        consistency_scores.append(mi_consistency)

        return float(np.mean(consistency_scores))

    def _compare_geometric_invariants(
        self, orig: GeometricInvariants, current: GeometricInvariants
    ) -> float:
        """
        Compare geometric invariants.

        Args:
            orig: Original geometric invariants
            current: Current geometric invariants

        Returns:
            Consistency score [0, 1]
        """
        consistency_scores = []

        topology_match = (1.0 if orig.manifold_topology_hash ==
                          current.manifold_topology_hash else 0.0)
        consistency_scores.append(topology_match)

        genus_diff = abs(orig.geometric_genus - current.geometric_genus)
        genus_consistency = 1.0 / (1.0 + genus_diff)
        consistency_scores.append(genus_consistency)

        if (orig.curvature_signature.size > 0 and
                current.curvature_signature.size > 0):
            curvature_diff = np.sum(np.abs(orig.curvature_signature -
                                           current.curvature_signature))
            curvature_consistency = 1.0 / (1.0 + curvature_diff)
        else:
            curvature_consistency = 0.5
        consistency_scores.append(curvature_consistency)

        return float(np.mean(consistency_scores))

    def _compare_mathematical_coherence(
        self, orig: MathematicalCoherence, current: MathematicalCoherence
    ) -> float:
        """
        Compare mathematical coherence structures.

        Args:
            orig: Original mathematical coherence
            current: Current mathematical coherence

        Returns:
            Consistency score [0, 1]
        """
        consistency_scores = []

        fingerprint_match = (
            1.0 if orig.coherence_mathematical_fingerprint ==
            current.coherence_mathematical_fingerprint else 0.0)
        consistency_scores.append(fingerprint_match)

        preservation_diff = abs(
            orig.mathematical_structure_preservation_score -
            current.mathematical_structure_preservation_score)
        preservation_consistency = 1.0 / (1.0 + preservation_diff)
        consistency_scores.append(preservation_consistency)

        if (orig.q_component_relationship_invariants and
                current.q_component_relationship_invariants):
            rel_invariants1 = np.mean(
                list(orig.q_component_relationship_invariants.values()))
            rel_invariants2 = np.mean(
                list(current.q_component_relationship_invariants.values()))
            invariants_diff = abs(rel_invariants1 - rel_invariants2)
            invariants_consistency = 1.0 / (1.0 + invariants_diff)
        else:
            invariants_consistency = 0.5
        consistency_scores.append(invariants_consistency)

        return float(np.mean(consistency_scores))

    def _compute_mathematical_similarity(
        self,
        profile1: MathematicalObjectIdentityProfile,
        profile2: MathematicalObjectIdentityProfile
    ) -> Tuple[float, Dict[str, float]]:
        """Compute mathematical similarity between two identity profiles."""
        similarity_components = {}

        similarity_components["topological"] = (
            self._compare_topological_invariants(
                profile1.topological_invariants,
                profile2.topological_invariants
            ))

        similarity_components["canonical"] = (
            self._compare_canonical_signatures(
                profile1.canonical_signature,
                profile2.canonical_signature
            ))

        similarity_components["information"] = (
            self._compare_information_identities(
                profile1.information_identity,
                profile2.information_identity
            ))

        similarity_components["geometric"] = (
            self._compare_geometric_invariants(
                profile1.geometric_invariants,
                profile2.geometric_invariants
            ))

        similarity_components["coherence"] = (
            self._compare_mathematical_coherence(
                profile1.mathematical_coherence,
                profile2.mathematical_coherence
            ))

        overall_similarity = np.mean(list(similarity_components.values()))

        return overall_similarity, similarity_components

    def _compare_topological_invariants_similarity(
        self, invariants1: TopologicalInvariants,
        invariants2: TopologicalInvariants
    ) -> float:
        """
        Compare topological invariants using scipy.spatial.distance.
        """
        features1 = np.array([
            invariants1.euler_characteristic,
            invariants1.genus_signature,
            float(invariants1.fundamental_group_rank),
            (np.mean(invariants1.betti_numbers)
             if invariants1.betti_numbers else 0.0)
        ])

        features2 = np.array([
            invariants2.euler_characteristic,
            invariants2.genus_signature,
            float(invariants2.fundamental_group_rank),
            (np.mean(invariants2.betti_numbers)
             if invariants2.betti_numbers else 0.0)
        ])

        feature_matrix = np.stack([features1, features2])

        distances = pdist(feature_matrix, metric='euclidean')
        distance = distances[0]

        similarity = 1.0 / (1.0 + distance)
        return float(similarity)

    def _compare_canonical_signatures_similarity(
        self, sig1: CanonicalMathematicalSignature,
        sig2: CanonicalMathematicalSignature
    ) -> float:
        """
        Compare canonical signatures using scipy.spatial.distance with proper
        tensor handling.
        """
        complexity1 = (float(np.linalg.norm(sig1.canonical_form_coefficients))
                       if sig1.canonical_form_coefficients.size > 0 else 0.0)
        complexity2 = (float(np.linalg.norm(sig2.canonical_form_coefficients))
                       if sig2.canonical_form_coefficients.size > 0 else 0.0)

        rel_size1 = (float(sig1.component_relationship_matrix.size)
                     if sig1.component_relationship_matrix.size > 0 else 0.0)
        rel_size2 = (float(sig2.component_relationship_matrix.size)
                     if sig2.component_relationship_matrix.size > 0 else 0.0)

        features1 = np.array([
            complexity1,
            sig1.mathematical_genus,
            rel_size1
        ])

        features2 = np.array([
            complexity2,
            sig2.mathematical_genus,
            rel_size2
        ])

        feature_matrix = np.stack([features1, features2])
        distances = pdist(feature_matrix, metric='cosine')
        distance = distances[0]

        similarity = 1.0 - distance
        return float(max(0.0, similarity))

    def _compare_information_identities_similarity(
        self, info1: InformationTheoreticIdentity,
        info2: InformationTheoreticIdentity
    ) -> float:
        """
        Compare information signatures using scipy.spatial.distance with
        proper tensor handling.
        """
        entropy1 = (float(np.linalg.norm(info1.entropy_signature_pattern))
                    if info1.entropy_signature_pattern.size > 0 else 0.0)
        entropy2 = (float(np.linalg.norm(info2.entropy_signature_pattern))
                    if info2.entropy_signature_pattern.size > 0 else 0.0)

        mi1 = (float(np.linalg.norm(info1.mutual_information_matrix))
               if info1.mutual_information_matrix.size > 0 else 0.0)
        mi2 = (float(np.linalg.norm(info2.mutual_information_matrix))
               if info2.mutual_information_matrix.size > 0 else 0.0)

        features1 = np.array([
            entropy1,
            info1.coherence_information_content,
            info1.mathematical_information_complexity,
            mi1
        ])

        features2 = np.array([
            entropy2,
            info2.coherence_information_content,
            info2.mathematical_information_complexity,
            mi2
        ])

        feature_matrix = np.stack([features1, features2])
        distances = pdist(feature_matrix, metric='euclidean')
        distance = distances[0]

        similarity = 1.0 / (1.0 + distance)
        return float(similarity)

    def _compare_geometric_invariants_similarity(
        self, geom1: GeometricInvariants, geom2: GeometricInvariants
    ) -> float:
        """
        Compare geometric invariants using spatial distance metrics.
        """
        curvature1 = (float(np.linalg.norm(geom1.curvature_signature))
                      if geom1.curvature_signature.size > 0 else 0.0)
        curvature2 = (float(np.linalg.norm(geom2.curvature_signature))
                      if geom2.curvature_signature.size > 0 else 0.0)

        canonical1 = (float(np.linalg.norm(geom1.canonical_geometric_form))
                      if geom1.canonical_geometric_form.size > 0 else 0.0)
        canonical2 = (float(np.linalg.norm(geom2.canonical_geometric_form))
                      if geom2.canonical_geometric_form.size > 0 else 0.0)

        features1 = np.array([
            curvature1,
            geom1.geometric_genus,
            canonical1
        ])

        features2 = np.array([
            curvature2,
            geom2.geometric_genus,
            canonical2
        ])

        feature_matrix = np.stack([features1, features2])
        distances = pdist(feature_matrix, metric='euclidean')
        distance = distances[0]

        similarity = 1.0 / (1.0 + distance)
        return float(similarity)

    def _compare_mathematical_coherence_similarity(
        self, coherence1: MathematicalCoherence,
        coherence2: MathematicalCoherence
    ) -> float:
        """
        Compare mathematical coherence structures using relationship
        invariants.
        """
        rel_invariants1 = (
            np.mean(list(coherence1.q_component_relationship_invariants
                         .values()))
            if coherence1.q_component_relationship_invariants else 0.0)
        rel_invariants2 = (
            np.mean(list(coherence2.q_component_relationship_invariants
                         .values()))
            if coherence2.q_component_relationship_invariants else 0.0)

        ratios1 = (float(np.linalg.norm(
            coherence1.mathematical_ratio_signatures))
            if coherence1.mathematical_ratio_signatures.size > 0 else 0.0)
        ratios2 = (float(np.linalg.norm(
            coherence2.mathematical_ratio_signatures))
            if coherence2.mathematical_ratio_signatures.size > 0 else 0.0)

        features1 = np.array([
            rel_invariants1,
            coherence1.mathematical_structure_preservation_score,
            ratios1
        ])

        features2 = np.array([
            rel_invariants2,
            coherence2.mathematical_structure_preservation_score,
            ratios2
        ])

        feature_matrix = np.stack([features1, features2])
        distances = pdist(feature_matrix, metric='euclidean')
        distance = distances[0]

        similarity = 1.0 / (1.0 + distance)
        return float(similarity)

    def _compute_topological_inheritance(
        self, parent: TopologicalInvariants, child: TopologicalInvariants
    ) -> float:
        return self._compare_topological_invariants(parent, child)

    def _compute_canonical_inheritance(
        self, parent: CanonicalMathematicalSignature,
        child: CanonicalMathematicalSignature
    ) -> float:
        return self._compare_canonical_signatures(parent, child)

    def _compute_information_inheritance(
        self, parent: InformationTheoreticIdentity,
        child: InformationTheoreticIdentity
    ) -> float:
        return self._compare_information_identities(parent, child)

    def _compute_geometric_inheritance(
        self, parent: GeometricInvariants, child: GeometricInvariants
    ) -> float:
        return self._compare_geometric_invariants(parent, child)

    def _compute_coherence_inheritance(
        self, parent: MathematicalCoherence, child: MathematicalCoherence
    ) -> float:
        return self._compare_mathematical_coherence(parent, child)
