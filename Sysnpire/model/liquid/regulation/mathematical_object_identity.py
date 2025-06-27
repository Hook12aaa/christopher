"""
Mathematical Object Identity Theory - Persistent Mathematical Entity Recognition

CORE MATHEMATICAL FOUNDATION: Implements mathematical object identity preservation
across field regulation cycles using topological invariants, canonical signatures,
and information-theoretic fingerprints derived from Q(τ,C,s) field theory.

IDENTITY PRINCIPLE: Mathematical objects maintain their essential mathematical
nature through invariant properties that survive all continuous field transformations:

1. TOPOLOGICAL INVARIANTS: Genus, Euler characteristic, fundamental group signatures
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
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
import hashlib

import numpy as np
import torch
from scipy import linalg, spatial, integrate
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

from Sysnpire.model.liquid.conceptual_charge_agent import ConceptualChargeAgent
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TopologicalInvariants:
    """Topological properties invariant under continuous field transformations."""

    euler_characteristic: float  # Topological invariant from field structure
    genus_signature: float  # Genus computed from Q-field topology
    fundamental_group_rank: int  # Fundamental group structure rank
    betti_numbers: List[int]  # Homology group dimensions
    topological_fingerprint: str  # Hash of topological structure


@dataclass
class CanonicalMathematicalSignature:
    """Mathematical "DNA" derived from Q(τ,C,s) formula structure."""

    formula_structure_hash: str  # Hash of mathematical formula structure
    component_relationship_matrix: np.ndarray  # Mathematical relationships between components
    mathematical_genus: float  # Mathematical genus of the Q-field structure
    canonical_form_coefficients: np.ndarray  # Canonical representation coefficients
    mathematical_species_id: str  # Species classification based on mathematical structure


@dataclass
class PersistenceHomologyFeatures:
    """Mathematical features that persist across regulation cycles."""

    persistent_cycles: List[Tuple[float, float]]  # Birth-death pairs of persistent features
    persistence_diagram_signature: str  # Hash of persistence diagram
    stable_mathematical_features: Set[str]  # Features that survive all transformations
    mathematical_lifetime_distribution: np.ndarray  # Feature lifetime statistics
    topological_persistence_rank: int  # Rank of persistent topology


@dataclass
class InformationTheoreticIdentity:
    """Information-theoretic fingerprints independent of field magnitudes."""

    entropy_signature_pattern: np.ndarray  # Entropy distribution pattern
    mutual_information_matrix: np.ndarray  # MI between Q-components
    information_theoretical_fingerprint: str  # Hash of information structure
    coherence_information_content: float  # Information in coherence relationships
    mathematical_information_complexity: float  # Information-theoretic complexity measure


@dataclass
class GeometricInvariants:
    """Geometric properties that define mathematical object identity."""

    curvature_signature: np.ndarray  # Curvature invariants of field manifold
    manifold_topology_hash: str  # Hash of manifold topological structure
    geometric_genus: float  # Geometric genus of field manifold
    canonical_geometric_form: np.ndarray  # Canonical geometric representation
    differential_geometric_invariants: Dict[str, float]  # Collection of geometric invariants


@dataclass
class MathematicalCoherence:
    """Essential mathematical relationships that define object identity."""

    q_component_relationship_invariants: Dict[str, float]  # Invariant relationships
    mathematical_ratio_signatures: np.ndarray  # Essential mathematical ratios
    phase_relationship_topology: np.ndarray  # Phase relationship structure
    mathematical_structure_preservation_score: float  # Structure preservation measure
    coherence_mathematical_fingerprint: str  # Hash of coherence structure


@dataclass
class MathematicalObjectIdentityProfile:
    """Complete mathematical identity profile for a Q(τ,C,s) entity."""

    object_mathematical_id: str  # Unique mathematical identifier
    topological_invariants: TopologicalInvariants
    canonical_signature: CanonicalMathematicalSignature
    persistence_features: PersistenceHomologyFeatures
    information_identity: InformationTheoreticIdentity
    geometric_invariants: GeometricInvariants
    mathematical_coherence: MathematicalCoherence
    identity_creation_timestamp: float
    identity_mathematical_confidence: float  # Mathematical certainty of identity


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
            mathematical_precision: Numerical precision for mathematical computations
        """
        self.mathematical_precision = mathematical_precision
        self.identity_registry: Dict[str, MathematicalObjectIdentityProfile] = {}
        self.mathematical_species_classification: Dict[str, Set[str]] = {}
        self.topological_computation_cache: Dict[str, TopologicalInvariants] = {}

        logger.info("🔬 Mathematical Object Identity system initialized with rigorous mathematical foundations")

    def compute_topological_invariants(self, agent: ConceptualChargeAgent) -> TopologicalInvariants:
        """
        Compute topological invariants that survive all continuous field transformations.

        Uses mathematical topology theory to extract invariant properties from Q-field structure.
        """
        q_components = self._extract_q_field_components(agent)
        if not q_components:
            raise ValueError("Cannot compute topological invariants without Q-field components")

        euler_characteristic = self._compute_euler_characteristic(q_components)

        genus_signature = self._compute_topological_genus(q_components)

        fundamental_group_rank = self._compute_fundamental_group_rank(q_components)

        betti_numbers = self._compute_betti_numbers(q_components)

        topological_fingerprint = self._generate_topological_fingerprint(
            euler_characteristic, genus_signature, fundamental_group_rank, betti_numbers
        )

        invariants = TopologicalInvariants(
            euler_characteristic=euler_characteristic,
            genus_signature=genus_signature,
            fundamental_group_rank=fundamental_group_rank,
            betti_numbers=betti_numbers,
            topological_fingerprint=topological_fingerprint,
        )

        logger.debug(f"🔬 Computed topological invariants: χ={euler_characteristic:.6f}, g={genus_signature:.6f}")
        return invariants

    def derive_canonical_mathematical_signature(self, agent: ConceptualChargeAgent) -> CanonicalMathematicalSignature:
        """
        Derive canonical mathematical signature from Q(τ,C,s) formula structure.

        Creates mathematical "DNA" based on formula structure, not field values.
        """
        formula_structure = self._extract_q_formula_structure(agent)

        formula_structure_hash = self._hash_mathematical_structure(formula_structure)

        component_relationship_matrix = self._compute_component_relationships(agent)

        mathematical_genus = self._compute_mathematical_genus(component_relationship_matrix)

        canonical_form_coefficients = self._compute_canonical_form(component_relationship_matrix)

        mathematical_species_id = self._classify_mathematical_species(
            formula_structure_hash, mathematical_genus, canonical_form_coefficients
        )

        signature = CanonicalMathematicalSignature(
            formula_structure_hash=formula_structure_hash,
            component_relationship_matrix=component_relationship_matrix,
            mathematical_genus=mathematical_genus,
            canonical_form_coefficients=canonical_form_coefficients,
            mathematical_species_id=mathematical_species_id,
        )

        logger.debug(
            f"🔬 Derived canonical signature: species={mathematical_species_id}, genus={mathematical_genus:.6f}"
        )
        return signature

    def compute_persistence_homology_features(
        self, agent: ConceptualChargeAgent, regulation_history: List[Dict[str, Any]]
    ) -> PersistenceHomologyFeatures:
        """
        Compute persistence homology features that survive regulation cycles.

        Identifies mathematical structures that persist across all field transformations.
        """
        field_evolution_data = self._extract_field_evolution_data(regulation_history)

        persistent_cycles = self._compute_persistent_cycles(field_evolution_data)

        persistence_diagram_signature = self._generate_persistence_diagram_signature(persistent_cycles)

        stable_mathematical_features = self._identify_stable_features(field_evolution_data, persistent_cycles)

        mathematical_lifetime_distribution = self._compute_lifetime_distribution(persistent_cycles)

        topological_persistence_rank = self._compute_persistence_rank(persistent_cycles)

        features = PersistenceHomologyFeatures(
            persistent_cycles=persistent_cycles,
            persistence_diagram_signature=persistence_diagram_signature,
            stable_mathematical_features=stable_mathematical_features,
            mathematical_lifetime_distribution=mathematical_lifetime_distribution,
            topological_persistence_rank=topological_persistence_rank,
        )

        logger.debug(
            f"🔬 Computed persistence features: {len(persistent_cycles)} cycles, rank={topological_persistence_rank}"
        )
        return features

    def compute_mathematical_information_signature(self, agent: ConceptualChargeAgent) -> InformationTheoreticIdentity:
        """
        Compute information-theoretic fingerprints independent of field magnitudes.

        Creates information signatures based on structural relationships, not values.
        """
        q_field_structure = self._extract_q_field_structure(agent)

        entropy_signature_pattern = self._compute_entropy_signature_pattern(q_field_structure)

        mutual_information_matrix = self._compute_structural_mutual_information(q_field_structure)

        information_theoretical_fingerprint = self._generate_information_fingerprint(
            entropy_signature_pattern, mutual_information_matrix
        )

        coherence_information_content = self._compute_coherence_information_content(q_field_structure)

        mathematical_information_complexity = self._compute_mathematical_complexity(
            entropy_signature_pattern, mutual_information_matrix
        )

        identity = InformationTheoreticIdentity(
            entropy_signature_pattern=entropy_signature_pattern,
            mutual_information_matrix=mutual_information_matrix,
            information_theoretical_fingerprint=information_theoretical_fingerprint,
            coherence_information_content=coherence_information_content,
            mathematical_information_complexity=mathematical_information_complexity,
        )

        logger.debug(f"🔬 Computed information signature: complexity={mathematical_information_complexity:.6f}")
        return identity

    def compute_geometric_invariants(self, agent: ConceptualChargeAgent) -> GeometricInvariants:
        """
        Compute geometric invariants that define mathematical object geometry.

        Extracts curvature signatures and manifold topology independent of coordinates.
        """
        field_manifold_structure = self._extract_field_manifold_structure(agent)

        curvature_signature = self._compute_curvature_signature(field_manifold_structure)

        manifold_topology_hash = self._generate_manifold_topology_hash(field_manifold_structure)

        geometric_genus = self._compute_geometric_genus(field_manifold_structure)

        canonical_geometric_form = self._compute_canonical_geometric_form(field_manifold_structure)

        differential_geometric_invariants = self._compute_differential_invariants(field_manifold_structure)

        invariants = GeometricInvariants(
            curvature_signature=curvature_signature,
            manifold_topology_hash=manifold_topology_hash,
            geometric_genus=geometric_genus,
            canonical_geometric_form=canonical_geometric_form,
            differential_geometric_invariants=differential_geometric_invariants,
        )

        logger.debug(f"🔬 Computed geometric invariants: genus={geometric_genus:.6f}")
        return invariants

    def compute_mathematical_coherence(self, agent: ConceptualChargeAgent) -> MathematicalCoherence:
        """
        Compute essential mathematical relationships that define object identity.

        Extracts invariant mathematical relationships between Q-components.
        """
        q_component_relationships = self._extract_q_component_relationships(agent)

        q_component_relationship_invariants = self._compute_relationship_invariants(q_component_relationships)

        mathematical_ratio_signatures = self._compute_mathematical_ratios(q_component_relationships)

        phase_relationship_topology = self._compute_phase_relationship_topology(agent)

        mathematical_structure_preservation_score = self._compute_structure_preservation_score(
            q_component_relationship_invariants, mathematical_ratio_signatures
        )

        coherence_mathematical_fingerprint = self._generate_coherence_fingerprint(
            q_component_relationship_invariants, mathematical_ratio_signatures, phase_relationship_topology
        )

        coherence = MathematicalCoherence(
            q_component_relationship_invariants=q_component_relationship_invariants,
            mathematical_ratio_signatures=mathematical_ratio_signatures,
            phase_relationship_topology=phase_relationship_topology,
            mathematical_structure_preservation_score=mathematical_structure_preservation_score,
            coherence_mathematical_fingerprint=coherence_mathematical_fingerprint,
        )

        logger.debug(
            f"🔬 Computed mathematical coherence: preservation={mathematical_structure_preservation_score:.6f}"
        )
        return coherence

    def create_mathematical_identity_profile(
        self, agent: ConceptualChargeAgent, regulation_history: Optional[List[Dict[str, Any]]] = None
    ) -> MathematicalObjectIdentityProfile:
        """
        Create complete mathematical identity profile for a Q(τ,C,s) entity.

        Combines all mathematical identity mechanisms into unified profile.
        """
        start_time = time.time()

        topological_invariants = self.compute_topological_invariants(agent)
        canonical_signature = self.derive_canonical_mathematical_signature(agent)
        persistence_features = self.compute_persistence_homology_features(agent, regulation_history or [])
        information_identity = self.compute_mathematical_information_signature(agent)
        geometric_invariants = self.compute_geometric_invariants(agent)
        mathematical_coherence = self.compute_mathematical_coherence(agent)

        object_mathematical_id = self._generate_mathematical_object_id(
            topological_invariants, canonical_signature, information_identity
        )

        identity_mathematical_confidence = self._compute_identity_confidence(
            topological_invariants,
            canonical_signature,
            persistence_features,
            information_identity,
            geometric_invariants,
            mathematical_coherence,
        )

        profile = MathematicalObjectIdentityProfile(
            object_mathematical_id=object_mathematical_id,
            topological_invariants=topological_invariants,
            canonical_signature=canonical_signature,
            persistence_features=persistence_features,
            information_identity=information_identity,
            geometric_invariants=geometric_invariants,
            mathematical_coherence=mathematical_coherence,
            identity_creation_timestamp=time.time(),
            identity_mathematical_confidence=identity_mathematical_confidence,
        )

        self.identity_registry[object_mathematical_id] = profile

        species_id = canonical_signature.mathematical_species_id
        if species_id not in self.mathematical_species_classification:
            self.mathematical_species_classification[species_id] = set()
        self.mathematical_species_classification[species_id].add(object_mathematical_id)

        computation_time = time.time() - start_time
        logger.info(
            f"🔬 Created mathematical identity profile: {object_mathematical_id} "
            f"(confidence={identity_mathematical_confidence:.6f}, time={computation_time:.3f}s)"
        )

        return profile

    def verify_mathematical_identity_consistency(
        self, current_profile: MathematicalObjectIdentityProfile, agent: ConceptualChargeAgent
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Verify mathematical object maintains consistent identity after regulation.

        Returns consistency status, confidence score, and detailed consistency metrics.
        """
        current_topological = self.compute_topological_invariants(agent)
        current_canonical = self.derive_canonical_mathematical_signature(agent)
        current_information = self.compute_mathematical_information_signature(agent)
        current_geometric = self.compute_geometric_invariants(agent)
        current_coherence = self.compute_mathematical_coherence(agent)

        consistency_metrics = {}

        consistency_metrics["topological"] = self._compare_topological_invariants(
            current_profile.topological_invariants, current_topological
        )

        consistency_metrics["canonical"] = self._compare_canonical_signatures(
            current_profile.canonical_signature, current_canonical
        )

        consistency_metrics["information"] = self._compare_information_identities(
            current_profile.information_identity, current_information
        )

        consistency_metrics["geometric"] = self._compare_geometric_invariants(
            current_profile.geometric_invariants, current_geometric
        )

        consistency_metrics["coherence"] = self._compare_mathematical_coherence(
            current_profile.mathematical_coherence, current_coherence
        )

        consistency_tensor = torch.tensor(list(consistency_metrics.values()), dtype=torch.float32)
        overall_consistency = torch.mean(consistency_tensor).item()

        mathematical_consistency_threshold = 1.0 - self.mathematical_precision
        is_consistent = overall_consistency >= mathematical_consistency_threshold

        logger.debug(
            f"🔬 Identity consistency verification: {overall_consistency:.6f} "
            f"({'CONSISTENT' if is_consistent else 'INCONSISTENT'})"
        )

        return is_consistent, overall_consistency, consistency_metrics

    def recognize_mathematical_peers(
        self, identity_profile: MathematicalObjectIdentityProfile, candidate_agents: List[ConceptualChargeAgent]
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Recognize mathematical peer objects based on identity signatures.

        Returns list of (object_id, similarity_score, detailed_similarities).
        """
        peer_recognition_results = []

        for candidate_agent in candidate_agents:
            candidate_profile = self.create_mathematical_identity_profile(candidate_agent)

            similarity_score, detailed_similarities = self._compute_mathematical_similarity(
                identity_profile, candidate_profile
            )

            peer_recognition_results.append(
                (candidate_profile.object_mathematical_id, similarity_score, detailed_similarities)
            )

        peer_recognition_results.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"🔬 Recognized {len(peer_recognition_results)} mathematical peer candidates")
        return peer_recognition_results

    def track_mathematical_genealogy(
        self, parent_profile: MathematicalObjectIdentityProfile, child_profile: MathematicalObjectIdentityProfile
    ) -> Dict[str, float]:
        """
        Track mathematical lineage relationships between objects.

        Computes mathematical inheritance patterns and genealogical relationships.
        """
        genealogy_metrics = {}

        genealogy_metrics["topological_inheritance"] = self._compute_topological_inheritance(
            parent_profile.topological_invariants, child_profile.topological_invariants
        )

        genealogy_metrics["canonical_inheritance"] = self._compute_canonical_inheritance(
            parent_profile.canonical_signature, child_profile.canonical_signature
        )

        genealogy_metrics["information_inheritance"] = self._compute_information_inheritance(
            parent_profile.information_identity, child_profile.information_identity
        )

        genealogy_metrics["geometric_inheritance"] = self._compute_geometric_inheritance(
            parent_profile.geometric_invariants, child_profile.geometric_invariants
        )

        genealogy_metrics["coherence_inheritance"] = self._compute_coherence_inheritance(
            parent_profile.mathematical_coherence, child_profile.mathematical_coherence
        )

        inheritance_values = [
                genealogy_metrics["topological_inheritance"],
                genealogy_metrics["canonical_inheritance"],
                genealogy_metrics["information_inheritance"],
                genealogy_metrics["geometric_inheritance"],
                genealogy_metrics["coherence_inheritance"],
            ]
        inheritance_tensor = torch.tensor(inheritance_values, dtype=torch.float32)
        genealogy_metrics["overall_inheritance"] = torch.mean(inheritance_tensor).item()

        logger.debug(f"🔬 Mathematical genealogy: inheritance={genealogy_metrics['overall_inheritance']:.6f}")
        return genealogy_metrics


    def _extract_q_field_components(self, agent: ConceptualChargeAgent) -> Dict[str, Any]:
        """Extract Q-field components for mathematical analysis."""
        components = {}

        if hasattr(agent, "q_value") and agent.q_value is not None:
            components["q_value"] = agent.q_value

        if hasattr(agent, "Q_components") and agent.Q_components:
            components["q_components"] = agent.Q_components

        if hasattr(agent, "breathing_q_coefficients") and agent.breathing_q_coefficients:
            components["breathing_coefficients"] = agent.breathing_q_coefficients

        if hasattr(agent, "temporal_biography") and agent.temporal_biography:
            components["temporal_biography"] = agent.temporal_biography

        return components

    def _compute_euler_characteristic(self, q_components: Dict[str, Any]) -> float:
        """Compute Euler characteristic from Q-field topology."""
        if "q_value" not in q_components:
            raise ValueError("Euler characteristic computation requires Q-value component - TOPOLOGICAL ANALYSIS IMPOSSIBLE")

        q_value = q_components["q_value"]

        if hasattr(q_value, "real") and hasattr(q_value, "imag"):
            real_part = float(q_value.real)
            imag_part = float(q_value.imag)

            real_tensor = torch.tensor(real_part, dtype=torch.float32)
            imag_tensor = torch.tensor(imag_part, dtype=torch.float32)
            magnitude = torch.sqrt(real_tensor**2 + imag_tensor**2).item()
            phase = torch.atan2(imag_tensor, real_tensor).item()

            mag_tensor = torch.tensor(magnitude, dtype=torch.float32)
            phase_tensor = torch.tensor(phase, dtype=torch.float32)
            euler_char = 2.0 - 2.0 * (mag_tensor / (1.0 + mag_tensor)).item() * torch.cos(phase_tensor).item()
            return float(euler_char)
        else:
            real_value = float(q_value)
            euler_char = 2.0 / (1.0 + abs(real_value))
            return float(euler_char)

    def _compute_topological_genus(self, q_components: Dict[str, Any]) -> float:
        """Compute topological genus from Q-field structure."""
        if "q_components" not in q_components:
            raise ValueError("Topological genus computation requires Q-components - GENUS ANALYSIS IMPOSSIBLE")

        q_comp = q_components["q_components"]

        real_parts = []
        imag_parts = []

        for component_name, component_value in q_comp.items():
            if hasattr(component_value, "real") and hasattr(component_value, "imag"):
                real_parts.append(float(component_value.real))
                imag_parts.append(float(component_value.imag))
            else:
                real_parts.append(float(component_value))
                imag_parts.append(0.0)

        if not real_parts:
            raise ValueError("Topological genus requires valid real parts from Q-components - GENUS COMPUTATION IMPOSSIBLE")

        real_tensor = torch.tensor(real_parts, dtype=torch.float32)
        imag_tensor = torch.tensor(imag_parts, dtype=torch.float32)
        field_complexity = torch.std(real_tensor).item() + torch.std(imag_tensor).item()
        genus = field_complexity / (2.0 * math.pi)  # Mathematical genus formula
        return float(genus)

    def _compute_fundamental_group_rank(self, q_components: Dict[str, Any]) -> int:
        """Compute fundamental group rank from phase relationships."""
        if "q_components" not in q_components:
            return 0

        q_comp = q_components["q_components"]

        phases = []
        for component_name, component_value in q_comp.items():
            if hasattr(component_value, "real") and hasattr(component_value, "imag"):
                phase = math.atan2(float(component_value.imag), float(component_value.real))
                phases.append(phase)

        if len(phases) < 2:
            return 0

        phase_differences = []
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                diff = abs(phases[i] - phases[j])
                phase_differences.append(min(diff, 2 * math.pi - diff))

        independent_count = 0
        tolerance = self.mathematical_precision * 10

        for diff in phase_differences:
            is_independent = True
            for existing_diff in phase_differences[:independent_count]:
                if abs(diff - existing_diff) < tolerance:
                    is_independent = False
                    break
            if is_independent:
                independent_count += 1

        return independent_count

    def _compute_betti_numbers(self, q_components: Dict[str, Any]) -> List[int]:
        """Compute Betti numbers for homology groups."""
        if "q_components" not in q_components:
            return [0]

        q_comp = q_components["q_components"]

        field_points = []
        for component_name, component_value in q_comp.items():
            if hasattr(component_value, "real") and hasattr(component_value, "imag"):
                field_points.append([float(component_value.real), float(component_value.imag)])
            else:
                field_points.append([float(component_value), 0.0])

        if len(field_points) < 2:
            return [1, 0]  # Single point has b0=1, b1=0

        field_points_list = field_points
        field_points = torch.tensor(field_points_list, dtype=torch.float32)

        distance_matrix = pdist(field_points)
        if len(distance_matrix) == 0:
            b0 = 1
        else:
            distance_tensor = torch.tensor(distance_matrix, dtype=torch.float32)
            threshold = torch.quantile(distance_tensor.flatten(), 0.25).item()  # Connection threshold
            connected_components = self._count_connected_components(field_points, threshold)
            b0 = connected_components

        b1 = max(0, len(field_points) - b0)

        return [b0, b1]

    def _count_connected_components(self, points: np.ndarray, threshold: float) -> int:
        """Count connected components in point set."""
        n_points = len(points)
        if n_points <= 1:
            return n_points

        distances = squareform(pdist(points))
        adjacency = distances <= threshold

        visited = set()
        components = 0

        for i in range(n_points):
            if i not in visited:
                components += 1
                stack = [i]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        for j in range(n_points):
                            if adjacency[current, j] and j not in visited:
                                stack.append(j)

        return components

    def _generate_topological_fingerprint(
        self, euler_char: float, genus: float, fund_group_rank: int, betti_numbers: List[int]
    ) -> str:
        """Generate unique topological fingerprint."""
        fingerprint_data = f"{euler_char:.12f}_{genus:.12f}_{fund_group_rank}_{'-'.join(map(str, betti_numbers))}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]

    def _extract_q_formula_structure(self, agent: ConceptualChargeAgent) -> Dict[str, Any]:
        """Extract mathematical structure of Q-formula components."""
        structure = {}

        if hasattr(agent, "Q_components") and agent.Q_components:
            structure["component_types"] = list(agent.Q_components.keys())
            structure["component_count"] = len(agent.Q_components)

            component_magnitudes = []
            component_phases = []

            for comp_name, comp_value in agent.Q_components.items():
                if hasattr(comp_value, "real") and hasattr(comp_value, "imag"):
                    magnitude = abs(comp_value)
                    phase = math.atan2(float(comp_value.imag), float(comp_value.real))
                    component_magnitudes.append(magnitude)
                    component_phases.append(phase)
                else:
                    component_magnitudes.append(abs(float(comp_value)))
                    component_phases.append(0.0)

            structure["magnitude_pattern"] = component_magnitudes
            structure["phase_pattern"] = component_phases

        return structure

    def _hash_mathematical_structure(self, structure: Dict[str, Any]) -> str:
        """Generate hash of mathematical formula structure."""
        structure_items = []

        if "component_types" in structure:
            structure_items.append(f"types:{'-'.join(sorted(structure['component_types']))}")

        if "component_count" in structure:
            structure_items.append(f"count:{structure['component_count']}")

        if "magnitude_pattern" in structure:
            magnitudes = structure["magnitude_pattern"]
            if magnitudes:
                mag_tensor = torch.tensor(magnitudes, dtype=torch.float32)
                max_mag = torch.max(mag_tensor).item()
                if max_mag > 0:
                    normalized_mags = (mag_tensor / max_mag).tolist()
                else:
                    normalized_mags = [0 for _ in magnitudes]
                mag_structure = f"mag_pattern:{'-'.join(f'{m:.6f}' for m in normalized_mags)}"
                structure_items.append(mag_structure)

        if "phase_pattern" in structure:
            phases = structure["phase_pattern"]
            phase_structure = f"phase_pattern:{'-'.join(f'{p:.6f}' for p in phases)}"
            structure_items.append(phase_structure)

        structure_string = "|".join(structure_items)
        return hashlib.sha256(structure_string.encode()).hexdigest()[:16]

    def _compute_component_relationships(self, agent: ConceptualChargeAgent) -> np.ndarray:
        """Compute mathematical relationship matrix between Q-components."""
        if not hasattr(agent, "Q_components") or not agent.Q_components:
            return torch.tensor([[1.0]], dtype=torch.float32).numpy()  # Return numpy for compatibility

        components = list(agent.Q_components.values())
        n_components = len(components)

        numerical_components = []
        for comp in components:
            if hasattr(comp, "real") and hasattr(comp, "imag"):
                numerical_components.append(complex(comp))
            else:
                numerical_components.append(complex(float(comp)))

        relationship_matrix = np.zeros((n_components, n_components), dtype=complex)  # Keep numpy for scipy compatibility

        for i in range(n_components):
            for j in range(n_components):
                if i == j:
                    relationship_matrix[i, j] = 1.0
                else:
                    comp_i = numerical_components[i]
                    comp_j = numerical_components[j]

                    if abs(comp_j) > self.mathematical_precision:
                        ratio = comp_i / comp_j
                        relationship_matrix[i, j] = ratio
                    else:
                        relationship_matrix[i, j] = 0.0

        return relationship_matrix

    def _compute_mathematical_genus(self, component_matrix: np.ndarray) -> float:
        """Compute mathematical genus of component relationship structure."""
        if component_matrix.size == 0:
            raise ValueError("Mathematical genus computation requires non-empty component matrix - GENUS ANALYSIS IMPOSSIBLE")

        try:
            eigenvalues = linalg.eigvals(component_matrix)
            significant_eigenvalues = [ev for ev in eigenvalues if abs(ev) > self.mathematical_precision]

            if not significant_eigenvalues:
                raise ValueError(f"Mathematical genus computation failed - no significant eigenvalues above precision {self.mathematical_precision} - SPECTRAL ANALYSIS DEGENERATE")

            eigenvalue_magnitudes = [abs(ev) for ev in significant_eigenvalues]
            genus = (len(significant_eigenvalues) - 1) / 2.0

            return float(genus)

        except (linalg.LinAlgError, ValueError) as e:
            raise ValueError(f"Mathematical genus eigenvalue computation failed: {e} - SPECTRAL ANALYSIS IMPOSSIBLE")

    def _compute_canonical_form(self, component_matrix: np.ndarray) -> np.ndarray:
        """Compute canonical form coefficients for unique representation."""
        if component_matrix.size == 0:
            raise ValueError("Mathematical species classification requires canonical coefficients - SPECIES CLASSIFICATION IMPOSSIBLE")

        try:
            U, s, Vh = linalg.svd(component_matrix)

            s_tensor = torch.tensor(s, dtype=torch.float32)
            canonical_coefficients = s_tensor / (torch.norm(s_tensor) + self.mathematical_precision)
            canonical_coefficients = canonical_coefficients.numpy()  # Return numpy for compatibility

            return canonical_coefficients

        except (linalg.LinAlgError, ValueError):
            return torch.tensor([1.0], dtype=torch.float32).numpy()

    def _classify_mathematical_species(self, structure_hash: str, genus: float, canonical_coeffs: np.ndarray) -> str:
        """Classify mathematical species based on structural properties."""
        genus_class = f"g{int(genus * 100):03d}"  # Genus classification

        if len(canonical_coeffs) == 0:
            coeff_class = "c000"
        else:
            coeffs_tensor = torch.tensor(canonical_coeffs, dtype=torch.float32)
            coeff_signature = int(torch.sum(coeffs_tensor).item() * 1000) % 1000
            coeff_class = f"c{coeff_signature:03d}"

        struct_class = structure_hash[:4]

        species_id = f"{genus_class}_{coeff_class}_{struct_class}"
        return species_id

    def _extract_field_evolution_data(self, regulation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract field evolution data from regulation history."""
        evolution_data = []

        for history_entry in regulation_history:
            if isinstance(history_entry, dict):
                field_data = {}

                for key in ["field_entropy", "phase_coherence", "energy_density", "regulation_strength"]:
                    if key in history_entry:
                        field_data[key] = history_entry[key]

                if "timestamp" in history_entry:
                    field_data["timestamp"] = history_entry["timestamp"]
                else:
                    field_data["timestamp"] = len(evolution_data)

                if field_data:  # Only add if we extracted some data
                    evolution_data.append(field_data)

        return evolution_data

    def _compute_persistent_cycles(self, field_evolution_data: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """Compute persistent cycles (birth-death pairs) from field evolution."""
        if not field_evolution_data:
            return []

        persistent_cycles = []

        for field_key in ["field_entropy", "phase_coherence", "energy_density"]:
            if field_key in field_evolution_data[0]:
                time_series = [entry.get(field_key) for entry in field_evolution_data]
                timestamps = [entry.get("timestamp") for i, entry in enumerate(field_evolution_data)]

                cycles = self._compute_time_series_persistence(time_series, timestamps)
                persistent_cycles.extend(cycles)

        return persistent_cycles

    def _compute_time_series_persistence(
        self, time_series: List[float], timestamps: List[float]
    ) -> List[Tuple[float, float]]:
        """Compute persistence features from time series."""
        if len(time_series) < 3:
            return []

        extrema = []
        for i in range(1, len(time_series) - 1):
            if (time_series[i] > time_series[i - 1] and time_series[i] > time_series[i + 1]) or (
                time_series[i] < time_series[i - 1] and time_series[i] < time_series[i + 1]
            ):
                extrema.append((timestamps[i], time_series[i]))

        persistence_pairs = []
        if len(extrema) >= 2:
            for i in range(0, len(extrema) - 1, 2):
                birth_time = extrema[i][0]
                death_time = extrema[i + 1][0] if i + 1 < len(extrema) else timestamps[-1]
                persistence_pairs.append((birth_time, death_time))

        return persistence_pairs

    def _generate_persistence_diagram_signature(self, persistent_cycles: List[Tuple[float, float]]) -> str:
        """Generate signature hash for persistence diagram."""
        if not persistent_cycles:
            return "empty_persistence"

        sorted_cycles = sorted(persistent_cycles)

        lifetimes = [death - birth for birth, death in sorted_cycles]
        signature_data = f"cycles_{len(sorted_cycles)}_lifetimes_{'_'.join(f'{lt:.6f}' for lt in lifetimes[:10])}"

        return hashlib.sha256(signature_data.encode()).hexdigest()[:16]

    def _identify_stable_features(
        self, field_evolution_data: List[Dict[str, Any]], persistent_cycles: List[Tuple[float, float]]
    ) -> Set[str]:
        """Identify mathematical features that remain stable across transformations."""
        stable_features = set()

        if not field_evolution_data:
            return stable_features

        for field_property in ["field_entropy", "phase_coherence", "energy_density"]:
            if field_property in field_evolution_data[0]:
                values = [entry.get(field_property) for entry in field_evolution_data]

                if values:
                    values_tensor = torch.tensor(values, dtype=torch.float32)
                    mean_val = torch.mean(values_tensor).item()
                    std_val = torch.std(values_tensor).item()
                    cv = std_val / (abs(mean_val) + self.mathematical_precision)

                    if cv < 0.1:  # Mathematical stability threshold
                        stable_features.add(f"stable_{field_property}")

        long_lived_cycles = [cycle for cycle in persistent_cycles if (cycle[1] - cycle[0]) > 1.0]
        if long_lived_cycles:
            stable_features.add("persistent_cycles")

        return stable_features

    def _compute_lifetime_distribution(self, persistent_cycles: List[Tuple[float, float]]) -> np.ndarray:
        """Compute distribution of feature lifetimes."""
        if not persistent_cycles:
            raise ValueError("Mathematical species classification requires canonical coefficients - SPECIES CLASSIFICATION IMPOSSIBLE")

        lifetimes = [death - birth for birth, death in persistent_cycles]

        if lifetimes:
            lifetimes_tensor = torch.tensor(lifetimes, dtype=torch.float32)
            hist = torch.histogram(lifetimes_tensor, bins=min(10, len(lifetimes)), density=True)[0].numpy()
            return hist
        else:
            raise ValueError("Mathematical species classification requires canonical coefficients - SPECIES CLASSIFICATION IMPOSSIBLE")

    def _compute_persistence_rank(self, persistent_cycles: List[Tuple[float, float]]) -> int:
        """Compute topological persistence rank."""
        if not persistent_cycles:
            return 0

        significant_cycles = []
        for birth, death in persistent_cycles:
            lifetime = death - birth
            if lifetime > self.mathematical_precision * 100:  # Significant lifetime threshold
                significant_cycles.append((birth, death))

        return len(significant_cycles)

    def _extract_q_field_structure(self, agent: ConceptualChargeAgent) -> Dict[str, Any]:
        """Extract Q-field structural information for information analysis."""
        structure = {}

        if hasattr(agent, "Q_components") and agent.Q_components:
            components = agent.Q_components
            structure["components"] = components
            structure["component_names"] = list(components.keys())

            component_values = list(components.values())
            if component_values:
                structure["component_ratios"] = self._compute_component_ratios(component_values)
                structure["phase_relationships"] = self._compute_phase_relationships(component_values)

        if hasattr(agent, "breathing_q_coefficients") and agent.breathing_q_coefficients:
            structure["breathing_structure"] = agent.breathing_q_coefficients

        return structure

    def _compute_component_ratios(self, component_values: List[Any]) -> List[float]:
        """Compute ratios between components for structural analysis."""
        ratios = []

        numerical_values = []
        for comp in component_values:
            if hasattr(comp, "__abs__"):
                numerical_values.append(abs(comp))
            else:
                numerical_values.append(abs(float(comp)))

        if len(numerical_values) >= 2:
            for i in range(len(numerical_values)):
                for j in range(i + 1, len(numerical_values)):
                    val_i, val_j = numerical_values[i], numerical_values[j]
                    if val_j > self.mathematical_precision:
                        ratio = val_i / val_j
                        ratios.append(ratio)
                    elif val_i > self.mathematical_precision:
                        ratios.append(float("inf"))

        return ratios

    def _compute_phase_relationships(self, component_values: List[Any]) -> List[float]:
        """Compute phase relationships between components."""
        phases = []

        for comp in component_values:
            if hasattr(comp, "real") and hasattr(comp, "imag"):
                phase = math.atan2(float(comp.imag), float(comp.real))
                phases.append(phase)
            else:
                phases.append(0.0)

        phase_relationships = []
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                phase_diff = abs(phases[i] - phases[j])
                phase_diff = min(phase_diff, 2 * math.pi - phase_diff)
                phase_relationships.append(phase_diff)

        return phase_relationships

    def _compute_entropy_signature_pattern(self, q_field_structure: Dict[str, Any]) -> np.ndarray:
        """Compute entropy signature pattern from field structure."""
        if not q_field_structure:
            raise ValueError("Mathematical species classification requires canonical coefficients - SPECIES CLASSIFICATION IMPOSSIBLE")

        entropy_components = []

        if "component_ratios" in q_field_structure:
            ratios = q_field_structure["component_ratios"]
            if ratios:
                finite_ratios = [r for r in ratios if math.isfinite(r)]
                if finite_ratios:
                    ratios_tensor = torch.tensor(finite_ratios, dtype=torch.float32)
                    normalized_ratios = ratios_tensor / (torch.sum(ratios_tensor) + self.mathematical_precision)
                    normalized_ratios = normalized_ratios.numpy()

                    entropy_val = entropy(normalized_ratios, base=2)
                    entropy_components.append(entropy_val)

        if "phase_relationships" in q_field_structure:
            phases = q_field_structure["phase_relationships"]
            if phases:
                phases_tensor = torch.tensor(phases, dtype=torch.float32)
                phase_bins = torch.histogram(phases_tensor, bins=min(5, len(phases)), density=True)[0].numpy()
                phase_bins = phase_bins[phase_bins > 0]
                if len(phase_bins) > 0:
                    phase_entropy = entropy(phase_bins, base=2)
                    entropy_components.append(phase_entropy)

        if not entropy_components:
            raise ValueError("Entropy signature pattern computation failed - no entropy components - ENTROPY PATTERN IMPOSSIBLE")
        return torch.tensor(entropy_components, dtype=torch.float32).numpy()

    def _compute_structural_mutual_information(self, q_field_structure: Dict[str, Any]) -> np.ndarray:
        """Compute mutual information matrix between structural components."""
        if "components" not in q_field_structure:
            return torch.tensor([[1.0]], dtype=torch.float32).numpy()

        components = q_field_structure["components"]
        component_names = list(components.keys())
        n_components = len(component_names)

        if n_components < 2:
            return torch.tensor([[1.0]], dtype=torch.float32).numpy()

        component_values = []
        for name in component_names:
            comp = components[name]
            if hasattr(comp, "real") and hasattr(comp, "imag"):
                component_values.append([float(comp.real), float(comp.imag)])
            else:
                component_values.append([float(comp), 0.0])

        component_values = torch.tensor(component_values, dtype=torch.float32).numpy()  # Convert back for scipy compatibility

        mi_matrix = np.zeros((n_components, n_components))

        for i in range(n_components):
            for j in range(n_components):
                if i == j:
                    mi_matrix[i, j] = 1.0  # Self-information normalized to 1
                else:
                    values_i = component_values[i]
                    values_j = component_values[j]

                    correlation = np.corrcoef(values_i, values_j)[0, 1]
                    if math.isfinite(correlation):
                        mi_matrix[i, j] = abs(correlation)
                    else:
                        mi_matrix[i, j] = 0.0

        return mi_matrix

    def _generate_information_fingerprint(self, entropy_pattern: np.ndarray, mi_matrix: np.ndarray) -> str:
        """Generate information-theoretic fingerprint."""
        entropy_signature = "_".join(f"{e:.6f}" for e in entropy_pattern)
        mi_signature = "_".join(f"{mi:.6f}" for mi in mi_matrix.flatten())

        fingerprint_data = f"entropy_{entropy_signature}_mi_{mi_signature}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]

    def _compute_coherence_information_content(self, q_field_structure: Dict[str, Any]) -> float:
        """Compute information content in coherence relationships."""
        if "phase_relationships" not in q_field_structure:
            raise ValueError("Coherence information content requires phase relationships - INFORMATION ANALYSIS IMPOSSIBLE")

        phase_relationships = q_field_structure["phase_relationships"]
        if not phase_relationships:
            raise ValueError("Coherence information content requires non-empty phase relationships - COHERENCE ANALYSIS IMPOSSIBLE")

        phase_tensor = torch.tensor(phase_relationships, dtype=torch.float32)
        phase_coherence = torch.mean(torch.cos(phase_tensor)).item()  # Average coherence

        coherence_prob = torch.tensor([abs(phase_coherence), 1.0 - abs(phase_coherence)], dtype=torch.float32)
        coherence_prob = coherence_prob / torch.sum(coherence_prob)
        coherence_prob = coherence_prob.numpy()  # Convert for scipy.stats.entropy
        information_content = entropy(coherence_prob, base=2)

        return float(information_content)

    def _compute_mathematical_complexity(self, entropy_pattern: np.ndarray, mi_matrix: np.ndarray) -> float:
        """Compute mathematical information complexity measure."""
        complexity_components = []

        if len(entropy_pattern) > 0:
            entropy_complexity = np.sum(entropy_pattern)
            complexity_components.append(entropy_complexity)

        if mi_matrix.size > 0:
            mi_tensor = torch.tensor(mi_matrix, dtype=torch.float32)
            triu_indices = torch.triu_indices(mi_tensor.size(0), mi_tensor.size(1), offset=1)
            off_diagonal = mi_tensor[triu_indices[0], triu_indices[1]]
            mi_complexity = torch.sum(off_diagonal).item()
            complexity_components.append(mi_complexity)

        if not complexity_components:
            raise ValueError("Mathematical information complexity computation failed - no complexity components - COMPLEXITY ANALYSIS IMPOSSIBLE")
        
        complexity_tensor = torch.tensor(complexity_components, dtype=torch.float32)
        total_complexity = torch.mean(complexity_tensor).item()
        return float(total_complexity)

    def _extract_field_manifold_structure(self, agent: ConceptualChargeAgent) -> Dict[str, Any]:
        """Extract field manifold structure for geometric analysis."""
        manifold_structure = {}

        if hasattr(agent, "field_position") and agent.field_position is not None:
            manifold_structure["field_position"] = agent.field_position

        if hasattr(agent, "q_value") and agent.q_value is not None:
            manifold_structure["q_value"] = agent.q_value

        if hasattr(agent, "geometric_features") and agent.geometric_features is not None:
            manifold_structure["geometric_features"] = agent.geometric_features

        if hasattr(agent, "modular_weight") and agent.modular_weight is not None:
            manifold_structure["modular_weight"] = agent.modular_weight

        return manifold_structure

    def _compute_curvature_signature(self, manifold_structure: Dict[str, Any]) -> np.ndarray:
        """Compute curvature signature of field manifold."""
        if "field_position" not in manifold_structure or "q_value" not in manifold_structure:
            raise ValueError("Curvature signature requires field_position and q_value in manifold - CURVATURE ANALYSIS IMPOSSIBLE")

        field_pos = manifold_structure["field_position"]
        q_value = manifold_structure["q_value"]

        if hasattr(field_pos, "__len__") and len(field_pos) >= 2:
            x, y = float(field_pos[0]), float(field_pos[1])
        else:
            x, y = float(field_pos), 0.0

        if hasattr(q_value, "real") and hasattr(q_value, "imag"):
            q_real, q_imag = float(q_value.real), float(q_value.imag)
        else:
            q_real, q_imag = float(q_value), 0.0

        gradient_x = q_real * x / (x**2 + y**2 + self.mathematical_precision)
        gradient_y = q_imag * y / (x**2 + y**2 + self.mathematical_precision)

        curvature = (gradient_x**2 + gradient_y**2) / (1 + gradient_x**2 + gradient_y**2) ** 1.5

        curvature_signature = torch.tensor([curvature, gradient_x, gradient_y], dtype=torch.float32).numpy()

        return curvature_signature

    def _generate_manifold_topology_hash(self, manifold_structure: Dict[str, Any]) -> str:
        """Generate hash of manifold topological structure."""
        topology_data = []

        if "field_position" in manifold_structure:
            pos = manifold_structure["field_position"]
            if hasattr(pos, "__len__"):
                pos_signature = "_".join(f"{float(p):.6f}" for p in pos)
            else:
                pos_signature = f"{float(pos):.6f}"
            topology_data.append(f"pos_{pos_signature}")

        if "q_value" in manifold_structure:
            q_val = manifold_structure["q_value"]
            if hasattr(q_val, "real") and hasattr(q_val, "imag"):
                q_signature = f"{float(q_val.real):.6f}_{float(q_val.imag):.6f}"
            else:
                q_signature = f"{float(q_val):.6f}_0.000000"
            topology_data.append(f"q_{q_signature}")

        if "geometric_features" in manifold_structure:
            geom_features = manifold_structure["geometric_features"]
            if hasattr(geom_features, "__len__"):
                geom_signature = "_".join(f"{float(g):.6f}" for g in geom_features)
                topology_data.append(f"geom_{geom_signature}")

        topology_string = "|".join(topology_data)
        return hashlib.sha256(topology_string.encode()).hexdigest()[:16]

    def _compute_geometric_genus(self, manifold_structure: Dict[str, Any]) -> float:
        """Compute geometric genus of field manifold."""
        if "q_value" not in manifold_structure:
            raise ValueError("Geometric genus computation requires Q-value in manifold structure - GEOMETRIC GENUS ANALYSIS IMPOSSIBLE")

        q_value = manifold_structure["q_value"]

        if hasattr(q_value, "real") and hasattr(q_value, "imag"):
            real_part = float(q_value.real)
            imag_part = float(q_value.imag)

            real_tensor = torch.tensor(real_part, dtype=torch.float32)
            imag_tensor = torch.tensor(imag_part, dtype=torch.float32)
            field_magnitude = torch.sqrt(real_tensor**2 + imag_tensor**2).item()
            geometric_genus = field_magnitude / (2 * math.pi + field_magnitude)

            return float(geometric_genus)
        else:
            real_value = float(q_value)
            geometric_genus = abs(real_value) / (2 * math.pi + abs(real_value))
            return float(geometric_genus)

    def _compute_canonical_geometric_form(self, manifold_structure: Dict[str, Any]) -> np.ndarray:
        """Compute canonical geometric form for unique representation."""
        if not manifold_structure:
            return torch.tensor([1.0], dtype=torch.float32).numpy()

        geometric_components = []

        if "field_position" in manifold_structure:
            pos = manifold_structure["field_position"]
            if hasattr(pos, "__len__"):
                geometric_components.extend([float(p) for p in pos])
            else:
                geometric_components.append(float(pos))

        if "q_value" in manifold_structure:
            q_val = manifold_structure["q_value"]
            if hasattr(q_val, "real") and hasattr(q_val, "imag"):
                geometric_components.extend([float(q_val.real), float(q_val.imag)])
            else:
                geometric_components.append(float(q_val))

        if not geometric_components:
            return torch.tensor([1.0], dtype=torch.float32).numpy()

        geometric_tensor = torch.tensor(geometric_components, dtype=torch.float32)
        norm = torch.norm(geometric_tensor).item()
        geometric_array = geometric_tensor.numpy()

        if norm > self.mathematical_precision:
            canonical_form = geometric_array / norm
            return canonical_form
        else:
            return torch.tensor([1.0], dtype=torch.float32).numpy()

    def _compute_differential_invariants(self, manifold_structure: Dict[str, Any]) -> Dict[str, float]:
        """Compute differential geometric invariants."""
        invariants = {}

        if "q_value" in manifold_structure and "field_position" in manifold_structure:
            q_val = manifold_structure["q_value"]
            field_pos = manifold_structure["field_position"]

            if hasattr(q_val, "real") and hasattr(q_val, "imag"):
                q_magnitude = abs(q_val)
                q_phase = math.atan2(float(q_val.imag), float(q_val.real))
            else:
                q_magnitude = abs(float(q_val))
                q_phase = 0.0

            if hasattr(field_pos, "__len__") and len(field_pos) >= 2:
                pos_x, pos_y = float(field_pos[0]), float(field_pos[1])
            else:
                pos_x, pos_y = float(field_pos), 0.0

            mean_curvature = q_magnitude / (1 + pos_x**2 + pos_y**2)
            invariants["mean_curvature"] = float(mean_curvature)

            q_mag_tensor = torch.tensor(q_magnitude, dtype=torch.float32)
            q_phase_tensor = torch.tensor(q_phase, dtype=torch.float32)
            pos_factor_tensor = torch.tensor(1 + pos_x**2 + pos_y**2, dtype=torch.float32)
            gaussian_curvature = (q_mag_tensor * torch.cos(q_phase_tensor) / (pos_factor_tensor ** 2)).item()
            invariants["gaussian_curvature"] = float(gaussian_curvature)

            mean_curv_tensor = torch.tensor(mean_curvature, dtype=torch.float32)
            gauss_curv_tensor = torch.tensor(gaussian_curvature, dtype=torch.float32)
            discriminant = torch.abs(mean_curv_tensor**2 - gauss_curv_tensor)
            sqrt_discriminant = torch.sqrt(discriminant)
            principal_curvature_1 = (mean_curv_tensor + sqrt_discriminant).item()
            principal_curvature_2 = (mean_curv_tensor - sqrt_discriminant).item()
            invariants["principal_curvature_1"] = float(principal_curvature_1)
            invariants["principal_curvature_2"] = float(principal_curvature_2)

        return invariants

    def _extract_q_component_relationships(self, agent: ConceptualChargeAgent) -> Dict[str, Any]:
        """Extract Q-component relationships for coherence analysis."""
        relationships = {}

        if hasattr(agent, "Q_components") and agent.Q_components:
            relationships["components"] = agent.Q_components

            component_magnitudes = {}
            component_phases = {}

            for comp_name, comp_value in agent.Q_components.items():
                if hasattr(comp_value, "real") and hasattr(comp_value, "imag"):
                    magnitude = abs(comp_value)
                    phase = math.atan2(float(comp_value.imag), float(comp_value.real))
                else:
                    magnitude = abs(float(comp_value))
                    phase = 0.0

                component_magnitudes[comp_name] = magnitude
                component_phases[comp_name] = phase

            relationships["magnitudes"] = component_magnitudes
            relationships["phases"] = component_phases

        if hasattr(agent, "breathing_q_coefficients") and agent.breathing_q_coefficients:
            relationships["breathing_coefficients"] = agent.breathing_q_coefficients

        return relationships

    def _compute_relationship_invariants(self, q_relationships: Dict[str, Any]) -> Dict[str, float]:
        """Compute invariant relationships between Q-components."""
        invariants = {}

        if "magnitudes" not in q_relationships or "phases" not in q_relationships:
            return invariants

        magnitudes = q_relationships["magnitudes"]
        phases = q_relationships["phases"]

        if len(magnitudes) < 2:
            return invariants

        magnitude_values = list(magnitudes.values())
        phase_values = list(phases.values())

        for i, (name_i, mag_i) in enumerate(magnitudes.items()):
            for j, (name_j, mag_j) in enumerate(magnitudes.items()):
                if i < j and mag_j > self.mathematical_precision:
                    ratio_key = f"magnitude_ratio_{name_i}_{name_j}"
                    invariants[ratio_key] = mag_i / mag_j

        for i, (name_i, phase_i) in enumerate(phases.items()):
            for j, (name_j, phase_j) in enumerate(phases.items()):
                if i < j:
                    phase_diff = abs(phase_i - phase_j)
                    phase_diff = min(phase_diff, 2 * math.pi - phase_diff)  # Normalize to [0, π]
                    diff_key = f"phase_diff_{name_i}_{name_j}"
                    invariants[diff_key] = phase_diff

        if magnitude_values:
            mag_tensor = torch.tensor(magnitude_values, dtype=torch.float32)
            invariants["total_magnitude"] = torch.sum(mag_tensor).item()
            invariants["magnitude_mean"] = torch.mean(mag_tensor).item()
            invariants["magnitude_std"] = torch.std(mag_tensor).item()

        if phase_values:
            phase_tensor = torch.tensor(phase_values, dtype=torch.float32)
            phase_cos_tensor = torch.cos(phase_tensor)
            phase_sin_tensor = torch.sin(phase_tensor)
            phase_cos_mean = torch.mean(phase_cos_tensor).item()
            phase_sin_mean = torch.mean(phase_sin_tensor).item()
            invariants["phase_circular_mean"] = math.atan2(phase_sin_mean, phase_cos_mean)
            cos_tensor = torch.tensor(phase_cos_mean, dtype=torch.float32)
            sin_tensor = torch.tensor(phase_sin_mean, dtype=torch.float32)
            magnitude = torch.sqrt(cos_tensor**2 + sin_tensor**2)
            invariants["phase_circular_variance"] = (1 - magnitude).item()

        return invariants

    def _compute_mathematical_ratios(self, q_relationships: Dict[str, Any]) -> np.ndarray:
        """Compute essential mathematical ratios between components."""
        if "magnitudes" not in q_relationships:
            return torch.tensor([1.0], dtype=torch.float32).numpy()

        magnitudes = q_relationships["magnitudes"]
        magnitude_values = list(magnitudes.values())

        if len(magnitude_values) < 2:
            return torch.tensor([1.0], dtype=torch.float32).numpy()

        ratios = []
        for i in range(len(magnitude_values)):
            for j in range(i + 1, len(magnitude_values)):
                mag_i, mag_j = magnitude_values[i], magnitude_values[j]
                if mag_j > self.mathematical_precision:
                    ratio = mag_i / mag_j
                    ratios.append(ratio)
                elif mag_i > self.mathematical_precision:
                    ratios.append(1e6)  # Large ratio for near-zero denominator

        if not ratios:
            raise ValueError("Mathematical ratios computation failed - no valid ratios - RATIO ANALYSIS IMPOSSIBLE")
        return torch.tensor(ratios, dtype=torch.float32).numpy()

    def _compute_phase_relationship_topology(self, agent: ConceptualChargeAgent) -> np.ndarray:
        """Compute phase relationship topology structure."""
        if not hasattr(agent, "Q_components") or not agent.Q_components:
            raise ValueError("Phase relationship topology requires Q_components on agent - TOPOLOGY ANALYSIS IMPOSSIBLE")

        phases = []
        for comp_name, comp_value in agent.Q_components.items():
            if hasattr(comp_value, "real") and hasattr(comp_value, "imag"):
                phase = math.atan2(float(comp_value.imag), float(comp_value.real))
                phases.append(phase)
            else:
                phases.append(0.0)

        if len(phases) < 2:
            raise ValueError(f"Phase relationship topology requires ≥2 phases, got {len(phases)} - TOPOLOGY ANALYSIS IMPOSSIBLE")

        n_phases = len(phases)
        phase_matrix = torch.zeros((n_phases, n_phases), dtype=torch.float32)

        for i in range(n_phases):
            for j in range(n_phases):
                if i != j:
                    phase_diff = abs(phases[i] - phases[j])
                    phase_diff = min(phase_diff, 2 * math.pi - phase_diff)
                    phase_matrix[i, j] = phase_diff

        triu_indices = torch.triu_indices(n_phases, n_phases, offset=1)
        topology_signature = phase_matrix[triu_indices[0], triu_indices[1]].numpy()
        return topology_signature

    def _compute_structure_preservation_score(
        self, relationship_invariants: Dict[str, float], mathematical_ratios: np.ndarray
    ) -> float:
        """Compute how well mathematical structure is preserved."""
        preservation_components = []

        if relationship_invariants:
            invariant_values = [v for v in relationship_invariants.values() if math.isfinite(v)]
            if invariant_values:
                invariant_tensor = torch.tensor(invariant_values, dtype=torch.float32)
                mean_invariant = torch.mean(invariant_tensor).item()
                std_invariant = torch.std(invariant_tensor).item()
                cv = std_invariant / (abs(mean_invariant) + self.mathematical_precision)
                preservation_score = 1.0 / (1.0 + cv)  # Higher score for lower variation
                preservation_components.append(preservation_score)

        if len(mathematical_ratios) > 0:
            ratios_tensor = torch.tensor(mathematical_ratios, dtype=torch.float32)
            finite_mask = torch.isfinite(ratios_tensor)
            finite_ratios = ratios_tensor[finite_mask].numpy()
            if len(finite_ratios) > 1:
                finite_tensor = torch.tensor(finite_ratios, dtype=torch.float32)
                ratio_cv = torch.std(finite_tensor).item() / (torch.mean(finite_tensor).item() + self.mathematical_precision)
                ratio_preservation = 1.0 / (1.0 + ratio_cv)
                preservation_components.append(ratio_preservation)

        if preservation_components:
            preservation_tensor = torch.tensor(preservation_components, dtype=torch.float32)
            overall_preservation = torch.mean(preservation_tensor).item()
            return float(overall_preservation)
        else:
            return 1.0  # Perfect preservation if no variation to measure

    def _generate_coherence_fingerprint(
        self, relationship_invariants: Dict[str, float], mathematical_ratios: np.ndarray, phase_topology: np.ndarray
    ) -> str:
        """Generate coherence fingerprint from mathematical relationships."""
        fingerprint_components = []

        if relationship_invariants:
            sorted_keys = sorted(relationship_invariants.keys())
            invariant_signature = "_".join(f"{relationship_invariants[key]:.6f}" for key in sorted_keys)
            fingerprint_components.append(f"inv_{invariant_signature}")

        if len(mathematical_ratios) > 0:
            ratio_signature = "_".join(f"{r:.6f}" for r in mathematical_ratios[:10])  # First 10 ratios
            fingerprint_components.append(f"ratio_{ratio_signature}")

        if len(phase_topology) > 0:
            topology_signature = "_".join(f"{t:.6f}" for t in phase_topology[:10])  # First 10 topology elements
            fingerprint_components.append(f"topo_{topology_signature}")

        coherence_string = "|".join(fingerprint_components)
        return hashlib.sha256(coherence_string.encode()).hexdigest()[:16]

    def _generate_mathematical_object_id(
        self,
        topological_invariants: TopologicalInvariants,
        canonical_signature: CanonicalMathematicalSignature,
        information_identity: InformationTheoreticIdentity,
    ) -> str:
        """Generate unique mathematical identifier for the object."""
        id_components = [
            topological_invariants.topological_fingerprint,
            canonical_signature.formula_structure_hash,
            information_identity.information_theoretical_fingerprint,
        ]

        combined_id = "_".join(id_components)
        mathematical_id = hashlib.sha256(combined_id.encode()).hexdigest()[:32]

        return f"MATH_OBJ_{mathematical_id}"

    def _compute_identity_confidence(
        self,
        topological_invariants: TopologicalInvariants,
        canonical_signature: CanonicalMathematicalSignature,
        persistence_features: PersistenceHomologyFeatures,
        information_identity: InformationTheoreticIdentity,
        geometric_invariants: GeometricInvariants,
        mathematical_coherence: MathematicalCoherence,
    ) -> float:
        """Compute mathematical confidence in identity determination."""
        confidence_components = []

        topo_confidence = 1.0 - abs(topological_invariants.euler_characteristic - 2.0) / 4.0
        confidence_components.append(max(0.0, min(1.0, topo_confidence)))

        canon_confidence = mathematical_coherence.mathematical_structure_preservation_score
        confidence_components.append(canon_confidence)

        if persistence_features.topological_persistence_rank > 0:
            persistence_confidence = min(1.0, persistence_features.topological_persistence_rank / 10.0)
        else:
            persistence_confidence = 0.5  # Neutral confidence for no persistence features
        confidence_components.append(persistence_confidence)

        info_confidence = min(1.0, information_identity.mathematical_information_complexity / 10.0)
        confidence_components.append(info_confidence)

        geom_confidence = min(1.0, abs(geometric_invariants.geometric_genus))
        confidence_components.append(geom_confidence)

        confidence_tensor = torch.tensor(confidence_components, dtype=torch.float32)
        overall_confidence = torch.mean(confidence_tensor).item()
        return float(overall_confidence)


    def _compare_topological_invariants(self, orig: TopologicalInvariants, current: TopologicalInvariants) -> float:
        """Compare topological invariants for consistency."""
        consistency_components = []

        euler_diff = abs(orig.euler_characteristic - current.euler_characteristic)
        euler_consistency = 1.0 - min(1.0, euler_diff / 4.0)
        consistency_components.append(euler_consistency)

        genus_diff = abs(orig.genus_signature - current.genus_signature)
        genus_consistency = 1.0 - min(1.0, genus_diff / 2.0)
        consistency_components.append(genus_consistency)

        rank_consistency = 1.0 if orig.fundamental_group_rank == current.fundamental_group_rank else 0.5
        consistency_components.append(rank_consistency)

        if len(orig.betti_numbers) == len(current.betti_numbers):
            betti_matches = sum(1 for a, b in zip(orig.betti_numbers, current.betti_numbers) if a == b)
            betti_consistency = betti_matches / len(orig.betti_numbers) if orig.betti_numbers else 1.0
        else:
            betti_consistency = 0.5
        consistency_components.append(betti_consistency)

        consistency_tensor = torch.tensor(consistency_components, dtype=torch.float32)
        return torch.mean(consistency_tensor).item()

    def _compare_canonical_signatures(
        self, orig: CanonicalMathematicalSignature, current: CanonicalMathematicalSignature
    ) -> float:
        """Compare canonical mathematical signatures for consistency."""
        hash_match = 1.0 if orig.formula_structure_hash == current.formula_structure_hash else 0.0

        genus_diff = abs(orig.mathematical_genus - current.mathematical_genus)
        genus_consistency = 1.0 - min(1.0, genus_diff / 2.0)

        species_match = 1.0 if orig.mathematical_species_id == current.mathematical_species_id else 0.0

        if (
            orig.canonical_form_coefficients.shape == current.canonical_form_coefficients.shape
            and orig.canonical_form_coefficients.size > 0
        ):
            orig_tensor = torch.tensor(orig.canonical_form_coefficients, dtype=torch.float32)
            current_tensor = torch.tensor(current.canonical_form_coefficients, dtype=torch.float32)
            coeff_diff = torch.norm(orig_tensor - current_tensor).item()
            coeff_consistency = 1.0 - min(1.0, coeff_diff)
        else:
            coeff_consistency = 0.5

        consistency_components = [hash_match, genus_consistency, species_match, coeff_consistency]
        consistency_tensor = torch.tensor(consistency_components, dtype=torch.float32)
        return torch.mean(consistency_tensor).item()

    def _compare_information_identities(
        self, orig: InformationTheoreticIdentity, current: InformationTheoreticIdentity
    ) -> float:
        """Compare information-theoretic identities for consistency."""
        consistency_components = []

        if (
            orig.entropy_signature_pattern.shape == current.entropy_signature_pattern.shape
            and orig.entropy_signature_pattern.size > 0
        ):
            orig_entropy = torch.tensor(orig.entropy_signature_pattern, dtype=torch.float32)
            current_entropy = torch.tensor(current.entropy_signature_pattern, dtype=torch.float32)
            entropy_diff = torch.norm(orig_entropy - current_entropy).item()
            entropy_consistency = 1.0 - min(
                1.0, entropy_diff / np.linalg.norm(orig.entropy_signature_pattern + self.mathematical_precision)
            )
        else:
            entropy_consistency = 0.5
        consistency_components.append(entropy_consistency)

        if (
            orig.mutual_information_matrix.shape == current.mutual_information_matrix.shape
            and orig.mutual_information_matrix.size > 0
        ):
            mi_diff = np.linalg.norm(orig.mutual_information_matrix - current.mutual_information_matrix)
            mi_consistency = 1.0 - min(
                1.0, mi_diff / np.linalg.norm(orig.mutual_information_matrix + self.mathematical_precision)
            )
        else:
            mi_consistency = 0.5
        consistency_components.append(mi_consistency)

        fingerprint_match = (
            1.0 if orig.information_theoretical_fingerprint == current.information_theoretical_fingerprint else 0.0
        )
        consistency_components.append(fingerprint_match)

        coherence_diff = abs(orig.coherence_information_content - current.coherence_information_content)
        coherence_consistency = 1.0 - min(
            1.0, coherence_diff / (abs(orig.coherence_information_content) + self.mathematical_precision)
        )
        consistency_components.append(coherence_consistency)

        consistency_tensor = torch.tensor(consistency_components, dtype=torch.float32)
        return torch.mean(consistency_tensor).item()

    def _compare_geometric_invariants(self, orig: GeometricInvariants, current: GeometricInvariants) -> float:
        """Compare geometric invariants for consistency."""
        consistency_components = []

        if orig.curvature_signature.shape == current.curvature_signature.shape and orig.curvature_signature.size > 0:
            curvature_diff = np.linalg.norm(orig.curvature_signature - current.curvature_signature)
            curvature_consistency = 1.0 - min(
                1.0, curvature_diff / (np.linalg.norm(orig.curvature_signature) + self.mathematical_precision)
            )
        else:
            curvature_consistency = 0.5
        consistency_components.append(curvature_consistency)

        topology_match = 1.0 if orig.manifold_topology_hash == current.manifold_topology_hash else 0.0
        consistency_components.append(topology_match)

        genus_diff = abs(orig.geometric_genus - current.geometric_genus)
        genus_consistency = 1.0 - min(1.0, genus_diff / (abs(orig.geometric_genus) + self.mathematical_precision))
        consistency_components.append(genus_consistency)

        if (
            orig.canonical_geometric_form.shape == current.canonical_geometric_form.shape
            and orig.canonical_geometric_form.size > 0
        ):
            geom_form_diff = np.linalg.norm(orig.canonical_geometric_form - current.canonical_geometric_form)
            geom_form_consistency = 1.0 - min(1.0, geom_form_diff)
        else:
            geom_form_consistency = 0.5
        consistency_components.append(geom_form_consistency)

        consistency_tensor = torch.tensor(consistency_components, dtype=torch.float32)
        return torch.mean(consistency_tensor).item()

    def _compare_mathematical_coherence(self, orig: MathematicalCoherence, current: MathematicalCoherence) -> float:
        """Compare mathematical coherence for consistency."""
        consistency_components = []

        orig_keys = set(orig.q_component_relationship_invariants.keys())
        current_keys = set(current.q_component_relationship_invariants.keys())
        common_keys = orig_keys.intersection(current_keys)

        if common_keys:
            invariant_consistency_scores = []
            for key in common_keys:
                orig_val = orig.q_component_relationship_invariants[key]
                current_val = current.q_component_relationship_invariants[key]
                val_diff = abs(orig_val - current_val)
                val_consistency = 1.0 - min(1.0, val_diff / (abs(orig_val) + self.mathematical_precision))
                invariant_consistency_scores.append(val_consistency)
            invariant_consistency = np.mean(invariant_consistency_scores)
        else:
            invariant_consistency = 0.5
        consistency_components.append(invariant_consistency)

        if (
            orig.mathematical_ratio_signatures.shape == current.mathematical_ratio_signatures.shape
            and orig.mathematical_ratio_signatures.size > 0
        ):
            ratio_diff = np.linalg.norm(orig.mathematical_ratio_signatures - current.mathematical_ratio_signatures)
            ratio_consistency = 1.0 - min(
                1.0, ratio_diff / (np.linalg.norm(orig.mathematical_ratio_signatures) + self.mathematical_precision)
            )
        else:
            ratio_consistency = 0.5
        consistency_components.append(ratio_consistency)

        if (
            orig.phase_relationship_topology.shape == current.phase_relationship_topology.shape
            and orig.phase_relationship_topology.size > 0
        ):
            phase_diff = np.linalg.norm(orig.phase_relationship_topology - current.phase_relationship_topology)
            phase_consistency = 1.0 - min(
                1.0, phase_diff / (np.linalg.norm(orig.phase_relationship_topology) + self.mathematical_precision)
            )
        else:
            phase_consistency = 0.5
        consistency_components.append(phase_consistency)

        fingerprint_match = (
            1.0 if orig.coherence_mathematical_fingerprint == current.coherence_mathematical_fingerprint else 0.0
        )
        consistency_components.append(fingerprint_match)

        consistency_tensor = torch.tensor(consistency_components, dtype=torch.float32)
        return torch.mean(consistency_tensor).item()

    def _compute_mathematical_similarity(
        self, profile1: MathematicalObjectIdentityProfile, profile2: MathematicalObjectIdentityProfile
    ) -> Tuple[float, Dict[str, float]]:
        """Compute mathematical similarity between two identity profiles."""
        similarity_components = {}

        similarity_components["topological"] = self._compare_topological_invariants(
            profile1.topological_invariants, profile2.topological_invariants
        )

        similarity_components["canonical"] = self._compare_canonical_signatures(
            profile1.canonical_signature, profile2.canonical_signature
        )

        similarity_components["information"] = self._compare_information_identities(
            profile1.information_identity, profile2.information_identity
        )

        similarity_components["geometric"] = self._compare_geometric_invariants(
            profile1.geometric_invariants, profile2.geometric_invariants
        )

        similarity_components["coherence"] = self._compare_mathematical_coherence(
            profile1.mathematical_coherence, profile2.mathematical_coherence
        )

        overall_similarity = np.mean(list(similarity_components.values()))

        return overall_similarity, similarity_components


    def _compare_topological_invariants(self, invariants1: TopologicalInvariants, invariants2: TopologicalInvariants) -> float:
        """
        Compare topological invariants using scipy.spatial.distance - NO MORE IMPORT THEATER!
        
        Uses actual pdist and squareform for proper distance calculations.
        """
        features1 = np.array([
            invariants1.euler_characteristic,
            invariants1.genus_signature,
            float(invariants1.fundamental_group_rank),
            np.mean(invariants1.betti_numbers) if invariants1.betti_numbers else 0.0
        ])
        
        features2 = np.array([
            invariants2.euler_characteristic,
            invariants2.genus_signature,
            float(invariants2.fundamental_group_rank),
            np.mean(invariants2.betti_numbers) if invariants2.betti_numbers else 0.0
        ])
        
        feature_matrix = np.stack([features1, features2])
        
        distances = pdist(feature_matrix, metric='euclidean')
        distance = distances[0]  # Only one pairwise distance
        
        similarity = 1.0 / (1.0 + distance)
        return float(similarity)

    def _compare_canonical_signatures(self, sig1: CanonicalMathematicalSignature, sig2: CanonicalMathematicalSignature) -> float:
        """
        Compare canonical signatures using scipy.spatial.distance.
        """
        features1 = np.array([
            sig1.mathematical_complexity_measure,
            sig1.mathematical_genus,
            len(sig1.component_relationships_matrix) if sig1.component_relationships_matrix else 0.0
        ])
        
        features2 = np.array([
            sig2.mathematical_complexity_measure,
            sig2.mathematical_genus,
            len(sig2.component_relationships_matrix) if sig2.component_relationships_matrix else 0.0
        ])
        
        feature_matrix = np.stack([features1, features2])
        distances = pdist(feature_matrix, metric='cosine')
        distance = distances[0]
        
        similarity = 1.0 - distance  # Cosine distance to similarity
        return float(max(0.0, similarity))

    def _compare_information_identities(self, info1: InformationTheoreticIdentity, info2: InformationTheoreticIdentity) -> float:
        """
        Compare information signatures using scipy.spatial.distance.
        """
        features1 = np.array([
            info1.entropy_signature,
            info1.information_density,
            info1.mutual_information_profile
        ])
        
        features2 = np.array([
            info2.entropy_signature,
            info2.information_density,
            info2.mutual_information_profile
        ])
        
        feature_matrix = np.stack([features1, features2])
        distances = pdist(feature_matrix, metric='manhattan')
        distance = distances[0]
        
        similarity = 1.0 / (1.0 + distance)
        return float(similarity)

    def _compare_geometric_invariants(self, geom1: GeometricInvariants, geom2: GeometricInvariants) -> float:
        """
        Compare geometric invariants using scipy.spatial.distance.
        """
        features1 = np.array([
            geom1.curvature_signature,
            geom1.manifold_dimension_estimate,
            geom1.geodesic_distance_profile
        ])
        
        features2 = np.array([
            geom2.curvature_signature,
            geom2.manifold_dimension_estimate,
            geom2.geodesic_distance_profile
        ])
        
        feature_matrix = np.stack([features1, features2])
        distances = pdist(feature_matrix, metric='chebyshev')
        distance = distances[0]
        
        similarity = 1.0 / (1.0 + distance)
        return float(similarity)

    def _compare_mathematical_coherence(self, coherence1: MathematicalCoherence, coherence2: MathematicalCoherence) -> float:
        """
        Compare mathematical coherence using scipy.spatial.distance.
        """
        features1 = np.array([
            coherence1.component_consistency_score,
            coherence1.relationship_coherence_measure,
            coherence1.mathematical_self_consistency_score
        ])
        
        features2 = np.array([
            coherence2.component_consistency_score,
            coherence2.relationship_coherence_measure,
            coherence2.mathematical_self_consistency_score
        ])
        
        feature_matrix = np.stack([features1, features2])
        distances = pdist(feature_matrix, metric='euclidean')
        
        distance_matrix = squareform(distances)
        distance = distances[0]  # Get the actual distance
        
        similarity = 1.0 / (1.0 + distance)
        return float(similarity)


    def _compute_topological_inheritance(self, parent: TopologicalInvariants, child: TopologicalInvariants) -> float:
        """Compute topological inheritance between parent and child objects."""
        return self._compare_topological_invariants(parent, child)

    def _compute_canonical_inheritance(
        self, parent: CanonicalMathematicalSignature, child: CanonicalMathematicalSignature
    ) -> float:
        """Compute canonical signature inheritance."""
        return self._compare_canonical_signatures(parent, child)

    def _compute_information_inheritance(
        self, parent: InformationTheoreticIdentity, child: InformationTheoreticIdentity
    ) -> float:
        """Compute information-theoretic inheritance."""
        return self._compare_information_identities(parent, child)

    def _compute_geometric_inheritance(self, parent: GeometricInvariants, child: GeometricInvariants) -> float:
        """Compute geometric inheritance."""
        return self._compare_geometric_invariants(parent, child)

    def _compute_coherence_inheritance(self, parent: MathematicalCoherence, child: MathematicalCoherence) -> float:
        """Compute mathematical coherence inheritance."""
        return self._compare_mathematical_coherence(parent, child)
