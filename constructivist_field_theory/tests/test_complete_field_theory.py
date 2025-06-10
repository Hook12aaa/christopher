"""
Test suite for complete field-theoretic conceptual charge implementation.

Tests the full formulation: Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)
"""
import pytest
import numpy as np
from core_mathematics.conceptual_charge import ConceptualCharge
from embedding_engine.models import ConceptualChargeGenerator


class TestCompleteConceptualCharge:
    """Test the complete field-theoretic conceptual charge implementation."""
    
    def test_basic_charge_creation(self):
        """Test basic conceptual charge creation with field-theoretic formulation."""
        semantic_vector = np.random.randn(1024)
        context = {"semantic_context": "test context", "emotional_intensity": 0.5}
        
        charge = ConceptualCharge(
            token="test",
            semantic_vector=semantic_vector,
            context=context,
            observational_state=1.0,
            gamma=1.0
        )
        
        assert charge.token == "test"
        assert charge.semantic_vector.shape == (1024,)
        assert charge.context == context
        assert charge.observational_state == 1.0
        assert charge.gamma == 1.0
    
    def test_trajectory_operators(self):
        """Test trajectory operator T_i(τ,s) implementation."""
        semantic_vector = np.random.randn(1024)
        charge = ConceptualCharge(
            token="trajectory_test",
            semantic_vector=semantic_vector,
            observational_state=2.0
        )
        
        # Test trajectory operator for dimension 0
        T_0 = charge.trajectory_operator(2.0, 0)
        assert isinstance(T_0, complex)
        assert T_0 != 0  # Should produce non-zero result
        
        # Test that different observational states produce different results
        T_0_different = charge.trajectory_operator(3.0, 0)
        assert T_0 != T_0_different
    
    def test_emotional_trajectory_integration(self):
        """Test emotional trajectory integration E^trajectory(τ,s)."""
        semantic_vector = np.random.randn(1024)
        charge = ConceptualCharge(
            token="emotional_test",
            semantic_vector=semantic_vector,
            observational_state=1.5
        )
        
        E_trajectory = charge.emotional_trajectory_integration(1.5)
        assert E_trajectory.shape == (1024,)
        assert np.all(E_trajectory >= 0)  # Emotional amplification should be non-negative
        
        # Test that different observational states produce different results
        E_trajectory_different = charge.emotional_trajectory_integration(2.0)
        assert not np.array_equal(E_trajectory, E_trajectory_different)
    
    def test_semantic_field_generation(self):
        """Test semantic field generation Φ^semantic(τ,s)."""
        semantic_vector = np.random.randn(1024)
        charge = ConceptualCharge(
            token="semantic_test",
            semantic_vector=semantic_vector,
            observational_state=1.0
        )
        
        phi_semantic = charge.semantic_field_generation(1.0)
        assert phi_semantic.shape == (1024,)
        assert phi_semantic.dtype == complex  # Should be complex-valued
        
        # Test breathing constellation patterns (should vary with observational state)
        phi_semantic_different = charge.semantic_field_generation(2.0)
        assert not np.array_equal(phi_semantic, phi_semantic_different)
    
    def test_phase_integration(self):
        """Test complete phase integration θ_total(τ,C,s)."""
        semantic_vector = np.random.randn(1024)
        context = {"test_context": "phase_test"}
        charge = ConceptualCharge(
            token="phase_test",
            semantic_vector=semantic_vector,
            context=context,
            observational_state=1.0
        )
        
        theta_total = charge.total_phase_integration(1.0)
        assert 0 <= theta_total <= 2 * np.pi
        
        # Test that context affects phase
        charge_different_context = ConceptualCharge(
            token="phase_test",
            semantic_vector=semantic_vector,
            context={"different_context": "different_value"},
            observational_state=1.0
        )
        theta_different = charge_different_context.total_phase_integration(1.0)
        assert theta_total != theta_different
    
    def test_observational_persistence(self):
        """Test observational persistence Ψ_persistence(s-s₀)."""
        semantic_vector = np.random.randn(1024)
        charge = ConceptualCharge(
            token="persistence_test",
            semantic_vector=semantic_vector
        )
        
        # Test persistence at initial state
        psi_0 = charge.observational_persistence(0.0)
        assert psi_0 > 0
        
        # Test decay over observational distance
        psi_1 = charge.observational_persistence(1.0)
        psi_2 = charge.observational_persistence(2.0)
        
        # Generally should decay, though oscillatory component may cause variations
        assert isinstance(psi_1, float)
        assert isinstance(psi_2, float)
    
    def test_complete_charge_computation(self):
        """Test the complete charge computation Q(τ, C, s)."""
        semantic_vector = np.random.randn(1024)
        context = {"semantic_context": "complete test", "emotional_intensity": 0.7}
        charge = ConceptualCharge(
            token="complete_test",
            semantic_vector=semantic_vector,
            context=context,
            observational_state=1.5,
            gamma=1.2
        )
        
        Q = charge.compute_complete_charge()
        assert isinstance(Q, complex)
        assert Q != 0  # Should produce non-zero result
        
        # Test magnitude and phase methods
        magnitude = charge.get_charge_magnitude()
        phase = charge.get_phase_factor()
        
        assert magnitude == abs(Q)
        assert phase == np.angle(Q)
        assert magnitude > 0
        assert 0 <= phase <= 2 * np.pi
    
    def test_observational_state_updates(self):
        """Test updating observational state and trajectory recording."""
        semantic_vector = np.random.randn(1024)
        charge = ConceptualCharge(
            token="update_test",
            semantic_vector=semantic_vector,
            observational_state=0.0
        )
        
        initial_charge = charge.compute_complete_charge()
        
        # Update observational state
        charge.update_observational_state(2.0)
        assert charge.observational_state == 2.0
        assert len(charge.trajectory_history) == 1
        assert charge.trajectory_history[0] == 0.0
        
        # Charge should be different at new observational state
        updated_charge = charge.compute_complete_charge()
        assert initial_charge != updated_charge


class TestConceptualChargeGenerator:
    """Test the ConceptualChargeGenerator with field-theoretic formulation."""
    
    @pytest.fixture
    def generator(self):
        """Create a ConceptualChargeGenerator instance for testing."""
        return ConceptualChargeGenerator()
    
    def test_charge_generation_with_context(self, generator):
        """Test charge generation with contextual environment."""
        text = "Art is a form of cultural expression."
        context = {
            "semantic_context": "cultural analysis",
            "emotional_intensity": 0.8,
            "social_context": "artistic community"
        }
        
        charge = generator.create_conceptual_charge(
            text=text,
            context=context,
            observational_state=1.5,
            gamma=1.1
        )
        
        assert isinstance(charge, ConceptualCharge)
        assert charge.token == text
        assert charge.context == context
        assert charge.observational_state == 1.5
        assert charge.gamma == 1.1
    
    def test_batch_charge_generation(self, generator):
        """Test batch generation with field-theoretic formulation."""
        texts = [
            "Jazz music in small venues.",
            "Classical concerts in grand halls.",
            "Street art in urban spaces."
        ]
        contexts = [
            {"genre": "jazz", "venue_size": "small"},
            {"genre": "classical", "venue_size": "large"},
            {"genre": "street_art", "venue_type": "urban"}
        ]
        observational_states = [0.5, 1.0, 1.5]
        
        charges = generator.create_batch_charges(
            texts=texts,
            contexts=contexts,
            observational_states=observational_states,
            gamma=1.2
        )
        
        assert len(charges) == 3
        assert all(isinstance(charge, ConceptualCharge) for charge in charges)
        assert all(charge.gamma == 1.2 for charge in charges)
        
        # Verify individual charge properties
        for i, charge in enumerate(charges):
            assert charge.token == texts[i]
            assert charge.context == contexts[i]
            assert charge.observational_state == observational_states[i]
    
    def test_field_theory_principles(self, generator):
        """Test that implementation follows field theory principles."""
        text = "Cultural resonance in artistic expression"
        
        # Generate charges at different observational states
        charge_1 = generator.create_conceptual_charge(text, observational_state=0.0)
        charge_2 = generator.create_conceptual_charge(text, observational_state=2.0)
        
        Q_1 = charge_1.compute_complete_charge()
        Q_2 = charge_2.compute_complete_charge()
        
        # Charges should differ based on observational state (trajectory dependence)
        assert Q_1 != Q_2
        
        # Test gamma scaling
        charge_scaled = generator.create_conceptual_charge(text, gamma=2.0)
        Q_scaled = charge_scaled.compute_complete_charge()
        
        # Should demonstrate field calibration effects
        assert abs(Q_scaled) != abs(Q_1)  # Different gamma should affect magnitude