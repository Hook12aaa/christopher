"""
Test suite for conceptual charge implementation.
"""
import pytest
import numpy as np
from core_mathematics.conceptual_charge import ConceptualCharge, EmotionalSpectrum, TemporalContext
from embedding_engine.models import ConceptualChargeGenerator


class TestConceptualCharge:
    """Test the ConceptualCharge class implementation."""
    
    def test_basic_charge_creation(self):
        """Test basic conceptual charge creation."""
        semantic_vector = np.random.randn(1024)
        emotional_context = EmotionalSpectrum(valence=0.5, arousal=0.3, dominance=0.7)
        temporal_context = TemporalContext(recency=0.8, duration=0.6, frequency=0.4)
        
        charge = ConceptualCharge(
            semantic_vector=semantic_vector,
            emotional_context=emotional_context,
            temporal_context=temporal_context
        )
        
        assert charge.semantic_vector.shape == (1024,)
        assert charge.emotional_context.valence == 0.5
        assert charge.temporal_context.recency == 0.8
        assert charge.vorticity.shape == (3,)
    
    def test_charge_magnitude(self):
        """Test charge magnitude calculation."""
        semantic_vector = np.ones(1024) * 0.1
        charge = ConceptualCharge(semantic_vector=semantic_vector)
        
        magnitude = charge.get_charge_magnitude()
        assert magnitude > 0
        assert isinstance(magnitude, float)
    
    def test_temporal_decay(self):
        """Test temporal decay function."""
        semantic_vector = np.random.randn(1024)
        temporal_context = TemporalContext(recency=0.5, duration=0.5, frequency=0.5)
        
        charge = ConceptualCharge(
            semantic_vector=semantic_vector,
            temporal_context=temporal_context
        )
        
        decay_0 = charge.temporal_decay(0.0)
        decay_1 = charge.temporal_decay(1.0)
        decay_2 = charge.temporal_decay(2.0)
        
        assert decay_0 == 1.0  # No decay at t=0
        assert 0 <= decay_1 <= 1.0
        assert 0 <= decay_2 <= 1.0
        assert decay_2 <= decay_1  # Decay increases over time
    
    def test_phase_factor(self):
        """Test phase factor calculation."""
        semantic_vector = np.random.randn(1024)
        emotional_context = EmotionalSpectrum(valence=0.5, arousal=0.3, dominance=0.7)
        temporal_context = TemporalContext(recency=0.8, duration=0.6, frequency=0.4)
        
        charge = ConceptualCharge(
            semantic_vector=semantic_vector,
            emotional_context=emotional_context,
            temporal_context=temporal_context
        )
        
        phase = charge.get_phase_factor()
        assert 0 <= phase <= 2 * np.pi
    
    def test_full_charge_vector(self):
        """Test full charge vector combination."""
        semantic_vector = np.random.randn(1024)
        emotional_context = EmotionalSpectrum(valence=0.5, arousal=0.3, dominance=0.7)
        temporal_context = TemporalContext(recency=0.8, duration=0.6, frequency=0.4)
        
        charge = ConceptualCharge(
            semantic_vector=semantic_vector,
            emotional_context=emotional_context,
            temporal_context=temporal_context
        )
        
        full_charge = charge.get_full_charge()
        # Should combine: semantic (1024) + emotional (3) + temporal (3) + vorticity (3)
        assert full_charge.shape == (1033,)


class TestConceptualChargeGenerator:
    """Test the ConceptualChargeGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a ConceptualChargeGenerator instance for testing."""
        return ConceptualChargeGenerator()
    
    def test_text_encoding(self, generator):
        """Test basic text encoding."""
        text = "This is a test sentence for semantic embedding."
        embeddings = generator.encode_text(text)
        
        assert embeddings.shape == (1024,)
        assert embeddings.dtype == np.float32
    
    def test_batch_text_encoding(self, generator):
        """Test batch text encoding."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        embeddings = generator.encode_text(texts)
        
        assert embeddings.shape == (3, 1024)
        assert embeddings.dtype == np.float32
    
    def test_conceptual_charge_creation(self, generator):
        """Test creating conceptual charge from text."""
        text = "Art is a form of cultural expression."
        emotional_context = {
            "valence": 0.7,
            "arousal": 0.5,
            "dominance": 0.6
        }
        temporal_context = {
            "recency": 0.9,
            "duration": 0.8,
            "frequency": 0.3
        }
        
        charge = generator.create_conceptual_charge(
            text=text,
            emotional_context=emotional_context,
            temporal_context=temporal_context
        )
        
        assert isinstance(charge, ConceptualCharge)
        assert charge.semantic_vector.shape == (1024,)
        assert charge.emotional_context.valence == 0.7
        assert charge.temporal_context.recency == 0.9
    
    def test_batch_charge_creation(self, generator):
        """Test creating multiple charges in batch."""
        texts = [
            "Jazz music in small venues.",
            "Classical concerts in grand halls.",
            "Street art in urban spaces."
        ]
        
        charges = generator.create_batch_charges(texts)
        
        assert len(charges) == 3
        assert all(isinstance(charge, ConceptualCharge) for charge in charges)
        assert all(charge.semantic_vector.shape == (1024,) for charge in charges)