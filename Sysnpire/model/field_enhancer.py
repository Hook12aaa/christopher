"""
Field Enhancer - Converts BGE embeddings to field-theoretic parameters

Transforms standard semantic embeddings into field theory parameters
for conceptual charge generation.
"""

import numpy as np
from typing import Dict, List
import sys
from pathlib import Path

# Add project root to import the core math
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core_mathematics.conceptual_charge import ConceptualCharge

class FieldEnhancer:
    """Converts BGE embeddings to field-theoretic conceptual charges."""
    
    def __init__(self, embedding_dim: int = 1024):
        self.embedding_dim = embedding_dim
        self.trajectory_dimensions = 3
        self.emotional_dimensions = 16
        self.semantic_dimensions = 8
    
    def enhance_embedding(self, embedding: np.ndarray, text: str, 
                         observational_state: float = 1.0, 
                         gamma: float = 1.0) -> ConceptualCharge:
        """Convert BGE embedding to conceptual charge."""
        
        # Extract field parameters from embedding
        field_params = self._extract_field_parameters(embedding, text)
        
        # Create conceptual charge
        semantic_vector = embedding[:self.semantic_dimensions]
        charge = ConceptualCharge(
            token=text,
            semantic_vector=semantic_vector,
            observational_state=observational_state,
            gamma=gamma
        )
        
        # Set field parameters
        charge.omega_base = field_params['omega_base']
        charge.phi_base = field_params['phi_base']
        charge.alpha_emotional = field_params['alpha_emotional']
        charge.v_emotional = field_params['v_emotional']
        charge.sigma_emotional_sq = field_params['sigma_emotional_sq']
        charge.beta_breathing = field_params['beta_breathing']
        charge.w_weights = field_params['w_weights']
        charge.sigma_persistence_sq = field_params['sigma_persistence']
        charge.alpha_persistence = field_params['alpha_persistence']
        charge.lambda_persistence = field_params['lambda_persistence']
        charge.beta_persistence = field_params['beta_persistence']
        
        return charge
    
    def _extract_field_parameters(self, embedding: np.ndarray, text: str) -> Dict[str, np.ndarray]:
        """Extract field parameters from BGE embedding."""
        
        # Use text hash for deterministic parameter extraction
        text_hash = hash(text) % 2**32
        np.random.seed(text_hash)
        
        # Trajectory operators from embedding structure
        omega_base = np.abs(embedding[:self.trajectory_dimensions]) + 0.5
        phi_base = (embedding[self.trajectory_dimensions:2*self.trajectory_dimensions] * np.pi) % (2*np.pi)
        
        # Emotional parameters from embedding middle section
        emotional_start = self.embedding_dim // 4
        emotional_section = embedding[emotional_start:emotional_start + self.emotional_dimensions]
        alpha_emotional = np.abs(emotional_section) + 0.1
        
        # Emotional alignment vectors
        v_emotional = embedding[emotional_start + self.emotional_dimensions:
                               emotional_start + 2*self.emotional_dimensions]
        v_emotional = v_emotional / (np.linalg.norm(v_emotional) + 1e-10)
        
        # Semantic breathing parameters
        semantic_start = self.embedding_dim // 2
        beta_breathing = embedding[semantic_start:semantic_start + self.semantic_dimensions] * 0.5
        w_weights = np.abs(embedding[semantic_start + self.semantic_dimensions:
                                   semantic_start + 2*self.semantic_dimensions]) + 0.1
        
        # Persistence parameters from embedding statistics
        sigma_persistence = 0.5 + np.std(embedding) * 2.0
        alpha_persistence = 0.1 + np.abs(np.mean(embedding))
        lambda_persistence = 0.05 + np.abs(embedding[-2]) * 0.1
        beta_persistence = 0.2 + np.abs(embedding[-1]) * 0.3
        
        return {
            'omega_base': omega_base,
            'phi_base': phi_base,
            'alpha_emotional': alpha_emotional,
            'v_emotional': v_emotional,
            'sigma_emotional_sq': np.ones(self.emotional_dimensions) * 2.0,
            'beta_breathing': beta_breathing,
            'w_weights': w_weights,
            'sigma_persistence': sigma_persistence**2,
            'alpha_persistence': alpha_persistence,
            'lambda_persistence': lambda_persistence,
            'beta_persistence': beta_persistence
        }
    
    def compute_charge_values(self, charge: ConceptualCharge) -> Dict[str, any]:
        """Compute field values for a conceptual charge."""
        Q = charge.compute_complete_charge()
        
        return {
            'complete_charge': Q,
            'magnitude': abs(Q),
            'phase': np.angle(Q),
            'trajectory_operators': [charge.trajectory_operator(charge.observational_state, i) 
                                   for i in range(self.trajectory_dimensions)],
            'emotional_trajectory': charge.emotional_trajectory_integration(charge.observational_state),
            'semantic_field': charge.semantic_field_generation(charge.observational_state),
            'phase_total': charge.total_phase_integration(charge.observational_state),
            'persistence': charge.observational_persistence(charge.observational_state)
        }