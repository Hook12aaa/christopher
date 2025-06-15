"""
Attention Deconstruction - Convert Transformer Attention to Emotional Field Effects

MATHEMATICAL FOUNDATION (README.md Section 3.1.3.2):
Standard Attention: Attention(Q, K, V) = softmax(QK^T / √d_k) · V
Geometric Interpretation: QK^T → alignment detection, softmax → amplification, V → transport

DECONSTRUCTION APPROACH:
1. QK^T analysis for emotional geometric patterns
2. Softmax geometric interpretation for field amplification  
3. Value transport conversion to field modulation
4. Static→Dynamic transformation for trajectory dependence

This module extracts the mathematical principles from transformer attention
and reconstructs them as explicit emotional field theory operations.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AttentionAnalysisResult:
    """Results from attention geometric analysis."""
    attention_weights: np.ndarray
    geometric_alignments: np.ndarray
    emotional_patterns: np.ndarray
    amplification_factors: np.ndarray
    field_effects: np.ndarray


class AttentionGeometryAnalyzer:
    """
    Analyze transformer attention mechanisms through geometric lens for emotional field extraction.
    
    MATHEMATICAL FOUNDATION:
    Decomposes attention operations into geometric components that reveal emotional patterns:
    1. QK^T → Directional alignment detection in semantic space
    2. Softmax → Exponential amplification of strong alignments  
    3. Weighted transport → Information flow along semantic geodesics
    
    EMOTIONAL INSIGHT:
    Emotional content naturally creates geometric patterns that attention mechanisms detect.
    By analyzing these patterns explicitly, we can extract emotional field effects.
    """
    
    def __init__(self, 
                 embedding_dimension: int = 1024,
                 attention_heads: int = 16,
                 emotional_threshold: float = 0.3):
        """
        Initialize attention geometry analyzer.
        
        Args:
            embedding_dimension: Dimension of semantic embeddings
            attention_heads: Number of attention heads to simulate
            emotional_threshold: Threshold for emotional pattern detection
        """
        self.embedding_dimension = embedding_dimension
        self.attention_heads = attention_heads
        self.emotional_threshold = emotional_threshold
        
        logger.info(f"Initialized AttentionGeometryAnalyzer for {embedding_dimension}D with {attention_heads} heads")
    
    def analyze_emotional_patterns(self,
                                 semantic_embedding: np.ndarray,
                                 token: str,
                                 context_embeddings: Optional[np.ndarray] = None) -> AttentionAnalysisResult:
        """
        Analyze emotional patterns using deconstructed attention mechanics.
        
        MATHEMATICAL PROCESS (README.md Section 3.1.3.2):
        1. Create Query-Key-Value matrices from semantic content
        2. Compute QK^T alignments for emotional pattern detection
        3. Apply geometric interpretation of softmax amplification
        4. Extract emotional field effects from attention patterns
        
        Args:
            semantic_embedding: Base semantic vector [D]
            token: Token identifier for pattern analysis
            context_embeddings: Optional context embeddings [N, D]
            
        Returns:
            AttentionAnalysisResult with extracted emotional patterns
        """
        try:
            # Step 1: Create Q, K, V matrices (Section 3.1.3.2.1)
            Q, K, V = self._create_qkv_matrices(semantic_embedding, token, context_embeddings)
            
            # Step 2: Compute QK^T alignments (geometric alignment detection)
            alignment_scores = self._compute_geometric_alignments(Q, K)
            
            # Step 3: Apply softmax geometric interpretation
            attention_weights = self._compute_attention_weights(alignment_scores)
            
            # Step 4: Extract emotional patterns from attention weights
            emotional_patterns = self._extract_emotional_patterns(
                attention_weights, semantic_embedding, token
            )
            
            # Step 5: Compute amplification factors
            amplification_factors = self._compute_amplification_factors(
                attention_weights, emotional_patterns
            )
            
            # Step 6: Convert to field effects
            field_effects = self._convert_to_field_effects(
                attention_weights, emotional_patterns, V
            )
            
            result = AttentionAnalysisResult(
                attention_weights=attention_weights,
                geometric_alignments=alignment_scores,
                emotional_patterns=emotional_patterns,
                amplification_factors=amplification_factors,
                field_effects=field_effects
            )
            
            logger.debug(f"Attention analysis complete for {token}: {len(emotional_patterns)} patterns detected")
            return result
            
        except Exception as e:
            logger.error(f"Attention analysis failed for {token}: {e}")
            # Return empty result
            return AttentionAnalysisResult(
                attention_weights=np.zeros((self.attention_heads, self.embedding_dimension)),
                geometric_alignments=np.zeros((self.attention_heads, self.embedding_dimension)),
                emotional_patterns=np.zeros(self.embedding_dimension),
                amplification_factors=np.ones(self.attention_heads),
                field_effects=np.zeros(self.embedding_dimension)
            )
    
    def _create_qkv_matrices(self,
                           semantic_embedding: np.ndarray,
                           token: str,
                           context_embeddings: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create Query, Key, Value matrices from semantic content.
        
        MATHEMATICAL APPROACH:
        - Query (Q): Represents what the token is "looking for" emotionally
        - Key (K): Represents what emotional content is available  
        - Value (V): Represents the emotional information to be transported
        """
        embedding_dim = len(semantic_embedding)
        
        # Create token-specific emotional query
        token_hash = hash(token) % 1000 / 1000.0
        Q = np.zeros((self.attention_heads, embedding_dim))
        
        for head in range(self.attention_heads):
            # Each head looks for different emotional aspects
            head_frequency = (head + 1) / self.attention_heads
            for i in range(embedding_dim):
                Q[head, i] = semantic_embedding[i] * np.cos(2 * np.pi * head_frequency * token_hash)
        
        # Create emotional key matrix
        K = np.zeros((self.attention_heads, embedding_dim))
        for head in range(self.attention_heads):
            K[head] = semantic_embedding * (1.0 + 0.1 * np.sin(2 * np.pi * head / self.attention_heads))
        
        # Value matrix contains the actual emotional content
        V = np.zeros((self.attention_heads, embedding_dim))
        for head in range(self.attention_heads):
            V[head] = semantic_embedding  # Base content
            
        # If context is available, incorporate it
        if context_embeddings is not None and len(context_embeddings) > 0:
            context_influence = np.mean(context_embeddings, axis=0)
            for head in range(self.attention_heads):
                V[head] += 0.1 * context_influence  # Add context influence
        
        return Q, K, V
    
    def _compute_geometric_alignments(self, Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Compute QK^T geometric alignments for emotional pattern detection.
        
        MATHEMATICAL FOUNDATION (README.md Section 3.1.3.2.1):
        qᵢ · kⱼ = ||qᵢ|| · ||kⱼ|| · cos(θᵢⱼ)
        - Dot products measure directional harmony in semantic space
        - High values indicate semantic-emotional alignment
        """
        alignment_scores = np.zeros((self.attention_heads, self.embedding_dimension))
        
        for head in range(self.attention_heads):
            # Compute alignment for each dimension
            for i in range(self.embedding_dimension):
                # QK^T computation with normalization
                q_norm = np.linalg.norm(Q[head])
                k_norm = np.linalg.norm(K[head])
                
                if q_norm > 0 and k_norm > 0:
                    alignment_scores[head, i] = np.dot(Q[head], K[head]) / (q_norm * k_norm)
                else:
                    alignment_scores[head, i] = 0.0
        
        return alignment_scores
    
    def _compute_attention_weights(self, alignment_scores: np.ndarray) -> np.ndarray:
        """
        Apply softmax geometric interpretation for attention weight computation.
        
        MATHEMATICAL FOUNDATION (README.md Section 3.1.3.2.1):
        softmax(αᵢ) = exp(αᵢ) / Σⱼ exp(αⱼ)
        - Exponential amplification of strong alignments
        - Suppression of weak connections
        """
        attention_weights = np.zeros_like(alignment_scores)
        
        for head in range(self.attention_heads):
            # Apply softmax to alignment scores
            scores = alignment_scores[head]
            
            # Prevent overflow in exponential
            scores_normalized = scores - np.max(scores)
            exp_scores = np.exp(scores_normalized)
            
            # Compute softmax
            sum_exp = np.sum(exp_scores)
            if sum_exp > 0:
                attention_weights[head] = exp_scores / sum_exp
            else:
                attention_weights[head] = np.ones(len(scores)) / len(scores)
        
        return attention_weights
    
    def _extract_emotional_patterns(self,
                                  attention_weights: np.ndarray,
                                  semantic_embedding: np.ndarray,
                                  token: str) -> np.ndarray:
        """
        Extract emotional patterns from attention weight distributions.
        
        EMOTIONAL INSIGHT (README.md Section 3.1.3.2.2):
        Emotional content creates characteristic attention patterns:
        - Clustering: Emotionally charged words cluster in semantic regions
        - Directional bias: Emotional contexts create directional alignments
        - Amplification: Emotionally coherent patterns get amplified
        """
        embedding_dim = len(semantic_embedding)
        emotional_patterns = np.zeros(embedding_dim)
        
        # Analyze attention weight patterns for emotional signatures
        for i in range(embedding_dim):
            # Collect attention weights across heads for dimension i
            head_weights = attention_weights[:, i]
            
            # Emotional pattern indicators:
            # 1. High variance → emotional significance
            # 2. High mean → emotional activation
            # 3. Pattern coherence → emotional consistency
            
            variance = np.var(head_weights)
            mean_weight = np.mean(head_weights)
            
            # Combine indicators for emotional pattern strength
            emotional_patterns[i] = mean_weight * (1.0 + variance)
        
        # Apply threshold for significant emotional patterns
        emotional_patterns = np.where(
            emotional_patterns > self.emotional_threshold,
            emotional_patterns,
            0.0
        )
        
        return emotional_patterns
    
    def _compute_amplification_factors(self,
                                     attention_weights: np.ndarray,
                                     emotional_patterns: np.ndarray) -> np.ndarray:
        """
        Compute amplification factors for emotional field effects.
        
        MATHEMATICAL FOUNDATION:
        Amplification represents how much attention mechanisms naturally amplify
        emotional content through their geometric operations.
        """
        amplification_factors = np.zeros(self.attention_heads)
        
        for head in range(self.attention_heads):
            # Compute correlation between attention weights and emotional patterns
            head_weights = attention_weights[head]
            
            if np.sum(emotional_patterns) > 0:
                # Correlation-based amplification
                correlation = np.corrcoef(head_weights, emotional_patterns)[0, 1]
                if not np.isnan(correlation):
                    amplification_factors[head] = 1.0 + abs(correlation)
                else:
                    amplification_factors[head] = 1.0
            else:
                amplification_factors[head] = 1.0
        
        return amplification_factors
    
    def _convert_to_field_effects(self,
                                attention_weights: np.ndarray,
                                emotional_patterns: np.ndarray,
                                V: np.ndarray) -> np.ndarray:
        """
        Convert attention patterns to explicit emotional field effects.
        
        FIELD CONVERSION (README.md Section 3.1.3.3.1):
        Transform attention-based emotional detection into dynamic field modulation:
        ℰ_modulation(x) = Σᵢ attention_weight[i] · field_influence_function(x, emotional_anchor[i])
        """
        embedding_dim = len(emotional_patterns)
        field_effects = np.zeros(embedding_dim)
        
        # Convert weighted attention transport to field modulation
        for i in range(embedding_dim):
            field_contribution = 0.0
            
            for head in range(self.attention_heads):
                # Weighted transport from attention mechanism
                attention_transport = attention_weights[head, i] * V[head, i]
                
                # Convert to field effect based on emotional pattern strength
                if emotional_patterns[i] > 0:
                    field_effect = attention_transport * emotional_patterns[i]
                    field_contribution += field_effect
            
            field_effects[i] = field_contribution / self.attention_heads
        
        return field_effects
    
    def extract_emotional_resonance_from_attention(self,
                                                 semantic_embedding: np.ndarray,
                                                 token: str,
                                                 coupling_mean: float) -> np.ndarray:
        """
        Extract emotional resonance pattern using attention deconstruction principles.
        
        This method provides the interface needed by the trajectory evolution module
        by applying attention deconstruction to extract emotional geometric patterns.
        
        Args:
            semantic_embedding: Base semantic vector
            token: Token identifier
            coupling_mean: Coupling strength from manifold analysis
            
        Returns:
            Emotional resonance pattern for trajectory computation
        """
        # Perform attention analysis
        analysis_result = self.analyze_emotional_patterns(
            semantic_embedding=semantic_embedding,
            token=token
        )
        
        # Extract resonance pattern from field effects
        emotional_resonance = analysis_result.field_effects
        
        # Modulate by coupling strength
        coupling_modulation = abs(coupling_mean)
        emotional_resonance *= coupling_modulation
        
        # Ensure pattern matches embedding dimension
        if len(emotional_resonance) != len(semantic_embedding):
            # Resize if necessary
            emotional_resonance = np.interp(
                np.linspace(0, 1, len(semantic_embedding)),
                np.linspace(0, 1, len(emotional_resonance)),
                emotional_resonance
            )
        
        logger.debug(f"Extracted emotional resonance for {token}: magnitude={np.linalg.norm(emotional_resonance):.4f}")
        return emotional_resonance


def create_attention_analyzer(embedding_dimension: int = 1024,
                           emotional_sensitivity: float = 0.3) -> AttentionGeometryAnalyzer:
    """
    Convenience function to create attention geometry analyzer.
    
    Args:
        embedding_dimension: Dimension of semantic embeddings
        emotional_sensitivity: Sensitivity threshold for emotional pattern detection
        
    Returns:
        Configured AttentionGeometryAnalyzer
    """
    return AttentionGeometryAnalyzer(
        embedding_dimension=embedding_dimension,
        attention_heads=16,
        emotional_threshold=emotional_sensitivity
    )