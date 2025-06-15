"""
Context Coupling - Context-Dependent Emotional Evolution

MATHEMATICAL FOUNDATION (README.md Section 3.1.3.3.7):
E^trajectory(τ, s | C) = E_base^trajectory(τ, s) · context_modulation(C, s)

CONTEXT EFFECTS:
1. Emotional Priming: Context sets emotional baseline expectations
2. Emotional Contagion: Context spreads emotional states across tokens
3. Emotional Contrast: Context can invert emotional polarity

This module implements context-dependent emotional field modulation
where the same semantic content can have different emotional responses
based on contextual priming and environmental influences.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContextualEmotionalState:
    """Contextual emotional state representation."""
    context_id: str
    emotional_baseline: complex
    priming_strength: float
    contagion_radius: float
    contrast_polarity: float
    decay_rate: float
    activation_timestamp: float


@dataclass
class EmotionalContagionEffect:
    """Emotional contagion effect between contexts."""
    source_context: str
    target_context: str
    contagion_strength: float
    transmission_vector: np.ndarray
    emotional_charge: complex
    distance_decay: float


class ContextualEmotionalModulator:
    """
    Modulate emotions based on contextual priming and environmental effects.
    
    MATHEMATICAL FOUNDATION:
    Implements context-dependent emotional evolution where emotional responses
    are modulated by contextual baseline, priming effects, and environmental
    emotional contagion patterns.
    
    CONTEXT MODULATION:
    E^trajectory(τ, s | C) = E_base^trajectory(τ, s) · [α_C + β_C·priming(C) + γ_C·contagion(C,s)]
    """
    
    def __init__(self,
                 context_memory_length: float = 10.0,
                 priming_decay_rate: float = 0.1,
                 contagion_radius: float = 5.0):
        """
        Initialize contextual emotional modulator.
        
        Args:
            context_memory_length: Length of contextual memory in observational units
            priming_decay_rate: Rate of emotional priming decay
            contagion_radius: Radius for emotional contagion effects
        """
        self.context_memory_length = context_memory_length
        self.priming_decay_rate = priming_decay_rate
        self.contagion_radius = contagion_radius
        
        # Context state tracking
        self.contextual_states = {}
        self.contagion_effects = []
        
        logger.info(f"Initialized ContextualEmotionalModulator: memory={context_memory_length}, contagion_radius={contagion_radius}")
    
    def modulate_emotion(self,
                        base_emotion: complex,
                        token: str,
                        context: str,
                        observational_state: float,
                        semantic_embedding: Optional[np.ndarray] = None) -> Tuple[complex, Dict[str, Any]]:
        """
        Apply contextual modulation to base emotional trajectory.
        
        MODULATION PROCESS (README.md Section 3.1.3.3.7):
        1. Retrieve or create contextual emotional state
        2. Apply emotional priming based on context baseline
        3. Apply emotional contagion from nearby contexts
        4. Apply emotional contrast effects if applicable
        5. Update contextual state for future interactions
        
        Args:
            base_emotion: Base emotional trajectory E_base^trajectory(τ, s)
            token: Token identifier for modulation tracking
            context: Context identifier C
            observational_state: Current observational state s
            semantic_embedding: Optional semantic embedding for context analysis
            
        Returns:
            Tuple of (modulated_emotion, modulation_analysis)
        """
        try:
            # Get or create contextual state
            contextual_state = self._get_contextual_state(context, observational_state)
            
            # Apply emotional priming
            priming_modulation = self._apply_emotional_priming(
                base_emotion, contextual_state, observational_state
            )
            
            # Apply emotional contagion
            contagion_modulation = self._apply_emotional_contagion(
                priming_modulation, context, observational_state
            )
            
            # Apply emotional contrast if applicable
            contrast_modulation = self._apply_emotional_contrast(
                contagion_modulation, contextual_state, observational_state
            )
            
            # Update contextual state
            self._update_contextual_state(
                context, contrast_modulation, observational_state
            )
            
            # Create modulation analysis
            modulation_analysis = {
                'base_emotion_magnitude': abs(base_emotion),
                'priming_effect': abs(priming_modulation) / abs(base_emotion) if abs(base_emotion) > 0 else 1.0,
                'contagion_effect': abs(contagion_modulation) / abs(priming_modulation) if abs(priming_modulation) > 0 else 1.0,
                'contrast_effect': abs(contrast_modulation) / abs(contagion_modulation) if abs(contagion_modulation) > 0 else 1.0,
                'total_modulation': abs(contrast_modulation) / abs(base_emotion) if abs(base_emotion) > 0 else 1.0,
                'contextual_baseline': contextual_state.emotional_baseline,
                'priming_strength': contextual_state.priming_strength,
                'context_id': context,
                'processing_stage': 'context_modulated'
            }
            
            logger.debug(f"Context modulation for {token} in {context}: factor={modulation_analysis['total_modulation']:.3f}")
            return contrast_modulation, modulation_analysis
            
        except Exception as e:
            logger.error(f"Context modulation failed for {token} in {context}: {e}")
            return base_emotion, {'error': str(e), 'processing_stage': 'failed'}
    
    def _get_contextual_state(self, 
                            context: str, 
                            observational_state: float) -> ContextualEmotionalState:
        """
        Retrieve or create contextual emotional state.
        
        CONTEXTUAL STATE MANAGEMENT:
        Maintains emotional baselines and priming effects for each context
        with proper decay and state evolution over observational time.
        """
        if context in self.contextual_states:
            state = self.contextual_states[context]
            
            # Apply decay to priming strength
            time_elapsed = observational_state - state.activation_timestamp
            decay_factor = np.exp(-self.priming_decay_rate * time_elapsed)
            state.priming_strength *= decay_factor
            state.activation_timestamp = observational_state
            
            return state
        else:
            # Create new contextual state
            # Context-specific emotional baseline
            context_hash = hash(context) % 1000 / 1000.0
            baseline_magnitude = 0.5 + 0.5 * context_hash
            baseline_phase = 2 * np.pi * context_hash
            emotional_baseline = baseline_magnitude * np.exp(1j * baseline_phase)
            
            state = ContextualEmotionalState(
                context_id=context,
                emotional_baseline=emotional_baseline,
                priming_strength=1.0,
                contagion_radius=self.contagion_radius,
                contrast_polarity=1.0 if context_hash > 0.5 else -1.0,
                decay_rate=self.priming_decay_rate,
                activation_timestamp=observational_state
            )
            
            self.contextual_states[context] = state
            logger.debug(f"Created contextual state for {context}: baseline={abs(emotional_baseline):.3f}")
            return state
    
    def _apply_emotional_priming(self,
                               base_emotion: complex,
                               contextual_state: ContextualEmotionalState,
                               observational_state: float) -> complex:
        """
        Apply emotional priming based on contextual baseline.
        
        EMOTIONAL PRIMING (README.md Section 3.1.3.3.7):
        Context sets emotional baseline expectations that bias emotional responses
        toward contextually appropriate patterns.
        
        PRIMING FORMULA:
        E_primed = E_base · (1 + α_priming · E_baseline · priming_strength)
        """
        # Priming strength based on context state
        priming_alpha = 0.3  # Priming coupling strength
        
        # Emotional priming effect
        priming_effect = priming_alpha * contextual_state.emotional_baseline * contextual_state.priming_strength
        
        # Apply priming modulation
        primed_emotion = base_emotion * (1.0 + priming_effect)
        
        return primed_emotion
    
    def _apply_emotional_contagion(self,
                                 primed_emotion: complex,
                                 context: str,
                                 observational_state: float) -> complex:
        """
        Apply emotional contagion from neighboring contexts.
        
        EMOTIONAL CONTAGION (README.md Section 3.1.3.3.7):
        Context spreads emotional states across nearby contexts through
        contagion effects that diminish with distance and time.
        
        CONTAGION FORMULA:
        E_contagion = E_primed + Σᵢ β_contagion · E_neighbor[i] · exp(-d[i]/σ_contagion)
        """
        contagion_effect = complex(0, 0)
        contagion_beta = 0.2  # Contagion coupling strength
        
        # Find relevant contagion effects
        active_contagions = self._get_active_contagion_effects(context, observational_state)
        
        for contagion in active_contagions:
            # Distance-based decay
            distance_factor = np.exp(-contagion.distance_decay)
            
            # Emotional transmission
            transmission_strength = contagion.contagion_strength * distance_factor
            transmitted_emotion = contagion.emotional_charge * transmission_strength
            
            contagion_effect += contagion_beta * transmitted_emotion
        
        # Apply contagion
        contagious_emotion = primed_emotion + contagion_effect
        
        return contagious_emotion
    
    def _apply_emotional_contrast(self,
                                contagious_emotion: complex,
                                contextual_state: ContextualEmotionalState,
                                observational_state: float) -> complex:
        """
        Apply emotional contrast effects based on contextual polarity.
        
        EMOTIONAL CONTRAST (README.md Section 3.1.3.3.7):
        Context can invert emotional polarity for contrast effects,
        creating emotional irony, sarcasm, or dramatic emphasis.
        
        CONTRAST FORMULA:
        E_contrast = E_contagious · [1 + γ_contrast · contrast_polarity · contrast_strength]
        """
        # Context-dependent contrast strength
        contrast_gamma = 0.1  # Contrast coupling strength
        
        # Determine contrast strength based on context characteristics
        # Strong contrast for certain contexts (irony, sarcasm)
        context_hash = hash(contextual_state.context_id) % 1000 / 1000.0
        contrast_strength = 0.5 if context_hash > 0.8 else 0.0  # High contrast for 20% of contexts
        
        # Apply contrast modulation
        contrast_factor = 1.0 + contrast_gamma * contextual_state.contrast_polarity * contrast_strength
        contrasted_emotion = contagious_emotion * contrast_factor
        
        return contrasted_emotion
    
    def _get_active_contagion_effects(self,
                                    target_context: str,
                                    observational_state: float) -> List[EmotionalContagionEffect]:
        """
        Get active emotional contagion effects affecting target context.
        
        CONTAGION FILTERING:
        Returns contagion effects that are still active based on decay
        and within effective transmission range.
        """
        active_contagions = []
        
        for contagion in self.contagion_effects:
            if contagion.target_context == target_context:
                # Check if still active (simple time-based decay)
                # In a full implementation, this would use proper observational state tracking
                if contagion.contagion_strength > 0.1:  # Threshold for meaningful effect
                    active_contagions.append(contagion)
        
        return active_contagions
    
    def _update_contextual_state(self,
                               context: str,
                               final_emotion: complex,
                               observational_state: float):
        """
        Update contextual state based on final emotional outcome.
        
        STATE EVOLUTION:
        Contextual states evolve based on emotional experiences
        to create dynamic contextual memory and adaptation.
        """
        if context in self.contextual_states:
            state = self.contextual_states[context]
            
            # Adaptive baseline update
            adaptation_rate = 0.05
            state.emotional_baseline += adaptation_rate * (final_emotion - state.emotional_baseline)
            
            # Update timestamp
            state.activation_timestamp = observational_state


class EmotionalPrimingManager:
    """
    Handle emotional priming effects within and across contexts.
    
    PRIMING MECHANISM:
    Manages how contextual emotional baselines influence subsequent
    emotional responses through priming and expectation effects.
    """
    
    def __init__(self, 
                 priming_window: float = 5.0,
                 cross_context_priming: bool = True):
        """
        Initialize emotional priming manager.
        
        Args:
            priming_window: Time window for priming effects
            cross_context_priming: Whether to allow cross-context priming
        """
        self.priming_window = priming_window
        self.cross_context_priming = cross_context_priming
        self.priming_history = []
    
    def create_priming_effect(self,
                            source_emotion: complex,
                            source_context: str,
                            target_context: str,
                            observational_state: float,
                            priming_strength: float = 1.0) -> Dict[str, Any]:
        """
        Create emotional priming effect from source to target context.
        
        PRIMING CREATION:
        Establishes priming relationship where source emotional state
        influences target context emotional baseline and responses.
        
        Args:
            source_emotion: Source emotional state for priming
            source_context: Source context identifier
            target_context: Target context identifier  
            observational_state: Current observational state
            priming_strength: Strength of priming effect
            
        Returns:
            Dict containing priming effect details
        """
        # Create priming effect
        priming_effect = {
            'source_emotion': source_emotion,
            'source_context': source_context,
            'target_context': target_context,
            'priming_strength': priming_strength,
            'creation_state': observational_state,
            'decay_rate': 0.1,
            'priming_type': 'contextual'
        }
        
        # Add to priming history
        self.priming_history.append(priming_effect)
        
        # Clean old priming effects
        self._clean_expired_priming(observational_state)
        
        logger.debug(f"Created priming effect: {source_context} → {target_context}, strength={priming_strength:.3f}")
        return priming_effect
    
    def get_active_priming(self,
                         target_context: str,
                         observational_state: float) -> List[Dict[str, Any]]:
        """
        Get active priming effects for target context.
        
        PRIMING RETRIEVAL:
        Returns priming effects that are still active and relevant
        for the target context at current observational state.
        """
        active_priming = []
        
        for effect in self.priming_history:
            if effect['target_context'] == target_context:
                # Check if still active
                time_elapsed = observational_state - effect['creation_state']
                if time_elapsed <= self.priming_window:
                    # Apply decay
                    decay_factor = np.exp(-effect['decay_rate'] * time_elapsed)
                    if decay_factor > 0.1:  # Threshold for meaningful priming
                        effect['current_strength'] = effect['priming_strength'] * decay_factor
                        active_priming.append(effect)
        
        return active_priming
    
    def _clean_expired_priming(self, observational_state: float):
        """Remove expired priming effects from history."""
        self.priming_history = [
            effect for effect in self.priming_history
            if observational_state - effect['creation_state'] <= self.priming_window * 2
        ]


class EmotionalContagionSimulator:
    """
    Simulate emotional contagion through contextual space.
    
    CONTAGION SIMULATION:
    Models how emotional states spread through contextual networks
    based on proximity, similarity, and transmission dynamics.
    """
    
    def __init__(self,
                 contagion_rate: float = 0.1,
                 transmission_decay: float = 0.2,
                 max_contagion_distance: float = 3.0):
        """
        Initialize emotional contagion simulator.
        
        Args:
            contagion_rate: Rate of emotional transmission
            transmission_decay: Decay rate over distance
            max_contagion_distance: Maximum distance for contagion effects
        """
        self.contagion_rate = contagion_rate
        self.transmission_decay = transmission_decay
        self.max_contagion_distance = max_contagion_distance
        self.context_network = {}
    
    def simulate_contagion(self,
                         source_contexts: Dict[str, complex],
                         target_context: str,
                         observational_state: float) -> List[EmotionalContagionEffect]:
        """
        Simulate emotional contagion from source contexts to target.
        
        CONTAGION SIMULATION PROCESS:
        1. Calculate distances between contexts in semantic space
        2. Determine transmission strength based on distance and similarity
        3. Apply decay factors for temporal and spatial distance
        4. Create contagion effects for significant transmissions
        
        Args:
            source_contexts: Dict mapping context_id to emotional_state
            target_context: Target context for contagion reception
            observational_state: Current observational state
            
        Returns:
            List of active contagion effects
        """
        contagion_effects = []
        
        for source_context, source_emotion in source_contexts.items():
            if source_context != target_context:
                # Calculate contextual distance (simplified)
                distance = self._calculate_context_distance(source_context, target_context)
                
                if distance <= self.max_contagion_distance:
                    # Calculate transmission strength
                    transmission_strength = self._calculate_transmission_strength(
                        source_emotion, distance, observational_state
                    )
                    
                    if transmission_strength > 0.01:  # Threshold for meaningful contagion
                        # Create contagion effect
                        effect = EmotionalContagionEffect(
                            source_context=source_context,
                            target_context=target_context,
                            contagion_strength=transmission_strength,
                            transmission_vector=self._get_transmission_vector(source_context, target_context),
                            emotional_charge=source_emotion,
                            distance_decay=distance * self.transmission_decay
                        )
                        
                        contagion_effects.append(effect)
        
        logger.debug(f"Simulated contagion for {target_context}: {len(contagion_effects)} effects")
        return contagion_effects
    
    def _calculate_context_distance(self, context1: str, context2: str) -> float:
        """
        Calculate distance between contexts in semantic space.
        
        DISTANCE CALCULATION:
        Uses hash-based distance approximation for context similarity.
        In full implementation, would use semantic embeddings.
        """
        # Simple hash-based distance (placeholder)
        hash1 = hash(context1) % 1000
        hash2 = hash(context2) % 1000
        
        # Normalize to [0, 1] and scale
        distance = abs(hash1 - hash2) / 1000.0 * self.max_contagion_distance
        return distance
    
    def _calculate_transmission_strength(self,
                                       source_emotion: complex,
                                       distance: float,
                                       observational_state: float) -> float:
        """
        Calculate emotional transmission strength based on multiple factors.
        
        TRANSMISSION FORMULA:
        strength = |source_emotion| · contagion_rate · exp(-distance/decay) · time_factor
        """
        # Base strength from emotional magnitude
        base_strength = abs(source_emotion) * self.contagion_rate
        
        # Distance decay
        distance_factor = np.exp(-distance / self.transmission_decay)
        
        # Time factor (simplified - in full implementation would use proper temporal dynamics)
        time_factor = 1.0
        
        transmission_strength = base_strength * distance_factor * time_factor
        return transmission_strength
    
    def _get_transmission_vector(self, source_context: str, target_context: str) -> np.ndarray:
        """
        Get transmission vector for directional contagion effects.
        
        TRANSMISSION VECTOR:
        Represents directional flow of emotional contagion
        in contextual semantic space.
        """
        # Simple 2D vector based on context hashes (placeholder)
        source_hash = hash(source_context) % 1000
        target_hash = hash(target_context) % 1000
        
        vector = np.array([
            (target_hash % 100) / 100.0 - 0.5,
            ((target_hash // 100) % 10) / 10.0 - 0.5
        ])
        
        # Normalize
        vector = vector / (np.linalg.norm(vector) + 1e-10)
        return vector


class ContextEmotionalAnalyzer:
    """
    Analyze emotional context interactions and patterns.
    
    CONTEXT ANALYSIS:
    Provides analytical tools for understanding context-dependent
    emotional evolution patterns and interaction dynamics.
    """
    
    def __init__(self):
        """Initialize context emotional analyzer."""
        pass
    
    def analyze_context_effects(self,
                              base_emotions: Dict[str, complex],
                              modulated_emotions: Dict[str, complex],
                              contexts: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze context effects on emotional responses.
        
        CONTEXT EFFECT ANALYSIS:
        Compares base emotional responses with context-modulated responses
        to quantify contextual influence patterns.
        
        Args:
            base_emotions: Base emotional responses by token
            modulated_emotions: Context-modulated emotional responses
            contexts: Context identifiers by token
            
        Returns:
            Dict containing comprehensive context effect analysis
        """
        analysis = {
            'total_tokens': len(base_emotions),
            'context_effects': {},
            'overall_modulation': 0.0,
            'strongest_context_effect': None,
            'context_consistency': 0.0
        }
        
        modulation_factors = []
        context_effects = {}
        
        for token in base_emotions:
            if token in modulated_emotions:
                base_mag = abs(base_emotions[token])
                mod_mag = abs(modulated_emotions[token])
                
                if base_mag > 0:
                    modulation_factor = mod_mag / base_mag
                    modulation_factors.append(modulation_factor)
                    
                    context = contexts.get(token, 'unknown')
                    if context not in context_effects:
                        context_effects[context] = []
                    context_effects[context].append(modulation_factor)
        
        # Overall modulation
        if modulation_factors:
            analysis['overall_modulation'] = np.mean(modulation_factors)
        
        # Context-specific effects
        for context, factors in context_effects.items():
            context_analysis = {
                'mean_modulation': np.mean(factors),
                'std_modulation': np.std(factors),
                'token_count': len(factors),
                'consistency': 1.0 / (1.0 + np.std(factors))
            }
            analysis['context_effects'][context] = context_analysis
        
        # Strongest context effect
        if context_effects:
            strongest_context = max(
                context_effects.keys(),
                key=lambda c: abs(context_effects[c][0] - 1.0)  # Largest deviation from 1.0
            )
            analysis['strongest_context_effect'] = strongest_context
        
        return analysis


def apply_context_modulation(base_emotion: complex,
                           token: str,
                           context: str,
                           observational_state: float,
                           modulator: Optional[ContextualEmotionalModulator] = None) -> Tuple[complex, Dict[str, Any]]:
    """
    Convenience function to apply complete context modulation.
    
    Args:
        base_emotion: Base emotional trajectory
        token: Token identifier
        context: Context identifier
        observational_state: Current observational state
        modulator: Optional pre-configured modulator
        
    Returns:
        Tuple of (modulated_emotion, analysis)
    """
    if modulator is None:
        modulator = ContextualEmotionalModulator()
    
    return modulator.modulate_emotion(
        base_emotion=base_emotion,
        token=token,
        context=context,
        observational_state=observational_state
    )


def create_contextual_modulator(context_memory: float = 10.0,
                              contagion_radius: float = 5.0) -> ContextualEmotionalModulator:
    """
    Convenience function to create contextual emotional modulator.
    
    Args:
        context_memory: Length of contextual memory
        contagion_radius: Radius for emotional contagion
        
    Returns:
        Configured ContextualEmotionalModulator
    """
    return ContextualEmotionalModulator(
        context_memory_length=context_memory,
        priming_decay_rate=0.1,
        contagion_radius=contagion_radius
    )