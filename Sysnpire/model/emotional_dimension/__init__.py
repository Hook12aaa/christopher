"""
Emotional Dimension - Field Modulation (Section 3.1.3)

Reconceptualizes emotion as field-modulating forces that transform semantic landscapes
through amplitude modulation, phase shifts, and metric warping.

Components:
- main.py: Primary interface for E^trajectory(τ, s) processing
- trajectory_evolution.py: Core emotional trajectory integration
- attention_deconstruction.py: Geometric analysis of attention mechanisms
- field_modulation.py: Metric tensor modulation and geometric distortions
- interference_patterns.py: Multi-emotion interference and phase coordination
- resonance_amplification.py: Frequency domain resonance detection and amplification
- context_coupling.py: Context-dependent emotional evolution and contagion
"""

from .main import (
    compute_emotional_trajectory,
    analyze_emotional_attention_patterns,
    compute_batch_emotional_trajectories,
    get_emotional_dimension_info
)
from .trajectory_evolution import EmotionalTrajectoryIntegrator, EmotionalTrajectoryParams
from .attention_deconstruction import AttentionGeometryAnalyzer
from .field_modulation import (
    EmotionalFieldModulator,
    MetricWarping,
    FieldEffectCalculator,
    GeometricDistortionManager
)
from .interference_patterns import (
    EmotionalInterferenceManager,
    PhaseCoordinator,
    ResonanceDetector,
    compute_multi_emotion_interference
)
from .resonance_amplification import (
    ResonanceCalculator,
    AmplificationEngine,
    FrequencyMatcher,
    ResonanceMemoryManager,
    apply_resonance_enhancement
)
from .context_coupling import (
    ContextualEmotionalModulator,
    EmotionalPrimingManager,
    EmotionalContagionSimulator,
    ContextEmotionalAnalyzer,
    apply_context_modulation
)

__all__ = [
    # Main interface
    'compute_emotional_trajectory',
    'analyze_emotional_attention_patterns', 
    'compute_batch_emotional_trajectories',
    'get_emotional_dimension_info',
    
    # Core trajectory evolution
    'EmotionalTrajectoryIntegrator',
    'EmotionalTrajectoryParams',
    
    # Attention deconstruction
    'AttentionGeometryAnalyzer',
    
    # Field modulation
    'EmotionalFieldModulator',
    'MetricWarping',
    'FieldEffectCalculator', 
    'GeometricDistortionManager',
    
    # Interference patterns
    'EmotionalInterferenceManager',
    'PhaseCoordinator',
    'ResonanceDetector',
    'compute_multi_emotion_interference',
    
    # Resonance amplification
    'ResonanceCalculator',
    'AmplificationEngine',
    'FrequencyMatcher',
    'ResonanceMemoryManager',
    'apply_resonance_enhancement',
    
    # Context coupling
    'ContextualEmotionalModulator',
    'EmotionalPrimingManager',
    'EmotionalContagionSimulator',
    'ContextEmotionalAnalyzer',
    'apply_context_modulation'
]