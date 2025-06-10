"""
Router for conceptual charge-related endpoints.
"""
from quart import Blueprint, request, jsonify
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from api.main import require_api_key
from embedding_engine.models import ConceptualChargeGenerator
import numpy as np

# Create blueprint for charges routes
blueprint = Blueprint("charges", __name__)

# Initialize the charge generator
charge_generator = ConceptualChargeGenerator()

# Dataclass definitions for request/response objects
@dataclass
class ChargeRequest:
    text: str
    context: Optional[Dict] = None
    observational_state: float = 0.0
    gamma: float = 1.0
    
    @classmethod
    async def from_request(cls):
        data = await request.get_json()
        return cls(
            text=data.get("text", ""),
            context=data.get("context"),
            observational_state=data.get("observational_state", 0.0),
            gamma=data.get("gamma", 1.0)
        )

@blueprint.route("/generate", methods=["POST"])
@require_api_key
async def generate_charge():
    """
    Generate a complete conceptual charge Q(τ, C, s) from the provided text.

    Request body:
    {
        "text": "string",
        "context": {
            "semantic_context": "optional semantic context",
            "emotional_intensity": float,  # optional emotional field strength
            "social_context": "optional social environment description"
        },
        "observational_state": float,  # current observational state s
        "gamma": float  # global field calibration factor
    }
    """
    try:
        req = await ChargeRequest.from_request()
        
        # Generate complete conceptual charge using field-theoretic formulation
        charge = charge_generator.create_conceptual_charge(
            text=req.text,
            context=req.context,
            observational_state=req.observational_state,
            gamma=req.gamma
        )
        
        # Compute complete charge components
        complete_charge = charge.compute_complete_charge()
        
        return jsonify({
            "charge_id": f"ch_{hash(req.text) % 1000000}",
            "text": req.text,
            "token": charge.token,
            "formulation": "Q(τ, C, s) = γ · T(τ, C, s) · E^trajectory(τ, s) · Φ^semantic(τ, s) · e^(iθ_total(τ,C,s)) · Ψ_persistence(s-s₀)",
            "components": {
                "gamma": charge.gamma,
                "context": charge.context,
                "observational_state": charge.observational_state,
                "semantic_vector_preview": charge.semantic_vector[:10].tolist(),  # First 10 dims
                "field_parameters": {
                    "omega_base_preview": charge.omega_base[:5].tolist(),
                    "emotional_alignment_preview": charge.v_emotional[:5].tolist(),
                    "breathing_depth_preview": charge.beta_breathing[:5].tolist()
                }
            },
            "charge_value": {
                "magnitude": abs(complete_charge),
                "phase": np.angle(complete_charge),
                "real_part": complete_charge.real,
                "imaginary_part": complete_charge.imag
            },
            "field_effects": {
                "trajectory_magnitude": np.mean([abs(charge.trajectory_operator(req.observational_state, i)) for i in range(3)]),
                "emotional_trajectory_mean": np.mean(charge.emotional_trajectory_integration(req.observational_state)),
                "semantic_field_mean": np.mean(np.abs(charge.semantic_field_generation(req.observational_state))),
                "observational_persistence": charge.observational_persistence(req.observational_state)
            }
        })
    except Exception as e:
        return jsonify({"error": f"Charge generation failed: {str(e)}"}), 500

@blueprint.route("/batch", methods=["POST"])
@require_api_key
async def generate_batch_charges():
    """
    Generate multiple conceptual charges in batch.
    
    Request body:
    {
        "texts": ["text1", "text2", ...],
        "emotional_contexts": [{...}, {...}, ...],  # Optional
        "temporal_contexts": [{...}, {...}, ...]    # Optional
    }
    """
    try:
        data = await request.get_json()
        texts = data.get("texts", [])
        emotional_contexts = data.get("emotional_contexts")
        temporal_contexts = data.get("temporal_contexts")
        
        # Generate batch charges
        charges = charge_generator.create_batch_charges(
            texts=texts,
            emotional_contexts=emotional_contexts,
            temporal_contexts=temporal_contexts
        )
        
        results = []
        for i, charge in enumerate(charges):
            results.append({
                "charge_id": f"ch_{hash(texts[i]) % 1000000}",
                "text": texts[i],
                "magnitude": charge.get_charge_magnitude(),
                "phase_factor": charge.get_phase_factor()
            })
        
        return jsonify({
            "charges": results,
            "count": len(results)
        })
    except Exception as e:
        return jsonify({"error": f"Batch charge generation failed: {str(e)}"}), 500

@blueprint.route("/analyze", methods=["POST"])
@require_api_key
async def analyze_charge():
    """
    Analyze properties of a conceptual charge.
    
    Request body:
    {
        "text": "string",
        "time_steps": [0.0, 0.5, 1.0, 2.0]  # Optional time steps for decay analysis
    }
    """
    try:
        data = await request.get_json()
        text = data.get("text", "")
        time_steps = data.get("time_steps", [0.0, 1.0, 2.0, 5.0, 10.0])
        
        # Generate charge for analysis
        charge = charge_generator.create_conceptual_charge(text=text)
        
        # Calculate decay over time
        decay_values = [charge.temporal_decay(t) for t in time_steps]
        
        return jsonify({
            "text": text,
            "analysis": {
                "magnitude": charge.get_charge_magnitude(),
                "phase_factor": charge.get_phase_factor(),
                "temporal_decay": {
                    "time_steps": time_steps,
                    "decay_values": decay_values
                },
                "vorticity_magnitude": float(np.linalg.norm(charge.vorticity))
            }
        })
    except Exception as e:
        return jsonify({"error": f"Charge analysis failed: {str(e)}"}), 500