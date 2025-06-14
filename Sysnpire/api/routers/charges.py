"""
Charges Router - FastAPI endpoints for conceptual charge generation

This module provides REST API endpoints for converting text and embeddings
into conceptual charges using the complete Q(τ, C, s) formula. Supports
field-theoretic charge generation with trajectory integration and field effects.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging
import uuid
import numpy as np
from datetime import datetime

from ..models import (
    ConceptualChargeRequest,
    ConceptualChargeResponse,
    BatchChargeResponse,
    TextInput,
    BatchTextInput,
    UniverseStats,
    ErrorResponse
)

from ...model.intial import BGEIngestion, MPNetIngestion
from ...model.charge_factory import ChargeFactory
from ...database.field_universe import FieldUniverse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

bge_ingestion = BGEIngestion()
mpnet_ingestion = MPNetIngestion()
charge_factory = ChargeFactory()
field_universe = FieldUniverse()


def charge_to_response(charge, charge_id: str = None) -> ConceptualChargeResponse:
    """Convert conceptual charge object to API response model."""
    if charge_id is None:
        charge_id = f"charge_{uuid.uuid4().hex[:12]}"
    
    if hasattr(charge, 'complete_charge'):
        complete_charge_value = charge.complete_charge
        if isinstance(complete_charge_value, complex):
            complete_charge_dict = {
                "magnitude": float(abs(complete_charge_value)),
                "phase": float(np.angle(complete_charge_value)),
                "real": float(complete_charge_value.real),
                "imaginary": float(complete_charge_value.imag)
            }
        else:
            complete_charge_dict = {
                "magnitude": float(abs(complete_charge_value)),
                "phase": 0.0,
                "real": float(complete_charge_value),
                "imaginary": 0.0
            }
    else:
        complete_charge_dict = {
            "magnitude": 1.0,
            "phase": 0.0,
            "real": 1.0,
            "imaginary": 0.0
        }
    
    return ConceptualChargeResponse(
        charge_id=charge_id,
        text=getattr(charge, 'text', 'Unknown text'),
        complete_charge=complete_charge_dict,
        field_position=getattr(charge, 'field_position', [0.0, 0.0, 0.0]),
        semantic_field=getattr(charge, 'semantic_field', [])[:10],
        emotional_trajectory=getattr(charge, 'emotional_trajectory', [])[:10],
        phase_total=getattr(charge, 'phase_total', 0.0),
        observational_state=getattr(charge, 'observational_state', 1.0),
        gamma=getattr(charge, 'gamma', 1.0),
        metadata=getattr(charge, 'metadata', {})
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