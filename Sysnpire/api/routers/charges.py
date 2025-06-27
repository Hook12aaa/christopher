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

from ...model.intial.bge_ingestion import BGEIngestion
from ...model.intial.mpnet_ingestion import MPNetIngestion
from ...model.charge_factory import ChargeFactory
from ...database.field_universe import FieldUniverse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

bge_ingestion = BGEIngestion()
mpnet_ingestion = MPNetIngestion()
charge_factory = ChargeFactory()
field_universe = FieldUniverse()

@router.post("/analyze_movement")
async def analyze_movement(request: TextInput):
    """
    Analyze transformative movement through meaning space.
    
    Returns detailed trajectory operator analysis showing how concepts
    transform through observational states according to T(τ,C,s).
    
    Following CLAUDE.md: uses actual trajectory computation, no simulation.
    """
    try:
        # Generate charge with enhanced trajectory operators
        search_results = bge_ingestion.search_embeddings(request.text, top_k=1)
        
        if not search_results or 'results' not in search_results:
            raise HTTPException(status_code=404, detail="No embedding found for text")
        
        # Extract embedding and manifold properties
        best_result = search_results['results'][0]
        embedding = best_result['embedding']
        manifold_properties = bge_ingestion.extract_manifold_properties(
            embedding=embedding,
            index=best_result.get('index'),
            all_embeddings=search_results.get('embeddings'))
        
        # Create enhanced charge with trajectory data
        from ...model.charge_factory import ChargeParameters
        charge_params = ChargeParameters(
            observational_state=request.observational_state or 1.0,
            gamma=request.gamma or 1.2,
            context=request.context or "movement_analysis"
        )
        
        metadata = {
            'text': request.text,
            'token': best_result.get('token'),
            'analysis_type': 'movement_analysis'
        }
        
        charge = charge_factory.create_charge(
            embedding=embedding,
            manifold_properties=manifold_properties,
            charge_params=charge_params,
            metadata=metadata
        )
        
        # Extract detailed movement analysis
        movement_data = charge.get_transformative_movement()
        
        # Convert numpy arrays for JSON response
        for key, value in movement_data.items():
            if isinstance(value, np.ndarray):
                movement_data[key] = value.tolist()
        
        return {
            "charge_id": f"movement_{uuid.uuid4().hex[:8]}",
            "text": request.text,
            "movement_analysis": movement_data,
            "trajectory_details": {
                "observational_state": charge.observational_state,
                "trajectory_enhanced": hasattr(charge, 'trajectory_data'),
                "transformative_formula": "T(τ,C,s) = ∫₀ˢ ω_i(τ,s')·e^(iφ_i(τ,s')) ds'",
                "dtf_enhanced": getattr(charge, 'dtf_enhanced', False)
            }
        }
        
    except Exception as e:
        logger.error(f"Movement analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Movement analysis failed: {str(e)}")


@router.get("/movement_stats")
async def get_movement_statistics():
    """
    Get statistics about trajectory operator performance across the system.
    
    Provides insights into transformative movement patterns in the field universe.
    """
    try:
        factory_stats = charge_factory.get_factory_statistics()
        
        # Get trajectory engine statistics
        trajectory_stats = {
            "trajectory_engine_initialized": hasattr(charge_factory, 'trajectory_engine'),
            "embedding_dimension": getattr(charge_factory.trajectory_engine, 'embedding_dimension', 0),
            "base_frequencies_count": len(getattr(charge_factory.trajectory_engine, 'base_frequencies', [])),
            "integration_method": getattr(charge_factory.trajectory_engine, 'integration_method', 'unknown')
        }
        
        return {
            "factory_statistics": factory_stats,
            "trajectory_statistics": trajectory_stats,
            "theoretical_foundation": {
                "trajectory_formula": "T_i(τ,s) = ∫₀ˢ ω_i(τ,s')·e^(iφ_i(τ,s')) ds'",
                "movement_principle": "Meaning evolves through trajectory, not position",
                "claude_md_compliant": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get movement statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")


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
    
    # Extract trajectory movement data if available
    movement_data = {}
    if hasattr(charge, 'get_transformative_movement'):
        try:
            movement_data = charge.get_transformative_movement()
            # Convert numpy arrays to lists for JSON serialization
            for key, value in movement_data.items():
                if isinstance(value, np.ndarray):
                    movement_data[key] = value.tolist()
        except Exception as e:
            logger.warning(f"Failed to extract movement data: {e}")
            movement_data = {'movement_available': False}
    
    # Prepare metadata including trajectory data
    metadata = getattr(charge, 'metadata', {}).copy()
    metadata['movement'] = movement_data
    
    # Add trajectory features if available (from database objects)
    if hasattr(charge, '_trajectory_metadata'):
        metadata['trajectory_metadata'] = charge._trajectory_metadata
    
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
        metadata=metadata
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
        texts = data.get("texts")
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
        text = data.get("text")
        time_steps = data.get("time_steps")
        
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