"""
Embeddings Router - FastAPI endpoints for semantic embedding generation

This module provides REST API endpoints for converting text into semantic
embeddings using BGE and MPNet models. Supports both single and batch processing
with comprehensive validation and response models.
"""

from fastapi import APIRouter, HTTPException
from typing import List
import logging
import uuid
from datetime import datetime

from ..models import (
    EmbeddingRequest, 
    SemanticEmbeddingResponse, 
    BatchEmbeddingResponse,
    TextInput,
    BatchTextInput,
    ModelComparisonRequest,
    ModelComparisonResponse,
    ErrorResponse
)

from ...model.intial.bge_ingestion import BGEIngestion
from ...model.intial.mpnet_ingestion import MPNetIngestion
from ..models import SemanticEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

bge_ingestion = BGEIngestion()
mpnet_ingestion = MPNetIngestion()


def embedding_to_response(embedding: SemanticEmbedding, model_type: str) -> SemanticEmbeddingResponse:
    """Convert SemanticEmbedding to API response model."""
    return SemanticEmbeddingResponse(
        text_id=embedding.text_id,
        text=embedding.text,
        vector=embedding.vector.tolist(),
        dimension=embedding.dimension,
        magnitude=embedding.magnitude,
        model_type=model_type,
        metadata=embedding.metadata or {}
    )


@router.post("/generate", 
             response_model=SemanticEmbeddingResponse,
             summary="Generate single semantic embedding",
             description="""
             Convert a single text input into a semantic embedding using the specified model.
             
             Supports both BGE (1024D) and MPNet (768D) models for different use cases:
             - BGE: General-purpose semantic embeddings with broad coverage
             - MPNet: Social-focused embeddings optimized for social construct analysis
             """)
async def generate_embedding(request: EmbeddingRequest):
    """
    Generate a semantic embedding for a single text input.
    
    This endpoint processes individual text strings through the selected
    embedding model and returns a complete semantic embedding with
    vector components, metadata, and computed properties.
    """
    try:
        if not isinstance(request.input_data, TextInput):
            raise HTTPException(
                status_code=400, 
                detail="Single text input required for this endpoint"
            )
        
        text_input = request.input_data
        
        if request.model_type == "bge":
            embedding = bge_ingestion.ingest_text(
                text=text_input.text,
                text_id=text_input.text_id,
                metadata=text_input.metadata
            )
        elif request.model_type == "mpnet":
            embedding = mpnet_ingestion.ingest_text(
                text=text_input.text,
                text_id=text_input.text_id,
                metadata=text_input.metadata
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model type: {request.model_type}"
            )
        
        logger.info(f"Generated {request.model_type} embedding for text: {text_input.text[:50]}...")
        
        return embedding_to_response(embedding, request.model_type)
        
    except ValueError as e:
        logger.error(f"Validation error in embedding generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during embedding generation")


@router.post("/batch", 
             response_model=BatchEmbeddingResponse,
             summary="Generate batch semantic embeddings",
             description="""
             Convert multiple text inputs into semantic embeddings efficiently.
             
             Processes up to 1000 texts in a single request with optimized batch processing.
             Maintains individual error handling while providing batch-level statistics.
             """)
async def generate_batch_embeddings(request: EmbeddingRequest):
    """
    Generate semantic embeddings for multiple text inputs in batch.
    
    This endpoint efficiently processes multiple texts through the selected
    embedding model, providing comprehensive batch statistics and individual
    embedding results.
    """
    try:
        if not isinstance(request.input_data, BatchTextInput):
            raise HTTPException(
                status_code=400,
                detail="Batch text input required for this endpoint"
            )
        
        batch_input = request.input_data
        
        start_time = datetime.utcnow()
        
        if request.model_type == "bge":
            embeddings = bge_ingestion.ingest_batch(
                texts=batch_input.texts,
                text_ids=batch_input.text_ids,
                metadata=batch_input.metadata
            )
        elif request.model_type == "mpnet":
            embeddings = mpnet_ingestion.ingest_batch(
                texts=batch_input.texts,
                text_ids=batch_input.text_ids,
                metadata=batch_input.metadata
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model type: {request.model_type}"
            )
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        embedding_responses = [
            embedding_to_response(emb, request.model_type) 
            for emb in embeddings
        ]
        
        successful_embeddings = len([emb for emb in embeddings if not emb.metadata.get('error')])
        failed_embeddings = len(embeddings) - successful_embeddings
        
        processing_stats = {
            "processing_time_seconds": processing_time,
            "throughput_embeddings_per_second": len(embeddings) / max(processing_time, 0.001),
            "successful_embeddings": successful_embeddings,
            "failed_embeddings": failed_embeddings,
            "average_magnitude": sum(emb.magnitude for emb in embeddings) / len(embeddings),
            "total_dimensions": embeddings[0].dimension if embeddings else 0
        }
        
        logger.info(f"Generated batch of {len(embeddings)} {request.model_type} embeddings "
                   f"in {processing_time:.2f}s")
        
        return BatchEmbeddingResponse(
            embeddings=embedding_responses,
            batch_size=len(embeddings),
            model_type=request.model_type,
            processing_stats=processing_stats
        )
        
    except ValueError as e:
        logger.error(f"Validation error in batch embedding generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during batch embedding generation")


@router.post("/compare", 
             response_model=ModelComparisonResponse,
             summary="Compare embeddings across models",
             description="""
             Generate embeddings for the same text using multiple models and compare results.
             
             Provides detailed comparison metrics including similarity scores, magnitude differences,
             and model-specific characteristics for informed model selection.
             """)
async def compare_models(request: ModelComparisonRequest):
    """
    Compare embedding models on the same text input.
    
    This endpoint generates embeddings using multiple models for the same text
    and provides comprehensive comparison analysis to help with model selection
    for specific use cases.
    """
    try:
        model_results = {}
        
        for model_type in request.models:
            if model_type == "bge":
                embedding = bge_ingestion.ingest_text(text=request.text)
                model_results[model_type] = embedding_to_response(embedding, "bge")
            elif model_type == "mpnet":
                embedding = mpnet_ingestion.ingest_text(text=request.text)
                model_results[model_type] = embedding_to_response(embedding, "mpnet")
            else:
                logger.warning(f"Skipping unsupported model type: {model_type}")
        
        if len(model_results) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least two valid models required for comparison"
            )
        
        comparison_metrics = {}
        
        if "similarity" in request.comparison_metrics:
            if "bge" in model_results and "mpnet" in model_results:
                bge_emb = bge_ingestion.ingest_text(text=request.text)
                mpnet_emb = mpnet_ingestion.ingest_text(text=request.text)
                
                min_dim = min(bge_emb.dimension, mpnet_emb.dimension)
                bge_truncated = bge_emb.vector[:min_dim]
                mpnet_truncated = mpnet_emb.vector[:min_dim]
                
                import numpy as np
                similarity = float(np.dot(bge_truncated, mpnet_truncated) / 
                                 (np.linalg.norm(bge_truncated) * np.linalg.norm(mpnet_truncated)))
                comparison_metrics["cross_model_similarity"] = similarity
        
        if "magnitude" in request.comparison_metrics:
            magnitudes = {model: result.magnitude for model, result in model_results.items()}
            comparison_metrics["magnitude_comparison"] = magnitudes
            comparison_metrics["magnitude_ratio"] = (
                magnitudes.get("bge", 1.0) / magnitudes.get("mpnet", 1.0) 
                if "bge" in magnitudes and "mpnet" in magnitudes else 1.0
            )
        
        if "social_score" in request.comparison_metrics:
            social_scores = {}
            for model, result in model_results.items():
                social_score = result.metadata.get("social_score", 0.0)
                social_scores[model] = social_score
            comparison_metrics["social_score_comparison"] = social_scores
        
        recommendations = {}
        if "bge" in model_results and "mpnet" in model_results:
            bge_magnitude = model_results["bge"].magnitude
            mpnet_magnitude = model_results["mpnet"].magnitude
            mpnet_social = model_results["mpnet"].metadata.get("social_score", 0.0)
            
            if mpnet_social > 0.1:
                recommendations["best_for_social_analysis"] = "mpnet"
                recommendations["reasoning"] = f"High social score ({mpnet_social:.3f}) detected"
            elif bge_magnitude > mpnet_magnitude * 1.2:
                recommendations["best_for_general_semantic"] = "bge"
                recommendations["reasoning"] = f"Higher semantic magnitude ({bge_magnitude:.3f})"
            else:
                recommendations["balanced_choice"] = "mpnet"
                recommendations["reasoning"] = "Balanced performance for social content"
        
        logger.info(f"Completed model comparison for text: {request.text[:50]}...")
        
        return ModelComparisonResponse(
            text=request.text,
            model_results=model_results,
            comparison_metrics=comparison_metrics,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error in model comparison: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during model comparison")


@router.get("/models", 
           summary="List available embedding models",
           description="Get information about available embedding models and their characteristics.")
async def list_models():
    """
    List available embedding models and their characteristics.
    
    Returns information about supported embedding models including
    dimensions, capabilities, and recommended use cases.
    """
    return {
        "available_models": {
            "bge": {
                "name": "BGE-Large-v1.5",
                "dimensions": 1024,
                "description": "General-purpose semantic embeddings with broad semantic coverage",
                "best_for": ["general_text", "technical_content", "broad_semantic_analysis"],
                "performance": "High accuracy across diverse text types"
            },
            "mpnet": {
                "name": "all-mpnet-base-v2", 
                "dimensions": 768,
                "description": "Social-focused embeddings optimized for social construct analysis",
                "best_for": ["social_content", "community_analysis", "cultural_text", "identity_formation"],
                "performance": "Optimized for social universe construction and relationship modeling"
            }
        },
        "default_model": "bge",
        "comparison_available": True
    }


@router.get("/stats",
           summary="Get embedding service statistics",
           description="Get statistics about embedding generation performance and cache status.")
async def get_embedding_stats():
    """
    Get statistics about embedding service performance.
    
    Returns information about cache performance, processing statistics,
    and system status for monitoring purposes.
    """
    bge_stats = bge_ingestion.get_cache_stats()
    mpnet_stats = mpnet_ingestion.get_social_analysis_stats()
    
    return {
        "bge_stats": bge_stats,
        "mpnet_stats": mpnet_stats,
        "service_status": "operational",
        "supported_operations": ["single", "batch", "comparison"]
    }