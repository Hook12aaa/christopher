"""
API Models - Pydantic models for FastAPI request/response validation

This module defines all data models used in the Sysnpire API for
request validation, response serialization, and automatic API documentation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import numpy as np


class ModelType(str, Enum):
    """Supported embedding model types."""
    BGE = "bge"
    MPNET = "mpnet"


class TextInput(BaseModel):
    """Single text input for embedding generation."""
    text: str = Field(..., min_length=1, max_length=10000, 
                      description="Input text to convert to embedding")
    text_id: Optional[str] = Field(None, description="Optional unique identifier for the text")
    metadata: Optional[Dict[str, Any]] = Field(None, 
                                              description="Optional metadata dictionary")


class BatchTextInput(BaseModel):
    """Batch text input for multiple embeddings."""
    texts: List[str] = Field(..., min_items=1, max_items=1000,
                            description="List of texts to convert to embeddings")
    text_ids: Optional[List[str]] = Field(None, 
                                         description="Optional list of unique identifiers")
    metadata: Optional[List[Dict[str, Any]]] = Field(None,
                                                    description="Optional list of metadata dictionaries")
    
    @validator('text_ids')
    def validate_text_ids_length(cls, v, values):
        """Validate that text_ids length matches texts length if provided."""
        if v is not None and 'texts' in values:
            if len(v) != len(values['texts']):
                raise ValueError('text_ids length must match texts length')
        return v
    
    @validator('metadata')
    def validate_metadata_length(cls, v, values):
        """Validate that metadata length matches texts length if provided."""
        if v is not None and 'texts' in values:
            if len(v) != len(values['texts']):
                raise ValueError('metadata length must match texts length')
        return v


class EmbeddingRequest(BaseModel):
    """Request for embedding generation."""
    input_data: Union[TextInput, BatchTextInput] = Field(..., 
                                                        description="Text input data")
    model_type: ModelType = Field(ModelType.BGE, 
                                 description="Type of embedding model to use")
    observational_state: float = Field(1.0, ge=0.1, le=5.0,
                                      description="Observational state parameter for charge generation")
    gamma: float = Field(1.0, ge=0.1, le=3.0,
                        description="Global field calibration factor")


class SemanticEmbeddingResponse(BaseModel):
    """Response containing a single semantic embedding."""
    text_id: str = Field(..., description="Unique identifier for the text")
    text: str = Field(..., description="Original input text")
    vector: List[float] = Field(..., description="Embedding vector components")
    dimension: int = Field(..., description="Dimensionality of the embedding")
    magnitude: float = Field(..., description="L2 norm of the embedding vector")
    model_type: str = Field(..., description="Type of model used for generation")
    metadata: Dict[str, Any] = Field(..., description="Embedding metadata")


class BatchEmbeddingResponse(BaseModel):
    """Response containing multiple semantic embeddings."""
    embeddings: List[SemanticEmbeddingResponse] = Field(..., 
                                                       description="List of generated embeddings")
    batch_size: int = Field(..., description="Number of embeddings in batch")
    model_type: str = Field(..., description="Type of model used for generation")
    processing_stats: Dict[str, Any] = Field(..., description="Batch processing statistics")


class ConceptualChargeRequest(BaseModel):
    """Request for conceptual charge generation."""
    input_data: Union[TextInput, BatchTextInput] = Field(...,
                                                        description="Text input for charge generation")
    model_type: ModelType = Field(ModelType.BGE,
                                 description="Embedding model for semantic vector generation")
    observational_state: float = Field(1.0, ge=0.1, le=5.0,
                                      description="Current observational state")
    gamma: float = Field(1.2, ge=0.1, le=3.0,
                        description="Global field calibration factor")
    context: Optional[str] = Field(None, description="Optional contextual information")


class ConceptualChargeResponse(BaseModel):
    """Response containing a conceptual charge."""
    charge_id: str = Field(..., description="Unique identifier for the charge")
    text: str = Field(..., description="Original input text")
    complete_charge: Dict[str, float] = Field(..., 
                                             description="Complete charge Q(τ, C, s) with magnitude and phase")
    field_position: List[float] = Field(..., description="Position in field universe")
    semantic_field: List[float] = Field(..., description="Semantic field components")
    emotional_trajectory: List[float] = Field(..., description="Emotional trajectory components")
    phase_total: float = Field(..., description="Total phase integration")
    observational_state: float = Field(..., description="Current observational state")
    gamma: float = Field(..., description="Applied gamma calibration")
    metadata: Dict[str, Any] = Field(..., description="Charge metadata and properties")


class BatchChargeResponse(BaseModel):
    """Response containing multiple conceptual charges."""
    charges: List[ConceptualChargeResponse] = Field(...,
                                                   description="List of generated charges")
    batch_size: int = Field(..., description="Number of charges in batch")
    universe_stats: Dict[str, Any] = Field(..., description="Universe statistics")


class SpaceAnalysisRequest(BaseModel):
    """Request for semantic space analysis."""
    embedding_ids: List[str] = Field(..., min_items=2,
                                    description="List of embedding IDs to analyze")
    analysis_type: str = Field("complete", 
                              description="Type of analysis to perform")
    include_field_prep: bool = Field(True,
                                    description="Include field theory preparation")


class SpaceAnalysisResponse(BaseModel):
    """Response containing semantic space analysis."""
    analysis_id: str = Field(..., description="Unique identifier for analysis")
    geometric_properties: Dict[str, Any] = Field(...,
                                                 description="Geometric space properties")
    topological_features: Dict[str, Any] = Field(...,
                                                 description="Topological characteristics")
    semantic_distributions: Dict[str, Any] = Field(...,
                                                   description="Semantic content distributions")
    dimensional_analysis: Dict[str, Any] = Field(...,
                                                description="Per-dimension characteristics")
    field_preparation: Dict[str, Any] = Field(...,
                                             description="Field theory preparation data")
    universe_parameters: Dict[str, Any] = Field(...,
                                               description="Recommended universe parameters")


class FieldVisualizationRequest(BaseModel):
    """Request for field visualization."""
    charge_ids: List[str] = Field(..., min_items=1, max_items=1000,
                                 description="List of charge IDs to visualize")
    visualization_type: str = Field("3d_field", 
                                   description="Type of visualization to generate")
    resolution: int = Field(50, ge=10, le=200,
                           description="Resolution for field visualization")


class FieldVisualizationResponse(BaseModel):
    """Response containing field visualization data."""
    visualization_id: str = Field(..., description="Unique identifier for visualization")
    visualization_type: str = Field(..., description="Type of visualization generated")
    field_data: Dict[str, Any] = Field(..., description="Field visualization data")
    charge_positions: List[Dict[str, Any]] = Field(...,
                                                  description="Charge positions and properties")
    field_statistics: Dict[str, Any] = Field(..., description="Field strength statistics")


class ResonanceAnalysisRequest(BaseModel):
    """Request for field resonance analysis."""
    charge_ids: List[str] = Field(..., min_items=2, max_items=100,
                                 description="List of charge IDs for resonance analysis")
    analysis_depth: str = Field("standard",
                               description="Depth of resonance analysis")


class ResonanceAnalysisResponse(BaseModel):
    """Response containing resonance analysis."""
    analysis_id: str = Field(..., description="Unique identifier for analysis")
    resonance_matrix: List[List[float]] = Field(...,
                                               description="Pairwise resonance strengths")
    resonance_patterns: Dict[str, Any] = Field(...,
                                              description="Identified resonance patterns")
    field_interactions: Dict[str, Any] = Field(...,
                                              description="Field interaction analysis")
    collective_behavior: Dict[str, Any] = Field(...,
                                               description="Collective field behavior")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class APIStatus(BaseModel):
    """API status response model."""
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    api_name: str = Field(..., description="Name of the API")
    documentation: str = Field(..., description="Documentation URL")
    available_endpoints: List[str] = Field(..., description="List of available endpoints")


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Health check timestamp")
    system_info: Dict[str, Any] = Field(..., description="System information")


class UniverseStats(BaseModel):
    """Universe statistics model."""
    total_charges: int = Field(..., description="Total number of charges in universe")
    field_strength_avg: float = Field(..., description="Average field strength")
    universe_size: float = Field(..., description="Universe size")
    charge_density: float = Field(..., description="Charge density")
    active_regions: int = Field(..., description="Number of active field regions")


class APIKeyRequest(BaseModel):
    """Request for API key generation."""
    name: str = Field(..., min_length=1, max_length=50,
                     description="Name for the new API key")
    permissions: Optional[List[str]] = Field(None,
                                           description="List of permissions for the key")


class APIKeyResponse(BaseModel):
    """Response containing API key information."""
    key_name: str = Field(..., description="Name of the API key")
    api_key: str = Field(..., description="Generated API key")
    created_at: str = Field(..., description="Creation timestamp")
    permissions: List[str] = Field(..., description="Key permissions")


class ModelComparisonRequest(BaseModel):
    """Request for model comparison analysis."""
    text: str = Field(..., min_length=1, max_length=5000,
                     description="Text to compare across models")
    models: List[ModelType] = Field([ModelType.BGE, ModelType.MPNET],
                                   description="Models to compare")
    comparison_metrics: List[str] = Field(["similarity", "magnitude", "social_score"],
                                         description="Metrics to compare")


class ModelComparisonResponse(BaseModel):
    """Response containing model comparison results."""
    text: str = Field(..., description="Original text compared")
    model_results: Dict[str, SemanticEmbeddingResponse] = Field(...,
                                                               description="Results per model")
    comparison_metrics: Dict[str, Any] = Field(...,
                                              description="Comparison analysis")
    recommendations: Dict[str, str] = Field(...,
                                           description="Model selection recommendations")