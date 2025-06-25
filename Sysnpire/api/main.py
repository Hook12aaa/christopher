"""
Main entry point for the Sysnpire Field Theory API.
FastAPI application with automatic documentation, validation, and authentication.
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uuid
import os
import logging
from datetime import datetime
from typing import Dict, List

from .models import APIStatus, HealthCheck, ErrorResponse, APIKeyRequest, APIKeyResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

security = HTTPBearer()

API_KEYS = {
    "development": f"dev_key_{str(uuid.uuid4())[:8]}",
}

if os.environ.get("SYSNPIRE_API_KEY"):
    API_KEYS["production"] = os.environ.get("SYSNPIRE_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown events."""
    logger.info("Starting Sysnpire Field Theory API")
    logger.info(f"Development API Key: {API_KEYS['development']}")
    yield
    logger.info("Shutting down Sysnpire Field Theory API")


app = FastAPI(
    title="Sysnpire Field Theory API",
    description="""
    **Sysnpire Field Theory API** - Transform text into mathematical fields using conceptual charge theory.
    
    This API provides endpoints for:
    - **Text-to-Embedding**: Convert text to semantic embeddings using BGE or MPNet models
    - **Conceptual Charges**: Generate field-theoretic representations with Q(τ, C, s) formula
    - **Semantic Analysis**: Comprehensive space analysis for universe construction
    - **Field Visualization**: 3D visualization of semantic fields and charge interactions
    - **Resonance Analysis**: Study field resonance patterns and collective behavior
    
    Built on cutting-edge research in Field Theory of Social Constructs.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key authentication."""
    token = credentials.credentials
    if token not in API_KEYS.values():
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


@app.get("/")
async def root():
    """
    Root endpoint returning API status and available endpoints.
    
    Returns basic information about the API including version,
    status, and links to documentation.
    """
    return APIStatus(
        status="online",
        version="1.0.0",
        api_name="Sysnpire Field Theory API",
        documentation="/docs",
        available_endpoints=[
            "/api/embeddings/",
            "/api/charges/",
            "/api/analysis/",
            "/api/visualization/",
            "/api/resonance/"
        ]
    )


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and uptime verification.
    
    Returns system health status, timestamp, and basic system information
    for monitoring services and load balancers.
    """
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        system_info={
            "api_version": "1.0.0",
            "python_version": "3.8+",
            "models_loaded": ["BGE-Large-v1.5", "all-mpnet-base-v2"],
            "available_endpoints": 5
        }
    )


@app.post("/api/auth/generate_key", 
          response_model=APIKeyResponse, 
          dependencies=[Depends(verify_api_key)],
          tags=["Authentication"])
async def generate_api_key(request: APIKeyRequest):
    """
    Generate a new API key with specified permissions.
    
    Requires an existing valid API key for authentication.
    Used for creating additional keys for different services or users.
    """
    new_key = f"sysnpire_key_{str(uuid.uuid4())}"
    API_KEYS[request.name] = new_key
    
    return APIKeyResponse(
        key_name=request.name,
        api_key=new_key,
        created_at=datetime.utcnow().isoformat(),
        permissions=request.permissions or ["read", "write"]
    )


@app.get("/api/auth/list_keys", 
         dependencies=[Depends(verify_api_key)],
         tags=["Authentication"])
async def list_api_keys():
    """
    List all API key names (admin endpoint).
    
    Returns the names of all registered API keys without exposing
    the actual key values for security purposes.
    """
    return {"key_names": list(API_KEYS.keys())}


from .routers import embeddings, charges, analysis, visualization, resonance

app.include_router(
    embeddings.router,
    prefix="/api/embeddings",
    tags=["Embeddings"],
    dependencies=[Depends(verify_api_key)]
)

app.include_router(
    charges.router,
    prefix="/api/charges", 
    tags=["Conceptual Charges"],
    dependencies=[Depends(verify_api_key)]
)

app.include_router(
    analysis.router,
    prefix="/api/analysis",
    tags=["Semantic Analysis"],
    dependencies=[Depends(verify_api_key)]
)

app.include_router(
    visualization.router,
    prefix="/api/visualization",
    tags=["Field Visualization"],
    dependencies=[Depends(verify_api_key)]
)

app.include_router(
    resonance.router,
    prefix="/api/resonance",
    tags=["Field Resonance"],
    dependencies=[Depends(verify_api_key)]
)


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors with structured response."""
    return HTTPException(
        status_code=404,
        detail="Endpoint not found. Visit /docs for API documentation."
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors with structured response."""
    logger.error(f"Internal server error: {exc}")
    return HTTPException(
        status_code=500,
        detail="Internal server error. Please try again later."
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )