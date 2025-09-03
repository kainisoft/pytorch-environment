"""
Health check router.

This module provides health check endpoints for monitoring
application status and dependencies.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.logging import get_logger
from app.services import get_ai_service

logger = get_logger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    message: str


class ModelInfoResponse(BaseModel):
    """Model information response model."""
    model_name: str
    device: str
    model_size: str
    is_loaded: bool
    parameters: Dict[str, int]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        HealthResponse: Application health status
    """
    logger.info("Health check requested")
    
    try:
        ai_service = get_ai_service()
        ai_health = ai_service.health_check()
        
        overall_status = "healthy" if ai_health["status"] == "healthy" else "degraded"
        
        return HealthResponse(
            status=overall_status,
            message="AI Chatbot Backend is running"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}"
        )


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with component status.
    
    Returns:
        Dict: Detailed health information
    """
    logger.info("Detailed health check requested")
    
    try:
        ai_service = get_ai_service()
        ai_health = ai_service.health_check()
        
        return {
            "status": "healthy",
            "components": {
                "database": "healthy",  # TODO: Add actual database health check
                "ai_model": ai_health,
                "api": "healthy"
            },
            "version": "1.0.0",
            "timestamp": ai_health.get("timestamp")
        }
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            "status": "unhealthy",
            "components": {
                "database": "unknown",
                "ai_model": {"status": "error", "error": str(e)},
                "api": "healthy"
            },
            "version": "1.0.0"
        }


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get AI model information.
    
    Returns:
        ModelInfoResponse: Model information and status
    """
    logger.info("Model info requested")
    
    try:
        ai_service = get_ai_service()
        model_info = ai_service.get_model_info()
        
        if not model_info.get("is_loaded"):
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        return ModelInfoResponse(
            model_name=model_info.get("model_name", "Unknown"),
            device=model_info.get("device", "Unknown"),
            model_size=model_info.get("model_size", "Unknown"),
            is_loaded=model_info.get("is_loaded", False),
            parameters=model_info.get("parameters", {})
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.post("/model/load")
async def load_model() -> Dict[str, Any]:
    """
    Load the AI model.
    
    Returns:
        Dict: Load operation result
    """
    logger.info("Model load requested")
    
    try:
        ai_service = get_ai_service()
        
        if ai_service.is_model_loaded():
            return {
                "status": "success",
                "message": "Model already loaded",
                "model_info": ai_service.get_model_info()
            }
        
        success = ai_service.load_model()
        
        if success:
            return {
                "status": "success",
                "message": "Model loaded successfully",
                "model_info": ai_service.get_model_info()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to load model"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


@router.post("/model/unload")
async def unload_model() -> Dict[str, str]:
    """
    Unload the AI model and free memory.
    
    Returns:
        Dict: Unload operation result
    """
    logger.info("Model unload requested")
    
    try:
        ai_service = get_ai_service()
        ai_service.cleanup()
        
        return {
            "status": "success",
            "message": "Model unloaded successfully"
        }
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload model: {str(e)}"
        )