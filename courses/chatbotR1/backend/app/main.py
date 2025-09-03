"""
Main FastAPI application module.

This module sets up the FastAPI application with CORS middleware,
routing, and basic configuration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.routers import chat, health
from app.services import get_ai_service, cleanup_ai_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting up AI Chatbot Backend...")
    setup_logging()
    
    # Initialize AI service (but don't load model yet - load on first request)
    try:
        ai_service = get_ai_service()
        logger.info("AI service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI service: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Chatbot Backend...")
    try:
        cleanup_ai_service()
        logger.info("AI service cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during AI service cleanup: {e}")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="AI Chatbot Application Backend",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])

    return app


# Create the FastAPI app instance
app = create_app()


@app.get("/")
async def root():
    """
    Root endpoint for basic health check.
    """
    return {"message": "AI Chatbot Backend API", "version": "1.0.0"}