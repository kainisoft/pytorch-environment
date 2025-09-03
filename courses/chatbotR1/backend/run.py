"""
Application entry point.

This script starts the FastAPI application using uvicorn server.
"""

import uvicorn

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def main():
    """
    Start the FastAPI application.
    """
    logger.info(
        "Starting AI Chatbot Backend",
        host=settings.HOST,
        port=settings.PORT,
        debug=settings.DEBUG
    )
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()