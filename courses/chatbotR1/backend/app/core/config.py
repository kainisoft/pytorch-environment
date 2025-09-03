"""
Application configuration settings.

This module handles environment variables and application configuration
using Pydantic settings.
"""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    
    # Project settings
    project_name: str = Field(default="AI Chatbot Backend", alias="PROJECT_NAME")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # Server settings
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    reload: bool = Field(default=False, alias="RELOAD")
    
    # CORS settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        alias="ALLOWED_ORIGINS"
    )
    
    # Database settings
    database_url: str = Field(
        default="sqlite:///./chatbot.db",
        alias="DATABASE_URL"
    )
    
    # AI Model settings
    model_path: str = Field(default="./models", alias="MODEL_PATH")
    model_name: str = Field(default="microsoft/DialoGPT-medium", alias="MODEL_NAME")
    device: str = Field(default="cpu", alias="DEVICE")
    max_context_length: int = Field(default=512, alias="MAX_CONTEXT_LENGTH")
    max_response_length: int = Field(default=150, alias="MAX_RESPONSE_LENGTH")
    
    # Security settings
    secret_key: str = Field(default="your-secret-key-here", alias="SECRET_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()