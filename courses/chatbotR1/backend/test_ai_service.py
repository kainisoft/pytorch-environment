#!/usr/bin/env python3
"""
Test script for AI Service functionality.

This script tests the AI service model loading, device selection,
and basic functionality without requiring the full FastAPI application.
"""

import sys
import os
import logging

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.ai_service import AIService, ModelConfig
from app.core.config import settings

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_device_selection():
    """Test device selection functionality."""
    print("\n=== Testing Device Selection ===")
    
    # Test auto device selection
    config = ModelConfig(device="auto")
    service = AIService(config)
    print(f"Auto device selection: {service.device}")
    
    # Test CPU device
    config = ModelConfig(device="cpu")
    service = AIService(config)
    print(f"CPU device: {service.device}")
    
    # Test CUDA device (if available)
    config = ModelConfig(device="cuda")
    service = AIService(config)
    print(f"CUDA device: {service.device}")
    
    # Test MPS device (if available)
    config = ModelConfig(device="mps")
    service = AIService(config)
    print(f"MPS device: {service.device}")


def test_model_config():
    """Test model configuration."""
    print("\n=== Testing Model Configuration ===")
    
    # Test default config
    config = ModelConfig()
    print(f"Default config: {config.to_dict()}")
    
    # Test custom config
    config = ModelConfig(
        model_name="microsoft/DialoGPT-small",
        device="cpu",
        max_context_length=256,
        temperature=0.8
    )
    print(f"Custom config: {config.to_dict()}")


def test_service_initialization():
    """Test AI service initialization."""
    print("\n=== Testing Service Initialization ===")
    
    # Test with default config
    service = AIService()
    print(f"Service initialized with device: {service.device}")
    print(f"Model loaded: {service.is_model_loaded()}")
    
    # Test health check before model loading
    health = service.health_check()
    print(f"Health check: {health}")
    
    # Test model info before loading
    info = service.get_model_info()
    print(f"Model info: {info}")


def test_model_loading():
    """Test model loading (optional - requires model download)."""
    print("\n=== Testing Model Loading (Optional) ===")
    
    # Use a smaller model for testing
    config = ModelConfig(
        model_name="microsoft/DialoGPT-small",  # Smaller model for testing
        device="cpu",  # Use CPU to avoid GPU memory issues
        max_context_length=128
    )
    
    service = AIService(config)
    
    print("Attempting to load model (this may take a while for first download)...")
    print("Note: This will download the model if not already cached.")
    
    # Uncomment the following lines to actually test model loading
    # WARNING: This will download the model (~350MB for DialoGPT-small)
    
    # success = service.load_model()
    # print(f"Model loading success: {success}")
    
    # if success:
    #     print(f"Model info after loading: {service.get_model_info()}")
    #     print(f"Health check after loading: {service.health_check()}")
    #     
    #     # Test cleanup
    #     service.cleanup()
    #     print(f"Model loaded after cleanup: {service.is_model_loaded()}")
    
    print("Model loading test skipped (uncomment code to run)")


def main():
    """Run all tests."""
    print("AI Service Test Suite")
    print("====================")
    
    try:
        test_device_selection()
        test_model_config()
        test_service_initialization()
        test_model_loading()
        
        print("\n=== All Tests Completed ===")
        print("AI Service foundation appears to be working correctly!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())