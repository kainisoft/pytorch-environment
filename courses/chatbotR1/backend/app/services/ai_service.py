"""
AI Service for managing PyTorch language models.

This module provides the AIService class for loading, managing, and using
PyTorch-based language models for chatbot responses.
"""

import gc
import logging
import os
import threading
from typing import Optional, Dict, Any, List
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    pipeline
)
from ..core.config import settings

logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration class for AI model settings."""
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        max_context_length: int = None,
        max_response_length: int = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: int = None
    ):
        self.model_name = model_name or settings.model_name
        self.device = device or settings.device
        self.max_context_length = max_context_length or settings.max_context_length
        self.max_response_length = max_response_length or settings.max_response_length
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.pad_token_id = pad_token_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_context_length": self.max_context_length,
            "max_response_length": self.max_response_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "pad_token_id": self.pad_token_id
        }


class AIService:
    """
    AI Service for managing PyTorch language models.
    
    This service handles model loading, device selection, memory management,
    and provides methods for generating AI responses.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize AI Service.
        
        Args:
            config: Model configuration. If None, uses default settings.
        """
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_loaded = False
        self.load_time = None
        self.model_info = {}
        self._lock = threading.Lock()
        
        # Initialize device
        self._setup_device()
        
        logger.info(f"AIService initialized with device: {self.device}")
    
    def _setup_device(self) -> None:
        """Setup and validate the compute device."""
        requested_device = self.config.device.lower()
        
        if requested_device == "auto":
            # Auto-detect best available device
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        elif requested_device == "cuda":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
        elif requested_device == "mps":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("MPS (Apple Silicon) device available")
            else:
                logger.warning("MPS requested but not available, falling back to CPU")
                self.device = "cpu"
        else:
            self.device = "cpu"
        
        logger.info(f"Using device: {self.device}")
    
    def load_model(self) -> bool:
        """
        Load the language model and tokenizer.
        
        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        with self._lock:
            if self.is_loaded:
                logger.info("Model already loaded")
                return True
            
            try:
                start_time = datetime.now()
                logger.info(f"Loading model: {self.config.model_name}")
                
                # Load tokenizer
                logger.info("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    cache_dir=settings.model_path
                )
                
                # Set pad token if not exists
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.config.pad_token_id = self.tokenizer.eos_token_id
                
                # Load model
                logger.info("Loading model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    cache_dir=settings.model_path,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
                
                # Move model to device if not using device_map
                if self.device != "cuda":
                    self.model = self.model.to(self.device)
                
                # Set model to evaluation mode
                self.model.eval()
                
                # Record load time and model info
                self.load_time = datetime.now()
                load_duration = (self.load_time - start_time).total_seconds()
                
                self.model_info = {
                    "model_name": self.config.model_name,
                    "device": str(self.device),
                    "model_size": self._get_model_size(),
                    "load_time": load_duration,
                    "loaded_at": self.load_time.isoformat(),
                    "parameters": self._count_parameters()
                }
                
                self.is_loaded = True
                logger.info(f"Model loaded successfully in {load_duration:.2f}s")
                logger.info(f"Model info: {self.model_info}")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                self.cleanup()
                return False
    
    def _get_model_size(self) -> str:
        """Get approximate model size in memory."""
        if not self.model:
            return "Unknown"
        
        try:
            param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
            total_size = param_size + buffer_size
            
            # Convert to human readable format
            for unit in ['B', 'KB', 'MB', 'GB']:
                if total_size < 1024.0:
                    return f"{total_size:.1f} {unit}"
                total_size /= 1024.0
            return f"{total_size:.1f} TB"
        except Exception:
            return "Unknown"
    
    def _count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        if not self.model:
            return {}
        
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                "total": total_params,
                "trainable": trainable_params,
                "non_trainable": total_params - trainable_params
            }
        except Exception:
            return {}
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.is_loaded and self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status."""
        base_info = {
            "is_loaded": self.is_loaded,
            "device": str(self.device),
            "config": self.config.to_dict()
        }
        
        if self.is_loaded and self.model_info:
            base_info.update(self.model_info)
        
        return base_info
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the AI service.
        
        Returns:
            Dict containing health status and diagnostics.
        """
        health_status = {
            "status": "healthy" if self.is_model_loaded() else "unhealthy",
            "model_loaded": self.is_model_loaded(),
            "device": str(self.device),
            "timestamp": datetime.now().isoformat()
        }
        
        if self.is_loaded:
            health_status.update({
                "model_name": self.config.model_name,
                "load_time": self.load_time.isoformat() if self.load_time else None,
                "memory_usage": self._get_memory_usage()
            })
        
        # Add device-specific diagnostics
        if self.device == "cuda" and torch.cuda.is_available():
            health_status["cuda_info"] = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved()
            }
        
        return health_status
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        memory_info = {}
        
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                memory_info["cuda"] = {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "max_allocated": torch.cuda.max_memory_allocated(),
                    "max_reserved": torch.cuda.max_memory_reserved()
                }
            
            # Add system memory info if available
            import psutil
            process = psutil.Process()
            memory_info["system"] = {
                "rss": process.memory_info().rss,
                "vms": process.memory_info().vms,
                "percent": process.memory_percent()
            }
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
        
        return memory_info
    
    def cleanup(self) -> None:
        """
        Clean up model and free memory.
        
        This method should be called when the model is no longer needed
        to free up GPU/system memory.
        """
        with self._lock:
            logger.info("Cleaning up AI service...")
            
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
            
            # Force garbage collection
            gc.collect()
            
            self.is_loaded = False
            self.load_time = None
            self.model_info = {}
            
            logger.info("AI service cleanup completed")
    
    def reload_model(self) -> bool:
        """
        Reload the model with current configuration.
        
        Returns:
            bool: True if reload successful, False otherwise.
        """
        logger.info("Reloading model...")
        self.cleanup()
        return self.load_model()
    
    def update_config(self, new_config: ModelConfig) -> bool:
        """
        Update model configuration and reload if necessary.
        
        Args:
            new_config: New model configuration.
            
        Returns:
            bool: True if update successful, False otherwise.
        """
        old_model_name = self.config.model_name
        old_device = self.config.device
        
        self.config = new_config
        
        # Check if we need to reload the model
        need_reload = (
            old_model_name != new_config.model_name or
            old_device != new_config.device
        )
        
        if need_reload and self.is_loaded:
            logger.info("Configuration changed, reloading model...")
            self._setup_device()
            return self.reload_model()
        elif not self.is_loaded:
            self._setup_device()
        
        return True


# Global AI service instance
_ai_service_instance: Optional[AIService] = None
_service_lock = threading.Lock()


def get_ai_service() -> AIService:
    """
    Get the global AI service instance (singleton pattern).
    
    Returns:
        AIService: The global AI service instance.
    """
    global _ai_service_instance
    
    with _service_lock:
        if _ai_service_instance is None:
            _ai_service_instance = AIService()
        return _ai_service_instance


def initialize_ai_service(config: Optional[ModelConfig] = None) -> AIService:
    """
    Initialize the global AI service with optional configuration.
    
    Args:
        config: Optional model configuration.
        
    Returns:
        AIService: The initialized AI service instance.
    """
    global _ai_service_instance
    
    with _service_lock:
        if _ai_service_instance is not None:
            _ai_service_instance.cleanup()
        
        _ai_service_instance = AIService(config)
        return _ai_service_instance


def cleanup_ai_service() -> None:
    """Clean up the global AI service instance."""
    global _ai_service_instance
    
    with _service_lock:
        if _ai_service_instance is not None:
            _ai_service_instance.cleanup()
            _ai_service_instance = None