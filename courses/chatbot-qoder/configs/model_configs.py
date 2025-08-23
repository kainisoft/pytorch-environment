"""
Model Configuration Module for Chatbot-Qoder Tutorial Series

This module contains configuration classes for different model architectures
used throughout the tutorial series. Each configuration class provides
sensible defaults and documentation for model hyperparameters.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BaseModelConfig:
    """Base configuration class for all models in the tutorial series."""
    
    # General model parameters
    vocab_size: int = 10000
    max_length: int = 512
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 10
    
    # Device configuration
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {key: value for key, value in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class MLPConfig(BaseModelConfig):
    """Configuration for Multi-Layer Perceptron models (Notebook 04)."""
    
    # MLP-specific parameters
    input_dim: int = 128
    hidden_dims: tuple = (256, 128, 64)
    output_dim: int = 10
    activation: str = "relu"  # "relu", "tanh", "sigmoid"
    batch_norm: bool = True
    
    # Text classification specific
    num_classes: int = 5
    embedding_dim: int = 128


@dataclass
class RNNConfig(BaseModelConfig):
    """Configuration for RNN-based models (Notebook 08)."""
    
    # RNN architecture parameters
    rnn_type: str = "LSTM"  # "RNN", "LSTM", "GRU"
    hidden_dim: int = 256
    num_layers: int = 2
    bidirectional: bool = False
    
    # Sequence-to-sequence parameters
    encoder_hidden_dim: int = 256
    decoder_hidden_dim: int = 256
    teacher_forcing_ratio: float = 0.5
    
    # Generation parameters
    max_decode_length: int = 50
    sos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class AttentionConfig(BaseModelConfig):
    """Configuration for attention-based models (Notebook 09)."""
    
    # Attention mechanism parameters
    attention_type: str = "additive"  # "additive", "multiplicative", "scaled_dot"
    attention_dim: int = 256
    num_heads: int = 8
    
    # Encoder-decoder with attention
    encoder_hidden_dim: int = 256
    decoder_hidden_dim: int = 256
    attention_hidden_dim: int = 256
    
    # Coverage mechanism
    use_coverage: bool = False
    coverage_weight: float = 1.0


@dataclass
class TransformerConfig(BaseModelConfig):
    """Configuration for Transformer models (Notebook 10)."""
    
    # Transformer architecture parameters
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    
    # Positional encoding
    max_position_embeddings: int = 512
    use_positional_encoding: bool = True
    
    # Regularization
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Generation parameters
    num_beams: int = 4
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95


@dataclass
class ChatbotConfig(BaseModelConfig):
    """Configuration for complete chatbot systems (Notebooks 11-12)."""
    
    # Model architecture
    model_type: str = "transformer"  # "rnn", "attention", "transformer"
    pretrained_model: Optional[str] = None
    
    # Conversation parameters
    max_context_length: int = 1024
    max_response_length: int = 128
    min_response_length: int = 5
    
    # Generation parameters
    do_sample: bool = True
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    
    # Safety and filtering
    use_safety_filter: bool = True
    max_toxicity_score: float = 0.7
    filter_repetitive_responses: bool = True
    
    # Response ranking
    use_response_ranking: bool = True
    ranking_model: str = "similarity"  # "similarity", "neural", "ensemble"


# Pre-defined configurations for different tutorial stages
TUTORIAL_CONFIGS = {
    "notebook_04_mlp": MLPConfig(
        input_dim=100,
        hidden_dims=(128, 64),
        output_dim=5,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=20
    ),
    
    "notebook_05_language_model": BaseModelConfig(
        vocab_size=5000,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        learning_rate=0.002,
        batch_size=64,
        num_epochs=15
    ),
    
    "notebook_08_seq2seq": RNNConfig(
        rnn_type="LSTM",
        hidden_dim=256,
        num_layers=2,
        bidirectional=True,
        teacher_forcing_ratio=0.5,
        learning_rate=0.001,
        batch_size=32
    ),
    
    "notebook_09_attention": AttentionConfig(
        attention_type="scaled_dot",
        num_heads=4,
        encoder_hidden_dim=256,
        decoder_hidden_dim=256,
        attention_hidden_dim=256,
        learning_rate=0.0005
    ),
    
    "notebook_10_transformer": TransformerConfig(
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=8,
        d_model=256,
        d_ff=1024,
        learning_rate=0.0001,
        batch_size=16
    ),
    
    "notebook_11_generative": ChatbotConfig(
        model_type="transformer",
        max_context_length=512,
        max_response_length=64,
        temperature=0.8,
        top_p=0.9,
        learning_rate=0.0001
    )
}


def get_config(config_name: str, **kwargs) -> BaseModelConfig:
    """
    Get a predefined configuration by name with optional parameter overrides.
    
    Args:
        config_name (str): Name of the configuration to retrieve
        **kwargs: Additional parameters to override in the configuration
    
    Returns:
        BaseModelConfig: Configuration object with specified parameters
    
    Example:
        >>> config = get_config("notebook_08_seq2seq", learning_rate=0.002)
        >>> print(config.learning_rate)
        0.002
    """
    if config_name not in TUTORIAL_CONFIGS:
        available_configs = list(TUTORIAL_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available_configs}")
    
    config = TUTORIAL_CONFIGS[config_name]
    
    # Override with provided kwargs
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        config = type(config).from_dict(config_dict)
    
    return config


def create_custom_config(base_config: str, **kwargs) -> BaseModelConfig:
    """
    Create a custom configuration based on a base configuration.
    
    Args:
        base_config (str): Name of the base configuration
        **kwargs: Parameters to override or add
    
    Returns:
        BaseModelConfig: Custom configuration object
    
    Example:
        >>> config = create_custom_config(
        ...     "notebook_08_seq2seq",
        ...     hidden_dim=512,
        ...     num_layers=3,
        ...     custom_param="value"
        ... )
    """
    return get_config(base_config, **kwargs)


# Export commonly used configurations
__all__ = [
    "BaseModelConfig",
    "MLPConfig", 
    "RNNConfig",
    "AttentionConfig",
    "TransformerConfig",
    "ChatbotConfig",
    "TUTORIAL_CONFIGS",
    "get_config",
    "create_custom_config"
]