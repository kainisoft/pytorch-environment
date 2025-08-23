"""
Configuration package for Chatbot-Qoder Tutorial Series

This package provides comprehensive configuration management for models,
training strategies, and data processing throughout the tutorial series.
"""

from .model_configs import (
    BaseModelConfig,
    MLPConfig,
    RNNConfig,
    AttentionConfig,
    TransformerConfig,
    ChatbotConfig,
    get_config,
    create_custom_config
)

from .training_configs import (
    TrainingConfig,
    FoundationTrainingConfig,
    LanguageModelTrainingConfig,
    ChatbotTrainingConfig,
    Seq2SeqTrainingConfig,
    TransformerTrainingConfig,
    FineTuningConfig,
    get_training_config,
    create_experiment_dir,
    save_config,
    load_config
)

from .data_configs import (
    DataConfig,
    TextClassificationDataConfig,
    LanguageModelDataConfig,
    ConversationDataConfig,
    RetrievalDataConfig,
    Seq2SeqDataConfig,
    TransformerDataConfig,
    get_data_config,
    validate_data_paths,
    create_data_directories,
    get_special_tokens
)

__all__ = [
    # Model configurations
    "BaseModelConfig",
    "MLPConfig",
    "RNNConfig", 
    "AttentionConfig",
    "TransformerConfig",
    "ChatbotConfig",
    "get_config",
    "create_custom_config",
    
    # Training configurations
    "TrainingConfig",
    "FoundationTrainingConfig",
    "LanguageModelTrainingConfig",
    "ChatbotTrainingConfig",
    "Seq2SeqTrainingConfig",
    "TransformerTrainingConfig",
    "FineTuningConfig",
    "get_training_config",
    "create_experiment_dir",
    "save_config",
    "load_config",
    
    # Data configurations
    "DataConfig",
    "TextClassificationDataConfig",
    "LanguageModelDataConfig",
    "ConversationDataConfig",
    "RetrievalDataConfig",
    "Seq2SeqDataConfig",
    "TransformerDataConfig",
    "get_data_config",
    "validate_data_paths",
    "create_data_directories",
    "get_special_tokens"
]