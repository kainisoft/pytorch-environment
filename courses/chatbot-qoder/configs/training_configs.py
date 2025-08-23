"""
Training Configuration Module for Chatbot-Qoder Tutorial Series

This module contains configuration classes for training strategies, optimization
parameters, and experiment settings used throughout the tutorial series.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import os


@dataclass
class TrainingConfig:
    """Base training configuration for all tutorial notebooks."""
    
    # Basic training parameters
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Optimization settings
    optimizer: str = "adam"  # "adam", "sgd", "adamw", "rmsprop"
    scheduler: str = "none"  # "none", "step", "cosine", "plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Gradient settings
    gradient_clip_norm: Optional[float] = 1.0
    accumulation_steps: int = 1
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 7
    min_delta: float = 0.001
    
    # Validation and evaluation
    validation_split: float = 0.2
    eval_every_n_epochs: int = 1
    save_best_model: bool = True
    
    # Logging and checkpointing
    log_every_n_steps: int = 100
    save_every_n_epochs: int = 5
    checkpoint_dir: str = "models/checkpoints"
    experiment_name: str = "default_experiment"
    
    # Device and performance
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    mixed_precision: bool = False
    dataloader_num_workers: int = 2
    pin_memory: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {key: value for key, value in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class FoundationTrainingConfig(TrainingConfig):
    """Training configuration for foundation notebooks (01-05)."""
    
    # Simplified training for learning purposes
    num_epochs: int = 15
    batch_size: int = 64
    learning_rate: float = 0.01
    
    # Less complex optimization
    optimizer: str = "sgd"
    scheduler: str = "step"
    early_stopping: bool = False
    
    # More frequent logging for educational purposes
    log_every_n_steps: int = 20
    eval_every_n_epochs: int = 1


@dataclass
class LanguageModelTrainingConfig(TrainingConfig):
    """Training configuration for language modeling (Notebook 05)."""
    
    # Language modeling specific parameters
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.002
    
    # Perplexity-based evaluation
    eval_metric: str = "perplexity"
    patience: int = 5
    
    # Text generation parameters
    generation_max_length: int = 100
    generation_temperature: float = 1.0
    generation_top_k: int = 50


@dataclass
class ChatbotTrainingConfig(TrainingConfig):
    """Training configuration for chatbot models (Notebooks 06-12)."""
    
    # Chatbot-specific training
    num_epochs: int = 25
    batch_size: int = 16
    learning_rate: float = 0.0005
    
    # Advanced optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clip_norm: float = 0.5
    
    # Conversation-specific parameters
    max_context_turns: int = 5
    response_max_length: int = 128
    
    # Evaluation metrics
    eval_metrics: List[str] = None
    
    def __post_init__(self):
        if self.eval_metrics is None:
            self.eval_metrics = ["bleu", "rouge", "perplexity"]


@dataclass
class Seq2SeqTrainingConfig(ChatbotTrainingConfig):
    """Training configuration for sequence-to-sequence models (Notebook 08)."""
    
    # Seq2seq specific parameters
    teacher_forcing_ratio: float = 0.5
    teacher_forcing_decay: float = 0.9
    teacher_forcing_min: float = 0.1
    
    # Beam search parameters
    beam_size: int = 4
    length_penalty: float = 0.6
    
    # Loss configuration
    label_smoothing: float = 0.1
    loss_function: str = "cross_entropy"  # "cross_entropy", "focal", "label_smoothing"


@dataclass
class TransformerTrainingConfig(ChatbotTrainingConfig):
    """Training configuration for transformer models (Notebooks 10-11)."""
    
    # Transformer-specific training
    num_epochs: int = 30
    batch_size: int = 8  # Smaller batch for memory efficiency
    learning_rate: float = 0.0001
    
    # Advanced optimization for transformers
    warmup_steps: int = 4000
    scheduler: str = "warmup_cosine"
    gradient_clip_norm: float = 1.0
    
    # Memory optimization
    mixed_precision: bool = True
    accumulation_steps: int = 4
    
    # Generation parameters
    generation_strategy: str = "beam"  # "greedy", "beam", "sampling"
    num_beams: int = 4
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9


@dataclass
class FineTuningConfig(TrainingConfig):
    """Configuration for fine-tuning pre-trained models (Notebook 12)."""
    
    # Fine-tuning specific parameters
    num_epochs: int = 5
    learning_rate: float = 0.00005
    warmup_ratio: float = 0.1
    
    # Regularization for fine-tuning
    weight_decay: float = 0.01
    dropout_rate: float = 0.1
    
    # Layer-wise learning rates
    use_layer_lr: bool = False
    layer_lr_decay: float = 0.95
    
    # Freezing strategy
    freeze_embeddings: bool = False
    freeze_encoder_layers: int = 0
    
    # Evaluation strategy
    eval_strategy: str = "steps"  # "epoch", "steps"
    eval_steps: int = 500
    save_strategy: str = "best"  # "best", "epoch", "steps"


# Pre-defined training configurations for different tutorial stages
TRAINING_CONFIGS = {
    "notebook_01_fundamentals": FoundationTrainingConfig(
        num_epochs=10,
        batch_size=128,
        learning_rate=0.01,
        experiment_name="pytorch_fundamentals"
    ),
    
    "notebook_04_mlp": FoundationTrainingConfig(
        num_epochs=20,
        batch_size=64,
        learning_rate=0.001,
        optimizer="adam",
        experiment_name="mlp_text_classification"
    ),
    
    "notebook_05_language_model": LanguageModelTrainingConfig(
        num_epochs=25,
        batch_size=32,
        learning_rate=0.002,
        experiment_name="character_language_model"
    ),
    
    "notebook_06_rule_based": TrainingConfig(
        # Rule-based doesn't need training, but config for evaluation
        num_epochs=0,
        experiment_name="rule_based_chatbot"
    ),
    
    "notebook_07_retrieval": TrainingConfig(
        # Minimal training for embedding fine-tuning
        num_epochs=5,
        batch_size=64,
        learning_rate=0.0001,
        experiment_name="retrieval_chatbot"
    ),
    
    "notebook_08_seq2seq": Seq2SeqTrainingConfig(
        num_epochs=30,
        batch_size=32,
        learning_rate=0.001,
        teacher_forcing_ratio=0.5,
        experiment_name="seq2seq_chatbot"
    ),
    
    "notebook_09_attention": Seq2SeqTrainingConfig(
        num_epochs=25,
        batch_size=16,
        learning_rate=0.0005,
        teacher_forcing_ratio=0.7,
        experiment_name="attention_chatbot"
    ),
    
    "notebook_10_transformer": TransformerTrainingConfig(
        num_epochs=20,
        batch_size=8,
        learning_rate=0.0001,
        warmup_steps=2000,
        experiment_name="transformer_chatbot"
    ),
    
    "notebook_11_generative": TransformerTrainingConfig(
        num_epochs=15,
        batch_size=4,
        learning_rate=0.00005,
        warmup_steps=1000,
        mixed_precision=True,
        accumulation_steps=8,
        experiment_name="generative_chatbot"
    ),
    
    "notebook_12_fine_tuning": FineTuningConfig(
        num_epochs=3,
        learning_rate=0.00002,
        warmup_ratio=0.1,
        weight_decay=0.01,
        experiment_name="fine_tuned_chatbot"
    )
}


def get_training_config(config_name: str, **kwargs) -> TrainingConfig:
    """
    Get a predefined training configuration by name with optional parameter overrides.
    
    Args:
        config_name (str): Name of the configuration to retrieve
        **kwargs: Additional parameters to override in the configuration
    
    Returns:
        TrainingConfig: Training configuration object with specified parameters
    
    Example:
        >>> config = get_training_config("notebook_08_seq2seq", learning_rate=0.002)
        >>> print(config.learning_rate)
        0.002
    """
    if config_name not in TRAINING_CONFIGS:
        available_configs = list(TRAINING_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available_configs}")
    
    config = TRAINING_CONFIGS[config_name]
    
    # Override with provided kwargs
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        config = type(config).from_dict(config_dict)
    
    return config


def create_experiment_dir(config: TrainingConfig) -> str:
    """
    Create directory structure for experiment logging and checkpoints.
    
    Args:
        config (TrainingConfig): Training configuration
    
    Returns:
        str: Path to the experiment directory
    """
    experiment_dir = os.path.join(config.checkpoint_dir, config.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ["checkpoints", "logs", "outputs", "configs"]
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    return experiment_dir


def save_config(config: TrainingConfig, save_path: str):
    """
    Save training configuration to a file.
    
    Args:
        config (TrainingConfig): Configuration to save
        save_path (str): Path to save the configuration file
    """
    import json
    
    config_dict = config.to_dict()
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(config_path: str, config_class: type = TrainingConfig) -> TrainingConfig:
    """
    Load training configuration from a file.
    
    Args:
        config_path (str): Path to the configuration file
        config_class (type): Configuration class to instantiate
    
    Returns:
        TrainingConfig: Loaded configuration object
    """
    import json
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return config_class.from_dict(config_dict)


# Export commonly used configurations and utilities
__all__ = [
    "TrainingConfig",
    "FoundationTrainingConfig",
    "LanguageModelTrainingConfig",
    "ChatbotTrainingConfig",
    "Seq2SeqTrainingConfig",
    "TransformerTrainingConfig",
    "FineTuningConfig",
    "TRAINING_CONFIGS",
    "get_training_config",
    "create_experiment_dir",
    "save_config",
    "load_config"
]