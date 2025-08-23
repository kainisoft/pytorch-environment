"""
Data Configuration Module for Chatbot-Qoder Tutorial Series

This module contains configuration classes for data processing, dataset
handling, and preprocessing parameters used throughout the tutorial series.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
import os


@dataclass
class DataConfig:
    """Base data configuration for all tutorial notebooks."""
    
    # File paths and directories
    data_dir: str = "data"
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Dataset split ratios (if files not provided separately)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Text preprocessing
    max_length: int = 512
    min_length: int = 1
    vocab_size: int = 10000
    lowercase: bool = True
    remove_punctuation: bool = False
    remove_special_chars: bool = False
    
    # Tokenization
    tokenizer_type: str = "word"  # "word", "char", "subword", "bpe"
    vocab_file: Optional[str] = None
    unk_token: str = "<UNK>"
    pad_token: str = "<PAD>"
    sos_token: str = "<SOS>"
    eos_token: str = "<EOS>"
    
    # Data loading
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {key: value for key, value in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class TextClassificationDataConfig(DataConfig):
    """Data configuration for text classification tasks (Notebook 04)."""
    
    # Classification specific
    num_classes: int = 5
    class_names: Optional[List[str]] = None
    balance_classes: bool = True
    
    # Text preprocessing for classification
    max_length: int = 128
    min_length: int = 5
    remove_stopwords: bool = False
    stemming: bool = False
    lemmatization: bool = False
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]


@dataclass
class LanguageModelDataConfig(DataConfig):
    """Data configuration for language modeling (Notebook 05)."""
    
    # Language modeling specific
    sequence_length: int = 100
    overlap: int = 50
    prediction_length: int = 1
    
    # Character vs word level
    tokenizer_type: str = "char"
    vocab_size: int = 128  # For character-level
    
    # Text generation
    generation_max_length: int = 200
    temperature: float = 1.0
    top_k: int = 50


@dataclass
class ConversationDataConfig(DataConfig):
    """Data configuration for conversation datasets (Notebooks 06-12)."""
    
    # Conversation specific parameters
    max_context_length: int = 512
    max_response_length: int = 128
    min_response_length: int = 5
    max_turns: int = 10
    
    # Context handling
    include_previous_turns: bool = True
    context_separator: str = " [SEP] "
    
    # Response filtering
    filter_short_responses: bool = True
    filter_repetitive: bool = True
    repetition_threshold: float = 0.8
    
    # Data augmentation
    augment_data: bool = False
    augmentation_factor: float = 2.0
    augmentation_methods: List[str] = None
    
    def __post_init__(self):
        if self.augmentation_methods is None:
            self.augmentation_methods = ["paraphrase", "synonym_replacement"]


@dataclass
class RetrievalDataConfig(ConversationDataConfig):
    """Data configuration for retrieval-based systems (Notebook 07)."""
    
    # Retrieval specific
    knowledge_base_file: str = "data/conversations/faq_knowledge.json"
    embedding_dim: int = 384
    similarity_threshold: float = 0.7
    top_k_responses: int = 5
    
    # Index building
    build_index: bool = True
    index_type: str = "faiss"  # "faiss", "annoy", "simple"
    
    # Response ranking
    use_reranking: bool = True
    rerank_top_k: int = 10


@dataclass
class Seq2SeqDataConfig(ConversationDataConfig):
    """Data configuration for sequence-to-sequence models (Notebook 08)."""
    
    # Seq2seq specific
    source_max_length: int = 256
    target_max_length: int = 128
    
    # Special tokens for seq2seq
    src_lang_token: Optional[str] = None
    tgt_lang_token: Optional[str] = None
    
    # Teacher forcing
    teacher_forcing_ratio: float = 0.5
    
    # Evaluation
    eval_bleu: bool = True
    eval_rouge: bool = True


@dataclass
class TransformerDataConfig(ConversationDataConfig):
    """Data configuration for transformer models (Notebooks 10-11)."""
    
    # Transformer specific
    block_size: int = 512
    stride: int = 256
    
    # Attention masks
    use_attention_mask: bool = True
    mask_padding: bool = True
    
    # Position embeddings
    max_position_embeddings: int = 512
    
    # For generative models
    generation_max_length: int = 100
    generation_min_length: int = 10


# Pre-defined data configurations for different tutorial stages
DATA_CONFIGS = {
    "notebook_01_fundamentals": DataConfig(
        data_dir="data/simple",
        max_length=50,
        vocab_size=1000,
        batch_size=64
    ),
    
    "notebook_03_preprocessing": DataConfig(
        data_dir="data/conversations",
        max_length=256,
        vocab_size=5000,
        tokenizer_type="word",
        train_file="simple_qa_pairs.json"
    ),
    
    "notebook_04_classification": TextClassificationDataConfig(
        data_dir="data/conversations",
        train_file="simple_qa_pairs.json",
        num_classes=5,
        max_length=128,
        batch_size=32,
        balance_classes=True
    ),
    
    "notebook_05_language_model": LanguageModelDataConfig(
        data_dir="data/conversations",
        train_file="simple_qa_pairs.json",
        tokenizer_type="char",
        sequence_length=100,
        vocab_size=128,
        batch_size=64
    ),
    
    "notebook_06_rule_based": ConversationDataConfig(
        data_dir="data/conversations",
        train_file="simple_qa_pairs.json",
        max_context_length=256,
        max_response_length=64
    ),
    
    "notebook_07_retrieval": RetrievalDataConfig(
        data_dir="data/conversations",
        train_file="simple_qa_pairs.json",
        knowledge_base_file="data/conversations/faq_knowledge.json",
        embedding_dim=384,
        top_k_responses=5
    ),
    
    "notebook_08_seq2seq": Seq2SeqDataConfig(
        data_dir="data/conversations",
        train_file="simple_qa_pairs.json",
        source_max_length=256,
        target_max_length=128,
        teacher_forcing_ratio=0.5,
        batch_size=32
    ),
    
    "notebook_09_attention": Seq2SeqDataConfig(
        data_dir="data/conversations",
        train_file="simple_qa_pairs.json",
        source_max_length=256,
        target_max_length=128,
        teacher_forcing_ratio=0.7,
        batch_size=16
    ),
    
    "notebook_10_transformer": TransformerDataConfig(
        data_dir="data/conversations",
        train_file="simple_qa_pairs.json",
        block_size=512,
        max_position_embeddings=512,
        batch_size=8
    ),
    
    "notebook_11_generative": TransformerDataConfig(
        data_dir="data/conversations",
        train_file="simple_qa_pairs.json",
        block_size=1024,
        generation_max_length=128,
        batch_size=4
    ),
    
    "notebook_12_fine_tuning": TransformerDataConfig(
        data_dir="data/conversations",
        train_file="simple_qa_pairs.json",
        block_size=512,
        generation_max_length=100,
        batch_size=8
    )
}


def get_data_config(config_name: str, **kwargs) -> DataConfig:
    """
    Get a predefined data configuration by name with optional parameter overrides.
    
    Args:
        config_name (str): Name of the configuration to retrieve
        **kwargs: Additional parameters to override in the configuration
    
    Returns:
        DataConfig: Data configuration object with specified parameters
    
    Example:
        >>> config = get_data_config("notebook_08_seq2seq", batch_size=64)
        >>> print(config.batch_size)
        64
    """
    if config_name not in DATA_CONFIGS:
        available_configs = list(DATA_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available_configs}")
    
    config = DATA_CONFIGS[config_name]
    
    # Override with provided kwargs
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        config = type(config).from_dict(config_dict)
    
    return config


def validate_data_paths(config: DataConfig) -> bool:
    """
    Validate that data files specified in configuration exist.
    
    Args:
        config (DataConfig): Data configuration to validate
    
    Returns:
        bool: True if all specified files exist, False otherwise
    """
    files_to_check = []
    
    if config.train_file:
        files_to_check.append(os.path.join(config.data_dir, config.train_file))
    if config.val_file:
        files_to_check.append(os.path.join(config.data_dir, config.val_file))
    if config.test_file:
        files_to_check.append(os.path.join(config.data_dir, config.test_file))
    if config.vocab_file:
        files_to_check.append(config.vocab_file)
    
    # Check for retrieval-specific files
    if hasattr(config, 'knowledge_base_file'):
        files_to_check.append(config.knowledge_base_file)
    
    missing_files = [f for f in files_to_check if not os.path.exists(f)]
    
    if missing_files:
        print(f"Warning: Missing data files: {missing_files}")
        return False
    
    return True


def create_data_directories(config: DataConfig):
    """
    Create necessary data directories based on configuration.
    
    Args:
        config (DataConfig): Data configuration
    """
    directories = [config.data_dir]
    
    # Add subdirectories based on data type
    subdirs = ["conversations", "embeddings", "corpora", "processed"]
    for subdir in subdirs:
        directories.append(os.path.join(config.data_dir, subdir))
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_special_tokens(config: DataConfig) -> Dict[str, str]:
    """
    Get special tokens dictionary from configuration.
    
    Args:
        config (DataConfig): Data configuration
    
    Returns:
        Dict[str, str]: Dictionary mapping token names to token strings
    """
    return {
        "unk_token": config.unk_token,
        "pad_token": config.pad_token,
        "sos_token": config.sos_token,
        "eos_token": config.eos_token
    }


def calculate_vocab_coverage(text_data: List[str], config: DataConfig) -> float:
    """
    Calculate vocabulary coverage for given text data and configuration.
    
    Args:
        text_data (List[str]): List of text strings
        config (DataConfig): Data configuration
    
    Returns:
        float: Vocabulary coverage ratio (0.0 to 1.0)
    """
    # This is a placeholder implementation
    # Real implementation would depend on the tokenizer used
    total_tokens = sum(len(text.split()) for text in text_data)
    unique_tokens = len(set(token for text in text_data for token in text.split()))
    
    coverage = min(unique_tokens / config.vocab_size, 1.0)
    return coverage


# Export commonly used configurations and utilities
__all__ = [
    "DataConfig",
    "TextClassificationDataConfig",
    "LanguageModelDataConfig", 
    "ConversationDataConfig",
    "RetrievalDataConfig",
    "Seq2SeqDataConfig",
    "TransformerDataConfig",
    "DATA_CONFIGS",
    "get_data_config",
    "validate_data_paths",
    "create_data_directories",
    "get_special_tokens",
    "calculate_vocab_coverage"
]