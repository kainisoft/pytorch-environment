"""
Utilities Package for Chatbot-Qoder Tutorial Series

This package provides comprehensive utilities for text processing, model management,
training, evaluation, and chatbot-specific functionality throughout the tutorial series.
"""

from .text_utils import (
    SimpleTokenizer,
    CharacterTokenizer,
    clean_text,
    build_vocabulary,
    encode_sequences,
    pad_sequences,
    create_attention_mask,
    text_to_sequences,
    calculate_text_statistics
)

from .model_helpers import (
    get_device,
    count_parameters,
    initialize_weights,
    MLP,
    SimpleRNN,
    SimpleAttention,
    save_checkpoint,
    load_checkpoint,
    create_model,
    freeze_parameters,
    get_model_summary
)

from .training_helpers import (
    EarlyStopping,
    TrainingLogger,
    get_optimizer,
    get_scheduler,
    train_epoch,
    evaluate_model,
    train_model,
    calculate_perplexity
)

from .evaluation_helpers import (
    bleu_score,
    rouge_score,
    perplexity,
    response_relevance,
    conversation_quality,
    calculate_metrics,
    human_evaluation_interface
)

from .chatbot_helpers import (
    ConversationContext,
    ResponseGenerator,
    RuleBasedChatbot,
    SafetyFilter,
    interactive_chat,
    load_conversation_data,
    save_conversation_log
)

__all__ = [
    # Text utilities
    "SimpleTokenizer",
    "CharacterTokenizer",
    "clean_text",
    "build_vocabulary",
    "encode_sequences",
    "pad_sequences",
    "create_attention_mask",
    "text_to_sequences",
    "calculate_text_statistics",
    
    # Model helpers
    "get_device",
    "count_parameters",
    "initialize_weights",
    "MLP",
    "SimpleRNN",
    "SimpleAttention",
    "save_checkpoint",
    "load_checkpoint",
    "create_model",
    "freeze_parameters",
    "get_model_summary",
    
    # Training helpers
    "EarlyStopping",
    "TrainingLogger",
    "get_optimizer",
    "get_scheduler",
    "train_epoch",
    "evaluate_model",
    "train_model",
    "calculate_perplexity",
    
    # Evaluation helpers
    "bleu_score",
    "rouge_score",
    "perplexity",
    "response_relevance",
    "conversation_quality",
    "calculate_metrics",
    "human_evaluation_interface",
    
    # Chatbot helpers
    "ConversationContext",
    "ResponseGenerator",
    "RuleBasedChatbot",
    "SafetyFilter",
    "interactive_chat",
    "load_conversation_data",
    "save_conversation_log"
]