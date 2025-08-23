"""
Model Configuration Classes for PyTorch Chatbot Tutorial

This module provides educational configuration classes for model parameters.
All configurations include comprehensive documentation and validation to help learners
understand model architecture decisions and hyperparameter effects.

Educational Focus:
    - Clear explanations of each parameter's role and impact
    - Detailed comments on parameter interactions and trade-offs
    - Examples of different configuration presets for various use cases
    - Validation methods with educational error messages
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import torch


@dataclass
class ModelConfig:
    """
    Educational configuration class for model parameters.
    
    This class demonstrates how to structure model configurations with proper
    validation and educational explanations for each parameter.
    
    Educational Purpose:
        - Shows best practices for model configuration management
        - Explains the role and impact of each hyperparameter
        - Demonstrates parameter validation techniques
        - Provides presets for different model complexities
        
    Learning Notes:
        - Larger models have more capacity but require more data and compute
        - Embedding dimension affects representation quality
        - Hidden dimension controls model expressiveness
        - Dropout helps prevent overfitting
        - Sequence length limits affect memory usage and context
    """
    
    # Core architecture parameters
    vocab_size: int
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    
    # Sequence parameters
    max_sequence_length: int = 512
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    
    # Architecture choices
    use_attention: bool = True
    use_layer_norm: bool = True
    use_residual_connections: bool = True
    activation_function: str = "relu"  # "relu", "gelu", "swish"
    
    # Advanced parameters
    tie_word_embeddings: bool = False
    use_positional_encoding: bool = True
    max_position_embeddings: int = 1024
    
    # Educational metadata
    model_name: str = "educational_chatbot"
    description: str = "Educational chatbot model for learning purposes"
    complexity_level: str = "medium"  # "simple", "medium", "complex"
    
    def __post_init__(self):
        """
        Validate configuration parameters with educational explanations.
        
        Educational Purpose:
            - Demonstrates proper parameter validation
            - Shows common configuration mistakes and how to catch them
            - Explains parameter relationships and constraints
        """
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate configuration parameters with educational error messages.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid with educational explanation
            
        Educational Purpose:
            - Shows how to validate model configurations
            - Explains why certain parameter combinations are problematic
            - Demonstrates defensive programming practices
        """
        errors = []
        
        # Validate basic parameters
        if self.vocab_size <= 0:
            errors.append("vocab_size must be positive (represents number of unique tokens)")
        
        if self.embedding_dim <= 0:
            errors.append("embedding_dim must be positive (dimension of token representations)")
        
        if self.hidden_dim <= 0:
            errors.append("hidden_dim must be positive (dimension of hidden layers)")
        
        if self.num_layers <= 0:
            errors.append("num_layers must be positive (number of transformer/RNN layers)")
        
        if not 0 <= self.dropout <= 1:
            errors.append("dropout must be between 0 and 1 (probability of dropping neurons)")
        
        # Validate attention parameters
        if self.use_attention and self.num_heads <= 0:
            errors.append("num_heads must be positive when using attention")
        
        if self.use_attention and self.hidden_dim % self.num_heads != 0:
            errors.append(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads}) for multi-head attention")
        
        # Validate sequence parameters
        if self.max_sequence_length <= 0:
            errors.append("max_sequence_length must be positive (maximum input length)")
        
        if self.use_positional_encoding and self.max_position_embeddings < self.max_sequence_length:
            errors.append("max_position_embeddings must be >= max_sequence_length for positional encoding")
        
        # Validate activation function
        valid_activations = ["relu", "gelu", "swish", "tanh", "sigmoid"]
        if self.activation_function.lower() not in valid_activations:
            errors.append(f"activation_function must be one of {valid_activations}")
        
        # Validate complexity level
        valid_complexity = ["simple", "medium", "complex"]
        if self.complexity_level.lower() not in valid_complexity:
            errors.append(f"complexity_level must be one of {valid_complexity}")
        
        if errors:
            error_msg = "Educational Configuration Validation Errors:\n"
            for i, error in enumerate(errors, 1):
                error_msg += f"  {i}. {error}\n"
            error_msg += "\nLearning Note: Proper validation prevents runtime errors and ensures model correctness."
            raise ValueError(error_msg)
        
        print("✓ Educational Configuration Validation Passed")
        return True
    
    def get_parameter_count_estimate(self) -> Dict[str, int]:
        """
        Estimate the number of parameters for this configuration.
        
        Returns:
            Dict containing parameter count breakdown
            
        Educational Purpose:
            - Shows how to estimate model size before training
            - Explains where parameters come from in neural networks
            - Helps understand memory and compute requirements
            
        Learning Notes:
            - Embedding layers contribute vocab_size * embedding_dim parameters
            - Linear layers contribute input_dim * output_dim + bias parameters
            - Attention mechanisms add significant parameters in transformers
        """
        params = {}
        
        # Embedding parameters
        params['token_embeddings'] = self.vocab_size * self.embedding_dim
        
        if self.use_positional_encoding:
            params['position_embeddings'] = self.max_position_embeddings * self.embedding_dim
        
        # Transformer/RNN layer parameters (simplified estimate)
        if self.use_attention:
            # Multi-head attention parameters per layer
            attention_params_per_layer = (
                4 * self.hidden_dim * self.hidden_dim +  # Q, K, V, O projections
                4 * self.hidden_dim  # biases
            )
            
            # Feed-forward parameters per layer
            ff_params_per_layer = (
                2 * self.hidden_dim * self.hidden_dim * 4 +  # Two linear layers (4x expansion)
                self.hidden_dim * 4 + self.hidden_dim  # biases
            )
            
            params['attention_layers'] = self.num_layers * attention_params_per_layer
            params['feedforward_layers'] = self.num_layers * ff_params_per_layer
            
            if self.use_layer_norm:
                params['layer_norm'] = self.num_layers * 2 * self.hidden_dim * 2  # 2 layer norms per layer
        else:
            # RNN/LSTM parameters (simplified)
            rnn_params_per_layer = 4 * self.hidden_dim * (self.embedding_dim + self.hidden_dim)
            params['rnn_layers'] = self.num_layers * rnn_params_per_layer
        
        # Output layer parameters
        params['output_layer'] = self.hidden_dim * self.vocab_size + self.vocab_size
        
        # Calculate totals
        params['total_estimated'] = sum(params.values())
        params['memory_mb_estimate'] = (params['total_estimated'] * 4) / (1024 * 1024)  # Assuming float32
        
        return params
    
    def print_summary(self) -> None:
        """
        Print a comprehensive summary of the configuration with educational insights.
        
        Educational Purpose:
            - Provides clear overview of model configuration
            - Shows parameter estimates and memory requirements
            - Explains the implications of different settings
        """
        print("=" * 60)
        print("EDUCATIONAL MODEL CONFIGURATION SUMMARY")
        print("=" * 60)
        
        print(f"Model Name: {self.model_name}")
        print(f"Description: {self.description}")
        print(f"Complexity Level: {self.complexity_level.title()}")
        print()
        
        print("Core Architecture:")
        print(f"  - Vocabulary Size: {self.vocab_size:,}")
        print(f"  - Embedding Dimension: {self.embedding_dim}")
        print(f"  - Hidden Dimension: {self.hidden_dim}")
        print(f"  - Number of Layers: {self.num_layers}")
        print(f"  - Dropout Rate: {self.dropout}")
        print()
        
        if self.use_attention:
            print("Attention Configuration:")
            print(f"  - Number of Heads: {self.num_heads}")
            print(f"  - Head Dimension: {self.hidden_dim // self.num_heads}")
            print(f"  - Use Layer Norm: {self.use_layer_norm}")
            print(f"  - Use Residual Connections: {self.use_residual_connections}")
            print()
        
        print("Sequence Configuration:")
        print(f"  - Max Sequence Length: {self.max_sequence_length}")
        print(f"  - Use Positional Encoding: {self.use_positional_encoding}")
        if self.use_positional_encoding:
            print(f"  - Max Position Embeddings: {self.max_position_embeddings}")
        print()
        
        print("Special Tokens:")
        print(f"  - Padding Token ID: {self.pad_token_id}")
        print(f"  - End-of-Sequence Token ID: {self.eos_token_id}")
        print(f"  - Beginning-of-Sequence Token ID: {self.bos_token_id}")
        print()
        
        # Parameter estimates
        param_counts = self.get_parameter_count_estimate()
        print("Parameter Estimates:")
        for component, count in param_counts.items():
            if component not in ['total_estimated', 'memory_mb_estimate']:
                print(f"  - {component.replace('_', ' ').title()}: {count:,}")
        print(f"  - Total Estimated: {param_counts['total_estimated']:,}")
        print(f"  - Estimated Memory (MB): {param_counts['memory_mb_estimate']:.2f}")
        print()
        
        # Educational insights
        print("Educational Insights:")
        if self.complexity_level == "simple":
            print("  - Simple configuration good for learning and small datasets")
            print("  - Fast training but limited model capacity")
        elif self.complexity_level == "medium":
            print("  - Balanced configuration for most educational purposes")
            print("  - Good trade-off between capacity and training time")
        else:
            print("  - Complex configuration for advanced experiments")
            print("  - High capacity but requires more data and compute")
        
        print(f"  - Model will use ~{param_counts['memory_mb_estimate']:.0f}MB of GPU memory")
        print("=" * 60)
    
    @classmethod
    def create_simple_config(cls, vocab_size: int) -> 'ModelConfig':
        """
        Create a simple configuration for educational purposes.
        
        Args:
            vocab_size: Size of the vocabulary
            
        Returns:
            ModelConfig: Simple configuration suitable for learning
            
        Educational Purpose:
            - Provides a starting point for beginners
            - Shows minimal viable configuration
            - Fast training for quick experimentation
        """
        return cls(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            max_sequence_length=256,
            complexity_level="simple",
            model_name="simple_educational_chatbot",
            description="Simple configuration for learning PyTorch and NLP basics"
        )
    
    @classmethod
    def create_medium_config(cls, vocab_size: int) -> 'ModelConfig':
        """
        Create a medium complexity configuration for educational purposes.
        
        Args:
            vocab_size: Size of the vocabulary
            
        Returns:
            ModelConfig: Medium configuration for balanced learning
            
        Educational Purpose:
            - Provides good balance of capacity and training speed
            - Suitable for most educational experiments
            - Shows realistic model sizes for practical applications
        """
        return cls(
            vocab_size=vocab_size,
            embedding_dim=256,
            hidden_dim=512,
            num_layers=4,
            num_heads=8,
            dropout=0.1,
            max_sequence_length=512,
            complexity_level="medium",
            model_name="medium_educational_chatbot",
            description="Medium configuration for comprehensive learning experience"
        )
    
    @classmethod
    def create_complex_config(cls, vocab_size: int) -> 'ModelConfig':
        """
        Create a complex configuration for advanced educational purposes.
        
        Args:
            vocab_size: Size of the vocabulary
            
        Returns:
            ModelConfig: Complex configuration for advanced learning
            
        Educational Purpose:
            - Shows larger model architectures
            - Suitable for advanced experiments and research
            - Demonstrates scaling effects in deep learning
        """
        return cls(
            vocab_size=vocab_size,
            embedding_dim=512,
            hidden_dim=1024,
            num_layers=6,
            num_heads=16,
            dropout=0.1,
            max_sequence_length=1024,
            complexity_level="complex",
            model_name="complex_educational_chatbot",
            description="Complex configuration for advanced learning and experimentation"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dict containing all configuration parameters
            
        Educational Purpose:
            - Shows how to serialize configurations
            - Useful for saving/loading model settings
            - Demonstrates configuration management best practices
        """
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'max_sequence_length': self.max_sequence_length,
            'pad_token_id': self.pad_token_id,
            'eos_token_id': self.eos_token_id,
            'bos_token_id': self.bos_token_id,
            'use_attention': self.use_attention,
            'use_layer_norm': self.use_layer_norm,
            'use_residual_connections': self.use_residual_connections,
            'activation_function': self.activation_function,
            'tie_word_embeddings': self.tie_word_embeddings,
            'use_positional_encoding': self.use_positional_encoding,
            'max_position_embeddings': self.max_position_embeddings,
            'model_name': self.model_name,
            'description': self.description,
            'complexity_level': self.complexity_level
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            ModelConfig: Configuration object
            
        Educational Purpose:
            - Shows how to deserialize configurations
            - Useful for loading saved model settings
            - Demonstrates configuration restoration
        """
        return cls(**config_dict)