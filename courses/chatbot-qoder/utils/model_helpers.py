"""
Model Helpers Module for Chatbot-Qoder Tutorial Series

This module provides utilities for creating, managing, and working with PyTorch
models throughout the tutorial series. Includes model factories, weight 
initialization, checkpointing, and analysis utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import Dict, Any, Optional, Tuple, List, Union
import os
import json
import math


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device (str): Device specification ("auto", "cpu", "cuda", "mps")
    
    Returns:
        torch.device: Selected device
    
    Example:
        >>> device = get_device("auto")
        >>> print(device)
        device(type='mps')  # On Apple Silicon Mac
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the parameters in a PyTorch model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        Dict[str, int]: Dictionary with parameter counts
    
    Example:
        >>> model = nn.Linear(100, 10)
        >>> params = count_parameters(model)
        >>> print(f"Total parameters: {params['total']}")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params
    }


def initialize_weights(model: nn.Module, method: str = "xavier_uniform") -> None:
    """
    Initialize model weights using specified method.
    
    Args:
        model (nn.Module): Model to initialize
        method (str): Initialization method
    
    Example:
        >>> model = nn.Linear(100, 10)
        >>> initialize_weights(model, "xavier_uniform")
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if method == "xavier_uniform":
                nn.init.xavier_uniform_(param)
            elif method == "xavier_normal":
                nn.init.xavier_normal_(param)
            elif method == "kaiming_uniform":
                nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif method == "kaiming_normal":
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif method == "normal":
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif method == "uniform":
                nn.init.uniform_(param, -0.1, 0.1)
        elif 'bias' in name:
            nn.init.zeros_(param)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for text classification.
    
    A simple feedforward neural network with configurable layers,
    activation functions, and regularization.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 activation: str = "relu",
                 dropout: float = 0.1,
                 batch_norm: bool = True):
        """
        Initialize MLP.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dims (List[int]): List of hidden layer dimensions
            output_dim (int): Output dimension
            activation (str): Activation function name
            dropout (float): Dropout probability
            batch_norm (bool): Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Skip activation and normalization for output layer
            if i < len(dims) - 2:
                # Batch normalization
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                
                # Activation function
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                
                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


class SimpleRNN(nn.Module):
    """
    Simple RNN/LSTM/GRU wrapper for educational purposes.
    
    Provides a clean interface for recurrent neural networks with
    support for bidirectional processing and variable length sequences.
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int = 1,
                 rnn_type: str = "LSTM",
                 bidirectional: bool = False,
                 dropout: float = 0.1,
                 padding_idx: int = 0):
        """
        Initialize RNN model.
        
        Args:
            vocab_size (int): Vocabulary size
            embedding_dim (int): Embedding dimension
            hidden_dim (int): Hidden state dimension
            num_layers (int): Number of RNN layers
            rnn_type (str): Type of RNN ("RNN", "LSTM", "GRU")
            bidirectional (bool): Whether to use bidirectional RNN
            dropout (float): Dropout probability
            padding_idx (int): Padding token index
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # RNN layer
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0,
                              bidirectional=bidirectional)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0,
                             bidirectional=bidirectional)
        else:  # RNN
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0,
                             bidirectional=bidirectional)
        
        # Output projection
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Linear(output_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                input_ids: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the RNN.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len)
            lengths (torch.Tensor, optional): Actual sequence lengths
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # RNN processing
        if lengths is not None:
            # Pack padded sequence for efficiency
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            rnn_output, _ = self.rnn(packed)
            rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)
        else:
            rnn_output, _ = self.rnn(embedded)
        
        # Output projection
        output = self.output_proj(rnn_output)  # (batch_size, seq_len, vocab_size)
        
        return output


class SimpleAttention(nn.Module):
    """
    Simple attention mechanism implementation for educational purposes.
    
    Implements basic additive (Bahdanau) attention between encoder and decoder states.
    """
    
    def __init__(self, hidden_dim: int, attention_dim: int):
        """
        Initialize attention mechanism.
        
        Args:
            hidden_dim (int): Hidden state dimension
            attention_dim (int): Attention projection dimension
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Attention projections
        self.encoder_proj = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.decoder_proj = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.attention_v = nn.Linear(attention_dim, 1, bias=False)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, 
                decoder_hidden: torch.Tensor,
                encoder_outputs: torch.Tensor,
                encoder_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.
        
        Args:
            decoder_hidden (torch.Tensor): Decoder hidden state (batch_size, hidden_dim)
            encoder_outputs (torch.Tensor): Encoder outputs (batch_size, seq_len, hidden_dim)
            encoder_mask (torch.Tensor, optional): Encoder padding mask
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Context vector and attention weights
        """
        batch_size, seq_len, hidden_dim = encoder_outputs.shape
        
        # Project encoder outputs and decoder hidden state
        encoder_proj = self.encoder_proj(encoder_outputs)  # (batch_size, seq_len, attention_dim)
        decoder_proj = self.decoder_proj(decoder_hidden).unsqueeze(1)  # (batch_size, 1, attention_dim)
        
        # Compute attention scores
        scores = self.attention_v(torch.tanh(encoder_proj + decoder_proj))  # (batch_size, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if encoder_mask is not None:
            scores = scores.masked_fill(encoder_mask == 0, -float('inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_dim)
        context = context.squeeze(1)  # (batch_size, hidden_dim)
        
        # Combine context with decoder hidden state
        combined = torch.cat([context, decoder_hidden], dim=1)  # (batch_size, hidden_dim * 2)
        output = self.output_proj(combined)  # (batch_size, hidden_dim)
        
        return output, attention_weights


def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   filepath: str,
                   **kwargs) -> None:
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer state
        epoch (int): Current epoch
        loss (float): Current loss
        filepath (str): Path to save checkpoint
        **kwargs: Additional information to save
    
    Example:
        >>> save_checkpoint(model, optimizer, 10, 0.5, "checkpoint.pth")
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str,
                   model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath (str): Path to checkpoint file
        model (nn.Module): Model to load state into
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
        device (torch.device, optional): Device to load onto
    
    Returns:
        Dict[str, Any]: Checkpoint information
    
    Example:
        >>> info = load_checkpoint("checkpoint.pth", model, optimizer)
        >>> print(f"Loaded epoch: {info['epoch']}")
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded: {filepath}")
    return checkpoint


def create_model(model_type: str,
                config: Dict[str, Any],
                device: Optional[torch.device] = None) -> nn.Module:
    """
    Factory function to create models based on configuration.
    
    Args:
        model_type (str): Type of model to create
        config (Dict[str, Any]): Model configuration
        device (torch.device, optional): Device to place model on
    
    Returns:
        nn.Module: Created model
    
    Example:
        >>> config = {"input_dim": 100, "hidden_dims": [64, 32], "output_dim": 10}
        >>> model = create_model("mlp", config)
    """
    if device is None:
        device = get_device()
    
    if model_type == "mlp":
        model = MLP(**config)
    elif model_type == "rnn":
        model = SimpleRNN(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(device)
    return model


def freeze_parameters(model: nn.Module, freeze_embeddings: bool = True) -> None:
    """
    Freeze model parameters for fine-tuning.
    
    Args:
        model (nn.Module): Model to freeze parameters
        freeze_embeddings (bool): Whether to freeze embedding layers
    
    Example:
        >>> freeze_parameters(model, freeze_embeddings=True)
    """
    for name, param in model.named_parameters():
        if freeze_embeddings and 'embedding' in name.lower():
            param.requires_grad = False
        elif 'classifier' not in name.lower() and 'output' not in name.lower():
            param.requires_grad = False


def get_model_summary(model: nn.Module) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model (nn.Module): Model to summarize
    
    Returns:
        str: Model summary string
    
    Example:
        >>> summary = get_model_summary(model)
        >>> print(summary)
    """
    summary = []
    summary.append(f"Model: {model.__class__.__name__}")
    summary.append("=" * 50)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
            summary.append(f"{name:30} {list(param.shape):20} {param_count:10,} (trainable)")
        else:
            summary.append(f"{name:30} {list(param.shape):20} {param_count:10,} (frozen)")
    
    summary.append("=" * 50)
    summary.append(f"Total parameters: {total_params:,}")
    summary.append(f"Trainable parameters: {trainable_params:,}")
    summary.append(f"Frozen parameters: {total_params - trainable_params:,}")
    
    return "\n".join(summary)


# Export commonly used functions and classes
__all__ = [
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
    "get_model_summary"
]