"""
Model Utilities for PyTorch Chatbot Tutorial

This module provides educational model handling utilities for the chatbot tutorial series.
All functions include comprehensive documentation and explanations to help learners
understand model architecture, initialization, and management concepts.

Educational Focus:
    - Clear explanations of model architecture patterns
    - Detailed comments on PyTorch model operations
    - Examples of model saving/loading best practices
    - Debugging tips for common model issues
"""

import torch
import torch.nn as nn
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a PyTorch model with educational insights.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dict containing parameter counts and analysis
        
    Educational Purpose:
        - Demonstrates how to analyze model complexity
        - Shows the difference between trainable and total parameters
        - Helps understand memory requirements and training time implications
        
    Learning Notes:
        - More parameters generally mean more model capacity
        - Trainable parameters are updated during training
        - Parameter count affects memory usage and training speed
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'memory_mb_estimate': (total_params * 4) / (1024 * 1024)  # Assuming float32
    }
    
    print(f"Educational Model Analysis:")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")
    print(f"- Non-trainable parameters: {param_info['non_trainable_parameters']:,}")
    print(f"- Estimated memory (MB): {param_info['memory_mb_estimate']:.2f}")
    
    return param_info


def initialize_weights(model: nn.Module, method: str = 'xavier_uniform') -> None:
    """
    Initialize model weights with educational explanations.
    
    Args:
        model: PyTorch model to initialize
        method: Initialization method ('xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
        
    Educational Purpose:
        - Demonstrates different weight initialization strategies
        - Explains why proper initialization is crucial for training
        - Shows how to apply initialization to different layer types
        
    Learning Notes:
        - Poor initialization can lead to vanishing/exploding gradients
        - Different activation functions work better with different initializations
        - Xavier initialization works well with tanh/sigmoid activations
        - Kaiming initialization works well with ReLU activations
    """
    print(f"Educational Weight Initialization:")
    print(f"- Method: {method}")
    print(f"- Initializing weights for better training convergence")
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:  # Linear layers, Conv layers
                if method == 'xavier_uniform':
                    nn.init.xavier_uniform_(param)
                elif method == 'xavier_normal':
                    nn.init.xavier_normal_(param)
                elif method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif method == 'kaiming_normal':
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
                
                print(f"  - Initialized {name}: {param.shape}")
                
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
            print(f"  - Initialized {name} to zero: {param.shape}")


def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                         epoch: int, loss: float, save_path: str,
                         additional_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Save model checkpoint with comprehensive information for educational purposes.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save
        epoch: Current training epoch
        loss: Current loss value
        save_path: Path to save the checkpoint
        additional_info: Additional information to save with checkpoint
        
    Educational Purpose:
        - Demonstrates proper model checkpointing practices
        - Shows what information should be saved for resuming training
        - Explains the importance of saving optimizer state
        
    Learning Notes:
        - Checkpoints allow resuming training from specific points
        - Saving optimizer state preserves learning rate schedules and momentum
        - Additional metadata helps track training progress and experiments
    """
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'model_architecture': str(model),
        'parameter_count': count_parameters(model)
    }
    
    # Add additional information if provided
    if additional_info:
        checkpoint.update(additional_info)
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    
    print(f"Educational Checkpoint Saved:")
    print(f"- Path: {save_path}")
    print(f"- Epoch: {epoch}")
    print(f"- Loss: {loss:.4f}")
    print(f"- Timestamp: {checkpoint['timestamp']}")
    print(f"- Model parameters: {checkpoint['parameter_count']['total_parameters']:,}")


def load_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                         checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """
    Load model checkpoint with educational error handling and information display.
    
    Args:
        model: PyTorch model to load state into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model onto
        
    Returns:
        Dict containing checkpoint information
        
    Educational Purpose:
        - Demonstrates proper checkpoint loading practices
        - Shows how to handle device compatibility issues
        - Explains the importance of model architecture matching
        
    Learning Notes:
        - Model architecture must match the saved checkpoint
        - Device compatibility is crucial for loading checkpoints
        - Checkpoint loading allows resuming training or inference
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Educational Checkpoint Loaded:")
        print(f"- Path: {checkpoint_path}")
        print(f"- Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"- Loss: {checkpoint.get('loss', 'Unknown')}")
        print(f"- Timestamp: {checkpoint.get('timestamp', 'Unknown')}")
        
        if 'parameter_count' in checkpoint:
            param_info = checkpoint['parameter_count']
            print(f"- Parameters: {param_info.get('total_parameters', 'Unknown'):,}")
        
        return checkpoint
        
    except FileNotFoundError as e:
        print(f"Educational Error: Checkpoint file not found - {e}")
        print("Learning Note: Ensure the checkpoint path is correct")
        print("Solution: Check file path and verify checkpoint exists")
        raise
        
    except RuntimeError as e:
        print(f"Educational Error: Model loading failed - {e}")
        print("Learning Note: This often occurs when model architecture doesn't match")
        print("Solution: Ensure model definition matches the saved checkpoint")
        raise


def get_model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> str:
    """
    Generate a comprehensive model summary for educational purposes.
    
    Args:
        model: PyTorch model to summarize
        input_size: Expected input size (batch_size, sequence_length, features)
        
    Returns:
        String containing detailed model summary
        
    Educational Purpose:
        - Provides detailed view of model architecture
        - Shows layer-by-layer parameter counts and output shapes
        - Helps understand data flow through the model
        
    Learning Notes:
        - Model summaries help debug architecture issues
        - Understanding data shapes is crucial for model design
        - Parameter distribution affects training dynamics
    """
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("EDUCATIONAL MODEL SUMMARY")
    summary_lines.append("=" * 80)
    
    # Model architecture
    summary_lines.append(f"Model Architecture: {model.__class__.__name__}")
    summary_lines.append(f"Expected Input Size: {input_size}")
    summary_lines.append("-" * 80)
    
    # Layer information
    summary_lines.append(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<15}")
    summary_lines.append("=" * 80)
    
    total_params = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            total_params += param_count
            
            layer_info = f"{name} ({module.__class__.__name__})"
            summary_lines.append(f"{layer_info:<25} {'TBD':<20} {param_count:<15,}")
    
    summary_lines.append("=" * 80)
    
    # Parameter summary
    param_info = count_parameters(model)
    summary_lines.append(f"Total params: {param_info['total_parameters']:,}")
    summary_lines.append(f"Trainable params: {param_info['trainable_parameters']:,}")
    summary_lines.append(f"Non-trainable params: {param_info['non_trainable_parameters']:,}")
    summary_lines.append(f"Estimated size (MB): {param_info['memory_mb_estimate']:.2f}")
    summary_lines.append("=" * 80)
    
    # Educational notes
    summary_lines.append("EDUCATIONAL NOTES:")
    summary_lines.append("- Parameter count affects model capacity and training time")
    summary_lines.append("- Memory usage scales with model size and batch size")
    summary_lines.append("- Deeper models can capture more complex patterns")
    summary_lines.append("- Balance model complexity with available data and compute")
    summary_lines.append("=" * 80)
    
    summary = "\n".join(summary_lines)
    print(summary)
    return summary


def freeze_layers(model: nn.Module, layer_names: list) -> None:
    """
    Freeze specific layers for transfer learning with educational explanations.
    
    Args:
        model: PyTorch model containing layers to freeze
        layer_names: List of layer names to freeze
        
    Educational Purpose:
        - Demonstrates transfer learning techniques
        - Shows how to selectively train parts of a model
        - Explains the benefits of freezing pre-trained layers
        
    Learning Notes:
        - Freezing prevents weight updates during training
        - Useful for transfer learning and fine-tuning
        - Can reduce training time and prevent overfitting
        - Pre-trained features often generalize well
    """
    frozen_count = 0
    total_params = 0
    
    print(f"Educational Layer Freezing:")
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # Check if this parameter belongs to a layer we want to freeze
        should_freeze = any(layer_name in name for layer_name in layer_names)
        
        if should_freeze:
            param.requires_grad = False
            frozen_count += param.numel()
            print(f"- Frozen: {name} ({param.numel():,} parameters)")
        else:
            param.requires_grad = True
            print(f"- Trainable: {name} ({param.numel():,} parameters)")
    
    print(f"\nFreezing Summary:")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Frozen parameters: {frozen_count:,}")
    print(f"- Trainable parameters: {total_params - frozen_count:,}")
    print(f"- Frozen percentage: {(frozen_count / total_params) * 100:.1f}%")


def compare_models(model1: nn.Module, model2: nn.Module, 
                  names: Tuple[str, str] = ("Model 1", "Model 2")) -> None:
    """
    Compare two models for educational analysis.
    
    Args:
        model1: First model to compare
        model2: Second model to compare
        names: Names for the models in the comparison
        
    Educational Purpose:
        - Demonstrates model comparison techniques
        - Shows how to analyze architectural differences
        - Helps understand trade-offs between different designs
        
    Learning Notes:
        - Model comparison helps choose appropriate architectures
        - Parameter count affects training time and memory usage
        - Architecture differences impact model capabilities
    """
    print(f"Educational Model Comparison:")
    print("=" * 60)
    
    # Get parameter information for both models
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)
    
    # Comparison table
    print(f"{'Metric':<25} {names[0]:<15} {names[1]:<15}")
    print("-" * 60)
    print(f"{'Total Parameters':<25} {params1['total_parameters']:<15,} {params2['total_parameters']:<15,}")
    print(f"{'Trainable Parameters':<25} {params1['trainable_parameters']:<15,} {params2['trainable_parameters']:<15,}")
    print(f"{'Memory (MB)':<25} {params1['memory_mb_estimate']:<15.2f} {params2['memory_mb_estimate']:<15.2f}")
    
    # Analysis
    print("\nEducational Analysis:")
    if params1['total_parameters'] > params2['total_parameters']:
        ratio = params1['total_parameters'] / params2['total_parameters']
        print(f"- {names[0]} has {ratio:.1f}x more parameters than {names[1]}")
        print(f"- {names[0]} likely has higher capacity but requires more compute")
    elif params2['total_parameters'] > params1['total_parameters']:
        ratio = params2['total_parameters'] / params1['total_parameters']
        print(f"- {names[1]} has {ratio:.1f}x more parameters than {names[0]}")
        print(f"- {names[1]} likely has higher capacity but requires more compute")
    else:
        print(f"- Both models have similar parameter counts")
    
    print("- Consider data size, compute resources, and task complexity when choosing")
    print("=" * 60)