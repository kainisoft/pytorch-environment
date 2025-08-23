"""
Training Utilities for PyTorch Chatbot Tutorial

This module provides educational training utilities for the chatbot tutorial series.
All functions include comprehensive documentation and explanations to help learners
understand training loops, optimization, and evaluation concepts.

Educational Focus:
    - Clear explanations of training loop components
    - Detailed comments on optimization and loss calculation
    - Examples of evaluation metrics and their interpretation
    - Debugging tips for common training issues
"""

import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple, Optional, Callable, Any
from torch.utils.data import DataLoader
import math


class EducationalTrainer:
    """
    Educational trainer class that demonstrates training loop best practices.
    
    This class provides a comprehensive training framework with detailed explanations
    of each component and step in the training process.
    
    Educational Purpose:
        - Demonstrates proper training loop structure
        - Shows how to handle training and validation phases
        - Explains metric tracking and logging
        - Provides debugging capabilities for training issues
    """
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 criterion: nn.Module, device: torch.device):
        """
        Initialize the educational trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer for parameter updates
            criterion: Loss function for training
            device: Device to run training on (CPU/GPU)
            
        Educational Notes:
            - Model should be moved to device before training
            - Optimizer manages parameter updates and learning rates
            - Criterion defines what the model should optimize for
            - Device selection affects training speed significantly
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Training history for educational analysis
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        print(f"Educational Trainer Initialized:")
        print(f"- Model: {model.__class__.__name__}")
        print(f"- Optimizer: {optimizer.__class__.__name__}")
        print(f"- Loss function: {criterion.__class__.__name__}")
        print(f"- Device: {device}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch with educational explanations.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Dict containing training metrics for the epoch
            
        Educational Purpose:
            - Demonstrates the training loop structure
            - Shows forward pass, loss calculation, and backpropagation
            - Explains gradient accumulation and parameter updates
        """
        self.model.train()  # Set model to training mode
        
        total_loss = 0.0
        num_batches = len(train_loader)
        start_time = time.time()
        
        print(f"\nEducational Training - Epoch {epoch}:")
        print(f"- Processing {num_batches} batches")
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            # Educational Note: Data must be on same device as model
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Zero gradients from previous iteration
            # Educational Note: PyTorch accumulates gradients, so we must clear them
            self.optimizer.zero_grad()
            
            # Forward pass
            # Educational Note: Model processes input and produces predictions
            outputs = self.model(input_ids)
            
            # Calculate loss
            # Educational Note: Loss measures how far predictions are from targets
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            
            # Backward pass (compute gradients)
            # Educational Note: Backpropagation calculates gradients for each parameter
            loss.backward()
            
            # Update parameters
            # Educational Note: Optimizer uses gradients to update model parameters
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Educational progress reporting
            if batch_idx % 10 == 0:
                current_loss = loss.item()
                progress = (batch_idx / num_batches) * 100
                print(f"  Batch {batch_idx}/{num_batches} ({progress:.1f}%) - Loss: {current_loss:.4f}")
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        # Store in history
        self.history['train_loss'].append(avg_loss)
        self.history['epoch_times'].append(epoch_time)
        
        print(f"- Epoch {epoch} completed in {epoch_time:.2f}s")
        print(f"- Average training loss: {avg_loss:.4f}")
        
        return {'loss': avg_loss, 'time': epoch_time}
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate the model for one epoch with educational explanations.
        
        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number
            
        Returns:
            Dict containing validation metrics for the epoch
            
        Educational Purpose:
            - Demonstrates proper validation procedures
            - Shows the difference between training and evaluation modes
            - Explains why gradients are disabled during validation
        """
        self.model.eval()  # Set model to evaluation mode
        
        total_loss = 0.0
        num_batches = len(val_loader)
        
        print(f"\nEducational Validation - Epoch {epoch}:")
        
        # Disable gradient computation for efficiency and correctness
        # Educational Note: Validation doesn't update parameters, so no gradients needed
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                # Forward pass only (no backward pass in validation)
                outputs = self.model(input_ids)
                
                # Calculate loss
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                
                total_loss += loss.item()
        
        # Calculate validation metrics
        avg_loss = total_loss / num_batches
        
        # Store in history
        self.history['val_loss'].append(avg_loss)
        
        print(f"- Validation completed")
        print(f"- Average validation loss: {avg_loss:.4f}")
        
        return {'loss': avg_loss}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Complete training loop with educational monitoring and explanations.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            save_path: Optional path to save best model
            
        Returns:
            Dict containing complete training history
            
        Educational Purpose:
            - Demonstrates complete training workflow
            - Shows epoch-by-epoch progress monitoring
            - Explains early stopping and model checkpointing concepts
        """
        print(f"Educational Training Started:")
        print(f"- Total epochs: {epochs}")
        print(f"- Training batches per epoch: {len(train_loader)}")
        print(f"- Validation batches per epoch: {len(val_loader)}")
        print("=" * 60)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            print(f"\nEPOCH {epoch}/{epochs}")
            print("-" * 40)
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Educational analysis
            current_val_loss = val_metrics['loss']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                print(f"✓ New best validation loss: {best_val_loss:.4f}")
                
                # Save best model if path provided
                if save_path:
                    from .model_utils import save_model_checkpoint
                    save_model_checkpoint(
                        self.model, self.optimizer, epoch, 
                        current_val_loss, save_path,
                        {'best_epoch': epoch, 'training_history': self.history}
                    )
            else:
                print(f"  Validation loss: {current_val_loss:.4f} (best: {best_val_loss:.4f})")
            
            # Educational insights
            if len(self.history['train_loss']) > 1:
                train_trend = self.history['train_loss'][-1] - self.history['train_loss'][-2]
                val_trend = self.history['val_loss'][-1] - self.history['val_loss'][-2]
                
                print(f"Educational Analysis:")
                print(f"- Training loss trend: {'↓' if train_trend < 0 else '↑'} {abs(train_trend):.4f}")
                print(f"- Validation loss trend: {'↓' if val_trend < 0 else '↑'} {abs(val_trend):.4f}")
                
                # Check for overfitting
                if train_trend < 0 and val_trend > 0:
                    print("⚠️  Potential overfitting detected (train↓, val↑)")
        
        print("\n" + "=" * 60)
        print("Educational Training Complete!")
        print(f"- Best validation loss: {best_val_loss:.4f}")
        print(f"- Total training time: {sum(self.history['epoch_times']):.2f}s")
        
        return self.history


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss with educational explanation.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity score
        
    Educational Purpose:
        - Explains the relationship between loss and perplexity
        - Shows how to interpret perplexity values
        - Demonstrates common NLP evaluation metrics
        
    Learning Notes:
        - Perplexity is 2^(cross-entropy loss)
        - Lower perplexity indicates better language modeling
        - Perplexity represents average branching factor
        - Human-level perplexity varies by task and domain
    """
    perplexity = math.exp(loss)
    
    print(f"Educational Perplexity Calculation:")
    print(f"- Cross-entropy loss: {loss:.4f}")
    print(f"- Perplexity: {perplexity:.2f}")
    print(f"- Interpretation: Model is uncertain between ~{perplexity:.0f} choices on average")
    
    return perplexity


def calculate_bleu_score(predictions: List[str], references: List[str]) -> float:
    """
    Calculate BLEU score with educational implementation and explanation.
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        
    Returns:
        BLEU score (0-1, higher is better)
        
    Educational Purpose:
        - Demonstrates BLEU score calculation from scratch
        - Explains n-gram matching concepts
        - Shows how to evaluate text generation quality
        
    Learning Notes:
        - BLEU measures n-gram overlap between prediction and reference
        - Higher BLEU scores indicate better text quality
        - BLEU has limitations and should be used with other metrics
        - Perfect BLEU score (1.0) means exact match
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    total_score = 0.0
    
    print(f"Educational BLEU Score Calculation:")
    print(f"- Comparing {len(predictions)} prediction-reference pairs")
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # Simple unigram BLEU for educational purposes
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        
        if len(pred_words) == 0:
            score = 0.0
        else:
            matches = len(pred_words.intersection(ref_words))
            score = matches / len(pred_words)
        
        total_score += score
        
        if i < 3:  # Show first few examples
            print(f"  Example {i+1}: {score:.3f} ({matches}/{len(pred_words)} words match)")
    
    avg_bleu = total_score / len(predictions)
    
    print(f"- Average BLEU score: {avg_bleu:.4f}")
    print(f"- Interpretation: {avg_bleu*100:.1f}% average word overlap")
    
    return avg_bleu


def evaluate_model_comprehensive(model: nn.Module, test_loader: DataLoader,
                               criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """
    Comprehensive model evaluation with multiple metrics and educational insights.
    
    Args:
        model: Trained PyTorch model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function for evaluation
        device: Device to run evaluation on
        
    Returns:
        Dict containing comprehensive evaluation metrics
        
    Educational Purpose:
        - Demonstrates comprehensive model evaluation
        - Shows multiple evaluation metrics and their interpretation
        - Explains the importance of thorough testing
        
    Learning Notes:
        - Multiple metrics provide different perspectives on model performance
        - Test set evaluation gives unbiased performance estimate
        - Different metrics may disagree - consider task requirements
    """
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    predictions = []
    references = []
    
    print(f"Educational Comprehensive Evaluation:")
    print(f"- Evaluating on {len(test_loader)} batches")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Forward pass
            outputs = model(input_ids)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            total_loss += loss.item()
            total_samples += input_ids.size(0)
            
            # For text generation evaluation (simplified)
            # In practice, you'd use proper decoding methods
            pred_tokens = torch.argmax(outputs, dim=-1)
            
            # Note: This is a simplified example - real implementation would
            # need proper tokenizer decoding
            
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    perplexity = calculate_perplexity(avg_loss)
    
    metrics = {
        'test_loss': avg_loss,
        'perplexity': perplexity,
        'total_samples': total_samples
    }
    
    print(f"\nEducational Evaluation Results:")
    print(f"- Test loss: {avg_loss:.4f}")
    print(f"- Perplexity: {perplexity:.2f}")
    print(f"- Total samples evaluated: {total_samples}")
    
    print(f"\nEducational Interpretation:")
    if perplexity < 10:
        print("- Excellent: Very low perplexity indicates strong language modeling")
    elif perplexity < 50:
        print("- Good: Reasonable perplexity for most applications")
    elif perplexity < 100:
        print("- Fair: Model shows some language understanding")
    else:
        print("- Poor: High perplexity suggests model needs improvement")
    
    return metrics


def learning_rate_scheduler_step(optimizer: torch.optim.Optimizer, 
                               scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                               metric: Optional[float] = None) -> float:
    """
    Handle learning rate scheduling with educational explanations.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler (optional)
        metric: Metric for schedulers that need it (e.g., ReduceLROnPlateau)
        
    Returns:
        Current learning rate
        
    Educational Purpose:
        - Demonstrates learning rate scheduling concepts
        - Shows different scheduler types and their effects
        - Explains when and why to adjust learning rates
        
    Learning Notes:
        - Learning rate scheduling can improve convergence
        - Different schedulers work better for different problems
        - Monitoring learning rate helps debug training issues
    """
    current_lr = optimizer.param_groups[0]['lr']
    
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is not None:
                scheduler.step(metric)
            else:
                print("Warning: ReduceLROnPlateau scheduler needs metric")
        else:
            scheduler.step()
        
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != current_lr:
            print(f"Educational Learning Rate Update:")
            print(f"- Previous LR: {current_lr:.6f}")
            print(f"- New LR: {new_lr:.6f}")
            print(f"- Change: {((new_lr - current_lr) / current_lr) * 100:+.1f}%")
        
        return new_lr
    
    return current_lr