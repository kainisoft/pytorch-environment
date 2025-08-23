"""
Training Helpers Module for Chatbot-Qoder Tutorial Series

This module provides utilities for training PyTorch models including training loops,
optimization helpers, early stopping, and learning rate scheduling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple, Callable
import time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Model to potentially save weights
        
        Returns:
            bool: Whether to stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
        
        return self.early_stop


class TrainingLogger:
    """Logger for tracking training metrics and progress."""
    
    def __init__(self):
        """Initialize training logger."""
        self.history = defaultdict(list)
        self.start_time = None
    
    def start_epoch(self):
        """Mark the start of an epoch."""
        self.start_time = time.time()
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int, phase: str = "train"):
        """
        Log metrics for current epoch.
        
        Args:
            metrics (Dict[str, float]): Metrics dictionary
            epoch (int): Current epoch number
            phase (str): Training phase ("train" or "val")
        """
        for key, value in metrics.items():
            self.history[f"{phase}_{key}"].append(value)
        
        if self.start_time:
            epoch_time = time.time() - self.start_time
            self.history[f"{phase}_time"].append(epoch_time)
    
    def print_epoch_summary(self, epoch: int, train_metrics: Dict[str, float], 
                           val_metrics: Optional[Dict[str, float]] = None):
        """
        Print summary of epoch metrics.
        
        Args:
            epoch (int): Current epoch
            train_metrics (Dict[str, float]): Training metrics
            val_metrics (Dict[str, float], optional): Validation metrics
        """
        summary = f"Epoch {epoch:3d} | "
        
        # Training metrics
        for key, value in train_metrics.items():
            summary += f"Train {key}: {value:.4f} | "
        
        # Validation metrics
        if val_metrics:
            for key, value in val_metrics.items():
                summary += f"Val {key}: {value:.4f} | "
        
        # Timing
        if f"train_time" in self.history:
            summary += f"Time: {self.history['train_time'][-1]:.2f}s"
        
        print(summary)
    
    def plot_history(self, metrics: List[str] = None, figsize: Tuple[int, int] = (12, 4)):
        """
        Plot training history.
        
        Args:
            metrics (List[str], optional): Metrics to plot
            figsize (Tuple[int, int]): Figure size
        """
        if metrics is None:
            metrics = ["loss", "accuracy"]
        
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Plot training metric
            train_key = f"train_{metric}"
            val_key = f"val_{metric}"
            
            if train_key in self.history:
                ax.plot(self.history[train_key], label=f"Train {metric}", marker='o', markersize=3)
            
            if val_key in self.history:
                ax.plot(self.history[val_key], label=f"Val {metric}", marker='s', markersize=3)
            
            ax.set_title(f"{metric.capitalize()} History")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def get_optimizer(model: nn.Module, 
                 optimizer_name: str = "adam",
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0001,
                 **kwargs) -> torch.optim.Optimizer:
    """
    Get optimizer for model parameters.
    
    Args:
        model (nn.Module): Model to optimize
        optimizer_name (str): Name of optimizer
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay
        **kwargs: Additional optimizer arguments
    
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    if optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, 
                        momentum=kwargs.get('momentum', 0.9), **kwargs)
    elif optimizer_name.lower() == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer: torch.optim.Optimizer,
                 scheduler_name: str = "none",
                 **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule
        scheduler_name (str): Name of scheduler
        **kwargs: Scheduler-specific arguments
    
    Returns:
        Optional[torch.optim.lr_scheduler._LRScheduler]: Configured scheduler
    """
    if scheduler_name.lower() == "none":
        return None
    elif scheduler_name.lower() == "step":
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name.lower() == "cosine":
        T_max = kwargs.get('T_max', 50)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name.lower() == "plateau":
        patience = kwargs.get('patience', 5)
        factor = kwargs.get('factor', 0.5)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def train_epoch(model: nn.Module,
               dataloader: DataLoader,
               criterion: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               gradient_clip_norm: Optional[float] = None) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        gradient_clip_norm (float, optional): Gradient clipping norm
    
    Returns:
        Dict[str, float]: Training metrics
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        if isinstance(batch, (list, tuple)):
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
            inputs, targets = batch[0], batch[1]
        else:
            inputs = batch.to(device)
            targets = inputs  # For language modeling
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        if outputs.dim() == 3:  # Sequence prediction (batch_size, seq_len, vocab_size)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        
        # Update parameters
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        
        # Calculate accuracy for classification tasks
        if outputs.dim() == 2 and outputs.size(1) > 1:
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
    
    # Calculate average metrics
    avg_loss = total_loss / total_samples
    metrics = {"loss": avg_loss}
    
    if correct_predictions > 0:
        accuracy = correct_predictions / total_samples
        metrics["accuracy"] = accuracy
    
    return metrics


def evaluate_model(model: nn.Module,
                  dataloader: DataLoader,
                  criterion: nn.Module,
                  device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on validation/test data.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Evaluation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to evaluate on
    
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
                inputs, targets = batch[0], batch[1]
            else:
                inputs = batch.to(device)
                targets = inputs  # For language modeling
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            if outputs.dim() == 3:  # Sequence prediction
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = targets.view(-1)
                loss = criterion(outputs_flat, targets_flat)
            else:
                loss = criterion(outputs, targets)
            
            # Track metrics
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # Calculate accuracy for classification tasks
            if outputs.dim() == 2 and outputs.size(1) > 1:
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == targets).sum().item()
    
    # Calculate average metrics
    avg_loss = total_loss / total_samples
    metrics = {"loss": avg_loss}
    
    if correct_predictions > 0:
        accuracy = correct_predictions / total_samples
        metrics["accuracy"] = accuracy
    
    return metrics


def train_model(model: nn.Module,
               train_loader: DataLoader,
               val_loader: Optional[DataLoader],
               config: Dict[str, Any],
               device: torch.device) -> Tuple[nn.Module, TrainingLogger]:
    """
    Complete training loop for a model.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader, optional): Validation data loader
        config (Dict[str, Any]): Training configuration
        device (torch.device): Device to train on
    
    Returns:
        Tuple[nn.Module, TrainingLogger]: Trained model and training history
    """
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, **config)
    scheduler = get_scheduler(optimizer, **config)
    early_stopping = EarlyStopping(
        patience=config.get('patience', 7),
        min_delta=config.get('min_delta', 0.001)
    ) if config.get('early_stopping', False) else None
    
    logger = TrainingLogger()
    
    # Training loop
    for epoch in range(config.get('num_epochs', 10)):
        logger.start_epoch()
        
        # Training phase
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_clip_norm=config.get('gradient_clip_norm')
        )
        logger.log_metrics(train_metrics, epoch, "train")
        
        # Validation phase
        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, criterion, device)
            logger.log_metrics(val_metrics, epoch, "val")
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = val_metrics["loss"] if val_metrics else train_metrics["loss"]
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Print progress
        if (epoch + 1) % config.get('log_every_n_epochs', 1) == 0:
            logger.print_epoch_summary(epoch + 1, train_metrics, val_metrics)
        
        # Early stopping
        if early_stopping is not None and val_metrics is not None:
            if early_stopping(val_metrics["loss"], model):
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    return model, logger


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss.
    
    Args:
        loss (float): Cross-entropy loss
    
    Returns:
        float: Perplexity score
    """
    return np.exp(loss)


# Export commonly used functions and classes
__all__ = [
    "EarlyStopping",
    "TrainingLogger",
    "get_optimizer",
    "get_scheduler",
    "train_epoch",
    "evaluate_model",
    "train_model",
    "calculate_perplexity"
]