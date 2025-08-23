"""
Training Configuration Classes for PyTorch Chatbot Tutorial

This module provides educational configuration classes for training parameters.
All configurations include comprehensive documentation and validation to help learners
understand training hyperparameters and optimization strategies.

Educational Focus:
    - Clear explanations of each training parameter's role and impact
    - Detailed comments on optimization strategies and learning schedules
    - Examples of different training configurations for various scenarios
    - Validation methods with educational error messages and guidance
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import torch
import math


@dataclass
class TrainingConfig:
    """
    Educational configuration class for training parameters.
    
    This class demonstrates how to structure training configurations with proper
    validation and educational explanations for each hyperparameter.
    
    Educational Purpose:
        - Shows best practices for training configuration management
        - Explains the role and impact of each training hyperparameter
        - Demonstrates optimization strategy selection
        - Provides presets for different training scenarios
        
    Learning Notes:
        - Learning rate is the most important hyperparameter to tune
        - Batch size affects gradient noise and memory usage
        - Scheduler helps achieve better convergence
        - Gradient clipping prevents exploding gradients
        - Early stopping prevents overfitting
    """
    
    # Core training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    
    # Optimization parameters
    optimizer_type: str = "adam"  # "adam", "sgd", "adamw"
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # "cosine", "linear", "exponential", "plateau"
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    min_learning_rate: float = 1e-6
    
    # Gradient management
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.0
    
    # Early stopping and checkpointing
    use_early_stopping: bool = True
    patience: int = 3
    min_delta: float = 1e-4
    save_best_model: bool = True
    save_every_n_epochs: int = 1
    
    # Evaluation and logging
    eval_every_n_steps: int = 500
    log_every_n_steps: int = 100
    eval_batch_size: Optional[int] = None
    
    # Mixed precision training
    use_mixed_precision: bool = False
    
    # Educational metadata
    config_name: str = "educational_training"
    description: str = "Educational training configuration for learning purposes"
    difficulty_level: str = "beginner"  # "beginner", "intermediate", "advanced"
    
    def __post_init__(self):
        """
        Validate and set derived parameters with educational explanations.
        
        Educational Purpose:
            - Demonstrates proper parameter validation and initialization
            - Shows how to set reasonable defaults based on other parameters
            - Explains parameter relationships and dependencies
        """
        # Set eval_batch_size to batch_size if not specified
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
        
        # Validate configuration
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate training configuration parameters with educational error messages.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid with educational explanation
            
        Educational Purpose:
            - Shows how to validate training configurations
            - Explains why certain parameter combinations are problematic
            - Demonstrates defensive programming for ML training
        """
        errors = []
        
        # Validate core parameters
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive (controls step size in optimization)")
        
        if self.learning_rate > 1.0:
            errors.append("learning_rate > 1.0 is usually too large (may cause divergence)")
        
        if self.batch_size <= 0:
            errors.append("batch_size must be positive (number of samples per gradient update)")
        
        if self.epochs <= 0:
            errors.append("epochs must be positive (number of complete passes through data)")
        
        # Validate optimizer parameters
        valid_optimizers = ["adam", "sgd", "adamw", "rmsprop"]
        if self.optimizer_type.lower() not in valid_optimizers:
            errors.append(f"optimizer_type must be one of {valid_optimizers}")
        
        if not 0 <= self.weight_decay <= 1:
            errors.append("weight_decay should be between 0 and 1 (L2 regularization strength)")
        
        if not 0 < self.beta1 < 1:
            errors.append("beta1 must be between 0 and 1 (momentum parameter for Adam)")
        
        if not 0 < self.beta2 < 1:
            errors.append("beta2 must be between 0 and 1 (second moment parameter for Adam)")
        
        # Validate scheduler parameters
        if self.use_scheduler:
            valid_schedulers = ["cosine", "linear", "exponential", "plateau", "step"]
            if self.scheduler_type.lower() not in valid_schedulers:
                errors.append(f"scheduler_type must be one of {valid_schedulers}")
            
            if self.warmup_steps < 0:
                errors.append("warmup_steps must be non-negative")
            
            if not 0 <= self.warmup_ratio <= 1:
                errors.append("warmup_ratio must be between 0 and 1")
            
            if self.min_learning_rate >= self.learning_rate:
                errors.append("min_learning_rate must be less than learning_rate")
        
        # Validate gradient parameters
        if self.gradient_clip_norm <= 0:
            errors.append("gradient_clip_norm must be positive (prevents exploding gradients)")
        
        if self.gradient_accumulation_steps <= 0:
            errors.append("gradient_accumulation_steps must be positive")
        
        # Validate regularization parameters
        if not 0 <= self.dropout_rate <= 1:
            errors.append("dropout_rate must be between 0 and 1")
        
        if not 0 <= self.label_smoothing <= 1:
            errors.append("label_smoothing must be between 0 and 1")
        
        # Validate early stopping parameters
        if self.use_early_stopping:
            if self.patience <= 0:
                errors.append("patience must be positive for early stopping")
            
            if self.min_delta < 0:
                errors.append("min_delta must be non-negative")
        
        # Validate logging parameters
        if self.eval_every_n_steps <= 0:
            errors.append("eval_every_n_steps must be positive")
        
        if self.log_every_n_steps <= 0:
            errors.append("log_every_n_steps must be positive")
        
        # Validate difficulty level
        valid_difficulty = ["beginner", "intermediate", "advanced"]
        if self.difficulty_level.lower() not in valid_difficulty:
            errors.append(f"difficulty_level must be one of {valid_difficulty}")
        
        if errors:
            error_msg = "Educational Training Configuration Validation Errors:\n"
            for i, error in enumerate(errors, 1):
                error_msg += f"  {i}. {error}\n"
            error_msg += "\nLearning Note: Proper validation prevents training failures and ensures optimal learning."
            raise ValueError(error_msg)
        
        print("✓ Educational Training Configuration Validation Passed")
        return True
    
    def get_optimizer(self, model_parameters) -> torch.optim.Optimizer:
        """
        Create optimizer based on configuration with educational explanations.
        
        Args:
            model_parameters: Model parameters to optimize
            
        Returns:
            torch.optim.Optimizer: Configured optimizer
            
        Educational Purpose:
            - Shows how to create optimizers from configuration
            - Explains different optimizer characteristics
            - Demonstrates parameter passing to optimizers
            
        Learning Notes:
            - Adam is generally good for most tasks (adaptive learning rates)
            - SGD with momentum works well for computer vision
            - AdamW includes better weight decay handling
            - RMSprop is good for RNNs and online learning
        """
        optimizer_params = {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        
        if self.optimizer_type.lower() == "adam":
            optimizer_params.update({
                'betas': (self.beta1, self.beta2),
                'eps': self.epsilon
            })
            optimizer = torch.optim.Adam(model_parameters, **optimizer_params)
            print(f"Educational Optimizer: Adam with lr={self.learning_rate}, weight_decay={self.weight_decay}")
            
        elif self.optimizer_type.lower() == "adamw":
            optimizer_params.update({
                'betas': (self.beta1, self.beta2),
                'eps': self.epsilon
            })
            optimizer = torch.optim.AdamW(model_parameters, **optimizer_params)
            print(f"Educational Optimizer: AdamW with lr={self.learning_rate}, weight_decay={self.weight_decay}")
            
        elif self.optimizer_type.lower() == "sgd":
            optimizer_params.update({
                'momentum': self.beta1  # Use beta1 as momentum for SGD
            })
            optimizer = torch.optim.SGD(model_parameters, **optimizer_params)
            print(f"Educational Optimizer: SGD with lr={self.learning_rate}, momentum={self.beta1}")
            
        elif self.optimizer_type.lower() == "rmsprop":
            optimizer_params.update({
                'alpha': self.beta2,  # Use beta2 as alpha for RMSprop
                'eps': self.epsilon
            })
            optimizer = torch.optim.RMSprop(model_parameters, **optimizer_params)
            print(f"Educational Optimizer: RMSprop with lr={self.learning_rate}, alpha={self.beta2}")
            
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        
        return optimizer
    
    def get_scheduler(self, optimizer: torch.optim.Optimizer, 
                     total_steps: Optional[int] = None) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler based on configuration.
        
        Args:
            optimizer: Optimizer to schedule
            total_steps: Total number of training steps (needed for some schedulers)
            
        Returns:
            Optional scheduler object
            
        Educational Purpose:
            - Shows how to create learning rate schedulers
            - Explains different scheduling strategies
            - Demonstrates scheduler configuration
            
        Learning Notes:
            - Cosine annealing provides smooth learning rate decay
            - Linear scheduling is simple and often effective
            - Plateau scheduling adapts to training progress
            - Warmup helps stabilize early training
        """
        if not self.use_scheduler:
            return None
        
        if self.scheduler_type.lower() == "cosine":
            if total_steps is None:
                raise ValueError("total_steps required for cosine scheduler")
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=total_steps - self.warmup_steps,
                eta_min=self.min_learning_rate
            )
            print(f"Educational Scheduler: Cosine annealing with T_max={total_steps - self.warmup_steps}")
            
        elif self.scheduler_type.lower() == "linear":
            if total_steps is None:
                raise ValueError("total_steps required for linear scheduler")
            
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=self.min_learning_rate / self.learning_rate,
                total_iters=total_steps - self.warmup_steps
            )
            print(f"Educational Scheduler: Linear decay over {total_steps - self.warmup_steps} steps")
            
        elif self.scheduler_type.lower() == "exponential":
            gamma = (self.min_learning_rate / self.learning_rate) ** (1.0 / (total_steps - self.warmup_steps))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            print(f"Educational Scheduler: Exponential decay with gamma={gamma:.6f}")
            
        elif self.scheduler_type.lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=self.patience,
                min_lr=self.min_learning_rate
            )
            print(f"Educational Scheduler: Reduce on plateau with patience={self.patience}")
            
        elif self.scheduler_type.lower() == "step":
            step_size = max(1, (total_steps - self.warmup_steps) // 3)  # 3 drops during training
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
            print(f"Educational Scheduler: Step decay every {step_size} steps")
            
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")
        
        return scheduler
    
    def get_effective_batch_size(self) -> int:
        """
        Calculate effective batch size considering gradient accumulation.
        
        Returns:
            int: Effective batch size
            
        Educational Purpose:
            - Shows how gradient accumulation affects effective batch size
            - Explains memory vs. batch size trade-offs
            - Demonstrates batch size calculation
            
        Learning Notes:
            - Effective batch size = batch_size * gradient_accumulation_steps
            - Larger effective batch sizes provide more stable gradients
            - Gradient accumulation allows large effective batches with limited memory
        """
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        
        print(f"Educational Batch Size Analysis:")
        print(f"- Actual batch size: {self.batch_size}")
        print(f"- Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"- Effective batch size: {effective_batch_size}")
        
        return effective_batch_size
    
    def print_summary(self) -> None:
        """
        Print a comprehensive summary of the training configuration.
        
        Educational Purpose:
            - Provides clear overview of training setup
            - Shows all important hyperparameters
            - Explains the implications of different settings
        """
        print("=" * 60)
        print("EDUCATIONAL TRAINING CONFIGURATION SUMMARY")
        print("=" * 60)
        
        print(f"Configuration Name: {self.config_name}")
        print(f"Description: {self.description}")
        print(f"Difficulty Level: {self.difficulty_level.title()}")
        print()
        
        print("Core Training Parameters:")
        print(f"  - Learning Rate: {self.learning_rate}")
        print(f"  - Batch Size: {self.batch_size}")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - Effective Batch Size: {self.get_effective_batch_size()}")
        print()
        
        print("Optimization Configuration:")
        print(f"  - Optimizer: {self.optimizer_type.upper()}")
        print(f"  - Weight Decay: {self.weight_decay}")
        if self.optimizer_type.lower() in ["adam", "adamw"]:
            print(f"  - Beta1 (momentum): {self.beta1}")
            print(f"  - Beta2 (second moment): {self.beta2}")
            print(f"  - Epsilon: {self.epsilon}")
        print()
        
        if self.use_scheduler:
            print("Learning Rate Scheduling:")
            print(f"  - Scheduler Type: {self.scheduler_type.title()}")
            print(f"  - Warmup Steps: {self.warmup_steps}")
            print(f"  - Warmup Ratio: {self.warmup_ratio}")
            print(f"  - Minimum Learning Rate: {self.min_learning_rate}")
            print()
        
        print("Gradient Management:")
        print(f"  - Gradient Clipping Norm: {self.gradient_clip_norm}")
        print(f"  - Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        print()
        
        print("Regularization:")
        print(f"  - Dropout Rate: {self.dropout_rate}")
        print(f"  - Label Smoothing: {self.label_smoothing}")
        print()
        
        if self.use_early_stopping:
            print("Early Stopping:")
            print(f"  - Patience: {self.patience} epochs")
            print(f"  - Minimum Delta: {self.min_delta}")
            print()
        
        print("Evaluation and Logging:")
        print(f"  - Evaluate Every: {self.eval_every_n_steps} steps")
        print(f"  - Log Every: {self.log_every_n_steps} steps")
        print(f"  - Save Every: {self.save_every_n_epochs} epochs")
        print(f"  - Mixed Precision: {self.use_mixed_precision}")
        print()
        
        # Educational insights
        print("Educational Insights:")
        if self.difficulty_level == "beginner":
            print("  - Conservative settings good for learning and stability")
            print("  - Lower learning rates prevent divergence")
        elif self.difficulty_level == "intermediate":
            print("  - Balanced settings for most practical applications")
            print("  - Good starting point for experimentation")
        else:
            print("  - Aggressive settings for experienced practitioners")
            print("  - May require careful monitoring and tuning")
        
        print(f"  - Effective batch size of {self.get_effective_batch_size()} provides stable gradients")
        print("=" * 60)
    
    @classmethod
    def create_beginner_config(cls) -> 'TrainingConfig':
        """
        Create a beginner-friendly training configuration.
        
        Returns:
            TrainingConfig: Conservative configuration for learning
            
        Educational Purpose:
            - Provides safe starting point for beginners
            - Uses conservative hyperparameters to prevent common issues
            - Focuses on stability over speed
        """
        return cls(
            learning_rate=0.0001,
            batch_size=16,
            epochs=5,
            optimizer_type="adam",
            weight_decay=0.01,
            use_scheduler=True,
            scheduler_type="cosine",
            warmup_steps=500,
            gradient_clip_norm=1.0,
            gradient_accumulation_steps=2,
            patience=3,
            difficulty_level="beginner",
            config_name="beginner_training",
            description="Conservative training configuration for beginners"
        )
    
    @classmethod
    def create_intermediate_config(cls) -> 'TrainingConfig':
        """
        Create an intermediate training configuration.
        
        Returns:
            TrainingConfig: Balanced configuration for most use cases
            
        Educational Purpose:
            - Provides good balance of speed and stability
            - Suitable for most educational experiments
            - Shows realistic hyperparameters for practical applications
        """
        return cls(
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            optimizer_type="adamw",
            weight_decay=0.01,
            use_scheduler=True,
            scheduler_type="cosine",
            warmup_steps=1000,
            gradient_clip_norm=1.0,
            gradient_accumulation_steps=1,
            patience=5,
            difficulty_level="intermediate",
            config_name="intermediate_training",
            description="Balanced training configuration for most applications"
        )
    
    @classmethod
    def create_advanced_config(cls) -> 'TrainingConfig':
        """
        Create an advanced training configuration.
        
        Returns:
            TrainingConfig: Aggressive configuration for experienced users
            
        Educational Purpose:
            - Shows advanced training techniques
            - Suitable for research and experimentation
            - Demonstrates optimization for performance
        """
        return cls(
            learning_rate=0.003,
            batch_size=64,
            epochs=20,
            optimizer_type="adamw",
            weight_decay=0.1,
            use_scheduler=True,
            scheduler_type="cosine",
            warmup_steps=2000,
            gradient_clip_norm=0.5,
            gradient_accumulation_steps=1,
            use_mixed_precision=True,
            patience=7,
            difficulty_level="advanced",
            config_name="advanced_training",
            description="Advanced training configuration for experienced practitioners"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dict containing all configuration parameters
        """
        return {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'optimizer_type': self.optimizer_type,
            'weight_decay': self.weight_decay,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'use_scheduler': self.use_scheduler,
            'scheduler_type': self.scheduler_type,
            'warmup_steps': self.warmup_steps,
            'warmup_ratio': self.warmup_ratio,
            'min_learning_rate': self.min_learning_rate,
            'gradient_clip_norm': self.gradient_clip_norm,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'dropout_rate': self.dropout_rate,
            'label_smoothing': self.label_smoothing,
            'use_early_stopping': self.use_early_stopping,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'save_best_model': self.save_best_model,
            'save_every_n_epochs': self.save_every_n_epochs,
            'eval_every_n_steps': self.eval_every_n_steps,
            'log_every_n_steps': self.log_every_n_steps,
            'eval_batch_size': self.eval_batch_size,
            'use_mixed_precision': self.use_mixed_precision,
            'config_name': self.config_name,
            'description': self.description,
            'difficulty_level': self.difficulty_level
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            TrainingConfig: Configuration object
        """
        return cls(**config_dict)