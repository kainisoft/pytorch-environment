"""
Visualization Utilities for PyTorch Chatbot Tutorial

This module provides educational visualization utilities for the chatbot tutorial series.
All functions include comprehensive documentation and explanations to help learners
understand training progress, model behavior, and result interpretation through visual analysis.

Educational Focus:
    - Clear explanations of visualization techniques for ML
    - Detailed comments on plot interpretation and insights
    - Examples of common visualization patterns in deep learning
    - Interactive plotting for better understanding
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
from matplotlib.patches import Rectangle


# Set up plotting style for educational clarity
plt.style.use('default')
sns.set_palette("husl")


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot comprehensive training history with educational annotations.
    
    Args:
        history: Dictionary containing training metrics over epochs
        save_path: Optional path to save the plot
        figsize: Figure size for the plot
        
    Educational Purpose:
        - Visualizes training progress over time
        - Shows relationship between training and validation metrics
        - Helps identify overfitting, underfitting, and convergence patterns
        - Demonstrates proper loss curve interpretation
        
    Learning Notes:
        - Training loss should generally decrease over time
        - Gap between train/val loss indicates overfitting
        - Oscillating losses may indicate learning rate issues
        - Plateauing suggests convergence or need for changes
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Educational Training History Analysis', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    axes[0, 0].set_title('Loss Curves', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add educational annotations
    if len(history['train_loss']) > 1:
        final_train_loss = history['train_loss'][-1]
        axes[0, 0].axhline(y=final_train_loss, color='blue', linestyle='--', alpha=0.5)
        axes[0, 0].text(0.02, 0.98, f'Final Train Loss: {final_train_loss:.4f}', 
                       transform=axes[0, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 2: Learning rate (if available)
    if 'learning_rates' in history and history['learning_rates']:
        axes[0, 1].plot(epochs[:len(history['learning_rates'])], history['learning_rates'], 'g-', linewidth=2)
        axes[0, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes,
                       fontsize=12, style='italic')
        axes[0, 1].set_title('Learning Rate Schedule', fontweight='bold')
    
    # Plot 3: Training time per epoch
    if 'epoch_times' in history and history['epoch_times']:
        axes[1, 0].bar(epochs[:len(history['epoch_times'])], history['epoch_times'], 
                      color='orange', alpha=0.7)
        axes[1, 0].set_title('Training Time per Epoch', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add average time annotation
        avg_time = np.mean(history['epoch_times'])
        axes[1, 0].axhline(y=avg_time, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].text(0.02, 0.98, f'Avg Time: {avg_time:.2f}s', 
                       transform=axes[1, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
    else:
        axes[1, 0].text(0.5, 0.5, 'Training Time\nData Not Available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes,
                       fontsize=12, style='italic')
        axes[1, 0].set_title('Training Time per Epoch', fontweight='bold')
    
    # Plot 4: Overfitting analysis
    if 'val_loss' in history and history['val_loss'] and len(history['val_loss']) > 1:
        train_val_gap = [v - t for t, v in zip(history['train_loss'], history['val_loss'])]
        axes[1, 1].plot(epochs[:len(train_val_gap)], train_val_gap, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('Overfitting Analysis (Val - Train Loss)', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Color regions to show overfitting
        axes[1, 1].fill_between(epochs[:len(train_val_gap)], train_val_gap, 0, 
                               where=[gap > 0 for gap in train_val_gap], 
                               color='red', alpha=0.3, label='Potential Overfitting')
        axes[1, 1].fill_between(epochs[:len(train_val_gap)], train_val_gap, 0, 
                               where=[gap <= 0 for gap in train_val_gap], 
                               color='green', alpha=0.3, label='Good Generalization')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Validation Loss\nNot Available\nfor Overfitting Analysis', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, style='italic')
        axes[1, 1].set_title('Overfitting Analysis', fontweight='bold')
    
    plt.tight_layout()
    
    # Add educational interpretation
    print("Educational Training History Interpretation:")
    if history['train_loss']:
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        print(f"- Training loss improved by {improvement:.1f}% ({initial_loss:.4f} → {final_loss:.4f})")
        
        if 'val_loss' in history and history['val_loss']:
            final_val_loss = history['val_loss'][-1]
            gap = final_val_loss - final_loss
            print(f"- Final validation gap: {gap:.4f}")
            if gap > 0.1:
                print("  ⚠️  Large gap suggests overfitting")
            elif gap < 0:
                print("  ✓ Validation loss lower than training (good generalization)")
            else:
                print("  ✓ Small gap indicates good generalization")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"- Plot saved to: {save_path}")
    
    plt.show()


def plot_attention_weights(attention_weights: torch.Tensor, 
                          input_tokens: List[str],
                          output_tokens: List[str],
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Visualize attention weights with educational explanations.
    
    Args:
        attention_weights: Tensor of shape (seq_len, seq_len) containing attention weights
        input_tokens: List of input tokens
        output_tokens: List of output tokens  
        save_path: Optional path to save the plot
        figsize: Figure size for the plot
        
    Educational Purpose:
        - Visualizes how attention mechanism focuses on different parts of input
        - Shows the relationship between input and output tokens
        - Helps understand what the model has learned to pay attention to
        - Demonstrates attention pattern interpretation
        
    Learning Notes:
        - Darker colors indicate higher attention weights
        - Diagonal patterns suggest position-based attention
        - Scattered patterns indicate content-based attention
        - Attention patterns reveal model's decision-making process
    """
    # Convert to numpy for plotting
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(input_tokens)))
    ax.set_yticks(range(len(output_tokens)))
    ax.set_xticklabels(input_tokens, rotation=45, ha='right')
    ax.set_yticklabels(output_tokens)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Add title and labels
    ax.set_title('Educational Attention Weight Visualization', fontweight='bold', pad=20)
    ax.set_xlabel('Input Tokens')
    ax.set_ylabel('Output Tokens')
    
    # Add text annotations for high attention weights
    threshold = np.max(attention_weights) * 0.7  # Show weights above 70% of max
    for i in range(len(output_tokens)):
        for j in range(len(input_tokens)):
            if attention_weights[i, j] > threshold:
                text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                             ha="center", va="center", color="white", fontweight='bold')
    
    plt.tight_layout()
    
    # Educational interpretation
    print("Educational Attention Analysis:")
    max_attention = np.max(attention_weights)
    avg_attention = np.mean(attention_weights)
    print(f"- Maximum attention weight: {max_attention:.3f}")
    print(f"- Average attention weight: {avg_attention:.3f}")
    
    # Find most attended input token for each output token
    for i, output_token in enumerate(output_tokens):
        max_input_idx = np.argmax(attention_weights[i])
        max_weight = attention_weights[i, max_input_idx]
        print(f"- '{output_token}' attends most to '{input_tokens[max_input_idx]}' ({max_weight:.3f})")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"- Attention plot saved to: {save_path}")
    
    plt.show()


def plot_model_performance_comparison(models_data: Dict[str, Dict[str, float]],
                                    metrics: List[str] = ['loss', 'perplexity', 'bleu'],
                                    save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Compare performance of multiple models with educational insights.
    
    Args:
        models_data: Dict mapping model names to their metrics
        metrics: List of metrics to compare
        save_path: Optional path to save the plot
        figsize: Figure size for the plot
        
    Educational Purpose:
        - Visualizes performance differences between model variants
        - Shows trade-offs between different metrics
        - Helps in model selection and architecture decisions
        - Demonstrates comparative analysis techniques
        
    Learning Notes:
        - Different models excel at different metrics
        - Consider multiple metrics for comprehensive evaluation
        - Trade-offs between model complexity and performance
        - Best model depends on specific use case requirements
    """
    # Prepare data for plotting
    model_names = list(models_data.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    fig.suptitle('Educational Model Performance Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))
    
    for i, metric in enumerate(metrics):
        # Extract metric values for all models
        values = []
        labels = []
        for model_name in model_names:
            if metric in models_data[model_name]:
                values.append(models_data[model_name][metric])
                labels.append(model_name)
        
        if values:
            # Create bar plot
            bars = axes[i].bar(labels, values, color=colors[:len(values)], alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            axes[i].set_title(f'{metric.capitalize()}', fontweight='bold')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Highlight best performing model
            if metric.lower() in ['loss']:  # Lower is better
                best_idx = np.argmin(values)
                best_color = 'green'
            else:  # Higher is better (bleu, accuracy, etc.)
                best_idx = np.argmax(values)
                best_color = 'green'
            
            bars[best_idx].set_edgecolor(best_color)
            bars[best_idx].set_linewidth(3)
        else:
            axes[i].text(0.5, 0.5, f'No {metric}\ndata available', 
                        ha='center', va='center', transform=axes[i].transAxes,
                        fontsize=12, style='italic')
            axes[i].set_title(f'{metric.capitalize()}', fontweight='bold')
    
    plt.tight_layout()
    
    # Educational analysis
    print("Educational Model Comparison Analysis:")
    for metric in metrics:
        values = []
        model_names_with_metric = []
        for model_name in model_names:
            if metric in models_data[model_name]:
                values.append(models_data[model_name][metric])
                model_names_with_metric.append(model_name)
        
        if values:
            if metric.lower() in ['loss', 'perplexity']:  # Lower is better
                best_idx = np.argmin(values)
                best_model = model_names_with_metric[best_idx]
                best_value = values[best_idx]
                print(f"- Best {metric}: {best_model} ({best_value:.4f})")
            else:  # Higher is better
                best_idx = np.argmax(values)
                best_model = model_names_with_metric[best_idx]
                best_value = values[best_idx]
                print(f"- Best {metric}: {best_model} ({best_value:.4f})")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"- Comparison plot saved to: {save_path}")
    
    plt.show()


def plot_loss_landscape(loss_values: np.ndarray, 
                       param1_range: np.ndarray, 
                       param2_range: np.ndarray,
                       param1_name: str = "Parameter 1",
                       param2_name: str = "Parameter 2",
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Visualize loss landscape for educational understanding of optimization.
    
    Args:
        loss_values: 2D array of loss values
        param1_range: Range of first parameter values
        param2_range: Range of second parameter values
        param1_name: Name of first parameter
        param2_name: Name of second parameter
        save_path: Optional path to save the plot
        figsize: Figure size for the plot
        
    Educational Purpose:
        - Visualizes the optimization landscape
        - Shows local minima, global minima, and saddle points
        - Helps understand why optimization can be challenging
        - Demonstrates the effect of different parameter values on loss
        
    Learning Notes:
        - Darker regions indicate lower loss (better performance)
        - Multiple dark regions suggest multiple local minima
        - Flat regions can slow down optimization
        - Sharp valleys can cause optimization instability
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 2D contour plot
    contour = ax1.contour(param1_range, param2_range, loss_values, levels=20)
    ax1.clabel(contour, inline=True, fontsize=8)
    im1 = ax1.contourf(param1_range, param2_range, loss_values, levels=50, cmap='viridis', alpha=0.7)
    
    ax1.set_xlabel(param1_name)
    ax1.set_ylabel(param2_name)
    ax1.set_title('Educational Loss Landscape (Contour)', fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Loss Value')
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(param1_range, param2_range)
    surf = ax2.plot_surface(X, Y, loss_values, cmap='viridis', alpha=0.8)
    
    ax2.set_xlabel(param1_name)
    ax2.set_ylabel(param2_name)
    ax2.set_zlabel('Loss Value')
    ax2.set_title('Educational Loss Landscape (3D)', fontweight='bold')
    
    plt.tight_layout()
    
    # Educational analysis
    min_loss = np.min(loss_values)
    max_loss = np.max(loss_values)
    min_idx = np.unravel_index(np.argmin(loss_values), loss_values.shape)
    
    print("Educational Loss Landscape Analysis:")
    print(f"- Minimum loss: {min_loss:.4f}")
    print(f"- Maximum loss: {max_loss:.4f}")
    print(f"- Loss range: {max_loss - min_loss:.4f}")
    print(f"- Optimal {param1_name}: {param1_range[min_idx[1]]:.4f}")
    print(f"- Optimal {param2_name}: {param2_range[min_idx[0]]:.4f}")
    
    # Analyze landscape characteristics
    gradient_x = np.gradient(loss_values, axis=1)
    gradient_y = np.gradient(loss_values, axis=0)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    print(f"- Average gradient magnitude: {np.mean(gradient_magnitude):.4f}")
    print(f"- Maximum gradient magnitude: {np.max(gradient_magnitude):.4f}")
    
    if np.max(gradient_magnitude) > 10 * np.mean(gradient_magnitude):
        print("  ⚠️  Steep gradients detected - may cause optimization instability")
    else:
        print("  ✓ Relatively smooth landscape - good for optimization")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"- Loss landscape plot saved to: {save_path}")
    
    plt.show()


def plot_token_embeddings(embeddings: torch.Tensor, 
                         tokens: List[str],
                         method: str = 'pca',
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualize token embeddings in 2D space with educational explanations.
    
    Args:
        embeddings: Tensor of shape (n_tokens, embedding_dim)
        tokens: List of token strings
        method: Dimensionality reduction method ('pca' or 'tsne')
        save_path: Optional path to save the plot
        figsize: Figure size for the plot
        
    Educational Purpose:
        - Visualizes high-dimensional embeddings in 2D space
        - Shows semantic relationships between tokens
        - Demonstrates dimensionality reduction techniques
        - Helps understand what the model has learned about language
        
    Learning Notes:
        - Similar tokens should cluster together in embedding space
        - Distance in embedding space reflects semantic similarity
        - PCA preserves global structure, t-SNE preserves local structure
        - Embedding quality affects downstream task performance
    """
    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    # Dimensionality reduction
    if method.lower() == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(embeddings)
        explained_variance = reducer.explained_variance_ratio_
        method_title = f'PCA (Explained Variance: {sum(explained_variance):.1%})'
    elif method.lower() == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        method_title = 't-SNE'
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot of embeddings
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        c=range(len(tokens)), cmap='tab20', s=100, alpha=0.7)
    
    # Add token labels
    for i, token in enumerate(tokens):
        ax.annotate(token, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=10, alpha=0.8)
    
    ax.set_title(f'Educational Token Embeddings Visualization ({method_title})', 
                fontweight='bold')
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Educational analysis
    print("Educational Embedding Analysis:")
    print(f"- Visualization method: {method.upper()}")
    print(f"- Original embedding dimension: {embeddings.shape[1]}")
    print(f"- Number of tokens: {len(tokens)}")
    
    if method.lower() == 'pca':
        print(f"- Variance explained by 2 components: {sum(explained_variance):.1%}")
        print(f"  - Component 1: {explained_variance[0]:.1%}")
        print(f"  - Component 2: {explained_variance[1]:.1%}")
    
    # Calculate pairwise distances for similarity analysis
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(embeddings_2d))
    
    # Find most similar token pairs
    n_pairs = min(5, len(tokens) * (len(tokens) - 1) // 2)
    similar_pairs = []
    
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            similar_pairs.append((distances[i, j], tokens[i], tokens[j]))
    
    similar_pairs.sort()
    
    print(f"\nMost similar token pairs (in 2D space):")
    for dist, token1, token2 in similar_pairs[:n_pairs]:
        print(f"- '{token1}' ↔ '{token2}': distance {dist:.3f}")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"- Embedding plot saved to: {save_path}")
    
    plt.show()