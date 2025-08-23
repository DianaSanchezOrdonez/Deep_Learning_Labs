import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# Configurar estilo de seaborn
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_training_history(json_path: str) -> Dict:
    """
    Carga el historial de entrenamiento desde un archivo JSON.
    
    Args:
        json_path: Ruta al archivo JSON con el historial de entrenamiento
        
    Returns:
        Diccionario con las métricas de entrenamiento
    """
    with open(json_path, 'r') as f:
        history = json.load(f)
    return history

def plot_training_metrics(history: Dict, 
                         title: str = "Training & Validation Metrics",
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    Grafica las métricas de entrenamiento usando seaborn.
    
    Args:
        history: Diccionario con las métricas de entrenamiento
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura
    """
    
    # Crear subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Preparar datos
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss comparison
    ax1.plot(epochs, history['train_loss'], label='Train Loss', 
             color='#e74c3c', linewidth=2, linestyle='--', alpha=0.8)
    ax1.plot(epochs, history['eval_loss'], label='Validation Loss', 
             color='#c0392b', linewidth=2, alpha=0.9)
    ax1.set_title('Loss Comparison', fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy comparison
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', 
             color='#3498db', linewidth=2, linestyle='--', alpha=0.8)
    ax2.plot(epochs, history['eval_acc'], label='Validation Accuracy', 
             color='#2980b9', linewidth=2, alpha=0.9)
    ax2.set_title('Accuracy Comparison', fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision, Recall, F1 (si están disponibles)
    if 'eval_precision' in history and 'eval_recall' in history and 'eval_f1' in history:
        ax3.plot(epochs, history['eval_precision'], label='Precision', 
                 color='#e67e22', linewidth=2, alpha=0.9)
        ax3.plot(epochs, history['eval_recall'], label='Recall', 
                 color='#f39c12', linewidth=2, alpha=0.9)
        ax3.plot(epochs, history['eval_f1'], label='F1-Score', 
                 color='#d35400', linewidth=2, alpha=0.9)
        ax3.set_title('Validation Metrics', fontweight='bold')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Precision, Recall, F1\ndata not available', 
                 ha='center', va='center', transform=ax3.transAxes,
                 fontsize=12, alpha=0.6)
        ax3.set_title('Validation Metrics', fontweight='bold')
    
    # 4. Learning curve (Overfitting analysis)
    train_acc_smooth = pd.Series(history['train_acc']).rolling(window=5, center=True).mean()
    val_acc_smooth = pd.Series(history['eval_acc']).rolling(window=5, center=True).mean()
    
    ax4.plot(epochs, train_acc_smooth, label='Train Acc (Smoothed)', 
             color='#3498db', linewidth=3, alpha=0.8)
    ax4.plot(epochs, val_acc_smooth, label='Val Acc (Smoothed)', 
             color='#2980b9', linewidth=3, alpha=0.8)
    
    # Highlight overfitting region
    gap = np.array(train_acc_smooth) - np.array(val_acc_smooth)
    overfitting_threshold = 5  # 5% gap
    overfitting_mask = gap > overfitting_threshold
    
    if np.any(overfitting_mask):
        overfitting_epochs = np.array(epochs)[overfitting_mask]
        ax4.axvspan(overfitting_epochs[0], epochs[-1], alpha=0.2, color='red', 
                   label=f'Potential Overfitting (>{overfitting_threshold}% gap)')
    
    ax4.set_title('Learning Curve Analysis', fontweight='bold')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()

def plot_metrics_distribution(history: Dict,
                            title: str = "Metrics Distribution",
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (14, 6)) -> None:
    """
    Crea gráficos de distribución de las métricas.
    
    Args:
        history: Diccionario con las métricas de entrenamiento
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura
    """
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Loss distribution
    loss_data = pd.DataFrame({
        'Train Loss': history['train_loss'],
        'Validation Loss': history['eval_loss']
    })
    
    sns.histplot(data=loss_data, ax=ax1, alpha=0.7, kde=True)
    ax1.set_title('Loss Distribution', fontweight='bold')
    ax1.set_xlabel('Loss Value')
    ax1.set_ylabel('Frequency')
    
    # Accuracy distribution
    acc_data = pd.DataFrame({
        'Train Accuracy': history['train_acc'],
        'Validation Accuracy': history['eval_acc']
    })
    
    sns.histplot(data=acc_data, ax=ax2, alpha=0.7, kde=True)
    ax2.set_title('Accuracy Distribution', fontweight='bold')
    ax2.set_xlabel('Accuracy (%)')
    ax2.set_ylabel('Frequency')
    
    # Box plot for comparison
    metrics_long = pd.concat([
        pd.DataFrame({'Value': history['train_loss'], 'Metric': 'Train Loss', 'Type': 'Loss'}),
        pd.DataFrame({'Value': history['eval_loss'], 'Metric': 'Val Loss', 'Type': 'Loss'}),
        pd.DataFrame({'Value': history['train_acc'], 'Metric': 'Train Acc', 'Type': 'Accuracy'}),
        pd.DataFrame({'Value': history['eval_acc'], 'Metric': 'Val Acc', 'Type': 'Accuracy'})
    ])
    
    sns.boxplot(data=metrics_long, x='Metric', y='Value', ax=ax3)
    ax3.set_title('Metrics Comparison', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()
