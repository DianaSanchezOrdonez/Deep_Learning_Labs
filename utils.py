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

def plot_loss_accuracy_combined(history: Dict,
                               title: str = "Training Progress",
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Crea un gráfico combinado de loss y accuracy similar al original.
    
    Args:
        history: Diccionario con las métricas de entrenamiento
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura
    """
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss en el eje izquierdo
    color_loss = '#e74c3c'
    ax1.set_xlabel('Épocas', fontweight='bold')
    ax1.set_ylabel('Loss', color=color_loss, fontweight='bold')
    
    line1 = ax1.plot(epochs, history['train_loss'], label='Train Loss', 
                     color=color_loss, linestyle='--', linewidth=2, alpha=0.8)
    line2 = ax1.plot(epochs, history['eval_loss'], label='Validation Loss', 
                     color='#c0392b', linewidth=2, alpha=0.9)
    
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy en el eje derecho
    ax2 = ax1.twinx()
    color_acc = '#3498db'
    ax2.set_ylabel('Accuracy (%)', color=color_acc, fontweight='bold')
    
    line3 = ax2.plot(epochs, history['train_acc'], label='Train Accuracy', 
                     color=color_acc, linestyle='--', linewidth=2, alpha=0.8)
    line4 = ax2.plot(epochs, history['eval_acc'], label='Validation Accuracy', 
                     color='#2980b9', linewidth=2, alpha=0.9)
    
    ax2.tick_params(axis='y', labelcolor=color_acc)
    
    # Combinar leyendas
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', bbox_to_anchor=(0.95, 0.5))
    
    plt.title(title, fontweight='bold', pad=20)
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

def print_training_summary(history: Dict) -> None:
    """
    Imprime un resumen de las métricas de entrenamiento.
    
    Args:
        history: Diccionario con las métricas de entrenamiento
    """
    
    print("=" * 60)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("=" * 60)
    
    # Métricas básicas
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['eval_acc'][-1]
    best_val_acc = max(history['eval_acc'])
    best_epoch = history['eval_acc'].index(best_val_acc) + 1
    
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['eval_loss'][-1]
    best_val_loss = min(history['eval_loss'])
    
    print(f"Épocas totales: {len(history['train_loss'])}")
    print(f"Mejor época (val acc): {best_epoch}")
    print()
    
    print("ACCURACY:")
    print(f"  Final Train Accuracy: {final_train_acc:.2f}%")
    print(f"  Final Val Accuracy:   {final_val_acc:.2f}%")
    print(f"  Best Val Accuracy:    {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Gap (Train - Val):    {final_train_acc - final_val_acc:.2f}%")
    print()
    
    print("LOSS:")
    print(f"  Final Train Loss: {final_train_loss:.4f}")
    print(f"  Final Val Loss:   {final_val_loss:.4f}")
    print(f"  Best Val Loss:    {best_val_loss:.4f}")
    print()
    
    # Test metrics si están disponibles
    if 'test_acc' in history and history['test_acc']:
        test_acc = history['test_acc'][0] if isinstance(history['test_acc'], list) else history['test_acc']
        test_loss = history['test_loss'][0] if isinstance(history['test_loss'], list) else history['test_loss']
        
        print("TEST RESULTS:")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Test Loss:     {test_loss:.4f}")
        
        if 'test_precision' in history:
            print(f"  Test Precision: {history['test_precision']:.4f}")
            print(f"  Test Recall:    {history['test_recall']:.4f}")
            print(f"  Test F1-Score:  {history['test_f1']:.4f}")
    
    print("=" * 60)

# Función de conveniencia para cargar y graficar en una sola llamada
def analyze_training_results(json_path: str, 
                           model_name: str = "Model",
                           save_plots: bool = False,
                           output_dir: str = "./") -> Dict:
    """
    Función de conveniencia para analizar completamente los resultados de entrenamiento.
    
    Args:
        json_path: Ruta al archivo JSON con el historial de entrenamiento
        model_name: Nombre del modelo para los títulos
        save_plots: Si guardar los gráficos
        output_dir: Directorio donde guardar los gráficos
        
    Returns:
        Diccionario con las métricas de entrenamiento
    """
    
    # Cargar datos
    history = load_training_history(json_path)
    
    # Imprimir resumen
    print_training_summary(history)
    
    # Crear gráficos
    if save_plots:
        save_path_1 = f"{output_dir}/{model_name}_detailed_metrics.png"
        save_path_2 = f"{output_dir}/{model_name}_combined_metrics.png"
        save_path_3 = f"{output_dir}/{model_name}_distribution.png"
    else:
        save_path_1 = save_path_2 = save_path_3 = None
    
    # Gráficos detallados
    plot_training_metrics(history, 
                         title=f"{model_name} - Detailed Training Metrics",
                         save_path=save_path_1)
    
    # Gráfico combinado (estilo original)
    plot_loss_accuracy_combined(history,
                               title=f"{model_name} - Training Progress",
                               save_path=save_path_2)
    
    # Distribuciones
    plot_metrics_distribution(history,
                             title=f"{model_name} - Metrics Distribution",
                             save_path=save_path_3)
    
    return history