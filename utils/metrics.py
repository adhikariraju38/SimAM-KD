"""
Metrics and Evaluation Utilities
Author: Raju Kumar Yadav (itsmeerajuyadav@gmail.com)

This module provides utilities for computing and logging metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import time
from sklearn.metrics import confusion_matrix, classification_report
import json
import os


def compute_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda',
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """
    Compute top-k accuracy on a dataset.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device
        topk: Tuple of k values for top-k accuracy

    Returns:
        Dictionary with accuracy metrics
    """
    model.eval()
    maxk = max(topk)

    correct = {k: 0 for k in topk}
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct_mask = pred.eq(targets.view(1, -1).expand_as(pred))

            for k in topk:
                correct[k] += correct_mask[:k].reshape(-1).float().sum(0).item()

            total += targets.size(0)

    accuracy = {f'top{k}_acc': 100. * correct[k] / total for k in topk}
    return accuracy


def compute_inference_time(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 32, 32),
    device: str = 'cuda',
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """
    Measure inference time.

    Args:
        model: Model to benchmark
        input_size: Input tensor size
        device: Device
        num_runs: Number of runs for timing
        warmup_runs: Number of warmup runs

    Returns:
        Dictionary with timing metrics (ms)
    """
    model = model.to(device)
    model.eval()

    x = torch.randn(*input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Timing runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'fps': float(1000 / np.mean(times)),
    }


def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get all predictions from a model.

    Returns:
        predictions: Predicted class labels
        targets: Ground truth labels
        probabilities: Class probabilities
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
            all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_probs)
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_normalized, annot=False, cmap='Blues', ax=ax,
        xticklabels=class_names, yticklabels=class_names
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Plot training curves.

    Args:
        history: Dictionary with training history
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Loss curve
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    if 'lr' in history:
        axes[2].plot(history['lr'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_comparison_bar(
    methods: List[str],
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot comparison bar chart.

    Args:
        methods: List of method names
        metrics: Dictionary mapping metric names to values
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_methods = len(methods)
    n_metrics = len(metrics)
    x = np.arange(n_methods)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=figsize)

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=metric_name)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Method Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def save_results(
    results: Dict,
    save_path: str,
    indent: int = 2,
):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=indent, default=str)


def load_results(load_path: str) -> Dict:
    """Load results from JSON file."""
    with open(load_path, 'r') as f:
        return json.load(f)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ResultsLogger:
    """Logger for experiment results."""

    def __init__(self, save_dir: str, experiment_name: str):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.results = {
            'experiment_name': experiment_name,
            'metrics': {},
            'history': {},
            'config': {},
        }

        os.makedirs(save_dir, exist_ok=True)

    def log_config(self, config: Dict):
        """Log experiment configuration."""
        self.results['config'] = config

    def log_metric(self, name: str, value: float):
        """Log a single metric."""
        self.results['metrics'][name] = value

    def log_metrics(self, metrics: Dict[str, float]):
        """Log multiple metrics."""
        self.results['metrics'].update(metrics)

    def log_history(self, history: Dict[str, List[float]]):
        """Log training history."""
        self.results['history'] = history

    def save(self):
        """Save results to file."""
        save_path = os.path.join(self.save_dir, f'{self.experiment_name}_results.json')
        save_results(self.results, save_path)
        print(f"Results saved to {save_path}")

    def get_summary(self) -> str:
        """Get a text summary of results."""
        lines = [
            f"Experiment: {self.experiment_name}",
            "-" * 40,
            "Metrics:",
        ]
        for name, value in self.results['metrics'].items():
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.4f}")
            else:
                lines.append(f"  {name}: {value}")
        return "\n".join(lines)
