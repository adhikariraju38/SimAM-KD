"""
Structured Pruning Module
Author: Raju Kumar Yadav (itsmeerajuyadav@gmail.com)

This module implements structured channel pruning for model compression.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple, Callable
import copy
from tqdm import tqdm

# Try to import torch_pruning, provide fallback if not available
try:
    import torch_pruning as tp
    TORCH_PRUNING_AVAILABLE = True
except ImportError:
    TORCH_PRUNING_AVAILABLE = False
    print("Warning: torch_pruning not installed. Install with: pip install torch-pruning")


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 32, 32), device: str = None) -> int:
    """
    Count FLOPs for a model.
    """
    try:
        from thop import profile
        # Get device from model parameters
        if device is None:
            device = next(model.parameters()).device
        x = torch.randn(*input_size).to(device)
        flops, _ = profile(model, inputs=(x,), verbose=False)
        return int(flops)
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        return 0
    except Exception as e:
        print(f"Warning: Could not count FLOPs: {e}")
        return 0


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / (1024 ** 2)


class StructuredPruner:
    """
    Structured Channel Pruning using torch_pruning library.
    """

    def __init__(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        importance_type: str = 'magnitude',  # 'magnitude', 'taylor', 'random'
        global_pruning: bool = False,
    ):
        """
        Args:
            model: Model to prune
            example_inputs: Example input tensor for graph tracing
            importance_type: Method to compute importance scores
            global_pruning: Whether to use global or layer-wise pruning
        """
        if not TORCH_PRUNING_AVAILABLE:
            raise RuntimeError("torch_pruning is required. Install with: pip install torch-pruning")

        self.model = model
        self.example_inputs = example_inputs
        self.importance_type = importance_type
        self.global_pruning = global_pruning

        # Set up importance scorer
        if importance_type == 'magnitude':
            self.importance = tp.importance.MagnitudeImportance(p=2)
        elif importance_type == 'taylor':
            self.importance = tp.importance.TaylorImportance()
        elif importance_type == 'random':
            self.importance = tp.importance.RandomImportance()
        else:
            raise ValueError(f"Unknown importance type: {importance_type}")

    def prune(
        self,
        pruning_ratio: float = 0.3,
        iterative_steps: int = 1,
        ignored_layers: Optional[List[nn.Module]] = None,
    ) -> nn.Module:
        """
        Perform structured pruning.

        Args:
            pruning_ratio: Target ratio of channels to prune
            iterative_steps: Number of iterative pruning steps
            ignored_layers: Layers to ignore during pruning

        Returns:
            Pruned model
        """
        model = copy.deepcopy(self.model)

        # Get ignored layers (typically classifier)
        if ignored_layers is None:
            ignored_layers = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and 'classifier' in name.lower():
                    ignored_layers.append(module)
                if isinstance(module, nn.Linear) and 'fc' in name.lower():
                    ignored_layers.append(module)

        # Calculate per-step pruning ratio for iterative pruning
        pruning_ratio_per_step = 1 - (1 - pruning_ratio) ** (1 / iterative_steps)

        # Create pruner
        pruner = tp.pruner.MagnitudePruner(
            model,
            self.example_inputs,
            importance=self.importance,
            iterative_steps=iterative_steps,
            pruning_ratio=pruning_ratio_per_step,
            global_pruning=self.global_pruning,
            ignored_layers=ignored_layers,
        )

        # Perform pruning
        for step in range(iterative_steps):
            pruner.step()

        return model

    def prune_and_finetune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        pruning_ratio: float = 0.3,
        finetune_epochs: int = 10,
        lr: float = 0.001,
        device: str = 'cuda',
    ) -> Tuple[nn.Module, Dict]:
        """
        Prune model and fine-tune to recover accuracy.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            pruning_ratio: Target pruning ratio
            finetune_epochs: Number of fine-tuning epochs
            lr: Learning rate for fine-tuning
            device: Device to use

        Returns:
            Pruned and fine-tuned model, metrics
        """
        # Get original metrics
        original_params = count_parameters(self.model)
        original_flops = count_flops(self.model)
        original_size = get_model_size_mb(self.model)
        original_acc = self._evaluate(self.model, val_loader, device)

        print(f"\nOriginal Model:")
        print(f"  Parameters: {original_params / 1e6:.2f}M")
        print(f"  FLOPs: {original_flops / 1e9:.3f}G")
        print(f"  Size: {original_size:.2f}MB")
        print(f"  Accuracy: {original_acc:.2f}%")

        # Prune
        print(f"\nPruning with ratio {pruning_ratio:.2f}...")
        pruned_model = self.prune(pruning_ratio)
        pruned_model = pruned_model.to(device)

        pruned_params = count_parameters(pruned_model)
        pruned_flops = count_flops(pruned_model)
        pruned_size = get_model_size_mb(pruned_model)
        pruned_acc_before = self._evaluate(pruned_model, val_loader, device)

        print(f"\nPruned Model (before fine-tuning):")
        print(f"  Parameters: {pruned_params / 1e6:.2f}M ({100 * (1 - pruned_params / original_params):.1f}% reduction)")
        print(f"  FLOPs: {pruned_flops / 1e9:.3f}G ({100 * (1 - pruned_flops / max(original_flops, 1)):.1f}% reduction)")
        print(f"  Size: {pruned_size:.2f}MB ({100 * (1 - pruned_size / original_size):.1f}% reduction)")
        print(f"  Accuracy: {pruned_acc_before:.2f}% ({pruned_acc_before - original_acc:+.2f}%)")

        # Fine-tune
        print(f"\nFine-tuning for {finetune_epochs} epochs...")
        history = self._finetune(
            pruned_model, train_loader, val_loader, finetune_epochs, lr, device
        )

        pruned_acc_after = self._evaluate(pruned_model, val_loader, device)

        print(f"\nPruned Model (after fine-tuning):")
        print(f"  Accuracy: {pruned_acc_after:.2f}% ({pruned_acc_after - original_acc:+.2f}%)")

        metrics = {
            'original': {
                'params': original_params,
                'flops': original_flops,
                'size_mb': original_size,
                'accuracy': original_acc,
            },
            'pruned': {
                'params': pruned_params,
                'flops': pruned_flops,
                'size_mb': pruned_size,
                'accuracy_before_ft': pruned_acc_before,
                'accuracy_after_ft': pruned_acc_after,
            },
            'reduction': {
                'params': 1 - pruned_params / original_params,
                'flops': 1 - pruned_flops / max(original_flops, 1),
                'size': 1 - pruned_size / original_size,
            },
            'finetune_history': history,
        }

        return pruned_model, metrics

    def _evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: str,
    ) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100. * correct / total

    def _finetune(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        device: str,
    ) -> Dict:
        """Fine-tune pruned model."""
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        best_acc = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in tqdm(train_loader, desc=f'FT Epoch {epoch}'):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            scheduler.step()

            train_acc = 100. * correct / total
            val_acc = self._evaluate(model, val_loader, device)

            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc

            print(f"  Epoch {epoch}: Train={train_acc:.2f}%, Val={val_acc:.2f}%")

        return history


class ManualChannelPruner:
    """
    Manual channel pruning without external dependencies.

    This is a fallback when torch_pruning is not available.
    """

    @staticmethod
    def prune_conv_layer(
        conv: nn.Conv2d,
        bn: Optional[nn.BatchNorm2d],
        indices_to_keep: torch.Tensor,
        prune_output: bool = True,
    ) -> Tuple[nn.Conv2d, Optional[nn.BatchNorm2d]]:
        """
        Prune a convolutional layer.

        Args:
            conv: Convolution layer to prune
            bn: Associated batch norm layer
            indices_to_keep: Channel indices to keep
            prune_output: If True, prune output channels; if False, prune input channels

        Returns:
            Pruned conv and bn layers
        """
        if prune_output:
            # Prune output channels
            new_out_channels = len(indices_to_keep)
            new_conv = nn.Conv2d(
                conv.in_channels, new_out_channels,
                conv.kernel_size, conv.stride, conv.padding,
                conv.dilation, conv.groups, conv.bias is not None
            )

            with torch.no_grad():
                new_conv.weight.copy_(conv.weight[indices_to_keep])
                if conv.bias is not None:
                    new_conv.bias.copy_(conv.bias[indices_to_keep])

            if bn is not None:
                new_bn = nn.BatchNorm2d(new_out_channels)
                with torch.no_grad():
                    new_bn.weight.copy_(bn.weight[indices_to_keep])
                    new_bn.bias.copy_(bn.bias[indices_to_keep])
                    new_bn.running_mean.copy_(bn.running_mean[indices_to_keep])
                    new_bn.running_var.copy_(bn.running_var[indices_to_keep])
            else:
                new_bn = None
        else:
            # Prune input channels
            new_in_channels = len(indices_to_keep)
            new_conv = nn.Conv2d(
                new_in_channels, conv.out_channels,
                conv.kernel_size, conv.stride, conv.padding,
                conv.dilation, conv.groups, conv.bias is not None
            )

            with torch.no_grad():
                new_conv.weight.copy_(conv.weight[:, indices_to_keep])
                if conv.bias is not None:
                    new_conv.bias.copy_(conv.bias)

            new_bn = bn

        return new_conv, new_bn

    @staticmethod
    def compute_channel_importance(
        conv: nn.Conv2d,
        method: str = 'l2',
    ) -> torch.Tensor:
        """
        Compute importance scores for each output channel.

        Args:
            conv: Convolution layer
            method: 'l1', 'l2', or 'random'

        Returns:
            Importance scores for each channel
        """
        weights = conv.weight.data  # (out_channels, in_channels, H, W)

        if method == 'l1':
            importance = weights.abs().sum(dim=[1, 2, 3])
        elif method == 'l2':
            importance = (weights ** 2).sum(dim=[1, 2, 3]).sqrt()
        elif method == 'random':
            importance = torch.rand(weights.size(0))
        else:
            raise ValueError(f"Unknown method: {method}")

        return importance


def prune_model_simple(
    model: nn.Module,
    pruning_ratio: float = 0.3,
    input_size: Tuple[int, ...] = (1, 3, 32, 32),
) -> nn.Module:
    """
    Simple pruning using torch_pruning if available.

    Args:
        model: Model to prune
        pruning_ratio: Ratio of channels to prune
        input_size: Input tensor size

    Returns:
        Pruned model
    """
    if TORCH_PRUNING_AVAILABLE:
        example_inputs = torch.randn(*input_size)
        pruner = StructuredPruner(model, example_inputs)
        return pruner.prune(pruning_ratio)
    else:
        print("Warning: torch_pruning not available. Returning original model.")
        return model
