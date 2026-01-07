
"""
SimAM-KD Pruning Script

Apply structured channel pruning to a trained model and fine-tune.

Usage:
    python prune.py --checkpoint checkpoints/best_model.pth --pruning-ratio 0.2
    python prune.py --checkpoint checkpoints/best_model.pth --pruning-ratio 0.3 --finetune-epochs 30

Authors: Raju Kumar Yadav, Rajesh Khanal, Safalta Kumari Yadav, Rikesh Kumar Shah, Bibek Kumar Gupta
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.student import MobileNetV3SimAM
from training.pruning import apply_structured_pruning, count_parameters, count_flops
from utils.data_loader import get_cifar_loaders


def parse_args():
    parser = argparse.ArgumentParser(description='SimAM-KD Pruning')

    # Input
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')

    # Pruning
    parser.add_argument('--pruning-ratio', type=float, default=0.2,
                        help='Pruning ratio (default: 0.2)')

    # Fine-tuning
    parser.add_argument('--finetune-epochs', type=int, default=30,
                        help='Fine-tuning epochs after pruning (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Fine-tuning learning rate (default: 0.01)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset (default: cifar10)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory (default: ./data)')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Data loading workers (default: 4)')

    # Output
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Output directory (default: ./checkpoints)')

    return parser.parse_args()


def evaluate(model, val_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / total


def fine_tune(model, train_loader, val_loader, epochs, lr, device):
    """Fine-tune pruned model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"  Fine-tune Epoch {epoch}/{epochs}: Val Acc = {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_acc


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get model config from checkpoint
    if 'args' in checkpoint:
        ckpt_args = checkpoint['args']
        num_classes = 10 if ckpt_args.get('dataset', 'cifar10') == 'cifar10' else 100
        attention_type = ckpt_args.get('attention', 'parallel')
        variant = ckpt_args.get('variant', 'small')
    else:
        num_classes = 100 if args.dataset == 'cifar100' else 10
        attention_type = 'parallel'
        variant = 'small'

    # Create model
    model = MobileNetV3SimAM(
        num_classes=num_classes,
        variant=variant,
        attention_type=attention_type,
        input_size=32
    )

    # Load weights
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'student_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['student_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    # Get data loaders
    train_loader, val_loader = get_cifar_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )

    # Evaluate before pruning
    print("\n" + "="*50)
    print("Before Pruning")
    print("="*50)

    original_acc = evaluate(model, val_loader, device)
    original_params = count_parameters(model)
    original_flops = count_flops(model, device=device)

    print(f"Accuracy: {original_acc:.2f}%")
    print(f"Parameters: {original_params/1e6:.2f}M")
    print(f"FLOPs: {original_flops/1e6:.2f}M")

    # Apply pruning
    print("\n" + "="*50)
    print(f"Applying {args.pruning_ratio*100:.0f}% Pruning")
    print("="*50)

    example_input = torch.randn(1, 3, 32, 32).to(device)
    pruned_model, stats = apply_structured_pruning(
        model=model,
        pruning_ratio=args.pruning_ratio,
        example_input=example_input
    )

    # Evaluate after pruning (before fine-tuning)
    pruned_acc_before = evaluate(pruned_model, val_loader, device)
    pruned_params = count_parameters(pruned_model)
    pruned_flops = count_flops(pruned_model, device=device)

    print(f"\nAfter Pruning (before fine-tuning):")
    print(f"Accuracy: {pruned_acc_before:.2f}% (drop: {original_acc - pruned_acc_before:.2f}%)")
    print(f"Parameters: {pruned_params/1e6:.2f}M ({(1-pruned_params/original_params)*100:.1f}% reduction)")
    print(f"FLOPs: {pruned_flops/1e6:.2f}M ({(1-pruned_flops/original_flops)*100:.1f}% reduction)")

    # Fine-tune
    print("\n" + "="*50)
    print(f"Fine-tuning for {args.finetune_epochs} epochs")
    print("="*50)

    final_acc = fine_tune(
        pruned_model, train_loader, val_loader,
        epochs=args.finetune_epochs,
        lr=args.lr,
        device=device
    )

    # Final results
    print("\n" + "="*50)
    print("Final Results")
    print("="*50)
    print(f"Original Accuracy: {original_acc:.2f}%")
    print(f"Final Accuracy: {final_acc:.2f}%")
    print(f"Accuracy Drop: {original_acc - final_acc:.2f}%")
    print(f"Parameter Reduction: {(1-pruned_params/original_params)*100:.1f}%")
    print(f"FLOPs Reduction: {(1-pruned_flops/original_flops)*100:.1f}%")

    # Save pruned model
    os.makedirs(args.output_dir, exist_ok=True)
    output_name = os.path.basename(args.checkpoint).replace('.pth', f'_pruned{int(args.pruning_ratio*100)}.pth')
    output_path = os.path.join(args.output_dir, output_name)

    torch.save({
        'state_dict': pruned_model.state_dict(),
        'accuracy': final_acc,
        'pruning_ratio': args.pruning_ratio,
        'original_params': original_params,
        'pruned_params': pruned_params,
        'param_reduction': (1 - pruned_params/original_params) * 100,
    }, output_path)

    print(f"\nPruned model saved to: {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()
