
"""
SimAM-KD Evaluation Script

Evaluate a trained model on CIFAR-10 or CIFAR-100.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --dataset cifar10
    python evaluate.py --checkpoint checkpoints/best_model.pth --dataset cifar100

Authors: Raju Kumar Yadav, Rajesh Khanal, Safalta Kumari Yadav, Rikesh Kumar Shah, Bibek Kumar Gupta
"""

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from models.student import MobileNetV3SimAM
from training.pruning import count_parameters, count_flops
from utils.data_loader import get_cifar_loaders


def parse_args():
    parser = argparse.ArgumentParser(description='SimAM-KD Evaluation')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset (default: cifar10)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory (default: ./data)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Data loading workers (default: 4)')

    # Model config (if not in checkpoint)
    parser.add_argument('--attention', type=str, default='parallel',
                        choices=['none', 'simam', 'ca', 'parallel'],
                        help='Attention type (default: parallel)')
    parser.add_argument('--variant', type=str, default='small',
                        choices=['small', 'large'],
                        help='MobileNetV3 variant (default: small)')

    return parser.parse_args()


def evaluate(model, val_loader, device, num_classes):
    """Full evaluation with per-class accuracy."""
    model.eval()

    correct = 0
    total = 0
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Overall accuracy
    overall_acc = 100. * correct / total

    # Per-class accuracy
    class_acc = 100. * class_correct / (class_total + 1e-8)

    return {
        'overall_accuracy': overall_acc,
        'class_accuracy': class_acc,
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets),
    }


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get model config
    if 'args' in checkpoint:
        ckpt_args = checkpoint['args']
        num_classes = 10 if ckpt_args.get('dataset', 'cifar10') == 'cifar10' else 100
        attention_type = ckpt_args.get('attention', args.attention)
        variant = ckpt_args.get('variant', args.variant)
        dataset = ckpt_args.get('dataset', args.dataset)
    else:
        num_classes = 100 if args.dataset == 'cifar100' else 10
        attention_type = args.attention
        variant = args.variant
        dataset = args.dataset

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
    model.eval()

    # Model info
    print("\n" + "="*50)
    print("Model Information")
    print("="*50)
    print(f"Architecture: MobileNetV3-{variant.capitalize()}")
    print(f"Attention: {attention_type}")
    print(f"Parameters: {count_parameters(model)/1e6:.2f}M")
    print(f"FLOPs: {count_flops(model, device=device)/1e6:.2f}M")

    if 'best_acc' in checkpoint:
        print(f"Checkpoint best accuracy: {checkpoint['best_acc']:.2f}%")

    # Load data
    _, val_loader = get_cifar_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )

    print(f"\nDataset: {dataset.upper()}")
    print(f"Test samples: {len(val_loader.dataset)}")

    # Evaluate
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)

    results = evaluate(model, val_loader, device, num_classes)

    print(f"\nOverall Accuracy: {results['overall_accuracy']:.2f}%")

    # Top-5 and Bottom-5 classes
    class_acc = results['class_accuracy']
    sorted_idx = np.argsort(class_acc)

    if num_classes <= 10:
        print("\nPer-class Accuracy:")
        for i in range(num_classes):
            print(f"  Class {i}: {class_acc[i]:.2f}%")
    else:
        print("\nTop-5 Classes:")
        for i in sorted_idx[-5:][::-1]:
            print(f"  Class {i}: {class_acc[i]:.2f}%")

        print("\nBottom-5 Classes:")
        for i in sorted_idx[:5]:
            print(f"  Class {i}: {class_acc[i]:.2f}%")

    print(f"\nMean Class Accuracy: {np.mean(class_acc):.2f}%")
    print(f"Std Class Accuracy: {np.std(class_acc):.2f}%")

    print("\nDone!")


if __name__ == '__main__':
    main()
