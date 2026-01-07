
"""
SimAM-KD Training Script

Train MobileNetV3 with attention mechanisms and knowledge distillation.

Usage:
    python train.py --dataset cifar10 --attention parallel
    python train.py --dataset cifar100 --attention parallel --temperature 4.0 --alpha 0.7
    python train.py --dataset cifar10 --attention parallel --no-kd  # Without KD

Authors: Raju Kumar Yadav, Rajesh Khanal, Safalta Kumari Yadav, Rikesh Kumar Shah, Bibek Kumar Gupta
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.student import MobileNetV3SimAM
from models.teacher import get_teacher_model
from training.distillation import DistillationTrainer, train_teacher
from utils.data_loader import get_cifar_loaders


def parse_args():
    parser = argparse.ArgumentParser(description='SimAM-KD Training')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to use (default: cifar10)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory (default: ./data)')

    # Model
    parser.add_argument('--attention', type=str, default='parallel',
                        choices=['none', 'simam', 'ca', 'parallel'],
                        help='Attention type (default: parallel)')
    parser.add_argument('--variant', type=str, default='small',
                        choices=['small', 'large'],
                        help='MobileNetV3 variant (default: small)')

    # Knowledge Distillation
    parser.add_argument('--no-kd', action='store_true',
                        help='Train without knowledge distillation')
    parser.add_argument('--teacher', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'wide_resnet'],
                        help='Teacher model (default: resnet50)')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='KD temperature (default: 4.0)')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='KD loss weight (default: 0.7)')

    # Training
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of epochs (default: 80)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay (default: 5e-4)')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Output
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Output directory (default: ./checkpoints)')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Dataset
    num_classes = 10 if args.dataset == 'cifar10' else 100
    train_loader, val_loader = get_cifar_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )
    print(f"Dataset: {args.dataset.upper()} ({num_classes} classes)")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Create student model
    student = MobileNetV3SimAM(
        num_classes=num_classes,
        variant=args.variant,
        attention_type=args.attention,
        input_size=32
    )
    student_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"\nStudent: MobileNetV3-{args.variant.capitalize()} + {args.attention.upper()}")
    print(f"Parameters: {student_params:.2f}M")

    # Experiment name
    if args.exp_name is None:
        kd_str = f"_T{args.temperature}_a{args.alpha}" if not args.no_kd else "_nokd"
        args.exp_name = f"{args.dataset}_{args.attention}{kd_str}"

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{args.exp_name}_best.pth")

    if args.no_kd:
        # Train without knowledge distillation
        print("\n" + "="*50)
        print("Training WITHOUT Knowledge Distillation")
        print("="*50)

        student = student.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            student.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(1, args.epochs + 1):
            # Training
            student.train()
            train_loss, correct, total = 0.0, 0, 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = student(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_acc = 100. * correct / total

            # Validation
            student.eval()
            val_loss, correct, total = 0.0, 0, 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = student(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            val_acc = 100. * correct / total
            scheduler.step()

            # Update history
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch}/{args.epochs}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

            # Save best
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'state_dict': student.state_dict(),
                    'best_acc': best_acc,
                    'args': vars(args),
                }, save_path)
                print(f"  -> New best: {best_acc:.2f}%")

        print(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")

    else:
        # Train with knowledge distillation
        print("\n" + "="*50)
        print("Training WITH Knowledge Distillation")
        print("="*50)

        # Create/load teacher
        print(f"\nTeacher: {args.teacher}")
        teacher = get_teacher_model(args.teacher, num_classes=num_classes, pretrained=True)
        teacher = teacher.to(device)
        teacher.eval()

        # Evaluate teacher
        teacher_correct, teacher_total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = teacher(inputs)
                _, predicted = outputs.max(1)
                teacher_total += targets.size(0)
                teacher_correct += predicted.eq(targets).sum().item()
        teacher_acc = 100. * teacher_correct / teacher_total
        print(f"Teacher accuracy: {teacher_acc:.2f}%")

        # Setup optimizer
        optimizer = optim.SGD(
            student.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Create trainer
        trainer = DistillationTrainer(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            temperature=args.temperature,
            alpha=args.alpha,
            use_amp=True
        )

        print(f"\nKD Settings: T={args.temperature}, alpha={args.alpha}")
        print(f"Training for {args.epochs} epochs...")

        # Train
        history = trainer.train(
            epochs=args.epochs,
            save_path=save_path,
            early_stopping=20
        )

        print(f"\nBest student accuracy: {trainer.best_acc:.2f}%")

    # Save training history
    history_path = os.path.join(args.output_dir, f"{args.exp_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to: {history_path}")

    print(f"\nModel saved to: {save_path}")
    print("Done!")


if __name__ == '__main__':
    main()
