"""
Main Experiment Runner for SimAM-KD Framework
Author: Raju Kumar Yadav (itsmeerajuyadav@gmail.com)

This script runs the complete experimental pipeline:
1. Train teacher model (or load pre-trained)
2. Train student models with different attention types
3. Apply knowledge distillation
4. Apply pruning and fine-tuning
5. Evaluate and compare results
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    MobileNetV3SimAM,
    mobilenetv3_simam_small,
    mobilenetv3_simam_large,
    get_teacher_model,
)
from training import (
    DistillationTrainer,
    train_teacher,
    StructuredPruner,
    count_parameters,
    count_flops,
    get_model_size_mb,
)
from utils import (
    get_data_loaders,
    compute_accuracy,
    compute_inference_time,
    plot_training_curves,
    plot_comparison_bar,
    save_results,
    ResultsLogger,
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def run_baseline_experiment(
    config: dict,
    train_loader,
    test_loader,
    device: str,
    results_dir: str,
):
    """Run baseline experiments without distillation."""
    print("\n" + "=" * 60)
    print("PHASE 1: Baseline Experiments (No Distillation)")
    print("=" * 60)

    results = {}
    attention_types = config.get('attention_types', ['none', 'simam', 'ca', 'parallel'])

    for att_type in attention_types:
        print(f"\n--- Training MobileNetV3-Small with {att_type.upper()} attention ---")

        model = mobilenetv3_simam_small(
            num_classes=config['num_classes'],
            attention_type=att_type,
            input_size=config.get('input_size', 32),
        ).to(device)

        # Count parameters
        params = count_parameters(model)
        flops = count_flops(model, input_size=(1, 3, config.get('input_size', 32), config.get('input_size', 32)))
        model_size = get_model_size_mb(model)

        print(f"  Parameters: {params / 1e6:.2f}M")
        print(f"  FLOPs: {flops / 1e9:.3f}G")
        print(f"  Model Size: {model_size:.2f}MB")

        # Training
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 0.001),
            weight_decay=config.get('weight_decay', 1e-4),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.get('baseline_epochs', 100)
        )
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

        for epoch in range(1, config.get('baseline_epochs', 100) + 1):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
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
            val_metrics = compute_accuracy(model, test_loader, device)
            val_acc = val_metrics['top1_acc']

            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"{results_dir}/baseline_{att_type}_best.pth")

            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: Train={train_acc:.2f}%, Val={val_acc:.2f}%")

        # Inference time
        timing = compute_inference_time(model, device=device)

        results[att_type] = {
            'accuracy': best_acc,
            'params': params,
            'flops': flops,
            'model_size_mb': model_size,
            'inference_time_ms': timing['mean_ms'],
            'fps': timing['fps'],
            'history': history,
        }

        print(f"  Best Accuracy: {best_acc:.2f}%")
        print(f"  Inference Time: {timing['mean_ms']:.2f}ms ({timing['fps']:.1f} FPS)")

    return results


def run_distillation_experiment(
    config: dict,
    train_loader,
    test_loader,
    device: str,
    results_dir: str,
):
    """Run knowledge distillation experiments."""
    print("\n" + "=" * 60)
    print("PHASE 2: Knowledge Distillation Experiments")
    print("=" * 60)

    results = {}

    # Prepare teacher
    teacher_name = config.get('teacher_model', 'resnet50')
    teacher_path = f"{results_dir}/teacher_{teacher_name}.pth"

    print(f"\n--- Preparing Teacher: {teacher_name} ---")
    teacher = get_teacher_model(
        teacher_name,
        num_classes=config['num_classes'],
        pretrained=True,
        input_size=config.get('input_size', 32),
    ).to(device)

    # Check if teacher already trained
    if os.path.exists(teacher_path):
        print("  Loading pre-trained teacher...")
        teacher.load_state_dict(torch.load(teacher_path, map_location=device)['state_dict'])
    else:
        print("  Training teacher from scratch...")
        teacher_history = train_teacher(
            teacher, train_loader, test_loader,
            epochs=config.get('teacher_epochs', 100),
            lr=0.1, device=device, save_path=teacher_path
        )

    # Evaluate teacher
    teacher_acc = compute_accuracy(teacher, test_loader, device)['top1_acc']
    print(f"  Teacher Accuracy: {teacher_acc:.2f}%")

    # Train students with KD
    attention_types = config.get('attention_types', ['none', 'simam', 'ca', 'parallel'])

    for att_type in attention_types:
        print(f"\n--- Training Student with KD ({att_type.upper()}) ---")

        student = mobilenetv3_simam_small(
            num_classes=config['num_classes'],
            attention_type=att_type,
            input_size=config.get('input_size', 32),
        ).to(device)

        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=config.get('kd_lr', 0.001),
            weight_decay=config.get('weight_decay', 1e-4),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.get('kd_epochs', 100)
        )

        trainer = DistillationTrainer(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            val_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            temperature=config.get('temperature', 4.0),
            alpha=config.get('alpha', 0.7),
        )

        history = trainer.train(
            epochs=config.get('kd_epochs', 100),
            save_path=f"{results_dir}/kd_{att_type}_best.pth",
            early_stopping=20,
        )

        # Metrics
        params = count_parameters(student)
        timing = compute_inference_time(student, device=device)

        results[att_type] = {
            'accuracy': trainer.best_acc,
            'params': params,
            'inference_time_ms': timing['mean_ms'],
            'fps': timing['fps'],
            'history': history,
        }

    return results


def run_pruning_experiment(
    config: dict,
    train_loader,
    test_loader,
    device: str,
    results_dir: str,
):
    """Run pruning experiments on the best KD model."""
    print("\n" + "=" * 60)
    print("PHASE 3: Pruning Experiments")
    print("=" * 60)

    results = {}
    pruning_ratios = config.get('pruning_ratios', [0.2, 0.3, 0.4])
    best_att_type = config.get('best_attention', 'simam')

    for ratio in pruning_ratios:
        print(f"\n--- Pruning with ratio {ratio:.0%} ---")

        # Load best KD model
        model = mobilenetv3_simam_small(
            num_classes=config['num_classes'],
            attention_type=best_att_type,
            input_size=config.get('input_size', 32),
        ).to(device)

        checkpoint = torch.load(
            f"{results_dir}/kd_{best_att_type}_best.pth",
            map_location=device
        )
        model.load_state_dict(checkpoint['student_state_dict'])

        original_params = count_parameters(model)
        original_acc = compute_accuracy(model, test_loader, device)['top1_acc']

        print(f"  Before pruning: {original_params/1e6:.2f}M params, {original_acc:.2f}% acc")

        # Prune
        example_inputs = torch.randn(1, 3, config.get('input_size', 32), config.get('input_size', 32)).to(device)

        try:
            pruner = StructuredPruner(model, example_inputs, importance_type='magnitude')
            pruned_model, metrics = pruner.prune_and_finetune(
                train_loader, test_loader,
                pruning_ratio=ratio,
                finetune_epochs=config.get('finetune_epochs', 20),
                lr=config.get('finetune_lr', 0.0001),
                device=device,
            )

            results[f'prune_{int(ratio*100)}'] = metrics

            # Save pruned model
            torch.save(
                pruned_model.state_dict(),
                f"{results_dir}/pruned_{int(ratio*100)}_{best_att_type}.pth"
            )
        except Exception as e:
            print(f"  Pruning failed: {e}")
            results[f'prune_{int(ratio*100)}'] = {'error': str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description='SimAM-KD Experiments')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['all', 'baseline', 'distillation', 'pruning'],
                       help='Which phase to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load config
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            'dataset': 'cifar10',
            'num_classes': 10,
            'batch_size': 128,
            'input_size': 32,
            'baseline_epochs': 100,
            'kd_epochs': 100,
            'teacher_epochs': 100,
            'finetune_epochs': 20,
            'lr': 0.001,
            'kd_lr': 0.001,
            'finetune_lr': 0.0001,
            'weight_decay': 1e-4,
            'temperature': 4.0,
            'alpha': 0.7,
            'teacher_model': 'resnet50',
            'attention_types': ['none', 'simam', 'ca', 'parallel'],
            'pruning_ratios': [0.2, 0.3, 0.4],
            'best_attention': 'simam',
        }

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"{args.results_dir}/{config['dataset']}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # Save config
    with open(f"{results_dir}/config.yaml", 'w') as f:
        yaml.dump(config, f)

    # Get data loaders
    print("\nLoading data...")
    train_loader, test_loader, _, num_classes = get_data_loaders(
        dataset=config['dataset'],
        batch_size=config['batch_size'],
        num_workers=4,
        data_dir=args.data_dir,
        augmentation='standard',
    )
    config['num_classes'] = num_classes
    print(f"Dataset: {config['dataset']} ({num_classes} classes)")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Run experiments
    all_results = {}

    if args.phase in ['all', 'baseline']:
        baseline_results = run_baseline_experiment(
            config, train_loader, test_loader, device, results_dir
        )
        all_results['baseline'] = baseline_results

    if args.phase in ['all', 'distillation']:
        kd_results = run_distillation_experiment(
            config, train_loader, test_loader, device, results_dir
        )
        all_results['distillation'] = kd_results

    if args.phase in ['all', 'pruning']:
        pruning_results = run_pruning_experiment(
            config, train_loader, test_loader, device, results_dir
        )
        all_results['pruning'] = pruning_results

    # Save all results
    save_results(all_results, f"{results_dir}/all_results.json")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    if 'baseline' in all_results:
        print("\nBaseline Results (No KD):")
        for att, res in all_results['baseline'].items():
            print(f"  {att}: {res['accuracy']:.2f}% ({res['params']/1e6:.2f}M params)")

    if 'distillation' in all_results:
        print("\nKnowledge Distillation Results:")
        for att, res in all_results['distillation'].items():
            print(f"  {att}: {res['accuracy']:.2f}%")

    if 'pruning' in all_results:
        print("\nPruning Results:")
        for key, res in all_results['pruning'].items():
            if 'error' not in res:
                print(f"  {key}: {res['pruned']['accuracy_after_ft']:.2f}% "
                      f"({res['reduction']['params']*100:.1f}% param reduction)")

    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
