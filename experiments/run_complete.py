"""
COMPLETE Experiment Suite for SimAM-KD Paper
Author: Raju Kumar Yadav (itsmeerajuyadav@gmail.com)

This script runs ALL experiments needed for the paper:
1. All attention types (none, simam, ca, parallel) with KD
2. Ablation study: Temperature (T=2,4,6,8)
3. Ablation study: Alpha (0.5, 0.7, 0.9)
4. Pruning ratios (10%, 15%, 20%, 25%, 30%)
5. CIFAR-100 experiments (optional)

Estimated time: 4-5 hours on RTX 4050
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import time
import json
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import mobilenetv3_simam_small, get_teacher_model
from training import count_parameters, get_model_size_mb, StructuredPruner
from utils import get_data_loaders, compute_accuracy, save_results


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class KDTrainer:
    """Knowledge Distillation Trainer"""

    def __init__(self, student, teacher, train_loader, val_loader, device,
                 temperature=4.0, alpha=0.9, lr=0.1):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.T = temperature
        self.alpha = alpha

        self.optimizer = torch.optim.SGD(
            student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True
        )
        self.scaler = GradScaler()
        self.best_acc = 0.0
        self.best_state = None
        self.history = {'train_acc': [], 'val_acc': [], 'train_loss': []}

    def train_epoch(self):
        self.student.train()
        total_loss, correct, total = 0, 0, 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                with torch.no_grad():
                    t_logits = self.teacher(inputs)
                s_logits = self.student(inputs)

                soft_loss = F.kl_div(
                    F.log_softmax(s_logits / self.T, dim=1),
                    F.softmax(t_logits / self.T, dim=1),
                    reduction='batchmean'
                ) * (self.T ** 2)
                hard_loss = F.cross_entropy(s_logits, targets)
                loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            _, pred = s_logits.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()

        return total_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def validate(self):
        self.student.eval()
        correct, total = 0, 0
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with autocast():
                outputs = self.student(inputs)
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
        return 100. * correct / total

    def train(self, epochs, verbose=True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-5
        )

        pbar = tqdm(range(1, epochs + 1), desc='Training', disable=not verbose)
        for epoch in pbar:
            train_loss, train_acc = self.train_epoch()
            val_acc = self.validate()
            scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_state = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}

            pbar.set_postfix({'val': f'{val_acc:.1f}%', 'best': f'{self.best_acc:.1f}%'})

        if self.best_state:
            self.student.load_state_dict(self.best_state)

        return self.best_acc, self.history


def finetune_pruned(model, train_loader, val_loader, device, epochs=30, lr=0.01):
    """Fine-tune pruned model"""
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in tqdm(range(1, epochs + 1), desc='Fine-tuning', leave=False):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss = criterion(model(inputs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        val_acc = compute_accuracy(model, val_loader, device)['top1_acc']
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, best_acc


def run_attention_comparison(train_loader, test_loader, teacher, device, results_dir, epochs=60):
    """Run KD for all attention types"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Attention Type Comparison")
    print("=" * 60)

    attention_types = ['none', 'simam', 'ca', 'parallel']
    results = {}

    for att in attention_types:
        print(f"\n>>> {att.upper()} attention...")
        set_seed(42)

        student = mobilenetv3_simam_small(num_classes=10, attention_type=att, input_size=32)
        trainer = KDTrainer(student, teacher, train_loader, test_loader, device,
                           temperature=4.0, alpha=0.9, lr=0.1)
        best_acc, history = trainer.train(epochs)

        # Save model
        torch.save({
            'state_dict': trainer.student.state_dict(),
            'best_acc': best_acc,
            'history': history,
        }, f"{results_dir}/attention_{att}.pth")

        results[att] = {
            'accuracy': best_acc,
            'params': count_parameters(student),
            'history': history,
        }
        print(f"    Result: {best_acc:.2f}%")

    return results


def run_temperature_ablation(train_loader, test_loader, teacher, device, results_dir, epochs=40):
    """Ablation study for temperature"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Temperature Ablation (T=2,4,6,8)")
    print("=" * 60)

    temperatures = [2.0, 4.0, 6.0, 8.0]
    results = {}

    for T in temperatures:
        print(f"\n>>> Temperature = {T}...")
        set_seed(42)

        student = mobilenetv3_simam_small(num_classes=10, attention_type='parallel', input_size=32)
        trainer = KDTrainer(student, teacher, train_loader, test_loader, device,
                           temperature=T, alpha=0.9, lr=0.1)
        best_acc, history = trainer.train(epochs)

        results[f'T={T}'] = {
            'temperature': T,
            'accuracy': best_acc,
            'history': history,
        }
        print(f"    Result: {best_acc:.2f}%")

    return results


def run_alpha_ablation(train_loader, test_loader, teacher, device, results_dir, epochs=40):
    """Ablation study for alpha"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Alpha Ablation (0.5, 0.7, 0.9)")
    print("=" * 60)

    alphas = [0.5, 0.7, 0.9]
    results = {}

    for alpha in alphas:
        print(f"\n>>> Alpha = {alpha}...")
        set_seed(42)

        student = mobilenetv3_simam_small(num_classes=10, attention_type='parallel', input_size=32)
        trainer = KDTrainer(student, teacher, train_loader, test_loader, device,
                           temperature=4.0, alpha=alpha, lr=0.1)
        best_acc, history = trainer.train(epochs)

        results[f'alpha={alpha}'] = {
            'alpha': alpha,
            'accuracy': best_acc,
            'history': history,
        }
        print(f"    Result: {best_acc:.2f}%")

    return results


def run_pruning_ablation(train_loader, test_loader, device, results_dir, model_path, epochs_ft=30):
    """Ablation study for pruning ratios"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Pruning Ratio Ablation")
    print("=" * 60)

    pruning_ratios = [0.10, 0.15, 0.20, 0.25, 0.30]
    results = {}

    # Load best KD model
    ckpt = torch.load(model_path, map_location=device)
    original_acc = ckpt['best_acc']
    original_state = ckpt['state_dict']

    student = mobilenetv3_simam_small(num_classes=10, attention_type='parallel', input_size=32)
    student.load_state_dict(original_state)
    original_params = count_parameters(student)

    print(f"Original model: {original_acc:.2f}%, {original_params/1e6:.2f}M params")

    for ratio in pruning_ratios:
        print(f"\n>>> Pruning {int(ratio*100)}%...")

        # Fresh copy
        student = mobilenetv3_simam_small(num_classes=10, attention_type='parallel', input_size=32).to(device)
        student.load_state_dict(original_state)

        # Prune
        example_inputs = torch.randn(1, 3, 32, 32).to(device)
        try:
            pruner = StructuredPruner(student, example_inputs)
            pruned = pruner.prune(pruning_ratio=ratio)
            pruned = pruned.to(device)

            pruned_params = count_parameters(pruned)
            param_reduction = (1 - pruned_params / original_params) * 100

            # Fine-tune
            pruned, final_acc = finetune_pruned(pruned, train_loader, test_loader, device, epochs_ft)

            results[f'{int(ratio*100)}%'] = {
                'pruning_ratio': ratio,
                'accuracy': final_acc,
                'params': pruned_params,
                'param_reduction': param_reduction,
                'accuracy_drop': original_acc - final_acc,
            }

            # Save
            torch.save(pruned.state_dict(), f"{results_dir}/pruned_{int(ratio*100)}.pth")
            print(f"    Result: {final_acc:.2f}% ({param_reduction:.1f}% reduction)")

        except Exception as e:
            print(f"    Error: {e}")
            results[f'{int(ratio*100)}%'] = {'error': str(e)}

    return results


def run_cifar100_experiment(device, results_dir, epochs=60):
    """Run experiments on CIFAR-100"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: CIFAR-100")
    print("=" * 60)

    # Load CIFAR-100
    train_loader, test_loader, _, _ = get_data_loaders(
        dataset='cifar100', batch_size=128, num_workers=4, augmentation='standard'
    )

    # Train teacher
    print("\n>>> Training teacher on CIFAR-100...")
    teacher = get_teacher_model('resnet50', num_classes=100, pretrained=True, input_size=32).to(device)

    # Quick teacher training
    optimizer = torch.optim.SGD(teacher.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_teacher_acc = 0
    for epoch in tqdm(range(1, 61), desc='Teacher'):
        teacher.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(teacher(inputs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        val_acc = compute_accuracy(teacher, test_loader, device)['top1_acc']
        if val_acc > best_teacher_acc:
            best_teacher_acc = val_acc

    print(f"    Teacher: {best_teacher_acc:.2f}%")

    # Train student with KD
    print("\n>>> Training student with KD on CIFAR-100...")
    student = mobilenetv3_simam_small(num_classes=100, attention_type='parallel', input_size=32)
    trainer = KDTrainer(student, teacher, train_loader, test_loader, device,
                       temperature=4.0, alpha=0.9, lr=0.1)
    best_acc, history = trainer.train(epochs)

    # Baseline without KD
    print("\n>>> Training baseline on CIFAR-100...")
    set_seed(42)
    baseline = mobilenetv3_simam_small(num_classes=100, attention_type='parallel', input_size=32).to(device)
    optimizer = torch.optim.SGD(baseline.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_baseline = 0
    for epoch in tqdm(range(1, epochs + 1), desc='Baseline'):
        baseline.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(baseline(inputs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        val_acc = compute_accuracy(baseline, test_loader, device)['top1_acc']
        if val_acc > best_baseline:
            best_baseline = val_acc

    results = {
        'teacher_accuracy': best_teacher_acc,
        'baseline_accuracy': best_baseline,
        'kd_accuracy': best_acc,
        'improvement': best_acc - best_baseline,
    }

    print(f"\n    CIFAR-100 Results:")
    print(f"    Baseline: {best_baseline:.2f}%")
    print(f"    With KD: {best_acc:.2f}% (+{best_acc - best_baseline:.2f}%)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-cifar100', action='store_true', help='Skip CIFAR-100 experiments')
    parser.add_argument('--epochs', type=int, default=60, help='Epochs for main experiments')
    parser.add_argument('--ablation-epochs', type=int, default=40, help='Epochs for ablation studies')
    args = parser.parse_args()

    print("=" * 60)
    print("COMPLETE EXPERIMENT SUITE FOR SIMAM-KD PAPER")
    print("Author: Raju Kumar Yadav")
    print("=" * 60)

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"./results/complete_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results: {results_dir}")

    # Load CIFAR-10
    print("\nLoading CIFAR-10...")
    train_loader, test_loader, _, _ = get_data_loaders(
        dataset='cifar10', batch_size=128, num_workers=4, augmentation='standard'
    )

    # Load teacher
    print("Loading teacher...")
    teacher = get_teacher_model('resnet50', num_classes=10, pretrained=False, input_size=32)
    ckpt = torch.load('./results/cifar10_20260106_082628/teacher_resnet50.pth', map_location=device)
    teacher.load_state_dict(ckpt['state_dict'])
    teacher = teacher.to(device)
    teacher.eval()
    teacher_acc = compute_accuracy(teacher, test_loader, device)['top1_acc']
    print(f"Teacher accuracy: {teacher_acc:.2f}%")

    all_results = {'teacher_accuracy': teacher_acc}

    # Experiment 1: Attention comparison
    start = time.time()
    all_results['attention_comparison'] = run_attention_comparison(
        train_loader, test_loader, teacher, device, results_dir, args.epochs
    )
    print(f"Experiment 1 done in {(time.time()-start)/60:.1f} min")

    # Experiment 2: Temperature ablation
    start = time.time()
    all_results['temperature_ablation'] = run_temperature_ablation(
        train_loader, test_loader, teacher, device, results_dir, args.ablation_epochs
    )
    print(f"Experiment 2 done in {(time.time()-start)/60:.1f} min")

    # Experiment 3: Alpha ablation
    start = time.time()
    all_results['alpha_ablation'] = run_alpha_ablation(
        train_loader, test_loader, teacher, device, results_dir, args.ablation_epochs
    )
    print(f"Experiment 3 done in {(time.time()-start)/60:.1f} min")

    # Experiment 4: Pruning ablation
    start = time.time()
    all_results['pruning_ablation'] = run_pruning_ablation(
        train_loader, test_loader, device, results_dir,
        f"{results_dir}/attention_parallel.pth", epochs_ft=30
    )
    print(f"Experiment 4 done in {(time.time()-start)/60:.1f} min")

    # Experiment 5: CIFAR-100 (optional)
    if not args.skip_cifar100:
        start = time.time()
        all_results['cifar100'] = run_cifar100_experiment(device, results_dir, args.epochs)
        print(f"Experiment 5 done in {(time.time()-start)/60:.1f} min")

    # Save all results
    # Remove non-serializable items
    def clean_results(obj):
        if isinstance(obj, dict):
            return {k: clean_results(v) for k, v in obj.items() if k != 'history'}
        return obj

    save_results(clean_results(all_results), f"{results_dir}/all_results.json")

    # Print summary
    print("\n" + "=" * 60)
    print("COMPLETE RESULTS SUMMARY")
    print("=" * 60)

    print("\n1. Attention Comparison:")
    for att, res in all_results['attention_comparison'].items():
        print(f"   {att}: {res['accuracy']:.2f}%")

    print("\n2. Temperature Ablation:")
    for k, res in all_results['temperature_ablation'].items():
        print(f"   {k}: {res['accuracy']:.2f}%")

    print("\n3. Alpha Ablation:")
    for k, res in all_results['alpha_ablation'].items():
        print(f"   {k}: {res['accuracy']:.2f}%")

    print("\n4. Pruning Ablation:")
    for k, res in all_results['pruning_ablation'].items():
        if 'error' not in res:
            print(f"   {k}: {res['accuracy']:.2f}% ({res['param_reduction']:.1f}% reduction)")

    if 'cifar100' in all_results:
        print("\n5. CIFAR-100:")
        c100 = all_results['cifar100']
        print(f"   Baseline: {c100['baseline_accuracy']:.2f}%")
        print(f"   With KD: {c100['kd_accuracy']:.2f}% (+{c100['improvement']:.2f}%)")

    print(f"\nAll results saved to: {results_dir}")
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
