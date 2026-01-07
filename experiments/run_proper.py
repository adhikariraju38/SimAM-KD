"""
PROPER Research Experiment Runner
Author: Raju Kumar Yadav (itsmeerajuyadav@gmail.com)

This runs experiments with proper settings for publishable results.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import mobilenetv3_simam_small, get_teacher_model
from training import count_parameters, get_model_size_mb
from utils import get_data_loaders, compute_accuracy, save_results


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class ProperKDTrainer:
    """Proper KD trainer with correct settings"""

    def __init__(self, student, teacher, train_loader, val_loader, device,
                 temperature=4.0, alpha=0.9, lr=0.001):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.teacher.eval()

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.temperature = temperature
        self.alpha = alpha  # Higher alpha = more focus on teacher

        # Use SGD with momentum - better for KD
        self.optimizer = torch.optim.SGD(
            student.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True
        )
        self.scaler = GradScaler()
        self.best_acc = 0.0
        self.best_state = None

    def train_epoch(self, epoch, total_epochs):
        self.student.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)
                student_logits = self.student(inputs)

                # KD Loss with proper scaling
                T = self.temperature
                soft_targets = F.softmax(teacher_logits / T, dim=1)
                soft_loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    soft_targets,
                    reduction='batchmean'
                ) * (T * T)

                hard_loss = F.cross_entropy(student_logits, targets)

                # Label smoothing helps
                loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*correct/total:.1f}%'})

        return total_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def validate(self):
        self.student.eval()
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with autocast():
                outputs = self.student(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return 100. * correct / total

    def train(self, epochs, save_path=None):
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-5
        )

        history = {'train_acc': [], 'val_acc': [], 'lr': []}

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch, epochs)
            val_acc = self.validate()
            scheduler.step()

            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['lr'].append(self.optimizer.param_groups[0]['lr'])

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_state = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}
                if save_path:
                    torch.save({
                        'student_state_dict': self.student.state_dict(),
                        'best_acc': self.best_acc,
                        'epoch': epoch,
                    }, save_path)

            if epoch % 10 == 0 or epoch == epochs:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch}: Train={train_acc:.1f}%, Val={val_acc:.1f}%, Best={self.best_acc:.1f}%, LR={lr:.6f}")

        # Load best model
        if self.best_state:
            self.student.load_state_dict(self.best_state)

        return self.best_acc, history


def proper_finetune(model, train_loader, val_loader, device, epochs=30, lr=0.001):
    """Proper fine-tuning after pruning"""
    model = model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f'FT {epoch}', leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        train_acc = 100. * correct / total

        # Validate
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
        val_acc = 100. * correct / total

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            print(f"    Epoch {epoch}: Train={train_acc:.1f}%, Val={val_acc:.1f}%, Best={best_acc:.1f}%")

    if best_state:
        model.load_state_dict(best_state)

    return model, best_acc


def main():
    print("=" * 60)
    print("PROPER RESEARCH EXPERIMENTS")
    print("Settings optimized for publishable results")
    print("=" * 60)

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"./results/proper_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print("\nLoading CIFAR-10...")
    BATCH_SIZE = 128  # Standard batch size for better gradients
    train_loader, test_loader, _, _ = get_data_loaders(
        dataset='cifar10', batch_size=BATCH_SIZE, num_workers=4, augmentation='standard'
    )

    # Load saved teacher
    print("\n" + "=" * 50)
    print("Loading Teacher (91.29% accuracy)")
    print("=" * 50)

    teacher = get_teacher_model('resnet50', num_classes=10, pretrained=False, input_size=32)
    ckpt = torch.load('./results/cifar10_20260106_082628/teacher_resnet50.pth', map_location=device)
    teacher.load_state_dict(ckpt['state_dict'])
    teacher = teacher.to(device)
    teacher.eval()

    teacher_acc = compute_accuracy(teacher, test_loader, device)['top1_acc']
    print(f"Teacher accuracy verified: {teacher_acc:.2f}%")

    # Baseline results from previous run
    baseline = {'none': 83.53, 'simam': 83.41, 'ca': 83.71, 'parallel': 83.97}

    # ============================================================
    # PROPER KD Training - 80 epochs
    # ============================================================
    print("\n" + "=" * 50)
    print("PROPER Knowledge Distillation (80 epochs)")
    print("Temperature=4.0, Alpha=0.9 (focus on teacher)")
    print("=" * 50)

    KD_EPOCHS = 80
    kd_results = {}

    for att_type in ['parallel']:  # Focus on best one first
        print(f"\n>>> Training {att_type.upper()} with KD...")
        start_time = time.time()

        student = mobilenetv3_simam_small(
            num_classes=10, attention_type=att_type, input_size=32
        )

        trainer = ProperKDTrainer(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            val_loader=test_loader,
            device=device,
            temperature=4.0,
            alpha=0.9,  # High alpha = more teacher influence
            lr=0.1,     # Higher LR with SGD
        )

        best_acc, history = trainer.train(
            epochs=KD_EPOCHS,
            save_path=f"{results_dir}/kd_{att_type}.pth"
        )

        elapsed = time.time() - start_time
        improvement = best_acc - baseline[att_type]

        kd_results[att_type] = {
            'accuracy': best_acc,
            'baseline': baseline[att_type],
            'improvement': improvement,
            'time_minutes': elapsed / 60,
            'history': history,
        }

        print(f"\n  RESULT: {best_acc:.2f}% (Baseline: {baseline[att_type]:.2f}%, Improvement: {improvement:+.2f}%)")

    # ============================================================
    # PROPER Pruning with more fine-tuning
    # ============================================================
    print("\n" + "=" * 50)
    print("PROPER Pruning (20% ratio, 30 epochs fine-tune)")
    print("=" * 50)

    pruning_results = {}

    try:
        from training import StructuredPruner
        import copy

        # Load best KD model
        best_att = 'parallel'
        student = mobilenetv3_simam_small(num_classes=10, attention_type=best_att, input_size=32).to(device)
        ckpt = torch.load(f"{results_dir}/kd_{best_att}.pth", map_location=device)
        student.load_state_dict(ckpt['student_state_dict'])

        before_acc = compute_accuracy(student, test_loader, device)['top1_acc']
        before_params = count_parameters(student)
        print(f"\nBefore pruning: {before_acc:.2f}%, {before_params/1e6:.2f}M params")

        # Try conservative pruning ratios
        for prune_ratio in [0.15, 0.20]:
            print(f"\n--- Pruning {int(prune_ratio*100)}% ---")

            # Make a copy for each pruning ratio
            model_copy = mobilenetv3_simam_small(num_classes=10, attention_type=best_att, input_size=32).to(device)
            model_copy.load_state_dict(ckpt['student_state_dict'])

            example_inputs = torch.randn(1, 3, 32, 32).to(device)
            pruner = StructuredPruner(model_copy, example_inputs)
            pruned_model = pruner.prune(pruning_ratio=prune_ratio)
            pruned_model = pruned_model.to(device)

            after_params = count_parameters(pruned_model)
            param_reduction = (1 - after_params / before_params) * 100

            acc_before_ft = compute_accuracy(pruned_model, test_loader, device)['top1_acc']
            print(f"  After pruning (before FT): {acc_before_ft:.2f}%")
            print(f"  Params: {after_params/1e6:.2f}M ({param_reduction:.1f}% reduction)")

            # Proper fine-tuning
            print(f"  Fine-tuning for 30 epochs...")
            pruned_model, acc_after_ft = proper_finetune(
                pruned_model, train_loader, test_loader, device,
                epochs=30, lr=0.01
            )

            pruning_results[f'{int(prune_ratio*100)}%'] = {
                'before_acc': before_acc,
                'after_prune_acc': acc_before_ft,
                'after_finetune_acc': acc_after_ft,
                'params_before': before_params,
                'params_after': after_params,
                'param_reduction': param_reduction,
            }

            # Save pruned model
            torch.save(pruned_model.state_dict(), f"{results_dir}/pruned_{int(prune_ratio*100)}_{best_att}.pth")

            print(f"  FINAL: {acc_after_ft:.2f}% with {param_reduction:.1f}% fewer params")

    except Exception as e:
        print(f"Pruning error: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS FOR PAPER")
    print("=" * 60)

    print("\n1. Knowledge Distillation Results:")
    print("┌─────────────┬──────────┬──────────┬─────────────┐")
    print("│ Method      │ Baseline │ With KD  │ Improvement │")
    print("├─────────────┼──────────┼──────────┼─────────────┤")
    for att, res in kd_results.items():
        print(f"│ {att:11} │ {res['baseline']:6.2f}%  │ {res['accuracy']:6.2f}%  │   {res['improvement']:+5.2f}%   │")
    print("└─────────────┴──────────┴──────────┴─────────────┘")

    print("\n2. Pruning Results:")
    print("┌──────────┬───────────┬────────────┬──────────────┐")
    print("│ Ratio    │ Before FT │ After FT   │ Param Reduc. │")
    print("├──────────┼───────────┼────────────┼──────────────┤")
    for ratio, res in pruning_results.items():
        print(f"│ {ratio:8} │ {res['after_prune_acc']:7.2f}%  │ {res['after_finetune_acc']:8.2f}%  │ {res['param_reduction']:10.1f}%  │")
    print("└──────────┴───────────┴────────────┴──────────────┘")

    # Save all results
    all_results = {
        'baseline': baseline,
        'teacher_accuracy': teacher_acc,
        'kd_results': {k: {kk: vv for kk, vv in v.items() if kk != 'history'} for k, v in kd_results.items()},
        'pruning_results': pruning_results,
    }
    save_results(all_results, f"{results_dir}/final_results.json")

    print(f"\nResults saved to: {results_dir}")
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
