"""
Knowledge Distillation Training Module
Authors: Raju Kumar Yadav, Rajesh Khanal, Safalta Kumari Yadav, Rikesh Kumar Shah, Bibek Kumar Gupta

Paper: SimAM-KD: Attention-Enhanced Knowledge Distillation for Efficient Image Classification
GitHub: https://github.com/adhikariraju38/SimAM-KD

This module implements knowledge distillation training for the SimAM-KD framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Tuple, Callable
import time
from tqdm import tqdm


class DistillationLoss(nn.Module):
    """
    Combined Knowledge Distillation Loss.

    L_total = alpha * L_KD + (1 - alpha) * L_CE

    Where:
    - L_KD: KL divergence between student and teacher soft labels
    - L_CE: Cross-entropy loss with hard labels
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        use_soft_labels: bool = True,
    ):
        """
        Args:
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss (1 - alpha for hard labels)
            use_soft_labels: Whether to use soft labels from teacher
        """
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.use_soft_labels = use_soft_labels
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.

        Args:
            student_logits: Student model outputs (B, num_classes)
            teacher_logits: Teacher model outputs (B, num_classes)
            labels: Ground truth labels (B,)

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Hard label loss (standard cross-entropy)
        hard_loss = self.ce_loss(student_logits, labels)

        if self.use_soft_labels and teacher_logits is not None:
            # Soft label loss (KL divergence)
            T = self.temperature
            soft_student = F.log_softmax(student_logits / T, dim=1)
            soft_teacher = F.softmax(teacher_logits / T, dim=1)

            soft_loss = F.kl_div(
                soft_student, soft_teacher, reduction='batchmean'
            ) * (T * T)

            # Combined loss
            total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

            loss_dict = {
                'total_loss': total_loss.item(),
                'hard_loss': hard_loss.item(),
                'soft_loss': soft_loss.item(),
            }
        else:
            total_loss = hard_loss
            loss_dict = {
                'total_loss': total_loss.item(),
                'hard_loss': hard_loss.item(),
                'soft_loss': 0.0,
            }

        return total_loss, loss_dict


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level Knowledge Distillation Loss.

    Matches intermediate feature representations between student and teacher.
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super(FeatureDistillationLoss, self).__init__()

        # Projection layer to match dimensions
        if student_dim != teacher_dim:
            self.projector = nn.Linear(student_dim, teacher_dim)
        else:
            self.projector = nn.Identity()

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute feature distillation loss.

        Args:
            student_features: Student intermediate features
            teacher_features: Teacher intermediate features

        Returns:
            MSE loss between projected student and teacher features
        """
        student_proj = self.projector(student_features)
        return F.mse_loss(student_proj, teacher_features)


class DistillationTrainer:
    """
    Trainer class for Knowledge Distillation.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        temperature: float = 4.0,
        alpha: float = 0.7,
        use_amp: bool = True,
        log_interval: int = 100,
    ):
        """
        Args:
            student: Student model to train
            teacher: Pre-trained teacher model (frozen)
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for student model
            scheduler: Learning rate scheduler
            device: Device to train on
            temperature: KD temperature
            alpha: Weight for soft labels
            use_amp: Use automatic mixed precision
            log_interval: Logging frequency
        """
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.teacher.eval()  # Teacher is always in eval mode

        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_interval = log_interval

        self.criterion = DistillationLoss(temperature, alpha)
        self.use_amp = use_amp and device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None

        self.best_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.student.train()

        total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    # Get teacher predictions
                    with torch.no_grad():
                        teacher_logits = self.teacher(inputs)

                    # Get student predictions
                    student_logits = self.student(inputs)

                    # Compute loss
                    loss, loss_dict = self.criterion(
                        student_logits, teacher_logits, targets
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Get teacher predictions
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)

                # Get student predictions
                student_logits = self.student(inputs)

                # Compute loss
                loss, loss_dict = self.criterion(
                    student_logits, teacher_logits, targets
                )

                loss.backward()
                self.optimizer.step()

            # Update metrics
            total_loss += loss_dict['total_loss']
            total_hard_loss += loss_dict['hard_loss']
            total_soft_loss += loss_dict['soft_loss']

            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            if (batch_idx + 1) % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'acc': f"{100. * correct / total:.2f}%"
                })

        # Epoch metrics
        n_batches = len(self.train_loader)
        metrics = {
            'train_loss': total_loss / n_batches,
            'train_hard_loss': total_hard_loss / n_batches,
            'train_soft_loss': total_soft_loss / n_batches,
            'train_acc': 100. * correct / total,
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.student.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(self.val_loader, desc='Validating'):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Get predictions
            student_logits = self.student(inputs)
            teacher_logits = self.teacher(inputs)

            # Compute loss
            loss, _ = self.criterion(student_logits, teacher_logits, targets)
            total_loss += loss.item()

            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_acc': 100. * correct / total,
        }

        return metrics

    def train(
        self,
        epochs: int,
        save_path: Optional[str] = None,
        early_stopping: int = 10,
    ) -> Dict:
        """
        Full training loop.

        Args:
            epochs: Number of epochs to train
            save_path: Path to save best model
            early_stopping: Patience for early stopping

        Returns:
            Training history
        """
        no_improve_count = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['train_acc'].append(train_metrics['train_acc'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_acc'].append(val_metrics['val_acc'])
            self.history['lr'].append(current_lr)

            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_acc']:.2f}%")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.2f}%")
            print(f"  LR: {current_lr:.6f}")

            # Save best model
            if val_metrics['val_acc'] > self.best_acc:
                self.best_acc = val_metrics['val_acc']
                no_improve_count = 0
                if save_path:
                    self.save_checkpoint(save_path, epoch, val_metrics)
                print(f"  New best accuracy: {self.best_acc:.2f}%")
            else:
                no_improve_count += 1

            # Early stopping
            if no_improve_count >= early_stopping:
                print(f"\nEarly stopping after {epoch} epochs")
                break

        print(f"\nTraining complete. Best accuracy: {self.best_acc:.2f}%")
        return self.history

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float],
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'metrics': metrics,
            'history': self.history,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"  Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_acc = checkpoint['best_acc']
        self.history = checkpoint['history']
        print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")


def train_teacher(
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.1,
    device: str = 'cuda',
    save_path: Optional[str] = None,
) -> Dict:
    """
    Train teacher model from scratch (for WideResNet or custom teachers).

    Args:
        teacher: Teacher model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        device: Device
        save_path: Path to save best model

    Returns:
        Training history
    """
    teacher = teacher.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        teacher.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        # Training
        teacher.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, targets in tqdm(train_loader, desc=f'Teacher Epoch {epoch}'):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = teacher(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total

        # Validation
        teacher.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = teacher(inputs)
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

        print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'state_dict': teacher.state_dict(),
                    'best_acc': best_acc,
                }, save_path)

    return history
