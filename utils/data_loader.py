"""
Data Loading Utilities
Author: Raju Kumar Yadav (itsmeerajuyadav@gmail.com)

This module provides data loading utilities for CIFAR-10, CIFAR-100,
and other benchmark datasets.
"""

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
import numpy as np


# CIFAR-10/100 normalization statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_cifar_transforms(
    train: bool = True,
    dataset: str = 'cifar10',
    augmentation: str = 'standard',
) -> transforms.Compose:
    """
    Get transforms for CIFAR datasets.

    Args:
        train: Whether this is for training or validation
        dataset: 'cifar10' or 'cifar100'
        augmentation: 'none', 'standard', or 'autoaugment'

    Returns:
        Transform composition
    """
    # Select normalization stats
    if dataset == 'cifar10':
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset == 'cifar100':
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if train:
        if augmentation == 'none':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif augmentation == 'standard':
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif augmentation == 'autoaugment':
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif augmentation == 'cutout':
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize(mean, std),
            ])
        else:
            raise ValueError(f"Unknown augmentation: {augmentation}")
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return transform


class Cutout:
    """
    Cutout data augmentation.

    Randomly masks out square regions of input during training.
    """

    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = './data',
    augmentation: str = 'standard',
    val_split: float = 0.0,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Get CIFAR-10 data loaders.

    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        data_dir: Directory to store/load data
        augmentation: Type of augmentation
        val_split: If > 0, split training set for validation

    Returns:
        train_loader, test_loader, (optional val_loader)
    """
    train_transform = get_cifar_transforms(True, 'cifar10', augmentation)
    test_transform = get_cifar_transforms(False, 'cifar10')

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    if val_split > 0:
        n_train = len(train_dataset)
        n_val = int(n_train * val_split)
        indices = list(range(n_train))
        np.random.shuffle(indices)

        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        val_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=test_transform
        )
        val_dataset = Subset(val_dataset, val_indices)
        train_dataset = Subset(train_dataset, train_indices)

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    else:
        val_loader = None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, val_loader


def get_cifar100_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = './data',
    augmentation: str = 'standard',
    val_split: float = 0.0,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Get CIFAR-100 data loaders.

    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        data_dir: Directory to store/load data
        augmentation: Type of augmentation
        val_split: If > 0, split training set for validation

    Returns:
        train_loader, test_loader, (optional val_loader)
    """
    train_transform = get_cifar_transforms(True, 'cifar100', augmentation)
    test_transform = get_cifar_transforms(False, 'cifar100')

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    if val_split > 0:
        n_train = len(train_dataset)
        n_val = int(n_train * val_split)
        indices = list(range(n_train))
        np.random.shuffle(indices)

        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        val_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=False, transform=test_transform
        )
        val_dataset = Subset(val_dataset, val_indices)
        train_dataset = Subset(train_dataset, train_indices)

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    else:
        val_loader = None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, val_loader


def get_data_loaders(
    dataset: str = 'cifar10',
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = './data',
    augmentation: str = 'standard',
    val_split: float = 0.0,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], int]:
    """
    Factory function to get data loaders.

    Args:
        dataset: 'cifar10' or 'cifar100'
        batch_size: Batch size
        num_workers: Number of workers
        data_dir: Data directory
        augmentation: Augmentation type
        val_split: Validation split ratio

    Returns:
        train_loader, test_loader, val_loader, num_classes
    """
    if dataset == 'cifar10':
        train_loader, test_loader, val_loader = get_cifar10_loaders(
            batch_size, num_workers, data_dir, augmentation, val_split
        )
        num_classes = 10
    elif dataset == 'cifar100':
        train_loader, test_loader, val_loader = get_cifar100_loaders(
            batch_size, num_workers, data_dir, augmentation, val_split
        )
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train_loader, test_loader, val_loader, num_classes


if __name__ == "__main__":
    # Test data loaders
    print("Testing data loaders...")

    for dataset in ['cifar10', 'cifar100']:
        train_loader, test_loader, _, num_classes = get_data_loaders(
            dataset=dataset, batch_size=128, data_dir='./data'
        )
        print(f"\n{dataset.upper()}:")
        print(f"  Num classes: {num_classes}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        # Get a batch
        images, labels = next(iter(train_loader))
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
