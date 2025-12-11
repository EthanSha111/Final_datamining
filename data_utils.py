"""
data_utils.py
--------------

Dataset loading and preprocessing utilities for MNIST and CIFAR-10, plus
basic reproducibility helpers.
"""

from __future__ import annotations

from typing import Tuple

import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_mnist_loaders(
    batch_size: int,
    num_workers: int = 2,
    data_root: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders for MNIST.

    Args:
        batch_size:  Batch size.
        num_workers: Number of worker processes for data loading.
        data_root:   Directory to store/fetch the dataset.

    Returns:
        train_loader, test_loader
    """
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_root, train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, transform=transform, download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_cifar10_loaders(
    batch_size: int,
    num_workers: int = 2,
    data_root: str = "./data",
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders for CIFAR-10.

    Args:
        batch_size:  Batch size.
        num_workers: Number of worker processes for data loading.
        data_root:   Directory to store/fetch the dataset.
        normalize:   Whether to apply standard CIFAR-10 normalization.

    Returns:
        train_loader, test_loader
    """
    transform_list = [transforms.ToTensor()]
    if normalize:
        # Standard CIFAR-10 normalization
        transform_list.append(
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            )
        )
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, transform=transform, download=True
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, transform=transform, download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


__all__ = ["set_seed", "get_mnist_loaders", "get_cifar10_loaders"]



