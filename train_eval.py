"""
train_eval.py
--------------

Training, evaluation, timing utilities, and a simple experiment runner
for comparing positional encoding variants on MNIST and CIFAR-10.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_utils import get_cifar10_loaders, get_mnist_loaders
from positional_encodings import (
    RoPEPositionalModule,
    CayleyStringPE,
    ReflectionStringPE,
    SparseCayleyStringPE,
)
from vit_models import VisionTransformer


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Returns:
        avg_loss, accuracy
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on a validation/test set.

    Returns:
        avg_loss, accuracy
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def measure_epoch_time(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Measure the wall-clock time (in seconds) for training one epoch.
    """
    start = time.time()
    train_one_epoch(model, loader, optimizer, device)
    end = time.time()
    return end - start


def measure_inference_time(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 10,
) -> float:
    """
    Measure average inference time per batch (in milliseconds) over a subset
    of the loader.
    """
    model.eval()
    times: list[float] = []

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            images = images.to(device)

            start = time.time()
            _ = model(images)
            end = time.time()

            times.append(end - start)

    if not times:
        return 0.0
    avg_ms = (sum(times) / len(times)) * 1000.0
    return avg_ms


@dataclass
class ExperimentConfig:
    dataset: str  # "mnist" or "cifar10"
    pos_variant: str  # e.g. "rope", "cayley_dense", "reflection", "cayley_sparse"

    img_size: int
    patch_size: int
    in_chans: int
    num_classes: int

    emb_dim: int
    depth: int
    n_heads: int

    batch_size: int
    epochs: int
    lr: float
    weight_decay: float = 1e-2

    f_sparse: float | None = None  # For sparse variants


def _build_pos_module(config: ExperimentConfig) -> nn.Module:
    """
    Instantiate the positional encoding module based on the variant string.
    """
    d_model = config.emb_dim
    variant = config.pos_variant.lower()

    if variant == "rope":
        return RoPEPositionalModule(d_model=d_model)
    if variant == "cayley_dense":
        return CayleyStringPE(d_model=d_model)
    if variant == "reflection":
        return ReflectionStringPE(d_model=d_model)
    if variant.startswith("cayley_sparse") or variant == "cayley_sparse":
        if config.f_sparse is None:
            raise ValueError("f_sparse must be provided for sparse Cayley-STRING variant.")
        return SparseCayleyStringPE(d_model=d_model, f=config.f_sparse)

    raise ValueError(f"Unknown positional variant: {config.pos_variant!r}")


def run_experiment(config: ExperimentConfig, device: torch.device | None = None) -> Dict:
    """
    Run a full training + evaluation experiment for a given configuration.

    Returns:
        results dict with config and metrics (loss, accuracy, timing).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    if config.dataset.lower() == "mnist":
        train_loader, test_loader = get_mnist_loaders(batch_size=config.batch_size)
    elif config.dataset.lower() == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(batch_size=config.batch_size)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset!r}")

    # Positional module
    pos_module = _build_pos_module(config)

    # Model
    model = VisionTransformer(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_chans=config.in_chans,
        num_classes=config.num_classes,
        emb_dim=config.emb_dim,
        depth=config.depth,
        n_heads=config.n_heads,
        pos_module=pos_module,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    epoch_times: list[float] = []
    last_train_loss = 0.0
    last_train_acc = 0.0
    last_val_loss = 0.0
    last_val_acc = 0.0

    for epoch in range(1, config.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_model(model, test_loader, device)
        end = time.time()

        epoch_time = end - start
        epoch_times.append(epoch_time)

        last_train_loss = train_loss
        last_train_acc = train_acc
        last_val_loss = val_loss
        last_val_acc = val_acc

        print(
            f"[Epoch {epoch}/{config.epochs}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
            f"time={epoch_time:.2f}s"
        )

    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    inference_time_ms = measure_inference_time(model, test_loader, device)

    results: Dict = {
        "config": asdict(config),
        "final_train_loss": last_train_loss,
        "final_train_acc": last_train_acc,
        "final_val_loss": last_val_loss,
        "final_val_acc": last_val_acc,
        "avg_epoch_time_sec": avg_epoch_time,
        "inference_time_ms_per_batch": inference_time_ms,
    }

    return results


__all__ = [
    "ExperimentConfig",
    "train_one_epoch",
    "eval_model",
    "measure_epoch_time",
    "measure_inference_time",
    "run_experiment",
]


