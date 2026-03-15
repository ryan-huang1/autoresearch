"""
One-time data preparation for autoresearch CIFAR-10 experiments.

Downloads CIFAR-10, creates a fixed train/validation split, and exposes
runtime utilities for dataloading and evaluation.

Usage:
    uv run prepare.py

Data and split metadata are stored in ~/.cache/autoresearch/.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300  # training time budget in seconds (5 minutes)
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
TRAIN_SIZE = 45_000
VAL_SIZE = 5_000
SPLIT_SEED = 1337

# CIFAR-10 normalization constants.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "cifar10")
SPLIT_DIR = os.path.join(CACHE_DIR, "splits")
SPLIT_PATH = os.path.join(SPLIT_DIR, "cifar10_train_val_split.pt")
DATASET_DIR = os.path.join(DATA_DIR, "cifar-10-batches-py")

CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def is_mps_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_default_device():
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_device(device=None):
    if device is None:
        return get_default_device()
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def get_device_name(device=None):
    return _resolve_device(device).type


def _supports_non_blocking(device):
    return _resolve_device(device).type == "cuda"


def synchronize_device(device=None):
    device = _resolve_device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def reset_peak_memory_stats(device=None):
    device = _resolve_device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_mb(device=None):
    device = _resolve_device(device)
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    if device.type == "mps" and hasattr(torch, "mps"):
        try:
            if hasattr(torch.mps, "driver_allocated_memory"):
                return torch.mps.driver_allocated_memory() / 1024 / 1024
            if hasattr(torch.mps, "current_allocated_memory"):
                return torch.mps.current_allocated_memory() / 1024 / 1024
        except RuntimeError:
            return 0.0
    return 0.0


def _resolve_num_workers():
    if sys.platform == "darwin":
        # macOS uses spawn semantics, so a conservative default avoids worker
        # process issues during local experimentation.
        return 0
    cpu_count = os.cpu_count() or 1
    return min(8, max(1, cpu_count // 2))


def _train_transform():
    return transforms.Compose([
        transforms.RandomCrop(IMAGE_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# ---------------------------------------------------------------------------
# Data download / split creation
# ---------------------------------------------------------------------------

def download_data():
    """Download CIFAR-10 train and test sets into the local cache."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATASET_DIR):
        print(f"Data: CIFAR-10 already present at {DATASET_DIR}")
    else:
        print("Data: downloading CIFAR-10...")
        datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
        datasets.CIFAR10(root=DATA_DIR, train=False, download=True)
        print(f"Data: download complete at {DATASET_DIR}")
        return

    # Make sure both train and test files exist even if the folder is already there.
    datasets.CIFAR10(root=DATA_DIR, train=True, download=False)
    datasets.CIFAR10(root=DATA_DIR, train=False, download=False)


def create_split():
    """Create and save a deterministic train/validation split."""
    if not os.path.exists(DATASET_DIR):
        raise RuntimeError("CIFAR-10 not found. Run `uv run prepare.py` first.")

    if os.path.exists(SPLIT_PATH):
        split = torch.load(SPLIT_PATH, map_location="cpu")
        train_indices = split["train_indices"]
        val_indices = split["val_indices"]
        print(
            f"Split: using existing fixed split at {SPLIT_PATH} "
            f"({len(train_indices)} train / {len(val_indices)} val)"
        )
        return

    os.makedirs(SPLIT_DIR, exist_ok=True)
    base_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=False)
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    permutation = torch.randperm(len(base_dataset), generator=generator)
    train_indices = permutation[:TRAIN_SIZE]
    val_indices = permutation[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    split = {
        "seed": SPLIT_SEED,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    torch.save(split, SPLIT_PATH)
    print(
        f"Split: saved fixed split to {SPLIT_PATH} "
        f"({len(train_indices)} train / {len(val_indices)} val)"
    )


def _load_split_indices():
    if not os.path.exists(DATASET_DIR):
        raise RuntimeError("CIFAR-10 not found. Run `uv run prepare.py` first.")
    if not os.path.exists(SPLIT_PATH):
        raise RuntimeError("Split metadata not found. Run `uv run prepare.py` first.")
    split = torch.load(SPLIT_PATH, map_location="cpu")
    return split["train_indices"], split["val_indices"]


def build_dataset(split, augment=False):
    """Return a dataset view for train/val/test."""
    assert split in {"train", "val", "test"}
    if split == "train":
        train_indices, _ = _load_split_indices()
        dataset = datasets.CIFAR10(
            root=DATA_DIR,
            train=True,
            download=False,
            transform=_train_transform() if augment else _eval_transform(),
        )
        return Subset(dataset, train_indices.tolist())

    if split == "val":
        _, val_indices = _load_split_indices()
        dataset = datasets.CIFAR10(
            root=DATA_DIR,
            train=True,
            download=False,
            transform=_eval_transform(),
        )
        return Subset(dataset, val_indices.tolist())

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=_eval_transform(),
    )
    return dataset


def _make_epoch_loader(split, batch_size, augment=False, device=None):
    dataset = build_dataset(split, augment=augment)
    num_workers = _resolve_num_workers()
    device = _resolve_device(device)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def make_dataloader(batch_size, split, augment=False, device=None):
    """
    Infinite dataloader yielding (images, labels, epoch).

    Images are moved to the requested device and normalized according to the
    fixed CIFAR-10 preprocessing defined above.
    """
    assert split in {"train", "val", "test"}
    device = _resolve_device(device)
    epoch = 1
    non_blocking = _supports_non_blocking(device)
    loader = _make_epoch_loader(split, batch_size, augment=augment and split == "train", device=device)

    while True:
        for images, labels in loader:
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            yield images, labels, epoch
        epoch += 1


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric harness)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_classifier(model, batch_size, split="val", device=None):
    """
    Evaluate a classifier on the requested split.

    Returns a dict with:
      - accuracy: top-1 accuracy in [0, 1]
      - loss: mean cross-entropy loss
      - num_examples: number of evaluated examples
    """
    assert split in {"val", "test"}
    device = _resolve_device(device)
    non_blocking = _supports_non_blocking(device)
    loader = _make_epoch_loader(split, batch_size, augment=False, device=device)

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.numel()

    if was_training:
        model.train()

    return {
        "accuracy": total_correct / total_examples,
        "loss": total_loss / total_examples,
        "num_examples": total_examples,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 data for autoresearch")
    _ = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    download_data()
    print()

    create_split()
    print()

    print("CIFAR-10 harness ready.")
    print(f"  image_size:  {IMAGE_SIZE}")
    print(f"  num_classes: {NUM_CLASSES}")
    print(f"  train_size:  {TRAIN_SIZE}")
    print(f"  val_size:    {VAL_SIZE}")
