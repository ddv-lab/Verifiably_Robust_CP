"""Model architectures and checkpoint loading, centralized in one place.

This is the canonical, checkpoint-compatible definition of the
CIFAR10/CIFAR100/TinyImageNet architectures -- both train.py and
run_experiment.py import `build_model` from here so they can't diverge.
"""

import os

import torch
import torch.nn as nn
from auto_LiRPA.utils import Flatten

_CHECKPOINT_FILENAMES = {
    "CIFAR10": "cifar10CNN.pth",
    "CIFAR100": "cifar100CNN.pth",
    "TINYNET": "tinynetCNN.pth",
}


class ConvNet(nn.Module):
    """TinyImageNet architecture. No pooling layers (strided convs only)."""

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(1024, 200),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


def _build_cifar10_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Dropout(0.4),
        Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def _build_cifar100_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Dropout(0.4),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Dropout(0.4),
        Flatten(),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 100),
    )


def build_model(dataset: str, arc: str = "CNN") -> nn.Module:
    """Build an (untrained) model for the given dataset. `arc` is currently
    always "CNN" -- kept as a parameter since config/CLI still expose it."""
    if arc != "CNN":
        raise ValueError(f"Unsupported architecture: {arc!r}. Only 'CNN' is supported.")
    if dataset == "CIFAR10":
        return _build_cifar10_cnn()
    if dataset == "CIFAR100":
        return _build_cifar100_cnn()
    if dataset == "TINYNET":
        return ConvNet()
    raise ValueError(f"No such dataset: {dataset!r}. Expected CIFAR10, CIFAR100, or TINYNET.")


def checkpoint_path(dataset: str, checkpoints_dir: str) -> str:
    return os.path.join(checkpoints_dir, _CHECKPOINT_FILENAMES[dataset])


def load_pretrained(dataset: str, arc: str, checkpoints_dir: str, device="cpu") -> nn.Module:
    """Build the model and load its shipped checkpoint.

    The CIFAR10/CIFAR100 checkpoints are dicts with a 'net'/'acc'/'epoch'
    structure; the TinyImageNet checkpoint is a raw state_dict. Detect the
    format from the loaded object rather than hardcoding it per dataset.
    """
    model = build_model(dataset, arc)
    state = torch.load(checkpoint_path(dataset, checkpoints_dir), map_location=device, weights_only=False)
    state_dict = state["net"] if isinstance(state, dict) and "net" in state else state
    model.load_state_dict(state_dict)
    return model.to(device)
