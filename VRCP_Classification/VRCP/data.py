"""Dataset loading and test-set truncation."""

import os

import torch
import torchvision
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms


def load_datasets(dataset: str, datasets_dir: str):
    """Load the train/test datasets for `dataset` (CIFAR10, CIFAR100, or TINYNET)."""
    if dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=datasets_dir, train=True, transform=torchvision.transforms.ToTensor(), download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=datasets_dir, train=False, transform=torchvision.transforms.ToTensor()
        )
    elif dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=datasets_dir, train=True, transform=torchvision.transforms.ToTensor(), download=True
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=datasets_dir, train=False, transform=torchvision.transforms.ToTensor()
        )
    elif dataset == "TINYNET":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = datasets.ImageFolder(root=os.path.join(datasets_dir, "tiny-imagenet-200", "train"), transform=transform)
        test_dataset = datasets.ImageFolder(root=os.path.join(datasets_dir, "tiny-imagenet-200", "val"), transform=transform)
    else:
        raise ValueError(f"No such dataset: {dataset!r}. Expected CIFAR10, CIFAR100, or TINYNET.")
    return train_dataset, test_dataset


def load_test_tensor(dataset: str, datasets_dir: str, n_test: int):
    """Load (and, if needed, randomly truncate to `n_test` points) the test
    set as a single dense tensor pair."""
    train_dataset, test_dataset = load_datasets(dataset, datasets_dir)

    if n_test < len(test_dataset):
        torch.manual_seed(0)
        test_dataset = random_split(test_dataset, [n_test, len(test_dataset) - n_test])[0]
    n_test = len(test_dataset)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=n_test, shuffle=False)
    _, (all_x_test, all_y_test) = next(enumerate(test_loader))

    return train_dataset, all_x_test, all_y_test
