"""Adversarial example generation and on-disk caching."""

import os
import pickle

import torch.nn as nn
from torchvision import transforms

from VRCP.paths import adversarial_examples_dir
from VRCP.utils import NormalizeLayer, pgd_attack

N_STEPS = 100  # number of PGD gradient steps


def generate_or_load_adversarial_examples(
    model, all_x_test, all_y_test, dataset, epsilon, robust, norm, root, device, gpu_capacity, n_test
):
    """Return (all_x_test_adv, all_x_test_adv_base), generating and caching
    them under Adversarial_Examples/... if not already cached (only for
    full-dataset runs, i.e. n_test == 10000).

    TinyImageNet's test tensor is stored already-normalized to
    mean=std=0.5 (per its dataset transform), which is *not* the [0,1]
    range `pgd_attack` requires. For that dataset we invert the
    normalization, attack a temporary model with a NormalizeLayer prepended
    (so the model still sees normalized input internally), then
    re-normalize the resulting adversarial examples.
    """
    directory = adversarial_examples_dir(root, dataset, epsilon, robust, norm_is_inf=(norm == "inf"))

    if n_test == 10000 and os.path.exists(directory):
        with open(os.path.join(directory, "data.pickle"), "rb") as f:
            return pickle.load(f)

    print("Generate adversarial test examples for the smoothed model:\n")
    if dataset == "TINYNET":
        inv_trans = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
            transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.]),
        ])
        temp_x_test = inv_trans(all_x_test)
        normalize_layer = NormalizeLayer([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], device=device)
        temp_model = nn.Sequential(normalize_layer, model)
        all_x_test_adv = pgd_attack(
            temp_model, temp_x_test, all_y_test, norm, N_STEPS, epsilon / 2, device, GPU_CAPACITY=gpu_capacity
        )
        standard_trans = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        all_x_test_adv = standard_trans(all_x_test_adv)
    else:
        all_x_test_adv = pgd_attack(
            model, all_x_test, all_y_test, norm, N_STEPS, epsilon, device, GPU_CAPACITY=gpu_capacity
        )

    all_x_test_adv_base = all_x_test_adv

    if n_test == 10000:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "data.pickle"), "wb") as f:
            pickle.dump([all_x_test_adv, all_x_test_adv_base], f)

    return all_x_test_adv, all_x_test_adv_base
