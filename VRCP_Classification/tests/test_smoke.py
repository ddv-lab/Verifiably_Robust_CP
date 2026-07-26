"""Fast, CPU-only smoke tests for the VRCP_Classification stack.

Catches import errors, auto_LiRPA API breaks, and gross numeric
regressions in the torch/auto_LiRPA/numpy stack. No dataset download,
checkpoint, or GPU is required; everything here uses small random tensors.
"""

import torch
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

import numpy as np

from VRCP import score_functions as scores
from VRCP.models import build_model
from VRCP.utils import calibration, evaluate_predictions, pgd_attack, prediction

DEVICE = "cpu"


def _compute_bounds(model, x, eps, norm=2):
    model.eval()
    dummy = torch.zeros_like(x)
    lirpa_model = BoundedModule(model, dummy, device=DEVICE)
    lirpa_model.eval()
    ptb = PerturbationLpNorm(norm=norm, eps=eps)
    my_input = BoundedTensor(x, ptb)
    with torch.no_grad():
        lb, ub = lirpa_model.compute_bounds(x=(my_input,), method="CROWN")
    return lirpa_model, lb, ub


# --- PGD attack generation ---------------------------------------------------

def test_pgd_l2_attack_shape_and_range():
    torch.manual_seed(0)
    model = build_model("CIFAR10").eval()
    x = torch.rand(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    x_adv = pgd_attack(model, x, y, norm="2", N_steps=2, max_norm=0.05, device=DEVICE, GPU_CAPACITY=4)
    assert x_adv.shape == x.shape
    assert x_adv.min() >= -1e-4 and x_adv.max() <= 1 + 1e-4


def test_pgd_linf_attack_shape_and_range():
    torch.manual_seed(0)
    model = build_model("CIFAR10").eval()
    x = torch.rand(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    x_adv = pgd_attack(model, x, y, norm="inf", N_steps=2, max_norm=0.02, device=DEVICE, GPU_CAPACITY=4)
    assert x_adv.shape == x.shape
    assert x_adv.min() >= -1e-4 and x_adv.max() <= 1 + 1e-4


# --- CROWN bound computation -------------------------------------------------

def test_crown_bounds_cifar100_pooling_architecture():
    torch.manual_seed(0)
    model = build_model("CIFAR100")
    x = torch.rand(2, 3, 32, 32)
    _, lb, ub = _compute_bounds(model, x, eps=0.01)
    assert torch.isfinite(lb).all() and torch.isfinite(ub).all()
    assert (lb <= ub + 1e-5).all()


def test_crown_bounds_tinyimagenet_architecture():
    torch.manual_seed(0)
    model = build_model("TINYNET")
    x = torch.rand(2, 3, 64, 64)
    _, lb, ub = _compute_bounds(model, x, eps=0.01)
    assert torch.isfinite(lb).all() and torch.isfinite(ub).all()
    assert (lb <= ub + 1e-5).all()


def test_crown_bounds_contain_clean_prediction():
    torch.manual_seed(0)
    model = build_model("CIFAR100").eval()
    x = torch.rand(2, 3, 32, 32)
    lirpa_model, lb, ub = _compute_bounds(model, x, eps=0.001)
    with torch.no_grad():
        clean_out = lirpa_model(x)
    assert (lb <= clean_out + 1e-4).all()
    assert (clean_out <= ub + 1e-4).all()


# --- Conformal scoring / calibration / prediction ---------------------------

def test_conformal_calibration_and_prediction_roundtrip():
    rng = np.random.default_rng(0)
    n_calib, n_test, num_classes = 200, 50, 10

    probs_calib = rng.dirichlet(np.ones(num_classes), size=n_calib)
    labels_calib = rng.integers(0, num_classes, size=n_calib)
    scores_calib = scores.class_probability_score(probs_calib, labels_calib)

    thresholds, _ = calibration(scores_simple=scores_calib, alpha=0.1, base=True)

    probs_test = rng.dirichlet(np.ones(num_classes), size=n_test)
    labels_test = rng.integers(0, num_classes, size=n_test)
    scores_test_all = np.stack(
        [scores.class_probability_score(probs_test, np.arange(num_classes), all_combinations=True)]
    )

    pred_sets = prediction(
        scores_simple=scores_test_all, num_of_scores=1, thresholds=thresholds.reshape(1, -1), base=True
    )
    result = evaluate_predictions(pred_sets[0], None, labels_test, num_of_classes=num_classes)

    coverage = result["Coverage"].iloc[0]
    avg_size = result["Size"].iloc[0]
    assert 0.0 <= coverage <= 1.0
    assert avg_size >= 0
