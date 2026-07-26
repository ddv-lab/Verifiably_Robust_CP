"""CROWN bound computation via auto_LiRPA, with on-disk caching.

Note: the "adversarial" bound below is computed around the *clean* input's
epsilon-ball, not the adversarial input. This is consistent with VRCP's
guarantee -- a CROWN bound valid over the clean point's epsilon-ball also
holds for any f(x+delta) with ||delta||<=epsilon, including the specific
adversarial point -- and is supported empirically: an end-to-end run
(CIFAR10, n_test=2000, delta=0.02) shows *_verif/*_verif_pre coverage
identical between clean and adversarial rows, while *_vanilla drops under
attack as expected.
"""

import os
import pickle

import numpy as np
import torch
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from tqdm import tqdm

from VRCP.paths import bounds_dir

_BOUND_CONFIG_BY_NUM_CLASSES = {
    10: {"batch_divisor": 384, "method": "CROWN-Optimized"},
    100: {"batch_size": 2, "method": "CROWN-Optimized"},
    200: {"batch_size": 2, "method": "CROWN"},
}


def compute_or_load_bounds(
    model, all_x_test, all_x_test_adv_base, dataset, epsilon, norm, num_of_classes, root, device, gpu_capacity, n_test
):
    """Return (lb_tensor, lb_tensor_adv, ub_tensor, ub_tensor_adv), each of
    shape (n_test, num_of_classes), generating and caching them under
    Bounds/... if not already cached (full-dataset runs only)."""
    norm_is_inf = norm == "inf"
    directory = bounds_dir(root, dataset, epsilon, norm_is_inf)
    lp_norm = np.inf if norm_is_inf else 2

    if n_test == 10000 and os.path.exists(directory):
        with open(os.path.join(directory, "data.pickle"), "rb") as f:
            return pickle.load(f)

    channels, rows, cols = all_x_test.size()[1], all_x_test.size()[2], all_x_test.size()[3]
    model.to(device)
    model.eval()

    cfg = _BOUND_CONFIG_BY_NUM_CLASSES.get(num_of_classes)
    if cfg is None:
        raise ValueError(f"No bound-computation config for num_of_classes={num_of_classes}.")
    batch_size = cfg.get("batch_size") or (gpu_capacity // cfg["batch_divisor"])
    bound_method = cfg["method"]

    num_of_batches = (n_test // batch_size) + 1 if n_test % batch_size != 0 else (n_test // batch_size)
    all_indices = torch.arange(n_test)

    dummy = torch.zeros_like(all_x_test[:batch_size].view(batch_size, channels, rows, cols)).to(device)
    lirpa_model = BoundedModule(model, dummy)
    lirpa_model.to(device)
    lirpa_model.eval()
    ptb = PerturbationLpNorm(norm=lp_norm, eps=epsilon)

    ub_tensor = torch.zeros(n_test, num_of_classes).to(device)
    ub_tensor_adv = torch.zeros(n_test, num_of_classes).to(device)
    lb_tensor = torch.zeros(n_test, num_of_classes).to(device)
    lb_tensor_adv = torch.zeros(n_test, num_of_classes).to(device)

    for i in tqdm(range(num_of_batches)):
        idxs = all_indices[(i * batch_size):((i + 1) * batch_size)]
        input_tensor = all_x_test[idxs].view(-1, channels, rows, cols).to(device)
        curr_batch_size = input_tensor.size()[0]

        my_input = BoundedTensor(input_tensor, ptb)
        if bound_method == "CROWN":
            with torch.no_grad():
                lb, ub = lirpa_model.compute_bounds(x=(my_input,), method=bound_method)
        else:
            lb, ub = lirpa_model.compute_bounds(x=(my_input,), method=bound_method)
        lb, ub = lb.to("cpu"), ub.to("cpu")

        sigmalb, sigmaub = _bounds_to_softmax_interval(lb, ub, curr_batch_size, num_of_classes)
        lb_tensor[(i * batch_size):(i * batch_size) + curr_batch_size] = torch.from_numpy(sigmalb)
        ub_tensor[(i * batch_size):(i * batch_size) + curr_batch_size] = torch.from_numpy(sigmaub)

        # Reuses `my_input` intentionally -- see module docstring.
        if bound_method == "CROWN":
            with torch.no_grad():
                lb_adv, ub_adv = lirpa_model.compute_bounds(x=(my_input,), method=bound_method)
        else:
            lb_adv, ub_adv = lirpa_model.compute_bounds(x=(my_input,), method=bound_method)
        lb_adv, ub_adv = lb_adv.to("cpu"), ub_adv.to("cpu")

        sigmalb_adv, sigmaub_adv = _bounds_to_softmax_interval(lb_adv, ub_adv, curr_batch_size, num_of_classes)
        lb_tensor_adv[(i * batch_size):(i * batch_size) + curr_batch_size] = torch.from_numpy(sigmalb_adv)
        ub_tensor_adv[(i * batch_size):(i * batch_size) + curr_batch_size] = torch.from_numpy(sigmaub_adv)

    if n_test == 10000:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "data.pickle"), "wb") as f:
            pickle.dump([lb_tensor, lb_tensor_adv, ub_tensor, ub_tensor_adv], f)

    return lb_tensor, lb_tensor_adv, ub_tensor, ub_tensor_adv


def _bounds_to_softmax_interval(lb, ub, curr_batch_size, num_of_classes):
    """Convert per-logit lower/upper bounds into worst-case bounds on the
    softmax probability of each class."""
    sigmalb = np.zeros((curr_batch_size, num_of_classes))
    sigmaub = np.zeros((curr_batch_size, num_of_classes))
    for k in range(curr_batch_size):
        for j in range(num_of_classes):
            temp = np.append(lb.numpy().copy(), ub[k][j].numpy())
            maxnum = max(temp)
            sigmalb[k][j] = np.exp(lb[k][j]) / (np.exp(lb[k][j]) + torch.sum(np.exp(ub[k])) - np.exp(ub[k][j]))
            sigmaub[k][j] = np.exp(ub[k][j] - maxnum) / (
                np.exp(ub[k][j] - maxnum) + torch.sum(np.exp(lb[k] - maxnum)) - np.exp(lb[k][j] - maxnum)
            )
    return sigmalb, sigmaub
