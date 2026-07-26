"""Thin CLI entrypoint for a VRCP classification experiment run.

Orchestration only -- see VRCP/data.py, VRCP/models.py, VRCP/adversarial.py,
VRCP/bounds.py, VRCP/experiment.py, and VRCP/reporting.py for the actual
logic, and config.yaml / vrcp_cls_config for configuration.

Every CLI flag defaults to None and, when set, overrides the matching
config.yaml value -- this keeps parameter sweeps (see INSTRUCTIONS.md, e.g.
Figure 2a/2b) working without editing config.yaml between runs:

    uv run python run_experiment.py --n_s 512 --delta 0.03 --dataset CIFAR10
"""

import argparse
import random
import warnings

import numpy as np
import torch

from vrcp_cls_config import Config, CFG_EXPERIMENT, CFG_PATH, CFG_RAND
from VRCP.adversarial import generate_or_load_adversarial_examples
from VRCP.bounds import compute_or_load_bounds
from VRCP.data import load_test_tensor
from VRCP.experiment import run_experiments
from VRCP.models import load_pretrained
from VRCP.paths import results_dir
from VRCP.reporting import save_and_plot_results
from VRCP.utils import calculate_accuracy

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="VRCP classification experiment")
    parser.add_argument("-a", "--alpha", type=float, default=None, help="Desired nominal marginal coverage")
    parser.add_argument("-d", "--delta", type=float, default=None, help="L_p bound on the adversarial noise")
    parser.add_argument("-s", "--splits", type=int, default=None, help="Number of experiments to estimate coverage")
    parser.add_argument("-r", "--ratio", type=float, default=None, help="Ratio between adversarial noise bound and smoothing noise")
    parser.add_argument("-n", "--norm", type=str, default=None, help="l_p norm used for epsilon ball: 2, inf")
    parser.add_argument("--n_s", type=int, default=None, help="Number of samples used for estimating smoothed score")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to be used: CIFAR100, CIFAR10, TINYNET")
    parser.add_argument("--arc", type=str, default=None, help="Architecture of classifier: CNN")
    parser.add_argument("--My_model", dest="my_model", action="store_true", default=None,
                         help="True for our trained model, False for Cohen's. Relevant only for CIFAR10")
    parser.add_argument("--batch_size", type=int, default=None, help="Number of images to send to GPU at once")
    parser.add_argument("--n_test", type=int, default=None, help="Number of test points")
    return parser.parse_args()


def main():
    args = parse_args()
    Config.load(cli_overrides=vars(args))
    experiment = Config.DATA[CFG_EXPERIMENT]
    path = Config.DATA[CFG_PATH]

    seed = Config.DATA[CFG_RAND]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = experiment["dataset"]
    if not experiment["my_model"]:
        raise NotImplementedError(
            "--My_model=False (Cohen et al.'s pretrained models) is not implemented; "
            "only My_model=True is supported."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    print("==> Preparing data..")
    _, all_x_test, all_y_test = load_test_tensor(dataset, path["datasets"], experiment["n_test"])
    n_test = all_x_test.shape[0]
    experiment["n_test"] = n_test
    experiment["save_results"] = experiment["splits"] == 50 and n_test == 10000
    num_of_classes = experiment["num_of_classes"]

    model = load_pretrained(dataset, experiment["arc"], path["checkpoints"], device=device)
    model.eval()

    acc, _, _ = calculate_accuracy(model, all_x_test, all_y_test, num_of_classes, k=1, device=device,
                                    GPU_CAPACITY=experiment["batch_size"])
    print(f"True Model accuracy: {acc * 100}%")

    all_x_test_adv, all_x_test_adv_base = generate_or_load_adversarial_examples(
        model, all_x_test, all_y_test, dataset, experiment["delta"], experiment["robust"], experiment["norm"],
        path["adversarial_examples"], device, experiment["batch_size"], n_test,
    )

    acc, _, _ = calculate_accuracy(model, all_x_test_adv_base, all_y_test, num_of_classes, k=1, device=device,
                                    GPU_CAPACITY=experiment["batch_size"])
    print(f"True Model accuracy on adversarial examples: {acc * 100}%")

    lb_tensor, lb_tensor_adv, ub_tensor, ub_tensor_adv = compute_or_load_bounds(
        model, all_x_test, all_x_test_adv_base, dataset, experiment["delta"], experiment["norm"], num_of_classes,
        path["bounds"], device, experiment["batch_size"], n_test,
    )

    results, quantiles = run_experiments(
        model, all_x_test, all_y_test, all_x_test_adv, all_x_test_adv_base,
        lb_tensor, lb_tensor_adv, ub_tensor, ub_tensor_adv,
        experiment["alpha"], experiment["delta"], experiment["splits"], experiment["n_s"],
        experiment["sigma_model"], experiment["sigma_smooth"], experiment["correction"],
        num_of_classes, n_test, device, experiment["batch_size"],
    )

    if experiment["save_results"]:
        output_dir = results_dir(
            path["results"], dataset, experiment["delta"], experiment["sigma_model"], experiment["sigma_smooth"],
            experiment["n_s"], experiment["robust"], experiment["arc"], experiment["alpha"],
            norm_is_inf=(experiment["norm"] == "inf"),
        )
    else:
        output_dir = path["results"]

    save_and_plot_results(
        results, quantiles, num_of_classes, experiment["alpha"], experiment["delta"], n_test, dataset,
        output_dir, experiment["save_results"], experiment["splits"],
    )


if __name__ == "__main__":
    main()
