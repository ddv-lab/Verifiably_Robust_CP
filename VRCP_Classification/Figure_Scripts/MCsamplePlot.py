"""Reproduces the "varied number of MC/smoothing samples" plots (Figure 2a).

Reads the per-n_smooth results CSVs written by run_experiment.py for a
sweep over --n_s (see INSTRUCTIONS.md), using the same path-building logic
as the experiment driver (VRCP.paths.results_dir).
"""

import argparse
import os
import sys

import matplotlib.font_manager  # noqa: F401  (registers fonts used below)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from VRCP.paths import results_dir

METHODS = ["HPS_verif", "HPS_verif_pre", "HPS_RSCP+", "PTT_HPS_RSCP+"]
METHOD_LABELS = ["VRCP_I", "VRCP_C", "RSCP+", "RSCP+ (PTT)"]


def load_results(dataset, epsilon, ratio, n_smooth_values, alpha, arc, results_root):
    noisy_results = pd.DataFrame()
    for n_smooth in n_smooth_values:
        directory = results_dir(
            results_root, dataset, epsilon, sigma_model=0, sigma_smooth=epsilon * ratio, n_smooth=n_smooth,
            robust=True, arc=arc, alpha=alpha, norm_is_inf=False,
        )
        path = os.path.join(directory, f"results10000{dataset}_{epsilon}.csv")
        results = pd.read_csv(path)
        results = results.drop(["Size list"], axis=1)
        results["Number of MC samples"] = str(n_smooth)
        results = results[(results["Method"].isin(METHODS)) & (results["noise_L2_norm"] != 0.0)]
        print(results.head())
        noisy_results = pd.concat([noisy_results, results])

    noisy_results = noisy_results.replace(METHODS, METHOD_LABELS)
    noisy_results.rename(
        columns={"noise_L2_norm": "Epsilon", "Size": "Average Set Size", "Coverage": "Marginal Coverage"},
        inplace=True,
    )
    return noisy_results


def plot(noisy_results, results_root):
    sns.set_theme(style="darkgrid")
    plt.style.use("ggplot")

    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"

    sns.lineplot(data=noisy_results, x="Number of MC samples", y="Marginal Coverage", hue="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(results_root, "variedMCsCoverage.pdf"))
    plt.close()

    sns.lineplot(data=noisy_results, x="Number of MC samples", y="Average Set Size", hue="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(results_root, "variedMCs.pdf"))
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot coverage/set-size vs. number of MC samples (Figure 2a)")
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--ratio", type=float, default=2, help="Must match the -r/--ratio used to generate results")
    parser.add_argument("--n_smooth_values", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--arc", default="CNN")
    parser.add_argument("--results_root", default="./Results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = load_results(args.dataset, args.epsilon, args.ratio, args.n_smooth_values, args.alpha, args.arc, args.results_root)
    plot(results, args.results_root)
