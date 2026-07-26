"""Reproduces the "varied epsilon" coverage/set-size plots (paper Figure 2b).

Reads the per-epsilon results CSVs written by run_experiment.py for a sweep
over --delta (see INSTRUCTIONS.md), using the same path-building logic as
the experiment driver (VRCP.paths.results_dir) so this script can't drift
out of sync with where results actually get written.
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


def load_results(dataset, epsilons, ratio, n_smooth, alpha, arc, results_root):
    noisy_results = pd.DataFrame()
    for eps in epsilons:
        directory = results_dir(
            results_root, dataset, eps, sigma_model=0, sigma_smooth=eps * ratio, n_smooth=n_smooth,
            robust=True, arc=arc, alpha=alpha, norm_is_inf=False,
        )
        path = os.path.join(directory, f"results10000{dataset}_{eps}.csv")
        results = pd.read_csv(path)
        results = results.drop(["Size list"], axis=1)
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

    sns.lineplot(data=noisy_results, x="Epsilon", y="Marginal Coverage", hue="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(results_root, "variedEpsilonCoverage.pdf"))
    plt.close()

    sns.lineplot(data=noisy_results, x="Epsilon", y="Average Set Size", hue="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(results_root, "variedEpsilon.pdf"))
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot coverage/set-size vs. epsilon (Figure 2b)")
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--epsilons", type=float, nargs="+",
                         default=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05])
    parser.add_argument("--ratio", type=float, default=2, help="Must match the -r/--ratio used to generate results")
    parser.add_argument("--n_smooth", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--arc", default="CNN")
    parser.add_argument("--results_root", default="./Results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = load_results(args.dataset, args.epsilons, args.ratio, args.n_smooth, args.alpha, args.arc, args.results_root)
    plot(results, args.results_root)
