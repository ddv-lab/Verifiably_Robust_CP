"""CSV and plot output for a completed experiment run."""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from VRCP.experiment import CALIBRATION_SCORES, POST_CAL


def save_and_plot_results(results, quantiles, num_of_classes, alpha, epsilon, n_test, dataset,
                           output_dir, save_results, n_experiments):
    """Write result CSVs and coverage/set-size plots to `output_dir`
    (the caller is responsible for computing the right directory via
    VRCP.paths.results_dir when save_results is True, or the plain results
    root otherwise)."""
    os.makedirs(output_dir, exist_ok=True)
    add_string = f"{n_test}{dataset}_{epsilon}"

    print(f"Saving results in: {output_dir}")
    results.to_csv(os.path.join(output_dir, f"results{add_string}.csv"))
    if save_results:
        with open(os.path.join(output_dir, f"quantiles_bounds{add_string}.pickle"), "wb") as f:
            pickle.dump([quantiles], f)

    colors_list = sns.color_palette("husl", len(CALIBRATION_SCORES) * 4)
    ax = sns.catplot(x="Black box", y="Coverage", hue="Method", palette=colors_list, col="noise_L2_norm",
                      data=results, kind="box", height=4, aspect=.7)

    lower_quantiles_mean = np.zeros(len(CALIBRATION_SCORES))
    upper_quantiles_mean = np.zeros(len(CALIBRATION_SCORES))
    for p in range(len(CALIBRATION_SCORES)):
        lower_quantiles_mean[p] = np.mean(quantiles[p, 0, :])
        upper_quantiles_mean[p] = np.mean(quantiles[p, 1, :])

    for graph in ax.axes[0]:
        graph.set(xlabel="Classifier", ylabel="Marginal coverage")
        graph.axhline(1 - alpha, ls="--", color="red")
        for p in range(len(CALIBRATION_SCORES)):
            graph.axhline(upper_quantiles_mean[p], ls="--", color=colors_list[p * 4 + 2])
            graph.axhline(lower_quantiles_mean[p], ls="--", color=colors_list[p * 4 + 2])
    ax.savefig(os.path.join(output_dir, f"Marginal{add_string}.pdf"))

    ax = sns.catplot(x="Black box", y="Size", hue="Method", col="noise_L2_norm",
                      data=results, kind="box", height=4, aspect=.7)
    for graph in ax.axes[0]:
        graph.set(xlabel="Classifier", ylabel="Set Size")
    ax.savefig(os.path.join(output_dir, f"Size{add_string}.pdf"))

    methods_list = []
    for p in range(len(CALIBRATION_SCORES)):
        score_name = CALIBRATION_SCORES[p]
        methods_list.append(score_name + "_vanilla")
        if POST_CAL:
            methods_list.append(score_name + "_verif")
        methods_list.append(score_name + "_verif_pre")
        methods_list.append(score_name + "_RSCP+")

    sizes = results[["Method", "noise_L2_norm", "Size list"]]
    mean_set_sizes = (
        sizes.groupby(["Method", "noise_L2_norm"])["Size list"].mean().reset_index()
        .set_index(["Method", "noise_L2_norm"])["Size list"].to_dict()
    )

    fig, axes = plt.subplots(nrows=2, ncols=len(methods_list), figsize=(30, 10), sharey=True)
    x_values = np.arange(num_of_classes + 1)
    xlabels = np.arange(num_of_classes + 1, 10)
    for i, method in enumerate(sizes["Method"].unique()):
        sns.barplot(x=x_values, y=mean_set_sizes[(method, 0)], width=1, ax=axes[0, i], color=colors_list[i])
        axes[0, i].set_title(method)
        axes[0, i].set_xlabel("Set Size")
        axes[0, i].set_ylabel("Frequency")
        axes[0, i].set_xticks(x_values, labels=xlabels)
    plt.xlim(0, num_of_classes)

    for i, method in enumerate(sizes["Method"].unique()):
        sns.barplot(x=x_values, y=mean_set_sizes[(method, epsilon)], width=1, ax=axes[1, i], color=colors_list[i])
        axes[1, i].set_title(f"{method}+ eps={epsilon} attack")
        axes[1, i].set_xlabel("Set Size")
        axes[1, i].set_ylabel("Frequency")
        axes[1, i].set_xticks(x_values, labels=xlabels)

    plt.savefig(os.path.join(output_dir, f"SizeHist{add_string}.pdf"))
    plt.close(fig)

    if save_results:
        _save_summary_stats(results, epsilon, n_experiments, output_dir)


def _save_summary_stats(results, epsilon, n_experiments, output_dir):
    results = results.drop(["Size list"], axis=1)

    clean_results = results[results["noise_L2_norm"] == 0]
    clean_means = clean_results.groupby("Method").mean()
    clean_stds = clean_results.groupby("Method").std()
    clean_means.to_csv(os.path.join(output_dir, "results_mean_clean.csv"))
    clean_stds.to_csv(os.path.join(output_dir, "results_std_clean.csv"))

    noisy_results = results[results["noise_L2_norm"] == epsilon]
    noisy_means = noisy_results.groupby("Method").mean()
    noisy_stds = noisy_results.groupby("Method").std()
    noisy_means.to_csv(os.path.join(output_dir, "results_mean_noisy.csv"))
    noisy_stds.to_csv(os.path.join(output_dir, "results_std_noisy.csv"))

    merged_df = pd.merge(noisy_means, noisy_stds, on=["Method"])
    z_value = 1.96
    sqrtn = np.sqrt(n_experiments)
    merged_df["Coverage_CI"] = merged_df.apply(
        lambda row: f"{row['Coverage_x']:.3f}±{(z_value / sqrtn) * row['Coverage_y']:.3f}", axis=1
    )
    merged_df["Size_CI"] = merged_df.apply(
        lambda row: f"{row['Size_x']:.3f}±{(z_value / sqrtn) * row['Size_y']:.3f}", axis=1
    )
    merged_df = merged_df.drop(["Coverage_x", "Coverage_y", "Size_x", "Size_y", "noise_L2_norm_x", "noise_L2_norm_y"], axis=1)
    merged_df.to_csv(os.path.join(output_dir, "results_CI_noisy.csv"))
