"""The core calibration/test-split experiment loop.

Kept as one function (rather than split further) since its local state --
thresholds, bounds, predicted sets, the running results table -- is
tightly coupled throughout.

Note: `predicted_adv_sets_verif_pre` is computed from
`scores_simple_clean_test`, not `scores_simple_adv_test`. This looks like
it could be a bug, but may instead reflect that a verified bound valid
over the clean point's epsilon-ball also covers the adversarial point
within it (see VRCP/bounds.py) -- don't change this without confirming
which is intended.
"""

import gc

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import VRCP.score_functions as scores
from VRCP.utils import calibration, get_scores, prediction, evaluate_predictions

CALIBRATION_SCORES = ["HPS", "PTT_HPS"]
POST_CAL = True
BERN_BETA = 0.001


def _build_scores_list(calibration_scores, model, x_valid, y_valid, alpha, sigma_model, num_of_classes, device, gpu_capacity):
    scores_list = []
    for score in calibration_scores:
        if score == "HPS":
            scores_list.append(scores.class_probability_score)
        elif score == "PTT_HPS":
            base_score = scores.class_probability_score
            ref_scores = get_scores(
                model, x_valid, np.arange(x_valid.shape[0]), 1, sigma_model, num_of_classes, [base_score],
                mode="base", device=device, GPU_CAPACITY=gpu_capacity,
            ).squeeze()
            ref_scores = ref_scores[np.arange(ref_scores.shape[0]), y_valid.numpy()]

            ptt_score = scores.ranking_score(ref_scores, base_score)
            ptt_score = scores.sigmoid_score(ptt_score, T=400, bias=1 - alpha)
            scores_list.append(ptt_score)
        else:
            raise ValueError(f"Undefined score function: {score!r}")
    return scores_list


def run_experiments(
    model, all_x_test, all_y_test, all_x_test_adv, all_x_test_adv_base,
    lb_tensor, lb_tensor_adv, ub_tensor, ub_tensor_adv,
    alpha, epsilon, n_experiments, n_smooth, sigma_model, sigma_smooth, correction,
    num_of_classes, n_test, device, gpu_capacity,
):
    """Run `n_experiments` random calibration/test splits, evaluating the
    four regimes (vanilla, verified, verified-pre, RSCP+) on both clean and
    adversarial test points. Returns (results_df, quantiles)."""
    all_indices = torch.arange(n_test)
    result_rows = []
    quantiles = np.zeros((len(CALIBRATION_SCORES), 2, n_experiments))
    bound_hoef = np.sqrt(-np.log(0.001) / 2 / n_smooth)

    for experiment in range(n_experiments):
        print("Experiment #", experiment + 1)
        test_ratio = 500 / n_test
        indices_test, indices_valid = train_test_split(all_indices, test_size=test_ratio, shuffle=True)
        x_test, x_valid = all_x_test[indices_test], all_x_test[indices_valid]
        y_test, y_valid = all_y_test[indices_test], all_y_test[indices_valid]
        x_test_adv = all_x_test_adv[indices_test]
        gc.collect()
        x_test_adv_base = all_x_test_adv_base[indices_test]
        gc.collect()
        new_n_test = x_test.shape[0]
        indices = torch.arange(new_n_test)

        scores_list = _build_scores_list(
            CALIBRATION_SCORES, model, x_valid, y_valid, alpha, sigma_model, num_of_classes, device, gpu_capacity
        )

        print("Calculating base scores on the clean test points:\n")
        scores_simple_clean_test = get_scores(
            model, x_test, indices, n_smooth, sigma_model, num_of_classes, scores_list,
            mode="base", device=device, GPU_CAPACITY=gpu_capacity,
        )
        print("Calculating smoothed scores on the clean test points:\n")
        smoothed_scores_clean_test, bern_bound_clean = get_scores(
            model, x_test, indices, n_smooth, sigma_smooth, num_of_classes, scores_list,
            mode="smoothed", device=device, GPU_CAPACITY=gpu_capacity,
        )
        print("Calculating base scores on the adversarial test points:\n")
        scores_simple_adv_test = get_scores(
            model, x_test_adv_base, indices, n_smooth, sigma_model, num_of_classes, scores_list,
            mode="base", device=device, GPU_CAPACITY=gpu_capacity,
        )
        print("Calculating smoothed scores on the adversarial test points:\n")
        smoothed_scores_adv_test, bern_bound_adv = get_scores(
            model, x_test_adv, indices, n_smooth, sigma_smooth, num_of_classes, scores_list,
            mode="smoothed", device=device, GPU_CAPACITY=gpu_capacity,
        )
        print("Calculating verified scores on the clean test points:\n")
        scores_verif_clean_test = get_scores(
            model, lb_tensor[indices_test], indices, n_smooth, sigma_model, num_of_classes, scores_list,
            mode="verif", device=device, GPU_CAPACITY=gpu_capacity,
        )
        test_ratio = (n_test / 2) / new_n_test
        idx1, idx2 = train_test_split(indices, test_size=test_ratio)
        print(f"Size of [holdout, calibration, test] split is "
              f"[{indices_valid.shape[0]}, {idx1.shape[0]}, {idx2.shape[0]}]")

        thresholds_base = np.zeros((len(scores_list), 3))
        thresholds_verif = np.zeros((len(scores_list), 3))
        thresholds = np.zeros((len(scores_list), 3))
        bounds = np.zeros((len(scores_list), 2))

        for p in range(len(scores_list)):
            thresholds_base[p], _ = calibration(
                scores_simple=scores_simple_clean_test[p, idx1, y_test[idx1]], alpha=alpha, correction=correction, base=True
            )
            thresholds_verif[p], _ = calibration(
                scores_simple=scores_verif_clean_test[p, idx1, y_test[idx1]], alpha=alpha, correction=correction, base=True
            )
            thresholds[p], bounds[p] = calibration(
                smoothed_scores=smoothed_scores_clean_test[p, idx1, y_test[idx1]], alpha=alpha - 2 * BERN_BETA,
                correction=correction, base=False,
            )

        for p in range(len(scores_list)):
            quantiles[p, 0, experiment] = bounds[p, 0]
            quantiles[p, 1, experiment] = bounds[p, 1]

        predicted_clean_sets_base = prediction(
            scores_simple=scores_simple_clean_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds_base, base=True
        )
        predicted_clean_sets_verif = prediction(
            ub=ub_tensor[indices_test[idx2]], num_of_scores=len(scores_list), thresholds=thresholds_base,
            base=False, verif=True, scores_list=scores_list,
        )
        predicted_clean_sets_verif_pre = prediction(
            scores_simple=scores_simple_clean_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds_verif, base=True
        )
        predicted_clean_sets = prediction(
            smoothed_scores=smoothed_scores_clean_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds,
            correction=correction, bound_hoef=bound_hoef, bounds_bern=bern_bound_clean, base=False,
        )

        predicted_adv_sets_base = prediction(
            scores_simple=scores_simple_adv_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds_base, base=True
        )
        predicted_adv_sets_verif = prediction(
            ub=ub_tensor_adv[indices_test[idx2]], num_of_scores=len(scores_list), thresholds=thresholds_base,
            base=False, verif=True, scores_list=scores_list,
        )
        # Uses clean scores intentionally -- see module docstring.
        predicted_adv_sets_verif_pre = prediction(
            scores_simple=scores_simple_clean_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds_verif, base=True
        )
        predicted_adv_sets = prediction(
            smoothed_scores=smoothed_scores_adv_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds,
            correction=correction, bound_hoef=bound_hoef, bounds_bern=bern_bound_adv, base=False,
        )

        for p in range(len(scores_list)):
            score_name = CALIBRATION_SCORES[p]
            methods_list = [score_name + "_vanilla", score_name + "_verif_pre", score_name + "_RSCP+"]
            predicted_clean_sets[p].insert(0, predicted_clean_sets_verif_pre[p])
            if POST_CAL:
                methods_list = [score_name + "_vanilla", score_name + "_verif", score_name + "_verif_pre", score_name + "_RSCP+"]
                predicted_clean_sets[p].insert(0, predicted_clean_sets_verif[p])
            predicted_clean_sets[p].insert(0, predicted_clean_sets_base[p])
            predicted_adv_sets[p].insert(0, predicted_adv_sets_verif_pre[p])
            if POST_CAL:
                predicted_adv_sets[p].insert(0, predicted_adv_sets_verif[p])
            predicted_adv_sets[p].insert(0, predicted_adv_sets_base[p])

            for r, method in enumerate(methods_list):
                res = evaluate_predictions(predicted_clean_sets[p][r], None, y_test[idx2].numpy(), num_of_classes=num_of_classes)
                res["Method"] = methods_list[r]
                res["noise_L2_norm"] = 0
                res["Black box"] = ""
                result_rows.append(res)

        for p in range(len(scores_list)):
            score_name = CALIBRATION_SCORES[p]
            methods_list = [score_name + "_vanilla", score_name + "_verif_pre", score_name + "_RSCP+"]
            if POST_CAL:
                methods_list = [score_name + "_vanilla", score_name + "_verif", score_name + "_verif_pre", score_name + "_RSCP+"]
            for r, method in enumerate(methods_list):
                res = evaluate_predictions(predicted_adv_sets[p][r], None, y_test[idx2].numpy(), num_of_classes=num_of_classes)
                res["Method"] = methods_list[r]
                res["noise_L2_norm"] = epsilon
                res["Black box"] = ""
                result_rows.append(res)

        del idx1, idx2, predicted_clean_sets, predicted_clean_sets_base, predicted_clean_sets_verif_pre
        del predicted_adv_sets, predicted_adv_sets_base, predicted_adv_sets_verif_pre, bounds, thresholds, thresholds_base
        if POST_CAL:
            del predicted_clean_sets_verif, predicted_adv_sets_verif
        gc.collect()

    results = pd.concat(result_rows, ignore_index=True)
    return results, quantiles
