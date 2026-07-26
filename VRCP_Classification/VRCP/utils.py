import gc
from typing import List, Literal

import numpy as np
import torch
import torchattacks
from numpy.random import default_rng
from scipy.stats import rankdata
from scipy.stats import norm
from scipy.stats.mstats import mquantiles
from torch.nn.functional import softmax
from tqdm import tqdm


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float], device="cpu"):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        :param device: device to place the mean/std tensors on
        """
        super().__init__()
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds


def evaluate_predictions(S, X, y, num_of_classes):
    # Marginal coverage
    marg_coverage = np.mean([y[i] in S[i] for i in range(len(y))])

    # Size list
    sets_by_size = {}
    for i, s in enumerate(S):
        ssize = len(s)
        if ssize not in sets_by_size:
            sets_by_size[ssize] = []
        sets_by_size[ssize].append((s, y[i]))

    freq_list = np.zeros(num_of_classes + 1)
    for i in range(num_of_classes + 1):
        if i in sets_by_size:
            freq_list[i] = len(sets_by_size[i])

    # Average set size
    size = np.mean([len(S[i]) for i in range(len(y))])

    import pandas as pd
    return pd.DataFrame({'Coverage': [marg_coverage], 'Size': [size], 'Size list': [freq_list]})


# calculate accuracy of the classifier
def calculate_accuracy(model, x, y, num_classes, k=1, device='cpu', GPU_CAPACITY=1024):
    # get size of the test set
    n = x.size()[0]
    # get classifier predictions on points
    model.eval()  # put in evaluation mode
    with torch.no_grad():
        outputs = model(x.to(device)).to(torch.device('cpu'))

    # transform the output into probabilities vector
    outputs = softmax(outputs, dim=1)

    # transform results to numpy array
    predictions = outputs.numpy()

    # get label ranks to calculate top k accuracy
    label_ranks = np.array([rankdata(-predictions[i, :], method='ordinal')[y[i]] - 1 for i in range(n)])

    # get probabilities of correct labels
    label_probs = np.array([predictions[i, y[i]] for i in range(n)])

    # calculate accuracy
    top_k_accuracy = np.sum(label_ranks <= (k - 1)) / float(n)

    # calculate average inverse probability score
    score = np.mean(1 - label_probs)

    # calculate the 0.9 quantile
    quantile = mquantiles(1 - label_probs, prob=0.9)
    return top_k_accuracy, score, quantile


def pgd_attack(model, x, y, norm: Literal["2", "inf"], N_steps=20, max_norm=0.125, device='cpu', GPU_CAPACITY=1024):
    """Generate PGD adversarial examples under an L2 or L-inf norm ball.

    Replaces the previous PGDL2/PGD_Linf pair, whose bodies were identical
    except for which torchattacks attack class they instantiated.
    """
    if x.min() < 0 or x.max() > 1:
        raise ValueError('Input values should be in the [0, 1] range.')
    if norm == "2":
        attack = torchattacks.PGDL2(model, eps=max_norm, alpha=1 / 255, steps=N_steps)
    elif norm == "inf":
        attack = torchattacks.PGD(model, eps=max_norm, alpha=1 / 255, steps=N_steps)
    else:
        raise ValueError(f"norm must be '2' or 'inf', got {norm!r}.")

    x_adv = torch.zeros_like(x)
    n = x.size()[0]
    batch_size = GPU_CAPACITY
    num_of_batches = (n // batch_size) + 1 if n % batch_size != 0 else (n // batch_size)

    print("Generating Adverserial Examples:")
    for j in tqdm(range(num_of_batches)):
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]

        x_adv_batch = attack(inputs, labels)
        x_adv_batch = x_adv_batch.to(torch.device('cpu'))
        x_adv[(j * batch_size):((j + 1) * batch_size)] = x_adv_batch.detach().clone()

        del x_adv_batch
        gc.collect()

    return x_adv


def calculate_Bern_bound(scores, beta=0.001):
    ns = scores.shape[1]
    square_sum = scores.sum(axis=1) ** 2
    sum_square = (scores ** 2).sum(axis=1)
    sample_variance = (sum_square - square_sum / ns) / (ns - 1)
    sample_variance[sample_variance < 0] = 0  # computation stability
    t = np.log(2 / beta)
    bound = np.sqrt(2 * sample_variance * t / ns) + (7 / 3) * t / (ns - 1)
    return bound


def get_scores(model, x, indices, n_smooth, sigma_smooth, num_of_classes, scores_list,
                mode: Literal["base", "verif", "smoothed"] = "base", device='cpu', GPU_CAPACITY=1024):
    """Compute non-conformity scores in one of three modes.

    Replaces the previous `base`/`verif` boolean-pair interface, whose
    cross-product allowed an invalid `base=True, verif=True` state.
      - "base": scores of the (possibly adversarial) classifier's raw output.
      - "verif": scores computed directly from precomputed CROWN bound
        tensors passed in as `x` (no forward pass).
      - "smoothed": randomized-smoothing scores, averaged over `n_smooth`
        noisy samples per point, with Bernstein-bound tracking.
    """
    n = x.size()[0]

    if mode in ("base", "verif"):
        scores_simple = np.zeros((len(scores_list), n, num_of_classes))
    else:
        smoothed_scores = np.zeros((len(scores_list), n, num_of_classes))
        bernstein_bounds = np.zeros((len(scores_list), n, num_of_classes))

    batch_size = GPU_CAPACITY // n_smooth
    num_of_batches = (n // batch_size) + 1 if n % batch_size != 0 else (n // batch_size)

    rng = default_rng()
    uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)

    if mode == "base":
        simple_outputs = np.zeros((n, num_of_classes))

    image_index = -1
    if mode == "verif":
        simple_outputs = x.cpu().numpy()
    else:
        rows = x.size()[2]
        cols = x.size()[3]
        channels = x.size()[1]
        for j in tqdm(range(num_of_batches)):
            inputs = x[(j * batch_size):((j + 1) * batch_size)]
            curr_batch_size = inputs.size()[0]

            if mode == "base":
                noisy_points = inputs.to(device)
            else:
                noises_test = torch.empty((curr_batch_size * n_smooth, channels, rows, cols))
                for k in range(curr_batch_size):
                    image_index = image_index + 1
                    torch.manual_seed(indices[image_index])
                    noises_test[(k * n_smooth):(k + 1) * n_smooth] = torch.randn(
                        (n_smooth, channels, rows, cols)) * sigma_smooth

                tmp = torch.zeros((inputs.size()[0] * n_smooth, *inputs.shape[1:]))
                x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)
                noisy_points = x_tmp + noises_test.to(device)

            model.eval()
            with torch.no_grad():
                noisy_points = model(noisy_points)

            noisy_outputs = softmax(noisy_points, dim=1).to(torch.device('cpu')).numpy()

            if mode == "base":
                simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = noisy_outputs
            else:
                batch_uniform = uniform_variables[(j * batch_size):((j + 1) * batch_size)]
                batch_uniform = np.repeat(batch_uniform, n_smooth)
                for p, score_func in enumerate(scores_list):
                    noisy_scores = score_func(noisy_outputs, np.arange(num_of_classes), batch_uniform, all_combinations=True)
                    smoothed_scores[p, (j * batch_size):((j + 1) * batch_size)] = noisy_scores.reshape(
                        -1, n_smooth, noisy_scores.shape[1]).mean(axis=1)
                    bernstein_bounds[p, (j * batch_size):((j + 1) * batch_size)] = calculate_Bern_bound(
                        noisy_scores.reshape(-1, n_smooth, noisy_scores.shape[1]), beta=0.001)
                del batch_uniform, noisy_scores
                gc.collect()

            if mode == "base":
                del noisy_points, noisy_outputs
            else:
                del noisy_points, noisy_outputs, noises_test, tmp
            gc.collect()

    if mode in ("base", "verif"):
        for p, score_func in enumerate(scores_list):
            scores_simple[p, :, :] = score_func(simple_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)
        return scores_simple

    return smoothed_scores, bernstein_bounds


def calibration(scores_simple=None, smoothed_scores=None, alpha=0.1, correction=0, base=False):
    if base:
        n_calib = scores_simple.shape[0]
    else:
        n_calib = smoothed_scores.shape[0]

    thresholds = np.zeros(3)
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
    bounds = np.zeros(2)

    # mquantiles returns a shape-(1,) array for a scalar `prob`; numpy>=2.0
    # no longer allows assigning that directly into a scalar slot (it used
    # to silently squeeze it), so index out the scalar explicitly.
    if base:
        thresholds[0] = mquantiles(scores_simple[:], prob=level_adjusted)[0]
    else:
        thresholds[1] = mquantiles(smoothed_scores[:], prob=level_adjusted)[0]
        thresholds[2] = mquantiles(smoothed_scores[:], prob=level_adjusted)[0]

        upper_thresh = norm.cdf(norm.ppf(thresholds[2], loc=0, scale=1) + correction, loc=0, scale=1)
        lower_thresh = norm.cdf(norm.ppf(thresholds[2], loc=0, scale=1) - correction, loc=0, scale=1)

        bounds[0] = np.size(smoothed_scores[:][smoothed_scores[:] <= lower_thresh]) / np.size(smoothed_scores[:])
        bounds[1] = np.size(smoothed_scores[:][smoothed_scores[:] <= upper_thresh]) / np.size(smoothed_scores[:])

    return thresholds, bounds


def prediction(scores_simple=None, smoothed_scores=None, ub=None, num_of_scores=2, thresholds=None, correction=0,
               bound_hoef=None, bounds_bern=None, base=False, verif=False, scores_list=None):
    scores_list = scores_list or []
    if base:
        n = scores_simple.shape[1]
    elif verif:
        n = ub.shape[0]
        num_of_classes = ub.shape[1]
        ub = ub.detach().to('cpu').numpy()
    else:
        n = smoothed_scores.shape[1]

    predicted_sets = []
    if verif:
        scores_verif = np.zeros((len(scores_list), n, num_of_classes))
        rng = default_rng()
        uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)
        for p, score_func in enumerate(scores_list):
            scores_verif[p, :, :] = score_func(ub, np.arange(num_of_classes), uniform_variables, all_combinations=True)
            S_hat_verif = [np.where(scores_verif[p, i, :] <= thresholds[p, 0])[0] for i in range(n)]
            predicted_sets.append(S_hat_verif)
    else:
        for p in range(num_of_scores):
            if base:
                S_hat_simple = [np.where(scores_simple[p, i, :] <= thresholds[p, 0])[0] for i in range(n)]
                predicted_sets.append(S_hat_simple)
            else:
                smoothed_S_hat_corrected_MC = [np.where(
                    norm.ppf(np.maximum(smoothed_scores[p, i, :] - bounds_bern[p, i, :], 0), loc=0, scale=1) - correction
                    <= norm.ppf(np.minimum(thresholds[p, 1] + bound_hoef, 1), loc=0, scale=1))[0] for i in range(n)]
                tmp_list = [smoothed_S_hat_corrected_MC]
                predicted_sets.append(tmp_list)
    return predicted_sets
