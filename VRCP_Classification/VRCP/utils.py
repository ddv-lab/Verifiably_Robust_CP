import torch
import numpy as np
import gc
import pandas as pd
from torch.nn.functional import softmax
from scipy.stats import rankdata
from numpy.random import default_rng
from scipy.stats.mstats import mquantiles
from scipy.stats import norm
from tqdm import tqdm
import torchattacks
import torch.nn as nn
from auto_LiRPA.perturbations import *
from typing import List

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(1024, 200)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds

def evaluate_predictions(S, X, y, num_of_classes=10):
    # Marginal coverage
    marg_coverage = np.mean([y[i] in S[i] for i in range(len(y))])

    # Size list
    sets_by_size = {}
    for i, s in enumerate(S):
        ssize = len(s)
        if ssize not in sets_by_size:
            sets_by_size[ssize] = []
        sets_by_size[ssize].append((s, y[i]))

    freq_list = np.zeros(num_of_classes+1)
    for i in range(num_of_classes+1):
        if i in sets_by_size:
            freq_list[i] = len(sets_by_size[i])

    # Average set size
    size = np.mean([len(S[i]) for i in range(len(y))])

    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Size': [size], 'Size list': [freq_list]})

    return out


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
    quantile = mquantiles(1-label_probs, prob=0.9)
    return top_k_accuracy, score, quantile

def PGDL2(model, x, y, N_steps=20, max_norm=0.125, device='cpu', GPU_CAPACITY=1024):
    if x.min() < 0 or x.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
    # create attack model
    attack = torchattacks.PGDL2(model, eps=max_norm, alpha=1/255, steps=N_steps)

    # create container for the adversarial examples
    x_adv = torch.zeros_like(x)

    # get number of data points
    n = x.size()[0]

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # start generating examples for each batch
    print("Generating Adverserial Examples:")

    image_index = -1
    for j in tqdm(range(num_of_batches)):
        # get inputs and labels of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]
    
        # generate adversarial examples for the batch
        x_adv_batch = attack(inputs, labels)

        # take only the one example for each point
        x_adv_batch = x_adv_batch[::1]

        # move back to CPU
        x_adv_batch = x_adv_batch.to(torch.device('cpu'))

        # put in the container
        x_adv[(j * batch_size):((j + 1) * batch_size)] = x_adv_batch.detach().clone()

        del x_adv_batch
        gc.collect()

    # return adversarial examples
    return x_adv

def PGD_Linf(model, x, y, N_steps=20, max_norm=0.125, device='cpu', GPU_CAPACITY=1024):
    if x.min() < 0 or x.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
    # create attack model
    attack = torchattacks.PGD(model, eps=max_norm, alpha=1/255, steps=N_steps)

    # create container for the adversarial examples
    x_adv = torch.zeros_like(x)

    # get number of data points
    n = x.size()[0]

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # start generating examples for each batch
    print("Generating Adverserial Examples:")

    for j in tqdm(range(num_of_batches)):
        # get inputs and labels of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]

        # generate adversarial examples for the batch
        x_adv_batch = attack(inputs, labels)

        # take only the one example for each point
        x_adv_batch = x_adv_batch[::1]

        # move back to CPU
        x_adv_batch = x_adv_batch.to(torch.device('cpu'))

        # put in the container
        x_adv[(j * batch_size):((j + 1) * batch_size)] = x_adv_batch.detach().clone()

        del x_adv_batch
        gc.collect()

    # return adversarial examples
    return x_adv

def calculate_Bern_bound(scores, beta=0.001):
    ns = scores.shape[1]
    square_sum = scores.sum(axis=1) ** 2
    sum_square = (scores ** 2).sum(axis=1)
    sample_variance = (sum_square - square_sum / ns) / (ns - 1)
    sample_variance[sample_variance<0] = 0 # computation stability
    t = np.log(2 / beta)
    bound = np.sqrt(2 * sample_variance * t / ns) + (7 / 3) * t  / (ns - 1)
    return bound

def get_scores(model, x, indices, n_smooth, sigma_smooth, num_of_classes, scores_list, base=False, verif=False, device='cpu', GPU_CAPACITY=1024):
    # get number of points
    n = x.size()[0]

    # create container for the scores
    if base or verif:
        scores_simple = np.zeros((len(scores_list), n, num_of_classes))
    else:
        smoothed_scores = np.zeros((len(scores_list), n, num_of_classes))
        bernstein_bounds = np.zeros((len(scores_list), n, num_of_classes))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # initiate random uniform variables for inverse quantile score
    rng = default_rng()
    uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)

    # create container for smoothed and base classifier outputs
    if base:
        simple_outputs = np.zeros((n, num_of_classes))

    image_index = -1
    if verif:
        simple_outputs = x.cpu().numpy()
    else:
        # get dimension of data
        rows = x.size()[2]
        cols = x.size()[3]
        channels = x.size()[1]
        for j in tqdm(range(num_of_batches)):
            # get inputs of batch
            inputs = x[(j * batch_size):((j + 1) * batch_size)]
            curr_batch_size = inputs.size()[0]

            if base:
                noisy_points = inputs.to(device)
            else:
                noises_test = torch.empty((curr_batch_size * n_smooth, channels, rows, cols))
                # get relevant noises for this batch
                for k in range(curr_batch_size):
                    image_index = image_index + 1
                    torch.manual_seed(indices[image_index])
                    noises_test[(k * n_smooth):(k + 1) * n_smooth] = torch.randn(
                        (n_smooth, channels, rows, cols)) * sigma_smooth

                # duplicate batch according to the number of added noises and send to device
                # the first n_smooth samples will be duplicates of x[0] and etc.
                tmp = torch.zeros((inputs.size()[0] * n_smooth, *inputs.shape[1:]))
                x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

                # add noise to points
                noisy_points = x_tmp + noises_test.to(device)

            # get classifier predictions on noisy points
            model.eval()  # put in evaluation mode
            with torch.no_grad():
                noisy_points = model(noisy_points)

            # transform the output into probabilities vector
            noisy_outputs = softmax(noisy_points, dim=1).to(torch.device('cpu')).numpy()
            
            if base:
                simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = noisy_outputs
            else:
                # get smoothed classifier outputs
                batch_uniform = uniform_variables[(j * batch_size):((j + 1) * batch_size)]
                batch_uniform = np.repeat(batch_uniform, n_smooth)
                for p, score_func in enumerate(scores_list):
                    # get scores for all noisy outputs for all classes
                    noisy_scores = score_func(noisy_outputs, np.arange(num_of_classes), batch_uniform, all_combinations=True)
                    # average n_smooth scores for eac points
                    smoothed_scores[p, (j * batch_size):((j + 1) * batch_size)] = noisy_scores.reshape(-1, n_smooth, noisy_scores.shape[1]).mean(axis=1)
                    bernstein_bounds[p, (j * batch_size):((j + 1) * batch_size)] = calculate_Bern_bound(noisy_scores.reshape(-1, n_smooth, noisy_scores.shape[1]), beta=0.001)
                # clean
                del batch_uniform, noisy_scores
                gc.collect()

            if base:
                del noisy_points, noisy_outputs
            else:
                del noisy_points, noisy_outputs, noises_test, tmp
            gc.collect()

    # run over all scores functions and compute scores of smoothed and base classifier
    for p, score_func in enumerate(scores_list):
        if base or verif:
            scores_simple[p, :, :] = score_func(simple_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)

    # return relevant scores
    if base or verif:
        return scores_simple
    else:
        return smoothed_scores, bernstein_bounds

def calibration(scores_simple=None, smoothed_scores=None, alpha=0.1, correction=0, base=False):
    # size of the calibration set
    if base:
        n_calib = scores_simple.shape[0]
    else:
        n_calib = smoothed_scores.shape[0]

    # create container for the calibration thresholds
    thresholds = np.zeros(3)
    
    # Compute thresholds
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
    bounds = np.zeros(2)

    if base:
        thresholds[0] = mquantiles(scores_simple[:], prob=level_adjusted)
    else:
        thresholds[1] = mquantiles(smoothed_scores[:], prob=level_adjusted)
        thresholds[2] = mquantiles(smoothed_scores[:], prob=level_adjusted)

        # calculate lower and upper bounds of correction of smoothed score
        upper_thresh = norm.cdf(norm.ppf(thresholds[2], loc=0, scale=1)+correction, loc=0, scale=1)
        lower_thresh = norm.cdf(norm.ppf(thresholds[2], loc=0, scale=1)-correction, loc=0, scale=1)

        bounds[0] = np.size(smoothed_scores[:][smoothed_scores[:] <= lower_thresh])/np.size(smoothed_scores[:])
        bounds[1] = np.size(smoothed_scores[:][smoothed_scores[:] <= upper_thresh]) / np.size(smoothed_scores[:])

    return thresholds, bounds

def prediction(scores_simple=None, smoothed_scores=None, ub=None, num_of_scores=2, thresholds=None, correction=0, bound_hoef=None, bounds_bern=None, base=False, verif=False, scores_list=[]):
    # get number of points
    if base:
        n = scores_simple.shape[1]
    elif verif:
        n = ub.shape[0]
        num_of_classes = ub.shape[1]
        ub = ub.detach().to('cpu').numpy()
    else:
        n = smoothed_scores.shape[1]

    # Generate prediction sets using the thresholds from the calibration
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
                smoothed_S_hat = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 1], loc=0, scale=1))[0] for i in range(n)]
                smoothed_S_hat_corrected = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) - correction <= norm.ppf(thresholds[p, 1], loc=0, scale=1))[0] for i in range(n)]
                # RSCP+ prediction sets using bernstein bounds (better performance)
                smoothed_S_hat_corrected_MC = [np.where(norm.ppf(np.maximum(smoothed_scores[p, i, :] - bounds_bern[p, i, :], 0), loc=0, scale=1) - correction <= norm.ppf(np.minimum(thresholds[p, 1] + bound_hoef, 1), loc=0, scale=1))[0] for i in range(n)]
                # RSCP+ prediction stes using hoeffding bounds
                smoothed_S_hat_corrected_MC_Hoef = [np.where(norm.ppf(np.maximum(smoothed_scores[p, i, :] - bound_hoef, 0), loc=0, scale=1) - correction <= norm.ppf(np.minimum(thresholds[p, 1] + bound_hoef, 1), loc=0, scale=1))[0] for i in range(n)]
                tmp_list = [smoothed_S_hat_corrected_MC]
                predicted_sets.append(tmp_list)
    return predicted_sets