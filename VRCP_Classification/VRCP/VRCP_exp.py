# this code is heavily adapted from the original RSCP (Gendler et al.) code
# retrieved from: https://github.com/Asafgendler/RSCP
# general imports

import gc
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
import random
import torch
import torchvision
import os
import pickle
import sys
import argparse
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt

from torch.nn.functional import softmax
import torch.nn as nn
from auto_LiRPA.utils import Flatten

sys.path.insert(0, './')
import VRCP.Score_Functions as scores
from VRCP.utils import *
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *

import warnings
warnings.filterwarnings("ignore")

# parameters
parser = argparse.ArgumentParser(description='Experiments')
parser.add_argument('-a', '--alpha', default=0.1, type=float, help='Desired nominal marginal coverage')
parser.add_argument('-d', '--delta', default=0.02, type=float, help='L2 bound on the adversarial noise')
parser.add_argument('-s', '--splits', default=50, type=int, help='Number of experiments to estimate coverage')
parser.add_argument('-r', '--ratio', default=2, type=float, help='Ratio between adversarial noise bound to smoothing noise')
parser.add_argument('-n', '--norm', default='2', type=str, help='l_p norm used for epsilon ball: 2, inf')
parser.add_argument('--n_s', default=256, type=int, help='Number of samples used for estimating smoothed score')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to be used: CIFAR100, CIFAR10, TINYNET')
parser.add_argument('--arc', default='CNN', type=str, help='Architecture of classifier : CNN')
parser.add_argument('--My_model', action='store_true', help='True for our trained model, False for Cohens. Relevent only for CIFAR10')
parser.add_argument('--batch_size', default=1024, type=int, help='Number of images to send to gpu at once')

args = parser.parse_args()

# parameters
alpha = args.alpha  # desired nominal marginal coverage
epsilon = args.delta  # L2 bound on the adversarial noise
n_experiments = args.splits  # number of experiments to estimate coverage
ratio = args.ratio  # ratio between adversarial noise bound to smoothed noise
norm = args.norm  # l_p norm used for epsilon ball
sigma_smooth = ratio * epsilon # sigma used for smoothing
sigma_model = 0 # sigma used for training the model is 0 in all our models
n_smooth = args.n_s  # number of samples used for smoothing
My_model = args.My_model
N_steps = 100  # number of gradient steps for PGD attack
dataset = args.dataset  # dataset to be used  CIFAR100', 'CIFAR10'
post_cal = True
calibration_scores = ['HPS', 'PTT_HPS'] # score functions
model_type = args.arc # Architecture of the model

# number of test points (if larger then available it takes the entire set)
n_test = 10000

# Validate parameters
assert dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'TINYNET', 'Dataset can only be CIFAR10, CIFAR100 or TINYNET.'
assert 0 <= alpha <= 1, 'Nominal level must be between 0 to 1'
assert not(n_smooth & (n_smooth - 1)), 'n_s must be a power of 2.'
assert not(args.batch_size & (args.batch_size - 1)), 'batch size must be a power of 2.'
assert args.batch_size >= n_smooth, 'batch size must be larger than n_s'
assert model_type == 'CNN', 'Architecture can only be CNN.'
assert norm == '2' or norm == 'inf', 'Norm can only be 2 (default) or inf'
assert sigma_model >= 0, 'std for training the model must be a non negative number.'
assert epsilon >= 0, 'L2 bound of noise must be non negative.'
assert isinstance(n_experiments, int) and n_experiments >= 1, 'number of splits must be a positive integer.'
assert ratio >= 0, 'ratio between sigma and delta must be non negative.'

# CIFAR100 has only my models
if dataset == "CIFAR100":
    My_model = True

# All our models are needs to be added a normalization layer
if My_model:
    robust = True

# The GPU used for our experiments can only handle the following quantities of images per batch
GPU_CAPACITY = args.batch_size

# Save results to final results directories only if full data is taken. Otherwise save locally.
if ((n_experiments == 50) and (n_test == 10000)):
    save_results = True
else:
    save_results = False

# calculate correction based on the Lipschitz constant
if sigma_smooth == 0:
    correction = 10000
else:
    correction = float(epsilon) / float(sigma_smooth)

# set random seed
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load datasets
if dataset == "CIFAR10":
    # Load train set
    train_dataset = torchvision.datasets.CIFAR10(root='./Datasets/',
                                                 train=True,
                                                 transform=torchvision.transforms.ToTensor(),
                                                 download=True)
    # load test set
    test_dataset = torchvision.datasets.CIFAR10(root='./Datasets/',
                                                train=False,
                                                transform=torchvision.transforms.ToTensor())
elif dataset == "CIFAR100":
    # Load train set
    train_dataset = torchvision.datasets.CIFAR100(root='./Datasets/',
                                                  train=True,
                                                  transform=torchvision.transforms.ToTensor(),
                                                  download=True)
    # load test set
    test_dataset = torchvision.datasets.CIFAR100(root='./Datasets/',
                                                 train=False,
                                                 transform=torchvision.transforms.ToTensor())
elif dataset == "TINYNET":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Load train set
    train_dataset = datasets.ImageFolder(root='Datasets/tiny-imagenet-200/train', transform=transform)
    # Load test set
    test_dataset = datasets.ImageFolder(root='Datasets/tiny-imagenet-200/val', transform=transform)
else:
    print("No such dataset")
    exit(1)

print("Train set is len: ", len(train_dataset))
print("Test set is len: ", len(test_dataset))

# cut the size of the test set if necessary
if n_test < len(test_dataset):
    torch.manual_seed(0)
    test_dataset = random_split(test_dataset, [n_test, len(test_dataset) - n_test])[0]

# save the sizes of each one of the sets
n_test = len(test_dataset)

# Create Data loader for test set
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=n_test,
                                          shuffle=False)

# convert test set into tensor
examples = enumerate(test_loader)
batch_idx, (all_x_test, all_y_test) = next(examples)

# get dimension of data
rows = all_x_test.size()[2]
cols = all_x_test.size()[3]
channels = all_x_test.size()[1]
print(f'Dimensions: Row={rows} Cols={cols} Channels={channels}')
if dataset == 'CIFAR100':
    num_of_classes = 100
elif dataset == 'TINYNET':
    num_of_classes = 200
else:
    num_of_classes = 10
min_pixel_value = 0.0
max_pixel_value = 1.0

# automatically choose device use gpu 0 if it is available o.w. use the cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print the chosen device
print("device: ", device)
# load my models
if My_model:
    if dataset == "CIFAR10":
        if model_type == 'CNN':
            model = nn.Sequential(
                nn.Conv2d(3, 16, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),
                nn.Dropout(0.4),
                Flatten(),
                nn.Linear(512,128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
            state = torch.load('./Checkpoints/cifar10CNN.pth') 
            model.load_state_dict(state['net'])
        else:
            print("No such architecture")
            exit(1)
    elif dataset == "CIFAR100":
        if model_type == 'CNN':
            model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),
                nn.Dropout(0.4),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),
                nn.Dropout(0.4),
                Flatten(),
                nn.Linear(1024,256),
                nn.ReLU(),
                nn.Linear(256, 100)
            )
            state = torch.load('./Checkpoints/cifar100CNN.pth')  
            model.load_state_dict(state['net'])
    elif dataset == "TINYNET":
        if model_type == 'CNN':
            model = ConvNet()
            state = torch.load('./Checkpoints/tinynetCNN.pth')  
            model.load_state_dict(state)
    else:
        print("No such architecture")
        exit(1)

# send model to device
model.to(device)

# put model in evaluation mode
model.eval()

# create indices for the test points
all_indices = torch.arange(n_test)

# directory to store adversarial examples and noises
directory = "./Adversarial_Examples/" + str(dataset) + "/epsilon_" + str(epsilon)

# normalization layer to my model
if robust:
    directory = directory + "/Robust"

# different attacks for different norms
if norm == "inf":
    directory = directory + "/" + str(norm)

print("Save results on directories: " + str(save_results))
print("Searching for adversarial examples in: " + str(directory))
if os.path.exists(directory):
    print("Are there saved adversarial examples: Yes")
else:
    print("Are there saved adversarial examples: No")

noises_base = torch.empty_like(all_x_test)
for k in range(n_test):
    torch.manual_seed(k)
    noises_base[k:(k + 1)] = torch.randn(
        (1, channels, rows, cols)) * sigma_model



# If there are no pre created adversarial examples, create new ones
if n_test != 10000 or not os.path.exists(directory):
    # Generate adversarial test examples
    print("Generate adversarial test examples for the smoothed model:\n")
    if dataset == "TINYNET":
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                        transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                            std = [ 1., 1., 1. ]),
                                    ])
        temp_x_test = invTrans(all_x_test)
        normalize_layer = NormalizeLayer([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        temp_model = nn.Sequential(normalize_layer, model)
        if norm == '2':
            all_x_test_adv = PGDL2(temp_model, temp_x_test, all_y_test, N_steps, epsilon/2, device, GPU_CAPACITY=GPU_CAPACITY)
        elif norm == 'inf':
            all_x_test_adv = PGD_Linf(temp_model, temp_x_test, all_y_test, N_steps, epsilon/2, device, GPU_CAPACITY=GPU_CAPACITY)
        standardTrans = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        all_x_test_adv = standardTrans(all_x_test_adv)
    else:
        if norm == '2':
            all_x_test_adv = PGDL2(model, all_x_test, all_y_test, N_steps, epsilon, device, GPU_CAPACITY=GPU_CAPACITY)
        elif norm == 'inf':
            all_x_test_adv = PGD_Linf(model, all_x_test, all_y_test, N_steps, epsilon, device, GPU_CAPACITY=GPU_CAPACITY)
    # Generate adversarial test examples for the base classifier
    all_x_test_adv_base = all_x_test_adv
    
    # Only store examples for full dataset
    if (n_test == 10000):
        os.makedirs(directory)
        with open(directory + "/data.pickle", 'wb') as f:
            pickle.dump([all_x_test_adv, all_x_test_adv_base], f)

# If there are pre created adversarial examples, load them
else:
    with open(directory + "/data.pickle", 'rb') as f:
        all_x_test_adv, all_x_test_adv_base = pickle.load(f)

# Calculate accuracy of classifier on clean test points
acc, _, _ = calculate_accuracy(model, all_x_test, all_y_test, num_of_classes, k=1, device=device, GPU_CAPACITY=GPU_CAPACITY)
print("True Model accuracy :" + str(acc * 100) + "%")

# Calculate accuracy of classifier on adversarial test points
acc, _, _ = calculate_accuracy(model, all_x_test_adv_base, all_y_test, num_of_classes, k=1, device=device, GPU_CAPACITY=GPU_CAPACITY)
print("True Model accuracy on adversarial examples :" + str(acc * 100) + "%")


directory = "./Bounds/" + str(dataset) + "/epsilon_" + str(epsilon)

if norm == "inf":
    directory = directory + "/" + str(norm)
    norm = np.inf
else:
    norm = 2

print("Searching for precomputed bounds in: " + str(directory))
if os.path.exists(directory):
    print("Are there saved bounds: Yes")
else:
    print("Are there saved bounds: No")

if (n_test != 10000) or not os.path.exists(directory):
    model.to(device)
    model.eval()
    if num_of_classes == 10:
        batch_size = GPU_CAPACITY // 384
        bound_method = "CROWN-Optimized"
    elif num_of_classes == 100:
        batch_size = 2
        bound_method = "CROWN-Optimized"
    elif num_of_classes == 200:
        batch_size = 2
        bound_method = "CROWN"

    if n_test % batch_size != 0:
        num_of_batches = (n_test // batch_size) + 1
    else:
        num_of_batches = (n_test // batch_size)

    print(model(all_x_test[0].view(1,channels,rows,cols).to(device)))
    dummy = torch.zeros_like(all_x_test[:batch_size].view(batch_size,channels,rows,cols)).to(device)
    lirpa_model = BoundedModule(model, dummy)
    print("bounded")
    lirpa_model.to(device)
    lirpa_model.eval()
    ptb = PerturbationLpNorm(norm=norm, eps=epsilon)
    print(lirpa_model(all_x_test[0].view(1,channels,rows,cols).to(device)))
    print(model(all_x_test[0].view(1,channels,rows,cols).to(device)))

    smx_clean = softmax(lirpa_model(all_x_test.to(device)),dim=1).detach().to('cpu').numpy()
    smx_adv = softmax(lirpa_model(all_x_test_adv_base.to(device)),dim=1).detach().to('cpu').numpy()

    ub_tensor = torch.zeros(n_test, num_of_classes).to(device)
    ub_tensor_adv = torch.zeros(n_test, num_of_classes).to(device)
    lb_tensor = torch.zeros(n_test, num_of_classes).to(device)
    lb_tensor_adv = torch.zeros(n_test, num_of_classes).to(device)
    for i in tqdm(range(num_of_batches)):
        idxs = all_indices[(i * batch_size):((i + 1) * batch_size)]
        input_tensor = all_x_test[idxs]
        curr_batch_size = input_tensor.size()[0]
        input_tensor = input_tensor.view(curr_batch_size,channels,rows,cols).to(device)

        my_input = BoundedTensor(input_tensor, ptb)
        # compute bounds for the current test image
        
        if bound_method == "CROWN":
            with torch.no_grad():
                lb, ub = lirpa_model.compute_bounds(x=(my_input,), method=bound_method)
        else:
            lb, ub = lirpa_model.compute_bounds(x=(my_input,), method=bound_method)
        lb = lb[:].to('cpu')
        ub = ub[:].to('cpu')
        # update the upper bound tensor
        sigmalb = np.zeros((curr_batch_size,num_of_classes))
        sigmaub = np.zeros((curr_batch_size,num_of_classes))

        for k in range(curr_batch_size):
            for j in range(num_of_classes):
                temp = lb.numpy().copy()
                temp = np.append(temp,ub[k][j].numpy())
                maxnum = max(temp)
                sigmalb[k][j] = (np.exp(lb[k][j])/(np.exp(lb[k][j])+torch.sum(np.exp(ub[k]))-np.exp(ub[k][j])))
                sigmaub[k][j]= (np.exp(ub[k][j]-maxnum)/(np.exp(ub[k][j]-maxnum)+torch.sum(np.exp(lb[k]-maxnum))-np.exp(lb[k][j]-maxnum)))
            lb_tensor[(i * batch_size)+k:(i * batch_size)+k+1] = torch.from_numpy(sigmalb[k])
            ub_tensor[(i * batch_size)+k:(i* batch_size)+k+1] = torch.from_numpy(sigmaub[k]) 

        adv_input_tensor = all_x_test_adv_base[idxs]
        adv_input_tensor = adv_input_tensor.view(curr_batch_size,channels,rows,cols).to(device)

        my_adv_input = BoundedTensor(adv_input_tensor, ptb)
        
        if bound_method == "CROWN":
            with torch.no_grad():
                lb_adv, ub_adv = lirpa_model.compute_bounds(x=(my_input,), method=bound_method)
        else:
            lb_adv, ub_adv = lirpa_model.compute_bounds(x=(my_input,), method=bound_method)
        lb_adv = lb_adv[:].to('cpu')
        ub_adv = ub_adv[:].to('cpu')

        sigmalb_adv = np.zeros((curr_batch_size,num_of_classes))
        sigmaub_adv = np.zeros((curr_batch_size,num_of_classes))

        for k in range(curr_batch_size):
            for j in range(num_of_classes):
                temp = lb_adv.numpy().copy()
                temp = np.append(temp,ub_adv[k][j].numpy())
                maxnum = max(temp)
                sigmalb_adv[k][j] = (np.exp(lb_adv[k][j])/(np.exp(lb_adv[k][j])+torch.sum(np.exp(ub_adv[k]))-np.exp(ub_adv[k][j])))
                sigmaub_adv[k][j]= (np.exp(ub_adv[k][j]-maxnum)/(np.exp(ub_adv[k][j]-maxnum)+torch.sum(np.exp(lb_adv[k]-maxnum))-np.exp(lb_adv[k][j]-maxnum)))
            lb_tensor_adv[(i * batch_size)+k:(i * batch_size)+k+1] = torch.from_numpy(sigmalb_adv[k])
            ub_tensor_adv[(i * batch_size)+k:(i* batch_size)+k+1] = torch.from_numpy(sigmaub_adv[k]) 

    if (n_test == 10000):
        os.makedirs(directory)
        with open(directory + "/data.pickle", 'wb') as f:
            pickle.dump([lb_tensor, lb_tensor_adv, ub_tensor, ub_tensor_adv], f)
    del ub, ub_adv, lb, lb_adv, input_tensor, my_input, my_adv_input
    gc.collect()
else:
    with open(directory + "/data.pickle", 'rb') as f:
        lb_tensor, lb_tensor_adv, ub_tensor, ub_tensor_adv = pickle.load(f)

if save_results:
    # directory to save results
    res_directory = "./Results/" + str(dataset) + "/epsilon_" + str(epsilon) + "/sigma_model_" + str(
        sigma_model) + "/sigma_smooth_" + str(sigma_smooth) + "/n_smooth_" + str(n_smooth)

    if robust:
        res_directory = res_directory + "/Robust"

    res_directory = res_directory + "/" + str(model_type)

    if alpha != 0.1:
        res_directory = res_directory + "/alpha_" + str(alpha)

    if norm == np.inf:
        res_directory = res_directory + "/inf"

    if not os.path.exists(res_directory):
        os.makedirs(res_directory)

if save_results:
    if not os.path.exists(res_directory):
        os.makedirs(res_directory)

# create dataframe for storing results
results = pd.DataFrame()

# container for storing bounds on "CP+SS"
quantiles = np.zeros((len(calibration_scores), 2, n_experiments))

bound_hoef = np.sqrt(-np.log(0.001) / 2 / n_smooth)

# run for n_experiments data splittings
print("\nRunning experiments for "+str(n_experiments)+" random splits:\n")
for experiment in tqdm(range(n_experiments)):
    print("Experiment #",experiment+1)
    # Split test data into calibration and test
    test_ratio = 500 / n_test
    indices_test, indices_valid = train_test_split(all_indices, test_size=test_ratio, shuffle=True)
    x_test, x_valid = all_x_test[indices_test], all_x_test[indices_valid]
    y_test, y_valid = all_y_test[indices_test], all_y_test[indices_valid]
    x_test_adv, x_valid_adv =  all_x_test_adv[indices_test], all_x_test_adv[indices_valid]
    gc.collect()
    x_test_adv_base, x_valid_adv_base = all_x_test_adv_base[indices_test], all_x_test_adv_base[indices_valid]
    gc.collect()
    new_n_test = x_test.shape[0]
    indices = torch.arange(new_n_test)
    scores_list = []
    for score in calibration_scores:
        if score == 'HPS':
            scores_list.append(scores.class_probability_score)
        elif score == 'PTT_HPS':
            base_score = scores.class_probability_score
            ref_scores = get_scores(model, x_valid, np.arange(x_valid.shape[0]), 1, sigma_model, num_of_classes, [base_score], base=True, device=device, GPU_CAPACITY=GPU_CAPACITY).squeeze()
            ref_scores = ref_scores[np.arange(ref_scores.shape[0]), y_valid.numpy()]

            PTT_score = scores.ranking_score(ref_scores, base_score)
            PTT_score = scores.sigmoid_score(PTT_score, T=400, bias=1-alpha)
            scores_list.append(PTT_score)
        else:
            print("Undefined score function")
            exit(1)
        print("Calculating base scores on the clean test points:\n")
        scores_simple_clean_test = get_scores(model, x_test, indices, n_smooth, sigma_model, num_of_classes, scores_list, base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)
        print("Calculating smoothed scores on the clean test points:\n")
        smoothed_scores_clean_test, Bern_bound_clean = get_scores(model, x_test, indices, n_smooth, sigma_smooth, num_of_classes, scores_list, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY)
        print("Calculating base scores on the adversarial test points:\n")
        scores_simple_adv_test = get_scores(model, x_test_adv_base, indices, n_smooth, sigma_model, num_of_classes, scores_list, base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)
        print("Calculating smoothed scores on the adversarial test points:\n")
        smoothed_scores_adv_test, Bern_bound_adv = get_scores(model, x_test_adv, indices, n_smooth, sigma_smooth, num_of_classes, scores_list, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY)
        print("Calculating verified scores on the clean test points:\n")
        scores_verif_clean_test = get_scores(model, lb_tensor[indices_test], indices, n_smooth, sigma_model, num_of_classes, scores_list, verif=True, device=device, GPU_CAPACITY=GPU_CAPACITY)
        print("Calculating verified scores on the adversarial test points:\n")
        scores_verif_adv_test = get_scores(model, lb_tensor_adv, indices, n_smooth, sigma_model, num_of_classes, scores_list, verif=True, device=device, GPU_CAPACITY=GPU_CAPACITY)

    test_ratio = ((n_test/2))/new_n_test
    idx1, idx2 = train_test_split(indices, test_size=test_ratio)
    print(f"Size of [holdout, calibration, test] split is [{indices_valid.shape[0]}, {idx1.shape[0]}, {idx2.shape[0]}]")
    
    thresholds_base = np.zeros((len(scores_list), 3))
    thresholds_verif = np.zeros((len(scores_list), 3))
    thresholds = np.zeros((len(scores_list), 3))
    bounds = np.zeros((len(scores_list), 2))

    # calibrate base model with the desired scores and get the thresholds
    for p in range(len(scores_list)):
        thresholds_base[p], _ = calibration(scores_simple=scores_simple_clean_test[p, idx1, y_test[idx1]], alpha=alpha, correction=correction, base=True)
        thresholds_verif[p], _ = calibration(scores_simple=scores_verif_clean_test[p, idx1, y_test[idx1]], alpha=alpha, correction=correction, base=True)
        thresholds[p], bounds[p] = calibration(smoothed_scores=smoothed_scores_clean_test[p, idx1, y_test[idx1]], alpha=alpha - 2*0.001, correction=correction, base=False)

    # put bounds in array of bounds
    for p in range(len(scores_list)):
        quantiles[p, 0, experiment] = bounds[p, 0]
        quantiles[p, 1, experiment] = bounds[p, 1]

    # generate prediction sets on the clean test set for base model (vanilla CP)
    predicted_clean_sets_base = prediction(scores_simple=scores_simple_clean_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds_base, base=True)
    
    # generate verified sets on the clean test set
    predicted_clean_sets_verif = prediction(ub=ub_tensor[indices_test[idx2]], num_of_scores=len(scores_list), thresholds=thresholds_base, base=False, verif=True, scores_list=scores_list)
    # generate pre-verified sets on the clean test set
    predicted_clean_sets_verif_pre = prediction(scores_simple=scores_simple_clean_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds_verif, base=True)

    # generate rscp+ prediction sets on the clean test set
    predicted_clean_sets = prediction(smoothed_scores=smoothed_scores_clean_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds, correction=correction, bound_hoef=bound_hoef, bounds_bern=Bern_bound_clean, base=False)

    # generate prediction sets on the adversarial test set for base model
    predicted_adv_sets_base = prediction(scores_simple=scores_simple_adv_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds_base, base=True)

    # generate verified sets on the adversarial test set
    predicted_adv_sets_verif = prediction(ub=ub_tensor_adv[indices_test[idx2]], num_of_scores=len(scores_list), thresholds=thresholds_base, base=False, verif=True, scores_list=scores_list)
    # generate verified sets on the adversarial test set
    predicted_adv_sets_verif_pre = prediction(scores_simple=scores_simple_clean_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds_verif, base=True)
    
    # generate rscp+ sets on the adversarial test set
    predicted_adv_sets = prediction(smoothed_scores=smoothed_scores_adv_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds, correction=correction, bound_hoef=bound_hoef, bounds_bern=Bern_bound_adv, base=False)
    # arrange results on clean test set in dataframe

    for p in range(len(scores_list)):
        score_name = calibration_scores[p]
        methods_list = [score_name + '_vanilla', score_name + '_verif_pre', score_name + '_RSCP+']
        predicted_clean_sets[p].insert(0, predicted_clean_sets_verif_pre[p])
        if post_cal:
            methods_list = [score_name + '_vanilla', score_name + '_verif', score_name + '_verif_pre', score_name + '_RSCP+']
            predicted_clean_sets[p].insert(0, predicted_clean_sets_verif[p])
        predicted_clean_sets[p].insert(0, predicted_clean_sets_base[p])
        predicted_adv_sets[p].insert(0, predicted_adv_sets_verif_pre[p])
        if post_cal:
            predicted_adv_sets[p].insert(0, predicted_adv_sets_verif[p])
        predicted_adv_sets[p].insert(0, predicted_adv_sets_base[p])
       
        for r, method in enumerate(methods_list):
            res = evaluate_predictions(predicted_clean_sets[p][r], None, y_test[idx2].numpy(), num_of_classes=num_of_classes)
            res['Method'] = methods_list[r]
            res['noise_L2_norm'] = 0
            res['Black box'] = ''
            # Add results to the list
            results = results.append(res)

    # arrange results on adversarial test set in dataframe
    for p in range(len(scores_list)):
        score_name = calibration_scores[p]
        methods_list = [score_name + '_vanilla', score_name + '_verif_pre', score_name + '_RSCP+']
        if post_cal:
            methods_list = [score_name + '_vanilla', score_name + '_verif', score_name + '_verif_pre', score_name + '_RSCP+']
        for r, method in enumerate(methods_list):
            res = evaluate_predictions(predicted_adv_sets[p][r], None, y_test[idx2].numpy(), num_of_classes=num_of_classes)
            res['Method'] = methods_list[r]
            res['noise_L2_norm'] = epsilon
            res['Black box'] = ''
            # Add results to the list
            results = results.append(res)

    # clean memory
    del idx1, idx2, predicted_clean_sets, predicted_clean_sets_base, predicted_clean_sets_verif_pre, predicted_adv_sets, predicted_adv_sets_base, predicted_adv_sets_verif_pre, bounds, thresholds, thresholds_base
    if post_cal:
        del predicted_clean_sets_verif, predicted_adv_sets_verif
    gc.collect()


add_string = ""

add_string = add_string+str(n_test)+dataset+"_"+str(epsilon)
if save_results:
    # save results
    print("Saving results in: " + str(res_directory))
    results.to_csv(res_directory + "/results"+add_string+".csv")
    with open(res_directory + "/quantiles_bounds"+add_string+".pickle", 'wb') as f:
        pickle.dump([quantiles], f)
else:
    # save results
    print("Saving results in main Results folder")
    results.to_csv("./Results/results"+add_string+".csv")
# plot results
# plot marginal coverage results
colors_list = sns.color_palette("husl", len(scores_list) * 4)
ax = sns.catplot(x="Black box", y="Coverage",
                 hue="Method", palette=colors_list, col="noise_L2_norm",
                 data=results, kind="box",
                 height=4, aspect=.7)

lower_quantiles_mean = np.zeros(len(scores_list))
upper_quantiles_mean = np.zeros(len(scores_list))
lower_quantiles_std = np.zeros(len(scores_list))
upper_quantiles_std = np.zeros(len(scores_list))

for p in range(len(scores_list)):
    lower_quantiles_mean[p] = np.mean(quantiles[p, 0, :])
    upper_quantiles_mean[p] = np.mean(quantiles[p, 1, :])
    lower_quantiles_std[p] = np.std(quantiles[p, 0, :])
    upper_quantiles_std[p] = np.std(quantiles[p, 1, :])

colors = ['green', 'blue']
for i, graph in enumerate(ax.axes[0]):
    graph.set(xlabel='Classifier', ylabel='Marginal coverage')
    graph.axhline(1 - alpha, ls='--', color="red")
    for p in range(len(scores_list)):
        graph.axhline(upper_quantiles_mean[p], ls='--', color=colors_list[p * 4 + 2])
        graph.axhline(lower_quantiles_mean[p], ls='--', color=colors_list[p * 4 + 2])


if save_results:
    ax.savefig(res_directory + "/Marginal"+add_string+".pdf")
else:
    ax.savefig("./Results/Marginal"+add_string+".pdf")

# plot avg set sizes results
ax = sns.catplot(x="Black box", y="Size",
                 hue="Method", col="noise_L2_norm",
                 data=results, kind="box",
                 height=4, aspect=.7)
for i, graph in enumerate(ax.axes[0]):
    graph.set(xlabel='Classifier', ylabel='Set Size')

if save_results:
    ax.savefig(res_directory + "/Size"+add_string+".pdf")
else:
    ax.savefig("./Results/Size"+add_string+".pdf")

methods_list = []
for p in range(len(scores_list)):
        score_name = calibration_scores[p]
        if post_cal:
            methods_list.append(score_name + '_vanilla')
            methods_list.append(score_name + '_verif')
            methods_list.append(score_name + '_verif_pre')
            methods_list.append(score_name + '_RSCP+')
        else:
            methods_list.append(score_name + '_vanilla')
            methods_list.append(score_name + '_verif_pre')
            methods_list.append(score_name + '_RSCP+')

# Plot size histograms
sizes = results[['Method','noise_L2_norm','Size list']]
set_sizes = sizes.groupby(['Method','noise_L2_norm'])['Size list']
mean_set_sizes = set_sizes.mean().reset_index().set_index(['Method', 'noise_L2_norm'])['Size list'].to_dict()

fig, axes = plt.subplots(nrows=2, ncols=len(methods_list), figsize=(30, 10), sharey=True)
x_values = np.arange(num_of_classes+1)
xlabels = np.arange(num_of_classes+1,10)
# Loop through each method and plot histogram
for i, method in enumerate(sizes['Method'].unique()):
# Plot the bar plot
    sns.barplot(x=x_values, y=mean_set_sizes[(method,0)], width=1, ax=axes[0,i], color=colors_list[i])
    axes[0,i].set_title(method)
    axes[0,i].set_xlabel('Set Size')
    axes[0,i].set_ylabel('Frequency')
    axes[0,i].set_xticks(x_values, labels=xlabels)
    
plt.xlim(0, num_of_classes)

for i, method in enumerate(sizes['Method'].unique()):
    sns.barplot(x=x_values, y=mean_set_sizes[(method,epsilon)], width=1, ax=axes[1,i], color=colors_list[i])
    axes[1,i].set_title(f'{method}+ eps={epsilon} attack')
    axes[1,i].set_xlabel('Set Size')
    axes[1,i].set_ylabel('Frequency')
    axes[1,i].set_xticks(x_values, labels=xlabels)

if save_results:
    plt.savefig(res_directory + "/SizeHist"+add_string+".pdf")
else:
    plt.savefig("./Results/SizeHist"+add_string+".pdf")


if save_results:
    # Save CI, mean and standard deviations of results
    results = results.drop(['Size list'],axis=1)
    clean_results = results[(results['noise_L2_norm']==0)]
    clean_means = clean_results.groupby("Method").mean()
    clean_stds = clean_results.groupby("Method").std()
    clean_means.to_csv(res_directory + "/results_mean_clean.csv")
    clean_stds.to_csv(res_directory + "/results_std_clean.csv")

    noisy_results = results[(results['noise_L2_norm']==epsilon)]
    noisy_means = noisy_results.groupby("Method").mean()
    noisy_stds = noisy_results.groupby("Method").std()
    noisy_means.to_csv(res_directory + "/results_mean_noisy.csv")
    noisy_stds.to_csv(res_directory + "/results_std_noisy.csv")

    merged_df = pd.merge(noisy_means, noisy_stds, on=['Method'])

    print(merged_df)

    z_value = 1.96  # for 95% confidence interval
    sqrtn = np.sqrt(n_experiments)
    merged_df['Coverage_CI'] = merged_df.apply(lambda row: f"{row['Coverage_x']:.3f}±{(z_value/sqrtn) * row['Coverage_y']:.3f}", axis=1)
    merged_df['Size_CI'] = merged_df.apply(lambda row: f"{row['Size_x']:.3f}±{(z_value/sqrtn) * row['Size_y']:.3f}", axis=1)
    print(merged_df)
    # Drop unnecessary columns
    merged_df.drop(['Coverage_x', 'Coverage_y', 'Size_x', 'Size_y', 'noise_L2_norm_x', 'noise_L2_norm_y'], axis=1, inplace=True)

    merged_df.to_csv(res_directory + "/results_CI_noisy.csv")