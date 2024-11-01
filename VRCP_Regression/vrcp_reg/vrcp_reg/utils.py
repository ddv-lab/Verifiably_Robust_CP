from vrcp_reg_config import *
from mpe_helper.data import Datasets, SingleLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import torch
from tqdm import tqdm


def enumerate_dataset(single_traj=False):
    loader = SingleLoader(Datasets.COMBINED, single_traj=single_traj, seeds=CONFIG[CFG_DATA]["world_ids"], shuffle=[False])
    data = loader.get_datasets()
    X = np.zeros(shape=(len(data), CONFIG[CFG_DATA]['state_dim']), dtype=np.float32)
    y = np.zeros(shape=(len(data), 1), dtype=np.float32)
    for i in range(len(data)):
        X[i], y[i] = data[i][0], data[i][1]
    return X, y


def load_datasets(x_full, y_full, x_single, y_single, seed):
    n_seeds = len(CONFIG[CFG_DATA]["world_ids"])
    # Always reserve the first portion n data points for training and ensure
    # they never end up in the random splits
    train_idx = np.arange(CONFIG[CFG_DATA]["n_train"])
    other_idx = shuffle(np.arange(CONFIG[CFG_DATA]["n_train"], n_seeds), random_state=seed)
    if CONFIG[CFG_DATA]["n_test"] > CONFIG[CFG_DATA]["n_cal"]:
        test_idx, cal_idx = other_idx[:CONFIG[CFG_DATA]["n_test"]], other_idx[CONFIG[CFG_DATA]["n_test"]:][:CONFIG[CFG_DATA]["n_cal"]]
    else:
        cal_idx, test_idx = other_idx[:CONFIG[CFG_DATA]["n_cal"]], other_idx[CONFIG[CFG_DATA]["n_cal"]:][:CONFIG[CFG_DATA]["n_test"]]

    x_train, y_train = x_full[train_idx], y_full[train_idx]
    x_test, y_test = x_single[test_idx], y_single[test_idx]
    x_cal, y_cal = x_single[cal_idx], y_single[cal_idx]

    # Scale all data to make it easier to train QR
    scalerX = MinMaxScaler()
    scalerX = scalerX.fit(x_full)
    x_train = scalerX.transform(x_train)
    x_test = scalerX.transform(x_test)
    x_cal = scalerX.transform(x_cal)

    scalerY = MinMaxScaler()
    scalerY = scalerY.fit(y_full)
    y_train = np.squeeze(scalerY.transform(y_train))
    y_test = np.squeeze(scalerY.transform(y_test))
    y_cal = np.squeeze(scalerY.transform(y_cal))

    x_train, y_train = shuffle(x_train, y_train, random_state=seed)
    x_test, y_test = shuffle(x_test, y_test, random_state=seed)
    x_cal, y_cal = shuffle(x_cal, y_cal, random_state=seed)
    return (x_train, y_train, x_test, y_test, x_cal, y_cal)


def fgsm_attack(x_test, data_grad):
    sign_data_grad = data_grad.sign()
    adv_x_test = x_test + CONFIG[CFG_DATA]["adv_prtb_eps"] * sign_data_grad
    adv_x_test = torch.clamp(adv_x_test, 0, 1)
    return adv_x_test


def apply_perturb(model, x_test, y_test):
    new_data = np.zeros_like(x_test)
    for i in tqdm(range(new_data.shape[0]), desc="Adv. Perturbation"):
        x_clean = torch.unsqueeze(torch.from_numpy(x_test[i]).to(device='cuda'), dim=0)
        x_clean.requires_grad = True
        y_clean = torch.FloatTensor([y_test[i]]).to(device='cuda')
        # Need to dig deep to get the model here due to CQR layers of abstraction
        model_out = model.model.base_model(x_clean)
        loss = model.loss_func(model_out, y_clean)
        model.model.base_model.zero_grad()
        loss.backward()
        data_grad = x_clean.grad.data
        x_adv = fgsm_attack(x_clean, data_grad)
        x_adv_np = x_adv.detach().cpu().numpy()
        new_data[i] = x_adv_np
    model.model.base_model.zero_grad()
    return new_data