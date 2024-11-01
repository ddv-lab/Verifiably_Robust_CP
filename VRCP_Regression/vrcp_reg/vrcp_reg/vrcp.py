from cqr import helper
from vrcp_reg.utils import enumerate_dataset, apply_perturb, load_datasets
from vrcp_reg_config import *
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import copy
import gc
from nonconformist.cp import IcpRegressor
from nonconformist.nc import BoundedQuantileRegErrFunc, RegressorNc, QuantileRegErrFunc
import numpy as np
import os
import random
import torch
from tqdm import tqdm


class VRCP:

    def __init__(self, pretrained=False, n_splits=50):
        self.n_splits = n_splits
        self.seed = self.set_seeds()
        self.split_coverage = np.zeros(shape=(self.n_splits, 4))
        self.split_interval = np.zeros_like(self.split_coverage)
        self.quantile_estimator = helper.AllQNet_RegressorAdapter(model=None,
                                                                fit_params=None,
                                                                in_shape=CONFIG[CFG_DATA]['state_dim'],
                                                                hidden_size=CONFIG[CFG_CQR]['model']['hidden_size'],
                                                                quantiles=CONFIG[CFG_CQR]['quantiles_net'],
                                                                learn_func=torch.optim.Adam,
                                                                epochs=CONFIG[CFG_CQR]['model']['epochs'],
                                                                batch_size=CONFIG[CFG_CQR]['model']['batch_size'],
                                                                dropout=CONFIG[CFG_CQR]['model']['dropout'],
                                                                lr=CONFIG[CFG_CQR]['model']['lr'],
                                                                wd=CONFIG[CFG_CQR]['model']['decay'],
                                                                test_ratio=CONFIG[CFG_CQR]['model']['cv_ratio'],
                                                                random_state=CONFIG[CFG_RAND]['cqr']['cross_val_seed'],
                                                                use_rearrangement=False)
        if pretrained:
            self.quantile_estimator.model.load_state_dict(torch.load(os.path.join(CONFIG[CFG_PATH]["model"], f"qr_model_{CONFIG[CFG_PATH]['dataset_name']}.pth")))
        self.x_full, self.y_full = enumerate_dataset(single_traj=False)
        self.x_single, self.y_single = enumerate_dataset(single_traj=True)

    def set_seeds(self):
        # Set seeds for reproducibility
        seed = CONFIG[CFG_RAND]['default_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    def report_metrics(self, arrays, texts):
        z_val = 1.96
        sqrt = np.sqrt(self.n_splits)
        for idx, array in enumerate(arrays):
            std = np.std(array)
            interval = std * (z_val/sqrt)
            print(f"AVERAGE {texts[idx]}: {np.mean(array)}±{interval}")

    def train_qr(self):
        x_train, y_train, *_ = load_datasets(self.x_full, self.y_full, self.x_single, self.y_single)
        print("Training model...")
        self.quantile_estimator.fit(x_train, y_train)
        torch.save(self.quantile_estimator.model.state_dict(), os.path.join(CONFIG[CFG_PATH]["model"], f"qr_model_{CONFIG[CFG_PATH]['dataset_name']}.pth"))
        print("Trained and saved model!")

    def compute_bounds(self, data, critical_val=None):
        dummy = torch.zeros(size=data.shape)
        model_copy = copy.deepcopy(self.quantile_estimator.model)

        lirpa_model = BoundedModule(model_copy, dummy)
        lirpa_model.to("cuda")
        lirpa_model.eval()
        ptb = PerturbationLpNorm(norm=np.inf, eps=CONFIG[CFG_DATA]["adv_prtb_eps"])

        all_bounds = np.zeros(shape=(data.shape[0], 2))
        for i in tqdm(range(data.shape[0]), desc="CROWN Bounds"):
            test_input = torch.unsqueeze(torch.from_numpy(data[i]), dim=0).to("cuda")
            my_input = BoundedTensor(test_input, ptb)
            bounds = lirpa_model.compute_bounds(x=(my_input,), method="CROWN")
            # Set bounds as worst case bounds
            # Since quantiles don't come back in order, need to capture and restore ordering
            model_output = torch.squeeze(lirpa_model(test_input), dim=0)
            ordering = model_output.argsort(dim=0).cpu()
            if critical_val:
                worst_lb = bounds[0][:, ordering[0]][0] - critical_val
                worst_ub = bounds[1][:, ordering[1]][0] + critical_val
            else:
                worst_lb = bounds[1][:, ordering[0]][0]
                worst_ub = bounds[0][:, ordering[1]][0]
            new_bounds = np.array([worst_lb.cpu(), worst_ub.cpu()])
            # Store bounds in the correct order
            all_bounds[i] = new_bounds[ordering]
        return all_bounds

    def run_splits(self, eval_vanilla=False):
        for iteration in range(self.n_splits):
            print(f"Launching split iteration: {iteration}")
            x_train, y_train, x_test, y_test, x_cal, y_cal = load_datasets(self.x_full, self.y_full, self.x_single, self.y_single, self.seed + iteration)
            
            print(f"Generating adversarially perturbed data")
            adv_data = apply_perturb(self.quantile_estimator, x_test, y_test)

            # Create CQR Nonconformist wrapper on top of QR model
            nc = RegressorNc(self.quantile_estimator, QuantileRegErrFunc())

            # Evaluate nominal vanilla CP coverage and interval lengths
            if eval_vanilla:
                y_lower, y_upper = helper.run_icp(nc, x_train, y_train, x_cal, y_cal, x_test, CONFIG[CFG_CP]['alpha'], train=False)
                self.split_coverage[iteration, 0], self.split_interval[iteration, 0] = helper.compute_coverage(y_test, y_lower, y_upper, CONFIG[CFG_CP]['alpha'], "VANILLA")
            
            # Evaluate miscoverage with adversarial data
            y_lower, y_upper = helper.run_icp(nc, x_train, y_train, x_cal, y_cal, adv_data, CONFIG[CFG_CP]['alpha'])
            self.split_coverage[iteration, 1], self.split_interval[iteration, 1] = helper.compute_coverage(y_test, y_lower, y_upper, CONFIG[CFG_CP]['alpha'], "ADVERSARIAL")


            # START VRCP--C
            # Get bounds for calibration method
            calib_bounds = self.compute_bounds(x_cal)
            
            # Re-create regressor wrapper with new bound-dependent NCF
            bounded_nc = RegressorNc(self.quantile_estimator, BoundedQuantileRegErrFunc(calib_bounds))

            # Evaluate new coverage on adversarial data using CROWN bounds
            y_lower, y_upper = helper.run_icp(bounded_nc, x_train, y_train, x_cal, y_cal, adv_data, CONFIG[CFG_CP]['alpha'], train=False)
            self.split_coverage[iteration, 2], self.split_interval[iteration, 2] = helper.compute_coverage(y_test, y_lower, y_upper, CONFIG[CFG_CP]['alpha'], "VRCP--C")
            # END VRCP--C


            # START VRCP--I
            inference_nc = RegressorNc(self.quantile_estimator, QuantileRegErrFunc())

            # Calibrate regressor on clean calibration data
            icp = IcpRegressor(inference_nc)
            icp.calibrate(x_cal, y_cal)
            cal_scores = np.sort(icp.cal_scores[0])
            index = int(np.ceil((1 - CONFIG[CFG_CP]['alpha']) * (cal_scores.shape[0] + 1))) - 1
            index = min(max(index, 0), cal_scores.shape[0] - 1)
            
            # Extract critical value of score distribution
            critical_val = cal_scores[index]

            # Compute bounds over test dataset
            test_bounds = self.compute_bounds(x_test, critical_val=critical_val)
            self.split_coverage[iteration, 3], self.split_interval[iteration, 3] = helper.compute_coverage(y_test, test_bounds[:, 0], test_bounds[:, 1], CONFIG[CFG_CP]['alpha'], "VRCP--I")
            # END VRCP--I
            gc.collect()

        if eval_vanilla:
            self.report_metrics([self.split_coverage[:, 0], self.split_interval[:, 0]], ["CLEAN COVERAGE", "CLEAN INTERVAL"])

        self.report_metrics([self.split_coverage[:, 1], self.split_interval[:, 1]], ["ADVERSARIAL COVERAGE", "ADVERSARIAL INTERVAL"])
        self.report_metrics([self.split_coverage[:, 2], self.split_interval[:, 2]], ["VRCP--C COVERAGE", "VRCP--C INTERVAL"])
        self.report_metrics([self.split_coverage[:, 3], self.split_interval[:, 3]], ["VRCP--I COVERAGE", "VRCP--I INTERVAL"])
        
if __name__ == "__main__":
    vrcp = VRCP(pretrained=True, n_splits=1)
    vrcp.run_splits()

