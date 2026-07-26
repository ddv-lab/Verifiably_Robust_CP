import os
import traceback

from strictyaml import load, YAMLError

import vrcp_cls_config
from .schema import SCHEMA

ROOT_DIR = os.path.dirname(vrcp_cls_config.__file__)
CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIR, "..", "config.yaml"))

CFG_EXPERIMENT = "experiment"
CFG_PATH = "path"
CFG_RAND = "rand"

_DATASET_NUM_CLASSES = {"CIFAR10": 10, "CIFAR100": 100, "TINYNET": 200}
_DATASET_IMAGE_DIM = {"CIFAR10": 32, "CIFAR100": 32, "TINYNET": 64}


# Using a class here purely to do some sketchy stateful stuff with Python
# (mirrors the pattern used in VRCP_Regression/vrcp_reg/vrcp_reg_config)
class Config:

    CONFIG_LOADED = False
    YAML = None
    DATA = None

    def load(cli_overrides=None):
        Config.YAML = Config._load_config()
        Config.DATA = Config.YAML.data
        if Config.YAML is not None:
            Config._apply_overrides(cli_overrides or {})
            Config._load_vars()

    def _load_config():
        try:
            with open(CONFIG_PATH, "r") as file:
                config = load(file.read(), SCHEMA)
                Config.CONFIG_LOADED = True
            return config
        except FileNotFoundError:
            print("The config file is missing. Please copy the default config from the repo.")
            exit(1)
        except YAMLError:
            print(traceback.format_exc())
            print("Your config file is invalid. Please check the stack trace.")
            exit(1)

    # CLI flags override YAML values so sweeps (e.g. `--n_s 512`) keep working
    # without editing config.yaml between runs.
    def _apply_overrides(cli_overrides):
        experiment = Config.DATA[CFG_EXPERIMENT]
        for key, value in cli_overrides.items():
            if value is None:
                continue
            if key not in experiment:
                continue
            experiment[key] = value

    def _load_vars():
        try:
            experiment = Config.DATA[CFG_EXPERIMENT]
            path = Config.DATA[CFG_PATH]

            dataset = experiment["dataset"]
            if dataset not in _DATASET_NUM_CLASSES:
                raise ValueError(f"Dataset can only be one of {sorted(_DATASET_NUM_CLASSES)}, got {dataset!r}.")
            if not (0 <= experiment["alpha"] <= 1):
                raise ValueError("Nominal level (alpha) must be between 0 and 1.")
            if experiment["delta"] < 0:
                raise ValueError("L-p bound of noise (delta) must be non-negative.")
            if experiment["ratio"] < 0:
                raise ValueError("Ratio between sigma and delta must be non-negative.")
            if experiment["norm"] not in ("2", "inf"):
                raise ValueError("Norm can only be '2' or 'inf'.")
            if experiment["arc"] != "CNN":
                raise ValueError("Architecture can only be 'CNN'.")
            if not isinstance(experiment["splits"], int) or experiment["splits"] < 1:
                raise ValueError("Number of splits must be a positive integer.")
            n_s = experiment["n_s"]
            if n_s & (n_s - 1):
                raise ValueError("n_s must be a power of 2.")
            batch_size = experiment["batch_size"]
            if batch_size & (batch_size - 1):
                raise ValueError("batch_size must be a power of 2.")
            if batch_size < n_s:
                raise ValueError("batch_size must be >= n_s.")

            # CIFAR100 only ships "My_model"-trained checkpoints.
            if dataset == "CIFAR100":
                experiment["my_model"] = True

            experiment["num_of_classes"] = _DATASET_NUM_CLASSES[dataset]
            experiment["image_dim"] = _DATASET_IMAGE_DIM[dataset]
            experiment["sigma_model"] = 0
            experiment["sigma_smooth"] = experiment["ratio"] * experiment["delta"]
            experiment["correction"] = (
                float(experiment["delta"]) / float(experiment["sigma_smooth"])
                if experiment["sigma_smooth"] != 0
                else 10000.0
            )
            experiment["robust"] = bool(experiment["my_model"])
            experiment["save_results"] = experiment["splits"] == 50 and experiment["n_test"] == 10000

            root = path["root"]
            path["datasets"] = os.path.join(root, path["datasets"])
            path["checkpoints"] = os.path.join(root, path["checkpoints"])
            path["adversarial_examples"] = os.path.join(root, path["adversarial_examples"])
            path["bounds"] = os.path.join(root, path["bounds"])
            path["results"] = os.path.join(root, path["results"])
        except Exception:
            print("Something went wrong loading the vars...")
            print(traceback.format_exc())
            exit(1)
