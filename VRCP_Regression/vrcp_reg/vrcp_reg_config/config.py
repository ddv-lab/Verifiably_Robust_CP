from .schema import SCHEMA
import vrcp_reg_config
import numpy as np
import os
from strictyaml import load, YAMLError
import traceback

ROOT_DIR = os.path.dirname(vrcp_reg_config.__file__)
CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIR, '..', 'config.yaml'))

CFG_GLOBAL = "global"
CFG_CP = "cp"
CFG_CQR = "cqr"
CFG_DATA = "data"
CFG_DEBUG = "debug"
CFG_ENV = "env"
CFG_PATH = "path"
CFG_RAND = "rand"
CFG_SIM = "sim"

# Using a class here purely to do some sketchy stateful stuff with Python
class Config:

    CONFIG_LOADED = False
    YAML = None
    DATA = None

    def load():
        Config.YAML = Config._load_config()
        Config.DATA = Config.YAML.data
        # Now load the vars if all was fine
        if Config.YAML is not None:
            Config._load_vars()

    def _load_config():
        config = None
        try:
            with open(CONFIG_PATH, 'r') as file:
                config = load(file.read(), SCHEMA)
                Config.CONFIG_LOADED = True
            return config
        except FileNotFoundError:
            print("The config file is missing. Please copy the default config from the repo.")
            exit(1)
        except YAMLError as e:
            print(traceback.format_exc())
            print("Your config file is invalid. Please check the stack trace.")
            exit(1)

    # Loads in required global runtime variables
    def _load_vars():
        try:
            # Configure dynamic paths
            Config.DATA[CFG_PATH]["root"] = os.path.join(Config.DATA[CFG_PATH]["root"], Config.DATA[CFG_PATH]["env_name"])
            Config.DATA[CFG_PATH]["data"] = os.path.join(Config.DATA[CFG_PATH]["root"], "obs", Config.DATA[CFG_PATH]["dataset_name"])
            Config.DATA[CFG_PATH]["policy"] = os.path.join(Config.DATA[CFG_PATH]["root"], "policies")
            Config.DATA[CFG_PATH]["model"] = os.path.join(Config.DATA[CFG_PATH]["root"], "models")
            Config.DATA[CFG_PATH]["log"] = os.path.join(Config.DATA[CFG_PATH]["root"], "logs")
            # Configure data params
            Config.DATA[CFG_DATA]["trajectory_len"] = Config.DATA[CFG_DATA]["prefix_len"] + Config.DATA[CFG_DATA]["suffix_len"]
            if Config.DATA[CFG_DATA]["eps_target"]:
                Config.DATA[CFG_DATA]["target_eps_val"] = Config.DATA[CFG_DATA]["eps_target"][Config.DATA[CFG_DATA]["target_idx"]]
            # TODO Update number of agents dynamically
            Config.DATA[CFG_ENV]['n_total_agents'] = Config.DATA[CFG_ENV]['n_agents']
            if Config.DATA[CFG_ENV]['n_adversaries']:
                Config.DATA[CFG_ENV]['n_total_agents'] += Config.DATA[CFG_ENV]['n_adversaries']
            Config.DATA[CFG_DATA]['state_dim'] = (Config.DATA[CFG_ENV]['n_total_agents'] * 4) + (Config.DATA[CFG_ENV]['n_landmarks'] * 2)
            # Load array of seeds
            # This is slow but we only load it once, so it should be fine...
            try:
                world_ids = np.array(np.load(os.path.join(str(Config.DATA[CFG_PATH]["data"]), "seeds.npy")))
                Config.DATA[CFG_DATA]["world_ids"] = world_ids
            except FileNotFoundError:
                # This is fine but best to warn user anyway.
                print("Seeds file not found... If you are training or generating data, this is expected.")
            if Config.DATA[CFG_DATA]["n_train"] + Config.DATA[CFG_DATA]["n_test"] + Config.DATA[CFG_DATA]["n_cal"] > Config.DATA[CFG_DATA]["n_states"]:
                raise ValueError("Number of data points across all splits is greater than total number of datapoints!")
        # TODO Properly handle exceptions here
        except Exception as e:
            print("Something went wrong loading the vars...")
            print(traceback.format_exc())
            exit(1)
