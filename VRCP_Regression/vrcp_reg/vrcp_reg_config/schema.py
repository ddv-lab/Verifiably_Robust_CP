from datetime import datetime
from strictyaml import Map, Any, EmptyNone, Datetime, Bool, Str, Int, Float, Seq, Optional


# TODO Copy in defaults from old args
SCHEMA = Map({
    "global": Map({
        "device": Str(),
        "n_splits": Int()
        }),
    "cp": Map({
        "alpha": Float(),
    }),
    "cqr": Map({
        "quantiles": Seq(Float()),
        "quantiles_net": Seq(Float()),
        "coverage_factor": Float(),
        "model": Map({
            "epochs": Int(),
            "lr": Float(),
            "batch_size": Int(),
            "hidden_size": Int(),
            "dropout": Float(),
            "decay": Float(),
            "cv_ratio": Float()
        })
    }),
    "data": Map({
        "n_states": Int(),
        "n_suffixes": Int(),
        "prefix_len": Int(),
        "suffix_len": Int(),
        "n_train": Int(),
        "n_test": Int(),
        "n_cal": Int(),
        "adv_prtb_eps": Float(),
        Optional("trajectory_len", default=None, drop_if_none=False): (EmptyNone() | Int()),
        Optional("eps_target", default=None, drop_if_none=False): (EmptyNone() | Seq(Float())),
        Optional("target_idx", default=None, drop_if_none=False): (EmptyNone() | Int()),
        Optional("target_eps_val", default=None, drop_if_none=False): (EmptyNone() | Float()),
        Optional("world_ids", default=None, drop_if_none=False): (EmptyNone() | Any()),
        Optional("state_dim", default=None, drop_if_none=False): (EmptyNone() | Int()),
    }),
    "env": Map({
        Optional("n_total_agents", default=None, drop_if_none=False): (EmptyNone() | Int()),
        "n_agents": Int(),
        "n_adversaries": Int(),
        "n_landmarks": Int(),
        "load_policies": Bool()
    }),
    "path": Map({
        "root": Str(),
        Optional("env_name", default="example"): Str(),
        Optional("dataset_name", default=datetime.now().strftime("%d-%m-%Y_%H:%M:%S")): Str(),
        Optional("data", default=None, drop_if_none=False): (EmptyNone() | Str()),
        Optional("policy", default=None, drop_if_none=False): (EmptyNone() | Str()),
        Optional("model", default=None, drop_if_none=False): (EmptyNone() | Str()),
        Optional("log", default=None, drop_if_none=False): (EmptyNone() | Str())
    }),
    "rand": Map({
        "default_seed": Int(),
        "cqr": Map({
            "cross_val_seed": Int(),
        }),
        "sim": Map({
            "noise_seed": Int(),
            "noisy": Bool(),
            "noise_var": Float(),
            "noise_func": Str(),
            "world_seed": Int(),
            "world_func": Str()
        })
    }),
    "sim": Map({
        "buffer_size": Int(),
        Optional("render_mode", default=None, drop_if_none=False): (EmptyNone() | Str()),
        "generator": Map({
            "parallel": Bool(),
            "n_threads": Int()
        }),
        "policy": Map({
            "eps_test": Float(),
            "eps_train": Float(),
            "lr": Float(),
            "gamma": Float(),
            "n_step": Int(),
            "target_update_freq": Int(),
            "epoch": Int(),
            "step_per_epoch": Int(),
            "step_per_collect": Int(),
            "update_per_step": Float(),
            "batch_size": Int(),
            Optional("hidden_sizes", default=[128, 128, 128, 128], drop_if_none=True): Seq(Int()),
            "training_num": Int(),
            "test_num": Int(),
            "render": Float()
        })
    }),
})
