
# VRCP Classification Experiments

#### Set up the environment (uv):

    uv sync

This creates `.venv/`, installs Python 3.11 (via uv, if not already present),
and installs all dependencies including `auto_LiRPA`, pinned to a specific
commit of https://github.com/Verified-Intelligence/auto_LiRPA (see
`pyproject.toml`'s `[tool.uv.sources]`). No separate `git clone` /
`setup.py install` step is needed, and no patch step is needed either --
the pinned commit does not exhibit the old CIFAR100/TinyImageNet
CROWN-bound pooling bug (verified by `tests/test_smoke.py`).

Run any command below with `uv run <command>`, e.g. `uv run pytest`.

#### Download TinyImageNet dataset:
The last execution of this script takes a while to complete, do not quit the process.

    cd Datasets
    bash tinyimagenet.sh
    cd ..

#### Run the smoke tests (optional, fast, CPU-only):

    uv run pytest tests/test_smoke.py

#### Train models (Optional)

 (The pre-trained models used in our experiments are already provided in the Checkpoints directory):

    uv run python train.py --dataset CIFAR10
    uv run python train.py --dataset CIFAR100
    uv run python train.py --dataset TINYNET


Adjust --batch_size parameter in accordance to GPU VRAM available.

#### Configuration

Experiment parameters live in `config.yaml`. Every CLI flag below overrides
the matching config.yaml value for that run, so config.yaml only needs to
hold your "default" settings -- the sweeps below work exactly as shown.

#### Table 1:

    uv run python run_experiment.py -a 0.1 -d 0.03 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    uv run python run_experiment.py -a 0.1 -d 0.02 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR100 --arc CNN --My_model
    uv run python run_experiment.py -a 0.1 -d 0.02 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset TINYNET --arc CNN --My_model

#### Table 2:

    uv run python run_experiment.py -a 0.1 -d 0.001 -s 50 -r 2 --n_s 1 -n inf --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model

#### Figure 1:

    uv run python run_experiment.py -a 0.1 -d 0.02 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model


#### Figure 2a:

    uv run python run_experiment.py -a 0.1 -d 0.03 -s 50 -r 2 --n_s 256 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    uv run python run_experiment.py -a 0.1 -d 0.03 -s 50 -r 2 --n_s 512 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    uv run python run_experiment.py -a 0.1 -d 0.03 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    uv run python run_experiment.py -a 0.1 -d 0.03 -s 50 -r 2 --n_s 2048 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    uv run python run_experiment.py -a 0.1 -d 0.03 -s 50 -r 2 --n_s 4096 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model

Then plot with: `uv run python Figure_Scripts/MCsamplePlot.py --epsilon 0.03`

#### Figure 2b:

    uv run python run_experiment.py -a 0.1 -d 0.005 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    uv run python run_experiment.py -a 0.1 -d 0.01 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    uv run python run_experiment.py -a 0.1 -d 0.015 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    uv run python run_experiment.py -a 0.1 -d 0.02 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    uv run python run_experiment.py -a 0.1 -d 0.025 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    uv run python run_experiment.py -a 0.1 -d 0.03 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    uv run python run_experiment.py -a 0.1 -d 0.035 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    uv run python run_experiment.py -a 0.1 -d 0.04 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    uv run python run_experiment.py -a 0.1 -d 0.045 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    uv run python run_experiment.py -a 0.1 -d 0.05 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model

Then plot with: `uv run python Figure_Scripts/epsilonPlot.py`
