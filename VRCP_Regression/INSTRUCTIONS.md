# VRCP Regression Experiments

#### Step 1: Set up the environment (uv)

    cd vrcp_reg
    uv sync

This creates `.venv/`, installs Python 3.11 (via uv, if not already
present), and installs all dependencies -- including `auto_LiRPA`, pinned
to the same commit as VRCP_Classification (see `pyproject.toml`'s
`[tool.uv.sources]`), and `pettingzoo`/`tianshou`/`gymnasium` at versions
confirmed compatible with the modernized numpy 2.x / Python 3.11 stack. No
`conda`, no separate `git clone` / `setup.py install` step, and no vendored
`lib/PettingZoo` or `lib/tianshou` copies (both were confirmed to be
unmodified upstream code at those versions, so plain pinned dependencies
replace them).

Run any command below with `uv run <command>`, e.g. `uv run pytest`.

#### Step 2: Run the smoke tests (optional, fast, CPU-only)

    uv run pytest tests/test_smoke.py

#### Step 3: Configure the root data path

`config.yaml`
    ...
    path:
      root: [CHANGE ME]
      env_name: 'spread'
      dataset_name: 'example'
    ...

Change the root path to the absolute path of the `data` folder within this folder.

Setting the environment name to one of adversary, spread or push will set up the correct environment accordingly. We provide trained RL policies with `dataset_name = example` however you may train your own too by running: `uv run python -m mpe_helper --mode train`

#### Step 4: Configure the parameters accordingly for each environment

The key parameters to change are:
    
    ...
    env:
      n_agents: X
      n_adversaries: Y
      n_landmarks: Z
    ...

For adversary, use X = 2, Y = 1, Z = 2.

For spread, use X = 3, Y = 0, Z = 3.

For push, use X = 1, Y = 1, Z = 1.

#### Step 5: Generate MPE datasets

    uv run python -m mpe_helper --mode generate

You can tweak the data generation parameters in the `sim` section of the config file.

#### Step 6: Run the VRCP code

    uv run python vrcp_reg/vrcp.py

This will run all experiments in Table 3. If you have trained your own polices/models, your results may vary. We provide well-trained models for each of the environments.
