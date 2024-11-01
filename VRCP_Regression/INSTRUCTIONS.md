# VRCP Regression Experiments

If you have already set up the conda environment for the classification experiments, please skip step 1.

#### Step 1: Set up conda environment:

    conda env create -f environment.yml
    conda activate VRCP
    git clone https://github.com/KaidiXu/auto_LiRPA
    cd auto_LiRPA
    python setup.py install
    cd ..

#### Step 2: Install regression specific libraries

    conda activate VRCP
    cd vrcp_reg
    pip install -r requirements.txt
    pip install -e .

#### Step 3: Install third-party libraries for MPE

    pip install -e lib/PettingZoo
    pip install -e lib/tianshou

#### Step 4: Configure the root data path

`config.yaml`
    ...
    path:
      root: [CHANGE ME]
      env_name: 'spread'
      dataset_name: 'example'
    ...

Change the root path to the absolute path of the `data` folder within this folder.

Setting the environment name to one of adversary, spread or push will set up the correct environment accordingly. We provide trained RL policies with `dataset_name = example` however you may train your own too by running: `python mpe_helper --mode train`

#### Step 5: Configure the parameters accordingly for each environment

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

#### Step 6: Generate MPE datasets

    python mpe_helper --mode generate

You can tweak the data generation parameters in the `sim` section of the config file.

#### Step 7: Run the VRCP code

    python vrcp_reg/vrcp.py

This will run all experiments in Table 3. If you have trained your own polices/models, your results may vary. We provide well-trained models for each of the environments.
    