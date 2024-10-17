
# VRCP Classification Experiments

#### Set up virtual environment:

    python3.9 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    git clone https://github.com/KaidiXu/auto_LiRPA
    cd auto_LiRPA
    python setup.py install
    cd ..

#### Download TinyImageNet dataset:
The last execution of this script takes a while to complete, do not quit the process.

    cd Datasets
    bash tinyimagenet.sh
    cd ..

#### Patch the auto_LirPA library
There is one line of code that causes issues when computing bounds for the CIFAR100 and TinyImageNet models.

    bash patch.sh


#### Train models (Optional)

 (The pre-trained models used in our experiments are already provided in the Checkpoints directory):

    python train.py --dataset CIFAR10
    python train.py --dataset CIFAR100
    python train.py --dataset TINYNET


Adjust --batch_size parameter in accordance to GPU VRAM available.

#### Table 1:

    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.03 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.02 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR100 --arc CNN --My_model
    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.02 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset TINYNET --arc CNN --My_model

#### Table 2:

    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.001 -s 50 -r 2 --n_s 1 -n inf --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model

#### Figure 1:

    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.02 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model


#### Figure 2a:

    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.03 -s 50 -r 2 --n_s 256 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.03 -s 50 -r 2 --n_s 512 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.03 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.03 -s 50 -r 2 --n_s 2048 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.03 -s 50 -r 2 --n_s 4096 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model

#### Figure 2b:

    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.005 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.01 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.015 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.02 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.025 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.03 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.035 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.04 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.045 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model
    python ./VRCP/VRCP_exp.py -a 0.1 -d 0.05 -s 50 -r 2 --n_s 1024 --batch_size 8192 --dataset CIFAR10 --arc CNN --My_model