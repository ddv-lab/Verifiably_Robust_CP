# Verifiably Robust Conformal Prediction (NeurIPS 2024)

Linus Jeary\*, Tom Kuipers\*, Mehran Hoseini, and Nicola Paoletti.

Department of Informatics, King's College London, UK


## Abstract

Conformal Prediction (CP) is a popular uncertainty quantification method that provides distribution-free, statistically valid prediction sets, assuming that training and test data are exchangeable. In such a case, CP’s prediction sets are guaranteed to cover the (unknown) true test output with a user-specified probability. Nevertheless, this guarantee is violated when the data is subjected to adversarial attacks, which often result in a significant loss of coverage. Recently, several approaches have been put forward to recover CP guarantees in this setting. These approaches leverage variations of randomised smoothing to produce conservative sets which account for the effect of the adversarial perturbations. They are, however, limited in that they only support ℓ<sub>2</sub>-bounded perturbations and classification tasks. This paper introduces VRCP (Verifiably Robust Conformal Prediction), a new framework that leverages recent neural network verification methods to recover coverage guarantees under adversarial attacks. Our VRCP method is the first to support perturbations bounded by arbitrary norms including ℓ<sub>1</sub>, ℓ<sub>2</sub>, and ℓ<sub>∞</sub>, as well as regression tasks. We evaluate and compare our approach on image classification tasks (CIFAR10, CIFAR100, and TinyImageNet) and regression tasks for deep reinforcement learning environments. In every case, VRCP achieves above nominal coverage and yields significantly more efficient and informative prediction regions than the SotA.

## How to use

The code (currently) exists as two separate codebases which directly reflect the two main sets of experiments and their case studies that exist in the paper.

To run either the classification or regression experiments, the full steps and setup instructions can be found in the corresponding INSTRUCTIONS.md files in each folder. 

## Future Steps

The code currently exists so as to replicate the experiments in the paper, however work has already begun on wrapping up both classification and regression codebases into a single, modular and configurable library for general use.

It is anticipated that this work will be completed by Q2 2025. Please continue to check back for updates.

## Contact

If you have any questions regarding the work or the code in this repository, please contact Tom Kuipers in the first instance (email: first.lastname (at) kcl.ac.uk).

## Acknowledgements

This work is supported by the “REXASI-PRO” H-EU project, call HORIZON-CL4-2021-HUMAN-01-01, Grant agreement ID: 101070028.
