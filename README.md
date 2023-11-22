# A machine learning enhanced Generalized Langevin Equation for Transient Anomalous Diffusion in Polymer Dynamics (NeurIPS ML for Physical Sciences Workshop 2023)


This project contains the source code needed to reproduce the results shown in the workshop paper submitted at the ML for Physical Sciences Workshop at the NeurIPS conference 2023 ([ML4PS workshop link](https://ml4physicalsciences.github.io/2023/)). \
The ML-GLE framework is able to emulate and extrapolate the long-term effective dynamics of slow diffusive polymers inside a polymer melt medium. It reproduces all the relevant statistical features and further enhances the Generalized Langevin Equation solution pertaining to transient anomlaous diffusion, allowing for simulating several orders of magnitute over the training dataset trajectory length, discovering therefore the diffusive properties, without performing a full-size simulation. 

![training_scheme](https://github.com/Gian-Michele-Cherchi/ml-gle/assets/43932730/256f5633-63a1-4c9b-8e2d-097f5b797982)

This code was developed using ```python 3.9.2 ``` on ```CentOS Linux Server```. The data preprocessing, training and generation was done using one ```NVIDIA A100``` GPU,  ```Torch 1.11``` and  ```CUDA 11.3```, while dataset CG simulation was performed exploiting the ```CPU cores```. 

## Poetry Env. Install
```Poetry 1.7.1``` is used as a dependency management. All package requirements are contained in the ```pyproject.toml``` file, Python version included. 
Install poetry following the instructions at https://python-poetry.org/docs/master/#installing-with-pipx.

Once Poetry is installed, install the poetry environment in the project folder. It should be fast. 
```
poetry install
```
Activate the environment with:
```
poetry shell
```

## Environment Variables


### DATASET

The training dataset is available at at https://huggingface.co/datasets/gian-michele/meltBR
