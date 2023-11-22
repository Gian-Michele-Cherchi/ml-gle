## ML-GLE: an ML-enhanced Generalized Langevin Equation framework for Transient Anomalous Diffusion in Polymer Dynamics (NeurIPS ML for Physical Sciences Workshop 2023)


This project contains the source code needed to reproduce the results shown in the workshop paper submitted at the ML for Physical Sciences Workshop at the NeurIPS conference 2023 ([ML4PS workshop link](https://ml4physicalsciences.github.io/2023/)). 

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
##### NOTE: You can create a new environment shell from scratch with  ```poetry init```, followed by ```poetry shell```, after initial configuration. 
## Environment Variables

Change the following fields in the ```config.yaml``` project configuration file:
- ```USERPATH```: path to the user folder
- ```PROJECTPATH```: path to the project folder
- ```RAW_DATAPATH```: path to the MD simulation trajectories in LAMMPS format
- ```READY_DATAPATH```: path to the ready training data in ```.pt``` format
- ```SAVEPATH```: path to save the model checkpoints

## Polymer Melt Dataset
The training data comes from a coarse-grained (CG) polymer melt simulation of a Polybutadyene Rubber simulated from 300K to 400K with 100 chains made of 100 monomers each. The simulations were made with LAMMPS using a classic Verlet Algorithm implementing a dissipative particle dynamics (DPD) for CG monomers with a timestep of $\\delta t = 50 fs$. We refer to the paper [Consistent and Transferable Force Fields for Statistical Copolymer Systems at the Mesoscale](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00945) for more detailed information on the simulation and CG methods. 

The simulated trajectories are downsampled on-the-fly with a coarse-grained timestep $\Delta t = {100}{ps}$ for a total simulation time of $T_{sim} = {10}{\micro s}$, which is necessary to observe and estimate useful statistical properties, like diffusion coefficients and correlations functions (ACFs). Reaching the latter simulation time also depends on hardware configuration and number of CPUs cores exploited, but it is in general a computationally expensive task. For this system it may take $\sim 12-20$ days.  

With the ML-GLE, only a small fraction of the simulation time is needed to extrapolate the effective single polymer dynamics and discover the diffusion coefficient. 

### Pre-processing 
The modes dataset is obtained from the LAMMPS output trajectories in binary format. Under the directory [src/data/](src/data/README.md)
### Post-processing

The training dataset is available at at https://huggingface.co/datasets/gian-michele/meltBR

## Train the NAR model for modes non-Markovian dynamics

## Generate effective polymer dynamics with the GLE





