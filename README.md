## ML-enhanced Generalized Langevin Equation framework for Transient Anomalous Diffusion in Polymer Dynamics (NeurIPS ML for Physical Sciences Workshop 2023)

This project contains the source code needed to reproduce the results shown in the workshop paper submitted at the ML for Physical Sciences Workshop at the NeurIPS conference 2023 ([ML4PS workshop link](https://ml4physicalsciences.github.io/2023/)). 

                                                 
<video width="210" height="100" src="https://github.com/Gian-Michele-Cherchi/ml-gle/assets/43932730/a02179c8-d518-44bb-8ef2-32f9887fd154"></video>


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
- ```PROJECTPATH```: path to the project folder
- ```DATAPATH```: path to the MD simulation trajectories in LAMMPS format
- ```SAVEPATH```: path to save the model checkpoints
-  ```DEVICE```: device for training and generation

## Train the NAR model for modes non-Markovian dynamics

The training dataset is available at at https://huggingface.co/datasets/gian-michele/meltBR

Modes training with the NAR model can be done by running the following:

```
python src/run_train.py train.mode=1
```
by specifying the mode number one wants to train, it is possible to override the default mode, specified in the config file in the train folder. 

With Hydra, modes training can be done with a single command line: 
```
python src/run_train.py --multirun train=train_conf train.mode=1,2,3
```
This launches jobs which are run sequentially, but CPU multiprocessing could be exploited with joblib lanucher [JobLib Launcher](https://hydra.cc/docs/plugins/joblib_launcher/), although further tests implementing this option were not performed. 

## Generate effective polymer dynamics with the GLE
Autoregressive Generation with the NAR model can be done with the following command, by specifying the number of modes, provided they are all properly trained and checkpoints saved: 
```
python src/run_gen.py  eval=gen_conf eval.nmodes=12
```
In addition, if one wishes to also fit the Transient GLE equation and generate the Center of Mass dynamics, one can pass the following argument: 
```
python src/run_gen.py  eval=gen_conf eval.nmodes=12 eval.gle=True
```
Autocorrelations are computed on the fly by default for each generated mode. 
In order to compute the MD simulated baselines for test and comparison (Autocorrelation functions and the Mean Square Displacements), see the [NACFs](src/nacfs_.py) and [MSDs](src/msd_.py) scripts.

## Polymer Melt Dataset
The training data comes from a coarse-grained (CG) polymer melt simulation of a Polybutadyene Rubber simulated from 300K to 400K with 100 chains made of 100 monomers each. The simulations were made with LAMMPS using a classic Verlet Algorithm implementing a dissipative particle dynamics (DPD) for CG monomers with a timestep of $\\delta t = 50 fs$. We refer to the paper [Consistent and Transferable Force Fields for Statistical Copolymer Systems at the Mesoscale](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00945) for more detailed information on the simulation and CG methods. 

The simulated trajectories are downsampled on-the-fly with a coarse-grained timestep $\Delta t = {100}{ps}$ for a total simulation time of $T_{sim} = {10}{\micro s}$, which is necessary to observe and estimate useful statistical properties, like diffusion coefficients and correlations functions (ACFs). Reaching the latter simulation time also depends on hardware configuration and number of CPUs cores exploited, but it is in general a computationally expensive task. For this system it may take $\sim 12-20$ days.  

With the ML-GLE, only a small fraction of the simulation time is needed to extrapolate the effective single polymer dynamics and discover the diffusion coefficient. 

### Pre-processing: LAMMPS -> TORCH TENSOR
Under the directory [data](src/data), the ```preproc.py``` script can be used to convert the LAMMPS dump.nc containing the polymer trajectories file in pytorch format ```.pt``` and obtain the normal modes trajectories as well in a separate ```.pt``` file, which can subsequently be used for training. 
### Post-processing: Configuration Reconstruction (.xyz format)
The ```postproc.py``` script, under the same directory, can instead be used to recontruct the polymer configuration in real space starting from the generated modes. The file is saved in ```.pt``` format and also ```.xyz``` format, for visualization. 






