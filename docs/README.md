# Documentation

**Table of Contents**:
- [READMEs](#readmes)
- [Detailed Installation Guide](#detailed-installation-guide)
- [Google Colab](#google-colab)
- [Run custom MJX environment](#run-custom-mjx-environment)


**Repository Link**: [RL-X](https://github.com/nico-bohlinger/RL-X)


## READMEs
Most documentation is available in the ```README.md``` files in the respective directories:
- [```/```](https://github.com/nico-bohlinger/RL-X/blob/master/README.md): Overview, Getting Started and Citation sections
- [```/experiments/```](https://github.com/nico-bohlinger/RL-X/blob/master/experiments/README.md): Information on the different ways to run experiments, experiment tracking and saving/loading of models
- [```/rl_x/algorithms/```](https://github.com/nico-bohlinger/RL-X/blob/master/algorithms/README.md): Information on the folder structure of algorithms, how to add new algorithms and how to mix and match them with environments
- [```/rl_x/algorithms/aqe/```](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/aqe/README.md): Implementation details of the Aggressive Q-Learning with Ensembles (AQE) algorithm
- [```/rl_x/algorithms/droq/```](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/droq/README.md): Implementation details of the Dropout Q-Functions (DroQ) algorithm
- [```/rl_x/algorithms/espo/```](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/espo/README.md): Implementation details of the Early Stopping Policy Optimization (ESPO) algorithm
- [```/rl_x/algorithms/mpo/```](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/mpo/README.md): Implementation details of the Maximum a Posteriori Policy Optimization (MPO) algorithm
- [```/rl_x/algorithms/ppo/```](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/ppo/README.md): Implementation details of the Proximal Policy Optimization (PPO) algorithm
- [```/rl_x/algorithms/redq/```](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/redq/README.md): Implementation details of the Randomized Ensembled Double Q-Learning (REDQ) algorithm
- [```/rl_x/algorithms/sac/```](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/sac/README.md): Implementation details of the Soft Actor Critic (SAC) algorithm
- [```/rl_x/algorithms/tqc/```](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/tqc/README.md): Implementation details of the Truncated Quantile Critics (TQC) algorithm
- [```/rl_x/environments/```](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/environments/README.md): Information on the folder structure of environments, how to add new environments and how to mix and match them with algorithms
- [```/rl_x/environments/custom_interface/```](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/environments/custom_interface/README.md): Implementation details of the custom environment interface with simple socket communication
- [```/rl_x/environments/custom_mujoco/```](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/environments/custom_mujoco/README.md): Implementation details of the custom MuJoCo environment examples (with and without MJX)
- [```/rl_x/environments/envpool/```](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/environments/gym/README.md): Details of the EnvPool environments
- [```/rl_x/environments/gym/```](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/environments/gym/README.md): Details of the Gymnasium environments
- [```/rl_x/runner/```](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/runner/README.md): Information on the folder structure of the runner class and how to use it to run experiments



## Detailed Installation Guide
### 1. Conda
For Linux, MacOS and Windows, a conda environment is recommended.  
All the code was tested with Python 3.11.4, other versions might work as well.
```
conda create -n rlx python=3.11.4
conda activate rlx
```

### 2. RL-X
For Linux, MacOS and Windows, RL-X has to be cloned.
```
git clone git@github.com:nico-bohlinger/RL-X.git
cd RL-X
```

### 3. Dependencies
For Linux, all dependencies can be installed with the following command:
```
pip install -e .[all]
```
For MacOS and Windows, EnvPool is currently not supported. Therefore, the following command has to be used:
```
pip install -e .
```

### 4. PyTorch
For Linux, MacOS and Windows, PyTorch has to be installed separately to use the CUDA 11.8 version such that there are no conflicts with JAX.
```
pip install "torch>=2.1.2" --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

### 5. JAX
For Linux, JAX with GPU support can be installed with the following command:
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
For MacOS and Windows, JAX with GPU support is not supported out-of-the-box. However, it can be done with some extra effort (see [here](https://github.com/google/jax) for more information).



## Google Colab
To run experiments in Google Colab take a look ```experiments/colab_experiment.ipynb``` or directly open it here:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nico-bohlinger/RL-X/blob/master/experiments/colab_experiment.ipynb) 



## Run custom MJX environment
```
python experiment.py --algorithm.name=ppo.flax --environment.name=custom_mujoco.ant_mjx --runner.track_console=True --environment.nr_envs=4000 --algorithm.nr_steps=10 --algorithm.minibatch_size=1000 --algorithm.nr_epochs=5 --algorithm.evaluation_frequency=-1
```
