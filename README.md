<img src="docs/assets/images/logo.png" align="right" width="25%"/>


# RL-X

A framework for Reinforcement Learning research.


│ [Overview](#overview) │ [Getting Started](#getting-started) │ [Documentation](https://nico-bohlinger.github.io/RL-X/) │ [Citation](#citation) │


## Overview
### Highlights

- 💡 **Perfect to understand and prototype algorithms**:
    - One algorithm = One directory -> No backtracking through  parent classes
    - Algorithms can be easily copied out of RL-X
- ⚒️ **Known DL libraries**: Implementations in PyTorch or mainly JAX (Flax)
- ⚡ **Maximum speed**: Utilizing JIT compilation with PyTorch and JAX 
- 🧪 **Mix and match and extend**: Generic interfaces between algorithms and environments
- ⛰️​ **Custom environments**: Examples for custom environments with MuJoCo or pure socket communication
- 🚀​ **GPU environments**: MuJoCo XLA (MJX) supports running 4000 parallel environments on a GPU
- 📈 **Experiments**: Checkpoints, Evaluation, Console log, Tensorboard, Weights & Biases, SLURM, Docker


### Implemented Algorithms
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) in PyTorch, Flax
- [Early Stopping Policy Optimization (ESPO)](https://arxiv.org/abs/2202.00079) in PyTorch, Flax
- [Soft Actor Critic (SAC)](https://arxiv.org/abs/1801.01290) in PyTorch, Flax
- [Randomized Ensembled Double Q-Learning (REDQ)](https://arxiv.org/abs/2101.05982) in Flax
- [Dropout Q-Functions (DroQ)](https://arxiv.org/abs/2110.02034) in Flax
- [Truncated Quantile Critics (TQC)](https://arxiv.org/abs/2005.04269) in Flax
- [Aggressive Q-Learning with Ensembles (AQE)](https://arxiv.org/abs/2111.09159) in Flax
- [Maximum a Posteriori Policy Optimization (MPO)](https://arxiv.org/pdf/1806.06920) in Flax
- [Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236.pdf) in Flax


### Usable Environments
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
    - MuJoCo
    - Atari
    - Classic control
- [EnvPool](https://github.com/sail-sg/envpool)
    - MuJoCo
    - Atari
    - Classic control
    - DeepMind Control Suite
- [Custom MuJoCo](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco)
    - Example of a custom MuJoCo environment
    - Example of a custom MuJoCo XLA (MJX) environment
- [Custom Interface](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_interface)
    - Prototype of a custom environment interface with socket communication

All listed environments are directly embedded in RL-X and can be used out-of-the-box.

For further information on the environments ([README](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/environments/README.md)) and algorithms ([README](https://github.com/nico-bohlinger/RL-X/blob/master/algorithms/README.md)) and how to add your own, read the respective README files.


## Getting Started
### Install
Default installation for a Linux system with a NVIDIA GPU:
```
conda create -n rlx python=3.11.4
conda activate rlx
git clone git@github.com:nico-bohlinger/RL-X.git
cd RL-X
pip install -e .[all]
pip install "torch>=2.1.2" --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip uninstall $(pip freeze | grep -i '\-cu12' | cut -d '=' -f 1) -y
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
For other configurations, see the [detailed installation guide](https://nico-bohlinger.github.io/RL-X/#detailed-installation-guide) in the documentation.


### Example
```
cd experiments
python experiment.py
```
Detailed instructions for running experiments can be found in the [README file](https://github.com/nico-bohlinger/RL-X/blob/master/experiments/README.md) in the experiments directory or in the [documentation](https://nico-bohlinger.github.io/RL-X).

Example for Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nico-bohlinger/RL-X/blob/master/experiments/colab_experiment.ipynb)


## Citation
If you use RL-X in your research, please cite the following [preprint](https://arxiv.org/abs/2310.13396):
```bibtex
@misc{bohlinger2023rlx,
      title={RL-X: A Deep Reinforcement Learning Library (not only) for RoboCup}, 
      author={Nico Bohlinger and Klaus Dorer},
      year={2023},
      eprint={2310.13396},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```