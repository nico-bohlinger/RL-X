<img src="docs/assets/images/logo.png" align="right" width="25%"/>


# RL-X

A framework for Reinforcement Learning research.


│ [Overview](#overview) │ [Getting Started](#getting-started) │ [Documentation](https://nico-bohlinger.github.io/RL-X/) │ [Citation](#citation) │


## Overview
### Highlights

- 💡 **Perfect to understand and prototype algorithms**:
    - One algorithm = One directory -> No backtracking through  parent classes
    - Algorithms can be easily copied out of RL-X
- ⚒️ **Known DL libraries**: Implementations in PyTorch and mainly JAX
- ⚡ **Maximum speed**: Just-In-Time (JIT) compilation and parallel environments
- 🧪 **Mix and match and extend**: Generic interfaces between algorithms and environments
- ⛰️​ **Custom environments**: Examples for MuJoCo, Isaac Lab, ManiSkill or pure socket communication
- 🚀​ **GPU environments**: MJX, Isaac Lab and ManiSkill can run thousands of parallel environments
- 🤖​ **Robot learning**: Training and deployment for the Unitree Go2 (quadruped) and G1 (humanoid) robots
- 📈 **Experiments**: Checkpoints, Evaluation, Console log, Tensorboard, Weights & Biases, SLURM, Docker


### Implemented Algorithms
- [Proximal Policy Optimization (PPO)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/ppo) in PyTorch, Flax
- [Proximal Policy Optimization + Differentiable Trust Region Layers (PPO+DTRL)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/ppo_dtrl) in Flax
- [Early Stopping Policy Optimization (ESPO)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/espo) in PyTorch, Flax
- [Deep Deterministic Policy Gradient (DDPG)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/ddpg) in Flax
- [Twin Delayed Deep Deterministic Gradient (TD3)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/td3) in Flax
- [Soft Actor Critic (SAC)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/sac) in PyTorch, Flax
- [Randomized Ensembled Double Q-Learning (REDQ)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/redq) in Flax
- [Dropout Q-Functions (DroQ)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/droq) in Flax
- [CrossQ](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/crossq) in Flax
- [Truncated Quantile Critics (TQC)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/tqc) in Flax
- [Aggressive Q-Learning with Ensembles (AQE)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/aqe) in Flax
- [Maximum a Posteriori Policy Optimization (MPO)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/mpo) in Flax
- [Deep Q-Network (DQN)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/dqn) in Flax
- [Deep Q-Network with Histogram Loss using Gaussians (DQN HL-Gauss)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/dqn_hl_gauss) in Flax
- [Double Deep Q-Network (DDQN)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/ddqn) in Flax
- [Parallelized Q-Network (PQN)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/pqn) in Flax


### Usable Environments
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
    - MuJoCo
    - Atari
    - Classic control
    - DeepMind Control Suite
- [EnvPool](https://github.com/sail-sg/envpool)
    - MuJoCo
    - Atari
    - Classic control
    - DeepMind Control Suite
- [Custom MuJoCo](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco)
    - Example of a custom MuJoCo environment
    - Example of a custom MuJoCo XLA (MJX) environment
- [Custom Robot Learning](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/robot_locomotion)
    - Example of custom MuJoCo and MJX environments for quadruped and humanoid locomotion learning and real robot deployment
- [Custom Isaac Lab](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_isaac_lab)
    - Example of a custom Isaac Lab environment
- [Custom ManiSkill](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_maniskill)
    - Example of a custom ManiSkill environment
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
pip install -e .[all] --config-settings editable_mode=compat
pip uninstall $(pip freeze | grep -i '\-cu12' | cut -d '=' -f 1) -y
pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip install -U "jax[cuda12]"
```
For other configurations, see the [detailed installation guide](https://nico-bohlinger.github.io/RL-X/#detailed-installation-guide) in the documentation.
As Isaac Lab needs to be installed separately, instructions can also be found there.


### Example
```
cd experiments
python experiment.py
```
Detailed instructions for running experiments can be found in the [README file](https://github.com/nico-bohlinger/RL-X/blob/master/experiments/README.md) in the experiments directory or in the [documentation](https://nico-bohlinger.github.io/RL-X).

Example for Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nico-bohlinger/RL-X/blob/master/experiments/colab_experiment.ipynb)


## Citation
If you use RL-X in your research, please cite the following [paper](https://arxiv.org/abs/2310.13396):
```bibtex
@incollection{bohlinger2023rlx,
      title={RL-X: A Deep Reinforcement Learning Library (not only) for RoboCup}, 
      author={Nico Bohlinger and Klaus Dorer},
      booktitle={Robot World Cup},
      pages={228--239},
      year={2023},
      publisher={Springer}
}
```
