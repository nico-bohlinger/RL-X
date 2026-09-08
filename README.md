<img src="docs/assets/images/logo_no_background.png" align="right" width="25%"/>


<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/images/readme_title_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/images/readme_title_light.svg">
  <img src="docs/assets/images/readme_title_light.svg" alt="RL-X" height="52">
</picture><br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/images/readme_separator_2px_native_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/images/readme_separator_2px_native_light.svg">
  <img src="docs/assets/images/readme_separator_2px_native_light.svg" alt="" width="72%" height="2" align="top">
</picture>

A framework for Reinforcement Learning research.


│ [Overview](#overview) │ [Getting Started](#getting-started) │ [Documentation](https://nico-bohlinger.github.io/RL-X/) │ [Citation](#citation) │


## Overview
### Highlights

- 💡 **Perfect to understand and prototype algorithms**:
    - One algorithm = One directory -> No backtracking through  parent classes
    - Algorithms can be easily copied out of RL-X
- ⚒️ **Known DL libraries**: Implementations in PyTorch and JAX
- ⚡ **Maximum speed**: Just-In-Time (JIT) compilation and parallel environments
- 🧪 **Mix and match and extend**: Generic interfaces between algorithms and environments
- ⛰️​ **Custom environments**: Examples for MuJoCo, Isaac Lab, ManiSkill or custom socket communication
- 🚀​ **GPU environments**: MJX, Warp, Isaac Lab and ManiSkill can run thousands of parallel environments
- 🤖​ **Robot learning**: Training and deployment for locomotion and motion tracking with the G1 and Go2
- ⚽ **RoboCup**: Training for the RoboCup soccer competition in MuJoCo and MJX
- 🕰️ **Memory architectures**: PPO with GRU, LSTM, Transformer, History Window, Mamba-2, Memory Actions
- 📈 **Experiments**: Checkpoints, Evaluation, Console log, Tensorboard, Weights & Biases, SLURM, Docker


### Implemented Algorithms
- [Proximal Policy Optimization (PPO)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/ppo)
- [Proximal Policy Optimization + Differentiable Trust Region Layers (PPO+DTRL)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/ppo_dtrl)
- [Proximal Policy Optimization + Gated Recurrent Unit (PPO+GRU)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/ppo_gru)
- [Proximal Policy Optimization + Long Short-Term Memory (PPO+LSTM)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/ppo_lstm)
- [Proximal Policy Optimization + Transformer (PPO+Transformer)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/ppo_transformer)
- [Proximal Policy Optimization + History Window (PPO+HistoryWindow)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/ppo_history_window)
- [Proximal Policy Optimization + Mamba-2 (PPO+Mamba-2)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/ppo_mamba2)
- [Proximal Policy Optimization + Memory Actions (PPO+MemoryActions)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/ppo_memory_actions)
- [Early Stopping Policy Optimization (ESPO)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/espo)
- [Trust Region Policy Optimization (TRPO)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/trpo)
- [Simple Policy Optimization (SPO)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/spo)
- [Flow Policy Optimization (FPO)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/fpo)
- [Diffusion Policy Policy Optimization (DPPO)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/dppo)
- [Diffusion Models for Maximum Entropy Reinforcement Learning (DIME)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/dime)
- [Relative Entropy Pathwise Policy Optimization (REPPO)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/reppo)
- [Deep Deterministic Policy Gradient (DDPG)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/ddpg)
- [Twin Delayed Deep Deterministic Gradient (TD3)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/td3)
- [Fast Twin Delayed Deep Deterministic Gradient (FastTD3)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/fasttd3)
- [Soft Actor Critic (SAC)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/sac)
- [Fast Soft Actor Critic (FastSAC)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/fastsac)
- [Flash Soft Actor Critic (FlashSAC)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/flashsac)
- [Randomized Ensembled Double Q-Learning (REDQ)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/redq)
- [Dropout Q-Functions (DroQ)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/droq)
- [Bigger, Regularized, Optimistic (BRO)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/bro)
- [CrossQ](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/crossq)
- [XQC](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/xqc)
- [Simplicity Bias (SimBa)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/simba)
- [Simplicity Bias V2 (SimBaV2)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/simbav2)
- [Truncated Quantile Critics (TQC)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/tqc)
- [Aggressive Q-Learning with Ensembles (AQE)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/aqe)
- [Maximum a Posteriori Policy Optimization (MPO)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/mpo)
- [Fast Maximum a Posteriori Policy Optimization (FastMPO)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/fastmpo)
- [Deep Q-Network (DQN)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/dqn)
- [Deep Q-Network with Histogram Loss using Gaussians (DQN HL-Gauss)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/dqn_hl_gauss)
- [Double Deep Q-Network (DDQN)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/ddqn)
- [Categorical Deep Q-Network (C51)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/c51)
- [Parallelized Q-Network (PQN)](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/algorithms/pqn)


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
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
    - Locomotion
- [Custom MuJoCo](rl_x/environments/custom_mujoco/README.md)
    - [Ant velocity-tracking](rl_x/environments/custom_mujoco/ant/): Examples with MuJoCo, MJX, MJX Warp and Warp Torch backends
    - [Robot locomotion](rl_x/environments/custom_mujoco/robot_locomotion/): Go2/G1 locomotion training and real-robot deployment
    - [Robot motion tracking](rl_x/environments/custom_mujoco/robot_motion_tracking/): BeyondMimic-style G1 tracking of retargeted LAFAN/OMOMO motions
    - [Custom RoboCup Soccer](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/robocup_soccer): Example of custom MuJoCo and MJX environments for the RoboCup soccer simulation 3D league and other humanoid soccer leagues
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
pip uninstall $(pip freeze | grep -E '\-cu12|\-cu13' | cut -d '=' -f 1) -y
pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip install "jax[cuda12]"
```
For other configurations, see the [detailed installation guide](https://nico-bohlinger.github.io/RL-X/#detailed-installation-guide) in the documentation.
As Isaac Lab needs to be installed separately, instructions can also be found there.
Similarly, ManiSkill might need additional steps, like downgrading numpy.


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
