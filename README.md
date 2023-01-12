# RL-X

A framework for Reinforcement Learning research.


## Highlights

- 💡 **Perfect to understand and prototype algorithms**:
    - One algorithm = One directory -> No backtracking through  parent classes
    - Algorithms can be easily copied out of RL-X
- ⚒️ **Known DL libraries**: Implementations in PyTorch, TorchScript or JAX (Flax)
- ⚡ **Maximum speed**: JAX versions utilize JIT compilation -> A lot faster than PyTorch
- 🧪 **Mix and match and extend**: Generic interfaces between algorithms and environments
- 📈 **Experiment tracking**: Console logging, Saving models, Tensorboard, Weights and Biases


## Implemented Algorithms
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) in PyTorch, TorchScript, Flax
- [Early Stopping Policy Optimization (ESPO)](https://arxiv.org/abs/2202.00079) in PyTorch, TorchScript, Flax
- [Soft Actor Critic (SAC)](https://arxiv.org/abs/1801.01290) in PyTorch, TorchScript, Flax
- [Randomized Ensembled Double Q-Learning (REDQ)](https://arxiv.org/abs/2101.05982) in Flax
- [Dropout Q-Functions (DroQ)](https://arxiv.org/abs/2110.02034) in Flax
- [Truncated Quantile Critics (TQC)](https://arxiv.org/abs/2005.04269) in Flax
- [Aggressive Q-Learning with Ensembles (AQE)](https://arxiv.org/abs/2111.09159) in Flax
- [Maximum a Posteriori Policy Optimization (MPO)](https://arxiv.org/pdf/1806.06920) in Flax


## Usable Environments
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
    - MuJoCo
- [EnvPool](https://github.com/sail-sg/envpool)
    - MuJoCo
    - Atari
    - Classic control
    - DeepMind Control Suite

Most of them have only one reference environment implemented.
To try out more just change the environment name in the create_env.py files or add a proper new directory for it.

For further infos on how to add more environments and algorithms read the respective README files.


## Install

```
git clone git@github.com:nico-bohlinger/RL-X.git
pip install -e .
```


## Example
```
cd experiments
python experiment.py
```
Detailed instructions can be found in the README file in the experiments directory.