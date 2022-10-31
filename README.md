# RL-X

A framework for Reinforcement Learning research.

## Implemented Algorithms
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) in PyTorch, TorchScript, Flax
- [Early Stopping Policy Optimization (ESPO)](https://arxiv.org/abs/2202.00079) in PyTorch, TorchScript, Flax
- [Soft Actor Critic (SAC)](https://arxiv.org/abs/1801.01290) in PyTorch, TorchScript, Flax
- [Randomized Ensembled Double Q-Learning (REDQ)](https://arxiv.org/abs/2101.05982) in Flax
- [Dropout Q-Functions (DroQ)](https://arxiv.org/abs/2110.02034) in Flax
- [Truncated Quantile Critics (TQC)](https://arxiv.org/abs/2005.04269) in Flax
- [Aggressive Q-Learning with Ensembles (AQE)](https://arxiv.org/abs/2111.09159) in Flax


## Install

```
git clone git@github.com:nico-bohlinger/RL-X.git
pip install -e .
```

## Example
```
cd experiments
python example.py
```
