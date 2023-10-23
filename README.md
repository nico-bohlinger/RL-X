# RL-X

A framework for Reinforcement Learning research.


## Highlights

- 💡 **Perfect to understand and prototype algorithms**:
    - One algorithm = One directory -> No backtracking through  parent classes
    - Algorithms can be easily copied out of RL-X
- ⚒️ **Known DL libraries**: Implementations in PyTorch, TorchScript or JAX (Flax)
- ⚡ **Maximum speed**: JAX versions utilize JIT compilation -> A lot faster than PyTorch
- 🧪 **Mix and match and extend**: Generic interfaces between algorithms and environments
- ⛰️​ **Custom environments**: Examples for custom environments with MuJoCo or pure socket communication
- 📈 **Experiment tracking**: Console logging, Model saving & loading, Tensorboard, Weights and Biases


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
    - Atari
    - Classic control
- [EnvPool](https://github.com/sail-sg/envpool)
    - MuJoCo
    - Atari
    - Classic control
    - DeepMind Control Suite
- [Custom MuJoCo](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco)
    - Example of a custom MuJoCo environment
- [Custom Interface](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_interface)
    - Prototype of a custom environment interface with socket communication

All listed environments are directly embedded in RL-X and can be used out-of-the-box.

For further information on the environments and algorithms and how to add your own, read the respective README files.


## Install

```
git clone git@github.com:nico-bohlinger/RL-X.git
cd RL-X
pip install -e .[all]
```

Tested with Python 3.9.7 and 3.11.4.

### Version Overview
The following RL-X versions can be installed:
- ```pip install -e .```: Installs only the core dependencies
- ```pip install -e .[all]```: Core and all optional dependencies (see below)
- ```pip install -e .[jax]```: Needed to run Flax-based algorithms also on NVIDIA GPUs
- ```pip install -e .[jax_cpu]```: Needed to run Flax-based algorithms on CPU only
- ```pip install -e .[envpool]```: Needed to run EnvPool environments

> Remember that multiple versions can be combined, e.g. ```pip install -e .[jax,envpool]```.

### OS Restrictions
- EnvPool is not supported on MacOS and Windows yet
- JAX with GPU support is not supported on MacOS and Windows out-of-the-box but can be done with some extra effort (see [here](https://github.com/google/jax) for more information)

To install the out-of-the-box most feature-rich version, use the following commands:
- Linux: ```pip install -e .[all]```
- MacOS: ```pip install -e .[jax_cpu]```
- Windows: ```pip install -e .[jax_cpu]```


## Example
```
cd experiments
python experiment.py
```
Detailed instructions can be found in the README file in the experiments directory.


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