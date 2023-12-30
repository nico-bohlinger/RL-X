# RL-X

A framework for Reinforcement Learning research.


## Highlights

- 💡 **Perfect to understand and prototype algorithms**:
    - One algorithm = One directory -> No backtracking through  parent classes
    - Algorithms can be easily copied out of RL-X
- ⚒️ **Known DL libraries**: Implementations in PyTorch or JAX (Flax)
- ⚡ **Maximum speed**: JAX versions utilize JIT compilation -> A lot faster than PyTorch's JIT
- 🧪 **Mix and match and extend**: Generic interfaces between algorithms and environments
- ⛰️​ **Custom environments**: Examples for custom environments with MuJoCo or pure socket communication
- 📈 **Experiment tracking**: Console logging, Model saving & loading, Tensorboard, Weights and Biases


## Implemented Algorithms
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) in PyTorch, Flax
- [Early Stopping Policy Optimization (ESPO)](https://arxiv.org/abs/2202.00079) in PyTorch, Flax
- [Soft Actor Critic (SAC)](https://arxiv.org/abs/1801.01290) in PyTorch, Flax
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
Default installation for a Linux system with a NVIDIA GPU:
```
conda create -n rlx python=3.11.4
conda activate rlx
git clone git@github.com:nico-bohlinger/RL-X.git
cd RL-X
pip install -e .[all]
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
For other configurations, see the detailed installation guide in ```INSTALL.md```.


## Example
```
cd experiments
python experiment.py
```
Detailed instructions for running experiments can be found in the README file in the experiments directory.


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