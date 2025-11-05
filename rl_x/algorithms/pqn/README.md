# Parallelized Q-Network

Contains the implementation of [Parallelized Q-Network (PQN)](https://arxiv.org/pdf/2407.04811v2).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Based on the LayerNorm MLP variation of the algorithm
- Default hyperparamters are the ones used for the Atari experiments in the paper

**Supported frameworks**
- JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |


## Resources
- Paper: [Simplifying Deep Temporal Difference Learning (Gallici et al., 2024)](https://arxiv.org/pdf/2407.04811v2)

- Repositories:
    - Source code of paper: [here](https://github.com/mttga/purejaxql)
    - CleanRL: [here](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/pqn.py)
