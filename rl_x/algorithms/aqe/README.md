# Aggressive Q-Learning with Ensembles

Contains the implementation of [Aggressive Q-Learning with Ensembles (AQE)](https://arxiv.org/pdf/2111.09159).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Supports setting different number of update steps for the critic
- Supports automatically adjusted temperature

**Supported frameworks**
- JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |


## Resources
- Paper: [Aggressive Q-Learning with Ensembles: Achieving Both High Sample Efficiency and High Asymptotic Performance (Wu et al., 2021)](https://arxiv.org/pdf/2111.09159)

- Basics from Soft Actor-Critic (SAC)
