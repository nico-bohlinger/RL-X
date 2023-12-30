# Dropout Q-Functions

Contains the implementation of [Dropout Q-Functions (DroQ)](https://arxiv.org/pdf/2110.02034).

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
- Paper: [Dropout Q-Functions for Doubly Efficient Reinforcement Learning (Hiraoka et al., 2022)](https://arxiv.org/pdf/2110.02034)

- Basics from Soft Actor-Critic (SAC)

- Repositories:
    - Source code of paper: [here](https://github.com/TakuyaHiraoka/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning)
    - Stable Baselines Jax: [here](https://github.com/araffin/sbx/tree/master/sbx/droq)
