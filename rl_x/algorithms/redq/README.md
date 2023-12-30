# Randomized Ensembled Double Q-Learning

Contains the implementation of [Randomized Ensembled Double Q-Learning (REDQ)](https://arxiv.org/pdf/2101.05982).

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
- Paper: [Randomized Ensembled Double Q-Learning: Learning Fast Without a Model (Chen et al., 2021)](https://arxiv.org/pdf/2101.05982)

- Basics from Soft Actor-Critic (SAC)

- Repositories:
    - Source code of paper: [here](https://github.com/watchernyu/REDQ)
    - JAXRL: [here](https://github.com/ikostrikov/jaxrl/tree/main/jaxrl/agents/redq)
