# Deep Q-Network (DQN)

Contains the implementation of [Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236.pdf).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- No special implementation details

**Supported frameworks**
- JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |


## Resources
- Paper: [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/pdf/1312.5602.pdf)
- Paper 2: [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236.pdf)

- Blog post by Julien Vitay: [here](https://julien-vitay.net/deeprl/2-Valuebased.html#deep-q-network-dqn)

- Blog post by Lilian Weng: [here](https://lilianweng.github.io/posts/2018-02-19-rl-overview/#deep-q-network)

- Repositories:
    - Dopamine: [here](https://github.com/google/dopamine/tree/master/dopamine/agents/dqn)
    - Stable Baselines3: [here](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn)
    - Stable Baselines Jax: [here](https://github.com/araffin/sbx/tree/master/sbx/dqn)
    - CleanRL: [here](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari_jax.py)
