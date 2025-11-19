# Categorical Deep Q-Network (C51)

Contains the implementation of [Categorical Deep Q-Network (C51)](https://arxiv.org/pdf/1707.06887).

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
- Paper: [A Distributional Perspective on Reinforcement Learning (Bellemare et al., 2017)](https://arxiv.org/pdf/1707.06887)

- Deep RL Bootcamp Frontiers Lecture I: [here](https://www.youtube.com/watch?v=bsuvM1jO-4w&t=933s&ab_channel=AIPrism)

- Blog post by Julien Vitay: [here](https://julien-vitay.net/deeprl/src/2.4-DQNvariants.html#sec-distributionalrl)

- Repositories:
    - Dopamine: [here](https://github.com/google/dopamine/tree/master/dopamine/jax/agents/quantile)
    - CleanRL: [here](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_atari_jax.py)
