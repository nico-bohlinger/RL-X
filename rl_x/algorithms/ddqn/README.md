# Double Deep Q-Network (DDQN)

Contains the implementation of [Double Deep Q-Network (DDQN)](https://arxiv.org/pdf/1509.06461.pdf).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Small differences in the hyperparameters compared to the original paper to stay consistent with the DQN implementation

**Supported frameworks**
- JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |


## Resources
- Paper: [Deep Reinforcement Learning with Double Q-learning (van Hasselt et al., 2016)](https://arxiv.org/pdf/1509.06461.pdf)

- Blog post by Julien Vitay: [here](https://julien-vitay.net/deeprl/2-Valuebased.html#double-dqn)
