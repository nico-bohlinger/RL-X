# Soft Actor-Critic

Contains the implementation of [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1801.01290).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Supports automatically adjusted temperature

**Supported frameworks**
- PyTorch, JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| PyTorch | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |


## Resources
- Paper: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor (Haarnoja et al., 2018)](https://arxiv.org/pdf/1801.01290)
- Paper 2: [Soft Actor-Critic Algorithms and Applications (Haarnoja et al., 2018)](https://arxiv.org/pdf/1812.05905)

- Spinning Up documentation: [here](https://spinningup.openai.com/en/latest/algorithms/sac.html)

- Blog post by Lilian Weng: [here](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#sac)

- Repositories:
    - Stable Baselines3: [here](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/sac.py)
    - Stable Baselines Jax: [here](https://github.com/araffin/sbx/tree/master/sbx/sac)
    - JAXRL: [here](https://github.com/ikostrikov/jaxrl/tree/main/jaxrl/agents/sac)
    - CleanRL: [here](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)
