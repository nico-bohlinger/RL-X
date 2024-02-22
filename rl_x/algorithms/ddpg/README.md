# Deep Deterministic Policy Gradient

Contains the implementation of [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Uses normal distributed action noise instead of the Ornstein-Uhlenbeck process

**Supported frameworks**
- JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |


## Resources
- Paper: [Continuous Control With Deep Reinforcement Learning (Lillicrap et al., 2015)](https://arxiv.org/pdf/1509.02971)

- Spinning Up documentation: [here](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

- Blog post by Lilian Weng: [here](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#ddpg)

- Repositories:
    - Stable Baselines3: [here](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ddpg/ddpg.py)
    - Stable Baselines Jax: [here](https://github.com/araffin/sbx/tree/master/sbx/ddpg)
    - JAXRL: [here](https://github.com/ikostrikov/jaxrl/tree/main/jaxrl/agents/ddpg)
    - CleanRL: [here](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py)
