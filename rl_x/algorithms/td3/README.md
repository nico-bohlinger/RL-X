# Twin Delayed Deep Deterministic Gradient

Contains the implementation of [Twin Delayed Deep Deterministic Gradient (TD3)](https://arxiv.org/pdf/1802.09477).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- No special implementation details

**Supported frameworks**
- JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |


## Resources
- Paper: [Addressing Function Approximation Error in Actor-Critic Methods (Fujimoto et al., 2018)](https://arxiv.org/pdf/1802.09477)

- Spinning Up documentation: [here](https://spinningup.openai.com/en/latest/algorithms/td3.html)

- Blog post by Lilian Weng: [here](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#td3)

- Repositories:
    - Stable Baselines3: [here](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/td3/td3.py)
    - Stable Baselines Jax: [here](https://github.com/araffin/sbx/tree/master/sbx/td3)
    - CleanRL: [here](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py)
