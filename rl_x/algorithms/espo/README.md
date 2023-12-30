# Early Stopping Policy Optimization

Contains the implementation of [Early Stopping Policy Optimization (ESPO)](https://arxiv.org/pdf/2202.00079).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Allows for calculating the delta with the mean or median of the ratio
- Flax version doesn't accurately linearly anneal the learning rate

**Supported frameworks**
- PyTorch, JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| PyTorch | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |


## Resources

- Paper: [You May Not Need Ratio Clipping in PPO (Sun et al., 2022)](https://arxiv.org/pdf/2202.00079)

- Basics from Proximal Policy Optimization (PPO)