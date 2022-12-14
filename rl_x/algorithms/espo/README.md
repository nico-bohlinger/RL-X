# Early Stopping Policy Optimization

Contains the implementation of [Early Stopping Policy Optimization (ESPO)](https://arxiv.org/pdf/2202.00079).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Allows for calculating the delta with the mean or median of the ratio

**Supported frameworks**
- PyTorch, TorchScript, JAX (Flax)

**Supported action and observation space types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| PyTorch | ✅ | ❌ | ✅ | ❌ |
| TorchScript | ✅ | ❌ | ✅ | ❌ |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ |


## Resources

- Paper: [You May Not Need Ratio Clipping in PPO (Sun et al., 2022)](https://arxiv.org/pdf/2202.00079)

- Basics from Proximal Policy Optimization (PPO)