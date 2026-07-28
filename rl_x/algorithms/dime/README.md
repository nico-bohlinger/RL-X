# Diffusion Models for Maximum Entropy Reinforcement Learning

Contains the implementation of [Diffusion Models for Maximum Entropy Reinforcement Learning (DIME)](https://alrhub.github.io/dime-website/).

On how the algorithm works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Adjusts the number of updates to preserve the samplewise update-to-data ratio with vectorized environments
- Rescales actions inside the algorithm instead of using an environment wrapper

**Supported frameworks**
- PyTorch, JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Continuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| PyTorch | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| JAX (Flax) full JIT | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |


## Resources

- Paper: [Diffusion Models for Maximum Entropy Reinforcement Learning](https://alrhub.github.io/dime-website/)
- Repository: [DIME](https://github.com/ALRhub/DIME)
