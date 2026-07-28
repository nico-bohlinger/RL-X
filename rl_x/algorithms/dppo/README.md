# Diffusion Policy Policy Optimization

Contains the implementation of [Diffusion Policy Policy Optimization (DPPO)](https://diffusion-ppo.github.io/).

On how the algorithm works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Replaces the offline observation normalization statistics with online running statistics
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

- Paper: [Diffusion Policy Policy Optimization](https://diffusion-ppo.github.io/)
- Repository: [Diffusion Policy Policy Optimization](https://github.com/irom-princeton/dppo)
