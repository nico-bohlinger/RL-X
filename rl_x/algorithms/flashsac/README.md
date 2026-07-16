# FlashSAC

Contains the implementation of [FlashSAC](https://arxiv.org/pdf/2604.04539).

On how the algorithm works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Rescales normalized policy actions inside the algorithm instead of requiring an action-rescaling environment wrapper

**Supported frameworks**
- JAX (Flax)
- JAX (Flax, full JIT)
- PyTorch

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| JAX (Flax, full JIT) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| PyTorch | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |


## Resources
- Paper: [FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control (Kim, Lee et al., 2026)](https://arxiv.org/pdf/2604.04539)

- Repositories:
    - Source code of the paper: [here](https://github.com/Holiday-Robot/FlashSAC)
