# FastTD3

Contains the implementation of [Fast Twin Delayed Deep Deterministic Gradient (FastTD3)](https://arxiv.org/pdf/2505.22642).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- No special implementation details

**Supported frameworks**
- PyTorch, JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| PyTorch | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| JAX (Flax) full JIT | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |


## Resources
- Paper: [FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control (Seo et al., 2025)](https://arxiv.org/pdf/2505.22642)

- Repositories:
    - Source code of the paper: [here](https://github.com/younggyoseo/FastTD3)
