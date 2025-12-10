# FastSAC

Contains the implementation of [Fast Soft Actor-Critic (FastSAC)](https://arxiv.org/pdf/2512.01996).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- No symmetry observations as this is not always available in every environment

**Supported frameworks**
- PyTorch, JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| PyTorch | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| JAX (Flax) full JIT | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |


## Resources
- Paper: [Learning Sim-to-Real Humanoid Locomotion in 15 Minutes (Seo et al., 2025)](https://arxiv.org/pdf/2512.01996)

- Repositories:
    - Source code of the paper: [here](https://github.com/amazon-far/holosoma/tree/main/src/holosoma/holosoma/agents/fast_sac)
