# FastMPO

Contains the implementation of Fast Maximum a Posteriori Policy Optimization (FastMPO).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- No special implementation details

**Supported frameworks**
- JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) full JIT | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |


## Resources
- MPO paper: [Maximum a Posteriori Policy Optimization (Abdolmaleki et al., 2018)](https://arxiv.org/pdf/1806.06920)
- FastTD3 paper: [FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control (Seo et al., 2025)](https://arxiv.org/pdf/2505.22642)
- FastSAC paper: [Learning Sim-to-Real Humanoid Locomotion in 15 Minutes (Seo et al., 2025)](https://arxiv.org/pdf/2512.01996)
