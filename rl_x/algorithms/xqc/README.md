# XQC

Contains the implementation of [XQC](https://arxiv.org/pdf/2509.25174).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Uses a fixed discount factor `gamma`. The official implementation instead derives it from the environment horizon `T` as `clip(1 - 5 / T, 0.95, 0.995)`

**Supported frameworks**
- JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |


## Resources
- Paper: [XQC: Well-conditioned Optimization Accelerates Deep Reinforcement Learning (Palenicek et al., 2026)](https://arxiv.org/pdf/2509.25174)

- Repositories:
    - Source code of the paper: [here](https://github.com/danielpalenicek/xqc)
