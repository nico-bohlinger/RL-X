# BRO

Contains the implementation of [BRO](https://arxiv.org/pdf/2405.16158).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Uses a configurable first reset step and reset interval instead of the fixed reset schedule from the official implementation

**Supported frameworks**
- JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |


## Resources
- Paper: [Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control (Nauman et al., 2024)](https://arxiv.org/pdf/2405.16158)

- Repositories:
    - Source code of the paper: [here](https://github.com/naumix/BiggerRegularizedOptimistic)
