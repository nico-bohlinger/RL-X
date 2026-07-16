# SimBaV2

Contains the implementation of [SimBaV2](https://arxiv.org/pdf/2502.15280).

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
- Paper: [Hyperspherical Normalization for Scalable Deep Reinforcement Learning (Lee et al., 2025)](https://arxiv.org/pdf/2502.15280)

- Repositories:
    - Source code of the paper: [here](https://github.com/dojeon-ai/SimbaV2)
