# SimBa

Contains the implementation of [SimBa](https://arxiv.org/pdf/2410.09754).

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
- Paper: [SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning (Lee et al., 2025)](https://arxiv.org/pdf/2410.09754)

- Repositories:
    - Source code of the paper: [here](https://github.com/SonyResearch/simba)
