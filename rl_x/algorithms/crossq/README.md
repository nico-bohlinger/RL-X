# CrossQ

Contains the implementation of [CrossQ](https://openreview.net/pdf?id=PczQtTsTIX).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Hyperparameters are exactly as in the paper but this means some standard parameters differ from the RL-X SAC implementation. So keep in mind, for a fair comparison between the two algorithms, these parameters have to be adjusted

**Supported frameworks**
- JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |


## Resources
- Paper: [Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity (Bhatt et al., 2024)](https://openreview.net/pdf?id=PczQtTsTIX)

- Repositories:
    - Source code of the paper: [here](https://github.com/adityab/CrossQ/tree/main)
