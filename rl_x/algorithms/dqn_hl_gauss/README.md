# Deep Q-Network with Histogram Loss using Gaussians (DQN HL-Gauss)

Contains the implementation of [Deep Q-Network with Histogram Loss using Gaussians (DQN HL-Gauss)](https://arxiv.org/pdf/2403.03950.pdf).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- No special implementation details

**Supported frameworks**
- JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |


## Resources
- Paper: [Stop Regressing: Training Value Functions via Classification for Scalable Deep RL (Farebrother et al., 2024)](https://arxiv.org/pdf/2403.03950.pdf)

- Original paper on the histogram loss with Gaussians: [Improving Regression Performance with Distributional Losses (Imani and White, 2018)](https://arxiv.org/pdf/1806.04613.pdf)
