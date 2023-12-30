# Truncated Quantile Critics

Contains the implementation of [Truncated Quantile Critics (TQC)](https://arxiv.org/pdf/2005.04269).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Supports automatically adjusted temperature

**Supported frameworks**
- JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |


## Resources
- Paper: [Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics (Kuznetsov et al., 2020)](https://arxiv.org/pdf/2005.04269)

- Basics from Soft Actor-Critic (SAC)

- Repositories:
    - Source code of paper: [here](https://github.com/SamsungLabs/tqc_pytorch)
    - Stable Baselines3 contrib: [here](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/tree/master/sb3_contrib/tqc)
    - Stable Baselines Jax: [here](https://github.com/araffin/sbx/tree/master/sbx/tqc)
