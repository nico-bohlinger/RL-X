# Proximal Policy Optimization + Differentiable Trust Region Layers

Contains the implementation of [Differentiable Trust Region Layers (DTRL)](https://arxiv.org/abs/2101.09207) in combination with [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- We assume diagonal covariances that are _not_ state dependent. 
- Uses a JAX implementation of the KL projection layer, with a L-BFGS for the covariance lagrangian multiplier computation and a custom vjp for computing gradients ([original code](https://github.com/boschresearch/trust-region-layers) in c++).
- Policy optimization based on the PPO-Clip version

**Supported frameworks**
- JAX (Flax)

**Supported observation space, action space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |


## Resources

- Paper: [Differentiable Trust Region Layers for Deep Reinforcement Learning (Otto et al., 2021)](https://arxiv.org/abs/2101.09207)