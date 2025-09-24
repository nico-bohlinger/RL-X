# Differentiable Trust Region Projection Layers + Proximal Policy Optimization

Contains the implementation of [Differentiable Trust Region Projection Layers (TRPLs)](https://arxiv.org/abs/2101.09207) using a KL divergence projection layer.


## RL-X implementation

**Implementation Details**
- We assume diagonal covariances that are _not_ state dependent. 
- Uses a jax implementation of the projection layer, with a L-BFGS for covariance lagrangian multiplier computation and a custom vjp for computing gradients ([original code](https://github.com/boschresearch/trust-region-layers) in c++).
- Policy optimization based on the PPO-Clip version: clipping the ratio of the new and old policy

**Supported frameworks**
- JAX (Flax)

**Supported observation space, action space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |

