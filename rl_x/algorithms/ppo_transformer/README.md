# Proximal Policy Optimization + Transformer

Contains the implementation of [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) in combination with a [Transformer](https://arxiv.org/abs/1706.03762) for the policy network.

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Based on the PPO-Clip version: Clipping the ratio of the new and old policy
- The hyperparameters and network architecture for the ```flax_full_jit``` version are tuned for strong performance on many parallel environments for the custom mjx robot locomotion environment

**Supported frameworks**
- JAX (Flax)

**Supported observation space, action space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) full JIT | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |


## Resources

- Paper: [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)