# Simple Policy Optimization

Contains the implementation of [Simple Policy Optimization (SPO)](https://proceedings.mlr.press/v267/xie25m.html).

SPO replaces PPO's clipped policy objective with a differentiable quadratic penalty on probability-ratio deviation. Its RL-X policy, critic, rollout collection, GAE, objective, optimizer and fully JIT-compiled training loop are self-contained in the SPO folder rather than dispatched through PPO.


## RL-X implementation

**Implementation details**
- Implements the paper's policy loss exactly: `-advantage * ratio + abs(advantage) * (ratio - 1)^2 / (2 * epsilon)`
- Logs the per-sample ratio-deviation penalty
- Supports the fully JIT-compiled JAX data path used by MJX environments

**Supported frameworks**
- JAX (Flax, fully JIT-compiled)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Continuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax, fully JIT-compiled) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |


## Resources

- Paper: [Simple Policy Optimization (Xie et al., 2025)](https://proceedings.mlr.press/v267/xie25m.html)
- Reference implementation: [Simple Policy Optimization](https://github.com/MyRepositories-hub/Simple-Policy-Optimization)
