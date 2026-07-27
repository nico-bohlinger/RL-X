# Simple Policy Optimization

Contains the implementation of [Simple Policy Optimization (SPO)](https://proceedings.mlr.press/v267/xie25m.html).

SPO replaces PPO's clipped policy objective with a differentiable quadratic penalty on probability-ratio deviation. Its RL-X policy, critic, rollout collection, GAE, objective, optimizer and fully JIT-compiled training loop are self-contained in the SPO folder rather than dispatched through PPO.


## RL-X implementation

**Implementation details**
- Implements the paper's policy loss exactly: `-advantage * ratio + abs(advantage) * (ratio - 1)^2 / (2 * epsilon)`
- Normalizes observations and discounted-return rewards with the reference clipping rules and normalizes advantages in each optimization minibatch
- Uses the authors' MuJoCo seven-layer Tanh policy, two-layer value network, clipped value loss, joint actor-critic gradient clipping, rollout horizon, optimizer settings, epoch count, and four-minibatch update geometry
- Keeps RL-X fully-jitted environments in their native action domain by default; `action_clipping_and_rescaling=True` enables the reference Gym-style environment-bound clipping path
- Logs the per-sample ratio-deviation penalty
- Supports RL-X's NumPy environment interface through `spo.flax` and the fully JIT-compiled JAX interface through `spo.flax_full_jit`

The port was audited against official reference revision `9fdeda315b00cee82d1dc4e10c01db200c97e909`.

**Supported frameworks**
- JAX (Flax)
- JAX (Flax, fully JIT-compiled)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Continuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| JAX (Flax, fully JIT-compiled) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |


## Resources

- Paper: [Simple Policy Optimization (Xie et al., 2025)](https://proceedings.mlr.press/v267/xie25m.html)
- Reference implementation: [Simple Policy Optimization](https://github.com/MyRepositories-hub/Simple-Policy-Optimization)
