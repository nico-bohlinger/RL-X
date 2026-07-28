# Trust Region Policy Optimization

Contains the implementation of [Trust Region Policy Optimization (TRPO)](https://proceedings.mlr.press/v37/schulman15.html).

On how the algorithm works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- No special implementation details

**Supported frameworks**
- PyTorch, JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Continuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| PyTorch | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| JAX (Flax) full JIT | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |


## Resources

- Paper: [Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html)
- Repository: [Stable-Baselines3 Contrib TRPO](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/tree/master/sb3_contrib/trpo)
