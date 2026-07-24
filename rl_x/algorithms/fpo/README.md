# Flow Policy Optimization

Contains a native RL-X implementation of [Flow Policy Optimization (FPO)](https://flowreinforce.github.io/).

The implementation follows the authors' Apache-2.0 JAX reference implementation. It samples actions by integrating a conditional flow, stores conditional-flow-matching losses under the behavior policy, and uses their difference as a PPO-style likelihood-ratio surrogate. The implementation is adapted to the RL-X fully JIT-compiled environment interface and supports training from rewards without a pretrained policy.


## RL-X implementation

**Implementation details**
- Conditional flow-matching policy with Euler integration
- Configurable flow steps, timestep embedding, flow samples per action, stochastic sampling and feather noise
- Advantage-weighted FPO ratio with PPO clipping
- GAE value targets, observation normalization and separate actor/critic diagnostics
- Tanh action squashing followed by environment-bound rescaling
- Fully JIT-compiled JAX rollout and update loop

**Supported frameworks**
- JAX (Flax, fully JIT-compiled)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Continuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax, fully JIT-compiled) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |


## Resources

- Paper and project: [Flow Matching Policy Gradients](https://flowreinforce.github.io/)
- Reference code: [akanazawa/fpo](https://github.com/akanazawa/fpo)
- The adapted reference implementation is Apache-2.0 licensed
