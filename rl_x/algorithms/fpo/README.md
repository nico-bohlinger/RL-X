# Flow Policy Optimization

Contains a native RL-X implementation of [Flow Policy Optimization (FPO)](https://flowreinforce.github.io/).

The implementation follows the authors' current G1 locomotion reference configuration. It samples actions by integrating a conditional flow, stores conditional-flow-matching losses under the behavior policy, and uses their difference as an asymmetric SPO likelihood-ratio surrogate. The implementation is adapted to the RL-X fully JIT-compiled environment interface and supports training from rewards without a pretrained policy.


## RL-X implementation

**Implementation details**
- Conditional flow-matching policy with Euler integration
- Sixty-four-step conditional flow with the reference G1 actor and critic networks
- Thirty-two per-action CFM samples with variance-preserving action-dimension reduction
- Per-sample ratios, symmetric CFM-loss clamps, negative-advantage protection and straight-through loss-difference bounding
- ASPO trust region: PPO for positive advantages and SPO for negative advantages
- GAE value targets, observation normalization and separate actor/critic diagnostics
- AdamW and the reference 32-epoch, four-minibatch G1 update geometry
- Latent-action clipping to `[-2, 2]`, matching the reference G1 runner
- Joint actor-critic gradient clipping matching the reference single-optimizer geometry
- Reference observation-normalizer epsilon and update horizon, configurable actor scaling, and actor EMA after the 500-update warmup
- NumPy-interface Flax rollout/update loop and fully JIT-compiled JAX rollout/update loop with the same CFM sampler, ASPO objective and EMA policy

The port was audited against official reference revision `b80112be1e8362263c4cd176e7aef21a275ff1c6`. The default profile is the authors' from-scratch G1 locomotion profile; it does not depend on a pretrained flow policy.

**Supported frameworks**
- JAX (Flax)
- JAX (Flax, fully JIT-compiled)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Continuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| JAX (Flax, fully JIT-compiled) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |


## Resources

- Paper and project: [Flow Matching Policy Gradients](https://flowreinforce.github.io/)
- Reference code: [amazon-far/fpo-control](https://github.com/amazon-far/fpo-control)
- The adapted reference implementation is Apache-2.0 licensed
