# Diffusion Models for Maximum Entropy Reinforcement Learning

Contains a fully JIT-compiled RL-X port of [DIME](https://alrhub.github.io/dime-website/), an off-policy maximum-entropy algorithm whose actor is a time-reversed diffusion sampler trained with adjoint matching.

The implementation is intentionally independent from FPO and DPPO. It owns its score policy, diffusion integrator, path-cost objective, distributional twin critic, replay loop, entropy coefficient, configuration, and trainer.


## RL-X implementation

**Implementation details**
- Sixteen-step overdamped time-reversed diffusion sampler with Euler-Maruyama integration and the reference PISGRAD score network
- Adjoint-matching running, stochastic, and terminal path costs
- Twin 101-atom CrossQ critic with Batch Renormalization, joint current/next batches and categorical Bellman projection
- Automatic entropy coefficient, learned timestep and per-action friction, and delayed actor updates
- Reference-scale replay capacity and an update geometry preserving the paper's update-to-data ratio under 4096 parallel environments
- Keeps the critic and diffusion path in DIME's normalized `[-1, 1]` action domain and only rescales at the environment boundary when `action_rescaling=True`
- Fully JIT-compiled per-environment replay and update loop

The port was audited against official reference revision `7082bb7e8ab0e7f8a036a985333541d08d9d3e7d`. The 32 updates after each 4096-environment vector step preserve the reference samplewise update-to-data ratio of two while avoiding a 4096-fold over-update.

**Supported frameworks**
- JAX (Flax, fully JIT-compiled)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Continuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax, fully JIT-compiled) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |


## Resources

- Paper and project: [Diffusion Models for Maximum Entropy Reinforcement Learning](https://alrhub.github.io/dime-website/)
- Reference code: [ALRhub/DIME](https://github.com/ALRhub/DIME)
