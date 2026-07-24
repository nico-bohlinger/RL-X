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
- Fully JIT-compiled per-environment replay and update loop

**Supported frameworks**
- JAX (Flax, fully JIT-compiled)


## Resources

- Paper and project: [Diffusion Models for Maximum Entropy Reinforcement Learning](https://alrhub.github.io/dime-website/)
- Reference code: [ALRhub/DIME](https://github.com/ALRhub/DIME)
