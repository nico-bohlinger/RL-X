# Diffusion Models for Maximum Entropy Reinforcement Learning

Contains a fully JIT-compiled RL-X port of [DIME](https://alrhub.github.io/dime-website/), an off-policy maximum-entropy algorithm whose actor is a time-reversed diffusion sampler trained with adjoint matching.

The implementation is intentionally independent from FPO and DPPO. It owns its score policy, diffusion integrator, path-cost objective, distributional twin critic, replay loop, entropy coefficient, configuration, and trainer.


## RL-X implementation

**Implementation details**
- Sixteen-step overdamped time-reversed diffusion sampler with Euler-Maruyama integration
- Adjoint-matching running, stochastic, and terminal path costs
- Twin 101-atom distributional critic with categorical Bellman projection
- Automatic entropy coefficient and delayed actor updates
- Fully JIT-compiled per-environment replay and update loop

**Supported frameworks**
- JAX (Flax, fully JIT-compiled)


## Resources

- Paper and project: [Diffusion Models for Maximum Entropy Reinforcement Learning](https://alrhub.github.io/dime-website/)
- Reference code: [ALRhub/DIME](https://github.com/ALRhub/DIME)
