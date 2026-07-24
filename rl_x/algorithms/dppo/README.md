# Diffusion Policy Policy Optimization

Contains a fully JIT-compiled RL-X implementation of the denoising-MDP policy-gradient formulation from [Diffusion Policy Policy Optimization (DPPO)](https://diffusion-ppo.github.io/).

The original DPPO reference implementation fine-tunes pretrained PyTorch diffusion policies. This RL-X version implements the from-scratch denoising-MDP baseline used by the FPO reference implementation and the massively parallel diffusion-policy comparison papers: every denoising transition is treated as a stochastic policy transition, its behavior log likelihood is stored, and PPO clipping is applied to denoising-step likelihood ratios. Its implementation is intentionally self-contained and does not import FPO policy, sampler, objective, trainer, or configuration code.


## RL-X implementation

**Implementation details**
- Own diffusion policy, stochastic denoising sampler, objective, trainer, and configuration
- Stores complete denoising paths and behavior transition likelihoods
- Applies PPO clipping to per-denoising-step likelihood ratios
- Uses configurable denoising-step noise and optional final-half-only policy updates

**Supported frameworks**
- JAX (Flax, fully JIT-compiled)


## Resources

- Paper and project: [Diffusion Policy Policy Optimization](https://diffusion-ppo.github.io/)
- Reference code: [irom-princeton/dppo](https://github.com/irom-princeton/dppo)
- From-scratch denoising-MDP comparison: [akanazawa/fpo](https://github.com/akanazawa/fpo)
