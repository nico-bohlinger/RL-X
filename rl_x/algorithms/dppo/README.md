# Diffusion Policy Policy Optimization

Contains a fully JIT-compiled RL-X implementation of the denoising-MDP policy-gradient formulation from [Diffusion Policy Policy Optimization (DPPO)](https://diffusion-ppo.github.io/).

The original DPPO repository includes a from-scratch MuJoCo locomotion configuration in addition to pretrained-policy fine-tuning. This RL-X version follows that from-scratch configuration: every DDPM denoising transition is treated as a stochastic policy transition, its behavior log likelihood is stored, and PPO clipping is applied to denoising-step likelihood ratios. Its implementation is intentionally self-contained and does not import FPO policy, sampler, objective, trainer, or configuration code.


## RL-X implementation

**Implementation details**
- Own diffusion policy, stochastic denoising sampler, objective, trainer, and configuration
- Cosine DDPM posterior with epsilon prediction, reconstructed-action clipping, clipped sampling noise, and a minimum denoising standard deviation
- Running-return reward scaling used by the reference from-scratch locomotion setup
- Stores complete denoising paths and behavior transition likelihoods
- Treats every denoising transition as a separate PPO sample
- Clamps per-action-dimension transition log probabilities before averaging, matching the reference objective
- Applies denoising-step discounting and denoising-step-dependent PPO clipping
- Uses the reference residual actor and value networks, separate actor and critic learning rates, observation normalization, and target-KL early stopping

**Supported frameworks**
- JAX (Flax, fully JIT-compiled)


## Resources

- Paper and project: [Diffusion Policy Policy Optimization](https://diffusion-ppo.github.io/)
- Reference code: [irom-princeton/dppo](https://github.com/irom-princeton/dppo)
