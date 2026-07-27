# Diffusion Policy Policy Optimization

Contains RL-X Flax implementations of the denoising-MDP policy-gradient formulation from [Diffusion Policy Policy Optimization (DPPO)](https://diffusion-ppo.github.io/).

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
- Uses the original stochastic DDPM evaluation rule: posterior noise remains active above the final denoising step, with the reference `1e-3` floor
- Keeps RL-X fully-jitted environments in their native action domain by default; set `action_rescaling=True` for environments that expect normalized actions mapped to their Box bounds
- Provides the same diffusion policy, denoising-path likelihood objective and timeout-correct GAE through both the NumPy and fully JIT-compiled JAX environment interfaces

The port was audited against official reference revision `cc7234ad7ff39a8f32de3af903606723a16f0648`. The default optimizer and network profile follows the repository's from-scratch Hopper configuration. RL-X replaces its offline normalization file with online running observation statistics so the algorithm can train from scratch on arbitrary RL-X environments.

**Supported frameworks**
- JAX (Flax)
- JAX (Flax, fully JIT-compiled)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Continuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| JAX (Flax, fully JIT-compiled) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |


## Resources

- Paper and project: [Diffusion Policy Policy Optimization](https://diffusion-ppo.github.io/)
- Reference code: [irom-princeton/dppo](https://github.com/irom-princeton/dppo)
