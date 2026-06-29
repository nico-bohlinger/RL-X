# Trust Region Inverse Reinforcement Learning + Trust Region Loss

Contains the implementation of [Trust Region Inverse Reinforcement Learning (TRIRL)](https://arxiv.org/pdf/2605.11020) with KL constrained PPO for policy optimization.

On how the algorithms works, refer to the [Resources](#resources) section.

## RL-X implementation

**Dataset**
- Hosted on HuggingFace https://huggingface.co/datasets/anishdiwan/trirl_dataset.
```bash
# Make sure git-xet is installed (https://hf.co/docs/hub/git-xet)
curl -sSfL https://hf.co/git-xet/install.sh | sh

# Place in the top level directory .../RL-X/
git clone https://huggingface.co/datasets/anishdiwan/trirl_dataset
```

**Implementation Details**
- Based on the PPO-Clip version: Clipping the ratio of the new and old policy
- The hyperparameters and network architecture for the ```flax_full_jit``` version are tuned for strong performance on many parallel environments for mujoco benchmark environments

**Supported frameworks**
- JAX (Flax)

**Supported observation space, action space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) full JIT | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |


## Resources

- Paper: [Trust Region Inverse Reinforcement Learning: Explicit Dual Ascent using Local Policy Updates (Diwan et al., 2026)](https://arxiv.org/pdf/2605.11020)

- Repositories:
    - Official Codebase: [here](https://github.com/anishhdiwan/trust-region-irl)